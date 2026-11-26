"""Persistent lingbot-map1.0 weights; independent KV cache for every memory segment."""
import contextlib
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from close_loop_test.common.map1.protocol import receive, send
from close_loop_test.common.map1.client import map1_builder_rgb


def load_model(checkpoint):
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    torch.manual_seed(42)
    model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True,
        enable_point=False, max_frame_num=1024, kv_cache_sliding_window=16, kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        use_sdpa=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to("cuda:0").eval()


def camera_to_world_poses(extrinsics):
    """The deployed lingbot-map1.0 checkpoint emits OpenCV c2w, despite upstream W2C docs.

    Keep decoded rotation and translation unchanged. Inverting here corrupts
    both trajectory and depth fusion; verified against recorded Habitat poses.
    """
    import numpy as np
    extrinsics = np.asarray(extrinsics)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError("Map1 camera poses must have shape [N,3,4]")
    poses = np.broadcast_to(np.eye(4), (len(extrinsics), 4, 4)).copy()
    poses[:, :3, :4] = extrinsics
    return poses


class Map1Stream:
    """Same prefix/causal forwards as inference_streaming, across IPC arrivals."""
    def __init__(self, model):
        self.model = model
        self.pending = []
        self.outputs = []
        self.shape = None
        model.clean_kv_cache()

    def online(self, frame):
        import numpy as np
        import torch
        if self.shape is not None and frame.shape != self.shape:
            raise ValueError("Map1 frame dimensions changed within segment")
        self.shape = frame.shape
        self.pending.append(frame)
        if not self.outputs and len(self.pending) < 8:
            return
        images = torch.from_numpy(np.stack(self.pending)).permute(0,3,1,2).float()[None] / 255
        device = next(self.model.parameters()).device
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            torch.compiler.cudagraph_mark_step_begin()
            output = self.model.forward(images.to(device), num_frame_for_scale=8,
                num_frame_per_block=images.shape[1], causal_inference=True)
            # Only geometry used by the memory builder; release dense auxiliary heads.
            self.outputs.append({key: output[key].detach().cpu() for key in ("pose_enc", "depth")})
        self.pending.clear()
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        raw = self.outputs[-1]
        ext, intrinsic = pose_encoding_to_extri_intri(raw["pose_enc"].float(), self.shape[:2])
        return dict(poses_c2w=camera_to_world_poses(ext[0].numpy()),
                    intrinsics=intrinsic[0].numpy(), depth=raw["depth"][0].float().numpy())

    def finish(self):
        import torch
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        if self.pending or not self.outputs:
            raise ValueError("Map1 scale initialization requires 8 frames")
        raw = {key: torch.cat([part[key] for part in self.outputs], dim=1)
               for key in ("pose_enc", "depth")}
        if self.model.pred_normalization:
            raw = self.model._normalize_predictions(raw)
        ext, intrinsic = pose_encoding_to_extri_intri(raw["pose_enc"].float(), self.shape[:2])
        return dict(poses_c2w=camera_to_world_poses(ext[0].numpy()),
                    intrinsics=intrinsic[0].numpy(), depth=raw["depth"][0].float().numpy())

    def close(self):
        self.pending.clear()
        self.outputs.clear()
        self.model.clean_kv_cache()


def serve(source, destination, loader=load_model, stream_factory=Map1Stream):
    import numpy as np
    model = key = prediction = None
    frames = []
    active = False
    stream = None
    while True:
        try:
            job = receive(source)
        except EOFError:
            if stream is not None:
                stream.close()
            return
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = {}
                op = job["op"]
                if op == "shutdown":
                    if stream is not None:
                        stream.close()
                    send(destination, dict(ok=True))
                    return
                if op == "begin":
                    if stream is not None:
                        stream.close()
                    stream = None
                    requested = job["checkpoint"]
                    if model is not None and key != requested:
                        raise ValueError("Map1 configuration changed; restart worker")
                    if model is None:
                        model = loader(requested)
                        key = requested
                    model.clean_kv_cache()
                    frames, prediction, active = [], None, True
                    stream = stream_factory(model)
                elif op == "frame":
                    if not active:
                        raise ValueError("Map1 segment not begun")
                    if len(frames) >= 1024:
                        raise ValueError("Map1 supports at most 1024 frames")
                    frame = map1_builder_rgb(job["rgb"])
                    if frames and frame.shape != frames[0].shape:
                        raise ValueError("Map1 frame dimensions changed within segment")
                    if stream is not None:
                        online = stream.online(frame)
                        if online is not None and job.get("return_online"):
                            result["online"] = dict(poses_c2w=online["poses_c2w"],
                                intrinsics=online["intrinsics"][-1], depth=online["depth"][-1])
                    frames.append(frame)
                elif op == "predict":
                    if not active or not 8 <= len(frames) <= 1024:
                        raise ValueError("Map1 requires 8..1024 frames")
                    started = time.monotonic()
                    count = len(frames)
                    height, width = frames[0].shape[:2]
                    prediction = stream.finish()
                    if stream is not None:
                        stream.close()
                    if np.asarray(prediction["poses_c2w"]).shape != (count, 4, 4) or np.asarray(prediction["intrinsics"]).shape != (count, 3, 3):
                        raise ValueError("Map1 frame/pose/intrinsic correspondence mismatch")
                    if np.asarray(prediction["depth"]).shape not in ((count,height,width),(count,height,width,1)):
                        raise ValueError("Map1 frame/depth correspondence mismatch")
                    if not all(np.isfinite(prediction[k]).all() for k in ("poses_c2w", "intrinsics", "depth")):
                        raise ValueError("Map1 returned nonfinite geometry")
                    frames, active = [], False
                    result = dict(poses_c2w=prediction["poses_c2w"], intrinsics=prediction["intrinsics"],
                        metadata=dict(backend="lingbot_map1", execution="streaming" if stream is not None else "batch", mode="segment_streaming",
                            complete=True, pose_convention="opencv_c2w", input_wh=[width,height], num_scale_frames=8,
                            input_interface="memory_rgb_v1", elapsed_s=time.monotonic()-started))
                    if job.get("include_depth"):
                        result["depth"] = prediction["depth"]
                elif op == "depth":
                    if prediction is None:
                        raise ValueError("Map1 prediction not ready")
                    index = job["index"]
                    if type(index) is not int or not 0 <= index < len(prediction["depth"]):
                        raise ValueError("Invalid Map1 depth index")
                    result = dict(depth=prediction["depth"][index])
                elif op == "abort":
                    frames, prediction, active = [], None, False
                    if stream is not None:
                        stream.close()
                        stream = None
                    if model is not None:
                        model.clean_kv_cache()
                else:
                    raise ValueError("unknown Map1 operation")
            send(destination, dict(ok=True, **result))
        except Exception as error:
            if stream is not None:
                stream.close()
                stream = None
            frames, prediction, active = [], None, False
            if model is not None:
                model.clean_kv_cache()
            traceback.print_exc(file=sys.stderr)
            send(destination, dict(ok=False, error=f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    serve(sys.stdin.buffer, sys.stdout.buffer)
