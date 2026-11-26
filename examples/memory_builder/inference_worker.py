"""One independent LingBot-Map streaming block, using SDPA and a fresh cache."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .pipeline import write_npz


def run_inference(request, output_path):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    paths = request["rgb_paths"]
    count = len(paths)
    if not 8 <= count <= request["max_frame_num"]:
        raise ValueError("each block needs 8..max_frame_num ordered RGB frames")
    device = torch.device(request["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=request["image_size"], patch_size=14
    )
    if images.ndim != 4 or images.shape[:2] != (count, 3):
        raise ValueError("LingBot-Map preprocessing changed frame count or RGB layout")
    if not torch.isfinite(images).all():
        raise ValueError("nonfinite preprocessed RGB")
    model = GCTStream(
        img_size=request["image_size"],
        patch_size=14,
        enable_camera=True,
        enable_depth=True,
        enable_3d_rope=True,
        max_frame_num=request["max_frame_num"],
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    )
    checkpoint = torch.load(
        request["checkpoint_path"], map_location="cpu", weights_only=True
    )
    state = checkpoint.get("model", checkpoint)
    # A partial checkpoint must never silently produce geometry.
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model = model.to(device).eval().requires_grad_(False)
    model.clean_kv_cache()
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)
    start = time.monotonic()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ),
    ):
        prediction = model.inference_streaming(
            images,
            num_scale_frames=8,
            keyframe_interval=1,
            output_device=torch.device("cpu"),
        )
    # This API returns camera-to-world, despite the decoder's historical name.
    pose, intrinsic = pose_encoding_to_extri_intri(
        prediction["pose_enc"].float(),
        image_size_hw=images.shape[-2:],
        build_intrinsics=True,
    )
    pose = pose.detach().float().cpu().numpy()
    if pose.shape != (1, count, 3, 4):
        raise ValueError(f"unexpected LingBot-Map pose shape: {pose.shape}")
    c2w = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    c2w[:, :3] = pose[0]
    depth = prediction["depth"].detach().float().cpu().numpy()
    if depth.shape == (1, count, *images.shape[-2:], 1):
        depth = depth[0, ..., 0]
    elif depth.shape == (1, count, *images.shape[-2:]):
        depth = depth[0]
    if depth.shape != (count, *images.shape[-2:]):
        raise ValueError(f"unexpected LingBot-Map depth shape: {depth.shape}")
    if not np.isfinite(c2w).all() or not np.isfinite(depth).all():
        raise ValueError("LingBot-Map returned nonfinite geometry")
    metadata = dict(
        pose_convention="opencv_camera_to_world",
        pose_decoder_inverted=False,
        mode="streaming",
        use_sdpa=True,
        inference_dtype=str(dtype),
        frame_count=count,
        elapsed_s=time.monotonic() - start,
        missing_checkpoint_keys=[],
        unexpected_checkpoint_keys=[],
    )
    write_npz(
        output_path,
        c2w_opencv=c2w,
        predicted_depth=depth,
        intrinsic=intrinsic.detach().float().cpu().numpy()[0],
        rgb_paths=np.asarray(paths),
        preprocessing=np.asarray("v1"),
        request_fingerprint=np.asarray(request["request_fingerprint"]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"Reconstructed {count} frames in {metadata['elapsed_s']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    args = parser.parse_args()
    run_inference(json.loads(args.request_json.read_text()), args.output_npz)


if __name__ == "__main__":
    main()
