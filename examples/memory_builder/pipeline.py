"""Build Spatial Memory from saved capture trajectories or RGB/odometry.

Contains build settings, sequence IO and fixed-trajectory capture.
Heavy reconstruction, feature extraction and rendering load only when used.
"""

from __future__ import annotations
from lingbot_navi.spatial_memory.settings import PreparationSettings
from lingbot_navi.spatial_memory.io import _json_default, write_json, write_npz
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import gc
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
from data_toolkit.render.camera import CameraSpec


@dataclass(frozen=True)
class RGBSequence:
    scene_id: str
    rgb_paths: tuple[Path, ...]
    camera_poses_c2w: np.ndarray
    timestamps_s: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    odometry_source: str = "metric_camera_odometry"

    def validate(self, *, check_images=True):
        from data_toolkit.render.geometry import rigid_transform

        n = len(self.rgb_paths)
        if n < 3 or not self.scene_id or "/" in self.scene_id or "\\" in self.scene_id:
            raise ValueError("sequence needs a scene ID and at least three frames")
        poses = np.asarray(self.camera_poses_c2w, dtype=np.float64)
        stamps = np.asarray(self.timestamps_s, dtype=np.float64)
        CameraSpec(self.width, self.height, self.intrinsics)
        if poses.shape != (n, 4, 4) or stamps.shape != (n,):
            raise ValueError(
                "RGB, synchronized camera odometry and timestamps must align"
            )
        for pose in poses:
            rigid_transform(pose)
        if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing")
        if check_images:
            from PIL import Image

            for path in self.rgb_paths:
                with Image.open(path) as image:
                    if image.size != (self.width, self.height):
                        raise ValueError(f"RGB size mismatch: {path}")
        return self

    def select(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or len(indices) < 3
            or np.any(indices < 0)
            or np.any(indices >= len(self.rgb_paths))
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("frame indices must be increasing, unique and in range")
        return RGBSequence(
            self.scene_id,
            tuple(self.rgb_paths[i] for i in indices),
            np.asarray(self.camera_poses_c2w)[indices],
            np.asarray(self.timestamps_s)[indices],
            np.asarray(self.intrinsics),
            self.width,
            self.height,
            self.odometry_source,
        )


@dataclass(frozen=True)
class TrajectoryAssets:
    ply: Path
    trajectory: Path


@dataclass(frozen=True)
class LingBotSettings:
    checkpoint: Path
    python: str = sys.executable
    device: str = "cuda:0"
    image_size: int = 518
    max_frame_num: int = 1024


@dataclass(frozen=True)
class FeatureSettings:
    model_path: Path
    device: str = "cuda:0"
    voxel_size_m: float = 0.04
    max_voxels: int = 100000
    input_width: int = 320
    input_height: int = 224
    min_depth_m: float = 0.1
    max_depth_m: float = 10.0
    seed: int = 0


@dataclass(frozen=True)
class MemoryProduct:
    output_dir: Path
    memory_manifest: Path
    prepared_memory: Path
    blocks: tuple[Path, ...]
    mode: str


def save_sequence(sequence, path):
    sequence.validate()
    path = Path(path).resolve()
    write_json(
        path,
        dict(
            format="lingbot-navi-rgb-odometry-v1",
            scene_id=sequence.scene_id,
            camera_axes="FLU",
            pose_type="camera_to_world",
            units="meter",
            odometry_source=sequence.odometry_source,
            width=sequence.width,
            height=sequence.height,
            intrinsics=np.asarray(sequence.intrinsics).tolist(),
            frames=[
                dict(
                    rgb_path=str(Path(rgb).resolve()),
                    timestamp_s=float(t),
                    camera_pose_c2w=np.asarray(pose).tolist(),
                )
                for rgb, t, pose in zip(
                    sequence.rgb_paths, sequence.timestamps_s, sequence.camera_poses_c2w
                )
            ],
        ),
    )


def load_sequence(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (
        data.get("format") != "lingbot-navi-rgb-odometry-v1"
        or data.get("camera_axes") != "FLU"
        or data.get("units") != "meter"
        or data.get("pose_type") != "camera_to_world"
    ):
        raise ValueError("expected metric camera-to-world FLU odometry")
    frames = data["frames"]
    return RGBSequence(
        data["scene_id"],
        tuple((path.parent / f["rgb_path"]).resolve() for f in frames),
        np.asarray([f["camera_pose_c2w"] for f in frames], dtype=np.float64),
        np.asarray([f["timestamp_s"] for f in frames], dtype=np.float64),
        np.asarray(data["intrinsics"], dtype=np.float64),
        data["width"],
        data["height"],
        data.get("odometry_source", "metric_camera_odometry"),
    ).validate()


def sequence_identity(sequence):
    images = []
    for path in sequence.rgb_paths:
        p = Path(path).resolve()
        stat = p.stat()
        images.append(dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    value = dict(
        scene_id=sequence.scene_id,
        images=images,
        poses=np.asarray(sequence.camera_poses_c2w).tolist(),
        timestamps=np.asarray(sequence.timestamps_s).tolist(),
        intrinsics=np.asarray(sequence.intrinsics).tolist(),
        dimensions=[sequence.width, sequence.height],
        odometry_source=sequence.odometry_source,
    )
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def ensure_request(output, request):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "request.json"
    normalized = json.loads(json.dumps(request, default=_json_default, allow_nan=False))
    if marker.exists():
        if json.loads(marker.read_text()) != normalized:
            raise ValueError(
                "output directory belongs to a different production request"
            )
    elif any(output.iterdir()):
        raise ValueError("refusing to adopt a nonempty unmarked output directory")
    else:
        write_json(marker, normalized)


def _asset_state(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def prepare_sequence(assets: TrajectoryAssets, output_dir: Path, *, device="cuda:0"):
    from .trajectory import load_trajectory

    return _prepare_sequence(assets, output_dir, load_trajectory(assets.trajectory), device)


def _prepare_sequence(assets, output_dir, trajectory, device):
    from data_toolkit.render import PERSPECTIVE_RENDERER

    output = Path(output_dir).resolve()
    ensure_request(
        output,
        dict(
            perspective_renderer=PERSPECTIVE_RENDERER,
            ply=_asset_state(assets.ply),
            trajectory=trajectory,
        ),
    )
    camera = CameraSpec(
        trajectory["width"], trajectory["height"], trajectory["intrinsics"]
    )
    poses = np.asarray([f["camera_pose_c2w"] for f in trajectory["frames"]])
    sequence = RGBSequence(
        trajectory["scene_id"],
        tuple(output / "rgb" / f"frame_{i:06d}.png" for i in range(len(poses))),
        poses,
        np.arange(len(poses), dtype=float),
        camera.K,
        camera.width,
        camera.height,
        "saved_capture_poses",
    )
    render_rgb(
        sequence,
        assets.ply,
        camera,
        device=device,
        trajectory=trajectory,
        preview_path=output.parent / "trajectory_bev.png",
    )
    save_sequence(sequence, output / "sequence.json")
    return sequence


def render_rgb(
    sequence, ply, camera, *, device="cuda:0", trajectory=None, preview_path=None
):
    """Use the shared perspective RGB backend; discard auxiliary range/alpha."""
    from data_toolkit.render import GaussianScene
    from data_toolkit.render.geometry import FLU_FROM_CV

    sequence.validate(check_images=False)
    renderer = GaussianScene(ply, device=device)
    try:
        for index, (path, pose) in enumerate(
            zip(sequence.rgb_paths, sequence.camera_poses_c2w)
        ):
            path = Path(path)
            if path.exists():
                with Image.open(path) as image:
                    if image.size != (sequence.width, sequence.height):
                        raise ValueError(f"Existing RGB has wrong dimensions: {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.png")
            pose_cv = np.asarray(pose).copy()
            pose_cv[:3, :3] = pose_cv[:3, :3] @ FLU_FROM_CV
            observation = renderer.render_camera(camera, pose_cv)
            observation.rgb.save(temporary)
            temporary.replace(path)
            if (index + 1) % 25 == 0:
                print(
                    f"Rendered {index + 1}/{len(sequence.rgb_paths)} capture frames",
                    flush=True,
                )
        if trajectory is not None:
            from examples.visualization.trajectory import render_trajectory_bev

            render_trajectory_bev(renderer, trajectory, preview_path)
    finally:
        renderer.close()
        del renderer
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _begin(source, output_dir, backend, features, preparation, *, segments):
    preparation.validate()
    output = Path(output_dir).resolve()
    trajectory = None
    if isinstance(source, TrajectoryAssets):
        from .trajectory import load_trajectory
        from data_toolkit.render import PERSPECTIVE_RENDERER

        trajectory = load_trajectory(source.trajectory)
        segments = trajectory["blocks"]
        identity = dict(
            perspective_renderer=PERSPECTIVE_RENDERER,
            ply=_asset_state(source.ply),
            trajectory=trajectory,
        )
    elif isinstance(source, RGBSequence):
        source.validate()
        from .trajectory import validate_blocks

        segments = validate_blocks(segments, len(source.rgb_paths))
        identity = dict(sequence=sequence_identity(source))
    else:
        raise TypeError("source must be TrajectoryAssets or RGBSequence")
    ensure_request(
        output,
        dict(
            schema_version=3,
            mode="v1_blocks",
            source=identity,
            backend=asdict(backend),
            features=asdict(features),
            preparation=asdict(preparation),
            blocks=segments,
        ),
    )
    if trajectory is None:
        sequence = source
        save_sequence(sequence, output / "sequence.json")
    else:
        capture = output / "capture"
        sequence = _prepare_sequence(source, capture, trajectory, backend.device)
        # Publish the same validated manifest at both existing locations.
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            manifest = Path(temporary) / "sequence.json"
            shutil.copy(capture / "sequence.json", manifest)
            os.replace(manifest, output / "sequence.json")
    write_json(output / "product.json", {"status": "running", "mode": "v1_blocks"})
    return output, sequence, segments, trajectory


def _finish(sequence, output, blocks, features, preparation, *, mode):
    from .memory import make_extractor, close_extractor, build_block_memory
    from .memory import package_memory

    extractor = make_extractor(features)
    try:
        paths = []
        for name, selected, reconstruction in blocks:
            paths.append(
                build_block_memory(
                    selected,
                    reconstruction,
                    output / "blocks" / name / "memory",
                    features,
                    extractor=extractor,
                )
            )
    finally:
        close_extractor(extractor)
        del extractor
    return package_memory(
        sequence, output, paths, mode=mode, preparation=preparation
    )


def produce_v1_blocks(
    source: TrajectoryAssets | RGBSequence,
    output_dir: Path,
    *,
    backend: LingBotSettings,
    features: FeatureSettings,
    segments: list[dict] | None = None,
    preparation: PreparationSettings = PreparationSettings(),
) -> MemoryProduct:
    """Split first; independent V1 model/cache and metric Sim(3) per block."""
    from .reconstruction import run_lingbot, align_reconstruction, lingbot_map_source_sha256

    output, sequence, segments, trajectory = _begin(
        source,
        output_dir,
        backend,
        features,
        preparation,
        segments=segments,
    )
    write_json(
        output / "segments.json", dict(mode="saved_frame_indices", segments=segments)
    )
    source_sha256 = lingbot_map_source_sha256()
    blocks = []
    for segment in segments:
        selected = sequence.select(segment["indices"])
        print(
            f"Reconstructing {segment['name']}: {len(selected.rgb_paths)} frames",
            flush=True,
        )
        if len(selected.rgb_paths) > backend.max_frame_num:
            raise ValueError(
                "V1 block exceeds max_frame_num; adjust the saved block indices"
            )
        work = output / "blocks" / segment["name"]
        prediction = run_lingbot(
            selected, backend, work / "inference", mode="streaming", source_sha256=source_sha256
        )
        reconstruction = align_reconstruction(selected, prediction, work)
        blocks.append((segment["name"], selected, reconstruction))
    product = _finish(
        sequence,
        output,
        blocks,
        features,
        preparation,
        mode="v1_blocks",
    )

    if trajectory is not None:
        from .memory import write_render_scene

        write_render_scene(sequence, source, output, trajectory=trajectory)
    return product
