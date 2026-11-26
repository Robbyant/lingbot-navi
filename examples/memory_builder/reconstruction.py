"""Streaming LingBot-Map reconstruction and metric odometry alignment."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .pipeline import (
    LingBotSettings,
    RGBSequence,
    ensure_request,
    write_json,
    write_npz,
    sequence_identity,
)


def validate_poses(value, name="poses"):
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError(f"{name} must be at least three synchronized 4x4 C2W poses")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} contains nonfinite values")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} has invalid homogeneous rows")
    r = poses[:, :3, :3]
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} has nonorthonormal rotations")
    if not np.allclose(np.linalg.det(r), 1, atol=2e-3):
        raise ValueError(f"{name} has improper rotations")
    return poses


def align_to_odometry(
    predicted,
    odometry,
    *,
    min_baseline_m=0.25,
    rotation_weight_m=0.25,
    robust_scale_m=0.10,
):
    """Fit camera-to-world FLU poses with robust position + orientation residuals."""
    pred = validate_poses(predicted, "predicted")
    odom = validate_poses(odometry, "odometry")
    if pred.shape != odom.shape:
        raise ValueError("predicted and odometry poses must have matching frame counts")
    if min_baseline_m <= 0 or rotation_weight_m <= 0 or robust_scale_m <= 0:
        raise ValueError("alignment thresholds must be positive")
    x, y = pred[:, :3, 3], odom[:, :3, 3]
    if np.max(np.linalg.norm(y - y[0], axis=1)) < min_baseline_m:
        raise ValueError("odometry translation baseline is too small to recover scale")
    xc, yc = x - np.mean(x, axis=0), y - np.mean(y, axis=0)
    variance = np.sum(xc * xc)
    if variance < 1e-10:
        raise ValueError("predicted trajectory has no observable translation scale")
    rp, ro = pred[:, :3, :3], odom[:, :3, :3]
    initial_r = Rotation.from_matrix(ro @ rp.transpose(0, 2, 1)).mean().as_matrix()
    initial_s = np.sum((xc @ initial_r.T) * yc) / variance
    if not np.isfinite(initial_s) or initial_s <= 0:
        raise ValueError(
            "predicted and odometry trajectories have incompatible directions"
        )
    initial_t = np.mean(y, axis=0) - initial_s * initial_r @ np.mean(x, axis=0)

    def unpack(v):
        return np.exp(v[0]), Rotation.from_rotvec(v[1:4]).as_matrix(), v[4:7]

    def residual(v):
        s, r, t = unpack(v)
        position = s * x @ r.T + t - y
        angular = (
            Rotation.from_matrix(ro.transpose(0, 2, 1) @ r @ rp).as_rotvec()
            * rotation_weight_m
        )
        return np.concatenate([position.ravel(), angular.ravel()])

    fit = least_squares(
        residual,
        np.r_[
            np.log(initial_s), Rotation.from_matrix(initial_r).as_rotvec(), initial_t
        ],
        loss="soft_l1",
        f_scale=robust_scale_m,
        max_nfev=500,
    )
    if not fit.success:
        raise ValueError(f"odometry alignment failed: {fit.message}")
    scale, rotation, translation = unpack(fit.x)
    aligned = pred.copy()
    aligned[:, :3, :3] = rotation @ rp
    aligned[:, :3, 3] = scale * x @ rotation.T + translation
    errors = np.linalg.norm(aligned[:, :3, 3] - y, axis=1)
    angles = Rotation.from_matrix(
        ro.transpose(0, 2, 1) @ aligned[:, :3, :3]
    ).magnitude()
    diagnostics = {
        "method": "odometry_pose_sim3_soft_l1",
        "reference": "synchronized_metric_camera_odometry",
        "uses_reference_depth": False,
        "scale_model_units_to_m": float(scale),
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "frame_count": len(pred),
        "odometry_baseline_m": float(np.max(np.linalg.norm(y - y[0], axis=1))),
        "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "position_median_m": float(np.median(errors)),
        "position_p95_m": float(np.quantile(errors, 0.95)),
        "rotation_median_deg": float(np.rad2deg(np.median(angles))),
        "rotation_p95_deg": float(np.rad2deg(np.quantile(angles, 0.95))),
        "rotation_weight_m_per_rad": rotation_weight_m,
        "robust_scale_m": robust_scale_m,
        "limitations": "one global scale per block; local geometry/odometry drift is not corrected",
    }
    return aligned, diagnostics


def normalize_poses(c2w_opencv):
    """OpenCV C2W -> first-camera-local FLU C2W, without metric scaling."""
    from data_toolkit.render.geometry import rigid_transform, FLU_FROM_CV

    poses = np.asarray(c2w_opencv, dtype=np.float64)
    basis = np.eye(4)
    basis[:3, :3] = FLU_FROM_CV.T
    result = basis.T @ np.linalg.inv(poses[0]) @ poses @ basis
    for pose in result:
        rigid_transform(pose)
    return result


def restore_depth(
    depth, width, height, *, preprocessing="v1", image_size=518, patch_size=14
):
    """Invert V1 crop preprocessing onto the original RGB grid."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("predicted axial depth must be a finite HxW image")
    if preprocessing == "v1":
        rw = image_size
        rh = round(height * (rw / width) / patch_size) * patch_size
        crop_top = max((rh - image_size) // 2, 0)
        vh, vw = min(rh, image_size), rw
        pad_top, pad_left = (depth.shape[0] - vh) // 2, (depth.shape[1] - vw) // 2
        if min(pad_top, pad_left) < 0:
            raise ValueError("V1 prediction cannot contain the expected crop")
        visible = depth[pad_top : pad_top + vh, pad_left : pad_left + vw]
        if visible.shape != (vh, vw):
            raise ValueError("V1 prediction crop shape mismatch")
        canvas = np.zeros((rh, rw), np.float32)
        canvas[crop_top : crop_top + vh] = np.where(visible > 0, visible, 0)
        valid = np.zeros((rh, rw), np.uint8)
        valid[crop_top : crop_top + vh] = 1
    else:
        raise ValueError("unknown depth preprocessing")
    restored = np.array(
        Image.fromarray(canvas).resize((width, height), Image.Resampling.BILINEAR)
    )
    mask = np.array(
        Image.fromarray(valid).resize((width, height), Image.Resampling.NEAREST)
    )
    restored[mask == 0] = 0
    return restored


def lingbot_map_source_sha256():
    spec = importlib.util.find_spec("lingbot_map")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Install the shared requirements.txt to use LingBot-Map")
    package = Path(next(iter(spec.submodule_search_locations)))
    source_hash = hashlib.sha256()
    for source_file in sorted(package.rglob("*.py")):
        source_hash.update(str(source_file.relative_to(package)).encode())
        source_hash.update(source_file.read_bytes())
    return source_hash.hexdigest()


def run_lingbot(
    sequence: RGBSequence, settings: LingBotSettings, output_dir: Path, *, mode: str, source_sha256: str
):
    """Run one independent V1 block with a fresh model and KV cache."""
    if mode != "streaming":
        raise ValueError("production supports V1 streaming blocks only")
    sequence.validate()
    n = len(sequence.rgb_paths)
    if not 8 <= n <= settings.max_frame_num:
        raise ValueError(
            f"{mode} requires 8..{settings.max_frame_num} exact frames, got {n}; no silent sampling"
        )
    checkpoint = Path(settings.checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError("LingBot-Map checkpoint is missing")
    worker = Path(__file__).with_name("inference_worker.py")
    request = dict(
        checkpoint_path=str(checkpoint),
        lingbot_map_source_sha256=source_sha256,
        rgb_paths=[str(Path(p).resolve()) for p in sequence.rgb_paths],
        mode=mode,
        device=settings.device,
        image_size=settings.image_size,
        patch_size=14,
        num_scale_frames=8,
        keyframe_interval=1,
        max_frame_num=settings.max_frame_num,
        kv_cache_sliding_window=64,
        camera_num_iterations=4,
        use_sdpa=True,
        include_preprocessed_images=False,
        input_identity=sequence_identity(sequence),
        runtime=asdict(settings),
        worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        checkpoint_state=[checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns],
    )
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True, default=str).encode()
    ).hexdigest()
    request["request_fingerprint"] = fingerprint
    output = Path(output_dir).resolve()
    ensure_request(output, request)
    archive = output / "prediction.npz"
    if not archive.exists():
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = (
            package_root + os.pathsep + environment.get("PYTHONPATH", "")
        )
        subprocess.run(
            [
                settings.python,
                "-m",
                "examples.memory_builder.inference_worker",
                "--request-json",
                str(output / "request.json"),
                "--output-npz",
                str(archive),
            ],
            cwd=output,
            env=environment,
            check=True,
        )
    with np.load(archive, allow_pickle=False) as d:
        if (
            str(d["request_fingerprint"].item()) != fingerprint
            or list(d["rgb_paths"]) != request["rgb_paths"]
        ):
            raise ValueError("prediction archive does not match exact ordered RGBs")
        poses = np.asarray(d["c2w_opencv"], dtype=np.float64)
        depth = np.asarray(d["predicted_depth"], dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        preprocessing = str(d["preprocessing"].item()) if "preprocessing" in d else "v1"
    if (
        poses.shape != (n, 4, 4)
        or depth.ndim != 3
        or depth.shape[0] != n
        or not np.isfinite(depth).all()
    ):
        raise ValueError("invalid reconstruction shapes or nonfinite depths")
    return dict(
        poses_c2w_opencv=poses,
        depth=depth,
        preprocessing=preprocessing,
        archive=archive,
        request_fingerprint=fingerprint,
    )


def align_reconstruction(sequence, prediction, output_dir):
    """One robust metric Sim(3), applied jointly to poses and depth."""
    output = Path(output_dir)
    raw = normalize_poses(prediction["poses_c2w_opencv"])
    poses, diagnostics = align_to_odometry(raw, sequence.camera_poses_c2w)
    scale = diagnostics["scale_model_units_to_m"]
    depth = (
        np.stack(
            [
                restore_depth(
                    d,
                    sequence.width,
                    sequence.height,
                    preprocessing=prediction["preprocessing"],
                )
                for d in prediction["depth"]
            ]
        )
        * scale
    )
    if not np.any(depth > 0):
        raise ValueError("prediction has no positive metric depth")
    diagnostics.update(
        odometry_source=sequence.odometry_source,
        prediction_archive=str(prediction.get("archive", "injected")),
        depth_source="lingbot_map_prediction",
    )
    write_npz(
        output / "reconstruction.npz",
        camera_poses_c2w=poses,
        depth_m=depth,
        odometry_c2w=sequence.camera_poses_c2w,
        raw_lingbot_c2w=raw,
        scale_model_units_to_m=scale,
    )
    write_json(output / "alignment.json", diagnostics)
    return output / "reconstruction.npz"
