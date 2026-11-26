"""Visual features, 3D association, voxel fusion and memory export."""

from __future__ import annotations
from lingbot_navi.spatial_memory.features import (validate_canvas, qwen_image_geometry, Qwen3_5ViTExtractor)
from lingbot_navi.spatial_memory.observations import (ObservationBatch, unproject_patches, prepare_observations)
from lingbot_navi.spatial_memory.fusion import (MemoryResult, BestObservationAccumulator)
from lingbot_navi.spatial_memory.preparation import (_rotation, _arrays, prepare_memory)
from lingbot_navi.spatial_memory.builder import MemoryBuilder
from lingbot_navi.spatial_memory.io import write_prepared_memory
from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from .pipeline import (
    FeatureSettings, MemoryProduct, PreparationSettings, write_json, write_npz,
)


from lingbot_navi.spatial_memory.fusion import _QUALIFYING, _FALLBACK


def make_extractor(settings: FeatureSettings):
    return Qwen3_5ViTExtractor(
        checkpoint_path=str(settings.model_path),
        device=settings.device,
        input_width=settings.input_width,
        input_height=settings.input_height,
    )


def close_extractor(extractor):
    if hasattr(extractor, "_encoder"):
        extractor._encoder = None
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_block_memory(sequence, reconstruction, output_dir, settings, *, extractor):
    """One block, first predicted camera frame, 4 cm voxels by default."""
    import torch
    from scipy.spatial.transform import Rotation

    if (
        settings.voxel_size_m <= 0
        or settings.max_voxels <= 0
        or not 0 < settings.min_depth_m < settings.max_depth_m <= 65.535
    ):
        raise ValueError("invalid voxel/depth settings")
    if int(extractor.feature_dim) != 2560:
        raise ValueError("LingBot-Navi requires 2560-dimensional merged visual features")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(reconstruction, allow_pickle=False) as d:
        poses = np.asarray(d["camera_poses_c2w"], dtype=np.float64)
        depths = np.asarray(d["depth_m"], dtype=np.float32)
    if poses.shape != (len(sequence.rgb_paths), 4, 4) or depths.shape != (
        len(sequence.rgb_paths),
        sequence.height,
        sequence.width,
    ):
        raise ValueError("reconstruction and RGB sequence do not align")
    if not np.isfinite(depths).all():
        raise ValueError("nonfinite metric depth")
    global_from_block = poses[0]
    block_from_global = np.linalg.inv(global_from_block)
    local_poses = block_from_global @ poses
    # Match the millimetre depth convention used by the validated demo builder.
    depths = np.rint(np.clip(depths, 0, 65.535) * 1000) / 1000
    accumulator = MemoryBuilder(
        voxel_size=settings.voxel_size_m,
        feature_dim=2560,
        max_voxels=settings.max_voxels,
        preferred_min_depth=2.0,
        preferred_max_depth=5.0,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    seen = set()
    used = 0
    for i, (path, pose) in enumerate(zip(sequence.rgb_paths, local_poses)):
        # Repeated rendered pixels add no new visual evidence; retain earliest.
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        result = extractor.extract(rgb)
        geometry = result["spatial_geometry"]
        observation = prepare_observations(
            result["features"],
            depths[i],
            sequence.intrinsics,
            pose,
            result["patch_rgb"],
            min_depth=settings.min_depth_m,
            max_depth=settings.max_depth_m,
            **geometry,
        )
        camera_pose = torch.from_numpy(
            np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()].astype(
                np.float32
            )
        )
        accumulator.add_observation(
            observation, frame_name=Path(path).name, frame_pose=camera_pose
        )
        used += 1
    memory = accumulator.finalize()
    if len(memory.positions) == 0 or not np.isfinite(memory.features).all():
        raise ValueError("no finite memory points after depth/voxel processing")
    arrays = dict(
        positions=memory.positions,
        features=memory.features,
        rgb=memory.rgb,
        counts=memory.counts,
    )
    if memory.frame_poses is not None:
        arrays["frame_poses"] = memory.frame_poses
    archive = output / "spatial_mem.npz"
    write_npz(archive, **arrays)
    metadata = dict(
        settings=asdict(settings),
        frame_count=len(sequence.rgb_paths),
        retained_rgb_count=used,
        point_count=len(memory.positions),
        feature_dim=2560,
        coordinate_frame="first_predicted_camera_local_meters",
        global_from_block=global_from_block.tolist(),
        block_from_global=block_from_global.tolist(),
        pose_source="lingbot_map_odometry_sim3",
        depth_source="lingbot_map_prediction",
        uses_reference_depth=False,
    )
    write_json(output / "metadata.json", metadata)
    return archive


def package_memory(
    sequence, output, block_paths, *, mode, preparation=PreparationSettings()
):
    output = Path(output)
    blocks = [
        dict(block_id=path.parent.parent.name, path=str(path.relative_to(output)))
        for path in block_paths
    ]
    scene_key = "lingbo:" + sequence.scene_id
    manifest = output / "memory.json"
    write_json(
        manifest,
        dict(
            scene_id=sequence.scene_id,
            scene_key=scene_key,
            source="lingbo",
            blocks=blocks,
        ),
    )
    arrays, local_transforms = prepare_memory(
        block_paths, [b["block_id"] for b in blocks], scene_key,
        preparation=preparation,
    )
    metadata = dict(
        format="lingbot-navi-memory-v1",
        scene_key=scene_key,
        block_ids=[b["block_id"] for b in blocks],
        feature_dim=arrays["memory_features"].shape[1],
        coordinate_frame="independent_block_local_meters",
        preparation=asdict(preparation),
    )
    prepared = output / "memory.npz"
    write_prepared_memory(prepared, arrays, metadata)
    world_transforms = []
    for path, local_transform in zip(block_paths, local_transforms):
        block_metadata = json.loads((path.parent / "metadata.json").read_text())
        world_transforms.append(
            local_transform @ np.asarray(block_metadata["block_from_global"])
        )
    first_from_world = world_transforms[0]
    write_json(
        output / "coordinate_transforms.json",
        dict(
            first_block_from_world=first_from_world.tolist(),
            world_from_first_block=np.linalg.inv(first_from_world).tolist(),
            block_order=[b["block_id"] for b in blocks],
            prepared_block_from_world=[t.tolist() for t in world_transforms],
            world_from_prepared_block=[np.linalg.inv(t).tolist() for t in world_transforms],
            note="Includes first-camera anchoring and build-time voxel-centering/rotation",
        ),
    )
    write_json(
        output / "product.json",
        dict(
            status="complete",
            mode=mode,
            scene_id=sequence.scene_id,
            source_frame_count=len(sequence.rgb_paths),
            block_count=len(blocks),
            memory_points=len(arrays["memory_xyz"]),
            feature_dim=arrays["memory_features"].shape[1],
            odometry_source=sequence.odometry_source,
            reference_depth_used=False,
            coordinate_frame="first_block_meters",
            preparation=asdict(preparation),
        ),
    )
    return MemoryProduct(output, manifest, prepared, tuple(block_paths), mode)


def write_render_scene(sequence, assets, output, *, trajectory):
    """Pair the generated memory with the existing single-question renderer."""
    from data_toolkit.render.geometry import FLU_FROM_CV

    output = Path(output)
    transforms = json.loads((output / "coordinate_transforms.json").read_text())
    world_from_first = np.asarray(transforms["world_from_first_block"])
    views = []
    for index in range(0, len(sequence.rgb_paths), 10):
        camera = sequence.camera_poses_c2w[index].copy()
        camera[:3, :3] = camera[:3, :3] @ FLU_FROM_CV
        views.append(
            dict(
                frame_index=index,
                image=str(Path(sequence.rgb_paths[index]).resolve()),
                camera_to_world_cv=camera.tolist(),
                K=sequence.intrinsics.tolist(),
                width=sequence.width,
                height=sequence.height,
            )
        )
    xyz = sequence.camera_poses_c2w[:, :3, 3]
    write_json(
        output / "scene.json",
        dict(
            name=sequence.scene_id,
            memory="memory.npz",
            coordinate_frame="first_block_meters",
            asset_from_first_block=world_from_first.tolist(),
            gs_asset=str(Path(assets.ply).resolve()),
            views=views,
            visualization=dict(
                floor_z=float(
                    trajectory.get("visualization", {}).get("floor_z", xyz[0, 2] - 1.25)
                ),
                bev_bounds=trajectory.get("visualization", {}).get(
                    "bev_bounds",
                    [
                        *(xyz[:, :2].min(0) - 2).tolist(),
                        *(xyz[:, :2].max(0) + 2).tolist(),
                    ],
                ),
            ),
        ),
    )
