"""CUDA HM3D RGBD renderer adapted from navigation_wm's mesh renderer.
Uses optical-center sensor world poses, including camera pitch; no Habitat/EGL.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .glb import (_parse_glb, decode_basis_image, prepare_photo_textured_glb,
                  resolve_basisu_path, write_glb)
DEFAULT_BASISU = os.environ.get("BASISU_PATH", "basisu")
# Match Habitat/Magnum defaults used by CameraSensor (see dump of projection_matrix).
NEAR = 0.01
FAR = 1000.0
# Texture atlas: HM3D scenes often exceed one 8K atlas at native res; we downscale.
ATLAS_MAX_SIDE = 8192
ATLAS_PAD = 1
ATLAS_SCALE_CANDIDATES = (1.0, 0.5, 0.25, 0.125, 0.0625)


def _default_basisu() -> Path | None:
    env = resolve_basisu_path(None)
    if env is not None:
        return env
    p = Path(DEFAULT_BASISU)
    return p if p.is_file() else None


def _decode_one_basis(args):
    """Worker: (idx, basis_bytes, basisu_path_str) -> (idx, png_bytes)."""
    idx, basis_bytes, basisu_path = args
    png = decode_basis_image(basis_bytes, Path(basisu_path))
    return idx, png


def prepare_photo_textured_glb_parallel(
    basis_glb_path: str | Path,
    cache_dir: str | Path | None = None,
    basisu_path: str | Path | None = None,
    *,
    force: bool = False,
    workers: int | None = None,
) -> Path:
    """Decode Basis textures (parallel) and cache ``photo_textured.glb`` next to the mesh."""
    import copy

    basis_glb_path = Path(basis_glb_path)
    if cache_dir is None:
        cache_dir = basis_glb_path.parent / ".viewer_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "photo_textured.glb"
    if out_path.is_file() and not force:
        return out_path

    if basisu_path is None:
        basisu_path = _default_basisu()
    else:
        basisu_path = Path(basisu_path)
    if basisu_path is None or not Path(basisu_path).is_file():
        raise FileNotFoundError(
            "basisu binary not found. Build/install basis_universal or set BASISU_PATH. "
            f"Expected default at {DEFAULT_BASISU}"
        )
    basisu_path = Path(basisu_path)

    meta, bin_data = _parse_glb(basis_glb_path)
    images = meta.get("images", [])
    buffer_views = meta.get("bufferViews", [])
    if not images:
        raise ValueError(f"No images in {basis_glb_path}")

    jobs = []
    png_by_image: dict[int, bytes] = {}
    for image_idx, image in enumerate(images):
        mime_type = image.get("mimeType", "")
        buffer_view_idx = image["bufferView"]
        buffer_view = buffer_views[buffer_view_idx]
        start = buffer_view.get("byteOffset", 0)
        end = start + buffer_view["byteLength"]
        payload = bin_data[start:end]
        if mime_type == "image/x-basis":
            jobs.append((image_idx, payload, str(basisu_path)))
        elif mime_type == "image/png":
            png_by_image[image_idx] = payload
        else:
            raise ValueError(f"Unsupported image mime type: {mime_type}")

    n_workers = max(1, int(workers if workers is not None else min(os.cpu_count() or 4, 16)))
    print(
        f"[mesh_cuda] decoding {len(jobs)} Basis textures "
        f"({n_workers} workers) → {out_path}",
        flush=True,
    )
    if jobs:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_decode_one_basis, j) for j in jobs]
            for fut in as_completed(futs):
                idx, png = fut.result()
                png_by_image[idx] = png
                done += 1
                if done == 1 or done == len(jobs) or done % 20 == 0:
                    print(f"[mesh_cuda] basis decode {done}/{len(jobs)}", flush=True)

    basis_buffer_view_indices = {
        images[image_idx]["bufferView"]
        for image_idx, image in enumerate(images)
        if image.get("mimeType") == "image/x-basis"
    }
    image_idx_by_buffer_view = {
        image["bufferView"]: image_idx for image_idx, image in enumerate(images)
    }

    new_bin = bytearray()
    new_buffer_views: list[dict] = []
    for buffer_view_idx, buffer_view in enumerate(buffer_views):
        old_start = buffer_view.get("byteOffset", 0)
        old_length = buffer_view["byteLength"]
        if buffer_view_idx in basis_buffer_view_indices:
            image_idx = image_idx_by_buffer_view[buffer_view_idx]
            chunk = png_by_image[image_idx]
        else:
            chunk = bin_data[old_start : old_start + old_length]
        new_buffer_view = dict(buffer_view)
        new_buffer_view["byteOffset"] = len(new_bin)
        new_buffer_view["byteLength"] = len(chunk)
        new_bin.extend(chunk)
        while len(new_bin) % 4 != 0:
            new_bin.append(0)
        new_buffer_views.append(new_buffer_view)

    new_meta = copy.deepcopy(meta)
    new_meta["bufferViews"] = new_buffer_views
    new_meta["buffers"] = [{"byteLength": len(new_bin)}]
    for image in new_meta.get("images", []):
        if image.get("mimeType") == "image/x-basis":
            image["mimeType"] = "image/png"
    for texture in new_meta.get("textures", []):
        extensions = texture.get("extensions") or {}
        basis_ext = extensions.pop("GOOGLE_texture_basis", None)
        if basis_ext is not None and "source" not in texture:
            texture["source"] = basis_ext["source"]
        if not extensions:
            texture.pop("extensions", None)
    if "extensionsUsed" in new_meta:
        new_meta["extensionsUsed"] = [
            e for e in new_meta["extensionsUsed"] if e != "GOOGLE_texture_basis"
        ]
        if not new_meta["extensionsUsed"]:
            del new_meta["extensionsUsed"]
    if "extensionsRequired" in new_meta:
        new_meta["extensionsRequired"] = [
            e for e in new_meta["extensionsRequired"] if e != "GOOGLE_texture_basis"
        ]
        if not new_meta["extensionsRequired"]:
            del new_meta["extensionsRequired"]

    write_glb(new_meta, bytes(new_bin), out_path)
    print(f"[mesh_cuda] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out_path


def _resolve_photo_textured_glb_unlocked(
    basis_glb_path: str | Path,
    *,
    force: bool = False,
    workers: int | None = None,
    basisu_path: str | Path | None = None,
) -> Path:
    """Return cached photo GLB, building it if needed."""
    basis_glb_path = Path(basis_glb_path)
    cache_dir = basis_glb_path.parent / ".viewer_cache"
    out_path = cache_dir / "photo_textured.glb"
    if out_path.is_file() and not force:
        return out_path
    # Prefer parallel decoder; fall back to visualize_hm3d_scene helper.
    try:
        return prepare_photo_textured_glb_parallel(
            basis_glb_path,
            cache_dir,
            basisu_path=_default_basisu() if basisu_path is None else basisu_path,
            force=force,
            workers=workers,
        )
    except Exception as exc:
        warnings.warn(
            f"parallel photo_textured prepare failed ({exc}); trying sequential",
            RuntimeWarning,
            stacklevel=2,
        )
        out = prepare_photo_textured_glb(
            basis_glb_path,
            cache_dir,
            _default_basisu() if basisu_path is None else Path(basisu_path),
            force=force,
        )
        if out is None:
            raise FileNotFoundError("Failed to prepare photo_textured.glb") from exc
        return out


def resolve_photo_textured_glb(basis_glb_path, **kwargs):
    """Serialize initial cache creation across workers rendering the same scene."""
    cache = Path(basis_glb_path).parent / ".viewer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "lingbotnavi_repack.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _resolve_photo_textured_glb_unlocked(basis_glb_path, **kwargs)


def _quat_xyzw_to_R(rot_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in rot_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def habitat_view_proj_matrices(
    position_xyz,
    rotation_xyzw,
    *,
    cam_h: float,
    hfov_deg: float,
    width: int,
    height: int,
    near: float = NEAR,
    far: float = FAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (view 4x4, proj 4x4) for Habitat OpenGL camera (Y-up, look -Z)."""
    ax, ay, az = (float(v) for v in position_xyz)
    cam = np.array([ax, ay + float(cam_h), az], dtype=np.float64)
    R = _quat_xyzw_to_R(rotation_xyzw)  # local → world
    # World → camera: p_cam = R.T @ (p_world - cam)
    Rt = R.T
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = Rt
    view[:3, 3] = -Rt @ cam

    aspect = float(width) / float(height)
    hfov = math.radians(float(hfov_deg))
    f = 1.0 / math.tan(hfov * 0.5)
    # OpenGL perspective with horizontal FOV:
    # P00 = f, P11 = f * aspect (= 1/tan(vfov/2))
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f
    proj[1, 1] = f * aspect
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return view.astype(np.float32), proj.astype(np.float32)


def _load_scene_batches(photo_glb: Path):
    """Load photo GLB; merge meshes that share a texture into draw batches.

    Returns list of dicts: ``{verts, faces, uvs, tex, aabb_*}``.
    ``tex`` is top-left RGB uint8; ``uvs`` are glTF-style (V=0 at top).
    """
    import trimesh
    from PIL import Image

    scene = trimesh.load(str(photo_glb), force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    world_tf: dict[str, np.ndarray] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph.get(node_name)
        world_tf[geom_name] = np.asarray(transform, dtype=np.float64)

    # key -> {tex, parts: [(verts, faces, uv), ...]}
    batches: dict[int, dict[str, Any]] = {}

    for name, geom in scene.geometry.items():
        if not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        transform = world_tf.get(name, np.eye(4, dtype=np.float64))
        verts = np.asarray(geom.vertices, dtype=np.float64)
        ones = np.ones((verts.shape[0], 1), dtype=np.float64)
        verts_w = (transform @ np.concatenate([verts, ones], axis=1).T).T[:, :3]
        # trimesh loads this GLB as Z-up (x, y, z)_tri = (x, -z, y)_habitat.
        # Convert back to Habitat Y-up so agent poses match Magnum/Habitat.
        # (x, y, z)_hab = (x_tri, z_tri, -y_tri)
        verts_w = np.stack(
            [verts_w[:, 0], verts_w[:, 2], -verts_w[:, 1]], axis=-1
        ).astype(np.float32)
        faces = np.asarray(geom.faces, dtype=np.int32)

        visual = geom.visual
        uv = None
        if hasattr(visual, "uv") and visual.uv is not None:
            uv = np.asarray(visual.uv, dtype=np.float32)

        img = None
        material = getattr(visual, "material", None)
        if material is not None:
            img = getattr(material, "image", None)
            if img is None:
                img = getattr(material, "baseColorTexture", None)
        if img is None:
            img = getattr(visual, "image", None)
        if img is not None and not isinstance(img, np.ndarray):
            img = np.asarray(Image.fromarray(np.asarray(img)).convert("RGB"))
        elif isinstance(img, np.ndarray) and img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif isinstance(img, np.ndarray) and img.shape[-1] == 4:
            img = img[:, :, :3]
        if img is None:
            img = np.full((4, 4, 3), 180, dtype=np.uint8)
        img = np.asarray(img, dtype=np.uint8)
        if uv is None:
            uv = np.zeros((len(verts_w), 2), dtype=np.float32)

        key = hash(img.tobytes())
        if key not in batches:
            batches[key] = {"tex": img, "parts": []}
        batches[key]["parts"].append((verts_w, faces, uv.astype(np.float32)))

    if not batches:
        raise RuntimeError(f"No drawable geometry in {photo_glb}")

    out = []
    for batch in batches.values():
        all_v, all_f, all_uv = [], [], []
        v_off = 0
        for verts, faces, uv in batch["parts"]:
            all_v.append(verts)
            all_f.append(faces + v_off)
            all_uv.append(uv)
            v_off += len(verts)
        verts = np.concatenate(all_v, axis=0)
        out.append(
            {
                "verts": verts,
                "faces": np.concatenate(all_f, axis=0).astype(np.int32),
                "uvs": np.concatenate(all_uv, axis=0).astype(np.float32),
                "tex": batch["tex"],
                "aabb_min": verts.min(axis=0).astype(np.float32),
                "aabb_max": verts.max(axis=0).astype(np.float32),
            }
        )
    return out


def _try_shelf_pack(
    sizes: list[tuple[int, int]],
    max_side: int,
    pad: int = ATLAS_PAD,
) -> tuple[list[tuple[int, int, int, int]], int, int] | None:
    """Shelf-pack (w,h) items. Returns placements (x,y,w,h), atlas_w, atlas_h or None."""
    items = sorted(enumerate(sizes), key=lambda t: (-t[1][1], -t[1][0]))
    shelves: list[dict[str, int]] = []
    placements: dict[int, tuple[int, int, int, int]] = {}

    for idx, (w, h) in items:
        if w + 2 * pad > max_side or h + 2 * pad > max_side:
            return None
        placed = False
        for shelf in shelves:
            if shelf["height"] < h + 2 * pad:
                continue
            if shelf["x"] + w + pad > max_side:
                continue
            x = shelf["x"]
            y = shelf["y"] + pad
            placements[idx] = (x, y, w, h)
            shelf["x"] = x + w + pad
            placed = True
            break
        if placed:
            continue
        y0 = 0 if not shelves else shelves[-1]["y"] + shelves[-1]["height"]
        if y0 + h + 2 * pad > max_side:
            return None
        shelf = {"y": y0, "height": h + 2 * pad, "x": pad}
        placements[idx] = (pad, y0 + pad, w, h)
        shelf["x"] = pad + w + pad
        shelves.append(shelf)

    atlas_h = shelves[-1]["y"] + shelves[-1]["height"]
    atlas_w = max(s["x"] for s in shelves)
    atlas_w = max(1, min(max_side, atlas_w))
    atlas_h = max(1, min(max_side, atlas_h))
    ordered = [placements[i] for i in range(len(sizes))]
    return ordered, atlas_w, atlas_h


def pack_texture_atlas(
    textures: list[np.ndarray],
    *,
    max_side: int = ATLAS_MAX_SIDE,
    pad: int = ATLAS_PAD,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], float]:
    """Pack top-left RGB textures into one atlas.

    Returns ``(atlas_uint8_topleft, uv_rects_opengl, scale)`` where each uv rect is
    ``(u0, v0, u1, v1)`` in OpenGL UV space (V=0 at bottom) for the *flipped* atlas
    that callers should upload after ``np.flip(atlas, 0)``.
    """
    from PIL import Image

    if not textures:
        atlas = np.zeros((1, 1, 3), dtype=np.uint8)
        return atlas, [(0.0, 0.0, 1.0, 1.0)], 1.0

    native = [(int(t.shape[1]), int(t.shape[0])) for t in textures]
    chosen_scale = None
    pack = None
    for scale in ATLAS_SCALE_CANDIDATES:
        sizes = [
            (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            for w, h in native
        ]
        pack = _try_shelf_pack(sizes, max_side=max_side, pad=pad)
        if pack is not None:
            chosen_scale = float(scale)
            break
    if pack is None or chosen_scale is None:
        raise RuntimeError(
            f"Cannot pack {len(textures)} textures into atlas max_side={max_side}"
        )

    placements, atlas_w, atlas_h = pack
    atlas = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
    uv_rects: list[tuple[float, float, float, float]] = []

    for idx, tex in enumerate(textures):
        x, y, w, h = placements[idx]
        if chosen_scale != 1.0:
            img = Image.fromarray(tex).resize((w, h), Image.Resampling.LANCZOS)
            rgb = np.asarray(img, dtype=np.uint8)
        else:
            rgb = tex[:h, :w, :3]
        atlas[y : y + h, x : x + w] = rgb
        # OpenGL UV after flipping atlas vertically for upload.
        u0 = x / float(atlas_w)
        u1 = (x + w) / float(atlas_w)
        v1 = 1.0 - (y / float(atlas_h))  # top of rect in GL
        v0 = 1.0 - ((y + h) / float(atlas_h))  # bottom of rect in GL
        uv_rects.append((u0, v0, u1, v1))

    return atlas, uv_rects, chosen_scale


def batches_to_atlas_draw(
    batches: list[dict[str, Any]],
    *,
    max_side: int = ATLAS_MAX_SIDE,
) -> dict[str, Any]:
    """Merge per-texture batches into one atlas-textured draw call."""
    textures = [b["tex"] for b in batches]
    atlas, uv_rects, scale = pack_texture_atlas(textures, max_side=max_side)

    all_v, all_f, all_uv = [], [], []
    v_off = 0
    for batch, rect in zip(batches, uv_rects):
        u0, v0, u1, v1 = rect
        uv = np.asarray(batch["uvs"], dtype=np.float32)
        # glTF UV: V=0 at top. Map into atlas OpenGL rect.
        u = u0 + np.clip(uv[:, 0], 0.0, 1.0) * (u1 - u0)
        v_img = np.clip(uv[:, 1], 0.0, 1.0)
        v = v1 + v_img * (v0 - v1)
        all_v.append(batch["verts"])
        all_f.append(batch["faces"] + v_off)
        all_uv.append(np.stack([u, v], axis=-1).astype(np.float32))
        v_off += len(batch["verts"])

    verts = np.concatenate(all_v, axis=0)
    return {
        "verts": verts,
        "faces": np.concatenate(all_f, axis=0).astype(np.int32),
        "uvs": np.concatenate(all_uv, axis=0),
        "atlas": atlas,  # top-left; flip before GPU upload
        "atlas_scale": scale,
        "aabb_min": verts.min(axis=0).astype(np.float32),
        "aabb_max": verts.max(axis=0).astype(np.float32),
    }


class MeshCudaRenderer:
    """Optical-pose-compatible RGBD renderer using nvdiffrast CUDA (no EGL).

    Default path packs all material textures into one atlas and issues a single
    rasterize/texture pass per frame.
    """

    def __init__(
        self,
        glb_path: str,
        *,
        height=(240, 320),
        cam_h: float = 1.2,
        hfov: float = 90.0,
        gpu_device_id: int = 0,
        force_repack: bool = False,
        basis_workers: int | None = None,
        atlas_max_side: int = ATLAS_MAX_SIDE,
    ):
        import torch
        import nvdiffrast.torch as dr

        self.frame_h = int(height[0] if isinstance(height, (tuple, list)) else height)
        self.frame_w = int(height[1] if isinstance(height, (tuple, list)) else 320)
        # build_sim uses height=(H, W)
        if isinstance(height, (tuple, list)) and len(height) == 2:
            self.frame_h, self.frame_w = int(height[0]), int(height[1])
        self.cam_h = float(cam_h)
        self.hfov = float(hfov)
        self.device = torch.device(f"cuda:{int(gpu_device_id)}")
        self._torch = torch
        self._dr = dr

        t0 = time.perf_counter()
        photo = resolve_photo_textured_glb(
            glb_path, force=force_repack, workers=basis_workers
        )
        print(f"[mesh_cuda] loading mesh {photo}", flush=True)
        batches = _load_scene_batches(photo)
        t_load = time.perf_counter()
        draw = batches_to_atlas_draw(batches, max_side=int(atlas_max_side))
        from .topology import cached_repair
        draw["verts"], draw["faces"], draw["uvs"], topology = cached_repair(
            draw["verts"], draw["faces"], draw["uvs"], photo.parent)
        print("[mesh_cuda] topology " + json.dumps(topology), flush=True)
        atlas = np.flip(draw["atlas"], axis=0).copy()  # OpenGL V for nvdiffrast
        print(
            f"[mesh_cuda] atlas {atlas.shape[1]}x{atlas.shape[0]} "
            f"scale={draw['atlas_scale']} from {len(batches)} textures; "
            f"verts={len(draw['verts'])} faces={len(draw['faces'])} "
            f"load_s={t_load - t0:.1f} atlas_s={time.perf_counter() - t_load:.1f}",
            flush=True,
        )

        self._verts = torch.as_tensor(
            draw["verts"], dtype=torch.float32, device=self.device
        )
        self._faces = torch.as_tensor(
            draw["faces"], dtype=torch.int32, device=self.device
        )
        self._uvs = torch.as_tensor(draw["uvs"], dtype=torch.float32, device=self.device)
        atlas_f = atlas.astype(np.float32) / 255.0
        self._tex = torch.as_tensor(atlas_f, dtype=torch.float32, device=self.device)[
            None, ...
        ]
        self._glctx = dr.RasterizeCudaContext(device=self.device)
        triangles = self._verts[self._faces.long()]
        self._face_origins = triangles[:,0].contiguous()
        self._face_normals = torch.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0], dim=-1)
        self._closed = False
        self._verts_h = torch.cat([self._verts, torch.ones((len(self._verts), 1), device=self.device)], dim=1)

    def render(self, position, rotation, height, width, hfov, near=0.01, far=1000.0, depth=False):
        """Optical-center world pose; depth is positive camera Z in metres."""
        torch, dr = self._torch, self._dr
        if self._closed:
            raise RuntimeError("Renderer is closed")
        view, proj = habitat_view_proj_matrices(position, rotation, cam_h=0,
            hfov_deg=hfov, width=width, height=height, near=near, far=far)
        with torch.inference_mode():
            # Match Habitat's back-face culling, including viewpoints close to walls.
            front = (self._face_normals * (torch.as_tensor(position, device=self.device) - self._face_origins)).sum(-1) > 0
            faces = self._faces[front].contiguous()
            if faces.shape[0] == 0:
                if depth:
                    return np.zeros((height,width), dtype=np.float32)
                out = np.zeros((height,width,4), dtype=np.uint8)
                out[:,:,3] = 255
                return out
            camera = self._verts_h @ torch.as_tensor(view, device=self.device).T
            clip = (camera @ torch.as_tensor(proj, device=self.device).T)[None]
            rast, _ = dr.rasterize(self._glctx, clip, faces, resolution=[height,width])
            mask = rast[..., 3:4] > 0
            if depth:
                z, _ = dr.interpolate((-camera[:,2:3])[None].contiguous(), rast, faces)
                out = torch.where(mask, z, torch.zeros_like(z))[0,:,:,0]
            else:
                uv, _ = dr.interpolate(self._uvs[None], rast, faces)
                color = dr.texture(self._tex, uv, filter_mode="linear")
                rgb = torch.where(mask, color[...,:3], torch.zeros_like(color[...,:3]))
                rgb = (rgb[0]*255).clamp(0,255).to(torch.uint8)
                out = torch.cat([rgb, torch.full_like(rgb[:,:,:1],255)], dim=-1)
            return torch.flip(out, dims=(0,)).cpu().numpy()

    def close(self):
        self._closed = True
        self._glctx = None
        self._verts = None
        self._verts_h = None
        self._faces = None
        self._face_origins = None
        self._face_normals = None
        self._uvs = None
        self._tex = None
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()
