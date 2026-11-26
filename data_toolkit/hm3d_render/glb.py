"""HM3D Basis repacking helpers adapted from navigation_wm/scripts/visualize_hm3d_scene.py."""
from __future__ import annotations
import os,shutil,tempfile,subprocess,struct,json,copy
from pathlib import Path

def resolve_basisu_path(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        path = cli_path.expanduser().resolve()
        return path if path.is_file() else None

    env_path = os.environ.get("BASISU_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path

    which_path = shutil.which("basisu")
    if which_path:
        return Path(which_path).resolve()

    return None

def decode_basis_image(basis_bytes: bytes, basisu_path: Path) -> bytes:
    """Decode an embedded Basis texture payload to PNG bytes."""

    with tempfile.TemporaryDirectory(prefix="hm3d_basis_") as tmp:
        work_dir = Path(tmp)
        basis_file = work_dir / "texture.basis"
        basis_file.write_bytes(basis_bytes)
        result = subprocess.run(
            [str(basisu_path), "-unpack", "-file", str(basis_file)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"basisu failed ({result.returncode}): {result.stderr or result.stdout}"
            )

        candidates = sorted(work_dir.glob("*_unpacked_rgb_RGBA32_*_0000.png"))
        if not candidates:
            candidates = sorted(work_dir.glob("*_unpacked_rgba_RGBA32_*_0000.png"))
        if not candidates:
            raise RuntimeError("basisu produced no RGBA32 PNG output")

        return candidates[0].read_bytes()

def write_glb(meta: dict, bin_chunk: bytes, out_path: Path) -> None:
    json_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_chunk = bin_chunk + b"\x00" * ((4 - len(bin_chunk) % 4) % 4)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + f".{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as f:
            f.write(b"glTF")
            f.write(struct.pack("<II", 2, total_length))
            f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
            f.write(json_bytes)
            f.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
            f.write(bin_chunk)
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def repack_basis_glb_with_png_textures(
    basis_glb_path: Path,
    out_path: Path,
    basisu_path: Path,
) -> None:
    """Build a browser-loadable GLB by replacing Basis textures with PNG."""

    meta, bin_data = _parse_glb(basis_glb_path)
    images = meta.get("images", [])
    buffer_views = meta.get("bufferViews", [])
    if not images:
        raise ValueError(f"No images found in {basis_glb_path}")

    png_by_image: dict[int, bytes] = {}
    for image_idx, image in enumerate(images):
        mime_type = image.get("mimeType", "")
        if mime_type == "image/x-basis":
            print(f"  Decoding Basis texture {image_idx + 1}/{len(images)} …", flush=True)
        buffer_view_idx = image["bufferView"]
        buffer_view = buffer_views[buffer_view_idx]
        start = buffer_view.get("byteOffset", 0)
        end = start + buffer_view["byteLength"]
        payload = bin_data[start:end]

        if mime_type == "image/x-basis":
            png_by_image[image_idx] = decode_basis_image(payload, basisu_path)
        elif mime_type == "image/png":
            png_by_image[image_idx] = payload
        else:
            raise ValueError(f"Unsupported image mime type: {mime_type}")

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

    for image_idx, image in enumerate(new_meta.get("images", [])):
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
            ext
            for ext in new_meta["extensionsUsed"]
            if ext != "GOOGLE_texture_basis"
        ]
        if not new_meta["extensionsUsed"]:
            del new_meta["extensionsUsed"]

    if "extensionsRequired" in new_meta:
        new_meta["extensionsRequired"] = [
            ext
            for ext in new_meta["extensionsRequired"]
            if ext != "GOOGLE_texture_basis"
        ]
        if not new_meta["extensionsRequired"]:
            del new_meta["extensionsRequired"]

    write_glb(new_meta, bytes(new_bin), out_path)

def prepare_photo_textured_glb(
    basis_glb_path: Path,
    cache_dir: Path,
    basisu_path: Path | None,
    *,
    force: bool = False,
) -> Path | None:
    """Decode ``basis.glb`` Basis textures to PNG and cache ``photo_textured.glb``."""

    if basisu_path is None or not basisu_path.is_file():
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "photo_textured.glb"
    if out_path.is_file() and not force:
        return out_path

    print(f"Decoding Basis textures with {basisu_path} …")
    repack_basis_glb_with_png_textures(basis_glb_path, out_path, basisu_path)
    return out_path

def _parse_glb(glb_path: Path) -> tuple[dict, bytes]:
    data = glb_path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"Invalid GLB: {glb_path}")

    chunk0_len = struct.unpack_from("<I", data, 12)[0]
    json_chunk = data[20 : 20 + chunk0_len].decode("utf-8").rstrip("\x00")
    meta = json.loads(json_chunk)

    chunk1_offset = 20 + chunk0_len
    chunk1_len = struct.unpack_from("<I", data, chunk1_offset)[0]
    bin_data = data[chunk1_offset + 8 : chunk1_offset + 8 + chunk1_len]
    return meta, bin_data
