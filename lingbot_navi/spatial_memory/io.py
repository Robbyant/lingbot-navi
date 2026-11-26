from pathlib import Path
import os, json, tempfile
import numpy as np

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, allow_nan=False, default=_json_default)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)



def write_prepared_memory(path, arrays, metadata, *, compressed=True):
    """Write the common model-input schema, preserving the caller's metadata."""
    payload = dict(arrays, metadata_json=np.asarray(json.dumps(metadata)))
    if compressed:
        write_npz(path, **payload)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **payload)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def write_evaluation_block(path, arrays, metadata, *, invalidate_metadata=False):
    """Publish an uncompressed evaluation block and its legacy metadata.

    Replace files instead of editing them so hardlinked snapshots remain intact.
    Metadata is published last; the pair is not a filesystem transaction.
    """
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_name('build_metadata.json')
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    meta_tmp = temporary.with_suffix('.json.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        digest = hashlib.sha256()
        with temporary.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        size = temporary.stat().st_size
        metadata.update(file_size_mb=size / 1000000.0,
                        artifact_integrity={path.name: dict(size_bytes=size, sha256=digest.hexdigest())})
        meta_tmp.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        if invalidate_metadata:
            metadata_path.unlink(missing_ok=True)
        os.replace(temporary, path)
        os.replace(meta_tmp, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
    return metadata
