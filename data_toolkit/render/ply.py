"""Read standard binary little-endian Gaussian PLY without a repository dependency."""

from pathlib import Path
import numpy as np


def read_gaussian_ply(path):
    path = Path(path)
    fields = []
    count = None
    vertex = False
    types = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "int": "<i4",
        "uint": "<u4",
    }
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        if handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Only binary_little_endian Gaussian PLY is supported")
        for _ in range(1024):
            line = handle.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            tokens = line.decode("ascii").split()
            if not tokens:
                continue
            if tokens[0] == "end_header":
                offset = handle.tell()
                break
            if tokens[0] in ("comment", "obj_info"):
                continue
            if tokens[0] == "element":
                if tokens[1] == "vertex" and count is None:
                    count = int(tokens[2])
                    vertex = True
                elif count is None and int(tokens[2]) != 0:
                    raise ValueError("The PLY vertex payload must be the first element")
                else:
                    vertex = False
            elif tokens[0] == "property" and vertex:
                if len(tokens) != 3 or tokens[1] not in types:
                    raise ValueError("Unsupported vertex property in Gaussian PLY")
                fields.append((tokens[2], types[tokens[1]]))
        else:
            raise ValueError("PLY header is too long")
    if count is None or count <= 0:
        raise ValueError("PLY contains no Gaussian vertices")
    dtype = np.dtype(fields)
    if path.stat().st_size < offset + count * dtype.itemsize:
        raise ValueError("Truncated PLY vertex payload")
    required = ["x", "y", "z"] + [f"scale_{i}" for i in range(3)] + ["opacity"]
    required += [f"rot_{i}" for i in range(4)] + [f"f_dc_{i}" for i in range(3)]
    if not set(required).issubset(dtype.names):
        raise ValueError(
            "PLY lacks standard Gaussian position/scale/opacity/rotation/SH fields"
        )
    a = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=(count,))
    xyz = np.column_stack([a[k] for k in ["x", "y", "z"]]).astype(np.float32)
    quats = np.column_stack([a[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    scales = np.exp(
        np.column_stack([a[f"scale_{i}"] for i in range(3)]).astype(np.float32)
    )
    opacity = 1 / (1 + np.exp(-np.clip(a["opacity"].astype(np.float32), -80, 80)))
    rest = sorted(
        [k for k in dtype.names if k.startswith("f_rest_")], key=lambda x: int(x[7:])
    )
    if rest != [f"f_rest_{i}" for i in range(len(rest))] or len(rest) not in (
        0,
        9,
        24,
        45,
    ):
        raise ValueError(
            "Expected contiguous channel-major SH coefficients, degree 0 through 3"
        )
    n_basis = len(rest) // 3 + 1
    degree = int(np.sqrt(n_basis) - 1)
    sh = np.empty((count, n_basis, 3), np.float32)
    sh[:, 0] = np.column_stack([a[f"f_dc_{i}"] for i in range(3)])
    for channel in range(3):
        for basis in range(1, n_basis):
            sh[:, basis, channel] = a[f"f_rest_{channel * (n_basis - 1) + basis - 1}"]
    return xyz, quats, scales, opacity, sh, degree
