"""Conform rendering-only T junctions before fixed-point rasterization.

Positions are shared across UV/material seams, but each new corner's UV is
interpolated on its own source face. Original assets and navigation are untouched.
"""
from __future__ import annotations
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np

VERSION = 3
TOLERANCE_M = 1e-6


def _repair_pass(verts, faces, uvs, *, tolerance_m=TOLERANCE_M):
    from scipy.spatial import cKDTree
    started = time.perf_counter()
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    uvs = np.asarray(uvs, dtype=np.float32)
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('tolerance_m must be positive and finite')
    if not np.isfinite(verts).all() or not np.isfinite(uvs).all():
        raise ValueError('non-finite rendering geometry')
    positions, inverse = np.unique(verts, axis=0, return_inverse=True)
    gf = inverse[faces]
    edges = np.concatenate([gf[:, [0, 1]], gf[:, [1, 2]], gf[:, [2, 0]]])
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    # A conforming shared edge occurs twice. T junctions appear as unmatched
    # long and short edges; closed internal edges need no candidate search.
    boundary = unique_edges[counts == 1]
    tree = cKDTree(positions)
    split = {}
    for ia, ib in boundary:
        if ia == ib:
            continue
        a, b = positions[ia].astype(float), positions[ib].astype(float)
        delta = b-a
        length2 = float(delta@delta)
        if length2 <= (2*tolerance_m)**2:
            continue
        candidates = tree.query_ball_point((a+b)*.5, np.sqrt(length2)*.5+tolerance_m)
        if len(candidates) <= 2:
            continue
        ids = np.asarray(candidates, dtype=np.int64)
        points = positions[ids].astype(float)
        fraction = ((points-a)@delta)/length2
        distance = np.linalg.norm(points-(a+fraction[:, None]*delta), axis=1)
        interior = ((fraction*np.sqrt(length2) > tolerance_m)
                    & ((1-fraction)*np.sqrt(length2) > tolerance_m)
                    & (distance <= tolerance_m))
        ids, fraction = ids[interior], fraction[interior]
        if len(ids):
            order = np.argsort(fraction, kind='stable')
            # Coincident/near-coincident edge vertices must not make slivers.
            selected = []
            last = -np.inf
            for i in order:
                if (fraction[i]-last)*np.sqrt(length2) > tolerance_m:
                    selected.append((int(ids[i]), float(fraction[i])))
                    last = fraction[i]
            split[(int(ia), int(ib))] = selected
    out_v = [verts]
    out_uv = [uvs]
    extra_v, extra_uv, out_f = [], [], []
    changed = 0
    next_index = len(verts)
    for face, geom in zip(faces, gf):
        ring = []
        any_split = False
        for j in range(3):
            k = (j+1)%3
            ring.append(int(face[j]))
            ia, ib = int(geom[j]), int(geom[k])
            samples = split.get((min(ia, ib), max(ia, ib)), ())
            if not samples:
                continue
            any_split = True
            for point, fraction in (samples if ia < ib else reversed(samples)):
                t = fraction if ia < ib else 1-fraction
                extra_v.append(positions[point])
                extra_uv.append((1-t)*uvs[face[j]]+t*uvs[face[k]])
                ring.append(next_index)
                next_index += 1
        if not any_split:
            out_f.append(face.tolist())
            continue
        # A centroid fan preserves every subdivided boundary segment. A fan
        # from an original corner would silently span splits on its own edge.
        center = next_index
        extra_v.append(verts[face].mean(axis=0))
        extra_uv.append(uvs[face].mean(axis=0))
        next_index += 1
        out_f.extend([center, ring[j], ring[(j+1)%len(ring)]] for j in range(len(ring)))
        changed += 1
    if extra_v:
        out_v.append(np.asarray(extra_v, np.float32))
        out_uv.append(np.asarray(extra_uv, np.float32))
    result = (np.concatenate(out_v), np.asarray(out_f, np.int32).reshape(-1, 3), np.concatenate(out_uv))
    stats = dict(version=VERSION, tolerance_m=tolerance_m, input_faces=len(faces),
                 output_faces=len(result[1]), split_edges=len(split), split_faces=changed,
                 elapsed_s=time.perf_counter()-started)
    return (*result, stats)


def repair_t_junctions(verts, faces, uvs, *, tolerance_m=TOLERANCE_M):
    # Splitting can expose a previously matched edge or create a centroid
    # coincident with another edge. Continue until the mesh is conforming.
    started = time.perf_counter()
    input_faces = len(faces)
    passes = []
    for _ in range(8):
        verts, faces, uvs, stats = _repair_pass(verts, faces, uvs, tolerance_m=tolerance_m)
        passes.append(stats)
        if stats['split_edges'] == 0:
            return verts, faces, uvs, dict(
                version=VERSION, tolerance_m=tolerance_m, input_faces=input_faces,
                output_faces=len(faces), split_edges=sum(p['split_edges'] for p in passes),
                split_faces=sum(p['split_faces'] for p in passes), passes=len(passes),
                final_unresolved_edges=0, elapsed_s=time.perf_counter()-started)
    raise RuntimeError('Rendering mesh T-junction repair did not converge after 8 passes')


def cached_repair(verts, faces, uvs, cache_dir):
    digest = hashlib.sha256(f'tjunction:{VERSION}:{TOLERANCE_M}'.encode())
    for a in (verts, faces, uvs):
        digest.update(np.ascontiguousarray(a).tobytes())
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'tjunction_v{VERSION}_{digest.hexdigest()[:24]}.npz'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            try:
                with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as data:
                    result = tuple(data[k].copy() for k in ('verts', 'faces', 'uvs'))
                    stats = json.loads(str(data['stats']))
                return (*result, dict(stats, cache_hit=True))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass  # Interrupted/corrupt caches are rebuilt under the same lock.
        v, f, uv, stats = repair_t_junctions(verts, faces, uvs)
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        try:
            # Some remote filesystems mishandle ZIP's header seek-back writes.
            # Finish the archive in memory, then publish a sequential file.
            archive = io.BytesIO()
            np.savez_compressed(archive, verts=v, faces=f, uvs=uv, stats=json.dumps(stats))
            temporary.write_bytes(archive.getvalue())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return v, f, uv, dict(stats, cache_hit=False)
