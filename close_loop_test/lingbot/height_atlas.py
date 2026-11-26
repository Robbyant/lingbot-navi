"""Frozen multi-height static OCC judge. No renderer or geometry-height queries."""
import io
import math
import numpy as np
from .occupancy import OccupancyNavigation
import hashlib,json

def bbox_sha(boxes):
    return hashlib.sha256(json.dumps(np.asarray(boxes,dtype=float).tolist(),sort_keys=True,separators=(",",":")).encode()).hexdigest()

PROTOCOL = 'lingbot_occ_static_visible_height_0to3_v1'


def filter_occ_mask(mask, free, key):
    """Filter a verified source mask in memory, preserving all published files."""
    filtered = mask & free
    removed = int(mask.sum()) - int(filtered.sum())
    if removed:
        print(f'[occ-mask-filter] {key}: removed={removed} retained={int(filtered.sum())}', flush=True)
    if not filtered.any():
        raise ValueError('Visibility mask has no free OCC cells after filtering: ' + key)
    return filtered


class HeightAtlas:
    def __init__(self, release):
        self.release = release
        self.index = release.atlas
        self.cache = {}
        self.navigations = {}
        self.scenes = {}
        if self.index.get('protocol') != PROTOCOL or self.index.get('complete') is not True:
            raise ValueError('Incomplete or unsupported multi-height atlas')
        profile = self.index.get('visibility', {})
        expected = dict(camera_height_reference='OCC ground_height', hfov_deg=90., width=320,
                        height=240, min_pixels=8, min_fraction=.01, depth_source='mesh.ply',
                        abs_depth_tolerance_m=.15, rel_depth_tolerance=.02,
                        orientation='nearest annotated point or median annotated point',
                        horizontal_sampling='OCC cell centers', target_point_voxel_m=.02,
                        max_target_points=8192)
        heights = profile.get('sampled_heights_m', [])
        if (any(profile.get(k) != v for k, v in expected.items()) or len(heights) != 61
                or not np.allclose(heights, np.arange(61) * .05, rtol=0, atol=1e-9)):
            raise ValueError('Unsupported multi-height visibility profile')
        if self.index.get('target_count') != len(release.regions):
            raise ValueError('Multi-height atlas target count mismatch')

    def navigation(self, sid):
        if sid not in self.navigations:
            self.scenes[sid] = self.release.scene(sid)
            self.navigations[sid] = OccupancyNavigation.load(self.scenes[sid])
        return self.navigations[sid]

    def object(self, sid, oid, boxes):
        key = sid + '/' + oid
        if key not in self.cache:
            meta = self.release.regions[key]
            nav = self.navigation(sid)
            scene = self.scenes[sid]
            if (meta.get('protocol') != PROTOCOL or meta.get('complete') is not True
                    or meta.get('OCC_free_required') is not True or meta.get('NavMesh_required') is not False
                    or meta.get('coordinate_frame') != 'Habitat x,-z; source OCC row/col'
                    or meta.get('radius_m') != 1. or meta.get('fallback_radius_m') != .5
                    or any(meta.get(k) != scene[k] for k in
                           ('scene_transform_id', 'occupancy_sha256', 'occupancy_metadata_sha256'))):
                raise ValueError('Multi-height judge provenance/eligibility mismatch: ' + key)
            region = meta['target_annotation']['success_region']
            if (region.get('protocol') != PROTOCOL or region.get('radius_m') != 1.
                    or region.get('fallback_radius_m') != .5):
                raise ValueError('Multi-height target success protocol mismatch: ' + key)
            row, col = meta.get('row_offset'), meta.get('col_offset')
            if type(row) is not int or type(col) is not int or min(row, col) < 0:
                raise ValueError('Invalid OCC mask offsets: ' + key)
            folder = self.release.atlas_root / self.index['objects'][key]['directory']
            relative = str((folder / 'visibility.npz').relative_to(self.release.root))
            path = self.release.path(relative, verify=True)
            raw = path.read_bytes()
            import hashlib
            if hashlib.sha256(raw).hexdigest() != meta['npz_sha256']:
                raise ValueError('Visibility mask changed while loading: ' + key)
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                visible = data['visible'].copy()
            if (visible.dtype != np.bool_ or visible.ndim != 2 or list(visible.shape) != meta['shape']
                    or int(visible.sum()) != meta['visible_cells']
                    or row + visible.shape[0] > nav.free.shape[0] or col + visible.shape[1] > nav.free.shape[1]
                    or not math.isclose(meta['resolution_m'], nav.resolution, rel_tol=0, abs_tol=1e-7)
                    or not np.allclose(meta['origin'], nav.origin + np.array([col, row])*nav.resolution,
                                       rtol=0, atol=1e-5)):
                raise ValueError('Visibility mask does not align with OCC: ' + key)
            visible = filter_occ_mask(visible, nav.free[row:row+visible.shape[0], col:col+visible.shape[1]], key)
            visible.setflags(write=False)
            self.cache[key] = meta, visible
        meta, visible = self.cache[key]
        # The enclosing region.json is SHA-pinned, including its target geometry.
        if bbox_sha(boxes) != bbox_sha(meta['target_annotation']['success_region']['bbox_union']):
            raise ValueError('Target geometry differs from frozen multi-height region: ' + key)
        return meta, visible

    def contains(self, positions, sid, oid, boxes):
        """Published mask is the final success set, including radius/fallback.

        Index with the original OCC grid, not the rounded cropped-mask origin.
        Cells are half-open; body height does not re-filter the height-union mask.
        """
        meta, visible = self.object(sid, oid, boxes)
        points = np.asarray(positions, dtype=float).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError('Invalid multi-height visibility query position')
        nav = self.navigation(sid)
        xy = points[:, [0, 2]] * [1., -1.]
        grid = np.floor((xy - nav.origin) / nav.resolution)
        local = grid - [meta['col_offset'], meta['row_offset']]
        valid = ((local[:, 0] >= 0) & (local[:, 1] >= 0)
                 & (local[:, 0] < visible.shape[1]) & (local[:, 1] < visible.shape[0]))
        ids = np.flatnonzero(valid)
        result = np.zeros(len(points), dtype=bool)
        cells = local[ids].astype(int)
        result[ids] = visible[cells[:, 1], cells[:, 0]]
        return result

    def goal_cells(self, sid, oid, boxes, navigation):
        meta, visible = self.object(sid, oid, boxes)
        expected = self.navigation(sid)
        if (navigation.provenance != expected.provenance
                or navigation.resolution != expected.resolution
                or not np.array_equal(navigation.origin, expected.origin)
                or not np.array_equal(navigation.free, expected.free)):
            raise ValueError('Goal distance navigation differs from the pinned OCC judge')
        rows, cols = np.nonzero(visible)
        return set(zip((cols + meta['col_offset']).tolist(), (rows + meta['row_offset']).tolist()))
