"""Raw OCC execution and distance arithmetic."""
from collections import OrderedDict
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def sha256_file(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()

def checked_json(path, expected):
    data = Path(path).read_bytes()
    if not expected or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'OCC provenance SHA256 mismatch: {path}')
    return json.loads(data)

class OccupancyNavigation:
    def __init__(self, state, metadata, provenance=None):
        self.resolution = float(metadata['resolution'])
        bounds = np.asarray(metadata['actual_bounds'], float)
        if (state.ndim != 2 or state.shape != (metadata['rows'], metadata['columns'])
                or not math.isfinite(self.resolution) or self.resolution <= 0
                or bounds.shape != (4,) or not np.isfinite(bounds).all()
                or not np.allclose(bounds[2:] - bounds[:2], np.array(state.shape[::-1]) * self.resolution,
                                   rtol=0, atol=self.resolution * .02)):
            raise ValueError('Invalid OCC dimensions, extent, or resolution')
        self.origin = bounds[:2]
        self.free = state == 1
        self.free.setflags(write=False)
        self.labels, _ = ndimage.label(self.free)
        self.provenance = dict(provenance or {})
        self._graph = None
        self._fields = OrderedDict()

    @classmethod
    def load(cls, scene):
        root = Path(scene['occupancy'])
        meta = checked_json(root / 'occupancy.json', scene['occupancy_metadata_sha256'])
        if (meta['scene_transform_id'] != scene['scene_transform_id'] or meta['up_axis'] != 'z'
                or meta['units'] not in ('m', 'meter', 'metre', 'meters', 'metres')):
            raise ValueError(f'OCC coordinate frame differs from scene: {scene["scene_id"]}')
        npz = root / 'occupancy.npz'
        if sha256_file(npz) != scene['occupancy_sha256']:
            raise ValueError(f'OCC NPZ SHA256 mismatch: {npz}')
        with np.load(npz, allow_pickle=False) as data:
            state = data['state']
        keys = ('occupancy', 'occupancy_sha256', 'occupancy_metadata_sha256',
                'occupancy_index_sha256', 'scene_transform_id', 'navigation_geometry')
        return cls(state, meta, {key: scene[key] for key in keys})

    def cell(self, xy):
        xy = np.asarray(xy, float)
        x, y = np.floor((xy - self.origin) / self.resolution).astype(int)
        return int(x), int(y)

    def center(self, cells):
        return self.origin + (np.asarray(cells) + .5) * self.resolution

    def _is_free(self, x, y):
        return 0 <= y < self.free.shape[0] and 0 <= x < self.free.shape[1] and self.free[y, x]

    def component(self, xy):
        if not np.isfinite(xy).all():
            return 0
        x, y = self.cell(xy)
        return int(self.labels[y, x]) if self._is_free(x, y) else 0

    def first_blocked_cell(self, start, end):
        """Return the first nonfree source cell touched by a horizontal segment."""
        a, b = np.asarray(start, float), np.asarray(end, float)
        if not np.isfinite([a, b]).all():
            raise ValueError('Nonfinite OCC motion')
        x, y = self.cell(a)
        end_x, end_y = self.cell(b)
        if not self._is_free(x, y):
            return (x, y)
        dx, dy = b - a
        sx, sy = int(np.sign(dx)), int(np.sign(dy))
        tx = (self.origin[0] + (x + (sx > 0)) * self.resolution - a[0]) / dx if sx else math.inf
        ty = (self.origin[1] + (y + (sy > 0)) * self.resolution - a[1]) / dy if sy else math.inf
        stepx = self.resolution / abs(dx) if sx else math.inf
        stepy = self.resolution / abs(dy) if sy else math.inf
        while (x, y) != (end_x, end_y):
            # An endpoint on a grid boundary belongs to its floor-indexed cell.
            next_x = tx if x != end_x else math.inf
            next_y = ty if y != end_y else math.inf
            if next_x + 1e-10 < next_y:
                x += sx
                tx += stepx
            elif next_y + 1e-10 < next_x:
                y += sy
                ty += stepy
            else:
                for side in ((x + sx, y), (x, y + sy)):
                    if not self._is_free(*side):
                        return side
                x, y = x + sx, y + sy
                tx, ty = tx + stepx, ty + stepy
            if not self._is_free(x, y):
                return (x, y)
        return None

    def segment_allowed(self, start, end):
        return self.first_blocked_cell(start, end) is None

    def project(self, start, target, max_distance):
        target = np.asarray(target, float)
        component = self.component(start)
        if not component or not np.isfinite(target).all():
            return None
        if self.component(target) == component:
            return target.copy()
        ys, xs = np.nonzero(self.labels == component)
        centers = self.center(np.column_stack((xs, ys)))
        distances = np.linalg.norm(centers - target, axis=1)
        nearest = int(distances.argmin())
        return centers[nearest] if distances[nearest] <= max_distance else None

    def _build_graph(self):
        ys, xs = np.nonzero(self.free)
        self._ids = np.full(self.free.shape, -1, np.int32)
        self._ids[ys, xs] = np.arange(len(xs))
        self._cells = np.column_stack((xs, ys))
        height, width = self.free.shape
        sources, ends, weights = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            x0, x1 = max(0, -dx), min(width, width - dx)
            y0, y1 = max(0, -dy), min(height, height - dy)
            a = (slice(y0, y1), slice(x0, x1))
            b = (slice(y0+dy, y1+dy), slice(x0+dx, x1+dx))
            valid = self.free[a] & self.free[b]
            if dx and dy:
                valid &= self.free[a[0], b[1]] & self.free[b[0], a[1]]
            start, end = self._ids[a][valid], self._ids[b][valid]
            sources.extend((start, end))
            ends.extend((end, start))
            weights.extend([np.full(len(start), self.resolution * math.hypot(dx, dy))] * 2)
        self._graph = csr_matrix((np.concatenate(weights), (np.concatenate(sources), np.concatenate(ends))),
                                 shape=(len(xs), len(xs)))

    def field(self, goals):
        if self._graph is None:
            self._build_graph()
        ids = np.unique([self._ids[y, x] for x, y in goals if self._is_free(x, y)])
        if not len(ids):
            return None
        key = ids.tobytes()
        if key not in self._fields:
            distances, parents, _ = dijkstra(self._graph, directed=True, indices=ids,
                                             min_only=True, return_predecessors=True)
            self._fields[key] = distances, parents
            if len(self._fields) > 8:
                self._fields.popitem(last=False)
        self._fields.move_to_end(key)
        return self._fields[key]

    def distance(self, xy, field):
        if field is None or not self.component(xy):
            return None
        x, y = self.cell(xy)
        d = float(field[0][self._ids[y, x]])
        return d if math.isfinite(d) else None

    def route(self, start, target, field):
        if self.distance(start, field) is None:
            return []
        x, y = self.cell(start)
        node = self._ids[y, x]
        points = [np.asarray(start).tolist()]
        while node >= 0:
            points.append(self.center(self._cells[node]).tolist())
            node = field[1][node]
        points.append(np.asarray(target).tolist())
        return points

def install_occupancy_filter(sim, navigation):
    import magnum as mn

    def move(start, requested):
        a, b = np.asarray(start, float), np.asarray(requested, float)
        cell = navigation.first_blocked_cell(a[[0, 2]] * [1, -1], b[[0, 2]] * [1, -1])
        sim.occupancy_last_block = None
        if cell is not None:
            sim.occupancy_last_block = dict(cell=list(cell), start=a.tolist(), requested=b.tolist(),
                                            reason='occ_nonfree_cell')
            return mn.Vector3(a)
        b[1] = a[1]
        return mn.Vector3(b)

    sim.occupancy_navigation = navigation
    sim.occupancy_last_block = None
    sim.agents[0].controls.move_filter_fn = move

def snap_start_to_free(position, navigation, identity):
    """Return a copied Habitat start and optional audit; never write release data."""
    original = np.asarray(position, dtype=float)
    if original.shape != (3,) or not np.isfinite(original).all():
        raise ValueError('Invalid body start: ' + identity)
    xy = original[[0, 2]] * [1., -1.]
    if navigation.component(xy):
        return original.tolist(), None
    rows, cols = np.nonzero(navigation.free)
    if not len(rows):
        raise ValueError('No free OCC cell for body start: ' + identity)
    centers = navigation.center(np.column_stack((cols, rows)))
    distances = np.sum((centers - xy)**2, axis=1)
    index = int(np.argmin(distances))
    adjusted = original.copy()
    adjusted[0], adjusted[2] = centers[index] * [1., -1.]
    audit = dict(original_position=original.tolist(), adjusted_position=adjusted.tolist(),
                 distance_m=float(np.sqrt(distances[index])),
                 cell=[int(cols[index]), int(rows[index])], method='nearest_free_occ_cell_center')
    print('[occ-start-adjust] ' + identity + ': ' + __import__('json').dumps(audit), flush=True)
    return adjusted.tolist(), audit
