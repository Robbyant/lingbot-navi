"""Read-only static multi-height release-v1 reader, with pinned assets."""
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

HEIGHT_ATLAS_PROTOCOL = 'lingbot_occ_static_visible_height_0to3_v1'

FORMAT = 'release-v1'
TRANSFORM = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseV1:
    def __init__(self, root, expected_sha256=None):
        self.root = Path(root).resolve(strict=True)
        raw = (self.root / 'manifest.json').read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and self.digest != expected_sha256:
            raise ValueError('release-v1 manifest SHA256 changed')
        self.manifest = json.loads(raw)
        if self.manifest.get('schema_version') != FORMAT or self.manifest.get('dataset_type') != 'evaluation':
            raise ValueError('Expected a release-v1 evaluation dataset')
        self.files = {}
        self._verified = {}
        for row in self.manifest['files']:
            name = row['path']
            self._relative(name)
            if name in self.files or not isinstance(row['bytes'], int) or row['bytes'] < 0:
                raise ValueError('Invalid or duplicate manifest entry: ' + name)
            if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
                raise ValueError('Invalid SHA256: ' + name)
            self.files[name] = row
        self.scene_ids = sorted(self.manifest['scene_ids'])
        if not self.scene_ids or len(set(self.scene_ids)) != len(self.scene_ids):
            raise ValueError('Empty or duplicate scene IDs')
        for sid in self.scene_ids:
            if len(PurePosixPath(sid).parts) != 1:
                raise ValueError('Invalid scene ID')
            self._relative(sid)
        self.atlas_relative = self.manifest['static_annotations_atlas']
        self.atlas = self.read_json(self.atlas_relative)
        if self.atlas.get("protocol") != HEIGHT_ATLAS_PROTOCOL:
            raise ValueError("Only static multi-height OCC release-v1 is supported")
        self.atlas_root = self.path(self.atlas_relative).parent
        self.regions = {}
        for key, row in self.atlas['objects'].items():
            sid, oid = key.split('/')
            self._relative(oid)
            if sid not in self.scene_ids:
                raise ValueError('Atlas references unknown scene: ' + key)
            self._relative(row['directory'])
            folder = self.atlas_root / row['directory']
            region_rel = str((folder / 'region.json').relative_to(self.root))
            mask_rel = str((folder / 'visibility.npz').relative_to(self.root))
            region = self.read_json(region_rel)
            self.path(mask_rel)
            if (self.files[region_rel]['sha256'] != row['region_sha256']
                    or self.files[mask_rel]['sha256'] != row['npz_sha256']
                    or region['npz_sha256'] != row['npz_sha256']):
                raise ValueError('Atlas/manifest checksum disagreement: ' + key)
            annotation = region['target_annotation']
            if region['scene'] != sid or region['object_id'] != oid or annotation['target_id'] != oid:
                raise ValueError('Atlas target identity mismatch: ' + key)
            boxes = annotation['success_region']['bbox_union']
            if not boxes or any(len(b) != 6 or not all(math.isfinite(v) for v in b)
                                or any(b[i] > b[i+3] for i in range(3)) for b in boxes):
                raise ValueError('Invalid target boxes: ' + key)
            self.regions[key] = region

    @staticmethod
    def _relative(name):
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name or name == '.':
            raise ValueError('Expected a normalized dataset-relative path: ' + str(name))

    def assert_manifest(self):
        if digest_file(self.root / 'manifest.json') != self.digest:
            raise ValueError('release-v1 manifest changed after loading')

    def path(self, relative, verify=False):
        self._relative(relative)
        if relative not in self.files:
            raise ValueError('Unlisted release asset: ' + relative)
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or path.is_symlink() or any(
                p.is_symlink() for p in path.parents if p != self.root and self.root in p.parents):
            raise ValueError('Release asset escapes root or uses a symlink: ' + relative)
        stat = path.stat()
        if not path.is_file():
            raise ValueError('Release asset is not a file: ' + relative)
        if stat.st_size != self.files[relative]['bytes']:
            raise ValueError('Release asset size mismatch: ' + relative)
        stamp = stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        if verify and self._verified.get(relative) != stamp:
            if digest_file(path) != self.files[relative]['sha256']:
                raise ValueError('Release asset SHA256 mismatch: ' + relative)
            after = path.stat()
            if stamp != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Release asset changed while hashing: ' + relative)
            self._verified[relative] = stamp
        return path

    def read_json(self, relative):
        # Hash exactly the bytes parsed, avoiding a second unverified read.
        path = self.path(relative)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.files[relative]['sha256']:
            raise ValueError('Release metadata SHA256 mismatch: ' + relative)
        return json.loads(raw)

    def task_files(self, split='val_seen'):
        self.assert_manifest()
        if split != 'val_seen':
            raise ValueError('release-v1 static episodes only support val_seen')
        return [self.path('tasks/static/' + sid + '.json') for sid in self.scene_ids
                if 'tasks/static/' + sid + '.json' in self.files]

    def scene(self, sid, verify_assets=False):
        if sid not in self.scene_ids:
            raise ValueError('Unknown release scene: ' + sid)
        source = self.read_json('scenes/' + sid + '/scene.json')
        coordinates = source['coordinates']
        if (source['scene_id'] != sid or coordinates['source_to_habitat'] != TRANSFORM
                or coordinates['source_unit_to_meter'] != 1. or coordinates['up_axis'] != 'Y'
                or coordinates['quaternion_order'] != 'xyzw' or coordinates['units'] != 'meter'):
            raise ValueError('Unsupported scene coordinates: ' + sid)
        assets = source['assets']
        paths = {name: self.path(assets[name], verify_assets) for name in
                 ('splat', 'depth_mesh', 'occupancy_array', 'occupancy_metadata')}
        metadata = self.read_json(assets['occupancy_metadata'])
        if paths['occupancy_array'].name != 'occupancy.npz' or paths['occupancy_metadata'] != paths['occupancy_array'].with_suffix('.json'):
            raise ValueError('OCC assets must be a paired occupancy.npz/occupancy.json')
        objects = {key.split('/')[1]: {'region': copy.deepcopy(value['target_annotation']['success_region'])}
                   for key, value in self.regions.items() if key.startswith(sid + '/')}
        return dict(scene_id=sid, splat_path=str(paths['splat']), source_mesh=str(paths['depth_mesh']),
                    ground_to_habitat=coordinates['source_to_habitat'], scene_unit_to_meter=1.,
                    scene_transform_id=metadata['scene_transform_id'], objects=objects,
                    occupancy=str(paths['occupancy_array'].parent),
                    occupancy_sha256=self.files[assets['occupancy_array']]['sha256'],
                    occupancy_metadata_sha256=self.files[assets['occupancy_metadata']]['sha256'],
                    occupancy_index_sha256=self.digest, navigation_geometry='raw_occ_supercover_v1',
                    **source['navigation'])

    def content(self, path):
        self.assert_manifest()
        path = Path(path).resolve(strict=True)
        if (path.parent != self.root / 'tasks/static' or path.stem not in self.scene_ids
                or path.suffix != '.json' or str(path.relative_to(self.root)) not in self.files):
            raise ValueError('Not a release-v1 static task file: ' + str(path))
        sid = path.stem
        data = self.read_json(str(path.relative_to(self.root)))
        if data['scene_id'] != sid:
            raise ValueError('Task file scene mismatch')
        goals = {}
        for key, region in self.regions.items():
            if not key.startswith(sid + '/'):
                continue
            annotation = region['target_annotation']
            oid = annotation['target_id']
            items = annotation['goal_observations']
            goal = copy.deepcopy(items[0]) if items else {}
            goal.update(object_id=oid, object_category=annotation['category'],
                        lang_desc=annotation['description_en'], lingbo_target_region=copy.deepcopy(annotation['success_region']))
            goal.pop('view_points', None)  # OCC scoring uses the exact region, never legacy viewpoints.
            goals.setdefault(sid + '_' + annotation['category'], []).append(goal)
        episodes, seen = [], set()
        for item in data['episodes']:
            eid = str(item['task_id'])
            if item['scene_id'] != sid or eid in seen or not item['goals']:
                raise ValueError('Invalid or duplicate episode: ' + eid)
            seen.add(eid)
            position, rotation = item['start_position'], item['start_rotation_xyzw']
            if (len(position) != 3 or len(rotation) != 4 or not all(math.isfinite(v) for v in position + rotation)
                    or not math.isclose(sum(v*v for v in rotation), 1., abs_tol=1e-5)
                    or not math.isfinite(item['start_sensor_pitch_rad'])
                    or item['start_sensor_pitch_protocol'] != 'source_pitch_local_x_body_yaw_only_v1'):
                raise ValueError('Invalid body start pose: ' + eid)
            from .occupancy import OccupancyNavigation, snap_start_to_free
            if not hasattr(self, '_start_navigation'):
                self._start_navigation = {}
            if sid not in self._start_navigation:
                self._start_navigation[sid] = OccupancyNavigation.load(self.scene(sid))
            position, start_adjustment = snap_start_to_free(position, self._start_navigation[sid], sid+'/'+eid)
            tasks, instructions = [], []
            for number, goal in enumerate(item['goals'], 1):
                key = sid + '/' + goal['target_id']
                if key not in self.regions or goal.get('step', number) != number:
                    raise ValueError('Invalid task target or order: ' + eid)
                annotation = self.regions[key]['target_annotation']
                if goal['category'] != annotation['category'] or goal['instruction_mode'] not in ('description', 'object'):
                    raise ValueError('Unsupported task category/modality: ' + eid)
                if goal['arrival_radius_m'] != 1. or annotation['success_region']['radius_m'] != 1.:
                    raise ValueError('release-v1 adapter currently supports the frozen 1m protocol only')
                text = goal['instruction_en']
                if not isinstance(text, str) or not text.strip().rstrip('.').strip():
                    raise ValueError('Missing task instruction: ' + eid)
                tasks.append([goal['category'], goal['instruction_mode'], goal['target_id']])
                instructions.append(text)
            episodes.append(dict(episode_id=eid, scene_id=sid, start_position=copy.deepcopy(position),
                start_rotation=copy.deepcopy(rotation), tasks=tasks, task_instructions=instructions,
                lingbo=dict(scene_id=sid, evaluation_ready=True, dataset_format=FORMAT,
                            start_sensor_pitch_rad=item['start_sensor_pitch_rad'], start_adjustment=start_adjustment)))
        return dict(episodes=episodes, goals=goals)

    def make_atlas(self):
        from .height_atlas import HeightAtlas
        return HeightAtlas(self)
