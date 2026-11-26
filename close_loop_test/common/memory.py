"""Online segments, LingBot-Navi queue policy, and release memory preparation."""
from lingbot_navi.spatial_memory.features import Qwen3_5ViTExtractor
from lingbot_navi.spatial_memory.preparation import prepare_memory, _arrays, _rotation_quaternion, rotation_matrix, quaternion_product
from lingbot_navi.spatial_memory.sampling import first_voxel_indices
from lingbot_navi.spatial_memory.settings import PreparationSettings
from lingbot_navi.spatial_memory.io import write_prepared_memory
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .policy.online_memory import OnlineMemoryBuilder
from .policy.spatial_memory_queue import enqueue_memory
from .policy.sampling import compact_memory

class OnlineMemory:
    def __init__(self, base_model, device, output, metadata, max_voxels=500000, scene='scene', map1=None):
        self.output, self.scene = Path(output), scene
        self.extractor = Qwen3_5ViTExtractor(base_model, device=device, resize_mode="aspect" if map1 is not None else "fixed")
        aug = metadata['joint_protocol'].get('block_augmentation', {})
        self.preparation = PreparationSettings(voxel_size=metadata['voxel_size'],
            rotation_degrees=tuple(aug.get('rotation_degrees', (0, 0, 180))) if aug.get('augment', True) else (0,0,0), seed=aug.get('seed',42))
        self.max_voxels = max_voxels
        self.paths = []
        self.builder = None
        self.records = []
        self.cache_key = None
        self.map1 = None
        if map1 is not None:
            from .map1.memory import Map1Memory
            self.map1 = Map1Memory(self, map1, device)

    def observe(self, bundle, frame_id=None):
        if self.map1 is not None:
            return self.map1.observe(bundle, frame_id)
        with np.load(bundle, allow_pickle=False) as a:
            rgb, depth, intrinsic, pose = (a[k] for k in ('rgb','depth','intrinsic','camera_pose'))
            agent = a['agent_position'].tolist()
        if self.builder is None:
            self.builder = OnlineMemoryBuilder(self.extractor, scene=self.scene,
                width=rgb.shape[1],height=rgb.shape[0],hfov=90,cam_height=1.5,max_voxels=self.max_voxels)
        kept = self.builder.push_metric_frame(rgb,depth,intrinsic,pose)
        self.records.append(agent)
        return dict(frames=self.builder.frames, processed=self.builder.processed, accepted=kept)

    def finish(self, segment, phase):
        if self.map1 is not None:
            return self.map1.finish(segment, phase)
        builder, records = self.builder, self.records
        self.builder, self.records = None, []
        if builder is None or builder.frames < 2:
            return dict(state='skipped', skip_reason='insufficient_frames',source_phase=phase,paths=self.paths)
        directory = self.output/'memories'/segment
        builder.finish(directory,self.preparation.voxel_size)
        path, merge = enqueue_memory(directory/'spatial_mem.npz',self.paths[-1] if self.paths else None,
                                      dict(n=len(records),agent_xyz=records))
        if merge: compact_memory(path,self.preparation.voxel_size)
        if str(path) not in self.paths:self.paths.append(str(path))
        self.cache_key=None
        return dict(state='done',path=str(path),queue_merge=merge,source_phase=phase,paths=self.paths)

    def prepare(self, paths):
        key=tuple((p,Path(p).stat().st_mtime_ns,Path(p).stat().st_size) for p in paths)
        if key==self.cache_key:return self.prepared,self.frames,self.local_xyz
        ids=['segment_%02d'%(i+1) for i in range(len(paths))]
        arrays, transforms=prepare_memory(paths,ids,'hm3d:'+self.scene,preparation=self.preparation)
        metadata=dict(format='lingbot-navi-memory-v1',block_ids=ids,feature_dim=2560,
            coordinate_frame='independent_block_local_meters',preparation=asdict(self.preparation))
        self.prepared=self.output/'memory.npz'
        write_prepared_memory(self.prepared, arrays, metadata, compressed=False)
        self.frames=[]
        for path,block_id in zip(paths,ids):
            xyz, _ = _arrays(path, 2560)
            if self.preparation.voxel_size:
                xyz = xyz[first_voxel_indices(xyz, self.preparation.voxel_size)]
            center = xyz.mean(axis=0, dtype=np.float64)
            q = _rotation_quaternion(self.preparation.seed, 'hm3d:'+self.scene,
                                     block_id, self.preparation.rotation_degrees)
            world_q = quaternion_product([0.,0.,0.,1.], q * [-1,-1,-1,1])
            self.frames.append(dict(name=Path(path).parent.name,memory_path=path,
                translation=center.tolist(),rotation_matrix=rotation_matrix(world_q).tolist(),scale=1.))
        bounds=arrays['block_cu_seqlens'];xyz=arrays['memory_xyz']
        self.local_xyz=[xyz[a:b] for a,b in zip(bounds[:-1],bounds[1:])]
        self.cache_key=key
        return self.prepared,self.frames,self.local_xyz

    def close(self):
        if self.map1 is not None:
            self.map1.close()
