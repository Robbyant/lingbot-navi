"""Fusion over already-normalized frames; adapters own deduplication and coordinates."""
from .fusion import BestObservationAccumulator

class MemoryBuilder:
    def __init__(self, **settings):
        self.accumulator = BestObservationAccumulator(**settings)

    @property
    def num_voxels(self):
        return self.accumulator.num_voxels

    def add_observation(self, observation, *, frame_name, frame_pose):
        self.accumulator.update(observation, frame_name=frame_name, frame_pose=frame_pose)

    def finalize(self):
        return self.accumulator.finalize()
