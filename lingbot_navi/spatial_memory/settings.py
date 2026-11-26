from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PreparationSettings:
    """Final model-input convention, separate from feature-fusion voxels."""

    voxel_size: float = 0.32
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 180.0)
    seed: int = 42

    def validate(self):
        if not np.isfinite(self.voxel_size) or self.voxel_size < 0:
            raise ValueError("voxel_size must be finite and nonnegative")
        angles = np.asarray(self.rotation_degrees)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("rotation_degrees must contain three finite angles")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
