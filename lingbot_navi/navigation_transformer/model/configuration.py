from dataclasses import dataclass, fields
from ...spatial_encoding import SpatialRoPESpec


@dataclass(frozen=True)
class ModelConfig:
    point_head_enabled: bool = False
    point_coordinate_frame: str = "first_block"
    spatial_feature_dim: int = 2560
    qwen_hidden_dim: int = 2560
    qwen_num_layers: int = 32
    qwen_num_heads: int = 16
    qwen_num_key_value_heads: int = 4
    qwen_head_dim: int = 256
    full_attention_layers: tuple = tuple(range(3, 32, 4))
    # Route-specific ranks/scales used by both conversion and unmerged inference.
    lora_routing_mode: str = "separate"
    spatial_tuning_mode: str = "full"
    lora_rank: int = 16
    lora_alpha: int = 32
    trajectory_lora_rank: int = 32
    trajectory_lora_alpha: int = 64
    enforce_supported_shape: bool = True
    spatial_rope_min_one_rad_distance_m: float = 0.01
    spatial_rope_max_one_rad_distance_m: float = 1000.0

    @classmethod
    def from_dict(cls, data):
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if "full_attention_layers" in values:
            values["full_attention_layers"] = tuple(values["full_attention_layers"])
        return cls(**values)

    def __post_init__(self):
        if (
            self.spatial_tuning_mode not in ("full", "lora")
            or self.lora_routing_mode != "separate"
        ):
            raise ValueError("Unsupported spatial / LoRA parameter layout")
        for key in (
            "lora_rank",
            "lora_alpha",
            "trajectory_lora_rank",
            "trajectory_lora_alpha",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(key + " must be a positive integer")
        if self.spatial_feature_dim != self.qwen_hidden_dim:
            raise ValueError("Memory feature width must equal language hidden width")
        if self.point_coordinate_frame not in ("first_block", "block_local"):
            raise ValueError("Unsupported point coordinate frame")
        SpatialRoPESpec(
            self.spatial_rope_min_one_rad_distance_m,
            self.spatial_rope_max_one_rad_distance_m,
        )


@dataclass(frozen=True)
class BlockConfig:
    num_interior_points: int = 78
    hidden_dim: int = 384
    position_gate_init: float = 0.1
    spatial_full: bool = True
    learn_block_scale: bool = False
