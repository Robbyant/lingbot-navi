"""Independent inference-time LoRA routes, preserving training arithmetic."""
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer, routes, config):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A, self.lora_B = nn.ModuleDict(), nn.ModuleDict()
        self.route_scaling = {}
        weight = base_layer.weight
        dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        for route in routes:
            rank = config.trajectory_lora_rank if route == "trajectory" else config.lora_rank
            alpha = config.trajectory_lora_alpha if route == "trajectory" else config.lora_alpha
            self.lora_A[route] = nn.Linear(base_layer.in_features, rank, bias=False,
                                         device=weight.device, dtype=dtype)
            self.lora_B[route] = nn.Linear(rank, base_layer.out_features, bias=False,
                                         device=weight.device, dtype=dtype)
            self.route_scaling[route] = alpha / rank

    def forward(self, hidden, *, route):
        result = self.base_layer(hidden)
        a, b = self.lora_A[route], self.lora_B[route]
        delta = b(a(hidden.to(dtype=a.weight.dtype))) * self.route_scaling[route]
        return result + delta.to(dtype=result.dtype)
