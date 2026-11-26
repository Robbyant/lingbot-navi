import copy
import torch
from torch import nn
from torch.nn import functional as F


class Float32Module(nn.Module):
    def _apply(self, fn, recurse=True):

        def preserve_fp32(value):
            converted = fn(value)
            if value.is_floating_point() and converted.dtype != torch.float32:
                if value.is_meta:
                    return converted.float()
                return value.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(preserve_fp32, recurse=recurse)


class SpatialLinear(nn.Linear):
    def forward(self, inputs):
        return F.linear(
            inputs,
            self.weight.to(inputs.dtype),
            None if self.bias is None else self.bias.to(inputs.dtype),
        )


class FullSpatialModule(Float32Module):
    def __init__(self, source, names):
        super().__init__()
        for name in names:
            module = copy.deepcopy(getattr(source, name)).float()
            self._replace_linears(module)
            self.add_module(name, module)
        self.requires_grad_(False)

    @staticmethod
    def _replace_linears(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                replacement = SpatialLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device="meta",
                )
                replacement.weight, replacement.bias = (child.weight, child.bias)
                setattr(module, name, replacement)
            else:
                FullSpatialModule._replace_linears(child)
