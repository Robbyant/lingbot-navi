"""Shared frozen Qwen3.5 visual encoder loading."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def load_vision(model_path, device, dtype, *, rotary_dtype=None):
    """Load shared visual weights while preserving caller position-encoding precision."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    path = Path(model_path)
    config_dict = json.loads((path / 'config.json').read_text())['vision_config']
    config = Qwen3_5VisionConfig(**config_dict)
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        vision = Qwen3_5VisionModel(config)
    index_path = path / 'model.safetensors.index.json'
    shards = defaultdict(list)
    prefix = 'model.visual.'
    if index_path.is_file():
        mapping = json.loads(index_path.read_text())['weight_map']
        for (key, shard) in mapping.items():
            if key.startswith(prefix):
                shards[shard].append(key)
    else:
        with safe_open(path / 'model.safetensors', framework='pt', device='cpu') as source:
            shards['model.safetensors'] = [k for k in source.keys() if k.startswith(prefix)]
    (weights, checksum) = ({}, hashlib.sha256())
    for (shard, keys) in sorted(shards.items()):
        with safe_open(path / shard, framework='pt', device='cpu') as source:
            for key in sorted(keys):
                tensor = source.get_tensor(key)
                weights[key[len(prefix):]] = tensor
    for (key, tensor) in sorted(weights.items()):
        checksum.update((prefix + key).encode())
        checksum.update(tensor.view(torch.uint8).numpy().tobytes())
    if not weights:
        raise ValueError(f'no Qwen3.5 visual weights in {path}')
    vision.load_state_dict(weights, strict=True, assign=True)
    head_dim = config.hidden_size // config.num_heads
    frequency = 1.0 / vision.rotary_pos_emb.theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2))
    vision.rotary_pos_emb.inv_freq = frequency
    vision = vision.to(device=device, dtype=dtype).eval().requires_grad_(False)
    # from_pretrained keeps this buffer in float32; the local planner's
    # established loader casts it with the model. Preserve both contracts.
    if rotary_dtype is not None:
        vision.rotary_pos_emb.inv_freq = frequency.to(device=device, dtype=rotary_dtype)
    identity = dict(config_sha256=digest(config_dict), visual_weights_sha256=checksum.hexdigest())
    return (vision, identity)
