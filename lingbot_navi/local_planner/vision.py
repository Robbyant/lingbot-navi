"""Frozen Qwen3.5 visual tower and the checkpoint's still-image preprocessing."""
from collections.abc import Mapping
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from lingbot_navi.vision import digest
from .head import RoutePlanningHead


IMAGE_PREPROCESSING_SCHEMA = "lingbotnavi_native_still_image_preprocessing_v1"


def file_digest(path):
    with Path(path).open("rb") as source:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def _resize_to_template_limit(image, max_pixels):
    if image.width * image.height <= max_pixels:
        return image
    from torchvision.transforms import Resize
    ratio = image.width / image.height
    height = math.sqrt(max_pixels / ratio)
    return Resize((int(height), int(height * ratio)))(image)


def _image_config(processor):
    return json.loads(processor.to_json_string())


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _check_native_implementation(contract, processor):
    from qwen_vl_utils import vision_process
    import transformers

    if contract.get("schema_version") != IMAGE_PREPROCESSING_SCHEMA:
        raise ValueError("unsupported image preprocessing contract; re-export prompt cache")
    # Pillow version equality is not required; retain the recorded contract
    # for provenance and processor identity, and validate the other fields.
    checks = dict(
        image_processor_class=f"{type(processor).__module__}.{type(processor).__name__}",
        image_patch_size=processor.patch_size, spatial_merge_size=processor.merge_size,
        max_ratio=vision_process.MAX_RATIO, processor_do_resize=False,
        template_load_images=True,
        qwen_vl_utils_version=version("qwen_vl_utils"),
        qwen_vl_utils_sha256=file_digest(vision_process.__file__),
        transformers_version=transformers.__version__,
    )
    for key, value in checks.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"native image preprocessing {key} differs from prompt cache; re-export it")
    if contract["spatial_merge_size"] != vision_process.SPATIAL_MERGE_SIZE:
        raise ValueError("qwen_vl_utils spatial merge changed from the prompt cache")
    minimum = _positive_integer(contract.get("min_pixels"), "min_pixels")
    maximum = _positive_integer(contract.get("max_pixels"), "max_pixels")
    if minimum > maximum:
        raise ValueError("native minimum image pixels exceed maximum image pixels")
    if "template_max_pixels" not in contract:
        raise ValueError("missing template resize policy; re-export prompt cache")
    cap = contract["template_max_pixels"]
    if cap is not None:
        _positive_integer(cap, "template_max_pixels")


class NativeCurrentImageProcessor:
    """PIL still images -> the same pixel_values/grid as the native VLM path."""

    def __init__(self, image_processor, contract):
        self.image_processor = image_processor
        self.contract = dict(contract)

    def __call__(self, images, return_tensors="pt"):
        from qwen_vl_utils import vision_process

        # Worker/global patches must not silently change merge or aspect checks.
        if (vision_process.SPATIAL_MERGE_SIZE != self.contract["spatial_merge_size"]
                or vision_process.MAX_RATIO != self.contract["max_ratio"]):
            raise ValueError("qwen_vl_utils image globals changed after processor initialization")
        if isinstance(images, Image.Image):
            images = [images]
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("native current-image preprocessing requires nonempty PIL still images")
        resized = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("current images must be PIL images; video/3D/array inputs are not supported")
            # Template._preprocess_inputs loads/converts to RGB before replace_tag.
            # This also preserves Swift's actual RGBA behavior (not fetch_image's
            # white-alpha composition when called directly on an RGBA image).
            image = image.convert("RGB")
            if self.contract["template_max_pixels"] is not None:
                image = _resize_to_template_limit(image, self.contract["template_max_pixels"])
            resized.append(vision_process.fetch_image(
                dict(image=image, min_pixels=self.contract["min_pixels"], max_pixels=self.contract["max_pixels"]),
                image_patch_size=self.contract["image_patch_size"]))
        return self.image_processor(images=resized, return_tensors=return_tensors, do_resize=False)

    def to_json_string(self):
        return json.dumps(dict(image_preprocessing=self.contract,
                               image_processor_config=_image_config(self.image_processor)),
                          sort_keys=True, indent=2) + "\n"


def load_current_image_processor(model_path, encoder_identity, max_pixels=None):
    """Load native HF processing and replay the prompt cache's resize policy.

    HF image processor config alone cannot recover old fetch_image settings:
    native Swift disabled its resize and used separately environment-patched
    qwen_vl_utils globals. Such old caches must be exported again, not guessed.
    """
    from transformers import AutoImageProcessor

    if not isinstance(encoder_identity, Mapping):
        raise ValueError("image preprocessing requires the prompt encoder identity")
    contract = encoder_identity.get("image_preprocessing")
    if not isinstance(contract, Mapping):
        raise ValueError("prompt cache lacks the native fetch_image resize policy; re-export prompt cache")
    recorded_config = encoder_identity.get("image_processor_config")
    if recorded_config is None:
        # Older Transformers/exporters may keep only the nested component.
        processor_config = encoder_identity.get("processor_config", {})
        if isinstance(processor_config, Mapping):
            recorded_config = processor_config.get("image_processor")
    if not isinstance(recorded_config, Mapping):
        raise ValueError("prompt cache lacks image_processor_config; re-export prompt cache")
    processor = AutoImageProcessor.from_pretrained(str(Path(model_path)), local_files_only=True)
    if _image_config(processor) != recorded_config:
        raise ValueError("native image_processor_config differs from prompt cache; re-export it")
    _check_native_implementation(contract, processor)
    if max_pixels is not None:
        _positive_integer(max_pixels, "max_pixels")
        if max_pixels != contract["max_pixels"]:
            raise ValueError("--max-pixels must equal the prompt cache native pixel budget; "
                             "a local-only override would change VLM preprocessing")
    return NativeCurrentImageProcessor(processor, contract)

class VisualLocalPlanner(nn.Module):
    def __init__(self, vision, config):
        super().__init__()
        if vision.config.out_hidden_size != config.vision_width:
            raise ValueError("ViT merged feature width does not match head config")
        self.vision = vision.requires_grad_(False).eval()
        self.head = RoutePlanningHead(config)

    def encode_observations(self, batch):
        parameter = next(self.vision.parameters())
        device = parameter.device
        cpu_grid = batch['image_grid_thw']
        grid = cpu_grid.to(device, non_blocking=True)
        with torch.no_grad():
            features = self.vision(batch['pixel_values'].to(device=device, dtype=parameter.dtype, non_blocking=True), grid_thw=grid).pooler_output.float()
        merge = self.vision.config.spatial_merge_size
        sizes = (cpu_grid.prod(-1) // (merge * merge)).tolist()
        images = features.split(sizes)
        locations = []
        for (temporal, height, width) in cpu_grid.tolist():
            if temporal != 1:
                raise ValueError('local planner expects observed RGB images, not video tensors')
            (height, width) = (height // merge, width // merge)
            (y, x) = torch.meshgrid((torch.arange(height, device=device) + 0.5) / height, (torch.arange(width, device=device) + 0.5) / width, indexing='ij')
            locations.append(torch.stack([x.flatten(), y.flatten()], -1) * 2 - 1)
        (token_rows, metadata_rows) = ([], [])
        for (row, ids) in enumerate(batch['image_ids']):
            token_rows.append(torch.cat([images[i] for i in ids]))
            frame = torch.as_tensor(np.column_stack([batch['frame_times'][row], batch['frame_roles'][row]]), device=device, dtype=torch.float32)
            metadata_rows.append(torch.cat([torch.cat([locations[i], frame[j].expand(len(locations[i]), -1)], -1) for (j, i) in enumerate(ids)]))
        lengths = torch.tensor([len(row) for row in token_rows], device=device)
        padded = pad_sequence(token_rows, batch_first=True)
        padding = torch.arange(padded.shape[1], device=device)[None] >= lengths[:, None]
        return dict(visual_tokens=padded, visual_metadata=pad_sequence(metadata_rows, batch_first=True), visual_padding_mask=padding)
