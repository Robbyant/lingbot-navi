from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def validate_canvas(height, width, multiple=32):
    if any(
        type(side) is not int or side <= 0 or side % multiple
        for side in (height, width)
    ):
        raise ValueError(
            f"Qwen canvas dimensions must be positive multiples of {multiple}"
        )


def qwen_image_geometry(width, height, *, input_height, input_width, multiple=32, resize_mode="fixed"):
    validate_canvas(input_height, input_width, multiple)
    if min(width, height) <= 0:
        raise ValueError("RGB dimensions must be positive")
    if resize_mode == "fixed":
        scale = min(input_width / width, input_height / height)
    elif resize_mode == "aspect":
        # Historical Map1 preprocessing: never enlarge; pad to whole patches.
        scale = min(1.0, max(input_width, input_height) / max(width, height))
    else:
        raise ValueError("resize_mode must be 'fixed' or 'aspect'")
    content_width, content_height = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    if resize_mode == "aspect":
        input_width = (content_width + multiple - 1) // multiple * multiple
        input_height = (content_height + multiple - 1) // multiple * multiple
    return dict(
        input_height=input_height,
        input_width=input_width,
        content_height=content_height,
        content_width=content_width,
        padding_top=(input_height - content_height) // 2,
        padding_left=(input_width - content_width) // 2,
    )


class Qwen3_5ViTExtractor:
    def __init__(self, checkpoint_path, *, device, input_width=640, input_height=480, resize_mode="fixed"):
        from transformers import AutoConfig, AutoImageProcessor
        from lingbot_navi.vision import load_vision

        if not Path(checkpoint_path).is_dir():
            raise FileNotFoundError(checkpoint_path)
        config = AutoConfig.from_pretrained(
            checkpoint_path, local_files_only=True
        ).vision_config
        self.feature_dim = int(config.out_hidden_size)
        self.patch_size = int(config.patch_size * config.spatial_merge_size)
        self.merge_size = int(config.spatial_merge_size)
        if self.feature_dim != 2560:
            raise ValueError(
                "Spatial Memory requires Qwen3.5-4B merged features (2560 dimensions)"
            )
        validate_canvas(input_height, input_width, self.patch_size)
        if resize_mode not in ("fixed", "aspect"):
            raise ValueError("resize_mode must be fixed or aspect")
        self.resize_mode = resize_mode
        self.width, self.height, self.device = input_width, input_height, device
        self._image_processor = AutoImageProcessor.from_pretrained(
            checkpoint_path, local_files_only=True
        )
        dtype = (
            torch.float16
            if torch.device(device).type == "cuda"
            and not torch.cuda.is_bf16_supported()
            else torch.bfloat16
        )
        self._encoder, _ = load_vision(
            checkpoint_path, torch.device(device), dtype, rotary_dtype=torch.float32
        )

    @torch.inference_mode()
    def extract(self, rgb):
        geometry = qwen_image_geometry(
            rgb.shape[1],
            rgb.shape[0],
            input_height=self.height,
            input_width=self.width,
            multiple=self.patch_size,
            resize_mode=self.resize_mode,
        )
        content = (
            Image.fromarray(rgb)
            .convert("RGB")
            .resize(
                (geometry["content_width"], geometry["content_height"]),
                Image.Resampling.BICUBIC,
            )
        )
        image = Image.new("RGB", (geometry["input_width"], geometry["input_height"]))
        image.paste(content, (geometry["padding_left"], geometry["padding_top"]))
        inputs = self._image_processor(
            images=image, do_resize=False, return_tensors="pt"
        )
        grid = inputs["image_grid_thw"].to(self.device)
        t, h, w = grid[0].tolist()
        if t != 1:
            raise ValueError("expected a single still image")
        output = self._encoder(
            inputs["pixel_values"].to(self.device), grid_thw=grid, return_dict=True
        )
        rows, cols = h // self.merge_size, w // self.merge_size
        if output.pooler_output.shape != (rows * cols, self.feature_dim):
            raise ValueError("vision patch grid differs from the calibrated canvas")
        features = (
            output.pooler_output.reshape(rows, cols, self.feature_dim).float().cpu()
        )
        pixels = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
        patches = pixels.unfold(0, self.patch_size, self.patch_size).unfold(
            1, self.patch_size, self.patch_size
        )
        return dict(
            features=features,
            patch_rgb=patches.reshape(rows, cols, 3, -1).mean(dim=-1),
            spatial_geometry=dict(
                **geometry, patch_height=self.patch_size, patch_width=self.patch_size
            ),
        )
