from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreprocessConfig:
    clip_frames: int
    frame_stride: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def preprocess_for_transformers(frames: torch.Tensor, processor, device: torch.device) -> dict[str, torch.Tensor]:
    video = frames.permute(0, 3, 1, 2)
    inputs = processor(video, return_tensors="pt")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def preprocess_for_torchhub(frames: torch.Tensor, config: PreprocessConfig, device: torch.device) -> torch.Tensor:
    video = frames.to(torch.float32).permute(0, 3, 1, 2) / 255.0
    resized = resize_short_side(video, config.image_size)
    cropped = center_crop(resized, config.image_size)
    mean = torch.tensor(config.mean, dtype=cropped.dtype, device=cropped.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, dtype=cropped.dtype, device=cropped.device).view(1, 3, 1, 1)
    normalized = (cropped - mean) / std
    return normalized.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def resize_short_side(video: torch.Tensor, size: int) -> torch.Tensor:
    height = video.shape[-2]
    width = video.shape[-1]
    if height == size and width == size:
        return video
    scale = size / min(height, width)
    new_height = round(height * scale)
    new_width = round(width * scale)
    return F.interpolate(video, size=(new_height, new_width), mode="bilinear", align_corners=False)


def center_crop(video: torch.Tensor, size: int) -> torch.Tensor:
    height = video.shape[-2]
    width = video.shape[-1]
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return video[..., top:top + size, left:left + size]
