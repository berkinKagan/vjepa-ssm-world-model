from dataclasses import dataclass
from typing import Any

import torch

from vjepa.preprocess import PreprocessConfig, preprocess_for_torchhub, preprocess_for_transformers


@dataclass
class VJEPAModel:
    model: Any
    processor: Any
    backend: str
    model_name: str
    device: torch.device
    precision: str

    def encode(self, frames: torch.Tensor, preprocess_config: PreprocessConfig) -> torch.Tensor:
        with torch.inference_mode():
            if self.backend == "transformers":
                inputs = preprocess_for_transformers(frames, self.processor, self.device)
                output = self._encode_transformers(inputs)
            elif self.backend == "torchhub":
                tensor = preprocess_for_torchhub(frames, preprocess_config, self.device)
                output = self.model(tensor)
            else:
                raise ValueError(f"Unsupported V-JEPA backend: {self.backend}")
        return extract_tensor_output(output).detach().cpu()

    def _encode_transformers(self, inputs: dict[str, torch.Tensor]):
        if hasattr(self.model, "get_vision_features"):
            try:
                return self.model.get_vision_features(**inputs)
            except TypeError:
                tensor = inputs.get("pixel_values_videos")
                if tensor is None:
                    tensor = inputs.get("pixel_values")
                return self.model.get_vision_features(tensor)
        return self.model(**inputs)


def load_pretrained_vjepa(config: dict) -> VJEPAModel:
    model_config = config["model"]
    backend = model_config.get("backend", "transformers")
    device = resolve_device(model_config.get("device", "auto"))
    precision = model_config.get("precision", "fp32")
    if backend == "transformers":
        return load_transformers_model(model_config, device, precision)
    if backend == "torchhub":
        return load_torchhub_model(model_config, device, precision)
    raise ValueError(f"Unsupported V-JEPA backend: {backend}")


def load_transformers_model(model_config: dict, device: torch.device, precision: str) -> VJEPAModel:
    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:
        raise RuntimeError("Install transformers to use the transformers backend") from exc
    model_name = model_config["name"]
    kwargs = {}
    dtype = resolve_dtype(precision, device)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    processor = AutoVideoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    freeze_model(model)
    return VJEPAModel(model=model, processor=processor, backend="transformers", model_name=model_name, device=device, precision=precision)


def load_torchhub_model(model_config: dict, device: torch.device, precision: str) -> VJEPAModel:
    repo = model_config.get("torchhub_repo", "facebookresearch/vjepa2")
    model_name = model_config.get("torchhub_name", model_config["name"])
    pretrained = bool(model_config.get("torchhub_pretrained", True))
    trust_repo = bool(model_config.get("torchhub_trust_repo", True))
    loaded = torch.hub.load(repo, model_name, pretrained=pretrained, trust_repo=trust_repo)
    model = loaded[0] if isinstance(loaded, tuple | list) else loaded
    model.to(device)
    model.eval()
    freeze_model(model)
    return VJEPAModel(model=model, processor=None, backend="torchhub", model_name=f"{repo}:{model_name}", device=device, precision=precision)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def extract_tensor_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, tuple | list):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Could not extract tensor from model output type: {type(output)!r}")
