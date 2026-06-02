import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.paths import PROJECT_ROOT, resolve_project_path

DEFAULT_SSM_CONFIG_PATH = PROJECT_ROOT / "configs" / "ssm.yaml"


def load_ssm_config(path: str | Path | dict = DEFAULT_SSM_CONFIG_PATH) -> dict:
    if isinstance(path, dict):
        return path
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_ssm_path(path: str | Path) -> Path:
    return resolve_project_path(path)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(value, path: str | Path) -> Path:
    destination = resolve_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    return destination


def save_yaml(value, path: str | Path) -> Path:
    destination = resolve_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
    return destination
