from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "vjepa.yaml"


def resolve_project_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_config_path(config: dict, key: str) -> Path:
    value = config["paths"][key]
    return resolve_project_path(value)
