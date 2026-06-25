import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_search.indexer import build_scene_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a text-searchable scene index from saved V-JEPA metadata and video frames")
    parser.add_argument("--config", default="configs/scene_search.yaml", help="Path to scene-search config")
    parser.add_argument("--embedding", required=True, help="Path to saved V-JEPA embedding artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = build_scene_index(config, args.embedding, progress_callback=print_progress)
    print(json.dumps(result, indent=2))


def print_progress(done: int, total: int) -> None:
    print(f"indexed {done}/{total} clips", flush=True)


if __name__ == "__main__":
    main()
