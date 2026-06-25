import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_search.retriever import search_scene_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search indexed V-JEPA video scenes with a text query")
    parser.add_argument("--config", default="configs/scene_search.yaml", help="Path to scene-search config")
    parser.add_argument("--query", required=True, help="Natural-language scene query")
    parser.add_argument("--top-k", type=int, default=None, help="Number of matching scenes to return")
    parser.add_argument("--index", default=None, help="Path to scene index JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = search_scene_index(config, args.query, args.index, args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
