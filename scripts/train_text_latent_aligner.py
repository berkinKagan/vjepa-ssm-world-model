import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_search.latent_aligner import ensure_text_latent_aligner
from scene_search.storage import load_scene_index_metadata
from utils.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the text-to-V-JEPA latent aligner from a built scene index")
    parser.add_argument("--config", default="configs/scene_search.yaml", help="Path to scene-search config")
    parser.add_argument("--index", required=True, help="Path to scene index JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    index_path = resolve_project_path(args.index)
    metadata = load_scene_index_metadata(index_path)
    caption_embeddings = np.load(index_path.with_suffix(".npy"))
    result = ensure_text_latent_aligner(config, caption_embeddings, metadata.get("segment_latents_file"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
