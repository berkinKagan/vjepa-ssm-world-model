import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_search.hybrid_retriever import hybrid_search_scene_index
from scene_search.indexer import build_scene_index
from scene_search.storage import load_scene_index_metadata
from utils.paths import DEFAULT_CONFIG_PATH, load_config
from vjepa.extractor import VJEPAExtractor

SELF_TEST_QUERIES = [
    "unboxing a gift with boxes",
    "standing on the couch",
    "arms outstretched on couch",
    "person holding a piece of paper",
    "small colorful object on the table"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug hybrid V-JEPA plus caption scene search")
    parser.add_argument("--config", default="configs/scene_search.yaml", help="Path to scene-search config")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--query", default=None, help="Scene query")
    parser.add_argument("--embedding", default=None, help="Path to saved V-JEPA embedding artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    embedding = args.embedding or find_or_extract_embedding(args.video)
    index_result = build_scene_index(config, embedding)
    metadata = load_scene_index_metadata(index_result["index_file"])
    query = args.query or SELF_TEST_QUERIES[0]
    result = hybrid_search_scene_index(config, query, index_result["index_file"], 10)
    print(json.dumps({"video_duration": metadata.get("video_duration"), "segment_count": metadata.get("segment_count"), "coverage_ratio": metadata.get("coverage_ratio"), "aligner_status": result.get("aligner_status"), "retrieval_mode": result.get("retrieval_mode")}, indent=2))
    for item in result["results"][:10]:
        print(f"{item['segment_index']} {item['start_time']:.2f}-{item['end_time']:.2f} caption={item.get('caption_score')} latent={item.get('latent_score')} final={item.get('final_score')} {item.get('searchable_summary')}")


def find_embedding(video_path: str) -> str:
    stem = Path(video_path).stem
    candidates = sorted(Path("outputs/embeddings").glob(f"{stem}*.pt"))
    if not candidates:
        candidates = sorted(Path("outputs/embeddings").glob(f"*{stem}*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No existing embedding found for video stem: {stem}")
    return str(candidates[-1])


def find_or_extract_embedding(video_path: str) -> str:
    try:
        return find_embedding(video_path)
    except FileNotFoundError:
        extractor = VJEPAExtractor(load_config(DEFAULT_CONFIG_PATH))
        result = extractor.extract_video(video_path)
        return str(result.embeddings_path)


if __name__ == "__main__":
    main()
