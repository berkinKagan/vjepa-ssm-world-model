import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.paths import DEFAULT_CONFIG_PATH, load_config
from vjepa.extractor import VJEPAExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen V-JEPA embeddings from a local video")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--output", default=None, help="Path for saved embeddings, ending in .pt or .npy")
    parser.add_argument("--metadata", default=None, help="Path for saved metadata JSON")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to V-JEPA extraction config")
    parser.add_argument("--max-clips", type=int, default=None, help="Override maximum number of sampled clips")
    parser.add_argument("--clip-stride-seconds", type=float, default=None, help="Override spacing between clip starts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_clips is not None:
        config["video"]["max_clips"] = args.max_clips
    if args.clip_stride_seconds is not None:
        config["video"]["clip_stride_seconds"] = args.clip_stride_seconds
    extractor = VJEPAExtractor(config)
    result = extractor.extract_video(args.video, output_path=args.output, metadata_path=args.metadata, progress_callback=print_progress)
    print(json.dumps({"embedding_file": str(result.embeddings_path), "metadata_file": str(result.metadata_path), "embedding_shape": result.metadata["embedding_shape"]}, indent=2))


def print_progress(done: int, total: int) -> None:
    print(f"processed {done}/{total} clips", flush=True)


if __name__ == "__main__":
    main()
