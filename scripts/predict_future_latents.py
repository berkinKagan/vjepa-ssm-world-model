import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_manager import DEFAULT_SSM_CONFIG_PATH
from ssm.predictor import predict_future_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future latent states from a saved V-JEPA embedding sequence")
    parser.add_argument("--embedding", required=True, help="Path to a saved V-JEPA embedding file")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved SSM checkpoint")
    parser.add_argument("--output", default=None, help="Path for saved future latent prediction")
    parser.add_argument("--config", default=str(DEFAULT_SSM_CONFIG_PATH), help="Path to SSM config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_future_latents(args.embedding, args.checkpoint, args.output, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
