import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_manager import DEFAULT_SSM_CONFIG_PATH
from ssm.evaluator import evaluate_ssm_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SSM/Mamba latent forecaster checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved SSM checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_SSM_CONFIG_PATH), help="Path to SSM config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_ssm_checkpoint(args.checkpoint, args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
