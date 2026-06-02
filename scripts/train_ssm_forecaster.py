import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_manager import DEFAULT_SSM_CONFIG_PATH
from ssm.trainer import train_ssm_forecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SSM/Mamba latent forecaster over saved V-JEPA embeddings")
    parser.add_argument("--config", default=str(DEFAULT_SSM_CONFIG_PATH), help="Path to SSM training config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_ssm_forecaster(args.config, progress_callback=print_progress)
    print(json.dumps({"best_checkpoint": str(result.best_checkpoint), "latest_checkpoint": str(result.latest_checkpoint), "latent_dim": result.latent_dim, "model_type": result.model_type}, indent=2))


def print_progress(epoch: int, total: int, metrics: dict) -> None:
    print(f"epoch {epoch}/{total} mse={metrics.get('mse', metrics.get('loss', 0.0))}", flush=True)


if __name__ == "__main__":
    main()
