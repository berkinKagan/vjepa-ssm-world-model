import json
from pathlib import Path


def save_training_curve(metrics_path: str | Path, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    epochs = [row["epoch"] for row in metrics]
    mse = [row.get("mse", row.get("mse_loss", 0.0)) for row in metrics]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, mse)
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    return destination
