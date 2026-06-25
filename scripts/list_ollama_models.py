import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_search.ollama_client import OllamaClient


def main() -> None:
    client = OllamaClient("http://127.0.0.1:11434")
    print(json.dumps(client.status(), indent=2))


if __name__ == "__main__":
    main()
