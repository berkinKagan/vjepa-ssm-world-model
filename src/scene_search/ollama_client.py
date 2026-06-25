import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scene_search.schemas import OllamaModelInfo


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        try:
            self.tags()
            return True
        except Exception:
            return False

    def status(self) -> dict:
        try:
            models = self.models()
            return {"running": True, "base_url": self.base_url, "models": [model.to_dict() for model in models]}
        except Exception as exc:
            return {"running": False, "base_url": self.base_url, "error": str(exc), "models": []}

    def tags(self) -> dict:
        return self.get_json("/api/tags")

    def models(self) -> list[OllamaModelInfo]:
        payload = self.tags()
        models = []
        for item in payload.get("models", []):
            name = item.get("name", "")
            models.append(OllamaModelInfo(name=name, size=item.get("size"), modified_at=item.get("modified_at"), vision_capable=is_vision_model_name(name)))
        return models

    def vision_models(self) -> list[OllamaModelInfo]:
        return [model for model in self.models() if model.vision_capable]

    def choose_vision_model(self, requested: str | None) -> str | None:
        models = self.models()
        if requested and any(model.name == requested for model in models):
            return requested if is_vision_model_name(requested) else None
        vision = [model.name for model in models if model.vision_capable]
        return vision[0] if vision else None

    def choose_text_model(self, requested: str | None) -> str | None:
        models = self.models()
        if requested and any(model.name == requested for model in models):
            return requested
        return models[0].name if models else None

    def generate(self, model: str, prompt: str, images: list[str | Path] | None = None) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if images:
            payload["images"] = [encode_image(path) for path in images]
        response = self.post_json("/api/generate", payload)
        return response.get("response", "").strip()

    def get_json(self, path: str) -> dict:
        request = Request(f"{self.base_url}{path}", method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Ollama is not reachable at {self.base_url}") from exc

    def post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc


def is_vision_model_name(name: str) -> bool:
    lowered = name.lower()
    markers = ["vision", "llava", "bakllava", "moondream", "minicpm-v", "qwen2.5vl", "qwen-vl", "gemma3"]
    return any(marker in lowered for marker in markers)


def encode_image(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")
