from pathlib import Path

from scene_search.ollama_client import OllamaClient

VISION_PROMPT = "Describe this video frame as part of a short video clip. Focus on visible people, objects, actions, interactions, location, and motion-relevant cues. Keep it concise and factual. Do not invent details."


class SceneCaptioner:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("caption_backend", "ollama_vision")
        self.client = OllamaClient(config["ollama_base_url"])

    def caption_clip(self, frame_paths: list[str], metadata: dict, clip: dict) -> tuple[str, str, str | None]:
        if self.backend == "placeholder":
            return placeholder_caption(metadata, clip), "placeholder", None
        if self.backend == "ollama_vision":
            return self.caption_with_ollama(frame_paths, metadata, clip)
        raise ValueError(f"Unsupported caption backend: {self.backend}")

    def caption_with_ollama(self, frame_paths: list[str], metadata: dict, clip: dict) -> tuple[str, str, str | None]:
        try:
            if not frame_paths:
                return placeholder_caption(metadata, clip, "No representative frame is available"), "placeholder", None
            model = self.client.choose_vision_model(self.config.get("ollama_model"))
            if not model:
                return placeholder_caption(metadata, clip, "No Ollama vision model is available"), "placeholder", None
            captions = [self.client.generate(model, VISION_PROMPT, images=[path]) for path in frame_paths]
            clean = [caption for caption in captions if caption]
            if not clean:
                return placeholder_caption(metadata, clip, "Ollama returned an empty caption"), "placeholder", model
            if len(clean) == 1:
                return clean[0], "ollama_vision", model
            summary = self.summarize_captions(clean)
            return summary or " ".join(clean), "ollama_vision", model
        except Exception as exc:
            return placeholder_caption(metadata, clip, str(exc)), "placeholder", None

    def summarize_captions(self, captions: list[str]) -> str:
        model = self.client.choose_text_model(self.config.get("ollama_text_model"))
        if not model:
            return " ".join(captions)
        prompt = "Summarize these frame captions into one concise factual video clip description:\n" + "\n".join(captions)
        return self.client.generate(model, prompt)


def placeholder_caption(metadata: dict, clip: dict, warning: str | None = None) -> str:
    name = metadata.get("video_filename") or Path(metadata.get("video_path", "unknown_video")).name
    start = float(clip["start_time"])
    end = float(clip["end_time"])
    text = f"Clip {clip['index']} from {start:.2f}s to {end:.2f}s of video {name}."
    return f"{text} {warning}." if warning else text
