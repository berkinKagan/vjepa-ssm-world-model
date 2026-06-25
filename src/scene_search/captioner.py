import json
from pathlib import Path

from scene_search.caption_cleaner import clean_caption
from scene_search.ollama_client import OllamaClient

VISION_PROMPT = "You are describing a short video segment using multiple frames from the same segment. Describe only what is visually observable. Do not name TV shows, actors, or fictional characters unless the name is visibly written on screen. Focus on people, objects, actions, interactions, location, spatial relations, and visible change across frames. Do not invent context. Return concise JSON with keys: people, objects, actions, location, spatial_relations, temporal_change, searchable_summary. The searchable_summary must be one short sentence optimized for scene retrieval."


class SceneCaptioner:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("caption_backend", "ollama_vision")
        self.client = OllamaClient(config["ollama_base_url"])

    def caption_clip(self, frame_paths: list[str], metadata: dict, clip: dict) -> tuple[dict, str, str | None]:
        if self.backend == "placeholder":
            raw = placeholder_caption(metadata, clip)
            return caption_payload(raw), "placeholder", None
        if self.backend == "ollama_vision":
            return self.caption_with_ollama(frame_paths, metadata, clip)
        raise ValueError(f"Unsupported caption backend: {self.backend}")

    def caption_with_ollama(self, frame_paths: list[str], metadata: dict, clip: dict) -> tuple[dict, str, str | None]:
        try:
            if not frame_paths:
                raw = placeholder_caption(metadata, clip, "No representative frame is available")
                return caption_payload(raw), "placeholder", None
            model = self.client.choose_vision_model(self.config.get("ollama_model"))
            if not model:
                raw = placeholder_caption(metadata, clip, "No Ollama vision model is available")
                return caption_payload(raw), "placeholder", None
            raw = self.generate_segment_caption(model, frame_paths)
            if not raw:
                raw = placeholder_caption(metadata, clip, "Ollama returned an empty caption")
                return caption_payload(raw), "placeholder", model
            return caption_payload(raw), "ollama_vision", model
        except Exception as exc:
            raw = placeholder_caption(metadata, clip, str(exc))
            return caption_payload(raw), "placeholder", None

    def generate_segment_caption(self, model: str, frame_paths: list[str]) -> str:
        try:
            return self.client.generate(model, VISION_PROMPT, images=frame_paths)
        except RuntimeError as exc:
            if len(frame_paths) <= 1 or "only supports one image" not in str(exc):
                raise
            return self.caption_single_image_sequence(model, frame_paths)

    def caption_single_image_sequence(self, model: str, frame_paths: list[str]) -> str:
        labels = frame_labels(len(frame_paths))
        observations = []
        for label, path in zip(labels, frame_paths):
            prompt = f"Describe this {label} frame from a short video segment. Focus only on visible people, objects, actions, interactions, location, and motion-relevant cues. Do not identify shows, actors, fictional characters, or production context. Keep it concise and factual. Do not invent context."
            text = self.client.generate(model, prompt, images=[path])
            observations.append({"frame": label, "caption": text})
        return json.dumps(sequence_payload(observations))


def placeholder_caption(metadata: dict, clip: dict, warning: str | None = None) -> str:
    name = metadata.get("video_filename") or Path(metadata.get("video_path", "unknown_video")).name
    start = float(clip["start_time"])
    end = float(clip["end_time"])
    text = f"Clip {clip['index']} from {start:.2f}s to {end:.2f}s of video {name}."
    return f"{text} {warning}." if warning else text


def caption_payload(raw_caption: str) -> dict:
    cleaned = clean_caption(raw_caption)
    summary = cleaned.get("searchable_summary", raw_caption)
    clean_text = ". ".join(value for value in cleaned.values() if value)
    return {"raw_caption": raw_caption, "clean_caption": clean_text, "searchable_summary": summary}


def frame_labels(count: int) -> list[str]:
    if count == 1:
        return ["middle"]
    if count == 2:
        return ["start", "end"]
    if count == 3:
        return ["start", "middle", "end"]
    return [f"frame {index + 1}" for index in range(count)]


def sequence_payload(observations: list[dict]) -> dict:
    captions = [item["caption"] for item in observations if item.get("caption")]
    summary = concise_summary(captions)
    return {"people": "", "objects": "", "actions": summary, "location": "", "spatial_relations": "", "temporal_change": " ".join(f"{item['frame']}: {item['caption']}" for item in observations if item.get("caption")), "searchable_summary": summary}


def concise_summary(captions: list[str]) -> str:
    if not captions:
        return ""
    middle = captions[len(captions) // 2]
    return middle.strip()
