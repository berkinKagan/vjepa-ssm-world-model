import json
import re


GENERIC_PATTERNS = [
    "The image shows",
    "This image depicts",
    "popular TV show",
    "the TV show",
    "TV show",
    "scene from",
    "from the show",
    "moment from the show",
    "the purpose of the image is to depict",
    "the purpose of the image is",
    "image is a still",
    "actor",
    "played by",
    "The Big Bang Theory"
]

NAME_PATTERNS = [
    "Sheldon",
    "Leonard",
    "Penny",
    "Howard",
    "Raj",
    "Amy",
    "Bernadette"
]


def parse_caption_payload(raw_caption: str) -> dict:
    text = strip_code_fence(raw_caption.strip())
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"searchable_summary": str(value)}
    except Exception:
        return {"searchable_summary": text}


def clean_caption(raw_caption: str) -> dict:
    payload = parse_caption_payload(raw_caption)
    cleaned = {key: clean_text(value) for key, value in payload.items()}
    summary = cleaned.get("searchable_summary") or build_summary(cleaned)
    cleaned["searchable_summary"] = clean_text(summary)
    return cleaned


def strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    stripped = text.strip("`")
    lines = stripped.splitlines()
    if lines and lines[0].lower().strip() == "json":
        lines = lines[1:]
    return "\n".join(lines).strip()


def clean_text(value) -> str:
    text = " ".join(value) if isinstance(value, list) else str(value)
    for pattern in GENERIC_PATTERNS + NAME_PATTERNS:
        text = re.sub(re.escape(pattern), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text.strip(" .")


def build_summary(payload: dict) -> str:
    values = [value for key, value in payload.items() if key != "searchable_summary" and value]
    return ". ".join(values)
