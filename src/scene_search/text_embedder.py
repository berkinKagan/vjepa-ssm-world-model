import hashlib

import numpy as np

from experiments.run_manager import resolve_device


class TextEmbedder:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.backend = "sentence_transformers"
        if model_name == "hashing":
            self.backend = "hashing"
            self.model = None
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=str(resolve_device(device)))
        except Exception:
            self.backend = "hashing"
            self.model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.stack([hash_embedding(text) for text in texts]).astype("float32")
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")


def hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    values = np.zeros(dim, dtype="float32")
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values
