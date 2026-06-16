from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from crag.config import EMBEDDING_MODEL_NAME


def load_model(local_only: bool = True) -> SentenceTransformer:
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_model_for_download() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[np.ndarray]:
    vectors = model.encode(texts, normalize_embeddings=True)
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


def serialize_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def deserialize_vector(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
