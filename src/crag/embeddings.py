from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from crag.config import EMBEDDING_MODEL_NAME

EMBEDDING_DIMENSIONS = 384
_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def load_model(local_only: bool = True) -> SentenceTransformer:
    if not local_only:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)

    previous_values = {name: os.environ.get(name) for name in _OFFLINE_ENV_VARS}
    missing_values = {name for name in _OFFLINE_ENV_VARS if name not in os.environ}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    finally:
        for name in _OFFLINE_ENV_VARS:
            if name in missing_values:
                os.environ.pop(name, None)
            else:
                previous_value = previous_values[name]
                if previous_value is not None:
                    os.environ[name] = previous_value


def load_model_for_download() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[np.ndarray]:
    vectors = model.encode(texts, normalize_embeddings=True)
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


def serialize_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def deserialize_vector(
    raw: bytes,
    expected_dimensions: int | None = EMBEDDING_DIMENSIONS,
) -> np.ndarray:
    float32_size = np.dtype(np.float32).itemsize
    if len(raw) % float32_size != 0:
        raise ValueError("Serialized vector byte length must be a multiple of float32 size")

    vector = np.frombuffer(raw, dtype=np.float32)
    if expected_dimensions is not None and vector.size != expected_dimensions:
        raise ValueError(
            f"Serialized vector has {vector.size} dimensions; expected {expected_dimensions}"
        )
    return vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
