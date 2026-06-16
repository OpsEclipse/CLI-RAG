import os

import numpy as np
import pytest

from crag.config import EMBEDDING_MODEL_NAME
from crag.embeddings import (
    cosine_similarity,
    deserialize_vector,
    load_model,
    serialize_vector,
)


def test_vector_round_trip():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    restored = deserialize_vector(serialize_vector(vector))

    assert restored.dtype == np.float32
    assert restored.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_cosine_similarity():
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)

    assert cosine_similarity(left, right) == pytest.approx(1.0)
    assert cosine_similarity(left, other) == pytest.approx(0.0)


def test_cosine_similarity_returns_zero_for_zero_norm_vectors():
    zero = np.array([0.0, 0.0], dtype=np.float32)
    nonzero = np.array([1.0, 0.0], dtype=np.float32)

    assert cosine_similarity(zero, nonzero) == pytest.approx(0.0)
    assert cosine_similarity(nonzero, zero) == pytest.approx(0.0)


def test_load_model_local_only_sets_offline_environment(monkeypatch):
    created_with = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            created_with.append(model_name)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr("crag.embeddings.SentenceTransformer", FakeSentenceTransformer)

    model = load_model(local_only=True)

    assert isinstance(model, FakeSentenceTransformer)
    assert created_with == [EMBEDDING_MODEL_NAME]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
