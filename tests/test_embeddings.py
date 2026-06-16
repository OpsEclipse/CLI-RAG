import os

import numpy as np
import pytest

from crag.config import EMBEDDING_MODEL_NAME
from crag.embeddings import (
    cosine_similarity,
    deserialize_vector,
    load_model,
    load_model_for_download,
    serialize_vector,
)


def test_vector_round_trip():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    restored = deserialize_vector(serialize_vector(vector), expected_dimensions=3)

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


def test_load_model_local_only_sets_offline_environment_during_construction(monkeypatch):
    constructor_env = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            constructor_env.append(
                (
                    model_name,
                    os.environ.get("HF_HUB_OFFLINE"),
                    os.environ.get("TRANSFORMERS_OFFLINE"),
                )
            )

    monkeypatch.setenv("HF_HUB_OFFLINE", "previous-hf")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr("crag.embeddings.SentenceTransformer", FakeSentenceTransformer)

    model = load_model(local_only=True)

    assert isinstance(model, FakeSentenceTransformer)
    assert constructor_env == [(EMBEDDING_MODEL_NAME, "1", "1")]
    assert os.environ["HF_HUB_OFFLINE"] == "previous-hf"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_load_model_for_download_does_not_inherit_local_only_environment(monkeypatch):
    constructor_env = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            constructor_env.append(
                (
                    model_name,
                    os.environ.get("HF_HUB_OFFLINE"),
                    os.environ.get("TRANSFORMERS_OFFLINE"),
                )
            )

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr("crag.embeddings.SentenceTransformer", FakeSentenceTransformer)

    load_model(local_only=True)
    load_model_for_download()

    assert constructor_env == [
        (EMBEDDING_MODEL_NAME, "1", "1"),
        (EMBEDDING_MODEL_NAME, None, None),
    ]


def test_deserialize_vector_rejects_malformed_bytes():
    with pytest.raises(ValueError, match="byte length"):
        deserialize_vector(b"abc", expected_dimensions=3)


def test_deserialize_vector_rejects_wrong_dimension():
    raw = np.array([0.1, 0.2], dtype=np.float32).tobytes()

    with pytest.raises(ValueError, match="expected 3"):
        deserialize_vector(raw, expected_dimensions=3)
