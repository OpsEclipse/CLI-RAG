import sqlite3

import numpy as np
import pytest
from typer.testing import CliRunner

import crag.search as search_module
from crag.db import connect, init_db
from crag.embeddings import serialize_vector
from crag.search import (
    hybrid_search,
    keyword_search,
    normalize_scores,
    save_last_search,
    semantic_search,
)


def seed_chunk(
    conn: sqlite3.Connection,
    file_name: str,
    text: str,
    topic: str,
    location: str,
    vector: np.ndarray,
) -> int:
    doc = conn.execute(
        "INSERT INTO documents(path, file_name, file_type, status) VALUES (?, ?, 'pptx', 'ready')",
        (f"/tmp/{file_name}", file_name),
    )
    document_id = int(doc.lastrowid)
    item = conn.execute(
        "INSERT INTO items(document_id, item_number, item_kind, topic, text) VALUES (?, 1, 'slide', ?, ?)",
        (document_id, topic, text),
    )
    item_id = int(item.lastrowid)
    chunk = conn.execute(
        "INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location) VALUES (?, ?, 0, ?, ?, ?)",
        (document_id, item_id, text, topic, location),
    )
    chunk_id = int(chunk.lastrowid)
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, ?, ?, ?)",
        (chunk_id, text, topic, file_name),
    )
    conn.execute(
        "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, 'test-model', ?)",
        (chunk_id, serialize_vector(vector)),
    )
    conn.commit()
    return chunk_id


def test_normalize_scores_handles_empty_and_equal_values():
    assert normalize_scores({}) == {}
    assert normalize_scores({1: 4.0, 2: 4.0}) == {1: 1.0, 2: 1.0}
    assert normalize_scores({1: 2.0, 2: 4.0}) == {1: 0.0, 2: 1.0}


def test_keyword_search_finds_matching_chunk(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "week-01.pptx",
        "Price elasticity measures responsiveness.",
        "Elasticity",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )

    results = keyword_search(conn, "elasticity", top=5)

    assert len(results) == 1
    assert results[0].file_name == "week-01.pptx"
    assert results[0].location == "S1"


def test_keyword_search_handles_punctuation_query(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "week-01.pptx",
        "Price elasticity measures responsiveness.",
        "Elasticity",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )

    results = keyword_search(conn, "what is elasticity?", top=5)

    assert [result.file_name for result in results] == ["week-01.pptx"]


def test_keyword_search_returns_empty_for_only_punctuation(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "week-01.pptx",
        "Price elasticity measures responsiveness.",
        "Elasticity",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )

    results = keyword_search(conn, '?!:"', top=5)

    assert results == []


def test_keyword_search_respects_file_filter(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "week-01.pptx",
        "Elasticity appears here.",
        "Elasticity",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )
    seed_chunk(
        conn,
        "week-02.pptx",
        "Elasticity appears here too.",
        "Elasticity",
        "S2",
        np.array([0, 1], dtype=np.float32),
    )

    results = keyword_search(conn, "elasticity", top=5, file_filter="week-02")

    assert [result.file_name for result in results] == ["week-02.pptx"]


def test_hybrid_search_uses_alpha_weighting(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "keyword.pptx",
        "alpha beta beta beta",
        "Keyword",
        "S1",
        np.array([0, 1], dtype=np.float32),
    )
    seed_chunk(
        conn,
        "semantic.pptx",
        "gamma delta",
        "Semantic",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )

    query_vector = np.array([1, 0], dtype=np.float32)
    results = hybrid_search(conn, "beta", query_vector, alpha=0.8, top=2)

    assert results[0].file_name == "semantic.pptx"


def test_hybrid_search_overfetches_candidates_before_combining(tmp_path, monkeypatch):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    first_chunk_id = seed_chunk(
        conn,
        "first.pptx",
        "alpha beta",
        "First",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )
    second_chunk_id = seed_chunk(
        conn,
        "second.pptx",
        "beta gamma",
        "Second",
        "S2",
        np.array([0, 1], dtype=np.float32),
    )
    seen_limits = []

    def fake_keyword_scores(conn, query, file_filter=None, top=20):
        seen_limits.append(("keyword", top))
        return {first_chunk_id: 1.0, second_chunk_id: 0.7}

    def fake_semantic_scores(conn, query_vector, file_filter=None, top=20):
        seen_limits.append(("semantic", top))
        return {first_chunk_id: 0.2, second_chunk_id: 1.0}

    monkeypatch.setattr(search_module, "keyword_scores", fake_keyword_scores)
    monkeypatch.setattr(search_module, "semantic_scores", fake_semantic_scores)

    hybrid_search(conn, "beta", np.array([1, 0], dtype=np.float32), alpha=0.5, top=1)

    assert seen_limits == [("keyword", 20), ("semantic", 20)]


def test_hybrid_search_rejects_alpha_outside_range(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    query_vector = np.array([1, 0], dtype=np.float32)

    with pytest.raises(ValueError, match="alpha"):
        hybrid_search(conn, "beta", query_vector, alpha=1.1)


def test_semantic_search_skips_malformed_vectors(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "broken.pptx",
        "Broken vector row.",
        "Broken",
        "S1",
        np.array([0, 1], dtype=np.float32),
    )
    conn.execute("UPDATE embeddings SET vector = ?", (b"abc",))
    valid_chunk_id = seed_chunk(
        conn,
        "valid.pptx",
        "Valid vector row.",
        "Valid",
        "S2",
        np.array([1, 0], dtype=np.float32),
    )

    results = semantic_search(
        conn, "valid", np.array([1, 0], dtype=np.float32), top=5
    )

    assert [result.chunk_id for result in results] == [valid_chunk_id]


def test_save_last_search_replaces_previous_rows(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "old.pptx",
        "old beta",
        "Old",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )
    current_chunk_id = seed_chunk(
        conn,
        "current.pptx",
        "current beta",
        "Current",
        "S2",
        np.array([0, 1], dtype=np.float32),
    )
    old_results = keyword_search(conn, "old", top=5)
    current_results = keyword_search(conn, "current", top=5)

    save_last_search(conn, old_results, mode="keyword")
    save_last_search(conn, current_results, mode="keyword")

    rows = conn.execute(
        "SELECT result_number, chunk_id, mode FROM last_search_results ORDER BY result_number"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(1, current_chunk_id, "keyword")]


def test_cli_keyword_search_does_not_load_embedding_model(tmp_path, monkeypatch):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(
        conn,
        "week-01.pptx",
        "Elasticity appears here.",
        "Elasticity",
        "S1",
        np.array([1, 0], dtype=np.float32),
    )
    conn.close()

    def fail_load_model(*_args, **_kwargs):
        raise AssertionError("keyword search should not load embeddings")

    monkeypatch.setattr("crag.config.DB_PATH", tmp_path / "crag.db")
    monkeypatch.setattr("crag.embeddings.load_model", fail_load_model)
    runner = CliRunner()

    result = runner.invoke(app_for_test(), ["search", "elasticity", "--keyword"])

    assert result.exit_code == 0
    assert "week-01.pptx" in result.stdout


def test_cli_rejects_keyword_and_semantic_together():
    runner = CliRunner()

    result = runner.invoke(
        app_for_test(), ["search", "elasticity", "--keyword", "--semantic"]
    )

    assert result.exit_code != 0
    assert "Choose only one" in result.stdout


def test_cli_rejects_top_less_than_one():
    runner = CliRunner()

    result = runner.invoke(
        app_for_test(), ["search", "elasticity", "--keyword", "--top", "0"]
    )

    assert result.exit_code != 0
    assert "top must be at least 1" in result.output


def app_for_test():
    from crag.cli import app

    return app
