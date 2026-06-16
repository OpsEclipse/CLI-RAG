import sqlite3

from crag.db import connect, init_db


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)

    init_db(conn)

    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
        ).fetchall()
    }
    assert "documents" in table_names
    assert "items" in table_names
    assert "chunks" in table_names
    assert "chunk_fts" in table_names
    assert "embeddings" in table_names
    assert "last_search_results" in table_names
    assert "last_list_results" in table_names
    assert "ingest_errors" in table_names


def test_foreign_keys_are_enabled(tmp_path):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)

    enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert enabled == 1
