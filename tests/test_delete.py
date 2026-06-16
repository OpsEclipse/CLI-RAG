from pathlib import Path

from typer.testing import CliRunner

from crag.cli import app
from crag.db import connect, init_db
from crag.delete import clear_index, delete_document_by_list_row, delete_document_by_path
from crag.embeddings import serialize_vector


def seed_document(conn, path="/tmp/week-01.pptx"):
    doc = conn.execute(
        "INSERT INTO documents(path, file_name, file_type, status) VALUES (?, 'week-01.pptx', 'pptx', 'ready')",
        (path,),
    )
    document_id = int(doc.lastrowid)
    item = conn.execute(
        "INSERT INTO items(document_id, item_number, item_kind, topic, text) VALUES (?, 1, 'slide', 'Topic', 'Text')",
        (document_id,),
    )
    item_id = int(item.lastrowid)
    chunk = conn.execute(
        "INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location) VALUES (?, ?, 0, 'Text', 'Topic', 'S1')",
        (document_id, item_id),
    )
    chunk_id = int(chunk.lastrowid)
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, 'Text', 'Topic', 'week-01.pptx')",
        (chunk_id,),
    )
    conn.execute(
        "INSERT INTO last_list_results(row_number, document_id) VALUES (1, ?)",
        (document_id,),
    )
    conn.commit()
    return document_id


def test_delete_document_by_path_removes_index_rows(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    deleted = delete_document_by_path(conn, "/tmp/week-01.pptx")

    assert deleted is True
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 0


def test_delete_document_by_list_row(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    deleted = delete_document_by_list_row(conn, 1)

    assert deleted is True
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_clear_index(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    clear_index(conn)

    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ingest_errors").fetchone()[0] == 0


def test_delete_document_by_path_removes_embeddings_and_last_search_rows(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    conn.execute(
        "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, 'test-model', ?)",
        (chunk_id, serialize_vector([1.0, 0.0])),
    )
    conn.execute(
        "INSERT INTO last_search_results(result_number, chunk_id, mode, score) VALUES (1, ?, 'keyword', 1.0)",
        (chunk_id,),
    )
    conn.commit()

    deleted = delete_document_by_path(conn, "/tmp/week-01.pptx")

    assert deleted is True
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM last_search_results").fetchone()[0] == 0


def test_delete_document_by_path_keeps_original_source_file(tmp_path):
    source = tmp_path / "week-01.pptx"
    source.write_text("original file")
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn, path=str(source))

    deleted = delete_document_by_path(conn, str(source))

    assert deleted is True
    assert source.exists()
    assert source.read_text() == "original file"


def test_delete_missing_path_returns_false(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    deleted = delete_document_by_path(conn, "/tmp/missing.pptx")

    assert deleted is False
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_delete_missing_list_row_returns_false(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    deleted = delete_document_by_list_row(conn, 99)

    assert deleted is False
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_cli_delete_all_prompts_unless_yes_is_passed(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    seed_document(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "--all"], input="n\n")

    conn = connect(db_path)
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert result.exit_code == 1
    assert "Delete all indexed files?" in result.stdout
    assert document_count == 1


def test_cli_delete_all_yes_clears_index(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    seed_document(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "--all", "--yes"])

    conn = connect(db_path)
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert result.exit_code == 0
    assert "Deleted indexed file. Original source file was not removed." in result.stdout
    assert document_count == 0


def test_cli_delete_by_row_deletes_most_recent_list_row(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    seed_document(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "1"])

    conn = connect(db_path)
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert result.exit_code == 0
    assert "Deleted indexed file. Original source file was not removed." in result.stdout
    assert document_count == 0


def test_cli_delete_by_path_deletes_path(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    source = tmp_path / "week-01.pptx"
    source.write_text("original file")
    conn = connect(db_path)
    init_db(conn)
    seed_document(conn, path=str(source))
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", str(source)])

    conn = connect(db_path)
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert result.exit_code == 0
    assert "Deleted indexed file. Original source file was not removed." in result.stdout
    assert document_count == 0
    assert source.exists()


def test_cli_delete_by_relative_path_matches_stored_absolute_path(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "crag.db"
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    source = course_dir / "week-01.pptx"
    source.write_text("original file")
    conn = connect(db_path)
    init_db(conn)
    seed_document(conn, path=str(source.resolve()))
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    monkeypatch.chdir(course_dir)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", source.name])

    conn = connect(db_path)
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert result.exit_code == 0
    assert "Deleted indexed file. Original source file was not removed." in result.stdout
    assert document_count == 0
    assert source.exists()


def test_cli_delete_missing_target_prints_message(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", str(Path("/tmp/missing.pptx"))])

    assert result.exit_code == 0
    assert "No matching indexed file found." in result.stdout


def test_cli_delete_missing_arguments_errors_clearly(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete"])

    assert result.exit_code == 2
    assert "Pass a row number, a file path, or --all." in result.stdout
