from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from crag.cli import app
from crag.db import connect, init_db
from crag.models import SearchResult
from crag.render import render_document_list, render_results, render_status


def seed_document(conn, file_name: str = "week-01.pptx") -> int:
    document = conn.execute(
        """
        INSERT INTO documents(path, file_name, file_type, status)
        VALUES (?, ?, 'pptx', 'ready')
        """,
        (f"/tmp/{file_name}", file_name),
    )
    document_id = int(document.lastrowid)
    item = conn.execute(
        """
        INSERT INTO items(document_id, item_number, item_kind, topic, text)
        VALUES (?, 1, 'slide', 'Elasticity', 'Price elasticity measures responsiveness.')
        """,
        (document_id,),
    )
    item_id = int(item.lastrowid)
    conn.execute(
        """
        INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location)
        VALUES (?, ?, 0, 'Price elasticity measures responsiveness.', 'Elasticity', 'S1')
        """,
        (document_id, item_id),
    )
    conn.commit()
    return document_id


def test_render_results_outputs_table():
    console = Console(record=True, width=120)
    results = [
        SearchResult(
            result_number=1,
            chunk_id=10,
            file_path=Path("/tmp/week-01.pptx"),
            file_name="week-01.pptx",
            location="S1",
            topic="Elasticity",
            snippet="Price elasticity measures responsiveness.",
            score=0.92,
        )
    ]

    render_results(console, "Hybrid Results", results)

    output = console.export_text()
    assert "Hybrid Results" in output
    assert "week-01.pptx" in output
    assert "S1" in output


def test_render_status_outputs_counts_and_offline_search(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)
    console = Console(record=True, width=120)

    render_status(console, conn)

    output = console.export_text()
    assert "Index Status" in output
    assert "Documents" in output
    assert "1" in output
    assert "Chunks" in output
    assert "Search online" in output
    assert "No" in output


def test_render_document_list_stores_last_list_results(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    document_id = seed_document(conn)
    console = Console(record=True, width=120)

    render_document_list(console, conn)

    output = console.export_text()
    rows = conn.execute(
        "SELECT row_number, document_id FROM last_list_results ORDER BY row_number"
    ).fetchall()
    assert "week-01.pptx" in output
    assert [tuple(row) for row in rows] == [(1, document_id)]


def test_render_document_list_errors_renders_errors_without_storing_rows(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)
    conn.execute(
        """
        INSERT INTO ingest_errors(path, error_type, message)
        VALUES ('/tmp/broken.pptx', 'RuntimeError', 'OCR failed')
        """
    )
    conn.commit()
    console = Console(record=True, width=120)

    render_document_list(console, conn, errors=True)

    output = console.export_text()
    rows = conn.execute("SELECT * FROM last_list_results").fetchall()
    assert "broken.pptx" in output
    assert "RuntimeError" in output
    assert "OCR failed" in output
    assert rows == []


def test_cli_status_outputs_index_status(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    seed_document(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Index Status" in result.stdout


def test_cli_list_outputs_files_and_stores_last_list_results(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    document_id = seed_document(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["list"])

    conn = connect(db_path)
    rows = conn.execute(
        "SELECT row_number, document_id FROM last_list_results ORDER BY row_number"
    ).fetchall()
    conn.close()
    assert result.exit_code == 0
    assert "week-01.pptx" in result.stdout
    assert [tuple(row) for row in rows] == [(1, document_id)]
