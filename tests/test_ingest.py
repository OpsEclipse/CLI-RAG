import json
from pathlib import Path

from typer.testing import CliRunner

from crag.cli import app
from crag.db import connect, init_db
from crag.ingest import ingest_file, scan_supported_files


class FakeOcrClient:
    def __init__(self, payload_path: Path):
        self.payload_path = payload_path

    def parse_file(self, path: Path) -> dict:
        return json.loads(self.payload_path.read_text())


def test_scan_supported_files_skips_unknown_extensions(temp_course_dir):
    pptx = temp_course_dir / "week-01.pptx"
    txt = temp_course_dir / "notes.txt"
    pptx.write_text("fake")
    txt.write_text("skip")

    files = scan_supported_files(temp_course_dir)

    assert files == [pptx]


def test_ingest_file_stores_document_items_chunks_and_raw_ocr(tmp_path, temp_course_dir):
    db_path = tmp_path / "crag.db"
    raw_dir = tmp_path / "raw"
    conn = connect(db_path)
    init_db(conn)
    source = temp_course_dir / "week-01.pptx"
    source.write_text("fake")
    fixture = Path("tests/fixtures/mistral_ocr_pptx.json")

    ingest_file(conn, source, FakeOcrClient(fixture), raw_dir)

    docs = conn.execute("SELECT * FROM documents").fetchall()
    items = conn.execute("SELECT * FROM items ORDER BY item_number").fetchall()
    chunks = conn.execute("SELECT * FROM chunks ORDER BY chunk_index").fetchall()
    fts_rows = conn.execute("SELECT rowid FROM chunk_fts").fetchall()

    assert len(docs) == 1
    assert docs[0]["file_name"] == "week-01.pptx"
    assert Path(docs[0]["raw_ocr_path"]).exists()
    assert len(items) == 2
    assert items[0]["item_number"] == 1
    assert items[0]["item_kind"] == "slide"
    assert items[0]["topic"] == "Price Elasticity"
    assert len(chunks) == 2
    assert chunks[0]["location"] == "S1"
    assert len(fts_rows) == 2


def test_ingest_cli_requires_mistral_api_key(monkeypatch, temp_course_dir):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", str(temp_course_dir)])

    assert result.exit_code != 0
    assert "Set MISTRAL_API_KEY" in result.output
