import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crag.cli import app
from crag.db import connect, init_db
from crag.ingest import ingest_file, scan_supported_files, write_raw_ocr


class FakeOcrClient:
    def __init__(self, payload_path: Path):
        self.payload_path = payload_path

    def parse_file(self, path: Path) -> dict:
        return json.loads(self.payload_path.read_text())


class PayloadOcrClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def parse_file(self, path: Path) -> dict:
        return self.payload


class SequenceOcrClient:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.index = 0

    def parse_file(self, path: Path) -> dict:
        payload = self.payloads[self.index]
        self.index += 1
        return payload


def test_scan_supported_files_skips_unknown_extensions(temp_course_dir):
    pptx = temp_course_dir / "week-01.pptx"
    txt = temp_course_dir / "notes.txt"
    pptx.write_text("fake")
    txt.write_text("skip")

    files = scan_supported_files(temp_course_dir)

    assert files == [pptx]


def test_write_raw_ocr_keeps_long_source_filename_short(tmp_path):
    raw_dir = tmp_path / "raw"
    source = tmp_path / f"{'very-long-source-name-' * 20}.pptx"

    raw_path = write_raw_ocr(source, {"pages": []}, raw_dir)

    assert len(raw_path.name) <= 120
    assert raw_path.suffix == ".json"


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


def test_reingest_same_file_replaces_old_rows_and_fts(tmp_path, temp_course_dir):
    db_path = tmp_path / "crag.db"
    raw_dir = tmp_path / "raw"
    conn = connect(db_path)
    init_db(conn)
    source = temp_course_dir / "week-01.pptx"
    source.write_text("fake")
    first_payload = json.loads(Path("tests/fixtures/mistral_ocr_pptx.json").read_text())
    second_payload = {
        "pages": [
            {
                "index": 0,
                "markdown": "# Updated Topic\n\nOnly the current content should remain.",
            }
        ]
    }

    ingest_file(conn, source, PayloadOcrClient(first_payload), raw_dir)
    ingest_file(conn, source, PayloadOcrClient(second_payload), raw_dir)

    docs = conn.execute("SELECT * FROM documents").fetchall()
    items = conn.execute("SELECT * FROM items").fetchall()
    chunks = conn.execute("SELECT * FROM chunks").fetchall()
    fts_rows = conn.execute("SELECT text FROM chunk_fts").fetchall()

    assert len(docs) == 1
    assert len(items) == 1
    assert len(chunks) == 1
    assert [row["text"] for row in fts_rows] == [
        "# Updated Topic\n\nOnly the current content should remain."
    ]


def test_failed_reingest_does_not_overwrite_existing_raw_ocr(
    tmp_path, temp_course_dir
):
    db_path = tmp_path / "crag.db"
    raw_dir = tmp_path / "raw"
    conn = connect(db_path)
    init_db(conn)
    source = temp_course_dir / "week-01.pptx"
    source.write_text("fake")
    first_payload = json.loads(Path("tests/fixtures/mistral_ocr_pptx.json").read_text())
    invalid_payload = {
        "pages": [
            {
                "index": "not-a-number",
                "markdown": "# Corrupted Replacement\n\nThis should not overwrite raw OCR.",
            }
        ]
    }
    client = SequenceOcrClient([first_payload, invalid_payload])

    ingest_file(conn, source, client, raw_dir)
    existing_raw_path = Path(
        conn.execute("SELECT raw_ocr_path FROM documents").fetchone()["raw_ocr_path"]
    )
    existing_raw_text = existing_raw_path.read_text()

    with pytest.raises(ValueError):
        ingest_file(conn, source, client, raw_dir)
    conn.rollback()

    assert existing_raw_path.read_text() == existing_raw_text
    assert len(list(raw_dir.glob("*.json"))) == 1


def test_ingest_cli_requires_mistral_api_key(monkeypatch, temp_course_dir):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", str(temp_course_dir)])

    assert result.exit_code != 0
    assert "Set MISTRAL_API_KEY" in result.output


def test_ingest_cli_missing_path_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    runner = CliRunner()
    missing_path = tmp_path / "missing"

    result = runner.invoke(app, ["ingest", str(missing_path)])

    assert result.exit_code != 0
    assert "Input path does not exist" in result.output


def test_ingest_cli_exits_nonzero_when_all_supported_files_fail(
    monkeypatch, tmp_path, temp_course_dir
):
    import crag.db as db_module

    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    source = temp_course_dir / "week-01.pptx"
    source.write_text("fake")

    class FailingMistralOcrClient:
        def __init__(self, api_key: str):
            pass

        def parse_file(self, path: Path) -> dict:
            raise RuntimeError("OCR failed")

    fake_ocr_module = types.SimpleNamespace(MistralOcrClient=FailingMistralOcrClient)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "crag.ocr", fake_ocr_module)
    monkeypatch.setattr(db_module, "connect", lambda: conn)
    monkeypatch.setattr(db_module, "ensure_app_dirs", lambda: None)
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", str(temp_course_dir)])
    error_count = conn.execute("SELECT COUNT(*) FROM ingest_errors").fetchone()[0]

    assert result.exit_code == 1
    assert "Ingested 0 file(s). Failed 1." in result.output
    assert error_count == 1


def test_ingest_cli_exits_nonzero_when_any_supported_file_fails(
    monkeypatch, tmp_path, temp_course_dir
):
    import crag.config as config_module
    import crag.db as db_module

    db_path = tmp_path / "crag.db"
    raw_dir = tmp_path / "raw"
    conn = connect(db_path)
    success = temp_course_dir / "week-01.pptx"
    failure = temp_course_dir / "week-02.pptx"
    success.write_text("fake")
    failure.write_text("fake")

    class PartlyFailingMistralOcrClient:
        def __init__(self, api_key: str):
            pass

        def parse_file(self, path: Path) -> dict:
            if path.name == "week-02.pptx":
                raise RuntimeError("OCR failed")
            return {
                "pages": [
                    {
                        "index": 0,
                        "markdown": "# Successful File\n\nThis file should be ingested.",
                    }
                ]
            }

    fake_ocr_module = types.SimpleNamespace(
        MistralOcrClient=PartlyFailingMistralOcrClient
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "crag.ocr", fake_ocr_module)
    monkeypatch.setattr(config_module, "RAW_OCR_DIR", raw_dir)
    monkeypatch.setattr(db_module, "connect", lambda: conn)
    monkeypatch.setattr(db_module, "ensure_app_dirs", lambda: None)
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", str(temp_course_dir)])
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    error_count = conn.execute("SELECT COUNT(*) FROM ingest_errors").fetchone()[0]

    assert result.exit_code == 1
    assert "Ingested 1 file(s). Failed 1." in result.output
    assert document_count == 1
    assert error_count == 1
