from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from crag.cli import app
from crag.db import connect, init_db
from crag.embeddings import serialize_vector
from crag.openers import OpenTarget, get_last_search_target, open_file


def test_get_last_search_target_returns_location(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    doc = conn.execute(
        "INSERT INTO documents(path, file_name, file_type, status) VALUES ('/tmp/week-01.pptx', 'week-01.pptx', 'pptx', 'ready')"
    )
    document_id = int(doc.lastrowid)
    item = conn.execute(
        "INSERT INTO items(document_id, item_number, item_kind, topic, text) VALUES (?, 1, 'slide', 'Elasticity', 'Price elasticity')",
        (document_id,),
    )
    item_id = int(item.lastrowid)
    chunk = conn.execute(
        "INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location) VALUES (?, ?, 0, 'Price elasticity', 'Elasticity', 'S1')",
        (document_id, item_id),
    )
    chunk_id = int(chunk.lastrowid)
    conn.execute(
        "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, 'test', ?)",
        (chunk_id, serialize_vector(np.array([1, 0], dtype=np.float32))),
    )
    conn.execute(
        "INSERT INTO last_search_results(result_number, chunk_id, mode, score) VALUES (1, ?, 'hybrid', 0.9)",
        (chunk_id,),
    )
    conn.commit()

    target = get_last_search_target(conn, 1)

    assert target.file_name == "week-01.pptx"
    assert target.location == "S1"
    assert target.topic == "Elasticity"


def test_get_last_search_target_raises_for_missing_result_number(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)

    with pytest.raises(ValueError, match="No search result found for 9"):
        get_last_search_target(conn, 9)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", "open"),
        ("linux", "xdg-open"),
    ],
)
def test_open_file_chooses_platform_command(monkeypatch, tmp_path, platform, expected):
    calls = []
    source_path = tmp_path / "week-01.pptx"
    source_path.write_text("slides")

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr("subprocess.run", fake_run)

    open_file(source_path)

    assert calls == [([expected, str(source_path)], True)]


def test_open_file_raises_before_platform_opener_when_source_missing(
    monkeypatch, tmp_path
):
    calls = []
    missing_path = tmp_path / "missing.pptx"

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(FileNotFoundError, match="Source file does not exist"):
        open_file(missing_path)

    assert calls == []


def test_open_file_uses_startfile_on_windows(monkeypatch, tmp_path):
    calls = []
    source_path = tmp_path / "week & 01.pptx"
    source_path.write_text("slides")

    def fake_startfile(path):
        calls.append(path)

    def fake_run(command, check):
        raise AssertionError("subprocess.run should not be used on Windows")

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("os.startfile", fake_startfile, raising=False)
    monkeypatch.setattr("subprocess.run", fake_run)

    open_file(source_path)

    assert calls == [str(source_path)]


def test_open_file_raises_for_unknown_platform(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr("sys.platform", "plan9")

    with pytest.raises(RuntimeError, match="Unsupported platform"):
        open_file(Path("/tmp/week-01.pptx"))


def test_cli_open_calls_opener_and_prints_target(monkeypatch, tmp_path):
    import crag.config
    import crag.openers

    target = OpenTarget(
        file_path=Path("/tmp/week-01.pptx"),
        file_name="week-01.pptx",
        location="S1",
        topic="Elasticity",
        snippet="Price elasticity",
    )
    calls = []

    def fake_get_last_search_target(conn, result_number):
        calls.append(("get", result_number))
        return target

    def fake_open_file(path):
        calls.append(("open", path))

    monkeypatch.setattr(crag.config, "DB_PATH", tmp_path / "crag.db")
    monkeypatch.setattr(crag.openers, "get_last_search_target", fake_get_last_search_target)
    monkeypatch.setattr(crag.openers, "open_file", fake_open_file)

    result = CliRunner().invoke(app, ["open", "1"])

    assert result.exit_code == 0
    assert calls == [("get", 1), ("open", Path("/tmp/week-01.pptx"))]
    assert "Opened: /tmp/week-01.pptx" in result.stdout
    assert "Go to: S1" in result.stdout
    assert "Topic: Elasticity" in result.stdout
    assert 'Match: "Price elasticity"' in result.stdout


def test_cli_open_missing_source_exits_cleanly(monkeypatch, tmp_path):
    import crag.config
    import crag.openers

    target = OpenTarget(
        file_path=tmp_path / "missing.pptx",
        file_name="missing.pptx",
        location="S1",
        topic="Elasticity",
        snippet="Price elasticity",
    )

    def fake_get_last_search_target(conn, result_number):
        return target

    monkeypatch.setattr(crag.config, "DB_PATH", tmp_path / "crag.db")
    monkeypatch.setattr(crag.openers, "get_last_search_target", fake_get_last_search_target)

    result = CliRunner().invoke(app, ["open", "1"])

    assert result.exit_code == 1
    assert "Source file does not exist" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
