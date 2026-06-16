import types

from typer.testing import CliRunner

from crag.cli import app
from crag.db import connect, init_db


def test_cli_help_shows_commands():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "search" in result.stdout
    assert "open" in result.stdout
    assert "status" in result.stdout
    assert "list" in result.stdout
    assert "delete" in result.stdout


def test_crag_help_shows_exam_workflow_commands():
    runner = CliRunner()

    result = runner.invoke(app, ["help"])

    assert result.exit_code == 0
    assert "CRAG Local Search" in result.stdout
    assert "crag ingest" in result.stdout
    assert "crag search" in result.stdout
    assert "--file" in result.stdout
    assert "crag open 1" in result.stdout
    assert "--keyword" in result.stdout
    assert "--semantic" in result.stdout
    assert "crag tui" in result.stdout
    assert "crag delete" in result.stdout


def test_cli_tui_runs_tui(monkeypatch):
    runner = CliRunner()
    calls = []
    fake_tui = types.SimpleNamespace(run_tui=lambda: calls.append("ran"))
    monkeypatch.setitem(__import__("sys").modules, "crag.tui", fake_tui)

    result = runner.invoke(app, ["tui"])

    assert result.exit_code == 0
    assert calls == ["ran"]


def test_delete_missing_positional_target_prints_message(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "3"])

    assert result.exit_code == 0
    assert "No matching indexed file found." in result.stdout


def test_delete_accepts_all_flag_with_confirmation(tmp_path, monkeypatch):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)
    init_db(conn)
    conn.close()
    monkeypatch.setattr("crag.config.DB_PATH", db_path)
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "--all", "--yes"])

    assert result.exit_code == 0
    assert "Deleted indexed file. Original source file was not removed." in result.stdout
