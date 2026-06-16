from typer.testing import CliRunner

from crag.cli import app


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


def test_delete_accepts_positional_target():
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "3"])

    assert result.exit_code == 0
    assert "target=3" in result.stdout


def test_delete_accepts_all_flag_with_confirmation():
    runner = CliRunner()

    result = runner.invoke(app, ["delete", "--all", "--yes"])

    assert result.exit_code == 0
    assert "all=True" in result.stdout
    assert "yes=True" in result.stdout
