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
