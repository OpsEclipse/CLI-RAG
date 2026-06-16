from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(path: Path) -> None:
    """Parse files and add them to the local index."""
    typer.echo(f"Ingest command registered for: {path}")


@app.command()
def search(
    query: str,
    keyword: bool = False,
    semantic: bool = False,
    alpha: float = 0.5,
    top: int = 5,
    file: Optional[str] = None,
) -> None:
    """Search the local index."""
    typer.echo(
        f"Search command registered: {query} keyword={keyword} semantic={semantic} alpha={alpha} top={top} file={file}"
    )


@app.command(name="open")
def open_result(result_number: int) -> None:
    """Open a source file from the most recent search."""
    typer.echo(f"Open command registered for result: {result_number}")


@app.command()
def status() -> None:
    """Show local index status."""
    typer.echo("Status command registered")


@app.command(name="list")
def list_documents(errors: bool = False) -> None:
    """List ingested files."""
    typer.echo(f"List command registered: errors={errors}")


@app.command()
def delete(target: Optional[str] = None, all: bool = False, yes: bool = False) -> None:
    """Delete indexed files from the local index."""
    typer.echo(f"Delete command registered: target={target} all={all} yes={yes}")


if __name__ == "__main__":
    app()
