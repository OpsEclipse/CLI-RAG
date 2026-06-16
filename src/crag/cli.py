from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(path: Path) -> None:
    """Parse files and add them to the local index."""
    import os

    if not path.exists():
        raise typer.BadParameter(f"Input path does not exist: {path}")

    from crag.config import RAW_OCR_DIR
    from crag.db import connect, ensure_app_dirs, init_db
    from crag.ingest import ingest_file, scan_supported_files

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set MISTRAL_API_KEY before running ingestion.")

    from crag.ocr import MistralOcrClient

    ensure_app_dirs()
    conn = connect()
    init_db(conn)
    client = MistralOcrClient(api_key)
    files = scan_supported_files(path)
    ready = 0
    failed = 0

    for file_path in files:
        try:
            ingest_file(conn, file_path, client, RAW_OCR_DIR)
            ready += 1
        except Exception as exc:
            failed += 1
            conn.rollback()
            conn.execute(
                "INSERT INTO ingest_errors(path, error_type, message) VALUES (?, ?, ?)",
                (str(file_path), type(exc).__name__, str(exc)),
            )
            conn.commit()

    typer.echo(
        f"Ingested {ready} file(s). Failed {failed}. Skipped unsupported files automatically."
    )
    if failed > 0:
        raise typer.Exit(1)


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
def delete(
    target: Optional[str] = typer.Argument(None),
    all_documents: bool = typer.Option(False, "--all"),
    yes: bool = False,
) -> None:
    """Delete indexed files from the local index."""
    typer.echo(
        f"Delete command registered: target={target} all={all_documents} yes={yes}"
    )


if __name__ == "__main__":
    app()
