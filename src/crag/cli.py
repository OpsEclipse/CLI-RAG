from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(path: Path) -> None:
    """Parse files and add them to the local index."""
    import os

    if not path.exists():
        raise typer.BadParameter(f"Input path does not exist: {path}")

    from crag.config import RAW_OCR_DIR
    from crag.db import connect, ensure_app_dirs, init_db
    from crag.embeddings import load_model_for_download
    from crag.ingest import ingest_file, scan_supported_files

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set MISTRAL_API_KEY before running ingestion.")

    from crag.ocr import MistralOcrClient

    ensure_app_dirs()
    conn = connect()
    init_db(conn)
    client = MistralOcrClient(api_key)
    embedding_model = load_model_for_download()
    files = scan_supported_files(path)
    ready = 0
    failed = 0

    for file_path in files:
        try:
            ingest_file(
                conn,
                file_path,
                client,
                RAW_OCR_DIR,
                embedding_model=embedding_model,
            )
            conn.commit()
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
    if keyword and semantic:
        typer.echo("Choose only one of --keyword or --semantic.")
        raise typer.Exit(2)
    if alpha < 0.0 or alpha > 1.0:
        raise typer.BadParameter("alpha must be between 0.0 and 1.0")
    if top < 1:
        raise typer.BadParameter("top must be at least 1")

    from crag import config
    from crag.db import connect, init_db
    from crag.render import render_results
    from crag.search import (
        hybrid_search,
        keyword_search,
        save_last_search,
        semantic_search,
    )

    conn = connect(config.DB_PATH)
    init_db(conn)
    console = Console()

    if keyword:
        mode = "keyword"
        title = "Keyword Results"
        results = keyword_search(conn, query, top=top, file_filter=file)
    else:
        from crag.embeddings import embed_texts, load_model

        model = load_model(local_only=True)
        query_vector = embed_texts(model, [query])[0]
        if semantic:
            mode = "semantic"
            title = "Semantic Results"
            results = semantic_search(
                conn, query, query_vector, top=top, file_filter=file
            )
        else:
            mode = "hybrid"
            title = "Hybrid Results"
            results = hybrid_search(
                conn, query, query_vector, alpha=alpha, top=top, file_filter=file
            )

    save_last_search(conn, results, mode=mode)
    render_results(console, title, results)


@app.command(name="open")
def open_result(result_number: int) -> None:
    """Open a source file from the most recent search."""
    from crag import config, openers
    from crag.db import connect, init_db

    conn = connect(config.DB_PATH)
    init_db(conn)
    try:
        target = openers.get_last_search_target(conn, result_number)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    try:
        openers.open_file(target.file_path)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.echo(f"Opened: {target.file_path}")
    typer.echo(f"Go to: {target.location}")
    typer.echo(f"Topic: {target.topic}")
    typer.echo(f'Match: "{target.snippet}"')


@app.command()
def status() -> None:
    """Show local index status."""
    from crag import config
    from crag.db import connect, init_db
    from crag.render import render_status

    conn = connect(config.DB_PATH)
    init_db(conn)
    render_status(Console(), conn)


@app.command(name="list")
def list_documents(errors: bool = False) -> None:
    """List ingested files."""
    from crag import config
    from crag.db import connect, init_db
    from crag.render import render_document_list

    conn = connect(config.DB_PATH)
    init_db(conn)
    render_document_list(Console(), conn, errors=errors)


@app.command()
def delete(
    target: Optional[str] = typer.Argument(None),
    all_documents: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete indexed files from the local index."""
    from crag import config
    from crag.db import connect, init_db
    from crag.delete import (
        clear_index,
        delete_document_by_list_row,
        delete_document_by_path,
    )

    conn = connect(config.DB_PATH)
    init_db(conn)

    if all_documents:
        if target is not None:
            typer.echo("Choose either --all or a target, not both.")
            raise typer.Exit(2)
        if not yes and not typer.confirm("Delete all indexed files?"):
            raise typer.Exit(1)
        clear_index(conn)
        typer.echo("Deleted indexed file. Original source file was not removed.")
        return

    if target is None:
        typer.echo("Pass a row number, a file path, or --all.")
        raise typer.Exit(2)

    if target.isdecimal():
        deleted = delete_document_by_list_row(conn, int(target))
    else:
        deleted = delete_document_by_path(conn, target)
        if not deleted:
            resolved_target = str(Path(target).expanduser().resolve())
            if resolved_target != target:
                deleted = delete_document_by_path(conn, resolved_target)

    if deleted:
        typer.echo("Deleted indexed file. Original source file was not removed.")
    else:
        typer.echo("No matching indexed file found.")


if __name__ == "__main__":
    app()
