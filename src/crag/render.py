from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console
from rich.table import Table

from crag.models import SearchResult


def render_results(console: Console, title: str, results: list[SearchResult]) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Loc")
    table.add_column("Topic")
    table.add_column("Match")
    table.add_column("Score", justify="right")

    for result in results:
        table.add_row(
            str(result.result_number),
            result.file_name,
            result.location,
            result.topic,
            result.snippet,
            f"{result.score:.3f}",
        )

    console.print(table)


def render_status(console: Console, conn: sqlite3.Connection) -> None:
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    item_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    last_ingested = conn.execute("SELECT MAX(updated_at) FROM documents").fetchone()[0]
    embedding_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    semantic_index = "Available" if embedding_count > 0 else "Unavailable"

    table = Table(title="Index Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Documents", str(document_count))
    table.add_row("Items", str(item_count))
    table.add_row("Chunks", str(chunk_count))
    table.add_row("Last ingested", str(last_ingested) if last_ingested else "Never")
    table.add_row("Semantic index", semantic_index)
    table.add_row("Search online", "No")

    console.print(table)


def render_document_list(
    console: Console, conn: sqlite3.Connection, errors: bool = False
) -> None:
    conn.execute("DELETE FROM last_list_results")
    conn.commit()

    if errors:
        _render_error_list(console, conn)
        return

    rows = conn.execute(
        """
        SELECT
            documents.id,
            documents.file_name,
            documents.file_type,
            documents.updated_at,
            documents.status,
            COUNT(items.id) AS item_count
        FROM documents
        LEFT JOIN items ON items.document_id = documents.id
        GROUP BY documents.id
        ORDER BY documents.file_name
        """
    ).fetchall()

    conn.executemany(
        """
        INSERT INTO last_list_results(row_number, document_id)
        VALUES (?, ?)
        """,
        [(index, int(row["id"])) for index, row in enumerate(rows, start=1)],
    )
    conn.commit()

    table = Table(title="Documents")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Type")
    table.add_column("Items", justify="right")
    table.add_column("Last Indexed")
    table.add_column("Status")

    for index, row in enumerate(rows, start=1):
        table.add_row(
            str(index),
            str(row["file_name"]),
            str(row["file_type"]),
            str(row["item_count"]),
            str(row["updated_at"]),
            str(row["status"]),
        )

    console.print(table)


def _render_error_list(console: Console, conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT path, error_type, message
        FROM ingest_errors
        ORDER BY created_at DESC, id DESC
        """
    ).fetchall()

    table = Table(title="Ingest Errors")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Error")
    table.add_column("Message")

    for index, row in enumerate(rows, start=1):
        table.add_row(
            str(index),
            Path(str(row["path"])).name,
            str(row["error_type"]),
            str(row["message"]),
        )

    console.print(table)
