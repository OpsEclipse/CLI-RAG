from __future__ import annotations

import sqlite3


def delete_document_by_path(conn: sqlite3.Connection, path: str) -> bool:
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
    if row is None:
        return False

    delete_document_by_id(conn, int(row["id"]))
    return True


def delete_document_by_list_row(conn: sqlite3.Connection, row_number: int) -> bool:
    row = conn.execute(
        """
        SELECT document_id
        FROM last_list_results
        WHERE row_number = ?
        """,
        (row_number,),
    ).fetchone()
    if row is None:
        return False

    delete_document_by_id(conn, int(row["document_id"]))
    return True


def delete_document_by_id(conn: sqlite3.Connection, document_id: int) -> None:
    conn.execute(
        """
        DELETE FROM chunk_fts
        WHERE rowid IN (
            SELECT id
            FROM chunks
            WHERE document_id = ?
        )
        """,
        (document_id,),
    )
    conn.execute(
        """
        DELETE FROM last_search_results
        WHERE chunk_id IN (
            SELECT id
            FROM chunks
            WHERE document_id = ?
        )
        """,
        (document_id,),
    )
    conn.execute(
        "DELETE FROM last_list_results WHERE document_id = ?",
        (document_id,),
    )
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()


def clear_index(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunk_fts")
    conn.execute("DELETE FROM last_search_results")
    conn.execute("DELETE FROM last_list_results")
    conn.execute("DELETE FROM ingest_errors")
    conn.execute("DELETE FROM documents")
    conn.commit()
