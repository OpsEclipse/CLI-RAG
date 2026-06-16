import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection


@dataclass(frozen=True)
class OpenTarget:
    file_path: Path
    file_name: str
    location: str
    topic: str
    snippet: str


def get_last_search_target(conn: Connection, result_number: int) -> OpenTarget:
    row = conn.execute(
        """
        SELECT
            documents.path AS file_path,
            documents.file_name AS file_name,
            chunks.location AS location,
            chunks.topic AS topic,
            chunks.text AS snippet
        FROM last_search_results
        JOIN chunks ON chunks.id = last_search_results.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        WHERE last_search_results.result_number = ?
        """,
        (result_number,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No search result found for {result_number}")

    return OpenTarget(
        file_path=Path(row["file_path"]),
        file_name=row["file_name"],
        location=row["location"],
        topic=row["topic"],
        snippet=_compact_snippet(row["snippet"]),
    )


def open_file(path: Path) -> None:
    file_path = str(path)
    if sys.platform == "darwin":
        command = ["open", file_path]
    elif sys.platform.startswith("linux"):
        command = ["xdg-open", file_path]
    elif sys.platform.startswith("win"):
        command = ["cmd", "/c", "start", "", file_path]
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    subprocess.run(command, check=True)


def _compact_snippet(text: str) -> str:
    snippet = re.sub(r"\s+", " ", text).strip()
    if len(snippet) <= 160:
        return snippet
    return f"{snippet[:157]}..."
