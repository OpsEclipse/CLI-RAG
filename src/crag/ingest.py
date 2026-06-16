from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from crag.config import RAW_OCR_DIR, SUPPORTED_EXTENSIONS


MAX_RAW_OCR_STEM_LENGTH = 60


def _payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip(".-_")
    return (stem or "document")[:MAX_RAW_OCR_STEM_LENGTH]


def scan_supported_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_topic(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or "Untitled"
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "Untitled"


def item_kind_for(path: Path) -> str:
    if path.suffix.lower() == ".pptx":
        return "slide"
    return "page"


def location_for(kind: str, number: int) -> str:
    prefix = "S" if kind == "slide" else "P"
    return f"{prefix}{number}"


def pages_from_ocr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, dict)]


def write_raw_ocr(
    path: Path, payload: dict[str, Any], raw_dir: Path = RAW_OCR_DIR
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    raw_path = (
        raw_dir / f"{_safe_stem(path)}-{source_digest}-{_payload_digest(payload)}.json"
    )
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return raw_path


def insert_fts(
    conn: sqlite3.Connection, chunk_id: int, text: str, topic: str, file_name: str
) -> None:
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, ?, ?, ?)",
        (chunk_id, text, topic, file_name),
    )


def ingest_file(
    conn: sqlite3.Connection,
    path: Path,
    ocr_client: Any,
    raw_dir: Path = RAW_OCR_DIR,
) -> int:
    source_path = path.resolve()
    payload = ocr_client.parse_file(path)
    path_text = str(source_path)
    existing_raw_paths = {
        str(row["raw_ocr_path"])
        for row in conn.execute(
            "SELECT raw_ocr_path FROM documents WHERE path = ? AND raw_ocr_path IS NOT NULL",
            (path_text,),
        )
    }
    raw_path = write_raw_ocr(source_path, payload, raw_dir)
    try:
        old_chunk_ids = [
            int(row["id"])
            for row in conn.execute(
                """
                SELECT chunks.id
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE documents.path = ?
                """,
                (path_text,),
            )
        ]
        for chunk_id in old_chunk_ids:
            conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
        conn.execute("DELETE FROM documents WHERE path = ?", (path_text,))

        document_cursor = conn.execute(
            """
            INSERT INTO documents(path, file_name, file_type, raw_ocr_path, status)
            VALUES (?, ?, ?, ?, 'ready')
            """,
            (path_text, path.name, path.suffix.lower().lstrip("."), str(raw_path)),
        )
        document_id = int(document_cursor.lastrowid)
        kind = item_kind_for(path)

        for page in pages_from_ocr(payload):
            item_number = int(page.get("index", 0)) + 1
            text = str(page.get("markdown", "")).strip()
            topic = extract_topic(text)
            item_cursor = conn.execute(
                """
                INSERT INTO items(document_id, item_number, item_kind, topic, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, item_number, kind, topic, text),
            )
            item_id = int(item_cursor.lastrowid)
            location = location_for(kind, item_number)
            chunk_cursor = conn.execute(
                """
                INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location)
                VALUES (?, ?, 0, ?, ?, ?)
                """,
                (document_id, item_id, text, topic, location),
            )
            chunk_id = int(chunk_cursor.lastrowid)
            insert_fts(conn, chunk_id, text, topic, path.name)

        conn.commit()
        return document_id
    except Exception:
        conn.rollback()
        if str(raw_path) not in existing_raw_paths:
            raw_path.unlink(missing_ok=True)
        raise
