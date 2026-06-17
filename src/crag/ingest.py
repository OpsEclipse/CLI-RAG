from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from crag.config import EMBEDDING_MODEL_NAME, RAW_OCR_DIR, SUPPORTED_EXTENSIONS
from crag.embeddings import cosine_similarity, embed_texts, serialize_vector


MAX_RAW_OCR_STEM_LENGTH = 60
INGEST_SAVEPOINT = "crag_ingest_file"
MAX_PDF_CHUNK_CHARS = 1400
MIN_PDF_CHUNK_CHARS = 30
PDF_CHUNK_SIMILARITY_THRESHOLD = 0.72
_MARKDOWN_BLOCK_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_REAL_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


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


def chunk_location(
    kind: str, item_number: int, chunk_index: int, chunk_count: int
) -> str:
    base_location = location_for(kind, item_number)
    if kind == "page" and chunk_count > 1:
        return f"{base_location}.{chunk_index + 1}"
    return base_location


def pages_from_ocr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        return []
    return [page for page in pages if isinstance(page, dict)]


def markdown_blocks(markdown: str) -> list[str]:
    blocks = [
        block.strip()
        for block in _MARKDOWN_BLOCK_RE.split(markdown.strip())
        if block.strip()
    ]
    if len(blocks) != 1:
        return blocks

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_RE.findall(blocks[0])
        if sentence.strip()
    ]
    if len(sentences) > 1:
        return sentences
    return blocks


def _would_exceed_chunk_limit(current_blocks: list[str], next_block: str) -> bool:
    if not current_blocks:
        return False
    current_text = "\n\n".join(current_blocks)
    return len(current_text) + len(next_block) + 2 > MAX_PDF_CHUNK_CHARS


def has_real_word(text: str) -> bool:
    return any(len(word) >= 2 for word in _REAL_WORD_RE.findall(text))


def is_noise_chunk(text: str) -> bool:
    stripped = text.strip()
    return not stripped or not has_real_word(stripped)


def join_chunk_texts(chunks: list[str]) -> str:
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def clean_pdf_chunks(chunks: list[str]) -> list[str]:
    cleaned: list[str] = []
    pending_prefix: list[str] = []

    for chunk in chunks:
        stripped = chunk.strip()
        if is_noise_chunk(stripped):
            continue
        if len(stripped) < MIN_PDF_CHUNK_CHARS:
            if cleaned:
                cleaned[-1] = join_chunk_texts([cleaned[-1], stripped])
            else:
                pending_prefix.append(stripped)
            continue
        if pending_prefix:
            stripped = join_chunk_texts([*pending_prefix, stripped])
            pending_prefix = []
        cleaned.append(stripped)

    if pending_prefix:
        if cleaned:
            cleaned[-1] = join_chunk_texts([cleaned[-1], *pending_prefix])
        else:
            cleaned = pending_prefix

    return cleaned


def semantic_pdf_chunks(markdown: str, embedding_model: Any | None) -> list[str]:
    blocks = markdown_blocks(markdown)
    if not blocks:
        return []
    if len(blocks) == 1:
        return clean_pdf_chunks(blocks)
    if embedding_model is None:
        return clean_pdf_chunks(blocks)

    vectors = embed_texts(embedding_model, blocks)
    chunks: list[list[str]] = [[blocks[0]]]
    previous_vector = vectors[0]

    for block, vector in zip(blocks[1:], vectors[1:]):
        similarity = cosine_similarity(previous_vector, vector)
        if (
            similarity >= PDF_CHUNK_SIMILARITY_THRESHOLD
            and not _would_exceed_chunk_limit(chunks[-1], block)
        ):
            chunks[-1].append(block)
        else:
            chunks.append([block])
        previous_vector = vector

    return clean_pdf_chunks(["\n\n".join(chunk_blocks) for chunk_blocks in chunks])


def chunks_for_item(path: Path, text: str, embedding_model: Any | None) -> list[str]:
    if path.suffix.lower() == ".pdf":
        return semantic_pdf_chunks(text, embedding_model)
    return [text]


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
    embedding_model: Any | None = None,
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
    if not conn.in_transaction:
        conn.execute("BEGIN")
    conn.execute(f"SAVEPOINT {INGEST_SAVEPOINT}")
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
        chunk_ids: list[int] = []
        chunk_texts: list[str] = []

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
            item_chunks = chunks_for_item(path, text, embedding_model)
            for chunk_index, chunk_text in enumerate(item_chunks):
                location = chunk_location(
                    kind, item_number, chunk_index, len(item_chunks)
                )
                chunk_cursor = conn.execute(
                    """
                    INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (document_id, item_id, chunk_index, chunk_text, topic, location),
                )
                chunk_id = int(chunk_cursor.lastrowid)
                insert_fts(conn, chunk_id, chunk_text, topic, path.name)
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)

        if embedding_model is not None and chunk_texts:
            vectors = embed_texts(embedding_model, chunk_texts)
            for chunk_id, vector in zip(chunk_ids, vectors):
                conn.execute(
                    """
                    INSERT INTO embeddings(chunk_id, model_name, vector)
                    VALUES (?, ?, ?)
                    """,
                    (chunk_id, EMBEDDING_MODEL_NAME, serialize_vector(vector)),
                )

        conn.execute(f"RELEASE SAVEPOINT {INGEST_SAVEPOINT}")
        return document_id
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {INGEST_SAVEPOINT}")
        conn.execute(f"RELEASE SAVEPOINT {INGEST_SAVEPOINT}")
        if str(raw_path) not in existing_raw_paths:
            raw_path.unlink(missing_ok=True)
        raise
