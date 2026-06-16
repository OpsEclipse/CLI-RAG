# Local Course Search CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `crag`, a local CLI that ingests course files with Mistral OCR, stores a local index, and searches it offline with hybrid ranking.

**Architecture:** Use a Python package with Typer commands, SQLite storage, SQLite FTS5 keyword search, and local `sentence-transformers` embeddings. Mistral OCR is used only during ingestion. Search never calls the internet.

**Tech Stack:** Python 3.11+, Typer, Rich, SQLite, Mistral SDK, Sentence Transformers, NumPy, Pytest.

---

## File Structure

- Create: `pyproject.toml`
  - Defines package metadata, CLI entry point, and dependencies.
- Create: `README.md`
  - Documents setup, ingestion, offline search, and exam-safe boundaries.
- Create: `src/crag/__init__.py`
  - Package marker.
- Create: `src/crag/cli.py`
  - Typer CLI commands: `ingest`, `search`, `open`, `status`, `list`, `delete`.
- Create: `src/crag/config.py`
  - Local paths and environment settings.
- Create: `src/crag/db.py`
  - SQLite connection, schema, and migrations.
- Create: `src/crag/models.py`
  - Dataclasses shared across modules.
- Create: `src/crag/ocr.py`
  - Mistral OCR client wrapper and fixture-friendly interface.
- Create: `src/crag/ingest.py`
  - File scanning, OCR parsing, chunking, and database writes.
- Create: `src/crag/search.py`
  - Keyword, semantic, and hybrid search.
- Create: `src/crag/embeddings.py`
  - Local embedding model loading and vector math.
- Create: `src/crag/render.py`
  - Rich tables and terminal output.
- Create: `src/crag/openers.py`
  - Opens original files and prints source locations.
- Create: `src/crag/delete.py`
  - Removes indexed documents and clears the index.
- Create: `tests/conftest.py`
  - Test database fixtures.
- Create: `tests/fixtures/mistral_ocr_pptx.json`
  - OCR fixture for a slide deck.
- Create: `tests/fixtures/mistral_ocr_pdf.json`
  - OCR fixture for a PDF.
- Create: `tests/test_db.py`
  - Schema and storage tests.
- Create: `tests/test_ingest.py`
  - File scan, OCR parsing, chunking, and status tests.
- Create: `tests/test_search.py`
  - Keyword, semantic, hybrid, and alpha tests.
- Create: `tests/test_cli.py`
  - CLI command tests.
- Create: `tests/test_delete.py`
  - Delete behavior tests.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/crag/__init__.py`
- Create: `src/crag/cli.py`
- Create: `tests/conftest.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Create the failing CLI smoke test**

```python
# tests/test_cli.py
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
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_cli.py::test_cli_help_shows_commands -v
```

Expected: FAIL because the `crag` package does not exist yet.

- [ ] **Step 3: Add package metadata**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "crag"
version = "0.1.0"
description = "Local course material search CLI"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "mistralai>=1.0.0",
  "numpy>=1.26.0",
  "rich>=13.7.0",
  "sentence-transformers>=3.0.0",
  "typer>=0.12.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
]

[project.scripts]
crag = "crag.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 4: Add a first README**

```markdown
# crag

`crag` is a local course search CLI.

It uses Mistral OCR during ingestion only. Ingestion is the setup step where files are parsed and indexed.

Search works offline after ingestion.

`crag` does not generate answers, summaries, or explanations. It only points to source material.

## Planned commands

```bash
crag ingest ./course-materials
crag search "price sensitivity"
crag search "price sensitivity" --keyword
crag search "price sensitivity" --semantic
crag search "price sensitivity" --alpha 0.7
crag open 1
crag status
crag list
crag list --errors
crag delete 3
crag delete --all
```
```

- [ ] **Step 5: Add the CLI skeleton**

```python
# src/crag/__init__.py
__version__ = "0.1.0"
```

```python
# src/crag/cli.py
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
```

- [ ] **Step 6: Add test fixture setup**

```python
# tests/conftest.py
from pathlib import Path

import pytest


@pytest.fixture
def temp_course_dir(tmp_path: Path) -> Path:
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    return course_dir
```

- [ ] **Step 7: Run the smoke test**

Run:

```bash
pytest tests/test_cli.py::test_cli_help_shows_commands -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md src tests
git commit -m "feat: scaffold crag cli"
```

---

## Task 2: SQLite Schema

**Files:**
- Create: `src/crag/config.py`
- Create: `src/crag/db.py`
- Create: `src/crag/models.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing schema tests**

```python
# tests/test_db.py
import sqlite3

from crag.db import connect, init_db


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)

    init_db(conn)

    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
        ).fetchall()
    }
    assert "documents" in table_names
    assert "items" in table_names
    assert "chunks" in table_names
    assert "chunk_fts" in table_names
    assert "embeddings" in table_names
    assert "last_search_results" in table_names
    assert "last_list_results" in table_names
    assert "ingest_errors" in table_names


def test_foreign_keys_are_enabled(tmp_path):
    db_path = tmp_path / "crag.db"
    conn = connect(db_path)

    enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert enabled == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_db.py -v
```

Expected: FAIL because `crag.db` does not exist.

- [ ] **Step 3: Add config paths**

```python
# src/crag/config.py
from pathlib import Path


APP_DIR = Path.home() / ".crag"
DB_PATH = APP_DIR / "crag.db"
RAW_OCR_DIR = APP_DIR / "raw_ocr"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".webp"}
```

- [ ] **Step 4: Add shared models**

```python
# src/crag/models.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchResult:
    result_number: int
    chunk_id: int
    file_path: Path
    file_name: str
    location: str
    topic: str
    snippet: str
    score: float


@dataclass(frozen=True)
class ParsedItem:
    item_number: int
    item_kind: str
    topic: str
    text: str
```

- [ ] **Step 5: Add database module**

```python
# src/crag/db.py
from pathlib import Path
import sqlite3

from crag.config import APP_DIR, DB_PATH, RAW_OCR_DIR


def ensure_app_dirs() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OCR_DIR.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            raw_ocr_path TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            warning TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            item_number INTEGER NOT NULL,
            item_kind TEXT NOT NULL,
            topic TEXT NOT NULL,
            text TEXT NOT NULL,
            UNIQUE(document_id, item_number, item_kind)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            topic TEXT NOT NULL,
            location TEXT NOT NULL,
            UNIQUE(item_id, chunk_index)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
            text,
            topic,
            file_name,
            content='',
            tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            vector BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS last_search_results (
            result_number INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            mode TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS last_list_results (
            row_number INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ingest_errors (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            error_type TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
```

- [ ] **Step 6: Run schema tests**

Run:

```bash
pytest tests/test_db.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/crag/config.py src/crag/db.py src/crag/models.py tests/test_db.py
git commit -m "feat: add local sqlite schema"
```

---

## Task 3: OCR Parsing and Ingestion

**Files:**
- Create: `src/crag/ocr.py`
- Create: `src/crag/ingest.py`
- Create: `tests/fixtures/mistral_ocr_pptx.json`
- Create: `tests/fixtures/mistral_ocr_pdf.json`
- Create: `tests/test_ingest.py`
- Modify: `src/crag/cli.py`

- [ ] **Step 1: Add OCR fixtures**

```json
// tests/fixtures/mistral_ocr_pptx.json
{
  "pages": [
    {
      "index": 0,
      "markdown": "# Price Elasticity\n\nPrice elasticity measures responsiveness of quantity demanded."
    },
    {
      "index": 1,
      "markdown": "# Opportunity Cost\n\nOpportunity cost is the value of the next best alternative."
    }
  ]
}
```

```json
// tests/fixtures/mistral_ocr_pdf.json
{
  "pages": [
    {
      "index": 0,
      "markdown": "# Demand Curves\n\nA demand curve shows how quantity demanded changes with price."
    }
  ]
}
```

- [ ] **Step 2: Write failing ingestion tests**

```python
# tests/test_ingest.py
import json
from pathlib import Path

from crag.db import connect, init_db
from crag.ingest import ingest_file, scan_supported_files


class FakeOcrClient:
    def __init__(self, payload_path: Path):
        self.payload_path = payload_path

    def parse_file(self, path: Path) -> dict:
        return json.loads(self.payload_path.read_text())


def test_scan_supported_files_skips_unknown_extensions(temp_course_dir):
    pptx = temp_course_dir / "week-01.pptx"
    txt = temp_course_dir / "notes.txt"
    pptx.write_text("fake")
    txt.write_text("skip")

    files = scan_supported_files(temp_course_dir)

    assert files == [pptx]


def test_ingest_file_stores_document_items_chunks_and_raw_ocr(tmp_path, temp_course_dir):
    db_path = tmp_path / "crag.db"
    raw_dir = tmp_path / "raw"
    conn = connect(db_path)
    init_db(conn)
    source = temp_course_dir / "week-01.pptx"
    source.write_text("fake")
    fixture = Path("tests/fixtures/mistral_ocr_pptx.json")

    ingest_file(conn, source, FakeOcrClient(fixture), raw_dir)

    docs = conn.execute("SELECT * FROM documents").fetchall()
    items = conn.execute("SELECT * FROM items ORDER BY item_number").fetchall()
    chunks = conn.execute("SELECT * FROM chunks ORDER BY chunk_index").fetchall()
    fts_rows = conn.execute("SELECT rowid FROM chunk_fts").fetchall()

    assert len(docs) == 1
    assert docs[0]["file_name"] == "week-01.pptx"
    assert Path(docs[0]["raw_ocr_path"]).exists()
    assert len(items) == 2
    assert items[0]["item_number"] == 1
    assert items[0]["item_kind"] == "slide"
    assert items[0]["topic"] == "Price Elasticity"
    assert len(chunks) == 2
    assert chunks[0]["location"] == "S1"
    assert len(fts_rows) == 2
```

- [ ] **Step 3: Run ingestion tests and verify they fail**

Run:

```bash
pytest tests/test_ingest.py -v
```

Expected: FAIL because ingestion modules do not exist.

- [ ] **Step 4: Add OCR client**

```python
# src/crag/ocr.py
from pathlib import Path
from typing import Any

from mistralai import Mistral


class MistralOcrClient:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def parse_file(self, path: Path) -> dict[str, Any]:
        uploaded = self.client.files.upload(
            file={
                "file_name": path.name,
                "content": path.read_bytes(),
            },
            purpose="ocr",
        )
        signed_url = self.client.files.get_signed_url(file_id=uploaded.id)
        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
        )
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)
```

- [ ] **Step 5: Add ingestion code**

```python
# src/crag/ingest.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from crag.config import RAW_OCR_DIR, SUPPORTED_EXTENSIONS


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
    return pages


def write_raw_ocr(path: Path, payload: dict[str, Any], raw_dir: Path = RAW_OCR_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    raw_path = raw_dir / f"{path.stem}-{digest}.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return raw_path


def insert_fts(conn: sqlite3.Connection, chunk_id: int, text: str, topic: str, file_name: str) -> None:
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, ?, ?, ?)",
        (chunk_id, text, topic, file_name),
    )


def ingest_file(conn: sqlite3.Connection, path: Path, ocr_client: Any, raw_dir: Path = RAW_OCR_DIR) -> int:
    payload = ocr_client.parse_file(path)
    raw_path = write_raw_ocr(path, payload, raw_dir)
    file_type = path.suffix.lower().lstrip(".")

    old_chunk_ids = [
        int(row["id"])
        for row in conn.execute(
            """
            SELECT chunks.id
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.path = ?
            """,
            (str(path),),
        )
    ]
    for chunk_id in old_chunk_ids:
        conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM documents WHERE path = ?", (str(path),))
    cur = conn.execute(
        """
        INSERT INTO documents(path, file_name, file_type, raw_ocr_path, status)
        VALUES (?, ?, ?, ?, 'ready')
        """,
        (str(path), path.name, file_type, str(raw_path)),
    )
    document_id = int(cur.lastrowid)
    kind = item_kind_for(path)

    for page in pages_from_ocr(payload):
        item_number = int(page.get("index", 0)) + 1
        text = str(page.get("markdown", "")).strip()
        topic = extract_topic(text)
        item_cur = conn.execute(
            """
            INSERT INTO items(document_id, item_number, item_kind, topic, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, item_number, kind, topic, text),
        )
        item_id = int(item_cur.lastrowid)
        location = location_for(kind, item_number)
        chunk_cur = conn.execute(
            """
            INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location)
            VALUES (?, ?, 0, ?, ?, ?)
            """,
            (document_id, item_id, text, topic, location),
        )
        chunk_id = int(chunk_cur.lastrowid)
        insert_fts(conn, chunk_id, text, topic, path.name)

    conn.commit()
    return document_id
```

- [ ] **Step 6: Wire CLI ingest**

Replace the `ingest` command in `src/crag/cli.py` with:

```python
@app.command()
def ingest(path: Path) -> None:
    """Parse files and add them to the local index."""
    import os

    from rich.console import Console

    from crag.config import RAW_OCR_DIR
    from crag.db import connect, ensure_app_dirs, init_db
    from crag.ingest import ingest_file, scan_supported_files
    from crag.ocr import MistralOcrClient

    console = Console()
    ensure_app_dirs()
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set MISTRAL_API_KEY before running ingestion.")

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
            conn.execute(
                "INSERT INTO ingest_errors(path, error_type, message) VALUES (?, ?, ?)",
                (str(file_path), type(exc).__name__, str(exc)),
            )
            conn.commit()

    console.print(f"Ingested {ready} file(s). Failed {failed}. Skipped unsupported files automatically.")
```

- [ ] **Step 7: Run ingestion tests**

Run:

```bash
pytest tests/test_ingest.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/crag/ocr.py src/crag/ingest.py src/crag/cli.py tests/fixtures tests/test_ingest.py
git commit -m "feat: add OCR ingestion pipeline"
```

---

## Task 4: Local Embeddings

**Files:**
- Create: `src/crag/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing embedding tests**

```python
# tests/test_embeddings.py
import numpy as np
import pytest

from crag.embeddings import cosine_similarity, deserialize_vector, serialize_vector


def test_vector_round_trip():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    restored = deserialize_vector(serialize_vector(vector))

    assert restored.dtype == np.float32
    assert restored.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_cosine_similarity():
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)

    assert cosine_similarity(left, right) == pytest.approx(1.0)
    assert cosine_similarity(left, other) == pytest.approx(0.0)
```

- [ ] **Step 2: Run embedding tests and verify they fail**

Run:

```bash
pytest tests/test_embeddings.py -v
```

Expected: FAIL because `crag.embeddings` does not exist.

- [ ] **Step 3: Add embedding helpers**

```python
# src/crag/embeddings.py
from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from crag.config import EMBEDDING_MODEL_NAME


def load_model(local_only: bool = True) -> SentenceTransformer:
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_model_for_download() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[np.ndarray]:
    vectors = model.encode(texts, normalize_embeddings=True)
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


def serialize_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def deserialize_vector(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
```

- [ ] **Step 4: Run embedding tests**

Run:

```bash
pytest tests/test_embeddings.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crag/embeddings.py tests/test_embeddings.py
git commit -m "feat: add local embedding helpers"
```

---

## Task 5: Search Engine

**Files:**
- Create: `src/crag/search.py`
- Create: `tests/test_search.py`
- Modify: `src/crag/cli.py`

- [ ] **Step 1: Write failing search tests**

```python
# tests/test_search.py
import sqlite3

import numpy as np
import pytest

from crag.db import connect, init_db
from crag.embeddings import serialize_vector
from crag.search import hybrid_search, keyword_search, normalize_scores


def seed_chunk(conn: sqlite3.Connection, file_name: str, text: str, topic: str, location: str, vector: np.ndarray) -> int:
    doc = conn.execute(
        "INSERT INTO documents(path, file_name, file_type, status) VALUES (?, ?, 'pptx', 'ready')",
        (f"/tmp/{file_name}", file_name),
    )
    document_id = int(doc.lastrowid)
    item = conn.execute(
        "INSERT INTO items(document_id, item_number, item_kind, topic, text) VALUES (?, 1, 'slide', ?, ?)",
        (document_id, topic, text),
    )
    item_id = int(item.lastrowid)
    chunk = conn.execute(
        "INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location) VALUES (?, ?, 0, ?, ?, ?)",
        (document_id, item_id, text, topic, location),
    )
    chunk_id = int(chunk.lastrowid)
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, ?, ?, ?)",
        (chunk_id, text, topic, file_name),
    )
    conn.execute(
        "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, 'test-model', ?)",
        (chunk_id, serialize_vector(vector)),
    )
    conn.commit()
    return chunk_id


def test_normalize_scores_handles_empty_and_equal_values():
    assert normalize_scores({}) == {}
    assert normalize_scores({1: 4.0, 2: 4.0}) == {1: 1.0, 2: 1.0}
    assert normalize_scores({1: 2.0, 2: 4.0}) == {1: 0.0, 2: 1.0}


def test_keyword_search_finds_matching_chunk(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(conn, "week-01.pptx", "Price elasticity measures responsiveness.", "Elasticity", "S1", np.array([1, 0], dtype=np.float32))

    results = keyword_search(conn, "elasticity", top=5)

    assert len(results) == 1
    assert results[0].file_name == "week-01.pptx"
    assert results[0].location == "S1"


def test_hybrid_search_uses_alpha_weighting(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_chunk(conn, "keyword.pptx", "alpha beta beta beta", "Keyword", "S1", np.array([0, 1], dtype=np.float32))
    seed_chunk(conn, "semantic.pptx", "gamma delta", "Semantic", "S1", np.array([1, 0], dtype=np.float32))

    query_vector = np.array([1, 0], dtype=np.float32)
    results = hybrid_search(conn, "beta", query_vector, alpha=0.8, top=2)

    assert results[0].file_name == "semantic.pptx"
```

- [ ] **Step 2: Run search tests and verify they fail**

Run:

```bash
pytest tests/test_search.py -v
```

Expected: FAIL because `crag.search` does not exist.

- [ ] **Step 3: Add search engine**

```python
# src/crag/search.py
from __future__ import annotations

import sqlite3

import numpy as np

from crag.embeddings import cosine_similarity, deserialize_vector
from crag.models import SearchResult


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def snippet(text: str, query: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    lowered = compact.lower()
    idx = lowered.find(query.lower())
    if idx == -1:
        return compact[:limit]
    start = max(0, idx - 35)
    end = min(len(compact), idx + len(query) + 70)
    return compact[start:end]


def result_from_chunk(conn: sqlite3.Connection, chunk_id: int, score: float, number: int, query: str) -> SearchResult:
    row = conn.execute(
        """
        SELECT chunks.id AS chunk_id, chunks.text, chunks.topic, chunks.location,
               documents.path, documents.file_name
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    return SearchResult(
        result_number=number,
        chunk_id=int(row["chunk_id"]),
        file_path=row["path"],
        file_name=row["file_name"],
        location=row["location"],
        topic=row["topic"],
        snippet=snippet(row["text"], query),
        score=score,
    )


def keyword_scores(conn: sqlite3.Connection, query: str, file_filter: str | None = None, top: int = 20) -> dict[int, float]:
    params: list[object] = [query]
    sql = """
        SELECT chunk_fts.rowid AS chunk_id, bm25(chunk_fts) * -1 AS score
        FROM chunk_fts
        JOIN chunks ON chunks.id = chunk_fts.rowid
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunk_fts MATCH ?
    """
    if file_filter:
        sql += " AND documents.file_name LIKE ?"
        params.append(f"%{file_filter}%")
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(top)
    return {int(row["chunk_id"]): float(row["score"]) for row in conn.execute(sql, params)}


def semantic_scores(conn: sqlite3.Connection, query_vector: np.ndarray, file_filter: str | None = None, top: int = 20) -> dict[int, float]:
    params: list[object] = []
    sql = """
        SELECT embeddings.chunk_id, embeddings.vector
        FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN documents ON documents.id = chunks.document_id
    """
    if file_filter:
        sql += " WHERE documents.file_name LIKE ?"
        params.append(f"%{file_filter}%")
    scores: dict[int, float] = {}
    for row in conn.execute(sql, params):
        vector = deserialize_vector(row["vector"])
        scores[int(row["chunk_id"])] = cosine_similarity(query_vector, vector)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top])


def keyword_search(conn: sqlite3.Connection, query: str, top: int = 5, file_filter: str | None = None) -> list[SearchResult]:
    scores = keyword_scores(conn, query, file_filter=file_filter, top=top)
    return [
        result_from_chunk(conn, chunk_id, score, number, query)
        for number, (chunk_id, score) in enumerate(scores.items(), start=1)
    ]


def semantic_search(conn: sqlite3.Connection, query: str, query_vector: np.ndarray, top: int = 5, file_filter: str | None = None) -> list[SearchResult]:
    scores = semantic_scores(conn, query_vector, file_filter=file_filter, top=top)
    return [
        result_from_chunk(conn, chunk_id, score, number, query)
        for number, (chunk_id, score) in enumerate(scores.items(), start=1)
    ]


def hybrid_search(conn: sqlite3.Connection, query: str, query_vector: np.ndarray, alpha: float = 0.5, top: int = 5, file_filter: str | None = None) -> list[SearchResult]:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    raw_keyword = keyword_scores(conn, query, file_filter=file_filter, top=top * 4)
    raw_semantic = semantic_scores(conn, query_vector, file_filter=file_filter, top=top * 4)
    keyword = normalize_scores(raw_keyword)
    semantic = normalize_scores(raw_semantic)
    chunk_ids = set(keyword) | set(semantic)
    combined = {
        chunk_id: alpha * semantic.get(chunk_id, 0.0) + (1 - alpha) * keyword.get(chunk_id, 0.0)
        for chunk_id in chunk_ids
    }
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top]
    return [
        result_from_chunk(conn, chunk_id, score, number, query)
        for number, (chunk_id, score) in enumerate(ranked, start=1)
    ]


def save_last_search(conn: sqlite3.Connection, results: list[SearchResult], mode: str) -> None:
    conn.execute("DELETE FROM last_search_results")
    for result in results:
        conn.execute(
            "INSERT INTO last_search_results(result_number, chunk_id, mode, score) VALUES (?, ?, ?, ?)",
            (result.result_number, result.chunk_id, mode, result.score),
        )
    conn.commit()
```

- [ ] **Step 4: Wire CLI search**

Replace the `search` command in `src/crag/cli.py` with:

```python
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
    from rich.console import Console

    from crag.db import connect, init_db
    from crag.embeddings import embed_texts, load_model
    from crag.render import render_results
    from crag.search import hybrid_search, keyword_search, save_last_search, semantic_search

    if keyword and semantic:
        raise typer.BadParameter("Choose only one mode: --keyword or --semantic.")
    if alpha < 0.0 or alpha > 1.0:
        raise typer.BadParameter("--alpha must be between 0.0 and 1.0.")

    console = Console()
    conn = connect()
    init_db(conn)

    if keyword:
        results = keyword_search(conn, query, top=top, file_filter=file)
        save_last_search(conn, results, "keyword")
        render_results(console, "Keyword Results", results)
        return

    try:
        model = load_model(local_only=True)
    except Exception as exc:
        console.print(
            "Semantic model is not available locally. Run ingestion once with internet so Hugging Face can cache the model."
        )
        console.print(f"Model load error: {exc}")
        raise typer.Exit(1) from exc
    query_vector = embed_texts(model, [query])[0]

    if semantic:
        results = semantic_search(conn, query, query_vector, top=top, file_filter=file)
        save_last_search(conn, results, "semantic")
        render_results(console, "Semantic Results", results)
        return

    results = hybrid_search(conn, query, query_vector, alpha=alpha, top=top, file_filter=file)
    save_last_search(conn, results, "hybrid")
    render_results(console, "Hybrid Results", results)
```

- [ ] **Step 5: Run search tests**

Run:

```bash
pytest tests/test_search.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/crag/search.py src/crag/cli.py tests/test_search.py
git commit -m "feat: add local search engine"
```

---

## Task 6: Rich Output, Status, and List

**Files:**
- Create: `src/crag/render.py`
- Create: `tests/test_render.py`
- Modify: `src/crag/cli.py`

- [ ] **Step 1: Write failing render test**

```python
# tests/test_render.py
from pathlib import Path

from rich.console import Console

from crag.models import SearchResult
from crag.render import render_results


def test_render_results_outputs_table():
    console = Console(record=True, width=120)
    results = [
        SearchResult(
            result_number=1,
            chunk_id=10,
            file_path=Path("/tmp/week-01.pptx"),
            file_name="week-01.pptx",
            location="S1",
            topic="Elasticity",
            snippet="Price elasticity measures responsiveness.",
            score=0.92,
        )
    ]

    render_results(console, "Hybrid Results", results)

    output = console.export_text()
    assert "Hybrid Results" in output
    assert "week-01.pptx" in output
    assert "S1" in output
```

- [ ] **Step 2: Run render test and verify it fails**

Run:

```bash
pytest tests/test_render.py -v
```

Expected: FAIL because `crag.render` does not exist.

- [ ] **Step 3: Add render helpers**

```python
# src/crag/render.py
from __future__ import annotations

import sqlite3

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
            f"{result.score:.2f}",
        )
    console.print(table)


def render_status(console: Console, conn: sqlite3.Connection) -> None:
    documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    last_ingested = conn.execute("SELECT MAX(updated_at) FROM documents").fetchone()[0] or "Never"

    table = Table(title="Index Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Documents", str(documents))
    table.add_row("Slides/pages", str(items))
    table.add_row("Text chunks", str(chunks))
    table.add_row("Last ingested", str(last_ingested))
    table.add_row("Semantic index", "Available" if embeddings else "Unavailable")
    table.add_row("Search online", "No")
    console.print(table)


def render_document_list(console: Console, conn: sqlite3.Connection, errors: bool = False) -> None:
    conn.execute("DELETE FROM last_list_results")
    if errors:
        rows = conn.execute(
            "SELECT id, path, error_type, message, created_at FROM ingest_errors ORDER BY created_at DESC"
        ).fetchall()
        table = Table(title="Ingest Errors")
        table.add_column("#", justify="right")
        table.add_column("File")
        table.add_column("Error")
        table.add_column("Message")
        for index, row in enumerate(rows, start=1):
            table.add_row(str(index), row["path"], row["error_type"], row["message"])
        console.print(table)
        return

    rows = conn.execute(
        """
        SELECT documents.id, documents.file_name, documents.file_type, documents.status,
               documents.updated_at, COUNT(items.id) AS item_count
        FROM documents
        LEFT JOIN items ON items.document_id = documents.id
        GROUP BY documents.id
        ORDER BY documents.file_name
        """
    ).fetchall()
    table = Table(title="Ingested Files")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Type")
    table.add_column("Items", justify="right")
    table.add_column("Last Indexed")
    table.add_column("Status")
    for index, row in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO last_list_results(row_number, document_id) VALUES (?, ?)",
            (index, row["id"]),
        )
        table.add_row(
            str(index),
            row["file_name"],
            row["file_type"].upper(),
            str(row["item_count"]),
            row["updated_at"],
            row["status"],
        )
    conn.commit()
    console.print(table)
```

- [ ] **Step 4: Wire CLI status and list**

Replace `status` and `list_documents` in `src/crag/cli.py` with:

```python
@app.command()
def status() -> None:
    """Show local index status."""
    from rich.console import Console

    from crag.db import connect, init_db
    from crag.render import render_status

    conn = connect()
    init_db(conn)
    render_status(Console(), conn)


@app.command(name="list")
def list_documents(errors: bool = False) -> None:
    """List ingested files."""
    from rich.console import Console

    from crag.db import connect, init_db
    from crag.render import render_document_list

    conn = connect()
    init_db(conn)
    render_document_list(Console(), conn, errors=errors)
```

- [ ] **Step 5: Run render tests**

Run:

```bash
pytest tests/test_render.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/crag/render.py src/crag/cli.py tests/test_render.py
git commit -m "feat: add rich terminal output"
```

---

## Task 7: Open Source Files

**Files:**
- Create: `src/crag/openers.py`
- Create: `tests/test_openers.py`
- Modify: `src/crag/cli.py`

- [ ] **Step 1: Write failing opener test**

```python
# tests/test_openers.py
import numpy as np

from crag.db import connect, init_db
from crag.embeddings import serialize_vector
from crag.openers import get_last_search_target


def test_get_last_search_target_returns_location(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    doc = conn.execute(
        "INSERT INTO documents(path, file_name, file_type, status) VALUES ('/tmp/week-01.pptx', 'week-01.pptx', 'pptx', 'ready')"
    )
    document_id = int(doc.lastrowid)
    item = conn.execute(
        "INSERT INTO items(document_id, item_number, item_kind, topic, text) VALUES (?, 1, 'slide', 'Elasticity', 'Price elasticity')",
        (document_id,),
    )
    item_id = int(item.lastrowid)
    chunk = conn.execute(
        "INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location) VALUES (?, ?, 0, 'Price elasticity', 'Elasticity', 'S1')",
        (document_id, item_id),
    )
    chunk_id = int(chunk.lastrowid)
    conn.execute(
        "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, 'test', ?)",
        (chunk_id, serialize_vector(np.array([1, 0], dtype=np.float32))),
    )
    conn.execute(
        "INSERT INTO last_search_results(result_number, chunk_id, mode, score) VALUES (1, ?, 'hybrid', 0.9)",
        (chunk_id,),
    )
    conn.commit()

    target = get_last_search_target(conn, 1)

    assert target.file_name == "week-01.pptx"
    assert target.location == "S1"
    assert target.topic == "Elasticity"
```

- [ ] **Step 2: Run opener test and verify it fails**

Run:

```bash
pytest tests/test_openers.py -v
```

Expected: FAIL because `crag.openers` does not exist.

- [ ] **Step 3: Add opener module**

```python
# src/crag/openers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import subprocess
import sys


@dataclass(frozen=True)
class OpenTarget:
    file_path: Path
    file_name: str
    location: str
    topic: str
    snippet: str


def get_last_search_target(conn: sqlite3.Connection, result_number: int) -> OpenTarget:
    row = conn.execute(
        """
        SELECT documents.path, documents.file_name, chunks.location, chunks.topic, chunks.text
        FROM last_search_results
        JOIN chunks ON chunks.id = last_search_results.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        WHERE last_search_results.result_number = ?
        """,
        (result_number,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No recent search result numbered {result_number}.")
    return OpenTarget(
        file_path=Path(row["path"]),
        file_name=row["file_name"],
        location=row["location"],
        topic=row["topic"],
        snippet=" ".join(row["text"].split())[:160],
    )


def open_file(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
        return
    if sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
        return
    if sys.platform.startswith("win"):
        subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
        return
    raise RuntimeError(f"Unsupported platform: {sys.platform}")
```

- [ ] **Step 4: Wire CLI open**

Replace `open_result` in `src/crag/cli.py` with:

```python
@app.command(name="open")
def open_result(result_number: int) -> None:
    """Open a source file from the most recent search."""
    from rich.console import Console

    from crag.db import connect, init_db
    from crag.openers import get_last_search_target, open_file

    console = Console()
    conn = connect()
    init_db(conn)
    target = get_last_search_target(conn, result_number)
    open_file(target.file_path)
    console.print(f"Opened: {target.file_path}")
    console.print(f"Go to: {target.location}")
    console.print(f"Topic: {target.topic}")
    console.print(f'Match: "{target.snippet}"')
```

- [ ] **Step 5: Run opener tests**

Run:

```bash
pytest tests/test_openers.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/crag/openers.py src/crag/cli.py tests/test_openers.py
git commit -m "feat: open source files from search results"
```

---

## Task 8: Delete Indexed Files

**Files:**
- Create: `src/crag/delete.py`
- Create: `tests/test_delete.py`
- Modify: `src/crag/cli.py`

- [ ] **Step 1: Write failing delete tests**

```python
# tests/test_delete.py
from crag.db import connect, init_db
from crag.delete import clear_index, delete_document_by_list_row, delete_document_by_path


def seed_document(conn, path="/tmp/week-01.pptx"):
    doc = conn.execute(
        "INSERT INTO documents(path, file_name, file_type, status) VALUES (?, 'week-01.pptx', 'pptx', 'ready')",
        (path,),
    )
    document_id = int(doc.lastrowid)
    item = conn.execute(
        "INSERT INTO items(document_id, item_number, item_kind, topic, text) VALUES (?, 1, 'slide', 'Topic', 'Text')",
        (document_id,),
    )
    item_id = int(item.lastrowid)
    chunk = conn.execute(
        "INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location) VALUES (?, ?, 0, 'Text', 'Topic', 'S1')",
        (document_id, item_id),
    )
    chunk_id = int(chunk.lastrowid)
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, 'Text', 'Topic', 'week-01.pptx')",
        (chunk_id,),
    )
    conn.execute(
        "INSERT INTO last_list_results(row_number, document_id) VALUES (1, ?)",
        (document_id,),
    )
    conn.commit()
    return document_id


def test_delete_document_by_path_removes_index_rows(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    deleted = delete_document_by_path(conn, "/tmp/week-01.pptx")

    assert deleted is True
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 0


def test_delete_document_by_list_row(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    deleted = delete_document_by_list_row(conn, 1)

    assert deleted is True
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_clear_index(tmp_path):
    conn = connect(tmp_path / "crag.db")
    init_db(conn)
    seed_document(conn)

    clear_index(conn)

    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ingest_errors").fetchone()[0] == 0
```

- [ ] **Step 2: Run delete tests and verify they fail**

Run:

```bash
pytest tests/test_delete.py -v
```

Expected: FAIL because `crag.delete` does not exist.

- [ ] **Step 3: Add delete module**

```python
# src/crag/delete.py
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
        "SELECT document_id FROM last_list_results WHERE row_number = ?",
        (row_number,),
    ).fetchone()
    if row is None:
        return False
    delete_document_by_id(conn, int(row["document_id"]))
    return True


def delete_document_by_id(conn: sqlite3.Connection, document_id: int) -> None:
    chunk_ids = [
        int(row["id"])
        for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
    ]
    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.execute(
        "DELETE FROM last_search_results WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    )
    conn.execute(
        "DELETE FROM last_list_results WHERE document_id NOT IN (SELECT id FROM documents)"
    )
    conn.commit()


def clear_index(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunk_fts")
    conn.execute("DELETE FROM ingest_errors")
    conn.execute("DELETE FROM last_search_results")
    conn.execute("DELETE FROM last_list_results")
    conn.execute("DELETE FROM documents")
    conn.commit()
```

- [ ] **Step 4: Wire CLI delete**

Replace `delete` in `src/crag/cli.py` with:

```python
@app.command()
def delete(target: Optional[str] = None, all: bool = False, yes: bool = False) -> None:
    """Delete indexed files from the local index."""
    from rich.console import Console

    from crag.db import connect, init_db
    from crag.delete import clear_index, delete_document_by_list_row, delete_document_by_path

    console = Console()
    conn = connect()
    init_db(conn)

    if all:
        if not yes and not typer.confirm("Clear the whole local index?"):
            console.print("Cancelled.")
            return
        clear_index(conn)
        console.print("Cleared the local index.")
        return

    if target is None:
        raise typer.BadParameter("Pass a list row, a file path, or --all.")

    if target.isdigit():
        deleted = delete_document_by_list_row(conn, int(target))
    else:
        deleted = delete_document_by_path(conn, target)

    if deleted:
        console.print("Deleted indexed file. Original source file was not removed.")
    else:
        console.print("No matching indexed file found.")
```

- [ ] **Step 5: Run delete tests**

Run:

```bash
pytest tests/test_delete.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/crag/delete.py src/crag/cli.py tests/test_delete.py
git commit -m "feat: delete indexed course files"
```

---

## Task 9: Semantic Index During Ingestion

**Files:**
- Modify: `src/crag/ingest.py`
- Modify: `src/crag/cli.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Add failing semantic indexing test**

Append to `tests/test_ingest.py`:

```python
import numpy as np


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]


def test_ingest_file_can_store_embeddings(tmp_path, temp_course_dir):
    db_path = tmp_path / "crag.db"
    raw_dir = tmp_path / "raw"
    conn = connect(db_path)
    init_db(conn)
    source = temp_course_dir / "week-01.pptx"
    source.write_text("fake")
    fixture = Path("tests/fixtures/mistral_ocr_pptx.json")

    ingest_file(conn, source, FakeOcrClient(fixture), raw_dir, embedding_model=FakeEmbeddingModel())

    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
pytest tests/test_ingest.py::test_ingest_file_can_store_embeddings -v
```

Expected: FAIL because `ingest_file` does not accept `embedding_model`.

- [ ] **Step 3: Update ingestion to store embeddings**

Change the function signature in `src/crag/ingest.py`:

```python
def ingest_file(
    conn: sqlite3.Connection,
    path: Path,
    ocr_client: Any,
    raw_dir: Path = RAW_OCR_DIR,
    embedding_model: Any | None = None,
) -> int:
```

Add this import:

```python
from crag.config import EMBEDDING_MODEL_NAME, RAW_OCR_DIR, SUPPORTED_EXTENSIONS
from crag.embeddings import embed_texts, serialize_vector
```

Inside `ingest_file`, collect chunk ids and texts:

```python
    chunk_ids: list[int] = []
    chunk_texts: list[str] = []
```

After each chunk insert, append:

```python
        chunk_ids.append(chunk_id)
        chunk_texts.append(text)
```

Before `conn.commit()`, add:

```python
    if embedding_model is not None and chunk_texts:
        vectors = embed_texts(embedding_model, chunk_texts)
        for chunk_id, vector in zip(chunk_ids, vectors):
            conn.execute(
                "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, ?, ?)",
                (chunk_id, EMBEDDING_MODEL_NAME, serialize_vector(vector)),
            )
```

- [ ] **Step 4: Update CLI ingest to load/download the model**

Inside `ingest` in `src/crag/cli.py`, add:

```python
    from crag.embeddings import load_model_for_download
```

Before the file loop, add:

```python
    embedding_model = load_model_for_download()
```

Change:

```python
            ingest_file(conn, file_path, client, RAW_OCR_DIR)
```

to:

```python
            ingest_file(conn, file_path, client, RAW_OCR_DIR, embedding_model=embedding_model)
```

- [ ] **Step 5: Run ingestion tests**

Run:

```bash
pytest tests/test_ingest.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/crag/ingest.py src/crag/cli.py tests/test_ingest.py
git commit -m "feat: build semantic index during ingestion"
```

---

## Task 10: End-to-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with setup and offline rules**

```markdown
# crag

`crag` is a local course search CLI.

It uses Mistral OCR during ingestion only. Ingestion is the setup step where files are parsed and indexed.

Search works offline after ingestion.

`crag` does not generate answers, summaries, or explanations. It only points to source material.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Ingest

```bash
export MISTRAL_API_KEY="your-key"
crag ingest ./course-materials
```

Ingestion may use the internet.

It also downloads the local embedding model from Hugging Face if it is not already cached.

## Search

```bash
crag search "price sensitivity"
crag search "price sensitivity" --keyword
crag search "price sensitivity" --semantic
crag search "price sensitivity" --alpha 0.7
```

Search must work without internet.

Default search is hybrid.

Hybrid uses:

```text
final_score = alpha * semantic_score + (1 - alpha) * keyword_score
```

The default alpha is `0.5`.

## Open

```bash
crag open 1
```

This opens the original source file and prints the slide or page location.

## Manage the index

```bash
crag status
crag list
crag list --errors
crag delete 3
crag delete /path/to/file.pptx
crag delete --all
crag delete --all --yes
```

Delete commands only remove indexed data. They do not delete the original source files.
```

- [ ] **Step 2: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run CLI help**

Run:

```bash
python -m crag.cli --help
```

Expected: command help displays all commands.

- [ ] **Step 4: Check repository state**

Run:

```bash
git status --short
```

Expected: only intended README changes are present before commit.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document crag setup and usage"
```

---

## Self-Review

Spec coverage:

- Mistral OCR ingestion is covered in Task 3.
- Raw OCR storage is covered in Task 3.
- Local embeddings from Hugging Face are covered in Tasks 4 and 9.
- Offline search is covered in Tasks 4 and 5.
- Hybrid alpha ranking is covered in Task 5.
- Rich tables are covered in Task 6.
- `crag open` source behavior is covered in Task 7.
- Status and list are covered in Task 6.
- Delete one file and delete all are covered in Task 8.

Placeholder scan:

- No incomplete markers or unfinished task sections are present.

Type consistency:

- `SearchResult` is defined in Task 2 and used in Tasks 5 and 6.
- `ParsedItem` is defined in Task 2 but not required by later tasks. It can stay as a small shared type for future parser cleanup.
- `ingest_file` gains `embedding_model` in Task 9 after its base behavior is already tested in Task 3.
