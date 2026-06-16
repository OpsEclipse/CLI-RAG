import os
from pathlib import Path


APP_DIR = Path.home() / ".crag"
DB_PATH = APP_DIR / "crag.db"
RAW_OCR_DIR = APP_DIR / "raw_ocr"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".webp"}


def load_dotenv(paths: tuple[Path, ...] = (Path(".env"), Path(".env.local"))) -> None:
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key_value = _parse_dotenv_line(line)
            if key_value is None:
                continue
            key, value = key_value
            os.environ.setdefault(key, value)


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_dotenv_quotes(value.strip())


def _strip_dotenv_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
