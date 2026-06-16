from pathlib import Path


APP_DIR = Path.home() / ".crag"
DB_PATH = APP_DIR / "crag.db"
RAW_OCR_DIR = APP_DIR / "raw_ocr"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".webp"}
