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
