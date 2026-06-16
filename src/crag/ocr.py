from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral


class MistralOcrClient:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def parse_file(self, path: Path) -> dict[str, Any]:
        with path.open("rb") as source:
            uploaded = self.client.files.upload(
                file={
                    "file_name": path.name,
                    "content": source,
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
