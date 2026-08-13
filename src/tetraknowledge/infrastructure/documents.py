from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class UnsupportedDocument(ValueError):
    pass


def document_metadata(filename: str) -> dict:
    """Derive stable, human-readable filters from corpus filenames."""

    normalized = Path(filename).name.lower()
    metadata = {"source": "document", "language": "es", "source_file": Path(filename).name}
    if "montecristo" in normalized or "monte_cristo" in normalized:
        metadata.update({"work": "El Conde de Montecristo", "work_key": "montecristo"})
    elif "quijote" in normalized:
        metadata.update({"work": "Don Quijote de la Mancha", "work_key": "don_quijote"})
    elif "morte_darthur" in normalized or "artur" in normalized:
        metadata.update({"work": "Le Morte d'Arthur", "work_key": "arturo"})
        if "morte_darthur" in normalized:
            metadata["language"] = "en"
    elif "batman" in normalized or "alfred" in normalized:
        metadata.update({"work": "Batman y Alfred", "work_key": "batman_alfred"})
    elif "tetrabank" in normalized or "polic" in normalized:
        metadata.update({"work": "TetraBank Policies", "work_key": "tetrabank"})
    else:
        metadata.update({"work": Path(filename).stem, "work_key": Path(filename).stem})
    return metadata


class AzureDocumentIntelligenceExtractor:
    """Optional OCR adapter, instantiated only when Azure credentials exist."""

    def __init__(self, endpoint: str, key: str):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        self.client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    def extract(self, payload: bytes) -> str:
        poller = self.client.begin_analyze_document("prebuilt-read", body=payload)
        result = poller.result()
        return "\n".join(line.content for page in result.pages for line in page.lines)


def extract_document(filename: str, content_type: str, payload: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"} or content_type.startswith("text/"):
        return payload.decode("utf-8", errors="replace")
    if suffix == ".pdf" or content_type == "application/pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
    raise UnsupportedDocument(f"Unsupported document format: {suffix or content_type}")


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    metadata: dict | None = None,
) -> list[dict]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Invalid chunk configuration")
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(
            {
                "content": " ".join(words[start:end]),
                "chunk_index": len(chunks),
                "metadata": {
                    **(metadata or {"source": "document"}),
                    "chunk_index": len(chunks),
                },
            }
        )
        if end == len(words):
            break
        start = end - chunk_overlap
    return chunks
