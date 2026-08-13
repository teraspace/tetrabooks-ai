from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
