import math
from abc import ABC, abstractmethod
from uuid import UUID

from tetraknowledge.application.query_planning import QueryPlan, build_query_plan
from tetraknowledge.domain import RetrievedChunk


def cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denom if denom else 0.0


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, chunks: list, top_k: int) -> list[RetrievedChunk]: ...


class NumpyVectorRetriever(Retriever):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query: str, chunks: list, top_k: int) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed(query)
        ranked = [(cosine(query_vector, c.embedding or []), c) for c in chunks]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(UUID(c.id), UUID(c.document_id), c.content, score, c.chunk_metadata)
            for score, c in ranked[:top_k]
        ]


class LexicalRetriever(Retriever):
    def search(self, query: str, chunks: list, top_k: int) -> list[RetrievedChunk]:
        terms = set(query.lower().split())
        ranked = []
        for c in chunks:
            ranked.append((len(terms & set(c.content.lower().split())) / max(len(terms), 1), c))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(UUID(c.id), UUID(c.document_id), c.content, score, c.chunk_metadata)
            for score, c in ranked[:top_k]
        ]


class HybridRetriever(Retriever):
    def __init__(self, embeddings):
        self.vector, self.lexical = NumpyVectorRetriever(embeddings), LexicalRetriever()

    def search(self, query: str, chunks: list, top_k: int) -> list[RetrievedChunk]:
        vector, lexical, scores = (
            self.vector.search(query, chunks, 10),
            self.lexical.search(query, chunks, 10),
            {},
        )
        by_id = {item.chunk_id: item for item in vector + lexical}
        for rank, item in enumerate(vector, 1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (60 + rank)
        for rank, item in enumerate(lexical, 1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (60 + rank)
        return [
            RetrievedChunk(item.chunk_id, item.document_id, item.content, score, item.metadata)
            for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for item in [by_id[item_id]]
        ]


class MultiQueryRetriever(Retriever):
    """Retrieve each planned sub-query, then fuse with work-level diversity."""

    def __init__(self, base: Retriever):
        self.base = base
        self.entity_anchor = LexicalRetriever()

    def search(self, query: str, chunks: list, top_k: int) -> list[RetrievedChunk]:
        plan: QueryPlan = build_query_plan(query)
        per_query = max(top_k, min(8, top_k * 2))
        candidates: dict[UUID, RetrievedChunk] = {}
        for subquery in plan.subqueries:
            for item in self.base.search(subquery, chunks, per_query):
                existing = candidates.get(item.chunk_id)
                if existing is None or item.score > existing.score:
                    candidates[item.chunk_id] = item
            # Entity names are exact lexical anchors. Keep these hits even when a
            # weak/demo embedding ranks semantically related passages higher.
            if plan.entities:
                for item in self.entity_anchor.search(subquery, chunks, per_query):
                    existing = candidates.get(item.chunk_id)
                    if existing is None or item.score > existing.score:
                        candidates[item.chunk_id] = item

        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        if len(plan.entities) < 2:
            return ranked[:top_k]

        # Round-robin by work guarantees that a comparison receives evidence for both sides.
        groups: dict[str, list[RetrievedChunk]] = {}
        for item in ranked:
            work = item.metadata.get("work", "unknown")
            groups.setdefault(work, []).append(item)
        selected: list[RetrievedChunk] = []
        while len(selected) < top_k and groups:
            for work in list(groups):
                if groups[work]:
                    selected.append(groups[work].pop(0))
                    if len(selected) == top_k:
                        break
                if not groups[work]:
                    del groups[work]
        return selected


class VectorStore:
    def index_documents(self, documents):
        raise NotImplementedError

    def search(self, query, top_k):
        raise NotImplementedError

    def delete(self, document_id):
        raise NotImplementedError

    def health_check(self):
        return True


class AzureAISearchVectorStore(VectorStore):
    def __init__(self, endpoint: str, api_key: str, index_name: str):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self.client = SearchClient(endpoint, index_name, AzureKeyCredential(api_key))

    def index_documents(self, documents):
        return self.client.upload_documents(documents)

    def search(self, query, top_k):
        return self.client.search(search_text=query, top=top_k)

    def delete(self, document_id):
        return self.client.delete_documents("id", [{"id": document_id}])
