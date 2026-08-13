import time

from sqlalchemy import select

from tetraknowledge.agents.graph import build_graph
from tetraknowledge.evaluation.metrics import reciprocal_rank, summarize
from tetraknowledge.infrastructure.database.models import (
    DocumentChunkModel,
    DocumentModel,
    EvaluationRunModel,
    KnowledgeBaseModel,
)
from tetraknowledge.infrastructure.documents import chunk_text, document_metadata, extract_document
from tetraknowledge.infrastructure.vector import (
    HybridRetriever,
    LexicalRetriever,
    MultiQueryRetriever,
    NumpyVectorRetriever,
)


class KnowledgeService:
    def __init__(self, session, embeddings, llm):
        self.session, self.embeddings, self.llm = session, embeddings, llm

    def create_kb(self, name, description):
        item = KnowledgeBaseModel(name=name, description=description)
        self.session.add(item)
        self.session.commit()
        return item

    def list_kbs(self):
        return self.session.scalars(
            select(KnowledgeBaseModel).order_by(KnowledgeBaseModel.created_at.desc())
        ).all()

    def get_kb(self, kb_id):
        return self.session.get(KnowledgeBaseModel, str(kb_id))

    def ingest(self, kb_id, filename, content_type, payload):
        text = extract_document(filename, content_type, payload)
        document = DocumentModel(
            knowledge_base_id=str(kb_id), filename=filename, content_type=content_type, content=text
        )
        self.session.add(document)
        self.session.flush()
        for item in chunk_text(text, metadata=document_metadata(filename)):
            self.session.add(
                DocumentChunkModel(
                    document_id=document.id,
                    content=item["content"],
                    chunk_metadata=item["metadata"],
                    embedding=self.embeddings.embed(item["content"]),
                    embedding_model=self.embeddings.model,
                    embedding_dimensions=self.embeddings.dimensions,
                    chunk_index=item["chunk_index"],
                )
            )
        self.session.commit()
        return document

    def chunks(self, kb_id):
        rows = self.session.execute(
            select(DocumentChunkModel, DocumentModel.filename)
            .join(DocumentModel)
            .where(DocumentModel.knowledge_base_id == str(kb_id))
        ).all()
        for chunk, filename in rows:
            # Enrich legacy rows created before work metadata was introduced.
            chunk.chunk_metadata = {
                **document_metadata(filename),
                **(chunk.chunk_metadata or {}),
            }
        return [chunk for chunk, _ in rows]

    def query(self, kb_id, question, top_k, mode, pipeline_version: int = 2):
        start = time.perf_counter()
        chunks = self.chunks(kb_id)
        base_retriever = {
            "vector": NumpyVectorRetriever(self.embeddings),
            "lexical": LexicalRetriever(),
            "hybrid": HybridRetriever(self.embeddings),
        }.get(mode, NumpyVectorRetriever(self.embeddings))
        retriever = base_retriever if pipeline_version == 1 else MultiQueryRetriever(base_retriever)
        result = build_graph(retriever, self.llm).invoke(
            {"question": question, "top_k": top_k, "_chunks": chunks}
        )
        return result, {
            "total": round((time.perf_counter() - start) * 1000, 2),
            "retrieval": None,
            "llm": None,
        }

    def evaluate(self, kb_id, strategy: str = "vector"):
        questions = [
            ("¿Cuál es el límite diario de Zafiro?", "7,350"),
            ("¿Qué exige Nébula?", "doble validación"),
            ("¿Qué autorización necesita Helios?", "Kappa-7"),
        ]
        chunks = self.chunks(kb_id)
        retriever = {
            "vector": NumpyVectorRetriever(self.embeddings),
            "lexical": LexicalRetriever(),
            "hybrid": HybridRetriever(self.embeddings),
        }.get(strategy, NumpyVectorRetriever(self.embeddings))
        rows = []
        for question, expected in questions:
            found = retriever.search(question, chunks, 5)
            ids = [str(item.chunk_id) for item in found]
            relevant = {
                str(item.chunk_id) for item in found if expected.lower() in item.content.lower()
            }
            rows.append(
                {"question": question, "hit": bool(relevant), "rr": reciprocal_rank(ids, relevant)}
            )
        run = EvaluationRunModel(
            knowledge_base_id=str(kb_id), strategy=strategy, summary=summarize(rows)
        )
        self.session.add(run)
        self.session.commit()
        return run
