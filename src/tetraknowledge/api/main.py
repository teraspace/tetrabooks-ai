import time
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from tetraknowledge.application.service import KnowledgeService
from tetraknowledge.config import get_settings
from tetraknowledge.infrastructure.database import SessionLocal
from tetraknowledge.infrastructure.database.models import EvaluationRunModel
from tetraknowledge.infrastructure.providers import embedding_provider, llm_provider

app = FastAPI(
    title="TetraBooks AI",
    version="0.1.0",
    description="Grounded document intelligence for books and long-form sources",
)
app.add_middleware(
    CORSMiddleware, allow_origins=get_settings().origins, allow_methods=["*"], allow_headers=["*"]
)
request_metrics = {"requests_total": 0, "errors_total": 0, "latencies": []}


@app.middleware("http")
async def observe_requests(request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        if response.status_code >= 500:
            request_metrics["errors_total"] += 1
        return response
    except Exception:
        request_metrics["errors_total"] += 1
        raise
    finally:
        request_metrics["requests_total"] += 1
        request_metrics["latencies"].append((time.perf_counter() - started) * 1000)


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class KnowledgeBaseResponse(BaseModel):
    id: UUID
    name: str
    description: str | None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_mode: str = Field(default="hybrid", pattern="^(vector|lexical|hybrid)$")
    pipeline_version: int = Field(default=2, ge=1, le=2)


class SourceResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    retrieval_mode: str
    pipeline_version: int
    latency_ms: dict
    validation: dict


def service_for(session):
    return KnowledgeService(session, embedding_provider(), llm_provider())


@app.get("/health")
def health():
    return {"status": "ok", "service": "tetrabooks-ai"}


@app.get("/ready")
def ready():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ready", "database": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc


@app.post(
    "/knowledge-bases", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED
)
def create_kb(payload: CreateKnowledgeBaseRequest):
    with SessionLocal() as session:
        return service_for(session).create_kb(payload.name, payload.description)


@app.get("/knowledge-bases", response_model=list[KnowledgeBaseResponse])
def list_kbs():
    with SessionLocal() as session:
        return service_for(session).list_kbs()


@app.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
def get_kb(kb_id: UUID):
    with SessionLocal() as session:
        item = service_for(session).get_kb(kb_id)
        if not item:
            raise HTTPException(404, "Knowledge base not found")
        return item


@app.post("/knowledge-bases/{kb_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload(kb_id: UUID, file: Annotated[UploadFile, File()]):
    payload = await file.read()
    if len(payload) > get_settings().max_document_bytes:
        raise HTTPException(413, "Document too large")
    with SessionLocal() as session:
        service = service_for(session)
        if not service.get_kb(kb_id):
            raise HTTPException(404, "Knowledge base not found")
        try:
            doc = service.ingest(
                kb_id,
                file.filename or "document",
                file.content_type or "application/octet-stream",
                payload,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "id": doc.id,
            "filename": doc.filename,
            "status": doc.status,
            "chunks": len(doc.chunks),
        }


@app.get("/knowledge-bases/{kb_id}/documents")
def list_documents(kb_id: UUID):
    with SessionLocal() as session:
        return [
            {"id": c.document_id, "content": c.content, "metadata": c.chunk_metadata}
            for c in service_for(session).chunks(kb_id)
        ]


@app.post("/knowledge-bases/{kb_id}/query", response_model=QueryResponse)
def query(kb_id: UUID, payload: QueryRequest):
    with SessionLocal() as session:
        service = service_for(session)
        if not service.get_kb(kb_id):
            raise HTTPException(404, "Knowledge base not found")
        result, latency = service.query(
            kb_id,
            payload.question,
            payload.top_k,
            payload.retrieval_mode,
            payload.pipeline_version,
        )
        return {
            "answer": result.get("answer", ""),
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "score": c.score,
                    "metadata": c.metadata,
                }
                for c in result.get("citations", [])
            ],
            "retrieval_mode": payload.retrieval_mode,
            "pipeline_version": payload.pipeline_version,
            "latency_ms": latency,
            "validation": result.get("validation", {}),
        }


@app.post("/knowledge-bases/{kb_id}/retrieve")
def retrieve(kb_id: UUID, payload: QueryRequest):
    with SessionLocal() as session:
        result, _ = service_for(session).query(
            kb_id,
            payload.question,
            payload.top_k,
            payload.retrieval_mode,
            payload.pipeline_version,
        )
        return {
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "score": c.score,
                    "metadata": c.metadata,
                }
                for c in result.get("citations", [])
            ]
        }


@app.post("/knowledge-bases/{kb_id}/evaluate")
def evaluate(kb_id: UUID, strategy: str = "vector"):
    with SessionLocal() as session:
        service = service_for(session)
        if not service.get_kb(kb_id):
            raise HTTPException(404, "Knowledge base not found")
        run = service.evaluate(kb_id, strategy)
        return {"id": run.id, "strategy": run.strategy, "summary": run.summary}


@app.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: UUID):
    with SessionLocal() as session:
        run = session.get(EvaluationRunModel, str(evaluation_id))
        if not run:
            raise HTTPException(404, "Evaluation not found")
        return {
            "id": run.id,
            "strategy": run.strategy,
            "summary": run.summary,
            "created_at": run.created_at,
        }


@app.get("/metrics/summary")
def metrics():
    latencies = request_metrics["latencies"]
    return {
        "requests_total": request_metrics["requests_total"],
        "errors_total": request_metrics["errors_total"],
        "average_latency": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "retrieval_success_rate": 0,
        "grounded_answer_rate": 0,
    }
