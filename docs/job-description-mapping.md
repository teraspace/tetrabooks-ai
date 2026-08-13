# Job description mapping

| Requirement | Implementation | Evidence | Interview demonstration |
|---|---|---|---|
| Python/FastAPI/REST | API routes and Pydantic schemas | `src/tetraknowledge/api/main.py` | Create a KB, upload a document and query it |
| RAG/vector search | Embeddings, vector/lexical/hybrid retrievers | `infrastructure/providers.py`, `vector.py` | Explain query embedding, top-k and citations |
| LangChain/LangGraph | Explicit retrieval-answer-validation graph | `agents/graph.py` | Trace retry and validation state |
| PostgreSQL/SQLAlchemy | Relational models and Alembic | `infrastructure/database`, `alembic` | Discuss metadata/index and migrations |
| Azure OpenAI | Configurable provider adapter | `infrastructure/providers.py` | Switch provider through environment variables |
| Azure AI Search | Configurable search adapter | `infrastructure/vector.py` | Explain hybrid retrieval and managed scaling |
| Document processing | TXT/Markdown/PDF extraction and chunking | `infrastructure/documents.py` | Show ingestion pipeline |
| Testing/CI | pytest, Ruff, GitHub Actions | `tests`, `.github/workflows/ci.yml` | Run tests without paid APIs |
| Production engineering | Docker, health checks, K8s probes, ADRs | `Dockerfile`, `deploy`, `docs/adr` | Discuss readiness, scaling and failure modes |
