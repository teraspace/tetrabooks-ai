# TetraBooks AI

TetraBooks AI is an independent fork of TetraKnowledge for document intelligence over books and long-form sources. It demonstrates ingestion, chunking, embeddings, retrieval, RAG, citations and a LangGraph workflow while keeping model and search vendors replaceable.

## Architecture

```mermaid
flowchart TD
  A[FastAPI] --> B[KnowledgeService]
  B --> C[LangGraph]
  C --> D[Query planner: NER + decomposition]
  D --> E[Multi-query retrieval + fusion/diversity]
  E --> H[PostgreSQL + pgvector / Azure AI Search]
  C --> F[LLM Provider: rule based, OpenAI, Azure OpenAI]
  C --> G[Validation + citations]
```

The local demo uses a deterministic hash embedding and rule-based LLM, so it runs without credentials. Production providers are selected with environment variables. The rule-based mode is deliberately a test double, not a claim that it replaces an LLM.

For multi-entity questions, the planner detects known ambiguous mentions, creates
entity-specific subqueries, retrieves each one independently, and fuses the results with
work-level diversity. For example, “Diferencias entre Fernando y Don Fernando” retrieves
both `El Conde de Montecristo` and `Don Quijote de la Mancha` instead of allowing one work
to dominate the top-k list. Each chunk exposes `work`, `work_key`, `language`, and
`source_file` metadata.

## Quickstart

```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
python scripts/seed_demo.py
uvicorn tetraknowledge.api.main:app --reload
```

Open http://localhost:8000/docs. For PostgreSQL/pgvector:

```bash
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://tetraknowledge:tetraknowledge@localhost:5434/tetraknowledge alembic upgrade head
```

## Example

```bash
KB_ID=$(curl -s -X POST localhost:8000/knowledge-bases -H 'content-type: application/json' -d '{"name":"TetraBank"}' | python -c 'import sys,json; print(json.load(sys.stdin)["id"])')
curl -X POST "localhost:8000/knowledge-bases/$KB_ID/documents" -F file=@demo_data/tetrabank_policies.txt
curl -X POST "localhost:8000/knowledge-bases/$KB_ID/query" -H 'content-type: application/json' -d '{"question":"Una cuenta Zafiro lleva 50 días inactiva y luego transfiere 3000 créditos. ¿Qué autorización necesita?","retrieval_mode":"hybrid"}'
```

## Providers

`LLM_PROVIDER=rule_based`, `openai` or `azure_openai`. Azure requires endpoint, key and deployment. `EMBEDDING_PROVIDER=hash` runs offline; `sentence_transformer` uses the configured multilingual model (for example `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, BGE or E5) when the optional dependency is installed. Changing embedding providers requires re-indexing the corpus. Azure AI Search is an optional adapter configured with `AZURE_SEARCH_*`.

## RAG and evaluation

The graph is `analyze_question -> retrieve_context -> answer -> validate_answer`, with a bounded rewrite path when retrieval is empty. Sources are returned as chunk IDs, document IDs, scores and metadata. The vector analogy is functional, not mathematical equivalence: retrieval selects external non-parametric memory at inference time; Transformer attention computes interactions inside a model forward pass.

Run the deterministic experiment with `python scripts/run_evaluation.py`. The benchmark scaffolding distinguishes retrieval failure from generation failure. Expand `QUESTIONS` to 50–80 cases for a statistically useful comparison.

## Engineering decisions

- Small explicit ports are used instead of hiding all behavior behind LangChain.
- Exact local retrieval is acceptable for a small corpus; HNSW/IVFFlat are scale decisions, not defaults by ideology.
- Documents are data, not instructions. The prompt defense protects the system policy from prompt injection.
- Paid providers are never required by tests.
- Docker runs as a non-root user and exposes `/health` and `/ready`.

See `docs/job-description-mapping.md`, `docs/adr/`, `deploy/k8s/`, and `docs/api.md` for the interview narrative.
