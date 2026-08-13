# API examples

The complete contract is available in Swagger at `/docs`.

```bash
curl localhost:8000/health
curl localhost:8000/ready
curl -X POST localhost:8000/knowledge-bases -H 'content-type: application/json' -d '{"name":"TetraBank","description":"Policies"}'
curl -X POST localhost:8000/knowledge-bases/{id}/documents -F file=@demo_data/tetrabank_policies.txt
curl -X POST localhost:8000/knowledge-bases/{id}/query -H 'content-type: application/json' -d '{"question":"What is the Zafiro limit?","top_k":3,"retrieval_mode":"hybrid"}'
```

## Multi-entity retrieval

The query path performs deterministic entity detection before retrieval. Known ambiguous
mentions are expanded into entity-specific subqueries, retrieved independently, and fused
with diversity by `metadata.work`. This is especially useful when multiple books share a
character name:

```bash
curl -X POST localhost:8000/knowledge-bases/{id}/query \
  -H 'content-type: application/json' \
  -d '{"question":"Diferencias entre Fernando y Don Fernando","top_k":6,"retrieval_mode":"hybrid"}'
```

The response exposes the work metadata on each citation, making it possible to audit which
source supported each side of a comparison. The current planner is intentionally
deterministic; a production NER model or LLM reranker can be added behind the same planner
and retriever interfaces.
