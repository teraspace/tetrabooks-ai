# ADR-002: pgvector first, Azure AI Search as an adapter

Local development uses PostgreSQL/pgvector because it keeps relational metadata and vectors together. Azure AI Search is an adapter for larger hybrid-search deployments. Exact search is reasonable for small corpora; HNSW/IVFFlat become useful as scale and latency requirements grow.
