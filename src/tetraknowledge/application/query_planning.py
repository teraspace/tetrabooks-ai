"""Small, deterministic query-planning layer for multi-entity questions.

This is intentionally independent of an LLM. It gives retrieval a reliable first pass
and can later be replaced or extended by an NER model without changing the retriever API.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryEntity:
    mention: str
    canonical_name: str
    work: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    original: str
    entities: tuple[QueryEntity, ...]
    subqueries: tuple[str, ...]


ENTITY_CATALOG = (
    QueryEntity(
        mention="Fernando",
        canonical_name="Fernando Mondego",
        work="El Conde de Montecristo",
        aliases=("fernando mondego", "fernando", "mondego"),
    ),
    QueryEntity(
        mention="Don Fernando",
        canonical_name="Don Fernando",
        work="Don Quijote de la Mancha",
        aliases=("don fernando",),
    ),
)


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def detect_entities(question: str) -> list[QueryEntity]:
    """Resolve the known ambiguous Fernando mentions before retrieval.

    The explicit ``Don Fernando`` check must run before the generic ``Fernando`` check.
    This avoids treating the longer mention as two independent entities.
    """

    normalized = " ".join(question.lower().split())
    found: list[QueryEntity] = []
    if _contains(normalized, "don fernando"):
        found.append(ENTITY_CATALOG[1])
    standalone_fernando = re.search(r"(?<!don )\bfernando\b", normalized)
    if _contains(normalized, "fernando mondego") or standalone_fernando:
        found.append(ENTITY_CATALOG[0])
    return found


def build_query_plan(question: str) -> QueryPlan:
    entities = detect_entities(question)
    subqueries = [question]
    for entity in entities:
        # Keep the bare mention as an exact lexical anchor. The disambiguated
        # variant adds work context for dense retrieval without replacing it.
        subqueries.append(entity.canonical_name)
        subqueries.append(f"{entity.canonical_name} {entity.work}")
    if len(entities) > 1:
        subqueries.append(" ".join(f"{e.canonical_name} {e.work}" for e in entities))
    return QueryPlan(question, tuple(entities), tuple(dict.fromkeys(subqueries)))
