def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1 / rank
    return 0.0


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "hit_rate": 0.0, "mrr": 0.0}
    return {
        "count": len(rows),
        "hit_rate": sum(row["hit"] for row in rows) / len(rows),
        "mrr": sum(row["rr"] for row in rows) / len(rows),
    }
