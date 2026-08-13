import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from tetraknowledge.infrastructure.documents import chunk_text
from tetraknowledge.infrastructure.providers import HashEmbeddingProvider

QUESTIONS = [
    ("¿Cuál es el límite diario de Zafiro?", "7,350", "direct"),
    ("¿Qué exige Nébula?", "doble validación", "paraphrase"),
    ("¿Qué autorización necesita una operación Helios de 3000?", "Kappa-7", "multi-hop"),
]


def main():
    text = Path(__file__).parents[1].joinpath("demo_data/tetrabank_policies.txt").read_text()
    chunks, emb, rows = chunk_text(text, 80, 10), HashEmbeddingProvider(), []
    for question, expected, kind in QUESTIONS:
        q = emb.embed(question)
        scores = [
            sum(a * b for a, b in zip(q, emb.embed(c["content"]), strict=True)) for c in chunks
        ]
        best = chunks[max(range(len(scores)), key=scores.__getitem__)]
        rows.append(
            {
                "strategy": "vector",
                "question_type": kind,
                "hit": expected.lower() in best["content"].lower(),
                "score": max(scores),
            }
        )
    Path("results").mkdir(exist_ok=True)
    with Path("results/experiment_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    Path("results/summary.md").write_text(
        "# Evaluation summary\n\n"
        "This deterministic smoke benchmark is intentionally small; expand it to "
        "50–80 questions for a serious comparison.\n"
    )
    print(rows)


if __name__ == "__main__":
    main()
