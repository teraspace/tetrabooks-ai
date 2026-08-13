import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from tetraknowledge.application.service import KnowledgeService
from tetraknowledge.infrastructure.database import Base, SessionLocal, engine
from tetraknowledge.infrastructure.providers import embedding_provider, llm_provider


def main():
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        service = KnowledgeService(session, embedding_provider(), llm_provider())
        kb = service.create_kb("TetraBank Policies", "Synthetic policies for RAG demonstration")
        payload = (
            Path(__file__).parents[1].joinpath("demo_data/tetrabank_policies.txt").read_bytes()
        )
        service.ingest(kb.id, "tetrabank_policies.txt", "text/plain", payload)
        print(kb.id)


if __name__ == "__main__":
    main()
