import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from tetraknowledge.application.service import KnowledgeService
from tetraknowledge.infrastructure.database import Base, SessionLocal, engine
from tetraknowledge.infrastructure.providers import embedding_provider, llm_provider


LITERARY_CORPUS = [
    ("Don Quijote de la Mancha", "don_quijote_de_la_mancha.txt"),
    ("El Conde de Montecristo", "el_conde_de_montecristo_es.txt"),
    ("Le Morte d'Arthur - Volume 1", "le_morte_darthur_volume_1.txt"),
    ("Le Morte d'Arthur - Volume 2", "le_morte_darthur_volume_2.txt"),
    ("Guia del Rey Arturo", "rey_arturo_guia_es.md"),
    ("Ficha de Batman y Alfred", "batman_alfred_ficha_es.md"),
]


def main():
    root = Path(__file__).parents[1]
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        service = KnowledgeService(session, embedding_provider(), llm_provider())

        kb = service.create_kb(
            "TetraBank Policies", "Synthetic policies for RAG demonstration"
        )
        payload = root.joinpath("demo_data/tetrabank_policies.txt").read_bytes()
        service.ingest(kb.id, "tetrabank_policies.txt", "text/plain", payload)
        print(kb.name)

        for name, filename in LITERARY_CORPUS:
            path = root.joinpath("corpus/raw", filename)
            kb = service.create_kb(name, f"Literary corpus document: {filename}")
            mime_type = "text/markdown" if path.suffix == ".md" else "text/plain"
            service.ingest(kb.id, path.name, mime_type, path.read_bytes())
            print(kb.name)


if __name__ == "__main__":
    main()
