import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from tetraknowledge.application.service import KnowledgeService
from tetraknowledge.infrastructure.database import Base, SessionLocal, engine
from tetraknowledge.infrastructure.providers import embedding_provider, llm_provider


LITERARY_CORPUS = [
    (
        "Don Quijote de la Mancha",
        "don_quijote_de_la_mancha.txt",
        "Don Quijote de la Mancha, de Miguel de Cervantes, presenta las aventuras de Alonso Quijano, quien adopta la identidad de Don Quijote y recorre La Mancha junto a Sancho Panza. La obra explora la tension entre imaginacion y realidad, los libros de caballeria, la amistad y la identidad.",
    ),
    (
        "El Conde de Montecristo",
        "el_conde_de_montecristo_es.txt",
        "El Conde de Montecristo, de Alexandre Dumas, sigue a Edmond Dantes, encarcelado injustamente y transformado por su encuentro con el abate Faria. Tras escapar, adopta una nueva identidad y planifica una compleja venganza contra quienes lo traicionaron.",
    ),
    (
        "Le Morte d'Arthur - Volume 1",
        "le_morte_darthur_volume_1.txt",
        "Le Morte d'Arthur, de Thomas Malory, reune relatos sobre Arturo, Merlin, Camelot, los caballeros de la Tabla Redonda y la espada Excalibur. Este primer volumen introduce el linaje de Arturo, su ascenso al trono y la formacion de su reino.",
    ),
    (
        "Le Morte d'Arthur - Volume 2",
        "le_morte_darthur_volume_2.txt",
        "Este volumen continua Le Morte d'Arthur con las aventuras de los caballeros de Camelot, sus juramentos, conflictos y busquedas. El corpus permite consultar personajes, lugares, relaciones y temas del ciclo arturico.",
    ),
    (
        "Guia del Rey Arturo",
        "rey_arturo_guia_es.md",
        "Guia de referencia sobre el ciclo arturico: Arturo, Merlin, Guinevere, Lancelot, Gawain, Mordred, Camelot, Excalibur y la Tabla Redonda. Incluye relaciones entre personajes, lugares y motivos narrativos para consultas de recuperacion semantica.",
    ),
    (
        "Ficha de Batman y Alfred",
        "batman_alfred_ficha_es.md",
        "Ficha de referencia sobre Bruce Wayne, Batman y Alfred Pennyworth. Resume sus identidades, roles, habilidades, relacion de confianza y la funcion de Alfred como mayordomo, mentor y apoyo estrategico de Batman.",
    ),
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

        for name, filename, fallback_text in LITERARY_CORPUS:
            path = root.joinpath("corpus/raw", filename)
            kb = service.create_kb(name, f"Literary corpus document: {filename}")
            if path.is_file():
                payload = path.read_bytes()
            else:
                payload = fallback_text.encode("utf-8")
            mime_type = "text/markdown" if filename.endswith(".md") else "text/plain"
            service.ingest(kb.id, filename, mime_type, payload)
            print(kb.name)


if __name__ == "__main__":
    main()
