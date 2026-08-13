import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import streamlit as st

from tetraknowledge.application.service import KnowledgeService
from tetraknowledge.infrastructure.database import SessionLocal
from tetraknowledge.infrastructure.providers import embedding_provider, llm_provider

st.set_page_config(page_title="TetraBooks AI", page_icon="📚")
st.title("TetraBooks AI")
st.caption("Long-form document intelligence demo · evolución del pipeline RAG")

with SessionLocal() as session:
    service = KnowledgeService(session, embedding_provider(), llm_provider())
    kbs = service.list_kbs()
    if not kbs:
        st.info("Run python scripts/seed_demo.py first.")
    else:
        kb = st.selectbox("Knowledge base", kbs, format_func=lambda item: item.name)
        pipeline = st.selectbox(
            "Pipeline version",
            [2, 1],
            format_func=lambda version: {
                2: "v2 · Entity-aware RAG (multi-query + diversity)",
                1: "v1 · Basic RAG (single retrieval)",
            }[version],
        )
        mode = st.selectbox("Retrieval mode", ["vector", "lexical", "hybrid"])
        question = st.text_input("Question")
        if question:
            result, latency = service.query(kb.id, question, 5, mode, pipeline)
            st.subheader("Answer")
            st.write(result.get("answer"))
            st.caption(f"Pipeline activa: v{pipeline}")
            st.metric("Latency (ms)", latency["total"])
            st.subheader("Retrieved chunks")
            for item in result.get("citations", []):
                with st.expander(f"Score {item.score:.3f}"):
                    st.write(item.content)
