from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict, total=False):
    question: str
    rewritten_question: str
    retrieved_chunks: list
    retrieval_attempts: int
    answer: str
    validation: dict
    citations: list
    top_k: int
    _chunks: list


def build_graph(retriever, llm):
    def analyze(state):
        return {"rewritten_question": state["question"], "retrieval_attempts": 0}

    def retrieve(state):
        found = retriever.search(state["rewritten_question"], state["_chunks"], state["top_k"])
        return {
            "retrieved_chunks": found,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    def route_after_retrieval(state):
        if state.get("retrieved_chunks") and state["retrieved_chunks"][0].score > 0:
            return "answer"
        return "answer" if state.get("retrieval_attempts", 0) >= 2 else "rewrite"

    def rewrite(state):
        return {
            "rewritten_question": state["question"] + " policy rule",
            "retrieval_attempts": state.get("retrieval_attempts", 0),
        }

    def answer(state):
        context = "\n".join(
            f"[{i + 1}] {c.content}" for i, c in enumerate(state.get("retrieved_chunks", []))
        )
        return {
            "answer": llm.answer(state["question"], context),
            "citations": state.get("retrieved_chunks", []),
        }

    def validate(state):
        answer = state.get("answer", "").lower()
        return {
            "validation": {
                "grounded": bool(state.get("retrieved_chunks")) and "no encontré" not in answer,
                "citation_count": len(state.get("citations", [])),
            }
        }

    graph = StateGraph(GraphState)
    for name, fn in [
        ("analyze_question", analyze),
        ("retrieve_context", retrieve),
        ("rewrite_query", rewrite),
        ("answer", answer),
        ("validate_answer", validate),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve_context")
    graph.add_conditional_edges(
        "retrieve_context", route_after_retrieval, {"answer": "answer", "rewrite": "rewrite_query"}
    )
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("answer", "validate_answer")
    graph.add_edge("validate_answer", END)
    return graph.compile()
