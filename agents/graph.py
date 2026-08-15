"""
agents/graph.py
=================

Weekend 3: wires the Retriever, Grader, and Analyst nodes into a LangGraph
StateGraph with a bounded retry loop.

Graph shape:

    retriever -> grader -> [route_after_grade]
                              |-- pass --------------> analyst -> END
                              |-- retry (< max) ------> retriever (loop)
                              |-- fail (retries used) -> analyst -> END

The retry_count increment happens in this module (not in retriever_node)
so the retry bookkeeping lives in one obvious place rather than being
threaded through every node that might loop.

Usage:
    from agents.graph import build_graph, run_query
    answer = run_query("What supply chain risks does NVIDIA mention?")
"""

from __future__ import annotations

import sys

from langgraph.graph import END, StateGraph
from loguru import logger

sys.path.insert(0, ".")
from agents.analyst import analyst_node  # noqa: E402
from agents.grader import grader_node, route_after_grade  # noqa: E402
from agents.retriever import retriever_node  # noqa: E402
from agents.state import AgentState  # noqa: E402

from common.tickers import detect_out_of_scope  # noqa: E402

def _route_entry(state: dict) -> str:
    """
    Entry-point router: skip retrieval/grading entirely for questions
    naming a clearly out-of-corpus company. No point spending a retrieval
    round + LLM grading call on a question we already know the answer to.
    """
    if detect_out_of_scope(state["question"]):
        return "analyst"
    return "retriever"

def _increment_retry(state: dict) -> dict:
    """Bump retry_count before looping back to the Retriever."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("analyst", analyst_node)

    graph.set_conditional_entry_point(
        _route_entry,
        {
            "retriever": "retriever",
            "analyst": "analyst",
        },
    )

    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges(
        "grader",
        route_after_grade,
        {
            "analyst": "analyst",
            "analyst_low_confidence": "analyst",
            "retry": "increment_retry",
        },
    )
    graph.add_edge("increment_retry", "retriever")
    graph.add_edge("analyst", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str) -> dict:
    """
    Run a question through the full agent loop. Returns the final state
    dict (includes final_answer, ticker, retrieved_chunks, low_confidence).
    """
    graph = get_graph()
    logger.info(f"Running query: {question!r}")
    result = graph.invoke({"question": question, "retry_count": 0})
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m agents.graph "your question here"')
        return 1

    question = " ".join(sys.argv[1:])
    result = run_query(question)

    print()
    print(f"Ticker filter used: {result.get('ticker') or '(none / cross-company)'}")
    print(f"Low confidence: {result.get('low_confidence', False)}")
    print(f"Retries used: {result.get('retry_count', 0)}")
    print()
    print("=== Answer ===")
    print(result.get("final_answer", "(no answer produced)"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
