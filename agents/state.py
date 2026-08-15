"""
agents/state.py
=================

Weekend 3: the shared state object every node in the graph reads from and
writes to. LangGraph passes this dict between nodes, merging each node's
return value into it.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # Input
    question: str

    # Set by the Retriever node (detected from the question, or None for
    # cross-company / unrecognized questions)
    ticker: str | None

    # Set by the Retriever node
    retrieved_chunks: list[dict]

    # Set by the Grader node: "pass" or "fail"
    grade: str

    # Incremented each time the graph loops back for a retry
    retry_count: int

    # Set by the Analyst node — the final response shown to the user
    final_answer: str

    # True if the Analyst had to proceed despite a failed grade after
    # exhausting retries — surfaced to the user as a confidence signal
    low_confidence: bool
