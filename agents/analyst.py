"""
agents/analyst.py
===================

Weekend 3: the Analyst node.

Synthesizes a final answer from whatever chunks the Retriever/Grader loop
settled on, with inline citations back to ticker + accession number so
every claim is traceable to a specific filing.

If the grade never passed (retries exhausted), the answer is prefixed
with a visible low-confidence disclaimer — the system tells the user its
context might be incomplete rather than presenting an unqualified answer.
"""

from __future__ import annotations

import sys
from common.tickers import detect_out_of_scope  # noqa: E402

from loguru import logger

sys.path.insert(0, ".")
from common.llm_client import get_llm_client  # noqa: E402

ANALYST_PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the excerpts below. Cite each claim with its excerpt number in brackets, e.g. [1]. If the excerpts don't fully answer the question, say what's missing rather than guessing.
Question: {question}
Excerpts:
{excerpts}
Answer (with [N] citations):"""

LOW_CONFIDENCE_PREFIX = (
    "⚠️ Note: retrieval confidence was low for this question — the excerpts "
    "below may not fully cover it. Treat this answer as a starting point, "
    "not a complete picture.\n\n"
)


def _format_excerpts(chunks: list[dict], max_chars_each: int = 600) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = c["chunk_text"][:max_chars_each].replace("\n", " ")
        lines.append(
            f"[{i+1}] ({c.get('ticker', '?')}, filing {c.get('accession_number', '?')}) {snippet}"
        )
    return "\n\n".join(lines)


def analyst_node(state: dict) -> dict:
    question = state["question"]

    out_of_scope_company = detect_out_of_scope(question)
    if out_of_scope_company:
        logger.info(f"Analyst: {out_of_scope_company} is out of corpus scope, "
                     f"skipping LLM call")
        return {
            "final_answer": (
                f"This system doesn't cover {out_of_scope_company} — it's "
                f"scoped to five companies' 10-Q filings: NVDA (NVIDIA), "
                f"AMD, INTC (Intel), TSLA (Tesla), and AAPL (Apple). "
                f"Try asking about one of those instead."
            ),
            "low_confidence": False,
        }

    chunks = state.get("retrieved_chunks", [])
    low_confidence = state.get("grade") != "pass"
    # ...rest of the function continues exactly as before

    if not chunks:
        logger.info("Analyst: no chunks available, returning graceful fallback")
        return {
            "final_answer": (
                "I couldn't find relevant information in the indexed filings "
                "to answer this question. Try rephrasing, or ask about one of: "
                "NVDA, AMD, INTC, TSLA, AAPL."
            ),
            "low_confidence": True,
        }

    llm = get_llm_client()
    prompt = ANALYST_PROMPT_TEMPLATE.format(
        question=question,
        excerpts=_format_excerpts(chunks),
    )

    logger.info(f"Analyst: synthesizing answer from {len(chunks)} chunks "
                f"(low_confidence={low_confidence})")
    response = llm.generate(prompt, role="analyst", temperature=0.1)

    if low_confidence:
        response = LOW_CONFIDENCE_PREFIX + response

    return {
        "final_answer": response,
        "low_confidence": low_confidence,
    }
