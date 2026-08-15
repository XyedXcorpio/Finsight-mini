"""
agents/grader.py
==================

Weekend 3: the Grader node.

Asks the LLM a single binary question: does the retrieved context contain
genuinely relevant material, even if partial? This mirrors the Adaptive
RAG grader_node pattern from PA3 — a lightweight relevance check that
decides whether to proceed to synthesis or retry with a broadened search.

TUNING NOTE: the first version of this prompt asked whether excerpts
"directly" and "substantively" address the question. On a 3B local model,
this consistently graded FAIL even on excerpts that were clearly on-topic
(e.g. AMD's mineral-sourcing paragraph, verified relevant in Weekend 2
testing) — the model was effectively grading "is this a complete answer"
rather than "is this relevant context," a harder and different bar than
intended. Loosened to "relevant, even if partial" with a concrete example
pair, which is a well-documented technique for stabilizing small models on
binary classification prompts. Incompleteness is fine — the Analyst is
already equipped to say what's missing rather than overclaim.
"""

from __future__ import annotations

import sys

from loguru import logger

sys.path.insert(0, ".")
from common.llm_client import get_llm_client  # noqa: E402

GRADE_PROMPT_TEMPLATE = """You are grading whether retrieved excerpts contain ANY genuinely relevant \
information for a question — not whether they fully answer it.
Question: {question}
Retrieved excerpts:
{excerpts}
Grade PASS if at least one excerpt discusses the question's topic, even \
partially or indirectly. Grade FAIL only if the excerpts are entirely \
off-topic or generic boilerplate unrelated to the question.
First check: is the question even about the same general subject as the \
excerpts (e.g. company finances, business risks, filings)? If the question \
is about something unrelated to finance/business entirely (recipes, \
sports, weather, etc.), grade FAIL regardless of how detailed the excerpts \
are — detailed financial content does not become relevant just because \
it's detailed.
Example (PASS): Question asks about mineral sourcing. An excerpt says \
"customers are seeking information about mineral sourcing in our supply \
chain" but gives no further detail. This is PASS — it is relevant, just \
incomplete. Incompleteness alone is never a reason to grade FAIL.
Example (FAIL): Question asks for a cookie recipe. Excerpts discuss \
supply chain risk and cash flow management. This is FAIL — the excerpts \
are detailed and on-topic for finance, but finance is not what the \
question asked about.
Respond with exactly one word: PASS or FAIL.
Answer:"""


def _format_excerpts(chunks: list[dict], max_chars_each: int = 400) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = c["chunk_text"][:max_chars_each].replace("\n", " ")
        lines.append(f"[{i+1}] ({c.get('ticker', '?')}) {snippet}")
    return "\n\n".join(lines)


def grader_node(state: dict) -> dict:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        logger.info("Grader: no chunks retrieved -> automatic FAIL")
        return {"grade": "fail"}

    llm = get_llm_client()
    prompt = GRADE_PROMPT_TEMPLATE.format(
        question=question,
        excerpts=_format_excerpts(chunks),
    )

    response = llm.generate(prompt, role="retriever_grader", temperature=0.0)
    verdict = "pass" if "PASS" in response.upper() else "fail"

    logger.info(f"Grader: {verdict.upper()} (raw response: {response[:50]!r})")
    return {"grade": verdict}


def route_after_grade(state: dict) -> str:
    """
    Conditional edge function: decides where the graph goes after grading.

    - PASS -> proceed to Analyst.
    - FAIL, retries remaining -> loop back to Retriever (broadened).
    - FAIL, retries exhausted -> proceed to Analyst anyway, flagged
      low_confidence, rather than failing outright.
    """
    MAX_RETRIES = 1
    grade = state.get("grade", "fail")
    retry_count = state.get("retry_count", 0)

    if grade == "pass":
        return "analyst"
    if retry_count < MAX_RETRIES:
        return "retry"
    return "analyst_low_confidence"