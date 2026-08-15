"""
agents/retriever.py
=====================

Weekend 3: the Retriever node.

Responsibilities:
1. On first call, detect a ticker from the question (if any) and run a
   ticker-filtered hybrid search.
2. On retry (after the Grader marks the first attempt "fail"), BROADEN
   the search by dropping the ticker filter — a filtered search that
   failed might succeed unfiltered if the relevant content exists but
   was mis-attributed, or if the question needed cross-company context
   the filter was wrongly excluding.
"""

from __future__ import annotations

import sys

from loguru import logger

sys.path.insert(0, ".")
from common.tickers import detect_ticker  # noqa: E402
from knowledge.hybrid_search import hybrid_search  # noqa: E402

TOP_K = 5


def retriever_node(state: dict) -> dict:
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    if retry_count == 0:
        # First attempt: detect ticker, filter if found.
        ticker = detect_ticker(question)
        logger.info(f"Retriever (attempt 1): ticker={ticker or 'none detected'}")
    else:
        # Retry: broaden by dropping any ticker filter, regardless of
        # what was detected the first time.
        ticker = None
        logger.info(f"Retriever (retry {retry_count}): broadened, no ticker filter")

    chunks = hybrid_search(question, k=TOP_K, ticker=ticker)
    logger.info(f"  -> {len(chunks)} chunks retrieved")

    return {
        "ticker": ticker,
        "retrieved_chunks": chunks,
    }
