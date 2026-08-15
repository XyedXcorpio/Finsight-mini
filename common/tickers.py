"""
common/tickers.py
===================

Weekend 3: detect which company (if any) a question is about, so the
Retriever can filter search results to that ticker.

This resolves a gap deliberately deferred from Weekend 2: hybrid_search
had no way to constrain results to one company, so a question specifically
about NVIDIA could surface AMD chunks that happened to rank nearby. This
module + hybrid_search's new `ticker` parameter close that gap.

We match both raw ticker symbols (NVDA) and common company names (nvidia)
since a real user is far more likely to type "NVIDIA" than "NVDA".
"""

from __future__ import annotations

import re

# Aliases map to the canonical ticker used everywhere else in the pipeline
# (Bronze/Silver `ticker` column, LanceDB `ticker` field, BM25 metadata).
TICKER_ALIASES: dict[str, str] = {
    "nvda": "NVDA", "nvidia": "NVDA",
    "amd": "AMD", "advanced micro devices": "AMD",
    "intc": "INTC", "intel": "INTC",
    "tsla": "TSLA", "tesla": "TSLA",
    "aapl": "AAPL", "apple": "AAPL",
}

KNOWN_TICKERS = sorted(set(TICKER_ALIASES.values()))

# Longest aliases first so "advanced micro devices" matches before a
# shorter substring could interfere.
_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(TICKER_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def detect_ticker(question: str) -> str | None:
    """
    Return the canonical ticker mentioned in a question, or None if the
    question doesn't clearly reference exactly one known company.

    If multiple different tickers are mentioned (a comparative question,
    e.g. "NVIDIA vs AMD"), we deliberately return None — filtering to a
    single ticker would be wrong for a cross-company question. The
    Retriever falls back to an unfiltered search in that case, which is
    the correct behavior for comparative queries.
    """
    matches = _ALIAS_PATTERN.findall(question)
    if not matches:
        return None

    resolved = {TICKER_ALIASES[m.lower()] for m in matches}
    if len(resolved) == 1:
        return resolved.pop()
    return None  # multiple distinct companies mentioned -> no single filter

# Common tickers/company names people might reasonably ask about that are
# NOT in this corpus. Not exhaustive — just enough well-known names to
# short-circuit the obvious cases with a guaranteed-correct answer instead
# of relying on the LLM to hedge correctly every time.
OUT_OF_SCOPE_ALIASES: dict[str, str] = {
    "googl": "GOOGL", "goog": "GOOGL", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "msft": "MSFT", "microsoft": "MSFT",
    "amzn": "AMZN", "amazon": "AMZN",
    "ibm": "IBM",
    "qcom": "QCOM", "qualcomm": "QCOM",
    "avgo": "AVGO", "broadcom": "AVGO",
    "twtr": "TWTR", "twitter": "TWTR",
    "netflix": "NFLX", "nflx": "NFLX",
}

_OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(OUT_OF_SCOPE_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def detect_out_of_scope(question: str) -> str | None:
    """
    Return the canonical name of a well-known, clearly out-of-corpus
    company mentioned in the question, or None. This is a deterministic
    regex check, not an LLM judgment call — after two prompt-only attempts
    misfired on topic-only questions (e.g. "mineral sourcing risks" being
    misread as an out-of-scope company), a fixed alias match is the
    reliable way to catch this specific case without touching how the
    Analyst handles everything else.
    """
    match = _OUT_OF_SCOPE_PATTERN.search(question)
    if match:
        return OUT_OF_SCOPE_ALIASES[match.group(1).lower()]
    return None