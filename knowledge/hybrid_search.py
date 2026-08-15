"""
knowledge/hybrid_search.py
============================

Weekend 2/3: query-time hybrid retrieval combining dense + lexical search,
with optional ticker filtering (added Weekend 3).

Reciprocal Rank Fusion (RRF) is the fusion method: instead of trying to
normalize and combine dense similarity scores with BM25 scores directly
(different scales, not comparable), RRF combines RANKS.

    RRF_score(doc) = sum over each ranker: 1 / (k + rank_in_that_ranker)

Ticker filtering (Weekend 3): applied INSIDE each ranker, not as a post-hoc
filter on fused results. Filtering after ranking would silently drop
relevant chunks that ranked outside the top-N before filtering was applied.
Dense filtering uses LanceDB's native `.where()`; BM25 filtering restricts
the candidate pool to matching indices before ranking.

Usage:
    from knowledge.hybrid_search import hybrid_search
    results = hybrid_search("What supply chain risks does NVIDIA mention?",
                             k=5, ticker="NVDA")

    python -m knowledge.hybrid_search "your query" [TICKER]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lancedb
from loguru import logger

sys.path.insert(0, ".")
from knowledge.embedder import embed_query  # noqa: E402

LANCEDB_DIR = "data/lancedb"
LANCEDB_TABLE = "chunks"
BM25_INDEX_PATH = Path("data/bm25_index.pkl")

RRF_K = 60
DENSE_TOP_N = 20
BM25_TOP_N = 20

_bm25_cache = None


def _load_bm25():
    global _bm25_cache
    if _bm25_cache is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_cache = pickle.load(f)
    return _bm25_cache


def dense_search(query: str, top_n: int = DENSE_TOP_N, ticker: str | None = None) -> list[dict]:
    """
    Vector similarity search via LanceDB. Returns chunks ranked by cosine
    similarity. If `ticker` is given, uses LanceDB's native `.where()` to
    restrict the search to that company BEFORE ranking — not after.
    """
    db = lancedb.connect(LANCEDB_DIR)
    table = db.open_table(LANCEDB_TABLE)

    query_vector = embed_query(query)
    search = table.search(query_vector)
    if ticker:
        search = search.where(f"ticker = '{ticker}'")
    results = search.limit(top_n).to_list()
    return results


def bm25_search(query: str, top_n: int = BM25_TOP_N, ticker: str | None = None) -> list[dict]:
    """
    Lexical search via BM25. Returns chunks ranked by term-overlap score,
    each carrying full metadata. If `ticker` is given, only chunks
    belonging to that company are considered at all — the candidate pool
    is restricted before ranking, not filtered after.
    """
    index = _load_bm25()
    bm25 = index["bm25"]
    chunk_ids = index["chunk_ids"]
    corpus_texts = index["corpus_texts"]
    metadata = index["metadata"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    candidate_indices = range(len(scores))
    if ticker:
        candidate_indices = [i for i in candidate_indices if metadata[i]["ticker"] == ticker]

    ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_n]

    results = []
    for i in ranked:
        if scores[i] <= 0:
            continue
        record = {
            "chunk_id": chunk_ids[i],
            "chunk_text": corpus_texts[i],
            "bm25_score": scores[i],
        }
        record.update(metadata[i])
        results.append(record)
    return results


def hybrid_search(query: str, k: int = 5, ticker: str | None = None) -> list[dict]:
    """
    Combine dense + BM25 results via Reciprocal Rank Fusion, optionally
    constrained to a single ticker.

    Returns the top `k` chunks by fused score, each annotated with which
    ranker(s) it came from.
    """
    dense_results = dense_search(query, ticker=ticker)
    bm25_results = bm25_search(query, ticker=ticker)

    dense_ranks = {r["chunk_id"]: i for i, r in enumerate(dense_results)}
    bm25_ranks = {r["chunk_id"]: i for i, r in enumerate(bm25_results)}

    all_chunk_ids = set(dense_ranks) | set(bm25_ranks)

    chunk_lookup = {r["chunk_id"]: r for r in dense_results}
    for r in bm25_results:
        chunk_lookup.setdefault(r["chunk_id"], r)

    fused = []
    for chunk_id in all_chunk_ids:
        score = 0.0
        in_dense = chunk_id in dense_ranks
        in_bm25 = chunk_id in bm25_ranks
        if in_dense:
            score += 1.0 / (RRF_K + dense_ranks[chunk_id])
        if in_bm25:
            score += 1.0 / (RRF_K + bm25_ranks[chunk_id])

        record = chunk_lookup[chunk_id]
        fused.append({
            "chunk_id": chunk_id,
            "chunk_text": record.get("chunk_text", ""),
            "ticker": record.get("ticker"),
            "accession_number": record.get("accession_number"),
            "rrf_score": score,
            "found_by": ("dense" if in_dense else "") + ("+bm25" if in_bm25 else ""),
        })

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:k]


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m knowledge.hybrid_search "your query here" [TICKER]')
        return 1

    ticker = None
    args = sys.argv[1:]
    if args[-1].isupper() and len(args[-1]) <= 5 and args[-1].isalpha():
        ticker = args[-1]
        args = args[:-1]

    query = " ".join(args)
    logger.info(f"Query: {query}" + (f"  [ticker={ticker}]" if ticker else ""))

    results = hybrid_search(query, k=5, ticker=ticker)

    print()
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (score={r['rrf_score']:.4f}, found_by={r['found_by']}) ---")
        print(f"Ticker: {r['ticker']}  |  Accession: {r['accession_number']}")
        print(r["chunk_text"][:300].replace("\n", " "))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
