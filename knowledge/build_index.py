"""
knowledge/build_index.py
==========================

Weekend 2: chunk every Silver filing, embed the chunks, and build both
halves of hybrid retrieval:
  - Dense: LanceDB table with vector embeddings, for semantic similarity search
  - Lexical: BM25 index over the same chunks, for exact keyword/term matching

Why hybrid, not just dense: dense embeddings are excellent at paraphrase and
conceptual similarity ("supply chain risk" matches "dependency on suppliers")
but weak on exact tokens — ticker symbols, dollar figures, specific product
names. BM25 is the reverse: exact-term matching, no paraphrase understanding.
Combining both (via reciprocal rank fusion at query time, in
knowledge/hybrid_search.py) covers both failure modes.

Usage:
    python -m knowledge.build_index
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lancedb
import pyarrow as pa
from loguru import logger
from rank_bm25 import BM25Okapi

sys.path.insert(0, ".")
from common.spark_utils import get_spark  # noqa: E402
from knowledge.chunker import semantic_chunk  # noqa: E402
from knowledge.embedder import embed_passages, embedding_dim  # noqa: E402

SILVER_PATH = "s3a://finsight-silver/sec_filings_clean"
LANCEDB_DIR = "data/lancedb"
LANCEDB_TABLE = "chunks"
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def load_filings() -> list[dict]:
    """Read all filings from Silver into a list of plain dicts (small dataset, fine in memory)."""
    spark = get_spark("build_index")
    df = spark.read.format("delta").load(SILVER_PATH)
    rows = df.select(
        "ticker", "filing_type", "accession_number", "clean_text", "word_count"
    ).collect()
    return [row.asDict() for row in rows]


def build_chunks(filings: list[dict]) -> list[dict]:
    """
    Chunk every filing's clean_text and flatten into one list of chunk
    records, each tagged with its source filing's metadata.
    """
    all_chunks: list[dict] = []
    chunk_counter = 0

    for filing in filings:
        ticker = filing["ticker"]
        accession = filing["accession_number"]
        text = filing["clean_text"]

        logger.info(f"Chunking {ticker} {accession} ({filing['word_count']} words)...")
        chunks = semantic_chunk(text, embed_fn=embed_passages)
        logger.info(f"  -> {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": f"{accession}_{chunk_counter}",
                "ticker": ticker,
                "filing_type": filing["filing_type"],
                "accession_number": accession,
                "chunk_text": chunk.text,
                "char_count": chunk.char_count,
                "paragraph_count": chunk.paragraph_count,
            })
            chunk_counter += 1

    return all_chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add a `vector` field to every chunk dict, embedding in one batched call."""
    logger.info(f"Embedding {len(chunks)} chunks (this is the slow part, be patient)...")
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_passages(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector.tolist()

    return chunks


def write_lancedb(chunks: list[dict]) -> None:
    """Write chunks + embeddings to a LanceDB table (dense retrieval side)."""
    dim = embedding_dim()
    logger.info(f"Writing {len(chunks)} chunks to LanceDB (dim={dim})...")

    db = lancedb.connect(LANCEDB_DIR)

    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("filing_type", pa.string()),
        pa.field("accession_number", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("char_count", pa.int64()),
        pa.field("paragraph_count", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    db.create_table(LANCEDB_TABLE, data=chunks, schema=schema, mode="overwrite")
    logger.info("LanceDB table written.")


def write_bm25_index(chunks: list[dict]) -> None:
    """
    Build and persist the BM25 (lexical) index alongside LanceDB.

    rank_bm25 has no built-in persistence, so we pickle the BM25Okapi object
    together with FULL metadata for every chunk (chunk_id, ticker,
    accession_number, filing_type, chunk_text) — not just chunk_id and text.

    Earlier version stored only chunk_ids + corpus_texts, which meant any
    chunk found ONLY by BM25 (never surfaced by dense search) showed up in
    hybrid_search results with ticker=None, accession_number=None — the
    metadata simply wasn't available anywhere in the BM25-only code path.
    Storing full metadata here fixes that at the source.
    """
    logger.info("Building BM25 index...")
    corpus_texts = [c["chunk_text"] for c in chunks]
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Full metadata per chunk, same order as tokenized_corpus/chunk_ids,
    # so hybrid_search can always attribute a BM25-only result correctly.
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "ticker": c["ticker"],
            "filing_type": c["filing_type"],
            "accession_number": c["accession_number"],
        }
        for c in chunks
    ]

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "corpus_texts": corpus_texts,
                "metadata": metadata,
            },
            f,
        )

    logger.info(f"BM25 index written to {BM25_INDEX_PATH}")


def main() -> int:
    filings = load_filings()
    logger.info(f"Loaded {len(filings)} filings from Silver")

    chunks = build_chunks(filings)
    logger.info(f"Total chunks across all filings: {len(chunks)}")

    if not chunks:
        logger.error("No chunks produced — check Silver table and chunker settings.")
        return 1

    chunks = embed_chunks(chunks)

    write_lancedb(chunks)
    write_bm25_index(chunks)

    avg_chunk_size = sum(c["char_count"] for c in chunks) / len(chunks)
    chunks_per_filing = len(chunks) / len(filings)
    logger.info(f"Average chunk size: {avg_chunk_size:.0f} chars")
    logger.info(f"Average chunks per filing: {chunks_per_filing:.1f}")

    logger.info("Index build complete. Next: knowledge/hybrid_search.py to test retrieval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())