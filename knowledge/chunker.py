"""
knowledge/chunker.py
=====================

Weekend 2: split Silver filing text into retrieval-sized chunks.

Design decision: paragraph-level semantic chunking, not sentence-level.

Your coursework's semantic splitter (PA3) embeds every SENTENCE and merges
based on cosine distance. That's precise but expensive — a 25,000-word 10-Q
has 1,000+ sentences, meaning 1,000+ embedding calls just to decide chunk
boundaries, before embedding the actual chunks for retrieval.

This chunker operates on PARAGRAPHS instead. SEC filings mostly have real
paragraph breaks (blank lines from the HTML extraction), so paragraphs are
a natural, much cheaper unit to measure semantic similarity between. Same
core idea as the coursework's approach — merge units while they stay
topically similar, split when they diverge — just at a coarser granularity
appropriate for a weekend project on a laptop.

KNOWN ISSUE (fixed here): SEC financial-statement discussion sections (e.g.
cash flow / liquidity narrative) are frequently a single unbroken run of
text with no blank-line breaks for thousands of characters — the HTML
extraction doesn't insert paragraph boundaries inside them. A first version
of this chunker treated such a run as ONE paragraph and put it in a chunk
unconditionally (the first paragraph of any chunk was never size-checked),
producing chunks up to 74,832 characters — 50x the intended ceiling. Fixed
by pre-splitting any oversized paragraph on sentence boundaries BEFORE
semantic merging begins, so nothing enters the merge loop already too big.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

MIN_PARAGRAPH_CHARS = 80          # shorter paragraphs are usually headers/noise
MAX_CHUNK_CHARS = 1500            # ceiling before we force a new chunk
SEMANTIC_DISTANCE_THRESHOLD = 0.35  # 1 - cosine_similarity; higher = more permissive merging

# Standard SEC "forward-looking statements" legal disclaimer language.
# This boilerplate appears near-verbatim in nearly every 10-Q's opening
# section and doesn't discuss anything company-specific — it's legal
# cover, not substantive content. It was polluting retrieval results:
# a query like "what supply chain risks does NVIDIA mention" returned
# this disclaimer because it generically mentions "risks" and "actual
# results," despite containing zero information about supply chains.
# We filter any paragraph matching 2+ of these signal phrases, since a
# single phrase alone could plausibly appear in real substantive text,
# but this combination is a strong, specific fingerprint of the
# boilerplate section specifically.
_BOILERPLATE_SIGNALS = [
    r"forward-looking statements",
    r"undertake no obligation to update",
    r"actual (results|future results) may (be materially different|differ materially)",
    r"safe harbor",
    r"cautionary statements",
    r"known and unknown risks",
]
_BOILERPLATE_PATTERN = re.compile("|".join(_BOILERPLATE_SIGNALS), re.IGNORECASE)


def _is_boilerplate(paragraph: str) -> bool:
    """True if a paragraph matches 2+ forward-looking-statements signal phrases."""
    matches = _BOILERPLATE_PATTERN.findall(paragraph)
    return len(matches) >= 2

# Simple sentence-boundary splitter for pre-splitting oversized paragraphs.
# Not perfect (doesn't handle "Mr. Smith" etc.), but good enough for breaking
# up a wall of financial-narrative text into sentence-ish pieces.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    char_count: int
    paragraph_count: int


def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """
    Break a single paragraph larger than max_chars into smaller pieces on
    sentence boundaries, greedily packing sentences up to the size limit.

    This runs BEFORE semantic merging, so oversized "paragraphs" (common in
    SEC financial-statement narrative with no blank-line breaks) never reach
    the merge loop as a single unsplittable unit.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = _SENTENCE_SPLIT.split(paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > max_chars and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current.strip())

    # Edge case: a single "sentence" (no punctuation at all) still over the
    # limit — hard-split on character count as a last resort so nothing
    # ever escapes with an unbounded size.
    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final_pieces.append(piece)
        else:
            for i in range(0, len(piece), max_chars):
                final_pieces.append(piece[i:i + max_chars])

    return final_pieces


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split cleaned filing text into paragraphs on blank-line boundaries.

    Filters out very short "paragraphs" (page numbers, lone headers left
    over from imperfect HTML extraction) and boilerplate forward-looking-
    statements disclaimers (see _is_boilerplate) since neither adds
    retrievable, company-specific content. Any resulting paragraph still
    larger than MAX_CHUNK_CHARS is pre-split on sentence boundaries — see
    _split_oversized_paragraph for why this matters.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    filtered = [
        p.strip() for p in raw_paragraphs
        if len(p.strip()) >= MIN_PARAGRAPH_CHARS and not _is_boilerplate(p)
    ]

    result: list[str] = []
    for para in filtered:
        result.extend(_split_oversized_paragraph(para, MAX_CHUNK_CHARS))
    return result


def semantic_chunk(
    text: str,
    embed_fn,
    distance_threshold: float = SEMANTIC_DISTANCE_THRESHOLD,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """
    Merge paragraphs into chunks based on semantic similarity.

    Algorithm:
    1. Split text into paragraphs (oversized ones pre-split on sentences).
    2. Embed every paragraph once.
    3. Walk through paragraphs in order. Keep adding to the current chunk
       as long as the new paragraph is semantically close to the chunk's
       running average embedding AND the chunk hasn't hit the size ceiling.
       Otherwise, close the current chunk and start a new one.

    Args:
        text: cleaned filing text (from Silver).
        embed_fn: callable taking list[str] -> np.ndarray of embeddings.
        distance_threshold: 1 - cosine_similarity. Paragraphs with distance
            ABOVE this from the running chunk average start a new chunk.
        max_chunk_chars: hard ceiling on chunk size regardless of semantic
            similarity.

    Returns:
        List of Chunk objects.
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    embeddings = embed_fn(paragraphs)

    chunks: list[Chunk] = []
    current_paragraphs: list[str] = [paragraphs[0]]
    current_embeddings: list[np.ndarray] = [embeddings[0]]

    def running_avg(embs: list[np.ndarray]) -> np.ndarray:
        return np.mean(embs, axis=0)

    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        return 1.0 - sim

    for para, emb in zip(paragraphs[1:], embeddings[1:]):
        current_text = "\n\n".join(current_paragraphs)
        would_exceed_size = len(current_text) + len(para) > max_chunk_chars

        chunk_avg = running_avg(current_embeddings)
        distance = cosine_distance(chunk_avg, emb)
        too_dissimilar = distance > distance_threshold

        if would_exceed_size or too_dissimilar:
            chunks.append(Chunk(
                text=current_text,
                char_count=len(current_text),
                paragraph_count=len(current_paragraphs),
            ))
            current_paragraphs = [para]
            current_embeddings = [emb]
        else:
            current_paragraphs.append(para)
            current_embeddings.append(emb)

    if current_paragraphs:
        current_text = "\n\n".join(current_paragraphs)
        chunks.append(Chunk(
            text=current_text,
            char_count=len(current_text),
            paragraph_count=len(current_paragraphs),
        ))

    return chunks


def fixed_size_chunk(
    text: str, chunk_size: int = 1000, overlap: int = 100
) -> list[Chunk]:
    """
    Simple fixed-size chunker for comparison against semantic_chunk().
    Matches the coursework's MyTextSplitter pattern.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        chunks.append(Chunk(
            text=piece,
            char_count=len(piece),
            paragraph_count=piece.count("\n\n") + 1,
        ))
        start = end - overlap
    return chunks