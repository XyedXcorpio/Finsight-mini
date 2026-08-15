"""
knowledge/embedder.py
=======================

Weekend 2: wraps a sentence-transformers model for embedding generation.

Model choice: BAAI/bge-small-en-v1.5 (384-dim), not bge-large (1024-dim).
bge-small is meaningfully faster on CPU with a small quality trade-off —
worth it for a weekend project embedding ~10 filings' worth of chunks.

Device: forced to CPU explicitly. The Quadro P620 is a Pascal-generation
card (compute capability 6.1 / sm_61) — current PyTorch wheels only ship
compiled kernels for sm_75 and newer, so CUDA auto-detection finds the GPU
but fails at the first actual kernel launch (torch.AcceleratorError: no
kernel image available). This is a different issue than the Ollama GPU
setup: Ollama bundles its own llama.cpp CUDA build which still supports
Pascal; PyTorch's official wheels have dropped it. Not worth chasing a fix
here — bge-small is a 33M-parameter model, genuinely fast on CPU, and our
corpus (10 filings) is tiny. GPU would help but isn't needed at this scale.

BGE models also have a quirk worth knowing: they expect a specific
instruction prefix on QUERY text (not on the documents/passages being
indexed) to get their best retrieval performance. We handle that here so
callers never have to remember it themselves.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily load the model once per process (loading takes a few seconds).
    Forces CPU explicitly — see module docstring for why.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    """
    Embed a list of document/passage texts (chunks going INTO the index).
    No instruction prefix — BGE's convention is prefix-free for passages.
    """
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single search query. Uses BGE's query instruction prefix,
    which measurably improves retrieval quality for this model family.
    """
    model = _get_model()
    prefixed = QUERY_INSTRUCTION + query
    return model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0]


def embedding_dim() -> int:
    """Return the embedding dimensionality (needed when creating the LanceDB table)."""
    return _get_model().get_sentence_embedding_dimension()