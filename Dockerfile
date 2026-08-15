# FinSight Mini — deployed container (Render)
#
# The free tier host has no GPU and no local Ollama daemon, so this
# container always runs with LLM_PROVIDER=groq (set below). Your local
# WSL dev flow is unaffected — this Dockerfile is only used for the
# deployed service; locally you still run `docker compose up -d` +
# `uvicorn api.main:app --reload` against Ollama as usual.
#
# Required environment variable (set in Render dashboard): GROQ_API_KEY
# Render also injects PORT at runtime — read dynamically below, not hardcoded.

FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal — no PySpark/Java here, since the deployed
# Space only serves the query-time API (LanceDB + BM25 + LangGraph),
# not the offline ingestion pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Pre-download the embedding model into the image at build time. Without
# this, SentenceTransformer() tries to fetch model weights from HF Hub on
# the FIRST query in a fresh container — Render's container has no
# persistent cache between deploys, so every cold start would hit the
# network, and unauthenticated HF Hub requests are rate-limited enough to
# cause outright request failures (502s) rather than just added latency.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Force the hosted LLM provider — no local model available on Spaces.
ENV LLM_PROVIDER=groq

COPY agents/ ./agents/
COPY common/ ./common/
COPY knowledge/ ./knowledge/
COPY api/ ./api/
COPY static/ ./static/

# Prebuilt indexes committed to the repo (LanceDB dir + BM25 pickle).
# These are query-time artifacts from Weekends 1-2, not regenerated here.
COPY data/lancedb/ ./data/lancedb/
COPY data/bm25_index.pkl ./data/bm25_index.pkl


EXPOSE 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}