---
title: FinSight Mini
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# FinSight Mini

Agentic RAG over SEC 10-Q filings (NVDA, AMD, INTC, TSLA, AAPL) using a
LangGraph Retriever → Grader → Analyst loop with hybrid (dense + BM25)
retrieval and ticker-aware filtering.

**Stack:** FastAPI · LangGraph · LanceDB · BM25 · BGE-small embeddings ·
Groq (`llama-3.1-8b-instant`)

Ask a question about supply chain risk, revenue trends, or risk factors
across any of the five covered companies, and get an answer grounded in
retrieved 10-Q filing excerpts with source attribution.

## Required Space secret

- `GROQ_API_KEY` — set under Space Settings → Repository secrets

## Local development

This Space runs the query-time API only. For the full pipeline
(ingestion, embedding, index building), see the main project README.

```bash
uvicorn api.main:app --reload --port 8000
```
