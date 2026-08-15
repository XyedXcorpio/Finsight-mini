# FinSight Mini

> A small, real agentic RAG system over SEC filings. Built as a weekend project to keep AI engineering skills sharp while job hunting — not a capstone, not a startup. Scoped deliberately small so it actually gets finished.

## What it does

Ask a question about a company's recent SEC filings (10-K / 10-Q), get a grounded, citation-backed answer:

> "What supply chain risks does NVIDIA mention in its most recent 10-Q?"

The system retrieves relevant filing sections using hybrid search (dense embeddings + BM25), grades its own retrieval quality, and generates an answer with the LLM — falling back to a broader search once if the first retrieval looks weak.

## Why it's scoped this way

This started as a 15-week, 4-agent, full-streaming-pipeline capstone plan. After the first weekend of pure infrastructure wrangling (GPU drivers, Docker networking, MLflow setup) with zero AI code written, the scope got cut hard:

| Cut | Kept instead |
|---|---|
| Streaming + CDC ingestion | One-shot batch script |
| 4-agent graph (Router/Retriever/Analyst/Critic) | 2-agent loop (Retriever ↔ Analyst, self-grading) |
| Separate MCP tool server | Plain Python functions as LangGraph tools |
| MLflow + Ragas formal eval | 15-20 question markdown eval table |
| Streamlit UI | FastAPI + plain HTML/JS (deploys free, no framework lock-in) |

What's **not** cut, because it's the actual interesting engineering: hybrid retrieval (dense + BM25 with rank fusion), semantic chunking, a self-correcting retrieval loop, and a clean local/cloud LLM provider abstraction.

## Architecture

```
SEC EDGAR (batch download)
        │
        ▼
  Bronze (raw) ──► Silver (cleaned) ──► Semantic chunks ──► LanceDB
  [Delta Lake, one-shot]                                    (dense + BM25)
                                                                   │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  Retriever node │
                                                          │  (grades its    │
                                                          │   own results)  │
                                                          └────────┬────────┘
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  Analyst node   │
                                                          │  (cites sources)│
                                                          └────────┬────────┘
                                                                   ▼
                                                          FastAPI + HTML/JS
```

**Local dev:** LLM calls go to Ollama (Llama 3.1 8B / Mistral 7B).
**Deployed (Hugging Face Spaces):** same code, LLM calls go to Groq's free API instead — see `common/llm_client.py`.

## Stack

| Layer | Tool |
|---|---|
| Batch processing | PySpark + Delta Lake (local, via MinIO) |
| Embeddings | sentence-transformers (bge) |
| Vector + lexical search | LanceDB + rank-bm25 |
| Agent framework | LangGraph (2 nodes) |
| LLM (dev) | Ollama |
| LLM (deployed) | Groq (free tier) |
| Backend | FastAPI |
| Frontend | Plain HTML/JS |
| Deployment | Hugging Face Spaces (free, Docker) |

## Project structure

```
finsight-mini/
├── common/            # llm_client.py — provider abstraction
├── pipelines/         # SEC download + Bronze/Silver batch scripts
├── knowledge/         # chunking, embedding, LanceDB indexing
├── agents/            # the 2-node LangGraph loop
├── app/                # FastAPI backend + static frontend + Dockerfile
├── eval/               # hand-written question set + results table
├── scripts/            # setup, smoke tests
└── BUILD_JOURNAL.md
```

## Build plan (4 weekends, ~3 hrs each)

- [ ] **Weekend 1** — Download SEC filings, Bronze → Silver Delta tables
- [ ] **Weekend 2** — Semantic chunking, embeddings, hybrid LanceDB index
- [ ] **Weekend 3** — Retriever ↔ Analyst LangGraph loop
- [ ] **Weekend 4** — FastAPI + frontend, eval table, deploy to HF Spaces

## Quickstart (once built)

```bash
make setup          # venv + deps
make up              # MinIO + Ollama
make models          # pull llama3.1:8b, mistral:7b
python -m pipelines.ingest_sec --tickers NVDA AMD --quarters 4
python -m knowledge.build_index
uvicorn app.main:app --reload
```

## License

MIT
