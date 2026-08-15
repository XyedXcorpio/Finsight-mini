# FinSight Mini — Build Journal

> Log decisions, bugs, and insights here after every session. Rough notes are fine — this file is for future-you, especially for interview prep.

Entry types: 🎯 Decision · 🐛 Bug · 💡 Insight · 📊 Benchmark · ❓ Open question · 📚 Reference

---

## Pre-history — Why this project was rescoped

**Original plan:** 15-week capstone. Streaming medallion architecture, 4-agent LangGraph system, MCP tool server, MLflow + Ragas eval, Streamlit UI.

**What happened:** Spent the first real session entirely on infrastructure (Docker, MinIO, Spark-Delta-S3A config, Ollama GPU passthrough) without writing a line of agent code. Hit a GPU offloading bug that took multiple back-and-forths to diagnose. Realized partway through that the actual constraints — 2-4 hrs/week, semester already finished, job-hunting in parallel, goal is "stay sharp + have something to show," not a capstone — didn't match a 15-week plan at all.

**🎯 Decision: cut to a 4-weekend project.**
- Cut streaming/CDC, 3 of 4 agents, the MCP server, MLflow/Ragas, Streamlit.
- Kept hybrid retrieval, semantic chunking, a self-grading retrieval loop, Delta Lake batch processing — the parts that are both genuinely interesting AND fast to build.
- **Why this is fine to put on a resume / in an interview**: recognizing scope creep and deliberately cutting to ship something real is itself a legitimate engineering judgment call. "I planned X, recognized it didn't fit my real constraints, and cut to Y" is a better story than an unfinished X.

**🎯 Decision: no Streamlit, deploy free on Hugging Face Spaces.**
- Streamlit Community Cloud doesn't fit well with a locally-hosted LLM, and a plain FastAPI + HTML/JS app reads as more substantial engineering anyway.
- Deployed version can't reach the local Ollama instance, so we added a provider abstraction (`common/llm_client.py`): same agent code, switches between Ollama (local dev) and Groq's free API (cloud deploy) via one env var.

---

## Weekend 1 — Data Foundation

### Decisions to make
- [ ] Confirm SEC EDGAR rate limits / contact-email policy before bulk downloading
- [ ] Schema for raw filing text — keep full HTML or strip to plain text at Bronze?

---

## Weekend 2 — Knowledge Base

### Decisions to make
- [ ] Chunking: semantic vs fixed-size — benchmark quickly, don't overthink
- [ ] Embedding model size — bge-large vs bge-small (CPU embedding speed matters here)

---

## Weekend 3 — Agent Loop

### Decisions to make
- [ ] Exact grading criteria for the Retriever's self-check (what counts as "good enough" context?)
- [ ] How many retry attempts before giving up gracefully

---

## Weekend 4 — Serving & Deploy

### Decisions to make
- [ ] Groq rate limits on free tier — confirm before relying on it for a live demo
- [ ] HF Spaces Docker resource limits — confirm LanceDB + embeddings fit in free tier RAM

---

# Resume Bullet Drafts

- [ ] Built a retrieval-augmented agent over SEC filings using hybrid (dense + BM25) search and a self-correcting LangGraph retrieval loop.
- [ ] Designed a provider-agnostic LLM abstraction enabling local development (Ollama) and free-tier cloud deployment (Groq) from identical agent code.
- [ ] Deployed a FastAPI-based agentic RAG system to Hugging Face Spaces at zero cost.

# Interview Story Bank

## "Tell me about a time you had to manage scope"
- The FinSight rescoping, above. This is a genuinely good story — use it.

## "Tell me about a technical trade-off you made"
- Local Ollama vs hosted Groq, and the abstraction that avoided forcing a single choice.

---

*Last updated: restart, see "Pre-history" above.*
