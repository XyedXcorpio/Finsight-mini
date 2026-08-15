"""
api/main.py
=============

Weekend 4: FastAPI backend for FinSight Mini.

Wraps the existing LangGraph agent pipeline (agents/graph.py) behind a
single POST endpoint, and serves the plain HTML/JS frontend as static
files from the same process — one container, one port, simplest possible
deployment to Hugging Face Spaces.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health   -> liveness check
    POST /query    -> {"question": str} -> {"answer", "sources", "ticker", "low_confidence"}
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

sys.path.insert(0, ".")
from agents.graph import run_query  # noqa: E402

app = FastAPI(title="FinSight Mini API", version="1.0.0")

# Same-origin on HF Spaces, but useful for local dev if the frontend is
# opened directly as a file:// page or served from a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class Source(BaseModel):
    chunk_id: str
    chunk_text: str
    ticker: str | None
    accession_number: str | None
    rrf_score: float
    found_by: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    ticker: str | None
    low_confidence: bool
    retries_used: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info(f"API received query: {question!r}")

    try:
        result = run_query(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return QueryResponse(
        answer=result.get("final_answer", "(no answer produced)"),
        sources=result.get("retrieved_chunks", []),
        ticker=result.get("ticker"),
        low_confidence=result.get("low_confidence", False),
        retries_used=result.get("retry_count", 0),
    )


# Serve the frontend last, so it doesn't shadow the API routes above.
static_dir = Path(__file__).parent.parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
