"""
common.llm_client
==================

One interface, two backends. This is the single most important design
decision in FinSight Mini: every agent node calls `LLMClient.generate(...)`
and never touches Ollama or Groq directly.

Why this exists:
- Locally, you develop against Ollama — free, private, but slow on a P620.
- Deployed on Hugging Face Spaces, there's no GPU and no Ollama — so the
  same agent code calls Groq's free API instead (fast, hosted, free tier).
- Switching between them is one environment variable: LLM_PROVIDER.
- No agent code changes between local dev and cloud deployment. This is
  the kind of provider abstraction real LLM apps use in production (the
  same pattern shows up as "model routing" or "provider adapters").

Usage:
    from common.llm_client import get_llm_client
    llm = get_llm_client()                      # reads LLM_PROVIDER from env
    answer = llm.generate("Summarize this filing...", model="analyst")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# Logical role -> concrete model name, per provider. Agents ask for a role
# ("router", "analyst") not a model string, so swapping providers or model
# versions never touches agent code.
MODEL_MAP = {
    "ollama": {
        # Single model for all roles. The Quadro P620 has 4GB VRAM — running
        # two different models means Ollama evicts and reloads on every role
        # switch, adding 15-25s of dead time per call, which was misread as
        # "slow inference" during setup. See BUILD_JOURNAL.md.
        "router": "llama3.2:3b",
        "retriever_grader": "llama3.2:3b",
        "analyst": "llama3.2:3b",
    },
    "groq": {
        # Groq's free tier serves Llama models at very high tok/s.
        "router": "llama-3.1-8b-instant",
        "retriever_grader": "llama-3.1-8b-instant",
        "analyst": "llama-3.1-8b-instant",
    },
}


class LLMClient(ABC):
    """Common interface every provider backend implements."""

    @abstractmethod
    def generate(self, prompt: str, *, role: str, temperature: float = 0.0) -> str:
        """Generate a completion for the given role (router/analyst/etc.)."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Local inference via Ollama. Used in development."""

    def __init__(self, host: str | None = None) -> None:
        import ollama as ollama_sdk

        self._sdk = ollama_sdk
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        logger.info(f"LLMClient: using Ollama at {self._host}")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def generate(self, prompt: str, *, role: str, temperature: float = 0.0) -> str:
        model = MODEL_MAP["ollama"][role]
        response = self._sdk.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response["response"].strip()


class GroqClient(LLMClient):
    """Hosted inference via Groq's free API. Used in the deployed demo."""

    def __init__(self, api_key: str | None = None) -> None:
        from groq import Groq

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "and set it in your .env file."
            )
        self._client = Groq(api_key=key)
        logger.info("LLMClient: using Groq (hosted)")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def generate(self, prompt: str, *, role: str, temperature: float = 0.0) -> str:
        model = MODEL_MAP["groq"][role]
        completion = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()


def get_llm_client() -> LLMClient:
    """
    Factory: returns the configured LLM backend based on LLM_PROVIDER env var.

    LLM_PROVIDER=ollama (default)  -> local dev
    LLM_PROVIDER=groq              -> cloud deployment (Hugging Face Spaces)
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "groq":
        return GroqClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'ollama' or 'groq')")
