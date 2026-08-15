"""
common.llm_client
==================

One interface, two backends: Ollama (local dev) and Groq (cloud deploy).
Switch via the LLM_PROVIDER env var. Agent code never touches either
provider directly.

Usage:
    from common.llm_client import get_llm_client
    llm = get_llm_client()
    answer = llm.generate("Summarize this filing...", role="analyst")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

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
        # llama-3.1-8b-instant is being decommissioned by Groq on 2026-08-16.
        # Migrated to their recommended replacement, openai/gpt-oss-20b — a
        # reasoning-capable MoE model (20B total params, ~3.6B active per
        # forward pass). Note: this is architecturally different from the
        # plain instruction-following Llama model our prompts were tuned
        # against (grader.py's PASS/FAIL prompt, analyst.py's citation
        # format) — worth re-verifying grader calibration specifically,
        # since we already found once that prompt framing meaningfully
        # affects this model family's PASS/FAIL bias (see BUILD_JOURNAL.md).
        "router": "openai/gpt-oss-20b",
        "retriever_grader": "openai/gpt-oss-20b",
        "analyst": "openai/gpt-oss-20b",
    },
}


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, role: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
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
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "groq":
        return GroqClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'ollama' or 'groq')")