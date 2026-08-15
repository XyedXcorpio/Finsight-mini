"""
scripts/hello_ollama.py
========================

Smoke test for Ollama (FinSight Mini). Verifies:
1. Ollama server is reachable.
2. Both required models are pulled.
3. Each model responds, with latency + tok/s reported so you can confirm
   the GPU (Quadro P620) is actually being used this time.

Run:
    python scripts/hello_ollama.py
"""

from __future__ import annotations

import sys
import time

import ollama

REQUIRED_MODELS = ["llama3.2:3b"]


def check_models_present() -> list[str]:
    try:
        response = ollama.list()
    except Exception as e:
        print(f"ERROR: cannot reach Ollama at localhost:11434: {e}")
        print("  Is the container running? Try:  docker compose ps")
        sys.exit(1)
    return [m["name"] for m in response.get("models", [])]


def smoke_test_model(model_name: str) -> None:
    prompt = "In one sentence, what is a vector database?"
    print(f">>> Testing {model_name}")

    start = time.perf_counter()
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"num_predict": 80, "temperature": 0.0},
    )
    elapsed = time.perf_counter() - start
    answer = response["response"].strip()
    tokens = response.get("eval_count", 0)
    tps = tokens / elapsed if elapsed > 0 else 0.0

    if tps < 2:
        verdict = "CPU only — GPU not active, something's still wrong"
    elif tps < 8:
        verdict = "Partial GPU offload — usable, not ideal"
    elif tps < 20:
        verdict = "Good — GPU active"
    else:
        verdict = "Excellent — full GPU"

    print(f"    latency : {elapsed:.2f}s")
    print(f"    tokens  : {tokens}  speed: {tps:.1f} tok/s  [{verdict}]")
    print(f"    answer  : {answer[:200]}")
    print()


def main() -> int:
    print(">>> Checking Ollama")
    available = check_models_present()
    print(f"    available models: {available or '(none)'}")
    print()

    missing = [m for m in REQUIRED_MODELS if m not in available]
    if missing:
        print(f"ERROR: missing models: {missing}")
        for m in missing:
            print(f"    docker exec finsight-ollama ollama pull {m}")
        return 1

    for model in REQUIRED_MODELS:
        smoke_test_model(model)

    print(">>> Ollama smoke test complete")
    print("    Target: >=8 tok/s means GPU is active and useful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
