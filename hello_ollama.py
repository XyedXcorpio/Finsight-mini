"""
scripts/hello_ollama.py
========================

Smoke test for Ollama (FinSight Mini).

IMPORTANT: measures speed using Ollama's own `eval_duration` field, not
wall-clock time. Wall-clock time includes model load time (can be 5-90s
on a cold start or when switching between models on limited VRAM), which
badly understates true generation speed. See BUILD_JOURNAL.md for the
debugging story behind this.

Run:
    python scripts/hello_ollama.py
"""

from __future__ import annotations

import sys

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
    print(f">>> Testing {model_name}")

    print("    warming up (loading into VRAM if needed)...")
    ollama.generate(model=model_name, prompt="hi", options={"num_predict": 5})

    prompt = "In one sentence, what is a vector database?"
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"num_predict": 80, "temperature": 0.0},
    )

    load_s = response.get("load_duration", 0) / 1e9
    eval_s = response.get("eval_duration", 0) / 1e9
    tokens = response.get("eval_count", 0)
    tps = tokens / eval_s if eval_s > 0 else 0.0
    answer = response["response"].strip()

    if tps < 2:
        verdict = "CPU only — GPU not active"
    elif tps < 8:
        verdict = "Partial GPU offload"
    elif tps < 20:
        verdict = "Good — GPU active"
    else:
        verdict = "Excellent — full GPU"

    print(f"    load_duration (warm) : {load_s:.3f}s")
    print(f"    eval_duration         : {eval_s:.3f}s")
    print(f"    tokens                : {tokens}")
    print(f"    TRUE tok/s            : {tps:.1f}  [{verdict}]")
    print(f"    answer                : {answer[:200]}")
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
    print("    Target: >=8 tok/s (measured on eval_duration, not wall clock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
