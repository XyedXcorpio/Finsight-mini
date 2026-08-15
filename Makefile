.PHONY: help setup up down models smoke smoke-spark smoke-ollama clean

help:
	@echo "FinSight Mini — commands"
	@echo "  make setup         Create venv, install deps"
	@echo "  make up            Start MinIO + Ollama"
	@echo "  make down          Stop services (keeps data)"
	@echo "  make models        Pull llama3.2:3b"
	@echo "  make smoke         Run both smoke tests"
	@echo "  make smoke-spark   Run only the Spark/Delta/MinIO smoke test"
	@echo "  make smoke-ollama  Run only the Ollama smoke test"
	@echo "  make clean         Stop services AND wipe volumes (destructive)"

setup:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"

up:
	docker compose up -d
	@echo "MinIO console: http://localhost:9001  (minioadmin/minioadmin)"
	@echo "Ollama API:    http://localhost:11434"

down:
	docker compose down

models:
	docker exec finsight-ollama ollama pull llama3.2:3b

smoke: smoke-spark smoke-ollama

smoke-spark:
	. .venv/bin/activate && python scripts/hello_spark.py

smoke-ollama:
	. .venv/bin/activate && python scripts/hello_ollama.py

clean:
	docker compose down -v
