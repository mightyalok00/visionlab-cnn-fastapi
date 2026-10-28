.PHONY: help setup test train serve docker-build docker-up clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

# Fallback for Windows shells if invoked under make
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
    PYTEST := .venv/Scripts/pytest.exe
    UVICORN := .venv/Scripts/uvicorn.exe
endif

help:
	@echo "VisionLab 2.0 Commands:"
	@echo "  make setup          Create venv and install dependencies"
	@echo "  make test           Run test suite with pytest"
	@echo "  make train          Train CNN variants and generate benchmark"
	@echo "  make serve          Start FastAPI server with hot-reload"
	@echo "  make docker-up      Build and run containerized service"
	@echo "  make clean          Remove pycache and test caches"

setup:
	@command -v uv >/dev/null 2>&1 && uv venv --python 3.11 && uv pip install -r requirements.txt || (python3.11 -m venv .venv && $(PYTHON) -m pip install -r requirements.txt)

test:
	$(PYTEST) -v

train:
	$(PYTHON) -m scripts.train --epochs 8 --batch-size 128

serve:
	$(UVICORN) app.main:app --reload --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
