PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python

.PHONY: help venv install-local install-cuda validate-data prepare-data test demo eval benchmark-summary

help:
	@echo "Available targets:"
	@echo "  make venv"
	@echo "  make install-local"
	@echo "  make install-cuda"
	@echo "  make validate-data"
	@echo "  make prepare-data"
	@echo "  make test"
	@echo "  make demo"
	@echo "  make eval"
	@echo "  make benchmark-summary"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install-local: venv
	$(PIP) install -e ".[demo,dev]"

install-cuda: venv
	$(PIP) install -e ".[cuda_train,demo,dev]"

validate-data:
	PYTHONPATH=src $(PYTHON_BIN) scripts/validate_dataset.py --source data/raw

prepare-data:
	PYTHONPATH=src $(PYTHON_BIN) scripts/prepare_dataset.py --source data/raw --output-dir data/processed

test:
	PYTHONPATH=src $(PYTHON_BIN) -m pytest tests

demo:
	PYTHONPATH=src $(PYTHON_BIN) scripts/run_demo.py --model-id google/gemma-4-4b-it --adapter-path artifacts/adapters/latest

eval:
	PYTHONPATH=src $(PYTHON_BIN) scripts/evaluate.py --config configs/eval.first_run.yaml --output artifacts/evals/first_run_eval.json

benchmark-summary:
	PYTHONPATH=src $(PYTHON_BIN) scripts/summarize_eval.py --input artifacts/evals/first_run_eval.json --output artifacts/evals/first_run_summary.md
