#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run bash scripts/setup_cuda_env.sh first."
  exit 1
fi

source .venv/bin/activate

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

mkdir -p artifacts/checkpoints artifacts/adapters artifacts/evals

PYTHONPATH=src python scripts/validate_dataset.py --source data/raw
PYTHONPATH=src python scripts/prepare_dataset.py --source data/raw --output-dir data/processed
PYTHONPATH=src python scripts/train_unsloth.py --config configs/train.first_run.yaml
PYTHONPATH=src python scripts/evaluate.py --config configs/eval.first_run.yaml --output "${EVAL_OUTPUT_PATH:-artifacts/evals/first_run_eval.json}"
PYTHONPATH=src python scripts/summarize_eval.py --input "${EVAL_OUTPUT_PATH:-artifacts/evals/first_run_eval.json}" --output artifacts/evals/first_run_summary.md

echo "First CUDA run complete."
