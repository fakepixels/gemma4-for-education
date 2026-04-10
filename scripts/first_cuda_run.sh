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

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p artifacts/checkpoints artifacts/adapters artifacts/evals

PYTHONPATH=src python scripts/validate_dataset.py --source data/raw
PYTHONPATH=src python scripts/prepare_dataset.py --source data/raw --output-dir data/processed

if [[ "${AUTO_OPTIMIZE_FIRST_RUN:-0}" == "1" ]]; then
  PYTHONPATH=src python scripts/optimize_first_run.py \
    --base-config "${TRAIN_CONFIG:-configs/train.first_run.yaml}" \
    --mode run \
    --goal "${OPTIMIZE_GOAL:-benchmark}"
else
  PYTHONPATH=src python scripts/train_unsloth.py --config "${TRAIN_CONFIG:-configs/train.first_run.yaml}"
fi

PYTHONPATH=src python scripts/evaluate.py --config configs/eval.first_run.yaml --output "${EVAL_OUTPUT_PATH:-artifacts/evals/first_run_eval.json}"
PYTHONPATH=src python scripts/summarize_eval.py --input "${EVAL_OUTPUT_PATH:-artifacts/evals/first_run_eval.json}" --output artifacts/evals/first_run_summary.md

echo "First CUDA run complete."
