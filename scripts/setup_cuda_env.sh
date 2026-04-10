#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Run this on a Linux CUDA machine."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cuda_train,demo,dev]"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

PYTHONPATH=src python scripts/validate_dataset.py --source data/raw
PYTHONPATH=src python scripts/prepare_dataset.py --source data/raw --output-dir data/processed

echo "CUDA environment ready."
nvidia-smi
