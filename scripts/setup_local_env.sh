#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demo,dev]"

PYTHONPATH=src python scripts/validate_dataset.py --source data/raw
PYTHONPATH=src python scripts/prepare_dataset.py --source data/raw --output-dir data/processed

echo "Local environment ready."
