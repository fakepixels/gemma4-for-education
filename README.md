# Gemma 4 Classroom Adaptation

Local-first middle school science lesson adaptation with Gemma 4 and Unsloth.

This project fine-tunes Gemma 4 to rewrite one teacher-provided science lesson for a specific reading level while preserving scientific meaning. The demo app generates three versions of the same lesson by calling the tuned model three times: `below`, `on`, and `above`.

## Why this project

Teachers in low-bandwidth classrooms often need to serve students with very different reading levels using the same lesson. Generic prompting can rewrite text, but it often drops facts, changes emphasis, or misses the target level. This repo narrows the problem to a single trustworthy transformation:

- keep the science correct
- change the reading complexity
- keep the teacher in control

## Repo layout

```text
configs/                  Training and evaluation configs
data/raw/                 Seed source passages and reference rewrites
data/processed/           Generated train/val/test JSONL files
scripts/                  Data prep, training, evaluation, and demo entrypoints
src/gemma4_classroom/     Shared prompt, inference, readability, and scoring code
artifacts/                Checkpoints, adapters, and eval reports
```

## Quickstart

### 1. Create a Python environment

```bash
bash scripts/setup_local_env.sh
```

For CUDA training, use a Linux GPU machine and install the training extras there:

```bash
bash scripts/setup_cuda_env.sh
```

You can also use [`Makefile`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/Makefile) targets such as `make install-local`, `make install-cuda`, `make prepare-data`, and `make test`.

## 2. Build the dataset

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --source data/raw --output-dir data/processed
```

This creates:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/dataset_manifest.json`

By default, the script ingests every raw dataset JSON file in [`data/raw/`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/data/raw) except the template file, so you can grow the corpus across multiple curated files.

Validate raw files before training:

```bash
PYTHONPATH=src python scripts/validate_dataset.py --source data/raw
```

## 3. Fine-tune with Unsloth

Edit [`configs/train.yaml`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/configs/train.yaml) for your GPU and model choice, then run:

```bash
PYTHONPATH=src python scripts/train_unsloth.py --config configs/train.yaml
```

The default config is set up for a Gemma 4 E2B or E4B style workflow with LoRA/QLoRA and single-target rewriting.

For the first real CUDA run, use:

```bash
bash scripts/first_cuda_run.sh
```

That script uses [`configs/train.first_run.yaml`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/configs/train.first_run.yaml) and [`configs/eval.first_run.yaml`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/configs/eval.first_run.yaml). The first-run profile is tuned for a **single NVIDIA L4 24 GB GPU**.

## 4. Evaluate base vs tuned

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval.yaml \
  --output artifacts/evals/baseline_vs_tuned.json
```

The evaluation script scores:

- fact coverage
- reading-level alignment
- length control
- a lightweight teacher usefulness proxy

## 5. Launch the local demo

```bash
PYTHONPATH=src python scripts/run_demo.py \
  --model-id google/gemma-4-E4B-it \
  --adapter-path artifacts/adapters/latest
```

The demo accepts one science lesson and returns:

- a below-level rewrite
- an on-level rewrite
- an above-level rewrite
- a teacher note listing preserved key concepts

## Dataset format

The training task is intentionally narrow. Each example has:

- `source_id`
- `topic`
- `grade_band`
- `target_level`
- `source_text`
- `must_keep_facts`
- `rewritten_text`

The generated training JSONL also includes:

- `prompt`
- `response`
- `text`

`text` is the full supervised training string used by the SFT trainer.

## Growing the dataset

The repo now includes:

- [`data/raw/science_adaptations_seed.json`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/data/raw/science_adaptations_seed.json) for the original starter set
- [`data/raw/science_adaptations_expanded.json`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/data/raw/science_adaptations_expanded.json) for a stronger scaffold across more science topics
- [`data/raw/dataset_template.json`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/data/raw/dataset_template.json) for adding new teacher-reviewed entries

Recommended data workflow:

1. Draft new source passages in the template format.
2. Validate them with `scripts/validate_dataset.py`.
3. Rebuild `data/processed/`.
4. Keep train/val/test splits at the source level, not the rewrite level.

## Environment files

- [`/.env.example`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/.env.example) contains the expected local environment variables.
- Copy it to `.env` on your own machine before running CUDA training.
- The setup scripts automatically source `.env` if it exists.

## First benchmark pass

After the first CUDA run, create a compact benchmark summary with:

```bash
PYTHONPATH=src python scripts/summarize_eval.py \
  --input artifacts/evals/first_run_eval.json \
  --output artifacts/evals/first_run_summary.md
```

The resulting Markdown file is designed to drop directly into the Kaggle writeup draft.

There is also a ready-to-fill benchmark section draft at [`docs/kaggle_benchmark_section_draft.md`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/docs/kaggle_benchmark_section_draft.md).

## Evaluation philosophy

This project is designed for a competition submission, so the key claim must stay crisp:

> Tuned Gemma 4 preserves science facts better and controls reading level more reliably than base Gemma 4 on classroom adaptation tasks.

The benchmark is built around that claim instead of broad educational QA.

## Suggested next steps

1. Expand the source corpus to 50 to 200 middle school science passages.
2. Replace part of the seed rewrites with teacher-reviewed adaptations.
3. Run a first LoRA training job on Gemma 4 E2B or E4B.
4. Compare base vs tuned on a fixed held-out set.
5. Use the strongest examples in the Kaggle writeup and demo video.
