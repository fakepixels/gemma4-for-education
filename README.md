# Gemma 4 Classroom Adaptation

Local-first middle school science lesson adaptation with Gemma 4 and Unsloth.

This repo fine-tunes Gemma 4 to do one job well: take a middle school science lesson, rewrite it for a target reading level, and avoid mangling the science in the process.

The model returns a structured response with two sections:

- `Adapted Lesson` for the student-facing rewrite
- `Key Concepts Preserved` for a short teacher-facing fact check

The demo app calls the tuned model three times to generate `below`, `on`, and `above` versions of the same lesson. One lesson in, three classroom-ready drafts out.

## Benchmark snapshot

The current held-out benchmark is the part where this stops being a nice idea and starts being evidence.

| Metric | Base Gemma 4 | Tuned Gemma 4 | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.417 | 1.000 | +0.583 |
| Avg teacher usefulness | 0.408 | 0.967 | +0.559 |
| Within target band rate | 0.417 | 0.833 | +0.416 |

This benchmark uses the project's science-aware classroom rubric on a fixed held-out set. The tuned model is materially better at staying factual, following the output contract, and landing in the requested reading band.

Per-level, the story is even clearer:

| Level | Base usefulness | Tuned usefulness | What changed |
|---|---:|---:|---|
| `below` | 0.725 | 0.900 | Better fact preservation, but still the messiest level |
| `on` | 0.000 | 1.000 | Biggest win: tuned becomes reliably classroom-usable |
| `above` | 0.500 | 1.000 | Strong gain in completeness and structure-following |

For the full writeup and qualitative examples, see:

- [`docs/kaggle_benchmark_section_polished.md`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/docs/kaggle_benchmark_section_polished.md)
- [`docs/qualitative_examples.md`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/docs/qualitative_examples.md)
- [`docs/benchmark_appendix.md`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/docs/benchmark_appendix.md)

## Why this project

Teachers in low-bandwidth classrooms already have enough problems. "Rewrite this lesson for three reading levels without breaking the science" should not need to be one of them.

Generic prompting can absolutely rewrite text. It can also quietly drop facts, drift off-level, and become very confident about all of it. This repo narrows the problem to one trustworthy transformation:

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

If you like `make`, there is a [`Makefile`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/Makefile) with targets like `make install-local`, `make install-cuda`, `make prepare-data`, and `make test`.

## 2. Build the dataset

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --source data/raw --output-dir data/processed
```

This creates:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/dataset_manifest.json`

By default, the script ingests every raw dataset JSON file in [`data/raw/`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/data/raw) except the template file, so you can keep growing the corpus without turning the pipeline into archaeology.

Validate raw files before training:

```bash
PYTHONPATH=src python scripts/validate_dataset.py --source data/raw
```

## 3. Fine-tune with Unsloth

Edit [`configs/train.yaml`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/configs/train.yaml) for your GPU and model choice, then run:

```bash
PYTHONPATH=src python scripts/train_unsloth.py --config configs/train.yaml
```

The default config is set up for a Gemma 4 E2B/E4B-style workflow with LoRA/QLoRA and single-target rewriting.

For the first real CUDA run, use:

```bash
bash scripts/first_cuda_run.sh
```

That script uses [`configs/train.first_run.yaml`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/configs/train.first_run.yaml) and [`configs/eval.first_run.yaml`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/configs/eval.first_run.yaml). The first-run profile is tuned for a **single NVIDIA L4 24 GB GPU**.

If you want the repo to be emotionally mature about OOMs and automatically step down through safer configs, use:

```bash
AUTO_OPTIMIZE_FIRST_RUN=1 OPTIMIZE_GOAL=benchmark bash scripts/first_cuda_run.sh
```

That adaptive path writes its candidate configs and logs under [`artifacts/tuning/`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/artifacts/tuning). You can also inspect the ladder without training:

```bash
PYTHONPATH=src python scripts/optimize_first_run.py --mode plan --goal benchmark
```

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

- a below-level adapted lesson
- an on-level adapted lesson
- an above-level adapted lesson
- a teacher note listing preserved key concepts

The product idea is intentionally simple. No agent maze. No twelve-tab orchestration diagram. Just a teacher, one lesson, and three better starting points.

## Dataset format

The training task is intentionally narrow. Each example has:

- `source_id`
- `topic`
- `grade_band`
- `target_level`
- `source_text`
- `must_keep_facts`
- `rewritten_text`

`rewritten_text` is normalized into a consistent two-section model output:

```text
Adapted Lesson
<student-facing rewrite>

Key Concepts Preserved
- <short fact 1>
- <short fact 2>
- <short fact 3>
```

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
4. Keep train/val/test splits at the source level, not the rewrite level. Leakage is boring and expensive.

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

The resulting Markdown file is designed to drop directly into the Kaggle writeup draft with minimal cleanup.

There is also a ready-to-fill benchmark section draft at [`docs/kaggle_benchmark_section_draft.md`](/Users/tinahe/Desktop/analysis/unsloth/gemma-4/docs/kaggle_benchmark_section_draft.md).

## Evaluation philosophy

This project is designed for a competition submission, so the key claim has to stay crisp:

> Tuned Gemma 4 preserves science facts better and controls reading level more reliably than base Gemma 4 on classroom adaptation tasks.

The benchmark is built around that claim instead of trying to prove the model can solve education in one repo.

## Suggested next steps

1. Expand the source corpus to 50 to 200 middle school science passages.
2. Replace part of the seed rewrites with teacher-reviewed adaptations.
3. Run a first LoRA training job on Gemma 4 E2B or E4B.
4. Compare base vs tuned on a fixed held-out set.
5. Use the strongest examples in the Kaggle writeup and demo video instead of asking the benchmark to be charismatic on its own.
