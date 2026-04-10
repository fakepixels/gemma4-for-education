# First CUDA Run

This is the shortest path to a real first fine-tuning pass on a Linux CUDA machine.

## Recommended box

- Linux
- Single NVIDIA L4 24 GB GPU
- Python 3.10 to 3.12
- CUDA drivers working before setup begins

This repo's first-run config is now explicitly tuned for **one L4 24 GB GPU** and a **Gemma 4 4B instruction model with 4-bit LoRA fine-tuning**.

## 1. Clone and enter the repo

```bash
git clone https://github.com/fakepixels/gemma4-for-education.git
cd gemma4-for-education
```

## 2. Set environment variables

```bash
cp .env.example .env
```

Fill in `HF_TOKEN` if the selected model download requires authentication.

## 3. Bootstrap the CUDA environment

```bash
bash scripts/setup_cuda_env.sh
```

## 4. Run the first training pass

```bash
bash scripts/first_cuda_run.sh
```

This will:

- validate raw data
- rebuild processed splits
- train with `configs/train.first_run.yaml` tuned for a single L4 24 GB GPU
- evaluate with `configs/eval.first_run.yaml`
- write a Markdown summary for benchmarks

## 5. Review the outputs

- `artifacts/adapters/first-run`
- `artifacts/evals/first_run_eval.json`
- `artifacts/evals/first_run_summary.md`

## First tuning goal

Do not optimize for maximum benchmark size on the first pass. Optimize for:

- the training job running end to end
- the adapter loading cleanly
- a visible improvement over the base model on held-out examples
- one or two high-quality qualitative examples for the Kaggle story

## L4-specific assumptions baked into the config

- 4-bit loading enabled
- `bfloat16` compute
- effective train batch size of 16 via `per_device_train_batch_size=4` and `gradient_accumulation_steps=4`
- moderate LoRA rank and four training epochs for a stable first pass
- checkpoint retention capped to avoid unnecessary disk growth
