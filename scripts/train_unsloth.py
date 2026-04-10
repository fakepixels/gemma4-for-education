from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import TrainingArguments

from gemma4_classroom.config import load_yaml


def resolve_unsloth_api():
    try:
        from unsloth import FastLanguageModel  # type: ignore

        return "FastLanguageModel", FastLanguageModel
    except ImportError:
        pass

    try:
        from unsloth import FastModel  # type: ignore

        return "FastModel", FastModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is not installed. Install on a CUDA Linux machine with `pip install -e \".[cuda_train]\"`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 with Unsloth for classroom adaptation.")
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


def training_args_from_config(cfg: dict) -> TrainingArguments:
    training_cfg = cfg["training"]
    optional_args = {
        "per_device_eval_batch_size": training_cfg.get("per_device_eval_batch_size"),
        "bf16": training_cfg.get("bf16"),
        "fp16": training_cfg.get("fp16"),
        "optim": training_cfg.get("optim"),
        "max_grad_norm": training_cfg.get("max_grad_norm"),
        "save_total_limit": training_cfg.get("save_total_limit"),
        "load_best_model_at_end": training_cfg.get("load_best_model_at_end"),
        "metric_for_best_model": training_cfg.get("metric_for_best_model"),
        "greater_is_better": training_cfg.get("greater_is_better"),
    }
    filtered_optional_args = {key: value for key, value in optional_args.items() if value is not None}

    return TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        warmup_steps=training_cfg["warmup_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        weight_decay=training_cfg["weight_decay"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        seed=training_cfg["seed"],
        report_to=training_cfg["report_to"],
        **filtered_optional_args,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    _, unsloth_api = resolve_unsloth_api()

    train_ds = load_dataset("json", data_files=cfg["data"]["train_path"], split="train")
    val_ds = load_dataset("json", data_files=cfg["data"]["val_path"], split="train")

    model, tokenizer = unsloth_api.from_pretrained(
        model_name=cfg["model"]["base_model_id"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    model = unsloth_api.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
    )

    training_args = training_args_from_config(cfg)

    try:
        from trl import SFTTrainer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("TRL is required. Install `pip install -e \".[cuda_train]\"`.") from exc

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field=cfg["data"]["text_field"],
        max_seq_length=cfg["model"]["max_seq_length"],
        packing=False,
        args=training_args,
    )

    trainer.train()
    adapter_dir = Path(cfg["training"]["adapter_output_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metadata = {
        "base_model_id": cfg["model"]["base_model_id"],
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "task": cfg["metadata"]["task_name"],
        "gpu_target": cfg["metadata"].get("gpu_target"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with (adapter_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps({"adapter_output_dir": str(adapter_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
