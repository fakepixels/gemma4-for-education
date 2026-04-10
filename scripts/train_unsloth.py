from __future__ import annotations

import argparse
import json
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

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=cfg["training"]["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        seed=cfg["training"]["seed"],
        report_to=cfg["training"]["report_to"],
    )

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
    }
    with (adapter_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps({"adapter_output_dir": str(adapter_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
