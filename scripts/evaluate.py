from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from gemma4_classroom.config import load_yaml
from gemma4_classroom.evaluation import score_table
from gemma4_classroom.prompting import build_inference_prompt


def resolve_unsloth_api():
    try:
        from unsloth import FastLanguageModel  # type: ignore

        return FastLanguageModel
    except ImportError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base vs tuned Gemma 4 models.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output", default="artifacts/evals/baseline_vs_tuned.json")
    return parser.parse_args()


def load_generation_model(
    model_id: str,
    adapter_path: str | None = None,
    use_4bit: bool = False,
    max_seq_length: int = 768,
):
    if adapter_path:
        unsloth_api = resolve_unsloth_api()
        if unsloth_api is None:
            raise RuntimeError("Unsloth is required to load the tuned Gemma 4 adapter for evaluation.")
        model, tokenizer = unsloth_api.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_seq_length,
            dtype="bfloat16" if torch.cuda.is_available() else None,
            load_in_4bit=False,
        )
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_id)
    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16
        if use_4bit and not adapter_path:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(get_model_device(model))
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if prompt in decoded:
        return decoded.split(prompt, 1)[1].strip()
    return decoded.strip()


def run_eval_rows(model, tokenizer, dataset, generation_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    total = len(dataset)
    for index, row in enumerate(dataset, start=1):
        print(
            f"[eval] generating {index}/{total} for target_level={row['target_level']} source_id={row['source_id']}",
            flush=True,
        )
        prompt = build_inference_prompt(
            source_text=row["source_text"],
            target_level=row["target_level"],
            must_keep_facts=row["must_keep_facts"],
        )
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=generation_cfg["max_new_tokens"],
            temperature=generation_cfg["temperature"],
            top_p=generation_cfg["top_p"],
        )
        rows.append(
            {
                "source_id": row["source_id"],
                "target_level": row["target_level"],
                "must_keep_facts": row["must_keep_facts"],
                "generated_text": generated_text,
            }
        )
    return rows


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset = load_dataset("json", data_files=cfg["data"]["test_path"], split="train")

    base_model, base_tokenizer = load_generation_model(
        cfg["model"]["base_model_id"],
        use_4bit=cfg["model"].get("use_4bit", False),
        max_seq_length=cfg["model"].get("max_seq_length", 768),
    )
    base_rows = run_eval_rows(base_model, base_tokenizer, dataset, cfg["generation"])
    release_model(base_model)

    tuned_model, tuned_tokenizer = load_generation_model(
        cfg["model"]["base_model_id"],
        adapter_path=cfg["model"]["adapter_path"],
        use_4bit=cfg["model"].get("use_4bit", False),
        max_seq_length=cfg["model"].get("max_seq_length", 768),
    )
    tuned_rows = run_eval_rows(tuned_model, tuned_tokenizer, dataset, cfg["generation"])
    release_model(tuned_model)

    report = {
        "base_model_id": cfg["model"]["base_model_id"],
        "adapter_path": cfg["model"]["adapter_path"],
        "base": score_table(base_rows),
        "tuned": score_table(tuned_rows),
        "base_examples": base_rows,
        "tuned_examples": tuned_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report["tuned"]["summary"], indent=2))


if __name__ == "__main__":
    main()
