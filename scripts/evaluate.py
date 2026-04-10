from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_classroom.config import load_yaml
from gemma4_classroom.evaluation import score_table
from gemma4_classroom.prompting import build_inference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base vs tuned Gemma 4 models.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--output", default="artifacts/evals/baseline_vs_tuned.json")
    return parser.parse_args()


def load_generation_model(model_id: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
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
    for row in dataset:
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


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset = load_dataset("json", data_files=cfg["data"]["test_path"], split="train")

    base_model, base_tokenizer = load_generation_model(cfg["model"]["base_model_id"])
    tuned_model, tuned_tokenizer = load_generation_model(
        cfg["model"]["base_model_id"],
        adapter_path=cfg["model"]["adapter_path"],
    )

    base_rows = run_eval_rows(base_model, base_tokenizer, dataset, cfg["generation"])
    tuned_rows = run_eval_rows(tuned_model, tuned_tokenizer, dataset, cfg["generation"])

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
