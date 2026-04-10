from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TuningProfile:
    name: str
    description: str
    overrides: dict[str, Any]


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def detect_vram_tier(gpu_vram_gb: float | None) -> str:
    if gpu_vram_gb is None:
        return "unknown"
    if gpu_vram_gb <= 24:
        return "24gb"
    if gpu_vram_gb <= 32:
        return "32gb"
    return "40gb_plus"


def _shared_safe_overrides() -> dict[str, Any]:
    return {
        "training": {
            "optim": "adamw_torch",
            "fp16": False,
            "bf16": True,
        }
    }


def build_profile_ladder(gpu_vram_gb: float | None, goal: str) -> list[TuningProfile]:
    tier = detect_vram_tier(gpu_vram_gb)
    goal = goal.lower()

    if goal not in {"benchmark", "smoke"}:
        raise ValueError(f"Unsupported goal: {goal}")

    profiles: list[TuningProfile] = []

    if tier == "40gb_plus":
        profiles.append(
            TuningProfile(
                name="benchmark-balanced",
                description="Largest first pass for roomy GPUs.",
                overrides={
                    **_shared_safe_overrides(),
                    "model": {"max_seq_length": 1024},
                    "lora": {"r": 16, "alpha": 32, "dropout": 0.0},
                    "training": {
                        "per_device_train_batch_size": 1,
                        "per_device_eval_batch_size": 1,
                        "gradient_accumulation_steps": 8,
                        "evaluation_strategy": "steps",
                        "save_strategy": "steps",
                        "load_best_model_at_end": True,
                    },
                    "metadata": {"gpu_target": "auto-40gb-plus"},
                },
            )
        )

    if tier in {"32gb", "40gb_plus", "unknown"}:
        profiles.append(
            TuningProfile(
                name="benchmark-safe-32gb",
                description="Safer benchmark profile for 32 GB consumer GPUs.",
                overrides={
                    **_shared_safe_overrides(),
                    "model": {"max_seq_length": 768},
                    "lora": {"r": 8, "alpha": 16, "dropout": 0.0},
                    "training": {
                        "per_device_train_batch_size": 1,
                        "per_device_eval_batch_size": 1,
                        "gradient_accumulation_steps": 16,
                        "evaluation_strategy": "no",
                        "save_strategy": "steps",
                        "save_steps": 20,
                        "load_best_model_at_end": False,
                        "num_train_epochs": 2,
                    },
                    "metadata": {"gpu_target": "auto-32gb-safe"},
                },
            )
        )

    profiles.append(
        TuningProfile(
            name="smoke-safe",
            description="Short sequence, low-rank LoRA, no eval; designed to fit and prove the stack works.",
            overrides={
                **_shared_safe_overrides(),
                "model": {"max_seq_length": 512},
                "lora": {"r": 8, "alpha": 16, "dropout": 0.0},
                "training": {
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                    "evaluation_strategy": "no",
                    "save_strategy": "no",
                    "load_best_model_at_end": False,
                    "num_train_epochs": 1,
                    "max_steps": 3 if goal == "smoke" else 12,
                },
                "metadata": {"gpu_target": "auto-smoke-safe"},
            },
        )
    )

    profiles.append(
        TuningProfile(
            name="smoke-minimal",
            description="Emergency fallback when the GPU is still right on the memory edge.",
            overrides={
                **_shared_safe_overrides(),
                "model": {"max_seq_length": 384},
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
                "training": {
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                    "evaluation_strategy": "no",
                    "save_strategy": "no",
                    "load_best_model_at_end": False,
                    "num_train_epochs": 1,
                    "max_steps": 2 if goal == "smoke" else 8,
                },
                "metadata": {"gpu_target": "auto-smoke-minimal"},
            },
        )
    )

    return profiles


def apply_profile(base_cfg: dict[str, Any], profile: TuningProfile) -> dict[str, Any]:
    merged = deep_merge(base_cfg, profile.overrides)
    metadata = dict(merged.get("metadata", {}))
    metadata["selected_profile"] = profile.name
    metadata["selected_profile_description"] = profile.description
    merged["metadata"] = metadata
    return merged
