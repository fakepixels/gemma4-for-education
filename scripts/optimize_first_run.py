from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from gemma4_classroom.config import dump_yaml, load_yaml
from gemma4_classroom.tuning import apply_profile, build_profile_ladder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run an adaptive first-run training ladder.")
    parser.add_argument("--base-config", default="configs/train.first_run.yaml")
    parser.add_argument("--mode", choices=["plan", "run"], default="plan")
    parser.add_argument("--goal", choices=["benchmark", "smoke"], default="benchmark")
    parser.add_argument("--gpu-vram-gb", type=float, default=None)
    parser.add_argument("--output-dir", default="artifacts/tuning")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def detect_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    first_line = result.stdout.strip().splitlines()[0]
    try:
        return round(float(first_line) / 1024.0, 2)
    except (IndexError, ValueError):
        return None


def run_training_attempt(
    python_executable: str,
    config_path: Path,
    log_path: Path,
) -> tuple[int, str]:
    command = [python_executable, "scripts/train_unsloth.py", "--config", str(config_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    log_text = log_path.read_text(encoding="utf-8")
    return process.returncode, log_text


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if "outofmemoryerror" in lowered or "cuda out of memory" in lowered:
        return "oom"
    if "keyboardinterrupt" in lowered:
        return "interrupted"
    return "error"


def build_attempt_summary(profile_name: str, config_path: Path, log_path: Path, status: str) -> dict[str, str]:
    return {
        "profile": profile_name,
        "config_path": str(config_path),
        "log_path": str(log_path),
        "status": status,
    }


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    gpu_vram_gb = args.gpu_vram_gb if args.gpu_vram_gb is not None else detect_vram_gb()
    profiles = build_profile_ladder(gpu_vram_gb, args.goal)

    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    for index, profile in enumerate(profiles, start=1):
        candidate_cfg = apply_profile(base_cfg, profile)
        config_path = config_dir / f"{index:02d}_{profile.name}.yaml"
        dump_yaml(config_path, candidate_cfg)
        plan_rows.append(
            {
                "order": index,
                "profile": profile.name,
                "description": profile.description,
                "config_path": str(config_path),
                "max_seq_length": candidate_cfg["model"]["max_seq_length"],
                "lora_r": candidate_cfg["lora"]["r"],
                "gradient_accumulation_steps": candidate_cfg["training"]["gradient_accumulation_steps"],
                "evaluation_strategy": candidate_cfg["training"].get("evaluation_strategy", "steps"),
                "max_steps": candidate_cfg["training"].get("max_steps"),
            }
        )

    plan_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_config": args.base_config,
        "mode": args.mode,
        "goal": args.goal,
        "gpu_vram_gb": gpu_vram_gb,
        "profiles": plan_rows,
    }
    dump_yaml(output_dir / "optimization_plan.yaml", plan_payload)

    if args.mode == "plan":
        print(json.dumps(plan_payload, indent=2))
        return

    attempts: list[dict[str, str]] = []
    winning_config: str | None = None
    for row in plan_rows:
        config_path = Path(row["config_path"])
        log_path = log_dir / f"{Path(row['config_path']).stem}.log"
        return_code, log_text = run_training_attempt(args.python_executable, config_path, log_path)
        if return_code == 0:
            winning_config = str(config_path)
            attempts.append(build_attempt_summary(row["profile"], config_path, log_path, "success"))
            break

        failure_type = classify_failure(log_text)
        attempts.append(build_attempt_summary(row["profile"], config_path, log_path, failure_type))
        if failure_type != "oom":
            break

    result_payload = {
        **plan_payload,
        "attempts": attempts,
        "winning_config": winning_config,
    }
    dump_yaml(output_dir / "optimization_result.yaml", result_payload)
    print(json.dumps(result_payload, indent=2))

    if winning_config is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
