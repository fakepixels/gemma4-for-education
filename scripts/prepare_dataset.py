from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gemma4_classroom.data import expand_seed_examples, load_seed_examples, write_jsonl

DEFAULT_VAL_SOURCE_IDS = [
    "food_webs_001",
    "forces_motion_001",
    "genetics_001",
]

DEFAULT_TEST_SOURCE_IDS = [
    "atoms_molecules_001",
    "cells_001",
    "ecosystems_001",
    "electric_circuits_001",
]


def split_by_source(rows: list[dict], val_source_ids: set[str], test_source_ids: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    train, val, test = [], [], []
    for row in rows:
        source_id = row["source_id"]
        if source_id in test_source_ids:
            test.append(row)
        elif source_id in val_source_ids:
            val.append(row)
        else:
            train.append(row)
    return train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test files for Gemma 4 classroom adaptation.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Raw dataset JSON file or directory. Repeat to pass multiple paths.",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-source-id", action="append", default=None)
    parser.add_argument("--test-source-id", action="append", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.17)
    parser.add_argument("--test-fraction", type=float, default=0.17)
    return parser.parse_args()


def resolve_source_files(source_args: list[str] | None) -> list[Path]:
    inputs = source_args or ["data/raw"]
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Source path not found: {raw_path}")
    data_files = [path for path in files if path.name != "dataset_template.json"]
    if not data_files:
        raise ValueError("No raw dataset JSON files found.")
    return data_files


def load_all_seed_examples(source_files: list[Path]) -> tuple[list[dict], list[str]]:
    combined: list[dict] = []
    source_names: list[str] = []
    for source_file in source_files:
        records = load_seed_examples(source_file)
        combined.extend(records)
        source_names.append(str(source_file))
    return combined, source_names


def choose_split_source_ids(source_ids: list[str], requested: list[str] | None, fraction: float, fallback_offset: int) -> list[str]:
    if requested:
        return requested
    sorted_ids = sorted(source_ids)
    target_count = max(1, math.ceil(len(sorted_ids) * fraction))
    rotated = sorted_ids[fallback_offset:] + sorted_ids[:fallback_offset]
    return rotated[:target_count]


def default_split_source_ids(available_ids: list[str], preferred_ids: list[str]) -> list[str]:
    available = set(available_ids)
    selected = [source_id for source_id in preferred_ids if source_id in available]
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = resolve_source_files(args.source)
    seeds, source_names = load_all_seed_examples(source_files)
    expanded = expand_seed_examples(seeds)
    source_ids = sorted({row["source_id"] for row in expanded})
    requested_test_ids = args.test_source_id or default_split_source_ids(source_ids, DEFAULT_TEST_SOURCE_IDS)
    test_source_ids = choose_split_source_ids(source_ids, requested_test_ids, args.test_fraction, fallback_offset=0)
    remaining_ids = [source_id for source_id in source_ids if source_id not in set(test_source_ids)]
    requested_val_ids = args.val_source_id or default_split_source_ids(remaining_ids, DEFAULT_VAL_SOURCE_IDS)
    val_source_ids = choose_split_source_ids(remaining_ids, requested_val_ids, args.val_fraction, fallback_offset=1)
    train, val, test = split_by_source(
        expanded,
        val_source_ids=set(val_source_ids),
        test_source_ids=set(test_source_ids),
    )

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)

    manifest = {
        "source_files": source_names,
        "num_seed_sources": len(seeds),
        "num_examples": len(expanded),
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "val_source_ids": val_source_ids,
        "test_source_ids": test_source_ids,
        "task": "single-target reading-level rewriting with fact preservation",
        "levels": ["below", "on", "above"],
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
