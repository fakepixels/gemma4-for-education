from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_classroom.data import LEVELS, load_seed_examples


REQUIRED_FIELDS = {
    "source_id",
    "topic",
    "grade_band",
    "source_text",
    "must_keep_facts",
    "rewrites",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw science adaptation dataset files.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Raw dataset JSON file or directory. Repeat to pass multiple paths.",
    )
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
    return [path for path in files if path.name != "dataset_template.json"]


def validate_record(record: dict, seen_ids: set[str]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        errors.append(f"{record.get('source_id', '<missing id>')}: missing fields {sorted(missing)}")
        return errors
    source_id = record["source_id"]
    if source_id in seen_ids:
        errors.append(f"{source_id}: duplicate source_id")
    seen_ids.add(source_id)
    if not isinstance(record["must_keep_facts"], list) or len(record["must_keep_facts"]) < 3:
        errors.append(f"{source_id}: must_keep_facts must contain at least 3 items")
    rewrites = record["rewrites"]
    for level in LEVELS:
        if level not in rewrites or not rewrites[level].strip():
            errors.append(f"{source_id}: missing rewrite for level `{level}`")
    return errors


def main() -> None:
    args = parse_args()
    source_files = resolve_source_files(args.source)
    seen_ids: set[str] = set()
    errors: list[str] = []
    counts: dict[str, int] = {}

    for source_file in source_files:
        records = load_seed_examples(source_file)
        counts[str(source_file)] = len(records)
        for record in records:
            errors.extend(validate_record(record, seen_ids))

    report = {
        "source_files": counts,
        "total_sources": sum(counts.values()),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
