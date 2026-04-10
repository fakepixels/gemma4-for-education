from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_classroom.data import LEVELS, load_seed_examples
from gemma4_classroom.evaluation import extract_student_facing_text
from gemma4_classroom.readability import estimate_level_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw readability alignment for classroom adaptation examples.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Raw dataset JSON file or directory. Repeat to pass multiple paths.",
    )
    parser.add_argument(
        "--fails-only",
        action="store_true",
        help="Only print examples that miss their target band.",
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


def main() -> None:
    args = parse_args()
    source_files = resolve_source_files(args.source)

    summary = {
        level: {"count": 0, "within": 0}
        for level in LEVELS
    }
    failing_examples: list[dict[str, object]] = []

    for source_file in source_files:
        for record in load_seed_examples(source_file):
            for level in LEVELS:
                metrics = estimate_level_alignment(
                    extract_student_facing_text(record["rewrites"][level]),
                    level,
                )
                summary[level]["count"] += 1
                summary[level]["within"] += int(metrics["within_target_band"])
                if not metrics["within_target_band"]:
                    failing_examples.append(
                        {
                            "source_file": str(source_file),
                            "source_id": record["source_id"],
                            "target_level": level,
                            "readability_grade": metrics["readability_grade"],
                            "avg_words_per_sentence": metrics["avg_words_per_sentence"],
                            "avg_word_length": metrics["avg_word_length"],
                            "lexical_diversity": metrics["lexical_diversity"],
                        }
                    )

    report = {
        "source_files": [str(path) for path in source_files],
        "summary": {
            level: {
                **counts,
                "rate": round(counts["within"] / counts["count"], 3) if counts["count"] else 0.0,
            }
            for level, counts in summary.items()
        },
        "failing_examples": failing_examples if args.fails_only else failing_examples,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
