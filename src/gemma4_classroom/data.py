from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gemma4_classroom.prompting import build_instruction, build_train_text


LEVELS = ("below", "on", "above")


def load_seed_examples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_seed_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for record in records:
        for level in LEVELS:
            response = record["rewrites"][level]
            expanded.append(
                {
                    "source_id": record["source_id"],
                    "topic": record["topic"],
                    "grade_band": record["grade_band"],
                    "target_level": level,
                    "source_text": record["source_text"],
                    "must_keep_facts": record["must_keep_facts"],
                    "rewritten_text": response,
                    "prompt": build_instruction(record["source_text"], level, record["must_keep_facts"]),
                    "response": response,
                    "text": build_train_text(
                        record["source_text"],
                        level,
                        record["must_keep_facts"],
                        response,
                    ),
                }
            )
    return expanded


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
