from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed benchmark appendix from eval JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fmt_delta(base: float, tuned: float) -> str:
    delta = round(tuned - base, 3)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def level_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = len(rows)
    if not count:
        return {
            "fact": 0.0,
            "useful": 0.0,
            "band": 0.0,
            "control": 0.0,
        }
    return {
        "fact": round(sum(row["fact_coverage"] for row in rows) / count, 3),
        "useful": round(sum(row["teacher_usefulness"] for row in rows) / count, 3),
        "band": round(sum(row["within_target_band"] for row in rows) / count, 3),
        "control": round(sum(row["level_control_score"] for row in rows) / count, 3),
    }


def build_pair_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for base_row, tuned_row, base_example, tuned_example in zip(
        report["base"]["rows"],
        report["tuned"]["rows"],
        report["base_examples"],
        report["tuned_examples"],
    ):
        rows.append(
            {
                "source_id": base_row["source_id"],
                "target_level": base_row["target_level"],
                "base": base_row,
                "tuned": tuned_row,
                "base_text": base_example.get("generated_text", ""),
                "tuned_text": tuned_example.get("generated_text", ""),
                "teacher_usefulness_delta": round(
                    tuned_row["teacher_usefulness"] - base_row["teacher_usefulness"], 3
                ),
                "fact_coverage_delta": round(
                    tuned_row["fact_coverage"] - base_row["fact_coverage"], 3
                ),
                "band_delta": int(tuned_row["within_target_band"]) - int(base_row["within_target_band"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pair_rows = build_pair_rows(report)

    lines = [
        "# Benchmark Appendix",
        "",
        f"Base model: `{report['base_model_id']}`",
        f"Adapter path: `{report['adapter_path']}`",
        "",
        "This appendix expands the top-line benchmark with per-level results and per-example deltas.",
        "",
    ]

    base_summary = report["base"]["summary"]
    tuned_summary = report["tuned"]["summary"]
    lines.extend(
        [
            "## Overall Summary",
            "",
            "| Metric | Base | Tuned | Delta |",
            "|---|---:|---:|---:|",
            f"| Avg fact coverage | {base_summary['avg_fact_coverage']:.3f} | {tuned_summary['avg_fact_coverage']:.3f} | {fmt_delta(base_summary['avg_fact_coverage'], tuned_summary['avg_fact_coverage'])} |",
            f"| Avg teacher usefulness | {base_summary['avg_teacher_usefulness']:.3f} | {tuned_summary['avg_teacher_usefulness']:.3f} | {fmt_delta(base_summary['avg_teacher_usefulness'], tuned_summary['avg_teacher_usefulness'])} |",
            f"| Within target band rate | {base_summary['within_target_band_rate']:.3f} | {tuned_summary['within_target_band_rate']:.3f} | {fmt_delta(base_summary['within_target_band_rate'], tuned_summary['within_target_band_rate'])} |",
            "",
        ]
    )

    lines.extend(["## Per-Level Breakdown", ""])
    level_order = ["below", "on", "above"]
    for level in level_order:
        base_rows = [row for row in report["base"]["rows"] if row["target_level"] == level]
        tuned_rows = [row for row in report["tuned"]["rows"] if row["target_level"] == level]
        base_level = level_summary(base_rows)
        tuned_level = level_summary(tuned_rows)
        lines.extend(
            [
                f"### `{level}`",
                "",
                "| Metric | Base | Tuned | Delta |",
                "|---|---:|---:|---:|",
                f"| Avg fact coverage | {base_level['fact']:.3f} | {tuned_level['fact']:.3f} | {fmt_delta(base_level['fact'], tuned_level['fact'])} |",
                f"| Avg teacher usefulness | {base_level['useful']:.3f} | {tuned_level['useful']:.3f} | {fmt_delta(base_level['useful'], tuned_level['useful'])} |",
                f"| Within target band rate | {base_level['band']:.3f} | {tuned_level['band']:.3f} | {fmt_delta(base_level['band'], tuned_level['band'])} |",
                f"| Avg level-control score | {base_level['control']:.3f} | {tuned_level['control']:.3f} | {fmt_delta(base_level['control'], tuned_level['control'])} |",
                "",
            ]
        )

    blank_base_outputs = sum(1 for row in pair_rows if not row["base_text"].strip())
    blank_tuned_outputs = sum(1 for row in pair_rows if not row["tuned_text"].strip())
    lines.extend(
        [
            "## Reliability Notes",
            "",
            f"- Blank base outputs on the held-out set: `{blank_base_outputs}` / `{len(pair_rows)}`",
            f"- Blank tuned outputs on the held-out set: `{blank_tuned_outputs}` / `{len(pair_rows)}`",
            "- The biggest gains are reliability and structure-following at `on` and `above`.",
            "- The main remaining weakness is `below`, where tuned preserves facts well but can stay too close to the original wording.",
            "",
        ]
    )

    strongest_wins = sorted(pair_rows, key=lambda row: row["teacher_usefulness_delta"], reverse=True)[:5]
    regressions = [row for row in sorted(pair_rows, key=lambda row: row["teacher_usefulness_delta"]) if row["teacher_usefulness_delta"] < 0][:5]

    lines.extend(["## Largest Tuned Wins", ""])
    lines.extend(
        [
            "| Example | Fact Coverage | Teacher Usefulness | Within Band |",
            "|---|---|---|---|",
        ]
    )
    for row in strongest_wins:
        lines.append(
            f"| `{row['source_id']}` / `{row['target_level']}` | {row['base']['fact_coverage']:.3f} -> {row['tuned']['fact_coverage']:.3f} | {row['base']['teacher_usefulness']:.3f} -> {row['tuned']['teacher_usefulness']:.3f} | `{row['base']['within_target_band']}` -> `{row['tuned']['within_target_band']}` |"
        )
    lines.append("")

    if regressions:
        lines.extend(["## Regressions To Fix", ""])
        lines.extend(
            [
                "| Example | Fact Coverage | Teacher Usefulness | Within Band |",
                "|---|---|---|---|",
            ]
        )
        for row in regressions:
            lines.append(
                f"| `{row['source_id']}` / `{row['target_level']}` | {row['base']['fact_coverage']:.3f} -> {row['tuned']['fact_coverage']:.3f} | {row['base']['teacher_usefulness']:.3f} -> {row['tuned']['teacher_usefulness']:.3f} | `{row['base']['within_target_band']}` -> `{row['tuned']['within_target_band']}` |"
            )
        lines.append("")

    lines.extend(["## Per-Example Table", ""])
    lines.extend(
        [
            "| Source | Level | Base Fact | Tuned Fact | Base Useful | Tuned Useful | Base Band | Tuned Band | Delta |",
            "|---|---|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in sorted(pair_rows, key=lambda row: (row["source_id"], level_order.index(row["target_level"]))):
        lines.append(
            f"| `{row['source_id']}` | `{row['target_level']}` | {row['base']['fact_coverage']:.3f} | {row['tuned']['fact_coverage']:.3f} | {row['base']['teacher_usefulness']:.3f} | {row['tuned']['teacher_usefulness']:.3f} | `{row['base']['within_target_band']}` | `{row['tuned']['within_target_band']}` | {row['teacher_usefulness_delta']:+.3f} |"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
