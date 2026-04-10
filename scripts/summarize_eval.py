from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize base-vs-tuned eval output into Markdown.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fmt_delta(base: float, tuned: float) -> str:
    delta = round(tuned - base, 3)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def main() -> None:
    args = parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    base = report["base"]["summary"]
    tuned = report["tuned"]["summary"]

    lines = [
        "# First Benchmark Summary",
        "",
        f"Base model: `{report['base_model_id']}`",
        f"Adapter path: `{report['adapter_path']}`",
        "",
        "| Metric | Base | Tuned | Delta |",
        "|---|---:|---:|---:|",
        f"| Avg fact coverage | {base['avg_fact_coverage']:.3f} | {tuned['avg_fact_coverage']:.3f} | {fmt_delta(base['avg_fact_coverage'], tuned['avg_fact_coverage'])} |",
        f"| Avg teacher usefulness | {base['avg_teacher_usefulness']:.3f} | {tuned['avg_teacher_usefulness']:.3f} | {fmt_delta(base['avg_teacher_usefulness'], tuned['avg_teacher_usefulness'])} |",
        f"| Within target band rate | {base['within_target_band_rate']:.3f} | {tuned['within_target_band_rate']:.3f} | {fmt_delta(base['within_target_band_rate'], tuned['within_target_band_rate'])} |",
        "",
        "## Interpretation",
        "",
        "- Use this table in the Kaggle writeup as the first proof point.",
        "- Pair it with one qualitative source passage where the tuned model preserves science facts more faithfully.",
        "- If tuned results are weaker than expected, improve raw data quality before expanding model size.",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
