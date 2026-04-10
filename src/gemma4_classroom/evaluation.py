from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gemma4_classroom.output_format import extract_key_concepts, extract_student_facing_text
from gemma4_classroom.readability import estimate_level_alignment


@dataclass
class ExampleScore:
    source_id: str
    target_level: str
    fact_coverage: float
    readability_grade: float
    avg_words_per_sentence: float
    level_control_score: float
    science_term_count: float
    complex_non_domain_ratio: float
    within_target_band: bool
    teacher_usefulness: float
def fact_coverage_score(must_keep_facts: list[str], output_text: str) -> float:
    lowered = output_text.lower()
    hits = 0
    for fact in must_keep_facts:
        fact_terms = [token.lower() for token in fact.replace(".", "").split() if len(token) > 3]
        if fact_terms and sum(token in lowered for token in fact_terms) >= max(1, len(fact_terms) // 2):
            hits += 1
    if not must_keep_facts:
        return 0.0
    return round(hits / len(must_keep_facts), 3)


def teacher_usefulness_proxy(fact_coverage: float, within_target_band: bool, output_text: str) -> float:
    key_concepts_bonus = 0.1 if extract_key_concepts(output_text) else 0.0
    target_bonus = 0.2 if within_target_band else 0.0
    length_penalty = 0.1 if len(output_text.split()) < 60 else 0.0
    score = max(0.0, min(1.0, fact_coverage * 0.7 + target_bonus + key_concepts_bonus - length_penalty))
    return round(score, 3)


def score_output(source_id: str, target_level: str, must_keep_facts: list[str], output_text: str) -> ExampleScore:
    student_facing_text = extract_student_facing_text(output_text)
    level_metrics = estimate_level_alignment(student_facing_text, target_level)
    fact_coverage = fact_coverage_score(must_keep_facts, output_text)
    usefulness = teacher_usefulness_proxy(
        fact_coverage,
        bool(level_metrics["within_target_band"]),
        output_text,
    )
    return ExampleScore(
        source_id=source_id,
        target_level=target_level,
        fact_coverage=fact_coverage,
        readability_grade=float(level_metrics["readability_grade"]),
        avg_words_per_sentence=float(level_metrics["avg_words_per_sentence"]),
        level_control_score=float(level_metrics["level_control_score"]),
        science_term_count=float(level_metrics["science_term_count"]),
        complex_non_domain_ratio=float(level_metrics["complex_non_domain_ratio"]),
        within_target_band=bool(level_metrics["within_target_band"]),
        teacher_usefulness=usefulness,
    )


def aggregate_scores(rows: list[ExampleScore]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "avg_fact_coverage": 0.0,
            "avg_teacher_usefulness": 0.0,
            "within_target_band_rate": 0.0,
        }
    count = len(rows)
    return {
        "count": count,
        "avg_fact_coverage": round(sum(row.fact_coverage for row in rows) / count, 3),
        "avg_teacher_usefulness": round(sum(row.teacher_usefulness for row in rows) / count, 3),
        "within_target_band_rate": round(sum(row.within_target_band for row in rows) / count, 3),
    }


def score_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        score_output(
            source_id=row["source_id"],
            target_level=row["target_level"],
            must_keep_facts=row["must_keep_facts"],
            output_text=row["generated_text"],
        )
        for row in rows
    ]
    return {
        "summary": aggregate_scores(scored),
        "rows": [row.__dict__ for row in scored],
    }
