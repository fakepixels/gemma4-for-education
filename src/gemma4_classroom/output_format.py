from __future__ import annotations

import re
from dataclasses import dataclass

ADAPTED_LESSON_HEADING = "Adapted Lesson"
KEY_CONCEPTS_HEADING = "Key Concepts Preserved"
MODEL_TURN_RE = re.compile(r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)", re.S)


@dataclass
class ParsedOutput:
    adapted_lesson: str
    key_concepts: list[str]


def _split_key_concepts_tail(tail: str) -> list[str]:
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    concepts: list[str] = []
    for line in lines:
        if line.startswith("-"):
            concept = line.lstrip("-").strip()
            if concept:
                concepts.append(concept)
            continue
        for part in line.split(";"):
            concept = part.strip(" -*:")
            if concept:
                concepts.append(concept)
    return concepts


def parse_model_output(text: str) -> ParsedOutput:
    cleaned = extract_model_completion(text).strip()
    if not cleaned:
        return ParsedOutput(adapted_lesson="", key_concepts=[])

    if ADAPTED_LESSON_HEADING in cleaned and KEY_CONCEPTS_HEADING in cleaned:
        _, lesson_tail = cleaned.split(ADAPTED_LESSON_HEADING, 1)
        lesson_body, key_tail = lesson_tail.split(KEY_CONCEPTS_HEADING, 1)
        return ParsedOutput(
            adapted_lesson=lesson_body.strip(" :\n"),
            key_concepts=_split_key_concepts_tail(key_tail),
        )

    if KEY_CONCEPTS_HEADING in cleaned:
        lesson_body, key_tail = cleaned.split(KEY_CONCEPTS_HEADING, 1)
        return ParsedOutput(
            adapted_lesson=lesson_body.strip(),
            key_concepts=_split_key_concepts_tail(key_tail),
        )

    return ParsedOutput(adapted_lesson=cleaned, key_concepts=[])


def extract_model_completion(text: str) -> str:
    cleaned = text.strip()
    turns = MODEL_TURN_RE.findall(cleaned)
    if turns:
        return turns[-1].strip()
    return cleaned


def format_model_output(adapted_lesson: str, key_concepts: list[str]) -> str:
    concept_lines = "\n".join(f"- {concept}" for concept in key_concepts if concept.strip())
    if concept_lines:
        return (
            f"{ADAPTED_LESSON_HEADING}\n"
            f"{adapted_lesson.strip()}\n\n"
            f"{KEY_CONCEPTS_HEADING}\n"
            f"{concept_lines}"
        ).strip()
    return f"{ADAPTED_LESSON_HEADING}\n{adapted_lesson.strip()}".strip()


def normalize_model_output(text: str) -> str:
    parsed = parse_model_output(text)
    return format_model_output(parsed.adapted_lesson, parsed.key_concepts)


def extract_student_facing_text(text: str) -> str:
    return parse_model_output(text).adapted_lesson


def extract_key_concepts(text: str) -> list[str]:
    return parse_model_output(text).key_concepts
