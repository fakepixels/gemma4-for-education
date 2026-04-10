from gemma4_classroom.evaluation import score_output
from gemma4_classroom.output_format import (
    extract_model_completion,
    extract_student_facing_text,
    normalize_model_output,
    parse_model_output,
)


def test_extract_student_facing_text_ignores_key_concepts_section():
    output_text = (
        "Plants use sunlight to make food. They also release oxygen.\n\n"
        "Key Concepts Preserved\n"
        "- Plants use sunlight.\n"
        "- Oxygen is released."
    )
    assert extract_student_facing_text(output_text) == "Plants use sunlight to make food. They also release oxygen."


def test_normalize_model_output_adds_structured_headings():
    normalized = normalize_model_output(
        "Plants use sunlight to make food.\n\nKey Concepts Preserved\n- Plants use sunlight."
    )
    parsed = parse_model_output(normalized)
    assert normalized.startswith("Adapted Lesson")
    assert parsed.adapted_lesson == "Plants use sunlight to make food."
    assert parsed.key_concepts == ["Plants use sunlight."]


def test_extract_model_completion_from_chat_transcript():
    transcript = (
        "<start_of_turn>system\nRules\n<end_of_turn>\n"
        "<start_of_turn>user\nPrompt\n<end_of_turn>\n"
        "<start_of_turn>model\nAdapted Lesson\nShort text.\n\nKey Concepts Preserved\n- Fact.\n<end_of_turn>"
    )
    assert extract_model_completion(transcript).startswith("Adapted Lesson")


def test_score_output_uses_body_for_readability_but_keeps_teacher_note_bonus():
    output_text = (
        "Adapted Lesson\n"
        "Plants make food. They use sunlight.\n\n"
        "Key Concepts Preserved\n"
        "- Plants use sunlight to make food.\n"
        "- Oxygen is released."
    )
    score = score_output(
        source_id="photosynthesis_demo",
        target_level="below",
        must_keep_facts=[
            "Plants use sunlight to make food.",
            "Photosynthesis releases oxygen.",
        ],
        output_text=output_text,
    )
    assert score.within_target_band is True
    assert score.teacher_usefulness > 0.0
    assert score.level_control_score > 0.6
