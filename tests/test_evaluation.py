from gemma4_classroom.evaluation import extract_student_facing_text, score_output


def test_extract_student_facing_text_ignores_key_concepts_section():
    output_text = (
        "Plants use sunlight to make food. They also release oxygen.\n\n"
        "Key Concepts Preserved\n"
        "- Plants use sunlight.\n"
        "- Oxygen is released."
    )
    assert extract_student_facing_text(output_text) == "Plants use sunlight to make food. They also release oxygen."


def test_score_output_uses_body_for_readability_but_keeps_teacher_note_bonus():
    output_text = (
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
