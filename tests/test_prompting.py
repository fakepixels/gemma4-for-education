from gemma4_classroom.prompting import build_inference_prompt, build_train_text
from gemma4_classroom.readability import estimate_level_alignment


def test_train_text_contains_required_sections():
    text = build_train_text(
        source_text="Water evaporates when heated by the Sun.",
        target_level="below",
        must_keep_facts=["The Sun heats water."],
        rewritten_text="The Sun warms water and some turns into vapor.\n\nKey Concepts Preserved\n- The Sun heats water.",
    )
    assert "Key Concepts Preserved" in text
    assert "target reading level: below" in text


def test_inference_prompt_ends_with_model_turn():
    prompt = build_inference_prompt(
        source_text="Cells are basic units of life.",
        target_level="on",
        must_keep_facts=["Cells are basic units of life."],
    )
    assert prompt.endswith("<start_of_turn>model")


def test_readability_band_estimation():
    below = estimate_level_alignment("Plants make food. Animals eat plants.", "below")
    assert below["within_target_band"] is True
