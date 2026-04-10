from __future__ import annotations

from textwrap import dedent

from gemma4_classroom.output_format import ADAPTED_LESSON_HEADING, KEY_CONCEPTS_HEADING

LEVEL_NAMES = {
    "below": "Level 1",
    "on": "Level 2",
    "above": "Level 3",
}

LEVEL_DEFINITIONS = {
    "below": "Level 1 is the most supported version: shorter sentences, simpler wording, and extra clarity.",
    "on": "Level 2 is the grade-level version: clear middle school language with normal science vocabulary.",
    "above": "Level 3 is the most challenging version: stronger vocabulary, more nuance, and more connected reasoning.",
}

SYSTEM_PROMPT = dedent(
    """
    You are an expert middle school science curriculum adapter.
    Rewrite the source lesson for the requested reading level.

    Rules:
    1. Preserve the science facts from the source.
    2. Do not add new scientific claims.
    3. Change vocabulary, sentence structure, and scaffolding to fit the target level.
    4. Write the student-facing rewrite under the heading "Adapted Lesson".
    5. Put the teacher-facing summary under the heading "Key Concepts Preserved".
    6. Keep the adapted lesson in paragraph form unless the prompt asks otherwise.
    7. Keep the key concepts short and factual.
    """
).strip()


def level_guidance(target_level: str) -> str:
    if target_level == "below":
        return dedent(
            """
            Reading-level guidance:
            - Use short, direct sentences.
            - Prefer common everyday words when possible.
            - Explain hard science words in simple language.
            - Keep the structure highly clear and supportive.
            """
        ).strip()
    if target_level == "on":
        return dedent(
            """
            Reading-level guidance:
            - Use medium-length sentences.
            - Keep the explanation clear but still academic.
            - Use science vocabulary when needed, but avoid unnecessary complexity.
            - Add enough context for a typical middle school reader to follow the idea.
            """
        ).strip()
    return dedent(
        """
        Reading-level guidance:
        - Use more precise academic vocabulary.
        - Use longer, more connected sentences when helpful.
        - Preserve nuance and relationships between ideas.
        - Sound challenging but still clear for a strong middle school reader.
        """
    ).strip()


def display_level_name(target_level: str) -> str:
    return LEVEL_NAMES[target_level]


def level_definition(target_level: str) -> str:
    return LEVEL_DEFINITIONS[target_level]


def build_instruction(source_text: str, target_level: str, must_keep_facts: list[str]) -> str:
    facts_block = "\n".join(f"- {fact}" for fact in must_keep_facts)
    guidance_block = level_guidance(target_level)
    level_name = display_level_name(target_level)
    level_meaning = level_definition(target_level)
    return dedent(
        f"""
        Adapt this middle school science lesson for the target reading level: {level_name}.

        Level definition:
        {level_meaning}

        Source lesson:
        {source_text}

        Facts that must stay true:
        {facts_block}

        {guidance_block}

        Output requirements:
        - Preserve all required facts.
        - Match the requested reading level.
        - Keep the topic and meaning the same.
        - Use this exact structure:
          {ADAPTED_LESSON_HEADING}
          <student-facing adapted lesson>

          {KEY_CONCEPTS_HEADING}
          - <short fact 1>
          - <short fact 2>
          - <short fact 3>
        """
    ).strip()


def build_train_text(source_text: str, target_level: str, must_keep_facts: list[str], rewritten_text: str) -> str:
    instruction = build_instruction(source_text, target_level, must_keep_facts)
    return dedent(
        f"""
        <bos><start_of_turn>system
        {SYSTEM_PROMPT}<end_of_turn>
        <start_of_turn>user
        {instruction}<end_of_turn>
        <start_of_turn>model
        {rewritten_text}<end_of_turn><eos>
        """
    ).strip()


def build_inference_prompt(source_text: str, target_level: str, must_keep_facts: list[str]) -> str:
    instruction = build_instruction(source_text, target_level, must_keep_facts)
    return dedent(
        f"""
        <bos><start_of_turn>system
        {SYSTEM_PROMPT}<end_of_turn>
        <start_of_turn>user
        {instruction}<end_of_turn>
        <start_of_turn>model
        """
    ).strip()
