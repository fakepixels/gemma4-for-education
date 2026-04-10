from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are an expert middle school science curriculum adapter.
    Rewrite the source lesson for the requested reading level.

    Rules:
    1. Preserve the science facts from the source.
    2. Do not add new scientific claims.
    3. Change vocabulary, sentence structure, and scaffolding to fit the target level.
    4. Keep the answer in paragraph form unless the prompt asks otherwise.
    5. End with a short section titled "Key Concepts Preserved" listing the main facts that stayed the same.
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


def build_instruction(source_text: str, target_level: str, must_keep_facts: list[str]) -> str:
    facts_block = "\n".join(f"- {fact}" for fact in must_keep_facts)
    guidance_block = level_guidance(target_level)
    return dedent(
        f"""
        Adapt this middle school science lesson for the target reading level: {target_level}.

        Source lesson:
        {source_text}

        Facts that must stay true:
        {facts_block}

        {guidance_block}

        Output requirements:
        - Preserve all required facts.
        - Match the requested reading level.
        - Keep the topic and meaning the same.
        - Finish with "Key Concepts Preserved".
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
