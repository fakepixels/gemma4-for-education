from __future__ import annotations

import argparse

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_classroom.prompting import build_inference_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Gemma 4 classroom adaptation demo.")
    parser.add_argument("--model-id", default="google/gemma-4-4b-it")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_model(model_id: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**encoded, max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if prompt in decoded:
        return decoded.split(prompt, 1)[1].strip()
    return decoded.strip()


def extract_teacher_note(outputs: dict[str, str]) -> str:
    shared_concepts = []
    for text in outputs.values():
        if "Key Concepts Preserved" in text:
            shared_concepts.append(text.split("Key Concepts Preserved", 1)[1].strip())
    if not shared_concepts:
        return "Review the three versions and confirm the main science facts stayed consistent."
    return "Preserved concepts across levels:\n\n" + "\n\n".join(shared_concepts)


def build_app(model, tokenizer) -> gr.Blocks:
    def generate_versions(source_text: str, must_keep_facts: str):
        facts = [line.strip("- ").strip() for line in must_keep_facts.splitlines() if line.strip()]
        outputs = {}
        for level in ("below", "on", "above"):
            prompt = build_inference_prompt(source_text=source_text, target_level=level, must_keep_facts=facts)
            outputs[level] = generate_text(model, tokenizer, prompt)
        teacher_note = extract_teacher_note(outputs)
        return outputs["below"], outputs["on"], outputs["above"], teacher_note

    with gr.Blocks(title="Gemma 4 Classroom Adaptation") as demo:
        gr.Markdown(
            """
            # Gemma 4 Classroom Adaptation
            Paste one middle school science lesson and generate three reading-level versions.

            This demo is designed for low-bandwidth classroom preparation. The teacher reviews and approves every output.
            """
        )
        with gr.Row():
            source_text = gr.Textbox(
                label="Source lesson",
                lines=14,
                placeholder="Paste a middle school science lesson here...",
            )
            must_keep_facts = gr.Textbox(
                label="Facts that must stay true",
                lines=8,
                placeholder="- The Sun's heat causes evaporation.\n- Plants release water vapor through transpiration.",
            )
        generate_button = gr.Button("Generate three versions")
        with gr.Row():
            below = gr.Textbox(label="Below-level version", lines=14)
            on = gr.Textbox(label="On-level version", lines=14)
            above = gr.Textbox(label="Above-level version", lines=14)
        teacher_note = gr.Textbox(label="Teacher note", lines=8)
        generate_button.click(
            fn=generate_versions,
            inputs=[source_text, must_keep_facts],
            outputs=[below, on, above, teacher_note],
        )
    return demo


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.model_id, args.adapter_path)
    app = build_app(model, tokenizer)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
