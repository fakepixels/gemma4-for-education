# Gemma 4 Classroom Adaptation: One Science Lesson, Three Reading Levels, Zero Cloud Dependence

## Fine-tuning Gemma 4 with Unsloth to help middle school science teachers serve mixed reading levels in low-bandwidth classrooms

In many classrooms, one teacher has to support students reading at very different levels from the same lesson. That problem is especially difficult in science, where simplifying too much can distort the content, while keeping the original wording can leave some students behind. It becomes even harder in low-connectivity settings, where cloud-first tools may be slow, unreliable, or unsuitable for the classroom workflow.

Our project focuses on one narrow, practical use case: a teacher pastes one middle school science lesson, and the system generates `below`, `on`, and `above` reading-level versions while preserving the same core science facts. The teacher remains fully in control and reviews every output.

## Why Gemma 4

Gemma 4 was a strong fit for this project because the challenge is not open-ended tutoring. It is controlled transformation. We needed a model that could be post-trained to follow a precise classroom format, preserve required facts, and support a local-first deployment story for low-bandwidth environments.

## Method

We fine-tuned `google/gemma-4-E4B-it` using **Unsloth LoRA**. The training task was single-target rewriting: each example contained a source lesson, a target level label (`below`, `on`, or `above`), a list of must-keep facts, and a reference adaptation. We also standardized the output format into two sections:

- `Adapted Lesson`
- `Key Concepts Preserved`

This made the task easier to control during both training and evaluation. The adapted lesson is student-facing. The key concepts section is teacher-facing and acts as a quick factual check.

## Dataset

The dataset is a compact middle school science adaptation corpus built around topics such as:

- ecosystems
- cells
- atoms and molecules
- electric circuits
- water cycle
- weather
- energy
- matter

Each source lesson was expanded into three reading-level versions. We split the data by source lesson rather than by rewrite variant, so the held-out set tested generalization to unseen passages rather than memorization of paraphrases.

## Evaluation

We compared base Gemma 4 against the fine-tuned adapter on a held-out benchmark. The evaluation measured:

- **Fact coverage:** whether required science facts were preserved
- **Reading-level alignment:** whether the adapted lesson matched the requested level
- **Teacher usefulness:** a classroom-oriented proxy combining factual fidelity, level control, and output completeness

We also updated the scoring pipeline to better reflect the real product. The benchmark now scores the student-facing lesson separately from the teacher note, extracts the final assistant answer cleanly, and uses a science-aware rubric for level control so necessary domain vocabulary is not unfairly penalized.

## Results

| Metric | Base Gemma 4 | Tuned Gemma 4 | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.417 | 1.000 | +0.583 |
| Avg teacher usefulness | 0.408 | 0.967 | +0.559 |
| Within target band rate | 0.417 | 0.833 | +0.416 |

These gains matter because the benchmark is tightly matched to the actual classroom workflow. We are not claiming a general win on all educational tasks. We are showing a measurable improvement on the exact transformation this teacher-facing tool is designed to perform.

## Qualitative Example

One of the clearest held-out examples was `electric_circuits_001` at the `on` level. Under the same prompt, the base model failed to produce a usable adapted lesson. The tuned model returned a complete structured response, preserved all required circuit facts, and matched the requested classroom format.

We saw similar behavior on held-out `on`-level examples such as `cells_001` and `atoms_molecules_001`. In these cases, the tuned model was more reliable at producing a classroom-ready output rather than an incomplete or malformed one.

## Demo

The demo is intentionally simple. A teacher pastes one science lesson and receives:

- a below-level adapted lesson
- an on-level adapted lesson
- an above-level adapted lesson
- a teacher note summarizing preserved concepts

This supports the story we want to tell in the submission: one lesson becomes three usable versions quickly, without relying on a cloud-heavy workflow.

## Limitations and Next Steps

Our strongest current gains are in reliability, structure-following, and factual preservation. The next improvement target is stronger stylistic separation at the level extremes, especially for some `below`-level examples where the base model still simplifies more aggressively. The next step is a focused data pass that sharpens low-level simplification and high-level enrichment without weakening factual control.

## Impact

The value of this project is not novelty for its own sake. It is practical leverage for a real classroom constraint. A low-bandwidth teacher workflow needs outputs that are reliable, grounded, and fast to review. Fine-tuning Gemma 4 let us move from a generic model to one that is better aligned with that exact need.
