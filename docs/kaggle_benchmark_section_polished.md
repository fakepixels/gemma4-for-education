# Benchmark Results

To test whether post-training improved the exact classroom workflow we cared about, we compared **base Gemma 4** with our **Unsloth fine-tuned Gemma 4 adapter** on a held-out set of middle school science passages. Each example asked the model to adapt one lesson for a target reading level: `below`, `on`, or `above`.

Our evaluation focused on three criteria:

- **Fact coverage:** whether the adapted lesson preserved the required science facts
- **Reading-level alignment:** whether the lesson matched the requested complexity band
- **Teacher usefulness:** a lightweight classroom proxy combining factual preservation, level control, and output completeness

## Results

| Metric | Base Gemma 4 | Tuned Gemma 4 | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.417 | 1.000 | +0.583 |
| Avg teacher usefulness | 0.408 | 0.967 | +0.559 |
| Within target band rate | 0.417 | 0.833 | +0.416 |

These gains are meaningful because our project is intentionally narrow. We are not benchmarking general educational QA or open-ended tutoring. We are measuring one high-value transformation: can the model reliably turn a teacher’s science lesson into a level-appropriate classroom version without changing the science?

## Qualitative Example

One of the clearest held-out examples was **`electric_circuits_001` at the `on` level**. Under the same prompt, the base model failed to produce a usable adapted lesson. The tuned model returned a complete structured response, preserved all five required circuit facts, and matched the requested classroom format:

- an adapted student-facing lesson
- a teacher-facing “Key Concepts Preserved” section

We saw the same pattern on other held-out `on`-level examples, including **`cells_001`** and **`atoms_molecules_001`**. In each case, the tuned model was more reliable at producing a complete classroom-ready output while preserving the science content.

## Why This Supports The Project Story

The core story of this project is that one teacher may need to serve several reading levels from the same lesson, often with limited time and unreliable connectivity. The benchmark supports that story directly. The tuned model is not simply “different” from the base model. It is **more dependable on the exact adaptation task the teacher needs**.

That is the technical claim behind the demo: one science lesson can become multiple level-appropriate versions quickly, and the tuned Gemma 4 model does a better job preserving facts, following the expected structure, and producing something a teacher could actually use.
