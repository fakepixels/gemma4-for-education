# Benchmark Results Draft

## Benchmark Results

To verify that our post-training work improved the model on the exact classroom task we cared about, we evaluated **base Gemma 4** against our **Unsloth fine-tuned Gemma 4 adapter** on a held-out set of middle school science passages. Each test example asked the model to rewrite a source lesson for one target reading level: `below`, `on`, or `above`.

We scored both models on three criteria:

- **Fact coverage:** how well the output preserved required science facts from the source lesson
- **Reading-level alignment:** whether the output matched the requested complexity band
- **Teacher usefulness:** a lightweight proxy combining fact preservation, target-level control, and output completeness

### Summary table

| Metric | Base Gemma 4 | Tuned Gemma 4 | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.417 | 1.000 | +0.583 |
| Avg teacher usefulness | 0.408 | 0.967 | +0.559 |
| Within target band rate | 0.417 | 0.833 | +0.416 |

These results matter because our project is not trying to solve every education task. It is focused on one narrow, high-value workflow: helping a teacher turn one science lesson into level-appropriate versions without changing the underlying science. A strong result here is more meaningful than a broad but noisy benchmark.

### Qualitative example

On the held-out example **`electric_circuits_001` at the `on` level**, the base model failed to produce a usable adapted lesson or preserved-facts section. Under the same prompt, the tuned model returned a complete classroom-ready response with the requested structure, preserved all five required science facts, and matched the target level band.

This was a representative pattern in our held-out set. Several `on`-level prompts, including `electric_circuits_001`, `atoms_molecules_001`, and `cells_001`, showed the same failure mode for the base model: incomplete or empty usable output. The tuned model consistently returned a structured lesson plus a teacher-facing fact check.

This is the pattern we wanted from fine-tuning:

- less factual drift
- tighter control over reading complexity
- outputs a teacher could use with lighter editing
- stronger adherence to the output format used by the classroom workflow

### Why this benchmark supports the story

Our submission is built around a teacher in a low-bandwidth classroom who needs to serve students with different reading levels from the same lesson. The benchmark shows that the tuned model is not just different from the base model. It is **more reliable on the exact transformation the teacher needs most**.

That is the core technical claim behind the demo: one lesson becomes three versions quickly, and the tuned model does a better job preserving the science while adapting the reading level.
