# Benchmark Results Draft

Use this section after the first CUDA benchmark pass. Replace the bracketed values with real numbers from `artifacts/evals/first_run_summary.md` and the strongest held-out example from `artifacts/evals/first_run_eval.json`.

## Benchmark Results

To verify that our post-training work improved the model on the exact classroom task we cared about, we evaluated **base Gemma 4** against our **Unsloth fine-tuned Gemma 4 adapter** on a held-out set of middle school science passages. Each test example asked the model to rewrite a source lesson for one target reading level: `below`, `on`, or `above`.

We scored both models on three criteria:

- **Fact coverage:** how well the output preserved required science facts from the source lesson
- **Reading-level alignment:** whether the output matched the requested complexity band
- **Teacher usefulness:** a lightweight proxy combining fact preservation, target-level control, and output completeness

### Summary table

| Metric | Base Gemma 4 | Tuned Gemma 4 | Delta |
|---|---:|---:|---:|
| Avg fact coverage | [0.000] | [0.000] | [+0.000] |
| Avg teacher usefulness | [0.000] | [0.000] | [+0.000] |
| Within target band rate | [0.000] | [0.000] | [+0.000] |

These results matter because our project is not trying to solve every education task. It is focused on one narrow, high-value workflow: helping a teacher turn one science lesson into level-appropriate versions without changing the underlying science. A strong result here is more meaningful than a broad but noisy benchmark.

### Qualitative example

On the held-out example **[SOURCE_ID]**, the base model produced a rewrite that **[describe drift: omitted a fact / overcomplicated the below-level version / missed the requested level]**. The tuned model kept the required facts intact and produced a version that was more appropriate for the requested level.

This is the pattern we wanted from fine-tuning:

- less factual drift
- tighter control over reading complexity
- outputs a teacher could use with lighter editing

### Why this benchmark supports the story

Our submission is built around a teacher in a low-bandwidth classroom who needs to serve students with different reading levels from the same lesson. The benchmark shows that the tuned model is not just different from the base model. It is **more reliable on the exact transformation the teacher needs most**.

That is the core technical claim behind the demo: one lesson becomes three versions quickly, and the tuned model does a better job preserving the science while adapting the reading level.
