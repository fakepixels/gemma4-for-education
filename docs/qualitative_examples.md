# Qualitative Examples

Use these examples in the Kaggle writeup, demo narration, or video captions. They come from the held-out benchmark artifact at `/workspace/gemma4-for-education/artifacts/evals/first_run_eval.json`.

## Best examples

### 1. `electric_circuits_001` at `Level 2`

Why it is strong:

- Base model teacher usefulness: `0.0`
- Tuned model teacher usefulness: `1.0`
- Base fact coverage: `0.0`
- Tuned fact coverage: `1.0`
- Base within-band: `False`
- Tuned within-band: `True`

What happened:

- Base model did not produce a usable adapted lesson.
- Tuned model returned a complete structured response with the adapted lesson and preserved facts.
- This is the cleanest example of post-training improving reliability, not just style.

Suggested narration:

> On the same prompt, the base model did not return a classroom-ready adaptation, while the tuned model produced a complete lesson and preserved every required circuit fact.

### 2. `cells_001` at `Level 2`

Why it is strong:

- Base model teacher usefulness: `0.0`
- Tuned model teacher usefulness: `1.0`
- Base fact coverage: `0.0`
- Tuned fact coverage: `1.0`
- Base within-band: `False`
- Tuned within-band: `True`

What happened:

- Base model again failed to produce a usable lesson body and fact check.
- Tuned model returned a full response that preserved the cell structures and organelle facts.
- This helps show the improvement was not isolated to a single topic.

Suggested narration:

> The improvement carried across topics. On cells, the tuned model preserved the biology facts and returned the expected teacher workflow format.

### 3. `atoms_molecules_001` at `Level 2`

Why it is strong:

- Base model teacher usefulness: `0.0`
- Tuned model teacher usefulness: `1.0`
- Base fact coverage: `0.0`
- Tuned fact coverage: `1.0`
- Base within-band: `False`
- Tuned within-band: `True`

What happened:

- Base model failed to produce a usable adapted lesson.
- Tuned model returned a complete structured answer with all required atom and molecule facts intact.
- This is a good supporting example because it shows the same gain in physical science, not just life science.

Suggested narration:

> The same pattern appears in chemistry-style content. The tuned model produces a complete adaptation where the base model does not.

## Useful supporting example

### `cells_001` at `Level 1`

Why it matters:

- Base model teacher usefulness: `0.8`
- Tuned model teacher usefulness: `1.0`
- Base fact coverage: `1.0`
- Tuned fact coverage: `1.0`
- Base within-band: `False`
- Tuned within-band: `True`

What happened:

- Both models preserved the science facts.
- The tuned model scored better because it stayed within the target band under the updated rubric.
- This is a useful example when you want to show a more subtle improvement rather than a hard failure from the base model.

## Cautionary examples

These are not ideal hero examples for the writeup, but they are useful internally:

- `atoms_molecules_001` at `Level 1`
- `ecosystems_001` at `Level 1`

In both cases, the base model simplified more aggressively, while the tuned model stayed closer to the source wording. These are good targets for the next small data pass focused on stronger `Level 1` style separation.
