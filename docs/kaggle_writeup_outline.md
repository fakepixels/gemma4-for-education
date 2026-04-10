# Kaggle Writeup Outline

## Title
Gemma 4 Classroom Adaptation: One Science Lesson, Three Reading Levels, Zero Cloud Dependence

## Subtitle
Fine-tuning Gemma 4 with Unsloth to help middle school science teachers serve mixed reading levels in low-bandwidth classrooms.

## Recommended structure

### 1. Problem
- One teacher often supports students with very different reading levels in the same room.
- Rewriting the same lesson by hand takes time teachers do not have.
- Cloud-first AI is not always practical in low-connectivity school settings.

### 2. Solution
- Build a local-first lesson adaptation assistant powered by Gemma 4.
- Teacher pastes one middle school science lesson.
- The system generates `below`, `on`, and `above` reading-level versions.
- The teacher stays in control and reviews every output.

### 3. Why Gemma 4
- Local deployment story for weak-connectivity settings.
- Strong open-model foundation for post-training.
- Clean adaptation target for instruction tuning.

### 4. Fine-tuning method
- Single-target rewriting with target labels `below`, `on`, and `above`.
- Unsloth LoRA/QLoRA training.
- Preservation constraints via required fact lists.
- Same prompt format used in training and inference.

### 5. Dataset
- Middle school science topics:
  - water cycle
  - ecosystems
  - cells
  - weather
  - energy
  - matter
- Each source lesson expanded into three reading-level rewrites.
- Train/validation/test split by source lesson, not by variant.

### 6. Evaluation
- Base Gemma 4 vs tuned Gemma 4.
- Fact coverage against must-keep facts.
- Readability alignment for requested target level.
- Teacher usefulness proxy and example-based qualitative review.

### 7. Results
- Include one compact table with before/after metrics.
- Show one strong example where the base model drifts or misses the target level and the tuned model stays grounded.

### 8. Live demo
- One lesson becomes three versions in seconds.
- Emphasize low-bandwidth classroom preparation and teacher oversight.

### 9. Limitations and next steps
- More teacher-reviewed data.
- Larger held-out evaluation set.
- Optional offline packaging on supported edge hardware.
