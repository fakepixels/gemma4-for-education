# Benchmark Appendix

Base model: `google/gemma-4-E4B-it`
Adapter path: `artifacts/adapters/first-run`

This appendix expands the top-line benchmark with per-level results and per-example deltas.

## Overall Summary

| Metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.417 | 1.000 | +0.583 |
| Avg teacher usefulness | 0.408 | 0.967 | +0.559 |
| Within target band rate | 0.417 | 0.833 | +0.416 |

## Per-Level Breakdown

### `below`

| Metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.750 | 1.000 | +0.250 |
| Avg teacher usefulness | 0.725 | 0.900 | +0.175 |
| Within target band rate | 0.750 | 0.500 | -0.250 |
| Avg level-control score | 0.812 | 0.688 | -0.124 |

### `on`

| Metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.000 | 1.000 | +1.000 |
| Avg teacher usefulness | 0.000 | 1.000 | +1.000 |
| Within target band rate | 0.000 | 1.000 | +1.000 |
| Avg level-control score | 0.600 | 0.938 | +0.338 |

### `above`

| Metric | Base | Tuned | Delta |
|---|---:|---:|---:|
| Avg fact coverage | 0.500 | 1.000 | +0.500 |
| Avg teacher usefulness | 0.500 | 1.000 | +0.500 |
| Within target band rate | 0.500 | 1.000 | +0.500 |
| Avg level-control score | 0.600 | 0.950 | +0.350 |

## Reliability Notes

- Blank base outputs on the held-out set: `7` / `12`
- Blank tuned outputs on the held-out set: `0` / `12`
- The biggest gains are reliability and structure-following at `on` and `above`.
- The main remaining weakness is `below`, where tuned preserves facts well but can stay too close to the original wording.

## Largest Tuned Wins

| Example | Fact Coverage | Teacher Usefulness | Within Band |
|---|---|---|---|
| `electric_circuits_001` / `on` | 0.000 -> 1.000 | 0.000 -> 1.000 | `False` -> `True` |
| `electric_circuits_001` / `above` | 0.000 -> 1.000 | 0.000 -> 1.000 | `False` -> `True` |
| `atoms_molecules_001` / `on` | 0.000 -> 1.000 | 0.000 -> 1.000 | `False` -> `True` |
| `atoms_molecules_001` / `above` | 0.000 -> 1.000 | 0.000 -> 1.000 | `False` -> `True` |
| `ecosystems_001` / `on` | 0.000 -> 1.000 | 0.000 -> 1.000 | `False` -> `True` |

## Regressions To Fix

| Example | Fact Coverage | Teacher Usefulness | Within Band |
|---|---|---|---|
| `atoms_molecules_001` / `below` | 1.000 -> 1.000 | 1.000 -> 0.800 | `True` -> `False` |
| `ecosystems_001` / `below` | 1.000 -> 1.000 | 1.000 -> 0.800 | `True` -> `False` |

## Per-Example Table

| Source | Level | Base Fact | Tuned Fact | Base Useful | Tuned Useful | Base Band | Tuned Band | Delta |
|---|---|---:|---:|---:|---:|---|---|---:|
| `atoms_molecules_001` | `below` | 1.000 | 1.000 | 1.000 | 0.800 | `True` | `False` | -0.200 |
| `atoms_molecules_001` | `on` | 0.000 | 1.000 | 0.000 | 1.000 | `False` | `True` | +1.000 |
| `atoms_molecules_001` | `above` | 0.000 | 1.000 | 0.000 | 1.000 | `False` | `True` | +1.000 |
| `cells_001` | `below` | 1.000 | 1.000 | 0.800 | 1.000 | `False` | `True` | +0.200 |
| `cells_001` | `on` | 0.000 | 1.000 | 0.000 | 1.000 | `False` | `True` | +1.000 |
| `cells_001` | `above` | 1.000 | 1.000 | 1.000 | 1.000 | `True` | `True` | +0.000 |
| `ecosystems_001` | `below` | 1.000 | 1.000 | 1.000 | 0.800 | `True` | `False` | -0.200 |
| `ecosystems_001` | `on` | 0.000 | 1.000 | 0.000 | 1.000 | `False` | `True` | +1.000 |
| `ecosystems_001` | `above` | 1.000 | 1.000 | 1.000 | 1.000 | `True` | `True` | +0.000 |
| `electric_circuits_001` | `below` | 0.000 | 1.000 | 0.100 | 1.000 | `True` | `True` | +0.900 |
| `electric_circuits_001` | `on` | 0.000 | 1.000 | 0.000 | 1.000 | `False` | `True` | +1.000 |
| `electric_circuits_001` | `above` | 0.000 | 1.000 | 0.000 | 1.000 | `False` | `True` | +1.000 |
