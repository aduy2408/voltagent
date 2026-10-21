# Retrieval benchmarks

This directory contains reusable evaluation inputs, outputs, and runners for the Kuphaldt retrieval experiments.

## Layout

- `questions/` contains the baseline set, the independently curated batches, and the merged 161-question benchmark.
- `results/` contains benchmark outputs. These are reproducible artifacts, not application runtime data.
- `scripts/` contains standalone runners that can be invoked from the repository root.

## Common commands

```bash
uv run python -m benchmarks.scripts.merge_question_sets
uv run python -m benchmarks.scripts.evaluate_retrieval --json
uv run python -m benchmarks.scripts.evaluate_dense_and_rerank --device auto --no-rerank
```

The dense runner writes indexes under `data/indexes/benchmark/` and results under `benchmarks/results/` by default. Override paths with `--index-root`, `--questions`, or `--output` when running on a remote machine.
