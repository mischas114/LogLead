# Enhanced Parquet Exports

When `demo/LO2_samples.py` is run with `--save-enhancers`, the script writes two additional Parquet files under the chosen output directory (default: `result/lo2/enhanced/`):

- `lo2_events_enhanced.parquet`
- `lo2_sequences_enhanced.parquet`

These "enhanced" tables contain the event- and sequence-level data **after** feature engineering has been applied. The extra columns can include:

- Normalised messages (`e_message_normalized`)
- Token lists (`e_words`, `e_trigrams`)
- Drain parser identifiers (`e_event_drain_id`)
- Length metrics (`e_chars_len`, `e_words_len`, etc.)
- Sequence aggregations (token bags, durations, lengths) when sequence data is present

## Why they exist

Saving the enhanced Parquets is a convenience feature. It lets you:

- Reload feature-rich data in notebooks without waiting for the enhancer pipeline to rerun.
- Share a snapshot of engineered features with teammates without exposing raw logs.
- Prototype alternative models that start from the engineered columns instead of recomputing them.

## How they interact with the MVP scripts

The core end-to-end flow (Steps 1–5 in `docs/LO2_e2e_pipeline.md`) still expects the **raw** loader exports:

- `lo2_events.parquet`
- `lo2_sequences.parquet`

Those files are produced during Step 1 by `demo/run_lo2_loader.py --save-parquet`. Later scripts (`LO2_samples.py`, `lo2_phase_f_explainability.py`) reapply feature engineering as part of their own processing so they always reflect the latest code. Because of that, the enhanced Parquets are *not* consumed automatically by the MVP scripts.

Put differently:

- **Keep** the enhanced Parquets if you want faster ad-hoc analysis or to persist engineered features.
- **Do not rely** on them when running the stock demo scripts—they will rebuild the features from the raw exports anyway.

## Recommended usage

- Use the enhanced Parquets in custom notebooks or tooling where you control the feature pipeline.
- Stick with the standard `lo2_events.parquet` / `lo2_sequences.parquet` inputs when following the documented MVP steps.
- If you prefer the scripts to consume the enhanced data directly, additional code changes would be required (e.g. conditional loading paths and schema checks).

In summary, the enhanced exports are a performance and convenience cache. They accelerate interactive work but do not alter the behaviour of the scripted MVP flow unless you modify the scripts to use them explicitly.
