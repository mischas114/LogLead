# Isolation Forest Next Steps

- **Expand training data:** Load all 666 new log files via `run_lo2_loader.py --save-parquet`; more correct runs improve the token statistics the Isolation Forest learns from.
- **Calibrate on recent runs:** Reserve the last 10 % of correct events with `--if-holdout-fraction 0.1` so score drift against fresh logs becomes visible before deployment.
- **Target alert rate:** Set `--if-threshold-percentile 99.5` to derive a data-driven score cutoff that yields a predictable alert volume.
- **Track quality:** Export `Precision@100`, `FP_rate@0.005`, and `PSI` with `--report-precision-at 100 --report-fp-alpha 0.005 --report-psi` so you can confirm precision rises, false positives shrink, and score drift stays ≤ 0.2.
- **Persist context:** Use `--save-model` and `--dump-metadata` to bundle the model, threshold, dataset sizes, and git commit into `models/` for reproducible scoring across sessions.
