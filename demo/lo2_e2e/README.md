# LO2 E2E Demo Flow

This folder groups the runnable scripts for the LO2 pipeline, covering data loading, enhancement, anomaly detection, and explainability artefacts.

## Quickstart

1. **Load raw runs to Parquet**
   ```bash
   python demo/lo2_e2e/run_lo2_loader.py --root /path/to/lo2_data --runs 5 --save-parquet --output-dir demo/result/lo2
   ```
2. **Generate enhancements and anomaly predictions**
   ```bash
   python demo/lo2_e2e/LO2_samples.py --phase full --save-enhancers --save-model models/lo2_if.joblib
   ```
   The `--save-model` argument stores the IsolationForest + vectorizer bundle for reuse; add `--overwrite-model` when replacing an existing dump.
3. **Create explainability artefacts**
   ```bash
   MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --root demo/result/lo2 --shap-sample 200
   ```

All outputs are written beneath `demo/result/lo2` by default. Adjust CLI options to mirror your dataset size and desired sampling behaviour.
