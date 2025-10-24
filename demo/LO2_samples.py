"""LO2 demo pipeline for enhancement, anomaly detection, and explainability."""

import argparse
import os
import random
from pathlib import Path

import polars as pl

from loglead import AnomalyDetector
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
import loglead.explainer as ex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LO2 enhancement pipeline with optional anomaly detection phases."
    )
    parser.add_argument(
        "--phase",
        choices=["enhancers", "if", "full"],
        default="full",
        help="Use 'enhancers' to stop after feature generation; 'if' trainiert IsolationForest; 'full' ergänzt LR/DT + XAI.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling enhanced records and optional down-sampling.",
    )
    parser.add_argument(
        "--if-contamination",
        type=float,
        default=0.1,
        help="IsolationForest contamination (Anteil erwarteter Anomalien).",
    )
    parser.add_argument(
        "--if-n-estimators",
        type=int,
        default=200,
        help="Anzahl Trees für IsolationForest.",
    )
    parser.add_argument(
        "--if-max-samples",
        default="auto",
        help="max_samples für IsolationForest (Ganzzahl oder 'auto').",
    )
    parser.add_argument(
        "--if-item",
        default="e_words",
        help="Spalte mit Tokenlisten für IsolationForest (z.B. e_words, e_trigrams, e_event_drain_id).",
    )
    parser.add_argument(
        "--if-numeric",
        default="",
        help="Kommagetrennte numerische Zusatzfeatures (z.B. e_chars_len).",
    )
    parser.add_argument(
        "--save-if",
        type=Path,
        default=Path("result/lo2/lo2_if_predictions.parquet"),
        help="Pfad für IsolationForest-Ergebnis (Parquet oder CSV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Keep working directory stable so relative paths resolve.
    script_dir = Path(__file__).resolve().parent
    orig_cwd = Path.cwd()
    os.chdir(script_dir)

    # Expected loader output locations.
    loader_output = (script_dir / "../demo/result/lo2").resolve()
    events_path = loader_output / "lo2_events.parquet"
    seq_path = loader_output / "lo2_sequences.parquet"

    if not events_path.exists():
        raise SystemExit(
            "Missing Parquet export. Run run_lo2_loader.py with --save-parquet before executing this script."
        )

    print(f"Reading LO2 events from {events_path}")
    df_events = pl.read_parquet(events_path)
    print(f"Number of LO2 events: {len(df_events)}")
    if "anomaly" in df_events.columns:
        event_ano = int(df_events["anomaly"].sum())
        print(f"Event anomalies: {event_ano} ({event_ano / max(len(df_events), 1) * 100:.2f}%)")

    # Optional sequence-level table.
    df_seqs = None
    if seq_path.exists():
        print(f"Reading LO2 sequences from {seq_path}")
        df_seqs = pl.read_parquet(seq_path)
        if len(df_seqs):
            seq_ano = int(df_seqs["anomaly"].sum()) if "anomaly" in df_seqs.columns else 0
            print(f"Sequence anomalies: {seq_ano} ({seq_ano / max(len(df_seqs), 1) * 100:.2f}%)")
    else:
        print("No lo2_sequences.parquet found; continuing with event-only workflow.")

    # Optional down-sampling for quick experimentation.
    MAX_EVENTS = 200_000
    if len(df_events) > MAX_EVENTS:
        print(f"Sampling down to {MAX_EVENTS} events for faster demos.")
        df_events = df_events.sample(n=MAX_EVENTS, seed=args.sample_seed)

    print("\nEnhancing events (normalization, tokens, parsers, lengths)...")
    enhancer = EventLogEnhancer(df_events)
    df_events = enhancer.normalize()
    df_events = enhancer.words()
    df_events = enhancer.trigrams()
    try:
        df_events = enhancer.parse_drain()
    except Exception as exc:  # drain parser can fail if templates missing
        print(f"Drain parsing skipped: {exc}")

    df_events = enhancer.length()

    random.seed(args.sample_seed)
    rand_idx = random.randint(0, len(df_events) - 1)
    print("\nSample enhanced record:")
    print(f"Original:  {df_events['m_message'][rand_idx]}")
    print(f"Words:     {df_events['e_words'][rand_idx]}")
    print(f"Trigrams:  {df_events['e_trigrams'][rand_idx]}")
    if "e_event_drain_id" in df_events.columns:
        print(f"Drain ID: {df_events['e_event_drain_id'][rand_idx]}")
    print(f"Len chars: {df_events['e_chars_len'][rand_idx]}")

    if df_seqs is not None and len(df_seqs):
        print("\nAggregating to sequence level...")
        seq_enhancer = SequenceEnhancer(df=df_events, df_seq=df_seqs)
        df_seqs = seq_enhancer.seq_len()
        df_seqs = seq_enhancer.duration()
        df_seqs = seq_enhancer.tokens(token="e_words")
        df_seqs = seq_enhancer.tokens(token="e_trigrams")

    if args.phase == "enhancers":
        print("\nEnhancer phase complete. Skipping anomaly detection and explainability.")
        return

    # Isolation Forest baseline (Phase D)
    print("\nTraining Isolation Forest on event words (Phase D)")
    numeric_cols = [col.strip() for col in args.if_numeric.split(",") if col.strip()]
    sad_if = AnomalyDetector(item_list_col=args.if_item, numeric_cols=numeric_cols or None)
    sad_if.train_df = df_events.filter(pl.col("test_case") == "correct")
    sad_if.test_df = df_events
    sad_if.prepare_train_test_data()

    max_samples = args.if_max_samples
    if isinstance(max_samples, str) and max_samples != "auto":
        if max_samples.isdigit():
            max_samples = int(max_samples)
        else:
            raise SystemExit("--if-max-samples muss 'auto' oder eine Ganzzahl sein.")

    sad_if.train_IsolationForest(
        filter_anos=True,
        n_estimators=args.if_n_estimators,
        contamination=args.if_contamination,
        max_samples=max_samples,
    )
    pred_if = sad_if.predict()

    # Add raw anomaly scores and dense ranking for inspection.
    score_if = (-sad_if.model.score_samples(sad_if.X_test)).tolist()
    pred_if = pred_if.with_columns(
        pl.Series(name="score_if", values=score_if)
    ).with_columns(
        pl.col("score_if").rank("dense", descending=True).alias("rank_if")
    )
    print("Top 5 IF-Runs (höchster Score zuerst):")
    print(pred_if.sort("score_if", descending=True).head(5))

    save_if_path = args.save_if
    if not save_if_path.is_absolute():
        save_if_path = (orig_cwd / save_if_path).resolve()
    save_if_path.parent.mkdir(parents=True, exist_ok=True)
    if save_if_path.suffix == ".csv":
        pred_if.write_csv(save_if_path)
    else:
        pred_if.write_parquet(save_if_path)
    print(f"IsolationForest-Ergebnis gespeichert unter {save_if_path}")

    if args.phase == "if":
        print("\nIsolation Forest abgeschlossen. Weitere Modelle übersprungen.")
        return

    print("\nTraining anomaly detector on events (words)")
    sad = AnomalyDetector()
    sad.item_list_col = "e_words"
    sad.test_train_split(df_events, test_frac=0.90)
    sad.prepare_train_test_data()
    sad.train_LR()
    df_pred = sad.predict()
    print("Event-level predictions ready.")

    print("Switching to trigrams + DecisionTree")
    sad.item_list_col = "e_trigrams"
    sad.prepare_train_test_data()
    sad.train_DT()
    df_pred = sad.predict()

    if df_seqs is not None and len(df_seqs):
        print("\nSequence-level anomaly detection with duration + length")
        sad_seq = AnomalyDetector()
        sad_seq.numeric_cols = ["seq_len", "duration_sec"]
        sad_seq.test_train_split(df_seqs, test_frac=0.90)
        sad_seq.train_LR()
        seq_pred = sad_seq.predict()
        print("Sequence-level predictions ready.")

        print("\nExplaining sequence model via SHAP (words vectorizer)")
        sad_seq.item_list_col = "e_words"
        sad_seq.numeric_cols = None
        sad_seq.prepare_train_test_data()
        sad_seq.train_LR()
        seq_pred = sad_seq.predict()
        explainer = ex.ShapExplainer(sad_seq, ignore_warning=True, plot_featurename_len=18)
        explainer.calc_shapvalues()
        explainer.plot(plottype="summary")
    else:
        print("\nNo sequence table available; skipping sequence-level AD and XAI.")

    print("\nLO2 sample pipeline complete.")


if __name__ == "__main__":
    main()
