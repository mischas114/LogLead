import argparse
from pathlib import Path
from typing import Optional

import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


def load_events(events_path: Path) -> pl.DataFrame:
    if not events_path.exists():
        raise FileNotFoundError(f"Events Parquet not found: {events_path}")
    df_events = pl.read_parquet(events_path)
    if "seq_id" not in df_events.columns:
        raise ValueError("Expected 'seq_id' column is missing; run the loader with --save-parquet first.")
    return df_events


def aggregate_events(df_events: pl.DataFrame) -> pl.DataFrame:
    group_cols = ["run", "test_case", "service"]

    df_agg = (
        df_events.group_by(group_cols)
        .agg(
            [
                pl.col("m_message").str.concat(" ").alias("event_seq"),
                pl.count().alias("n_lines"),
                (pl.col("m_timestamp").max() - pl.col("m_timestamp").min())
                .dt.total_seconds()
                .fill_null(0.0)
                .alias("span_s"),
            ]
        )
        .with_columns(
            (pl.col("test_case") != "correct").alias("y_error")
        )
        .select(group_cols + ["event_seq", "n_lines", "span_s", "y_error"])
        .sort(group_cols)
    )
    return df_agg


def maybe_join_metrics(df_agg: pl.DataFrame, metrics_path: Optional[Path]) -> pl.DataFrame:
    if not metrics_path:
        return df_agg

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics Parquet not found: {metrics_path}")

    df_metrics = pl.read_parquet(metrics_path)
    required_cols = {"run", "test_case", "metric_name", "value"}
    missing = required_cols - set(df_metrics.columns)
    if missing:
        raise ValueError(f"Metrics file missing required columns: {sorted(missing)}")

    df_metric_agg = (
        df_metrics.group_by(["run", "test_case", "metric_name"])
        .agg(
            [
                pl.col("value").quantile(0.95).alias("p95"),
                pl.col("value").std().alias("std"),
            ]
        )
        .with_columns(
            pl.concat_str([pl.col("metric_name"), pl.lit("_p95")]).alias("metric_p95_col"),
            pl.concat_str([pl.col("metric_name"), pl.lit("_std")]).alias("metric_std_col"),
        )
    )

    df_metric_p95 = df_metric_agg.select(
        ["run", "test_case", "metric_p95_col", "p95"]
    ).pivot(
        values="p95",
        index=["run", "test_case"],
        columns="metric_p95_col",
    )

    df_metric_std = df_metric_agg.select(
        ["run", "test_case", "metric_std_col", "std"]
    ).pivot(
        values="std",
        index=["run", "test_case"],
        columns="metric_std_col",
    )

    df_metrics_wide = df_metric_p95.join(df_metric_std, on=["run", "test_case"], how="outer")

    df_joined = df_agg.join(df_metrics_wide, on=["run", "test_case"], how="left")
    return df_joined


def vectorize(df_features: pl.DataFrame, standardize: bool) -> tuple[sparse.csr_matrix, list[int], CountVectorizer, Optional[StandardScaler]]:
    event_text = df_features["event_seq"].to_list()
    if not event_text:
        raise ValueError("No event sequences available for vectorization.")

    vectorizer = CountVectorizer(token_pattern=r"\S+")
    bow_matrix = vectorizer.fit_transform(event_text)

    numeric_cols = [
        name
        for name in df_features.columns
        if name in {"n_lines", "span_s"} or name.endswith(("_p95", "_std"))
    ]

    numeric_frame = df_features.select(numeric_cols).fill_null(0) if numeric_cols else None
    numeric_data = numeric_frame.to_numpy() if numeric_frame is not None else None

    scaler: Optional[StandardScaler] = None
    if standardize and numeric_data is not None and numeric_data.size:
        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_data)

    if numeric_data is not None and numeric_data.size:
        numeric_sparse = sparse.csr_matrix(numeric_data)
    else:
        numeric_sparse = sparse.csr_matrix((len(event_text), 0))
    feature_matrix = sparse.hstack([bow_matrix, numeric_sparse], format="csr")

    labels = df_features["y_error"].cast(pl.Int64).to_list()
    return feature_matrix, labels, vectorizer, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C feature engineering smoke-test for LO2 data.")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("demo/result/lo2/lo2_events.parquet"),
        help="Path to the event-level Parquet produced by run_lo2_loader.py",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional Parquet with metrics (same run/test_case columns)",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply StandardScaler to numeric feature columns before stacking.",
    )
    args = parser.parse_args()

    df_events = load_events(args.events)
    print(f"Loaded events: {df_events.shape}")

    df_agg = aggregate_events(df_events)
    print(f"Aggregated sequences: {df_agg.shape}")

    df_features = maybe_join_metrics(df_agg, args.metrics)

    X, y, vectorizer, scaler = vectorize(df_features, args.standardize)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: positives={sum(y)} | total={len(y)}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    if scaler:
        print("Numeric columns were standardized (mean=0, std=1).")
    else:
        print("Numeric columns kept in raw scale.")


if __name__ == "__main__":
    main()
