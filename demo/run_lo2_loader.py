import argparse
from pathlib import Path

import polars as pl

from loglead.loaders import LO2Loader

# Quick tweak guide for the CLI options:
#   --root             -> Mandatory; points at the LO2 directory that has run folders.
#   --runs             -> Limit how many run_* folders you scan (helpful for smoke tests).
#   --errors-per-run   -> Number of error directories to sample per run when duplicates stay allowed.
#   --allow-duplicates -> Keep repeated error types across runs; disable for broader coverage.
#   --single-error-type-> Pin the loader to one error folder (use "random" for a surprise pick).
#   --single-service   -> Match the suffix in filenames like oauth2-oauth2-<service>.log.
#   --head             -> Control how many rows you print from each dataframe preview.
#   --load-metrics     -> Also parse metrics/*.json if those files are present.
#   --save-parquet     -> Persist the processed tables to Parquet (uses --output-dir).
#   --output-dir       -> Target directory for Parquet exports when you enable saving.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LO2Loader on a local LO2 log directory and inspect the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="Path to the root directory that contains the LO2 runs.")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to scan before stopping.")
    parser.add_argument(
        "--errors-per-run",
        type=int,
        default=1,
        help="How many error directories to sample per run (ignored when single-error-type is set).",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Permit the same error type to appear across multiple runs (matches LO2Loader default).",
    )
    parser.add_argument(
        "--single-error-type",
        default="",
        help="Force using exactly this error directory name on every run. Use 'random' to pick one automatically.",
    )
    parser.add_argument(
        "--single-service",
        default="",
        help="Only load log files whose name contains oauth2-oauth2-<service>. Leave empty to use the Loader default.",
    )
    parser.add_argument(
        "--service-types",
        nargs="+",
        choices=["client", "code", "key", "refresh-token", "service", "token", "user"],
        help="Filter log files to the given service types. Overrides --single-service when provided.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to show from the event and sequence dataframes.",
    )
    parser.add_argument(
        "--load-metrics",
        action="store_true",
        help="Attempt to load metrics from metrics/*.json inside each run/test-case directory.",
    )
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="Save loader.df and loader.df_seq to Parquet files in the specified --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/lo2",
        help="Directory where Parquet files should be written when --save-parquet is enabled.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    single_error = args.single_error_type if args.single_error_type else None

    if args.single_service and args.service_types:
        raise SystemExit("Cannot use --single-service together with --service-types. Pick one of the options.")

    loader = LO2Loader(
        filename=str(root),
        n_runs=args.runs,
        errors_per_run=args.errors_per_run,
        dup_errors=args.allow_duplicates or bool(single_error),
        single_error_type=single_error,
        single_service=args.single_service,
        service_types=args.service_types,
    )

    loader.execute()

    if args.load_metrics:
        loader.load_metrics()

    print("\n=== Event-level dataframe (loader.df) ===")
    print(f"Rows: {len(loader.df)} | Columns: {loader.df.columns}")
    print(loader.df.head(args.head))

    if loader.df_seq is not None:
        print("\n=== Sequence-level dataframe (loader.df_seq) ===")
        print(f"Rows: {len(loader.df_seq)} | Columns: {loader.df_seq.columns}")
        print(loader.df_seq.head(args.head))
    else:
        print("\nSequence dataframe not generated (loader.df_seq is None).")

    if args.load_metrics:
        if getattr(loader, "metrics_df", None) is not None:
            print("\n=== Metrics dataframe (loader.metrics_df) ===")
            print(f"Rows: {len(loader.metrics_df)} | Columns: {loader.metrics_df.columns}")
            print(loader.metrics_df.head(args.head))
        else:
            print("\nNo metrics were found under metrics/*.json directories.")

    # Provide a quick glimpse at unique cases to verify sampling behaviour.
    print("\nRuns present:", loader.df.select(pl.col("run").unique()).to_series().to_list())
    print("Test cases present:", loader.df.select(pl.col("test_case").unique()).to_series().to_list())
    print("Services present:", loader.df.select(pl.col("service").unique()).to_series().to_list())

    if args.save_parquet:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        events_path = out_dir / "lo2_events.parquet"
        loader.df.write_parquet(events_path)
        print(f"\nSaved loader.df to {events_path}")
        if loader.df_seq is not None:
            seq_path = out_dir / "lo2_sequences.parquet"
            loader.df_seq.write_parquet(seq_path)
            print(f"Saved loader.df_seq to {seq_path}")
        if args.load_metrics and getattr(loader, "metrics_df", None) is not None:
            metrics_path = out_dir / "lo2_metrics.parquet"
            loader.metrics_df.write_parquet(metrics_path)
            print(f"Saved loader.metrics_df to {metrics_path}")


if __name__ == "__main__":
    main()
