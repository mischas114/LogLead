import argparse
import os
import time
from pathlib import Path

import polars as pl

from loglead.loaders import LO2Loader
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer

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
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of runs to scan before stopping (None oder ≤0 = alle).",
    )
    parser.add_argument(
        "--errors-per-run",
        type=int,
        default=None,
        help="How many error directories to include per run (None oder ≤0 = alle).",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        dest="allow_duplicates",
        default=True,
        help="Permit the same error type to appear across multiple runs.",
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_false",
        dest="allow_duplicates",
        help="Disable duplicate error sampling across runs.",
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
        help="Persist sequence tables (lo2_sequences*.parquet) in the specified --output-dir.",
    )
    parser.add_argument(
        "--save-events",
        action="store_true",
        help="Persist the event-level table to Parquet (disabled by default to reduce footprint).",
    )
    parser.add_argument(
        "--save-base-sequences",
        action="store_true",
        help="Persist the raw sequence table (lo2_sequences.parquet) alongside the enhanced export.",
    )
    parser.add_argument(
        "--enhancer-run-batch",
        type=int,
        default=0,
        help="Optional Anzahl von Run-IDs pro Batch für das Feature-Engineering. 0 = alles auf einmal.",
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
    UNLIMITED_SENTINEL = 10**9
    runs_limit = args.runs if args.runs and args.runs > 0 else UNLIMITED_SENTINEL
    errors_limit = args.errors_per_run if args.errors_per_run and args.errors_per_run > 0 else UNLIMITED_SENTINEL
    print(f"[Loader] Run limit: {'unlimited' if runs_limit >= UNLIMITED_SENTINEL else runs_limit}")
    print(
        f"[Loader] Errors per run limit: {'unlimited' if errors_limit >= UNLIMITED_SENTINEL else errors_limit}"
    )

    loader = LO2Loader(
        filename=str(root),
        n_runs=runs_limit,
        errors_per_run=errors_limit,
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

    total_runs_available = sum(1 for entry in root.iterdir() if entry.is_dir())
    runs_loaded = (
        int(loader.df.select(pl.col("run").n_unique()).item()) if loader.df is not None and len(loader.df) else 0
    )
    total_events = len(loader.df) if loader.df is not None else 0
    downsampling_applied = False
    if args.runs and args.runs > 0 and runs_limit < total_runs_available:
        downsampling_applied = True
    if args.errors_per_run and args.errors_per_run > 0:
        downsampling_applied = True

    print("\n[Summary] Loader diagnostics:")
    print(
        f"  runs_available={total_runs_available} | runs_loaded={runs_loaded} | run_limit="
        f"{'unlimited' if runs_limit >= UNLIMITED_SENTINEL else runs_limit}"
    )
    print(
        f"  errors_per_run_limit={'unlimited' if errors_limit >= UNLIMITED_SENTINEL else errors_limit} "
        f"| allow_duplicates={args.allow_duplicates}"
    )
    print(f"  total_events_loaded={total_events}")
    print(f"[Summary] Downsampling occurred: {'yes' if downsampling_applied else 'no'}")

    enhanced_seq_path: Path | None = None
    enhanced_preview: pl.DataFrame | None = None
    enhanced_seq: pl.DataFrame | None = None
    if args.save_parquet and loader.df_seq is not None and len(loader.df_seq):
        print("\n[Enhance] Aggregating sequence features (words/trigrams/timestamps)...")

        def _enhance_batch(events_batch: pl.DataFrame, seq_batch: pl.DataFrame) -> pl.DataFrame:
            def _timed_step(label: str, func, *, soft_fail: bool = False):
                start = time.perf_counter()
                try:
                    result = func()
                except Exception as exc:
                    level = "WARNING" if soft_fail else "ERROR"
                    print(f"[Enhance][{level}] {label} failed after {time.perf_counter() - start:.2f}s: {exc}")
                    if soft_fail:
                        return None
                    raise
                else:
                    print(f"[Enhance] {label} finished in {time.perf_counter() - start:.2f}s")
                    return result

            enhancer = EventLogEnhancer(events_batch)
            events_batch = _timed_step("EventLogEnhancer.normalize", enhancer.normalize)
            events_batch = _timed_step("EventLogEnhancer.words", enhancer.words)
            events_batch = _timed_step("EventLogEnhancer.trigrams", enhancer.trigrams)
            parsed = _timed_step("EventLogEnhancer.parse_drain", enhancer.parse_drain, soft_fail=True)
            if parsed is not None:
                events_batch = parsed
            events_batch = _timed_step("EventLogEnhancer.length", enhancer.length)

            seq_enhancer_local = SequenceEnhancer(df=events_batch, df_seq=seq_batch)
            seq_batch = _timed_step("SequenceEnhancer.seq_len", seq_enhancer_local.seq_len)
            if "start_time" not in seq_batch.columns:
                seq_batch = _timed_step("SequenceEnhancer.start_time", seq_enhancer_local.start_time)
            seq_batch = _timed_step("SequenceEnhancer.duration", seq_enhancer_local.duration)
            seq_batch = _timed_step("SequenceEnhancer.tokens(e_words)", lambda: seq_enhancer_local.tokens(token="e_words"))
            seq_batch = _timed_step("SequenceEnhancer.tokens(e_trigrams)", lambda: seq_enhancer_local.tokens(token="e_trigrams"))
            if "e_event_drain_id" in events_batch.columns:
                events_enhanced = _timed_step(
                    "SequenceEnhancer.events(e_event_drain_id)",
                    lambda: seq_enhancer_local.events("e_event_drain_id"),
                    soft_fail=True,
                )
                if events_enhanced is not None:
                    seq_batch = events_enhanced
            if "start_time_right" in seq_batch.columns:
                seq_batch = seq_batch.drop("start_time_right")

            test_case_expr = pl.col("seq_id").str.split("__").list.get(1)
            if "test_case" not in seq_batch.columns:
                seq_batch = seq_batch.with_columns(test_case_expr.alias("test_case"))
            else:
                seq_batch = seq_batch.with_columns(
                    pl.col("test_case").fill_null(test_case_expr).alias("test_case")
                )

            required_columns = {"seq_id", "start_time", "end_time", "duration"}
            missing = required_columns - set(seq_batch.columns)
            if missing:
                raise ValueError(f"Missing required sequence columns after enhancement: {sorted(missing)}")

            print(
                f"[Enhance] Batch ready with columns={len(seq_batch.columns)} rows={seq_batch.height}"
            )
            return seq_batch

        def _validate_parquet(path: Path) -> None:
            print(f"[Enhance] Validating parquet output: {path}")
            if not path.exists():
                print(f"[Enhance][WARNING] Validation skipped; file not found: {path}")
                return
            try:
                import pyarrow.parquet as pq
            except Exception as exc:
                print(f"[Enhance][WARNING] Could not import pyarrow for validation: {exc}")
                return

            try:
                metadata = pq.read_metadata(str(path))
            except Exception as exc:
                print(f"[Enhance][ERROR] Validation failed for {path}: {exc}")
                raise
            else:
                print(
                    f"[Enhance] Validation OK: rows={metadata.num_rows} row_groups={metadata.num_row_groups}"
                )

        if args.enhancer_run_batch and args.enhancer_run_batch > 0:
            run_list = loader.df.select(pl.col("run").unique()).to_series().to_list()
            batches = [
                run_list[idx : idx + args.enhancer_run_batch]
                for idx in range(0, len(run_list), args.enhancer_run_batch)
            ]
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            writer = None
            target_path = out_dir / "lo2_sequences_enhanced.parquet"
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError as exc:
                    print(f"[Enhance] Could not remove existing {target_path}: {exc}")
            try:
                for batch_idx, run_subset in enumerate(batches, start=1):
                    print(f"[Enhance] Batch {batch_idx}/{len(batches)} (runs={len(run_subset)})")
                    events_batch = loader.df.filter(pl.col("run").is_in(run_subset))
                    if events_batch.is_empty():
                        continue
                    seq_ids = events_batch.select(pl.col("seq_id").unique()).to_series().to_list()
                    seq_batch = loader.df_seq.filter(pl.col("seq_id").is_in(seq_ids))
                    if seq_batch.is_empty():
                        del events_batch, seq_ids, seq_batch
                        continue
                    enhanced_batch = _enhance_batch(events_batch, seq_batch)
                    if enhanced_preview is None:
                        enhanced_preview = enhanced_batch.head(5)
                    table = enhanced_batch.to_arrow()
                    if writer is None:
                        import pyarrow.parquet as pq

                        writer = pq.ParquetWriter(
                            str(target_path),
                            table.schema,
                            compression="zstd",
                        )
                        enhanced_seq_path = target_path
                    writer.write_table(table)
                    del events_batch, seq_ids, seq_batch, enhanced_batch, table
            finally:
                if writer is not None:
                    writer.close()
                else:
                    enhanced_seq_path = None
            if enhanced_seq_path:
                _validate_parquet(enhanced_seq_path)
        else:
            event_df = loader.df.clone() if args.save_events else loader.df
            enhanced_seq = _enhance_batch(event_df, loader.df_seq.clone())
            if not args.save_events:
                del event_df
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            enhanced_seq_path = out_dir / "lo2_sequences_enhanced.parquet"
            enhanced_seq.write_parquet(enhanced_seq_path)
            enhanced_preview = enhanced_seq.head(5)
            enhanced_seq = None
            _validate_parquet(enhanced_seq_path)

        if enhanced_seq_path and enhanced_preview is not None:
            print(
                f"[Enhance] Done. Columns include: {enhanced_preview.columns}"
            )
        elif enhanced_seq_path:
            print("[Enhance] Done. Enhanced sequences written to disk.")
    elif not args.save_parquet:
        print("[Enhance] Skipped (save_parquet not set).")
    else:
        print("[Enhance] Sequence table empty; skipping aggregation.")

    if args.save_parquet:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.save_events:
            events_path = out_dir / "lo2_events.parquet"
            loader.df.write_parquet(events_path)
            print(f"\nSaved loader.df to {events_path}")
        else:
            print("\nSkipping event-level Parquet export (use --save-events to enable).")

        if loader.df_seq is not None:
            if args.save_base_sequences:
                seq_path = out_dir / "lo2_sequences.parquet"
                loader.df_seq.write_parquet(seq_path)
                print(f"Saved base sequences to {seq_path}")
            else:
                print("Skipping base sequence export (use --save-base-sequences to enable).")

        if args.load_metrics and getattr(loader, "metrics_df", None) is not None:
            metrics_path = out_dir / "lo2_metrics.parquet"
            loader.metrics_df.write_parquet(metrics_path)
            print(f"Saved loader.metrics_df to {metrics_path}")

        if enhanced_seq_path:
            print(f"Enhanced sequences gespeichert unter {enhanced_seq_path}")
            if enhanced_preview is not None:
                print(enhanced_preview)
        else:
            print("No enhanced sequences were produced.")


if __name__ == "__main__":
    main()
