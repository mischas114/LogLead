#!/usr/bin/env python3
"""
Merge loader batch outputs into a single Parquet without loading everything into RAM.

This script opens each `_lo2_seq_batch_*.parquet` sequentially, streams the data
into a single ParquetWriter, and (optionally) removes the batch files afterwards.

Usage:
    python tools/merge_sequence_batches.py \
        --input-dir demo/result/lo2 \
        --output demo/result/lo2/lo2_sequences_enhanced.parquet \
        [--cleanup]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge loader batch Parquet files into a single enhanced parquet.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing `_lo2_seq_batch_*.parquet` files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination parquet path (e.g. demo/result/lo2/lo2_sequences_enhanced.parquet).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove batch files after successful merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    batch_files = sorted(input_dir.glob("_lo2_seq_batch_*.parquet"))
    if not batch_files:
        raise SystemExit(f"No batch parquet files found in {input_dir}")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[merge] Found {len(batch_files)} batch files. Writing to {output_path}")

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for idx, batch_path in enumerate(batch_files, start=1):
        print(f"[merge] Processing {batch_path.name} ({idx}/{len(batch_files)})")
        df = pl.read_parquet(batch_path)
        total_rows += df.height
        table = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        writer.write_table(table)
        del df, table

    if writer is not None:
        writer.close()

    print(f"[merge] Done. Total rows: {total_rows}")
    preview = pl.read_parquet(output_path).head(5)
    print("[merge] Preview:")
    print(preview)

    if args.cleanup:
        removed = 0
        for file_path in batch_files:
            try:
                file_path.unlink()
                removed += 1
            except OSError as exc:
                print(f"[merge] Could not delete {file_path}: {exc}")
        print(f"[merge] Removed {removed}/{len(batch_files)} batch files.")


if __name__ == "__main__":
    main()
