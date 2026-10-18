#!/usr/bin/env python3
"""Build Criteo_x4 with raw numerical fields for NumEmb experiments.

I1..I13 are read from the original CSV files, missing values are filled with
zero, and values are stored as Int64 without log, bucketing, clipping, or
standardization. C1..C26 and Label are copied row-for-row from the validated
low-frequency-filtered, non-hashed Criteo_x4 v2.2 semantics Parquet dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .criteo_reference import (
    CATEGORICAL,
    NUMERIC,
    SPLITS,
    ExactReader,
    read_json,
    write_json,
)


class RawNumericStats:
    def __init__(self, sample_stride: int, n_bins: int):
        self.sample_stride = sample_stride
        self.n_bins = n_bins
        width = len(NUMERIC)
        self.count = 0
        self.sums = np.zeros(width, dtype=np.float64)
        self.sums_sq = np.zeros(width, dtype=np.float64)
        self.minima = np.full(width, np.inf, dtype=np.float64)
        self.maxima = np.full(width, -np.inf, dtype=np.float64)
        self.samples = []

    def update(self, frame: pl.DataFrame) -> None:
        values = frame.select(NUMERIC).to_numpy().astype(np.float64, copy=False)
        self.sums += values.sum(axis=0, dtype=np.float64)
        self.sums_sq += np.square(values).sum(axis=0, dtype=np.float64)
        self.minima = np.minimum(self.minima, values.min(axis=0))
        self.maxima = np.maximum(self.maxima, values.max(axis=0))
        indices = np.arange(values.shape[0], dtype=np.int64) + self.count
        selected = values[indices % self.sample_stride == 0]
        if selected.size:
            self.samples.append(selected)
        self.count += values.shape[0]

    def finalize(self) -> Dict:
        if not self.count:
            raise RuntimeError("Cannot finalize empty training statistics")
        sample = np.concatenate(self.samples, axis=0)
        means = self.sums / self.count
        variances = np.maximum(
            self.sums_sq / self.count - np.square(means),
            0.0,
        )
        stds = np.sqrt(variances)
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        features = {}
        for index, name in enumerate(NUMERIC):
            edges = np.unique(
                np.quantile(sample[:, index], quantiles, method="linear")
            )
            edges[0] = self.minima[index]
            edges[-1] = self.maxima[index]
            if edges.size < 2:
                value = float(edges[0])
                edges = np.asarray([value - 1e-6, value + 1e-6])
            features[name] = {
                "mean": float(means[index]),
                "std": float(max(stds[index], 1e-12)),
                "min": int(self.minima[index]),
                "max": int(self.maxima[index]),
                "raw_bin_edges": edges.tolist(),
                "effective_bins": int(edges.size - 1),
            }
        return {
            "train_rows": self.count,
            "sample_stride": self.sample_stride,
            "quantile_sample_rows": int(sample.shape[0]),
            "n_bins_requested": self.n_bins,
            "numeric_transform": "identity",
            "missing_value": 0,
            "storage_dtype": "int64",
            "features": features,
        }


def transform_split(
    split: str,
    raw_root: Path,
    reference_root: Path,
    output: Path,
    chunk_rows: int,
    compression: str,
    stats: RawNumericStats | None,
) -> Dict:
    reference_path = reference_root / f"{split}.parquet"
    parquet = pq.ParquetFile(reference_path)
    reader = ExactReader(raw_root / f"{split}.csv", chunk_rows)
    writer = None
    rows = 0
    try:
        for group in range(parquet.num_row_groups):
            sparse = pl.from_arrow(
                parquet.read_row_group(
                    group,
                    columns=CATEGORICAL + ["Label"],
                )
            )
            raw = reader.take(sparse.height).select(
                [
                    pl.col(name).cast(pl.Int64).alias(name)
                    for name in NUMERIC
                ]
            )
            if stats is not None:
                stats.update(raw)
            frame = raw.hstack(sparse)
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    output,
                    table.schema,
                    compression=compression,
                    write_statistics=True,
                    use_dictionary=CATEGORICAL,
                )
            writer.write_table(table, row_group_size=frame.height)
            rows += frame.height
            print(
                f"[{split}] row_group={group:03d}/{parquet.num_row_groups - 1:03d} "
                f"rows={rows:,}",
                flush=True,
            )
        reader.assert_exhausted()
    finally:
        if writer is not None:
            writer.close()
    result = pq.ParquetFile(output)
    if result.metadata.num_rows != rows:
        raise RuntimeError(
            f"{split} row mismatch: metadata={result.metadata.num_rows}, wrote={rows}"
        )
    return {
        "rows": rows,
        "row_groups": result.num_row_groups,
        "bytes": output.stat().st_size,
    }


def build_metadata(
    reference_map: Mapping,
    reference_vocab: Mapping,
    dataset_id: str,
    stats: Mapping,
):
    categorical_specs = {}
    for item in reference_map["features"]:
        name, spec = next(iter(item.items()))
        if name in CATEGORICAL:
            categorical_specs[name] = copy.deepcopy(spec)

    features = []
    for name in NUMERIC:
        numeric = stats["features"][name]
        features.append(
            {
                name: {
                    "source": "dense",
                    "type": "numeric",
                    "numeric_pre_transform": "identity",
                    "numeric_missing_value": 0,
                    "numeric_storage_dtype": "int64",
                    "numeric_mean": numeric["mean"],
                    "numeric_std": numeric["std"],
                    "numeric_min": numeric["min"],
                    "numeric_max": numeric["max"],
                    "numeric_bin_edges": numeric["raw_bin_edges"],
                }
            }
        )
    for name in CATEGORICAL:
        spec = categorical_specs[name]
        spec["source"] = "sparse"
        features.append({name: spec})

    categorical_total = sum(
        int(categorical_specs[name]["vocab_size"])
        for name in CATEGORICAL
    )
    feature_map = {
        "dataset_id": dataset_id,
        "num_fields": len(NUMERIC) + len(CATEGORICAL),
        "total_features": len(NUMERIC) + categorical_total,
        "input_length": len(NUMERIC) + len(CATEGORICAL),
        "labels": list(reference_map["labels"]),
        "features": features,
    }
    feature_vocab = {
        name: copy.deepcopy(reference_vocab[name])
        for name in CATEGORICAL
    }
    return feature_map, feature_vocab


def fingerprint(path: Path) -> Dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-id", default="Criteo_x4_NumEmbRaw")
    parser.add_argument("--chunk-rows", type=int, default=1_048_576)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--compression", default="snappy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    reference = args.reference_root.resolve()
    target = args.output_root
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    if target.parent.resolve() != source:
        raise ValueError("--output-root must be a direct child of --source-root")

    raw_files = [source / f"{split}.csv" for split in SPLITS]
    reference_files = [reference / f"{split}.parquet" for split in SPLITS]
    required = raw_files + reference_files + [
        reference / "feature_map.json",
        reference / "feature_vocab.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")

    reference_map = read_json(reference / "feature_map.json")
    reference_vocab = read_json(reference / "feature_vocab.json")
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging output already exists: {staging}")
    staging.mkdir()

    stats_builder = RawNumericStats(args.sample_stride, args.n_bins)
    split_stats = {}
    try:
        for split in SPLITS:
            split_stats[split] = transform_split(
                split,
                source,
                reference,
                staging / f"{split}.parquet",
                args.chunk_rows,
                args.compression,
                stats_builder if split == "train" else None,
            )
        numeric_stats = stats_builder.finalize()
        feature_map, feature_vocab = build_metadata(
            reference_map,
            reference_vocab,
            args.dataset_id,
            numeric_stats,
        )
        write_json(staging / "numeric_stats.json", numeric_stats)
        write_json(staging / "feature_map.json", feature_map)
        write_json(staging / "feature_vocab.json", feature_vocab)
        write_json(
            staging / "manifest.json",
            {
                "dataset_id": args.dataset_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source),
                "reference_root": str(reference),
                "output_root": str(target),
                "preprocessing": {
                    "numerical_fields": NUMERIC,
                    "numeric_missing_value": 0,
                    "numeric_transform": "identity",
                    "numeric_storage_dtype": "int64",
                    "numeric_feature_type": "numeric",
                    "categorical_fields": CATEGORICAL,
                    "categorical_ids": (
                        "preserved from train-only min_count=10, non-hashed "
                        "reference Parquet"
                    ),
                    "labels": "preserved from reference Parquet",
                },
                "raw_source_files": {
                    path.stem: fingerprint(path)
                    for path in raw_files
                },
                "reference_files": {
                    path.stem: fingerprint(path)
                    for path in reference_files
                },
                "numeric_statistics": {
                    "fit_split": "train_only",
                    "sample_stride": args.sample_stride,
                    "n_bins_requested": args.n_bins,
                    "metadata_only": True,
                },
                "total_features": feature_map["total_features"],
                "splits": split_stats,
            },
        )
        if target.exists():
            raise FileExistsError(f"Output appeared during build: {target}")
        staging.rename(target)
    except Exception:
        print(f"BUILD_FAILED staging_preserved={staging}", flush=True)
        raise

    print(
        f"BUILD_OK output={target} total_features={feature_map['total_features']} "
        f"train_rows={numeric_stats['train_rows']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
