#!/usr/bin/env python3
"""Convert the five NeurIPS 2018 AutoML lifelong tasks for FuxiCTR.

The archive exposes five labelled chronological batches per task:
train1, test1, test2, test3 and test4.  Later test5--test9 batches do not
contain labels and are intentionally excluded.  The output protocol is:

  train = train1 + test1 + test2
  valid = test3
  test  = test4

Only Numerical and Categorical fields are retained.  Time and Multi-value
fields are excluded, matching the field counts used by numerical-embedding
papers (task A: 23+51, task E: 6+25).  Categorical vocabularies and numeric
statistics are fitted on the training split only.  Outputs are encoded
Parquet files plus feature_map.json, so FuxiCTR can load them directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TASK_PREFIX = {"A": "AA", "B": "B", "C": "C", "D": "D", "E": "E"}
SPLIT_BATCHES = {
    "train": ("train1", "test1", "test2"),
    "valid": ("test3",),
    "test": ("test4",),
}
KEEP_TYPES = {"Numerical", "Categorical"}
MISSING_STRINGS = {"", "NaN", "nan", "NA", "N/A", "?", "null", "None"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--tasks", default="A,B,C,D,E")
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--max-rows-per-batch",
        type=int,
        default=None,
        help="Smoke-test only: truncate each source batch.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def fingerprint(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_types(path: Path) -> List[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not values:
        raise ValueError(f"Empty feature type file: {path}")
    return values


def build_columns(types: Sequence[str]) -> Tuple[List[str], List[int], Dict[str, str], List[Dict[str, object]]]:
    counters = {"Numerical": 0, "Categorical": 0}
    all_columns = [f"raw_{i + 1}" for i in range(len(types))]
    selected_indices: List[int] = []
    selected_types: Dict[str, str] = {}
    original_fields: List[Dict[str, object]] = []
    for index, feature_type in enumerate(types):
        kept = feature_type in KEEP_TYPES
        output_name = None
        if kept:
            counters[feature_type] += 1
            prefix = "I" if feature_type == "Numerical" else "C"
            output_name = f"{prefix}{counters[feature_type]}"
            selected_indices.append(index)
            selected_types[output_name] = feature_type
        original_fields.append(
            {
                "original_index": index,
                "original_type": feature_type,
                "kept": kept,
                "output_name": output_name,
            }
        )
    return all_columns, selected_indices, selected_types, original_fields


def source_paths(source_root: Path, prefix: str, batch: str) -> Tuple[Path, Path]:
    task_dir = source_root / prefix
    return task_dir / f"{prefix}_{batch}.data", task_dir / f"{prefix}_{batch}.solution"


def data_chunks(
    path: Path,
    all_columns: Sequence[str],
    selected_indices: Sequence[int],
    chunk_rows: int,
    max_rows: int | None,
) -> Iterator[pd.DataFrame]:
    selected_names = [all_columns[i] for i in selected_indices]
    yield from pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=list(all_columns),
        usecols=selected_names,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_rows,
        nrows=max_rows,
        engine="c",
        on_bad_lines="error",
    )


def label_chunks(path: Path, chunk_rows: int, max_rows: int | None) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["Label"],
        usecols=[0],
        dtype=np.int8,
        chunksize=chunk_rows,
        nrows=max_rows,
        engine="c",
        on_bad_lines="error",
    )


def paired_chunks(
    data_path: Path,
    label_path: Path,
    all_columns: Sequence[str],
    selected_indices: Sequence[int],
    chunk_rows: int,
    max_rows: int | None,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    data_iter = data_chunks(data_path, all_columns, selected_indices, chunk_rows, max_rows)
    label_iter = label_chunks(label_path, chunk_rows, max_rows)
    for chunk_index, pair in enumerate(zip_longest(data_iter, label_iter), start=1):
        frame, labels = pair
        if frame is None or labels is None:
            raise ValueError(f"Chunk count mismatch: {data_path} vs {label_path}")
        if len(frame) != len(labels):
            raise ValueError(
                f"Row mismatch in chunk {chunk_index}: {data_path}={len(frame)}, "
                f"{label_path}={len(labels)}"
            )
        yield frame, labels


def is_missing(series: pd.Series) -> pd.Series:
    return series.isin(MISSING_STRINGS)


def fit_vocabularies(
    source_root: Path,
    prefix: str,
    all_columns: Sequence[str],
    selected_indices: Sequence[int],
    selected_types: Mapping[str, str],
    chunk_rows: int,
    max_rows: int | None,
) -> Dict[str, Dict[str, int]]:
    selected_names = list(selected_types)
    raw_to_output = dict(zip([all_columns[i] for i in selected_indices], selected_names))
    categorical_raw = [
        raw for raw, output in raw_to_output.items() if selected_types[output] == "Categorical"
    ]
    vocabularies: Dict[str, Dict[str, int]] = {
        raw_to_output[raw]: {"__PAD__": 0} for raw in categorical_raw
    }
    for batch in SPLIT_BATCHES["train"]:
        data_path, _ = source_paths(source_root, prefix, batch)
        for frame in data_chunks(data_path, all_columns, selected_indices, chunk_rows, max_rows):
            for raw_name in categorical_raw:
                output_name = raw_to_output[raw_name]
                vocabulary = vocabularies[output_name]
                values = frame[raw_name]
                unique_values = pd.unique(values[~is_missing(values)])
                for token in unique_values:
                    token = str(token)
                    if token not in vocabulary:
                        vocabulary[token] = len(vocabulary)
            del frame
    return vocabularies


class NumericStats:
    def __init__(self, names: Iterable[str], sample_stride: int, n_bins: int) -> None:
        self.names = list(names)
        self.sample_stride = sample_stride
        self.n_bins = n_bins
        self.state = {
            name: {
                "count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "min": math.inf,
                "max": -math.inf,
                "samples": [],
                "log_squared_tokens": set(),
            }
            for name in self.names
        }

    def update(self, frame: pd.DataFrame) -> None:
        for name in self.names:
            values = frame[name].to_numpy(dtype=np.float64, copy=False)
            state = self.state[name]
            state["count"] += int(values.size)
            state["sum"] += float(values.sum(dtype=np.float64))
            state["sumsq"] += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
            state["min"] = min(state["min"], float(values.min()))
            state["max"] = max(state["max"], float(values.max()))
            state["samples"].append(values[:: self.sample_stride].copy())
            integer = np.trunc(
                np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            ).astype(np.int64)
            transformed = np.where(
                integer > 2,
                np.floor(np.square(np.log(np.maximum(integer, 1)))).astype(np.int64),
                integer,
            )
            state["log_squared_tokens"].update(
                int(token) for token in np.unique(transformed) if token > 0
            )

    def log_squared_vocabularies(self) -> Dict[str, Dict[str, int]]:
        vocabularies: Dict[str, Dict[str, int]] = {}
        for name, state in self.state.items():
            vocabulary = {"__PAD__": 0, "__OOV__": 1}
            for token in sorted(state["log_squared_tokens"]):
                vocabulary[str(token)] = len(vocabulary)
            vocabularies[name] = vocabulary
        return vocabularies

    def finalize(self) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for name, state in self.state.items():
            count = int(state["count"])
            mean = float(state["sum"] / max(count, 1))
            variance = max(float(state["sumsq"] / max(count, 1) - mean * mean), 0.0)
            samples = np.concatenate(state["samples"]) if state["samples"] else np.array([0.0])
            quantiles = np.quantile(samples, np.linspace(0.0, 1.0, self.n_bins + 1))
            edges = np.unique(quantiles.astype(np.float64)).tolist()
            if len(edges) == 1:
                edges = [edges[0], edges[0] + 1.0]
            result[name] = {
                "count": count,
                "mean": mean,
                "std": math.sqrt(variance),
                "min": float(state["min"]),
                "max": float(state["max"]),
                "bin_edges": edges,
                "sample_count": int(samples.size),
            }
        return result


def encode_frame(
    raw_frame: pd.DataFrame,
    labels: pd.DataFrame,
    all_columns: Sequence[str],
    selected_indices: Sequence[int],
    selected_types: Mapping[str, str],
    vocabularies: MutableMapping[str, Dict[str, int]],
    fit_vocabulary: bool,
) -> pd.DataFrame:
    output: Dict[str, pd.Series] = {}
    selected_names = list(selected_types)
    for raw_name, output_name in zip([all_columns[i] for i in selected_indices], selected_names):
        values = raw_frame[raw_name]
        if selected_types[output_name] == "Numerical":
            output[output_name] = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(np.float32)
        else:
            vocabulary = vocabularies[output_name]
            if fit_vocabulary:
                unique_values = pd.unique(values[~is_missing(values)])
                for token in unique_values:
                    token = str(token)
                    if token not in vocabulary:
                        vocabulary[token] = len(vocabulary)
            output[output_name] = values.map(vocabulary).fillna(0).astype(np.int64)
    encoded = pd.DataFrame(output, copy=False)
    encoded["Label"] = labels["Label"].astype(np.float32).to_numpy(copy=False)
    return encoded


def transform_split(
    split: str,
    output_path: Path,
    source_root: Path,
    prefix: str,
    all_columns: Sequence[str],
    selected_indices: Sequence[int],
    selected_types: Mapping[str, str],
    vocabularies: MutableMapping[str, Dict[str, int]],
    chunk_rows: int,
    max_rows: int | None,
    compression: str,
    stats: NumericStats | None,
    fit_vocabulary: bool,
) -> Dict[str, object]:
    writer: pq.ParquetWriter | None = None
    rows = 0
    positives = 0
    source_counts: Dict[str, int] = {}
    try:
        for batch in SPLIT_BATCHES[split]:
            data_path, label_path = source_paths(source_root, prefix, batch)
            batch_rows = 0
            for raw_frame, labels in paired_chunks(
                data_path, label_path, all_columns, selected_indices, chunk_rows, max_rows
            ):
                encoded = encode_frame(
                    raw_frame,
                    labels,
                    all_columns,
                    selected_indices,
                    selected_types,
                    vocabularies,
                    fit_vocabulary,
                )
                if stats is not None:
                    stats.update(encoded)
                table = pa.Table.from_pandas(encoded, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                writer.write_table(table)
                chunk_count = len(encoded)
                rows += chunk_count
                batch_rows += chunk_count
                positives += int(labels["Label"].sum())
                del raw_frame, labels, encoded, table
            source_counts[batch] = batch_rows
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise ValueError(f"No rows written for {prefix}/{split}")
    return {
        "rows": rows,
        "positives": positives,
        "positive_rate": positives / rows,
        "source_batches": source_counts,
    }


def build_feature_map(
    dataset_id: str,
    selected_types: Mapping[str, str],
    vocabularies: Mapping[str, Mapping[str, int]],
    numeric_stats: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    features = []
    total_features = 0
    for name, feature_type in selected_types.items():
        if feature_type == "Numerical":
            stat = numeric_stats[name]
            features.append(
                {
                    name: {
                        "source": "dense",
                        "type": "numeric",
                        "numeric_pre_transform": "identity",
                        "numeric_missing_value": 0.0,
                        "numeric_storage_dtype": "float32",
                        "numeric_mean": stat["mean"],
                        "numeric_std": stat["std"],
                        "numeric_min": stat["min"],
                        "numeric_max": stat["max"],
                        "numeric_bin_edges": stat["bin_edges"],
                    }
                }
            )
            total_features += 1
        else:
            vocab_size = len(vocabularies[name])
            features.append(
                {name: {"source": "sparse", "type": "categorical", "padding_idx": 0, "vocab_size": vocab_size}}
            )
            total_features += vocab_size
    return {
        "dataset_id": dataset_id,
        "num_fields": len(selected_types),
        "total_features": total_features,
        "input_length": len(selected_types),
        "labels": ["Label"],
        "features": features,
    }


def save_vocabularies(root: Path, vocabularies: Mapping[str, Mapping[str, int]]) -> None:
    vocab_root = root / "vocab"
    vocab_root.mkdir()
    for name, vocabulary in vocabularies.items():
        table = pa.table(
            {
                "token": pa.array(list(vocabulary.keys()), type=pa.string()),
                "id": pa.array(list(vocabulary.values()), type=pa.int64()),
            }
        )
        pq.write_table(table, vocab_root / f"{name}.parquet", compression="zstd")


def prepare_task(task: str, args: argparse.Namespace) -> Dict[str, object]:
    prefix = TASK_PREFIX[task]
    dataset_id = f"AutoML_{task}"
    type_path = args.source_root / prefix / f"{prefix}_feat.type"
    types = read_types(type_path)
    all_columns, selected_indices, selected_types, original_fields = build_columns(types)
    info_text = (args.source_root / prefix / f"{prefix}_public.info").read_text()
    expected = int(info_text.split("feat_num =")[1].splitlines()[0])
    if expected != len(types):
        raise ValueError(f"Feature count mismatch for {task}: info={expected}, types={len(types)}")

    target = args.output_root / dataset_id
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {target}")
    staging = args.output_root / f".{dataset_id}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory exists: {staging}")
    staging.mkdir(parents=True)

    categorical_names = [
        name for name, kind in selected_types.items() if kind == "Categorical"
    ]
    vocabularies: Dict[str, Dict[str, int]] = {
        name: {"__PAD__": 0} for name in categorical_names
    }
    print(f"[{task}] fitting vocabularies while writing train.parquet", flush=True)
    numeric_names = [name for name, kind in selected_types.items() if kind == "Numerical"]
    stats_builder = NumericStats(numeric_names, args.sample_stride, args.n_bins)
    split_stats: Dict[str, Dict[str, object]] = {}
    try:
        for split in ("train", "valid", "test"):
            print(f"[{task}] writing {split}.parquet", flush=True)
            split_stats[split] = transform_split(
                split,
                staging / f"{split}.parquet",
                args.source_root,
                prefix,
                all_columns,
                selected_indices,
                selected_types,
                vocabularies,
                args.chunk_rows,
                args.max_rows_per_batch,
                args.compression,
                stats_builder if split == "train" else None,
                split == "train",
            )
        numeric_stats = stats_builder.finalize()
        feature_map = build_feature_map(
            dataset_id, selected_types, vocabularies, numeric_stats
        )
        save_vocabularies(staging, vocabularies)
        write_json(staging / "feature_map.json", feature_map)
        write_json(staging / "numeric_stats.json", numeric_stats)
        write_json(
            staging / "log_squared_bucket_vocab.json",
            stats_builder.log_squared_vocabularies(),
        )
        source_files = {}
        for split, batches in SPLIT_BATCHES.items():
            for batch in batches:
                data_path, label_path = source_paths(args.source_root, prefix, batch)
                source_files[f"{batch}.data"] = fingerprint(data_path)
                source_files[f"{batch}.solution"] = fingerprint(label_path)
        manifest = {
            "dataset_id": dataset_id,
            "task": task,
            "source_prefix": prefix,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(args.source_root.resolve()),
            "output_root": str(target.resolve()),
            "split_policy": SPLIT_BATCHES,
            "unlabelled_batches_excluded": [
                "test5",
                "test6",
                "test7",
                "test8",
                "test9",
            ],
            "feature_policy": {
                "kept_types": sorted(KEEP_TYPES),
                "dropped_types": sorted(set(types) - KEEP_TYPES),
                "categorical_encoding": (
                    "exact train-only first-occurrence vocabulary; OOV/missing=0"
                ),
                "numeric_encoding": (
                    "raw float32 identity; missing=0; train-only statistics"
                ),
                "log_squared_bucket_vocabulary": (
                    "train-only positive transformed tokens; PAD=0; OOV=1"
                ),
            },
            "original_feature_count": len(types),
            "kept_feature_count": len(selected_types),
            "numerical_feature_count": sum(
                value == "Numerical" for value in selected_types.values()
            ),
            "categorical_feature_count": sum(
                value == "Categorical" for value in selected_types.values()
            ),
            "original_fields": original_fields,
            "split_statistics": split_stats,
            "categorical_vocab_sizes": {
                name: len(vocabulary)
                for name, vocabulary in vocabularies.items()
            },
            "source_files": source_files,
            "converter": fingerprint(Path(__file__)),
        }
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        print(f"[{task}] complete: {target}", flush=True)
        return manifest
    except Exception:
        print(f"[{task}] failed; staging retained at {staging}", flush=True)
        raise
    finally:
        del vocabularies
        gc.collect()


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.chunk_rows <= 0 or args.sample_stride <= 0 or args.n_bins <= 0:
        raise ValueError("chunk-rows, sample-stride and n-bins must be positive")
    tasks = [
        item.strip().upper()
        for item in args.tasks.split(",")
        if item.strip()
    ]
    unknown = sorted(set(tasks) - set(TASK_PREFIX))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = [prepare_task(task, args) for task in tasks]
    write_json(
        args.output_root / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": [item["dataset_id"] for item in manifests],
            "split_policy": SPLIT_BATCHES,
        },
    )


if __name__ == "__main__":
    main()
