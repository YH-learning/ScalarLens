#!/usr/bin/env python3
"""Build an isolated Criteo_x4 copy with the executable FuxiCTR v2.2.0 semantics.

For I1..I13, missing values become 0 and:
    floor(log(x) ** 2) if x > 2 else int(x)

Labels and the 26 categorical IDs are preserved from the already validated
Parquet files. Existing source files are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import polars as pl
import pyarrow.parquet as pq


NUMERIC = [f"I{i}" for i in range(1, 14)]
CATEGORICAL = [f"C{i}" for i in range(1, 27)]
SPLITS = ("train", "valid", "test")
PUBLIC_NUMERIC_SIZES = {
    "I1": 44, "I2": 99, "I3": 122, "I4": 41, "I5": 220,
    "I6": 112, "I7": 80, "I8": 69, "I9": 92, "I10": 6,
    "I11": 27, "I12": 37, "I13": 72,
}
PUBLIC_TOTAL_FEATURES = 910_747


def bucket(values: np.ndarray, legacy: bool = False) -> np.ndarray:
    """Apply the official converter, or the old incorrect converter for QA."""
    values = np.asarray(values, dtype=np.int64)
    result = np.ones_like(values) if legacy else values.copy()
    mask = values > 2
    if np.any(mask):
        positive = values[mask].astype(np.float64, copy=False)
        result[mask] = np.floor(np.log(positive) ** 2).astype(np.int64)
    return result


def iter_numeric(path: Path, chunk_rows: int) -> Iterable[pl.DataFrame]:
    frame = (
        pl.scan_csv(
            path,
            null_values=[""],
            infer_schema_length=10_000,
            low_memory=True,
        )
        .select(
            [
                pl.col(name).fill_null(0).cast(pl.Int64).alias(name)
                for name in NUMERIC
            ]
        )
    )
    yield from frame.collect_batches(
        chunk_size=chunk_rows,
        maintain_order=True,
    )


class ExactReader:
    """Serve exact row counts when CSV and Parquet batch boundaries differ."""

    def __init__(self, path: Path, chunk_rows: int):
        self.path = path
        self.iterator = iter(iter_numeric(path, chunk_rows))
        self.batch = None
        self.offset = 0
        self.rows = 0

    def take(self, rows: int) -> pl.DataFrame:
        remaining = rows
        parts: List[pl.DataFrame] = []
        while remaining:
            if self.batch is None or self.offset == self.batch.height:
                try:
                    self.batch = next(self.iterator)
                    self.offset = 0
                except StopIteration as error:
                    raise RuntimeError(
                        f"Raw CSV ended early at {self.rows:,} rows: {self.path}"
                    ) from error
            available = self.batch.height - self.offset
            size = min(available, remaining)
            parts.append(self.batch.slice(self.offset, size))
            self.offset += size
            self.rows += size
            remaining -= size
        return parts[0] if len(parts) == 1 else pl.concat(parts, rechunk=False)

    def assert_exhausted(self) -> None:
        if self.batch is not None and self.offset < self.batch.height:
            raise RuntimeError(f"Raw CSV has extra rows after {self.rows:,}: {self.path}")
        try:
            extra = next(self.iterator)
        except StopIteration:
            return
        if extra.height:
            raise RuntimeError(f"Raw CSV has extra rows after {self.rows:,}: {self.path}")


def collect_counts(path: Path, chunk_rows: int) -> Tuple[Dict[str, Counter], int]:
    counters = {name: Counter() for name in NUMERIC}
    rows = 0
    for block_id, frame in enumerate(iter_numeric(path, chunk_rows)):
        values = bucket(frame.to_numpy())
        for index, name in enumerate(NUMERIC):
            tokens, counts = np.unique(values[:, index], return_counts=True)
            counters[name].update(
                {int(token): int(count) for token, count in zip(tokens, counts)}
            )
        rows += frame.height
        print(f"[fit] block={block_id:05d} rows={rows:,}", flush=True)
    if not rows:
        raise RuntimeError(f"Empty training file: {path}")
    return counters, rows


def build_vocabs(
    counters: Mapping[str, Counter],
    min_count: int,
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, int]]]:
    integer_vocabs = {}
    json_vocabs = {}
    for name in NUMERIC:
        # Public v2.2.0 passes fill_na=0 to Tokenizer as na_value=0.
        # Token 0 is excluded from its vocabulary and later maps to OOV.
        # This applies to both missing values and literal zeros.
        kept = sorted(
            (
                (int(token), int(count))
                for token, count in counters[name].items()
                if count >= min_count and int(token) != 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        integer_vocab = {
            token: index for index, (token, _) in enumerate(kept, 1)
        }
        oov = len(integer_vocab) + 1
        size = oov + 1
        if size != PUBLIC_NUMERIC_SIZES[name]:
            raise RuntimeError(
                f"{name} vocab_size={size}, expected={PUBLIC_NUMERIC_SIZES[name]}; "
                f"lowest-kept={kept[-10:]}"
            )
        json_vocab = {"__PAD__": 0}
        json_vocab.update(
            {str(token): index for token, index in integer_vocab.items()}
        )
        json_vocab["__OOV__"] = oov
        integer_vocabs[name] = integer_vocab
        json_vocabs[name] = json_vocab
        print(
            f"[vocab] {name} size={size} real={len(integer_vocab)} oov={oov}",
            flush=True,
        )
    return integer_vocabs, json_vocabs


def lookup_for(vocab: Mapping[int, int]) -> Tuple[np.ndarray, int]:
    oov = len(vocab) + 1
    maximum = max((token for token in vocab if token >= 0), default=-1)
    lookup = np.full(maximum + 1, oov, dtype=np.int64)
    for token, index in vocab.items():
        if token >= 0:
            lookup[token] = index
    return lookup, oov


def encode(values: np.ndarray, lookup: np.ndarray, oov: int) -> np.ndarray:
    output = np.full(values.shape, oov, dtype=np.int64)
    valid = (values >= 0) & (values < lookup.size)
    output[valid] = lookup[values[valid]]
    return output


def update_metadata(
    feature_map: Mapping,
    feature_vocab: Mapping,
    dataset_id: str,
    numeric_vocabs: Mapping[str, Mapping[str, int]],
) -> Tuple[Dict, Dict]:
    result_map = copy.deepcopy(feature_map)
    result_vocab = copy.deepcopy(feature_vocab)
    result_map["dataset_id"] = dataset_id
    total = 0
    seen = set()
    for item in result_map["features"]:
        if len(item) != 1:
            raise RuntimeError(f"Unexpected feature map item: {item}")
        name, spec = next(iter(item.items()))
        seen.add(name)
        if name in numeric_vocabs:
            spec["vocab_size"] = max(numeric_vocabs[name].values()) + 1
        total += int(spec["vocab_size"])
    expected = set(NUMERIC + CATEGORICAL)
    if seen != expected:
        raise RuntimeError(
            f"Feature fields differ: missing={expected - seen}, extra={seen - expected}"
        )
    result_map["total_features"] = total
    if total != PUBLIC_TOTAL_FEATURES:
        raise RuntimeError(
            f"total_features={total}, expected={PUBLIC_TOTAL_FEATURES}"
        )
    for name, vocab in numeric_vocabs.items():
        result_vocab[name] = dict(vocab)
    return result_map, result_vocab


def transform_split(
    split: str,
    source: Path,
    output: Path,
    vocabs: Mapping[str, Mapping[int, int]],
    chunk_rows: int,
    compression: str,
) -> Dict:
    parquet = pq.ParquetFile(source / f"{split}.parquet")
    reader = ExactReader(source / f"{split}.csv", chunk_rows)
    lookups = {name: lookup_for(vocabs[name]) for name in NUMERIC}
    writer = None
    rows = 0
    changed = 0
    try:
        for group in range(parquet.num_row_groups):
            old_frame = pl.from_arrow(parquet.read_row_group(group))
            raw = reader.take(old_frame.height).to_numpy()
            values = bucket(raw)
            columns = []
            for index, name in enumerate(NUMERIC):
                lookup, oov = lookups[name]
                ids = encode(values[:, index], lookup, oov)
                changed += int(np.count_nonzero(ids != old_frame[name].to_numpy()))
                columns.append(pl.Series(name=name, values=ids, dtype=pl.Int64))
            new_frame = old_frame.with_columns(columns)
            table = new_frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    output,
                    table.schema,
                    compression=compression,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=new_frame.height)
            rows += new_frame.height
            print(
                f"[{split}] row_group={group:03d}/{parquet.num_row_groups - 1:03d} "
                f"rows={rows:,} changed_numeric_positions={changed:,}",
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
        "changed_numeric_positions": changed,
        "bytes": output.stat().st_size,
    }


def read_json(path: Path) -> Dict:
    with path.open() as file:
        return json.load(file)


def write_json(path: Path, payload: Mapping) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=False)


def self_test(source: Path) -> None:
    probe = np.asarray([0, 1, 2, 3, 4, 7, 10, 100], dtype=np.int64)
    expected = np.asarray([0, 1, 2, 1, 1, 3, 5, 21], dtype=np.int64)
    actual = bucket(probe)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"Formula failed: {actual} != {expected}")

    payload = read_json(source / "feature_vocab.json")
    raw = next(iter_numeric(source / "train.csv", 10_000))
    current = pl.read_parquet(source / "train.parquet", n_rows=raw.height)
    old_values = bucket(raw.to_numpy(), legacy=True)
    mismatches = 0
    for index, name in enumerate(NUMERIC):
        vocab = {
            int(token): int(token_id)
            for token, token_id in payload[name].items()
            if not token.startswith("__")
        }
        lookup, _ = lookup_for(vocab)
        ids = encode(old_values[:, index], lookup, int(payload[name]["__OOV__"]))
        mismatches += int(np.count_nonzero(ids != current[name].to_numpy()))
    if mismatches:
        raise AssertionError(f"Raw/Parquet alignment mismatches={mismatches}")
    print(
        f"SELF_TEST_OK formula={actual.tolist()} aligned_rows={raw.height} "
        f"numeric_fields={len(NUMERIC)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-id", default="Criteo_x4_001_v220_semantics")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=1_048_576)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    target = args.output_root
    if args.self_test:
        self_test(source)
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    if target.parent.resolve() != source:
        raise ValueError("--output-root must be a direct child of --source-root")
    required = [source / f"{split}.csv" for split in SPLITS]
    required += [source / f"{split}.parquet" for split in SPLITS]
    required += [source / "feature_map.json", source / "feature_vocab.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")

    counters, train_rows = collect_counts(source / "train.csv", args.chunk_rows)
    integer_vocabs, numeric_json_vocabs = build_vocabs(
        counters, args.min_count
    )
    feature_map, feature_vocab = update_metadata(
        read_json(source / "feature_map.json"),
        read_json(source / "feature_vocab.json"),
        args.dataset_id,
        numeric_json_vocabs,
    )

    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging output already exists: {staging}")
    staging.mkdir()
    split_stats = {}
    for split in SPLITS:
        split_stats[split] = transform_split(
            split,
            source,
            staging / f"{split}.parquet",
            integer_vocabs,
            args.chunk_rows,
            args.compression,
        )
    write_json(staging / "feature_map.json", feature_map)
    write_json(staging / "feature_vocab.json", feature_vocab)
    write_json(
        staging / "manifest.json",
        {
            "dataset_id": args.dataset_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source),
            "output_root": str(target),
            "preprocessing": {
                "numeric_missing_value": 0,
                "numeric_transform": "floor(log(x)^2) if x > 2 else int(x)",
                "numeric_feature_type": "categorical",
                "min_count": args.min_count,
                "vocab_fit_split": "train_only",
                "categorical_ids": "preserved from source Parquet",
                "numeric_tokenizer_na_value": 0,
                "numeric_zero_encoding": "OOV",
                "labels": "preserved from source Parquet",
            },
            "train_rows_used_for_numeric_vocab": train_rows,
            "numeric_vocab_sizes": PUBLIC_NUMERIC_SIZES,
            "total_features": feature_map["total_features"],
            "splits": split_stats,
        },
    )
    if target.exists():
        raise FileExistsError(f"Output appeared during build: {target}")
    staging.rename(target)
    print(
        f"BUILD_OK output={target} total_features={feature_map['total_features']} "
        f"train_rows={train_rows:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
