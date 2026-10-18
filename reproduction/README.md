# Reproduction guide

This directory holds **inputs and evidence**, not generated training outputs:

- `paper/`: 1,539 per-seed records, archived mean/std values, and the paper's comparison table.
- `metadata/`: dataset schemas, training environment, source provenance and relocation records.
- `checks/`: concise checks from the public-source release; verbose logs are generated under `outputs/validation/`.
- `licenses/`: retained upstream licenses.

## Datasets

Download data separately and set `DATA_ROOT` to a directory containing `AutoML_A`, `AutoML_E`, and `Criteo_x4_NumEmbRaw`.

```text
DATA_ROOT/
  AutoML_A/
    feature_map.json
    train.parquet
    valid.parquet
    test.parquet
  AutoML_E/             # same four files
  Criteo_x4_NumEmbRaw/   # same four files
```

A feature map contains `dataset_id`, `labels`, and a list of single-field dictionaries under `features`. Numerical fields have `type=numeric` and strictly increasing, train-only `numeric_bin_edges`. Categorical fields have `type=categorical`, `vocab_size` and `padding_idx`. Parquet columns must use those names; categorical values must already be integer IDs. Schema snapshots are in `metadata/feature_maps/`.

### AutoML-A and AutoML-E

Obtain and extract the original AutoML3 input archive. [AutoML3Final](https://github.com/djajetic/AutoML3Final) describes the source challenge and processing. `AUTOML_SOURCE` must point to the **data**, not the cloned processing repository. It must contain `AA/` and `E/`, including files such as `AA_feat.type`, `AA_public.info`, `AA_train1.data` and `AA_train1.solution` under `AA/`.

```bash
python -m scalarlens.data.automl \
  --source-root "$AUTOML_SOURCE" --output-root "$DATA_ROOT" \
  --tasks A,E --chunk-rows 50000 --sample-stride 64 --n-bins 16
```

The chronological protocol is **train=train1+test1+test2, valid=test3, test=test4**. Unlabeled test5–test9 are excluded. Numerical and categorical fields are retained; time and multivalue fields are excluded. A has 23 numerical and 51 categorical fields; E has 6 and 25. Vocabularies and statistics are fitted only on train. Values remain on their original scale: no log or z-score is applied. Means/stds stored in metadata do not imply normalization. `--max-rows-per-batch` is only a debugging option, not the paper protocol.

### Criteo x4

Use the paper's existing x4 split and categorical encoding, not a new random split. [RecData](https://github.com/reczoo/RecData) provides source dataset information. Set `CRITEO_SOURCE` to the original prepared x4 directory, containing `train/valid/test.csv`, corresponding encoded `train/valid/test.parquet`, `feature_map.json` and `feature_vocab.json`.

**A raw `train.txt` download alone is not sufficient for these converters.** The split/encoding reference is an explicit prerequisite; this repository does not redistribute it or claim to recreate it from arbitrary Criteo files.

```bash
python -m scalarlens.data.criteo_reference \
  --source-root "$CRITEO_SOURCE" \
  --output-root "$CRITEO_SOURCE/Criteo_x4_001_v220_semantics" --min-count 10

python -m scalarlens.data.criteo_raw \
  --source-root "$CRITEO_SOURCE" \
  --reference-root "$CRITEO_SOURCE/Criteo_x4_001_v220_semantics" \
  --output-root "$CRITEO_SOURCE/Criteo_x4_NumEmbRaw" \
  --dataset-id Criteo_x4_NumEmbRaw --sample-stride 32 --n-bins 16
```

Both output directories must be direct children of the source directory. The second converter reads raw I1–I13 values from CSV, fills missing values with zero, and preserves the reference categorical IDs, labels and row order. Categories use train frequency threshold 10 with no hashing. Link the final directory under `DATA_ROOT` if necessary. Do not use log-squared-bucket numerical inputs as the raw dataset.

### Dataset checks

```bash
python -m scalarlens.data --dataset-dir "$DATA_ROOT/AutoML_A" \
  --output outputs/data_audit/AutoML_A.json
```

Historical row counts and feature-map hashes are in `metadata/datasets.json`. Relocated file paths and serialized metadata can change byte hashes without changing the data. The original feature-map hash is retained as provenance, not claimed as a universal hash for every regenerated serialization.

## Results and statistics

The archive contains 3 datasets × 9 backbones × 19 methods × 3 seeds. Mean and **sample** standard deviation (`ddof=1`) are recomputed from per-seed records. AUC/Logloss means are truncated to four decimals for the paper table; standard deviations are rounded to four decimals. Rankings use untruncated test AUC; exact ties follow the documented method order.

The old `summary_513.csv` has some rank columns inherited from an earlier method set. Only its mean/std values are used for validation. Published ranks are recomputed across the 19 displayed methods. ContextFiLM is an ablation, not a ranked baseline.

The ScalarLens archive wins 25 of 27 dataset/backbone settings, with mean rank 1.074074. Reaggregation checks all four validation/test metrics, all 513 AUC/SD/Logloss table cells, and a 54-cell stratified sample. `Fast` in the source records is ScalarLens; the public training name is `scalarlens`. `narydis_full` maps to the public `narydis` preset.

Machine-specific paths in archived CSV/metadata are replaced with relative identifiers. No numerical metric, seed, epoch or duration is changed. `metadata/result_relocation.json` records the original per-seed file hash; `metadata/source_imports.json` records imported source hashes and public-layout transformations.

## Verification

```bash
python scripts/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/verify.py
python scripts/verify_wheel.py
```

The suite checks exact encoder forward/backward/Adam equivalence, RNG preservation, shared 16-dimensional numerical LR input, nine backbone forward/backward paths, all 27 real schema combinations using artificial rows, archived table arithmetic, ten CPU training tasks including a repeated seed, and installation/training from a wheel outside the checkout.

The original experiment used Python 3.10.16, PyTorch 2.6.0, and PPU-ZW810E accelerators. Local verification uses the environment recorded in `checks/`. Real-schema equivalence is **not** real-data training; synthetic scores are **not** benchmark scores; CPU verification is **not** a GPU speed reproduction. Full real-data GPU retraining was not performed during source cleanup.

## Efficiency

Run each candidate in a fresh process on the same idle accelerator, at least three repetitions:

```bash
python -m scalarlens.benchmark --dataset-dir "$DATA_ROOT/Criteo_x4_NumEmbRaw" \
  --backbone DCNv2 --gpu 0 --runtime none --output outputs/speed/reference.json
python -m scalarlens.benchmark --dataset-dir "$DATA_ROOT/Criteo_x4_NumEmbRaw" \
  --backbone DCNv2 --gpu 0 --runtime portable-packed --fused-adam \
  --output outputs/speed/optimized.json
```

Default timing uses batch 4096, 20 warmup steps, 400 training steps, 100 inference steps, and 20-step timing blocks. Cold first-call compilation is reported separately. End-to-end timing includes the iterator, host-to-device transfer and full model; training also includes loss, backward, clipping and Adam. Timing is not a convergence experiment: separately rerun full training to check accuracy.

`portable-packed` uses full-step compilation for DCNv2 and module-only compilation for DeepFM. Backend/compiler compatibility is device-dependent. The optimized path disables per-step input checks after preflight; sampled preflight does not prove every custom input row is valid.
