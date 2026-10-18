# ScalarLens

**Numerical Embeddings with Stable Coordinates and Contextual Responses for CTR Prediction**

ScalarLens maps raw numerical values to learnable local coordinates, then uses numerical and categorical context to produce numerical embeddings for a CTR predictor. It is a numerical encoder, not a replacement for the downstream backbone.

This repository contains the encoder, nine CTR backbones, dataset preparation tools, paper presets, and the per-seed records behind the main comparison table.

## Repository at a glance

```text
src/scalarlens/    Encoder, backbones, training, data tools, and result analysis
src/fuxictr/       Bundled framework with the shared numerical-embedding LR interface
scripts/          Verification and source packaging
tests/            Numerical equivalence and integration tests
reproduction/     Paper records, dataset schemas, provenance, and licenses
```

There is no separate `data/` versus `dataset/` directory: all data **code** is in `scalarlens.data`. Store downloaded datasets outside the checkout, or in the ignored `datasets/` directory. All generated results, logs, and release archives go in the ignored `outputs/` directory.

## Installation

Use Python **3.10–3.12**. The paper used PyTorch 2.6.0. Linux with a compatible accelerator is recommended for full training; CPU/macOS can run the tests and synthetic example.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# CUDA 12.6 example: choose a PyTorch wheel compatible with your driver.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# For CPU or macOS, use: python -m pip install torch==2.6.0

python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

Do **not** separately install FuxiCTR: the required version and numerical/LR integration are bundled here. Triton is optional, used only by accelerator-specific optimized execution; it is not needed for CPU or reference execution. Compatible Linux systems can install the explicit extra with `python -m pip install '.[cuda]'`. Vendor accelerators may require their own PyTorch/compiler distribution.

## Quick start

### Recompute the paper table — no dataset or PyTorch required

From the repository root:

```bash
PYTHONPATH=src python3 -m scalarlens.results \
  --source reproduction/paper --output outputs/paper
```

This recomputes statistics from **1,539 archived per-seed records**: three datasets, nine backbones, 19 methods, and three seeds. It checks all 513 AUC/SD/Logloss table cells and saves a 54-cell stratified sample audit. Expected status: `PASS`, with no aggregation or table errors.

Read `outputs/paper/summary.md` for the complete results. Machine-readable outputs include `per_seed.csv`, `summary.csv`, `ranking.csv`, `sample_check.csv`, and `verification.json`. This is a reconstruction of archived results, **not a new training run**.

### Train a small synthetic example

After installation:

```bash
python -m scalarlens.data --dataset-dir outputs/smoke/data --make-smoke
scalarlens-train --dataset-dir outputs/smoke/data \
  --output-dir outputs/smoke/train --backbone DCNv2 \
  --method scalarlens --seed 2026 --gpu -1 --workers 0 \
  --epochs 2 --batch-size 128
```

The example verifies data loading, training, best-checkpoint reload, and test evaluation. Its scores are not benchmark results. Choose a new output directory when repeating a completed task.

## Prepare real datasets

Prepared datasets must contain `feature_map.json`, `train.parquet`, `valid.parquet`, and `test.parquet`. Categorical columns contain encoded integer IDs. Numerical columns retain their original scale; initial quantile boundaries are fitted on the training split only.

Set a dataset root, for example `export DATA_ROOT="$PWD/datasets"`. Dataset acquisition, exact split rules, and converter commands are documented in [the reproduction guide](reproduction/README.md#datasets).

```bash
python -m scalarlens.data --dataset-dir "$DATA_ROOT/AutoML_A" \
  --output outputs/data_check.json
```

The validator checks split row counts from Parquet metadata and samples up to 8,192 rows per split for finite values, categorical ID ranges, and binary labels. It does not claim to compare every source row.

## Reproduce training

### One run

```bash
scalarlens-train --dataset-dir "$DATA_ROOT/AutoML_A" \
  --output-dir outputs/automl_a_dcnv2_seed2026 \
  --backbone DCNv2 --method scalarlens --seed 2026 --gpu 0
```

Each task saves its resolved `config.json`, `train.log`, `result.json`, `result.csv`, and checkpoints in the selected output directory. Model selection uses **validation AUC only**; test metrics are evaluated after reloading the best checkpoint.

Supported backbones: **DNN, DeepFM, WideDeep, NFM, PNN, DCN, DCNv2, AutoInt, FiBiNET**.

Verified training presets: **`scalarlens`, `field`, `deer`, `daes_gate`, `narydis`**. `narydis` is the full variant, not NaryDis-Lite. The archived table covers all 19 methods; the training CLI does not claim verified presets for the remaining methods.

### Three datasets × nine backbones × three seeds

```bash
# Inspect the 81-task ScalarLens plan first.
scalarlens-matrix --data-root "$DATA_ROOT" \
  --output-dir outputs/scalarlens_matrix --gpus 0 1 2 3 --dry-run

scalarlens-matrix --data-root "$DATA_ROOT" \
  --output-dir outputs/scalarlens_matrix --gpus 0 1 2 3

# Optional 405-task comparison of the five verified presets.
scalarlens-matrix --data-root "$DATA_ROOT" \
  --output-dir outputs/core_comparison --gpus 0 1 2 3 \
  --methods field deer daes_gate narydis scalarlens

# Summarize new results separately, including their deltas from the paper.
scalarlens-reproduce --runs outputs/scalarlens_matrix \
  --source reproduction/paper --output outputs/new_summary
```

The launcher runs one process per GPU slot. Repeating the same command skips completed tasks with matching identities and finite metrics; inspect `failures.json` before resuming failed tasks. Do not run two launchers on the same output directory or GPU slots. Non-streaming loading requires sufficient host RAM, especially with multiple concurrent processes.

### Fixed protocol

| Setting | Value |
|---|---|
| Seeds | 2026, 2027, 2028 |
| Epoch limit / early-stop patience | 100 / 2 |
| Selection | Validation AUC, maximize |
| Batch size / optimizer / learning rate | 4096 / Adam / 0.001 |
| Embedding dimension / MLP | 16 / [256, 128, 64] |
| Batch normalization / dropout / regularization | Off / 0 / 0 |
| Input | Raw numerical scale; non-streaming Parquet; global shuffle |
| ScalarLens intervals / modes / rank / steps | 16 / 8 / 16 / 3 |
| Mode hidden width / readout width | 32 / 32 |

The single configuration source is [`src/scalarlens/assets/paper.yaml`](src/scalarlens/assets/paper.yaml). In models with an LR branch, numerical LR consumes the **same 16-dimensional numerical embeddings** and projects them to scalar scores. Categorical LR remains native; there is no independent one-dimensional numerical encoder.

## Use the encoder directly

```python
import torch
from scalarlens import ScalarLensEmbedding, StableCoordinates

# Replace these toy boundaries with boundaries estimated from training data.
edges = torch.linspace(-3, 3, 17).repeat(2, 1)
local = StableCoordinates(edges, d_embedding=16)
encoder = ScalarLensEmbedding(local, ["N1", "N2"], 16, {"C1": 16})

values = torch.randn(32, 2)
context = {"C1": torch.randn(32, 16)}
embeddings = encoder(values, context)  # [batch, numerical fields, embedding dim]
```

For FuxiCTR integration, use `numeric_embedding=scalarlens` and `numeric_embedding_plugins=[scalarlens.plugin]`. Historical names such as `Fast` remain only in archived records and the private reference implementation needed for exact regression tests.

## Verify and package

```bash
python -m pytest -q
python scripts/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/verify.py
python scripts/verify_wheel.py
python scripts/package.py
```

The checks cover original-versus-public encoder outputs, gradients and Adam updates; initialization/RNG preservation; the shared LR interface; nine backbones; real dataset schemas; CPU training; dependency declarations; and archived statistics. The wheel check installs and trains outside the source checkout. See [verification details](reproduction/README.md#verification).

The source archive is written to `outputs/ScalarLens-source.zip`. It excludes environments, datasets, weights, logs, build products, and Git metadata. The repository can be pushed normally; these generated files are ignored.

## Optimized execution

Reference-quality training is the default. Optional packed input transfer, prefetching, compilation, and fused Adam do not change the propagation structure:

```bash
scalarlens-train --dataset-dir "$DATA_ROOT/Criteo_x4_NumEmbRaw" \
  --output-dir outputs/criteo_optimized --backbone DCNv2 --gpu 0 \
  --runtime portable-packed --fused-adam
```

See [the timing protocol](reproduction/README.md#efficiency) before comparing speed. Module compilation alone is not the end-to-end optimized setting. Accelerator paths require a compatible compiler and valid input data; CPU tests do not establish GPU speedups.

## License and provenance

Code is distributed under Apache-2.0, with upstream notices retained. See [NOTICE](NOTICE) and [third-party licenses](reproduction/licenses). Datasets and checkpoints are not redistributed. Reference methods are the implementations used in our experiments, not a claim that every method is official author code.

The [paper records](reproduction/paper) preserve all reported metric values, seeds, and epochs. Machine-specific result paths have been replaced by relative identifiers; original hashes and transformations are recorded in [metadata](reproduction/metadata). Full real-dataset GPU retraining is separate from the local verification suite.
