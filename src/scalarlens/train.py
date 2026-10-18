"""Train a paper preset using relocated, preprocessed Parquet data."""
import argparse
import csv
import json
import logging
import time
from pathlib import Path

from .backbones import BACKBONES, get_backbone
from .protocol import METHODS, training_params


def run(args):
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.dataloaders import RankDataLoader
    from fuxictr.pytorch.torch_utils import seed_everything

    data = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Completed output exists: {result_path}. Choose a fresh --output-dir.")
    metadata = json.loads((data / "feature_map.json").read_text())
    dataset_id = metadata["dataset_id"]
    params = training_params(args.backbone, args.method)
    params.update(dataset_id=dataset_id, data_root=str(data.parent), model_root=str(output / "checkpoints"),
                  model_id=f"{args.method}_{args.backbone}_{args.seed}", gpu=args.gpu, seed=args.seed,
                  num_workers=args.workers, train_data=str(data / "train.parquet"),
                  valid_data=str(data / "valid.parquet"), test_data=str(data / "test.parquet"))
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.execution_mode:
        if args.method != "scalarlens":
            raise ValueError("--execution-mode applies to ScalarLens")
        params["numeric_embedding_params"]["execution_mode"] = args.execution_mode
    runtime_mode = getattr(args, 'runtime', 'none')
    fused_adam = getattr(args, 'fused_adam', False)
    if runtime_mode != 'none' or fused_adam:
        import torch
        if args.method != 'scalarlens' or args.gpu < 0 or not torch.cuda.is_available():
            raise ValueError('Optimized endpoint execution requires ScalarLens and a CUDA-compatible device.')
        from .data import validate
        validate(data)  # sampled preflight; users remain responsible for full data validity
    params['runtime_mode'] = runtime_mode
    params['fused_adam'] = fused_adam
    for split in ("train", "valid", "test"):
        if not Path(params[split + "_data"]).is_file():
            raise FileNotFoundError(params[split + "_data"])
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(output / "train.log"), logging.StreamHandler()], force=True)
    (output / "config.json").write_text(json.dumps(params, indent=2) + "\n")
    seed_everything(args.seed)
    feature_map = FeatureMap(dataset_id, str(data))
    feature_map.load(str(data / "feature_map.json"), params)
    feature_map.data_dir = str(data)
    started = time.monotonic()
    model = get_backbone(args.backbone)(feature_map, **params)
    Path(model.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if fused_adam:
        import torch
        model.optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], fused=True)
    if runtime_mode != 'none':
        from .runtime import enable_optimized_runtime
        enable_optimized_runtime(model, runtime_mode)
    best_epochs = []
    original_checkpoint = model.checkpoint_and_earlystop

    def record_epoch(logs, *extra, **kw):
        previous = model._best_metric
        original_checkpoint(logs, *extra, **kw)
        if model._best_metric != previous:
            best_epochs.append(model._epoch_index + 1)

    model.checkpoint_and_earlystop = record_epoch
    loading = time.monotonic()
    def loaders(stage):
        if runtime_mode == 'none':
            return RankDataLoader(feature_map, stage=stage, **params).make_iterator()
        from .runtime import build_data_iterators
        return build_data_iterators(feature_map, params, runtime_mode, stage, model.device)

    train, valid = loaders('train')
    loading_seconds = time.monotonic() - loading
    model.fit(train, validation_data=valid, **params)
    val = model.evaluate(valid)
    stopped_epoch = model._epoch_index + 1
    del train, valid
    test = loaders('test')
    measured = model.evaluate(test)
    result = dict(dataset=dataset_id, backbone=args.backbone, method=args.method, seed=args.seed,
                  val_auc=float(val["AUC"]), val_logloss=float(val["logloss"]),
                  test_auc=float(measured["AUC"]), test_logloss=float(measured["logloss"]),
                  best_epoch=best_epochs[-1] if best_epochs else None, stopped_epoch=stopped_epoch,
                  elapsed_seconds=time.monotonic() - started, loading_seconds=loading_seconds,
                  checkpoint=str(model.checkpoint), selection_metric="validation AUC")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    with (output / "result.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, default="DCNv2")
    parser.add_argument("--method", choices=METHODS, default="scalarlens")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--execution-mode", choices=("reference", "auto", "compiled"))
    parser.add_argument('--runtime', choices=('none', 'packed-only', 'portable-packed', 'reduce-overhead-packed'), default='none')
    parser.add_argument('--fused-adam', action='store_true')
    run(parser.parse_args())


if __name__ == "__main__":
    main()
