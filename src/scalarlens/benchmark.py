"""CUDA endpoint timing, with input movement and compilation cold start separated."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time


def timed_first(iterator, step, synchronize):
    started = time.perf_counter()
    batch = next(iterator)
    loaded = time.perf_counter()
    step(batch)
    synchronize()
    ended = time.perf_counter()
    return dict(data_ms=(loaded-started)*1000, step_ms=(ended-loaded)*1000,
                total_ms=(ended-started)*1000)


def timed_blocks(iterator, step, synchronize, warmup, steps, block_size):
    for _ in range(warmup):
        step(next(iterator))
    synchronize()
    blocks = []
    remaining = steps
    while remaining:
        count = min(block_size, remaining)
        started = time.perf_counter()
        for _ in range(count):
            step(next(iterator))
        synchronize()
        blocks.append((time.perf_counter()-started)*1000/count)
        remaining -= count
    return dict(steps=steps, block_size=block_size, mean_ms=statistics.mean(blocks),
                median_ms=statistics.median(blocks), stdev_ms=statistics.stdev(blocks) if len(blocks)>1 else 0.,
                block_means_ms=blocks)


def main():
    import torch
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.torch_utils import seed_everything
    from .backbones import get_backbone
    from .data import validate
    from .protocol import training_params
    from .runtime import enable_optimized_runtime, build_data_iterators
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backbone', choices=('DCNv2', 'DeepFM'), default='DCNv2')
    p.add_argument('--runtime', choices=('none', 'packed-only', 'portable-packed', 'reduce-overhead-packed'), default='none')
    p.add_argument('--fused-adam', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--train-steps', type=int, default=400)
    p.add_argument('--infer-steps', type=int, default=100)
    p.add_argument('--block-size', type=int, default=20)
    a = p.parse_args()
    if not torch.cuda.is_available() or a.gpu < 0:
        p.error('Endpoint benchmark requires a CUDA-compatible accelerator; CPU timing is not substituted.')
    if a.output.exists():
        raise FileExistsError(a.output)
    if min(a.train_steps, a.infer_steps, a.block_size) <= 0 or a.warmup < 0:
        p.error('Step counts/block size must be positive; warmup nonnegative.')
    validate(a.dataset_dir)
    torch.set_num_threads(1)
    torch.cuda.set_device(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(a.seed)
    params = training_params(a.backbone)
    dataset_id = json.loads((a.dataset_dir/'feature_map.json').read_text())['dataset_id']
    params.update(dataset_id=dataset_id, gpu=a.gpu, verbose=0, num_workers=a.workers,
                  seed=a.seed, train_data=str(a.dataset_dir/'train.parquet'),
                  valid_data=str(a.dataset_dir/'valid.parquet'), test_data=str(a.dataset_dir/'test.parquet'))
    params['numeric_embedding_params']['execution_mode'] = 'reference'
    params['numeric_embedding_params']['deer_execution_mode'] = 'reference'
    with tempfile.TemporaryDirectory(prefix='scalarlens-endpoint-') as temporary:
        params['model_root'] = temporary
        feature_map = FeatureMap(dataset_id, str(a.dataset_dir))
        feature_map.load(str(a.dataset_dir/'feature_map.json'), params)
        started = time.perf_counter()
        model = get_backbone(a.backbone)(feature_map, **params)
        model._max_gradient_norm = 10.
        if a.fused_adam:
            model.optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], fused=True)
        enable_optimized_runtime(model, a.runtime)
        train, valid = build_data_iterators(feature_map, params, a.runtime, 'train', model.device)
        setup_ms = (time.perf_counter()-started)*1000
        train_steps = min(a.train_steps, len(train)-a.warmup-1)
        infer_steps = min(a.infer_steps, len(valid)-a.warmup-1)
        if min(train_steps, infer_steps) <= 0:
            raise ValueError('Insufficient batches for warmup + measurement. Do not duplicate rows to hide this.')
        sync = lambda: torch.cuda.synchronize(a.gpu)
        iterator = iter(train)
        model.train()
        first_train = timed_first(iterator, model.train_step, sync)
        steady_train = timed_blocks(iterator, model.train_step, sync, a.warmup, train_steps, a.block_size)
        iterator = iter(valid)
        model.eval()

        def infer(batch):
            with torch.inference_mode():
                return model(batch)['y_pred']

        first_inference = timed_first(iterator, infer, sync)
        steady_inference = timed_blocks(iterator, infer, sync, a.warmup, infer_steps, a.block_size)
        result = dict(dataset=dataset_id, backbone=a.backbone, runtime=a.runtime, fused_adam=a.fused_adam,
                      seed=a.seed, batch_size=params['batch_size'], device=torch.cuda.get_device_name(a.gpu),
                      torch=torch.__version__, tf32=False, setup_ms=setup_ms,
                      scope='iterator + H2D + full model; train includes loss/backward/clip/Adam; not convergence accuracy',
                      first_train=first_train, steady_train=steady_train,
                      first_inference=first_inference, steady_inference=steady_inference)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
