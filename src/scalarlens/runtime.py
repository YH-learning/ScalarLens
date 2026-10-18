"""Relocated FP32 packed/prefetched runtime from the measured endpoint snapshot.

Opt-in CUDA implementation. CPU regression does not certify GPU speedups.
"""
import logging
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.dataloaders.parquet_dataloader import ParquetDataset

class PackedPinnedParquetDataLoader(TorchDataLoader):
    """Non-streaming Parquet loader with one pinned 2-D tensor per batch.

    It preserves the exact values and row order of FuxiCTR's standard Parquet
    loader.  Column views are materialized only after a single asynchronous
    host-to-device copy in ``CUDAPrefetchLoader``.
    """

    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.feature_map = feature_map
        self.dataset = ParquetDataset(feature_map, data_path)
        loader_kwargs = dict(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=True,
        )
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
        super().__init__(**loader_kwargs)
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

class CUDAPrefetchLoader:
    """Re-iterable double-buffered loader producing the standard input dict."""

    def __init__(self, loader, feature_map, device):
        self.loader = loader
        self.feature_map = feature_map
        self.device = torch.device(device)
        self.columns = list(feature_map.features.keys()) + feature_map.labels

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)

        def preload():
            try:
                host_batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                device_batch = host_batch.to(self.device, non_blocking=True)
                return {
                    name: device_batch[:, self.feature_map.get_column_index(name)]
                    for name in self.columns
                }

        next_batch = preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            for tensor in batch.values():
                tensor.record_stream(torch.cuda.current_stream(self.device))
            next_batch = preload()
            yield batch

def enable_optimized_runtime(model, mode):
    """Enable an FP32 compilation path without changing model parameters.

    This is deliberately opt-in. Dataset validation remains a separate data
    preflight responsibility; the model class keeps validation enabled by
    default outside this deployment path.
    """
    if mode == "none":
        return
    if mode not in (
        "packed-only",
        "module-compile-packed",
        "default-packed",
        "reduce-overhead",
        "reduce-overhead-packed",
        "portable-packed",
    ):
        raise ValueError(f"unsupported compile runtime: {mode}")
    torch._dynamo.config.cache_size_limit = 64
    scalar_count = 0
    scalar_module = None
    for module in model.modules():
        if module.__class__.__name__ == "FeatureEmbeddingDict":
            module.categorical_lookup_mode = "auto"
            module.reuse_context_embeddings = True
        elif module.__class__.__name__ == "ScalarLensEmbedding":
            scalar_count += 1
            scalar_module = module
            module.validate_inputs = False
            module.set_execution_mode("eager")
        elif module.__class__.__name__ == "StableCoordinates":
            module.set_execution_mode("reference")
    if scalar_count != 1:
        raise RuntimeError(
            "reduce-overhead runtime is restricted to one ScalarLens module; "
            f"found {scalar_count}"
        )
    if mode == "packed-only":
        logging.info(
            "Enabled ScalarLens exact packed runtime without compiler: input "
            "checks disabled after dataset preflight; parameters and objective "
            "unchanged."
        )
        return
    if mode == "module-compile-packed" or (
        mode == "portable-packed" and model.__class__.__name__ == "DeepFM"
    ):
        wrapped = torch.compile(
            scalar_module, fullgraph=True, dynamic=False,
            mode="reduce-overhead",
        )
        for owner in model.modules():
            layers = getattr(owner, "numeric_embedding_layers", None)
            if layers is None:
                continue
            for key, value in layers.items():
                if value is scalar_module:
                    layers[key] = wrapped
                    logging.info(
                        "Enabled ScalarLens FP32 module-only compiled packed "
                        "runtime; full backbone remains eager."
                    )
                    return
        raise RuntimeError("ScalarLens owner slot not found")

    compile_mode = "reduce-overhead"
    if mode == "default-packed":
        compile_mode = None
    elif mode == "portable-packed":
        # The portable path uses whole-step reduce-overhead for DCNv2 and the
        # module-only branch above for DeepFM.  The latter avoids an invalid
        # accelerator Triton reduction for AutoML-A's short-field shape.
        compile_mode = "reduce-overhead"

    compile_kwargs = dict(fullgraph=False, dynamic=False)
    if compile_mode is not None:
        compile_kwargs["mode"] = compile_mode
    original_forward = model.forward
    model.forward = torch.compile(original_forward, **compile_kwargs)

    def compiled_train_step(batch_data):
        model.optimizer.zero_grad(set_to_none=True)
        return_dict = model(batch_data)
        y_true = model.get_labels(batch_data)
        loss = model.compute_loss(return_dict, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model._max_gradient_norm)
        model.optimizer.step()
        return loss

    model.train_step = torch.compile(compiled_train_step, **compile_kwargs)
    logging.info(
        "Enabled ScalarLens FP32 compiled runtime (mode=%s): input checks "
        "disabled after dataset preflight; parameters and objective unchanged.",
        compile_mode or "default",
    )

def build_data_iterators(feature_map, params, runtime_mode, stage, device):
    """Build standard or packed/prefetched iterators without changing samples."""
    loader_params = dict(params)
    use_packed = runtime_mode in (
        "packed-only",
        "module-compile-packed",
        "default-packed",
        "reduce-overhead-packed",
        "portable-packed",
    )
    if use_packed:
        if params.get("data_format") != "parquet" or params.get("streaming"):
            raise ValueError(
                "reduce-overhead-packed requires non-streaming Parquet data"
            )
        loader_params["data_loader"] = PackedPinnedParquetDataLoader
    iterator = RankDataLoader(
        feature_map, stage=stage, **loader_params
    ).make_iterator()
    if not use_packed:
        return iterator
    if stage == "train":
        train_gen, valid_gen = iterator
        return (
            CUDAPrefetchLoader(train_gen, feature_map, device),
            CUDAPrefetchLoader(valid_gen, feature_map, device),
        )
    return CUDAPrefetchLoader(iterator, feature_map, device)
