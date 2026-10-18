"""Execution-only batching of ordinary categorical embedding lookups.

Original Parameters, state_dict keys, optimizer groups and padding semantics stay
unchanged. This helper does not pack or mutate parameter storage.
"""
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import module as module_hooks
from fuxictr.utils import not_in_whitelist


def grouped_categorical_lookup(owner, inputs, feature_source, feature_type):
    if any(getattr(module_hooks, name, {}) for name in (
        "_global_forward_hooks", "_global_forward_pre_hooks",
        "_global_backward_hooks", "_global_backward_pre_hooks")):
        return {}
    needed_context = set()
    if getattr(owner, "reuse_context_embeddings", False):
        for numeric in owner.numeric_embedding_layers.values():
            needed_context.update(getattr(numeric, "context_features", ()))
    groups = defaultdict(list)
    for name, layer in owner.embedding_layers.items():
        if name not in inputs or owner._feature_map.features[name]["type"] != "categorical":
            continue
        spec = owner._feature_map.features[name]
        visible = (not not_in_whitelist(spec["source"], feature_source)
                   and not not_in_whitelist(spec["type"], feature_type))
        if not visible and name not in needed_context:
            continue
        if (type(layer) is not nn.Embedding or layer.padding_idx != 0
            or layer.max_norm is not None or layer.sparse or layer.scale_grad_by_freq
            or name in owner.feature_encoders or inputs[name].ndim != 1
            or layer._forward_hooks or layer._forward_pre_hooks
            or layer._backward_hooks or layer._backward_pre_hooks):
            continue
        if getattr(owner, "categorical_lookup_mode", "reference") == "auto" and not layer.weight.is_cuda:
            continue
        key = (layer.weight.device, layer.weight.dtype, layer.embedding_dim)
        groups[key].append((name, layer))
    result = {}
    for entries in groups.values():
        if len(entries) < 2:
            continue
        ids = torch.stack([inputs[name].long() for name, _ in entries], dim=1)
        weights = [layer.weight for _, layer in entries]
        sizes = [w.shape[0] for w in weights]
        offsets = [0]
        for size in sizes[:-1]:
            offsets.append(offsets[-1] + size)
        offset = torch.tensor(offsets, dtype=ids.dtype, device=ids.device)
        size_tensor = torch.tensor(sizes, dtype=ids.dtype, device=ids.device)
        # A bad local ID must never silently address another field's table.
        torch._assert_async(((ids >= 0) & (ids < size_tensor)).all(),
                            "Grouped categorical embedding ID out of range")
        values = F.embedding(ids + offset, torch.cat(weights, dim=0))
        # Preserve even a nonzero padding value, but do not update its gradient.
        values = torch.where((ids == 0).unsqueeze(-1), values.detach(), values)
        result.update({name: values[:, i] for i, (name, _) in enumerate(entries)})
    return result
