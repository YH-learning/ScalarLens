"""Batched categorical lookup preserving each original parameter and optimizer slot."""
import torch
from torch.nn import functional as F

def grouped_lookup(indices, weights, group_size=0):
    count = len(weights)
    group_size = group_size or count
    outputs = []
    for start in range(0, count, group_size):
        group = weights[start:start + group_size]
        offsets, total = [], 0
        for weight in group:
            offsets.append(total)
            total += weight.shape[0]
        offset = torch.tensor(offsets, device=indices.device, dtype=indices.dtype)
        local_ids = indices[:, start:start + len(group)]
        values = F.embedding(local_ids + offset, torch.cat(group, dim=0))
        # Preserve each table's padding value and suppress its parameter gradient.
        outputs.append(torch.where((local_ids == 0).unsqueeze(-1), values.detach(), values))
    return torch.cat(outputs, dim=1)
