"""Equivalent DEER interpolation candidates; boundaries retain original arithmetic."""
import torch
from torch.nn import functional as F

def location(x, boundaries, lower, upper):
    clipped = torch.maximum(torch.minimum(x, upper), lower)
    interval = (clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)).sum(-1)
    ids = interval + torch.arange(boundaries.shape[0], device=x.device) * boundaries.shape[1]
    flat = boundaries.reshape(-1)
    left = flat[ids]
    right = flat[ids + 1]
    alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(0., 1.)
    return interval, ids, alpha

def flat_embedding(x, boundaries, knots, lower, upper):
    _, ids, alpha = location(x, boundaries, lower, upper)
    pair = F.embedding(torch.stack((ids, ids + 1), dim=-1), knots.flatten(0, 1))
    return (1. - alpha).unsqueeze(-1) * pair[:, :, 0] + alpha.unsqueeze(-1) * pair[:, :, 1]

def flat_index(x, boundaries, knots, lower, upper):
    _, ids, alpha = location(x, boundaries, lower, upper)
    table = knots.flatten(0, 1)
    left, right = table[ids], table[ids + 1]
    return (1. - alpha).unsqueeze(-1) * left + alpha.unsqueeze(-1) * right

def dense_basis(x, boundaries, knots, lower, upper):
    interval, _, alpha = location(x, boundaries, lower, upper)
    nodes = torch.arange(knots.shape[1], device=x.device)
    basis = (interval.unsqueeze(-1) == nodes) * (1. - alpha).unsqueeze(-1)
    basis = basis + ((interval + 1).unsqueeze(-1) == nodes) * alpha.unsqueeze(-1)
    return torch.bmm(basis.transpose(0, 1), knots).transpose(0, 1)

class _StableKnotLookup(torch.autograd.Function):
    """Direct lookup forward, fused per-knot batch reduction backward."""
    @staticmethod
    def forward(ctx, knots, interval):
        ctx.save_for_backward(interval)
        ctx.knot_shape = knots.shape
        ids = interval + torch.arange(knots.shape[0], device=knots.device) * knots.shape[1]
        return knots.flatten(0, 1)[ids], knots.flatten(0, 1)[ids + 1]

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        (interval,) = ctx.saved_tensors
        nodes = torch.arange(ctx.knot_shape[1], device=interval.device)
        left_mask = interval.unsqueeze(-1) == nodes
        right_mask = (interval + 1).unsqueeze(-1) == nodes
        contributions = torch.where(left_mask.unsqueeze(-1), grad_left.unsqueeze(2), 0.)
        contributions = contributions + torch.where(
            right_mask.unsqueeze(-1), grad_right.unsqueeze(2), 0.)
        return contributions.sum(dim=0), None

def stable_index(x, boundaries, knots, lower, upper):
    interval, _, alpha = location(x, boundaries, lower, upper)
    left, right = _StableKnotLookup.apply(knots, interval)
    return (1. - alpha).unsqueeze(-1) * left + alpha.unsqueeze(-1) * right

def _gather_pair(knots, interval):
    ids = interval + torch.arange(knots.shape[0], device=knots.device) * knots.shape[1]
    return knots.flatten(0, 1)[ids], knots.flatten(0, 1)[ids + 1]

def _knot_contributions(interval, grad_left, grad_right, n_knots):
    nodes = torch.arange(n_knots, device=interval.device)
    return (torch.where((interval.unsqueeze(-1) == nodes).unsqueeze(-1),
                        grad_left.unsqueeze(2), 0.)
            + torch.where(((interval + 1).unsqueeze(-1) == nodes).unsqueeze(-1),
                          grad_right.unsqueeze(2), 0.))

_gather_pair_compiled = torch.compile(_gather_pair, fullgraph=True, dynamic=False)
_knot_contributions_compiled = torch.compile(_knot_contributions, fullgraph=True, dynamic=False)

class _PreciseKnotLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, knots, interval, use_compile):
        ctx.save_for_backward(interval)
        ctx.n_knots = knots.shape[1]
        ctx.use_compile = use_compile
        fn = _gather_pair_compiled if knots.is_cuda and use_compile else _gather_pair
        return fn(knots, interval)

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        (interval,) = ctx.saved_tensors
        fn = _knot_contributions_compiled if grad_left.is_cuda and ctx.use_compile else _knot_contributions
        contributions = fn(interval, grad_left, grad_right, ctx.n_knots)
        # Preserve the original PyTorch batch reduction rather than replacing
        # it with atomic scatter or a compiler-selected reduction tree.
        return contributions.sum(dim=0), None, None

def precise_index(x, boundaries, knots, lower, upper, use_compile=True):
    interval, _, alpha = location(x, boundaries, lower, upper)
    left, right = _PreciseKnotLookup.apply(knots, interval, use_compile)
    # Deliberately retain original FP32 interpolation arithmetic outside fusion.
    return (1. - alpha).unsqueeze(-1) * left + alpha.unsqueeze(-1) * right
