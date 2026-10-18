"""Experimental split compilation: retain interpolation FP32 operation order."""
import torch
from .deer_engineering import location, _StableKnotLookup, precise_index

def _pair(knots, interval):
    return _StableKnotLookup.apply(knots, interval)

_pair_compiled = torch.compile(_pair, fullgraph=True, dynamic=False)

def split_index(x, boundaries, knots, lower, upper):
    if not x.is_cuda:
        return precise_index(x, boundaries, knots, lower, upper, use_compile=False)
    interval, _, alpha = location(x, boundaries, lower, upper)
    left, right = _pair_compiled(knots, interval)
    return (1. - alpha).unsqueeze(-1) * left + alpha.unsqueeze(-1) * right
