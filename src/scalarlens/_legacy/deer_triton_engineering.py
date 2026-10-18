"""Fused FP32 knot interpolation with segmented gradients; experimental until verified."""
import torch
import triton
import triton.language as tl
def _gather_pair(knots, interval):
    ids = interval + torch.arange(knots.shape[0], device=knots.device) * knots.shape[1]
    return knots.flatten(0, 1)[ids], knots.flatten(0, 1)[ids + 1]

def _knot_contributions(interval, grad_left, grad_right, n_knots):
    nodes = torch.arange(n_knots, device=interval.device)
    return (torch.where((interval.unsqueeze(-1) == nodes).unsqueeze(-1), grad_left.unsqueeze(2), 0.)
            + torch.where(((interval + 1).unsqueeze(-1) == nodes).unsqueeze(-1), grad_right.unsqueeze(2), 0.))

def _torch_fallback(x, boundaries, knots, lower, upper):
    clipped = torch.maximum(torch.minimum(x, upper), lower)
    interval = (clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)).sum(-1)
    expanded = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
    left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
    right = expanded.gather(2, (interval + 1).unsqueeze(-1)).squeeze(-1)
    alpha = ((clipped-left)/(right-left).clamp_min(1e-12)).clamp(0., 1.)
    table = knots.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    index = interval.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, knots.size(-1))
    l = table.gather(2, index).squeeze(2)
    r = table.gather(2, index+1).squeeze(2)
    return (1.-alpha).unsqueeze(-1)*l + alpha.unsqueeze(-1)*r

@triton.jit
def _forward(K, I, A, Y, N: tl.constexpr, F: tl.constexpr, S: tl.constexpr,
             D: tl.constexpr, BLOCK: tl.constexpr):
    z = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q, d = z // D, z % D
    f = q % F
    i = tl.load(I + q, q < N, 0)
    a = tl.load(A + q, q < N, 0)
    left = tl.load(K + (f * S + i) * D + d, q < N, 0)
    right = tl.load(K + (f * S + i + 1) * D + d, q < N, 0)
    value = (1. - a) * left + a * right
    tl.store(Y + z, value, z < N * D)

@triton.jit
def _alpha_backward(K, I, G, GA, N: tl.constexpr, F: tl.constexpr,
                    S: tl.constexpr, D: tl.constexpr, DD: tl.constexpr, Q: tl.constexpr):
    q = tl.program_id(0) * Q + tl.arange(0, Q)
    d = tl.arange(0, DD)
    i = tl.load(I + q, q < N, 0)
    f = q % F
    left = tl.load(K + ((f * S + i) * D)[:, None] + d[None, :],
                   (q[:, None] < N) & (d[None, :] < D), 0)
    right = tl.load(K + ((f * S + i + 1) * D)[:, None] + d[None, :],
                    (q[:, None] < N) & (d[None, :] < D), 0)
    g = tl.load(G + q[:, None] * D + d[None, :],
                (q[:, None] < N) & (d[None, :] < D), 0)
    ga = tl.sum(g * right, axis=1) - tl.sum(g * left, axis=1)
    tl.store(GA + q, ga, q < N)

@triton.jit
def _knots_partial(I, A, G, PART, B: tl.constexpr, F: tl.constexpr,
                   S: tl.constexpr, D: tl.constexpr, DD: tl.constexpr, R: tl.constexpr):
    field_node = tl.program_id(0)
    f, node = field_node // S, field_node % S
    part = tl.program_id(1)
    b = part * R + tl.arange(0, R)
    d = tl.arange(0, DD)
    q = b * F + f
    i = tl.load(I + q, b < B, -2)
    a = tl.load(A + q, b < B, 0)
    g = tl.load(G + q[:, None] * D + d[None, :],
                (b[:, None] < B) & (d[None, :] < D), 0)
    left = tl.where((i == node)[:, None], (1. - a[:, None]) * g, 0.)
    right = tl.where((i + 1 == node)[:, None], a[:, None] * g, 0.)
    value = tl.sum(left + right, axis=0)
    tl.store(PART + (part * F * S + field_node) * D + d, value, d < D)

@triton.jit
def _knots_finish(PART, GK, N: tl.constexpr, P: tl.constexpr,
                  PP: tl.constexpr, BLOCK: tl.constexpr):
    z = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.arange(0, PP)
    value = tl.load(PART + p[:, None] * N + z[None, :],
                    (p[:, None] < P) & (z[None, :] < N), 0)
    tl.store(GK + z, tl.sum(value, axis=0), z < N)

class _Interpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, knots, interval, alpha):
        ctx.save_for_backward(knots, interval, alpha)
        knots, interval, alpha = knots.contiguous(), interval.contiguous(), alpha.contiguous()
        b, f = interval.shape
        s, d = knots.shape[1:]
        output = torch.empty((b, f, d), device=knots.device, dtype=knots.dtype)
        if b:
            _forward[(triton.cdiv(b*f*d, 256),)](
                knots, interval, alpha, output, b*f, f, s, d, 256, enable_fp_fusion=False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        knots, interval, alpha = ctx.saved_tensors
        g = grad_output.contiguous()
        if torch.is_grad_enabled():
            # Higher-order autograd uses ordinary differentiable Torch ops.
            # These differentiable helpers intentionally retain Torch autograd.
            left, right = _gather_pair(knots, interval)
            gk = _knot_contributions(interval, g * (1. - alpha).unsqueeze(-1),
                                     g * alpha.unsqueeze(-1), knots.shape[1]).sum(0)
            ga = (g * right).sum(-1) - (g * left).sum(-1)
            return gk, None, ga
        b, f = interval.shape
        s, d = knots.shape[1:]
        ga = torch.empty_like(alpha)
        gk = torch.empty_like(knots)
        if not b:
            gk.zero_()
            return gk, None, ga
        dd = triton.next_power_of_2(d)
        _alpha_backward[(triton.cdiv(b*f, 16),)](
            knots, interval, g, ga, b*f, f, s, d, dd, 16, enable_fp_fusion=False)
        parts = triton.cdiv(b, 256)
        partial = torch.empty((parts, f*s*d), device=g.device, dtype=g.dtype)
        _knots_partial[(f*s, parts)](
            interval, alpha, g, partial, b, f, s, d, dd, 256, enable_fp_fusion=False)
        _knots_finish[(triton.cdiv(f*s*d, 128),)](
            partial, gk, f*s*d, parts, triton.next_power_of_2(parts), 128,
            enable_fp_fusion=False)
        return gk, None, ga

def triton_index(x, boundaries, knots, lower, upper):
    if not x.is_cuda or x.dtype != torch.float32 or knots.dtype != torch.float32:
        return _torch_fallback(x, boundaries, knots, lower, upper)
    clipped = torch.maximum(torch.minimum(x, upper), lower)
    interval = (clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)).sum(-1)
    # Keep the small boundary tensor's original gather/reduction. Flattened
    # boundary indexing creates expensive native atomic index backward here.
    expanded = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
    left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
    right = expanded.gather(2, (interval + 1).unsqueeze(-1)).squeeze(-1)
    alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(0., 1.)
    return _Interpolation.apply(knots, interval, alpha)
