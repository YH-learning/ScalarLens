"""Experimental context-conditioned numerical geometry for AtlasDEER."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn


class ContextDeformableDEEREmbedding(nn.Module):
    """One shared DEER curve sampled at context-conditioned rank offsets.

    Categorical embeddings are softly pooled into a low-rank context vector.
    That vector controls both sampling offsets and mixture weights for K nearby
    points on the shared DEER knot curve. The output is attached as a bounded
    residual whose per-field scale is initialized to zero, so initialization
    exactly reproduces vanilla DEER without duplicating its knot table.

    This is an experimental research candidate, not a paper reproduction.
    """

    requires_context_embeddings = True

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        context_features: Sequence[str],
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_samples: int = 3,
        context_rank: int = 4,
        max_offset_bins: float = 1.0,
        initial_offset_bins: float = 0.75,
        residual_scale_limit: float = 1.0,
        router_init_std: float = 1e-2,
    ):
        super().__init__()
        if initial_edges.ndim != 2 or initial_edges.size(1) < 2:
            raise ValueError(
                "initial_edges must have shape [features, bins + 1]"
            )
        n_bins = initial_edges.size(1) - 1
        if not 0 <= min_width < 1.0 / n_bins:
            raise ValueError("min_width is too large for the number of bins")
        if width_temperature <= 0:
            raise ValueError("width_temperature must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        if boundary_init_noise is not None and boundary_init_noise < 0:
            raise ValueError("boundary_init_noise must be nonnegative")
        if n_samples < 2:
            raise ValueError("n_samples must be at least two")
        if context_rank < 1:
            raise ValueError("context_rank must be positive")
        if max_offset_bins <= 0:
            raise ValueError("max_offset_bins must be positive")
        if not 0 <= initial_offset_bins <= max_offset_bins:
            raise ValueError(
                "initial_offset_bins must be in [0, max_offset_bins]"
            )
        if not 0 < residual_scale_limit <= 1:
            raise ValueError("residual_scale_limit must be in (0, 1]")
        if router_init_std < 0:
            raise ValueError("router_init_std must be nonnegative")

        self.n_features = int(initial_edges.size(0))
        self.n_bins = int(n_bins)
        self.d_embedding = int(d_embedding)
        self.context_features = tuple(str(x) for x in context_features)
        self.min_width = float(min_width)
        self.width_temperature = float(width_temperature)
        self.n_samples = int(n_samples)
        self.context_rank = int(context_rank)
        self.max_offset_bins = float(max_offset_bins)
        self.residual_scale_limit = float(residual_scale_limit)

        lower = initial_edges[:, 0].float()
        upper = initial_edges[:, -1].float()
        widths = (
            initial_edges[:, 1:] - initial_edges[:, :-1]
        ).clamp_min(1e-12)
        fractions = widths / (upper - lower).unsqueeze(-1)
        raw_widths = fractions.log()
        if boundary_init_noise:
            raw_widths = raw_widths + torch.empty_like(
                raw_widths
            ).uniform_(-boundary_init_noise, boundary_init_noise)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.raw_widths = nn.Parameter(raw_widths)
        self.knot_embeddings = nn.Parameter(
            torch.empty(self.n_features, self.n_bins + 1, self.d_embedding)
        )
        nn.init.normal_(self.knot_embeddings, std=float(embedding_init_std))

        self.default_context = nn.Parameter(
            torch.zeros(self.d_embedding)
        )
        self.context_projection = nn.Linear(
            self.d_embedding, self.context_rank
        )
        self.context_attention_logits = nn.Parameter(
            torch.zeros(len(self.context_features))
        )
        self.offset_weight = nn.Parameter(torch.empty(
            self.n_features, self.n_samples, self.context_rank
        ))
        self.gate_weight = nn.Parameter(torch.empty(
            self.n_features, self.n_samples, self.context_rank
        ))
        nn.init.normal_(self.offset_weight, std=float(router_init_std))
        nn.init.normal_(self.gate_weight, std=float(router_init_std))
        self.gate_bias = nn.Parameter(torch.linspace(
            0.25, -0.25, self.n_samples
        ).repeat(self.n_features, 1))
        self.raw_residual_scales = nn.Parameter(
            torch.zeros(self.n_features)
        )
        fixed = torch.linspace(
            -initial_offset_bins,
            initial_offset_bins,
            self.n_samples,
        )
        self.register_buffer("fixed_offset_bins", fixed)

    def current_boundaries(self) -> torch.Tensor:
        fractions = torch.softmax(
            self.raw_widths / self.width_temperature, dim=-1
        )
        if self.min_width > 0:
            fractions = fractions + self.min_width
            fractions = fractions / fractions.sum(dim=-1, keepdim=True)
        cumulative = torch.cumsum(fractions, dim=-1)
        internal_and_upper = self.lower.unsqueeze(-1) + cumulative * (
            self.upper - self.lower
        ).unsqueeze(-1)
        return torch.cat(
            [self.lower.unsqueeze(-1), internal_and_upper], dim=-1
        )

    def residual_scales(self) -> torch.Tensor:
        return self.residual_scale_limit * torch.tanh(
            self.raw_residual_scales
        )

    def _base_and_rank(self, x: torch.Tensor):
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1)
            >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        left = expanded_boundaries.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = (
            (clipped - left) / (right - left).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        expanded_knots = self.knot_embeddings.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.d_embedding
        )
        left_embedding = expanded_knots.gather(2, index).squeeze(2)
        right_embedding = expanded_knots.gather(
            2, index + 1
        ).squeeze(2)
        base = (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )
        rank = (interval.to(alpha.dtype) + alpha) / self.n_bins
        return base, rank

    def _context_vector(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        vectors = []
        if context_embeddings:
            for name in self.context_features:
                value = context_embeddings.get(name)
                if value is None:
                    continue
                if value.ndim != 2 or value.shape != (
                    x.size(0), self.d_embedding
                ):
                    raise ValueError(
                        f"Invalid context embedding shape for {name!r}: "
                        f"{tuple(value.shape)}"
                    )
                vectors.append(value)
        if not vectors:
            return self.default_context.unsqueeze(0).expand(
                x.size(0), -1
            )
        stacked = torch.stack(vectors, dim=1)
        weights = torch.softmax(
            self.context_attention_logits[: len(vectors)], dim=0
        )
        return (stacked * weights.view(1, -1, 1)).sum(dim=1)

    def _interpolate_rank(self, rank: torch.Tensor) -> torch.Tensor:
        scaled = rank.clamp(0.0, 1.0) * self.n_bins
        interval = torch.floor(scaled).long().clamp(
            min=0, max=self.n_bins - 1
        )
        alpha = (scaled - interval.to(scaled.dtype)).clamp(0.0, 1.0)
        knots = self.knot_embeddings.unsqueeze(0).expand(
            rank.size(0), -1, -1, -1
        )
        index = interval.unsqueeze(-1).expand(
            -1, -1, -1, self.d_embedding
        )
        left = knots.gather(2, index)
        right = knots.gather(2, index + 1)
        return (
            (1.0 - alpha).unsqueeze(-1) * left
            + alpha.unsqueeze(-1) * right
        )

    def forward(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        base, rank = self._base_and_rank(x)
        context = self.context_projection(
            self._context_vector(x, context_embeddings)
        )
        offset_delta = torch.tanh(torch.einsum(
            "br,fkr->bfk", context, self.offset_weight
        )) * self.max_offset_bins
        offsets = (
            self.fixed_offset_bins.view(1, 1, -1) + offset_delta
        ) / self.n_bins
        sample_rank = (rank.unsqueeze(-1) + offsets).clamp(0.0, 1.0)
        samples = self._interpolate_rank(sample_rank)
        logits = (
            torch.einsum("br,fkr->bfk", context, self.gate_weight)
            + self.gate_bias.unsqueeze(0)
        )
        weights = torch.softmax(logits, dim=-1)
        alternative = (weights.unsqueeze(-1) * samples).sum(dim=2)
        scale = self.residual_scales().view(1, -1, 1)
        return base + scale * (alternative - base)


class SparseTransportDEEREmbedding(nn.Module):
    """Transport each selected numeric field on one shared DEER curve.

    A train-only diagnostic supplies a sparse mapping from numeric fields to
    their most relevant categorical context.  For an active field ``f`` the
    selected categorical embedding is normalized, then two field-specific
    linear probes predict a bounded displacement and transport gate::

        u' = clip(u + tanh(w_delta^T LN(c)) * delta_max / n_bins)
        e(x, c) = e(u) + sigmoid(w_gate^T LN(c) + b) * (e(u') - e(u))

    The method keeps a single knot table and adds only ``2 * F * (d + 1)``
    router parameters.  Unmapped fields are exactly vanilla DEER.  Unlike the
    pooled-context prototype, every numeric field observes only one explicitly
    diagnosed context, and normalization prevents small categorical embedding
    norms from collapsing the conditional signal.

    This is an experimental research candidate, not a paper reproduction.
    """

    requires_context_embeddings = True

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        numeric_features: Sequence[str],
        field_context_map: Mapping[str, str],
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        max_shift_bins: float = 0.75,
        gate_init: float = 0.25,
        router_init_std: float = 0.05,
        context_norm_eps: float = 1e-8,
        detach_context: bool = False,
    ):
        super().__init__()
        if initial_edges.ndim != 2 or initial_edges.size(1) < 2:
            raise ValueError(
                "initial_edges must have shape [features, bins + 1]"
            )
        self.numeric_features = tuple(str(x) for x in numeric_features)
        if len(self.numeric_features) != initial_edges.size(0):
            raise ValueError(
                "numeric_features must align with initial_edges rows"
            )
        unknown = sorted(set(field_context_map) - set(self.numeric_features))
        if unknown:
            raise ValueError(
                f"field_context_map has unknown numeric fields: {unknown}"
            )
        if len(set(self.numeric_features)) != len(self.numeric_features):
            raise ValueError("numeric_features must be unique")
        n_bins = initial_edges.size(1) - 1
        if not 0 <= min_width < 1.0 / n_bins:
            raise ValueError("min_width is too large for the number of bins")
        if width_temperature <= 0:
            raise ValueError("width_temperature must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        if boundary_init_noise is not None and boundary_init_noise < 0:
            raise ValueError("boundary_init_noise must be nonnegative")
        if max_shift_bins <= 0:
            raise ValueError("max_shift_bins must be positive")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0, 1)")
        if router_init_std < 0:
            raise ValueError("router_init_std must be nonnegative")
        if context_norm_eps <= 0:
            raise ValueError("context_norm_eps must be positive")

        self.n_features = int(initial_edges.size(0))
        self.n_bins = int(n_bins)
        self.d_embedding = int(d_embedding)
        self.min_width = float(min_width)
        self.width_temperature = float(width_temperature)
        self.max_shift_bins = float(max_shift_bins)
        self.context_norm_eps = float(context_norm_eps)
        self.detach_context = bool(detach_context)

        normalized_map = {
            str(field): str(context)
            for field, context in field_context_map.items()
            if context is not None and str(context)
        }
        self.field_context_map = normalized_map
        self.context_features = tuple(dict.fromkeys(normalized_map.values()))
        context_to_index = {
            name: index for index, name in enumerate(self.context_features)
        }
        assignment = torch.full((self.n_features,), -1, dtype=torch.long)
        for field_index, field in enumerate(self.numeric_features):
            context = normalized_map.get(field)
            if context is not None:
                assignment[field_index] = context_to_index[context]
        self.register_buffer("context_assignment", assignment)
        self.register_buffer("active_fields", assignment.ge(0))

        lower = initial_edges[:, 0].float()
        upper = initial_edges[:, -1].float()
        widths = (
            initial_edges[:, 1:] - initial_edges[:, :-1]
        ).clamp_min(1e-12)
        fractions = widths / (upper - lower).unsqueeze(-1)
        raw_widths = fractions.log()
        if boundary_init_noise:
            raw_widths = raw_widths + torch.empty_like(
                raw_widths
            ).uniform_(-boundary_init_noise, boundary_init_noise)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.raw_widths = nn.Parameter(raw_widths)
        self.knot_embeddings = nn.Parameter(
            torch.empty(self.n_features, self.n_bins + 1, self.d_embedding)
        )
        nn.init.normal_(self.knot_embeddings, std=float(embedding_init_std))

        self.shift_weight = nn.Parameter(
            torch.empty(self.n_features, self.d_embedding)
        )
        self.shift_bias = nn.Parameter(torch.zeros(self.n_features))
        self.gate_weight = nn.Parameter(
            torch.empty(self.n_features, self.d_embedding)
        )
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_bias = nn.Parameter(
            torch.full((self.n_features,), gate_logit)
        )
        if router_init_std == 0:
            nn.init.zeros_(self.shift_weight)
            nn.init.zeros_(self.gate_weight)
        else:
            # Initialize a non-zero router without perturbing downstream model
            # initialization in same-seed DEER comparisons.
            rng_state = torch.random.get_rng_state()
            nn.init.normal_(self.shift_weight, std=float(router_init_std))
            nn.init.normal_(self.gate_weight, std=float(router_init_std))
            torch.random.set_rng_state(rng_state)
        with torch.no_grad():
            inactive = ~self.active_fields
            self.shift_weight[inactive].zero_()
            self.shift_bias[inactive].zero_()
            self.gate_weight[inactive].zero_()

    def current_boundaries(self) -> torch.Tensor:
        fractions = torch.softmax(
            self.raw_widths / self.width_temperature, dim=-1
        )
        if self.min_width > 0:
            fractions = fractions + self.min_width
            fractions = fractions / fractions.sum(dim=-1, keepdim=True)
        cumulative = torch.cumsum(fractions, dim=-1)
        internal_and_upper = self.lower.unsqueeze(-1) + cumulative * (
            self.upper - self.lower
        ).unsqueeze(-1)
        return torch.cat(
            [self.lower.unsqueeze(-1), internal_and_upper], dim=-1
        )

    def _base_and_rank(self, x: torch.Tensor):
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1)
            >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        left = expanded_boundaries.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = (
            (clipped - left) / (right - left).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        expanded_knots = self.knot_embeddings.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.d_embedding
        )
        left_embedding = expanded_knots.gather(2, index).squeeze(2)
        right_embedding = expanded_knots.gather(2, index + 1).squeeze(2)
        base = (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )
        rank = (interval.to(alpha.dtype) + alpha) / self.n_bins
        return base, rank

    def _interpolate_rank(self, rank: torch.Tensor) -> torch.Tensor:
        scaled = rank.clamp(0.0, 1.0) * self.n_bins
        interval = torch.floor(scaled).long().clamp(
            min=0, max=self.n_bins - 1
        )
        alpha = (scaled - interval.to(scaled.dtype)).clamp(0.0, 1.0)
        knots = self.knot_embeddings.unsqueeze(0).expand(
            rank.size(0), -1, -1, -1
        )
        index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.d_embedding
        )
        left = knots.gather(2, index).squeeze(2)
        right = knots.gather(2, index + 1).squeeze(2)
        return (
            (1.0 - alpha).unsqueeze(-1) * left
            + alpha.unsqueeze(-1) * right
        )

    def _field_contexts(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        output = x.new_zeros(
            x.size(0), self.n_features, self.d_embedding
        )
        if not self.context_features:
            return output
        if not context_embeddings:
            raise ValueError(
                "SparseTransportDEEREmbedding requires context embeddings"
            )
        vectors = []
        for name in self.context_features:
            value = context_embeddings.get(name)
            if value is None:
                raise ValueError(f"Missing context embedding for {name!r}")
            if value.ndim != 2 or value.shape != (
                x.size(0), self.d_embedding
            ):
                raise ValueError(
                    f"Invalid context embedding shape for {name!r}: "
                    f"{tuple(value.shape)}"
                )
            if self.detach_context:
                value = value.detach()
            vectors.append(value)
        stacked = torch.stack(vectors, dim=1)
        active_indices = self.active_fields.nonzero(as_tuple=False).flatten()
        selected = stacked[:, self.context_assignment[active_indices], :]
        selected = torch.nn.functional.layer_norm(
            selected,
            (self.d_embedding,),
            eps=self.context_norm_eps,
        )
        output[:, active_indices, :] = selected
        return output

    def transport_statistics(
        self,
        x: torch.Tensor,
        context_embeddings: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._field_contexts(x, context_embeddings)
        shift = torch.tanh(
            torch.einsum("bfd,fd->bf", context, self.shift_weight)
            + self.shift_bias
        ) * self.max_shift_bins
        gate = torch.sigmoid(
            torch.einsum("bfd,fd->bf", context, self.gate_weight)
            + self.gate_bias
        )
        active = self.active_fields.to(x.dtype).unsqueeze(0)
        return shift * active, gate * active

    def forward(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        base, rank = self._base_and_rank(x)
        shift, gate = self.transport_statistics(x, context_embeddings or {})
        transported = self._interpolate_rank(
            rank + shift / self.n_bins
        )
        return base + gate.unsqueeze(-1) * (transported - base)
