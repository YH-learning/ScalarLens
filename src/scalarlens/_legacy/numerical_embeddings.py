"""Numerical feature embeddings for FuxiCTR.

The RTDL-backed methods are thin adapters around the official
``rtdl-num-embeddings`` package. AutoDis follows Guo et al. (KDD 2021).
The research reproductions below cite their fidelity boundaries in docstrings.
"""

from collections import OrderedDict
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

try:
    from .atlasdeer import (
        ContextDeformableDEEREmbedding,
        SparseTransportDEEREmbedding,
    )
except ImportError:  # plugin loader imports this file outside its package
    import importlib.util

    _atlas_path = Path(__file__).resolve().parent / "atlasdeer.py"
    _atlas_spec = importlib.util.spec_from_file_location(
        "_fuxictr_numemb_atlasdeer", _atlas_path
    )
    if _atlas_spec is None or _atlas_spec.loader is None:
        raise ImportError(f"Cannot load AtlasDEER implementation from {_atlas_path}")
    _atlas_module = importlib.util.module_from_spec(_atlas_spec)
    _atlas_spec.loader.exec_module(_atlas_module)
    ContextDeformableDEEREmbedding = (
        _atlas_module.ContextDeformableDEEREmbedding
    )
    SparseTransportDEEREmbedding = (
        _atlas_module.SparseTransportDEEREmbedding
    )

try:
    from .coupled_scattering import CoupledScatteringEmbedding
except ImportError:  # file-based registry loading outside the src package
    import importlib.util

    _scatter_path = Path(__file__).resolve().parent / "coupled_scattering.py"
    _scatter_spec = importlib.util.spec_from_file_location(
        "_fuxictr_numemb_coupled_scattering", _scatter_path
    )
    if _scatter_spec is None or _scatter_spec.loader is None:
        raise ImportError(f"Cannot load coupled scattering from {_scatter_path}")
    _scatter_module = importlib.util.module_from_spec(_scatter_spec)
    _scatter_spec.loader.exec_module(_scatter_module)
    CoupledScatteringEmbedding = _scatter_module.CoupledScatteringEmbedding

try:
    from .fast_coupled_embedding import FastCoupledEmbedding, InteractionResidualEmbedding
except ImportError:
    import importlib.util

    _fast_path = Path(__file__).resolve().parent / "fast_coupled_embedding.py"
    _fast_spec = importlib.util.spec_from_file_location("_fuxictr_numemb_fast_coupled", _fast_path)
    if _fast_spec is None or _fast_spec.loader is None:
        raise ImportError(f"Cannot load fast coupled embedding from {_fast_path}")
    _fast_module = importlib.util.module_from_spec(_fast_spec)
    _fast_spec.loader.exec_module(_fast_module)
    FastCoupledEmbedding = _fast_module.FastCoupledEmbedding
    InteractionResidualEmbedding = _fast_module.InteractionResidualEmbedding

from fuxictr.pytorch.layers.embeddings.numerical_embedding import (
    register_numerical_embedding,
)
from fuxictr.pytorch.layers import FeatureEmbeddingDict

try:
    import rtdl_num_embeddings as rtdl_num
except ImportError as exc:  # pragma: no cover - environment setup failure
    raise ImportError(
        "Reference embeddings require rtdl-num-embeddings==0.0.12. "
        "Install the repository requirements.txt before training."
    ) from exc


SUPPORTED_NUM_EMBEDDINGS = (
    "field",
    "linear",
    "linear_relu",
    "autodis",
    "piecewise_linear",
    "piecewise_linear_relu",
    "periodic",
    "narydis_lite",
    "mnse",
    "bspline",
    "spline_response_init",
    "quantile_spline",
    "adaptive_spline",
    "narydis_full",
    "deer",
    "deer_scattering",
    "deer_scattering_fast",
    "deer_story_control",
    "deer_interaction_residual",
    "liftdeer",
    "deer_graphframe",
    "deer_graphres",
    "deer_moe",
    "deer_deform",
    "deer_transport",
    "deer_response",
    "deer_response_init",
    "atlas",
    "atlas_sobolev",
    "atlas_hp",
    "atlas_residual",
    "deer_jump",
    "deer_crossbasis",
    "deer_residual",
    "deer_residual_lr",
    "deer_multigrid",
    "deer_switch",
    "deer_dualchart",
    "deer_adapter",
    "deer_schwarz",
    "wave_spline",
    "wavespec",
    "deer_sid",
    "deer_rqvae",
    "deer_prq",
    "deer_prq_gate",
    "deer_prq_taskgate",
    "deer_prq_sid",
    "deer_multigrid_lr",
    "deer_subspace",
    "deer_diffusion",
    "deer_trend",
    "deer_asinh",
    "deer_atom",
    "deer_mobius",
    "deer_mobius_shared",
    "deer_quadratic",
    "deer_hermite",
    "deer_weno",
    "deer_bubble",
    "deer_multires",
    "deer_keras",
    "dae",
    "daes_gate",
    "daes_tran",
    "plr",
    "quantile_bucket",
    "log_squared_bucket",
)


class FieldEmbedding(nn.Module):
    """FuxiCTR's historical numeric path: one vector scaled by each value."""

    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        nn.init.normal_(self.weight, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight


class AutoDisEmbedding(nn.Module):
    """Vectorized AutoDis for multiple numerical fields.

    For field ``j``, the module computes
    ``h=LeakyReLU(x*w)``, ``p=softmax((h@W+alpha*h)/tau)`` and ``p@M``.
    ``M`` is the field-specific table of meta-embeddings.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_meta_embeddings: int = 16,
        temperature: float = 0.1,
        residual_weight: float = 1.0,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        if n_meta_embeddings < 2:
            raise ValueError("n_meta_embeddings must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.residual_weight = float(residual_weight)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.proj_weight = nn.Parameter(torch.empty(n_features, n_meta_embeddings))
        self.proj_matrix = nn.Parameter(
            torch.empty(n_features, n_meta_embeddings, n_meta_embeddings)
        )
        self.meta_embeddings = nn.Parameter(
            torch.empty(n_features, n_meta_embeddings, d_embedding)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj_weight)
        for matrix in self.proj_matrix:
            nn.init.xavier_uniform_(matrix)
        nn.init.normal_(self.meta_embeddings, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.leaky_relu(x.unsqueeze(-1) * self.proj_weight)
        logits = torch.einsum("fnm,bfm->bfn", self.proj_matrix, hidden)
        logits = logits + self.residual_weight * hidden
        assignment = torch.softmax(logits / self.temperature, dim=-1)
        return torch.einsum("bfn,fnd->bfd", assignment, self.meta_embeddings)


class QuantileBucketEmbedding(nn.Module):
    """Hard quantile-bucket lookup baseline with feature-specific tables."""

    def __init__(self, bins: List[torch.Tensor], d_embedding: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(len(edges) - 1, d_embedding) for edges in bins]
        )
        for i, edges in enumerate(bins):
            self.register_buffer(f"edges_{i}", edges.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, table in enumerate(self.embeddings):
            edges = getattr(self, f"edges_{i}")
            index = torch.bucketize(x[:, i].contiguous(), edges[1:-1], right=False)
            outputs.append(table(index))
        return torch.stack(outputs, dim=1)


class LogSquaredBucketEmbedding(nn.Module):
    """Criteo log-squared bucketing followed by exact train-vocabulary lookup.

    The transform matches the RecZoo/FuxiCTR Criteo_x4 recipe: integer values
    greater than two become floor(log(x) squared); all other integers are kept.
    Each transformed token is then mapped through a field-specific vocabulary.
    Non-positive, rare, unseen, and non-finite values use that field's OOV id.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        vocabularies: Dict[str, Dict[str, int]],
        d_embedding: int,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        self.numeric_features = tuple(numeric_features)
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        self.embeddings = nn.ModuleList()
        for i, feature in enumerate(self.numeric_features):
            if feature not in vocabularies:
                raise ValueError(f"Missing vocabulary for numeric feature {feature!r}")
            vocabulary = vocabularies[feature]
            if "__OOV__" not in vocabulary:
                raise ValueError(f"Vocabulary for {feature!r} has no __OOV__ id")
            token_to_id = {
                int(token): int(index)
                for token, index in vocabulary.items()
                if token not in {"__PAD__", "__OOV__"} and int(token) > 0
            }
            min_token = min(token_to_id, default=0)
            max_token = max(token_to_id, default=0)
            oov_id = int(vocabulary["__OOV__"])
            lookup = torch.full(
                (max_token - min_token + 1,), oov_id, dtype=torch.long
            )
            for token, index in token_to_id.items():
                lookup[token - min_token] = index
            self.register_buffer(f"lookup_{i}", lookup)
            self.register_buffer(
                f"min_token_{i}", torch.tensor(min_token, dtype=torch.long)
            )
            self.register_buffer(
                f"oov_id_{i}", torch.tensor(oov_id, dtype=torch.long)
            )
            vocab_size = max(int(index) for index in vocabulary.values()) + 1
            table = nn.Embedding(vocab_size, d_embedding, padding_idx=0)
            nn.init.normal_(table.weight, std=embedding_init_std)
            with torch.no_grad():
                table.weight[0].zero_()
            self.embeddings.append(table)

    @staticmethod
    def bucket_tokens(x: torch.Tensor) -> torch.Tensor:
        safe = torch.nan_to_num(
            x.detach().double(), nan=0.0, posinf=0.0, neginf=0.0
        )
        integer = torch.trunc(safe).long()
        positive = integer > 2
        transformed = torch.floor(
            torch.log(integer.clamp_min(1).double()).square()
        ).long()
        return torch.where(positive, transformed, integer)

    def bucket_indices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != len(self.numeric_features):
            raise ValueError(
                f"Expected [batch, {len(self.numeric_features)}], got {tuple(x.shape)}"
            )
        tokens = self.bucket_tokens(x)
        indices = []
        for i in range(len(self.numeric_features)):
            lookup = getattr(self, f"lookup_{i}")
            min_token = getattr(self, f"min_token_{i}")
            oov_id = getattr(self, f"oov_id_{i}")
            positions = tokens[:, i] - min_token
            valid = (positions >= 0) & (positions < lookup.numel())
            safe_positions = positions.clamp(0, lookup.numel() - 1)
            mapped = lookup[safe_positions]
            indices.append(torch.where(valid, mapped, oov_id))
        return torch.stack(indices, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = self.bucket_indices(x)
        return torch.stack(
            [table(indices[:, i]) for i, table in enumerate(self.embeddings)],
            dim=1,
        )


def _to_nary_digits(
    x: torch.Tensor,
    base: int,
    max_digits: int,
    multiplier: float,
    signed: bool,
) -> torch.Tensor:
    """Convert values to little-endian base-n digits without preprocessing state."""

    if base < 2:
        raise ValueError("All N-ary bases must be at least 2")
    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    safe_x = torch.nan_to_num(
        x.detach() * multiplier,
        nan=0.0,
        posinf=9.0e18,
        neginf=-9.0e18,
    ).clamp(min=-9.0e18, max=9.0e18)
    integer = torch.round(safe_x).long()
    remainder = integer.abs()
    digits = []
    for _ in range(max_digits):
        digits.append(torch.remainder(remainder, base))
        remainder = torch.div(remainder, base, rounding_mode="floor")
    encoded = torch.stack(digits, dim=-1).to(dtype=x.dtype)
    if signed:
        encoded = torch.cat(
            [encoded, (integer < 0).to(dtype=x.dtype).unsqueeze(-1)], dim=-1
        )
    return encoded


def _normalized_digit_weights(digits: torch.Tensor) -> torch.Tensor:
    denominator = digits.sum(dim=-1, keepdim=True)
    weights = digits / denominator.clamp_min(1.0)
    fallback = torch.zeros_like(weights)
    fallback[..., 0] = 1.0
    return torch.where(denominator > 0, weights, fallback)


class NaryDisLiteEmbedding(nn.Module):
    """Efficient multi-base N-ary encoding inspired by NaryDis and MNSE.

    This engineering variant keeps deterministic hybrid-base encoding and
    field/base-specific positional embeddings, but replaces hierarchical
    attention and contrastive regularization with normalized pooling.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        bases: Sequence[int] = (2, 3, 5, 7, 10),
        max_digits: int = 12,
        multiplier: float = 100.0,
        signed: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()
        self.bases = tuple(int(base) for base in bases)
        if not self.bases or any(base < 2 for base in self.bases):
            raise ValueError("bases must contain integers greater than one")
        if pooling not in ("mean", "sum"):
            raise ValueError("pooling must be 'mean' or 'sum'")
        self.max_digits = int(max_digits)
        self.multiplier = float(multiplier)
        self.signed = bool(signed)
        self.pooling = pooling
        sequence_length = self.max_digits + int(self.signed)
        self.positional_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(n_features, sequence_length, d_embedding)
                )
                for _ in self.bases
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embeddings in self.positional_embeddings:
            nn.init.normal_(embeddings, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ary_representations = []
        for base, embeddings in zip(self.bases, self.positional_embeddings):
            digits = _to_nary_digits(
                x, base, self.max_digits, self.multiplier, self.signed
            )
            weights = _normalized_digit_weights(digits)
            ary_representations.append(
                torch.einsum("bfl,fld->bfd", weights, embeddings)
            )
        stacked = torch.stack(ary_representations, dim=2)
        if self.pooling == "sum":
            return stacked.sum(dim=2)
        return stacked.mean(dim=2)


class BSplineEmbedding(nn.Module):
    """B-spline function-basis encoding followed by learned coefficients.

    Inputs are clipped to each feature's training range, mapped to ``[0, 1]``,
    expanded with a clamped uniform B-spline basis, and aggregated with a
    field-specific coefficient embedding table.
    """

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        d_embedding: int,
        n_basis: int = 8,
        degree: int = 3,
    ):
        super().__init__()
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.n_basis <= self.degree:
            raise ValueError("n_basis must be greater than degree")
        self.register_buffer("lower", lower.float())
        self.register_buffer("upper", upper.float())
        n_internal = self.n_basis - self.degree - 1
        internal = (
            torch.linspace(0.0, 1.0, n_internal + 2)[1:-1]
            if n_internal > 0
            else torch.empty(0)
        )
        knots = torch.cat(
            [
                torch.zeros(self.degree + 1),
                internal,
                torch.ones(self.degree + 1),
            ]
        )
        self.register_buffer("knots", knots)
        self.coefficients = nn.Parameter(
            torch.empty(len(lower), self.n_basis, d_embedding)
        )
        nn.init.normal_(self.coefficients, std=1e-4)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        denominator = (self.upper - self.lower).clamp_min(1e-12)
        unit_x = ((x - self.lower) / denominator).clamp(0.0, 1.0)
        expanded = unit_x.unsqueeze(-1)
        basis = (
            (expanded >= self.knots[:-1]) & (expanded < self.knots[1:])
        ).to(dtype=x.dtype)
        for order in range(1, self.degree + 1):
            left_denominator = self.knots[order:-1] - self.knots[: -order - 1]
            right_denominator = self.knots[order + 1 :] - self.knots[1:-order]
            left_scale = torch.where(
                left_denominator > 0,
                (expanded - self.knots[: -order - 1])
                / left_denominator.clamp_min(1e-12),
                torch.zeros_like(left_denominator),
            )
            right_scale = torch.where(
                right_denominator > 0,
                (self.knots[order + 1 :] - expanded)
                / right_denominator.clamp_min(1e-12),
                torch.zeros_like(right_denominator),
            )
            basis = left_scale * basis[..., :-1] + right_scale * basis[..., 1:]
        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.0
        return torch.where((unit_x >= 1.0).unsqueeze(-1), endpoint, basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfn,fnd->bfd", self.basis(x), self.coefficients)



class ResponseInitializedSplineEmbedding(BSplineEmbedding):
    """B-spline with one control axis initialized by train-only response geometry.

    The empirical-Bayes response curve lives on quantile edges, whereas the
    B-spline controls live at uniform-raw Greville abscissae. The response is
    first transported between those coordinates, standardized per field, and
    then replaces exactly one random control-table direction. Orthogonal random
    components are preserved. The method adds no parameters or inference cost
    relative to the underlying B-spline.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_basis: int = 8,
        degree: int = 3,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        embedding_init_std: float = 1e-4,
    ):
        if response_scale < 0:
            raise ValueError("response_scale must be nonnegative")
        if response_clip <= 0:
            raise ValueError("response_clip must be positive")
        if degree < 1:
            raise ValueError("degree must be positive for response initialization")
        super().__init__(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        expected = tuple(initial_edges.shape)
        if tuple(response.shape) != expected:
            raise ValueError(
                f"response_knots must have shape {expected}, got {tuple(response.shape)}"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")
        greville = torch.stack(
            [
                self.knots[i + 1 : i + self.degree + 1].mean()
                for i in range(self.n_basis)
            ]
        )
        raw_controls = self.lower.unsqueeze(-1) + greville.unsqueeze(0) * (
            self.upper - self.lower
        ).unsqueeze(-1)
        response_controls = []
        for field in range(initial_edges.size(0)):
            edges = initial_edges[field]
            points = raw_controls[field]
            interval = torch.bucketize(points.contiguous(), edges[1:-1])
            left = edges[interval]
            right = edges[interval + 1]
            alpha = ((points - left) / (right - left).clamp_min(1e-12)).clamp(
                0.0, 1.0
            )
            values = (
                (1.0 - alpha) * response[field, interval]
                + alpha * response[field, interval + 1]
            )
            response_controls.append(values)
        response_controls = torch.stack(response_controls)
        centered = response_controls - response_controls.mean(
            dim=-1, keepdim=True
        )
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        standardized = centered / rms.clamp_min(1e-6)
        standardized = standardized.clamp(
            -float(response_clip), float(response_clip)
        )
        standardized = torch.where(
            rms <= 1e-6, torch.zeros_like(standardized), standardized
        )
        self.register_buffer(
            "standardized_response_controls", standardized.float()
        )
        direction = torch.ones(d_embedding, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(1e-12)
        self.register_buffer("response_direction", direction)
        self.response_scale = float(response_scale)
        with torch.no_grad():
            random_parallel = torch.einsum(
                "fmd,d->fm", self.coefficients, direction
            )
            target_parallel = (
                self.response_scale
                * float(embedding_init_std)
                * float(d_embedding) ** 0.5
                * standardized
            )
            self.coefficients.add_(
                (target_parallel - random_parallel).unsqueeze(-1) * direction
            )




class QuantileSplineEmbedding(nn.Module):
    """Cubic B-splines on a fixed or learnable empirical-rank coordinate.

    Raw values are first transported through a piecewise-linear empirical CDF
    defined by training-only quantile edges.  A clamped B-spline basis is then
    evaluated in rank space and mapped to the shared embedding dimension.  In
    adaptive mode the ordered raw-value mesh is trained exactly as in DEER.

    Initialization consumes the same random knot table as a same-bin DEER and
    deterministically projects it to Greville abscissae.  Thus downstream model
    parameters remain seed-aligned while the spline starts as a smooth
    approximation of the corresponding random DEER curve.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        n_basis: int = 8,
        degree: int = 3,
        learnable_boundaries: bool = True,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        if initial_edges.ndim != 2 or initial_edges.size(1) < 2:
            raise ValueError("initial_edges must have shape [features, bins + 1]")
        self.n_bins = int(initial_edges.size(1) - 1)
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        self.learnable_boundaries = bool(learnable_boundaries)
        self.min_width = float(min_width)
        self.width_temperature = float(width_temperature)
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if self.n_basis <= self.degree:
            raise ValueError("n_basis must be greater than degree")
        if not 0 <= self.min_width < 1.0 / self.n_bins:
            raise ValueError("min_width is too large for the number of bins")
        if self.width_temperature <= 0:
            raise ValueError("width_temperature must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")

        edges = initial_edges.float()
        self.register_buffer("lower", edges[:, 0].clone())
        self.register_buffer("upper", edges[:, -1].clone())
        widths = (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-12)
        fractions = widths / (self.upper - self.lower).unsqueeze(-1)
        raw_widths = fractions.log()
        if self.learnable_boundaries:
            self.raw_widths = nn.Parameter(raw_widths)
        else:
            self.register_buffer("raw_widths", raw_widths)

        n_internal = self.n_basis - self.degree - 1
        internal = (
            torch.linspace(0.0, 1.0, n_internal + 2)[1:-1]
            if n_internal > 0
            else torch.empty(0)
        )
        spline_knots = torch.cat(
            [
                torch.zeros(self.degree + 1),
                internal,
                torch.ones(self.degree + 1),
            ]
        )
        self.register_buffer("spline_knots", spline_knots)

        # Consume exactly the DEER knot-table RNG stream, then project those
        # knots to B-spline controls without any additional randomness.
        reference_knots = torch.empty(
            edges.size(0), self.n_bins + 1, d_embedding
        )
        nn.init.normal_(reference_knots, std=float(embedding_init_std))
        greville = torch.stack(
            [
                spline_knots[i + 1 : i + self.degree + 1].mean()
                for i in range(self.n_basis)
            ]
        )
        positions = greville * self.n_bins
        left_index = positions.floor().long().clamp_max(self.n_bins)
        right_index = (left_index + 1).clamp_max(self.n_bins)
        alpha = (positions - left_index).view(1, self.n_basis, 1)
        controls = (
            (1.0 - alpha) * reference_knots[:, left_index, :]
            + alpha * reference_knots[:, right_index, :]
        )
        self.coefficients = nn.Parameter(controls)

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
        return torch.cat([self.lower.unsqueeze(-1), internal_and_upper], dim=-1)

    def rank_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded.gather(2, (interval + 1).unsqueeze(-1)).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        return (interval.to(x.dtype) + alpha) / self.n_bins

    def basis_from_rank(self, rank: torch.Tensor) -> torch.Tensor:
        expanded = rank.clamp(0.0, 1.0).unsqueeze(-1)
        basis = (
            (expanded >= self.spline_knots[:-1])
            & (expanded < self.spline_knots[1:])
        ).to(dtype=rank.dtype)
        for order in range(1, self.degree + 1):
            left_denominator = (
                self.spline_knots[order:-1]
                - self.spline_knots[: -order - 1]
            )
            right_denominator = (
                self.spline_knots[order + 1 :]
                - self.spline_knots[1:-order]
            )
            left_scale = torch.where(
                left_denominator > 0,
                (expanded - self.spline_knots[: -order - 1])
                / left_denominator.clamp_min(1e-12),
                torch.zeros_like(left_denominator),
            )
            right_scale = torch.where(
                right_denominator > 0,
                (self.spline_knots[order + 1 :] - expanded)
                / right_denominator.clamp_min(1e-12),
                torch.zeros_like(right_denominator),
            )
            basis = left_scale * basis[..., :-1] + right_scale * basis[..., 1:]
        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.0
        return torch.where((rank >= 1.0).unsqueeze(-1), endpoint, basis)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis_from_rank(self.rank_coordinate(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfn,fnd->bfd", self.basis(x), self.coefficients)


class NaryDisEmbedding(nn.Module):
    """Paper-aligned NaryDis with hierarchical attention and InfoNCE.

    Equations (3)--(10) of Chen et al. (CIKM 2022) are implemented: gated
    intra-ary pooling, field/value-aware inter-ary attention, and an optional
    contrastive regularizer exposed through ``get_auxiliary_loss``.
    """

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        d_embedding: int,
        bases: Sequence[int] = tuple(range(2, 11)),
        max_digits: int = 12,
        multiplier: float = 100.0,
        signed: bool = True,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 5e-5,
        attention_hidden_dims: Sequence[int] = (80, 40),
        ssl_hidden_dim: int = 64,
        ssl_temperature: float = 0.1,
        ssl_weight: float = 0.0,
        ssl_max_samples: int = 512,
        mask_zero_digits: bool = True,
    ):
        super().__init__()
        self.bases = tuple(int(base) for base in bases)
        if len(self.bases) < 2 or any(base < 2 for base in self.bases):
            raise ValueError("narydis_full requires at least two valid bases")
        if temperature <= 0 or min_temperature <= 0:
            raise ValueError("NaryDis temperatures must be positive")
        if ssl_temperature <= 0 or ssl_weight < 0:
            raise ValueError("Invalid NaryDis SSL settings")
        self.max_digits = int(max_digits)
        self.multiplier = float(multiplier)
        self.signed = bool(signed)
        self.initial_temperature = float(temperature)
        self.min_temperature = float(min_temperature)
        self.temperature_decay = float(temperature_decay)
        self.ssl_temperature = float(ssl_temperature)
        self.ssl_weight = float(ssl_weight)
        self.ssl_max_samples = int(ssl_max_samples)
        if self.ssl_max_samples < 0:
            raise ValueError("ssl_max_samples must be non-negative")
        self.mask_zero_digits = bool(mask_zero_digits)
        self.register_buffer("lower", lower.float())
        self.register_buffer("upper", upper.float())
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        n_features = len(lower)
        sequence_length = self.max_digits + int(self.signed)
        self.positional_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(n_features, sequence_length, d_embedding)
                )
                for _ in self.bases
            ]
        )
        self.intra_ary_gates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_features, sequence_length))
                for _ in self.bases
            ]
        )
        self.field_vectors = nn.Parameter(torch.empty(n_features, d_embedding))
        hidden_dims = tuple(int(value) for value in attention_hidden_dims)
        if len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("attention_hidden_dims must contain two positive values")
        self.inter_ary_attention = nn.Sequential(
            nn.Linear(3 * d_embedding, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )
        self.ssl_projector = nn.Sequential(
            nn.Linear(d_embedding, ssl_hidden_dim),
            nn.ReLU(),
            nn.Linear(ssl_hidden_dim, ssl_hidden_dim),
            nn.ReLU(),
            nn.Linear(ssl_hidden_dim, ssl_hidden_dim),
        )
        self._last_auxiliary_loss: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embeddings in self.positional_embeddings:
            nn.init.normal_(embeddings, std=1e-4)
        for gates in self.intra_ary_gates:
            nn.init.ones_(gates)
        nn.init.normal_(self.field_vectors, std=1e-4)
        for module in (self.inter_ary_attention, self.ssl_projector):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _current_temperature(self) -> float:
        annealed = self.initial_temperature - self.temperature_decay * int(
            self.training_steps.item()
        )
        return max(self.min_temperature, annealed)

    def _intra_ary_representations(self, x: torch.Tensor) -> torch.Tensor:
        representations = []
        temperature = self._current_temperature()
        for base, embeddings, gates in zip(
            self.bases, self.positional_embeddings, self.intra_ary_gates
        ):
            digits = _to_nary_digits(
                x, base, self.max_digits, self.multiplier, self.signed
            )
            logits = digits * gates.unsqueeze(0) / temperature
            if self.mask_zero_digits:
                active = digits > 0
                empty = ~active.any(dim=-1, keepdim=True)
                fallback = torch.zeros_like(active)
                fallback[..., 0] = True
                active = active | (empty & fallback)
                logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1)
            representations.append(
                torch.einsum("bfl,fld->bfd", weights, embeddings)
            )
        return torch.stack(representations, dim=2)

    def _contrastive_loss(self, ary_embeddings: torch.Tensor) -> torch.Tensor:
        if self.ssl_max_samples and ary_embeddings.size(0) > self.ssl_max_samples:
            sample = torch.randperm(
                ary_embeddings.size(0), device=ary_embeddings.device
            )[: self.ssl_max_samples]
            ary_embeddings = ary_embeddings[sample]
        pair = torch.randperm(len(self.bases), device=ary_embeddings.device)[:2]
        first = self.ssl_projector(ary_embeddings[:, :, pair[0], :])
        second = self.ssl_projector(ary_embeddings[:, :, pair[1], :])
        first = F.normalize(first, dim=-1)
        second = F.normalize(second, dim=-1)
        logits = torch.einsum("bfd,cfd->fbc", first, second)
        logits = logits / self.ssl_temperature
        batch_size, n_features = first.shape[:2]
        labels = torch.arange(batch_size, device=first.device).repeat(n_features)
        return F.cross_entropy(logits.reshape(n_features * batch_size, batch_size), labels)

    def get_auxiliary_loss(self) -> torch.Tensor:
        if self._last_auxiliary_loss is None:
            return self.field_vectors.new_zeros(())
        return self._last_auxiliary_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ary_embeddings = self._intra_ary_representations(x)
        normalized_x = (
            (x - self.lower) / (self.upper - self.lower).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        value_aware = normalized_x.unsqueeze(-1) * self.field_vectors.unsqueeze(0)
        expanded_value = value_aware.unsqueeze(2).expand_as(ary_embeddings)
        attention_input = torch.cat(
            [ary_embeddings, expanded_value, ary_embeddings * expanded_value], dim=-1
        )
        scores = self.inter_ary_attention(attention_input).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        output = (weights.unsqueeze(-1) * ary_embeddings).sum(dim=2)
        if self.training and self.ssl_weight > 0:
            self._last_auxiliary_loss = self.ssl_weight * self._contrastive_loss(
                ary_embeddings
            )
        else:
            self._last_auxiliary_loss = output.new_zeros(())
        if self.training:
            self.training_steps.add_(1)
        return output


try:
    from .deer_engineering import precise_index as _deer_flat_index
except ImportError:
    import importlib.util
    _deer_path = Path(__file__).resolve().parent / "deer_engineering.py"
    _deer_spec = importlib.util.spec_from_file_location("_fuxictr_numemb_deer_engineering", _deer_path)
    _deer_module = importlib.util.module_from_spec(_deer_spec)
    _deer_spec.loader.exec_module(_deer_module)
    _deer_flat_index = _deer_module.precise_index


class DEEREmbedding(nn.Module):
    """Dynamic Explicit Embedding Representation with learnable bucket widths.

    Positive normalized widths parameterize ordered boundaries end-to-end. A
    value is represented by linear interpolation between the two embeddings
    adjacent to its active interval. This follows the learnable-boundary and
    interpolation description in Cheng (CIKM 2022); no official code exists.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__()
        if initial_edges.ndim != 2 or initial_edges.size(1) < 2:
            raise ValueError("initial_edges must have shape [features, bins + 1]")
        if not 0 <= min_width < 1.0 / (initial_edges.size(1) - 1):
            raise ValueError("min_width is too large for the number of bins")
        if width_temperature <= 0:
            raise ValueError("width_temperature must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        if boundary_init_noise is not None and boundary_init_noise < 0:
            raise ValueError("boundary_init_noise must be nonnegative")
        self.n_bins = initial_edges.size(1) - 1
        self.min_width = float(min_width)
        self.width_temperature = float(width_temperature)
        lower = initial_edges[:, 0]
        upper = initial_edges[:, -1]
        initial_widths = (initial_edges[:, 1:] - initial_edges[:, :-1]).clamp_min(
            1e-12
        )
        initial_fractions = initial_widths / (upper - lower).unsqueeze(-1)
        self.register_buffer("lower", lower.float())
        self.register_buffer("upper", upper.float())
        raw_widths = initial_fractions.log()
        if boundary_init_noise is not None:
            noise = torch.empty_like(raw_widths)
            nn.init.uniform_(noise, -1.0, 1.0)
            raw_widths = raw_widths + noise * float(boundary_init_noise)
        self.raw_widths = nn.Parameter(raw_widths)
        self.knot_embeddings = nn.Parameter(
            torch.empty(initial_edges.size(0), self.n_bins + 1, d_embedding)
        )
        nn.init.normal_(
            self.knot_embeddings, std=float(embedding_init_std)
        )

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
        return torch.cat([self.lower.unsqueeze(-1), internal_and_upper], dim=-1)
    def current_knot_embeddings(self) -> torch.Tensor:
        return self.knot_embeddings


    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        knots = self.current_knot_embeddings()
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(0.0, 1.0)
        expanded_embeddings = knots.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, knots.size(-1)
        )
        left_embedding = expanded_embeddings.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_embeddings.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (1.0 - alpha).unsqueeze(-1) * left_embedding + alpha.unsqueeze(
            -1
        ) * right_embedding


    def set_execution_mode(self, mode):
        if mode not in ("reference", "flat", "compiled", "auto"):
            raise ValueError("DEER execution mode must be reference, flat, compiled or auto")
        self.deer_execution_mode = mode
        object.__setattr__(self, "_deer_compiled_core", None)
        object.__setattr__(self, "_deer_boundary_cache", None)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        state["_deer_compiled_core"] = None
        state["_deer_boundary_cache"] = None
        return state

    def _runtime_boundaries(self):
        if self.training or torch.is_grad_enabled():
            return self.current_boundaries()
        # Reuse only no-grad evaluation boundaries. Standard optimizer updates,
        # load_state_dict, buffer changes and device/dtype changes invalidate.
        values = (self.raw_widths, self.lower, self.upper)
        try:
            key = tuple((id(t), t._version, t.device, t.dtype) for t in values)
        except RuntimeError:  # Tensors created under inference_mode have no version.
            return self.current_boundaries()
        key += (self.width_temperature, self.min_width)
        cache = getattr(self, "_deer_boundary_cache", None)
        if cache is None or cache[0] != key:
            cache = (key, self.current_boundaries())
            object.__setattr__(self, "_deer_boundary_cache", cache)
        return cache[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = getattr(self, "deer_execution_mode", "reference")
        # Training microbenchmarks did not show a consistent kernel win.
        # Default auto therefore optimizes no-grad evaluation only.
        if mode == "reference" or (mode == "auto" and (
                x.device.type != "cuda" or self.training or torch.is_grad_enabled())):
            return self.forward_reference(x)
        flat_index = _deer_flat_index
        boundaries = self._runtime_boundaries()
        knots = self.current_knot_embeddings()
        if mode == "flat":
            return flat_index(x, boundaries, knots, self.lower, self.upper, use_compile=False)
        try:
            if getattr(self, "_deer_compiled_core", None) is None:
                import importlib.util
                import sys
                module_name = "_fuxictr_numemb_deer_triton_runtime"
                module = sys.modules.get(module_name)
                if module is None:
                    path = Path(__file__).resolve().parent / "deer_triton_engineering.py"
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except Exception:
                        sys.modules.pop(module_name, None)
                        raise
                object.__setattr__(self, "_deer_compiled_core", module.triton_index)
            return self._deer_compiled_core(x, boundaries, knots, self.lower, self.upper)
        except Exception as exc:
            missing = isinstance(exc, ModuleNotFoundError) and exc.name == "triton"
            compiler = (isinstance(exc, torch._dynamo.exc.TorchDynamoException)
                        or (type(exc).__module__.startswith("triton.") and
                            type(exc).__name__ in ("CompilationError", "OutOfResources")))
            if mode != "auto" or not (missing or compiler) or "out of memory" in str(exc).lower():
                raise
            import warnings
            warnings.warn("DEER compiler unavailable; using original interpolation: "
                          + str(exc).splitlines()[0], RuntimeWarning)
            self.set_execution_mode("reference")
            return self.forward_reference(x)


class LiftDEEREmbedding(DEEREmbedding):
    """DEER in a CDF 5/3 lifting basis with learnable scale shrinkage.

    The ordered DEER knot table is interpreted as a signal on a path graph and
    transformed, without loss, into one coarse table and complementary detail
    bands.  Each lifting level uses linear prediction followed by the CDF 5/3
    update, so affine knot signals have zero detail coefficients and the
    transform is exactly invertible.  Field/scale/channel gains modulate only
    complementary detail bands::

        E = inverse_lift(C, {g_l D_l}).

    Gains initialize to one, making the layer bitwise-equivalent to a same-seed
    DEER before optimization.  The lifting coefficients contain exactly as many
    values as the original knot table; the only parameter overhead is
    ``F * L * d`` scale gains.  Because routing is static, the reconstructed
    table can be folded before inference and lookup cost remains DEER's O(d).
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        n_levels: Optional[int] = None,
        gain_limit: float = 1.0,
        gain_gradient_scale: float = 32.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        max_levels = 0
        remaining_intervals = int(self.n_bins)
        while remaining_intervals >= 2 and remaining_intervals % 2 == 0:
            max_levels += 1
            remaining_intervals //= 2
        if n_levels is None:
            n_levels = min(4, max_levels)
        self.n_levels = int(n_levels)
        if self.n_levels < 1:
            raise ValueError("liftdeer requires at least one lifting level")
        if self.n_levels > max_levels:
            raise ValueError(
                f"n_bins={self.n_bins} supports at most {max_levels} "
                f"dyadic lifting levels, got {self.n_levels}"
            )
        if gain_limit <= 0:
            raise ValueError("gain_limit must be positive")
        if gain_gradient_scale <= 0:
            raise ValueError("gain_gradient_scale must be positive")
        self.gain_limit = float(gain_limit)
        self.gain_gradient_scale = float(gain_gradient_scale)

        # DEER has already consumed the reference RNG stream.  Reparameterize
        # that exact table deterministically, without drawing more randomness.
        reference_knots = self.knot_embeddings.detach().clone()
        del self.knot_embeddings
        coarse, details = self.lifting_analysis(reference_knots, self.n_levels)
        self.coarse_embeddings = nn.Parameter(coarse)
        self.detail_embeddings = nn.ParameterList(
            [nn.Parameter(detail) for detail in details]
        )
        self.raw_detail_gains = nn.Parameter(
            torch.zeros(
                initial_edges.size(0), self.n_levels, d_embedding,
                dtype=reference_knots.dtype,
            )
        )

    @staticmethod
    def _predict_odd(even: torch.Tensor) -> torch.Tensor:
        """Predict odd path nodes by linear interpolation of even neighbours."""
        return 0.5 * (even[:, :-1, :] + even[:, 1:, :])

    @staticmethod
    def _update_even(detail: torch.Tensor) -> torch.Tensor:
        """CDF 5/3 update with symmetric boundary extension."""
        left = torch.cat([detail[:, :1, :], detail], dim=1)
        right = torch.cat([detail, detail[:, -1:, :]], dim=1)
        return 0.25 * (left + right)

    @classmethod
    def lifting_analysis(cls, knots: torch.Tensor, n_levels: int):
        """Return coarsest coefficients and fine-to-coarse detail bands."""
        current = knots
        details = []
        for _ in range(int(n_levels)):
            if current.size(1) < 3 or current.size(1) % 2 != 1:
                raise ValueError(
                    "Each lifting level requires an odd number of path nodes"
                )
            even = current[:, ::2, :]
            odd = current[:, 1::2, :]
            detail = odd - cls._predict_odd(even)
            current = even + cls._update_even(detail)
            details.append(detail)
        return current, details

    @classmethod
    def lifting_synthesis(
        cls,
        coarse: torch.Tensor,
        details: Sequence[torch.Tensor],
        gains: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exactly reconstruct a fine path signal, optionally scaling bands."""
        current = coarse
        for level in range(len(details) - 1, -1, -1):
            detail = details[level]
            if gains is not None:
                detail = detail * gains[:, level, :].unsqueeze(1)
            even = current - cls._update_even(detail)
            odd = detail + cls._predict_odd(even)
            paired = torch.stack([even[:, :-1, :], odd], dim=2)
            current = torch.cat(
                [paired.flatten(1, 2), even[:, -1:, :]], dim=1
            )
        return current

    def current_detail_gains(self) -> torch.Tensor:
        gains = 1.0 + self.gain_limit * torch.tanh(self.raw_detail_gains)
        if self.gain_gradient_scale != 1.0:
            gains = gains.detach() + self.gain_gradient_scale * (
                gains - gains.detach()
            )
        return gains

    def current_knot_embeddings(self) -> torch.Tensor:
        return self.lifting_synthesis(
            self.coarse_embeddings,
            self.detail_embeddings,
            self.current_detail_gains(),
        )

    def unscaled_knot_embeddings(self) -> torch.Tensor:
        """Reconstruct the underlying table with every detail gain fixed to one."""
        return self.lifting_synthesis(
            self.coarse_embeddings, self.detail_embeddings, gains=None
        )

    def detail_energy(self) -> torch.Tensor:
        """Per-field and per-level RMS detail energy for diagnostics."""
        return torch.stack(
            [detail.square().mean(dim=(1, 2)).sqrt() for detail in self.detail_embeddings],
            dim=1,
        )


class ResponsePriorDEEREmbedding(DEEREmbedding):
    """DEER augmented with a train-only empirical-Bayes response field.

    For each numerical field, an offline train-split pass estimates a smoothed
    label log-odds residual at the DEER knots.  At inference the scalar response
    field is interpolated on the *same learned mesh* as DEER and injected along
    one deterministic unit direction::

        e_f(x) = e_DEER,f(x) + gamma_f r_f(x) v.

    ``gamma_f`` is the only additional trainable parameter per field.  The
    response knots and direction are buffers, so inference remains a two-knot
    O(d) interpolation and initialization consumes exactly the same RNG stream
    as vanilla DEER.  Statistics must be computed from the training split only.
    This is an experimental research method, not a paper reproduction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        prior_scale: float = 0.01,
        prior_scale_limit: float = 0.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if prior_scale < 0:
            raise ValueError("prior_scale must be nonnegative")
        if prior_scale_limit < 0:
            raise ValueError("prior_scale_limit must be nonnegative")
        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        expected = (initial_edges.size(0), self.n_bins + 1)
        if tuple(response.shape) != expected:
            raise ValueError(
                f"response_knots must have shape {expected}, got {tuple(response.shape)}"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")
        self.register_buffer("response_knots", response.float())
        self.prior_scale_limit = float(prior_scale_limit)
        if self.prior_scale_limit > 0:
            ratio = min(float(prior_scale) / self.prior_scale_limit, 1.0 - 1e-6)
            raw = 0.5 * np.log((1.0 + ratio) / (1.0 - ratio))
            self.raw_prior_scales = nn.Parameter(
                torch.full((initial_edges.size(0),), float(raw))
            )
        else:
            self.prior_scales = nn.Parameter(
                torch.full((initial_edges.size(0),), float(prior_scale))
            )
        direction = torch.ones(d_embedding, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(1e-12)
        self.register_buffer("prior_direction", direction)

    def current_prior_scales(self) -> torch.Tensor:
        if self.prior_scale_limit > 0:
            return self.prior_scale_limit * torch.tanh(self.raw_prior_scales)
        return self.prior_scales

    def response_field(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        expanded_response = self.response_knots.unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        left_response = expanded_response.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right_response = expanded_response.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        return (1.0 - alpha) * left_response + alpha * right_response

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        response = self.response_field(x)
        residual = response * self.current_prior_scales().unsqueeze(0)
        return base + residual.unsqueeze(-1) * self.prior_direction



class ResponseInitializedDEEREmbedding(DEEREmbedding):
    """DEER with one axis initialized by train-only response geometry.

    A smoothed empirical-Bayes log-odds curve is computed on the training
    split. The random knot table is decomposed into one fixed unit direction
    and its orthogonal complement. Only the random component along that
    direction is replaced by the standardized response curve. The target-aware
    coordinate has the same scale as a random coordinate, consumes no extra
    random numbers, adds no parameters, and has vanilla DEER inference cost.
    Every knot and boundary remains fully trainable after initialization.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if response_scale < 0:
            raise ValueError("response_scale must be nonnegative")
        if response_clip <= 0:
            raise ValueError("response_clip must be positive")
        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        expected = (initial_edges.size(0), self.n_bins + 1)
        if tuple(response.shape) != expected:
            raise ValueError(
                f"response_knots must have shape {expected}, got {tuple(response.shape)}"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")
        centered = response - response.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        standardized = centered / rms.clamp_min(1e-6)
        standardized = standardized.clamp(
            -float(response_clip), float(response_clip)
        )
        standardized = torch.where(
            rms <= 1e-6, torch.zeros_like(standardized), standardized
        )
        self.register_buffer(
            "standardized_response_knots", standardized.float()
        )
        direction = torch.ones(d_embedding, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(1e-12)
        self.register_buffer("response_direction", direction)
        self.response_scale = float(response_scale)
        with torch.no_grad():
            random_parallel = torch.einsum(
                "fkd,d->fk", self.knot_embeddings, direction
            )
            target_parallel = (
                self.response_scale
                * float(embedding_init_std)
                * float(d_embedding) ** 0.5
                * standardized
            )
            self.knot_embeddings.add_(
                (target_parallel - random_parallel).unsqueeze(-1) * direction
            )



class GraphFrameDEEREmbedding(DEEREmbedding):
    """DEER initialized as a local/diffusive/supervised graph frame.

    The ordered knot values of every numerical field form a signal on a path
    graph. Vanilla DEER initializes all latent channels as independent local
    signals. This layer keeps part of those channels local, diffuses a second
    part with a graph heat step, and initializes the remaining channels with a
    train-only empirical-Bayes response signal split into coarse-to-fine heat
    bands. The bands are orthogonalized independently for every field.

    After initialization there is only the ordinary trainable DEER knot table:
    no expert, router, or auxiliary forward path remains. Consequently the
    parameter count and inference FLOPs are exactly those of DEER, while the
    initial frame exposes local, Sobolev-smooth, and task-aligned coordinates.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_smooth: int = 4,
        n_response: int = 3,
        smoothing_steps: int = 4,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if n_smooth < 0 or n_response < 1:
            raise ValueError("n_smooth must be nonnegative and n_response positive")
        if n_response > 4:
            raise ValueError("n_response cannot exceed the four graph heat bands")
        if n_smooth + n_response >= d_embedding:
            raise ValueError("at least one local channel must remain")
        if smoothing_steps < 1:
            raise ValueError("smoothing_steps must be positive")
        if response_scale < 0 or response_clip <= 0:
            raise ValueError("invalid response initialization scale or clip")

        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        expected = tuple(initial_edges.shape)
        if tuple(response.shape) != expected:
            raise ValueError(
                f"response_knots must have shape {expected}, got {tuple(response.shape)}"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")

        self.n_smooth = int(n_smooth)
        self.n_response = int(n_response)
        self.n_local = int(d_embedding - n_smooth - n_response)
        self.smoothing_steps = int(smoothing_steps)
        self.response_scale = float(response_scale)
        self.response_clip = float(response_clip)

        with torch.no_grad():
            if self.n_smooth:
                start = self.n_local
                stop = start + self.n_smooth
                local_signal = self.knot_embeddings[:, :, start:stop].clone()
                smooth_signal = self._diffuse_path(
                    local_signal, self.smoothing_steps
                )
                local_mean = local_signal.mean(dim=1, keepdim=True)
                smooth_mean = smooth_signal.mean(dim=1, keepdim=True)
                local_rms = (local_signal - local_mean).square().mean(
                    dim=1, keepdim=True
                ).sqrt()
                smooth_rms = (smooth_signal - smooth_mean).square().mean(
                    dim=1, keepdim=True
                ).sqrt()
                smooth_signal = local_mean + (smooth_signal - smooth_mean) * (
                    local_rms / smooth_rms.clamp_min(1e-12)
                )
                self.knot_embeddings[:, :, start:stop].copy_(smooth_signal)

            response_frame = self._response_heat_frame(response.float())
            response_frame = response_frame.clamp(
                -self.response_clip, self.response_clip
            )
            response_start = d_embedding - self.n_response
            self.knot_embeddings[:, :, response_start:].copy_(
                self.response_scale
                * float(embedding_init_std)
                * response_frame
            )
        self.register_buffer("initial_response_frame", response_frame)

    @staticmethod
    def _diffuse_path(signal: torch.Tensor, steps: int) -> torch.Tensor:
        output = signal
        for _ in range(int(steps)):
            left = torch.cat([output[:, :1], output[:, :-1]], dim=1)
            right = torch.cat([output[:, 1:], output[:, -1:]], dim=1)
            output = 0.25 * left + 0.5 * output + 0.25 * right
        return output

    def _response_heat_frame(self, response: torch.Tensor) -> torch.Tensor:
        centered = response - response.mean(dim=1, keepdim=True)
        heat_1 = self._diffuse_path(centered, 1)
        heat_4 = self._diffuse_path(centered, 4)
        heat_16 = self._diffuse_path(centered, 16)
        candidates = (
            heat_16,
            heat_4 - heat_16,
            heat_1 - heat_4,
            centered - heat_1,
        )
        orthogonal = []
        for candidate in candidates[: self.n_response]:
            vector = candidate - candidate.mean(dim=1, keepdim=True)
            for basis in orthogonal:
                projection = (vector * basis).mean(
                    dim=1, keepdim=True
                )
                vector = vector - projection * basis
            rms = vector.square().mean(dim=1, keepdim=True).sqrt()
            vector = torch.where(
                rms > 1e-6,
                vector / rms.clamp_min(1e-6),
                torch.zeros_like(vector),
            )
            orthogonal.append(vector)
        return torch.stack(orthogonal, dim=-1)


class ResidualGraphFrameDEEREmbedding(DEEREmbedding):
    """Vanilla DEER plus variance-calibrated graph-frame initialization.

    All random local DEER coordinates are retained. A path-graph heat diffusion
    of the same random table is fused residually and renormalized to the
    original per-field/channel variance. Orthogonal coarse-to-fine response
    bands, estimated from the training split only, are then added along fixed
    DCT directions. After this initialization the module is exactly an ordinary
    DEER table, so it adds neither learned parameters nor inference operations.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_response: int = 3,
        smoothing_steps: int = 4,
        smooth_scale: float = 0.5,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 1 <= n_response <= min(4, d_embedding):
            raise ValueError("n_response must lie in [1, min(4, d_embedding)]")
        if smoothing_steps < 1:
            raise ValueError("smoothing_steps must be positive")
        if smooth_scale < 0 or response_scale < 0 or response_clip <= 0:
            raise ValueError("invalid graph-frame scale or clip")
        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        if tuple(response.shape) != tuple(initial_edges.shape):
            raise ValueError(
                "response_knots must have the same shape as initial_edges"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")

        self.n_response = int(n_response)
        self.smoothing_steps = int(smoothing_steps)
        self.smooth_scale = float(smooth_scale)
        self.response_scale = float(response_scale)
        self.response_clip = float(response_clip)

        positions = torch.arange(d_embedding, dtype=torch.float32) + 0.5
        frequencies = torch.arange(self.n_response, dtype=torch.float32)
        directions = torch.cos(
            torch.pi * frequencies[:, None] * positions[None, :] / d_embedding
        )
        directions[0].mul_(2.0 ** -0.5)
        directions.mul_((2.0 / d_embedding) ** 0.5)

        with torch.no_grad():
            original = self.knot_embeddings.clone()
            if self.smooth_scale:
                smooth = GraphFrameDEEREmbedding._diffuse_path(
                    original, self.smoothing_steps
                )
                fused = original + self.smooth_scale * smooth
                original_mean = original.mean(dim=1, keepdim=True)
                original_rms = (original - original_mean).square().mean(
                    dim=1, keepdim=True
                ).sqrt()
                fused_mean = fused.mean(dim=1, keepdim=True)
                fused_rms = (fused - fused_mean).square().mean(
                    dim=1, keepdim=True
                ).sqrt()
                fused = original_mean + (fused - fused_mean) * (
                    original_rms / fused_rms.clamp_min(1e-12)
                )
                self.knot_embeddings.copy_(fused)

            frame = self._response_heat_frame(response.float()).clamp(
                -self.response_clip, self.response_clip
            )
            response_delta = torch.einsum("fkr,rd->fkd", frame, directions)
            response_delta.mul_(
                self.response_scale
                * float(embedding_init_std)
                * (float(d_embedding) / self.n_response) ** 0.5
            )
            self.knot_embeddings.add_(response_delta)

        self.register_buffer("initial_response_frame", frame)
        self.register_buffer("response_directions", directions)

    def _response_heat_frame(self, response: torch.Tensor) -> torch.Tensor:
        centered = response - response.mean(dim=1, keepdim=True)
        if self.n_response == 1:
            rms = centered.square().mean(dim=1, keepdim=True).sqrt()
            standardized = torch.where(
                rms > 1e-6,
                centered / rms.clamp_min(1e-6),
                torch.zeros_like(centered),
            )
            return standardized.unsqueeze(-1)
        heat_1 = GraphFrameDEEREmbedding._diffuse_path(centered, 1)
        heat_4 = GraphFrameDEEREmbedding._diffuse_path(centered, 4)
        heat_16 = GraphFrameDEEREmbedding._diffuse_path(centered, 16)
        candidates = (
            heat_16,
            heat_4 - heat_16,
            heat_1 - heat_4,
            centered - heat_1,
        )
        orthogonal = []
        for candidate in candidates[: self.n_response]:
            vector = candidate - candidate.mean(dim=1, keepdim=True)
            for basis in orthogonal:
                vector = vector - (vector * basis).mean(
                    dim=1, keepdim=True
                ) * basis
            rms = vector.square().mean(dim=1, keepdim=True).sqrt()
            vector = torch.where(
                rms > 1e-6,
                vector / rms.clamp_min(1e-6),
                torch.zeros_like(vector),
            )
            orthogonal.append(vector)
        return torch.stack(orthogonal, dim=-1)




class AdaptiveSobolevDEEREmbedding(ResponseInitializedDEEREmbedding):
    """Shared-control interpolation continuum between DEER and a smooth spline.

    A single response-initialized knot table defines two views of each field:
    an adaptive piecewise-linear finite-element chart and a cubic B-spline
    chart. Per-field, per-channel gates learn their convex combination,

        e(x) = (1 - g) e_local(x) + g e_smooth(x).

    The spline controls are differentiable resamples of the same DEER table,
    so the layer preserves every latent channel in both charts without
    doubling learned coefficients. Relative to DEER it adds only F*d gate
    scalars. This is the finite-element analogue of choosing local versus
    Sobolev-smooth coordinates independently in each latent channel.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_basis: int = 8,
        degree: int = 3,
        smooth_logit_init: float = 0.0,
        gate_temperature: float = 1.0,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            response_scale=response_scale,
            response_clip=response_clip,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if degree < 1 or n_basis <= degree:
            raise ValueError("invalid B-spline basis")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        n_internal = self.n_basis - self.degree - 1
        internal = (
            torch.linspace(0.0, 1.0, n_internal + 2)[1:-1]
            if n_internal > 0
            else torch.empty(0)
        )
        spline_knots = torch.cat(
            [
                torch.zeros(self.degree + 1),
                internal,
                torch.ones(self.degree + 1),
            ]
        )
        greville = torch.stack(
            [
                spline_knots[i + 1 : i + self.degree + 1].mean()
                for i in range(self.n_basis)
            ]
        )
        self.register_buffer("spline_knots", spline_knots)
        self.register_buffer("greville", greville)
        self.raw_smooth_logits = nn.Parameter(
            torch.full(
                (initial_edges.size(0), d_embedding),
                float(smooth_logit_init),
            )
        )
        self.gate_temperature = float(gate_temperature)

    def smooth_gates(self) -> torch.Tensor:
        return torch.sigmoid(
            self.raw_smooth_logits / self.gate_temperature
        )

    def smooth_basis(self, x: torch.Tensor) -> torch.Tensor:
        unit_x = (
            (x - self.lower)
            / (self.upper - self.lower).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        expanded = unit_x.unsqueeze(-1)
        basis = (
            (expanded >= self.spline_knots[:-1])
            & (expanded < self.spline_knots[1:])
        ).to(dtype=x.dtype)
        for order in range(1, self.degree + 1):
            left_denominator = (
                self.spline_knots[order:-1]
                - self.spline_knots[:-order - 1]
            )
            right_denominator = (
                self.spline_knots[order + 1:]
                - self.spline_knots[1:-order]
            )
            left_scale = torch.where(
                left_denominator > 0,
                (expanded - self.spline_knots[:-order - 1])
                / left_denominator.clamp_min(1e-12),
                torch.zeros_like(left_denominator),
            )
            right_scale = torch.where(
                right_denominator > 0,
                (self.spline_knots[order + 1:] - expanded)
                / right_denominator.clamp_min(1e-12),
                torch.zeros_like(right_denominator),
            )
            basis = (
                left_scale * basis[..., :-1]
                + right_scale * basis[..., 1:]
            )
        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.0
        return torch.where(
            (unit_x >= 1.0).unsqueeze(-1), endpoint, basis
        )

    def smooth_controls(self) -> torch.Tensor:
        boundaries = self.current_boundaries()
        raw_points = self.lower.unsqueeze(-1) + self.greville.unsqueeze(0) * (
            self.upper - self.lower
        ).unsqueeze(-1)
        interval = (
            raw_points.unsqueeze(-1)
            >= boundaries[:, 1:-1].unsqueeze(1)
        ).sum(dim=-1)
        left = boundaries.gather(1, interval)
        right = boundaries.gather(1, interval + 1)
        alpha = (
            (raw_points - left) / (right - left).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        knots = self.current_knot_embeddings()
        gather_index = interval.unsqueeze(-1).expand(
            -1, -1, knots.size(-1)
        )
        left_embedding = knots.gather(1, gather_index)
        right_embedding = knots.gather(1, gather_index + 1)
        return (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = DEEREmbedding.forward(self, x)
        smooth = torch.einsum(
            "bfm,fmd->bfd", self.smooth_basis(x), self.smooth_controls()
        )
        gates = self.smooth_gates().unsqueeze(0)
        return local + gates * (smooth - local)


class HPMixtureEmbedding(nn.Module):
    """Adaptive hp finite-element mixture with full-capacity experts.

    The p=1 expert is response-initialized DEER on a movable mesh. The p=3
    expert is a cubic B-spline on a fixed raw-coordinate mesh. A tiny router
    chooses the smooth expert per field, latent channel, and empirical-rank
    position. Both experts retain all d channels; the router adds only 2*F*d
    scalars. Variance normalization keeps the random-feature scale constant
    along the local-to-smooth interpolation path.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_basis: int = 8,
        degree: int = 3,
        smooth_logit_init: float = 0.0,
        gate_temperature: float = 1.0,
        router_gradient_scale: float = 1.0,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        variance_preserving: bool = True,
    ):
        super().__init__()
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if router_gradient_scale <= 0:
            raise ValueError("router_gradient_scale must be positive")
        self.local = ResponseInitializedDEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            response_scale=response_scale,
            response_clip=response_clip,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        state_after_local = torch.random.get_rng_state()
        self.smooth = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_local)
        shape = (initial_edges.size(0), d_embedding)
        self.raw_smooth_logits = nn.Parameter(
            torch.full(shape, float(smooth_logit_init))
        )
        self.rank_slopes = nn.Parameter(torch.zeros(shape))
        self.gate_temperature = float(gate_temperature)
        self.router_gradient_scale = float(router_gradient_scale)
        self.variance_preserving = bool(variance_preserving)

    def rank_position(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.local.current_boundaries()
        clipped = torch.maximum(
            torch.minimum(x, self.local.upper), self.local.lower
        )
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
        return (interval.to(x.dtype) + alpha) / self.local.n_bins

    def smooth_gates(self, x: torch.Tensor) -> torch.Tensor:
        rank = self.rank_position(x).unsqueeze(-1) - 0.5
        logits = (
            self.raw_smooth_logits.unsqueeze(0)
            + rank * self.rank_slopes.unsqueeze(0)
        )
        gate = torch.sigmoid(logits / self.gate_temperature)
        if self.router_gradient_scale != 1.0:
            gate = gate.detach() + self.router_gradient_scale * (
                gate - gate.detach()
            )
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        smooth = self.smooth(x)
        gate = self.smooth_gates(x)
        output = (1.0 - gate) * local + gate * smooth
        if self.variance_preserving:
            scale = torch.sqrt(
                (1.0 - gate).square() + gate.square()
            ).clamp_min(1e-6)
            output = output / scale
        return output


class ResponseAnchoredResidualEmbedding(ResponseInitializedDEEREmbedding):
    """Response-geometric DEER anchor plus a zero-initialized smooth corrector.

    The response-initialized movable P1 mesh supplies a strong, task-aware
    coordinate system.  An independent cubic B-spline is added as an additive
    Schwarz correction,

        e(x) = e_response-DEER(x) + lambda * e_spline-correction(x).

    The correction coefficients start at exactly zero, so construction is
    function preserving and keeps the downstream RNG stream identical to the
    response-DEER anchor.  A straight-through gradient multiplier decouples
    the correction's forward amplitude from its learning time scale.  This
    retains the local inductive bias while letting smooth-preferring backbones
    acquire low-order structure without averaging incompatible random codes.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_basis: int = 8,
        degree: int = 3,
        correction_scale: float = 1.0,
        correction_gradient_scale: Optional[float] = None,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        if correction_scale <= 0:
            raise ValueError("correction_scale must be positive")
        if correction_gradient_scale is not None and correction_gradient_scale <= 0:
            raise ValueError("correction_gradient_scale must be positive")
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            response_scale=response_scale,
            response_clip=response_clip,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        state_after_anchor = torch.random.get_rng_state()
        self.smooth_correction = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_anchor)
        nn.init.zeros_(self.smooth_correction.coefficients)
        self.correction_scale = float(correction_scale)
        self.correction_gradient_scale = float(
            correction_scale if correction_gradient_scale is None
            else correction_gradient_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = super().forward(x)
        correction = self.smooth_correction(x)
        weighted_correction = (
            self.correction_gradient_scale * correction
            + (self.correction_scale - self.correction_gradient_scale)
            * correction.detach()
        )
        return anchor + weighted_correction


class OrthogonalAtlasEmbedding(nn.Module):
    """Orthogonal atlas of local, smooth, ordinal, and response coordinates.

    The embedding space is a direct sum of four non-interfering charts:
    adaptive piecewise-linear DEER coordinates, smooth B-spline coordinates,
    a fixed empirical-rank coordinate, and a fixed train-only empirical-Bayes
    response coordinate. The two scalar anchors cannot be overwritten by the
    trainable charts, while each downstream backbone can select the geometry it
    needs with an ordinary linear projection. Output width remains unchanged.

    Initialization consumes exactly one full-width DEER knot table. Its columns
    deterministically initialize both trainable charts, so downstream backbone
    random states are aligned with vanilla DEER. The method uses fewer learned
    knot coefficients than full-width DEER and has fixed-basis O(d) inference.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        local_dim: Optional[int] = None,
        n_basis: int = 8,
        degree: int = 3,
        rank_scale: float = 1.0,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
    ):
        super().__init__()
        if d_embedding < 4:
            raise ValueError("atlas requires d_embedding >= 4")
        if degree < 1 or n_basis <= degree:
            raise ValueError("invalid B-spline basis")
        if rank_scale < 0 or response_scale < 0:
            raise ValueError("anchor scales must be nonnegative")
        if response_clip <= 0:
            raise ValueError("response_clip must be positive")
        learned_dim = d_embedding - 2
        if local_dim is None:
            local_dim = learned_dim // 2
        local_dim = int(local_dim)
        smooth_dim = learned_dim - local_dim
        if local_dim < 1 or smooth_dim < 1:
            raise ValueError("local_dim must leave at least one smooth dimension")

        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        expected = tuple(initial_edges.shape)
        if tuple(response.shape) != expected:
            raise ValueError(
                f"response_knots must have shape {expected}, got {tuple(response.shape)}"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")

        reference = torch.empty(
            initial_edges.size(0), initial_edges.size(1), d_embedding
        )
        nn.init.normal_(reference, std=float(embedding_init_std))
        state_after_reference = torch.random.get_rng_state()
        self.local = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=local_dim,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        self.smooth = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=smooth_dim,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_reference)
        with torch.no_grad():
            self.local.knot_embeddings.copy_(reference[:, :, :local_dim])
            greville = torch.stack(
                [
                    self.smooth.knots[
                        i + 1 : i + self.smooth.degree + 1
                    ].mean()
                    for i in range(self.smooth.n_basis)
                ]
            )
            raw_controls = initial_edges[:, :1] + greville.unsqueeze(0) * (
                initial_edges[:, -1:] - initial_edges[:, :1]
            )
            smooth_reference = reference[
                :, :, local_dim : local_dim + smooth_dim
            ]
            controls = []
            for field in range(initial_edges.size(0)):
                edges = initial_edges[field]
                points = raw_controls[field]
                interval = torch.bucketize(
                    points.contiguous(), edges[1:-1]
                )
                left = edges[interval]
                right = edges[interval + 1]
                alpha = (
                    (points - left) / (right - left).clamp_min(1e-12)
                ).clamp(0.0, 1.0)
                controls.append(
                    (1.0 - alpha).unsqueeze(-1)
                    * smooth_reference[field, interval]
                    + alpha.unsqueeze(-1)
                    * smooth_reference[field, interval + 1]
                )
            self.smooth.coefficients.copy_(torch.stack(controls))

        centered = response - response.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        standardized = centered / rms.clamp_min(1e-6)
        standardized = standardized.clamp(
            -float(response_clip), float(response_clip)
        )
        standardized = torch.where(
            rms <= 1e-6, torch.zeros_like(standardized), standardized
        )
        self.register_buffer("anchor_edges", initial_edges.float())
        self.register_buffer(
            "standardized_response_knots", standardized.float()
        )
        self.local_dim = local_dim
        self.smooth_dim = smooth_dim
        self.embedding_init_std = float(embedding_init_std)
        self.rank_scale = float(rank_scale)
        self.response_scale = float(response_scale)

    def anchor_coordinates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = self.anchor_edges
        clipped = torch.maximum(
            torch.minimum(x, edges[:, -1]), edges[:, 0]
        )
        interval = (
            clipped.unsqueeze(-1) >= edges[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_edges = edges.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_edges.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_edges.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = (
            (clipped - left) / (right - left).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        rank = (interval.to(x.dtype) + alpha) / (edges.size(1) - 1)
        expanded_response = self.standardized_response_knots.unsqueeze(
            0
        ).expand(x.size(0), -1, -1)
        left_response = expanded_response.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right_response = expanded_response.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        response = (
            (1.0 - alpha) * left_response + alpha * right_response
        )
        return (12.0 ** 0.5) * (rank - 0.5), response

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank, response = self.anchor_coordinates(x)
        anchors = self.embedding_init_std * torch.stack(
            [self.rank_scale * rank, self.response_scale * response],
            dim=-1,
        )
        return torch.cat(
            [self.local(x), self.smooth(x), anchors], dim=-1
        )




class DiscontinuousResidualDEEREmbedding(DEEREmbedding):
    """Continuous DEER plus a low-rank piecewise-constant jump field.

    Many numerical columns are measures with both continuous and atomic parts:
    rounded counts, missing-value sentinels, and business thresholds.  A
    continuous interpolant cannot represent their jumps without distorting its
    neighbouring slopes.  Inspired by discontinuous Galerkin enrichment, this
    layer decomposes the representation into a continuous DEER field and one
    interval-local correction::

        e_f(x) = e_DEER,f(x) + sum_r a[f, k(x), r] v[r].

    The directions ``v`` are a fixed orthonormal DCT basis; only ``F*K*R``
    scalar coefficients are added.  Zero coefficients exactly recover DEER,
    consume no random numbers, and have useful first-step gradients.  Runtime
    remains one interval lookup and O(dR) arithmetic.  This is an experimental
    research method, not a paper reproduction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        residual_rank: int = 2,
        residual_scale_limit: float = 0.5,
        coefficient_init_std: float = 0.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if residual_rank < 1 or residual_rank > d_embedding:
            raise ValueError("residual_rank must be in [1, d_embedding]")
        if residual_scale_limit <= 0:
            raise ValueError("residual_scale_limit must be positive")
        if coefficient_init_std < 0:
            raise ValueError("coefficient_init_std must be nonnegative")
        self.residual_rank = int(residual_rank)
        self.residual_scale_limit = float(residual_scale_limit)
        self.raw_jump_coefficients = nn.Parameter(
            torch.empty(
                initial_edges.size(0), self.n_bins, self.residual_rank
            )
        )
        if coefficient_init_std == 0:
            nn.init.zeros_(self.raw_jump_coefficients)
        else:
            state = torch.random.get_rng_state()
            nn.init.normal_(
                self.raw_jump_coefficients, std=float(coefficient_init_std)
            )
            torch.random.set_rng_state(state)
        positions = torch.arange(d_embedding, dtype=torch.float32) + 0.5
        frequencies = torch.arange(self.residual_rank, dtype=torch.float32)
        phases = torch.pi * frequencies[:, None] * positions[None, :] / d_embedding
        directions = torch.cos(phases)
        directions[0].mul_(2.0 ** -0.5)
        directions.mul_((2.0 / d_embedding) ** 0.5)
        self.register_buffer("jump_directions", directions)

    def jump_coefficients(self) -> torch.Tensor:
        return self.residual_scale_limit * torch.tanh(
            self.raw_jump_coefficients
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        coefficients = self.jump_coefficients().unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        active = coefficients.gather(
            2,
            interval.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, self.residual_rank
            ),
        ).squeeze(2)
        residual = torch.einsum("bfr,rd->bfd", active, self.jump_directions)
        return output + residual


class CrossBasisDEEREmbedding(DEEREmbedding):
    """Partition-of-unity fusion of adaptive DEER and smooth B-splines.

    DEER is an adaptive piecewise-linear finite-element basis, whereas the
    cubic B-spline expert is a smooth fixed-grid basis.  A tiny router forms a
    convex partition of unity

        e_f(x) = (1-g_f(x)) e_DEER,f(x) + g_f(x) e_spline,f(x).

    ``g_f(x)`` is parameterized by low-order Legendre polynomials of the
    continuous DEER rank coordinate.  With ``gate_order=0`` this reduces to a
    scalar field gate.  Orders 1--3 add only ``F * order`` parameters and let
    dense and tail regions select different approximation biases.  A router
    temperature controls adaptation speed without changing its initial gate.

    The spline expert is initialized inside a restored RNG scope, so adding it
    never changes downstream backbone initialization.  Inference remains
    O(d) for a fixed small number of spline bases.  This is an experimental
    research method, not a paper reproduction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        gate_init: float = 0.25,
        gate_limit: float = 1.0,
        gate_order: int = 0,
        gate_temperature: float = 1.0,
        variance_preserving: bool = False,
        gate_geometry_power: Optional[float] = None,
        gate_geometry_scale: float = 1.0,
        gate_geometry_scores: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 0.0 < gate_limit <= 1.0:
            raise ValueError("gate_limit must be in (0, 1]")
        if not 0.0 < gate_init < gate_limit:
            raise ValueError("gate_init must be in (0, gate_limit)")
        if not 0 <= gate_order <= 3:
            raise ValueError("gate_order must be in [0, 3]")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if gate_geometry_power is not None and gate_geometry_power <= 0:
            raise ValueError("gate_geometry_power must be positive")
        if gate_geometry_scale <= 0:
            raise ValueError("gate_geometry_scale must be positive")
        state = torch.random.get_rng_state()
        self.spline = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state)
        self.gate_limit = float(gate_limit)
        self.gate_order = int(gate_order)
        self.gate_temperature = float(gate_temperature)
        self.variance_preserving = bool(variance_preserving)
        if gate_geometry_power is None:
            ratios = torch.full(
                (initial_edges.size(0),), float(gate_init) / self.gate_limit
            )
        else:
            if gate_geometry_scores is None:
                widths = (
                    initial_edges[:, 1:] - initial_edges[:, :-1]
                ).clamp_min(0.0)
                totals = widths.sum(dim=-1, keepdim=True)
                probabilities = widths / totals.clamp_min(1e-12)
                entropy = -(
                    probabilities * probabilities.clamp_min(1e-12).log()
                ).sum(dim=-1) / np.log(widths.size(-1))
                entropy = torch.where(
                    totals.squeeze(-1) > 1e-12,
                    entropy,
                    torch.zeros_like(entropy),
                )
            else:
                entropy = torch.as_tensor(
                    gate_geometry_scores, dtype=initial_edges.dtype
                )
                if entropy.shape != (initial_edges.size(0),):
                    raise ValueError(
                        "gate_geometry_scores must have one value per feature"
                    )
            ratios = (
                float(gate_geometry_scale)
                * entropy.clamp(0.0, 1.0).pow(float(gate_geometry_power))
            )
        ratios = ratios.clamp(1e-4, 1.0 - 1e-4)
        logits = torch.log(ratios) - torch.log1p(-ratios)
        self.register_buffer("initial_gate_ratios", ratios.float())
        self.gate_logits = nn.Parameter(self.gate_temperature * logits.float())
        if self.gate_order > 0:
            self.gate_coefficients = nn.Parameter(
                torch.zeros(initial_edges.size(0), self.gate_order)
            )

    def rank_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous [0,1] coordinate induced by the current DEER mesh."""
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded.gather(2, (interval + 1).unsqueeze(-1)).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(0.0, 1.0)
        return (interval.to(x.dtype) + alpha) / float(self.n_bins)

    def gates(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.gate_logits
        if x is not None and self.gate_order > 0:
            z = 2.0 * self.rank_coordinate(x) - 1.0
            basis = [z]
            if self.gate_order >= 2:
                basis.append(0.5 * (3.0 * z.square() - 1.0))
            if self.gate_order >= 3:
                basis.append(0.5 * (5.0 * z.pow(3) - 3.0 * z))
            router_basis = torch.stack(basis, dim=-1)
            logits = logits.unsqueeze(0) + torch.einsum(
                "bfo,fo->bf", router_basis, self.gate_coefficients
            )
        return self.gate_limit * torch.sigmoid(logits / self.gate_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = super().forward(x)
        smooth = self.spline(x)
        gate = self.gates(x)
        if gate.ndim == 1:
            gate = gate.unsqueeze(0)
        gate = gate.unsqueeze(-1)
        output = local + gate * (smooth - local)
        if self.variance_preserving:
            normalizer = ((1.0 - gate).square() + gate.square()).sqrt()
            output = output / normalizer.clamp_min(1e-12)
        return output


class ResidualBasisDEEREmbedding(DEEREmbedding):
    """DEER with a zero-initialized smooth residual basis.

    This is an additive multi-basis expansion rather than an expert average::

        e_f(x) = e_DEER,f(x) + lambda * e_spline,f(x).

    The cubic B-spline coefficients start at exactly zero.  Consequently the
    module is functionally identical to DEER at initialization, preserves the
    same downstream RNG stream, and learns only corrections that the adaptive
    piecewise-linear mesh does not capture efficiently.  ``lambda`` is a fixed
    conditioning constant, so no router or extra nonlinear network is needed.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        residual_scale: float = 1.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        state = torch.random.get_rng_state()
        self.spline = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state)
        nn.init.zeros_(self.spline.coefficients)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.residual_scale * self.spline(x)


class LowRankSmoothResidualDEEREmbedding(DEEREmbedding):
    """Adaptive DEER anchor plus a low-rank smooth B-spline correction.

    This is the dual two-level decomposition to :class:`MultigridResidualEmbedding`.
    DEER supplies the safe adaptive coarse solution.  A zero-initialized spline
    detail with ``rank`` channels is lifted through a shared orthogonal matrix.
    It therefore starts exactly at DEER, preserves DEER-aligned backbone RNG,
    and adds only ``F*M*r + r*d`` coefficient parameters (besides the fixed
    spline knots).  The construction is useful when raw marginals are strongly
    non-uniform and a uniform spline should act only as a regular correction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        correction_rank: int = 2,
        residual_scale: float = 1.0,
        zero_factor: str = "coefficients",
        latent_init_std: float = 0.05,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 1 <= correction_rank <= d_embedding:
            raise ValueError("correction_rank must lie in [1, d_embedding]")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if zero_factor not in ("coefficients", "directions"):
            raise ValueError("zero_factor must be 'coefficients' or 'directions'")
        if latent_init_std <= 0:
            raise ValueError("latent_init_std must be positive")
        state_after_deer = torch.random.get_rng_state()
        self.smooth_correction = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=correction_rank,
            n_basis=n_basis,
            degree=degree,
        )
        self.smooth_directions = nn.Parameter(
            torch.empty(correction_rank, d_embedding)
        )
        if zero_factor == "coefficients":
            nn.init.orthogonal_(self.smooth_directions)
            nn.init.zeros_(self.smooth_correction.coefficients)
        else:
            nn.init.normal_(
                self.smooth_correction.coefficients, std=float(latent_init_std)
            )
            nn.init.zeros_(self.smooth_directions)
        torch.random.set_rng_state(state_after_deer)
        self.correction_rank = int(correction_rank)
        self.residual_scale = float(residual_scale)
        self.zero_factor = zero_factor
        self.latent_init_std = float(latent_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.smooth_correction(x) @ self.smooth_directions
        return super().forward(x) + self.residual_scale * detail


class MultigridResidualEmbedding(nn.Module):
    """Smooth coarse basis with a zero-initialized adaptive DEER correction.

    This mirrors a two-level finite-element/multigrid decomposition::

        e_f(x) = e_spline,f(x) + lambda * e_DEER-correction,f(x).

    The uniform cubic B-spline is the coarse, well-conditioned approximation.
    The DEER mesh supplies local adaptive degrees of freedom whose knot
    embeddings start at zero.  Hence the module is exactly a B-spline at
    initialization, while gradients immediately train the correction values
    and subsequently its movable mesh.  There is no router, the output width
    is unchanged, and evaluation remains O(d) for fixed basis sizes.

    Construction restores the RNG state following the coarse B-spline, so
    downstream backbone initialization is exactly aligned with a B-spline
    control under the same seed.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        correction_scale: float = 1.0,
        correction_init_std: float = 0.0,
        variance_preserving: bool = False,
        correction_gradient_scale: Optional[float] = None,
    ):
        super().__init__()
        if correction_scale <= 0:
            raise ValueError("correction_scale must be positive")
        if correction_gradient_scale is not None and correction_gradient_scale <= 0:
            raise ValueError("correction_gradient_scale must be positive")
        if correction_init_std < 0:
            raise ValueError("correction_init_std must be nonnegative")
        self.spline = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        state_after_spline = torch.random.get_rng_state()
        self.correction = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        torch.random.set_rng_state(state_after_spline)
        if correction_init_std == 0:
            nn.init.zeros_(self.correction.knot_embeddings)
        else:
            with torch.no_grad():
                self.correction.knot_embeddings.mul_(
                    float(correction_init_std) / float(embedding_init_std)
                )
        self.correction_scale = float(correction_scale)
        self.correction_init_std = float(correction_init_std)
        self.variance_preserving = bool(variance_preserving)
        self.correction_gradient_scale = float(
            correction_scale if correction_gradient_scale is None
            else correction_gradient_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.correction(x)
        weighted_correction = (
            self.correction_gradient_scale * correction
            + (self.correction_scale - self.correction_gradient_scale)
            * correction.detach()
        )
        output = self.spline(x) + weighted_correction
        if self.variance_preserving:
            output = output / (1.0 + self.correction_scale ** 2) ** 0.5
        return output


class ProjectedDualChartEmbedding(nn.Module):
    """Aligned local--smooth expert selection for numerical features.

    A vanilla DEER expert is initialized first and therefore preserves the
    established local/adaptive representation and downstream RNG stream.  A
    cubic B-spline expert is then least-squares projected into the *same*
    latent coordinate system at the initial DEER knots.  This removes the
    arbitrary rotation mismatch that makes mixtures of independently random
    numerical embeddings hard to optimize.

    A tiny static gate per field and latent channel learns which approximation
    geometry to use::

        e(x) = e_local(x) + g * (e_smooth(x) - e_local(x)).

    The forward gate is an ordinary sigmoid. ``gate_gradient_scale`` only
    improves its conditioning in backward, without changing forward values.
    No input-dependent router is used; inference is deterministic and O(d).
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        gate_init: float = 0.1,
        gate_temperature: float = 0.25,
        gate_gradient_scale: float = 32.0,
        channelwise_gate: bool = True,
        projection_mode: str = "restriction",
        projection_ridge: float = 1e-2,
    ):
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if gate_gradient_scale <= 0:
            raise ValueError("gate_gradient_scale must be positive")
        if projection_mode not in ("restriction", "ridge"):
            raise ValueError("projection_mode must be restriction or ridge")
        if projection_ridge <= 0:
            raise ValueError("projection_ridge must be positive")

        # Local-first initialization makes this a strict extension of vanilla
        # DEER and keeps all downstream parameters seed-aligned with it.
        self.local = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        state_after_local = torch.random.get_rng_state()
        self.smooth = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_local)

        # Multigrid restriction is the stable default: sample the local
        # piecewise-linear function at B-spline Greville nodes and use those
        # bounded values as coarse controls.  A ridge-Galerkin projection is
        # retained as an ablation.  Plain least squares is deliberately absent:
        # raw-scale quantiles can make its design matrix rank deficient.
        with torch.no_grad():
            design = self.smooth.basis(initial_edges.t().contiguous()).permute(
                1, 0, 2
            )
            if projection_mode == "restriction":
                greville = torch.stack(
                    [
                        self.smooth.knots[
                            i + 1 : i + self.smooth.degree + 1
                        ].mean()
                        for i in range(self.smooth.n_basis)
                    ]
                )
                points = self.local.lower.unsqueeze(-1) + greville.unsqueeze(0) * (
                    self.local.upper - self.local.lower
                ).unsqueeze(-1)
                projected = self.local(points.t().contiguous()).permute(1, 0, 2)
            else:
                projected_fields = []
                for field in range(initial_edges.size(0)):
                    matrix = design[field].double()
                    target = self.local.knot_embeddings[field].double()
                    gram = matrix.t() @ matrix
                    ridge_scale = gram.trace() / float(self.smooth.n_basis)
                    system = gram + (
                        float(projection_ridge)
                        * ridge_scale.clamp_min(1e-12)
                        * torch.eye(self.smooth.n_basis, dtype=gram.dtype)
                    )
                    solution = torch.linalg.solve(system, matrix.t() @ target)
                    projected_fields.append(
                        solution.to(self.smooth.coefficients.dtype)
                    )
                projected = torch.stack(projected_fields)
            self.smooth.coefficients.copy_(projected)
            fitted = torch.einsum("fkm,fmd->fkd", design, projected)
            error = (
                fitted - self.local.knot_embeddings.detach()
            ).square().mean(dim=(1, 2)).sqrt()
        self.register_buffer("initial_projection_rmse", error.float())
        self.projection_mode = projection_mode
        self.projection_ridge = float(projection_ridge)

        gate_shape = (
            (initial_edges.size(0), d_embedding)
            if channelwise_gate
            else (initial_edges.size(0), 1)
        )
        initial_logit = np.log(gate_init) - np.log1p(-gate_init)
        self.gate_logits = nn.Parameter(
            torch.full(
                gate_shape,
                float(gate_temperature * initial_logit),
            )
        )
        self.gate_temperature = float(gate_temperature)
        self.gate_gradient_scale = float(gate_gradient_scale)
        self.channelwise_gate = bool(channelwise_gate)

    def gates(self) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logits / self.gate_temperature)
        if self.gate_gradient_scale != 1.0:
            gate = gate.detach() + self.gate_gradient_scale * (
                gate - gate.detach()
            )
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        smooth = self.smooth(x)
        gate = self.gates().unsqueeze(0)
        return local + gate * (smooth - local)


class ResidualFunctionAdapterEmbedding(nn.Module):
    """Full-rank residual adapter between local and smooth function spaces.

    DEER and a cubic B-spline describe complementary numerical geometries but
    their latent coordinates need not align. Scalar or channel-wise convex
    gates cannot correct a rotation mismatch. This module instead learns one
    small matrix per numerical field::

        e(x) = e_local(x) + (e_smooth(x) - e_local(x)) A_f.

    The smooth expert is initialized by stable multigrid restriction of the
    local DEER function, so both charts start in the same coordinate system.
    Every adapter matrix starts at zero, making the initial forward function
    and all downstream RNG exactly identical to vanilla DEER. A bounded
    full-rank map can subsequently recover the smooth chart (A=I), retain the
    local chart (A=0), or learn a rotated combination unavailable to diagonal
    gates.

    Only F*d*d adapter scalars and F*M*d spline controls are added. The adapter
    can be absorbed into the smooth coefficients after training; evaluation
    remains two fixed-width basis lookups and one addition. This construction
    is inspired by residual adapters in NLP and structural re-parameterization
    in computer vision.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        adapter_limit: float = 1.5,
        adapter_gradient_scale: float = 32.0,
    ):
        super().__init__()
        if adapter_limit <= 0:
            raise ValueError("adapter_limit must be positive")
        if adapter_gradient_scale <= 0:
            raise ValueError("adapter_gradient_scale must be positive")
        self.local = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        state_after_local = torch.random.get_rng_state()
        self.smooth = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_local)

        # Stable multigrid restriction aligns latent coordinates without the
        # rank-deficient least-squares systems encountered on raw features.
        with torch.no_grad():
            greville = torch.stack(
                [
                    self.smooth.knots[
                        i + 1 : i + self.smooth.degree + 1
                    ].mean()
                    for i in range(self.smooth.n_basis)
                ]
            )
            points = self.local.lower.unsqueeze(-1) + greville.unsqueeze(0) * (
                self.local.upper - self.local.lower
            ).unsqueeze(-1)
            controls = self.local(points.t().contiguous()).permute(1, 0, 2)
            self.smooth.coefficients.copy_(controls)

        self.raw_adapter = nn.Parameter(
            torch.zeros(
                initial_edges.size(0), d_embedding, d_embedding
            )
        )
        self.adapter_limit = float(adapter_limit)
        self.adapter_gradient_scale = float(adapter_gradient_scale)

    def adapter_matrices(self, scale_gradient: bool = True) -> torch.Tensor:
        matrices = self.adapter_limit * torch.tanh(
            self.raw_adapter / self.adapter_limit
        )
        if scale_gradient and self.adapter_gradient_scale != 1.0:
            matrices = matrices.detach() + self.adapter_gradient_scale * (
                matrices - matrices.detach()
            )
        return matrices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        smooth = self.smooth(x)
        matrices = self.adapter_matrices(scale_gradient=True)
        correction = torch.einsum(
            "bfd,fde->bfe", smooth - local, matrices
        )
        return local + correction



class HierarchicalSchwarzEmbedding(nn.Module):
    """Full-width smooth space with an adaptive high-frequency correction.

    A numerical feature often needs a smooth global trend and a local adaptive
    correction at the same time. Routing between two full embeddings makes
    this choice an optimization problem, while splitting latent channels
    reduces the capacity of both experts. This module instead uses a
    hierarchical function-space decomposition inspired by additive Schwarz
    methods, wavelets, and multigrid finite elements::

        e(x) = S(x; C) + alpha * L(x; R).

    S is a cubic B-spline on the raw coordinate and L is a DEER interpolant on
    the quantile mesh. A single same-seed DEER table E is first restricted to
    the smooth space, then the adaptive nodal residual is initialized as
    R = E - S(edges). Consequently, for alpha=1 the combined embedding exactly
    reproduces every DEER knot while exposing both full-width spaces to task
    gradients from the first batch. There is no router, annealing schedule, or
    auxiliary loss.

    The layer adds F*M*d smooth controls to DEER's parameters and evaluates
    two compact bases. Both costs are linear in embedding width and independent
    of vocabulary size.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        detail_scale: float = 1.0,
    ):
        super().__init__()
        if detail_scale <= 0:
            raise ValueError("detail_scale must be positive")
        self.detail = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        state_after_reference = torch.random.get_rng_state()
        reference_knots = self.detail.knot_embeddings.detach().clone()
        self.smooth = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_reference)

        with torch.no_grad():
            greville = torch.stack(
                [
                    self.smooth.knots[
                        i + 1 : i + self.smooth.degree + 1
                    ].mean()
                    for i in range(self.smooth.n_basis)
                ]
            )
            control_points = (
                self.detail.lower.unsqueeze(-1)
                + greville.unsqueeze(0)
                * (self.detail.upper - self.detail.lower).unsqueeze(-1)
            )
            controls = self.detail(
                control_points.t().contiguous()
            ).permute(1, 0, 2)
            self.smooth.coefficients.copy_(controls)

            nodal_points = (
                self.detail.current_boundaries().detach().t().contiguous()
            )
            smooth_at_nodes = self.smooth(nodal_points).permute(1, 0, 2)
            self.detail.knot_embeddings.copy_(
                reference_knots - smooth_at_nodes
            )
            reconstructed = (
                self.smooth(nodal_points).permute(1, 0, 2)
                + self.detail(nodal_points).permute(1, 0, 2)
            )
            error = (reconstructed - reference_knots).abs().amax()

        self.register_buffer("initial_nodal_max_error", error.float())
        self.detail_scale = float(detail_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth(x) + self.detail_scale * self.detail(x)


class WaveSplineEmbedding(nn.Module):
    """Smooth spline trend plus balanced multiresolution semantic IDs.

    The smooth branch represents low-frequency variation with a compact
    B-spline. The detail branch treats empirical-quantile intervals as leaves
    of a balanced binary tree and assigns one signed Haar coefficient at every
    scale::

        e(x) = S(x) + gamma * sum_l w_l H_l(rank(x)).

    Each value receives a hierarchy of interval identities, from a coarse
    left/right split to a fine adjacent-bin split. This is the same
    multiresolution principle used by wavelets, image pyramids, and hierarchical
    token IDs, but remains a vocabulary-free numerical embedding. The Haar
    coefficients start at zero, so the layer is exactly a B-spline initially;
    unlike a zero-gated expert, every active coefficient receives a direct task
    gradient on the first batch.

    With L levels, the detail table has F*(2**L-1)*d parameters and each
    example activates only L vectors. Both parameter count and lookup cost are
    independent of the number of distinct numerical values.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        n_basis: int = 8,
        degree: int = 3,
        n_levels: Optional[int] = None,
        detail_scale: float = 1.0,
        level_decay: float = 1.0,
        detail_init_std: float = 0.0,
        local_order: int = 0,
        local_scale: float = 1.0,
        local_init_std: float = 0.0,
    ):
        super().__init__()
        if initial_edges.ndim != 2 or initial_edges.size(1) < 3:
            raise ValueError("initial_edges must have shape [features, bins + 1]")
        if detail_scale <= 0:
            raise ValueError("detail_scale must be positive")
        if not 0 < level_decay <= 1:
            raise ValueError("level_decay must be in (0, 1]")
        if detail_init_std < 0:
            raise ValueError("detail_init_std must be nonnegative")
        if not 0 <= local_order <= 2:
            raise ValueError("local_order must be in [0, 2]")
        if local_scale <= 0:
            raise ValueError("local_scale must be positive")
        if local_init_std < 0:
            raise ValueError("local_init_std must be nonnegative")
        n_bins = int(initial_edges.size(1) - 1)
        max_levels = int(np.floor(np.log2(n_bins)))
        if n_levels is None:
            n_levels = max_levels
        n_levels = int(n_levels)
        if not 1 <= n_levels <= max_levels:
            raise ValueError(
                f"n_levels must be between 1 and {max_levels} for {n_bins} bins"
            )
        if n_bins % (2 ** n_levels) != 0:
            raise ValueError("n_bins must be divisible by 2 ** n_levels")

        self.smooth = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        self.register_buffer("edges", initial_edges.float().clone())
        self.n_bins = n_bins
        self.n_levels = n_levels
        self.detail_scale = float(detail_scale)
        self.local_order = int(local_order)
        self.local_scale = float(local_scale)
        n_features = initial_edges.size(0)
        n_nodes = 2 ** n_levels - 1
        self.detail_coefficients = nn.Parameter(
            torch.empty(n_features, n_nodes, d_embedding)
        )
        if detail_init_std == 0:
            nn.init.zeros_(self.detail_coefficients)
        else:
            state = torch.random.get_rng_state()
            nn.init.normal_(
                self.detail_coefficients, std=float(detail_init_std)
            )
            torch.random.set_rng_state(state)
        if self.local_order:
            self.local_coefficients = nn.Parameter(
                torch.empty(
                    n_features, n_bins, self.local_order, d_embedding
                )
            )
            if local_init_std == 0:
                nn.init.zeros_(self.local_coefficients)
            else:
                state = torch.random.get_rng_state()
                nn.init.normal_(
                    self.local_coefficients, std=float(local_init_std)
                )
                torch.random.set_rng_state(state)
        else:
            self.register_parameter("local_coefficients", None)
        weights = level_decay ** torch.arange(n_levels, dtype=torch.float32)
        weights = weights / weights.square().sum().sqrt()
        self.register_buffer("level_weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.smooth(x)
        clipped = torch.maximum(
            torch.minimum(x, self.edges[:, -1]), self.edges[:, 0]
        )
        interval = (
            clipped.unsqueeze(-1) >= self.edges[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        field = torch.arange(
            interval.size(1), device=interval.device
        ).unsqueeze(0).expand_as(interval)
        detail = torch.zeros_like(smooth)
        offset = 0
        for level in range(self.n_levels):
            groups = 2 ** level
            span = self.n_bins // groups
            half = span // 2
            parent = interval // span
            sign = torch.where(
                interval.remainder(span) < half,
                torch.ones_like(interval),
                -torch.ones_like(interval),
            )
            coefficient = self.detail_coefficients[field, offset + parent]
            detail = detail + (
                self.level_weights[level] * sign.unsqueeze(-1) * coefficient
            )
            offset += groups
        output = smooth + self.detail_scale * detail
        if self.local_order:
            expanded_edges = self.edges.unsqueeze(0).expand(
                x.size(0), -1, -1
            )
            left = expanded_edges.gather(
                2, interval.unsqueeze(-1)
            ).squeeze(-1)
            right = expanded_edges.gather(
                2, (interval + 1).unsqueeze(-1)
            ).squeeze(-1)
            alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
                0.0, 1.0
            )
            bubble = 4.0 * alpha * (1.0 - alpha)
            local_basis = [bubble]
            if self.local_order == 2:
                local_basis.append(bubble * (2.0 * alpha - 1.0))
            basis = torch.stack(local_basis, dim=-1)
            coefficients = self.local_coefficients[field, interval]
            local = (basis.unsqueeze(-1) * coefficients).sum(dim=-2)
            output = output + self.local_scale * local
        return output

class WaveSpecEmbedding(WaveSplineEmbedding):
    """Orthogonal supervised chart on a multiresolution spectral element.

    The unsupervised chart is a WaveSpline representation projected onto a
    (d-1)-dimensional subspace. Its orthogonal complement is initialized with a
    train-only empirical-Bayes response curve on the same quantile mesh. This
    is a direct-sum construction rather than a gated mixture: the task-aware
    coordinate cannot destructively interfere with the smooth, hierarchical,
    or cell-interior coordinates, while an ordinary downstream linear map can
    select either chart.

    Only one scalar knot value per field and quantile boundary is added.
    Inference remains a two-knot interpolation plus the WaveSpline lookups, and
    construction consumes exactly the same random-number stream as WaveSpline.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        response_knots: Sequence[Sequence[float]],
        n_basis: int = 8,
        degree: int = 3,
        n_levels: Optional[int] = None,
        detail_scale: float = 1.0,
        level_decay: float = 1.0,
        detail_init_std: float = 0.0,
        local_order: int = 0,
        local_scale: float = 1.0,
        local_init_std: float = 0.0,
        response_scale: float = 1.0,
        response_clip: float = 3.0,
        embedding_init_std: float = 1e-4,
    ):
        if d_embedding < 2:
            raise ValueError("wavespec requires d_embedding >= 2")
        if response_scale < 0:
            raise ValueError("response_scale must be nonnegative")
        if response_clip <= 0:
            raise ValueError("response_clip must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
            n_levels=n_levels,
            detail_scale=detail_scale,
            level_decay=level_decay,
            detail_init_std=detail_init_std,
            local_order=local_order,
            local_scale=local_scale,
            local_init_std=local_init_std,
        )
        response = torch.as_tensor(response_knots, dtype=initial_edges.dtype)
        if tuple(response.shape) != tuple(initial_edges.shape):
            raise ValueError(
                "response_knots must have the same shape as initial_edges"
            )
        if not torch.isfinite(response).all():
            raise ValueError("response_knots must be finite")
        centered = response - response.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        standardized = centered / rms.clamp_min(1e-6)
        standardized = standardized.clamp(
            -float(response_clip), float(response_clip)
        )
        standardized = torch.where(
            rms <= 1e-6, torch.zeros_like(standardized), standardized
        )
        direction = torch.ones(d_embedding, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(1e-12)
        self.register_buffer("response_direction", direction)
        initial_response = (
            float(response_scale)
            * float(embedding_init_std)
            * float(d_embedding) ** 0.5
            * standardized
        )
        self.response_knot_values = nn.Parameter(initial_response.float())

        # Reserve exactly one orthogonal coordinate for the response chart.
        # Projection is deterministic and therefore preserves downstream RNG.
        with torch.no_grad():
            parameters = [self.smooth.coefficients, self.detail_coefficients]
            if self.local_coefficients is not None:
                parameters.append(self.local_coefficients)
            for parameter in parameters:
                parallel = torch.einsum(
                    "...d,d->...", parameter, self.response_direction
                )
                parameter.sub_(
                    parallel.unsqueeze(-1) * self.response_direction
                )

    def response_field(self, x: torch.Tensor) -> torch.Tensor:
        clipped = torch.maximum(
            torch.minimum(x, self.edges[:, -1]), self.edges[:, 0]
        )
        interval = (
            clipped.unsqueeze(-1) >= self.edges[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_edges = self.edges.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_edges.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_edges.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        values = self.response_knot_values.unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        left_value = values.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right_value = values.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        return (1.0 - alpha) * left_value + alpha * right_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        response = self.response_field(x)
        return base + response.unsqueeze(-1) * self.response_direction



class ProlongatedSwitchEmbedding(nn.Module):
    """Function-preserving switch between smooth and adaptive numerical bases.

    A coarse cubic B-spline is initialized first. Its learned random function is
    then prolonged onto the quantile DEER mesh by evaluating the spline at
    every adaptive knot. Both experts therefore begin in the same embedding
    coordinate system, so switching never exposes the backbone to unrelated
    random codes.

    Soft routing learns a convex continuation between bases. Stochastic routing
    uses a straight-through binary Concrete switch during training and a
    deterministic switch at inference. An optional delayed double-well penalty
    polarizes gates after the task loss has supplied a preferred direction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        gate_init: float = 0.5,
        gate_temperature: float = 0.25,
        routing: str = "soft",
        anchor_gradient: bool = False,
        polarization_weight: float = 0.0,
        polarization_warmup_steps: int = 0,
        polarization_ramp_steps: int = 1,
    ):
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if routing not in ("soft", "stochastic"):
            raise ValueError("routing must be soft or stochastic")
        if polarization_weight < 0:
            raise ValueError("polarization_weight must be nonnegative")
        if polarization_warmup_steps < 0 or polarization_ramp_steps <= 0:
            raise ValueError("invalid polarization schedule")

        self.spline = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        state_after_spline = torch.random.get_rng_state()
        self.local = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        torch.random.set_rng_state(state_after_spline)

        # Multigrid prolongation initializes the adaptive expert from exact
        # coarse function values at its knots without consuming random numbers.
        with torch.no_grad():
            knot_values = self.spline(initial_edges.t().contiguous())
            self.local.knot_embeddings.copy_(knot_values.permute(1, 0, 2))

        initial_logit = np.log(gate_init) - np.log1p(-gate_init)
        self.gate_logits = nn.Parameter(
            torch.full(
                (initial_edges.size(0),),
                float(gate_temperature * initial_logit),
            )
        )
        self.gate_temperature = float(gate_temperature)
        self.routing = routing
        self.anchor_gradient = bool(anchor_gradient)
        self.polarization_weight = float(polarization_weight)
        self.polarization_warmup_steps = int(polarization_warmup_steps)
        self.polarization_ramp_steps = int(polarization_ramp_steps)
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        self._last_auxiliary_loss: Optional[torch.Tensor] = None

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def get_auxiliary_loss(self) -> torch.Tensor:
        if self._last_auxiliary_loss is None:
            return self.gate_logits.new_zeros(())
        return self._last_auxiliary_loss

    def _route(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.training and self.routing == "stochastic":
            uniform = torch.rand_like(probabilities).clamp(1e-6, 1.0 - 1e-6)
            logistic = uniform.log() - torch.log1p(-uniform)
            relaxed = torch.sigmoid(
                (self.gate_logits + logistic) / self.gate_temperature
            )
            hard = (relaxed >= 0.5).to(relaxed.dtype)
            return hard.detach() - relaxed.detach() + relaxed
        if not self.training and self.routing == "stochastic":
            return (probabilities >= 0.5).to(probabilities.dtype)
        return probabilities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.gates()
        route = self._route(probabilities).view(1, -1, 1)
        local = self.local(x)
        smooth = self.spline(x)
        if self.anchor_gradient:
            output = smooth + route * (local - smooth.detach())
        else:
            output = local + route * (smooth - local)

        if self.training and self.polarization_weight > 0:
            step = int(self.training_steps.item())
            progress = max(
                0.0,
                min(
                    1.0,
                    (step - self.polarization_warmup_steps)
                    / float(self.polarization_ramp_steps),
                ),
            )
            self._last_auxiliary_loss = (
                self.polarization_weight
                * progress
                * (probabilities * (1.0 - probabilities)).mean()
            )
        else:
            self._last_auxiliary_loss = output.new_zeros(())
        if self.training:
            self.training_steps.add_(1)
        return output


class LowRankMultigridEmbedding(nn.Module):
    """Parameter-efficient multigrid correction in a shared low-rank subspace.

    The coarse path is a full-width B-spline.  The adaptive DEER detail has
    only ``rank`` channels and is lifted to the embedding space by a shared
    learned matrix.  This separable tensor representation changes the adaptive
    parameter cost from ``F*(K+1)*d`` to ``F*(K+1)*r + r*d``.  Detail knot
    values start at zero, so initialization is exactly the B-spline control and
    the downstream RNG stream is unchanged.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        correction_rank: int = 2,
        correction_scale: float = 1.0,
    ):
        super().__init__()
        if not 1 <= correction_rank <= d_embedding:
            raise ValueError("correction_rank must lie in [1, d_embedding]")
        if correction_scale <= 0:
            raise ValueError("correction_scale must be positive")
        self.spline = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_embedding,
            n_basis=n_basis,
            degree=degree,
        )
        state_after_spline = torch.random.get_rng_state()
        self.correction = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=correction_rank,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        self.detail_directions = nn.Parameter(
            torch.empty(correction_rank, d_embedding)
        )
        nn.init.orthogonal_(self.detail_directions)
        torch.random.set_rng_state(state_after_spline)
        nn.init.zeros_(self.correction.knot_embeddings)
        self.correction_rank = int(correction_rank)
        self.correction_scale = float(correction_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.correction(x) @ self.detail_directions
        return self.spline(x) + self.correction_scale * detail


class ComplementarySubspaceEmbedding(nn.Module):
    """Parameter-efficient direct sum of adaptive and smooth function spaces.

    The output channels are partitioned between an adaptive DEER expert and a
    cubic B-spline expert and concatenated, rather than averaged.  A downstream
    backbone therefore assigns independent weights to local/adaptive and
    global/smooth coordinates.  With ``K+1`` DEER knots and ``M`` spline bases,
    parameter cost is ``F[K + (1-r)(K+1)d + rMd]``; for the default
    ``K=16, M=8, r=0.5`` this is smaller than a 16-dimensional DEER.

    Initialization explicitly restores the RNG state that a full-width DEER
    would have produced, keeping downstream backbone weights seed-aligned.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_basis: int = 8,
        degree: int = 3,
        spline_fraction: float = 0.5,
    ):
        super().__init__()
        if d_embedding < 2:
            raise ValueError("d_embedding must be at least two")
        if not 0.0 < spline_fraction < 1.0:
            raise ValueError("spline_fraction must be in (0, 1)")
        d_spline = max(1, min(d_embedding - 1, int(round(d_embedding * spline_fraction))))
        d_deer = d_embedding - d_spline
        state_before = torch.random.get_rng_state()
        _ = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        state_after_full_deer = torch.random.get_rng_state()
        torch.random.set_rng_state(state_before)
        self.deer = DEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_deer,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        self.spline = BSplineEmbedding(
            lower=initial_edges[:, 0],
            upper=initial_edges[:, -1],
            d_embedding=d_spline,
            n_basis=n_basis,
            degree=degree,
        )
        torch.random.set_rng_state(state_after_full_deer)
        self.d_deer = d_deer
        self.d_spline = d_spline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.deer(x), self.spline(x)], dim=-1)


class HierarchicalBubbleDEEREmbedding(DEEREmbedding):
    """DEER with a low-rank, endpoint-zero local finite-element residual.

    Vanilla DEER is a continuous piecewise-linear (P1) finite element on a
    learned one-dimensional mesh.  This module performs hierarchical
    p-refinement by adding the quadratic bubble ``4 t (1 - t)`` inside each
    active interval.  The bubble vanishes at both endpoints, so adjacent
    intervals remain continuous and the learned DEER knot values keep their
    exact meaning.

    Interval amplitudes and field-specific embedding directions are low-rank::

        e(x) = e_DEER(x) + 4 t (1-t) sum_r a[f,k,r] v[f,r].

    Zero amplitudes recover DEER exactly while non-zero direction
    initialization gives the amplitudes a useful first-step gradient.  The
    extension adds ``F * R * (K + d)`` parameters and preserves O(dR) local
    evaluation without evaluating a dense bank of experts.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        residual_rank: int = 1,
        coefficient_init_std: float = 0.0,
        direction_init_scale: float = 1.0,
        residual_scale_limit: float = 0.5,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if residual_rank < 1:
            raise ValueError("residual_rank must be positive")
        if coefficient_init_std < 0:
            raise ValueError("coefficient_init_std must be nonnegative")
        if direction_init_scale <= 0:
            raise ValueError("direction_init_scale must be positive")
        if residual_scale_limit <= 0:
            raise ValueError("residual_scale_limit must be positive")
        self.residual_rank = int(residual_rank)
        self.residual_scale_limit = float(residual_scale_limit)
        n_features = initial_edges.size(0)
        self.raw_bubble_coefficients = nn.Parameter(
            torch.empty(n_features, self.n_bins, self.residual_rank)
        )
        self.bubble_directions = nn.Parameter(
            torch.empty(n_features, self.residual_rank, d_embedding)
        )
        if coefficient_init_std == 0:
            nn.init.zeros_(self.raw_bubble_coefficients)
        else:
            nn.init.normal_(
                self.raw_bubble_coefficients,
                std=float(coefficient_init_std),
            )
        # A deterministic orthonormal DCT basis avoids consuming the global RNG.
        # Consequently a same-seed DEER comparison also preserves initialization
        # of every downstream backbone parameter.
        positions = torch.arange(d_embedding, dtype=torch.float32) + 0.5
        frequencies = torch.arange(self.residual_rank, dtype=torch.float32)
        phases = torch.pi * frequencies[:, None] * positions[None, :] / d_embedding
        basis = torch.cos(phases)
        basis[0].mul_(2.0 ** -0.5)
        basis.mul_((2.0 / d_embedding) ** 0.5 * float(direction_init_scale))
        with torch.no_grad():
            self.bubble_directions.copy_(basis.unsqueeze(0).expand(n_features, -1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        knots = self.current_knot_embeddings()
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )

        expanded_embeddings = knots.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, knots.size(-1)
        )
        left_embedding = expanded_embeddings.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_embeddings.gather(
            2, gather_index + 1
        ).squeeze(2)
        linear = (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )

        coefficients = self.residual_scale_limit * torch.tanh(
            self.raw_bubble_coefficients
        )
        expanded_coefficients = coefficients.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        active_coefficients = expanded_coefficients.gather(
            2,
            interval.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, self.residual_rank
            ),
        ).squeeze(2)
        residual = torch.einsum(
            "bfr,frd->bfd", active_coefficients, self.bubble_directions
        )
        bubble = 4.0 * alpha * (1.0 - alpha)
        return linear + bubble.unsqueeze(-1) * residual


class MixtureGridDEEREmbedding(DEEREmbedding):
    """DEER with a residual mixture of two monotone coordinate grids.

    The base and alternative experts share the same knot embeddings and differ
    only in their ordered boundary grids. A continuous, field-wise gate mixes
    their interpolants in empirical-rank space::

        e(x) = e_0(x) + tanh(s_f) * sigmoid(a_f(q_f(x)-.5) + c_f)
                           * (e_1(x) - e_0(x)).

    ``q_f`` is the fractional base-grid position in ``[0, 1]``. With
    ``s_f=0`` the module is exactly vanilla DEER, while the perturbed second
    grid gives the mixture scale a non-zero gradient at initialization. The
    extension adds only ``F * (B + 3)`` trainable parameters for ``F`` fields
    and ``B`` intervals; knot embeddings are not duplicated.

    This is an experimental research candidate, not a paper reproduction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        expert_boundary_init_noise: float = 0.02,
        mixture_scale_limit: float = 1.0,
        gate_slope_init_std: float = 0.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if expert_boundary_init_noise < 0:
            raise ValueError("expert_boundary_init_noise must be nonnegative")
        if not 0 < mixture_scale_limit <= 1.0:
            raise ValueError("mixture_scale_limit must be in (0, 1]")
        if gate_slope_init_std < 0:
            raise ValueError("gate_slope_init_std must be nonnegative")

        alternative_widths = self.raw_widths.detach().clone()
        if expert_boundary_init_noise > 0:
            noise = torch.empty_like(alternative_widths)
            nn.init.uniform_(noise, -1.0, 1.0)
            alternative_widths = (
                alternative_widths
                + float(expert_boundary_init_noise) * noise
            )
        self.alternative_raw_widths = nn.Parameter(alternative_widths)
        self.raw_mixture_scales = nn.Parameter(
            torch.zeros(initial_edges.size(0))
        )
        self.gate_slopes = nn.Parameter(torch.empty(initial_edges.size(0)))
        if gate_slope_init_std == 0:
            nn.init.zeros_(self.gate_slopes)
        else:
            nn.init.normal_(
                self.gate_slopes, std=float(gate_slope_init_std)
            )
        self.gate_biases = nn.Parameter(torch.zeros(initial_edges.size(0)))
        self.mixture_scale_limit = float(mixture_scale_limit)

    def current_alternative_boundaries(self) -> torch.Tensor:
        fractions = torch.softmax(
            self.alternative_raw_widths / self.width_temperature, dim=-1
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

    def mixture_scales(self) -> torch.Tensor:
        return self.mixture_scale_limit * torch.tanh(
            self.raw_mixture_scales
        )

    def _interpolate(
        self,
        clipped: torch.Tensor,
        boundaries: torch.Tensor,
        knots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(
            clipped.size(0), -1, -1
        )
        left = expanded_boundaries.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        expanded_knots = knots.unsqueeze(0).expand(
            clipped.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, knots.size(-1)
        )
        left_embedding = expanded_knots.gather(
            2, gather_index
        ).squeeze(2)
        right_embedding = expanded_knots.gather(
            2, gather_index + 1
        ).squeeze(2)
        output = (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )
        rank_position = (interval.to(alpha.dtype) + alpha) / self.n_bins
        return output, rank_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        knots = self.current_knot_embeddings()
        base, rank_position = self._interpolate(
            clipped, self.current_boundaries(), knots
        )
        alternative, _ = self._interpolate(
            clipped, self.current_alternative_boundaries(), knots
        )
        gate = torch.sigmoid(
            self.gate_slopes.unsqueeze(0) * (rank_position - 0.5)
            + self.gate_biases.unsqueeze(0)
        )
        scale = self.mixture_scales().view(1, -1, 1)
        return base + scale * gate.unsqueeze(-1) * (alternative - base)


class DiffusionDEEREmbedding(DEEREmbedding):
    """DEER with a bounded, field-wise diffusion step on the knot graph.

    The ordered knot table is a path graph. Before interpolation, one explicit
    heat-equation step filters neighbouring knot embeddings:

        E' = E + tau * L(E).

    tau is one learned scalar per numerical field and is bounded by
    diffusion_limit * tanh(raw_tau). Positive values smooth local
    oscillations; negative values perform a bounded sharpening step. Zero
    exactly recovers DEER and has a non-zero gradient. The filtered table can
    be fused after training, so inference keeps DEER's two-knot O(d) lookup.

    This is an experimental research candidate, not a paper reproduction.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        diffusion_limit: float = 0.25,
        diffusion_init_std: float = 0.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 0 < diffusion_limit <= 0.5:
            raise ValueError("diffusion_limit must be in (0, 0.5]")
        if diffusion_init_std < 0:
            raise ValueError("diffusion_init_std must be nonnegative")
        self.diffusion_limit = float(diffusion_limit)
        self.raw_diffusion_times = nn.Parameter(
            torch.empty(initial_edges.size(0))
        )
        if diffusion_init_std == 0:
            nn.init.zeros_(self.raw_diffusion_times)
        else:
            nn.init.normal_(
                self.raw_diffusion_times, std=float(diffusion_init_std)
            )

    def diffusion_times(self) -> torch.Tensor:
        return self.diffusion_limit * torch.tanh(self.raw_diffusion_times)

    def current_knot_embeddings(self) -> torch.Tensor:
        knots = self.knot_embeddings
        laplacian = torch.empty_like(knots)
        laplacian[:, 0] = knots[:, 1] - knots[:, 0]
        laplacian[:, -1] = knots[:, -2] - knots[:, -1]
        laplacian[:, 1:-1] = (
            knots[:, :-2] - 2.0 * knots[:, 1:-1] + knots[:, 2:]
        )
        return knots + self.diffusion_times().view(-1, 1, 1) * laplacian

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        knots = self.current_knot_embeddings()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = (
            (clipped - left) / (right - left).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        expanded_knots = knots.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, knots.size(-1)
        )
        left_embedding = expanded_knots.gather(
            2, gather_index
        ).squeeze(2)
        right_embedding = expanded_knots.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )



class TrendResidualDEEREmbedding(DEEREmbedding):
    """DEER with a bounded field-wise scale on local trend residuals.

    Each knot table is decomposed into the global linear trend joining its
    endpoint embeddings and a local residual. One learned scalar per field
    controls the residual amplitude::

        E'_k = T_k + s_f (E_k - T_k),
        s_f = 1 + scale_limit * tanh(raw_scale_f).

    Here ``T_k`` evaluates the endpoint chord at the learned boundary of knot
    ``k``. A zero raw scale exactly recovers DEER with a non-zero gradient.
    The transformed knots can be fused after training, preserving DEER's
    two-knot O(d) inference. This is an experimental research candidate.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        residual_scale_limit: float = 0.5,
        residual_scale_init_std: float = 0.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 0 < residual_scale_limit < 1.0:
            raise ValueError("residual_scale_limit must be in (0, 1)")
        if residual_scale_init_std < 0:
            raise ValueError("residual_scale_init_std must be nonnegative")
        self.residual_scale_limit = float(residual_scale_limit)
        self.raw_residual_scales = nn.Parameter(
            torch.empty(initial_edges.size(0))
        )
        if residual_scale_init_std == 0:
            nn.init.zeros_(self.raw_residual_scales)
        else:
            nn.init.normal_(
                self.raw_residual_scales,
                std=float(residual_scale_init_std),
            )

    def residual_scales(self) -> torch.Tensor:
        return 1.0 + self.residual_scale_limit * torch.tanh(
            self.raw_residual_scales
        )

    def current_knot_embeddings(self) -> torch.Tensor:
        boundaries = self.current_boundaries()
        normalized_positions = (
            (boundaries - self.lower.unsqueeze(-1))
            / (self.upper - self.lower).clamp_min(1e-12).unsqueeze(-1)
        ).clamp(0.0, 1.0)
        normalized_positions = torch.cat(
            [
                torch.zeros_like(normalized_positions[:, :1]),
                normalized_positions[:, 1:-1],
                torch.ones_like(normalized_positions[:, -1:]),
            ], dim=-1
        )
        left_endpoint = self.knot_embeddings[:, :1]
        right_endpoint = self.knot_embeddings[:, -1:]
        trend = (
            (1.0 - normalized_positions).unsqueeze(-1) * left_endpoint
            + normalized_positions.unsqueeze(-1) * right_endpoint
        )
        scale = self.residual_scales().view(-1, 1, 1)
        return trend + scale * (self.knot_embeddings - trend)



class AtomicDEEREmbedding(DEEREmbedding):
    """DEER for mixed discrete-continuous numerical distributions.

    A continuous interpolant must assign the same limit to an isolated mass
    point and its numerical neighbours. Real tabular fields violate this
    assumption: missing sentinels, zero inflation, and quantized defaults can
    carry separate semantics. This layer uses the measure decomposition

        e(x) = e_continuous(x) + 1[x = a_f] r_f,

    where ``e_continuous`` is DEER, ``a_f`` is a train-split atom, and ``r_f``
    is one residual embedding per field. Atom residuals start at zero, making
    initialization exactly DEER. The full extension adds F*d parameters. With
    ``atom_rank=1``, residuals factor as a field coefficient times one shared
    direction and add only F+d parameters. Both forms require one exact
    comparison and one vector addition per value.

    Optional train-split atom frequencies disable rare or constant atoms. No
    validation or test statistics are consumed by the layer.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        atom_values: Optional[Sequence[float]] = None,
        atom_frequencies: Optional[Sequence[float]] = None,
        atom_scores: Optional[Sequence[float]] = None,
        min_atom_score: float = 0.0,
        min_atom_frequency: float = 0.0,
        max_atom_frequency: float = 1.0,
        atom_tolerance: float = 0.0,
        center_atoms: bool = False,
        atom_rank: Optional[int] = None,
        atom_direction_init_std: Optional[float] = None,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        n_features = initial_edges.size(0)
        if not 0.0 <= min_atom_frequency <= max_atom_frequency <= 1.0:
            raise ValueError("atom-frequency bounds must lie in [0, 1]")
        if min_atom_score < 0:
            raise ValueError("min_atom_score must be nonnegative")
        if atom_tolerance < 0:
            raise ValueError("atom_tolerance must be nonnegative")
        if atom_rank is not None and atom_rank < 1:
            raise ValueError("atom_rank must be positive or None")
        values = (
            initial_edges[:, 0].detach().clone()
            if atom_values is None
            else torch.as_tensor(atom_values, dtype=initial_edges.dtype)
        )
        if values.numel() != n_features:
            raise ValueError("atom_values must contain one value per field")
        if atom_frequencies is None:
            frequencies = torch.ones(n_features, dtype=initial_edges.dtype)
        else:
            frequencies = torch.as_tensor(
                atom_frequencies, dtype=initial_edges.dtype
            )
        if frequencies.numel() != n_features:
            raise ValueError("atom_frequencies must contain one value per field")
        if atom_scores is None:
            scores = torch.full(
                (n_features,), float("inf"), dtype=initial_edges.dtype
            )
        else:
            scores = torch.as_tensor(atom_scores, dtype=initial_edges.dtype)
        if scores.numel() != n_features:
            raise ValueError("atom_scores must contain one value per field")
        active = (
            (frequencies >= min_atom_frequency)
            & (frequencies <= max_atom_frequency)
            & (scores >= min_atom_score)
        )
        self.atom_tolerance = float(atom_tolerance)
        self.center_atoms = bool(center_atoms)
        self.register_buffer("atom_values", values.reshape(n_features))
        self.register_buffer(
            "atom_frequencies", frequencies.reshape(n_features)
        )
        self.register_buffer("active_atoms", active.reshape(n_features))
        self.register_buffer("atom_scores", scores.reshape(n_features))
        self.atom_rank = atom_rank
        if atom_rank is None:
            self.atom_residuals = nn.Parameter(
                torch.zeros(n_features, d_embedding)
            )
        else:
            init_std = (
                d_embedding ** -0.5
                if atom_direction_init_std is None
                else float(atom_direction_init_std)
            )
            self.atom_coefficients = nn.Parameter(
                torch.zeros(n_features, atom_rank)
            )
            self.atom_directions = nn.Parameter(
                torch.empty(atom_rank, d_embedding)
            )
            nn.init.normal_(self.atom_directions, std=init_std)

    def atom_residual_matrix(self) -> torch.Tensor:
        if self.atom_rank is None:
            return self.atom_residuals
        return self.atom_coefficients @ self.atom_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        distance = (x - self.atom_values.unsqueeze(0)).abs()
        basis = (distance <= self.atom_tolerance).to(output.dtype)
        active = self.active_atoms.to(output.dtype).unsqueeze(0)
        basis = basis * active
        if self.center_atoms:
            basis = basis - self.atom_frequencies.unsqueeze(0) * active
        residuals = self.atom_residual_matrix()
        return output + basis.unsqueeze(-1) * residuals.unsqueeze(0)


class AsinhWarpDEEREmbedding(DEEREmbedding):
    """DEER with a safe field-wise blend toward asinh interpolation.

    Raw numerical values with heavy tails are often easier to interpolate in
    a logarithmic coordinate. For every active interval this layer computes
    both the original DEER coordinate and an ``asinh`` coordinate, then uses
    one projected coefficient per field to form a convex blend. Coefficients
    start at exactly zero, so initialization and all original DEER gradients
    are unchanged; a field can only opt into the robust warp when useful.

    ``asinh`` is monotone, defined for signed inputs, linear near zero, and
    logarithmic in the tails. The method adds one scalar per field and keeps
    the same two-knot O(d) interpolation cost, plus scalar ``asinh`` calls.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        warp_limit: float = 1.0,
        warp_init: float = 0.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 0 < warp_limit <= 1.0:
            raise ValueError("warp_limit must be in (0, 1]")
        if not 0 <= warp_init <= warp_limit:
            raise ValueError("warp_init must be in [0, warp_limit]")
        self.warp_limit = float(warp_limit)
        self.raw_warp_coefficients = nn.Parameter(
            torch.full((initial_edges.size(0),), float(warp_init))
        )
        robust_scale = torch.median(initial_edges.abs(), dim=1).values
        fallback = 0.5 * (initial_edges[:, -1] - initial_edges[:, 0]).abs()
        robust_scale = torch.where(
            robust_scale > 1e-6, robust_scale, fallback
        ).clamp_min(1e-6)
        self.register_buffer("asinh_scale", robust_scale.float())

    def warp_coefficients(self) -> torch.Tensor:
        return self.raw_warp_coefficients.clamp(0.0, self.warp_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        width = (right - left).clamp_min(1e-12)
        linear_alpha = ((clipped - left) / width).clamp(0.0, 1.0)
        scale = self.asinh_scale.unsqueeze(0)
        warped_value = torch.asinh(clipped / scale)
        warped_left = torch.asinh(left / scale)
        warped_right = torch.asinh(right / scale)
        warped_alpha = (
            (warped_value - warped_left)
            / (warped_right - warped_left).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        blend = self.warp_coefficients().unsqueeze(0)
        alpha = linear_alpha + blend * (warped_alpha - linear_alpha)

        expanded_knots = self.knot_embeddings.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.knot_embeddings.size(-1)
        )
        left_embedding = expanded_knots.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_knots.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )
class MobiusDEEREmbedding(DEEREmbedding):
    """DEER with one shape parameter per interval.

    The learned boundaries and knot embeddings are identical to DEER. Only
    the local interpolation coordinate changes:

        phi(t; a) = t / (t + exp(a) * (1 - t)).

    a=0 recovers DEER. For every finite a, phi preserves both endpoints, is
    strictly increasing on [0, 1], and remains inside [0, 1]. Thus the output
    stays in the convex hull of the two active knot embeddings and cannot
    overshoot. The extension adds one scalar per interval while preserving
    two-knot local support and O(d) evaluation.

    This is an experimental research candidate, not a claim of paper-faithful
    reproduction. log_scale_limit bounds the local condition number and
    prevents nearly step-like warps during early screening.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        warp_init_std: float = 0.0,
        log_scale_limit: float = 4.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if warp_init_std < 0:
            raise ValueError("warp_init_std must be nonnegative")
        if log_scale_limit <= 0:
            raise ValueError("log_scale_limit must be positive")
        self.log_scale_limit = float(log_scale_limit)
        self.raw_log_scales = nn.Parameter(
            torch.empty(initial_edges.size(0), self.n_bins)
        )
        if warp_init_std == 0:
            nn.init.zeros_(self.raw_log_scales)
        else:
            nn.init.normal_(self.raw_log_scales, std=float(warp_init_std))

    def interval_log_scales(self) -> torch.Tensor:
        return self.raw_log_scales.clamp(
            -self.log_scale_limit, self.log_scale_limit
        )

    @staticmethod
    def warp_coordinate(
        alpha: torch.Tensor, log_scale: torch.Tensor
    ) -> torch.Tensor:
        scale = torch.exp(log_scale)
        denominator = alpha + scale * (1.0 - alpha)
        return alpha / denominator.clamp_min(torch.finfo(alpha.dtype).tiny)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        log_scales = self.interval_log_scales().unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        active_log_scale = log_scales.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        warped_alpha = self.warp_coordinate(alpha, active_log_scale)
        expanded_embeddings = self.knot_embeddings.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.knot_embeddings.size(-1)
        )
        left_embedding = expanded_embeddings.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_embeddings.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (
            (1.0 - warped_alpha).unsqueeze(-1) * left_embedding
            + warped_alpha.unsqueeze(-1) * right_embedding
        )


class SharedMobiusDEEREmbedding(MobiusDEEREmbedding):
    """Low-variance Mobius DEER with one shape scalar per numeric field.

    The interval-wise version is expressive but can fit unrelated local
    fluctuations on smaller datasets. This variant shares the same monotone
    coordinate warp across all intervals of a field. It therefore adds only
    ``n_features`` scalars to DEER, while retaining exact identity
    initialization, endpoint preservation, convex-hull interpolation, and
    O(d) evaluation.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        warp_init_std: float = 0.0,
        log_scale_limit: float = 4.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
            warp_init_std=warp_init_std,
            log_scale_limit=log_scale_limit,
        )
        self.raw_log_scales = nn.Parameter(
            self.raw_log_scales[:, :1].detach().clone()
        )

    def interval_log_scales(self) -> torch.Tensor:
        return super().interval_log_scales().expand(-1, self.n_bins)


class QuadraticDEEREmbedding(DEEREmbedding):
    """DEER with a field-shared, endpoint-preserving quadratic warp.

    For local coordinate ``t``, use ``t + beta * t * (1 - t)`` with
    ``beta = shape_limit * tanh(raw_shape)`` and ``shape_limit < 1``.
    The map fixes both endpoints, is strictly increasing, and stays in [0, 1].
    It adds one scalar per field and avoids exponentials in the hot path.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        shape_init_std: float = 0.0,
        shape_limit: float = 0.99,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if shape_init_std < 0:
            raise ValueError("shape_init_std must be nonnegative")
        if not 0 < shape_limit < 1:
            raise ValueError("shape_limit must lie strictly between zero and one")
        self.shape_limit = float(shape_limit)
        self.raw_shapes = nn.Parameter(torch.empty(initial_edges.size(0), 1))
        if shape_init_std == 0:
            nn.init.zeros_(self.raw_shapes)
        else:
            nn.init.normal_(self.raw_shapes, std=float(shape_init_std))

    @staticmethod
    def warp_coordinate(
        alpha: torch.Tensor, raw_shape: torch.Tensor, shape_limit: float
    ) -> torch.Tensor:
        beta = float(shape_limit) * torch.tanh(raw_shape)
        return alpha + beta * alpha * (1.0 - alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        raw_shape = self.raw_shapes.unsqueeze(0).expand(x.size(0), -1, -1)
        warped_alpha = self.warp_coordinate(
            alpha, raw_shape.squeeze(-1), self.shape_limit
        )
        expanded_embeddings = self.knot_embeddings.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.knot_embeddings.size(-1)
        )
        left_embedding = expanded_embeddings.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_embeddings.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (
            (1.0 - warped_alpha).unsqueeze(-1) * left_embedding
            + warped_alpha.unsqueeze(-1) * right_embedding
        )


class HermiteDEEREmbedding(DEEREmbedding):
    """DEER plus a zero-initialized nonuniform cubic Hermite residual.

    The active DEER segment supplies the linear path. Neighboring knot
    embeddings define finite-difference endpoint slopes on the learned,
    potentially nonuniform boundary grid. A bounded field-by-dimension gate
    blends the resulting cubic Hermite path with the original linear path.
    Zero gates recover DEER exactly; the extension adds only F*d parameters.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        gate_init_std: float = 0.0,
        gate_limit: float = 1.0,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if gate_init_std < 0:
            raise ValueError("gate_init_std must be nonnegative")
        if not 0 < gate_limit <= 1:
            raise ValueError("gate_limit must lie in (0, 1]")
        self.gate_limit = float(gate_limit)
        self.raw_gates = nn.Parameter(
            torch.empty(initial_edges.size(0), d_embedding)
        )
        if gate_init_std == 0:
            nn.init.zeros_(self.raw_gates)
        else:
            nn.init.normal_(self.raw_gates, std=float(gate_init_std))

    @staticmethod
    def _gather_values(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        expanded = values.unsqueeze(0).expand(index.size(0), *values.shape)
        return expanded.gather(2, index.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _gather_embeddings(
        embeddings: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        expanded = embeddings.unsqueeze(0).expand(
            index.size(0), -1, -1, -1
        )
        gather_index = index.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, embeddings.size(-1)
        )
        return expanded.gather(2, gather_index).squeeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        i0 = (interval - 1).clamp_min(0)
        i1 = interval
        i2 = interval + 1
        i3 = (interval + 2).clamp_max(self.n_bins)

        x0 = self._gather_values(boundaries, i0)
        x1 = self._gather_values(boundaries, i1)
        x2 = self._gather_values(boundaries, i2)
        x3 = self._gather_values(boundaries, i3)
        p0 = self._gather_embeddings(self.knot_embeddings, i0)
        p1 = self._gather_embeddings(self.knot_embeddings, i1)
        p2 = self._gather_embeddings(self.knot_embeddings, i2)
        p3 = self._gather_embeddings(self.knot_embeddings, i3)

        h = (x2 - x1).clamp_min(1e-12)
        t = ((clipped - x1) / h).clamp(0.0, 1.0)
        m1 = (p2 - p0) / (x2 - x0).clamp_min(1e-12).unsqueeze(-1)
        m2 = (p3 - p1) / (x3 - x1).clamp_min(1e-12).unsqueeze(-1)
        t2 = t.square()
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        cubic = (
            h00.unsqueeze(-1) * p1
            + (h10 * h).unsqueeze(-1) * m1
            + h01.unsqueeze(-1) * p2
            + (h11 * h).unsqueeze(-1) * m2
        )
        linear = (1.0 - t).unsqueeze(-1) * p1 + t.unsqueeze(-1) * p2
        gate = self.gate_limit * torch.tanh(self.raw_gates).unsqueeze(0)
        return linear + gate * (cubic - linear)


class WENODEEREmbedding(DEEREmbedding):
    """Shock-aware adaptive-order interpolation on the learned DEER mesh.

    This module transfers the central idea of weighted essentially
    non-oscillatory (WENO) finite-volume schemes to numerical embeddings.
    Vanilla DEER is retained as a robust first-order candidate. A
    Catmull--Rom/Hermite candidate provides higher-order interpolation, while
    a dimensionless local smoothness sensor decides how much of that candidate
    is safe for every sample, field, and (optionally) latent channel::

        e(x) = e_linear(x) + omega(beta) * s * (e_cubic(x) - e_linear(x))
        omega(beta) = (1 + beta / tau) ** (-p).

    beta is the ratio between second-difference energy and first-difference
    energy over the four-knot stencil. It is zero for affine knot signals and
    O(1) around rough or discontinuous signals. Thus the representation uses
    high order in smooth regions and automatically falls back to DEER near
    embedding-space shocks. Boundary intervals are kept linear because a
    complete stencil is unavailable.

    The method introduces no trainable parameter beyond DEER, consumes no
    additional RNG, and preserves O(d) work per feature for fixed stencil
    width. Detaching the sensor prevents optimization from gaming the routing
    statistic while gradients still train both interpolation candidates.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        smoothness_scale: float = 0.1,
        smoothness_power: float = 2.0,
        cubic_strength: float = 1.0,
        channelwise_sensor: bool = False,
        detach_sensor: bool = True,
        linear_boundaries: bool = True,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if smoothness_scale <= 0:
            raise ValueError("smoothness_scale must be positive")
        if smoothness_power <= 0:
            raise ValueError("smoothness_power must be positive")
        if not 0.0 <= cubic_strength <= 1.0:
            raise ValueError("cubic_strength must lie in [0, 1]")
        self.smoothness_scale = float(smoothness_scale)
        self.smoothness_power = float(smoothness_power)
        self.cubic_strength = float(cubic_strength)
        self.channelwise_sensor = bool(channelwise_sensor)
        self.detach_sensor = bool(detach_sensor)
        self.linear_boundaries = bool(linear_boundaries)

    @staticmethod
    def _gather_embeddings(
        embeddings: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        expanded = embeddings.unsqueeze(0).expand(
            index.size(0), -1, -1, -1
        )
        gather_index = index.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, embeddings.size(-1)
        )
        return expanded.gather(2, gather_index).squeeze(2)

    def interpolation_terms(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return linear/cubic candidates, WENO weight, and active interval."""
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        i0 = (interval - 1).clamp_min(0)
        i1 = interval
        i2 = interval + 1
        i3 = (interval + 2).clamp_max(self.n_bins)

        knots = self.current_knot_embeddings()
        p0 = self._gather_embeddings(knots, i0)
        p1 = self._gather_embeddings(knots, i1)
        p2 = self._gather_embeddings(knots, i2)
        p3 = self._gather_embeddings(knots, i3)

        expanded_boundaries = boundaries.unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        left = expanded_boundaries.gather(
            2, interval.unsqueeze(-1)
        ).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        t = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )

        # Slopes live in empirical-rank coordinates. This avoids the extreme
        # conditioning induced by raw-value gaps in heavy-tailed features.
        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)
        t2 = t.square()
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        cubic = (
            h00.unsqueeze(-1) * p1
            + h10.unsqueeze(-1) * m1
            + h01.unsqueeze(-1) * p2
            + h11.unsqueeze(-1) * m2
        )
        linear = (1.0 - t).unsqueeze(-1) * p1 + t.unsqueeze(-1) * p2

        d0 = p1 - p0
        d1 = p2 - p1
        d2 = p3 - p2
        curvature_energy = (d1 - d0).square() + (d2 - d1).square()
        variation_energy = d0.square() + d1.square() + d2.square()
        if not self.channelwise_sensor:
            curvature_energy = curvature_energy.sum(dim=-1, keepdim=True)
            variation_energy = variation_energy.sum(dim=-1, keepdim=True)
        beta = curvature_energy / variation_energy.clamp_min(1e-12)
        if self.detach_sensor:
            beta = beta.detach()
        weight = (
            1.0 + beta / self.smoothness_scale
        ).pow(-self.smoothness_power)
        if self.linear_boundaries:
            complete_stencil = (
                (interval > 0) & (interval < self.n_bins - 1)
            ).unsqueeze(-1)
            weight = weight * complete_stencil.to(weight.dtype)
        return linear, cubic, weight, interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear, cubic, weight, _ = self.interpolation_terms(x)
        return linear + self.cubic_strength * weight * (cubic - linear)




class ResidualQuantizedDEEREmbedding(DEEREmbedding):
    """DEER augmented with coarse-to-fine residual semantic codes.

    In rank mode, deterministic residual K-means codebooks act on DEER's
    empirical-rank coordinate. Prefix codes address trainable token embeddings
    which form a variance-normalized residual adapter around continuous DEER.
    The adapter initializes to exactly zero, so optimization starts from a
    same-seed DEER and can opt into hierarchical discontinuous corrections.

    In latent mode, RQ-VAE-style hard residual quantization acts directly on
    the DEER latent. Codebooks use deterministic residual K-means on samples of
    the initial curve. A straight-through mixture keeps a configurable
    continuous remainder; codebook, commitment, and utilization terms are
    exposed as an auxiliary loss.

    Projection mode removes only the residual projection onto each selected
    code direction after field-wise centering. Unlike full codeword
    subtraction, it preserves complementary geometry while retaining a
    discrete coarse-to-fine semantic ID.

    All modes preserve [batch, fields, embedding_dim]. Extra initializations
    restore the RNG state after constructing DEER so downstream backbone
    parameters remain aligned with the DEER baseline.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        n_codebooks: int = 3,
        codebook_size: int = 4,
        quantization_space: str = "rank",
        sid_scale_init: float = 0.0,
        sid_scale_limit: float = 1.0,
        quantized_fraction: float = 0.5,
        adaptive_fraction: bool = False,
        task_gate: bool = False,
        task_gate_init: float = 0.0,
        task_gate_limit: float = 0.5,
        task_code_adapter: bool = False,
        task_code_scale: float = 1.0,
        quality_slope_init: float = 1.0,
        quality_center: float = 0.5,
        preserve_output_norm: bool = False,
        codebook_weight: float = 1.0,
        commitment_weight: float = 0.25,
        balance_weight: float = 0.0,
        balance_temperature: float = 0.1,
        kmeans_iterations: int = 8,
        projection_topk: int = 1,
        projection_temperature: float = 0.1,
        embedding_init_std: float = 1e-4,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        boundary_init_noise: Optional[float] = None,
        variance_preserving: bool = True,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if n_codebooks < 1:
            raise ValueError("n_codebooks must be positive")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least two")
        if quantization_space not in ("rank", "latent", "projection"):
            raise ValueError(
                "quantization_space must be rank, latent, or projection"
            )
        if not 0.0 <= quantized_fraction <= 1.0:
            raise ValueError("quantized_fraction must be in [0, 1]")
        if adaptive_fraction and quantization_space != "projection":
            raise ValueError("adaptive_fraction requires projection space")
        if adaptive_fraction and not 0.0 < quantized_fraction < 1.0:
            raise ValueError(
                "adaptive quantized_fraction must lie strictly inside (0, 1)"
            )
        if task_gate and quantization_space != "projection":
            raise ValueError("task_gate requires projection space")
        if task_gate and adaptive_fraction:
            raise ValueError("task_gate and adaptive_fraction are mutually exclusive")
        if task_gate_limit <= 0:
            raise ValueError("task_gate_limit must be positive")
        if abs(task_gate_init) >= task_gate_limit:
            raise ValueError("abs(task_gate_init) must be below task_gate_limit")
        if task_code_adapter and quantization_space != "projection":
            raise ValueError("task_code_adapter requires projection space")
        if task_code_scale <= 0:
            raise ValueError("task_code_scale must be positive")
        if quality_slope_init < 0 or not 0.0 <= quality_center <= 1.0:
            raise ValueError("invalid quality-gate initialization")
        if sid_scale_limit <= 0:
            raise ValueError("sid_scale_limit must be positive")
        if abs(sid_scale_init) >= sid_scale_limit:
            raise ValueError("abs(sid_scale_init) must be below sid_scale_limit")
        if codebook_weight < 0 or commitment_weight < 0 or balance_weight < 0:
            raise ValueError("auxiliary loss weights must be nonnegative")
        if balance_temperature <= 0:
            raise ValueError("balance_temperature must be positive")
        if kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if not 1 <= projection_topk <= codebook_size:
            raise ValueError("projection_topk must lie in [1, codebook_size]")
        if projection_temperature <= 0:
            raise ValueError("projection_temperature must be positive")

        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantization_space = quantization_space
        self.sid_scale_limit = float(sid_scale_limit)
        self.quantized_fraction = float(quantized_fraction)
        self.adaptive_fraction = bool(adaptive_fraction)
        self.task_gate = bool(task_gate)
        self.task_gate_init = float(task_gate_init)
        self.task_gate_limit = float(task_gate_limit)
        self.task_code_adapter = bool(task_code_adapter)
        self.task_code_scale = float(task_code_scale)
        self.quality_center = float(quality_center)
        self.preserve_output_norm = bool(preserve_output_norm)
        self._last_quantized_fraction: Optional[torch.Tensor] = None
        self.codebook_weight = float(codebook_weight)
        self.commitment_weight = float(commitment_weight)
        self.balance_weight = float(balance_weight)
        self.balance_temperature = float(balance_temperature)
        self.projection_topk = int(projection_topk)
        self.projection_temperature = float(projection_temperature)
        self.variance_preserving = bool(variance_preserving)
        self._last_auxiliary_loss: Optional[torch.Tensor] = None
        n_features = initial_edges.size(0)

        state_after_deer = torch.random.get_rng_state()
        if quantization_space == "rank":
            rank_centers = []
            first = (
                torch.arange(self.codebook_size, dtype=torch.float32) + 0.5
            ) / self.codebook_size
            rank_centers.append(first)
            for level in range(1, self.n_codebooks):
                scale = float(self.codebook_size ** level)
                rank_centers.append((first - 0.5) / scale)
            self.register_buffer(
                "rank_codebooks", torch.stack(rank_centers, dim=0)
            )
            self.code_embeddings = nn.Parameter(
                torch.empty(
                    n_features,
                    self.n_codebooks,
                    self.codebook_size,
                    d_embedding,
                )
            )
            nn.init.normal_(
                self.code_embeddings,
                std=float(embedding_init_std)
                / float(np.sqrt(self.n_codebooks)),
            )
            initial_raw_scale = np.arctanh(
                float(sid_scale_init) / float(sid_scale_limit)
            )
            self.raw_sid_scales = nn.Parameter(
                torch.full(
                    (n_features, d_embedding), float(initial_raw_scale)
                )
            )
            self.latent_codebooks = None
        else:
            initializer = (
                self._projection_residual_kmeans_initialization
                if quantization_space == "projection"
                else self._residual_kmeans_initialization
            )
            codebooks = initializer(
                self.knot_embeddings.detach(),
                self.n_codebooks,
                self.codebook_size,
                int(kmeans_iterations),
            )
            self.latent_codebooks = nn.Parameter(codebooks)
            self.code_embeddings = None
            self.raw_sid_scales = None
        if self.task_code_adapter:
            # Geometry chooses the hierarchical IDs; zero-initialized task
            # tokens learn collaborative residuals without perturbing DEER at
            # initialization.
            self.task_code_embeddings = nn.Parameter(
                torch.zeros(
                    n_features,
                    self.n_codebooks,
                    self.codebook_size,
                    d_embedding,
                )
            )
        else:
            self.register_parameter("task_code_embeddings", None)
        if self.adaptive_fraction:
            fraction_logit = np.log(
                self.quantized_fraction / (1.0 - self.quantized_fraction)
            )
            raw_slope = (
                np.log(np.expm1(float(quality_slope_init)))
                if quality_slope_init > 0
                else -20.0
            )
            self.fraction_logits = nn.Parameter(
                torch.full((n_features, 1), float(fraction_logit))
            )
            self.raw_quality_slopes = nn.Parameter(
                torch.full((n_features, 1), float(raw_slope))
            )
        else:
            self.register_parameter("fraction_logits", None)
            self.register_parameter("raw_quality_slopes", None)
        torch.random.set_rng_state(state_after_deer)
        if self.task_gate:
            initial_raw_gate = np.arctanh(
                self.task_gate_init / self.task_gate_limit
            )
            self.raw_task_scales = nn.Parameter(
                torch.full((n_features, 1), float(initial_raw_gate))
            )
        else:
            self.register_parameter("raw_task_scales", None)

    @staticmethod
    def _sample_knot_curve(
        knots: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        positions = torch.linspace(
            0.0, knots.size(1) - 1, n_samples,
            device=knots.device, dtype=knots.dtype,
        )
        left = positions.floor().long()
        right = positions.ceil().long()
        alpha = positions - left
        return (
            (1.0 - alpha).view(1, -1, 1) * knots[:, left]
            + alpha.view(1, -1, 1) * knots[:, right]
        )

    @classmethod
    def _residual_kmeans_initialization(
        cls,
        knots: torch.Tensor,
        n_codebooks: int,
        codebook_size: int,
        n_iterations: int,
    ) -> torch.Tensor:
        """Deterministic per-field residual K-means on the DEER curve."""
        n_samples = max(128, 32 * codebook_size)
        residual = cls._sample_knot_curve(knots, n_samples)
        stages = []
        seed_indices = (
            (
                torch.arange(codebook_size, device=knots.device)
                + 0.5
            )
            * n_samples
            / codebook_size
        ).long().clamp_max(n_samples - 1)
        for _ in range(n_codebooks):
            centers = residual[:, seed_indices].clone()
            for _ in range(n_iterations):
                distances = (
                    residual.unsqueeze(2) - centers.unsqueeze(1)
                ).square().sum(dim=-1)
                assignments = distances.argmin(dim=-1)
                updated = centers.clone()
                for code in range(codebook_size):
                    mask = assignments == code
                    counts = mask.sum(dim=1)
                    sums = (
                        residual * mask.unsqueeze(-1).to(residual.dtype)
                    ).sum(dim=1)
                    nonempty = counts > 0
                    updated[nonempty, code] = (
                        sums[nonempty]
                        / counts[nonempty].unsqueeze(-1).to(residual.dtype)
                    )
                centers = updated
            stages.append(centers)
            distances = (
                residual.unsqueeze(2) - centers.unsqueeze(1)
            ).square().sum(dim=-1)
            assignments = distances.argmin(dim=-1)
            selected = centers.gather(
                1,
                assignments.unsqueeze(-1).expand(-1, -1, centers.size(-1)),
            )
            residual = residual - selected
        return torch.stack(stages, dim=1)

    @classmethod
    def _projection_residual_kmeans_initialization(
        cls,
        knots: torch.Tensor,
        n_codebooks: int,
        codebook_size: int,
        n_iterations: int,
    ) -> torch.Tensor:
        """Initialize directions by K-means and projection residual updates."""
        n_samples = max(128, 32 * codebook_size)
        samples = cls._sample_knot_curve(knots, n_samples)
        residual = samples - samples.mean(dim=1, keepdim=True)
        stages = []
        seed_indices = (
            (torch.arange(codebook_size, device=knots.device) + 0.5)
            * n_samples
            / codebook_size
        ).long().clamp_max(n_samples - 1)
        for _ in range(n_codebooks):
            centers = residual[:, seed_indices].clone()
            for _ in range(n_iterations):
                distances = (
                    residual.unsqueeze(2) - centers.unsqueeze(1)
                ).square().sum(dim=-1)
                assignments = distances.argmin(dim=-1)
                updated = centers.clone()
                for code in range(codebook_size):
                    mask = assignments == code
                    counts = mask.sum(dim=1)
                    sums = (
                        residual * mask.unsqueeze(-1).to(residual.dtype)
                    ).sum(dim=1)
                    nonempty = counts > 0
                    updated[nonempty, code] = (
                        sums[nonempty]
                        / counts[nonempty].unsqueeze(-1).to(residual.dtype)
                    )
                centers = updated
            stages.append(centers)
            distances = (
                residual.unsqueeze(2) - centers.unsqueeze(1)
            ).square().sum(dim=-1)
            assignments = distances.argmin(dim=-1)
            selected = centers.gather(
                1,
                assignments.unsqueeze(-1).expand(
                    -1, -1, centers.size(-1)
                ),
            )
            coefficient = (residual * selected).sum(
                dim=-1, keepdim=True
            ) / selected.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            residual = residual - coefficient * selected
        return torch.stack(stages, dim=1)

    def rank_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        return (interval.to(x.dtype) + alpha) / float(self.n_bins)

    @staticmethod
    def _gather_codes(
        table: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        expanded = table.unsqueeze(0).expand(indices.size(0), -1, -1, -1)
        gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, table.size(-1)
        )
        return expanded.gather(2, gather_index).squeeze(2)

    def _balance_penalty(self, probabilities: torch.Tensor) -> torch.Tensor:
        usage = probabilities.mean(dim=0)
        return (
            usage
            * (usage.clamp_min(1e-12).log() + np.log(self.codebook_size))
        ).sum(dim=-1).mean()

    def _rank_forward(
        self, x: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        residual = self.rank_coordinates(x)
        sid = continuous.new_zeros(continuous.shape)
        balance = continuous.new_zeros(())
        for level in range(self.n_codebooks):
            centers = self.rank_codebooks[level].to(residual.dtype)
            distances = (residual.unsqueeze(-1) - centers).square()
            indices = distances.argmin(dim=-1)
            sid = sid + self._gather_codes(
                self.code_embeddings[:, level], indices
            )
            residual = residual - centers[indices]
            if self.training and self.balance_weight > 0:
                probabilities = torch.softmax(
                    -distances / self.balance_temperature, dim=-1
                )
                balance = balance + self._balance_penalty(probabilities)
        scales = self.sid_scale_limit * torch.tanh(
            self.raw_sid_scales
        ).unsqueeze(0)
        if self.variance_preserving:
            output = (continuous + scales * sid) / torch.sqrt(
                1.0 + scales.square()
            )
        else:
            output = continuous + scales * sid
        if self.training and self.balance_weight > 0:
            self._last_auxiliary_loss = (
                self.balance_weight * balance / self.n_codebooks
            )
        else:
            self._last_auxiliary_loss = output.new_zeros(())
        return output

    def _latent_forward(self, continuous: torch.Tensor) -> torch.Tensor:
        residual = continuous
        quantized = continuous.new_zeros(continuous.shape)
        balance = continuous.new_zeros(())
        for level in range(self.n_codebooks):
            table = self.latent_codebooks[:, level]
            distances = (
                residual.unsqueeze(2) - table.unsqueeze(0)
            ).square().sum(dim=-1)
            indices = distances.argmin(dim=-1)
            selected = self._gather_codes(table, indices)
            quantized = quantized + selected
            residual = residual - selected
            if self.training and self.balance_weight > 0:
                probabilities = torch.softmax(
                    -distances / self.balance_temperature, dim=-1
                )
                balance = balance + self._balance_penalty(probabilities)

        output = continuous + self.quantized_fraction * (
            quantized - continuous
        ).detach()
        if self.training:
            auxiliary = output.new_zeros(())
            if self.codebook_weight > 0:
                auxiliary = auxiliary + self.codebook_weight * F.mse_loss(
                    quantized, continuous.detach()
                )
            if self.commitment_weight > 0:
                auxiliary = auxiliary + self.commitment_weight * F.mse_loss(
                    continuous, quantized.detach()
                )
            if self.balance_weight > 0:
                auxiliary = auxiliary + self.balance_weight * balance / self.n_codebooks
            self._last_auxiliary_loss = auxiliary
        else:
            self._last_auxiliary_loss = output.new_zeros(())
        return output

    def _projection_components(
        self, continuous: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        anchor = self.current_knot_embeddings().mean(dim=1).unsqueeze(0).detach()
        # Codebook fitting follows VQ stop-gradient semantics: only the
        # commitment term, when enabled, is allowed to update the DEER encoder.
        residual = continuous.detach() - anchor
        projected = continuous.new_zeros(continuous.shape)
        balance = continuous.new_zeros(())
        codes = []
        for level in range(self.n_codebooks):
            table = self.latent_codebooks[:, level]
            normalized_residual = F.normalize(residual, dim=-1, eps=1e-12)
            normalized_table = F.normalize(table, dim=-1, eps=1e-12)
            similarities = torch.einsum(
                "bfd,fkd->bfk", normalized_residual, normalized_table
            )
            top_values, top_indices = similarities.topk(
                self.projection_topk, dim=-1
            )
            codes.append(top_indices[..., 0])
            if self.projection_topk == 1:
                selected = self._gather_codes(table, top_indices[..., 0])
            else:
                expanded = table.unsqueeze(0).expand(
                    continuous.size(0), -1, -1, -1
                )
                gather_index = top_indices.unsqueeze(-1).expand(
                    -1, -1, -1, table.size(-1)
                )
                candidates = expanded.gather(2, gather_index)
                weights = torch.softmax(
                    top_values / self.projection_temperature, dim=-1
                )
                selected = (
                    candidates * weights.unsqueeze(-1)
                ).sum(dim=2)
            coefficient = (residual * selected).sum(
                dim=-1, keepdim=True
            ) / selected.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            projection = coefficient * selected
            projected = projected + projection
            residual = residual - projection
            if self.training and self.balance_weight > 0:
                probabilities = torch.softmax(
                    similarities / self.projection_temperature, dim=-1
                )
                balance = balance + self._balance_penalty(probabilities)
        return anchor + projected, codes, balance

    def _projection_forward(self, continuous: torch.Tensor) -> torch.Tensor:
        reconstructed, codes, balance = self._projection_components(continuous)
        if self.task_gate:
            fraction = self.task_gate_limit * torch.tanh(
                self.raw_task_scales
            ).unsqueeze(0)
        elif self.adaptive_fraction:
            anchor = self.current_knot_embeddings().mean(dim=1).unsqueeze(0)
            total_energy = (
                (continuous - anchor).square().sum(dim=-1).clamp_min(1e-12)
            )
            residual_energy = (
                (continuous - reconstructed).square().sum(dim=-1)
            )
            quality = (1.0 - residual_energy / total_energy).clamp(0.0, 1.0)
            slopes = F.softplus(self.raw_quality_slopes).unsqueeze(0)
            fraction = torch.sigmoid(
                self.fraction_logits.unsqueeze(0)
                + slopes * (quality.detach().unsqueeze(-1) - self.quality_center)
            )
        else:
            fraction = continuous.new_tensor(self.quantized_fraction)
        self._last_quantized_fraction = fraction.detach()
        output = continuous + fraction * (
            reconstructed - continuous
        ).detach()
        if self.task_code_adapter:
            task_residual = continuous.new_zeros(continuous.shape)
            for level, indices in enumerate(codes):
                task_residual = task_residual + self._gather_codes(
                    self.task_code_embeddings[:, level], indices
                )
            output = output + self.task_code_scale * task_residual / float(
                np.sqrt(self.n_codebooks)
            )
        if self.preserve_output_norm:
            target_norm = continuous.norm(dim=-1, keepdim=True)
            actual_norm = output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            output = output * (
                target_norm / actual_norm
            ).detach()
        if self.training:
            auxiliary = output.new_zeros(())
            if self.codebook_weight > 0:
                auxiliary = auxiliary + self.codebook_weight * F.mse_loss(
                    reconstructed, continuous.detach()
                )
            if self.commitment_weight > 0:
                auxiliary = auxiliary + self.commitment_weight * F.mse_loss(
                    continuous, reconstructed.detach()
                )
            if self.balance_weight > 0:
                auxiliary = auxiliary + self.balance_weight * balance / self.n_codebooks
            self._last_auxiliary_loss = auxiliary
        else:
            self._last_auxiliary_loss = output.new_zeros(())
        return output

    def get_auxiliary_loss(self) -> torch.Tensor:
        if self._last_auxiliary_loss is None:
            return self.raw_widths.new_zeros(())
        return self._last_auxiliary_loss

    def semantic_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Return [batch, fields, levels] hard codes for diagnostics."""
        codes = []
        if self.quantization_space == "rank":
            residual = self.rank_coordinates(x)
            for level in range(self.n_codebooks):
                centers = self.rank_codebooks[level].to(residual.dtype)
                indices = (
                    residual.unsqueeze(-1) - centers
                ).square().argmin(dim=-1)
                codes.append(indices)
                residual = residual - centers[indices]
        elif self.quantization_space == "latent":
            residual = DEEREmbedding.forward(self, x)
            for level in range(self.n_codebooks):
                table = self.latent_codebooks[:, level]
                indices = (
                    residual.unsqueeze(2) - table.unsqueeze(0)
                ).square().sum(dim=-1).argmin(dim=-1)
                codes.append(indices)
                residual = residual - self._gather_codes(table, indices)
        else:
            continuous = DEEREmbedding.forward(self, x)
            _, codes, _ = self._projection_components(continuous)
        return torch.stack(codes, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        continuous = DEEREmbedding.forward(self, x)
        if self.quantization_space == "rank":
            return self._rank_forward(x, continuous)
        if self.quantization_space == "latent":
            return self._latent_forward(continuous)
        return self._projection_forward(continuous)

class MultiResolutionDEEREmbedding(DEEREmbedding):
    """Parameter-efficient DEER with coarse controls and low-rank detail.

    A fixed linear prolongation matrix maps ``n_controls`` coarse embeddings
    to the DEER knot grid. A rank-r factorization adds field-specific local
    detail. The learned DEER boundaries are unchanged. At inference the
    generated knots can be fused, leaving the same two-knot lookup and O(d)
    sample complexity as DEER.
    """

    def __init__(
        self,
        initial_edges: torch.Tensor,
        d_embedding: int,
        min_width: float = 1e-4,
        width_temperature: float = 1.0,
        embedding_init_std: float = 1e-4,
        boundary_init_noise: Optional[float] = None,
        n_controls: int = 8,
        residual_rank: int = 1,
        residual_direction_init_std: float = 1e-2,
    ):
        super().__init__(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=min_width,
            width_temperature=width_temperature,
            embedding_init_std=embedding_init_std,
            boundary_init_noise=boundary_init_noise,
        )
        if not 2 <= n_controls <= self.n_bins + 1:
            raise ValueError("n_controls must be in [2, n_bins + 1]")
        if residual_rank < 0:
            raise ValueError("residual_rank must be nonnegative")
        if residual_direction_init_std <= 0:
            raise ValueError("residual_direction_init_std must be positive")
        n_features = initial_edges.size(0)
        del self.knot_embeddings
        self.n_controls = int(n_controls)
        self.residual_rank = int(residual_rank)

        positions = torch.linspace(0.0, self.n_controls - 1, self.n_bins + 1)
        left = positions.floor().long()
        right = positions.ceil().long()
        fraction = positions - left
        basis = torch.zeros(self.n_bins + 1, self.n_controls)
        basis.scatter_add_(1, left.unsqueeze(1), (1.0 - fraction).unsqueeze(1))
        basis.scatter_add_(1, right.unsqueeze(1), fraction.unsqueeze(1))
        self.register_buffer("control_basis", basis)

        self.control_embeddings = nn.Parameter(
            torch.empty(n_features, self.n_controls, d_embedding)
        )
        nn.init.normal_(self.control_embeddings, std=float(embedding_init_std))
        self.residual_coefficients = nn.Parameter(
            torch.zeros(n_features, self.n_bins + 1, self.residual_rank)
        )
        self.residual_directions = nn.Parameter(
            torch.empty(n_features, self.residual_rank, d_embedding)
        )
        if self.residual_rank > 0:
            nn.init.normal_(
                self.residual_directions,
                std=float(residual_direction_init_std),
            )

    def current_knot_embeddings(self) -> torch.Tensor:
        knots = torch.einsum(
            "kc,fcd->fkd", self.control_basis, self.control_embeddings
        )
        if self.residual_rank > 0:
            knots = knots + torch.einsum(
                "fkr,frd->fkd",
                self.residual_coefficients,
                self.residual_directions,
            )
        return knots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        knots = self.current_knot_embeddings()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        expanded_knots = knots.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, knots.size(-1)
        )
        left_embedding = expanded_knots.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_knots.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (
            (1.0 - alpha).unsqueeze(-1) * left_embedding
            + alpha.unsqueeze(-1) * right_embedding
        )


class KerasDEEREmbedding(nn.Module):
    """Corrected PyTorch port of the user-provided Keras DEER layer.

    Unlike DEEREmbedding, every field shares one fixed value range and the
    interval logits start near zero, so the initial partition is almost
    equal-width. The original Keras snippet constructed n_bins + 2 boundary
    values for only n_bins + 1 embeddings; this port uses the mathematically
    consistent n_bins + 1 knots and clamps out-of-range inputs before
    interpolation.

    max_value should cover raw, unnormalized inputs. The factory therefore
    defaults it to 1e6 when input_normalized=false unless explicitly set.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_bins: int = 16,
        min_value: float = 0.0,
        max_value: float = 1.0,
        boundary_init_limit: float = 0.05,
        embedding_init_std: float = 0.05,
    ):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value")
        if boundary_init_limit < 0:
            raise ValueError("boundary_init_limit must be nonnegative")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        self.n_bins = int(n_bins)
        self.register_buffer(
            "lower", torch.full((n_features,), float(min_value))
        )
        self.register_buffer(
            "upper", torch.full((n_features,), float(max_value))
        )
        self.boundary_weights = nn.Parameter(torch.empty(n_features, self.n_bins))
        self.boundary_embeddings = nn.Parameter(
            torch.empty(n_features, self.n_bins + 1, d_embedding)
        )
        nn.init.uniform_(
            self.boundary_weights,
            -float(boundary_init_limit),
            float(boundary_init_limit),
        )
        nn.init.normal_(
            self.boundary_embeddings, std=float(embedding_init_std)
        )

    def current_boundaries(self) -> torch.Tensor:
        fractions = torch.softmax(self.boundary_weights, dim=-1)
        cumulative = torch.cumsum(fractions, dim=-1)
        internal_and_upper = self.lower.unsqueeze(-1) + cumulative * (
            self.upper - self.lower
        ).unsqueeze(-1)
        return torch.cat([self.lower.unsqueeze(-1), internal_and_upper], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.current_boundaries()
        clipped = torch.maximum(torch.minimum(x, self.upper), self.lower)
        interval = (
            clipped.unsqueeze(-1) >= boundaries[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded_boundaries = boundaries.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded_boundaries.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded_boundaries.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        alpha = ((clipped - left) / (right - left).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        expanded_embeddings = self.boundary_embeddings.unsqueeze(0).expand(
            x.size(0), -1, -1, -1
        )
        gather_index = interval.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.boundary_embeddings.size(-1)
        )
        left_embedding = expanded_embeddings.gather(2, gather_index).squeeze(2)
        right_embedding = expanded_embeddings.gather(
            2, gather_index + 1
        ).squeeze(2)
        return (1.0 - alpha).unsqueeze(-1) * left_embedding + alpha.unsqueeze(
            -1
        ) * right_embedding


def _resample_bin_edges(bins: List[torch.Tensor], n_bins: int) -> torch.Tensor:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    resampled = []
    for edges in bins:
        values = F.interpolate(
            edges.view(1, 1, -1),
            size=n_bins + 1,
            mode="linear",
            align_corners=True,
        ).view(-1)
        if not torch.all(values[1:] > values[:-1]):
            lower, upper = edges[0], edges[-1]
            values = torch.linspace(lower.item(), upper.item(), n_bins + 1)
        resampled.append(values)
    return torch.stack(resampled)


class DistributionAwareEmbedding(nn.Module):
    """DAE reproduction using expected batch order statistics.

    During training, equally spaced order statistics from each batch update a
    running expectation of feature quantiles. Distances to those quantiles are
    transformed by a field-specific MLP into meta-embedding weights. Evaluation
    freezes the accumulated statistics. The original authors did not release
    code, so update behavior is explicit and checkpointed here.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_quantiles: int = 16,
        hidden_dim: int = 32,
        temperature: float = 1.0,
        momentum: Optional[float] = None,
        normalize_distances: bool = False,
    ):
        super().__init__()
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be at least two")
        if hidden_dim < 1 or temperature <= 0:
            raise ValueError("Invalid DAE network settings")
        if momentum is not None and not 0 < momentum <= 1:
            raise ValueError("momentum must be in (0, 1]")
        self.n_quantiles = int(n_quantiles)
        self.temperature = float(temperature)
        self.momentum = None if momentum is None else float(momentum)
        self.normalize_distances = bool(normalize_distances)
        self.register_buffer(
            "quantile_levels", torch.linspace(0.0, 1.0, self.n_quantiles)
        )
        self.register_buffer(
            "running_quantiles", torch.zeros(n_features, self.n_quantiles)
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.first_weight = nn.Parameter(
            torch.empty(n_features, self.n_quantiles, hidden_dim)
        )
        self.first_bias = nn.Parameter(torch.zeros(n_features, hidden_dim))
        self.second_weight = nn.Parameter(
            torch.empty(n_features, hidden_dim, self.n_quantiles)
        )
        self.second_bias = nn.Parameter(torch.zeros(n_features, self.n_quantiles))
        self.meta_embeddings = nn.Parameter(
            torch.empty(n_features, self.n_quantiles, d_embedding)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for field_weight in self.first_weight:
            nn.init.xavier_uniform_(field_weight)
        for field_weight in self.second_weight:
            nn.init.xavier_uniform_(field_weight)
        nn.init.zeros_(self.first_bias)
        nn.init.zeros_(self.second_bias)
        nn.init.normal_(self.meta_embeddings, std=1e-4)

    @torch.no_grad()
    def _batch_order_statistics(self, x: torch.Tensor) -> torch.Tensor:
        quantiles = torch.quantile(
            x.detach().float(), self.quantile_levels.to(x.device), dim=0
        )
        return quantiles.transpose(0, 1)

    @torch.no_grad()
    def _update_quantiles(self, batch_quantiles: torch.Tensor) -> None:
        if self.num_updates.item() == 0:
            self.running_quantiles.copy_(batch_quantiles)
        elif self.momentum is None:
            count = self.num_updates.to(dtype=batch_quantiles.dtype) + 1.0
            self.running_quantiles.add_(
                (batch_quantiles - self.running_quantiles) / count
            )
        else:
            self.running_quantiles.lerp_(batch_quantiles, self.momentum)
        self.num_updates.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_quantiles(self._batch_order_statistics(x))
            quantiles = self.running_quantiles
        elif self.num_updates.item() > 0:
            quantiles = self.running_quantiles
        else:
            quantiles = self._batch_order_statistics(x)
        distances = torch.abs(x.unsqueeze(-1) - quantiles.unsqueeze(0))
        if self.normalize_distances:
            scale = (quantiles[:, -1] - quantiles[:, 0]).clamp_min(1e-6)
            distances = distances / scale.view(1, -1, 1)
        hidden = F.relu(
            torch.einsum("bfq,fqh->bfh", distances, self.first_weight)
            + self.first_bias
        )
        logits = (
            torch.einsum("bfh,fhq->bfq", hidden, self.second_weight)
            + self.second_bias
        )
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return torch.einsum("bfq,fqd->bfd", weights, self.meta_embeddings)


class DAESEmbedding(nn.Module):
    """Paper-aligned DAES reproduction for streaming numerical features.

    The module implements quantile-space thermometer encoding and the gate/tran
    modulation equations from Liu et al. (arXiv:2602.03223). FuxiCTR consumes
    streams in batches, so the reservoir is updated with an exact batch-boundary
    merge: conditional on the previous uniform reservoir, a hypergeometric draw
    selects how many current-batch rows enter the new uniform reservoir. This
    has the same reservoir distribution at batch boundaries as item-wise
    reservoir sampling while avoiding Python work per sample.

    The paper selects three categorical fields but does not specify how their
    embeddings are combined into the single context vector used by its
    equations. This reproduction mean-pools the selected field embeddings.
    """

    requires_context_embeddings = True

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        context_features: Sequence[str],
        n_meta_embeddings: int = 10,
        reservoir_size: int = 100000,
        beta: float = 0.5,
        modulation: str = "gate",
        quantile_refresh_interval: int = 1,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if n_meta_embeddings < 2:
            raise ValueError("n_meta_embeddings must be at least two")
        if reservoir_size < n_meta_embeddings + 1:
            raise ValueError(
                "reservoir_size must exceed the number of meta-embeddings"
            )
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if modulation not in {"gate", "tran"}:
            raise ValueError("modulation must be 'gate' or 'tran'")
        if quantile_refresh_interval < 1:
            raise ValueError("quantile_refresh_interval must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")

        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)
        self.context_features = tuple(str(name) for name in context_features)
        self.n_meta_embeddings = int(n_meta_embeddings)
        self.reservoir_size = int(reservoir_size)
        self.beta = float(beta)
        self.modulation = modulation
        self.quantile_refresh_interval = int(quantile_refresh_interval)

        self.meta_embeddings = nn.Parameter(
            torch.empty(
                self.n_features,
                self.n_meta_embeddings,
                self.d_embedding,
            )
        )
        self.default_context = nn.Parameter(torch.empty(self.d_embedding))
        if self.modulation == "gate":
            self.modulation_weight = nn.Parameter(
                torch.empty(
                    self.n_features,
                    self.n_meta_embeddings,
                    self.d_embedding,
                )
            )
        else:
            self.modulation_weight = nn.Parameter(
                torch.empty(
                    self.n_features,
                    self.n_meta_embeddings,
                    self.n_meta_embeddings,
                    self.d_embedding,
                )
            )

        self.register_buffer(
            "reservoir",
            torch.zeros(self.reservoir_size, self.n_features),
        )
        self.register_buffer("num_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "cached_quantiles",
            torch.zeros(self.n_features, self.n_meta_embeddings + 1),
        )
        self.register_buffer(
            "quantiles_initialized", torch.zeros((), dtype=torch.bool)
        )
        self.register_buffer(
            "batches_since_refresh", torch.zeros((), dtype=torch.long)
        )
        self.reset_parameters(float(embedding_init_std))

    def reset_parameters(self, embedding_init_std: float = 1e-4) -> None:
        nn.init.normal_(self.meta_embeddings, std=embedding_init_std)
        nn.init.normal_(self.default_context, std=embedding_init_std)
        if self.modulation == "gate":
            for weight in self.modulation_weight:
                nn.init.xavier_uniform_(weight)
        else:
            for feature_index in range(self.n_features):
                nn.init.xavier_uniform_(
                    self.modulation_weight[feature_index].reshape(
                        self.n_meta_embeddings * self.n_meta_embeddings,
                        self.d_embedding,
                    )
                )

    @torch.no_grad()
    def _update_reservoir(self, x: torch.Tensor) -> bool:
        """Merge one batch into an exactly uniform fixed-size reservoir."""

        batch = x.detach().float()
        batch_size = int(batch.size(0))
        seen = int(self.num_seen.item())
        position = 0
        changed = False

        if seen < self.reservoir_size:
            take = min(self.reservoir_size - seen, batch_size)
            if take:
                self.reservoir[seen : seen + take].copy_(batch[:take])
                seen += take
                position += take
                changed = True

        remaining = batch_size - position
        if remaining:
            selected = int(
                np.random.hypergeometric(
                    ngood=remaining,
                    nbad=seen,
                    nsample=self.reservoir_size,
                )
            )
            if selected:
                new_rows = np.random.choice(
                    remaining, size=selected, replace=False
                )
                old_slots = np.random.choice(
                    self.reservoir_size, size=selected, replace=False
                )
                new_rows = torch.as_tensor(
                    new_rows + position,
                    dtype=torch.long,
                    device=batch.device,
                )
                old_slots = torch.as_tensor(
                    old_slots,
                    dtype=torch.long,
                    device=batch.device,
                )
                self.reservoir.index_copy_(
                    0, old_slots, batch.index_select(0, new_rows)
                )
                changed = True
            seen += remaining

        self.num_seen.fill_(seen)
        return changed

    @torch.no_grad()
    def _batch_quantiles(self, samples: torch.Tensor) -> torch.Tensor:
        levels = torch.linspace(
            0.0,
            1.0,
            self.n_meta_embeddings + 1,
            device=samples.device,
        )
        return torch.quantile(
            samples.detach().float(), levels, dim=0
        ).transpose(0, 1)

    @torch.no_grad()
    def _refresh_quantiles(self) -> None:
        filled = min(int(self.num_seen.item()), self.reservoir_size)
        if filled < 1:
            return
        self.cached_quantiles.copy_(
            self._batch_quantiles(self.reservoir[:filled])
        )
        self.quantiles_initialized.fill_(True)
        self.batches_since_refresh.zero_()

    def _quantile_coordinates(
        self, x: torch.Tensor, quantiles: torch.Tensor
    ) -> torch.Tensor:
        """Equation (8): cumulative thermometer encoding in quantile space."""

        lower = quantiles[:, 0]
        upper = quantiles[:, -1]
        clipped = torch.maximum(torch.minimum(x, upper), lower)
        interval = (
            clipped.unsqueeze(-1) >= quantiles[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded = quantiles.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        width = right - left
        fraction = torch.where(
            width.abs() > 1e-12,
            (clipped - left) / width.clamp_min(1e-12),
            torch.zeros_like(clipped),
        ).clamp(0.0, 1.0)
        positions = torch.arange(
            self.n_meta_embeddings, device=x.device
        ).view(1, 1, -1)
        raw = (positions < interval.unsqueeze(-1)).to(x.dtype)
        raw = raw + (
            positions == interval.unsqueeze(-1)
        ).to(x.dtype) * fraction.unsqueeze(-1)
        return raw

    def _context_vector(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if not context_embeddings:
            return self.default_context.unsqueeze(0).expand(x.size(0), -1)
        vectors = []
        for feature in self.context_features:
            embedding = context_embeddings.get(feature)
            if embedding is None:
                continue
            if embedding.ndim != 2 or embedding.size(0) != x.size(0):
                raise ValueError(
                    f"Invalid context embedding shape for {feature!r}: "
                    f"{tuple(embedding.shape)}"
                )
            if embedding.size(1) != self.d_embedding:
                raise ValueError(
                    f"Context embedding {feature!r} has dim "
                    f"{embedding.size(1)}, expected {self.d_embedding}"
                )
            vectors.append(embedding)
        if not vectors:
            return self.default_context.unsqueeze(0).expand(x.size(0), -1)
        return torch.stack(vectors, dim=1).mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.training:
            changed = self._update_reservoir(x)
            self.batches_since_refresh.add_(1)
            if (
                changed
                and (
                    not bool(self.quantiles_initialized.item())
                    or self.batches_since_refresh.item()
                    >= self.quantile_refresh_interval
                )
            ):
                self._refresh_quantiles()

        if bool(self.quantiles_initialized.item()):
            quantiles = self.cached_quantiles
        else:
            quantiles = self._batch_quantiles(x)

        raw = self._quantile_coordinates(x, quantiles)
        context = self._context_vector(x, context_embeddings)
        if self.modulation == "gate":
            modulation = torch.sigmoid(
                torch.einsum(
                    "bd,fmd->bfm", context, self.modulation_weight
                )
            )
            transformed = raw * modulation
        else:
            transformed = torch.sigmoid(
                torch.einsum(
                    "bd,fijd,bfj->bfi",
                    context,
                    self.modulation_weight,
                    raw,
                )
            )
        weights = self.beta * transformed + (1.0 - self.beta) * raw
        return torch.einsum(
            "bfm,fmd->bfd", weights, self.meta_embeddings
        )


def _load_bins(
    numeric_features: Iterable[str], feature_map, device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    bins = []
    for feature in numeric_features:
        edges = feature_map.features[feature].get("numeric_bin_edges")
        if edges is None:
            raise ValueError(
                f"Feature {feature!r} has no numeric_bin_edges. "
                "Prepare train-only boundaries with the scalarlens.data converters."
            )
        tensor = torch.as_tensor(edges, dtype=torch.float32, device=device)
        if tensor.ndim != 1 or len(tensor) < 2 or not torch.all(tensor[1:] > tensor[:-1]):
            raise ValueError(f"Invalid numeric_bin_edges for feature {feature!r}: {edges}")
        bins.append(tensor)
    return bins


def build_numerical_embedding(
    method: str,
    numeric_features: List[str],
    feature_map,
    d_embedding: int,
    params: Optional[Dict] = None,
) -> nn.Module:
    """Construct one numerical embedding module with a common output shape."""

    params = dict(params or {})
    method = method.lower()
    if method not in SUPPORTED_NUM_EMBEDDINGS:
        raise ValueError(
            f"Unknown numerical_embedding={method!r}; "
            f"choose one of {SUPPORTED_NUM_EMBEDDINGS}"
        )
    n_features = len(numeric_features)
    if method == "deer_story_control":
        import importlib.util
        module_name = "_fuxictr_numemb_story_controls"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                module_name, Path(__file__).with_name("story_controls.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        module = sys.modules[module_name]
        core_params = dict(params)
        core_params["execution_mode"] = "reference"
        core = build_numerical_embedding(
            "deer_scattering_fast", numeric_features, feature_map,
            d_embedding, core_params)
        kind = params.get("control_kind", "norm")
        if kind in ("no_norm", "frozen_message"):
            return module.FastFunctionalControl(core, kind)
        target = sum(p.numel() for name, p in core.named_parameters()
                     if not name.startswith("local_encoder."))
        with torch.random.fork_rng(devices=[]):
            return module.LocalContextControl(
                core.local_encoder, numeric_features, d_embedding,
                core.context_dims, kind, target, core.norm_eps,
                context_rank=int(params.get("context_rank", 16)))
    if method in ("deer_scattering", "deer_scattering_fast", "deer_interaction_residual"):
        context_features = params.get("context_features")
        if context_features is None:
            context_features = [
                name for name, spec in feature_map.features.items()
                if spec.get("type") == "categorical"
            ]
        elif isinstance(context_features, str):
            context_features = [name.strip() for name in context_features.split(",") if name.strip()]
        else:
            context_features = list(context_features)
        if len(set(context_features)) != len(context_features):
            raise ValueError("deer_scattering context_features must be unique")
        invalid = [
            name for name in context_features
            if name not in feature_map.features
            or feature_map.features[name].get("type") != "categorical"
        ]
        if invalid:
            raise ValueError(f"deer_scattering context_features must be categorical: {invalid}")
        context_dims = {
            name: int(feature_map.features[name].get("embedding_dim", d_embedding))
            for name in context_features
        }
        local_params = dict(params)
        if method != "deer_scattering_fast":
            local_params.setdefault("deer_execution_mode", "reference")
        local_encoder = build_numerical_embedding(
            "deer", numeric_features, feature_map, d_embedding, local_params
        )
        # New parameters must not shift the existing categorical/backbone RNG stream.
        # Parameters are constructed on CPU, as in all existing registry builders.
        with torch.random.fork_rng(devices=[]):
            if method in ("deer_scattering_fast", "deer_interaction_residual"):
                embedding_cls = FastCoupledEmbedding
                extra = {}
                if method == "deer_interaction_residual":
                    embedding_cls = InteractionResidualEmbedding
                    extra = dict(counterfactual=params.get("counterfactual", True),
                                 residual_init=float(params.get("residual_init", 0.)))
                return embedding_cls(
                    local_encoder=local_encoder, numeric_features=numeric_features,
                    d_embedding=d_embedding, context_dims=context_dims,
                    modes_per_field=params.get("modes_per_field", 8),
                    coupling_rank=params.get("coupling_rank", 16),
                    n_steps=params.get("n_steps", 3),
                    mode_hidden_dim=params.get("mode_hidden_dim", 32),
                    readout_hidden_dim=params.get("readout_hidden_dim", 32),
                    norm_eps=float(params.get("norm_eps", 1e-8)),
                    coupling_mode=str(params.get("coupling_mode", "joint")),
                    validate_inputs=bool(params.get("validate_inputs", True)),
                    execution_mode=str(params.get("execution_mode",
                        "auto" if method == "deer_scattering_fast" else "reference")),
                    use_numeric_values=bool(params.get("use_numeric_values", True)),
                    use_categorical_context=bool(params.get("use_categorical_context", True)),
                    **extra,
                )
            return CoupledScatteringEmbedding(
                local_encoder=local_encoder,
                numeric_features=numeric_features,
                d_embedding=d_embedding,
                context_dims=context_dims,
                modes_per_field=params.get("modes_per_field", 4),
                n_frequencies=params.get("n_frequencies", 4),
                detuning_hidden_dim=params.get("detuning_hidden_dim", 64),
                readout_hidden_dim=params.get("readout_hidden_dim", 64),
                damping_floor=float(params.get("damping_floor", 0.1)),
                damping_init=float(params.get("damping_init", 0.5)),
                coupling_init_scale=float(params.get("coupling_init_scale", 1.0)),
                norm_eps=float(params.get("norm_eps", 1e-8)),
                coupling_mode=str(params.get("coupling_mode", "joint")),
                solve_chunk_size=params.get("solve_chunk_size", 64),
            )
    if method == "field":
        return FieldEmbedding(n_features, d_embedding)
    if method == "linear":
        return rtdl_num.LinearEmbeddings(n_features, d_embedding)
    if method == "linear_relu":
        return rtdl_num.LinearReLUEmbeddings(n_features, d_embedding)
    if method == "autodis":
        return AutoDisEmbedding(
            n_features=n_features,
            d_embedding=d_embedding,
            n_meta_embeddings=int(params.get("n_meta_embeddings", 16)),
            temperature=float(params.get("temperature", 0.1)),
            residual_weight=float(params.get("residual_weight", 1.0)),
            leaky_relu_slope=float(params.get("leaky_relu_slope", 0.01)),
        )
    if method in ("narydis_lite", "mnse"):
        return NaryDisLiteEmbedding(
            n_features=n_features,
            d_embedding=d_embedding,
            bases=params.get("bases", (2, 3, 5, 7, 10)),
            max_digits=int(params.get("max_digits", 12)),
            multiplier=float(params.get("multiplier", 100.0)),
            signed=bool(params.get("signed", True)),
            pooling=str(params.get("pooling", "mean")),
        )
    if method == "dae":
        momentum = params.get("momentum")
        return DistributionAwareEmbedding(
            n_features=n_features,
            d_embedding=d_embedding,
            n_quantiles=int(params.get("n_quantiles", 16)),
            hidden_dim=int(params.get("hidden_dim", 32)),
            temperature=float(params.get("temperature", 1.0)),
            momentum=None if momentum is None else float(momentum),
            normalize_distances=bool(params.get("normalize_distances", False)),
        )
    if method in ("daes_gate", "daes_tran"):
        context_features = params.get("context_features")
        if context_features is None:
            n_context_fields = int(params.get("n_context_fields", 3))
            categorical = [
                (name, int(spec.get("vocab_size", 0)))
                for name, spec in feature_map.features.items()
                if spec.get("type") == "categorical"
            ]
            categorical.sort(key=lambda item: (item[1], item[0]))
            context_features = [
                name for name, _ in categorical[:n_context_fields]
            ]
        elif isinstance(context_features, str):
            context_features = [
                name.strip()
                for name in context_features.split(",")
                if name.strip()
            ]
        missing_context = [
            name
            for name in context_features
            if name not in feature_map.features
            or feature_map.features[name].get("type") != "categorical"
        ]
        if missing_context:
            raise ValueError(
                f"DAES context_features must be categorical: {missing_context}"
            )
        return DAESEmbedding(
            n_features=n_features,
            d_embedding=d_embedding,
            context_features=context_features,
            n_meta_embeddings=int(params.get("n_meta_embeddings", 10)),
            reservoir_size=int(params.get("reservoir_size", 100000)),
            beta=float(params.get("beta", 0.5)),
            modulation="gate" if method == "daes_gate" else "tran",
            quantile_refresh_interval=int(
                params.get("quantile_refresh_interval", 1)
            ),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        )
    if method in ("periodic", "plr"):
        return rtdl_num.PeriodicEmbeddings(
            n_features=n_features,
            d_embedding=d_embedding,
            n_frequencies=int(params.get("n_frequencies", 16)),
            frequency_init_scale=float(params.get("frequency_init_scale", 0.05)),
            activation=method == "plr",
            lite=bool(params.get("lite", False)),
        )
    if method == "deer_keras":
        input_normalized = params.get("input_normalized", True)
        if isinstance(input_normalized, str):
            input_normalized = input_normalized.strip().lower() in {
                "1", "true", "yes", "y"
            }
        default_max = 1.0 if input_normalized else 1.0e6
        return KerasDEEREmbedding(
            n_features=n_features,
            d_embedding=d_embedding,
            n_bins=int(params.get("n_bins", 16)),
            min_value=float(params.get("min_value", 0.0)),
            max_value=float(params.get("max_value", default_max)),
            boundary_init_limit=float(params.get("boundary_init_limit", 0.05)),
            embedding_init_std=float(params.get("embedding_init_std", 0.05)),
        )

    if method == "log_squared_bucket":
        vocabularies = params.get("vocabularies")
        if vocabularies is None:
            vocab_path = params.get("vocab_path")
            if not vocab_path:
                raise ValueError(
                    "log_squared_bucket requires vocab_path or vocabularies"
                )
            vocab_path = Path(vocab_path)
            if not vocab_path.is_absolute():
                vocab_path = Path(feature_map.data_dir) / vocab_path
            with vocab_path.open() as file:
                vocabularies = json.load(file)
        return LogSquaredBucketEmbedding(
            numeric_features=numeric_features,
            vocabularies=vocabularies,
            d_embedding=d_embedding,
            embedding_init_std=float(
                params.get("embedding_init_std", 1e-4)
            ),
        )

    bins = _load_bins(numeric_features, feature_map)
    if method == "bspline":
        lower = torch.stack([edges[0] for edges in bins])
        upper = torch.stack([edges[-1] for edges in bins])
        return BSplineEmbedding(
            lower=lower,
            upper=upper,
            d_embedding=d_embedding,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
        )
    if method == "atlas_hp":
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        response_knots = params.get("response_knots")
        if response_knots is None:
            prior_path = params.get("response_prior_path")
            if not prior_path:
                raise ValueError(
                    "atlas_hp requires response_prior_path or response_knots"
                )
            with Path(prior_path).open() as file:
                prior = json.load(file)
            if int(prior.get("n_bins", n_bins)) != n_bins:
                raise ValueError(
                    f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                )
            knot_map = prior.get("knot_scores", {})
            missing = [name for name in numeric_features if name not in knot_map]
            if missing:
                raise ValueError(
                    f"Response prior is missing numerical fields: {missing}"
                )
            response_knots = [knot_map[name] for name in numeric_features]
        return HPMixtureEmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            smooth_logit_init=float(params.get("smooth_logit_init", 0.0)),
            gate_temperature=float(params.get("gate_temperature", 1.0)),
            router_gradient_scale=float(params.get("router_gradient_scale", 1.0)),
            response_scale=float(params.get("response_scale", 1.0)),
            response_clip=float(params.get("response_clip", 3.0)),
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
            boundary_init_noise=params.get("boundary_init_noise"),
            variance_preserving=bool(params.get("variance_preserving", True)),
        )
    if method == "atlas_residual":
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        response_knots = params.get("response_knots")
        if response_knots is None:
            prior_path = params.get("response_prior_path")
            if not prior_path:
                raise ValueError(
                    "atlas_residual requires response_prior_path or response_knots"
                )
            with Path(prior_path).open() as file:
                prior = json.load(file)
            if int(prior.get("n_bins", n_bins)) != n_bins:
                raise ValueError(
                    f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                )
            knot_map = prior.get("knot_scores", {})
            missing = [name for name in numeric_features if name not in knot_map]
            if missing:
                raise ValueError(
                    f"Response prior is missing numerical fields: {missing}"
                )
            response_knots = [knot_map[name] for name in numeric_features]
        gradient_scale = params.get("correction_gradient_scale")
        return ResponseAnchoredResidualEmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            correction_scale=float(params.get("correction_scale", 1.0)),
            correction_gradient_scale=(
                None if gradient_scale is None else float(gradient_scale)
            ),
            response_scale=float(params.get("response_scale", 1.0)),
            response_clip=float(params.get("response_clip", 3.0)),
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
            boundary_init_noise=params.get("boundary_init_noise"),
        )
    if method == "atlas_sobolev":
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        response_knots = params.get("response_knots")
        if response_knots is None:
            prior_path = params.get("response_prior_path")
            if not prior_path:
                raise ValueError(
                    "atlas_sobolev requires response_prior_path or response_knots"
                )
            with Path(prior_path).open() as file:
                prior = json.load(file)
            if int(prior.get("n_bins", n_bins)) != n_bins:
                raise ValueError(
                    f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                )
            knot_map = prior.get("knot_scores", {})
            missing = [name for name in numeric_features if name not in knot_map]
            if missing:
                raise ValueError(
                    f"Response prior is missing numerical fields: {missing}"
                )
            response_knots = [knot_map[name] for name in numeric_features]
        return AdaptiveSobolevDEEREmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            smooth_logit_init=float(params.get("smooth_logit_init", 0.0)),
            gate_temperature=float(params.get("gate_temperature", 1.0)),
            response_scale=float(params.get("response_scale", 1.0)),
            response_clip=float(params.get("response_clip", 3.0)),
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
            boundary_init_noise=params.get("boundary_init_noise"),
        )
    if method == "atlas":
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        response_knots = params.get("response_knots")
        if response_knots is None:
            prior_path = params.get("response_prior_path")
            if not prior_path:
                raise ValueError(
                    "atlas requires response_prior_path or response_knots"
                )
            with Path(prior_path).open() as file:
                prior = json.load(file)
            if int(prior.get("n_bins", n_bins)) != n_bins:
                raise ValueError(
                    f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                )
            knot_map = prior.get("knot_scores", {})
            missing = [name for name in numeric_features if name not in knot_map]
            if missing:
                raise ValueError(
                    f"Response prior is missing numerical fields: {missing}"
                )
            response_knots = [knot_map[name] for name in numeric_features]
        local_dim = params.get("local_dim")
        return OrthogonalAtlasEmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            local_dim=None if local_dim is None else int(local_dim),
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            rank_scale=float(params.get("rank_scale", 1.0)),
            response_scale=float(params.get("response_scale", 1.0)),
            response_clip=float(params.get("response_clip", 3.0)),
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
            boundary_init_noise=params.get("boundary_init_noise"),
        )
    if method == "spline_response_init":
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        response_knots = params.get("response_knots")
        if response_knots is None:
            prior_path = params.get("response_prior_path")
            if not prior_path:
                raise ValueError(
                    "spline_response_init requires response_prior_path or response_knots"
                )
            with Path(prior_path).open() as file:
                prior = json.load(file)
            if int(prior.get("n_bins", n_bins)) != n_bins:
                raise ValueError(
                    f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                )
            knot_map = prior.get("knot_scores", {})
            missing = [name for name in numeric_features if name not in knot_map]
            if missing:
                raise ValueError(
                    f"Response prior is missing numerical fields: {missing}"
                )
            response_knots = [knot_map[name] for name in numeric_features]
        return ResponseInitializedSplineEmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            response_scale=float(params.get("response_scale", 1.0)),
            response_clip=float(params.get("response_clip", 3.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        )
    if method == "wavespec":
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        response_knots = params.get("response_knots")
        if response_knots is None:
            prior_path = params.get("response_prior_path")
            if not prior_path:
                raise ValueError(
                    "wavespec requires response_prior_path or response_knots"
                )
            with Path(prior_path).open() as file:
                prior = json.load(file)
            if int(prior.get("n_bins", n_bins)) != n_bins:
                raise ValueError(
                    f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                )
            knot_map = prior.get("knot_scores", {})
            missing = [
                name for name in numeric_features if name not in knot_map
            ]
            if missing:
                raise ValueError(
                    f"Response prior is missing numerical fields: {missing}"
                )
            response_knots = [knot_map[name] for name in numeric_features]
        requested_levels = params.get("n_levels")
        return WaveSpecEmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            response_knots=response_knots,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            n_levels=(
                None if requested_levels is None else int(requested_levels)
            ),
            detail_scale=float(params.get("detail_scale", 1.0)),
            level_decay=float(params.get("level_decay", 1.0)),
            detail_init_std=float(params.get("detail_init_std", 0.0)),
            local_order=int(params.get("local_order", 0)),
            local_scale=float(params.get("local_scale", 1.0)),
            local_init_std=float(params.get("local_init_std", 0.0)),
            response_scale=float(params.get("response_scale", 1.0)),
            response_clip=float(params.get("response_clip", 3.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        )
    if method in ("quantile_spline", "adaptive_spline"):
        n_bins = int(params.get("n_bins", 16))
        initial_edges = _resample_bin_edges(bins, n_bins)
        return QuantileSplineEmbedding(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
            learnable_boundaries=method == "adaptive_spline",
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        )
    if method == "narydis_full":
        lower = torch.stack([edges[0] for edges in bins])
        upper = torch.stack([edges[-1] for edges in bins])
        return NaryDisEmbedding(
            lower=lower,
            upper=upper,
            d_embedding=d_embedding,
            bases=params.get("bases", tuple(range(2, 11))),
            max_digits=int(params.get("max_digits", 12)),
            multiplier=float(params.get("multiplier", 100.0)),
            signed=bool(params.get("signed", True)),
            temperature=float(params.get("temperature", 1.0)),
            min_temperature=float(params.get("min_temperature", 0.1)),
            temperature_decay=float(params.get("temperature_decay", 5e-5)),
            attention_hidden_dims=params.get("attention_hidden_dims", (80, 40)),
            ssl_hidden_dim=int(params.get("ssl_hidden_dim", 64)),
            ssl_temperature=float(params.get("ssl_temperature", 0.1)),
            ssl_weight=float(params.get("ssl_weight", 0.0)),
            ssl_max_samples=int(params.get("ssl_max_samples", 512)),
            mask_zero_digits=bool(params.get("mask_zero_digits", True)),
        )
    if method in (
        "deer_asinh",
        "deer_atom",
        "deer",
        "liftdeer",
        "deer_graphframe",
        "deer_graphres",
        "deer_moe",
        "deer_deform",
        "deer_transport",
        "deer_response",
        "deer_response_init",
        "deer_jump",
        "deer_crossbasis",
        "deer_residual",
        "deer_residual_lr",
        "deer_multigrid",
        "deer_switch",
        "deer_dualchart",
        "deer_adapter",
        "deer_schwarz",
        "wave_spline",
        "deer_sid",
        "deer_rqvae",
        "deer_prq",
        "deer_prq_gate",
        "deer_prq_taskgate",
        "deer_prq_sid",
        "deer_multigrid_lr",
        "deer_subspace",
        "deer_trend",
        "deer_diffusion",
        "deer_mobius",
        "deer_mobius_shared",
        "deer_quadratic",
        "deer_hermite",
        "deer_weno",
        "deer_bubble",
        "deer_multires",
    ):
        n_bins = int(params.get("n_bins", 16))
        boundary_scheme = str(
            params.get("boundary_scheme", "quantile")
        ).lower()
        if boundary_scheme == "quantile":
            initial_edges = _resample_bin_edges(bins, n_bins)
        elif boundary_scheme == "fixed":
            min_value = float(params.get("min_value", 0.0))
            max_value = float(params.get("max_value", 1.0))
            if max_value <= min_value:
                raise ValueError("max_value must be greater than min_value")
            initial_edges = torch.linspace(
                min_value, max_value, n_bins + 1
            ).repeat(n_features, 1)
        else:
            raise ValueError(
                "boundary_scheme must be either 'quantile' or 'fixed'"
            )
        boundary_init_noise = params.get("boundary_init_noise")
        embedding_class = {
            "deer_asinh": AsinhWarpDEEREmbedding,
            "deer_atom": AtomicDEEREmbedding,
            "deer": DEEREmbedding,
            "liftdeer": LiftDEEREmbedding,
            "deer_graphframe": GraphFrameDEEREmbedding,
            "deer_graphres": ResidualGraphFrameDEEREmbedding,
            "deer_moe": MixtureGridDEEREmbedding,
            "deer_deform": ContextDeformableDEEREmbedding,
            "deer_transport": SparseTransportDEEREmbedding,
            "deer_response": ResponsePriorDEEREmbedding,
            "deer_response_init": ResponseInitializedDEEREmbedding,
            "deer_jump": DiscontinuousResidualDEEREmbedding,
            "deer_crossbasis": CrossBasisDEEREmbedding,
            "deer_residual": ResidualBasisDEEREmbedding,
            "deer_residual_lr": LowRankSmoothResidualDEEREmbedding,
            "deer_multigrid": MultigridResidualEmbedding,
            "deer_switch": ProlongatedSwitchEmbedding,
            "deer_dualchart": ProjectedDualChartEmbedding,
            "deer_adapter": ResidualFunctionAdapterEmbedding,
            "deer_schwarz": HierarchicalSchwarzEmbedding,
            "wave_spline": WaveSplineEmbedding,
            "deer_sid": ResidualQuantizedDEEREmbedding,
            "deer_rqvae": ResidualQuantizedDEEREmbedding,
            "deer_prq": ResidualQuantizedDEEREmbedding,
            "deer_prq_gate": ResidualQuantizedDEEREmbedding,
            "deer_prq_sid": ResidualQuantizedDEEREmbedding,
            "deer_prq_taskgate": ResidualQuantizedDEEREmbedding,
            "deer_multigrid_lr": LowRankMultigridEmbedding,
            "deer_subspace": ComplementarySubspaceEmbedding,
            "deer_trend": TrendResidualDEEREmbedding,
            "deer_diffusion": DiffusionDEEREmbedding,
            "deer_mobius": MobiusDEEREmbedding,
            "deer_mobius_shared": SharedMobiusDEEREmbedding,
            "deer_quadratic": QuadraticDEEREmbedding,
            "deer_hermite": HermiteDEEREmbedding,
            "deer_weno": WENODEEREmbedding,
            "deer_bubble": HierarchicalBubbleDEEREmbedding,
            "deer_multires": MultiResolutionDEEREmbedding,
        }[method]
        embedding_kwargs = dict(
            initial_edges=initial_edges,
            d_embedding=d_embedding,
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
            boundary_init_noise=(
                None
                if boundary_init_noise is None
                else float(boundary_init_noise)
            ),
        )
        if method in (
            "deer_sid",
            "deer_rqvae",
            "deer_prq",
            "deer_prq_sid",
            "deer_prq_gate",
            "deer_prq_taskgate",
        ):
            embedding_kwargs.update(
                n_codebooks=int(params.get("n_codebooks", 3)),
                codebook_size=int(params.get("codebook_size", 4)),
                quantization_space=(
                    "rank"
                    if method == "deer_sid"
                    else "projection"
                    if method in (
                        "deer_prq",
                        "deer_prq_gate",
                        "deer_prq_taskgate",
                        "deer_prq_sid",
                    )
                    else "latent"
                ),
                sid_scale_init=float(params.get("sid_scale_init", 0.0)),
                sid_scale_limit=float(params.get("sid_scale_limit", 1.0)),
                quantized_fraction=float(
                    params.get("quantized_fraction", 0.5)
                ),
                adaptive_fraction=bool(
                    params.get("adaptive_fraction", method == "deer_prq_gate")
                ),
                task_gate=bool(
                    params.get("task_gate", method == "deer_prq_taskgate")
                ),
                task_gate_init=float(params.get("task_gate_init", 0.0)),
                task_gate_limit=float(params.get("task_gate_limit", 0.5)),
                task_code_adapter=bool(
                    params.get("task_code_adapter", method == "deer_prq_sid")
                ),
                task_code_scale=float(
                    params.get("task_code_scale", 1.0)
                ),
                quality_slope_init=float(
                    params.get("quality_slope_init", 1.0)
                ),
                quality_center=float(params.get("quality_center", 0.5)),
                preserve_output_norm=bool(
                    params.get("preserve_output_norm", False)
                ),
                codebook_weight=float(params.get("codebook_weight", 1.0)),
                commitment_weight=float(
                    params.get("commitment_weight", 0.25)
                ),
                balance_weight=float(params.get("balance_weight", 0.0)),
                balance_temperature=float(
                    params.get("balance_temperature", 0.1)
                ),
                kmeans_iterations=int(params.get("kmeans_iterations", 8)),
                projection_topk=int(params.get("projection_topk", 1)),
                projection_temperature=float(
                    params.get("projection_temperature", 0.1)
                ),
                variance_preserving=bool(
                    params.get("variance_preserving", True)
                ),
            )
        elif method == "liftdeer":
            requested_levels = params.get("n_levels")
            embedding_kwargs.update(
                n_levels=(
                    None if requested_levels is None else int(requested_levels)
                ),
                gain_limit=float(params.get("gain_limit", 1.0)),
                gain_gradient_scale=float(
                    params.get("gain_gradient_scale", 32.0)
                ),
            )
        elif method in ("deer_mobius", "deer_mobius_shared"):
            embedding_kwargs.update(
                warp_init_std=float(params.get("warp_init_std", 0.0)),
                log_scale_limit=float(
                    params.get("log_scale_limit", 4.0)
                ),
            )
        elif method == "deer_quadratic":
            embedding_kwargs.update(
                shape_init_std=float(params.get("shape_init_std", 0.0)),
                shape_limit=float(params.get("shape_limit", 0.99)),
            )
        elif method == "deer_hermite":
            embedding_kwargs.update(
                gate_init_std=float(params.get("gate_init_std", 0.0)),
                gate_limit=float(params.get("gate_limit", 1.0)),
            )
        elif method == "deer_weno":
            embedding_kwargs.update(
                smoothness_scale=float(params.get("smoothness_scale", 0.1)),
                smoothness_power=float(params.get("smoothness_power", 2.0)),
                cubic_strength=float(params.get("cubic_strength", 1.0)),
                channelwise_sensor=bool(
                    params.get("channelwise_sensor", False)
                ),
                detach_sensor=bool(params.get("detach_sensor", True)),
                linear_boundaries=bool(params.get("linear_boundaries", True)),
            )
        elif method == "deer_bubble":
            embedding_kwargs.update(
                residual_rank=int(params.get("residual_rank", 1)),
                coefficient_init_std=float(
                    params.get("coefficient_init_std", 0.0)
                ),
                direction_init_scale=float(
                    params.get("direction_init_scale", 1.0)
                ),
                residual_scale_limit=float(
                    params.get("residual_scale_limit", 0.5)
                ),
            )
        elif method == "deer_diffusion":
            embedding_kwargs.update(
                diffusion_limit=float(params.get("diffusion_limit", 0.25)),
                diffusion_init_std=float(
                    params.get("diffusion_init_std", 0.0)
                ),
            )
        elif method == "deer_trend":
            embedding_kwargs.update(
                residual_scale_limit=float(
                    params.get("residual_scale_limit", 0.5)
                ),
                residual_scale_init_std=float(
                    params.get("residual_scale_init_std", 0.0)
                ),
            )
        elif method == "deer_atom":
            embedding_kwargs.update(
                atom_values=params.get("atom_values"),
                atom_frequencies=params.get("atom_frequencies"),
                atom_scores=params.get("atom_scores"),
                min_atom_score=float(params.get("min_atom_score", 0.0)),
                min_atom_frequency=float(
                    params.get("min_atom_frequency", 0.0)
                ),
                max_atom_frequency=float(
                    params.get("max_atom_frequency", 1.0)
                ),
                atom_rank=(
                    None
                    if params.get("atom_rank") is None
                    else int(params["atom_rank"])
                ),
                atom_direction_init_std=params.get("atom_direction_init_std"),
                atom_tolerance=float(params.get("atom_tolerance", 0.0)),
                center_atoms=bool(params.get("center_atoms", False)),
            )
        elif method == "deer_moe":
            embedding_kwargs.update(
                expert_boundary_init_noise=float(
                    params.get("expert_boundary_init_noise", 0.02)
                ),
                mixture_scale_limit=float(
                    params.get("mixture_scale_limit", 1.0)
                ),
                gate_slope_init_std=float(
                    params.get("gate_slope_init_std", 0.0)
                ),
            )
        elif method == "deer_deform":
            context_features = params.get("context_features")
            if context_features is None:
                n_context_fields = int(params.get("n_context_fields", 8))
                categorical = [
                    (name, int(spec.get("vocab_size", 0)))
                    for name, spec in feature_map.features.items()
                    if spec.get("type") == "categorical"
                    and int(spec.get("vocab_size", 0)) > 1
                ]
                categorical.sort(key=lambda item: (item[1], item[0]))
                context_features = [
                    name for name, _ in categorical[:n_context_fields]
                ]
            elif isinstance(context_features, str):
                context_features = [
                    name.strip()
                    for name in context_features.split(",")
                    if name.strip()
                ]
            invalid = [
                name for name in context_features
                if name not in feature_map.features
                or feature_map.features[name].get("type") != "categorical"
            ]
            if invalid:
                raise ValueError(
                    f"DEER deform context_features must be categorical: {invalid}"
                )
            embedding_kwargs.update(
                context_features=context_features,
                n_samples=int(params.get("n_samples", 3)),
                context_rank=int(params.get("context_rank", 4)),
                max_offset_bins=float(params.get("max_offset_bins", 1.0)),
                initial_offset_bins=float(
                    params.get("initial_offset_bins", 0.75)
                ),
                residual_scale_limit=float(
                    params.get("residual_scale_limit", 1.0)
                ),
                router_init_std=float(
                    params.get("router_init_std", 1e-2)
                ),
            )
        elif method == "deer_crossbasis":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                gate_init=float(params.get("gate_init", 0.25)),
                gate_limit=float(params.get("gate_limit", 1.0)),
                gate_order=int(params.get("gate_order", 0)),
                gate_temperature=float(params.get("gate_temperature", 1.0)),
                variance_preserving=bool(params.get("variance_preserving", False)),
                gate_geometry_power=(
                    None
                    if params.get("gate_geometry_power") is None
                    else float(params.get("gate_geometry_power"))
                ),
                gate_geometry_scale=float(params.get("gate_geometry_scale", 1.0)),
                gate_geometry_scores=(
                    None
                    if params.get("gate_geometry_power") is None
                    else [
                        0.0
                        if edge.numel() <= 2
                        else float(
                            -(
                                ((edge[1:] - edge[:-1]).clamp_min(0.0)
                                 / (edge[-1] - edge[0]).clamp_min(1e-12))
                                * ((edge[1:] - edge[:-1]).clamp_min(1e-12)
                                   / (edge[-1] - edge[0]).clamp_min(1e-12)).log()
                            ).sum()
                            / np.log(edge.numel() - 1)
                        )
                        for edge in bins
                    ]
                ),
            )
        elif method == "deer_residual":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                residual_scale=float(params.get("residual_scale", 1.0)),
            )
        elif method == "deer_residual_lr":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                correction_rank=int(params.get("correction_rank", 2)),
                residual_scale=float(params.get("residual_scale", 1.0)),
                zero_factor=str(params.get("zero_factor", "coefficients")),
                latent_init_std=float(params.get("latent_init_std", 0.05)),
            )
        elif method == "deer_multigrid":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                correction_scale=float(params.get("correction_scale", 1.0)),
                correction_init_std=float(params.get("correction_init_std", 0.0)),
                variance_preserving=bool(params.get("variance_preserving", False)),
                correction_gradient_scale=(
                    None if params.get("correction_gradient_scale") is None
                    else float(params.get("correction_gradient_scale"))
                ),
            )
        elif method == "deer_switch":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                gate_init=float(params.get("gate_init", 0.5)),
                gate_temperature=float(params.get("gate_temperature", 0.25)),
                routing=str(params.get("routing", "soft")),
                anchor_gradient=bool(params.get("anchor_gradient", False)),
                polarization_weight=float(params.get("polarization_weight", 0.0)),
                polarization_warmup_steps=int(params.get("polarization_warmup_steps", 0)),
                polarization_ramp_steps=int(params.get("polarization_ramp_steps", 1)),
            )
        elif method == "deer_dualchart":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                gate_init=float(params.get("gate_init", 0.1)),
                gate_temperature=float(params.get("gate_temperature", 0.25)),
                gate_gradient_scale=float(
                    params.get("gate_gradient_scale", 32.0)
                ),
                channelwise_gate=bool(params.get("channelwise_gate", True)),
                projection_mode=str(
                    params.get("projection_mode", "restriction")
                ),
                projection_ridge=float(params.get("projection_ridge", 1e-2)),
            )
        elif method == "deer_adapter":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                adapter_limit=float(params.get("adapter_limit", 1.5)),
                adapter_gradient_scale=float(
                    params.get("adapter_gradient_scale", 32.0)
                ),
            )
        elif method == "deer_schwarz":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                detail_scale=float(params.get("detail_scale", 1.0)),
            )
        elif method == "wave_spline":
            requested_levels = params.get("n_levels")
            embedding_kwargs = dict(
                initial_edges=initial_edges,
                d_embedding=d_embedding,
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                n_levels=(None if requested_levels is None else int(requested_levels)),
                detail_scale=float(params.get("detail_scale", 1.0)),
                level_decay=float(params.get("level_decay", 1.0)),
                detail_init_std=float(params.get("detail_init_std", 0.0)),
                local_order=int(params.get("local_order", 0)),
                local_scale=float(params.get("local_scale", 1.0)),
                local_init_std=float(params.get("local_init_std", 0.0)),
            )
        elif method == "deer_multigrid_lr":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                correction_rank=int(params.get("correction_rank", 2)),
                correction_scale=float(params.get("correction_scale", 1.0)),
            )
        elif method == "deer_subspace":
            embedding_kwargs.update(
                n_basis=int(params.get("n_basis", 8)),
                degree=int(params.get("degree", 3)),
                spline_fraction=float(params.get("spline_fraction", 0.5)),
            )
        elif method == "deer_jump":
            embedding_kwargs.update(
                residual_rank=int(params.get("residual_rank", 2)),
                residual_scale_limit=float(
                    params.get("residual_scale_limit", 0.5)
                ),
                coefficient_init_std=float(
                    params.get("coefficient_init_std", 0.0)
                ),
            )
        elif method in ("deer_response", "deer_response_init", "deer_graphframe", "deer_graphres"):
            response_knots = params.get("response_knots")
            if response_knots is None:
                prior_path = params.get("response_prior_path")
                if not prior_path:
                    raise ValueError(
                        "deer_response requires response_prior_path or response_knots"
                    )
                prior_path = Path(prior_path)
                with prior_path.open() as file:
                    prior = json.load(file)
                if int(prior.get("n_bins", n_bins)) != n_bins:
                    raise ValueError(
                        f"Response prior uses {prior.get('n_bins')} bins, expected {n_bins}"
                    )
                knot_map = prior.get("knot_scores", {})
                missing = [name for name in numeric_features if name not in knot_map]
                if missing:
                    raise ValueError(
                        f"Response prior is missing numerical fields: {missing}"
                    )
                response_knots = [knot_map[name] for name in numeric_features]
            embedding_kwargs.update(response_knots=response_knots)
            if method == "deer_response":
                embedding_kwargs.update(
                    prior_scale=float(params.get("prior_scale", 0.01)),
                    prior_scale_limit=float(params.get("prior_scale_limit", 0.0)),
                )
            elif method == "deer_response_init":
                embedding_kwargs.update(
                    response_scale=float(params.get("response_scale", 1.0)),
                    response_clip=float(params.get("response_clip", 3.0)),
                )
            elif method == "deer_graphframe":
                embedding_kwargs.update(
                    n_smooth=int(params.get("n_smooth", 4)),
                    n_response=int(params.get("n_response", 3)),
                    smoothing_steps=int(params.get("smoothing_steps", 4)),
                    response_scale=float(params.get("response_scale", 1.0)),
                    response_clip=float(params.get("response_clip", 3.0)),
                )
            else:
                embedding_kwargs.update(
                    n_response=int(params.get("n_response", 3)),
                    smoothing_steps=int(params.get("smoothing_steps", 4)),
                    smooth_scale=float(params.get("smooth_scale", 0.5)),
                    response_scale=float(params.get("response_scale", 1.0)),
                    response_clip=float(params.get("response_clip", 3.0)),
                )
        elif method == "deer_transport":
            field_context_map = params.get("field_context_map", {})
            if not isinstance(field_context_map, dict):
                raise ValueError(
                    "deer_transport requires field_context_map as a mapping"
                )
            invalid_numeric = [
                name for name in field_context_map
                if name not in numeric_features
            ]
            invalid_context = [
                name for name in field_context_map.values()
                if name not in feature_map.features
                or feature_map.features[name].get("type") != "categorical"
            ]
            if invalid_numeric or invalid_context:
                raise ValueError(
                    "Invalid deer_transport mapping: "
                    f"numeric={invalid_numeric}, context={invalid_context}"
                )
            embedding_kwargs.update(
                numeric_features=numeric_features,
                field_context_map=field_context_map,
                max_shift_bins=float(params.get("max_shift_bins", 0.75)),
                gate_init=float(params.get("gate_init", 0.25)),
                router_init_std=float(params.get("router_init_std", 0.05)),
                context_norm_eps=float(params.get("context_norm_eps", 1e-8)),
                detach_context=bool(params.get("detach_context", False)),
            )
        elif method == "deer_asinh":
            embedding_kwargs.update(
                warp_limit=float(params.get("warp_limit", 1.0)),
                warp_init=float(params.get("warp_init", 0.0)),
            )
        elif method == "deer_multires":
            embedding_kwargs.update(
                n_controls=int(params.get("n_controls", 8)),
                residual_rank=int(params.get("residual_rank", 1)),
                residual_direction_init_std=float(
                    params.get("residual_direction_init_std", 1e-2)
                ),
            )
        module = embedding_class(**embedding_kwargs)
        if method == "deer":
            module.set_execution_mode(str(params.get("deer_execution_mode", "auto")))
        return module
    if method == "quantile_bucket":
        return QuantileBucketEmbedding(bins, d_embedding)
    return rtdl_num.PiecewiseLinearEmbeddings(
        bins=bins,
        d_embedding=d_embedding,
        activation=method == "piecewise_linear_relu",
        version=str(params.get("version", "B")),
    )


def _make_registry_builder(method: str):
    def builder(*, numeric_features, feature_map, d_embedding, params):
        return build_numerical_embedding(
            method,
            numeric_features,
            feature_map,
            d_embedding,
            params,
        )

    return builder


for _method in SUPPORTED_NUM_EMBEDDINGS:
    register_numerical_embedding(
        _method, _make_registry_builder(_method), override=True
    )


class MixedFeatureEmbedding(nn.Module):
    """FuxiCTR categorical embeddings plus a selectable numerical tokenizer."""

    def __init__(
        self,
        feature_map,
        embedding_dim: int,
        numerical_embedding: str = "field",
        numerical_embedding_params: Optional[Dict] = None,
        embedding_initializer: str = "partial(nn.init.normal_, std=1e-4)",
    ):
        super().__init__()
        self.feature_map = feature_map
        self.numeric_features = [
            name
            for name, spec in feature_map.features.items()
            if spec["type"] == "numeric"
        ]
        if not self.numeric_features:
            raise ValueError(
                "NumEmb requires at least one feature with type='numeric'. "
                "Use the AvazuNum dataset or feature_specs overrides."
            )
        non_numeric_features = [
            name
            for name, spec in feature_map.features.items()
            if spec["type"] not in ("numeric", "meta")
        ]
        self.other_embeddings = FeatureEmbeddingDict(
            feature_map,
            embedding_dim,
            embedding_initializer=embedding_initializer,
            required_feature_columns=non_numeric_features,
        )
        self.numeric_embedding = build_numerical_embedding(
            numerical_embedding,
            self.numeric_features,
            feature_map,
            embedding_dim,
            numerical_embedding_params,
        )

    def forward(self, inputs, flatten_emb: bool = False) -> torch.Tensor:
        other = self.other_embeddings(inputs)
        numeric_x = torch.stack(
            [inputs[name].float().view(-1) for name in self.numeric_features], dim=1
        )
        if getattr(
            self.numeric_embedding, "requires_context_embeddings", False
        ):
            context_embeddings = {
                name: other[name]
                for name in self.numeric_embedding.context_features
                if name in other
            }
            numeric = self.numeric_embedding(
                numeric_x, context_embeddings=context_embeddings
            )
        else:
            numeric = self.numeric_embedding(numeric_x)
        numeric_by_name = {
            name: numeric[:, i, :] for i, name in enumerate(self.numeric_features)
        }
        merged = OrderedDict()
        for name, spec in self.feature_map.features.items():
            if spec["type"] == "meta":
                continue
            if name in numeric_by_name:
                merged[name] = numeric_by_name[name]
            elif name in other:
                merged[name] = other[name]
        return self.other_embeddings.dict2tensor(merged, flatten_emb=flatten_emb)
