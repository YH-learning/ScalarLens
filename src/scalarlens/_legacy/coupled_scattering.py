"""Input-conditioned coupled scattering for numerical embeddings.

This is a digital, task-trained architecture inspired by coupled-mode scattering,
not a physical simulator or a claim of CTR gains. DEER supplies local tokens;
the input changes the Hermitian operator itself, not just its right-hand side.
There is no log transform, batch statistic, label prior, or auxiliary objective.
"""
import math
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ModeProjection(nn.Module):
    """Field-specific nonlinear detuning head; real parameters work with Adam."""

    def __init__(self, input_dim: int, hidden_dim: int, n_modes: int):
        super().__init__()
        self.weight_in = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias_in = nn.Parameter(torch.zeros(hidden_dim))
        self.weight_out = nn.Parameter(torch.empty(hidden_dim, n_modes))
        self.bias_out = nn.Parameter(torch.linspace(-1.0, 1.0, n_modes))
        nn.init.xavier_normal_(self.weight_in)
        nn.init.xavier_normal_(self.weight_out)

    def forward(self, token):
        return F.silu(token @ self.weight_in + self.bias_in) @ self.weight_out + self.bias_out


class CoupledScatteringEmbedding(nn.Module):
    """Solve [Gamma + i(H(x,c) - omega I)] a = b, then read per-field tokens.

    H = J + diag(delta(x,c)), J is Hermitian, Gamma >= damping_floor I.
    Consequently sigma_min(A) >= damping_floor in exact arithmetic. Positive
    damping guarantees nonsingularity, not bounded training loss or CTR gains.

    Each numerical/categorical field has its own set of modes. Categories are
    never mean-pooled. The learned drive b is sample-independent: all sample
    conditioning of the scattering response occurs through the operator.

    coupling_mode:
      joint       : numerical and categorical modes all interact (main model).
      numeric     : numerical fields interact; context cannot reach the readout.
      independent : only modes belonging to the same field interact.

    Input/output: [B, F] -> [B, F, D]. Context values are existing categorical
    embeddings [B, D_c]. Missing configured context is an error, never silently
    replaced by a static vector. Missing numerical values must be imputed by the
    dataset's documented preprocessing before this layer. DEER's range clipping
    remains unchanged.

    Model weights stay float32/float64; autocast is disabled around the complex
    solve. Calling model.half()/bfloat16() is intentionally unsupported.
    """

    requires_context_embeddings = True

    def __init__(
        self,
        local_encoder: nn.Module,
        numeric_features: Sequence[str],
        d_embedding: int,
        context_dims: Optional[Mapping[str, int]] = None,
        modes_per_field: int = 4,
        n_frequencies: int = 4,
        detuning_hidden_dim: int = 64,
        readout_hidden_dim: int = 64,
        damping_floor: float = 0.1,
        damping_init: float = 0.5,
        coupling_init_scale: float = 1.0,
        norm_eps: float = 1e-8,
        coupling_mode: str = "joint",
        solve_chunk_size: int = 64,
    ):
        super().__init__()
        self.numeric_features = tuple(numeric_features)
        self.context_dims = dict(context_dims or {})
        self.context_features = tuple(self.context_dims)
        if not self.numeric_features or len(set(self.numeric_features)) != len(self.numeric_features):
            raise ValueError("numeric_features must be nonempty and unique")
        if set(self.numeric_features) & set(self.context_features):
            raise ValueError("numeric and context features must be disjoint")
        for name, value in (
            ("d_embedding", d_embedding), ("modes_per_field", modes_per_field),
            ("n_frequencies", n_frequencies), ("detuning_hidden_dim", detuning_hidden_dim),
            ("readout_hidden_dim", readout_hidden_dim), ("solve_chunk_size", solve_chunk_size),
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if modes_per_field < 2:
            raise ValueError("modes_per_field must be >= 2 for repeated input injection")
        if any(isinstance(d, bool) or int(d) != d or d < 1 for d in self.context_dims.values()):
            raise ValueError("context embedding dimensions must be positive integers")
        if not (math.isfinite(damping_floor) and math.isfinite(damping_init)
                and 0 < damping_floor < damping_init):
            raise ValueError("require finite 0 < damping_floor < damping_init")
        if not math.isfinite(coupling_init_scale) or coupling_init_scale <= 0:
            raise ValueError("coupling_init_scale must be finite and positive")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if coupling_mode not in ("joint", "numeric", "independent"):
            raise ValueError("coupling_mode must be joint, numeric, or independent")
        self.local_encoder = local_encoder
        self.n_features = len(self.numeric_features)
        self.d_embedding = int(d_embedding)
        self.modes_per_field = int(modes_per_field)
        self.n_frequencies = int(n_frequencies)
        self.n_numeric_modes = self.n_features * self.modes_per_field
        self.n_modes = (self.n_features + len(self.context_features)) * self.modes_per_field
        self.damping_floor = float(damping_floor)
        self.norm_eps = float(norm_eps)
        self.coupling_mode = coupling_mode
        self.solve_chunk_size = int(solve_chunk_size)

        # Separate heads retain field identity, including differently sized contexts.
        node_dims = [self.d_embedding] * self.n_features + list(self.context_dims.values())
        self.detuning_heads = nn.ModuleList([
            ModeProjection(dim, int(detuning_hidden_dim), self.modes_per_field)
            for dim in node_dims
        ])
        # Parameterize only the strict upper triangle (no unused diagonal weights).
        rows, cols = torch.triu_indices(self.n_modes, self.n_modes, offset=1)
        self.register_buffer("coupling_rows", rows)
        self.register_buffer("coupling_cols", cols)
        owners = torch.arange(self.n_modes) // self.modes_per_field
        if coupling_mode == "joint":
            enabled = torch.ones_like(rows, dtype=torch.bool)
        elif coupling_mode == "numeric":
            enabled = ((owners[rows] < self.n_features) & (owners[cols] < self.n_features))
            enabled |= owners[rows] == owners[cols]
        else:
            enabled = owners[rows] == owners[cols]
        self.register_buffer("coupling_mask", enabled)
        self.coupling_real = nn.Parameter(torch.empty(rows.numel()))
        self.coupling_imag = nn.Parameter(torch.empty(rows.numel()))
        scale = float(coupling_init_scale) / math.sqrt(2 * self.n_modes)
        nn.init.normal_(self.coupling_real, std=scale)
        nn.init.normal_(self.coupling_imag, std=scale)
        gap = float(damping_init - damping_floor)
        inverse_softplus = gap + math.log(-math.expm1(-gap))
        self.raw_damping = nn.Parameter(torch.full((self.n_modes,), inverse_softplus))
        frequencies = (torch.zeros(1) if self.n_frequencies == 1
                       else torch.linspace(-2.0, 2.0, self.n_frequencies))
        self.frequencies = nn.Parameter(frequencies)
        self.drive_real = nn.Parameter(torch.empty(self.n_frequencies, self.n_modes))
        self.drive_imag = nn.Parameter(torch.empty(self.n_frequencies, self.n_modes))
        nn.init.normal_(self.drive_real, std=1.0 / math.sqrt(2 * self.n_modes))
        nn.init.normal_(self.drive_imag, std=1.0 / math.sqrt(2 * self.n_modes))
        self.register_buffer("identity", torch.eye(self.n_modes))

        # Unrestricted nonlinear readout, not a zero-initialized residual gate.
        # FM sums O(F^2) dot products. Calibrate the entire embedding's initial
        # scale by field count and dimension; a positive learned gain can grow
        # freely. This is not a cap or an additive DEER bypass.
        output_gain = 1.0 / math.sqrt(self.n_features * self.d_embedding)
        raw_gain = output_gain + math.log(-math.expm1(-output_gain))
        self.raw_output_gain = nn.Parameter(torch.full((self.n_features, 1), raw_gain))
        readout_dim = self.d_embedding + 3 * self.n_frequencies * self.modes_per_field
        self.readout_weight_in = nn.Parameter(torch.empty(
            self.n_features, readout_dim, int(readout_hidden_dim)))
        self.readout_bias_in = nn.Parameter(torch.zeros(self.n_features, int(readout_hidden_dim)))
        self.readout_weight_out = nn.Parameter(torch.empty(
            self.n_features, int(readout_hidden_dim), self.d_embedding))
        self.readout_bias_out = nn.Parameter(torch.zeros(self.n_features, self.d_embedding))
        for matrix in self.readout_weight_in:
            nn.init.xavier_normal_(matrix)
        for matrix in self.readout_weight_out:
            nn.init.xavier_normal_(matrix)

    def get_extra_state(self):
        # Reject equal-shaped checkpoints with different field order or semantics.
        return {
            "version": 2, "numeric_features": self.numeric_features,
            "context_dims": tuple(self.context_dims.items()),
            "modes_per_field": self.modes_per_field,
            "n_frequencies": self.n_frequencies,
            "coupling_mode": self.coupling_mode,
            "damping_floor": self.damping_floor, "norm_eps": self.norm_eps,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Scattering checkpoint field layout/configuration does not match")

    def _normalize_token(self, token):
        # Per-sample/per-field RMS normalization; no cross-row information.
        return token * torch.rsqrt(token.square().mean(dim=-1, keepdim=True) + self.norm_eps)

    def damping(self):
        return self.damping_floor + F.softplus(self.raw_damping)

    def coupling_matrix(self):
        weights = torch.complex(self.coupling_real, self.coupling_imag)
        weights = weights * self.coupling_mask.to(weights.dtype)
        upper = weights.new_zeros((self.n_modes, self.n_modes))
        upper = upper.index_put((self.coupling_rows, self.coupling_cols), weights)
        return upper + upper.mH

    def _build_system(self, x, context_embeddings):
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected numerical input [B, {self.n_features}], got {tuple(x.shape)}")
        dtype = self.coupling_real.dtype
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("Scattering weights must stay float32/float64; use autocast, not model.half()")
        if x.device != self.coupling_real.device:
            raise ValueError("input and scattering module must be on the same device")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite numerical input; impute missing values before scattering")
        x = x.to(dtype)
        local = self.local_encoder(x)
        if local.shape != (x.shape[0], self.n_features, self.d_embedding):
            raise ValueError("local encoder returned an incompatible shape")
        if not torch.isfinite(local).all():
            raise FloatingPointError("local encoder produced non-finite values")
        local = self._normalize_token(local)
        tokens = list(local.unbind(dim=1))
        contexts = context_embeddings or {}
        for feature, dim in self.context_dims.items():
            if feature not in contexts:
                raise KeyError(f"missing configured categorical context: {feature}")
            token = contexts[feature]
            if token.shape != (x.shape[0], dim):
                raise ValueError(f"context {feature!r} must have shape [{x.shape[0]}, {dim}]")
            if token.device != x.device:
                raise ValueError(f"context {feature!r} must be on the same device as numerical input")
            if not torch.isfinite(token).all():
                raise ValueError(f"non-finite context embedding: {feature}")
            tokens.append(self._normalize_token(token.to(dtype)))
        detuning = torch.cat([head(token) for head, token in zip(self.detuning_heads, tokens)], dim=-1)
        hamiltonian = self.coupling_matrix().unsqueeze(0) + torch.diag_embed(detuning)
        shifted = hamiltonian.unsqueeze(1) - self.frequencies[None, :, None, None] * self.identity
        system = torch.diag_embed(self.damping())[None, None] + 1j * shifted
        drive = torch.complex(self.drive_real, self.drive_imag)
        rhs = drive[None, :, :, None].expand(x.shape[0], -1, -1, -1)
        return local, system, rhs

    def build_system(self, x, context_embeddings=None):
        """Inspectable operator/RHS for residual and input-conditioning checks."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._build_system(x, context_embeddings)

    @staticmethod
    def _real_block_system(system, rhs):
        # Exact real representation of a complex linear system. The remote PPU
        # supports real LU but aborts the process on complex cuSOLVER getrf.
        # [Ar -Ai; Ai Ar] [zr; zi] = [br; bi], with unchanged model/gradients.
        top = torch.cat((system.real, -system.imag), dim=-1)
        bottom = torch.cat((system.imag, system.real), dim=-1)
        block = torch.cat((top, bottom), dim=-2)
        real_rhs = torch.cat((rhs.real, rhs.imag), dim=-2)
        return block, real_rhs

    @classmethod
    def _solve_system(cls, system, rhs):
        # The target PPU's complex LU is unsupported and large real LU can be
        # inaccurate. Use float64 normal equations + Cholesky on every device:
        # R.T R z = R.T b has exactly the same solution since R is nonsingular.
        # Normal equations square the condition number, hence float64, one
        # residual-correction step, and a hard check against the ORIGINAL system.
        # No jitter/ridge is added: that would change the scattering operator.
        block, real_rhs = cls._real_block_system(system, rhs)
        block, real_rhs = block.double(), real_rhs.double()
        transpose = block.mT
        factor = torch.linalg.cholesky(transpose @ block)
        solution = torch.cholesky_solve(transpose @ real_rhs, factor)
        residual = real_rhs - block @ solution
        solution = solution + torch.cholesky_solve(transpose @ residual, factor)
        residual_norm = (block @ solution - real_rhs).square().sum((-2, -1)).sqrt()
        rhs_norm = real_rhs.square().sum((-2, -1)).sqrt().clamp_min(1e-12)
        if not torch.isfinite(solution).all() or torch.any(residual_norm / rhs_norm > 1e-6):
            raise FloatingPointError("scattering solver did not reach relative residual <= 1e-6")
        real, imag = solution.split(system.shape[-1], dim=-2)
        return torch.complex(real, imag).to(system.dtype)

    def system_response(self, x, context_embeddings=None):
        """Return normalized local tokens and complex response [B, Q, M]."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            local, system, rhs = self._build_system(x, context_embeddings)
            response = self._solve_system(system, rhs).squeeze(-1)
            if not torch.isfinite(response).all():
                raise FloatingPointError("non-finite scattering solve; no silent fallback is applied")
            return local, response

    def forward(self, x, context_embeddings=None):
        # Exact sample-wise chunking + recomputation: a training batch of 4096
        # must not retain 4096 large factorizations. This does not micro-step the
        # optimizer, change the batch loss, or reduce the number of coupled modes.
        if x.shape[0] <= self.solve_chunk_size:
            return self._forward_block(x, context_embeddings)
        outputs = []
        for start in range(0, x.shape[0], self.solve_chunk_size):
            stop = start + self.solve_chunk_size
            context = None if context_embeddings is None else {
                name: value[start:stop] for name, value in context_embeddings.items()
            }
            values = x[start:stop]
            if self.training and torch.is_grad_enabled():
                output = checkpoint(
                    self._forward_block, values, context,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                output = self._forward_block(values, context)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def _forward_block(self, x, context_embeddings=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            local, response = self.system_response(x, context_embeddings)
            numerical = response[:, :, :self.n_numeric_modes].reshape(
                x.shape[0], self.n_frequencies, self.n_features, self.modes_per_field
            ).permute(0, 2, 1, 3).flatten(start_dim=2)
            observables = torch.cat((
                local, numerical.real, numerical.imag, numerical.abs().square()), dim=-1)
            hidden = F.silu(torch.einsum("bfi,fih->bfh", observables, self.readout_weight_in)
                            + self.readout_bias_in)
            output = torch.einsum("bfh,fhd->bfd", hidden, self.readout_weight_out) + self.readout_bias_out
            output = output * F.softplus(self.raw_output_gain)
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite scattering readout")
            return output

    @torch.no_grad()
    def diagnostics(self, x, context_embeddings=None):
        """Explicit diagnostics only; no batch-dependent state or graph caching."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, system, rhs = self._build_system(x, context_embeddings)
            response = self._solve_system(system, rhs)
            block, real_rhs = self._real_block_system(system, rhs)
            solution = torch.cat((response.real, response.imag), dim=-2).double()
            residual = (block.double() @ solution - real_rhs.double()).square().sum((-2, -1)).sqrt()
            rhs_norm = real_rhs.double().square().sum((-2, -1)).sqrt().clamp_min(1e-12)
            return {
                "n_modes": self.n_modes,
                "damping_min": self.damping().min().item(),
                "damping_max": self.damping().max().item(),
                "relative_residual_max": (residual / rhs_norm).max().item(),
                "response_abs_max": response.abs().max().item(),
                "coupling_mode": self.coupling_mode,
            }
