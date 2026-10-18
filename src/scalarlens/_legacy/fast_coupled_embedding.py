"""Fast real-valued, input-modulated low-rank numerical-field interaction.

This is a finite-depth learned recurrent model, NOT an equivalent solver for
the complex scattering system. No matrix inverse, decomposition, sample loop,
or complex/double conversion is used in forward. All fields are processed in
one batch and categorical tokens retain their identities.
"""
import math
import warnings
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FastCoupledEmbedding(nn.Module):
    """DEER local tokens -> gated low-rank exchanges -> field-wise readout.

    d(x,c), g(x,c) = vectorized field MLPs
    s0 = tanh(d)
    s[t+1] = (1-alpha[t])*s[t] + alpha[t]*tanh(d + g*((s[t] U) V))
    alpha=sigmoid(a), g=sigmoid(g_logits).

    Finite states remain within [-1,1] for finite intermediates. This does not
    guarantee contraction, bounded parameter gradients, or improved CTR AUC.
    Input affects both the drive and the propagation operator. The low-rank
    coupling acts on individual field modes, never a pre-pooled category token.
    Default complexity of propagation: O(B * steps * modes * rank).
    """

    requires_context_embeddings = True

    def __init__(
        self, local_encoder: nn.Module, numeric_features: Sequence[str],
        d_embedding: int, context_dims: Optional[Mapping[str, int]] = None,
        modes_per_field: int = 8, coupling_rank: int = 16, n_steps: int = 3,
        mode_hidden_dim: int = 32, readout_hidden_dim: int = 32,
        norm_eps: float = 1e-8, coupling_mode: str = "joint",
        validate_inputs: bool = True, execution_mode: str = "reference",
        use_numeric_values: bool = True,
        use_categorical_context: bool = True,
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
            ("coupling_rank", coupling_rank), ("n_steps", n_steps),
            ("mode_hidden_dim", mode_hidden_dim), ("readout_hidden_dim", readout_hidden_dim),
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(isinstance(d, bool) or int(d) != d or d < 1 for d in self.context_dims.values()):
            raise ValueError("context dimensions must be positive integers")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if coupling_mode not in ("joint", "numeric", "independent"):
            raise ValueError("coupling_mode must be joint, numeric, or independent")
        self.local_encoder = local_encoder
        self.n_features = len(self.numeric_features)
        self.n_fields = self.n_features + len(self.context_features)
        self.d_embedding = int(d_embedding)
        self.modes_per_field = int(modes_per_field)
        self.n_numeric_modes = self.n_features * self.modes_per_field
        self.n_modes = self.n_fields * self.modes_per_field
        self.coupling_rank = int(coupling_rank)
        self.n_steps = int(n_steps)
        self.norm_eps = float(norm_eps)
        self.coupling_mode = coupling_mode
        self.validate_inputs = bool(validate_inputs)
        if not isinstance(use_numeric_values, bool) or not isinstance(use_categorical_context, bool):
            raise ValueError("factorial information switches must be boolean")
        self.use_numeric_values = use_numeric_values
        self.use_categorical_context = use_categorical_context
        self.set_execution_mode(execution_mode)
        # Standard configs use equal dimensions, so this dictionary is empty.
        # Nonstandard context dimensions get explicit, field-specific adapters.
        self.context_adapters = nn.ModuleDict({
            str(i): nn.Linear(dim, self.d_embedding, bias=False)
            for i, dim in enumerate(self.context_dims.values()) if dim != self.d_embedding
        })
        h = int(mode_hidden_dim)
        self.mode_weight_in = nn.Parameter(torch.empty(self.n_fields, self.d_embedding, h))
        self.mode_bias_in = nn.Parameter(torch.zeros(self.n_fields, h))
        self.mode_weight_out = nn.Parameter(torch.empty(self.n_fields, h, 2 * self.modes_per_field))
        self.mode_bias_out = nn.Parameter(torch.zeros(self.n_fields, 2 * self.modes_per_field))
        self.coupling_down = nn.Parameter(torch.empty(self.n_modes, self.coupling_rank))
        self.coupling_up = nn.Parameter(torch.empty(self.coupling_rank, self.n_modes))
        self.raw_update_rate = nn.Parameter(torch.zeros(self.n_steps))
        readout_dim = self.d_embedding + self.modes_per_field * (self.n_steps + 1)
        rh = int(readout_hidden_dim)
        self.readout_weight_in = nn.Parameter(torch.empty(self.n_features, readout_dim, rh))
        self.readout_bias_in = nn.Parameter(torch.zeros(self.n_features, rh))
        self.readout_weight_out = nn.Parameter(torch.empty(self.n_features, rh, self.d_embedding))
        self.readout_bias_out = nn.Parameter(torch.zeros(self.n_features, self.d_embedding))
        gain = 1. / math.sqrt(self.n_features * self.d_embedding)
        raw_gain = gain + math.log(-math.expm1(-gain))
        self.raw_output_gain = nn.Parameter(torch.full((self.n_features, 1), raw_gain))
        for stack in (self.mode_weight_in, self.mode_weight_out,
                      self.readout_weight_in, self.readout_weight_out):
            for matrix in stack:
                nn.init.xavier_normal_(matrix)
        nn.init.xavier_normal_(self.coupling_down)
        nn.init.xavier_normal_(self.coupling_up)

    def get_extra_state(self):
        return {
            "version": "real_lowrank_v1", "numeric_features": self.numeric_features,
            "context_dims": tuple(self.context_dims.items()),
            "d_embedding": self.d_embedding, "modes_per_field": self.modes_per_field,
            "coupling_rank": self.coupling_rank, "n_steps": self.n_steps,
            "coupling_mode": self.coupling_mode, "norm_eps": self.norm_eps,
            "use_numeric_values": self.use_numeric_values,
            "use_categorical_context": self.use_categorical_context,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Fast coupled checkpoint layout/configuration does not match")

    @staticmethod
    def _field_linear(x, weight, bias):
        # Batched GEMM: no field-wise Python/kernel launch loop.
        return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1) + bias

    def _normalize(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.norm_eps)

    def _encode(self, x, context_embeddings, return_raw=False, return_tokens=False):
        if x.ndim != 2 or x.shape[1] != self.n_features or x.shape[0] == 0:
            raise ValueError(f"expected nonempty numerical input [B, {self.n_features}]")
        if x.device != self.coupling_down.device:
            raise ValueError("numerical input and module must be on the same device")
        if not return_tokens and self.validate_inputs and not torch.isfinite(x).all():
            raise ValueError("non-finite numerical input; impute in the data pipeline")
        x = x.to(self.mode_weight_in.dtype)
        local = self.local_encoder(x)
        if local.shape != (x.shape[0], self.n_features, self.d_embedding):
            raise ValueError("incompatible local encoder shape")
        local = local if self.use_numeric_values else local * 0.0
        contexts = context_embeddings or {}
        values = []
        for i, (feature, dim) in enumerate(self.context_dims.items()):
            if feature not in contexts:
                raise KeyError(f"missing configured categorical context: {feature}")
            token = contexts[feature]
            if token.shape != (x.shape[0], dim) or token.device != x.device:
                raise ValueError(f"context {feature!r} must be [B, {dim}] on input device")
            token = token.to(local.dtype)
            if str(i) in self.context_adapters:
                token = self.context_adapters[str(i)](token)
            token = token if self.use_categorical_context else token * 0.0
            values.append(token)
        tokens = torch.cat((local, torch.stack(values, dim=1)), dim=1) if values else local
        if return_tokens:
            return tokens
        # One check across all fields, not a device synchronization per field.
        if self.validate_inputs and not torch.isfinite(tokens).all():
            raise ValueError("non-finite local/context embedding")
        tokens = self._normalize(tokens)
        hidden = F.silu(self._field_linear(tokens, self.mode_weight_in, self.mode_bias_in))
        drive, logits = self._field_linear(hidden, self.mode_weight_out, self.mode_bias_out).chunk(2, dim=-1)
        result = (tokens[:, :self.n_features], drive.reshape(x.shape[0], self.n_modes),
                  logits.reshape(x.shape[0], self.n_modes).sigmoid())
        return (local, *result) if return_raw else result

    def _message(self, state):
        if self.coupling_mode == "joint":
            return (state @ self.coupling_down) @ self.coupling_up
        if self.coupling_mode == "numeric":
            numeric = (state[:, :self.n_numeric_modes] @ self.coupling_down[:self.n_numeric_modes]) @ self.coupling_up[:, :self.n_numeric_modes]
            return torch.cat((numeric, torch.zeros_like(state[:, self.n_numeric_modes:])), dim=1)
        # Only same-field blocks of U V; cross-field propagation is disabled.
        down = self.coupling_down.reshape(self.n_fields, self.modes_per_field, self.coupling_rank)
        up = self.coupling_up.reshape(self.coupling_rank, self.n_fields, self.modes_per_field).permute(1, 0, 2)
        blocks = torch.bmm(down, up)
        grouped = state.reshape(state.shape[0], self.n_fields, self.modes_per_field)
        return torch.bmm(grouped.transpose(0, 1), blocks).transpose(0, 1).reshape_as(state)

    def propagate(self, x, context_embeddings=None):
        """Inspection interface: normalized local tokens, gates, finite states."""
        local, drive, gate = self._encode(x, context_embeddings)
        state = torch.tanh(drive)
        states = []
        rates = self.raw_update_rate.sigmoid()
        for i in range(self.n_steps):
            proposal = torch.tanh(drive + gate * self._message(state))
            state = (1. - rates[i]) * state + rates[i] * proposal
            states.append(state)
        return local, gate, states

    def forward_reference(self, x, context_embeddings=None):
        local, gate, states = self.propagate(x, context_embeddings)
        numeric_gate = gate[:, :self.n_numeric_modes].reshape(x.shape[0], self.n_features, self.modes_per_field)
        numeric_states = [
            state[:, :self.n_numeric_modes].reshape(x.shape[0], self.n_features, self.modes_per_field)
            for state in states
        ]
        features = torch.cat([local, numeric_gate, *numeric_states], dim=-1)
        hidden = F.silu(self._field_linear(features, self.readout_weight_in, self.readout_bias_in))
        output = self._field_linear(hidden, self.readout_weight_out, self.readout_bias_out)
        output = output * F.softplus(self.raw_output_gain)
        if self.validate_inputs and not torch.isfinite(output).all():
            raise FloatingPointError("non-finite fast coupled readout")
        return output


    def set_execution_mode(self, mode):
        """Runtime-only switch; parameters, RNG and checkpoint schema unchanged."""
        if mode not in ("reference", "eager", "compiled", "auto"):
            raise ValueError("execution_mode must be reference, eager, compiled or auto")
        self.execution_mode = mode
        object.__setattr__(self, "_compiled_core", None)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        state["_compiled_core"] = None
        return state

    def _forward_core(self, tokens, x):
        # DEER routing remains eager and unchanged, including knot tie semantics.
        normalized = self._normalize(tokens)
        hidden = F.silu(self._field_linear(normalized, self.mode_weight_in, self.mode_bias_in))
        drive, logits = self._field_linear(hidden, self.mode_weight_out, self.mode_bias_out).chunk(2, dim=-1)
        batch = tokens.shape[0]
        drive = drive.reshape(batch, self.n_modes)
        gate = logits.reshape(batch, self.n_modes).sigmoid()
        state = torch.tanh(drive)
        rates = self.raw_update_rate.sigmoid()
        numeric_states = []
        n = self.n_numeric_modes
        for i in range(self.n_steps):
            if i == self.n_steps - 1 and self.coupling_mode == "joint":
                # Final categorical destinations cannot affect any returned value.
                message = (state @ self.coupling_down) @ self.coupling_up[:, :n]
                proposal = torch.tanh(drive[:, :n] + gate[:, :n] * message)
                state = (1. - rates[i]) * state[:, :n] + rates[i] * proposal
            else:
                proposal = torch.tanh(drive + gate * self._message(state))
                state = (1. - rates[i]) * state + rates[i] * proposal
            numeric_states.append(state[:, :n].reshape(batch, self.n_features, self.modes_per_field))
        features = torch.cat([
            normalized[:, :self.n_features],
            gate[:, :n].reshape(batch, self.n_features, self.modes_per_field),
            *numeric_states], dim=-1)
        hidden = F.silu(self._field_linear(features, self.readout_weight_in, self.readout_bias_in))
        output = self._field_linear(hidden, self.readout_weight_out, self.readout_bias_out)
        output = output * F.softplus(self.raw_output_gain)
        if self.validate_inputs:
            # Keep all three checks, but transfer their flags together once.
            flags = torch.stack((torch.isfinite(x).all(), torch.isfinite(tokens).all(),
                                 torch.isfinite(output).all()))
            return output, flags
        return output, None

    def forward(self, x, context_embeddings=None):
        mode = getattr(self, "execution_mode", "reference")
        if mode == "reference" or (mode == "auto" and x.device.type != "cuda"):
            return self.forward_reference(x, context_embeddings)
        tokens = self._encode(x, context_embeddings, return_tokens=True)
        if mode in ("compiled", "auto"):
            if self._compiled_core is None:
                object.__setattr__(self, "_compiled_core", torch.compile(
                    self._forward_core, fullgraph=True, dynamic=False))
            try:
                output, flags = self._compiled_core(tokens, x)
            except torch._dynamo.exc.TorchDynamoException as exc:
                # Unsupported compiler/device/shape combinations may use the
                # equivalent eager core; never hide OOM or model runtime errors.
                if mode != "auto" or "out of memory" in str(exc).lower():
                    raise
                warnings.warn("Fast compiler unavailable; using equivalent eager core: "
                              + str(exc).splitlines()[0], RuntimeWarning)
                self.set_execution_mode("eager")
                output, flags = self._forward_core(tokens, x)
        else:
            output, flags = self._forward_core(tokens, x)
        if flags is not None:
            finite_input, finite_tokens, finite_output = flags.detach().cpu().tolist()
            if not finite_input:
                raise ValueError("non-finite numerical input; impute in the data pipeline")
            if not finite_tokens:
                raise ValueError("non-finite local/context embedding")
            if not finite_output:
                raise FloatingPointError("non-finite fast coupled readout")
        return output

class InteractionResidualEmbedding(FastCoupledEmbedding):
    """Keep exact DEER local tokens; learn a cross-field residual.

    y_f = DEER_f(x_f) + tanh(a_f) * (R_joint - R_independent)_f.
    Both readouts share weights, drive and gates. The independent reference
    uses only diagonal field blocks of the SAME low-rank propagation matrix.
    This is a computational interaction contrast, not a causal estimate.
    At a=0 the full output is exactly DEER. No extra labels or batch statistics.
    counterfactual=False is the plain-residual ablation with R_joint only.
    """

    def __init__(self, *args, counterfactual=True, residual_init=0., **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(counterfactual, bool):
            raise ValueError("counterfactual must be boolean")
        if not math.isfinite(residual_init) or abs(residual_init) >= 1:
            raise ValueError("residual_init must be finite and in (-1,1)")
        self.counterfactual = counterfactual
        self.residual_scale = nn.Parameter(torch.full(
            (self.n_features, 1), math.atanh(residual_init)))

    def get_extra_state(self):
        result = super().get_extra_state()
        result.update(version="interaction_residual_v1",
                      counterfactual=self.counterfactual)
        return result

    def _independent_blocks(self):
        down = self.coupling_down.reshape(
            self.n_fields, self.modes_per_field, self.coupling_rank)
        up = self.coupling_up.reshape(
            self.coupling_rank, self.n_fields, self.modes_per_field).permute(1, 0, 2)
        return torch.bmm(down, up)

    def _states_from_drive(self, drive, gate, blocks=None):
        state = torch.tanh(drive)
        states = []
        for rate in self.raw_update_rate.sigmoid():
            if blocks is None:
                message = self._message(state)
            else:
                grouped = state.reshape(-1, self.n_fields, self.modes_per_field)
                message = torch.bmm(grouped.transpose(0, 1), blocks).transpose(
                    0, 1).reshape_as(state)
            state = (1-rate)*state + rate*torch.tanh(drive + gate*message)
            states.append(state)
        return states

    def _readout_view(self, local, gate, states):
        batch = local.shape[0]
        numeric_gate = gate[:, :self.n_numeric_modes].reshape(
            batch, self.n_features, self.modes_per_field)
        views = [s[:, :self.n_numeric_modes].reshape(
            batch, self.n_features, self.modes_per_field) for s in states]
        features = torch.cat([local, numeric_gate, *views], dim=-1)
        hidden = F.silu(self._field_linear(
            features, self.readout_weight_in, self.readout_bias_in))
        return self._field_linear(
            hidden, self.readout_weight_out, self.readout_bias_out
        ) * F.softplus(self.raw_output_gain)

    def forward(self, x, context_embeddings=None):
        raw_local, local, drive, gate = self._encode(
            x, context_embeddings, return_raw=True)
        joint = self._readout_view(
            local, gate, self._states_from_drive(drive, gate))
        if self.counterfactual:
            independent = self._readout_view(local, gate, self._states_from_drive(
                drive, gate, self._independent_blocks()))
            residual = joint - independent
        else:
            residual = joint
        output = raw_local + self.residual_scale.tanh() * residual
        if self.validate_inputs and not torch.isfinite(output).all():
            raise FloatingPointError("non-finite interaction residual")
        return output
