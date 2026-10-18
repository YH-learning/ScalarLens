"""Predeclared paper controls; not recommended new production methods."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def field_linear(x, w, b):
    return torch.bmm(x.transpose(0, 1), w).transpose(0, 1) + b

class LocalContextControl(nn.Module):
    """DEER + scale, local capacity, or a one-shot FiLM adaptation.

    FiLM sees the same ordered numeric/category tokens as Fast. All its
    parameters are used; width is selected by parameter count, never labels.
    This is a tabular control inspired by FiLM, not its original CV system.
    """
    requires_context_embeddings = True

    def __init__(self, local_encoder, numeric_features, d_embedding,
                 context_dims, kind, target_extra_params, norm_eps=1e-8,
                 context_rank=16):
        super().__init__()
        if kind not in ("norm", "local_mlp", "film"):
            raise ValueError(kind)
        self.local_encoder = local_encoder
        self.numeric_features = tuple(numeric_features)
        self.n_features = len(numeric_features)
        self.d_embedding = d_embedding
        self.context_dims = dict(context_dims) if kind == "film" else {}
        self.context_features = tuple(self.context_dims)
        self.kind, self.norm_eps = kind, float(norm_eps)
        if any(d != d_embedding for d in self.context_dims.values()):
            raise ValueError("This control requires common embedding dimensions")
        f, d, r = self.n_features, d_embedding, context_rank
        self.target_extra_params = int(target_extra_params)
        gain = 1. / math.sqrt(f*d)
        self.raw_output_gain = nn.Parameter(torch.full((f, 1), gain + math.log(-math.expm1(-gain))))
        self.hidden_dim = 0
        if kind != "norm":
            total_fields = f + len(self.context_features)
            def count(h):
                n = f*(d*h + h + h*d + d + 1)
                if kind == "film":
                    n += total_fields*d*r + r + f*(r*2*h + 2*h)
                return n
            self.hidden_dim = min(range(1, 4097),
                                  key=lambda h: abs(count(h)-self.target_extra_params))
            h = self.hidden_dim
            self.weight_in = nn.Parameter(torch.empty(f, d, h))
            self.bias_in = nn.Parameter(torch.zeros(f, h))
            self.weight_out = nn.Parameter(torch.empty(f, h, d))
            self.bias_out = nn.Parameter(torch.zeros(f, d))
            for w in (self.weight_in, self.weight_out):
                for matrix in w: nn.init.xavier_normal_(matrix)
            if kind == "film":
                self.context_weight = nn.Parameter(torch.empty(total_fields*d, r))
                self.context_bias = nn.Parameter(torch.zeros(r))
                nn.init.xavier_normal_(self.context_weight)
                self.film_weight = nn.Parameter(torch.empty(f, r, 2*h))
                self.film_bias = nn.Parameter(torch.zeros(f, 2*h))
                for matrix in self.film_weight: nn.init.xavier_normal_(matrix)
            actual = sum(p.numel() for n,p in self.named_parameters()
                         if not n.startswith("local_encoder."))
            assert actual == count(h), (actual, count(h))
            if abs(actual/self.target_extra_params - 1) > .05:
                raise ValueError("Cannot match useful extra capacity within 5%")
        self.extra_params = sum(p.numel() for n,p in self.named_parameters()
                                if not n.startswith("local_encoder."))

    def get_extra_state(self):
        return dict(version="story_control_v1", kind=self.kind,
                    numeric_features=self.numeric_features,
                    context_dims=tuple(self.context_dims.items()),
                    d_embedding=self.d_embedding, hidden_dim=self.hidden_dim,
                    target_extra_params=self.target_extra_params, norm_eps=self.norm_eps)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Story control checkpoint configuration mismatch")

    def normalize(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x, context_embeddings=None):
        z = self.normalize(self.local_encoder(x))
        if self.kind == "norm":
            return z * F.softplus(self.raw_output_gain)
        hidden = field_linear(z, self.weight_in, self.bias_in)
        if self.kind == "film":
            contexts = context_embeddings or {}
            cats = []
            for key, dim in self.context_dims.items():
                token = contexts[key]
                if token.shape != (x.shape[0], dim):
                    raise ValueError("Bad context shape: " + key)
                cats.append(token.to(z.dtype))
            tokens = torch.cat([z, self.normalize(torch.stack(cats, 1))], 1) if cats else z
            context = torch.tanh(tokens.flatten(1) @ self.context_weight + self.context_bias)
            context = context.unsqueeze(1).expand(-1, self.n_features, -1)
            gamma, beta = field_linear(context, self.film_weight, self.film_bias).chunk(2, -1)
            hidden = (1 + torch.tanh(gamma)) * hidden + beta
        output = field_linear(F.silu(hidden), self.weight_out, self.bias_out)
        return output * F.softplus(self.raw_output_gain)

class FastFunctionalControl(nn.Module):
    """Same initialized Fast parameter tensors, two forward ablations.

    frozen_message removes refreshed feedback. Categorical-destination columns
    then have no output path; nominal parameter count is identical but effective
    active capacity is NOT claimed identical. Record this explicitly.
    """
    requires_context_embeddings = True

    def __init__(self, core, kind):
        super().__init__()
        if kind not in ("no_norm", "frozen_message"): raise ValueError(kind)
        if core.coupling_mode != "joint": raise ValueError("joint core required")
        self.core, self.kind = core, kind
        self.context_features = core.context_features
        self.numeric_features = core.numeric_features
        self.n_features = core.n_features
        self.d_embedding = core.d_embedding

    def get_extra_state(self):
        return dict(version="fast_functional_control_v1", kind=self.kind)

    def set_extra_state(self, state):
        if state != self.get_extra_state(): raise RuntimeError("Ablation mismatch")

    def forward(self, x, context_embeddings=None):
        c = self.core
        tokens = c._encode(x, context_embeddings, return_tokens=True)
        z = tokens if self.kind == "no_norm" else c._normalize(tokens)
        hidden = F.silu(field_linear(z, c.mode_weight_in, c.mode_bias_in))
        drive, logits = field_linear(hidden, c.mode_weight_out, c.mode_bias_out).chunk(2,-1)
        batch, n = x.shape[0], c.n_numeric_modes
        drive, gate = drive.reshape(batch,-1), logits.reshape(batch,-1).sigmoid()
        state = drive.tanh()
        fixed = c._message(state) if self.kind == "frozen_message" else None
        states = []
        for alpha in c.raw_update_rate.sigmoid():
            message = fixed if fixed is not None else c._message(state)
            state = (1-alpha)*state + alpha*torch.tanh(drive + gate*message)
            states.append(state[:,:n].reshape(batch,c.n_features,c.modes_per_field))
        features = torch.cat([z[:,:c.n_features],
                              gate[:,:n].reshape(batch,c.n_features,c.modes_per_field),
                              *states], -1)
        hidden = F.silu(field_linear(features,c.readout_weight_in,c.readout_bias_in))
        output = field_linear(hidden,c.readout_weight_out,c.readout_bias_out)*F.softplus(c.raw_output_gain)
        if c.validate_inputs and not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite story ablation output")
        return output
