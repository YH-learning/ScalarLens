"""Ordered, learnable scalar coordinates with adjacent-node interpolation."""
from pathlib import Path
from typing import Optional
import torch
from torch import nn
from .deer_engineering import precise_index as _deer_flat_index

class StableCoordinates(nn.Module):
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
            message = str(exc)
            pointer_device = (isinstance(exc, ValueError)
                              and "Pointer argument" in message
                              and "cannot be accessed from Triton" in message)
            if mode != "auto" or not (missing or compiler or pointer_device) \
                    or "out of memory" in message.lower():
                raise
            import warnings
            warnings.warn("DEER compiler unavailable; using original interpolation: "
                          + message.splitlines()[0], RuntimeWarning)
            self.set_execution_mode("reference")
            return self.forward_reference(x)
