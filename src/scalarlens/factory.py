"""Construct ScalarLens while preserving the historical initialization stream."""
import inspect
import torch

from . import ScalarLensEmbedding, StableCoordinates


def build_scalarlens(numeric_features, feature_map, d_embedding, params=None):
    from ._legacy import numerical_embeddings as frozen
    params = dict(params or {})
    names = list(numeric_features)
    bins = frozen._load_bins(names, feature_map)
    if params.get("boundary_scheme", "quantile") != "quantile":
        raise ValueError("Paper ScalarLens uses train-quantile initial boundaries")
    edges = frozen._resample_bin_edges(bins, int(params.get("n_bins", 16)))
    coordinate = StableCoordinates(
        edges, d_embedding,
        min_width=float(params.get("min_width", 1e-4)),
        width_temperature=float(params.get("width_temperature", 1)),
        embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        boundary_init_noise=params.get("boundary_init_noise"),
    )
    coordinate.set_execution_mode(params.get("deer_execution_mode", "auto"))
    context = params.get("context_features")
    if context is None:
        context = [name for name, spec in feature_map.features.items() if spec["type"] == "categorical"]
    if isinstance(context, str):
        context = [name.strip() for name in context.split(",") if name.strip()]
    if len(context) != len(set(context)):
        raise ValueError("Context fields must be unique")
    invalid = [name for name in context if name not in feature_map.features or feature_map.features[name]["type"] != "categorical"]
    if invalid:
        raise ValueError(f"Context fields must be categorical: {invalid}")
    dims = {name: int(feature_map.features[name].get("embedding_dim", d_embedding)) for name in context}
    accepted = set(inspect.signature(ScalarLensEmbedding.__init__).parameters) - {"self", "local_encoder", "numeric_features", "d_embedding", "context_dims"}
    kwargs = {name: value for name, value in params.items() if name in accepted}
    # PyYAML's YAML 1.1 resolver reads historical `1e-8` as a string. The
    # original registry explicitly casts this field; preserve that behavior.
    if "norm_eps" in kwargs:
        kwargs["norm_eps"] = float(kwargs["norm_eps"])
    kwargs.setdefault("execution_mode", "auto")
    # The original registry isolates response initialization from the subsequent
    # categorical/backbone parameter initialization, while constructing DEER first.
    with torch.random.fork_rng(devices=[]):
        return ScalarLensEmbedding(coordinate, names, d_embedding, context_dims=dims, **kwargs)
