"""Registry and builder API for pluggable numerical feature embeddings."""

from importlib import import_module
from typing import Callable, Dict, Iterable, List, Optional

from torch import nn


_NUMERICAL_EMBEDDING_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_numerical_embedding(
    name: str,
    builder: Optional[Callable[..., nn.Module]] = None,
    *,
    override: bool = False,
):
    """Register a numerical embedding builder.

    A builder receives keyword arguments numeric_features, feature_map,
    d_embedding and params and must return an nn.Module whose output shape is
    [batch_size, n_numeric_features, d_embedding].
    """

    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Numerical embedding name must be non-empty")

    def decorator(fn):
        if normalized_name in _NUMERICAL_EMBEDDING_REGISTRY and not override:
            raise KeyError(
                f"Numerical embedding {normalized_name!r} is already registered"
            )
        _NUMERICAL_EMBEDDING_REGISTRY[normalized_name] = fn
        return fn

    return decorator(builder) if builder is not None else decorator


def list_numerical_embeddings() -> List[str]:
    """Return registered numerical embedding names in sorted order."""

    return sorted(_NUMERICAL_EMBEDDING_REGISTRY)


def build_numerical_embedding(
    name: str,
    numeric_features: Iterable[str],
    feature_map,
    d_embedding: int,
    params: Optional[dict] = None,
    plugin_modules: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Import requested plugin modules and construct a registered embedding."""

    if isinstance(plugin_modules, str):
        plugin_modules = [plugin_modules]
    for module_name in plugin_modules or []:
        import_module(module_name)

    normalized_name = str(name).strip().lower()
    if normalized_name not in _NUMERICAL_EMBEDDING_REGISTRY:
        available = ", ".join(list_numerical_embeddings()) or "<none>"
        raise ValueError(
            f"Unknown numeric_embedding={normalized_name!r}. "
            f"Registered embeddings: {available}. "
            "Add the implementation module to numeric_embedding_plugins."
        )
    module = _NUMERICAL_EMBEDDING_REGISTRY[normalized_name](
        numeric_features=list(numeric_features),
        feature_map=feature_map,
        d_embedding=int(d_embedding),
        params=dict(params or {}),
    )
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"Builder for {normalized_name!r} returned "
            f"{type(module).__name__}, expected nn.Module"
        )
    return module
