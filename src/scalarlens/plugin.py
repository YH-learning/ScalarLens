"""Register the public `scalarlens` FuxiCTR embedding name."""
from ._legacy import numerical_embeddings as _frozen_registry
from fuxictr.pytorch.layers.embeddings.numerical_embedding import register_numerical_embedding
from .factory import build_scalarlens

register_numerical_embedding("scalarlens", build_scalarlens, override=True)
