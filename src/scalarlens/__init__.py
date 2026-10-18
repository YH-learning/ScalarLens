"""Public numerical embedding API, independent of a specific CTR backbone."""
def __getattr__(name):
    if name == "StableCoordinates":
        from .coordinates import StableCoordinates
        return StableCoordinates
    if name == "ScalarLensEmbedding":
        from .embedding import ScalarLensEmbedding
        return ScalarLensEmbedding
    raise AttributeError(name)

__version__ = "0.1.0"
__all__ = ["ScalarLensEmbedding", "StableCoordinates"]
