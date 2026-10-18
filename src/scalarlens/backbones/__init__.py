"""FuxiCTR 2.3.10 consumers with the shared numerical-token LR interface."""
from importlib import import_module

BACKBONES = ("DNN", "DeepFM", "WideDeep", "NFM", "PNN", "DCN", "DCNv2", "AutoInt", "FiBiNET")


def get_backbone(name):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose {BACKBONES}")
    return getattr(import_module(f"{__name__}.{name.lower()}"), name)
