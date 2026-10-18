"""Frozen paper training defaults and explicit public method identifiers."""
from copy import deepcopy
from importlib.resources import files
import yaml

METHODS = ("scalarlens", "field", "deer", "daes_gate", "narydis")
ORIGINAL_PREFIX = {"scalarlens": "ScalarLens", "deer": "DEER", "daes_gate": "DAES", "narydis": "NaryDis"}


def training_params(backbone, method="scalarlens"):
    if method not in METHODS:
        raise ValueError(f"Training presets supported: {METHODS}")
    config = yaml.safe_load(files("scalarlens").joinpath("assets/paper.yaml").read_text())
    params = deepcopy(config["Base"])
    if method == "field":
        params.update(model=backbone, numeric_embedding="field", numeric_embedding_params={})
    else:
        params.update(deepcopy(config[f"{ORIGINAL_PREFIX[method]}_{backbone}"]))
    if method == "scalarlens":
        params["numeric_embedding"] = "scalarlens"
        # The frozen framework enables these exact engineering paths based on
        # the historical method name. Preserve them explicitly after renaming.
        params["numeric_embedding_params"].setdefault("reuse_context_embeddings", True)
        params["numeric_embedding_params"].setdefault("categorical_lookup_mode", "auto")
    params["numeric_embedding_plugins"] = ["scalarlens.plugin"]
    # Use the exact source feature-map metadata for every raw-scale field.
    params["numeric_embedding_params"].setdefault("standardize", False)
    return params
