"""
Utilities to load models for scripts.
"""

import omegaconf

from compositional_learning_experiments.models import sequence, tree


def model_class(name: str):
    """
    Get the model constructor associated with a model name.

    The function signature of the return value depends on "name," so it's better
    to use the typical constructor whenever possible.
    """
    if name == "SiameseLSTM":
        return sequence.SiameseLSTM
    if name == "SiameseTransformer":
        return sequence.SiameseTransformer
    if name == "TreeTransformer":
        return sequence.TreeTransformer
    if name == "TreeRNN":
        return tree.TreeRNN
    if name == "VectorQuantizedTreeRNN":
        return tree.VectorQuantizedTreeRNN
    if name == "RoundingTreeRNN":
        return tree.RoundingTreeRNN

    raise ValueError("Unrecognized model type:", name)


def new_model_from_config(cfg: omegaconf.DictConfig):
    """
    Initialize a new model from a config file.
    """

    dict_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    args = {
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        **dict_cfg["model"],
        **dict_cfg["data"],
    }
    constructor = model_class(cfg.model_meta.name)
    return constructor(**args)
