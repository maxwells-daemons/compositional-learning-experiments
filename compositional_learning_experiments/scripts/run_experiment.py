#!/usr/bin/env python

"""
A script to train models of all types and for all tasks this project supports.
"""

import json
import logging
import os

import hydra
import omegaconf
import numpy as np
import pytorch_lightning as pl
import torch

from compositional_learning_experiments import models, data


@hydra.main(config_path="../config/config.yaml", strict=False)
def main(cfg: omegaconf.DictConfig):
    cfg.data.train_path = hydra.utils.to_absolute_path(cfg.data.train_path)
    cfg.data.val_path = hydra.utils.to_absolute_path(cfg.data.val_path)
    cfg.data.test_path = hydra.utils.to_absolute_path(cfg.data.test_path)

    logger = logging.getLogger(__name__)
    logger.info("Beginning experiment")
    logger.info("hparams:\n" + cfg.pretty())

    args = {
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        **cfg.model,
        **cfg.data,
    }

    if cfg.model_meta.name == "SiameseLSTM":
        model = models.sequence.SiameseLSTM(**args)
    elif cfg.model_meta.name == "SiameseTransformer":
        model = models.sequence.SiameseTransformer(**args)
    elif cfg.model_meta.name == "TreeTransformer":
        model = models.sequence.TreeTransformer(**args)
    elif cfg.model_meta.name == "TreeRNN":
        model = models.tree.TreeRNN(**args)
    elif cfg.model_meta.name == "VectorQuantizedTreeRNN":
        model = models.tree.VectorQuantizedTreeRNN(**args)
    elif cfg.model_meta.name == "RoundingTreeRNN":
        model = models.tree.RoundingTreeRNN(**args)
    else:
        raise ValueError("Unrecognized model type:", cfg.model_meta.name)

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.getcwd(), name="", version=""),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(save_top_k=-1),
        **cfg.trainer,
    )

    logger.info("=== Beginning training ===")
    trainer.fit(model)

    def evaluate_depths(path, depths):
        results = {}
        for depth in depths:
            logger.info(f"-- Evaluating for depth {depth} --")
            batch_size = (
                cfg.training.batch_size if cfg.training.batch_size > 1 else None
            )
            dataset = data.load_dataset(path, [depth], model.data_format)
            loader = model.make_dataloader(dataset, False)
            results[depth] = trainer.test(model, loader)

        return results

    results = {}
    logger.info("=== Evaluating on training set ===")
    results["train"] = evaluate_depths(cfg.data.train_path, cfg.data.train_depths)

    logger.info("=== Evaluating on validation set ===")
    results["val"] = evaluate_depths(cfg.data.val_path, cfg.data.val_depths)

    logger.info("=== Evaluating on test set ===")
    results["test"] = evaluate_depths(cfg.data.test_path, cfg.data.test_depths)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
