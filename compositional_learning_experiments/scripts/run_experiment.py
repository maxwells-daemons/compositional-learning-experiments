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

    model = models.loading.new_model_from_config(cfg)
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.getcwd(), name="", version=""),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(save_top_k=-1),
        **omegaconf.OmegaConf.to_container(cfg.trainer, resolve=True),
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

    # Workaround to add metrics to tensorboard
    train_acc = np.mean(
        [results["train"][d]["accuracy"] for d in cfg.data.train_depths]
    )
    val_acc = np.mean([results["val"][d]["accuracy"] for d in cfg.data.val_depths])
    test_acc = np.mean([results["test"][d]["accuracy"] for d in cfg.data.test_depths])

    model.logger.experiment.add_scalar("metric/train_acc", train_acc, 1)
    model.logger.experiment.add_scalar("metric/val_acc", val_acc, 1)
    model.logger.experiment.add_scalar("metric/test_acc", test_acc, 1)
    model.logger.experiment.flush()


if __name__ == "__main__":
    main()
