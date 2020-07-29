#!/usr/bin/env python

"""
A script to train models of all types and for all tasks this project supports.
"""

import os

import hydra
import omegaconf
import pytorch_lightning as pl

from compositional_learning_experiments.models import seq2seq, equation_verification


@hydra.main(config_path="config/config.yaml", strict=False)
def main(cfg: omegaconf.DictConfig):
    print(cfg.pretty())

    assert cfg.model_meta.domain == cfg.task.domain

    args = {
        "task_name": cfg.task.name,
        "train_dataset": hydra.utils.to_absolute_path(cfg.task.train_path),
        "val_dataset": hydra.utils.to_absolute_path(cfg.task.val_path),
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        **cfg.model,
    }

    if cfg.model_meta.name == "EncoderDecoderRNN":
        model = seq2seq.EncoderDecoderRNN(**args)
    elif cfg.model_meta.name == "AttentionRNN":
        model = seq2seq.AttentionRNN(**args)
    elif cfg.model_meta.name == "Seq2SeqTransformer":
        model = seq2seq.Transformer(epochs=cfg.trainer.max_epochs, **args)
    elif cfg.model_meta.name == "SiameseLSTM":
        model = equation_verification.SiameseLSTM(
            test_dataset=hydra.utils.to_absolute_path(cfg.task.test_path), **args
        )
    elif cfg.model_meta.name == "SiameseTransformer":
        model = equation_verification.SiameseTransformer(
            test_dataset=hydra.utils.to_absolute_path(cfg.task.test_path), **args
        )
    else:
        raise ValueError("Unrecognized model type:", cfg.model_meta.name)

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.getcwd(), name="", version=""),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(monitor="val/loss"),
        **cfg.trainer,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
