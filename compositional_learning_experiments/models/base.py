"""
Provides basic functionality for models on equation verification data.
"""

import abc
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch


class EquationVerificationModel(pl.LightningModule, abc.ABC):
    """
    An abstract base class for equation verification models.

    Parameters
    ----------
    does_batch
        True if this model uses batched computation.
        If so, tensors with batch axes are expected for `compute_test_metrics`.
    """

    data_format: str

    def __init__(self, does_batch: bool):
        super().__init__()
        self._does_batch = does_batch

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        val_metrics = {f"val/{k}": torch.tensor(v) for (k, v) in aggregated.items()}
        return {"log": val_metrics}

    @abc.abstractmethod
    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_fit_start(self):
        # See: https://github.com/PyTorchLightning/pytorch-lightning/issues/1228
        metrics_placeholder = {
            "metric/train_acc": 0,
            "metric/val_acc": 0,
            "metric/test_acc": 0,
        }

        self.logger.log_hyperparams(self.hparams, metrics=metrics_placeholder)

    def compute_test_metrics(self, prob_equal, target, left_embed, right_embed):
        """
        Compute a set of test metrics, which should be returned from test_step.
        """
        positive = target > 0
        negative = positive.bitwise_not()
        predicted_positive = prob_equal > 0.5
        predicted_negative = predicted_positive.bitwise_not()

        true_positive = positive.bitwise_and(predicted_positive)
        true_negative = negative.bitwise_and(predicted_negative)
        false_positive = negative.bitwise_and(predicted_positive)
        false_negative = positive.bitwise_and(predicted_negative)

        accuracy = (predicted_positive == target).float().mean()

        if self._does_batch:
            l2_dists = (left_embed - right_embed).norm(p=2, dim=1)
            left_magnitudes = left_embed.norm(p=2, dim=1)
            right_magnitudes = right_embed.norm(p=2, dim=1)
            dot_products = (left_embed * right_embed).sum(1)
        else:
            l2_dists = (left_embed - right_embed).norm(p=2, dim=0)
            left_magnitudes = left_embed.norm(p=2)
            right_magnitudes = right_embed.norm(p=2)
            dot_products = (left_embed * right_embed).sum()

        positive_l2_accum = (l2_dists * positive).sum()
        negative_l2_accum = (l2_dists * negative).sum()

        cosine_similarities = dot_products / (left_magnitudes * right_magnitudes)
        positive_cosine_accum = (cosine_similarities * positive).sum()
        negative_cosine_accum = (cosine_similarities * negative).sum()

        return {
            "accuracy": accuracy.cpu(),
            "true_positive": true_positive.sum().cpu(),
            "false_positive": false_positive.sum().cpu(),
            "true_negative": true_negative.sum().cpu(),
            "false_negative": false_negative.sum().cpu(),
            "positive_l2_accum": positive_l2_accum.cpu(),
            "negative_l2_accum": negative_l2_accum.cpu(),
            "positive_cosine_accum": positive_cosine_accum.cpu(),
            "negative_cosine_accum": negative_cosine_accum.cpu(),
        }

    def test_epoch_end(self, outputs):
        agg_key_funcs = {
            "true_positive": np.sum,
            "false_positive": np.sum,
            "true_negative": np.sum,
            "false_negative": np.sum,
            "positive_l2_accum": np.sum,
            "negative_l2_accum": np.sum,
            "positive_cosine_accum": np.sum,
            "negative_cosine_accum": np.sum,
        }
        aggregated = pl.loggers.base.merge_dicts(outputs, agg_key_funcs=agg_key_funcs)

        positive = aggregated["true_positive"] + aggregated["false_negative"]
        negative = aggregated["true_negative"] + aggregated["false_positive"]

        precision = aggregated["true_positive"] / (
            aggregated["true_positive"] + aggregated["false_positive"]
        )
        recall = aggregated["true_positive"] / positive

        # Mean L2 distance for all examples which are (or are not) matching
        mean_positive_l2 = aggregated["positive_l2_accum"] / positive
        mean_negative_l2 = aggregated["negative_l2_accum"] / negative

        # Similar with cosine similarity
        mean_positive_cosine = aggregated["positive_cosine_accum"] / positive
        mean_negative_cosine = aggregated["negative_cosine_accum"] / negative

        test_metrics = {
            "accuracy": aggregated["accuracy"],
            "precision": precision,
            "recall": recall,
            "mean_positive_l2": mean_positive_l2,
            "mean_negative_l2": mean_negative_l2,
            "mean_positive_cosine": mean_positive_cosine,
            "mean_negative_cosine": mean_negative_cosine,
        }

        return {k: torch.tensor(v) for (k, v) in test_metrics.items()}

    @abc.abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    @abc.abstractmethod
    def val_dataloader(self):
        raise NotImplementedError

    @abc.abstractmethod
    def make_dataloader(self, dataset, train: bool):
        pass


class SymmetricBilinearForm(torch.nn.Module):
    """
    A learnable symmetric bilinear similarity metric on an activation space.

    Parameters
    ----------
    dims : int
        The dimensionality of the input space.
    """

    def __init__(self, dims: int):
        super(SymmetricBilinearForm, self).__init__()
        self.dims = dims
        weight_init = torch.nn.init.xavier_normal_(torch.empty(dims, dims)).triu()
        self.weights = torch.nn.Parameter(weight_init)

    def forward(self, vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
        """
        Given two batches of vectors of the same shape, compute the batch of
        elementwise similarities.

        Parameters
        ----------
        vec_1 : torch.Tensor
            The first batch of vectors, with shape [batch_size, self.dims].
        vec_1 : torch.Tensor
            The seocnd batch of vectors, with shape [batch_size, self.dims].

        Returns
        -------
        torch.Tensor
            One similarity value per (paired) element in the batch.
        """
        # Upper-triangular is sufficient to parameterize a symmetric matrix
        upper_weights = self.weights.triu()
        # Avoid doubling variance of diagonal elements
        bilinear_weights = (upper_weights + upper_weights.triu(1).T).unsqueeze(0)
        return torch.nn.functional.bilinear(vec_1, vec_2, bilinear_weights).squeeze(1)


class DotProductSimilarity(torch.nn.Module):
    def __init__(self):
        super(DotProductSimilarity, self).__init__()

    def forward(self, vec_1, vec_2):
        return (vec_1 * vec_2).sum(1)


def make_similarity_metric(name: str, dims: Optional[int]):
    if name == "dot_product":
        return DotProductSimilarity()

    if name == "symmetric_bilinear":
        assert dims is not None
        return SymmetricBilinearForm(dims)

    raise ValueError("Unrecognized similarity metric")
