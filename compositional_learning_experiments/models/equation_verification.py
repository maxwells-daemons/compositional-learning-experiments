"""
Code for models and training on the equation verification task.
"""

import abc
import itertools
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchtext

from compositional_learning_experiments.data import equation_verification
from compositional_learning_experiments.models import seq2seq


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


class SequenceBase(pl.LightningModule, abc.ABC):
    """
    An abstract base class for sequence-style equation verification models.
    """

    def __init__(
        self,
        task_name: str,
        batch_size: int,
        learning_rate: float,
        train_dataset: str = "recursiveMemNet/data/40k_train.json",
        val_dataset: str = "recursiveMemNet/data/40k_val_shallow.json",
        test_dataset: str = "recursiveMemNet/data/40k_test.json",
    ):
        super().__init__()

        train_ds = equation_verification.get_split_sequence(train_dataset)
        val_ds = equation_verification.get_split_sequence(val_dataset)
        test_ds_lengths = equation_verification.get_split_sequence_lengths(test_dataset)

        all_examples = itertools.chain.from_iterable(
            [train_ds.examples, val_ds.examples]
            + [ds.examples for ds in test_ds_lengths.values()]
        )

        def get_length(example):
            return max(len(example.left), len(example.right))

        # Include space for <init> and <eos> tokens
        self.max_input_length = max(map(get_length, all_examples)) + 2

        self.text_field = equation_verification.TEXT_FIELD
        self.target_field = equation_verification.TARGET_FIELD
        self.text_field.build_vocab(train_ds, val_ds, *test_ds_lengths.values())

        self.pad_i = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.init_i = self.text_field.vocab.stoi[self.text_field.init_token]
        self.eos_i = self.text_field.vocab.stoi[self.text_field.eos_token]

        def batch_sort_key(example):  # Sort batches to minimize worst-case padding
            return max(len(example.left), len(example.right))

        def make_iterator(ds, train):
            return torchtext.data.BucketIterator(
                ds,
                batch_size,
                device=self.device,
                train=train,
                shuffle=train,
                sort_key=batch_sort_key,
            )

        self.train_iter = make_iterator(train_ds, True)
        self.val_iter = make_iterator(val_ds, False)
        self.test_iter_lengths = {
            length: make_iterator(dataset, False)
            for length, dataset in test_ds_lengths.items()
        }

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        logit = self(batch.left, batch.right)
        target = batch.target.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit.detach() > 0) == batch.target).float().mean()

        return {"loss": loss, "log": {"train/loss": loss, "train/accuracy": accuracy}}

    # Validation
    def validation_step(self, batch, batch_idx):
        logit = self(batch.left, batch.right)
        target = batch.target.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit > 0) == batch.target).float().mean()

        return {"loss": loss.cpu(), "accuracy": accuracy.cpu()}

    def validation_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        val_metrics = {f"val/{k}": torch.tensor(v) for (k, v) in aggregated.items()}

        return {"log": val_metrics}

    # Testing
    def test_step(self, batch, batch_idx):
        logit = self(batch.left, batch.right)
        target = batch.target.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit > 0) == batch.target).float().mean()

        return {"loss": loss.cpu(), "accuracy": accuracy.cpu()}

    def test_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        test_metrics = {f"test/{k}": torch.tensor(v) for (k, v) in aggregated.items()}

        return {"log": test_metrics}

    # Data
    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter


def test(model: pl.LightningModule, trainer: pl.Trainer) -> plt.Figure:
    """
    Run a model on the test set, plotting the accuracy vs equation depth.
    """
    fig, ax = plt.subplots()

    lengths = []
    accuracies = []

    for length, dataset in model.test_iter_lengths.items():
        accuracy = trainer.test(model, dataset)["test/accuracy"]
        lengths.append(length)
        accuracies.append(accuracy)

    ax.scatter(lengths, accuracies)
    fig.suptitle("Test accuracies by depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Accuracy")

    return fig


# TODO: make work with non-transformer models
def test_checkpoint(checkpoint: str):
    model = SiameseTransformer.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(gpus=1)
    fig = test(model, trainer)
    fig.show()


class SiameseLSTM(SequenceBase):
    """
    Uses a shared RNN encoder to embed both sequences, then compares the representations
    with a symmetric bilinear model.

    Parameters
    ----------
    rnn_base : str
        "lstm", "gru", or "rnn".
    d_model : int
        The dimension of embedding and recurrent layers in this model.
    num_layers : int
        The number of stacked recurrent layers in the input and output RNNs.
    dropout : float
        How much dropout to apply in the recurrent layers.
    bidirectional : bool
        If true, the encoder is bidirectional and the decoder is twice as wide
        (to ensure the hidden dimensions match).
    """

    def __init__(
        self,
        task_name: str,
        batch_size: int,
        learning_rate: float,
        rnn_base: str,
        d_model: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        train_dataset: str = "recursiveMemNet/data/40k_train.json",
        val_dataset: str = "recursiveMemNet/data/40k_val_shallow.json",
        test_dataset: str = "recursiveMemNet/data/40k_test.json",
    ):
        super().__init__(
            task_name=task_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
        self.save_hyperparameters()
        self.hparams.model_name = "SiameseLSTM"

        d_output = 2 * d_model if bidirectional else d_model
        self.similarity_metric = SymmetricBilinearForm(d_output)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.text_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.pad_i,
        )

        rnn_base_constructor = seq2seq.get_rnn_constructor(rnn_base)
        self.encoder = rnn_base_constructor(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    # Training
    def embed_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Transform a variable-length sequence into a fixed-size embedding vector.
        """
        lengths = (sequence != self.pad_i).sum(0)
        input_embedding = self.dropout(self.embedding(sequence))
        input_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_embedding, lengths, enforce_sorted=False
        )

        hidden: torch.Tensor
        _, hidden = self.encoder(input_packed)  # type: ignore

        # Ignore LSTM cell state
        if isinstance(hidden, tuple):  # type: ignore
            hidden = hidden[0]

        if self.hparams.bidirectional:
            hidden = seq2seq.stack_bidirectional_context(hidden[-2:, :, :]).squeeze(0)
        else:
            hidden = hidden[-1, :, :]

        return hidden

    def forward(
        self, left_sequence: torch.Tensor, right_sequence: torch.Tensor
    ) -> torch.Tensor:
        left_embed = self.dropout(self.embed_sequence(left_sequence))
        right_embed = self.dropout(self.embed_sequence(right_sequence))
        logit = self.similarity_metric(left_embed, right_embed)
        return logit

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)


class SiameseTransformer(SequenceBase):
    """
    Use a shared Transformer encoder to encode both sequences, using the representation
    at the <init> token as the sequence representation, and compre with a symmetric
    bilinear model.
    """

    def __init__(
        self,
        task_name: str,
        batch_size: int,
        learning_rate: float,
        d_model: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        train_dataset: str = "recursiveMemNet/data/40k_train.json",
        val_dataset: str = "recursiveMemNet/data/40k_val_shallow.json",
        test_dataset: str = "recursiveMemNet/data/40k_test.json",
    ):
        super().__init__(
            task_name=task_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
        self.save_hyperparameters()
        self.hparams.model_name = "SiameseTransformer"

        self.similarity_metric = SymmetricBilinearForm(d_model)
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.text_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.pad_i,
        )

        self.positional_encoding = seq2seq.PositionalEncoding(
            self.max_input_length, d_model
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout
        )
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )

    def embed_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Transform a variable-length sequence into a fixed-size embedding vector.
        """
        pad_mask = (sequence == self.pad_i).T
        # TODO: restore
        # input_embedding = self.embedding(sequence)
        input_embedding = self.positional_encoding(self.embedding(sequence))
        encoded_sequence = self.encoder(input_embedding, src_key_padding_mask=pad_mask)
        sequence_rep = encoded_sequence[0, :, :]
        return sequence_rep

    def forward(
        self, left_sequence: torch.Tensor, right_sequence: torch.Tensor
    ) -> torch.Tensor:
        left_embed = self.embed_sequence(left_sequence)
        right_embed = self.embed_sequence(right_sequence)
        logit = self.similarity_metric(left_embed, right_embed)
        return logit

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# TODO: remove
if __name__ == "__main__":
    test_checkpoint("outputs/2020-07-29/11-24-57/checkpoints/epoch=20.ckpt")
