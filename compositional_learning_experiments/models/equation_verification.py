"""
Code for models and training on the equation verification task.
"""

import abc
import itertools
import math
from typing import List, Optional

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


class DotProductSimilarity(torch.nn.Module):
    def __init__(self):
        super(DotProductSimilarity, self).__init__()

    def forward(self, vec_1, vec_2):
        return (vec_1 * vec_2).sum(1)


# Sequence models
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
    def forward(self, batch: torchtext.data.Batch) -> torch.Tensor:
        """
        Given two batches of vectors of the same shape, compute the batch of
        elementwise similarities.

        Parameters
        ----------
        batch : torchtext.data.Batch
            A batch of sequence-format input data.

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
        logit = self(batch)
        target = batch.target.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit.detach() > 0) == batch.target).float().mean()

        return {"loss": loss, "log": {"train/loss": loss, "train/accuracy": accuracy}}

    # Validation
    def validation_step(self, batch, batch_idx):
        logit = self(batch)
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
        logit = self(batch)
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
    similarity_metric : str
        Which similarity metric to use. One of ["dot_product", "symmetric_bilinear"].
    """

    similarity_metric: torch.nn.Module

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
        similarity_metric: str,
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

        if similarity_metric == "dot_product":
            self.similarity_metric = DotProductSimilarity()
        elif similarity_metric == "symmetric_bilinear":
            self.similarity_metric = SymmetricBilinearForm(d_output)
        else:
            raise ValueError("Unrecognized similarity metric")

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

    def forward(self, batch: torchtext.data.Batch) -> torch.Tensor:
        left_embed = self.dropout(self.embed_sequence(batch.left))
        right_embed = self.dropout(self.embed_sequence(batch.right))
        logit = self.similarity_metric(left_embed, right_embed)
        return logit

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)


class SiameseTransformer(SequenceBase):
    """
    Use a shared Transformer encoder to encode both sequences, using the representation
    at the <init> token as the sequence representation, and compre with a symmetric
    bilinear model.

    Parameters
    ----------
    d_model : int
        The dimension of embedding and transformer layers in this model.
    nhead : int
        Number of heads in each self-attention step.
    dropout : float
        How much dropout to apply in the embedding and transformer layers.
    num_layers : int
        The number of stacked recurrent layers in the input and output RNNs.
    similarity_metric : str
        Which similarity metric to use. One of ["dot_product", "symmetric_bilinear"].
    root_representation : bool
        If true, the representation of the sequence is taken from the root token.
        Otherwise, it is taken from the <init> token.
    """

    similarity_metric: torch.nn.Module

    def __init__(
        self,
        task_name: str,
        batch_size: int,
        learning_rate: float,
        d_model: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        similarity_metric: str,
        root_representation: bool,
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

        if similarity_metric == "dot_product":
            self.similarity_metric = DotProductSimilarity()
        elif similarity_metric == "symmetric_bilinear":
            self.similarity_metric = SymmetricBilinearForm(d_model)
        else:
            raise ValueError("Unrecognized similarity metric")

        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.text_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.pad_i,
        )

        self.dropout = torch.nn.Dropout(p=dropout)
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

    def embed_token(self, sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence)
        scaled = embedded * math.sqrt(self.hparams.d_model)
        dropped = self.dropout(scaled)
        return self.positional_encoding(dropped)

    def embed_sequence(
        self, sequence: torch.Tensor, root_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform a variable-length sequence into a fixed-size embedding vector.

        Parameters
        ----------
        sequence : torch.Tensor
            A sequence of token indices, of shape [sequence_length, batch].
        root_index : torch.Tensor
            The index in each sequence where the root token is, of shape [batch].

        Returns
        -------
        torch.Tensor
            A fixed-size representation of each sequence, of size [batch, d_model].
        """
        pad_mask = (sequence == self.pad_i).T
        input_embedding = self.embed_token(sequence)
        encoded_sequence = self.encoder(input_embedding, src_key_padding_mask=pad_mask)

        if self.hparams.root_representation:
            # Sequence representation index varies by batch element (root_index)
            gather_index = (
                root_index.unsqueeze(1).repeat(1, self.hparams.d_model).unsqueeze(0)
            )
            return encoded_sequence.gather(0, gather_index).squeeze(0)

        # Sequence representation is the starting <index> token
        return encoded_sequence[0, :, :]

    def forward(self, batch: torchtext.data.Batch) -> torch.Tensor:
        left_embed = self.embed_sequence(batch.left, batch.left_root_index)
        right_embed = self.embed_sequence(batch.right, batch.right_root_index)
        logit = self.similarity_metric(left_embed, right_embed)
        return logit

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def plot_attentions(self, tokens: List[str], log_step: int) -> None:
        string = "".join(tokens)
        tokens_tensor = self.text_field.numericalize([tokens], device=self.device)
        embeddings = self.positional_encoding(self.embedding(tokens_tensor))

        for i, layer in enumerate(self.encoder.layers):
            attentions = layer.self_attn(embeddings, embeddings, embeddings)[1][0]
            figure = seq2seq.plot_attention(
                tokens, tokens, attentions.detach().cpu().numpy()
            )

            name = f"{string}/layer_{i}"
            self.logger.experiment.add_figure(name, figure, log_step)

            embeddings = layer(embeddings)

    def validation_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        val_metrics = {f"val/{k}": torch.tensor(v) for (k, v) in aggregated.items()}

        for token_set in equation_verification.SEQUENCE_TEST_STRINGS:
            self.plot_attentions(token_set, self.current_epoch)

        return {"log": val_metrics}


# Tree models
class TreeBase(pl.LightningModule, abc.ABC):
    """
    An abstract base class for tree-style equation verification models.
    Because these models compute on nonhomogenous trees, they do not support batching.

    NOTE: because these models rely on curriculum learning, they must be run for
    1 epoch and have the repeats of each depth set in the constructor.
    """

    def __init__(
        self,
        task_name: str,
        learning_rate: float,
        epochs: int,
        train_dataset: str = "recursiveMemNet/data/40k_train.json",
        val_dataset: str = "recursiveMemNet/data/40k_val_shallow.json",
        test_dataset: str = "recursiveMemNet/data/40k_test.json",
        batch_size: int = 1,  # Present for compatibility with the existing interface
    ):
        super().__init__()

        if batch_size != 1:
            raise ValueError("Tree models do not support batching")

        self.train_dataset = equation_verification.TreeCurriculum(
            train_dataset, repeats=epochs
        )
        self.val_dataset = equation_verification.TreeCurriculum(val_dataset)

    @abc.abstractmethod
    def forward(self, tree: equation_verification.ExpressionTree) -> torch.Tensor:
        """
        Given an expression tree rooted at equality and with extant left and right
        subtrees, return the logit that the two subtrees evaluate to the same quantity.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, example, batch_idx):
        tree, label = example
        target = label.type_as(logit)
        logit = self(tree)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit.detach() > 0) == label).float().mean()

        return {"loss": loss, "log": {"train/loss": loss, "train/accuracy": accuracy}}

    # Validation
    def validation_step(self, example, batch_idx):
        tree, label = example
        target = label.type_as(logit)
        logit = self(tree)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit > 0) == label).float().mean()

        return {"loss": loss.cpu(), "accuracy": accuracy.cpu()}

    def validation_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        val_metrics = {f"val/{k}": torch.tensor(v) for (k, v) in aggregated.items()}

        return {"log": val_metrics}

    # Data
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=None)
