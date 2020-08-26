"""
Code to build and train sequence models on equation verification data.
"""

import itertools
import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchtext

from compositional_learning_experiments import models, data


def get_rnn_constructor(variant: str):
    return {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU,}[variant]


def stack_bidirectional_context(context: torch.Tensor) -> torch.Tensor:
    """
    Concatenate the forward and backward representations from a bidirectional RNN
    on the hidden dimension axis.
    Parameters
    ----------
    context : torch.Tensor
        A tensor with shape [layers * 2, batch_size, dims].
        May also be a 2-tuple of such tensors (as in an LSTM).
    Returns
    -------
    torch.Tensor
        context reshaped to have shape [layers, batch_size, dims * 2].
    """
    if isinstance(context, tuple):  # LSTM; see below for detailed view
        hidden, cell = context
        num_layers, batch_size, d_model = hidden.shape
        num_layers //= 2
        hidden = hidden.view([num_layers, 2, batch_size, d_model])
        hidden = hidden.permute([0, 2, 1, 3])
        hidden = hidden.reshape([num_layers, batch_size, 2 * d_model])
        cell = cell.view([num_layers, 2, batch_size, d_model])
        cell = cell.permute([0, 2, 1, 3])
        cell = cell.reshape([num_layers, batch_size, 2 * d_model])
        return (hidden, cell)
    else:  # RNN & GRU
        num_layers, batch_size, d_model = context.shape
        num_layers //= 2
        # [layers * directions, batch, dims] -> [layers, directions, batch, dims]
        context = context.view([num_layers, 2, batch_size, d_model])
        # [layers, directions, batch, dims] -> [layers, batch, directions, dims]
        context = context.permute([0, 2, 1, 3])
        # [layers, batch, directions, dims] -> [layers, batch, directions * dims]
        return context.reshape([num_layers, batch_size, 2 * d_model])


def plot_attention(
    src_string: List[str], generated_string: List[str], attention_map: np.ndarray
) -> plt.Figure:
    """
    Create a pyplot figure visualizing the attention scores at each step of decoding.
    Adapted from: https://bastings.github.io/annotated_encoder_decoder/.
    Parameters
    ----------
    src_string : List[str]
        The input string, as a list of tokens.
    generated_string : List[str]
        The generated tokens, including the <eos> token but not the <init> token.
    Returns
    -------
    plt.Figure
        A figure displaying the attention map.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    heatmap = ax.pcolor(attention_map.T, cmap="viridis")

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(attention_map.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(attention_map.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(generated_string, minor=False, rotation="vertical")
    ax.set_yticklabels(src_string, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    plt.tight_layout()
    return fig


class PositionalEncoding(torch.nn.Module):
    """
    Applies sinusoidal positional encoding to an input tensor.
    Adapted from:
    https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    Parameters
    ----------
    d_model : int
        The last dimension of the input tensor.
        This will also be the dimension of each place's positional encoding.
    """

    def __init__(self, max_length: int, d_model: int):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return x


class TreePositionalEncoding(torch.nn.Module):
    def __init__(self, max_depth: int, d_model: int):
        super(TreePositionalEncoding, self).__init__()
        assert d_model % ((max_depth + 1) * 2) == 0
        self.repeats = d_model // ((max_depth + 1) * 2)
        self.d_model = d_model

    def forward(self, sequence, tree_encoding):
        # TODO: add learnable parameters
        scaled = tree_encoding.type_as(sequence) * math.sqrt(self.d_model)
        repeated = scaled.repeat([1, 1, self.repeats])
        return sequence + repeated


class SequenceBase(models.base.EquationVerificationModel):
    """
    An abstract base class for sequence-style equation verification models.
    """

    def __init__(
        self,
        batch_size: int,
        train_dataset: torchtext.data.Dataset,
        val_dataset: torchtext.data.Dataset,
        test_dataset: torchtext.data.Dataset,
    ):
        super().__init__(True)

        # Include space for <init> and <eos> tokens
        all_examples = itertools.chain.from_iterable(
            [train_dataset.examples, val_dataset.examples, test_dataset.examples]
        )
        self.max_input_length = max(map(SequenceBase.example_length, all_examples)) + 2

        self.text_field = data._TEXT_FIELD
        self.target_field = data._TARGET_FIELD
        self.text_field.build_vocab(train_dataset, val_dataset, test_dataset)

        self.pad_i = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.init_i = self.text_field.vocab.stoi[self.text_field.init_token]
        self.eos_i = self.text_field.vocab.stoi[self.text_field.eos_token]

        self.train_iter = self.make_dataloader(train_dataset, True)
        self.val_iter = self.make_dataloader(val_dataset, False)

    @staticmethod
    def example_length(example):
        return max(len(example.left), len(example.right))

    def make_dataloader(self, dataset, train):
        return torchtext.data.BucketIterator(
            dataset,
            self.hparams.batch_size,
            device=self.device,
            train=train,
            shuffle=train,
            sort_key=SequenceBase.example_length,
        )

    def training_step(self, batch, batch_idx):
        logit, _, _ = self(batch)
        target = batch.target.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit.detach() > 0) == batch.target).float().mean()

        return {"loss": loss, "log": {"train/loss": loss, "train/accuracy": accuracy}}

    def validation_step(self, batch, batch_idx):
        logit, _, _ = self(batch)
        target = batch.target.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit > 0) == batch.target).float().mean()

        return {"loss": loss.cpu(), "accuracy": accuracy.cpu()}

    def test_step(self, batch, batch_idx):
        logit, left_embed, right_embed = self(batch)
        prob_equal = torch.nn.functional.sigmoid(logit)
        target = batch.target.type_as(logit)
        return self.compute_test_metrics(prob_equal, target, left_embed, right_embed)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter


class ParenthesesBase(SequenceBase):
    data_format = "parentheses"

    def __init__(
        self,
        batch_size: int,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
    ):
        train_dataset = data.load_parentheses_dataset(train_path, train_depths)
        val_dataset = data.load_parentheses_dataset(val_path, val_depths)
        test_dataset = data.load_parentheses_dataset(test_path, test_depths)
        super().__init__(batch_size, train_dataset, val_dataset, test_dataset)


class SiameseLSTM(ParenthesesBase):
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
        rnn_base: str,
        d_model: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        similarity_metric: str,
        learning_rate: float,
        batch_size: int,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
    ):
        self.save_hyperparameters()
        super().__init__(
            batch_size=batch_size,
            train_path=train_path,
            train_depths=train_depths,
            val_path=val_path,
            val_depths=val_depths,
            test_path=test_path,
            test_depths=test_depths,
        )
        self.hparams.model_name = "SiameseLSTM"

        d_output = 2 * d_model if bidirectional else d_model

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_output
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.text_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.pad_i,
        )
        rnn_base_constructor = get_rnn_constructor(rnn_base)
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
            hidden = stack_bidirectional_context(hidden[-2:, :, :]).squeeze(0)
        else:
            hidden = hidden[-1, :, :]

        return hidden

    def forward(self, batch):
        left_embed = self.dropout(self.embed_sequence(batch.left))
        right_embed = self.dropout(self.embed_sequence(batch.right))
        logit = self.similarity_metric(left_embed, right_embed)
        return logit, left_embed, right_embed

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)


class SiameseTransformer(ParenthesesBase):
    """
    Use a shared Transformer encoder to encode both sequences, using the representation
    at the <init> token or the tree root as the sequence representation, and compare
    with a symmetric bilinear model.

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

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        similarity_metric: str,
        root_representation: bool,
        learning_rate: float,
        batch_size: int,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
    ):
        self.save_hyperparameters()
        super().__init__(
            batch_size=batch_size,
            train_path=train_path,
            train_depths=train_depths,
            val_path=val_path,
            val_depths=val_depths,
            test_path=test_path,
            test_depths=test_depths,
        )
        self.hparams.model_name = "SiameseTransformer"

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_model
        )
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.text_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.pad_i,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout
        )
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )
        self.positional_encoding = PositionalEncoding(self.max_input_length, d_model)

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

    def forward(self, batch):
        left_embed = self.embed_sequence(batch.left, batch.left_root_index)
        right_embed = self.embed_sequence(batch.right, batch.right_root_index)
        logit = self.similarity_metric(left_embed, right_embed)
        return logit, left_embed, right_embed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def plot_attentions(self, tokens: List[str], log_step: int) -> None:
        string = "".join(tokens)
        tokens_tensor = self.text_field.numericalize([tokens], device=self.device)
        embeddings = self.positional_encoding(self.embedding(tokens_tensor))

        for i, layer in enumerate(self.encoder.layers):
            attentions = layer.self_attn(embeddings, embeddings, embeddings)[1][0]
            figure = plot_attention(tokens, tokens, attentions.detach().cpu().numpy())

            name = f"{string}/layer_{i}"
            self.logger.experiment.add_figure(name, figure, log_step)

            embeddings = layer(embeddings)


class TreeTransformer(SequenceBase):
    """
    Use a shared Transformer encoder to encode both sequences, using the representation
    at the <init> token as the sequence representation, and compare with a symmetric
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

    data_format = "positional_encoding"

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        similarity_metric: str,
        learning_rate: float,
        batch_size: int,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
    ):
        self.save_hyperparameters()
        train_dataset = data.load_positional_encoding_dataset(train_path, train_depths)
        val_dataset = data.load_positional_encoding_dataset(val_path, val_depths)
        test_dataset = data.load_positional_encoding_dataset(test_path, test_depths)
        super().__init__(batch_size, train_dataset, val_dataset, test_dataset)

        self.hparams.model_name = "SiameseTransformer"

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_model
        )
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.text_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.pad_i,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout
        )
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )
        self.positional_encoding = TreePositionalEncoding(
            max_depth=max(train_depths + val_depths + test_depths), d_model=d_model
        )

    def embed_token(
        self, sequence: torch.Tensor, tree_encoding: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(sequence)
        scaled = embedded * math.sqrt(self.hparams.d_model)
        dropped = self.dropout(scaled)
        return self.positional_encoding(dropped, tree_encoding)

    def embed_sequence(
        self, sequence: torch.Tensor, positional_encoding: torch.Tensor
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
        input_embedding = self.embed_token(sequence, positional_encoding)
        encoded_sequence = self.encoder(input_embedding, src_key_padding_mask=pad_mask)

        # Sequence representation is the root of the expression
        return encoded_sequence[1, :, :]

    def forward(self, batch):
        left_embed = self.embed_sequence(batch.left, batch.left_positional_encoding)
        right_embed = self.embed_sequence(batch.right, batch.right_positional_encoding)
        logit = self.similarity_metric(left_embed, right_embed)
        return logit, left_embed, right_embed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def plot_attentions(
        self, tokens: List[str], tree_encoding: torch.Tensor, log_step: int
    ) -> None:
        string = "".join(tokens)
        tokens_tensor = self.text_field.numericalize([tokens], device=self.device)
        embeddings = self.positional_encoding(
            self.embedding(tokens_tensor), tree_encoding.unsqueeze(1)
        )

        for i, layer in enumerate(self.encoder.layers):
            attentions = layer.self_attn(embeddings, embeddings, embeddings)[1][0]
            figure = plot_attention(tokens, tokens, attentions.detach().cpu().numpy())

            name = f"{string}/layer_{i}"
            self.logger.experiment.add_figure(name, figure, log_step)

            embeddings = layer(embeddings)
