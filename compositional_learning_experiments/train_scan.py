"""
Code to run experiments using the SCAN dataset.
"""

import abc
import os
import math
from typing import List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn
import torch.utils.tensorboard
import torchtext

from compositional_learning_experiments import parse_scan

# Derived from the SCAN grammar; see Figure 6/7 of https://arxiv.org/pdf/1711.00350.pdf.
MAX_INPUT_LENGTH = 9
MAX_OUTPUT_LENGTH = 50  # NOTE: includes <eos> and <init> tokens


def tokenize(string: str) -> List[str]:
    """
    Tokenize a string of SCAN text.

    Parameters
    ----------
    string : str
        A line in either the SCAN input or output grammar.

    Returns
    -------
    List[str]
        A list of tokens in the string, including surrounding spaces for reversibility.
    """
    return list(map(lambda s: f" {s} ", str.split(string)))


FIELDS = {
    "input": torchtext.data.ReversibleField(
        sequential=True,
        fix_length=MAX_INPUT_LENGTH,
        tokenize=tokenize,
        pad_token=" <pad> ",
        unk_token=" <unk> ",
    ),
    "target": torchtext.data.ReversibleField(
        sequential=True,
        is_target=True,
        pad_token=" <pad> ",
        unk_token=" <unk> ",
        init_token=" <init> ",
        eos_token=" <eos> ",
        fix_length=MAX_OUTPUT_LENGTH,
        tokenize=tokenize,
    ),
}

# Closed over by make_example, but precomputed for efficiency
_FIELDS_ASSOCIATION_LIST = list(FIELDS.items())

TEST_COMMANDS = [
    "turn left",
    "turn right",
    "jump",
    "walk",
    "look",
    "run",
    "walk left",
    "run right",
    "turn right twice",
    "turn left thrice",
    "jump left twice",
    "look right thrice",
    "turn opposite left",
    "walk opposite right",
    "look around left",
    "jump around right",
    "walk and run",
    "look after jump",
    "walk left and run right",
    "look twice after jump thrice",
    "jump opposite left after walk around left",
    "walk around right thrice after jump opposite left twice",
]


def make_example(line: str) -> torchtext.data.Example:
    """
    Parse a line of text in the SCAN data format into a torchtext Example.

    Parameters
    ----------
    line : str
        A line of text of the format "IN: S1 OUT: S2", where S1 and S2 are strings
        in the SCAN input / output grammars respectively.

    Returns
    -------
    torchtext.data.Example
        A torchtext Example object containing "input" and "target" fields.
    """
    split = parse_scan.split_example(line)
    return torchtext.data.Example.fromlist(split, _FIELDS_ASSOCIATION_LIST)


def scan_dataset(file: str) -> torchtext.data.Dataset:
    """
    Create a (non-split) dataset from a file in the SCAN data format.

    Parameters
    ----------
    file : str
        Filename of the data to load.

    Returns
    -------
    torchtext.data.Dataset
        A torchtext Dataset containing Examples with "input" and "target" fields.
    """
    with open(file, "r") as f:
        lines = f.readlines()

    examples = map(make_example, lines)
    dataset = torchtext.data.Dataset(list(examples), FIELDS)

    return dataset


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
    dropout : float
        Dropout to apply to the input.
    """

    def __init__(self, d_model: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(MAX_OUTPUT_LENGTH, d_model)
        position = torch.arange(0, MAX_OUTPUT_LENGTH).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SCANBase(pl.LightningModule, abc.ABC):
    """
    An abstract base class for models that will be trained on SCAN.

    All sequences are assumed to use [SEQUENCE_LENGTH, BATCH_SIZE, DIM] convention.
    This class handles data loading and training/validation conventions.

    Parameters
    ----------
    train_dataset : str
        A path to the dataset to train on.
    val_dataset : str
        A path to the dataset to validate on.
    batch_size : int
        Batch size to use for training and validation.

    Attributes
    ----------
    input_field : torchtext.data.Field
        A Field containing vocabulary and parsing information for input text.
    output_field : torchtext.data.Field
        A Field containing vocabulary and parsing information for output text.
    input_pad_i : int
        Token index of the <pad> character in input text.
    target_pad_i : int
        Token index of the <pad> character in output text.
    target_init_i : int
        Token index of the <init> character in output text.
    target_eos_i : int
        Token index of the <eos> character in output text.
    train_iter : torchtext.dataset.Iterator
        An Iterator looping over the training set.
    val_iter : torchtext.dataset.Iterator
        An Iterator looping over the validation set.
    """

    input_field: torchtext.data.Field
    target_field: torchtext.data.Field
    input_pad_i: int
    target_pad_i: int
    target_init_i: int
    target_eos_i: int
    train_iter: torchtext.data.Iterator
    val_iter: torchtext.data.Iterator

    def __init__(
        self, train_dataset: str, val_dataset: str, batch_size: int,
    ):
        super().__init__()

        train_dataset = scan_dataset(train_dataset)
        val_dataset = scan_dataset(val_dataset)

        self.input_field = FIELDS["input"]
        self.target_field = FIELDS["target"]
        self.input_field.build_vocab(train_dataset, val_dataset)
        self.target_field.build_vocab(train_dataset, val_dataset)

        self.input_pad_i = self.input_field.vocab.stoi[self.input_field.pad_token]
        self.target_pad_i = self.target_field.vocab.stoi[self.target_field.pad_token]
        self.target_init_i = self.target_field.vocab.stoi[self.target_field.init_token]
        self.target_eos_i = self.target_field.vocab.stoi[self.target_field.eos_token]

        self.train_iter = torchtext.data.Iterator(
            train_dataset, batch_size, device=self.device, train=True
        )
        self.val_iter = torchtext.data.Iterator(
            val_dataset, batch_size, device=self.device, train=False, sort=False,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.target_pad_i)

    @abc.abstractmethod
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Predict logits for an encoder-decoder teacher-forcing training setup.

        The subclass must apply masking during decoding so that output tokens
        cannot see themselves or future tokens.

        Parameters
        ----------
        src : torch.Tensor
            A batch of input sequences, given as word indices.
            Shaped as [SEQUENCE_LENGTH, BATCH_SIZE].
        tgt : torch.Tensor
            A batch of target sequences, given as word indices and without the final
            token (<eos> or <pad>). Shaped as [SEQUENCE_LENGTH, BATCH_SIZE].

        Returns
        -------
        torch.Tensor
            For each place in each sequence, a set of logits for the *next* token
            in the sequence (i.e. beginning with the first word following <init> and
            ideally ending with <eos>). Shaped as [SEQUENCE_LENGTH, BATCH_SIZE, DIM].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer_greedy(self, src: str) -> str:
        """
        Infer an output sequence from an input string using greedy decoding.
        NOTE: does not handle <pad> tokens in the input!

        Parameters
        ----------
        src : str
            An input string in the SCAN input grammar.

        Returns
        -------
        str
            The predicted output in the SCAN output grammar.
        """
        raise NotImplementedError

    def compute_batch_accuracy(
        self, predicted_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the token and sequence accuracy for a batch of predictions and targets.

        Parameters
        ----------
        predicted_tokens : torch.Tensor
            A batch of predicted integer word indices with shape
            (sequence_length, batch_size).
        target_tokens : torch.Tensor
            A batch of target integer word indices (usually all tokens but <init>,
            including <eos>). Should have shape (sequence_length, batch_size).

        Returns
        -------
        float
            The proportion of correct tokens in the output.
        float
            The proportion of sequences produced totally correctly.
        """
        with torch.no_grad():
            batch_size = target_tokens.size(1)
            unpadded_places = (target_tokens == self.target_pad_i).bitwise_not()
            wrong_places = (target_tokens != predicted_tokens) & unpadded_places

            n_wrong_tokens = wrong_places.sum()
            token_accuracy = 1.0 - torch.true_divide(
                n_wrong_tokens, unpadded_places.sum()
            )

            n_wrong_sequences = wrong_places.any(0).sum()
            sequence_accuracy = 1.0 - torch.true_divide(n_wrong_sequences, batch_size)

        return token_accuracy, sequence_accuracy

    # Training
    def training_step(self, batch, batch_idx):
        tgt_input = batch.target[:-1]
        tgt_goal = batch.target[1:]
        tgt_goal_flat = tgt_goal.view(-1)

        predicted_logits = self(batch.input, tgt_input)
        pred_flat = predicted_logits.view(tgt_goal_flat.size(0), -1)
        loss = self.loss_fn(pred_flat, tgt_goal_flat)

        predicted_tokens = predicted_logits.detach().argmax(-1)
        token_accuracy, sequence_accuracy = self.compute_batch_accuracy(
            predicted_tokens, tgt_goal
        )

        return {
            "loss": loss,
            "log": {
                "train/loss": loss,
                "train/token_accuracy": token_accuracy,
                "train/sequence_accuracy": sequence_accuracy,
            },
        }

    def validation_step(self, batch, batch_idx):
        tgt_input = batch.target[:-1]
        tgt_goal = batch.target[1:]
        tgt_goal_flat = tgt_goal.view(-1)

        predicted_logits = self(batch.input, tgt_input)
        pred_flat = predicted_logits.view(tgt_goal_flat.size(0), -1)
        loss = self.loss_fn(pred_flat, tgt_goal_flat)

        predicted_tokens = predicted_logits.detach().argmax(-1)
        token_accuracy, sequence_accuracy = self.compute_batch_accuracy(
            predicted_tokens, tgt_goal
        )

        return {
            "loss": loss.cpu(),
            "token_accuracy": token_accuracy.cpu(),
            "sequence_accuracy": sequence_accuracy.cpu(),
        }

    def validation_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        val_metrics = {f"val/{k}": torch.tensor(v) for (k, v) in aggregated.items()}

        for command in TEST_COMMANDS:
            output = self.infer_greedy(command)
            self.logger.experiment.add_text(command, output, self.current_epoch)

        return {"log": val_metrics}

    # Data
    def setup(self, stage):
        # Adding dummy values now makes live values show in the "metrics" UI later
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/loss": 0.0,
                "train/token_accuracy": 0.0,
                "train/sequence_accuracy": 0.0,
                "val/loss": 0.0,
                "val/token_accuracy": 0.0,
                "val/sequence_accuracy": 0.0,
            },
        )

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter


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


def get_rnn_constructor(variant: str):
    return {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU,}[variant]


class EncoderDecoderRNN(SCANBase):
    """
    An encoder-decoder ("seq2seq") RNN without attention.

    Parameters
    ----------
    rnn_base : torch.nn.RNNBase constructor
        A constructor of a class conforming to the interface of torch.nn.RNN.
    d_model : int
        The dimension of embedding and recurrent layers in this model.
    num_layers : int
        The number of stacked recurrent layers in the input and output RNNs.
    dropout : float
        How much dropout to apply in the recurrent layers.
    bidirectional_encoder : bool
        If true, the encoder is bidirectional and the decoder is twice as wide
        (to ensure the hidden dimensions match).
    """

    def __init__(
        self,
        train_dataset: str,
        val_dataset: str,
        batch_size: int,
        learning_rate: float,
        rnn_base: str,
        d_model: int,
        num_layers: int,
        dropout: float,
        bidirectional_encoder: bool,
    ):
        super().__init__(train_dataset, val_dataset, batch_size)
        self.save_hyperparameters()
        self.hparams.model_name = "EncoderDecoderRNN"

        self.decoder_width = (2 * d_model) if bidirectional_encoder else d_model
        rnn_base_constructor = get_rnn_constructor(rnn_base)
        self.encoder = rnn_base_constructor(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
        )
        self.decoder = rnn_base_constructor(
            input_size=d_model,
            hidden_size=self.decoder_width,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.input_pipeline = torch.nn.Sequential(
            torch.nn.Embedding(
                num_embeddings=len(self.input_field.vocab),
                embedding_dim=d_model,
                padding_idx=self.input_pad_i,
            ),
            self.dropout,
        )
        self.target_pipeline = torch.nn.Sequential(
            torch.nn.Embedding(
                num_embeddings=len(self.target_field.vocab),
                embedding_dim=d_model,
                padding_idx=self.target_pad_i,
            ),
            self.dropout,
        )
        self.output = torch.nn.Linear(self.decoder_width, len(self.target_field.vocab))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        input_enc = self.input_pipeline(src)
        target_enc = self.target_pipeline(tgt)
        src_lengths = src.size(0) - (src == self.input_pad_i).sum(0)
        tgt_lengths = tgt.size(0) - (tgt == self.target_pad_i).sum(0)
        src_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_enc, src_lengths, enforce_sorted=False
        )
        tgt_packed = torch.nn.utils.rnn.pack_padded_sequence(
            target_enc, tgt_lengths, enforce_sorted=False
        )

        _, context = self.encoder(src_packed)
        if self.hparams.bidirectional_encoder:
            context = stack_bidirectional_context(context)

        packed_output, _ = self.decoder(tgt_packed, context)
        # NOTE: this makes a prediction for every point in the sequence, including
        # padding values. These must be ignored later.
        predicted_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            total_length=MAX_OUTPUT_LENGTH - 1,  # Does not predict <init> token
        )
        predicted_logits = self.output(predicted_embeddings)
        return predicted_logits

    def infer_greedy(self, src: str) -> str:
        src_tokens = self.input_field.preprocess(src)
        src_tensor = self.input_field.numericalize([src_tokens], device=self.device)

        generated_tokens = torch.tensor(
            self.target_init_i, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            input_enc = self.input_pipeline(src_tensor)
            _, context = self.encoder(input_enc)
            if self.hparams.bidirectional_encoder:
                context = stack_bidirectional_context(context)

            # Manually unroll decoder with sequence/batch dimensions of 1
            for _ in range(MAX_OUTPUT_LENGTH):
                target_enc = self.target_pipeline(generated_tokens[-1])
                predicted_embeddings, context = self.decoder(
                    target_enc.unsqueeze(0).unsqueeze(0), context
                )
                predicted_logits = self.output(predicted_embeddings)
                new_token = predicted_logits[0, 0, :].argmax()
                generated_tokens = torch.cat([generated_tokens, new_token.unsqueeze(0)])

                if new_token == self.target_eos_i:
                    break

        return self.target_field.reverse(generated_tokens.unsqueeze(1))[0]

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)


class BahdanauAttention(torch.nn.Module):
    """
    Implement additive attention (Bahdanau et al. 2015).

    Parameters
    ----------
    query_size : int
        The size of vectors that will be turned into queries.
    key_size : int
        The size of vectors that will be turned into keys.
    attention_dim : int
        The number of internal dimensions to use for the score computation.
    """

    def __init__(self, query_size: int, key_size: int, attention_dim: int):
        super().__init__()
        self.project_query = torch.nn.Linear(query_size, attention_dim)
        self.project_key = torch.nn.Linear(key_size, attention_dim)
        self.compute_score = torch.nn.Linear(attention_dim, 1)

    def forward(
        self,
        query: Optional[torch.Tensor] = None,
        query_input: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
        key_inputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the attention scores between a batch of queries and a batch of key
        sequences.

        Parameters
        ----------
        query : Optional[torch.Tensor] (default: None)
            A precomputed batch of query vectors, of shape [batch_size, attention_dim].
            Must be defined iff query_input is not.
        query_input : Optional[torch.Tensor] (default: None)
            A batch of inputs to be turned into query vectors, of shape
            [batch_size, query_size]. Must be defined iff query is not.
        keys : Optional[torch.Tensor] (default: None)
            A precomputed batch of sequences of key vectors, of shape
            [sequence_length, batch_size, attention_dim].
            Must be defined iff key_input is not.
        key_inputs : Optional[torch.Tensor] (default: None)
            A batch of sequences of inputs to be turned into key vectors, of shape
            [sequence_length, batch_size, key_size]. Must be defined iff keys is not.
        mask : Optional[torch.Tensor] (default: None)
            A tensor of shape [sequence_length, batch_size] which is 0 for positions
            which may be attented to and -inf for positions which may not.

        Returns
        -------
        torch.Tensor
            An attention score between each key in the sequence and the query.
            Has shape [sequence_length, batch_size].
        """
        if query_input is None:
            assert query is not None
        else:
            assert query is None
            query = self.project_query(query_input)

        if key_inputs is None:
            assert keys is not None
        else:
            assert keys is None
            keys = self.project_key(key_inputs)

        # NOTE: addition is broadcast; there is a single query and a sequence of keys
        scores = self.compute_score(torch.tanh(query + keys)).squeeze(2)

        if mask is not None:
            scores += mask

        return scores.softmax(0)


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


class AttentionRNN(SCANBase):
    """
    An encoder-decoder ("seq2seq") RNN with attention.

    Parameters
    ----------
    rnn_base : torch.nn.RNNBase constructor
        A constructor of a class conforming to the interface of torch.nn.RNN.
    d_model : int
        The dimension of embedding and recurrent layers in this model.
    num_layers : int
        The number of stacked recurrent layers in the input and output RNNs.
    dropout : float
        How much dropout to apply in the recurrent layers.
    bidirectional_encoder : bool
        If true, the encoder is bidirectional and the decoder is twice as wide
        (to ensure the hidden dimensions match).
    """

    def __init__(
        self,
        train_dataset: str,
        val_dataset: str,
        batch_size: int,
        learning_rate: float,
        rnn_base: str,
        d_model: int,
        num_layers: int,
        dropout: float,
        bidirectional_encoder: bool,
        attention_dim: int,
    ):
        super().__init__(train_dataset, val_dataset, batch_size)
        self.save_hyperparameters()
        self.hparams.model_name = "AttentionRNN"

        self.decoder_width = (2 * d_model) if bidirectional_encoder else d_model
        self.decoder_input_size = d_model + self.decoder_width
        rnn_base_constructor = get_rnn_constructor(rnn_base)
        self.encoder = rnn_base_constructor(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
        )
        self.decoder = rnn_base_constructor(
            input_size=self.decoder_input_size,
            hidden_size=self.decoder_width,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.attention = BahdanauAttention(
            query_size=self.decoder_width,
            key_size=self.decoder_width,
            attention_dim=attention_dim,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.input_pipeline = torch.nn.Sequential(
            torch.nn.Embedding(
                num_embeddings=len(self.input_field.vocab),
                embedding_dim=d_model,
                padding_idx=self.input_pad_i,
            ),
            self.dropout,
        )
        self.target_pipeline = torch.nn.Sequential(
            torch.nn.Embedding(
                num_embeddings=len(self.target_field.vocab),
                embedding_dim=d_model,
                padding_idx=self.target_pad_i,
            ),
            self.dropout,
        )
        self.output = torch.nn.Linear(self.decoder_width, len(self.target_field.vocab))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        input_enc = self.input_pipeline(src)
        target_enc = self.target_pipeline(tgt)
        src_lengths = src.size(0) - (src == self.input_pad_i).sum(0)
        src_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input_enc, src_lengths, enforce_sorted=False
        )

        memory, hidden = self.encoder(src_packed)
        memory, _ = torch.nn.utils.rnn.pad_packed_sequence(
            memory, total_length=MAX_INPUT_LENGTH, padding_value=0.0
        )
        memory_mask = torch.zeros(src.shape, device=self.device)
        memory_mask[torch.where(src == self.input_pad_i)] = -math.inf
        keys = self.attention.project_key(memory)
        if self.hparams.bidirectional_encoder:
            hidden = stack_bidirectional_context(hidden)

        outputs = []
        for i in range(MAX_OUTPUT_LENGTH - 1):
            # NOTE: we only use the last layer's hidden state to query
            if isinstance(hidden, tuple):
                query_input = hidden[0][-1]
            else:
                query_input = hidden[-1]
            attention_scores = self.attention(
                query_input=query_input, keys=keys, mask=memory_mask,
            )
            # [batch_size, decoder_width]
            context = (attention_scores.unsqueeze(2) * memory).sum(0)
            inputs = torch.cat([target_enc[i], context], dim=1)
            predicted_embeddings, hidden = self.decoder(inputs.unsqueeze(0), hidden)
            outputs.append(self.output(predicted_embeddings).squeeze(0))

        return torch.stack(outputs)

    def infer_greedy(self, src: str) -> str:
        generated_tensor, _ = self.infer_and_attend_greedy(src)
        generated_str = self.target_field.reverse(generated_tensor.unsqueeze(1))[0]
        return generated_str

    def infer_and_attend_greedy(self, src: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform greedy inference, yielding a prediction string and a per-step
        attention map.

        Parameters
        ----------
        src : str
            An input string in the SCAN input grammar.

        Returns
        -------
        torch.Tensor
            The generated tokens.
        torch.Tensor
            For each step of inference (i.e. not including <init>), the attention
            distribution over input tokens. Has shape [src_length, tgt_length - 1].
        """
        src_tokens = self.input_field.preprocess(src)
        src_tensor = self.input_field.numericalize([src_tokens], device=self.device)

        generated_tokens = [torch.tensor(self.target_init_i, device=self.device)]
        step_attentions = []

        with torch.no_grad():
            input_enc = self.input_pipeline(src_tensor)
            memory, hidden = self.encoder(input_enc)
            keys = self.attention.project_key(memory)
            if self.hparams.bidirectional_encoder:
                hidden = stack_bidirectional_context(hidden)

            for _ in range(MAX_OUTPUT_LENGTH - 1):
                target_enc = self.target_pipeline(generated_tokens[-1]).unsqueeze(0)
                attention_scores = self.attention(query_input=hidden[-1], keys=keys)
                context = (attention_scores.unsqueeze(2) * memory).sum(0)
                inputs = torch.cat([target_enc, context], dim=1)
                predicted_embeddings, hidden = self.decoder(inputs.unsqueeze(0), hidden)
                predicted_logits = self.output(predicted_embeddings)
                new_token = predicted_logits[0, 0, :].argmax()

                generated_tokens.append(new_token)
                step_attentions.append(attention_scores[:, 0])

                if new_token == self.target_eos_i:
                    break

        generated_tensor = torch.stack(generated_tokens)
        attention_map = torch.stack(step_attentions)

        return generated_tensor, attention_map

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # On validation, also plot test command attention maps
    def validation_epoch_end(self, outputs):
        aggregated = pl.loggers.base.merge_dicts(outputs)
        val_metrics = {f"val/{k}": torch.tensor(v) for (k, v) in aggregated.items()}

        for command in TEST_COMMANDS:
            output_tensor, attention_map = self.infer_and_attend_greedy(command)
            src_tokens = self.input_field.preprocess(command)
            output = self.target_field.reverse(output_tensor.unsqueeze(1))[0]
            # Remove <init> token
            output_tokens = [self.target_field.vocab.itos[x] for x in output_tensor[1:]]
            fig = plot_attention(src_tokens, output_tokens, attention_map.cpu().numpy())

            self.logger.experiment.add_text(command, output, self.current_epoch)
            self.logger.experiment.add_figure(command, fig, self.current_epoch)
            plt.clf()

        return {"log": val_metrics}


class Transformer(SCANBase):
    """
    A typical sequence-to-sequence transformer for the SCAN task.
    Model parameters and forward() are as they are in torch.nn.Transformer.
    """

    def __init__(
        self,
        train_dataset: str,
        val_dataset: str,
        batch_size: int,
        learning_rate: float,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
    ):
        super().__init__(train_dataset, val_dataset, batch_size)
        self.save_hyperparameters()
        self.hparams.model_name = "Transformer"

        self.scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = torch.nn.Transformer(  # type: ignore
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.input_embedding = torch.nn.Embedding(
            num_embeddings=len(self.input_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.input_pad_i,
        )
        self.target_embedding = torch.nn.Embedding(
            num_embeddings=len(self.target_field.vocab),
            embedding_dim=d_model,
            padding_idx=self.target_pad_i,
        )
        self.output = torch.nn.Linear(d_model, len(self.target_field.vocab))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        input_pad_mask = src.T == self.input_pad_i
        target_pad_mask = tgt.T == self.target_pad_i

        input_enc = self.positional_encoding(self.input_embedding(src) * self.scale)
        target_enc = self.positional_encoding(self.target_embedding(tgt) * self.scale)
        future_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(0)
        ).type_as(input_enc)

        predicted_embeddings = self.transformer(
            input_enc,
            target_enc,
            tgt_mask=future_mask,
            src_key_padding_mask=input_pad_mask,
            tgt_key_padding_mask=target_pad_mask,
            memory_key_padding_mask=input_pad_mask,
        )
        predicted_logits = self.output(predicted_embeddings)
        return predicted_logits

    def infer_greedy(self, src: str) -> str:
        src_tokens = self.input_field.preprocess(src)
        src_tensor = self.input_field.numericalize([src_tokens], device=self.device)

        generated_tokens = torch.tensor(
            self.target_init_i, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            input_enc = self.positional_encoding(
                self.input_embedding(src_tensor) * self.scale
            )
            memory = self.transformer.encoder(input_enc)

            while (
                generated_tokens[-1] != self.target_eos_i
                and generated_tokens.numel() < MAX_OUTPUT_LENGTH
            ):
                future_mask = self.transformer.generate_square_subsequent_mask(
                    generated_tokens.numel()
                ).type_as(input_enc)
                target_enc = self.positional_encoding(
                    self.target_embedding(generated_tokens.unsqueeze(1)) * self.scale
                )
                predicted_embeddings = self.transformer.decoder(
                    target_enc, memory, tgt_mask=future_mask
                )
                predicted_logits = self.output(predicted_embeddings)
                new_token = predicted_logits[-1, 0, :].argmax()
                generated_tokens = torch.cat([generated_tokens, new_token.unsqueeze(0)])

        return self.target_field.reverse(generated_tokens.unsqueeze(1))[0]

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )


@hydra.main(config_path="config/config.yaml", strict=False)
def main(cfg: omegaconf.DictConfig):
    if cfg.model.name == "Transformer":
        model = Transformer(
            train_dataset=hydra.utils.to_absolute_path(cfg.split.train),
            val_dataset=hydra.utils.to_absolute_path(cfg.split.val),
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.name == "EncoderDecoderRNN":
        model = EncoderDecoderRNN(
            train_dataset=hydra.utils.to_absolute_path(cfg.split.train),
            val_dataset=hydra.utils.to_absolute_path(cfg.split.val),
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            rnn_base=cfg.model.rnn_base,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional_encoder=cfg.model.bidirectional_encoder,
        )
    elif cfg.model.name == "AttentionRNN":
        model = AttentionRNN(
            train_dataset=hydra.utils.to_absolute_path(cfg.split.train),
            val_dataset=hydra.utils.to_absolute_path(cfg.split.val),
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            rnn_base=cfg.model.rnn_base,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional_encoder=cfg.model.bidirectional_encoder,
            attention_dim=cfg.model.attention_dim,
        )
    else:
        raise ValueError("Unrecognized model type")

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.getcwd(), name="", version=""),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(monitor="val/loss"),
        **cfg.trainer,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
