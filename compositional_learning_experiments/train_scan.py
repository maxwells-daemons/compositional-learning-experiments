"""
Code to run experiments using the SCAN dataset.
"""

import abc
import os
import logging
import math
from typing import List, Optional, Tuple

import hydra
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


logger = logging.getLogger(__name__)

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
    # TODO: handle multiple model types
    model = SCANTransformer(
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
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(os.getcwd(), name="", version=""),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(monitor="val/loss"),
        **cfg.trainer,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
