"""
Code to build and train tree models on equation verification data.
"""

from collections import Counter
from typing import List

import torch
import torchtext

from compositional_learning_experiments import models, data


class UnaryModule(torch.nn.Module):
    """
    An MLP applied to single tokens.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(d_model, d_model),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer_stack(inputs)
        normalized = output / output.norm(p=2)
        return normalized


class BinaryModule(torch.nn.Module):
    """
    Linearly combines two tokens and then applies an MLP to them.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(2 * d_model, 2 * d_model),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(2 * d_model, d_model),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([left, right], dim=-1)
        output = self.layer_stack(inputs)
        normalized = output / output.norm(p=2)
        return normalized


class TreeBase(models.base.EquationVerificationModel):
    """
    An abstract base class for tree-style equation verification models.
    Because these models compute on nonhomogenous trees, they do not support batching.

    NOTE: because these models rely on curriculum learning, they must be run for
    1 epoch and have the repeats of each depth set in the constructor.
    """

    data_format = "tree"

    def __init__(
        self,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
        batch_size: int = 1,  # Present for compatibility with the existing interface
    ):
        if batch_size != 1:
            raise ValueError("Tree models do not support batching")
        super().__init__(False)

        self.train_dataset = data.TreeDataset(train_path, train_depths)
        self.val_dataset = data.TreeDataset(val_path, val_depths)
        self.test_dataset = data.TreeDataset(test_path, test_depths)

        leaf_vocab = self.train_dataset.leaf_vocab.union(
            self.val_dataset.leaf_vocab.union(self.test_dataset.leaf_vocab)
        )
        unary_vocab = self.train_dataset.unary_vocab.union(
            self.val_dataset.unary_vocab.union(self.test_dataset.unary_vocab)
        )
        binary_vocab = self.train_dataset.binary_vocab.union(
            self.val_dataset.binary_vocab.union(self.test_dataset.binary_vocab)
        )

        self.leaf_vocab = torchtext.vocab.Vocab(Counter(leaf_vocab))
        self.unary_vocab = torchtext.vocab.Vocab(Counter(unary_vocab))
        self.binary_vocab = torchtext.vocab.Vocab(Counter(binary_vocab))

    def training_step(self, example, batch_idx):
        tree, label = example
        logit, _, _ = self(tree)
        target = label.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit.detach() > 0) == label).float().mean()

        return {
            "loss": loss,
            "log": {"train/loss": loss, "train/accuracy": accuracy},
        }

    def validation_step(self, example, batch_idx):
        tree, label = example
        logit, _, _ = self(tree)
        target = label.type_as(logit)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        accuracy = ((logit.detach() > 0) == label).float().mean()

        return {
            "loss": loss.cpu(),
            "accuracy": accuracy.cpu(),
        }

    def test_step(self, example, batch_idx):
        tree, target = example
        logit, left_embed, right_embed = self(tree)
        prob_equal = torch.nn.functional.sigmoid(logit)
        return self.compute_test_metrics(prob_equal, target, left_embed, right_embed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=None)

    def make_dataloader(self, dataset, train):
        return torch.utils.data.DataLoader(dataset, batch_size=None)


class TreeRNN(TreeBase):
    def __init__(
        self,
        learning_rate: float,
        d_model: int,
        dropout: float,
        similarity_metric: str,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
        batch_size: int = 1,  # Present for compatibility with the existing interface
    ):
        self.save_hyperparameters()
        super().__init__(
            train_path,
            train_depths,
            val_path,
            val_depths,
            test_path,
            test_depths,
            batch_size,
        )
        self.hparams.model_name = "TreeRNN"

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_model
        )
        self.leaf_embedding = torch.nn.Embedding(
            num_embeddings=len(self.leaf_vocab), embedding_dim=d_model, max_norm=1
        )
        self.unary_modules = torch.nn.ModuleDict(
            {name: UnaryModule(d_model, dropout) for name in self.unary_vocab.itos}
        )
        self.binary_modules = torch.nn.ModuleDict(
            {name: BinaryModule(d_model, dropout) for name in self.binary_vocab.itos}
        )

    def embed_tree(self, tree: data.ExpressionTree) -> torch.Tensor:
        if tree.left is None and tree.right is None:
            index = self.leaf_vocab.stoi[tree.label]
            return self.leaf_embedding(torch.tensor(index, device=self.device))

        if tree.left is None:
            right_rep = self.embed_tree(tree.right)
            module = self.unary_modules[tree.label]
            return module(right_rep)

        if tree.right is None:
            left_rep = self.embed_tree(tree.left)
            module = self.unary_modules[tree.label]
            return module(left_rep)

        left_rep = self.embed_tree(tree.left)
        right_rep = self.embed_tree(tree.right)
        module = self.binary_modules[tree.label]
        return module(left_rep, right_rep)

    def forward(self, tree):
        left_embed = self.embed_tree(tree.left)
        right_embed = self.embed_tree(tree.right)
        logit = self.similarity_metric(
            left_embed.unsqueeze(0), right_embed.unsqueeze(0)
        ).squeeze()
        return logit, left_embed, right_embed

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
