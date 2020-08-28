"""
Code to build and train tree models on equation verification data.
"""

from collections import Counter
from typing import List, Tuple

import torch
import torchtext

from compositional_learning_experiments import models, data


# Referenced: https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
class VectorQuantizeStraightThrough(torch.autograd.Function):
    """
    Computes the nearest vectors in a codebook to a set of embeddings. Returns 3 values:
        - Codebook entries, tracking gradients of the codebook and zeroing gradients to
          the embeddings.
        - Codebook entries, using a straight-through estimator of gradients to the
          embeddings and zeroing gradients to the codebook.
        - Indices of the codebook entries.

    Note that the first 2 are identical but have different gradient dynamics. The first
    updates the codebook & should be used for codebook loss, the second ignores the
    codebook but passes gradients backwards and should be used for other losses.
    """

    @staticmethod
    def forward(context, codebook, embeddings):
        with torch.no_grad():
            # Broadcast embeddings to shape [codebook_size, codebook_dims]
            deltas = embeddings - codebook
            distances = deltas.norm(p=2, dim=-1)  # Shape: [codebook_size]
            _, index = distances.kthvalue(1)  # Scalar

            quantized_codebook = codebook[index]  # Propagates gradients to codebook
            quantized_straight_through = quantized_codebook.clone()  # Does not

        context.save_for_backward(codebook, index)
        context.mark_non_differentiable(index)
        return quantized_codebook, quantized_straight_through, index

    @staticmethod
    def backward(
        context,
        grad_quantized_codebook,
        grad_quantized_straight_through,
        grad_index=None,
    ):
        # Codebook tracks gradients from grad_quantized_codebook
        if context.needs_input_grad[0]:
            codebook, index = context.saved_tensors
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook[index, :].add_(grad_quantized_codebook.clone())
        else:
            grad_codebook = None

        # Embeddings use straight-through gradients from grad_quantized_straight_through
        if context.needs_input_grad[1]:
            grad_embeddings = grad_quantized_straight_through.clone()
        else:
            grad_embeddings = None

        return grad_codebook, grad_embeddings


class VectorQuantize(torch.nn.Module):
    """
    A layer which performs vector quantization as in VQ-VAE.
    Read VectorQuantizeStraightThrough's docstring for a full description.
    """

    def __init__(self, codebook_size: int, codebook_dims: int):
        super().__init__()

        # Initialize codebook with unit norm
        codebook = torch.empty([codebook_size, codebook_dims])
        torch.nn.init.uniform_(codebook, -1, 1)
        codebook = codebook / codebook.norm(p=2, dim=-1, keepdim=True)
        self.codebook = torch.nn.Parameter(data=codebook, requires_grad=True)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return VectorQuantizeStraightThrough.apply(self.codebook, embeddings)  # type: ignore


class RoundingStraightThrough(torch.autograd.Function):
    """
    Rounds tensors to a given precision and backpropagates using the straight-through
    gradient estimator.
    """

    @staticmethod
    def forward(ctx, input, precision):
        return (input / precision).round() * precision

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output if ctx.needs_input_grad[0] else None
        grad_precision = None
        return grad_input, grad_precision


class RoundingLayer(torch.nn.Module):
    """
    A wrapper layer for RoundingStraightThrough.
    """

    def __init__(self, precision: float):
        super().__init__()
        self.precision = precision

    def forward(self, inputs: torch.Tensor):
        return RoundingStraightThrough.apply(inputs, self.precision)  # type: ignore


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
    """
    A simple TreeRNN-style recurisve neural network.
    """

    class UnaryModule(torch.nn.Module):
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
        self.hparams.model_name = "TreeRNN"
        super().__init__(
            train_path,
            train_depths,
            val_path,
            val_depths,
            test_path,
            test_depths,
            batch_size,
        )

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_model
        )
        self.leaf_embedding = torch.nn.Embedding(
            num_embeddings=len(self.leaf_vocab), embedding_dim=d_model, max_norm=1
        )
        self.unary_modules = torch.nn.ModuleDict(
            {name: self.UnaryModule(d_model, dropout) for name in self.unary_vocab.itos}
        )
        self.binary_modules = torch.nn.ModuleDict(
            {
                name: self.BinaryModule(d_model, dropout)
                for name in self.binary_vocab.itos
            }
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


class VectorQuantizedTreeRNN(TreeBase):
    """
    A TreeRNN that applies VQ-VAE-style quantization at each submodule.
    """

    class UnaryModule(torch.nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.layer_stack = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model),
                torch.nn.Tanh(),
                torch.nn.Linear(d_model, d_model),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.layer_stack(inputs)

    class BinaryModule(torch.nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.layer_stack = torch.nn.Sequential(
                torch.nn.Linear(2 * d_model, 2 * d_model),
                torch.nn.Tanh(),
                torch.nn.Linear(2 * d_model, d_model),
            )

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            inputs = torch.cat([left, right], dim=-1)
            return self.layer_stack(inputs)

    def __init__(
        self,
        codebook_size: int,
        codebook_loss_weight: float,
        commitment_loss_weight: float,
        d_model: int,
        similarity_metric: str,
        learning_rate: float,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
        batch_size: int = 1,  # Present for compatibility with the existing interface
    ):
        self.save_hyperparameters()
        self.hparams.model_name = "VectorQuantizedTreeRNN"
        super().__init__(
            train_path,
            train_depths,
            val_path,
            val_depths,
            test_path,
            test_depths,
            batch_size,
        )

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_model
        )
        self.quantize = VectorQuantize(codebook_size, d_model)
        self.leaf_embedding = torch.nn.Embedding(
            num_embeddings=len(self.leaf_vocab), embedding_dim=d_model
        )
        self.unary_modules = torch.nn.ModuleDict(
            {name: self.UnaryModule(d_model) for name in self.unary_vocab.itos}
        )
        self.binary_modules = torch.nn.ModuleDict(
            {name: self.BinaryModule(d_model) for name in self.binary_vocab.itos}
        )

    def embed_tree_with_losses(
        self, tree: data.ExpressionTree
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recursively computes the representation of a given subtree and the sum of
        all codebook and commitment losses under that subtree.

        Returns
        -------
        representation : torch.Tensor
            The representation of this subtree, extracted at the root.
            Will be one of the codebook vectors.
        codebook_loss : torch.Tensor
            A loss encouraging codebook vectors to be close to matching activations.
        commitment_loss : torch.Tensor
            A loss encouraging activations to stay close to their codebook vectors.
        """
        if tree.left is None and tree.right is None:
            embed_index = self.leaf_vocab.stoi[tree.label]
            activation = self.leaf_embedding(
                torch.tensor(embed_index, device=self.device)
            )
            child_codebook_loss = torch.tensor(0, device=self.device)
            child_commitment_loss = torch.tensor(0, device=self.device)
        elif tree.left is None:
            (
                right_rep,
                child_codebook_loss,
                child_commitment_loss,
            ) = self.embed_tree_with_losses(tree.right)
            module = self.unary_modules[tree.label]
            activation = module(right_rep)
        elif tree.right is None:
            (
                left_rep,
                child_codebook_loss,
                child_commitment_loss,
            ) = self.embed_tree_with_losses(tree.left)
            module = self.unary_modules[tree.label]
            activation = module(left_rep)
        else:
            (
                left_rep,
                left_codebook_loss,
                left_commitment_loss,
            ) = self.embed_tree_with_losses(tree.left)
            (
                right_rep,
                right_codebook_loss,
                right_commitment_loss,
            ) = self.embed_tree_with_losses(tree.right)

            module = self.binary_modules[tree.label]
            activation = module(left_rep, right_rep)
            child_codebook_loss = left_codebook_loss + right_codebook_loss
            child_commitment_loss = left_commitment_loss + right_commitment_loss

        (root_quantized_codebook, root_quantized_straight_through, _,) = self.quantize(
            activation
        )

        # Pull selected codebook vector closer to the activation
        root_codebook_loss = torch.nn.functional.mse_loss(
            input=root_quantized_codebook, target=activation.detach()
        )
        codebook_loss = root_codebook_loss + child_codebook_loss

        # Pull activation closer to the selected codebook vector
        root_commitment_loss = torch.nn.functional.mse_loss(
            input=activation, target=root_quantized_codebook.detach()
        )
        commitment_loss = root_commitment_loss + child_commitment_loss

        # Root rep. is the quantized activation, with straight-through gradient estimate
        return root_quantized_straight_through, codebook_loss, commitment_loss

    def forward_with_losses(
        self, tree: data.ExpressionTree
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            left_embed,
            left_codebook_loss,
            left_commitment_loss,
        ) = self.embed_tree_with_losses(tree.left)

        (
            right_embed,
            right_codebook_loss,
            right_commitment_loss,
        ) = self.embed_tree_with_losses(tree.right)

        logit = self.similarity_metric(
            left_embed.unsqueeze(0), right_embed.unsqueeze(0)
        ).squeeze()
        total_codebook_loss = left_codebook_loss + right_codebook_loss
        total_commitment_loss = left_commitment_loss + right_commitment_loss

        return (
            logit,
            left_embed,
            right_embed,
            total_codebook_loss,
            total_commitment_loss,
        )

    def training_step(self, example, batch_idx):
        tree, label = example
        logit, _, _, codebook_loss, commitment_loss = self.forward_with_losses(tree)
        target = label.type_as(logit)
        prediction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, target
        )
        accuracy = ((logit.detach() > 0) == label).float().mean()
        loss = (
            prediction_loss
            + self.hparams.codebook_loss_weight * codebook_loss
            + self.hparams.commitment_loss_weight * commitment_loss
        )

        return {
            "loss": loss,
            "log": {
                "train/loss": loss,
                "train/loss/prediction": prediction_loss,
                "train/loss/codebook": codebook_loss,
                "train/loss/commitment": commitment_loss,
                "train/accuracy": accuracy,
            },
        }

    def forward(self, tree):
        logit, left_embed, right_embed, _, _ = self.forward_with_losses(tree)
        return logit, left_embed, right_embed

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class RoundingTreeRNN(TreeBase):
    """
    A TreeRNN that rounds activations to a specified precision after each module.
    """

    class UnaryModule(torch.nn.Module):
        def __init__(self, d_model: int, d_inner: int, normalize: bool):
            super().__init__()
            self.layer_stack = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_inner),
                torch.nn.Tanh(),
                torch.nn.Linear(d_inner, d_model),
            )
            self.normalize = normalize

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output = self.layer_stack(inputs)
            if self.normalize:
                output = output / output.norm(p=2)
            return output

    class BinaryModule(torch.nn.Module):
        def __init__(self, d_model: int, d_inner: int, normalize: bool):
            super().__init__()
            self.layer_stack = torch.nn.Sequential(
                torch.nn.Linear(2 * d_model, d_inner),
                torch.nn.Tanh(),
                torch.nn.Linear(d_inner, d_model),
            )
            self.normalize = normalize

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            inputs = torch.cat([left, right], dim=-1)
            output = self.layer_stack(inputs)
            if self.normalize:
                output = output / output.norm(p=2)
            return output

    def __init__(
        self,
        precision: float,
        normalize: bool,
        l1_penalty: float,
        l2_penalty: float,
        d_model: int,
        d_inner: int,
        similarity_metric: str,
        learning_rate: float,
        train_path: str,
        train_depths: List[int],
        val_path: str,
        val_depths: List[int],
        test_path: str,
        test_depths: List[int],
        batch_size: int = 1,  # Present for compatibility with the existing interface
    ):
        self.save_hyperparameters()
        self.hparams.model_name = "RoundingTreeRNN"
        super().__init__(
            train_path,
            train_depths,
            val_path,
            val_depths,
            test_path,
            test_depths,
            batch_size,
        )

        self.similarity_metric = models.base.make_similarity_metric(
            similarity_metric, d_model
        )
        self.rounding = RoundingLayer(precision)
        self.leaf_embedding = torch.nn.Embedding(
            num_embeddings=len(self.leaf_vocab),
            embedding_dim=d_model,
            max_norm=1.0 if normalize else None,
        )
        self.unary_modules = torch.nn.ModuleDict(
            {
                name: self.UnaryModule(d_model, d_inner, normalize)
                for name in self.unary_vocab.itos
            }
        )
        self.binary_modules = torch.nn.ModuleDict(
            {
                name: self.BinaryModule(d_model, d_inner, normalize)
                for name in self.binary_vocab.itos
            }
        )

    def activation_penalties(
        self, activation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hparams.l1_penalty > 0:
            l1 = activation.norm(p=1)
        else:
            l1 = torch.tensor(0.0, device=self.device)

        if self.hparams.l2_penalty > 0:
            l2 = activation.norm(p=2)
        else:
            l2 = torch.tensor(0.0, device=self.device)

        return l1, l2

    def embed_tree_with_losses(
        self, tree: data.ExpressionTree
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recursively computes the representation of a given subtree and the sum of
        all L1 and L2 activation norms under that subtree.

        Returns
        -------
        representation : torch.Tensor
            The representation of this subtree, extracted at the root.
            Will be one of the codebook vectors.
        l1 : torch.Tensor
            The total accumulated L1 norm of activations under this subtree.
        l2 : torch.Tensor
            The total accumulated L2 norm of activations under this subtree.
        """
        if tree.left is None and tree.right is None:
            embed_index = self.leaf_vocab.stoi[tree.label]
            activation = self.leaf_embedding(
                torch.tensor(embed_index, device=self.device)
            )
            child_l1 = torch.tensor(0, device=self.device)
            child_l2 = torch.tensor(0, device=self.device)
        elif tree.left is None:
            (right_rep, child_l1, child_l2,) = self.embed_tree_with_losses(tree.right)
            module = self.unary_modules[tree.label]
            activation = module(right_rep)
        elif tree.right is None:
            (left_rep, child_l1, child_l2,) = self.embed_tree_with_losses(tree.left)
            module = self.unary_modules[tree.label]
            activation = module(left_rep)
        else:
            (left_rep, left_child_l1, left_child_l2,) = self.embed_tree_with_losses(
                tree.left
            )
            (right_rep, right_child_l1, right_child_l2,) = self.embed_tree_with_losses(
                tree.right
            )

            module = self.binary_modules[tree.label]
            activation = module(left_rep, right_rep)
            child_l1 = left_child_l1 + right_child_l1
            child_l2 = left_child_l2 + right_child_l2

        root_l1, root_l2 = self.activation_penalties(activation)
        total_l1 = child_l1 + root_l1
        total_l2 = child_l2 + root_l2
        root_quantized = self.rounding(activation)

        return root_quantized, total_l1, total_l2

    def forward_with_losses(
        self, tree: data.ExpressionTree
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (left_embed, left_l1, left_l2,) = self.embed_tree_with_losses(tree.left)
        (right_embed, right_l1, right_l2,) = self.embed_tree_with_losses(tree.right)

        logit = self.similarity_metric(
            left_embed.unsqueeze(0), right_embed.unsqueeze(0)
        ).squeeze()
        total_l1 = left_l1 + right_l1
        total_l2 = left_l2 + right_l2

        return (
            logit,
            left_embed,
            right_embed,
            total_l1,
            total_l2,
        )

    def training_step(self, example, batch_idx):
        tree, label = example
        logit, _, _, total_l1, total_l2 = self.forward_with_losses(tree)
        target = label.type_as(logit)
        prediction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, target
        )
        accuracy = ((logit.detach() > 0) == label).float().mean()

        l1_loss = self.hparams.l1_penalty * total_l1
        l2_loss = self.hparams.l2_penalty * total_l2

        loss = prediction_loss
        if self.hparams.l1_penalty > 0:
            loss = loss + l1_loss
        if self.hparams.l2_penalty > 0:
            loss = loss + l2_loss

        return {
            "loss": loss,
            "log": {
                "train/loss": loss,
                "train/loss/prediction": prediction_loss,
                "train/loss/l1": l1_loss,
                "train/loss/l2": l2_loss,
                "train/accuracy": accuracy,
                "train/total_l1": total_l1,
                "train/total_l2": total_l2,
            },
        }

    def forward(self, tree):
        logit, left_embed, right_embed, _, _ = self.forward_with_losses(tree)
        return logit, left_embed, right_embed

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
