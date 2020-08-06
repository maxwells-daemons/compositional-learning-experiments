"""
Code to work with the equation verification dataset from [Arabshahi et al. 2020].

The dataset can be loaded in two formats:
 - Sequence: mathematical expressions are provided as a textual sequence with
   parentheses indicating structure (e.g. "(2 * y) + (x * 3)").
 - Tree: mathematical expressions are provided as a binary tree's pre-order traversal,
   with tensors for vertex labels, parent indices, child indices, and depth.

In both cases, each example is provided as (left expression, right expression, label).
The task is to predict whether the two expressions are equivalent.
"""

import itertools
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import graphviz
import torch
import torchtext

TRAIN_DEPTHS = [1, 2, 3, 4, 5, 6, 7]
TEST_DEPTHS = [8, 9, 10, 11, 12, 13, 14, 15]


def to_token(func: str, var: str) -> str:
    """
    Convert a vertex in a serialized equation tree into an unambiguous token.
    """
    if var:
        return var
    return func


class ExpressionTree:
    """
    A binary tree encoding a mathematical expression.

    Parameters
    ----------
    label : str
        The label for the root vertex.
    left : Optional[ExpressionTree] (default: None)
        If applicable, the left child of this expression.
    right : Optional[ExpressionTree] (default: None)
        If applicable, the right child of this expression.
    index : int (default: 0)
        The index of this tree in the pre-order traversal of a larger expression tree.
    """

    label: str
    left: Optional["ExpressionTree"]
    right: Optional["ExpressionTree"]
    index: int

    def __init__(
        self,
        label: str,
        left: Optional["ExpressionTree"] = None,
        right: Optional["ExpressionTree"] = None,
        index: int = 0,
    ):
        self.label = label
        self.left = left
        self.right = right
        self.index = index

    def __repr__(self):
        return (
            f"ExpressionTree(label='{self.label}', "
            f"index={self.index}, "
            f"left={repr(self.left)}, "
            f"right={repr(self.right)})"
        )

    def __str__(self):
        if not (self.left or self.right):  # Leaf node
            return self.label
        if not self.left:  # Unary, right-populated
            return f"{self.label} ({str(self.right)})"
        if not self.right:  # Unary, left-populated
            return f"{self.label} ({str(self.left)})"
        return f"({str(self.left)}) {self.label} ({str(self.right)})"

    def to_prefix(self, blank_token: bool = False) -> str:
        """
        Get a string representation of this tree as a prefix-style ("root left right")
        pre-order traversal of its labels, without parentheses.
        """
        string = self.label
        if self.left:
            string += " " + self.left.to_prefix()
        else:
            if blank_token:
                string += " BLANK"
        if self.right:
            string += " " + self.right.to_prefix()
        else:
            if blank_token:
                string += " BLANK"
        return string

    def positional_encoding(self, max_depth: int = 15) -> torch.Tensor:
        """
        Make a tree-structured positional encoding for this tree, which aligns
        with its pre-order prefix string representation, including the <init>
        and <eos> tokens.

        Encoding is the one from [Shiv 2019]:
        "Novel positional encodings to enable tree-based transformers".
        The output will be of shape [sequence_length + 2, 2 * max_depth];
        the extra 2 elements are for the <init> and <eos> tokens.
        """
        root_encoding = torch.zeros([2 * max_depth], dtype=torch.bool)
        inner_encoding = self._subtree_positional_encoding(root_encoding, max_depth)

        # Add 2 sequence elements (first and last) and 2 places (last two) for one-hot
        # encoding the <init> and <eos> tokens
        full_encoding = torch.zeros(
            [inner_encoding.size(0) + 2, inner_encoding.size(1) + 2], dtype=torch.bool
        )
        full_encoding[1:-1, :-2] = inner_encoding
        full_encoding[0, -2] = 1  # <init>: [0, 0, ..., 0, 0, 1, 0]
        full_encoding[-1, -1] = 1  # <eos>: [0, 0, ..., 0, 0, 0, 1]

        return full_encoding

    def _subtree_positional_encoding(
        self, root_encoding: torch.Tensor, max_depth: int
    ) -> torch.Tensor:
        """
        Construct the positional encoding for the subtree under this vertex.
        """
        tree_encoding = root_encoding.unsqueeze(0)

        if self.left:
            left_encoding = root_encoding.roll(2)
            left_encoding[:2] = torch.tensor([1, 0])
            left_subtree = self.left._subtree_positional_encoding(
                left_encoding, max_depth
            )
            tree_encoding = torch.cat([tree_encoding, left_subtree], 0)

        if self.right:
            right_encoding = root_encoding.roll(2)
            right_encoding[:2] = torch.tensor([0, 1])
            right_subtree = self.right._subtree_positional_encoding(
                right_encoding, max_depth
            )
            tree_encoding = torch.cat([tree_encoding, right_subtree], 0)

        return tree_encoding

    def plot_graph(self) -> graphviz.Digraph:
        """
        Plot this equation tree as a graphviz graph.
        """
        graph = graphviz.Digraph()
        graph.graph_attr["label"] = str(self)
        graph.graph_attr["labelloc"] = "t"
        graph.graph_attr["rankdir"] = "BT"
        self._plot_subtree(graph)
        return graph

    def _plot_subtree(self, graph: graphviz.Digraph) -> None:
        """
        Recursively plot the subtree under this vertex.
        An internal helper method for plot_graph().
        """
        graph.node(str(self.index), self.label)
        if self.left:
            self.left._plot_subtree(graph)
            graph.edge(str(self.left.index), str(self.index))
        if self.right:
            self.right._plot_subtree(graph)
            graph.edge(str(self.right.index), str(self.index))

    @staticmethod
    def from_serialized(serialized: Dict[str, str]) -> "ExpressionTree":
        """
        Create an expression tree from the serialization format used in recursiveMemNet.

        Parameters
        ----------
        serialized
            The serialized equation, parsed from JSON.

        Returns
        -------
        ExpressionTree
            The tree containing the mathematical expression, including an equality
            at the root.
        """
        funcs_rev = serialized["func"].split(",")
        vars_rev = serialized["vars"].split(",")
        funcs_rev.reverse()
        vars_rev.reverse()

        tree, _ = ExpressionTree._from_partial_serialized(funcs_rev, vars_rev, -1)
        assert tree
        return tree

    @staticmethod
    def _from_partial_serialized(
        funcs_rev: List[str], vars_rev: List[str], last_index: int
    ) -> Tuple[Optional["ExpressionTree"], int]:
        """
        Perform one (recursive) step in parsing a serialized expression tree.

        An internal helper method for from_serialized().

        Parameters
        ----------
        funcs_rev : List[str]
            A reversed list of "func" symbols remaining to be parsed. Modified inplace.
        vars_rev : List[str]
            A reversed list of "vars" symbols remaining to be parsed. Modified inplace.

        Returns
        -------
        Optional[ExpressionTree]
            As much of a subtree as can be parsed from funcs_rev and vars_rev,
            stopping at an empty list or a leaf node.
        """
        assert len(funcs_rev) == len(vars_rev)
        if not vars_rev:  # No more symbols to parse
            return None, -1

        root_label = to_token(funcs_rev.pop(), vars_rev.pop())
        if root_label == "#":
            return None, -1

        root_index = last_index + 1
        left, left_index = ExpressionTree._from_partial_serialized(
            funcs_rev, vars_rev, root_index
        )
        right, right_index = ExpressionTree._from_partial_serialized(
            funcs_rev, vars_rev, left_index
        )
        new_index = max(root_index, left_index, right_index)
        return ExpressionTree(root_label, left, right, root_index), new_index

    def leaf_vocab(self) -> Set[str]:
        """
        Get the set of leaf tokens under this subtree.
        """
        if self.left is None:
            if self.right is None:
                return {self.label}

            return self.right.leaf_vocab()

        if self.right is None:
            return self.left.leaf_vocab()

        return self.left.leaf_vocab().union(self.right.leaf_vocab())

    def unary_vocab(self) -> Set[str]:
        """
        Get the set of of unary functions under this subtree.
        """
        if self.left is None:
            if self.right is None:
                return set()

            return {self.label}.union(self.right.unary_vocab())

        if self.right is None:
            return {self.label}.union(self.left.unary_vocab())

        return self.left.unary_vocab().union(self.right.unary_vocab())

    def binary_vocab(self) -> Set[str]:
        """
        Get the set of of binary functions under this subtree.
        """
        if self.left is None:
            if self.right is None:
                return set()

            return self.right.binary_vocab()

        if self.right is None:
            return self.left.binary_vocab()

        return (
            {self.label}
            .union(self.left.binary_vocab())
            .union(self.right.binary_vocab())
        )


TARGET_FIELD = torchtext.data.Field(
    sequential=False, use_vocab=False, dtype=torch.int32, is_target=True
)

# Code to produce sequence datasets
def tokenize(equation_string: str) -> List[str]:
    """
    Tokenize an equation string, including spaces around each token for reversibility.
    """
    tokens = equation_string.replace("(", " ( ").replace(")", " ) ").strip().split()
    return [" " + token + " " for token in tokens]


TEXT_FIELD = torchtext.data.ReversibleField(
    sequential=True,
    tokenize=tokenize,
    pad_token=" <pad> ",
    unk_token=" <unk> ",
    init_token=" <init> ",
    eos_token=" <eos> ",
)
INDEX_FIELD = torchtext.data.Field(
    sequential=False,
    use_vocab=False,
    dtype=torch.long,
    is_target=False,
    batch_first=True,
)


class StackField(torchtext.data.RawField):
    def __init__(self):
        super().__init__()

    def process(self, batch, *args, **kwargs):
        pad_length = max([x.size(0) for x in batch])

        def pad(tensor):
            length = tensor.size(0)
            n_pad = pad_length - length

            # NOTE: we pad with the root encoding, (which seems like a huge problem),
            # but these positions will be masked by the transformer.
            return torch.nn.functional.pad(tensor, [0, 0, 0, n_pad], value=0)

        return torch.stack([pad(x) for x in batch], 1)


POSITIONAL_ENCODING_FIELD = StackField()

SEQUENCE_FIELDS = {
    "left": TEXT_FIELD,
    "right": TEXT_FIELD,
    "target": TARGET_FIELD,
    "left_root_index": INDEX_FIELD,
    "right_root_index": INDEX_FIELD,
}
_SEQUENCE_ASSOC_LIST = list(SEQUENCE_FIELDS.items())

TREE_TRANSFORMER_FIELDS = {
    "left": TEXT_FIELD,
    "right": TEXT_FIELD,
    "target": TARGET_FIELD,
    "left_positional_encoding": POSITIONAL_ENCODING_FIELD,
    "right_positional_encoding": POSITIONAL_ENCODING_FIELD,
}
_TREE_TRANSFORMER_ASSOC_LIST = list(TREE_TRANSFORMER_FIELDS.items())


def sequence_root_index(example: ExpressionTree) -> int:
    """
    Get the index of the root token in a sequence expression.
    """
    if example.left is None or example.right is None:
        return 1  # Just the <init> token

    return len(tokenize(str(example.left))) + 3  # <init> token and () around left


def make_sequence_example(
    example: Dict[str, Any], prefix: bool = False
) -> torchtext.data.Example:
    """
    Make a Sequence-style example from a serialized equation verification example.
    If `prefix`, use a parenthesis-free prefix representation with positional encoding.
    """
    tree = ExpressionTree.from_serialized(example["equation"])
    assert tree.left is not None
    assert tree.right is not None
    label = int(example["label"] == "1")

    if prefix:
        left = tree.left.to_prefix()
        right = tree.right.to_prefix()
        left_positional_encoding = tree.left.positional_encoding()
        right_positional_encoding = tree.right.positional_encoding()

        return torchtext.data.Example.fromlist(
            [left, right, label, left_positional_encoding, right_positional_encoding],
            _TREE_TRANSFORMER_ASSOC_LIST,
        )

    left = str(tree.left)
    right = str(tree.right)

    left_root_index = sequence_root_index(tree.left)
    right_root_index = sequence_root_index(tree.right)
    assert tokenize(tree.left.label)[0] == tokenize(left)[left_root_index - 1]
    assert tokenize(tree.right.label)[0] == tokenize(right)[right_root_index - 1]

    return torchtext.data.Example.fromlist(
        [left, right, label, left_root_index, right_root_index], _SEQUENCE_ASSOC_LIST
    )


def get_split_sequence(file: str, prefix: bool = False) -> torchtext.data.Dataset:
    """
    Make a Sequence-style dataset from a file of equation verification data.
    """
    with open(file, "r") as f:
        raw_data = json.load(f)

    data_flat = itertools.chain.from_iterable(raw_data)
    examples = [make_sequence_example(ex, prefix) for ex in data_flat]

    fields = TREE_TRANSFORMER_FIELDS if prefix else SEQUENCE_FIELDS
    return torchtext.data.Dataset(examples, fields)


def get_split_sequence_lengths(
    file: str, prefix: bool = False
) -> Dict[int, torchtext.data.Dataset]:
    """
    Make a collection of sequence-style datasets, indexed by depth, from a data file.
    """
    with open(file, "r") as f:
        raw_data = json.load(f)

    datasets_and_lengths = {}
    for length, data in enumerate(raw_data):
        if not data:
            continue

        examples = [make_sequence_example(ex, prefix) for ex in data]
        dataset = torchtext.data.Dataset(examples, SEQUENCE_FIELDS)
        datasets_and_lengths[length + 1] = dataset

    return datasets_and_lengths


# Code to produce tree datasets
class TreeCurriculum(torch.utils.data.IterableDataset):  # type: ignore
    """
    A dataset yielding single tree examples in a fixed curriculum of increasing depth.
    Note that these examples cannot be batched in their current form.

    Parameters
    ----------
    file : str
        File to load trees from.
    repeats : int
        How many times to repeat each stage of the curriculum.
        Equivalent to epoch count.
    """

    def __init__(self, file: str, repeats: int = 1):
        super(TreeCurriculum, self).__init__()

        with open(file, "r") as f:
            raw_data = json.load(f)

        self.leaf_vocab: Set[str] = set()
        self.unary_vocab: Set[str] = set()
        self.binary_vocab: Set[str] = set()

        self.data = []
        self.data_by_depth = {}
        for depth, split in enumerate(raw_data):
            if not split:
                continue

            deserialized = []
            for serialized in split:
                tree = ExpressionTree.from_serialized(serialized["equation"])
                label = torch.tensor(serialized["label"] == "1")
                example = (tree, label)
                deserialized.append(example)

                self.leaf_vocab = self.leaf_vocab.union(tree.leaf_vocab())
                self.binary_vocab = self.binary_vocab.union(tree.binary_vocab())
                self.unary_vocab = self.unary_vocab.union(tree.unary_vocab())

            self.data.extend(deserialized * repeats)
            self.data_by_depth[depth] = deserialized

    def __iter__(self):
        return iter(self.data)


class TreesOfDepth(torch.utils.data.IterableDataset):  # type: ignore
    def __init__(self, file: str, depth: int):
        super(TreesOfDepth, self).__init__()

        with open(file, "r") as f:
            raw_data = json.load(f)

        self.data = []
        for serialized in raw_data[depth - 1]:
            tree = ExpressionTree.from_serialized(serialized["equation"])
            label = torch.tensor(serialized["label"] == "1")
            example = (tree, label)
            self.data.append(example)

    def __iter__(self):
        return iter(self.data)


# Expression trees to periodically visualize behavior on
TEST_TREES = [
    ExpressionTree(label="2", index=1, left=None, right=None),
    ExpressionTree(
        label="Pow",
        index=1,
        left=ExpressionTree(label="var_0", index=2, left=None, right=None),
        right=ExpressionTree(label="1/2", index=3, left=None, right=None),
    ),
    ExpressionTree(
        label="Mul",
        index=1,
        left=ExpressionTree(
            label="Pow",
            index=2,
            left=ExpressionTree(label="1", index=3, left=None, right=None),
            right=ExpressionTree(label="1/2", index=4, left=None, right=None),
        ),
        right=ExpressionTree(
            label="Add",
            index=5,
            left=ExpressionTree(label="var_1", index=6, left=None, right=None),
            right=ExpressionTree(label="var_0", index=7, left=None, right=None),
        ),
    ),
    ExpressionTree(
        label="Pow",
        index=4,
        left=ExpressionTree(
            label="Add",
            index=5,
            left=ExpressionTree(label="0", index=6, left=None, right=None),
            right=ExpressionTree(
                label="Pow",
                index=7,
                left=ExpressionTree(label="var_0", index=8, left=None, right=None),
                right=ExpressionTree(label="1", index=9, left=None, right=None),
            ),
        ),
        right=ExpressionTree(
            label="Add",
            index=10,
            left=ExpressionTree(label="0", index=11, left=None, right=None),
            right=ExpressionTree(label="var_1", index=12, left=None, right=None),
        ),
    ),
]
