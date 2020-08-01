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
from typing import Any, Dict, List, Optional, Tuple

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


SEQUENCE_FIELDS = {
    "left": TEXT_FIELD,
    "right": TEXT_FIELD,
    "target": TARGET_FIELD,
    "left_root_index": INDEX_FIELD,
    "right_root_index": INDEX_FIELD,
}
_SEQUENCE_ASSOC_LIST = list(SEQUENCE_FIELDS.items())

# fmt: off
SEQUENCE_TEST_STRINGS = [
    [" <init> ", " 2 ", " <eos> "],
    [" <init> ", " ( ", " var_0 ", " ) ", " Pow ", " ( ", " 1/2 ", " ) ", " <eos> "],
    [
        " <init> ",
        " ( ", " ( ", " 1 ", " ) ", " Pow ", " ( ", " 1/2 ", " ) ", " ) ",
        " Mul ",
        " ( ", " ( ", " var_1 ", " ) ", " Add ", " ( ", " var_0 ", " ) ", " ) ",
        " <eos> ",
    ],
    [' <init> ',
     ' ( ', ' ( ', ' var_0 ', ' ) ', ' Pow ', ' ( ', ' 1 ', ' ) ', ' ) ',
     ' Mul ',
     ' ( ', ' ( ', ' var_2 ', ' ) ',
     ' Add ',
     ' ( ', ' ( ', ' 0 ', ' ) ', ' Add ', ' ( ', ' var_1 ', ' ) ', ' ) ', ' ) ',
     ' <eos> ']
]
# fmt: on


def sequence_root_index(example: ExpressionTree) -> int:
    """
    Get the index of the root token in a sequence expression.
    """
    if example.left is None or example.right is None:
        return 1  # Just the <init> token

    return len(tokenize(str(example.left))) + 3  # <init> token and () around left


def make_sequence_example(example: Dict[str, Any]) -> torchtext.data.Example:
    """
    Make a Sequence-style example from a serialized equation verification example.
    """
    tree = ExpressionTree.from_serialized(example["equation"])
    assert tree.left is not None
    assert tree.right is not None
    left = str(tree.left)
    right = str(tree.right)
    label = int(example["label"] == "1")

    left_root_index = sequence_root_index(tree.left)
    right_root_index = sequence_root_index(tree.right)
    assert tokenize(tree.left.label)[0] == tokenize(left)[left_root_index - 1]
    assert tokenize(tree.right.label)[0] == tokenize(right)[right_root_index - 1]

    return torchtext.data.Example.fromlist(
        [left, right, label, left_root_index, right_root_index], _SEQUENCE_ASSOC_LIST
    )


def get_split_sequence(file: str) -> torchtext.data.Dataset:
    """
    Make a Sequence-style dataset from a file of equation verification data.
    """
    with open(file, "r") as f:
        raw_data = json.load(f)

    data_flat = itertools.chain.from_iterable(raw_data)
    examples = map(make_sequence_example, data_flat)
    return torchtext.data.Dataset(list(examples), SEQUENCE_FIELDS)


def get_split_sequence_lengths(file: str) -> Dict[int, torchtext.data.Dataset]:
    """
    Make a collection of sequence-style datasets, indexed by depth, from a data file.
    """
    with open(file, "r") as f:
        raw_data = json.load(f)

    datasets_and_lengths = {}
    for length, data in enumerate(raw_data):
        if not data:
            continue

        examples = map(make_sequence_example, data)
        dataset = torchtext.data.Dataset(list(examples), SEQUENCE_FIELDS)
        datasets_and_lengths[length] = dataset

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

        self.data = []
        for split in raw_data:
            if not split:
                continue

            deserialized = list(map(self.deserialize, split))
            self.data.extend(deserialized * repeats)

    def __iter__(self):
        return iter(self.data)

    @staticmethod
    def deserialize(serialized):
        tree = ExpressionTree.from_serialized(serialized["equation"])
        label = torch.tensor(serialized["label"] == "1")
        return (tree, label)
