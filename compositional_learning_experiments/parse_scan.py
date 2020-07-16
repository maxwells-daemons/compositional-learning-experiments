"""
Code to parse examples from the SCAN dataset into an intermediate representation
sufficient to reproduce the input & output accurately, as well as convert into other
forms (such as various network structures).

Notes
-----
This does not follow the original grammar exactly, but is similar and can be easily
converted in both directions. The BNF for the grammar this module uses is as follows:

<Command> ::= AND   <RepeatedAction> <RepeatedAction>
            | AFTER <RepeatedAction> <RepeatedAction>
            | JUST  <RepeatedAction>

<RepeatedAction> ::= ONCE   <DirectedAction>
                   | TWICE  <DirectedAction>
                   | THRICE <DirectedAction>

<DirectedAction> ::= UNDIRECTED <AtomicAction>
                   | DIRECTED   <AtomicAction> <Direction>
                   | OPPOSITE   <AtomicAction> <Direction>
                   | AROUND     <AtomicAction> <Direction>

<AtomicAction> ::= TURN | WALK | LOOK | RUN | JUMP
<Direction> ::= LEFT | RIGHT

The most notable deviations from the original phrase-structured grammar are:
    - TURN is included as an atomic action to simplify the DirectedAction category.
    - There is a new Direction category to further break down directional commands.

Examples
--------
>>> ir = Command.parser.parse("walk left and jump right twice")
>>> ir.to_input()
"walk left and jump right twice"
>>> ir.to_output()
"I_TURN_LEFT I_WALK I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP"
>>> ir.plot_graph()
<graphviz.dot.Digraph object at ...>
"""

import abc
from enum import Enum
from typing import List, Optional, Tuple

import graphviz
import parsy


def split_example(example: str) -> Tuple[str, str]:
    """
    Split an example from the SCAN dataset into an input/target pair.

    Parameters
    ----------
    example : str
        The example.

    Returns
    -------
    str
        The input.
    str
        The target.
    """
    inp = example.split("OUT:")[0].lstrip("IN:").strip()
    out = example.split("OUT:")[1].strip()
    return inp, out


class Node(abc.ABC):
    """
    A superclass for all categories in the IR grammar. For a complete description
    of allowed values, see the BNF in the module docstring.

    Attributes
    ----------
    children : List[Node] (default: [])
        A list of all nodes this node depends on.
    kind : Kind
        Which variant of this category is being concretely represented.
    """

    class Kind(Enum):
        """
        Enumerates the acceptable variants of a given category.
        """

        pass

    children: "List[Node]" = []
    kind: Kind

    def __init__(self, kind: Kind):
        self.kind = kind

    def _name(self) -> str:
        """
        Create a human-readable name for this concrete node.
        """
        return f"{self.__class__.__name__}: {self.kind.value}"

    def __repr__(self) -> str:
        return f"{self._name()} {self.children}"

    def _add_subtree_nodes(self, graph: graphviz.Digraph, idx: int) -> int:
        """
        A helper method to recursively add this node and all of its children to a graph.

        Attributes
        ----------
        graph : graphviz.Digraph
            The graph to add nodes to.
        idx : int
            An index uniquely identifying this node.

        Returns
        -------
        int
            The maximum index in this subtree. Used to ensure names are unique.
        """
        graph.node(str(idx), self._name())

        max_idx = idx
        for child in self.children:
            child_idx = max_idx + 1
            max_idx = child._add_subtree_nodes(graph, child_idx)
            graph.edge(str(child_idx), str(idx))

        return max_idx

    def plot_graph(self) -> graphviz.Digraph:
        """
        Create a GraphViz graph of the IR as a directed tree.
        """
        graph = graphviz.Digraph()
        self._add_subtree_nodes(graph, 0)
        return graph

    @abc.abstractmethod
    def to_input(self) -> str:
        """
        Convert this example into a SCAN input.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_output(self) -> str:
        """
        Convert this example into a SCAN output.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def parser():
        """
        Get a Parsy parser for this category, including its constituent nodes.
        """
        raise NotImplementedError


class AtomicAction(Node):
    class Kind(Enum):
        TURN = "turn"
        WALK = "walk"
        LOOK = "look"
        RUN = "run"
        JUMP = "jump"

    @staticmethod
    @parsy.generate("AtomicAction")
    def parser():
        kind = yield (
            parsy.string("turn").result(AtomicAction.Kind.TURN)
            | parsy.string("walk").result(AtomicAction.Kind.WALK)
            | parsy.string("look").result(AtomicAction.Kind.LOOK)
            | parsy.string("run").result(AtomicAction.Kind.RUN)
            | parsy.string("jump").result(AtomicAction.Kind.JUMP)
        )
        return AtomicAction(kind)

    def to_input(self) -> str:
        return self.kind.value

    def to_output(self) -> str:
        return f"I_{self.kind.name}"


class Direction(Node):
    class Kind(Enum):
        LEFT = "left"
        RIGHT = "right"

    @staticmethod
    @parsy.generate("Direction")
    def parser():
        kind = yield (
            parsy.string("left").result(Direction.Kind.LEFT)
            | parsy.string("right").result(Direction.Kind.RIGHT)
        )
        return Direction(kind)

    def to_input(self) -> str:
        return self.kind.value

    def to_output(self) -> str:
        return self.kind.name


class DirectedAction(Node):
    direction: Optional[Direction]  # None iff UNDIRECTED
    action: AtomicAction

    class Kind(Enum):
        UNDIRECTED = "UNDIRECTED"
        DIRECTED = "DIRECTED"
        OPPOSITE = "opposite"
        AROUND = "around"

    def __init__(
        self, kind, direction: Optional[Direction], action: AtomicAction,
    ):
        # UNDIRECTED must have no direction, and everything else must have a direction
        assert bool(direction) == (kind != DirectedAction.Kind.UNDIRECTED)

        super().__init__(kind)
        self.direction = direction
        self.action = action
        self.children = [action, direction] if direction else [action]

    @staticmethod
    @parsy.generate("DirectedAction")
    def parser():
        action = yield AtomicAction.parser

        kind = yield (
            parsy.whitespace
            >> (
                parsy.string("opposite").result(DirectedAction.Kind.OPPOSITE)
                | parsy.string("around").result(DirectedAction.Kind.AROUND)
            )
        ).optional()
        if kind:
            direction = yield (parsy.whitespace >> Direction.parser)
            return DirectedAction(kind, direction, action)

        direction = yield (parsy.whitespace >> Direction.parser).optional()
        if direction:
            return DirectedAction(DirectedAction.Kind.DIRECTED, direction, action)

        return DirectedAction(DirectedAction.Kind.UNDIRECTED, None, action)

    def to_input(self) -> str:
        act = self.action.to_input()
        if self.kind == DirectedAction.Kind.UNDIRECTED:
            return act
        if self.kind == DirectedAction.Kind.DIRECTED:
            return f"{act} {self.direction.to_input()}"  # type: ignore
        return f"{act} {self.kind.value} {self.direction.to_input()}"  # type: ignore

    def to_output(self) -> str:
        act = self.action.to_output()
        if self.kind == DirectedAction.Kind.UNDIRECTED:
            return act

        turn = "I_TURN_" + self.direction.to_output()  # type: ignore
        if self.kind == DirectedAction.Kind.DIRECTED:
            if self.action.kind == AtomicAction.Kind.TURN:
                return turn
            return f"{turn} {act}"

        if self.kind == DirectedAction.Kind.OPPOSITE:
            turn_twice = f"{turn} {turn}"
            if self.action.kind == AtomicAction.Kind.TURN:
                return turn_twice
            return f"{turn_twice} {act}"

        # AROUND
        if self.action.kind == AtomicAction.Kind.TURN:
            return f"{turn} {turn} {turn} {turn}"
        return f"{turn} {act} {turn} {act} {turn} {act} {turn} {act}"


class RepeatedAction(Node):
    action: DirectedAction

    class Kind(Enum):
        ONCE = "ONCE"
        TWICE = "twice"
        THRICE = "thrice"

    def __init__(self, kind, action: DirectedAction):
        super().__init__(kind)
        self.action = action
        self.children = [action]

    @staticmethod
    @parsy.generate("RepeatedAction")
    def parser():
        action = yield DirectedAction.parser
        kind = yield (
            parsy.whitespace
            >> (
                parsy.string("twice").result(RepeatedAction.Kind.TWICE)
                | parsy.string("thrice").result(RepeatedAction.Kind.THRICE)
            )
        ).optional()
        kind = kind or RepeatedAction.Kind.ONCE
        return RepeatedAction(kind, action)

    def to_input(self) -> str:
        act = self.action.to_input()
        if self.kind == RepeatedAction.Kind.ONCE:
            return act
        return f"{act} {self.kind.value}"

    def to_output(self) -> str:
        act = self.action.to_output()
        if self.kind == RepeatedAction.Kind.TWICE:
            return f"{act} {act}"
        if self.kind == RepeatedAction.Kind.THRICE:
            return f"{act} {act} {act}"
        return act


class Command(Node):
    action_1: DirectedAction
    action_2: Optional[DirectedAction]  # None iff JUST

    class Kind(Enum):
        AND = "and"
        AFTER = "after"
        JUST = "JUST"

    def __init__(
        self, kind, action_1: DirectedAction, action_2: Optional[DirectedAction]
    ):
        assert bool(action_2) == (kind != Command.Kind.JUST)

        super().__init__(kind)
        self.action_1 = action_1
        self.action_2 = action_2
        self.children = [action_1, action_2] if self.action_2 else [action_1]  # type: ignore

    @staticmethod
    @parsy.generate("Command")
    def parser():
        act_1 = yield RepeatedAction.parser
        kind = yield (
            parsy.whitespace
            >> (
                parsy.string("and").result(Command.Kind.AND)
                | parsy.string("after").result(Command.Kind.AFTER)
            )
        ).optional()

        if kind:
            act_2 = yield (parsy.whitespace >> RepeatedAction.parser)
            return Command(kind, act_1, act_2)
        return Command(Command.Kind.JUST, act_1, None)

    def to_input(self) -> str:
        act_1 = self.action_1.to_input()
        if self.kind == Command.Kind.JUST:
            return act_1

        act_2 = self.action_2.to_input()  # type: ignore
        return f"{act_1} {self.kind.value} {act_2}"

    def to_output(self) -> str:
        act_1 = self.action_1.to_output()
        if self.kind == Command.Kind.JUST:
            return act_1

        act_2 = self.action_2.to_output()  # type: ignore
        if self.kind == Command.Kind.AND:
            return f"{act_1} {act_2}"
        return f"{act_2} {act_1}"


# Testing code
def test_parsing_correctness(
    dataset_file: str = "SCAN/simple_split/tasks_train_simple.txt",
):
    with open(dataset_file, "r") as f:
        for line in f.readlines():
            inp, out = split_example(line)
            ir = Command.parser.parse(inp)
            assert inp == ir.to_input()
            assert out == ir.to_output()
