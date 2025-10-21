from enum import StrEnum
from fractions import Fraction
from typing import NamedTuple, Union, Literal

import rustworkx as rx


class NodeType(StrEnum):
    B = "B"  # Boundary
    Z = "Z"  # Z spider
    X = "X"  # X spider
    H = "H"  # H box


class NodeInfo(NamedTuple):
    type: NodeType
    phase: Union[Fraction, Literal[0]] = 0


class Diagram(rx.PyGraph):
    """
    A ZX diagram as an open graph.

    For allowed node types, see the NodeType enum.
    Node data must be an instance of NodeInfo.
    """

    def type(self, node_idx: int) -> NodeType:
        return self.get_node_data(node_idx).type

    def phase(self, node_idx: int) -> NodeType:
        return self.get_node_data(node_idx).phase
