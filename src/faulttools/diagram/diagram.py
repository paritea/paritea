from enum import StrEnum
from fractions import Fraction
from recordclass import RecordClass

import rustworkx as rx


class NodeType(StrEnum):
    B = "B"  # Boundary
    Z = "Z"  # Z spider
    X = "X"  # X spider
    H = "H"  # H box


class NodeInfo(RecordClass):
    type: NodeType
    phase: Fraction = Fraction(0, 1)


class Diagram(rx.PyGraph[NodeInfo, None]):
    """
    A ZX diagram as an open graph.

    For allowed node types, see the NodeType enum.
    Node data must be an instance of NodeInfo.
    """

    def type(self, node_idx: int) -> NodeType:
        return self.get_node_data(node_idx).type

    def phase(self, node_idx: int) -> Fraction:
        return self.get_node_data(node_idx).phase

    def add_to_phase(self, node_idx: int, phase: Fraction):
        old_phase = self.get_node_data(node_idx).phase
        self.get_node_data(node_idx).phase = (old_phase + phase) % 2
