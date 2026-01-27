from collections.abc import Iterable
from dataclasses import dataclass

from paritea.diagram import Diagram, NodeType


@dataclass(init=False)
class CircuitBuilder:
    n_qubits: int
    diagram: Diagram
    current_qubit_nodes: list[int | None]
    row_offset: int

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.diagram = Diagram()
        self.current_qubit_nodes = [None for _ in range(n_qubits)]
        self.row_offset = 0

    def append_single_qubit(
        self, qs: Iterable[int], n_type: NodeType, *, connect: bool = True, offset: bool = True
    ) -> list[int]:
        new_nodes = [self.diagram.add_node(n_type, x=self.row_offset, y=q) for q in qs]
        if offset:
            self.row_offset += 1
        for q, next_node in zip(qs, new_nodes):
            if connect and self.current_qubit_nodes[q] is not None:
                self.diagram.add_edge(self.current_qubit_nodes[q], next_node)
            self.current_qubit_nodes[q] = next_node

        return new_nodes

    def append_cnot(self, cts: list[tuple[int, int]]) -> list[int]:
        new_nodes = []
        for c, t in cts:
            (c_node,) = self.append_single_qubit([c], n_type=NodeType.Z, offset=False)
            (t_node,) = self.append_single_qubit([t], n_type=NodeType.X, offset=True)
            self.diagram.add_edge(c_node, t_node)
            new_nodes.append(c_node)
            new_nodes.append(t_node)

        return new_nodes

    def append_measure(self, qs: Iterable[int], n_type: NodeType) -> list[int]:
        return self.append_single_qubit(qs, n_type)

    def append_reset(self, qs: Iterable[int], n_type: NodeType) -> list[int]:
        return self.append_single_qubit(qs, n_type, connect=False)
