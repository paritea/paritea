from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction

from paritea import Pauli, PauliString
from paritea.diagram import Diagram, NodeType
from paritea.noise import Fault, NoiseModel


@dataclass(init=True, frozen=True)
class NoiseModelBuilder[CostT]:
    single_q_cost: tuple[CostT | None, CostT | None, CostT | None] = (None, None, None)
    two_q_cost: dict[tuple[Pauli, Pauli], CostT] = field(default_factory=dict)
    measure_cost: tuple[CostT | None, CostT | None, CostT | None] = (None, None, None)
    reset_cost: tuple[CostT | None, CostT | None, CostT | None] = (None, None, None)

    def for_single_q(self, qs: Iterable[int]) -> Iterable[tuple[dict[int, Pauli], CostT]]:
        for q in qs:
            for p, cost in zip((Pauli.X, Pauli.Z, Pauli.Y), self.single_q_cost):
                if cost is not None:
                    yield {q: p}, cost

    def for_two_q(self, qs: Iterable[tuple[int, int]]) -> Iterable[tuple[dict[int, Pauli], CostT]]:
        for q1, q2 in qs:
            for (p1, p2), cost in self.two_q_cost.items():
                yield {q1: p1, q2: p2}, cost

    def for_measure(self, qs: Iterable[int]) -> Iterable[tuple[dict[int, Pauli], CostT]]:
        for q in qs:
            for p, cost in zip((Pauli.X, Pauli.Z, Pauli.Y), self.measure_cost):
                if cost is not None:
                    yield {q: p}, cost

    def for_reset(self, qs: Iterable[int]) -> Iterable[tuple[dict[int, Pauli], CostT]]:
        for q in qs:
            for p, cost in zip((Pauli.X, Pauli.Z, Pauli.Y), self.reset_cost):
                if cost is not None:
                    yield {q: p}, cost


class PauliBasis(StrEnum):
    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(init=False)
class CircuitBuilder[CostT]:
    n_qubits: int
    diagram: Diagram
    current_qubit_nodes: list[int | None]
    row_offset: int

    _noise_model_builder: NoiseModelBuilder[CostT]
    atomic_faults: dict[Fault, list[CostT]]
    _fault_prot_queue: list[tuple[set, dict[int, Pauli], CostT]]

    def __init__(self, n_qubits: int, builder: NoiseModelBuilder[CostT] | None = None):
        self.n_qubits = n_qubits
        self.diagram = Diagram()
        self.current_qubit_nodes = [None for _ in range(n_qubits)]
        self.row_offset = 0
        self._noise_model_builder = builder or NoiseModelBuilder()
        self.atomic_faults = defaultdict(list)
        self._fault_prot_queue = []

    def _queue_for_edge_replacement(self, fault_prot: dict[int, Pauli], cost: CostT):
        self._fault_prot_queue.append((set(fault_prot.keys()), fault_prot, cost))

    def _flush_edge_for_qubit(self, qubit: int, edge: int) -> None:
        flushed_indices = set()
        for i, (qubits, fault_prot, cost) in enumerate(self._fault_prot_queue):
            if qubit not in qubits:
                continue

            fault_prot[edge] = fault_prot.pop(qubit)
            qubits.remove(qubit)
            if len(qubits) != 0:
                continue

            self.atomic_faults[Fault(PauliString(fault_prot))].append(cost)
            flushed_indices.add(i)

        if len(flushed_indices) > 0:
            self._fault_prot_queue = [prot for i, prot in enumerate(self._fault_prot_queue) if i not in flushed_indices]

    def _append_single_qubit(
        self,
        qs: Iterable[int],
        n_type: NodeType,
        phase: Fraction | None = None,
        *,
        connect: bool = True,
        offset: bool = True,
    ) -> list[int]:
        new_nodes = [self.diagram.add_node(n_type, phase, x=self.row_offset, y=q) for q in qs]
        if offset:
            self.row_offset += 1
        for q, next_node in zip(qs, new_nodes):
            if connect and self.current_qubit_nodes[q] is not None:
                e = self.diagram.add_edge(self.current_qubit_nodes[q], next_node)
                self._flush_edge_for_qubit(q, e)
            self.current_qubit_nodes[q] = next_node

        return new_nodes

    def append_boundary(self, qs: Iterable[int]) -> list[int]:
        return self._append_single_qubit(qs, NodeType.B)

    def append_cnot(self, cts: list[tuple[int, int]]) -> list[int]:
        new_nodes = []
        for c, t in cts:
            (c_node,) = self._append_single_qubit([c], n_type=NodeType.Z, offset=False)
            (t_node,) = self._append_single_qubit([t], n_type=NodeType.X, offset=True)
            self.diagram.add_edge(c_node, t_node)
            new_nodes.append(c_node)
            new_nodes.append(t_node)

        for prot, cost in self._noise_model_builder.for_two_q(cts):
            self._queue_for_edge_replacement(prot, cost)

        return new_nodes

    def append_measure(self, qs: Iterable[int], basis: PauliBasis) -> list[int]:
        """Appends a measurement in the given basis postselected to the +1 eigenstate of
        the basis."""
        for prot, cost in self._noise_model_builder.for_measure(qs):
            self._queue_for_edge_replacement(prot, cost)

        match basis:
            case PauliBasis.X:
                return self._append_single_qubit(qs, NodeType.Z)
            case PauliBasis.Y:
                return self._append_single_qubit(qs, NodeType.Z, phase=Fraction(-1, 2))
            case PauliBasis.Z:
                return self._append_single_qubit(qs, NodeType.X)
            case _:
                raise NotImplementedError(f"Unknown basis: {basis}")

    def append_measure_comp(self, qs: Iterable[int]) -> list[int]:
        return self.append_measure(qs, PauliBasis.Z)

    def append_reset(self, qs: Iterable[int], basis: PauliBasis) -> list[int]:
        """Appends a reset into the +1 eigenstate of the given basis."""
        match basis:
            case PauliBasis.X:
                nodes = self._append_single_qubit(qs, NodeType.Z, connect=False)
            case PauliBasis.Y:
                nodes = self._append_single_qubit(qs, NodeType.Z, phase=Fraction(1, 2))
            case PauliBasis.Z:
                nodes = self._append_single_qubit(qs, NodeType.X, connect=False)
            case _:
                raise NotImplementedError(f"Unknown basis: {basis}")
        for prot, cost in self._noise_model_builder.for_reset(qs):
            self._queue_for_edge_replacement(prot, cost)
        return nodes

    def append_reset_zero(self, qs: Iterable[int]) -> list[int]:
        return self.append_reset(qs, PauliBasis.Z)

    def append_x(self, qs: Iterable[int]) -> list[int]:
        nodes = self._append_single_qubit(qs, NodeType.X)
        for prot, cost in self._noise_model_builder.for_single_q(qs):
            self._queue_for_edge_replacement(prot, cost)
        return nodes

    def append_z(self, qs: Iterable[int]) -> list[int]:
        nodes = self._append_single_qubit(qs, NodeType.Z)
        for prot, cost in self._noise_model_builder.for_single_q(qs):
            self._queue_for_edge_replacement(prot, cost)
        return nodes

    def append_sqrt_x(self, qs: Iterable[int]) -> list[int]:
        nodes = self._append_single_qubit(qs, NodeType.X, phase=Fraction(1, 2))
        for prot, cost in self._noise_model_builder.for_single_q(qs):
            self._queue_for_edge_replacement(prot, cost)
        return nodes

    def append_sqrt_z(self, qs: Iterable[int]) -> list[int]:
        nodes = self._append_single_qubit(qs, NodeType.Z, phase=Fraction(1, 2))
        for prot, cost in self._noise_model_builder.for_single_q(qs):
            self._queue_for_edge_replacement(prot, cost)
        return nodes

    def append_h(self, qs: Iterable[int]) -> list[int]:
        nodes1 = self._append_single_qubit(qs, NodeType.X, phase=Fraction(1, 2))
        nodes2 = self._append_single_qubit(qs, NodeType.Z, phase=Fraction(1, 2))
        nodes3 = self._append_single_qubit(qs, NodeType.X, phase=Fraction(1, 2))
        for prot, cost in self._noise_model_builder.for_single_q(qs):
            self._queue_for_edge_replacement(prot, cost)
        return nodes1 + nodes2 + nodes3

    def finish(self) -> tuple[Diagram, NoiseModel[CostT]]:
        return self.diagram, NoiseModel(self.diagram, self.atomic_faults)
