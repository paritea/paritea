from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from typing import NamedTuple

from galois import GF2

from faulttools.diagram import Diagram
from faulttools.pauli import Pauli, PauliString


class Fault(NamedTuple):
    """
    A fault, described by 1. diagram edges it flips and 2. detectors it violates / flips.

    Note that detectors may have once been part of a diagram, and two faults may be equivalent even though one is found
    to violate detectors and the other is not.
    """

    edge_flips: PauliString
    detector_flips: frozenset[int] = frozenset()

    @staticmethod
    def edge_flip(edge_idx: int, flip: Pauli) -> "Fault":
        return Fault(PauliString.unary(edge_idx, flip), frozenset())

    def is_trivial(self) -> bool:
        return len(self.detector_flips) == 0 and self.edge_flips.is_trivial()

    def compile(self, edge_idx_map: Mapping[int, int], detector_idx_map: Mapping[int, int]) -> GF2:
        num_edges = len(edge_idx_map)
        compiled = GF2.Zeros(num_edges * 2 + len(detector_idx_map))
        for edge, pauli in self.edge_flips.items():
            idx = edge_idx_map[edge]
            if pauli == Pauli.Z or pauli == Pauli.Y:
                compiled[idx] = 1
            if pauli == Pauli.X or pauli == Pauli.Y:
                compiled[idx + num_edges] = 1

        for detector in self.detector_flips:
            compiled[detector_idx_map[detector] + num_edges * 2] = 1

        return compiled

    @staticmethod
    def compiled_to_int(compiled: GF2) -> int:
        out = 0
        for bit in compiled.tolist():
            out = (out << 1) | bit
        return out

    def to_int(self, edge_idx_map: Mapping[int, int], detector_idx_map: Mapping[int, int]) -> int:
        return Fault.compiled_to_int(self.compile(edge_idx_map, detector_idx_map))


class NoiseModel[T]:
    _diagram: Diagram
    _atomic_faults: dict[Fault, list[T]]

    @staticmethod
    def weighted_edge_flip_noise(
        diagram: Diagram,
        w_x: int | None = None,
        w_y: int | None = None,
        w_z: int | None = None,
        idealised_edges: list[int] | None = None,
    ) -> "NoiseModel[int]":
        idealised_edges = idealised_edges or []
        atomic_faults: dict[Fault, list[int]] = defaultdict(list)
        for edge_idx in diagram.edge_indices():
            if edge_idx in idealised_edges:
                continue

            atomic_faults[Fault.edge_flip(edge_idx, Pauli.X)].append(w_x or 1)
            atomic_faults[Fault.edge_flip(edge_idx, Pauli.Y)].append(w_y or 1)
            atomic_faults[Fault.edge_flip(edge_idx, Pauli.Z)].append(w_z or 1)

        return NoiseModel(diagram=diagram, atomic_faults=atomic_faults)

    def __init__(self, diagram: Diagram, atomic_faults: dict[Fault, list[T]]) -> None:
        self._diagram = diagram
        self._atomic_faults = atomic_faults

    @property
    def diagram(self) -> Diagram:
        return self._diagram

    def num_faults(self):
        return sum(len(vs) for vs in self._atomic_faults.values())

    def atomic_faults(self) -> Iterable[Fault]:
        return self._atomic_faults.keys()

    def atomic_faults_with_weight(self) -> Iterable[tuple[Fault, T]]:
        for fault, values in self._atomic_faults.items():
            for value in values:
                yield fault, value

    def compress(self, reweight_func: Callable[[T, T], T]) -> None:
        for fault, values in self._atomic_faults.items():
            if len(values) == 0 or fault.is_trivial():
                continue

            compressed = reduce(reweight_func, values)
            values.clear()
            values.append(compressed)

    def transform_faults(self, fault_transform: Callable[[Fault], Fault]) -> "NoiseModel":
        new_faults: dict[Fault, list[T]] = {}
        for fault in self._atomic_faults:
            new_fault = fault_transform(fault)
            if new_fault not in new_faults:
                new_faults[new_fault] = []
            new_faults[new_fault].extend(self._atomic_faults[fault])

        return NoiseModel(self._diagram, new_faults)
