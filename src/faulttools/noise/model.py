from collections.abc import Callable, Iterable, Mapping
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
    _atomic_faults: list[tuple[Fault, T]]

    @staticmethod
    def weighted_edge_flip_noise(
        diagram: Diagram,
        w_x: int | None = None,
        w_y: int | None = None,
        w_z: int | None = None,
        idealised_edges: list[int] | None = None,
    ) -> "NoiseModel[int]":
        idealised_edges = idealised_edges or []
        atomic_faults: list[tuple[Fault, int]] = []
        for edge_idx in diagram.edge_indices():
            if edge_idx in idealised_edges:
                continue

            atomic_faults.append((Fault.edge_flip(edge_idx, Pauli.X), w_x or 1))
            atomic_faults.append((Fault.edge_flip(edge_idx, Pauli.Y), w_y or 1))
            atomic_faults.append((Fault.edge_flip(edge_idx, Pauli.Z), w_z or 1))

        return NoiseModel(diagram=diagram, atomic_faults=atomic_faults)

    def __init__(self, diagram: Diagram, atomic_faults: list[tuple[Fault, T]]) -> None:
        self._diagram = diagram
        self._atomic_faults = atomic_faults

    @property
    def diagram(self) -> Diagram:
        return self._diagram

    def atomic_faults(self) -> Iterable[Fault]:
        return (x[0] for x in self._atomic_faults)

    def atomic_faults_with_weight(self) -> list[tuple[Fault, T]]:
        return self._atomic_faults

    def compress(self, reweight_func: Callable[[T, T], T]) -> None:
        faults: dict[Fault, T] = {}
        for fault, v in self._atomic_faults:
            if fault.is_trivial():
                continue

            existing_v = faults.get(fault)
            faults[fault] = v if existing_v is None else reweight_func(existing_v, v)
        self._atomic_faults = list(faults.items())
