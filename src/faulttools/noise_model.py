from typing import List, Optional, Iterable, Tuple, NamedTuple, Set

from .diagram import Diagram
from .pauli import PauliString, Pauli

class Fault(NamedTuple):
    """
    A fault, described by 1. diagram edges it flips and 2. detectors it violates / flips.

    Note that detectors may have once been part of a diagram, and two faults may be equivalent even though one is found
    to violate detectors and the other is not.
    """
    edge_flips: PauliString
    detector_flips: Set[int]

    @staticmethod
    def edge_flip(edge_idx: int, flip: Pauli) -> "Fault":
        return Fault(PauliString.edge_flip(edge_idx, flip), set())

class NoiseModel:
    _diagram: Diagram
    _atomic_weights: List[Tuple[Fault, int]]

    @staticmethod
    def edge_flip_noise(  # TODO test this
        diagram: Diagram,
        w_x: Optional[int] = None,
        w_y: Optional[int] = None,
        w_z: Optional[int] = None,
        idealised_edges: Optional[List[int]] = None,
    ) -> "NoiseModel":
        atomic_weights: List[Tuple[Fault, int]] = []
        for edge_idx in diagram.edge_indices():
            if edge_idx in idealised_edges:
                continue

            atomic_weights.append((Fault.edge_flip(edge_idx, Pauli.X), w_x or 1))
            atomic_weights.append((Fault.edge_flip(edge_idx, Pauli.Y), w_y or 1))
            atomic_weights.append((Fault.edge_flip(edge_idx, Pauli.Z), w_z or 1))

        return NoiseModel(diagram=diagram, atomic_weights=atomic_weights)

    def __init__(self, diagram: Diagram, atomic_weights: List[Tuple[Fault, int]]) -> None:
        self._diagram = diagram
        self._atomic_weights = atomic_weights

    def diagram(self) -> Diagram:
        return self._diagram

    def atomic_faults(self) -> Iterable[Fault]:
        return map(lambda x: x[0], self._atomic_weights)

    def atomic_weights(self) -> List[Tuple[Fault, int]]:
        return self._atomic_weights
