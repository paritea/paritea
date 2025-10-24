from typing import List, Optional, Iterable, Tuple

from .diagram import Diagram
from .pauli import PauliString, Pauli


class NoiseModel:  # TODO incorporate detector flip information into atomic fault description
    _diagram: Diagram
    _atomic_weights: List[Tuple[PauliString, int]]

    @staticmethod
    def edge_flip_noise(  # TODO test this
        diagram: Diagram,
        w_x: Optional[int] = None,
        w_y: Optional[int] = None,
        w_z: Optional[int] = None,
        idealised_edges: Optional[List[int]] = None,
    ) -> "NoiseModel":
        atomic_weights: List[Tuple[PauliString, int]] = []
        for edge_idx in diagram.edge_indices():
            if edge_idx in idealised_edges:
                continue

            atomic_weights.append((PauliString.edge_flip(edge_idx, Pauli.X), w_x or 1))
            atomic_weights.append((PauliString.edge_flip(edge_idx, Pauli.Y), w_y or 1))
            atomic_weights.append((PauliString.edge_flip(edge_idx, Pauli.Z), w_z or 1))

        return NoiseModel(diagram=diagram, atomic_weights=atomic_weights)

    def __init__(self, diagram: Diagram, atomic_weights: List[Tuple[PauliString, int]]) -> None:
        self._diagram = diagram
        self._atomic_weights = atomic_weights

    def diagram(self) -> Diagram:
        return self._diagram

    def atomic_faults(self) -> Iterable[PauliString]:
        return map(lambda x: x[0], self._atomic_weights)

    def atomic_weights(self) -> List[Tuple[PauliString, int]]:
        return self._atomic_weights
