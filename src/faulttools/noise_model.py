from typing import Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple, Union

from galois import GF2

from .diagram import Diagram
from .diagram.conversion import DiagramParam, to_diagram
from .pauli import Pauli, PauliString
from .util import canonicalize_input


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
        return Fault(PauliString.unary(edge_idx, flip), set())

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


class NoiseModel:
    _diagram: Diagram
    _atomic_weights: List[Tuple[Fault, int]]

    @staticmethod
    def edge_flip_noise(
        diagram: Diagram,
        w_x: Optional[int] = None,
        w_y: Optional[int] = None,
        w_z: Optional[int] = None,
        idealised_edges: Optional[List[int]] = None,
    ) -> "NoiseModel":
        idealised_edges = idealised_edges or []
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


type NoiseModelParam = Union[NoiseModel, DiagramParam]


def to_noise_model(obj: NoiseModelParam) -> NoiseModel:
    if isinstance(obj, NoiseModel):
        return obj
    else:
        return NoiseModel.edge_flip_noise(to_diagram(obj))


def noise_model_params(*param_names: str):
    return canonicalize_input(**{name: to_noise_model for name in param_names})
