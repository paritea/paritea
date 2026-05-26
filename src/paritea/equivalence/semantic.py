import numpy as np

from paritea import PauliString
from paritea.diagram import Diagram
from paritea.web import compute_stabilisers
from galois import GF2

def _reduce_stabilisers(stabilisers: list[PauliString], boundary_idx_map: dict[int, int]) -> GF2:
    np_stabilisers = np.zeros((len(stabilisers), len(boundary_idx_map) * 2), dtype=int)
    for i, stab in enumerate(stabilisers):
        np_stabilisers[i, :] = stab.restrict(boundary_idx_map.keys()).compile(boundary_idx_map)

    return GF2(np_stabilisers).row_reduce(eye="left")

def is_semantic_equivalence(d1: Diagram, d2: Diagram) -> bool:
    d1_edge_idx_map = {d1.incident_edges(b)[0]: i for i, b in enumerate(d1.io_sorted())}
    d2_edge_idx_map = {d2.incident_edges(b)[0]: i for i, b in enumerate(d2.io_sorted())}

    rref_1 = _reduce_stabilisers(compute_stabilisers(d1), d1_edge_idx_map)
    rref_2 = _reduce_stabilisers(compute_stabilisers(d2), d2_edge_idx_map)

    return np.array_equal(rref_1, rref_2)
