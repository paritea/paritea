import numpy as np
from galois import GF2

from .compute import compute_detecting_regions, compute_pauli_webs, compute_stabilisers
from ..pauli import PauliString


def row_reduce_webs(webs: list[PauliString], idx_map: dict[int, int]) -> GF2:
    np_stabilisers = np.zeros((len(webs), len(idx_map) * 2), dtype=int)
    for i, stab in enumerate(webs):
        np_stabilisers[i, :] = stab.restrict(idx_map.keys()).compile(idx_map)

    return GF2(np_stabilisers).row_reduce(eye="left")

__all__ = [
    "compute_detecting_regions",
    "compute_pauli_webs",
    "compute_stabilisers",
    "row_reduce_webs",
]
