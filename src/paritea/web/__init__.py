import numpy as np
from galois import GF2

from paritea.pauli import PauliString

from .compute import compute_detecting_regions, compute_pauli_webs, compute_stabilisers


def row_reduce_webs(webs: list[PauliString], idx_map: dict[int, int]) -> GF2:
    """Produces a row-reduction of the provided webs, only including the keys of the
    index map, at the index indicated by the keys entry in the map."""
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
