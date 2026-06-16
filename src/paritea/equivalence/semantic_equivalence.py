import numpy as np

from paritea.diagram import Diagram
from paritea.web import compute_stabilisers, row_reduce_webs


def is_semantic_equivalence(d1: Diagram, d2: Diagram) -> bool:
    d1_edge_idx_map = {d1.incident_edges(b)[0]: i for i, b in enumerate(d1.io_sorted())}
    d2_edge_idx_map = {d2.incident_edges(b)[0]: i for i, b in enumerate(d2.io_sorted())}
    rref_1 = row_reduce_webs(compute_stabilisers(d1), d1_edge_idx_map)
    rref_2 = row_reduce_webs(compute_stabilisers(d2), d2_edge_idx_map)

    return np.array_equal(rref_1, rref_2)
