from collections import defaultdict
from typing import Dict, List, NamedTuple

import numpy as np

from pyzx import Mat2
from pyzx.linalg import Z2

from ..diagram import Diagram, NodeType
from ..pauli import Pauli, PauliString


class GraphOrdering(NamedTuple):
    graph_to_ordering: Dict[int, int]
    ordering_to_graph: Dict[int, int]

    z_boundaries: Dict[int, int]
    internal_spiders: List[int]
    pi_2_spiders: List[int]

    def ord(self, s: int) -> int:
        return self.graph_to_ordering[s]

    def graph(self, o: int) -> int:
        return self.ordering_to_graph[o]


def determine_ordering(d: Diagram) -> GraphOrdering:
    boundaries = d.boundary_nodes()
    z_boundaries = {d.neighbors(b)[0]: b for b in boundaries}
    internal_spiders = list(set(d.node_indices()).difference(boundaries).difference(z_boundaries.keys()))
    pi_2_spiders = list(filter(lambda _v: d.phase(_v).denominator == 2, internal_spiders))

    graph_to_ordering: Dict[int, int] = dict()
    ordering_to_graph: Dict[int, int] = dict()
    idx = 0
    for boundary in z_boundaries.keys():
        graph_to_ordering[boundary] = idx
        ordering_to_graph[idx] = boundary
        idx += 1
    for internal in set(internal_spiders).difference(pi_2_spiders):
        graph_to_ordering[internal] = idx
        ordering_to_graph[idx] = internal
        idx += 1
    for pi_2_spider in pi_2_spiders:
        graph_to_ordering[pi_2_spider] = idx
        ordering_to_graph[idx] = pi_2_spider
        idx += 1

    return GraphOrdering(graph_to_ordering, ordering_to_graph, z_boundaries, internal_spiders, pi_2_spiders)


def create_firing_verification(d: Diagram, ordering: GraphOrdering) -> Mat2:
    num_z_boundaries = len(ordering.z_boundaries)
    num_non_boundary_spiders = num_z_boundaries + len(ordering.internal_spiders)
    adj_matrix = Mat2.zeros(num_non_boundary_spiders, num_non_boundary_spiders)

    for s, t in d.edge_list():
        if d.type(s) != NodeType.B and d.type(t) != NodeType.B:
            adj_matrix[ordering.ord(s), ordering.ord(t)] += 1
            adj_matrix[ordering.ord(t), ordering.ord(s)] += 1

    m_d = Mat2.zeros(adj_matrix.rows(), adj_matrix.cols() + num_z_boundaries)
    m_d[0:num_z_boundaries, 0:num_z_boundaries] = Mat2.id(num_z_boundaries)
    m_d[:, num_z_boundaries:] = adj_matrix
    num_pi_2 = len(ordering.pi_2_spiders)
    slice_key = (slice(m_d.rows() - num_pi_2, m_d.rows()), slice(m_d.cols() - num_pi_2, m_d.cols()))
    m_d[slice_key] = Mat2(
        (np.array(m_d[slice_key].data, dtype=bool) ^ np.array(Mat2.id(num_pi_2).data, dtype=bool)).tolist()
    )

    return m_d


def convert_firing_assignment_to_web(d: Diagram, ordering: GraphOrdering, v: List[Z2]) -> PauliString:
    web: Dict[int, Pauli] = defaultdict(lambda: Pauli.I)

    for adj_vertex, g_vertex in ordering.ordering_to_graph.items():
        g_type = d.type(g_vertex)
        # Fire all green spiders with full red edges and thus their red neighbours
        if g_type == NodeType.Z and v[adj_vertex + len(ordering.z_boundaries)] == 1:
            for _n in d.neighbors(g_vertex):
                web[d.edge_indices_from_endpoints(g_vertex, _n)[0]] *= Pauli.X
        # Fire all red spiders with full green edges and thus their green neighbours
        if g_type == NodeType.X and v[adj_vertex + len(ordering.z_boundaries)] == 1:
            for _n in d.neighbors(g_vertex):
                web[d.edge_indices_from_endpoints(g_vertex, _n)[0]] *= Pauli.Z

    # Fire all green output edges
    for g_z_boundary, g_boundary in ordering.z_boundaries.items():
        adj_z_boundary = ordering.ord(g_z_boundary)
        if v[adj_z_boundary] == 1:
            web[d.edge_indices_from_endpoints(g_z_boundary, g_boundary)[0]] *= Pauli.Z

    return PauliString(web)
