from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Tuple

from pyzx.graph.base import upair

from faulttools.diagram import Diagram, NodeType
from faulttools.pauli import Pauli


@dataclass(init=True, frozen=True)
class ExtraIdNode:
    node: int


@dataclass(init=True, frozen=True)
class ExpandedHadamard:
    r1_node: int
    r2_node: int
    r3_node: int
    origin: int
    flipped_decomposition: bool


class AdditionalNodes:
    extra_id_nodes: List[ExtraIdNode]
    expanded_hadamards: List[ExpandedHadamard]

    def __init__(self, extra_id_nodes: List[ExtraIdNode], expanded_hadamards: List[ExpandedHadamard]):
        self.extra_id_nodes = extra_id_nodes
        self.expanded_hadamards = expanded_hadamards

    @staticmethod
    def empty() -> "AdditionalNodes":
        return AdditionalNodes([], [])

    def add_extra_id_node(self, node: int):
        self.extra_id_nodes.append(ExtraIdNode(node))

    def add_expanded_hadamard(self, expanded_hadamard: ExpandedHadamard):
        self.expanded_hadamards.append(expanded_hadamard)

    def _remove_extra_id_node(
        self, adj: Dict[int, Dict[int, any]], web: Dict[Tuple[int, int], Pauli], id_node: ExtraIdNode
    ):
        v1, v2 = adj[id_node.node].keys()
        web[upair(v1, v2)] = web.get(upair(v1, id_node.node), Pauli.I)
        adj[v1][v2] = True
        adj[v2][v1] = True
        web.pop(upair(v1, id_node.node), "")
        web.pop(upair(id_node.node, v2), "")
        del adj[v1][id_node.node]
        del adj[id_node.node][v1]
        del adj[id_node.node][v2]
        del adj[v2][id_node.node]

    def _remove_expanded_hadamard(
        self, adj: Dict[int, Dict[int, any]], web: Dict[Tuple[int, int], Pauli], hadamard: ExpandedHadamard
    ):
        w1, w2, w3 = hadamard.r1_node, hadamard.r2_node, hadamard.r3_node
        w1_left, w1_right = adj[w1].keys()
        l = w1_left if w1_right == w2 else w1_right
        w3_left, w3_right = adj[w3].keys()
        r = w3_right if w3_left == w2 else w3_left

        web[upair(l, hadamard.origin)] = web.get(upair(l, w1), Pauli.I)
        web[upair(hadamard.origin, r)] = web.get(upair(r, w3), Pauli.I)
        if hadamard.origin not in adj:
            adj[hadamard.origin] = dict()
        adj[l][hadamard.origin] = True
        adj[hadamard.origin][l] = True
        adj[hadamard.origin][r] = True
        adj[r][hadamard.origin] = True

        web.pop(upair(l, w1), "")
        web.pop(upair(w1, w2), "")
        web.pop(upair(w2, w3), "")
        web.pop(upair(w3, r), "")
        del adj[l][w1]
        del adj[w1][l]
        del adj[w1][w2]
        del adj[w2][w1]
        del adj[w2][w3]
        del adj[w3][w2]
        del adj[w3][r]
        del adj[r][w3]

    def remove_from(self, d: Diagram, web: Dict[Tuple[int, int], Pauli]) -> None:
        adj = {n1: {n2: True for n2 in d.neighbors(n1)} for n1 in d.node_indices()}
        for id_node in self.extra_id_nodes:
            self._remove_extra_id_node(adj, web, id_node)
        for hadamard in self.expanded_hadamards:
            self._remove_expanded_hadamard(adj, web, hadamard)


def _place_node_between(d: Diagram, _type: NodeType, n1: int, n2: int) -> int:
    node = d.add_node(_type)
    d.remove_edge(n1, n2)
    d.add_edges([(n1, node), (node, n2)])

    return node


_euler_decomposition_xzx = [NodeType.X, NodeType.Z, NodeType.X]
_euler_decomposition_zxz = [NodeType.Z, NodeType.X, NodeType.Z]


def _euler_expand_edges(d: Diagram) -> Iterable[ExpandedHadamard]:
    """A cut down version of pyzx.euler_expansion which does not add global scalars and does not prematurely 'merge'
    spiders.
    """

    def _decompose_between(_v1: int, _v2: int, *, _flip: bool) -> Tuple[int, int, int]:
        # Change decomposition to avoid introducing more X-spiders due to adjacent Z-spider
        pattern = _euler_decomposition_xzx if _flip else _euler_decomposition_zxz

        _w2 = _place_node_between(d, pattern[1], _v1, _v2)
        d.add_to_phase(_w2, Fraction(1, 2))
        _w1 = _place_node_between(d, pattern[0], _v1, _w2)
        d.add_to_phase(_w1, Fraction(1, 2))
        _w3 = _place_node_between(d, pattern[2], _w2, _v2)
        d.add_to_phase(_w3, Fraction(1, 2))

        return _w1, _w2, _w3

    expanded_hadamards = []
    for v in list(d.node_indices()):
        if d.type(v) != NodeType.H:
            continue

        v1, v2 = d.neighbors(v)

        d.remove_node(v)
        d.add_edge(v1, v2)

        flip = d.type(v1) == d.type(v2) and d.type(v1) == NodeType.X
        w1, w2, w3 = _decompose_between(v1, v2, _flip=flip)

        expanded_hadamards.append(ExpandedHadamard(w1, w2, w3, origin=v, flipped_decomposition=flip))

    return expanded_hadamards


def _ensure_red_green(d: Diagram) -> Iterable[int]:
    new_nodes = []
    # Introduce intermediate nodes
    for s, t in list(d.edge_list()):
        if d.type(s) == d.type(t):
            new_type = NodeType.Z if d.type(s) == NodeType.X else NodeType.X
            new_nodes.append(_place_node_between(d, new_type, s, t))

    # Introduce intermediate nodes for boundary <-> boundary connections
    for s, t in list(d.edge_list()):
        if d.type(s) == d.type(t) and d.type(s) == NodeType.B:
            new_nodes.append(_place_node_between(d, NodeType.X, s, t))

    # Ensure boundaries are not connected to a red spider
    boundaries = d.boundary_nodes()
    for boundary in boundaries:
        neighbour = list(d.neighbors(boundary))[0]
        if d.type(neighbour) == NodeType.X:
            new_nodes.append(_place_node_between(d, NodeType.Z, boundary, neighbour))

    # Ensure boundaries are not connected to green spiders with nonzero phase or more than one boundary connection
    for boundary in boundaries:
        neighbour = list(d.neighbors(boundary))[0]
        neighbour_boundaries = [v for v in d.neighbors(neighbour) if d.type(v) == NodeType.B]
        if d.phase(neighbour) != 0 or len(neighbour_boundaries) > 1:
            new_x = _place_node_between(d, NodeType.X, boundary, neighbour)
            new_nodes.append(new_x)
            new_nodes.append(_place_node_between(d, NodeType.Z, boundary, new_x))

    return new_nodes


def to_red_green_form(d: Diagram) -> AdditionalNodes:
    if d.has_parallel_edges():
        raise AssertionError("Can only work on diagrams containing no parallel edges.")

    # Convert all H-edges and Hadamards to red and green spiders
    additional_nodes = AdditionalNodes.empty()
    for hadamard in _euler_expand_edges(d):
        additional_nodes.add_expanded_hadamard(hadamard)

    # Verify that diagram is clifford
    offending_vertices = list(
        filter(
            lambda n: d.phase(n).denominator > 2
            or (d.type(n) != NodeType.Z and d.type(n) != NodeType.X and d.type(n) != NodeType.B),
            d.node_indices(),
        )
    )
    if len(offending_vertices) > 0:
        raise AssertionError(
            f"Given diagram is not a Clifford diagram up to hadamard expansion. The following "
            f"vertices are either not of type X,Z,BOUNDARY or have a non-clifford "
            f"phase: {', '.join(map(str, offending_vertices))}"
        )

    for node in _ensure_red_green(d):
        additional_nodes.add_extra_id_node(node)

    return additional_nodes
