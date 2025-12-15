from fractions import Fraction
from typing import Dict, Protocol, Tuple, Union

from pyzx import EdgeType as PyZxEdgeType
from pyzx import Graph
from pyzx import VertexType as PyZxVertexType
from pyzx.graph.base import BaseGraph

from ..diagram import Diagram, NodeType

pyzx_v_type_to_node_type = {
    PyZxVertexType.BOUNDARY: NodeType.B,
    PyZxVertexType.Z: NodeType.Z,
    PyZxVertexType.X: NodeType.X,
    PyZxVertexType.H_BOX: NodeType.H,
}

node_type_to_pyzx_v_type = {
    NodeType.B: PyZxVertexType.BOUNDARY,
    NodeType.Z: PyZxVertexType.Z,
    NodeType.X: PyZxVertexType.X,
    NodeType.H: PyZxVertexType.H_BOX,
}


class SupportsPyZXIndex(Protocol):
    def set_pyzx_index(self, node_idx: int, pyzx_idx: int) -> None: ...
    def pyzx_index(self, node_idx: int) -> int: ...


class DiagramWithPyZXIndex(Diagram, SupportsPyZXIndex, Protocol):
    pass


def from_pyzx(pyzx_graph: BaseGraph, convert_had_edges: bool = False) -> Diagram:
    """
    Note: To obtain a graph that is convertible using 'to_pyzx' see 'from_pyzx_reversible'.

    :param pyzx_graph: The PyZX graph to convert to a diagram.
    :param convert_had_edges: Whether to handle hadamard edges via conversion to H-Boxes (True) or throwing (False).
    :return: The converted diagram.
    """
    return _from_pyzx(pyzx_graph, convert_had_edges, reversible=False)


def from_pyzx_reversible(pyzx_graph: BaseGraph, convert_had_edges: bool = False) -> DiagramWithPyZXIndex:
    """
    :param pyzx_graph: The PyZX graph to convert to a diagram.
    :param convert_had_edges: Whether to handle hadamard edges via conversion to H-Boxes (True) or throwing (False).
    :return: The converted diagram.
    """
    return _from_pyzx(pyzx_graph, convert_had_edges, reversible=True)


def _from_pyzx(
    pyzx_graph: BaseGraph, convert_had_edges: bool = False, reversible: bool = False
) -> Union[Diagram, DiagramWithPyZXIndex]:
    if reversible:
        diagram: DiagramWithPyZXIndex = Diagram(additional_keys=["pyzx_index"])
    else:
        diagram: Diagram = Diagram()

    vertex_to_id = {}
    for v in pyzx_graph.vertices():
        if not isinstance(v, int):
            raise ValueError(f"Unsupported PyZX vertex instance: {type(v)}")

        v_type = pyzx_graph.type(v)
        if v_type not in pyzx_v_type_to_node_type:
            raise ValueError(f"Unsupported PyZX vertex type: {v_type.name}")

        v_phase = pyzx_graph.phase(v)
        if Fraction(v_phase, 1).denominator > 2:
            raise ValueError(f"Unsupported PyZX vertex phase: {v_phase} for vertex {v}")

        node = diagram.add_node(pyzx_v_type_to_node_type[v_type], phase=v_phase)
        if reversible:
            diagram.set_x(node, pyzx_graph.qubit(v)).set_y(node, pyzx_graph.row(v)).set_pyzx_index(node, v)
        vertex_to_id[v] = node

    for edge in pyzx_graph.edges():
        source, target = pyzx_graph.edge_st(edge)
        e_type = pyzx_graph.edge_type(edge)

        if e_type == PyZxEdgeType.SIMPLE:
            diagram.add_edge(vertex_to_id[source], vertex_to_id[target])
        elif e_type == PyZxEdgeType.HADAMARD:
            if convert_had_edges:
                h = diagram.add_node(NodeType.H)
                diagram.add_edge(vertex_to_id[source], h)
                diagram.add_edge(h, vertex_to_id[target])
            else:
                raise ValueError(
                    f"Unsupported PyZX edge type: {e_type.name}. Try explicitly converting the PyZX diagram and passing convert_had_edges=True."
                )
        else:
            raise ValueError(f"Unsupported PyZX edge type: {e_type.name}")

    if len(pyzx_graph.inputs()) > 0 or len(pyzx_graph.outputs()) > 0:
        diagram.set_io(
            inputs=[vertex_to_id[i] for i in pyzx_graph.inputs()],
            outputs=[vertex_to_id[o] for o in pyzx_graph.outputs()],
            virtual=False,
        )

    return diagram


def to_pyzx(d: Diagram, with_mapping: bool = False) -> Union[BaseGraph, Tuple[BaseGraph, Dict[int, int]]]:
    """
    Constructs a PyZX diagram from the given diagram instance, reassigning original node ids and positions. Does not
    convert original hadamard edges back.

    :returns: The PyZX diagram and a mapping from the original node indices to the new node indices.
    """

    g = Graph(backend="simple")

    mapping: Dict[int, int] = {}

    for n in d.node_indices():
        if hasattr(d, "pyzx_index"):
            pyzx_id = d.pyzx_index(n)
            g.add_vertex_indexed(pyzx_id)
        else:
            pyzx_id = g.add_vertex()
        mapping[n] = pyzx_id
        g.set_type(pyzx_id, node_type_to_pyzx_v_type[d.type(n)])
        g.set_qubit(pyzx_id, d.y(n))
        g.set_row(pyzx_id, d.x(n))

    g.add_edge_table({(mapping[s], mapping[t]): [1, 0] for s, t in d.edge_list()})

    if with_mapping:
        return g, mapping
    else:
        return g
