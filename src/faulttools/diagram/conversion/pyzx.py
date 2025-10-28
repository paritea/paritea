from fractions import Fraction

from pyzx.graph.base import BaseGraph
from pyzx import VertexType as PyZxVertexType, EdgeType as PyZxEdgeType

from ..diagram import Diagram, NodeType, NodeInfo

pyzx_v_type_to_node_type = {
    PyZxVertexType.BOUNDARY: NodeType.B,
    PyZxVertexType.Z: NodeType.Z,
    PyZxVertexType.X: NodeType.X,
    PyZxVertexType.H_BOX: NodeType.H,
}


def from_pyzx(pyzx_graph: BaseGraph, convert_had_edges: bool = False) -> Diagram:
    """
    :param pyzx_graph: The PyZX graph to convert to a diagram.
    :param convert_had_edges: Whether to handle hadamard edges via conversion to H-Boxes (True) or throwing (False).
    :return: The converted diagram.
    """

    diagram = Diagram()

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

        vertex_to_id[v] = diagram.add_node(NodeInfo(type=pyzx_v_type_to_node_type[v_type], phase=v_phase))

    for edge in pyzx_graph.edges():
        source, target = pyzx_graph.edge_st(edge)
        e_type = pyzx_graph.edge_type(edge)

        if e_type == PyZxEdgeType.SIMPLE:
            diagram.add_edge(vertex_to_id[source], vertex_to_id[target], None)
        elif e_type == PyZxEdgeType.HADAMARD:
            if convert_had_edges:
                h = diagram.add_node(NodeInfo(type=NodeType.H))
                diagram.add_edge(vertex_to_id[source], h, None)
                diagram.add_edge(h, vertex_to_id[target], None)
            else:
                raise ValueError(
                    f"Unsupported PyZX edge type: {e_type.name}. Try explicitly converting the PyZX diagram and passing convert_had_edges=True."
                )
        else:
            raise ValueError(f"Unsupported PyZX edge type: {e_type.name}")

    return diagram
