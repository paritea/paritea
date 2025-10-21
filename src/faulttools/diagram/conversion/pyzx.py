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


def from_pyzx(pyzx_graph: BaseGraph) -> Diagram:
    diagram = Diagram()

    vertex_to_id = {}
    for v in pyzx_graph.vertices():
        if not isinstance(v, int):
            raise ValueError(f"Unsupported PyZX vertex instance: {type(v)}")

        v_type = pyzx_graph.type(v)
        if v_type not in pyzx_v_type_to_node_type:
            raise ValueError(f"Unsupported PyZX vertex type: {v_type}")

        v_phase = pyzx_graph.phase(v)
        if v_phase != 0 and not isinstance(v_phase, Fraction):
            raise ValueError(f"Unsupported PyZX vertex phase: {v_phase} for vertex {v}")

        vertex_to_id[v] = diagram.add_node(NodeInfo(type=pyzx_v_type_to_node_type[v_type], phase=v_phase))

    for edge in pyzx_graph.edges():
        source, target = pyzx_graph.edge_st(edge)
        e_type = pyzx_graph.edge_type(edge)

        if e_type != PyZxEdgeType.SIMPLE:
            raise ValueError(f"Unsupported PyZX edge type: {e_type}")

        diagram.add_edge(source, target, None)

    return diagram
