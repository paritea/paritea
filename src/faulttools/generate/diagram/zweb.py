import pyzx as zx
from pyzx.graph.graph_s import GraphS


def zweb(qubits: int, depth: int, position_factor: int = 1) -> GraphS:
    """
    Generate a rectangle of interconnected Z spiders, which has the useful trait that every possible sub-rectangle forms
    a detecting region for X-errors.
    """

    g = GraphS()

    inputs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit=q * position_factor, row=0) for q in range(qubits)]
    internal_spiders = [
        [g.add_vertex(zx.VertexType.Z, qubit=q * position_factor, row=r * position_factor + 1) for q in range(qubits)]
        for r in range(depth)
    ]
    outputs = [
        g.add_vertex(zx.VertexType.BOUNDARY, qubit=q * position_factor, row=depth * position_factor + 1)
        for q in range(qubits)
    ]
    all_spiders = [inputs, *internal_spiders, outputs]

    edges = [(all_spiders[r][q], all_spiders[r + 1][q]) for r in range(depth + 1) for q in range(qubits)]
    edges.extend([(all_spiders[r + 1][q], all_spiders[r + 1][q + 1]) for q in range(qubits - 1) for r in range(depth)])
    g.add_edges(edges, zx.EdgeType.SIMPLE)

    g.auto_detect_io()

    return g
