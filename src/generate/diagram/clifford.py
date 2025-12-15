import random
from fractions import Fraction
from typing import Optional

import pyzx as zx
from pyzx.graph.graph_s import GraphS


def _random_clifford_phase():
    """Return a random Clifford phase (multiple of π/2)."""
    return random.choice([0, Fraction(1, 2), 1, Fraction(3, 2)])


def _add_random_spider(graph, spider_type, qubit=None):
    """Add a spider of given type ('Z' or 'X') to the graph."""
    phase = _random_clifford_phase()
    vtype = zx.VertexType.Z if spider_type == "Z" else zx.VertexType.X
    return graph.add_vertex(ty=vtype, phase=phase, qubit=qubit, row=0.5)


def _connect_spiders(graph: GraphS, v1, v2, hadamard=False):
    if hadamard:
        h = graph.add_vertex(zx.VertexType.H_BOX)
        graph.add_edges([(v1, h), (h, v2)])
    else:
        graph.add_edge((v1, v2), zx.EdgeType.SIMPLE)


def clifford(qubits: Optional[int] = None, spiders: Optional[int] = None) -> GraphS:
    """Generate a random clifford ZX-diagram, possibly non-unitary."""
    if qubits is None:
        qubits = random.randint(2, 5)
    if spiders is None:
        spiders = random.randint(5, 15)

    g = GraphS()

    # Create inputs and outputs
    inputs = [g.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=i, row=0) for i in range(qubits)]
    outputs = [g.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=i, row=1) for i in range(qubits)]

    # Create input-layer Z-spiders (right after inputs)
    input_zs = [g.add_vertex(ty=zx.VertexType.Z, qubit=i, row=0.1) for i in range(qubits)]
    for i in range(qubits):
        _connect_spiders(g, inputs[i], input_zs[i])

    # Create output-layer Z-spiders (right before outputs)
    output_zs = [g.add_vertex(ty=zx.VertexType.Z, qubit=i, row=0.9) for i in range(qubits)]
    for i in range(qubits):
        _connect_spiders(g, output_zs[i], outputs[i])

    # Create internal Clifford spiders
    internal_spiders = []
    for _ in range(spiders):
        internal_spiders.append(_add_random_spider(g, random.choice(["Z", "X"]), qubit=random.randint(0, qubits - 1)))

    # Connect some spiders randomly
    all_spiders = input_zs + internal_spiders + output_zs
    for _ in range(spiders * 2):
        v1, v2 = random.sample(all_spiders, 2)
        _connect_spiders(g, v1, v2, hadamard=random.random() < 0.3)

    if random.random() < 0.5:
        # Remove some outputs or inputs
        if random.random() < 0.5 and outputs:
            v = random.choice(outputs)
            g.remove_vertex(v)
            outputs.remove(v)
        else:
            if inputs:
                v = random.choice(inputs)
                g.remove_vertex(v)
                inputs.remove(v)

    return g
