from copy import deepcopy
from typing import List, Tuple

from faulttools import PauliString, Pauli
from faulttools.diagram import Diagram, NodeInfo, NodeType


def generate_shor_extraction(stabilisers: List[PauliString], qubits: int, repeat: int = 1) -> Diagram:
    """
    Generates a ZX diagram measuring the given stabilisers one-by-one using Shor-style syndrome extraction. Diagrams are
    post-selected on all-zero extraction measurements.

    Currently only supports measuring X and Z on single qubits. Uses an assumed fault-free cat state preparation.
    """

    stabiliser_mapping: List[Tuple[Diagram, List[int]]] = []
    for stabiliser in stabilisers:
        stabiliser_diagram = Diagram()
        mapped = [-1 for _ in range(qubits)]
        controls = [-1 for _ in range(qubits)]

        cat_x = stabiliser_diagram.add_node(NodeInfo(NodeType.X))

        # Generate Pauli boxes
        for idx, pauli in stabiliser.items():
            if pauli == Pauli.I:
                continue
            elif pauli == Pauli.Y:
                raise ValueError("Measuring Pauli Y is not supported!")

            target_node = stabiliser_diagram.add_node(NodeInfo(NodeType.Z))
            c = stabiliser_diagram.add_node(NodeInfo(NodeType.X))
            if pauli == Pauli.X:
                # X-Pauli is provided with retracted H-box to facilitate single nodes in array
                h = stabiliser_diagram.add_node(NodeInfo(NodeType.H))
                stabiliser_diagram.add_edge(target_node, h, None)
                stabiliser_diagram.add_edge(h, c, None)
            else:
                stabiliser_diagram.add_edge(target_node, c, None)

            mapped[idx] = target_node
            controls[idx] = c

        # Connect to cat state
        for c in controls:
            measure = stabiliser_diagram.add_node(NodeInfo(NodeType.X))
            if c == -1:
                stabiliser_diagram.add_edge(cat_x, measure, None)
            else:
                stabiliser_diagram.add_edge(cat_x, c, None)
                stabiliser_diagram.add_edge(c, measure, None)

        stabiliser_mapping.append((stabiliser_diagram, mapped))

    d = Diagram()
    current_qubit_nodes = [d.add_node(NodeInfo(NodeType.B)) for _ in range(qubits)]
    for _ in range(repeat):
        for stabiliser_diagram, mapping in stabiliser_mapping:
            trans = d.compose(
                deepcopy(stabiliser_diagram), {b: (n, None) for b, n in zip(current_qubit_nodes, mapping) if n != -1}
            )
            current_qubit_nodes = [trans[n] if n != -1 else current_qubit_nodes[i] for i, n in enumerate(mapping)]

    # Add boundary nodes on the other side
    for i in range(qubits):
        n = d.add_node(NodeInfo(NodeType.B))
        d.add_edge(current_qubit_nodes[i], n, None)

    return d
