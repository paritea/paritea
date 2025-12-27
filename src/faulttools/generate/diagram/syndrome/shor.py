from typing import Literal, overload

from faulttools import Pauli, PauliString
from faulttools.diagram import Diagram, NodeType


@overload
def shor_extraction(
    stabilisers: list[PauliString],
    *,
    qubits: int,
    repeat: int = 1,
    granular: bool = False,
    partition: Literal[False] = False,
) -> Diagram: ...
@overload
def shor_extraction(
    stabilisers: list[PauliString],
    *,
    qubits: int,
    repeat: int = 1,
    granular: bool = False,
    partition: Literal[True],
) -> tuple[Diagram, list[list[int]]]: ...
def shor_extraction(
    stabilisers: list[PauliString],
    *,
    qubits: int,
    repeat: int = 1,
    granular: bool = False,
    partition: bool = False,
) -> Diagram | tuple[Diagram, list[list[int]]]:
    """
    Generates a ZX diagram measuring the given stabilisers one-by-one using Shor-style syndrome extraction. Diagrams are
    post-selected on all-zero extraction measurements.

    Currently only supports measuring X and Z on single qubits. Uses an assumed fault-free cat state preparation.
    """

    d = Diagram()
    row_offset = 0
    # Initial row of boundaries
    current_qubit_nodes = [d.add_node(NodeType.B, x=row_offset, y=i) for i in range(qubits)]
    inputs = current_qubit_nodes.copy()
    node_list: list[list[int]] = []
    row_offset += 1

    for _ in range(repeat):
        new_nodes = []
        for stabiliser in stabilisers:
            stabiliser_diagram = Diagram()
            mapped = [-1 for _ in range(qubits)]
            controls = [-1 for _ in range(qubits)]

            cat_x = stabiliser_diagram.add_node(NodeType.X, x=row_offset, y=qubits + 2 + qubits / 2)
            row_offset += 1

            # Generate Pauli boxes
            for idx, pauli in stabiliser.items():
                if pauli == Pauli.I:
                    continue
                elif pauli == Pauli.Y:
                    raise ValueError("Measuring Pauli Y is not supported!")

                target_node = stabiliser_diagram.add_node(NodeType.Z, x=row_offset, y=idx)
                c = stabiliser_diagram.add_node(NodeType.X, x=row_offset, y=idx + qubits + 2)
                if pauli == Pauli.X:
                    # X-Pauli is provided with retracted H-box to facilitate single nodes in array
                    h = stabiliser_diagram.add_node(NodeType.H, x=row_offset, y=qubits + 1)
                    stabiliser_diagram.add_edge(target_node, h)
                    stabiliser_diagram.add_edge(h, c)
                else:
                    stabiliser_diagram.add_edge(target_node, c)

                mapped[idx] = target_node
                controls[idx] = c
                row_offset += 1

            # Connect to cat state
            for i, c in enumerate(controls):
                measure = stabiliser_diagram.add_node(NodeType.X, x=row_offset, y=qubits + i + 2)
                if c == -1:
                    stabiliser_diagram.add_edge(cat_x, measure)
                else:
                    stabiliser_diagram.add_edge(cat_x, c)
                    stabiliser_diagram.add_edge(c, measure)

            row_offset += 1

            # Append to overall diagram
            trans = d.compose(stabiliser_diagram, {b: n for b, n in zip(current_qubit_nodes, mapped) if n != -1})
            current_qubit_nodes = [trans[n] if n != -1 else current_qubit_nodes[i] for i, n in enumerate(mapped)]
            if granular:
                node_list.append(list(trans.values()))
            else:
                new_nodes.extend(trans.values())

        if not granular:
            node_list.append(new_nodes)

    # Add boundary nodes on the other side
    for i in range(qubits):
        b = d.add_node(NodeType.B, x=row_offset, y=i)
        d.add_edge(current_qubit_nodes[i], b)
        current_qubit_nodes[i] = b

    outputs = current_qubit_nodes.copy()
    d.set_io(inputs, outputs, virtual=False)

    if partition:
        return d, node_list

    return d
