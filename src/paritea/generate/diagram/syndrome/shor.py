from typing import Literal, overload

from mpl_toolkits.axes_grid1.axes_size import Fraction

from paritea import Pauli, PauliString
from paritea.diagram import Diagram, NodeType


@overload
def shor_extraction(
    stabilisers: list[PauliString],
    *,
    qubits: int,
    repeat: int = 1,
    partition: Literal[False] = False,
    granular: bool = False,
) -> Diagram: ...
@overload
def shor_extraction(
    stabilisers: list[PauliString],
    *,
    qubits: int,
    repeat: int = 1,
    partition: Literal[True],
    granular: bool = False,
) -> tuple[Diagram, list[list[int]]]: ...
def shor_extraction(
    stabilisers: list[PauliString],
    *,
    qubits: int,
    repeat: int = 1,
    partition: bool = False,
    granular: bool = False,
) -> Diagram | tuple[Diagram, list[list[int]]]:
    """
    Generates a ZX diagram measuring the given stabilisers one-by-one using Shor-style syndrome extraction. Diagrams are
    post-selected on all-zero extraction measurements.

    Currently only supports measuring stabilisers made up of X and Z. Uses an assumed fault-free cat state preparation.

    :param partition: Whether to return partitions for the diagram.
    :param granular: Whether to take a partition to be an entire measurement round (False) or an individual stabiliser
    measurement (True).
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
            first = [-1 for _ in range(qubits)]
            last = [-1 for _ in range(qubits)]
            controls = [-1 for _ in range(qubits)]

            cat_z = stabiliser_diagram.add_node(NodeType.Z, x=row_offset, y=qubits + 2 + qubits / 2)
            row_offset += 1

            # Generate Pauli boxes
            for idx, pauli in stabiliser.items():
                if pauli == Pauli.I:
                    continue

                target_node = stabiliser_diagram.add_node(NodeType.X, x=row_offset, y=idx)
                c = stabiliser_diagram.add_node(NodeType.Z, x=row_offset, y=idx + qubits + 1)
                stabiliser_diagram.add_edge(target_node, c)

                if pauli == Pauli.X:
                    h1 = stabiliser_diagram.add_node(NodeType.H, x=row_offset - 0.5, y=idx)
                    h2 = stabiliser_diagram.add_node(NodeType.H, x=row_offset + 0.5, y=idx)
                    stabiliser_diagram.add_edge(target_node, h1)
                    stabiliser_diagram.add_edge(target_node, h2)
                    first[idx] = h1
                    last[idx] = h2
                elif pauli == Pauli.Y:
                    x1 = stabiliser_diagram.add_node(NodeType.X, phase=Fraction(1, 2), x=row_offset - 0.5, y=idx)
                    x2 = stabiliser_diagram.add_node(NodeType.X, phase=Fraction(-1, 2), x=row_offset + 0.5, y=idx)
                    stabiliser_diagram.add_edge(target_node, x1)
                    stabiliser_diagram.add_edge(target_node, x2)
                    first[idx] = x1
                    last[idx] = x2
                else:
                    first[idx] = target_node
                    last[idx] = target_node

                controls[idx] = c
                row_offset += 1

            # Connect to cat state
            for i, c in enumerate(controls):
                h = stabiliser_diagram.add_node(NodeType.H, x=row_offset, y=qubits + i + 1)
                measure = stabiliser_diagram.add_node(NodeType.X, x=row_offset + 1, y=qubits + i + 1)
                stabiliser_diagram.add_edge(h, measure)
                if c == -1:
                    stabiliser_diagram.add_edge(cat_z, h)
                else:
                    stabiliser_diagram.add_edge(cat_z, c)
                    stabiliser_diagram.add_edge(c, h)

            row_offset += 2

            # Append to overall diagram
            trans = d.compose(stabiliser_diagram, {b: n for b, n in zip(current_qubit_nodes, first) if n != -1})
            current_qubit_nodes = [trans[n] if n != -1 else current_qubit_nodes[i] for i, n in enumerate(last)]
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
