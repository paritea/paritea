from typing import List

import stim

from faulttools.diagram import NodeType, Diagram


def convert_circuit(circuit: stim.Circuit) -> Diagram:
    n = circuit.num_qubits
    d = Diagram()
    row_offset = 0
    # Input boundaries
    current_qubit_nodes: List[int | None] = []
    for idx in range(n):
        current_qubit_nodes.append(d.add_node(NodeType.B, x=row_offset, y=idx))
    row_offset += 1

    for instr in circuit:
        match instr.name:
            # Regular gates
            case "H":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    h = d.add_node(NodeType.H, x=row_offset, y=tar.qubit_value)
                    d.add_edges([(current_qubit_nodes[tar.qubit_value], h)])
                    current_qubit_nodes[tar.qubit_value] = h
                row_offset += 1
            case "CX" | "CNOT" | "ZCX":
                for group in instr.target_groups():
                    assert len(group) == 2
                    for tar in group:
                        assert tar.is_qubit_target and not tar.is_inverted_result_target

                    control = d.add_node(NodeType.Z, x=row_offset, y=group[0].qubit_value)
                    target = d.add_node(NodeType.X, x=row_offset, y=group[1].qubit_value)
                    d.add_edges(
                        [
                            (current_qubit_nodes[group[0].qubit_value], control),
                            (current_qubit_nodes[group[1].qubit_value], target),
                            (control, target),
                        ]
                    )
                    current_qubit_nodes[group[0].qubit_value] = control
                    current_qubit_nodes[group[1].qubit_value] = target
                    row_offset += 1
            # Measurements / Resets
            case "R" | "RZ" | "MR" | "MRZ" | "M" | "MZ":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    # Silent measurement # TODO this is postselected
                    m = d.add_node(NodeType.X, x=row_offset, y=tar.qubit_value)
                    d.add_edge(current_qubit_nodes[tar.qubit_value], m)
                    r = d.add_node(NodeType.X, x=row_offset + 1, y=tar.qubit_value)
                    current_qubit_nodes[tar.qubit_value] = r

                row_offset += 2
            # Error mechanisms
            case "X_ERROR":
                pass
            case "DEPOLARIZE1":
                pass
            case "DEPOLARIZE2":
                pass
            # Flow generator indicators
            case "DETECTOR":
                pass
            case "OBSERVABLE_INCLUDE":
                pass
            # Irrelevant instructions
            case "TICK":
                continue
            case "QUBIT_COORDS":
                continue
            # Incompatibilities
            case "REPEAT":
                raise NotImplementedError(
                    "The REPEAT instruction is not yet supported. Use circuit.flattened() to avoid repeat blocks."
                )
            case _:
                raise NotImplementedError(f"The instruction {instr.name} is not supported.")

    # Output boundaries
    for idx in range(n):
        b = d.add_node(NodeType.B, x=row_offset, y=idx)
        d.add_edge(current_qubit_nodes[idx], b)
    row_offset += 1

    return d
