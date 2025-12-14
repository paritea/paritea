import math
from typing import List, Tuple

import stim

from faulttools import Pauli
from faulttools.diagram import NodeType, Diagram
from faulttools.noise import Fault, NoiseModel


def from_stim(circuit: stim.Circuit) -> Tuple[Diagram, NoiseModel]:
    n = circuit.num_qubits
    d = Diagram()
    row_offset = 0
    # Everything starts in |0> state
    current_qubit_nodes: List[int | None] = []
    for idx in range(n):
        current_qubit_nodes.append(d.add_node(NodeType.X, x=row_offset, y=idx))
    initial_nodes = current_qubit_nodes.copy()
    row_offset += 1

    atomic_faults: List[Tuple[Fault, float]] = []
    scheduled_pauli_faults_per_qubit: List[List[Tuple[Pauli, float]]] = [[] for _ in range(n)]

    def flush_faults_on_qubit(q: int, edge: int) -> None:
        for fault, prob in scheduled_pauli_faults_per_qubit[q]:
            atomic_faults.append((Fault.edge_flip(edge, fault), prob))
        scheduled_pauli_faults_per_qubit[q] = []

    for instr in circuit:
        match instr.name:
            # Regular gates
            case "H":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    h = d.add_node(NodeType.H, x=row_offset, y=tar.qubit_value)
                    e = d.add_edge(current_qubit_nodes[tar.qubit_value], h)
                    current_qubit_nodes[tar.qubit_value] = h
                    flush_faults_on_qubit(tar.qubit_value, e)
                row_offset += 1
            case "CX" | "CNOT" | "ZCX":
                for group in instr.target_groups():
                    assert len(group) == 2
                    for tar in group:
                        assert tar.is_qubit_target and not tar.is_inverted_result_target

                    q_control = group[0].qubit_value
                    q_target = group[1].qubit_value
                    control = d.add_node(NodeType.Z, x=row_offset, y=q_control)
                    target = d.add_node(NodeType.X, x=row_offset, y=q_target)
                    e_control, e_target, _ = d.add_edges(
                        [
                            (current_qubit_nodes[q_control], control),
                            (current_qubit_nodes[q_target], target),
                            (control, target),
                        ]
                    )
                    current_qubit_nodes[q_control] = control
                    current_qubit_nodes[q_target] = target
                    flush_faults_on_qubit(q_control, e_control)
                    flush_faults_on_qubit(q_target, e_target)
                    row_offset += 1
            # Measurements / Resets
            case "R" | "RZ" | "MR" | "MRZ" | "M" | "MZ":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    # Silent measurement # TODO this is postselected
                    m = d.add_node(NodeType.X, x=row_offset, y=tar.qubit_value)
                    e = d.add_edge(current_qubit_nodes[tar.qubit_value], m)
                    flush_faults_on_qubit(tar.qubit_value, e)
                    r = d.add_node(NodeType.X, x=row_offset + 1, y=tar.qubit_value)
                    current_qubit_nodes[tar.qubit_value] = r

                row_offset += 2
            # Error mechanisms
            case "X_ERROR":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    scheduled_pauli_faults_per_qubit[tar.qubit_value].append((Pauli.X, args[0]))
            case "Y_ERROR":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    scheduled_pauli_faults_per_qubit[tar.qubit_value].append((Pauli.Y, args[0]))
            case "Z_ERROR":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    scheduled_pauli_faults_per_qubit[tar.qubit_value].append((Pauli.Z, args[0]))
            case "DEPOLARIZE1":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    p = args[0]
                    if p > 0.75:
                        raise RuntimeError("Cannot approximate single-qubit depolarizing channel with p > 0.75!")
                    # Apply the magic formula
                    independent_p = 0.5 - 0.5 * math.sqrt(1 - 4 / 3 * p)
                    scheduled_pauli_faults_per_qubit[tar.qubit_value].append((Pauli.X, independent_p))
                    scheduled_pauli_faults_per_qubit[tar.qubit_value].append((Pauli.Y, independent_p))
                    scheduled_pauli_faults_per_qubit[tar.qubit_value].append((Pauli.Z, independent_p))
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

    # Output boundaries; remove unused qubit world lines
    for idx in range(n):
        if current_qubit_nodes[idx] == initial_nodes[idx]:
            d.remove_node(initial_nodes[idx])
            current_qubit_nodes[idx] = None
        else:
            b = d.add_node(NodeType.B, x=row_offset, y=idx)
            d.add_edge(current_qubit_nodes[idx], b)
            current_qubit_nodes[idx] = b
    row_offset += 1
    d.set_io(
        [], [b for b in current_qubit_nodes if b is not None], virtual=False
    )  # TODO throw when a qubit is used but not measured at the end

    # Build noise model
    noise = NoiseModel(d, atomic_faults)

    return d, noise
