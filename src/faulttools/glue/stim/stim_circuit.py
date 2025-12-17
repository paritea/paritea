import math
from dataclasses import dataclass

import stim

from faulttools.diagram import Diagram, NodeType
from faulttools.noise import Fault, NoiseModel
from faulttools.pauli import Pauli


@dataclass(init=True, kw_only=True)
class _DiagramBuildingState:
    n_qubits: int
    current_qubit_nodes: list[int | None]
    row_offset: int


def _h(d: Diagram, state: _DiagramBuildingState, instr: stim.CircuitInstruction) -> list[tuple[int, int]]:
    updated_qubits = []
    for tar in instr.targets_copy():
        assert tar.is_qubit_target and not tar.is_inverted_result_target
        h = d.add_node(NodeType.H, x=state.row_offset, y=tar.qubit_value)
        e = d.add_edge(state.current_qubit_nodes[tar.qubit_value], h)
        state.current_qubit_nodes[tar.qubit_value] = h
        updated_qubits.append((tar.qubit_value, e))
    state.row_offset += 1

    return updated_qubits


def _cnot(d: Diagram, state: _DiagramBuildingState, instr: stim.CircuitInstruction) -> list[tuple[int, int]]:
    updated_qubits = []
    for group in instr.target_groups():
        assert len(group) == 2
        for tar in group:
            assert tar.is_qubit_target and not tar.is_inverted_result_target

        q_control = group[0].qubit_value
        q_target = group[1].qubit_value
        control = d.add_node(NodeType.Z, x=state.row_offset, y=q_control)
        target = d.add_node(NodeType.X, x=state.row_offset, y=q_target)
        e_control, e_target, _ = d.add_edges(
            [
                (state.current_qubit_nodes[q_control], control),
                (state.current_qubit_nodes[q_target], target),
                (control, target),
            ]
        )
        state.current_qubit_nodes[q_control] = control
        state.current_qubit_nodes[q_target] = target
        updated_qubits.append((q_control, e_control))
        updated_qubits.append((q_target, e_target))
        state.row_offset += 1

    return updated_qubits


def from_stim(circuit: stim.Circuit) -> tuple[Diagram, NoiseModel]:
    d = Diagram()
    state = _DiagramBuildingState(
        n_qubits=circuit.num_qubits,
        row_offset=0,
        current_qubit_nodes=[],
    )
    # Everything starts in |0> state
    state.current_qubit_nodes = [d.add_node(NodeType.X, x=0, y=idx) for idx in range(state.n_qubits)]
    initial_nodes = state.current_qubit_nodes.copy()
    state.row_offset += 1

    atomic_faults: list[tuple[Fault, float]] = []
    scheduled_pauli_faults_per_qubit: list[list[tuple[Pauli, float]]] = [[] for _ in range(state.n_qubits)]

    def flush_faults_on_qubit(q: int, edge: int) -> None:
        for fault, prob in scheduled_pauli_faults_per_qubit[q]:
            atomic_faults.append((Fault.edge_flip(edge, fault), prob))
        scheduled_pauli_faults_per_qubit[q] = []

    for instr in circuit:
        match instr.name:
            # Regular gates
            case "H":
                for qubit, e in _h(d, state, instr):
                    flush_faults_on_qubit(qubit, e)
            case "CX" | "CNOT" | "ZCX":
                for qubit, e in _cnot(d, state, instr):
                    flush_faults_on_qubit(qubit, e)
            # Measurements / Resets
            case "R" | "RZ" | "MR" | "MRZ" | "M" | "MZ":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    # Silent measurement # TODO this is postselected
                    m = d.add_node(NodeType.X, x=state.row_offset, y=tar.qubit_value)
                    e = d.add_edge(state.current_qubit_nodes[tar.qubit_value], m)
                    flush_faults_on_qubit(tar.qubit_value, e)
                    r = d.add_node(NodeType.X, x=state.row_offset + 1, y=tar.qubit_value)
                    state.current_qubit_nodes[tar.qubit_value] = r

                state.row_offset += 2
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
            case "TICK" | "QUBIT_COORDS":
                continue
            # Incompatibilities
            case "REPEAT":
                raise NotImplementedError(
                    "The REPEAT instruction is not yet supported. Use circuit.flattened() to avoid repeat blocks."
                )
            case _:
                raise NotImplementedError(f"The instruction {instr.name} is not supported.")

    # Output boundaries; remove unused qubit world lines
    for idx in range(state.n_qubits):
        if state.current_qubit_nodes[idx] == initial_nodes[idx]:
            d.remove_node(initial_nodes[idx])
            state.current_qubit_nodes[idx] = None
        else:
            b = d.add_node(NodeType.B, x=state.row_offset, y=idx)
            d.add_edge(state.current_qubit_nodes[idx], b)
            state.current_qubit_nodes[idx] = b
    state.row_offset += 1
    d.set_io(
        [], [b for b in state.current_qubit_nodes if b is not None], virtual=False
    )  # TODO throw when a qubit is used but not measured at the end

    # Build noise model
    noise = NoiseModel(d, atomic_faults)

    return d, noise
