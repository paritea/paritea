import math
from dataclasses import dataclass

import stim

from faulttools.diagram import Diagram, NodeType
from faulttools.noise import Fault, NoiseModel
from faulttools.pauli import Pauli


@dataclass(init=True, kw_only=True)
class _DiagramBuildingState:
    n_qubits: int
    d: Diagram
    current_qubit_nodes: list[int | None]
    row_offset: int

    def is_initialized(self, qubit: int) -> bool:
        return self.current_qubit_nodes[qubit] is not None

    def ensure_initialized(self, *qubits: int) -> None:
        for qubit in qubits:
            if self.is_initialized(qubit):
                return
            # Everything starts in |0> state in stim
            self.current_qubit_nodes[qubit] = self.d.add_node(NodeType.X, x=self.row_offset - 1, y=qubit)


def _h(state: _DiagramBuildingState, instr: stim.CircuitInstruction) -> list[tuple[int, int]]:
    updated_qubits = []
    for tar in instr.targets_copy():
        assert tar.is_qubit_target and not tar.is_inverted_result_target
        q = tar.qubit_value
        state.ensure_initialized(q)
        h = state.d.add_node(NodeType.H, x=state.row_offset, y=q)
        e = state.d.add_edge(state.current_qubit_nodes[q], h)
        state.current_qubit_nodes[q] = h
        updated_qubits.append((q, e))
    state.row_offset += 1

    return updated_qubits


def _cnot(state: _DiagramBuildingState, instr: stim.CircuitInstruction) -> list[tuple[int, int]]:
    updated_qubits = []
    for group in instr.target_groups():
        assert len(group) == 2
        for tar in group:
            assert tar.is_qubit_target and not tar.is_inverted_result_target

        q_control = group[0].qubit_value
        q_target = group[1].qubit_value
        state.ensure_initialized(q_control, q_target)
        control = state.d.add_node(NodeType.Z, x=state.row_offset, y=q_control)
        target = state.d.add_node(NodeType.X, x=state.row_offset, y=q_target)
        e_control, e_target, _ = state.d.add_edges(
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
    state = _DiagramBuildingState(
        n_qubits=circuit.num_qubits,
        d=Diagram(),
        row_offset=1,
        current_qubit_nodes=[None for _ in range(circuit.num_qubits)],
    )

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
                for qubit, e in _h(state, instr):
                    flush_faults_on_qubit(qubit, e)
            case "CX" | "CNOT" | "ZCX":
                for qubit, e in _cnot(state, instr):
                    flush_faults_on_qubit(qubit, e)
            # Measurements / Resets
            case "R" | "RZ":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    q = tar.qubit_value
                    if state.is_initialized(q):
                        # Silent measurement
                        m = state.d.add_node(NodeType.X, x=state.row_offset, y=q)
                        e = state.d.add_edge(state.current_qubit_nodes[q], m)
                        flush_faults_on_qubit(q, e)
                    r = state.d.add_node(NodeType.X, x=state.row_offset + 1, y=q)
                    state.current_qubit_nodes[q] = r

                state.row_offset += 2
            case "MR" | "MRZ" | "M" | "MZ":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    # Silent measurement
                    q = tar.qubit_value
                    state.ensure_initialized(q)
                    m = state.d.add_node(NodeType.X, x=state.row_offset, y=q)
                    e = state.d.add_edge(state.current_qubit_nodes[q], m)
                    flush_faults_on_qubit(q, e)
                    r = state.d.add_node(NodeType.X, x=state.row_offset + 1, y=q)
                    state.current_qubit_nodes[q] = r

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

    # Remove unused qubit world lines and latest measurements
    down_shift = 0
    for qubit in range(state.n_qubits):
        cur_node = state.current_qubit_nodes[qubit]
        if cur_node is None:
            for node in state.d.node_indices():
                if state.d.y(node) >= qubit - down_shift:  # Assume everything is aligned with a qubit world line
                    state.d.set_y(node, state.d.y(node) - 1)
            down_shift += 1
            continue

        if len(state.d.neighbors(cur_node)) != 0:  # Overapproximation to detect whether last op was a measurement
            raise RuntimeError(f"Last operation on used qubit {qubit} was not detected to be a measurement!")
        state.d.remove_node(cur_node)

    # All stim diagrams we support are closed
    state.d.set_io([], [], virtual=False)

    # Build noise model
    noise = NoiseModel(state.d, atomic_faults)

    return state.d, noise
