import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import stim
import sympy
from sympy import Expr, Symbol, symbols

from faulttools.diagram import Diagram, NodeType
from faulttools.noise import Fault, NoiseModel
from faulttools.pauli import Pauli, PauliString


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


def from_stim(
    circuit: stim.Circuit,
) -> tuple[Diagram, NoiseModel[Expr], Iterable[Symbol], list[int], dict[int, PauliString], list[PauliString]]:
    state = _DiagramBuildingState(
        n_qubits=circuit.num_qubits,
        d=Diagram(),
        row_offset=1,
        current_qubit_nodes=[None for _ in range(circuit.num_qubits)],
    )

    p: Symbol = symbols("p")
    atomic_faults: list[tuple[Fault, Expr]] = []
    fault_prot_queue: list[tuple[set, dict[int, Pauli], Expr]] = []

    def queue_for_edge_replacement(fault_prot: dict[int, Pauli], prob: Expr):
        fault_prot_queue.append((set(fault_prot.keys()), fault_prot, prob))

    def flush_edge_for_qubit(qubit: int, edge: int) -> None:
        nonlocal fault_prot_queue
        flushed_indices = []
        for i, prot in enumerate(fault_prot_queue):
            qubits, fault_prot, prob = prot
            if qubit not in qubits:
                continue

            fault_prot[edge] = fault_prot.pop(qubit)
            qubits.remove(qubit)
            if len(qubits) != 0:
                continue

            atomic_faults.append((Fault(PauliString(fault_prot)), prob))
            flushed_indices.append(i)

        if len(flushed_indices) > 0:
            fault_prot_queue = [prot for i, prot in enumerate(fault_prot_queue) if i not in flushed_indices]

    measurement_nodes: list[int] = []
    detectors: list[list[int]] = []
    observables: dict[int, list[int]] = {}

    for instr in circuit:
        match instr.name:
            # Regular gates
            case "H":
                for q, e in _h(state, instr):
                    flush_edge_for_qubit(q, e)
            case "CX" | "CNOT" | "ZCX":
                for q, e in _cnot(state, instr):
                    flush_edge_for_qubit(q, e)
            # Measurements / Resets
            case "R" | "RZ":
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    q = tar.qubit_value
                    if state.is_initialized(q):
                        # Silent measurement
                        m = state.d.add_node(NodeType.X, x=state.row_offset, y=q)
                        e = state.d.add_edge(state.current_qubit_nodes[q], m)
                        flush_edge_for_qubit(q, e)
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
                    flush_edge_for_qubit(q, e)
                    measurement_nodes.append(m)
                    r = state.d.add_node(NodeType.X, x=state.row_offset + 1, y=q)
                    state.current_qubit_nodes[q] = r

                state.row_offset += 2
            # Error mechanisms
            case "X_ERROR":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    queue_for_edge_replacement({tar.qubit_value: Pauli.X}, p)
            case "Y_ERROR":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    queue_for_edge_replacement({tar.qubit_value: Pauli.Y}, p)
            case "Z_ERROR":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    queue_for_edge_replacement({tar.qubit_value: Pauli.Z}, p)
            case "DEPOLARIZE1":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for tar in instr.targets_copy():
                    assert tar.is_qubit_target and not tar.is_inverted_result_target
                    # if p > 0.75:
                    #    raise RuntimeError("Cannot approximate single-qubit depolarizing channel with p > 0.75!")
                    # Apply the magic formula to decorrelate depolarization, see https://algassert.com/post/2001
                    independent_p = 0.5 - 0.5 * sympy.sqrt(1 - 4 / 3 * p)
                    queue_for_edge_replacement({tar.qubit_value: Pauli.X}, independent_p)
                    queue_for_edge_replacement({tar.qubit_value: Pauli.Y}, independent_p)
                    queue_for_edge_replacement({tar.qubit_value: Pauli.Z}, independent_p)
            case "DEPOLARIZE2":
                args = instr.gate_args_copy()
                assert len(args) == 1
                for group in instr.target_groups():
                    assert len(group) == 2
                    for tar in group:
                        assert tar.is_qubit_target and not tar.is_inverted_result_target
                    # if p > 15 / 16:
                    #    raise RuntimeError("Cannot approximate double-qubit depolarizing channel with p > 15/16!")
                    # Apply the magic formula to decorrelate depolarization, see https://algassert.com/post/2001
                    independent_p = 0.5 - 0.5 * sympy.sqrt(sympy.sqrt(sympy.sqrt(1 - (16 * p) / 15)))

                    for p1, p2 in itertools.product([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], repeat=2):
                        if p1 == Pauli.I and p2 == Pauli.I:
                            continue
                        queue_for_edge_replacement({group[0].qubit_value: p1, group[1].qubit_value: p2}, independent_p)
            # Flow generator indicators
            case "DETECTOR":
                detector = []
                for tar in instr.targets_copy():
                    assert tar.is_measurement_record_target and not tar.is_inverted_result_target
                    detector.append(measurement_nodes[tar.value])
                detectors.append(detector)
            case "OBSERVABLE_INCLUDE":
                args = instr.gate_args_copy()
                assert len(args) == 1
                obs_idx = int(args[0])
                if obs_idx not in observables:
                    observables[obs_idx] = []
                for tar in instr.targets_copy():
                    assert tar.is_measurement_record_target and not tar.is_inverted_result_target
                    observables[obs_idx].append(measurement_nodes[tar.value])
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
    for q in range(state.n_qubits):
        cur_node = state.current_qubit_nodes[q]
        if cur_node is None:
            for node in state.d.node_indices():
                if state.d.y(node) >= q - down_shift:  # Assume everything is aligned with a qubit world line
                    state.d.set_y(node, state.d.y(node) - 1)
            down_shift += 1
            continue

        if len(state.d.neighbors(cur_node)) != 0:  # Overapproximation to detect whether last op was a measurement
            raise RuntimeError(f"Last operation on used qubit {q} was not detected to be a measurement!")
        state.d.remove_node(cur_node)

    # All stim diagrams we support are closed
    state.d.set_io([], [], virtual=False)

    # Build noise model
    noise = NoiseModel(state.d, atomic_faults)

    # Build anticommutation Pauli strings for observables and detectors
    def build_pauli_string(measurement_record: list[int]) -> PauliString:
        return PauliString(
            {
                state.d.incident_edges(meas_node)[0]: Pauli.Z if state.d.type(meas_node) == NodeType.Z else Pauli.X
                for meas_node in measurement_record
            }
        )

    anticommutation_observables = {i: build_pauli_string(observable) for i, observable in observables.items()}
    anticommutation_detectors = [build_pauli_string(detector) for detector in detectors]

    return state.d, noise, [p], measurement_nodes, anticommutation_observables, anticommutation_detectors
