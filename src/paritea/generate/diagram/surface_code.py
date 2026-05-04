# The code in this file is directly derived from generators in https://github.com/quantumlib/Stim, with open boundaries,
# translated to Python. See https://github.com/quantumlib/Stim/blob/main/LICENSE for license details.
# Details on the stabiliser measurement cycles etc. can be found in https://arxiv.org/pdf/1404.3747.

from itertools import chain
from typing import Literal, NamedTuple, overload

from paritea.diagram import Diagram
from paritea.generate.circuit_builder import CircuitBuilder


class _Coord(NamedTuple):
    x: float
    y: float

    def __add__(self, other: "_Coord") -> "_Coord":
        return _Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "_Coord") -> "_Coord":
        return _Coord(self.x - other.x, self.y - other.y)

    def __eq__(self, other: "_Coord") -> bool:
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: "_Coord") -> bool:
        if self.x != other.x:
            return self.x < other.x

        return self.y < other.y


@overload
def surface_code_memory_experiment(
    *, distance: int, rounds: int = 1, partition: Literal[True]
) -> tuple[Diagram, list[list[int]]]: ...
@overload
def surface_code_memory_experiment(*, distance: int, rounds: int = 1, partition: Literal[False] = False) -> Diagram: ...
def surface_code_memory_experiment(
    *, distance: int, rounds: int = 1, partition: bool = False
) -> Diagram | tuple[Diagram, list[list[int]]]:
    """
    Returns a circuit diagram protecting the X logical using the rotated surface code.

    :param distance: The distance (side length) of the surface code.
    :param rounds: The number of times to repeat the diagram.
    :param partition: Whether to return partitions that aid during Pauli web computation.
    :return:
    """
    if rounds < 1:
        raise ValueError(f"Need rounds >= 1, {rounds} given.")
    if distance < 2:
        raise ValueError(f"Need distance >= 2, {distance} given.")

    # Place data qubits
    data_coords: set[_Coord] = set()
    x_observable: list[_Coord] = []
    z_observable: list[_Coord] = []
    for x in range(distance):
        actual_x = x + 0.5
        for y in range(distance):
            actual_y = y + 0.5
            q = _Coord(actual_x * 2, actual_y * 2)
            data_coords.add(q)
            if x == 0:
                x_observable.append(q)
            if y == 0:
                z_observable.append(q)

    # Place ancilla qubits
    x_measure_coords: set[_Coord] = set()
    z_measure_coords: set[_Coord] = set()
    for x in range(distance + 1):
        for y in range(distance + 1):
            q = _Coord(x * 2, y * 2)
            on_boundary_x = x == 0 or x == distance
            on_boundary_y = y == 0 or y == distance
            parity = x % 2 != y % 2
            if (on_boundary_x and parity) or (on_boundary_y and not parity):
                continue
            if parity:
                x_measure_coords.add(q)
            else:
                z_measure_coords.add(q)

    # Define interaction orders so that hook errors run against the error grain instead of with it.
    z_order = [_Coord(1, 1), _Coord(1, -1), _Coord(-1, 1), _Coord(-1, -1)]
    x_order = [_Coord(1, 1), _Coord(-1, 1), _Coord(1, -1), _Coord(-1, -1)]

    diagram, partitions = _generate_rotated_surface_code(
        distance=distance,
        rounds=rounds,
        data_coords=data_coords,
        x_measure_coords=x_measure_coords,
        z_measure_coords=z_measure_coords,
        z_order=z_order,
        x_order=x_order,
    )

    if partition:
        return diagram, partitions

    return diagram


def _generate_rotated_surface_code(
    *,
    distance: int,
    rounds: int = 1,
    data_coords: set[_Coord],
    x_measure_coords: set[_Coord],
    z_measure_coords: set[_Coord],
    z_order: list[_Coord],
    x_order: list[_Coord],
) -> tuple[Diagram, list[list[int]]]:
    def coord_to_index(_q: _Coord) -> int:
        _q = _q - _Coord(0, _q.x % 2)
        return int(_q.x + _q.y * (distance + 0.5))

    # Index the measurement qubits and data qubits.
    p2q: dict[_Coord, int] = {}
    q2p: dict[int, _Coord] = {}
    for coord in chain(data_coords, x_measure_coords, z_measure_coords):
        q = coord_to_index(coord)
        p2q[coord] = q
        q2p[q] = coord

    # Make target lists for various types of qubits.
    data_qubits: list[int] = sorted(p2q[q] for q in data_coords)
    measurement_qubits: list[int] = sorted([p2q[q] for q in x_measure_coords] + [p2q[q] for q in z_measure_coords])
    x_measurement_qubits: list[int] = sorted(p2q[q] for q in x_measure_coords)
    all_qubits = sorted(data_qubits + measurement_qubits)

    # Reverse index the measurement order used for defining detectors.
    data_coord_to_order: dict[_Coord, int] = {}
    measure_coord_to_order: dict[_Coord, int] = {}
    for i, q in enumerate(data_qubits):
        data_coord_to_order[q2p[q]] = i
    for i, q in enumerate(measurement_qubits):
        measure_coord_to_order[q2p[q]] = i

    # List out CNOT gate targets using given interaction orders.
    cnot_cts: list[list[tuple[int, int]]] = [[], [], [], []]
    for k in range(4):
        for measure in x_measure_coords:
            data = measure + x_order[k]
            if data in p2q:
                cnot_cts[k].append((p2q[measure], p2q[data]))
        for measure in z_measure_coords:
            data = measure + z_order[k]
            if data in p2q:
                cnot_cts[k].append((p2q[data], p2q[measure]))

    partitions = []
    # Start building the circuit
    builder = CircuitBuilder(max(all_qubits) + 1)
    inputs = builder.append_boundary(data_qubits)
    for _ in range(rounds):
        new_nodes = []

        new_nodes.extend(builder.append_reset_zero(measurement_qubits))
        new_nodes.extend(builder.append_h(x_measurement_qubits))
        for cts in cnot_cts:
            new_nodes.extend(builder.append_cnot(cts))
        new_nodes.extend(builder.append_h(x_measurement_qubits))
        new_nodes.extend(builder.append_measure_comp(measurement_qubits))

        partitions.append(new_nodes)
    outputs = builder.append_boundary(data_qubits)
    builder.diagram.set_io(inputs, outputs, virtual=False)

    return builder.diagram, partitions
