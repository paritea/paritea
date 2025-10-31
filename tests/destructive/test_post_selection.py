import time
from copy import deepcopy
from typing import List, Tuple

import generate
from faulttools import build_flip_operators, NoiseModel, pushout, PauliString, Pauli
from faulttools.destructive import post_select
from faulttools.diagram import Diagram, NodeInfo, NodeType
from faulttools.diagram.conversion import from_pyzx


def test_post_selection_zweb():
    d = from_pyzx(generate.zweb(4, 4))
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.edge_flip_noise(d)
    pushed_out_noise_model = pushout(noise_model, flip_ops)

    t = time.time()
    lmao = list(post_select(pushed_out_noise_model, flip_ops.stab_gen_set, len(flip_ops.region_flip_ops)))
    print(f"Generating post selection took {time.time() - t:.2f}s")


def generate_shor_extraction(stabilisers: List[PauliString], qubits: int, repeat: int = 1) -> Diagram:
    """
    Generates a ZX diagram measuring the given stabilisers one-by-one.
    """

    stabiliser_mapping: List[Tuple[Diagram, List[int]]] = []
    for stabiliser in stabilisers:
        stabiliser_diagram = Diagram()
        mapped = [-1 for _ in range(qubits)]
        controls = [-1 for _ in range(qubits)]

        cat_x = stabiliser_diagram.add_node(NodeInfo(NodeType.X))  # TODO fault tolerantly prepare arbitrary cat state

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


steane_code_stabilisers = [
    PauliString("IIIXXXX"),
    PauliString("IXXIIXX"),
    PauliString("XIXIXIX"),
    PauliString("IIIZZZZ"),
    PauliString("IZZIIZZ"),
    PauliString("ZIZIZIZ"),
]


def generate_rotated_planar_surface_code_stabilisers(L: int) -> List[PauliString]:
    """
    Generates the stabilisers of an LxL rotated planar surface code.
    """

    plaquettes = []

    def qubit(x: int, y: int) -> int:
        return x + y * L

    # Generate bulk plaquettes in checkerboard, starting with X
    for i in range(L - 1):
        for j in range(L - 1):
            p_type = Pauli.X if (i + j) % 2 == 0 else Pauli.Z
            plaquettes.append(
                PauliString(
                    {qubit(i, j): p_type, qubit(i + 1, j): p_type, qubit(i, j + 1): p_type, qubit(i + 1, j + 1): p_type}
                )
            )

    # Generate boundary plaquettes
    for i in range(0, L - 1, 2):
        # Top Z plaquette
        plaquettes.append(PauliString({qubit(i, 0): Pauli.Z, qubit(i + 1, 0): Pauli.Z}))
        # Right X plaquette
        plaquettes.append(PauliString({qubit(L - 1, i): Pauli.X, qubit(L - 1, i + 1): Pauli.X}))

    for i in range(1, L, 2):
        # Bottom Z plaquette
        plaquettes.append(PauliString({qubit(i, L - 1): Pauli.Z, qubit(i + 1, L - 1): Pauli.Z}))
        # Left X plaquette
        plaquettes.append(PauliString({qubit(0, i): Pauli.X, qubit(0, i + 1): Pauli.X}))

    return plaquettes


def test_post_selection_shor_extraction():
    L = 5
    # Generates a shor extraction of the [7,1,3] Steane code stabilisers
    t = time.time()
    d = generate_shor_extraction(
        generate_rotated_planar_surface_code_stabilisers(L),
        qubits=L**2,
        repeat=1,
    )
    print(f"Generating circuit took {time.time() - t:.2f}s")

    t = time.time()
    flip_ops = build_flip_operators(d)
    print(f"Flip ops took {time.time() - t:.2f}s")

    t = time.time()
    noise_model = NoiseModel.edge_flip_noise(d)
    print(f"Noise model took {time.time() - t:.2f}s")

    t = time.time()
    pushed_out_noise_model = pushout(noise_model, flip_ops)
    print(f"Pushing out took {time.time() - t:.2f}s")

    t = time.time()
    lmao = list(post_select(pushed_out_noise_model, flip_ops.stab_gen_set, len(flip_ops.region_flip_ops)))
    print(f"Generating post selection took {time.time() - t:.2f}s")
    print(f"Found {len(lmao)} undetectable faults under post selection")
