from paritea import Pauli, PauliString


def rotated_planar_surface_code_stabilisers(L: int) -> list[PauliString]:
    """
    Generates the stabilisers (plaquettes) of an LxL rotated planar surface code, using row-major qubit indexing.
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
