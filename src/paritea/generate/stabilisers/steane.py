from paritea import PauliString


def steane_code_stabilisers() -> list[PauliString]:
    """
    The stabilisers of the 7-qubit CSS Steane code.
    """

    return [
        PauliString("IIIXXXX"),
        PauliString("IXXIIXX"),
        PauliString("XIXIXIX"),
        PauliString("IIIZZZZ"),
        PauliString("IZZIIZZ"),
        PauliString("ZIZIZIZ"),
    ]
