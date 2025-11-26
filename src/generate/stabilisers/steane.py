from typing import List

from faulttools import PauliString


def steane_code_stabilisers() -> List[PauliString]:
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
