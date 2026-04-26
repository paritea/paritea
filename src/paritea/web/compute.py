from paritea.diagram import Diagram
from paritea.pauli import PauliString
from paritea._bindings import _compute_pauli_webs

def compute_stabilisers(diagram: Diagram) -> list[PauliString]:
    """
    :return: A set of stabilising webs for the given diagram that forms a basis for the diagrams stabilisers when
        restricted to its boundary. A full basis for all stabilising webs is only obtained combining the return value
        with a basis for the diagrams detecting regions.
    """
    return _compute_pauli_webs(diagram, stabilisers=True, detecting_regions=False)[0]


def compute_detecting_regions(diagram: Diagram) -> list[PauliString]:
    """
    :return: A basis for the detecting regions of the given diagram.
    """
    return _compute_pauli_webs(diagram, stabilisers=False, detecting_regions=True)[1]


def compute_pauli_webs(diagram: Diagram) -> tuple[list[PauliString], list[PauliString]]:
    """
    See .compute_stabilisers and .compute_detecting_regions of this package.
    """
    return _compute_pauli_webs(diagram, stabilisers=True, detecting_regions=True)
