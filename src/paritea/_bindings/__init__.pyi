from paritea.diagram import Diagram
from paritea.pauli import PauliString


def _compute_pauli_webs(a: Diagram, *, stabilisers: bool, detecting_regions: bool) -> tuple[list[PauliString] | None, list[PauliString] | None]: ...
