from paritea.diagram import Diagram
from paritea.pauli import PauliString

def _compute_pauli_webs(
    a: Diagram, *, stabilisers: bool, detecting_regions: bool
) -> tuple[list[PauliString] | None, list[PauliString] | None]: ...

def _compute_pauli_webs_through_partitions(
    a: Diagram, *, partitions: list[list[int]]
) -> tuple[list[PauliString], list[PauliString]]: ...
