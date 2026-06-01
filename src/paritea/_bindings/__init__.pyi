from paritea.diagram import Diagram
from paritea.pauli import PauliString

def _compute_pauli_webs(
    a: Diagram, *, stabilisers: bool, detecting_regions: bool
) -> tuple[list[PauliString] | None, list[PauliString] | None]: ...
def _compute_pauli_webs_through_partitions(
    a: Diagram, *, partitions: list[list[int]]
) -> tuple[list[PauliString], list[PauliString]]: ...

def _check_fault_equivalence(
    *,
    nm1_sigs: list[tuple[int, int]],
    nm2_sigs: list[tuple[int, int]],
    d1_boundaries: int,
    d1_detectors: int,
    d2_boundaries: int,
    d2_detectors: int,
    until: int | None,
    quiet: bool,
) -> int | None: ...
