from paritea import PauliString
from paritea._bindings import _compute_pauli_webs_through_partitions
from paritea.diagram import Diagram


def pauli_webs_through_partitions(
    d: Diagram, *, partitions: list[list[int]]
) -> tuple[list[PauliString], list[PauliString]]:
    """
    Computes the Pauli webs of the given diagram with the provided partitions by calculating the Pauli webs of each
    subdiagram individually and combining the results. Provided partitions must cover all nodes except for boundaries.
    """

    return _compute_pauli_webs_through_partitions(d, partitions=partitions)
