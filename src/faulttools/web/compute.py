from typing import Iterable, Optional, Tuple

from pyzx import Mat2
from .red_green import to_red_green_form
from .firing_assignments import determine_ordering, create_firing_verification, convert_firing_assignment_to_web
from ..diagram import Diagram
from ..pauli import PauliString

from copy import deepcopy


def _compute(
    diagram: Diagram, stabilisers: bool, detecting_regions: bool
) -> Tuple[Optional[Iterable[PauliString]], Optional[Iterable[PauliString]]]:
    """
    Performs full stabiliser and detecting region computation, depending on the given flags. Enabling both flags in one
    call is preferred to enabling them in separate calls as they may share basic computations.
    """

    d = deepcopy(diagram)

    additional_nodes = to_red_green_form(d)
    ordering = determine_ordering(d)
    m_d = create_firing_verification(d, ordering)

    # Compute row span of valid firing assignment space
    sol_row_basis = Mat2(m_d.nullspace())

    stabs = None
    if stabilisers:
        # Search for solutions that do not highlight boundary edges, i.e. detecting regions
        boundary_selected_basis = sol_row_basis.transpose()[: len(ordering.z_boundaries) * 2, :]

        pivot_cols = []
        boundary_selected_basis.gauss(pivot_cols=pivot_cols)
        stab_sols = [sol_row_basis.data[i] for i in pivot_cols]
        webs = list(map(lambda v: convert_firing_assignment_to_web(d, ordering, v), stab_sols))
        stabs = []
        for web in webs:
            cleaned_web = additional_nodes.remove_from(d, web)
            stabs.append(
                PauliString({diagram.edge_indices_from_endpoints(*edge)[0]: p for edge, p in cleaned_web.items()})
            )

    regions = None
    if detecting_regions:
        # Search for solutions that do not highlight boundary edges, i.e. detecting regions
        boundary_selected_basis = sol_row_basis.transpose()[: len(ordering.z_boundaries) * 2, :]
        boundary_nullspace_vectors = boundary_selected_basis.nullspace()
        # Empty nullspace of boundary edges -> no webs that highlight no boundary edges -> no detecting regions
        if len(boundary_nullspace_vectors) == 0:
            region_sols = []
        else:
            region_sols = (Mat2(boundary_nullspace_vectors) * sol_row_basis).data
        webs = list(map(lambda v: convert_firing_assignment_to_web(d, ordering, v), region_sols))
        regions = []
        for web in webs:
            cleaned_web = additional_nodes.remove_from(d, web)
            regions.append(
                PauliString({diagram.edge_indices_from_endpoints(*edge)[0]: p for edge, p in cleaned_web.items()})
            )

    return stabs, regions


def compute_stabilisers(diagram: Diagram) -> Iterable[PauliString]:
    return _compute(diagram, stabilisers=True, detecting_regions=False)[0]


def compute_detecting_regions(diagram: Diagram) -> Iterable[PauliString]:
    return _compute(diagram, stabilisers=False, detecting_regions=True)[1]


def compute_pauli_webs(diagram: Diagram) -> Tuple[Iterable[PauliString], Iterable[PauliString]]:
    return _compute(diagram, stabilisers=True, detecting_regions=True)
