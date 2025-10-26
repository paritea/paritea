import dataclasses
from typing import List, Iterable, Tuple, Mapping, Set

from .diagram import Diagram, NodeType
from .pauli import PauliString, Pauli
from .web import compute_pauli_webs


@dataclasses.dataclass(init=True, frozen=True)
class FlipOperators:
    diagram: Diagram
    stab_flip_ops: List[PauliString]
    region_flip_ops: List[PauliString]
    stab_gen_set: List[PauliString]
    region_gen_set: List[PauliString]
    region_flip_op_stab_flip_map: Mapping[int, Set[int]]


def _flip_operators(
    web_generating_set: Iterable[PauliString],
    restriction_func=lambda x: x,
) -> Tuple[List[PauliString], List[PauliString]]:
    """
    Calculates flip operators for the given collection of webs, which is presumed to be a minimal generating set for
    some space under the given restriction function (defaults to identity).

    Note that this function might change the generating set in use, which will be returned.

    :return: The flip operators and the new generating set, such that the items at the same indices correspond.
    """

    flip_ops = []
    new_gen_set = list(web_generating_set)
    for curr_gen_idx in range(len(new_gen_set)):
        curr_gen = restriction_func(new_gen_set[curr_gen_idx])
        flip_op = None
        for edge, p in curr_gen.items():
            if p != Pauli.I:
                flip_op = PauliString.edge_flip(edge, p)
        if flip_op is None:
            raise AssertionError(f"No flip operator found for generator {curr_gen}!")

        flip_ops.append(flip_op)
        new_gen_set = [
            new_gen_set[i]
            if i == curr_gen_idx or restriction_func(new_gen_set[i]).commutes(flip_op)
            else new_gen_set[i] * flip_op
            for i in range(len(new_gen_set))
        ]

    return flip_ops, new_gen_set


def build_flip_operators(d: Diagram) -> FlipOperators:
    """
    Builds flip operators, obtaining new generating sets for the stabilising and detecting webs of the diagram.
    """
    boundary_edges = d.boundary_edges()
    # To establish stabilisers (i.e. input-output relationships), we need to ensure that every boundary node has exactly
    # one connected boundary edge.
    assert all(
        [
            len(d.incident_edge_index_map(b)) == 1
            and d.type(list(d.incident_edge_index_map(b).values())[0][1]) != NodeType.B
            for b in d.boundary_nodes()
        ]
    ), "The diagram must allocate boundary nodes and edges one-to-one!"

    stabs, regions = compute_pauli_webs(d)

    stab_flip_ops, stab_gen_set = _flip_operators(stabs, lambda w: w.restrict(boundary_edges))
    region_flip_ops, region_gen_set = _flip_operators(regions)

    region_flip_op_stab_flip_map = {
        i: {j for j in range(len(stab_gen_set)) if not region_flip_ops[i].commutes(stab_gen_set[j])}
        for i in range(len(region_flip_ops))
    }

    return FlipOperators(d, stab_flip_ops, region_flip_ops, stab_gen_set, region_gen_set, region_flip_op_stab_flip_map)
