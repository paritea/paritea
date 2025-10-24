from typing import Iterable, Tuple, List

from faulttools import PauliString, Pauli
from faulttools.diagram import NodeType
from faulttools.noise_model import NoiseModel
from faulttools.web import compute_pauli_webs


def flip_operators(
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

        new_gen_set = [
            new_gen_set[i]
            if i == curr_gen_idx or restriction_func(new_gen_set[i]).commutes(flip_op)
            else new_gen_set[i] * flip_op
            for i in range(len(new_gen_set))
        ]

    return flip_ops, new_gen_set


def dagger_form(model: NoiseModel) -> NoiseModel: # TODO guarantee diagram form in which no two boundaries are connected
    d = model.diagram()
    boundary_nodes = d.filter_nodes(lambda ni: ni.type == NodeType.B)
    boundary_edges = []
    for b in boundary_nodes:
        boundary_edges += d.incident_edges(b)
    stabs, regions = compute_pauli_webs(model.diagram())

    stab_flip_ops, stab_gen_set = flip_operators(stabs, lambda w: w.restrict(boundary_edges))
    region_flip_ops, region_gen_set = flip_operators(regions)

    region_flip_op_stab_flip_map = {
        i: {j for j in range(len(stab_gen_set)) if not region_flip_ops[i].commutes(stab_gen_set[j])}
        for i in range(len(region_flip_ops))
    }

    new_faults = []
    for atomic_fault, atomic_weight in model.atomic_weights():
        # Obtain web flip description of original atomic fault
        flipped_regions = {i for i in range(len(region_gen_set)) if not atomic_fault.commutes(region_gen_set[i])}
        orig_flipped_stabs = {i for i in range(len(stab_gen_set)) if not atomic_fault.commutes(stab_gen_set[i])}

        # Compute which stabilisers a composed region flip operator would flip
        curr_flipped_stabs = set()
        for flipped_region in flipped_regions:
            curr_flipped_stabs.symmetric_difference_update(region_flip_op_stab_flip_map[flipped_region])

        # Construct composed region flip operator
        new_fault = PauliString()
        for flipped_region in flipped_regions:
            new_fault *= region_flip_ops[flipped_region]

        # Add flips for missing stabilisers
        for missing_stab in orig_flipped_stabs.difference(curr_flipped_stabs):
            new_fault *= stab_flip_ops[missing_stab]

        # Remove flips for extra stabilisers
        for extra_stab in curr_flipped_stabs.difference(orig_flipped_stabs):
            new_fault *= stab_flip_ops[extra_stab]

        new_faults.append((new_fault, atomic_weight))

    return NoiseModel(d, new_faults)
