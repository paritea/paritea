from faulttools import PauliString
from faulttools.flip_operators import FlipOperators
from faulttools.noise_model import NoiseModel, Fault


def dagger_form(model: NoiseModel, flip_ops: FlipOperators) -> NoiseModel:
    assert model.diagram() is flip_ops.diagram

    new_faults = []
    for atomic_fault, atomic_weight in model.atomic_weights():
        atomic_fault_flips = atomic_fault.edge_flips
        # Obtain web flip description of original atomic fault
        flipped_regions = {
            i
            for i in range(len(flip_ops.region_gen_set))
            if not atomic_fault_flips.commutes(flip_ops.region_gen_set[i])
        }
        orig_flipped_stabs = {
            i for i in range(len(flip_ops.stab_gen_set)) if not atomic_fault_flips.commutes(flip_ops.stab_gen_set[i])
        }

        # Compute which stabilisers a composed region flip operator would flip
        curr_flipped_stabs = set()
        for flipped_region in flipped_regions:
            curr_flipped_stabs.symmetric_difference_update(flip_ops.region_flip_op_stab_flip_map[flipped_region])

        new_fault_edge_flips = PauliString()
        # Add flips for missing stabilisers
        for missing_stab in orig_flipped_stabs.difference(curr_flipped_stabs):
            new_fault_edge_flips *= flip_ops.stab_flip_ops[missing_stab]
        # Remove flips for extra stabilisers
        for extra_stab in curr_flipped_stabs.difference(orig_flipped_stabs):
            new_fault_edge_flips *= flip_ops.stab_flip_ops[extra_stab]

        new_fault = Fault(new_fault_edge_flips, atomic_fault.detector_flips.union(flipped_regions))
        new_faults.append((new_fault, atomic_weight))

    return NoiseModel(model.diagram(), new_faults)
