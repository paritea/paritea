from collections import defaultdict

from faulttools import PauliString
from faulttools.flip_operators import FlipOperators
from faulttools.noise import Fault, NoiseModel


def push_out[T](model: NoiseModel[T], flip_ops: FlipOperators) -> NoiseModel[T]:
    assert model.diagram is flip_ops.diagram

    new_faults: dict[Fault, list[T]] = defaultdict(list)
    for fault, values in model.atomic_faults_with_values():
        flipped_regions = {
            i for i in range(len(flip_ops.region_gen_set)) if not fault.edge_flips.commutes(flip_ops.region_gen_set[i])
        }

        new_fault_edge_flips = PauliString()
        for stabiliser, flip_op in zip(flip_ops.stab_gen_set, flip_ops.stab_flip_ops):
            if not fault.edge_flips.commutes(stabiliser):
                new_fault_edge_flips *= flip_op

        new_fault = Fault(new_fault_edge_flips, fault.detector_flips.union(flipped_regions))
        new_faults[new_fault].extend(values)

    return NoiseModel(model.diagram, new_faults)
