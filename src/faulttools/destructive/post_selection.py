import itertools
import math
from functools import reduce
from typing import Any, Generator, List

from tqdm import tqdm

from faulttools import NoiseModel, PauliString
from faulttools.equivalence.check_fault_equivalence import _compile_atomic_faults, AugmentedStabilisers, _stabilisers
from faulttools.noise_model import Fault


def post_select(noise: NoiseModel, stabilisers: List[PauliString], num_detectors: int) -> Generator[int, Any, None]:
    """
    Produces all undetectable faults that may occur under the given noise model, post-selecting on non-zero detector
    parities (i.e. all combined faults that flip detecting regions are eliminated).

    :param noise: The noise model in pushed out form.
    :param stabilisers: The stabiliser generators of the diagram underlying the given noise model.
    :param num_detectors: A hint for the number of detectors to expect. Can be obtained as the number of flip operators.
    """
    d = noise.diagram()
    edge_idx_map = {d.incident_edges(b)[0]: i for i, b in enumerate(sorted(d.boundary_nodes()))}
    detector_idx_map = {i: i for i in range(num_detectors)}
    detector_masks = {i: 1 << (num_detectors - detector_idx_map[i] - 1) for i in range(num_detectors)}
    all_detector_mask = (1 << num_detectors) - 1

    compiled_stabilisers = _stabilisers(stabilisers, edge_idx_map)

    stabs = AugmentedStabilisers.from_stabilisers(compiled_stabilisers, len(detector_idx_map))
    compiled_faults = _compile_atomic_faults(noise, stabs, edge_idx_map, detector_idx_map)
    fault_ints = {Fault.compiled_to_int(f) for f in compiled_faults}

    detectable = {f for f in fault_ints if f & all_detector_mask > 0}
    undetectable_atomic = {f for f in fault_ints if f & all_detector_mask == 0}
    undetectable = {0}.union(undetectable_atomic)

    for f in undetectable:
        yield f

    tqdm.write(
        f"{len(detectable)} detected, {len(undetectable)} undetectable, {len(detectable) + len(undetectable)} total"
    )

    for curr_detector in tqdm(range(num_detectors), desc="Detectors: ", leave=True, miniters=1):
        detected_faults = [f for f in detectable if f & detector_masks[curr_detector] > 0]
        for comb in tqdm(
            itertools.combinations(detected_faults, r=2),
            total=math.comb(len(detected_faults), 2),
            desc="Combinations: ",
            leave=True,
        ):
            combined_fault = reduce(lambda s1, s2: s1 ^ s2, comb)
            if combined_fault in detectable or combined_fault in undetectable:
                continue  # Already discovered fault

            if combined_fault & all_detector_mask > 0:
                detectable.add(combined_fault)
                continue  # Undiscovered fault, but not yielded since it is detectable

            undetectable.add(combined_fault)

            # yield combined_fault
            # new_undetectable = {combined_fault}
            # while len(new_undetectable) > 0:
            #    produced_faults = set()
            #    for f1, f2 in itertools.product(new_undetectable, undetectable_atomic):
            #        new_combination = f1 ^ f2
            #
            #        if new_combination in undetectable:
            #            continue
            #
            #        undetectable.add(new_combination)
            #        produced_faults.add(new_combination)
            #        yield new_combination
            #
            #     print(len(produced_faults))
            #   print(len(new_undetectable))
            #    new_undetectable = produced_faults

        detectable.difference_update(detected_faults)
        tqdm.write(
            f"{len(detectable)} detected, {len(undetectable)} undetectable, {len(detectable) + len(undetectable)} total"
        )
