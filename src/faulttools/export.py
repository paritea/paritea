from typing import List, Dict, Tuple

import stim

from faulttools import PauliString, build_flip_operators, FlipOperators
from faulttools.noise import NoiseModel
from faulttools.pushout import push_out


def _change_basis_for_logicals(
    flip_ops: FlipOperators, logicals: List[PauliString]
) -> Tuple[FlipOperators, Dict[int, int]]:
    if len(logicals) > len(flip_ops.region_gen_set):
        raise ValueError("Too many logicals given!")

    stab_gen_set = flip_ops.stab_gen_set.copy()
    stab_flip_ops = flip_ops.stab_flip_ops.copy()
    region_gen_set = flip_ops.region_gen_set.copy()
    region_flip_ops = flip_ops.region_flip_ops.copy()

    region_idx_to_logical_map: Dict[int, int] = {}
    for i_l, logical in enumerate(logicals):
        flip_indices = [i for i, region in enumerate(region_gen_set) if not region.commutes(logical)]
        if len(flip_indices) == 0:
            raise RuntimeError(f"Cannot make logical {logical} anticommute with any region!")
        region_idx_to_logical_map[flip_indices[0]] = i_l
        if len(flip_indices) > 1:
            for flip_idx in flip_indices[1:]:
                region_gen_set[flip_idx] *= region_gen_set[flip_indices[0]]
                region_flip_ops[flip_idx] *= region_flip_ops[flip_indices[0]]

    region_flip_op_stab_flip_map = {
        i: {j for j in range(len(stab_gen_set)) if not region_flip_ops[i].commutes(stab_gen_set[j])}
        for i in range(len(region_flip_ops))
    }

    return FlipOperators(
        flip_ops.diagram,
        stab_flip_ops,
        region_flip_ops,
        stab_gen_set,
        region_gen_set,
        region_flip_op_stab_flip_map,
    ), region_idx_to_logical_map


def export_to_stim_dem(nm: NoiseModel, *, logicals: List[PauliString]) -> stim.DetectorErrorModel:
    d = nm.diagram()
    flip_ops, region_idx_to_logical_map = _change_basis_for_logicals(build_flip_operators(d), logicals)

    pushed_out = push_out(nm, flip_ops)

    dem_str = ""
    for fault, p in pushed_out.atomic_weights():
        if len(fault.detector_flips) == 0:
            continue

        dem_str += f"error({p}) "
        for detector in fault.detector_flips:

            if detector in region_idx_to_logical_map:
                dem_str += f"L{region_idx_to_logical_map[detector]} "
            else:
                dem_str += f"D{detector} "
        dem_str += "\n"
    dem = stim.DetectorErrorModel(dem_str)

    return dem
