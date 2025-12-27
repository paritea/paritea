import numpy as np
import stim
from galois import GF2

from faulttools import FlipOperators, Pauli, PauliString, build_flip_operators
from faulttools.noise import NoiseModel
from faulttools.pushout import push_out


def _flip_ops_for_detecting_operators(
    flip_ops: FlipOperators, measurement_nodes: list[int], operators: list[PauliString]
) -> FlipOperators:
    measurement_edges = {flip_ops.diagram.incident_edges(n)[0]: i for i, n in enumerate(measurement_nodes)}

    region_gen_set: list[PauliString] = []
    region_flip_ops: list[PauliString] = []
    for operator in operators:
        compiled_gen_set = GF2(
            [p.restrict(measurement_edges.keys()).compile(measurement_edges) for p in flip_ops.region_gen_set]
        )
        b = GF2.Zeros(len(measurement_edges) * 2)
        for e, p in operator.items():
            if p == Pauli.X or p == Pauli.Y:
                b[measurement_edges[e]] = 1
            if p == Pauli.Z or p == Pauli.Y:
                b[measurement_edges[e] + len(operator)] = 1
        rref = GF2(np.vstack([compiled_gen_set, [b]])).transpose().row_reduce()
        solution = PauliString({})
        solution_flip_op = PauliString({})
        for row in rref:
            nonzero_indices = np.nonzero(row)[0]
            if len(nonzero_indices) > 0 and row[-1] == 1:
                if nonzero_indices[0] == len(compiled_gen_set):  # Got a full 0 row except for augmenting column
                    raise RuntimeError(f"Could not solve detecting regions for operator {operator}!")
                solution *= flip_ops.region_gen_set[nonzero_indices[0]]
                solution_flip_op *= flip_ops.region_flip_ops[nonzero_indices[0]]
        region_gen_set.append(solution)
        region_flip_ops.append(solution_flip_op)

    region_flip_op_stab_flip_map = {
        i: {j for j in range(len(flip_ops.stab_gen_set)) if not region_flip_ops[i].commutes(flip_ops.stab_gen_set[j])}
        for i in range(len(region_flip_ops))
    }

    return FlipOperators(
        flip_ops.diagram,
        flip_ops.stab_flip_ops.copy(),
        region_flip_ops,
        flip_ops.stab_gen_set.copy(),
        region_gen_set,
        region_flip_op_stab_flip_map,
    )


def push_out_for_measurement_detectors[T](
    nm: NoiseModel[T], *, measurement_nodes: list[int], logicals: list[PauliString], detectors: list[PauliString]
) -> tuple[NoiseModel[T], set[int], set[int]]:
    d = nm.diagram
    flip_ops = _flip_ops_for_detecting_operators(build_flip_operators(d), measurement_nodes, logicals + detectors)
    new_nm = push_out(nm, flip_ops)

    return new_nm, set(range(len(logicals))), set(range(len(logicals), len(logicals) + len(detectors)))


def export_to_stim_dem(
    nm: NoiseModel[float], *, logical_regions: set[int], detector_regions: set[int]
) -> stim.DetectorErrorModel:
    dem_str = ""
    for fault, p in nm.atomic_faults_with_values_unpacked():
        if len(fault.detector_flips) == 0:
            continue

        dem_part = ""
        for detector in fault.detector_flips:
            if detector in logical_regions:
                dem_part += f"L{detector} "
            elif detector in detector_regions:
                dem_part += f"D{detector} "
            else:
                raise ValueError(f"Region {detector} is neither known as a logical nor as a detector")

        if dem_part != "":
            dem_str += f"error({p}) {dem_part}\n"
    dem = stim.DetectorErrorModel(dem_str)

    return dem
