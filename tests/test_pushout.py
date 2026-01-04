import pytest

from paritea import FlipOperators, build_flip_operators, generate, push_out
from paritea.glue.pyzx import from_pyzx
from paritea.noise import NoiseModel


def group_fault_values_by_flips[T](
    nm: NoiseModel[T], flip_ops: FlipOperators
) -> dict[tuple[frozenset[int], frozenset[int]], list[T]]:
    result: dict[tuple[frozenset[int], frozenset[int]], list[T]] = {}
    for fault, values in nm.atomic_faults_with_values():
        flipped_stabs = frozenset(i for i, r in enumerate(flip_ops.stab_gen_set) if not fault.edge_flips.commutes(r))
        flipped_regions = frozenset(
            i for i, r in enumerate(flip_ops.region_gen_set) if not fault.edge_flips.commutes(r)
        )

        if not fault.detector_flips.isdisjoint(flipped_regions):
            raise RuntimeError("Given detector indices are not disjoint, shift the region indices saved in faults!")

        key = (flipped_stabs, fault.detector_flips.union(flipped_regions))
        if key not in result:
            result[key] = []
        result[key].extend(values)

    for values in result.values():
        values.sort()

    return result


def change_stabiliser_basis(flip_ops: FlipOperators) -> FlipOperators:
    if len(flip_ops.stab_gen_set) >= 2:
        stab_gen_set = [flip_ops.stab_gen_set[0]]
        stab_flip_ops = [flip_ops.stab_flip_ops[0]]
        for op, flip_op in zip(flip_ops.stab_gen_set[1:], flip_ops.stab_flip_ops[1:]):
            stab_gen_set.append(op * flip_ops.stab_gen_set[0])
            stab_flip_ops.append(flip_op * flip_ops.stab_flip_ops[0])
    else:
        stab_gen_set = flip_ops.stab_gen_set.copy()
        stab_flip_ops = flip_ops.stab_flip_ops.copy()

    return FlipOperators(
        diagram=flip_ops.diagram,
        stab_flip_ops=stab_flip_ops,
        stab_gen_set=stab_gen_set,
        region_gen_set=flip_ops.region_gen_set.copy(),
    )


def compare_noise_models[T](nm1: NoiseModel[T], nm2: NoiseModel[T], flip_ops: FlipOperators) -> None:
    assert nm1.diagram is nm2.diagram
    assert nm1.diagram is flip_ops.diagram

    changed_flip_ops = change_stabiliser_basis(flip_ops)

    boundary_edges = nm1.diagram.boundary_edges()
    assert nm1.num_faults() == nm2.num_faults()
    assert all(set(f.edge_flips.keys()).issubset(boundary_edges) for f in nm2.atomic_faults())

    nm1_grouped = group_fault_values_by_flips(nm1, changed_flip_ops)
    nm2_grouped = group_fault_values_by_flips(nm2, changed_flip_ops)
    keys_diff = set(nm1_grouped.keys()).symmetric_difference(nm2_grouped.keys())
    assert len(keys_diff) == 0, f"Fault keys are not the same! Symmetric diff: {keys_diff}"
    for flips in nm1_grouped:
        assert nm1_grouped[flips] == nm2_grouped[flips], f"Value difference for fault flips {flips}!"


def test_simple_zweb():
    d = from_pyzx(generate.zweb(2, 2))
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.weighted_edge_flip_noise(d)

    pushed_out = push_out(noise_model, flip_ops)

    compare_noise_models(noise_model, pushed_out, flip_ops)


@pytest.mark.parametrize("repeat", [1, 2, 5, 10])
def test_shor_extraction_steane(repeat: int):
    d = generate.shor_extraction(generate.steane_code_stabilisers(), qubits=7, repeat=repeat)
    d.infer_io_from_boundaries()
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.weighted_edge_flip_noise(d)

    pushed_out = push_out(noise_model, flip_ops)

    compare_noise_models(noise_model, pushed_out, flip_ops)


@pytest.mark.parametrize(("size", "repeat"), [(3, 1), (3, 2), (3, 5), (5, 1), (5, 2)])
def test_shor_extraction_surface(size: int, repeat: int):
    d = generate.shor_extraction(generate.rotated_planar_surface_code_stabilisers(size), qubits=size**2, repeat=repeat)
    d.infer_io_from_boundaries()
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.weighted_edge_flip_noise(d)

    pushed_out = push_out(noise_model, flip_ops)

    compare_noise_models(noise_model, pushed_out, flip_ops)
