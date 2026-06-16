from collections.abc import Mapping
from typing import NamedTuple, Iterable, overload, Literal

import numpy as np
from galois import GF2

from paritea import build_flip_operators, push_out
from paritea._bindings import _check_fault_equivalence
from paritea.noise import Fault, NoiseModel
from paritea.utils import NoiseModelParam, noise_model_params
from paritea.web import row_reduce_webs


class Stabilisers:
    def __init__(self, stabiliser_rref: GF2):
        self.rref = stabiliser_rref
        self.indices = self.rref.argmax(axis=1).view(np.ndarray)


class AugmentedStabilisers:
    _rref: GF2
    _indices: np.ndarray

    @staticmethod
    def from_stabilisers(stabilisers: Stabilisers, num_sinks: int) -> "AugmentedStabilisers":
        self = AugmentedStabilisers()
        self._rref = GF2(np.hstack([stabilisers.rref, GF2.Zeros((len(stabilisers.rref), num_sinks))]))
        self._indices = stabilisers.indices

        return self

    def normalise_single(self, compiled_fault: GF2) -> GF2:  # TODO remove
        return self.normalise(GF2([compiled_fault]))[0]

    def normalise(self, compiled_faults: GF2) -> GF2:
        return compiled_faults + compiled_faults[:, self._indices] @ self._rref


def _compile_atomic_faults(
    atomics: list[tuple[Fault, list[int]]],
    stabilisers: AugmentedStabilisers,
    boundaries_to_idx: Mapping[int, int],
    detector_to_idx: Mapping[int, int],
) -> list[tuple[int, int]]:
    normalised_faults: list[tuple[int, int]] = []
    for f, vs in atomics:
        if f.is_trivial():
            continue

        compiled = f.compile(boundaries_to_idx, detector_to_idx)
        normalised = stabilisers.normalise_single(compiled)
        normalised_int = Fault.compiled_to_int(normalised)
        normalised_faults.extend((normalised_int, v) for v in vs)

    return normalised_faults


class Violation(NamedTuple):
    is_nm1: bool
    """Whether the violation of fault equivalence results from the first noise model
    (true) or the second (false)."""
    weight: int
    """The total weight of the violating combination."""
    faults: Iterable[tuple[Fault, int]] | None
    """The atomic faults from which the violating combination was constructed."""


def _is_fault_equivalence(
    noise_1: NoiseModel[int],
    noise_2: NoiseModel[int],
    num_detectors_1: int,
    num_detectors_2: int,
    stabilisers: Stabilisers,
    *,
    until: int,
    resolve_reason: bool,
    quiet: bool,
) -> Violation | None:
    """
    Given two noise models noise_1 and noise_2 (required to be in pushed out form), determine if they are fault
    equivalent. This requires their underlying diagrams to be semantically equivalent, so stabilisers are only supplied
    once.

    Note that currently only equally weighted noise models are supported.

    :param noise_1: First noise model to check
    :param noise_2: Second noise model to check
    :param num_detectors_1: Size of detector basis in the diagram attached to noise_1
    :param num_detectors_2: Size of detector basis in the diagram attached to noise_2
    :param stabilisers: A stabiliser basis for the diagrams attached to noise_1 and noise_2
    :param until: Up to which weight (exclusive) to check the equivalence
    :param quiet: Whether to silence additional informative output
    """
    atomic_weights_1 = {w for _, w in noise_1.atomic_faults_with_values_unpacked()}
    atomic_weights_2 = {w for _, w in noise_2.atomic_faults_with_values_unpacked()}
    negative_weights_1 = {w for w in atomic_weights_1 if w < 0}
    negative_weights_2 = {w for w in atomic_weights_2 if w < 0}
    if len(negative_weights_1) > 0 or len(negative_weights_2) > 0:
        raise ValueError(
            "Cannot process noise models with negative weights, but the following negative weights were given: "
            f"Noise model 1: {negative_weights_1}; Noise model 2: {negative_weights_2}."
        )

    d1, d2 = noise_1.diagram, noise_2.diagram
    d1_edge_idx_map = {d1.incident_edges(b)[0]: i for i, b in enumerate(d1.io_sorted())}
    d1_detector_idx_map = {i: i for i in range(num_detectors_1)}
    d2_edge_idx_map = {d2.incident_edges(b)[0]: i for i, b in enumerate(d2.io_sorted())}
    d2_detector_idx_map = {i: i for i in range(num_detectors_2)}

    if not quiet:
        print("Compiling atomic faults for d1...")
    d1_stabs = AugmentedStabilisers.from_stabilisers(stabilisers, num_detectors_1)
    nm1_atomics = list(noise_1.atomic_faults_with_values())
    g1_sig_nf = _compile_atomic_faults(nm1_atomics, d1_stabs, d1_edge_idx_map, d1_detector_idx_map)
    if not quiet:
        print(f"Retrieved {len(g1_sig_nf)} atomic faults for d1!")

    if not quiet:
        print("Compiling atomic faults for d2...")
    d2_stabs = AugmentedStabilisers.from_stabilisers(stabilisers, num_detectors_2)
    nm2_atomics = list(noise_2.atomic_faults_with_values())
    g2_sig_nf = _compile_atomic_faults(nm2_atomics, d2_stabs, d2_edge_idx_map, d2_detector_idx_map)
    if not quiet:
        print(f"Retrieved {len(g2_sig_nf)} atomic faults for d2!")

    violating_weight = _check_fault_equivalence(
        nm1_sigs=g1_sig_nf,
        nm2_sigs=g2_sig_nf,
        d1_detectors=num_detectors_1,
        d2_detectors=num_detectors_2,
        until=until,
        resolve_reason=resolve_reason,
        quiet=quiet,
    )

    if violation is None:
        return None

    idx_sum = 0
    violating_faults: list[tuple[Fault, int]] = []
    for f, vs in (nm1_atomics if violation.is_nm1 else nm2_atomics):
        for i, v in enumerate(vs):
            if idx_sum + i in violation.faults:
                violating_faults.append((f, v))
        idx_sum += len(vs)

    return Violation(
        is_nm1=violation.is_nm1,
        weight=violation.weight,
        faults=violating_faults,
    )


@overload
def is_fault_equivalence(
    noise_1: NoiseModelParam[int],
    noise_2: NoiseModelParam[int],
    *,
    until: int | None = None,
    resolve_reason: Literal[False] = False,
    quiet: bool = True,
) -> bool: ...
@overload
def is_fault_equivalence(
    noise_1: NoiseModelParam[int],
    noise_2: NoiseModelParam[int],
    *,
    until: int | None = None,
    resolve_reason: Literal[True],
    quiet: bool = True,
) -> Violation | None: ...
@noise_model_params("noise_1", "noise_2")
def is_fault_equivalence(
    noise_1: NoiseModelParam[int],
    noise_2: NoiseModelParam[int],
    *,
    until: int | None = None,
    resolve_reason: bool = False,
    quiet: bool = True,
) -> bool | Violation | None:
    """
    Given two noise models noise_1 and noise_2 (required to be in pushed out form),
    determine if they are fault equivalent. This requires their underlying diagrams to
    be semantically equivalent, so stabilisers are only supplied once.

    Note that currently only equally weighted noise models are supported. Further note
    that for the result to be sound, the boundary nodes of the two diagrams must be
    ordered the same.

    :param noise_1: First noise model to check
    :param noise_2: Second noise model to check
    :param until: Up to which weight (exclusive) to check the equivalence
    :param resolve_reason: Whether to return the reason / fault combination failing the
        fault equivalence check, if there is one.
    :param quiet: Whether to silence additional informative output
    """
    d1, d2 = noise_1.diagram, noise_2.diagram
    d1_edge_idx_map = {d1.incident_edges(b)[0]: i for i, b in enumerate(d1.io_sorted())}
    d2_edge_idx_map = {d2.incident_edges(b)[0]: i for i, b in enumerate(d2.io_sorted())}

    flip_ops_1 = build_flip_operators(d1)
    flip_ops_2 = build_flip_operators(d2)
    stabs_1_rref = row_reduce_webs(flip_ops_1.stab_gen_set, d1_edge_idx_map)
    stabs_2_rref = row_reduce_webs(flip_ops_2.stab_gen_set, d2_edge_idx_map)

    if not np.array_equal(stabs_1_rref, stabs_2_rref):
        raise ValueError("The two circuits given have different stabilisers and thus different semantics!")

    pushed_out_noise_1 = push_out(noise_1, flip_ops_1)
    pushed_out_noise_2 = push_out(noise_2, flip_ops_2)

    violation = _is_fault_equivalence(
        noise_1=pushed_out_noise_1,
        noise_2=pushed_out_noise_2,
        num_detectors_1=len(flip_ops_1.region_gen_set),
        num_detectors_2=len(flip_ops_2.region_gen_set),
        stabilisers=Stabilisers(stabs_1_rref),
        until=until,
        resolve_reason=resolve_reason,
        quiet=quiet,
    )

    if resolve_reason:
        return violation

    return violation is None
