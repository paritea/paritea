from collections.abc import Mapping
from typing import NamedTuple, Iterable

import numpy as np
from galois import GF2

from paritea import build_flip_operators, push_out
from paritea._bindings import _check_fault_equivalence, _Violation
from paritea.diagram import Diagram
from paritea.noise import Fault
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
    d1: Diagram,
    d2: Diagram,
    nm1_atomics: list[tuple[Fault, list[int]]],
    nm2_atomics: list[tuple[Fault, list[int]]],
    num_detectors_1: int,
    num_detectors_2: int,
    stabilisers: Stabilisers,
    *,
    until: int,
    provenance: bool,
    quiet: bool,
) -> _Violation | None:
    """
    Given two noise models noise_1 and noise_2 (required to be in pushed out form), determine if they are fault
    equivalent. This requires their underlying diagrams to be semantically equivalent, so stabilisers are only supplied
    once.

    Note that currently only equally weighted noise models are supported.

    :param nm1_atomics: Atomic faults of the first noise model to check
    :param nm2_atomics: Atomic faults of the second noise model to check
    :param num_detectors_1: Size of detector basis in the diagram attached to noise_1
    :param num_detectors_2: Size of detector basis in the diagram attached to noise_2
    :param stabilisers: A stabiliser basis for the diagrams attached to noise_1 and noise_2
    :param until: Up to which weight (exclusive) to check the equivalence
    :param provenance: Whether to resolve provenance of fault equivalence failures
    :param quiet: Whether to silence additional informative output
    """
    negative_weights_1 = {w for _, ws in nm1_atomics for w in ws if w < 0}
    negative_weights_2 = {w for _, ws in nm2_atomics for w in ws if w < 0}
    if len(negative_weights_1) > 0 or len(negative_weights_2) > 0:
        raise ValueError(
            "Cannot process noise models with negative weights, but the following negative weights were given: "
            f"Noise model 1: {negative_weights_1}; Noise model 2: {negative_weights_2}."
        )

    d1_edge_idx_map = {d1.incident_edges(b)[0]: i for i, b in enumerate(d1.io_sorted())}
    d1_detector_idx_map = {i: i for i in range(num_detectors_1)}
    d2_edge_idx_map = {d2.incident_edges(b)[0]: i for i, b in enumerate(d2.io_sorted())}
    d2_detector_idx_map = {i: i for i in range(num_detectors_2)}

    if not quiet:
        print("Compiling atomic faults for d1...")
    d1_stabs = AugmentedStabilisers.from_stabilisers(stabilisers, num_detectors_1)
    g1_sig_nf = _compile_atomic_faults(nm1_atomics, d1_stabs, d1_edge_idx_map, d1_detector_idx_map)
    if not quiet:
        print(f"Retrieved {len(g1_sig_nf)} atomic faults for d1!")

    if not quiet:
        print("Compiling atomic faults for d2...")
    d2_stabs = AugmentedStabilisers.from_stabilisers(stabilisers, num_detectors_2)
    g2_sig_nf = _compile_atomic_faults(nm2_atomics, d2_stabs, d2_edge_idx_map, d2_detector_idx_map)
    if not quiet:
        print(f"Retrieved {len(g2_sig_nf)} atomic faults for d2!")

    return _check_fault_equivalence(
        nm1_sigs=g1_sig_nf,
        nm2_sigs=g2_sig_nf,
        d1_detectors=num_detectors_1,
        d2_detectors=num_detectors_2,
        until=until,
        provenance=provenance,
        quiet=quiet,
    )


@noise_model_params("noise_1", "noise_2")
def check_fault_equivalence(
    noise_1: NoiseModelParam[int],
    noise_2: NoiseModelParam[int],
    *,
    until: int | None = None,
    provenance: bool = False,
    quiet: bool = True,
) -> Violation | None:
    """
    Given two noise models noise_1 and noise_2 (required to be in pushed out form),
    determine if they are fault equivalent. This requires their underlying diagrams to
    be semantically equivalent.

    Note that for the result to be sound, the boundary nodes of the two diagrams must be
    ordered the same.

    :param noise_1: First noise model to check
    :param noise_2: Second noise model to check
    :param until: Up to which weight (exclusive) to check the equivalence
    :param provenance: Whether to populate the fault combination that is failing the
        fault equivalence check in the violation, if there is one.
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

    nm1_unpushed_atomics = list(noise_1.atomic_faults_with_values())
    nm1_pushed_atomics = list(pushed_out_noise_1.atomic_faults_with_values())
    nm2_unpushed_atomics = list(noise_2.atomic_faults_with_values())
    nm2_pushed_atomics = list(pushed_out_noise_2.atomic_faults_with_values())

    violation = _is_fault_equivalence(
        d1=d1,
        d2=d2,
        nm1_atomics=nm1_pushed_atomics,
        nm2_atomics=nm2_pushed_atomics,
        num_detectors_1=len(flip_ops_1.region_gen_set),
        num_detectors_2=len(flip_ops_2.region_gen_set),
        stabilisers=Stabilisers(stabs_1_rref),
        until=until,
        provenance=provenance,
        quiet=quiet,
    )

    if violation is None:
        return None

    violating_faults: list[tuple[Fault, int]] | None = None
    if provenance:
        if violation.faults is None:
            raise RuntimeError("Expected provenance to be resolved, got nothing!")

        idx_sum = 0
        violating_faults = []
        for f, vs in (nm1_unpushed_atomics if violation.is_nm1 else nm2_unpushed_atomics):
            for i, v in enumerate(vs):
                if idx_sum + i in violation.faults:
                    violating_faults.append((f, v))
            idx_sum += len(vs)

    return Violation(
        is_nm1=violation.is_nm1,
        weight=violation.weight,
        faults=violating_faults,
    )

@noise_model_params("noise_1", "noise_2")
def is_fault_equivalence(
    noise_1: NoiseModelParam[int],
    noise_2: NoiseModelParam[int],
    *,
    until: int | None = None,
    quiet: bool = True,
) -> bool:
    """
    Returns true iff the noise models are fault equivalent. A convenience wrapper around
    check_fault_equivalence.

    :param noise_1: First noise model to check
    :param noise_2: Second noise model to check
    :param until: Up to which weight (exclusive) to check the equivalence
    :param quiet: Whether to silence additional informative output
    """

    violation = check_fault_equivalence(
        noise_1=noise_1,
        noise_2=noise_2,
        until=until,
        provenance=False,
        quiet=quiet,
    )

    return violation is None
