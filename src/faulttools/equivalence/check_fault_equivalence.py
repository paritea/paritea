from typing import List, Mapping

import numpy as np
from galois import GF2

from .. import NoiseModel, pushout, build_flip_operators
from .enumeration import _smallest_size_iteration
from ..noise_model import noise_model_params, NoiseModelParam
from ..pauli import Pauli, PauliString


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


def _stabilisers(stabilisers: List[PauliString], boundary_idx_map: Mapping[int, int]) -> Stabilisers:
    num_boundaries = len(boundary_idx_map)
    np_stabilisers = np.zeros((len(stabilisers), num_boundaries * 2), dtype=int)
    for i, stab in enumerate(stabilisers):
        for boundary, pauli in stab.restrict(boundary_idx_map.keys()).items():
            idx = boundary_idx_map[boundary]
            if pauli == Pauli.Z or pauli == Pauli.Y:
                np_stabilisers[i, idx] = 1
            if pauli == Pauli.X or pauli == Pauli.Y:
                np_stabilisers[i, idx + num_boundaries] = 1

    return Stabilisers(GF2(np_stabilisers).row_reduce(eye="left"))


def _compile_atomic_faults(
    noise: NoiseModel,
    stabilisers: AugmentedStabilisers,
    boundaries_to_idx: Mapping[int, int],
    detector_to_idx: Mapping[int, int],
) -> List[GF2]:
    normalised_faults: List[GF2] = []
    for f in noise.atomic_faults():
        if f.is_trivial():
            continue

        compiled = f.compile(boundaries_to_idx, detector_to_idx)
        normalised_faults.append(stabilisers.normalise_single(compiled))
    sig_nf = [GF2(l) for l in np.unique(normalised_faults, axis=0)]

    return sig_nf


def _is_fault_equivalence(
    noise_1: NoiseModel,
    noise_2: NoiseModel,
    num_detectors_1: int,
    num_detectors_2: int,
    stabilisers: List[PauliString],
    quiet: bool = True,
) -> bool:
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
    :param quiet: Whether to silence additional informative output
    """
    atomic_weights_1 = {t[1] for t in noise_1.atomic_weights()}
    atomic_weights_2 = {t[1] for t in noise_2.atomic_weights()}
    if atomic_weights_1 != {1} or atomic_weights_2 != {1}:
        raise ValueError(
            f"Both given noise models must be equally and normally weighted."
            f"Weight sets detected: {atomic_weights_1} and {atomic_weights_2}"
        )

    # TODO index boundaries the same way / force and use boundary bijection

    d1, d2 = noise_1.diagram(), noise_2.diagram()
    d1_edge_idx_map = {d1.incident_edges(b)[0]: i for i, b in enumerate(sorted(d1.boundary_nodes()))}
    d1_detector_idx_map = {i: i for i in range(num_detectors_1)}
    d2_edge_idx_map = {d2.incident_edges(b)[0]: i for i, b in enumerate(sorted(d2.boundary_nodes()))}
    d2_detector_idx_map = {i: i for i in range(num_detectors_2)}

    compiled_stabilisers = _stabilisers(stabilisers, d1_edge_idx_map)

    if not quiet:
        print("Compiling atomic faults for d1...")
    g1_stabs = AugmentedStabilisers.from_stabilisers(compiled_stabilisers, len(d1_detector_idx_map))
    g1_sig_nf = _compile_atomic_faults(noise_1, g1_stabs, d1_edge_idx_map, d1_detector_idx_map)
    if not quiet:
        print(f"Retrieved {len(g1_sig_nf)} unique faults for d1!")

    if not quiet:
        print("Compiling atomic faults for d2...")
    g2_stabs = AugmentedStabilisers.from_stabilisers(compiled_stabilisers, len(d2_detector_idx_map))
    g2_sig_nf = _compile_atomic_faults(noise_2, g2_stabs, d2_edge_idx_map, d2_detector_idx_map)
    if not quiet:
        print(f"Retrieved {len(g2_sig_nf)} unique faults for d2!")

    if not quiet:
        print("Checking if d1 is fault-bound by d2...")
    g1_g2_weight = _smallest_size_iteration(
        g1_sig_nf, g2_sig_nf, num_detectors_1, len(d2_edge_idx_map), num_detectors_2, quiet=quiet
    )
    if g1_g2_weight is not None:
        return False

    if not quiet:
        print("Checking if d2 is fault-bound by d1...")
    g2_g1_weight = _smallest_size_iteration(
        g2_sig_nf, g1_sig_nf, num_detectors_2, len(d1_edge_idx_map), num_detectors_1, quiet=quiet
    )
    return g2_g1_weight is None


@noise_model_params("noise_1", "noise_2")
def is_fault_equivalence(
    noise_1: NoiseModelParam,
    noise_2: NoiseModelParam,
    quiet: bool = True,
) -> bool:
    """
    Given two noise models noise_1 and noise_2 (required to be in pushed out form), determine if they are fault
    equivalent. This requires their underlying diagrams to be semantically equivalent, so stabilisers are only supplied
    once.

    Note that currently only equally weighted noise models are supported.
    Further note that for the result to be sound, the boundary nodes of the two diagrams must be ordered the same.

    :param noise_1: First noise model to check
    :param noise_2: Second noise model to check
    :param quiet: Whether to silence additional informative output
    """
    flip_ops_1 = build_flip_operators(noise_1.diagram())
    pushed_out_noise_1 = pushout(noise_1, flip_ops_1)

    flip_ops_2 = build_flip_operators(noise_2.diagram())
    pushed_out_noise_2 = pushout(noise_2, flip_ops_2)

    return _is_fault_equivalence(
        noise_1=pushed_out_noise_1,
        noise_2=pushed_out_noise_2,
        num_detectors_1=len(flip_ops_1.region_gen_set),
        num_detectors_2=len(flip_ops_2.region_gen_set),
        stabilisers=flip_ops_1.stab_gen_set,  # TODO assert stabiliser space equality
        quiet=quiet,
    )
