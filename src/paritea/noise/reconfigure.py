from bitgauss import BitMatrix

from paritea.noise import Fault, NoiseModel


def try_reconfigure_matchable[T](
    nm: NoiseModel[T],
    *,
    only_detectors: set[int] | None = None,
) -> tuple[NoiseModel[T], bool]:
    """Reconfigures the detectors that faults flip such that the noise model is
    matchable (i.e. every fault flips at most two detectors) if possible. Returns the
    new noise model and whether the operation was successful. In case it was not, the
    returned model may still contain more matchable faults than the original."""

    max_detector = max(d for f in nm.atomic_faults() for d in f.detector_flips)
    af = list(nm.atomic_faults_with_values())

    def _from_detectors(detectors: frozenset[int]) -> list[bool]:
        return [(i in detectors.intersection(only_detectors or [])) for i in range(max_detector)]

    # Goal is faults (columns) flipping at most two detectors (rows), so construct rows
    # from faults and compute the graphic form on the transpose.
    m = BitMatrix.from_list([_from_detectors(f.detector_flips) for f, _ in af])
    m.transpose_inplace()
    m, hyper = m.graphic_form_partial()
    m.transpose_inplace()

    def _to_detectors(old_flips: frozenset[int], row: list[bool]) -> frozenset[int]:
        new_flips = frozenset(i for i, b in enumerate(row) if b)
        return new_flips.union(old_flips.difference(only_detectors or []))

    new_nm = NoiseModel(
        nm.diagram,
        {
            Fault(f.edge_flips, _to_detectors(f.detector_flips, row)): vs
            for (f, vs), row in zip(af, m.to_list())
        },
    )
    return new_nm, len(hyper) == 0
