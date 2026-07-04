import sinter
from stim import CompiledDemSampler, DetectorErrorModel
from math import sqrt
from scipy.stats import norm


class DemWrappingCircuit:
    """
    A wrapper for a detector error model to duck type as a circuit for the purposes of sinter.collect calls.
    """

    def __init__(self, dem: DetectorErrorModel, num_detectors: int, num_observables: int):
        self.dem = dem
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def __str__(self):
        return "some-stable-value"

    def compile_detector_sampler(self):
        # Note that the DEM has to be compiled on demand as the compiled version cannot be pickled for transfer to
        # sampling processes.
        return DemSamplerDuckTypingAsDetectorSampler(self.dem.compile_sampler())


class DemSamplerDuckTypingAsDetectorSampler:
    """
    A wrapper for a detector error model sampler to duck type as a circuit detector sampler for the purposes of
    sinter.collect calls. In particular, the DEM sampler always samples with separated observables and thus does not
    know the corresponding keyword argument.
    """

    def __init__(self, dem_sampler: CompiledDemSampler):
        self.dem_sampler = dem_sampler

    def sample(self, shots, separate_observables, **kwargs):
        if not separate_observables:
            raise ValueError("Can only sample with separate observables!")
        # Ignore observable separation since it is always active for a dem sampler
        det_data, obs_data, _ = self.dem_sampler.sample(shots, **kwargs)

        return det_data, obs_data


def wrap_dem_as_sinter_task(dem: DetectorErrorModel, *_, **kwargs) -> sinter.Task:
    """
    Wraps a detector error model as a sinter.Task object that can be sampled via sinter.collect calls.

    Callers may provide all keyword arguments of sinter.Task except 'circuit' and 'detector_error_model'.

    :param dem: the detector error model to wrap
    :param kwargs: any sinter.Task keyword arguments except 'circuit' and 'detector_error_model'
    :return: the sinter.Task
    """
    if "circuit" in kwargs or "detector_error_model" in kwargs:
        raise ValueError("Circuit and DEM for sinter task are determined by this function and may not be provided!")

    return sinter.Task(
        circuit=DemWrappingCircuit(dem, num_detectors=dem.num_detectors, num_observables=dem.num_observables),
        detector_error_model=dem,
        **kwargs,
    )


def _wilson_interval(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a single binomial proportion. Returns (low, high)."""
    if n == 0:
        raise ValueError("n must be > 0")

    p = k / n
    z = norm.ppf(1 - (1 - confidence) / 2)

    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + (z * z) / (4 * n * n)) / denom

    return center - margin, center + margin


def _confidence_interval_for_difference(
    a_errors: int,
    a_shots: int,
    b_errors: int,
    b_shots: int,
    confidence: float = 0.90,
):
    """Newcombe CI for difference of proportions: (p1 - p2).
    Uses Wilson intervals and combines them as: [L1 - U2, U1 - L2]."""

    if not (0 < confidence < 1):
        raise ValueError("confidence_level must be in (0, 1)")

    a_l, a_h = _wilson_interval(a_errors, a_shots, confidence)
    b_l, b_h = _wilson_interval(b_errors, b_shots, confidence)

    return a_l - b_h, a_h - b_l


def error_rates_equal(
    a: sinter.TaskStats,
    b: sinter.TaskStats,
    *,
    delta: float = 1e-5,
    confidence: float = 0.95,
) -> bool:
    """Asserts that the error rates derived from the two sinter products is at most
    `delta`, with the given `confidence`."""
    if confidence < 0 or confidence > 1:
        raise ValueError(f"Only 0 <= confidence <= 1 is allowed, {confidence} given.")

    low, high = _confidence_interval_for_difference(a.errors, a.shots, b.errors, b.shots, confidence)

    return low >= -delta and high <= delta
