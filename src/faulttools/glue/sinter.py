import sinter
from stim import DetectorErrorModel, CompiledDemSampler


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
        assert separate_observables
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
