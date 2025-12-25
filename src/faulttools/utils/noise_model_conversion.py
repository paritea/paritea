from faulttools.noise import NoiseModel
from faulttools.util import canonicalize_input
from faulttools.utils import DiagramParam, to_diagram

type NoiseModelParam[T] = NoiseModel[T] | DiagramParam


def to_noise_model[T](obj: NoiseModelParam[T]) -> NoiseModel[T]:
    if isinstance(obj, NoiseModel):
        return obj
    else:
        return NoiseModel.weighted_edge_flip_noise(to_diagram(obj))


def noise_model_params(*param_names: str):
    return canonicalize_input(**dict.fromkeys(param_names, to_noise_model))
