from faulttools.noise import NoiseModel
from faulttools.util import canonicalize_input
from faulttools.utils import DiagramParam, to_diagram

type NoiseModelParam = NoiseModel | DiagramParam


def to_noise_model(obj: NoiseModelParam) -> NoiseModel:
    if isinstance(obj, NoiseModel):
        return obj
    else:
        return NoiseModel.edge_flip_noise(to_diagram(obj))


def noise_model_params(*param_names: str):
    return canonicalize_input(**dict.fromkeys(param_names, to_noise_model))
