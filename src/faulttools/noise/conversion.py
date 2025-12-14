from typing import Union

from ..diagram.conversion import to_diagram, DiagramParam
from .model import NoiseModel
from ..util import canonicalize_input

type NoiseModelParam = Union[NoiseModel, DiagramParam]


def to_noise_model(obj: NoiseModelParam) -> NoiseModel:
    if isinstance(obj, NoiseModel):
        return obj
    else:
        return NoiseModel.edge_flip_noise(to_diagram(obj))


def noise_model_params(*param_names: str):
    return canonicalize_input(**{name: to_noise_model for name in param_names})
