from typing import Union

from .pyzx import from_pyzx as from_pyzx, from_pyzx_reversible as from_pyzx_reversible, to_pyzx as to_pyzx
from .stim import from_stim as from_stim
from .. import Diagram

import pyzx as zx
import stim

type DiagramParam = Union[Diagram, zx.graph.base.BaseGraph]


def to_diagram(obj: DiagramParam) -> Diagram:
    if isinstance(obj, Diagram):
        return obj
    elif isinstance(obj, zx.graph.base.BaseGraph):
        return from_pyzx(obj)
    elif isinstance(obj, stim.Circuit):
        return from_stim(obj)[0]
    else:
        raise TypeError(f"Cannot automatically convert type {type(obj)} to {Diagram.__name__}")
