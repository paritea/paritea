from typing import Union

from .pyzx import from_pyzx as from_pyzx
from .. import Diagram

import pyzx as zx

type DiagramParam = Union[Diagram, zx.graph.base.BaseGraph]


def to_diagram(obj: DiagramParam) -> Diagram:
    if isinstance(obj, Diagram):
        return obj
    elif isinstance(obj, zx.graph.base.BaseGraph):
        return from_pyzx(obj)
    else:
        raise TypeError(f"Cannot automatically convert type {type(obj)} to {Diagram.__name__}")
