from typing import Union

import pyzx as zx

from .. import Diagram
from .pyzx import from_pyzx as from_pyzx
from .pyzx import from_pyzx_reversible as from_pyzx_reversible
from .pyzx import to_pyzx as to_pyzx

type DiagramParam = Union[Diagram, zx.graph.base.BaseGraph]


def to_diagram(obj: DiagramParam) -> Diagram:
    if isinstance(obj, Diagram):
        return obj
    elif isinstance(obj, zx.graph.base.BaseGraph):
        return from_pyzx(obj)
    else:
        raise TypeError(f"Cannot automatically convert type {type(obj)} to {Diagram.__name__}")
