import pyzx as zx
import stim

from faulttools.diagram import Diagram
from faulttools.glue.pyzx import from_pyzx
from faulttools.glue.stim import from_stim

type DiagramParam = Diagram | zx.graph.base.BaseGraph


def to_diagram(obj: DiagramParam) -> Diagram:
    if isinstance(obj, Diagram):
        return obj
    elif isinstance(obj, zx.graph.base.BaseGraph):
        return from_pyzx(obj)
    elif isinstance(obj, stim.Circuit):
        return from_stim(obj)[0]
    else:
        raise TypeError(f"Cannot automatically convert type {type(obj)} to {Diagram.__name__}")
