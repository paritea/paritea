import pyzx as zx

from paritea.diagram import Diagram
from paritea.glue.pyzx import from_pyzx

type DiagramParam = Diagram | zx.graph.base.BaseGraph


def to_diagram(obj: DiagramParam) -> Diagram:
    if isinstance(obj, Diagram):
        return obj
    elif isinstance(obj, zx.graph.base.BaseGraph):
        return from_pyzx(obj)
    else:
        raise TypeError(f"Cannot automatically convert type {type(obj)} to {Diagram.__name__}")
