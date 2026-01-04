import pyzx as zx
from pyzx.pauliweb import PauliWeb

from .diagram import Diagram
from .glue.pyzx import to_pyzx
from .pauli import PauliString


def draw(d: Diagram, *, web: PauliString | None = None) -> None:
    """Draws the diagram using PyZX (using the standard conversion). Remaps a given webs indexing to be passed to
    PyZX (this may raise an exception if PyZX does not support drawing webs in the current mode)."""
    g, node_map = to_pyzx(d, with_mapping=True)

    edge_idx_to_pyzx_s_t = {
        d.edge_indices_from_endpoints(s, t)[0]: (node_map[s], node_map[t]) for s, t in d.edge_list()
    }

    def to_pyzx_web(_web: PauliString) -> PauliWeb:
        w = PauliWeb(g)
        for e, p in web.items():
            w.add_edge(edge_idx_to_pyzx_s_t[e], p)
        return w

    if web is not None:
        zx.draw(g, labels=True, pauli_web=to_pyzx_web(web))
    else:
        zx.draw(g, labels=True)