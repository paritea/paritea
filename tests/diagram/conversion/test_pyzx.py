import pyzx.generate
from pyzx import Graph

from faulttools.diagram.conversion import from_pyzx


def test_empty():
    d = from_pyzx(Graph())
    assert d.num_nodes() == 0
    assert d.num_edges() == 0


def test_random():
    g = pyzx.generate.cliffordT(qubits=4, depth=4)
    d = from_pyzx(g)

    assert d.num_nodes() == g.num_vertices()
    assert d.num_edges() == g.num_edges()
