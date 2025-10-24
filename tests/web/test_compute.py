from pyzx import Graph, VertexType

import generate
from faulttools import PauliString
from faulttools.diagram.conversion import from_pyzx
from faulttools.web import compute_pauli_webs


def test_identity_webs():
    g = Graph()
    b1 = g.add_vertex(VertexType.BOUNDARY)
    b2 = g.add_vertex(VertexType.BOUNDARY)
    n = g.add_vertex(VertexType.Z)
    g.add_edges([(b1, n), (n, b2)])
    d = from_pyzx(g)

    stabs, regions = compute_pauli_webs(d)

    assert set(stabs) == {PauliString("XX"), PauliString("ZZ")}
    assert len(list(regions)) == 0


def test_zweb_webs():
    d = from_pyzx(generate.zweb(2, 2))
    stabs, regions = compute_pauli_webs(d)

    assert len(list(stabs)) == 4
    assert len(list(regions)) == 1
