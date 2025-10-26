import pytest
from pyzx import Graph, VertexType

from faulttools import build_flip_operators
from faulttools.diagram.conversion import from_pyzx


def test_asserts_no_connected_boundaries():
    g = Graph()
    bs = g.add_vertices(4)
    z = g.add_vertex(VertexType.Z)

    # Connect two boundaries trivially, and two non-trivially
    g.add_edge((bs[0], bs[2]))
    g.add_edges([(bs[1], z), (z, bs[3])])
    d = from_pyzx(g)

    with pytest.raises(AssertionError, match="The diagram must allocate boundary nodes and edges one-to-one!"):
        build_flip_operators(d)
