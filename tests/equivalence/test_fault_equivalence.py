from fractions import Fraction

import pytest
import pyzx as zx

from faulttools.diagram.conversion import from_pyzx
from faulttools.equivalence import is_fault_equivalence


@pytest.mark.skip(reason="Identity wires are currently not supported")
def test_id_spider_simp():
    """
    Simplification of a single spider to an identity wire.
    From https://arxiv.org/pdf/2506.17181.
    """
    g1 = zx.Graph()
    z = g1.add_vertex(zx.VertexType.Z)
    g1.add_edges([(g1.add_vertex(zx.VertexType.BOUNDARY), z), (z, g1.add_vertex(zx.VertexType.BOUNDARY))])

    g2 = zx.Graph()
    g2.add_edges([(g2.add_vertex(zx.VertexType.BOUNDARY), g2.add_vertex(zx.VertexType.BOUNDARY))])

    assert is_fault_equivalence(g1, g2)


def test_2_pi_2_fuse():
    """
    Fusing two pi/2 spiders into a single pi spider.
    """
    g1 = zx.Graph()
    z1, z2 = g1.add_vertex(zx.VertexType.Z, phase=Fraction(1, 2)), g1.add_vertex(zx.VertexType.Z, phase=Fraction(1, 2))
    g1.add_edges([(g1.add_vertex(zx.VertexType.BOUNDARY), z1), (z1, z2), (z2, g1.add_vertex(zx.VertexType.BOUNDARY))])

    g2 = zx.Graph()
    z = g2.add_vertex(zx.VertexType.Z, phase=1)
    g2.add_edges([(g2.add_vertex(zx.VertexType.BOUNDARY), z), (z, g2.add_vertex(zx.VertexType.BOUNDARY))])

    assert is_fault_equivalence(g1, g2)


@pytest.mark.parametrize("fan_out", [2, 4, 10, 69])
def test_no_leg_spider_fuse(fan_out):
    """
    Fusing a spider with exactly one leg into its neighbor with a variable number of legs.
    From https://arxiv.org/pdf/2506.17181.
    """
    g1 = zx.Graph()
    bz, z = g1.add_vertex(zx.VertexType.Z), g1.add_vertex(zx.VertexType.Z)
    g1.add_edge((bz, z))
    for _ in range(fan_out):
        g1.add_edge((z, g1.add_vertex(zx.VertexType.BOUNDARY)))

    g2 = g1.clone()
    g2.remove_vertex(bz)

    assert is_fault_equivalence(g1, g2)


@pytest.mark.parametrize("ring_size", [3, 4, 5, 6, 7, 8, 9, 10])
def test_collapse_ring(ring_size):
    """
    Collapsing a ring of spiders into a single spider.
    From https://arxiv.org/pdf/2506.17181 and generalised for ring size > 5.
    """
    g1 = zx.Graph()
    b_spiders_2 = [g1.add_vertex(zx.VertexType.BOUNDARY) for _ in range(ring_size)]
    z = g1.add_vertex(zx.VertexType.Z, qubit=0, row=0)
    for i in range(ring_size):
        g1.add_edge((z, b_spiders_2[i]))

    g2 = zx.Graph()
    b_spiders = [g2.add_vertex(zx.VertexType.BOUNDARY) for _ in range(ring_size)]
    z_spiders = [g2.add_vertex(zx.VertexType.Z) for _ in range(ring_size)]
    for i in range(ring_size):
        g2.add_edge((z_spiders[i - 1], z_spiders[i]))
        g2.add_edge((z_spiders[i], b_spiders[i]))

    if ring_size <= 5:
        assert is_fault_equivalence(g1, g2, quiet=False)
    else:
        assert not is_fault_equivalence(g1, g2, quiet=False)


def _add_cat_state(g: zx.graph.base.BaseGraph, size: int, qubit: int = 0, row: int = 0) -> tuple[int, list[int]]:
    z = g.add_vertex(zx.VertexType.Z, qubit=qubit, row=row)
    boundaries = [g.add_vertex(zx.VertexType.BOUNDARY, qubit=qubit + i, row=row + 1) for i in range(size)]
    g.add_edges([(z, b) for b in boundaries])

    return z, boundaries


def _add_cz_layer(g: zx.graph.base.BaseGraph, boundaries: list[int]) -> list[int]:
    """
    Adds a layer of CZ gates to the graph, by converting the given boundaries to Z-spiders.
    Let n = boundaries / 2, then boundaries[i] will be connected to boundaries[i+n].
    :returns The new boundaries
    """
    n = len(boundaries) // 2
    new_bs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit=i, row=2 * (n + 1)) for i in range(2 * n)]
    for i in range(n):
        g.set_type(boundaries[i], zx.VertexType.Z)
        g.set_type(boundaries[i + n], zx.VertexType.Z)
        g.add_edges(
            [
                (boundaries[i], boundaries[i + n]),
                (boundaries[i], new_bs[i]),
                (boundaries[i], new_bs[i + n]),
            ]
        )

    return new_bs


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_cat_state_decomposition(n):
    """
    Expanding a cat state with 2n legs into two cat states with n legs.
    From https://arxiv.org/pdf/2506.17181.
    """
    g1 = zx.Graph()
    _add_cat_state(g1, size=2 * n, qubit=0, row=0)

    g2 = zx.Graph()
    _, bs1 = _add_cat_state(g2, size=n, qubit=2, row=0)
    _, bs2 = _add_cat_state(g2, size=n, qubit=6, row=0)
    _add_cz_layer(g2, [*bs1, *bs2])

    assert is_fault_equivalence(g1, g2, quiet=False)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_cat_state_decomposition_in_context(n):
    """
    Expanding a cat state with 2n legs into two cat states with n legs, in a CZ context that forms new detecting
    regions with the expanded states.
    Based on https://arxiv.org/pdf/2506.17181.
    """

    g1 = zx.Graph()
    _, bs = _add_cat_state(g1, size=2 * n, qubit=0, row=0)
    _add_cz_layer(g1, bs)

    g2 = zx.Graph()
    _, bs1 = _add_cat_state(g2, size=n, qubit=2, row=0)
    _, bs2 = _add_cat_state(g2, size=n, qubit=6, row=0)
    new_bs = _add_cz_layer(g2, [*bs1, *bs2])
    _add_cz_layer(g2, new_bs)

    assert is_fault_equivalence(g1, g2, quiet=False)


def test_cnot_fuse():
    """
    Fusing a series of CNOT gates on the control qubits is NOT a fault equivalent rewrite.
    Based on https://arxiv.org/pdf/2410.17240.
    """

    c = zx.Circuit(5)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 0, 2)
    c.add_gate("CNOT", 0, 3)
    c.add_gate("CNOT", 0, 4)
    g1 = c.to_graph()

    g2 = g1.copy()
    zx.simplify.spider_simp(g2)

    assert not is_fault_equivalence(g1, g2, quiet=False)


def test_cnot_target_fuse_flagged():
    """
    Fusing a series of CNOT gates on the target qubits is a fault equivalent rewrite provided that additional flag
    qubits are employed.

    Based on https://arxiv.org/pdf/2410.17240, https://doi.org/10.22331/q-2018-02-08-53 and external contribution.
    """
    c1 = zx.Circuit(4)

    c1.add_gate("InitAncilla", 4)
    c1.add_gate("InitAncilla", 5)
    c1.add_gate("H", 4)

    c1.add_gate("CNOT", 5, 4)

    for i in range(4):
        c1.add_gate("CNOT", i, 4)

    c1.add_gate("CNOT", 5, 4)

    c1.add_gate("H", 4)
    c1.add_gate("PostSelect", 4)
    c1.add_gate("PostSelect", 5)

    g1 = c1.to_graph(compress_rows=True)
    zx.simplify.id_simp(g1)
    zx.basicrules.color_change(g1, 4)
    zx.basicrules.color_change(g1, 20)

    c2 = zx.Circuit(4)

    c2.add_gate("InitAncilla", 4)
    c2.add_gate("H", 4)

    for i in range(4):
        c2.add_gate("CNOT", i, 4)

    c2.add_gate("H", 4)
    c2.add_gate("PostSelect", 4)

    g2 = c2.to_graph(compress_rows=True)
    zx.simplify.id_simp(g2)
    zx.basicrules.color_change(g2, 4)
    zx.basicrules.color_change(g2, 15)
    zx.simplify.spider_simp(g2)

    assert is_fault_equivalence(g1, g2, quiet=False)


def test_parallel_syndrome_extraction():
    """
    Parallel syndrome extraction.
    From https://arxiv.org/pdf/1804.06995, Figure II.3.C.
    """
    c1 = zx.Circuit(5)

    c1.add_gate("InitAncilla", 5)
    c1.add_gate("InitAncilla", 6)
    c1.add_gate("InitAncilla", 7)
    c1.add_gate("H", 5)
    c1.add_gate("H", 6)

    c1.add_gate("H", 0)
    c1.add_gate("CNOT", 0, 5)
    c1.add_gate("H", 0)
    c1.add_gate("CNOT", 7, 5)
    c1.add_gate("CNOT", 1, 5)
    c1.add_gate("H", 3)
    c1.add_gate("CNOT", 3, 5)
    c1.add_gate("H", 3)
    c1.add_gate("CNOT", 7, 5)
    c1.add_gate("CNOT", 2, 5)

    c1.add_gate("CNOT", 2, 6)
    c1.add_gate("CNOT", 7, 6)
    c1.add_gate("H", 1)
    c1.add_gate("CNOT", 1, 6)
    c1.add_gate("H", 1)
    c1.add_gate("CNOT", 3, 6)
    c1.add_gate("CNOT", 7, 6)
    c1.add_gate("H", 4)
    c1.add_gate("CNOT", 4, 6)
    c1.add_gate("H", 4)

    c1.add_gate("H", 5)
    c1.add_gate("H", 6)
    c1.add_gate("PostSelect", 5)
    c1.add_gate("PostSelect", 6)
    c1.add_gate("PostSelect", 7)

    g1 = c1.to_graph(compress_rows=True)
    zx.simplify.id_simp(g1)
    zx.basicrules.color_change(g1, 12)
    zx.basicrules.color_change(g1, 20)
    zx.basicrules.color_change(g1, 32)
    zx.basicrules.color_change(g1, 40)
    zx.basicrules.color_change(g1, 5)
    zx.basicrules.color_change(g1, 6)
    zx.basicrules.color_change(g1, 44)
    zx.basicrules.color_change(g1, 45)

    c2 = zx.Circuit(5)

    c2.add_gate("InitAncilla", 5)
    c2.add_gate("InitAncilla", 6)
    c2.add_gate("H", 5)
    c2.add_gate("H", 6)

    c2.add_gate("H", 0)
    c2.add_gate("CNOT", 0, 5)
    c2.add_gate("H", 0)
    c2.add_gate("CNOT", 1, 5)
    c2.add_gate("H", 3)
    c2.add_gate("CNOT", 3, 5)
    c2.add_gate("H", 3)
    c2.add_gate("CNOT", 2, 5)

    c2.add_gate("CNOT", 2, 6)
    c2.add_gate("H", 1)
    c2.add_gate("CNOT", 1, 6)
    c2.add_gate("H", 1)
    c2.add_gate("CNOT", 3, 6)
    c2.add_gate("H", 4)
    c2.add_gate("CNOT", 4, 6)
    c2.add_gate("H", 4)

    c2.add_gate("H", 5)
    c2.add_gate("H", 6)
    c2.add_gate("PostSelect", 5)
    c2.add_gate("PostSelect", 6)

    g2 = c2.to_graph(compress_rows=False)
    zx.simplify.id_simp(g2)
    zx.basicrules.color_change(g2, 11)
    zx.basicrules.color_change(g2, 17)
    zx.basicrules.color_change(g2, 25)
    zx.basicrules.color_change(g2, 31)
    zx.basicrules.color_change(g2, 5)
    zx.basicrules.color_change(g2, 6)
    zx.basicrules.color_change(g2, 35)
    zx.basicrules.color_change(g2, 36)
    zx.basicrules.fuse(g2, 13, 16)
    zx.basicrules.fuse(g2, 24, 27)

    assert is_fault_equivalence(
        from_pyzx(g1, convert_had_edges=True), from_pyzx(g2, convert_had_edges=True), quiet=False
    )
