from fractions import Fraction

from paritea.diagram import Diagram, NodeType
from paritea.equivalence.semantic_equivalence import is_semantic_equivalence
from paritea.generate import surface_code_memory_experiment


def test_removing_pi_2_phase():
    # Replacing two pi/2 phase spiders with one is a change in semantics
    d1 = Diagram()
    z1, z2 = d1.add_node(NodeType.Z, phase=Fraction(1, 2)), d1.add_node(NodeType.Z, phase=Fraction(1, 2))
    d1.add_edges([(d1.add_node(NodeType.B), z1), (z1, z2), (z2, d1.add_node(NodeType.B))])

    d2 = Diagram()
    z = d2.add_node(NodeType.Z, phase=Fraction(1, 2))
    d2.add_edges([(d2.add_node(NodeType.B), z), (z, d2.add_node(NodeType.B))])

    assert not is_semantic_equivalence(d1, d2)


def test_surface_code_multi_round():
    """Tests that one round of a surface code memory experiment implements the same+
    linear map as two rounds."""
    d1 = surface_code_memory_experiment(distance=3, rounds=1)
    d2 = surface_code_memory_experiment(distance=3, rounds=2)

    assert is_semantic_equivalence(d1, d2)
