from copy import deepcopy

from paritea.diagram import Diagram, NodeType


def test_additional_keys_deepcopy():
    d = Diagram(additional_keys=["foo"])
    n = d.add_node(NodeType.Z)
    d.set_foo(n, "bar")
    assert d.foo(n) == "bar"

    d2 = deepcopy(d)
    assert d2.foo(n) == "bar"

    d2.set_foo(n, "baz")
    assert d2.foo(n) == "baz"
    assert d.foo(n) == "bar"
