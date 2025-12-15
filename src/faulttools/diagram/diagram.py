from collections.abc import Iterable, Mapping
from copy import deepcopy
from enum import StrEnum
from fractions import Fraction
from typing import Any, Protocol, Self, runtime_checkable

import rustworkx as rx
from recordclass import RecordClass


class NodeType(StrEnum):
    B = "B"  # Boundary
    Z = "Z"  # Z spider
    X = "X"  # X spider
    H = "H"  # H box


class _NodeInfo(RecordClass):
    type: NodeType
    phase: Fraction


class SupportsPositioning(Protocol):
    def set_x(self, node_idx: int, x: int) -> None: ...
    def set_y(self, node_idx: int, y: int) -> None: ...
    def x(self, node_idx: int) -> int: ...
    def y(self, node_idx: int) -> int: ...


@runtime_checkable
class Diagram(SupportsPositioning, Protocol):
    """
    A ZX diagram as an open graph.

    For allowed node types, see the NodeType enum.
    Node data must be an instance of NodeInfo.
    """

    def __init__(self, *, additional_keys: Iterable[str] | None = None):
        self._g = rx.PyGraph[_NodeInfo, None]()
        self._x: dict[int, int] = dict()
        self._y: dict[int, int] = dict()
        self._io: tuple[list[int], list[int]] | None = None
        self._is_io_virtual: bool = True
        # Additional untyped keys for node index mappings
        self.additional_keys = set(additional_keys or [])
        for key in self.additional_keys:
            setattr(self, f"_{key}", dict())
            setattr(self, f"{key}", lambda idx, _key=key: getattr(self, f"_{_key}").get(idx))
            setattr(self, f"set_{key}", lambda idx, arg, _key=key: getattr(self, f"_{_key}").update({idx: arg}) or self)
        self._rebind_methods()

    def _rebind_methods(self):
        # Delegations from the wrapped graph, which must be rebound on each new instance
        self.num_nodes = self._g.num_nodes
        self.has_node = self._g.has_node
        self.node_indices = self._g.node_indices
        self.num_edges = self._g.num_edges
        self.has_edge = self._g.has_edge
        self.edge_list = self._g.edge_list
        self.edge_indices = self._g.edge_indices
        self.edge_indices_from_endpoints = self._g.edge_indices_from_endpoints
        self.get_edge_endpoints_by_index = self._g.get_edge_endpoints_by_index
        self.incident_edges = self._g.incident_edges
        self.has_parallel_edges = self._g.has_parallel_edges
        self.add_edges = self._g.add_edges_from_no_data
        self.remove_edge = self._g.remove_edge
        self.neighbors = self._g.neighbors

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        result._rebind_methods()
        return result

    def add_node(
        self,
        t: NodeType,
        phase: Fraction | None = None,
        x: int | None = None,
        y: int | None = None,
        **kwargs: dict[str, Any],
    ) -> int:
        idx = self._g.add_node(_NodeInfo(t, phase or Fraction(0, 1)))
        if x is not None:
            self._x[idx] = x
        if y is not None:
            self._y[idx] = y
        for key, arg in kwargs.items():
            if key not in self.additional_keys:
                raise ValueError(f"Custom key {key} is not supported on this diagram.")
            getattr(self, f"_{key}")[idx] = arg

        return idx

    def remove_node(self, idx: int) -> None:
        self._g.remove_node(idx)
        self._x.pop(idx, "")
        self._y.pop(idx, "")
        for key in self.additional_keys:
            getattr(self, f"_{key}").pop(idx, "")

    def add_edge(self, a: int, b: int) -> int:
        return self._g.add_edge(a, b, None)

    def compose(self, other: Self, node_map: Mapping[int, int]) -> dict[int, int]:
        """
        Add another diagram into this diagram.

        :param Self other: The other Diagram to add onto this diagram.
        :param Mapping[int, int] node_map: A map from nodes in this diagram to nodes in the other, where an edge should
            be created.

        :returns: new_node_ids: A dictionary mapping node index from the other
            Diagram to the equivalent node index in this Diagram after they've
            been combined.
        """

        new_node_ids = self._g.compose(other._g, {i: (o, None) for i, o in node_map.items()})
        for other_node, new_this_node in new_node_ids.items():
            self._x[new_this_node] = other._x[other_node]
            self._y[new_this_node] = other._y[other_node]
            for key in self.additional_keys.intersection(other.additional_keys):
                getattr(self, f"_{key}")[new_this_node] = getattr(other, f"_{key}")[other_node]

        return new_node_ids

    ### Properties ###

    def type(self, node_idx: int) -> NodeType:
        return self._g.get_node_data(node_idx).type

    def phase(self, node_idx: int) -> Fraction:
        return self._g.get_node_data(node_idx).phase

    def set_x(self, node_idx: int, x: int) -> Self:
        self._x[node_idx] = x
        return self

    def set_y(self, node_idx: int, y: int) -> Self:
        self._y[node_idx] = y
        return self

    def x(self, node_idx: int) -> int:
        return self._x.get(node_idx, -1)

    def y(self, node_idx: int) -> int:
        return self._y.get(node_idx, -1)

    def set_io(self, inputs: list[int], outputs: list[int], *, virtual: bool) -> Self:
        """
        Sets the boundary node indices regarded as inputs / outputs. Their order directly determines their index through
        isomorphic conversion to a states outputs, i.e. they are indexed as <...all-inputs><...all-outputs>.
        """
        if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
            raise ValueError(
                f"IO may not contain duplicate node indices. Unique I/O:"
                f" {len(set(inputs))}/{len(set(outputs))}, Given I/O: {len(inputs)}/{len(outputs)}"
            )
        if not virtual:
            boundaries = set(self.boundary_nodes())
            unique_io = set(inputs).union(set(outputs))
            if unique_io != boundaries:
                raise ValueError(
                    f"The provided IO must be a 1-1 allocation of boundary nodes, or be virtual. "
                    f"Surplus IO: {unique_io.difference(boundaries)}. "
                    f"Unaccounted boundaries: {boundaries.difference(unique_io)}"
                )

        self._io = (inputs, outputs)
        self._is_io_virtual = virtual
        return self

    def io(self) -> tuple[list[int], list[int]]:
        if self._io is None:
            raise ValueError("IO is not set!")

        return self._io

    def io_sorted(self) -> list[int]:
        if self._io is None:
            return sorted(self.boundary_nodes())

        return self._io[0] + self._io[1]

    def virtualize_io(self) -> None:
        if self.is_io_virtual():
            return

        inputs, outputs = self.io()
        new_inputs = [self._g.neighbors(inp)[0] for inp in inputs]
        new_outputs = [self._g.neighbors(out)[0] for out in outputs]
        for b in inputs + outputs:
            self.remove_node(b)

        self.set_io(new_inputs, new_outputs, virtual=True)

    def realize_io(self) -> tuple[list[int], list[int]]:
        if not self.is_io_virtual():
            return self.io()

        inputs, outputs = self.io()
        new_inputs = [self.add_node(NodeType.B) for _ in inputs]
        new_outputs = [self.add_node(NodeType.B) for _ in outputs]
        for inp, new_inp in zip(inputs, new_inputs):
            self.add_edge(inp, new_inp)
        for out, new_out in zip(outputs, new_outputs):
            self.add_edge(out, new_out)
        self.set_io(new_inputs, new_outputs, virtual=False)

        return new_inputs, new_outputs

    def is_io_virtual(self) -> bool:
        return self._is_io_virtual

    ### Convenience ###

    def add_to_phase(self, node_idx: int, phase: Fraction):
        old_phase = self._g.get_node_data(node_idx).phase
        self._g.get_node_data(node_idx).phase = (old_phase + phase) % 2

    def boundary_nodes(self) -> rx.NodeIndices:
        return self._g.filter_nodes(lambda ni: ni.type == NodeType.B)

    def boundary_edges(self) -> set[int]:
        boundary_edges: list[int] = []
        for b in self.boundary_nodes():
            boundary_edges += self._g.incident_edges(b)
        return set(boundary_edges)
