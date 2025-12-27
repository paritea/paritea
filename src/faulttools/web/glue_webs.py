import dataclasses
from copy import deepcopy
from dataclasses import field
from itertools import starmap

import numpy as np
from galois import GF2

from faulttools import PauliString
from faulttools.diagram import Diagram, NodeType
from faulttools.web import compute_pauli_webs


@dataclasses.dataclass(init=True)
class _SubgraphTracker:
    inc_edges: dict[int, "_SubgraphTracker | None"] = field(default_factory=dict)


def auto_partition(d: Diagram, node_sets: list[list[int]]) -> tuple[list[PauliString], list[PauliString]]:
    if d.is_io_virtual():
        raise ValueError("This function can only process diagrams with real IO!")
    if len(node_sets) == 0:
        raise ValueError("No node sets given!")

    allocated_nodes: set[int] = set()
    cut_edges: dict[int, _SubgraphTracker] = {}
    subgraphs: list[tuple[Diagram, dict[int, int]]] = []
    sg_trackers: list[_SubgraphTracker] = []
    for node_set in node_sets:
        if not allocated_nodes.isdisjoint(node_set):
            raise RuntimeError(
                f"Not all node sets are disjoint! Duplicate nodes at least: {allocated_nodes.intersection(node_set)}"
            )
        allocated_nodes.update(node_set)

        subgraph, node_map = d.subgraph(node_set, preserve_data=False)
        tracker = _SubgraphTracker()
        sg_trackers.append(tracker)

        io_nodes = []
        for node in node_set:
            for e, adj in d.incident_edge_index_map(node).items():
                _, other, _ = adj
                if other in node_set:
                    continue

                io_nodes.append(node)
                if e in cut_edges:
                    neighbour = cut_edges[e]
                    tracker.inc_edges[e] = neighbour
                    neighbour.inc_edges[e] = tracker
                else:
                    tracker.inc_edges[e] = None
                    cut_edges[e] = tracker

        reverse_node_map = {n: sn for sn, n in node_map.items()}
        subgraph.set_io([], [reverse_node_map[n] for n in io_nodes], virtual=True)

        edge_map: dict[int, int] = {}
        for se in subgraph.edge_indices():
            s, t = subgraph.get_edge_endpoints_by_index(se)
            edge_map[se] = d.edge_indices_from_endpoints(node_map[s], node_map[t])[0]
        _, real_sub_outputs = subgraph.realize_io()
        for b, d_edge in zip(real_sub_outputs, tracker.inc_edges):
            edge_map[subgraph.incident_edges(b)[0]] = d_edge
        subgraphs.append((subgraph, edge_map))

    unallocated_nodes = [n for n in set(d.node_indices()).difference(allocated_nodes) if d.type(n) != NodeType.B]
    if len(unallocated_nodes) > 0:
        raise ValueError(f"Not all nodes were allocated: {unallocated_nodes}")

    def _find_webs(sg: Diagram, edge_map: dict[int, int]) -> tuple[list[PauliString], list[PauliString]]:
        st, re = compute_pauli_webs(sg)
        new_st = [PauliString({edge_map[e]: p for e, p in s.items()}) for s in st]
        new_re = [PauliString({edge_map[e]: p for e, p in r.items()}) for r in re]
        return new_st, new_re

    webs = list(starmap(_find_webs, subgraphs))

    def _zip_webs_with(
        cur_stabs: list[PauliString],
        next_stabs: list[PauliString],
        zipped_edges: list[int],
        new_boundaries: list[int],
    ) -> tuple[list[PauliString], list[PauliString]]:
        zip_idx_map = {e: i for i, e in enumerate(zipped_edges)}
        zip_idx_map_2 = {e: i for i, e in enumerate(new_boundaries)}
        cur_stabs_compiled = [s.restrict(zipped_edges).compile(zip_idx_map) for s in cur_stabs]
        cur_stabs_compiled_2 = [s.restrict(new_boundaries).compile(zip_idx_map_2) for s in cur_stabs]

        next_stabs_compiled = [s.restrict(zipped_edges).compile(zip_idx_map) for s in next_stabs]
        next_stabs_compiled_2 = [s.restrict(new_boundaries).compile(zip_idx_map_2) for s in next_stabs]

        if len(cur_stabs_compiled) == 0 and len(next_stabs_compiled) == 0:
            return [], []

        # Compute solutions; Add trivial exclusive null spaces
        new_stabs = []
        new_regions = []

        all_compiled = GF2(cur_stabs_compiled + next_stabs_compiled).transpose()
        solutions = all_compiled.null_space()  # Row-matrix of combination vectors for valid Pauli webs

        all_compiled_2 = GF2(cur_stabs_compiled_2 + next_stabs_compiled_2)
        comp_sols = solutions @ all_compiled_2
        stacked = GF2(np.hstack([comp_sols, GF2.Identity(len(comp_sols))]))
        T = stacked.row_reduce()[:, -len(comp_sols) :]

        solutions_basis_changed = T @ solutions

        # Add non-trivial shared null space solutions
        for solution in solutions_basis_changed:
            converted = solution.tolist()
            cur_activations = converted[: len(cur_stabs)]
            next_activations = converted[len(cur_stabs) :]

            next_web: PauliString = PauliString()
            for idx, activated in enumerate(cur_activations):
                if activated:
                    next_web = next_web * cur_stabs[idx]
            save_this = next_web.restrict(zipped_edges)
            for idx, activated in enumerate(next_activations):
                if activated:
                    next_web = next_web * next_stabs[idx]
            next_web = next_web * save_this

            if next_web.restrict(new_boundaries).is_trivial():
                new_regions.append(next_web)
            else:
                new_stabs.append(next_web)

        if len(new_stabs) != len(new_boundaries):
            raise AssertionError(
                f"Something went wrong, I got the wrong number of stabilisers to form a basis (got {len(new_stabs)}, "
                f"need {len(new_boundaries)})!"
            )

        return new_stabs, new_regions

    cur_stabs, cur_regions = webs[0]
    main_tracker = sg_trackers[0]
    while any(main_tracker.inc_edges.values()):
        neighbour = next(n for n in main_tracker.inc_edges.values() if n is not None)
        edges_to_neighbor = [e for e, t in main_tracker.inc_edges.items() if id(t) == id(neighbour)]

        new_inc_edges = {}
        for e in main_tracker.inc_edges:
            if e in edges_to_neighbor:
                continue
            new_inc_edges[e] = main_tracker.inc_edges[e]
        for e in neighbour.inc_edges:
            if e in edges_to_neighbor:
                continue
            new_inc_edges[e] = neighbour.inc_edges[e]
        main_tracker.inc_edges = new_inc_edges

        neighbor_stabs, neighbor_regions = webs[sg_trackers.index(neighbour)]
        nex_stabs, nex_regions = _zip_webs_with(
            cur_stabs, neighbor_stabs, edges_to_neighbor, list(main_tracker.inc_edges.keys())
        )

        cur_stabs = nex_stabs
        cur_regions.extend(neighbor_regions)
        cur_regions.extend(nex_regions)

    return cur_stabs, cur_regions


def webs_through_cuts(
    diagram: Diagram, subgraph_instructions: list[tuple[set[int], list[int], list[int]]]
) -> tuple[list[PauliString], list[PauliString]]:
    """
    Currently assumes that all subgraphs glue [i.outputs, i+1.inputs], and that the diagram does not contain trivial
    edges.
    """
    if diagram.is_io_virtual():
        raise ValueError("This function can only process diagrams with real IO!")
    if len(subgraph_instructions) == 0:
        raise ValueError("No subgraph instructions given!")
    for i in range(len(subgraph_instructions) - 1):
        out = subgraph_instructions[i][2]
        inp = subgraph_instructions[i + 1][1]
        if len(inp) != len(out) or set(inp) != set(out):
            raise ValueError("Concatenated subgraphs must be connected via IO edges!")

    d = deepcopy(diagram)

    subgraphs: list[tuple[Diagram, dict[int, int]]] = []
    remaining_nodes = set(d.node_indices())
    for i, instr in enumerate(subgraph_instructions):
        nodes, input_edges, output_edges = instr
        if not remaining_nodes.issuperset(nodes):
            raise ValueError(f"#Instr[{i}] tried to reserve nodes that were already taken!")
        remaining_nodes.difference_update(nodes)

        sub_inputs = []
        if i != 0:
            for edge in input_edges:
                s, t = d.get_edge_endpoints_by_index(edge)
                if not ((s in nodes) ^ (t in nodes)):
                    raise ValueError(f"#Instr[{i}] The IO edge #{edge} does not cut across the subgraph boundary!")
                sub_inputs.append(s if s in nodes else t)
        else:
            for edge in input_edges:
                s, t = d.get_edge_endpoints_by_index(edge)
                if not ((d.type(s) == NodeType.B) ^ (d.type(t) == NodeType.B)):
                    raise ValueError(f"#Instr[{i}] Exactly one of #{s} and #{t} must be a boundary!")
                sub_inputs.append(t if d.type(s) == NodeType.B else s)

        sub_outputs = []
        if i < len(subgraph_instructions) - 1:
            for edge in output_edges:
                s, t = d.get_edge_endpoints_by_index(edge)
                if not ((s in nodes) ^ (t in nodes)):
                    raise ValueError(f"#Instr[{i}] The IO edge #{edge} does not cut across the subgraph boundary!")
                sub_outputs.append(s if s in nodes else t)
        else:
            for edge in output_edges:
                s, t = d.get_edge_endpoints_by_index(edge)
                if not ((d.type(s) == NodeType.B) ^ (d.type(t) == NodeType.B)):
                    raise ValueError(f"#Instr[{i}] Exactly one of #{s} and #{t} must be a boundary!")
                sub_outputs.append(t if d.type(s) == NodeType.B else s)

        if not set(sub_inputs).issubset(nodes) or not set(sub_outputs).issubset(nodes):
            raise AssertionError("Something went wrong, generated IO must be a subset of included nodes!")

        subgraph, node_map = d.subgraph(list(nodes), preserve_data=False)
        reverse_node_map = {n: sn for sn, n in node_map.items()}
        subgraph.set_io(
            [reverse_node_map[n] for n in sub_inputs],
            [reverse_node_map[n] for n in sub_outputs],
            virtual=True,
        )

        edge_map: dict[int, int] = {}
        for se in subgraph.edge_indices():
            s, t = subgraph.get_edge_endpoints_by_index(se)
            edge_map[se] = d.edge_indices_from_endpoints(node_map[s], node_map[t])[0]
        real_sub_inputs, real_sub_outputs = subgraph.realize_io()
        for b, d_edge in zip(real_sub_inputs, input_edges):
            edge_map[subgraph.incident_edges(b)[0]] = d_edge
        for b, d_edge in zip(real_sub_outputs, output_edges):
            edge_map[subgraph.incident_edges(b)[0]] = d_edge

        subgraphs.append((subgraph, edge_map))

    if len([n for n in remaining_nodes if d.type(n) != NodeType.B]) > 0:
        raise ValueError("Not all nodes were allocated!")

    def _find_webs(sg: Diagram, em: dict[int, int]) -> tuple[list[PauliString], list[PauliString]]:
        st, re = compute_pauli_webs(sg)
        new_st = [PauliString({em[e]: p for e, p in s.items()}) for s in st]
        new_re = [PauliString({em[e]: p for e, p in r.items()}) for r in re]
        return new_st, new_re

    webs = list(starmap(_find_webs, subgraphs))

    cur_stabs, cur_regions = webs[0]
    d_boundary_edges = subgraph_instructions[0][1]
    for i in range(1, len(subgraph_instructions)):
        next_stabs, next_regions = webs[i]
        # Register already closed regions
        cur_regions.extend(next_regions)

        prev_outputs = subgraph_instructions[i - 1][2]
        prev_idx_map = {e: i for i, e in enumerate(prev_outputs)}
        cur_stabs_exclusive, cur_stabs_shared = [], []
        for stab in cur_stabs:
            if stab.restrict(prev_outputs).is_trivial():
                cur_stabs_exclusive.append(stab)
            else:
                cur_stabs_shared.append(stab)
        cur_stabs_shared_compiled = [s.restrict(prev_outputs).compile(prev_idx_map) for s in cur_stabs_shared]

        next_inputs, next_outputs = subgraph_instructions[i][1], subgraph_instructions[i][2]
        next_idx_map = {e: i for i, e in enumerate(next_inputs)}
        next_stabs_exclusive, next_stabs_shared = [], []
        for stab in next_stabs:
            if stab.restrict(next_inputs).is_trivial():
                next_stabs_exclusive.append(stab)
            else:
                next_stabs_shared.append(stab)
        next_stabs_shared_compiled = [s.restrict(next_inputs).compile(next_idx_map) for s in next_stabs_shared]

        all_compiled = GF2(cur_stabs_shared_compiled + next_stabs_shared_compiled)
        solutions = all_compiled.transpose().null_space().row_reduce()

        # Compute solutions; Add trivial exclusive null spaces
        new_stabs = cur_stabs_exclusive + next_stabs_exclusive
        new_regions = []

        # Add non-trivial shared null space solutions
        for solution in solutions:
            converted = solution.tolist()
            cur_activations = converted[: len(cur_stabs_shared)]
            next_activations = converted[len(cur_stabs_shared) :]

            next_web: PauliString = PauliString()
            for idx, activated in enumerate(cur_activations):
                if activated:
                    next_web = next_web * cur_stabs_shared[idx]
            save_this = next_web.restrict(next_inputs)
            for idx, activated in enumerate(next_activations):
                if activated:
                    next_web = next_web * next_stabs_shared[idx]
            next_web = next_web * save_this

            if next_web.restrict(d_boundary_edges + next_outputs).is_trivial():
                new_regions.append(next_web)
            else:
                new_stabs.append(next_web)

        if len(new_stabs) != len(d_boundary_edges) + len(next_outputs):
            raise AssertionError(
                f"Something went wrong, I got the wrong number of stabilisers to form a basis (got {len(new_stabs)}, "
                f"need {len(d_boundary_edges) + len(next_outputs)})!"
            )

        cur_stabs = new_stabs
        cur_regions.extend(new_regions)

    return cur_stabs, cur_regions
