import dataclasses
from dataclasses import field
from itertools import starmap

import numpy as np
from galois import GF2

from faulttools import PauliString
from faulttools.diagram import Diagram, NodeType
from faulttools.web import compute_pauli_webs


@dataclasses.dataclass(init=True, repr=False)
class _SubgraphTracker:
    inc_edges: dict[int, "_SubgraphTracker | None"] = field(default_factory=dict)


def _find_webs(sg: Diagram, edge_map: dict[int, int]) -> tuple[list[PauliString], list[PauliString]]:
    st, re = compute_pauli_webs(sg)
    new_st = [PauliString({edge_map[e]: p for e, p in s.items()}) for s in st]
    new_re = [PauliString({edge_map[e]: p for e, p in r.items()}) for r in re]
    return new_st, new_re


def _zip_webs(
    cur_stabs: list[PauliString],
    next_stabs: list[PauliString],
    zipped_edges: list[int],
    new_boundaries: list[int],
) -> tuple[list[PauliString], list[PauliString]]:
    # Prepare and compile stabilisers for both subdiagrams
    zip_idx_map = {e: i for i, e in enumerate(zipped_edges)}
    boundary_idx_map = {e: i for i, e in enumerate(new_boundaries)}
    cur_stabs_compiled = [s.restrict(zipped_edges).compile(zip_idx_map) for s in cur_stabs]
    cur_stabs_boundary_compiled = [s.restrict(new_boundaries).compile(boundary_idx_map) for s in cur_stabs]

    next_stabs_compiled = [s.restrict(zipped_edges).compile(zip_idx_map) for s in next_stabs]
    next_stabs_boundary_compiled = [s.restrict(new_boundaries).compile(boundary_idx_map) for s in next_stabs]

    if len(cur_stabs_compiled) == 0 and len(next_stabs_compiled) == 0:
        return [], []

    # Compute matchings over shared edges
    all_compiled = GF2(cur_stabs_compiled + next_stabs_compiled).transpose()
    solutions = all_compiled.null_space()  # Row-matrix of combination vectors for valid matches

    # Compute a basis change to extract the maximum number of detecting regions possible
    all_boundary_compiled = GF2(cur_stabs_boundary_compiled + next_stabs_boundary_compiled)
    boundary_solutions = solutions @ all_boundary_compiled
    stacked = GF2(np.hstack([boundary_solutions, GF2.Identity(len(boundary_solutions))]))
    basis_change = stacked.row_reduce()[:, -len(boundary_solutions) :]
    solutions_basis_changed = basis_change @ solutions

    # Extract webs from matching information
    new_stabs = []
    new_regions = []
    for solution in solutions_basis_changed:
        converted = solution.tolist()
        cur_activations = converted[: len(cur_stabs)]
        next_activations = converted[len(cur_stabs) :]

        next_web: PauliString = PauliString()
        for idx, activated in enumerate(cur_activations):
            if activated:
                next_web *= cur_stabs[idx]
        shared_edges = next_web.restrict(zipped_edges)
        for idx, activated in enumerate(next_activations):
            if activated:
                next_web *= next_stabs[idx]
        next_web *= shared_edges

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


def pauli_webs_through_partitions(
    d: Diagram, *, partitions: list[list[int]]
) -> tuple[list[PauliString], list[PauliString]]:
    if d.is_io_virtual():
        raise ValueError("This function can only process diagrams with real IO!")

    if len(partitions) == 0:
        raise ValueError("No partitions given!")

    allocated_nodes: set[int] = set()
    cut_edges: dict[int, _SubgraphTracker] = {}
    subgraphs: list[tuple[Diagram, dict[int, int]]] = []
    sg_trackers: list[_SubgraphTracker] = []
    # Extract subgraphs from partitions and build partition neighbour tracking graph
    for part in partitions:
        if not allocated_nodes.isdisjoint(part):
            raise RuntimeError(
                f"Not all partitions are disjoint! Duplicate nodes at least: {allocated_nodes.intersection(part)}"
            )
        allocated_nodes.update(part)

        subgraph, node_map = d.subgraph(part, preserve_data=False)
        tracker = _SubgraphTracker()
        sg_trackers.append(tracker)

        io_nodes = []
        for node in part:
            for e, adj in d.incident_edge_index_map(node).items():
                _, other, _ = adj
                if other in part:
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

    # Find webs for all subdiagrams
    webs = list(starmap(_find_webs, subgraphs))

    # Zip all webs together
    cur_stabs, cur_regions = webs[0]
    main_tracker = sg_trackers[0]
    while any(main_tracker.inc_edges.values()):
        neighbour = next(n for n in main_tracker.inc_edges.values() if n is not None)
        edges_to_neighbour = [e for e, t in main_tracker.inc_edges.items() if t is neighbour]

        main_tracker.inc_edges = {e: tr for e, tr in main_tracker.inc_edges.items() if e not in edges_to_neighbour}
        main_tracker.inc_edges |= {e: tr for e, tr in neighbour.inc_edges.items() if e not in edges_to_neighbour}

        neighbour_stabs, neighbour_regions = webs[sg_trackers.index(neighbour)]
        nex_stabs, nex_regions = _zip_webs(
            cur_stabs, neighbour_stabs, edges_to_neighbour, list(main_tracker.inc_edges.keys())
        )

        cur_stabs = nex_stabs
        cur_regions.extend(neighbour_regions)
        cur_regions.extend(nex_regions)

    return cur_stabs, cur_regions
