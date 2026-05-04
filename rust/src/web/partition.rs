use crate::diagram::{Diagram, NodeType};
use crate::pauli::PauliString;
use crate::web::compute_pauli_webs;
use bitgauss::BitMatrix;
use itertools::{Itertools, enumerate};
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::stable_graph::EdgeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;
use std::collections::{BTreeSet, HashMap};

#[derive(Default, Clone)]
struct SubgraphTracker {
    inc_edges: HashMap<EdgeIndex, Option<usize>>,
}

fn find_webs(
    sg: &Diagram,
    edge_map: HashMap<EdgeIndex, EdgeIndex>,
) -> (Vec<PauliString>, Vec<PauliString>) {
    let remap_edges =
        |s: PauliString| PauliString(s.0.into_iter().map(|(k, v)| (edge_map[&k], v)).collect());

    let (st, re) = compute_pauli_webs(sg);
    (
        st.into_iter().map(remap_edges).collect(),
        re.into_iter().map(remap_edges).collect(),
    )
}

pub fn zip_webs(
    cur_stabs: Vec<PauliString>,
    next_stabs: Vec<PauliString>,
    zipped_edges: Vec<EdgeIndex>,
    new_boundaries: Vec<EdgeIndex>,
) -> (Vec<PauliString>, Vec<PauliString>) {
    // Prepare and compile stabilisers for both subdiagrams
    let zip_idx_map = HashMap::from_iter(enumerate(zipped_edges.clone()).map(|(i, e)| (e, i)));
    let boundary_idx_map =
        HashMap::from_iter(enumerate(new_boundaries.clone()).map(|(i, e)| (e, i)));
    let cur_stabs_compiled = cur_stabs
        .iter()
        .map(|s| s.restrict(&zipped_edges).compile(&zip_idx_map))
        .collect_vec();
    let cur_stabs_boundary_compiled = cur_stabs
        .iter()
        .map(|s| s.restrict(&new_boundaries).compile(&boundary_idx_map))
        .collect_vec();

    let next_stabs_compiled = next_stabs
        .iter()
        .map(|s| s.restrict(&zipped_edges).compile(&zip_idx_map))
        .collect_vec();
    let next_stabs_boundary_compiled = next_stabs
        .iter()
        .map(|s| s.restrict(&new_boundaries).compile(&boundary_idx_map))
        .collect_vec();

    if cur_stabs_compiled.len() == 0 && next_stabs_compiled.len() == 0 {
        return (vec![], vec![]);
    }

    // Compute matchings over shared edges
    let all_compiled =
        BitMatrix::hstack_from_iter(cur_stabs_compiled.iter().chain(next_stabs_compiled.iter()));
    // Row-matrix of combination vectors for valid matches
    let solutions = BitMatrix::vstack_from_iter(&all_compiled.nullspace());

    // Compute a basis change to extract the maximum number of detecting regions possible
    let all_boundary_compiled = BitMatrix::hstack_from_iter(
        cur_stabs_boundary_compiled
            .iter()
            .chain(next_stabs_boundary_compiled.iter()),
    )
    .transposed();
    let boundary_solutions = &solutions * &all_boundary_compiled;
    let mut proxy = BitMatrix::identity(boundary_solutions.rows());
    let mut stacked = boundary_solutions.clone();
    stacked.gauss_with_proxy(true, 1, &mut proxy);
    let basis_change = proxy;
    let solutions_basis_changed = &basis_change * &solutions;

    // Extract webs from matching information
    let mut new_stabs = Vec::new();
    let mut new_regions = Vec::new();
    for i in 0..solutions_basis_changed.rows() {
        let solution = solutions_basis_changed
            .row(i)
            .iter()
            .take(solutions_basis_changed.cols())
            .collect_vec();

        let mut next_web: PauliString = PauliString::default();
        // Activate current stabilisers (the first `cur_stabs.len()` entries of the solution)
        for (idx, &activated) in enumerate(solution.iter().take(cur_stabs.len())) {
            if activated {
                next_web = next_web * &cur_stabs[idx]
            }
        }

        // Activate next stabilisers (the remaining entries of the solution)
        let shared_edges = next_web.restrict(&zipped_edges);
        for (idx, &activated) in enumerate(solution.iter().dropping(cur_stabs.len())) {
            if activated {
                next_web = next_web * &next_stabs[idx]
            }
        }
        // Reapply action on shared edges since they cancelled out previously
        next_web = next_web * &shared_edges;

        if next_web.restrict(&new_boundaries).is_trivial() {
            new_regions.push(next_web);
        } else {
            new_stabs.push(next_web);
        }
    }

    if new_stabs.len() != new_boundaries.len() {
        panic!(
            "Something went wrong, I got the wrong number of stabilisers to form a basis (got {}, need {})!",
            new_stabs.len(),
            new_boundaries.len()
        );
    }

    (new_stabs, new_regions)
}

/// Computes the Pauli webs of the given diagram with the provided partitions by calculating the
/// Pauli webs of each subdiagram individually and combining the results. Provided partitions must
/// cover all nodes except for boundaries.
pub fn pauli_webs_through_partitions(
    d: &Diagram,
    partitions: Vec<BTreeSet<NodeIndex>>,
) -> (Vec<PauliString>, Vec<PauliString>) {
    if d.is_io_virtual() {
        panic!("This function can only process diagrams with real IO!"); // TODO result
    }

    if partitions.len() == 0 {
        panic!("No partitions given!"); // TODO result
    }

    let mut allocated_nodes = BTreeSet::<NodeIndex>::new();
    let mut cut_edges = HashMap::<EdgeIndex, usize>::new();
    let mut subgraphs = Vec::<(Diagram, HashMap<EdgeIndex, EdgeIndex>)>::new();
    let mut sg_trackers = Vec::<SubgraphTracker>::new();

    // Extract subgraphs from partitions and build partition neighbour tracking graph
    for part in partitions {
        if !allocated_nodes.is_disjoint(&part.iter().copied().collect()) {
            panic!(
                "Not all partitions are disjoint! Duplicate nodes at least: {:?}",
                allocated_nodes.intersection(&part)
            ); // TODO result
        }
        allocated_nodes.extend(&part);

        let (mut subgraph, node_map) = d.subgraph(part.iter().copied());
        let reverse_node_map = node_map
            .iter()
            .map(|(&k, &v)| (v, k))
            .collect::<HashMap<_, _>>();

        let mut tracker = SubgraphTracker::default();
        let tracker_id = sg_trackers.len();

        let mut io_nodes = Vec::new();
        for &node in part.iter() {
            for e in d.edges(node) {
                if part.contains(&e.target()) {
                    continue;
                };

                io_nodes.push((node, e.id()));
                if cut_edges.contains_key(&e.id()) {
                    let neighbour_id = cut_edges[&e.id()];
                    let neighbour = &mut sg_trackers[neighbour_id];
                    tracker.inc_edges.insert(e.id(), Some(neighbour_id));
                    neighbour.inc_edges.insert(e.id(), Some(tracker_id));
                } else {
                    tracker.inc_edges.insert(e.id(), None);
                    cut_edges.insert(e.id(), tracker_id);
                }
            }
        }

        subgraph.set_virtual_io(
            vec![],
            io_nodes.iter().map(|&n| node_map[&n.0]).collect_vec(),
        );

        let mut edge_map = HashMap::<EdgeIndex, EdgeIndex>::new();
        for se in subgraph.edge_indices() {
            let (s, t) = subgraph.edge_endpoints(se).unwrap();
            edge_map.insert(
                se,
                d.edges_connecting(reverse_node_map[&s], reverse_node_map[&t])
                    .next()
                    .unwrap()
                    .id(),
            );
        }

        let (_, real_sub_outputs) = subgraph.realize_io();
        for (&b, &(_, d_edge)) in real_sub_outputs.iter().zip(io_nodes.iter()) {
            edge_map.insert(subgraph.edges(b).next().unwrap().id(), d_edge);
        }
        sg_trackers.push(tracker);
        subgraphs.push((subgraph, edge_map))
    }

    {
        let nodes = BTreeSet::from_iter(d.node_indices());
        let unallocated_nodes = nodes
            .difference(&allocated_nodes)
            .filter(|&&n| d.node_type(n) != NodeType::B)
            .collect_vec();
        if unallocated_nodes.len() > 0 {
            panic!("Not all nodes were allocated: {:?}", unallocated_nodes);
        }
    }

    // Find webs for all subdiagrams
    let webs = subgraphs
        .into_iter()
        .map(|(d, edge_map)| find_webs(&d, edge_map))
        .collect_vec();

    // Zip all webs together
    let (mut cur_stabs, mut cur_regions) = webs[0].clone();
    let main_tracker_id = 0;
    while let Some(neighbour_id) = sg_trackers[main_tracker_id]
        .inc_edges
        .values()
        .find_map(|&s| s)
    {
        let edges_to_neighbour = sg_trackers[main_tracker_id]
            .inc_edges
            .iter()
            .filter_map(|(&e, &tracker_id)| {
                if tracker_id == Some(neighbour_id) {
                    Some(e)
                } else {
                    None
                }
            })
            .collect_vec();

        let new_edges: HashMap<EdgeIndex, Option<usize>> = {
            let neighbour = &sg_trackers[neighbour_id];
            sg_trackers[main_tracker_id]
                .inc_edges
                .iter()
                .filter(|(e, _)| !edges_to_neighbour.contains(e))
                .chain(
                    neighbour
                        .inc_edges
                        .iter()
                        .filter(|(e, _)| !edges_to_neighbour.contains(e)),
                )
                .map(|(&e, &t_id)| (e, t_id))
                .collect()
        };

        let (neighbour_stabs, neighbour_regions) = webs[neighbour_id].clone();
        let (nex_stabs, nex_regions) = zip_webs(
            cur_stabs,
            neighbour_stabs,
            edges_to_neighbour,
            new_edges.keys().copied().collect_vec(),
        );
        sg_trackers[main_tracker_id].inc_edges = new_edges;

        cur_stabs = nex_stabs;
        cur_regions.extend(neighbour_regions);
        cur_regions.extend(nex_regions);
    }

    (cur_stabs, cur_regions)
}
