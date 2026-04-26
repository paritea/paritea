use crate::diagram::{Diagram, NodeData, NodeType, Phase};
use crate::pauli::Pauli;
use crate::util::upair;
use fraction::{CheckedDiv, Zero};
use itertools::Itertools;
use rustworkx_core::petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct ExtraIdNode {
    node: NodeIndex,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpandedHadamard {
    r1_node: NodeIndex,
    r2_node: NodeIndex,
    r3_node: NodeIndex,
    origin: NodeIndex,
    flipped_decomposition: bool,
}

pub struct AdditionalNodes {
    extra_id_nodes: Vec<ExtraIdNode>,
    expanded_hadamards: Vec<ExpandedHadamard>,
}

impl AdditionalNodes {
    pub fn new(
        extra_id_nodes: Vec<ExtraIdNode>,
        expanded_hadamards: Vec<ExpandedHadamard>,
    ) -> Self {
        AdditionalNodes {
            extra_id_nodes,
            expanded_hadamards,
        }
    }

    pub fn empty() -> Self {
        AdditionalNodes {
            extra_id_nodes: Vec::new(),
            expanded_hadamards: Vec::new(),
        }
    }

    pub fn add_extra_id_node(&mut self, node: NodeIndex) {
        self.extra_id_nodes.push(ExtraIdNode { node });
    }

    pub fn add_expanded_hadamard(&mut self, expanded_hadamard: ExpandedHadamard) {
        self.expanded_hadamards.push(expanded_hadamard);
    }

    fn remove_extra_id_node(
        adj: &mut HashMap<NodeIndex, HashMap<NodeIndex, bool>>,
        web: &mut HashMap<(NodeIndex, NodeIndex), Pauli>,
        id_node: ExtraIdNode,
    ) {
        let (v1, v2) = adj[&id_node.node].keys().copied().collect_tuple().unwrap();
        web.insert(
            upair(v1, v2),
            *web.get(&upair(v1, id_node.node)).unwrap_or(&Pauli::I),
        );
        adj.get_mut(&v1).unwrap().insert(v2, true);
        adj.get_mut(&v2).unwrap().insert(v1, true);
        web.remove(&upair(v1, id_node.node));
        web.remove(&upair(id_node.node, v2));
        adj.get_mut(&v1).unwrap().remove(&id_node.node);
        adj.get_mut(&id_node.node).unwrap().remove(&v1);
        adj.get_mut(&id_node.node).unwrap().remove(&v2);
        adj.get_mut(&v2).unwrap().remove(&id_node.node);
    }

    fn remove_expanded_hadamard(
        adj: &mut HashMap<NodeIndex, HashMap<NodeIndex, bool>>,
        web: &mut HashMap<(NodeIndex, NodeIndex), Pauli>,
        hadamard: ExpandedHadamard,
    ) {
        let (w1, w2, w3) = (hadamard.r1_node, hadamard.r2_node, hadamard.r3_node);
        let (w1_left, w1_right) = adj[&w1].keys().copied().collect_tuple().unwrap();
        let l = if w1_right == w2 { w1_left } else { w1_right };
        let (w3_left, w3_right) = adj[&w3].keys().copied().collect_tuple().unwrap();
        let r = if w3_left == w2 { w3_right } else { w3_left };
        web.insert(
            upair(l, hadamard.origin),
            *web.get(&upair(l, w1)).unwrap_or(&Pauli::I),
        );
        web.insert(
            upair(hadamard.origin, r),
            *web.get(&upair(w3, r)).unwrap_or(&Pauli::I),
        );
        if !adj.contains_key(&hadamard.origin) {
            adj.insert(hadamard.origin, HashMap::new());
        }
        adj.get_mut(&l).unwrap().insert(hadamard.origin, true);
        adj.get_mut(&hadamard.origin).unwrap().insert(l, true);
        adj.get_mut(&r).unwrap().insert(hadamard.origin, true);
        adj.get_mut(&hadamard.origin).unwrap().insert(r, true);
        web.remove(&upair(l, w1));
        web.remove(&upair(w1, w2));
        web.remove(&upair(w2, w3));
        web.remove(&upair(w3, r));
        adj.get_mut(&l).unwrap().remove(&w1);
        adj.get_mut(&w1).unwrap().remove(&l);
        adj.get_mut(&w1).unwrap().remove(&w2);
        adj.get_mut(&w2).unwrap().remove(&w1);
        adj.get_mut(&w2).unwrap().remove(&w3);
        adj.get_mut(&w3).unwrap().remove(&w2);
        adj.get_mut(&w3).unwrap().remove(&r);
        adj.get_mut(&r).unwrap().remove(&w3);
    }

    pub fn remove_from(&self, d: &Diagram, web: &mut HashMap<(NodeIndex, NodeIndex), Pauli>) {
        let mut adj = d
            .node_indices()
            .map(|n| (n, d.neighbors(n).map(|m| (m, true)).collect()))
            .collect();
        for &id_node in &self.extra_id_nodes {
            Self::remove_extra_id_node(&mut adj, web, id_node);
        }
        for &hadamard in &self.expanded_hadamards {
            Self::remove_expanded_hadamard(&mut adj, web, hadamard);
        }
    }
}

fn place_node_between(d: &mut Diagram, nt: NodeType, n1: NodeIndex, n2: NodeIndex) -> NodeIndex {
    let node = d.add_node(NodeData::from_type(nt));
    d.remove_edge_between(n1, n2);
    d.add_edges([(n1, node), (node, n2)]);

    node
}

const _EULER_DECOMPOSITION_XZX: [NodeType; 3] = [NodeType::X, NodeType::Z, NodeType::X];
const _EULER_DECOMPOSITION_ZXZ: [NodeType; 3] = [NodeType::Z, NodeType::X, NodeType::Z];

/// A cut down version of pyzx.euler_expansion which does not add global scalars and does not
/// prematurely 'merge' spiders.
fn euler_expand_edges(d: &mut Diagram) -> Vec<ExpandedHadamard> {
    let mut expanded_hadamards = Vec::new();
    for v in d.node_indices().collect_vec() {
        if d.node_type(v) != NodeType::H {
            continue;
        }

        let (v1, v2) = d.neighbors(v).collect_tuple().unwrap();

        d.remove_node(v);
        d.add_edge(v1, v2);

        let flip = d.node_type(v1) == d.node_type(v2) && d.node_type(v1) == NodeType::X;

        // Change decomposition to avoid introducing more X-spiders due to adjacent Z-spider
        let pattern = if flip {
            _EULER_DECOMPOSITION_XZX
        } else {
            _EULER_DECOMPOSITION_ZXZ
        };

        let w2 = place_node_between(d, pattern[1], v1, v2);
        d.add_to_phase(w2, Phase::new(1u32, 2u32)); // TODO convert to fraction
        let w1 = place_node_between(d, pattern[0], v1, w2);
        d.add_to_phase(w1, Phase::new(1u32, 2u32));
        let w3 = place_node_between(d, pattern[2], w2, v2);
        d.add_to_phase(w3, Phase::new(1u32, 2u32));

        expanded_hadamards.push(ExpandedHadamard {
            r1_node: w1,
            r2_node: w2,
            r3_node: w3,
            origin: v,
            flipped_decomposition: flip,
        });
    }

    expanded_hadamards
}

fn ensure_red_green(d: &mut Diagram) -> Vec<NodeIndex> {
    let mut new_nodes = Vec::new();
    // Introduce intermediate nodes
    for (s, t) in d.edge_list().collect_vec() {
        if d.node_type(s) == d.node_type(t) {
            let new_type = if d.node_type(s) == NodeType::X {
                NodeType::Z
            } else {
                NodeType::X
            };
            new_nodes.push(place_node_between(d, new_type, s, t));
        }
    }

    // Introduce intermediate nodes for boundary <-> boundary connections
    for (s, t) in d.edge_list().collect_vec() {
        if d.node_type(s) == NodeType::B && d.node_type(t) == NodeType::B {
            new_nodes.push(place_node_between(d, NodeType::X, s, t));
        }
    }

    // Ensure boundaries are not connected to a red spider
    let boundaries = d.boundary_nodes().collect_vec();
    for &boundary in &boundaries {
        let neighbour = d.neighbors(boundary).next().unwrap();
        if d.node_type(neighbour) == NodeType::X {
            new_nodes.push(place_node_between(d, NodeType::Z, boundary, neighbour));
        }
    }

    // Ensure boundaries are not connected to green spiders with nonzero phase or more than one boundary connection
    for boundary in boundaries {
        let neighbour = d.neighbors(boundary).next().unwrap();
        let neighbour_boundaries = d
            .neighbors(neighbour)
            .filter(|&v| d.node_type(v) == NodeType::B)
            .collect_vec();
        if !d.phase(neighbour).is_zero() || neighbour_boundaries.len() > 1 {
            let new_x = place_node_between(d, NodeType::X, boundary, neighbour);
            new_nodes.push(new_x);
            new_nodes.push(place_node_between(d, NodeType::Z, boundary, new_x));
        }
    }

    new_nodes
}

pub fn to_red_green_form(d: &mut Diagram) -> AdditionalNodes {
    if d.has_parallel_edges() {
        panic!("Can only work on diagrams containing no parallel edges."); // TODO result
    }

    // Convert all H-edges and Hadamards to red and green spiders
    let mut additional_nodes = AdditionalNodes::empty();
    for hadamard in euler_expand_edges(d) {
        additional_nodes.add_expanded_hadamard(hadamard);
    }

    // Verify that diagram is clifford
    let offending_vertices = d
        .node_indices()
        .filter(|&n| {
            d.phase(n).checked_div(&Phase::new(2u32, 1u32)).is_none()
                || (d.node_type(n) != NodeType::Z
                    && d.node_type(n) != NodeType::X
                    && d.node_type(n) != NodeType::B)
        })
        .collect_vec();
    if !offending_vertices.is_empty() {
        panic!(
            "Given diagram is not a Clifford diagram up to hadamard expansion. The following \
             vertices are either not of type X,Z,BOUNDARY or have a non-clifford \
             phase: {}",
            offending_vertices
                .iter()
                .map(|v| v.index().to_string())
                .join(", ")
        );
    }

    for node in ensure_red_green(d) {
        additional_nodes.add_extra_id_node(node);
    }

    additional_nodes
}
