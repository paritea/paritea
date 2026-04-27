use crate::diagram::{Diagram, NodeType};
use crate::pauli::Pauli;
use crate::util::upair;
use bitgauss::BitMatrix;
use rustc_hash::FxHashMap;
use rustworkx_core::petgraph::graph::NodeIndex;
use std::collections::HashSet;

pub struct GraphOrdering {
    graph_to_ordering: FxHashMap<NodeIndex, usize>,
    ordering_to_graph: FxHashMap<usize, NodeIndex>,

    pub z_boundaries: FxHashMap<NodeIndex, NodeIndex>,
    pub internal_spiders: Vec<NodeIndex>,
    pub pi_2_spiders: Vec<NodeIndex>,
}

impl GraphOrdering {
    pub fn ord(&self, s: NodeIndex) -> usize {
        self.graph_to_ordering[&s]
    }

    pub fn graph(&self, o: usize) -> NodeIndex {
        self.ordering_to_graph[&o]
    }
}

pub fn determine_ordering(d: &Diagram) -> GraphOrdering {
    let boundaries = d.boundary_nodes().collect::<HashSet<_>>();
    let z_boundaries = boundaries
        .iter()
        .map(|&b| (d.neighbors(b).next().unwrap(), b))
        .collect::<FxHashMap<_, _>>();
    let internal_spiders = d
        .node_indices()
        .collect::<HashSet<_>>()
        .difference(&boundaries)
        .copied()
        .collect::<HashSet<NodeIndex>>()
        .difference(&z_boundaries.keys().copied().collect())
        .copied()
        .collect::<HashSet<_>>();
    let pi_2_spiders = internal_spiders
        .iter()
        .filter(|&&s| d.phase(s).denom() == Some(&2))
        .copied()
        .collect::<HashSet<_>>();

    let mut graph_to_ordering: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut ordering_to_graph: FxHashMap<usize, NodeIndex> = FxHashMap::default();
    let mut idx = 0;
    for boundary in z_boundaries.keys() {
        graph_to_ordering.insert(*boundary, idx);
        ordering_to_graph.insert(idx, *boundary);
        idx += 1;
    }
    for &internal in internal_spiders.difference(&pi_2_spiders) {
        graph_to_ordering.insert(internal, idx);
        ordering_to_graph.insert(idx, internal);
        idx += 1;
    }
    for &pi_2_spider in &pi_2_spiders {
        graph_to_ordering.insert(pi_2_spider, idx);
        ordering_to_graph.insert(idx, pi_2_spider);
        idx += 1;
    }

    GraphOrdering {
        graph_to_ordering,
        ordering_to_graph,
        z_boundaries,
        internal_spiders: internal_spiders.into_iter().collect(),
        pi_2_spiders: pi_2_spiders.into_iter().collect(),
    }
}

pub fn create_firing_verification(d: &Diagram, ordering: &GraphOrdering) -> BitMatrix {
    let num_z_boundaries = ordering.z_boundaries.len();
    let num_non_boundary_spiders = num_z_boundaries + ordering.internal_spiders.len();

    let (rows, cols) = (
        num_non_boundary_spiders,
        num_non_boundary_spiders + num_z_boundaries,
    );
    let mut m_d = BitMatrix::zeros(rows, cols);
    // Init with identity, padded down and right with zeros
    for i in 0..num_z_boundaries {
        m_d.set_bit(i, i, true);
    }
    // Store adjacency matrix to the right of the identity
    for (s, t) in d.edge_list() {
        if d.node_type(s) != NodeType::B && d.node_type(t) != NodeType::B {
            let i_row = ordering.ord(s);
            let i_col = ordering.ord(t);
            m_d.set_bit(i_row, i_col + num_z_boundaries, true);
            m_d.set_bit(i_col, i_row + num_z_boundaries, true);
        }
    }
    // Subtract identity for pi/2 spiders in the bottom right corner
    let num_pi_2 = ordering.pi_2_spiders.len();
    for i in 0..num_pi_2 {
        let (r, c) = (rows - num_pi_2 + i, cols - num_pi_2 + i);
        m_d.set_bit(r, c, m_d.bit(r, c) ^ true);
    }

    m_d
}

pub fn convert_firing_assignment_to_web_prototype(
    d: &Diagram,
    ordering: &GraphOrdering,
    v: Vec<bool>,
) -> FxHashMap<(NodeIndex, NodeIndex), Pauli> {
    let mut prot = FxHashMap::default(); // TODO defaultdict

    for (&adj_vertex, &g_vertex) in ordering.ordering_to_graph.iter() {
        let g_type = d.node_type(g_vertex);
        // Fire all green spiders with full red edges and thus their red neighbours
        if g_type == NodeType::Z && v[adj_vertex + ordering.z_boundaries.len()] == true {
            for _n in d.neighbors(g_vertex) {
                prot.insert(
                    upair(g_vertex, _n),
                    prot.get(&upair(g_vertex, _n)).copied().unwrap_or(Pauli::I) * Pauli::X,
                );
            }
        }
        // Fire all red spiders with full green edges and thus their green neighbours
        if g_type == NodeType::X && v[adj_vertex + ordering.z_boundaries.len()] == true {
            for _n in d.neighbors(g_vertex) {
                prot.insert(
                    upair(g_vertex, _n),
                    prot.get(&upair(g_vertex, _n)).copied().unwrap_or(Pauli::I) * Pauli::Z,
                );
            }
        }
    }

    // Fire all green output edges
    for (&g_z_boundary, &g_boundary) in ordering.z_boundaries.iter() {
        let adj_z_boundary = ordering.ord(g_z_boundary);
        if v[adj_z_boundary] == true {
            prot.insert(
                upair(g_z_boundary, g_boundary),
                prot.get(&upair(g_z_boundary, g_boundary))
                    .copied()
                    .unwrap_or(Pauli::I)
                    * Pauli::Z,
            );
        }
    }

    prot
}
