//! A recreation of the diagram class and constructs native to paritea.
use delegate::delegate;
use derive_more::Display;
use fraction::Fraction;
use itertools::Itertools;
use rustworkx_core::graph_ext::HasParallelEdgesUndirected;
use rustworkx_core::petgraph::Undirected;
use rustworkx_core::petgraph::graph::{EdgeIndex, NodeIndex};
use rustworkx_core::petgraph::prelude::StableUnGraph;
use rustworkx_core::petgraph::stable_graph::{Edges, EdgesConnecting};
use std::collections::HashMap;

/// A ZX spiders phase, represented as a fraction of pi.
pub type Phase = Fraction;

/// The type of a single spider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum NodeType {
    /// A boundary node
    B,
    /// An X spider
    X,
    /// A Z spider
    Z,
    /// An H-Box
    H,
}

#[derive(Debug, Clone)]
struct NodeData(NodeType, Phase);

impl NodeData {
    pub fn new(nt: NodeType, phase: Phase) -> Self {
        Self(nt, phase)
    }

    pub fn from_type(nt: NodeType) -> Self {
        Self(nt, Phase::new(0u32, 1u32))
    }
}

#[derive(Debug, Clone, Default)]
/// A reconstruction of a ZX diagram in the paritea internal representation.
pub struct Diagram {
    graph: StableUnGraph<NodeData, ()>,
    io: (Vec<NodeIndex>, Vec<NodeIndex>),
    is_io_virtual: bool,
}

impl Diagram {
    /// Whether the IO indices of the diagram refer to boundary nodes (real) or to internal nodes
    /// that are connected to the given output (virtual).
    pub fn is_io_virtual(&self) -> bool {
        self.is_io_virtual
    }

    /// Sets the node indices regarded as inputs / outputs. Their order directly determines their
    /// index through isomorphic conversion to a states outputs, i.e. they are indexed as
    /// <...all-inputs><...all-outputs>.
    pub fn set_virtual_io(&mut self, inputs: Vec<NodeIndex>, outputs: Vec<NodeIndex>) {
        if !self.boundary_nodes().collect_vec().is_empty() {
            panic!("Graph may not contain any boundaries when setting virtual IO!");
        }

        self.io = (inputs, outputs);
        self.is_io_virtual = true;
    }

    /// Converts a diagrams virtual IO to real IO by creating boundary nodes and connecting them
    /// to the virtual IO nodes.
    pub fn realize_io(&mut self) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
        if !self.is_io_virtual {
            return self.io.clone();
        };

        let (inputs, outputs) = self.io.clone();

        let mut new_inputs = Vec::new();
        for &inp in inputs.iter() {
            let new_inp = self.add_node(NodeType::B, None);
            new_inputs.push(new_inp);
            self.add_edge(inp, new_inp);
        }
        let mut new_outputs = Vec::new();
        for &out in outputs.iter() {
            let new_out = self.add_node(NodeType::B, None);
            new_outputs.push(new_out);
            self.add_edge(out, new_out);
        }

        self.io = (new_inputs.clone(), new_outputs.clone());
        self.is_io_virtual = false;

        (new_inputs, new_outputs)
    }

    /// Add a node with the specified type and optionally a phase
    pub fn add_node(&mut self, node_type: NodeType, phase: Option<Phase>) -> NodeIndex {
        let nd = if let Some(phase) = phase {
            NodeData::new(node_type, phase)
        } else {
            NodeData::from_type(node_type)
        };
        self.add_node_inner(nd)
    }

    /// Get the node type of a node
    pub fn node_type(&self, node: NodeIndex) -> NodeType {
        self.graph[node].0
    }

    /// Get the phase of a node
    pub fn phase(&self, node: NodeIndex) -> Phase {
        self.graph[node].1
    }

    /// Add a phase to the phase of the node
    pub fn add_to_phase(&mut self, node: NodeIndex, phase: Phase) {
        self.graph[node].1 += phase;
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex) -> EdgeIndex {
        self.graph.add_edge(a, b, ())
    }

    /// Add a range of edges between nodes
    pub fn add_edges(&mut self, points: impl IntoIterator<Item = (NodeIndex, NodeIndex)>) {
        for (a, b) in points {
            self.graph.add_edge(a, b, ());
        }
    }

    /// Remove an edge between two nodes. Panics if the edge is not found.
    pub fn remove_edge_between(&mut self, a: NodeIndex, b: NodeIndex) {
        if let Some(e) = self.graph.find_edge(a, b) {
            self.graph.remove_edge(e);
        } else {
            panic!(
                "Can't remove edge between {:?} and {:?}: Edge not found!",
                a, b
            );
        }
    }

    /// All the edges in the diagram by their endpoints
    pub fn edge_list(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex)> {
        self.graph
            .edge_indices()
            .map(|e| self.graph.edge_endpoints(e).unwrap())
    }

    /// All the boundary nodes of the diagram
    pub fn boundary_nodes(&self) -> impl Iterator<Item = NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&e| self.graph[e].0 == NodeType::B)
    }

    /// Returns the subdiagram consisting of the specified nodes, as well as a map between the nodes
    /// in the current diagram and the corresponding ones in the subdiagram.
    pub fn subgraph(
        &self,
        nodes: impl IntoIterator<Item = NodeIndex>,
    ) -> (Self, HashMap<NodeIndex, NodeIndex>) {
        let mut outgraph = Self::default();
        let node_map: HashMap<NodeIndex, NodeIndex> = nodes
            .into_iter()
            .map(|n| (n, outgraph.add_node_inner(self.graph[n].clone())))
            .collect();
        for (a, b) in self.edge_list() {
            if let (Some(&a_sub), Some(&b_sub)) = (node_map.get(&a), node_map.get(&b)) {
                outgraph.add_edge(a_sub, b_sub);
            }
        }
        (outgraph, node_map)
    }

    delegate! {
        to self.graph {
            /// Count all nodes
            pub fn node_count(&self) -> usize;
            /// All node indices
            pub fn node_indices(&self) -> impl Iterator<Item=NodeIndex>;
            #[call(add_node)]
            fn add_node_inner(&mut self, weight: NodeData) -> NodeIndex;
            /// Remove a node
            pub fn remove_node(&mut self, n: NodeIndex);
            /// All neighbours of a node
            pub fn neighbors(&self, a: NodeIndex) -> impl Iterator<Item=NodeIndex>;
            /// Count all edges
            pub fn edge_count(&self) -> usize;
            /// All edges in the diagram, as an iterator over edge references
            pub fn edges(&self, a: NodeIndex) -> Edges<'_, (), Undirected>;
            /// All edges connecting the two nodes, as an iterator over edge references
            pub fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> EdgesConnecting<'_, (), Undirected>;
            /// All edge indices in the diagram
            pub fn edge_indices(&self) -> impl Iterator<Item=EdgeIndex>;
            /// The source and target of an edge, if present
            pub fn edge_endpoints(&self, e: EdgeIndex) -> Option<(NodeIndex, NodeIndex)>;
            /// Remove an edge with the given index
            pub fn remove_edge(&mut self, e: EdgeIndex);
            /// Whether there are any two nodes that have more and one edge between them
            pub fn has_parallel_edges(&self) -> bool;
        }
    }
}
