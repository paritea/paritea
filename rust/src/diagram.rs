use delegate::delegate;
use fraction::Fraction;
use rustworkx_core::graph_ext::HasParallelEdgesUndirected;
use rustworkx_core::petgraph::Undirected;
use rustworkx_core::petgraph::graph::{EdgeIndex, NodeIndex};
use rustworkx_core::petgraph::prelude::StableUnGraph;
use rustworkx_core::petgraph::stable_graph::EdgesConnecting;

pub type Phase = Fraction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    B,
    X,
    Z,
    H,
}

#[derive(Debug, Clone)]
pub struct NodeData(NodeType, Phase);

impl NodeData {
    pub fn new(nt: NodeType, phase: Phase) -> Self {
        Self(nt, phase)
    }

    pub fn from_type(nt: NodeType) -> Self {
        Self(nt, Phase::new(0u32, 1u32))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Diagram {
    graph: StableUnGraph<NodeData, ()>,
    is_io_virtual: bool,
}

impl Diagram {
    pub fn is_io_virtual(&self) -> bool {
        self.is_io_virtual
    }

    pub fn node_type(&self, node: NodeIndex) -> NodeType {
        self.graph[node].0
    }

    pub fn phase(&self, node: NodeIndex) -> Phase {
        self.graph[node].1
    }

    pub fn add_to_phase(&mut self, node: NodeIndex, phase: Phase) {
        self.graph[node].1 = self.graph[node].1 + phase;
    }

    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex) -> EdgeIndex {
        self.graph.add_edge(a, b, ())
    }

    pub fn add_edges(&mut self, points: impl IntoIterator<Item = (NodeIndex, NodeIndex)>) {
        for (a, b) in points {
            self.graph.add_edge(a, b, ());
        }
    }

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

    pub fn edge_list(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex)> {
        self.graph
            .edge_indices()
            .map(|e| self.graph.edge_endpoints(e).unwrap())
    }

    pub fn boundary_nodes(&self) -> impl Iterator<Item = NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&e| self.graph[e].0 == NodeType::B)
    }

    delegate! {
        to self.graph {
            pub fn node_count(&self) -> usize;
            pub fn node_indices(&self) -> impl Iterator<Item=NodeIndex>;
            pub fn add_node(&mut self, weight: NodeData) -> NodeIndex;
            pub fn remove_node(&mut self, n: NodeIndex);
            pub fn neighbors(&self, a: NodeIndex) -> impl Iterator<Item=NodeIndex>;
            pub fn edge_count(&self) -> usize;
            pub fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> EdgesConnecting<(), Undirected>;
            pub fn remove_edge(&mut self, e: EdgeIndex);
            pub fn has_parallel_edges(&self) -> bool;
        }
    }
}
