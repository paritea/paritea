/// A recreation of the diagram class native to paritea.
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
pub struct Diagram {
    graph: StableUnGraph<NodeData, ()>,
    io: (Vec<NodeIndex>, Vec<NodeIndex>),
    is_io_virtual: bool,
}

impl Diagram {
    pub fn is_io_virtual(&self) -> bool {
        self.is_io_virtual
    }

    /// Sets the boundary node indices regarded as inputs / outputs. Their order directly determines
    /// their index through isomorphic conversion to a states outputs, i.e. they are indexed as
    /// <...all-inputs><...all-outputs>.
    pub fn set_virtual_io(&mut self, inputs: Vec<NodeIndex>, outputs: Vec<NodeIndex>) {
        if !self.boundary_nodes().collect_vec().is_empty() {
            panic!("Graph may not contain any boundaries when setting virtual IO!");
        }

        self.io = (inputs, outputs);
        self.is_io_virtual = true;
    }

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

    pub fn add_node(&mut self, node_type: NodeType, phase: Option<Phase>) -> NodeIndex {
        let nd = if let Some(phase) = phase {
            NodeData::new(node_type, phase)
        } else {
            NodeData::from_type(node_type)
        };
        self.add_node_inner(nd)
    }

    pub fn node_type(&self, node: NodeIndex) -> NodeType {
        self.graph[node].0
    }

    pub fn phase(&self, node: NodeIndex) -> Phase {
        self.graph[node].1
    }

    pub fn add_to_phase(&mut self, node: NodeIndex, phase: Phase) {
        self.graph[node].1 += phase;
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
            pub fn node_count(&self) -> usize;
            pub fn node_indices(&self) -> impl Iterator<Item=NodeIndex>;
            #[call(add_node)]
            fn add_node_inner(&mut self, weight: NodeData) -> NodeIndex;
            pub fn remove_node(&mut self, n: NodeIndex);
            pub fn neighbors(&self, a: NodeIndex) -> impl Iterator<Item=NodeIndex>;
            pub fn edge_count(&self) -> usize;
            pub fn edges(&self, a: NodeIndex) -> Edges<'_, (), Undirected>;
            pub fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> EdgesConnecting<'_, (), Undirected>;
            pub fn edge_indices(&self) -> impl Iterator<Item=EdgeIndex>;
            pub fn edge_endpoints(&self, e: EdgeIndex) -> Option<(NodeIndex, NodeIndex)>;
            pub fn remove_edge(&mut self, e: EdgeIndex);
            pub fn has_parallel_edges(&self) -> bool;
        }
    }
}
