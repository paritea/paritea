//! Rust implementations for algorithms used in the paritea Python library.
use rustworkx_core::petgraph::graph::NodeIndex;

pub mod diagram;
pub mod pauli;
pub mod web;

fn sorted_pair(a: NodeIndex, b: NodeIndex) -> (NodeIndex, NodeIndex) {
    if a < b { (a, b) } else { (b, a) }
}
