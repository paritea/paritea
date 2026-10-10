//! Rust implementations for algorithms used in the paritea Python library.
use rustworkx_core::petgraph::graph::NodeIndex;

#[cfg(feature = "py")]
mod bindings;
pub mod diagram;
pub mod fault_equivalence;
pub mod pauli;
pub mod web;

fn sorted_pair(a: NodeIndex, b: NodeIndex) -> (NodeIndex, NodeIndex) {
    if a < b { (a, b) } else { (b, a) }
}
