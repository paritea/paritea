use rustworkx_core::petgraph::graph::NodeIndex;

pub fn upair(a: NodeIndex, b: NodeIndex) -> (NodeIndex, NodeIndex) {
    if a < b { (a, b) } else { (b, a) }
}
