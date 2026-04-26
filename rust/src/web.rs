pub mod compute;
mod firing_assignments;
mod red_green;

pub use compute::{compute, compute_detecting_regions, compute_pauli_webs, compute_stabilisers};
