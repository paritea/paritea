//! Methods for computing Pauli webs.

mod compute;
mod firing_assignments;
mod partition;
mod red_green;

pub use compute::compute;
pub use partition::pauli_webs_through_partitions;
