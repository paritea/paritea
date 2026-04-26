use crate::diagram::Diagram;
use crate::firing_assignments::{
    convert_firing_assignment_to_web_prototype, create_firing_verification, determine_ordering,
};
use crate::pauli::{Pauli, PauliString};
use crate::red_green::to_red_green_form;
use bitgauss::{BitMatrix, BitVec};
use itertools::Itertools;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::prelude::EdgeRef;
use std::collections::HashMap;
use std::ops::Mul;

/// Performs full stabiliser and detecting region computation, depending on the given flags.
/// Enabling both flags in one call is preferred to enabling them in separate calls as they may
/// share basic computations.
fn compute(
    diagram: &Diagram,
    stabilisers: bool,
    detecting_regions: bool,
) -> (Option<Vec<PauliString>>, Option<Vec<PauliString>>) {
    if diagram.is_io_virtual() {
        panic!("This function does not accept diagrams with virtual IO!"); // TODO result
    }

    let to_pauli_string = |prototype: HashMap<(NodeIndex, NodeIndex), Pauli>| -> PauliString {
        prototype
            .into_iter()
            .map(|(edge, p)| {
                (
                    diagram
                        .edges_connecting(edge.0, edge.1)
                        .next()
                        .unwrap()
                        .id(),
                    p,
                )
            })
            .collect()
    };

    let mut d = diagram.clone();
    let additional_nodes = to_red_green_form(&mut d);
    let ordering = determine_ordering(&d);
    let m_d = create_firing_verification(&d, &ordering);

    // Compute row span of valid firing assignment space
    let sol_row_basis = BitMatrix::vstack_from_iter(&m_d.nullspace());

    let b: Vec<bool> = BitVec::new().into();

    BitMatrix::from_bool_vec(&vec![b]);

    let stabs = if !stabilisers {
        None
    } else {
        // Search for solutions that highlight boundary edges, i.e. stabilisers
        let transp = sol_row_basis.transposed();
        let mut boundary_selected_basis = BitMatrix::from_bool_vec(
            &(0..(ordering.z_boundaries.len() * 2))
                .map(|i| transp.row(i).to_vec().into())
                .collect_vec(),
        );
        boundary_selected_basis.gauss(true);

        let mut pivot_cols = Vec::new();
        for i in 0..boundary_selected_basis.rows() {
            let row = boundary_selected_basis.row(i);
            if let Some(nonzero_index) = row.iter().position(|x| x) {
                pivot_cols.push(nonzero_index);
            }
        }
        let mut web_prototypes = pivot_cols
            .into_iter()
            .map(|col| sol_row_basis.row(col).to_vec())
            .map(|v| convert_firing_assignment_to_web_prototype(&d, &ordering, v))
            .collect::<Vec<_>>();
        for web_prototype in web_prototypes.iter_mut() {
            additional_nodes.remove_from(&d, web_prototype);
        }
        Some(web_prototypes.into_iter().map(to_pauli_string).collect())
    };

    let regions = if !detecting_regions {
        None
    } else {
        // Search for solutions that do not highlight boundary edges, i.e. detecting regions
        let transp = sol_row_basis.transposed();
        let mut boundary_selected_basis = BitMatrix::from_bool_vec(
            &(0..(ordering.z_boundaries.len() * 2))
                .map(|i| transp.row(i).to_vec().into())
                .collect_vec(),
        );
        let boundary_nullspace_vectors =
            BitMatrix::vstack_from_iter(&boundary_selected_basis.nullspace());
        // Empty nullspace of boundary edges -> no webs that highlight no boundary edges -> no detecting regions
        let region_sols = if boundary_nullspace_vectors.rows() == 0 {
            Vec::new()
        } else {
            let r = boundary_selected_basis.mul(&sol_row_basis);
            (0..r.rows()).map(|i| r.row(i).to_vec()).collect::<Vec<_>>()
        };

        let mut web_prototypes = region_sols
            .into_iter()
            .map(|v| convert_firing_assignment_to_web_prototype(&d, &ordering, v))
            .collect::<Vec<_>>();
        for web_prototype in &mut web_prototypes {
            additional_nodes.remove_from(&d, web_prototype);
        }
        Some(web_prototypes.into_iter().map(to_pauli_string).collect())
    };

    (stabs, regions)
}

/// A set of stabilising webs for the given diagram that forms a basis for the diagrams stabilisers
/// when restricted to its boundary. A full basis for all stabilising webs is only obtained
/// combining the return value with a basis for the diagrams detecting regions.
pub fn compute_stabilisers(diagram: Diagram) -> Vec<PauliString> {
    let (Some(stabilisers), None) = compute(&diagram, true, false) else {
        unreachable!("Should have passed the right flags!");
    };
    stabilisers
}

/// A basis for the detecting regions of the given diagram.
pub fn compute_detecting_regions(diagram: Diagram) -> Vec<PauliString> {
    let (None, Some(regions)) = compute(&diagram, false, true) else {
        unreachable!("Should have passed the right flags!");
    };
    regions
}

/// See [`compute_stabilisers`] and [`compute_detecting_regions`].
pub fn compute_pauli_webs(diagram: Diagram) -> (Vec<PauliString>, Vec<PauliString>) {
    let (Some(stabilisers), Some(regions)) = compute(&diagram, true, true) else {
        unreachable!("Should have passed the right flags!");
    };
    (stabilisers, regions)
}
