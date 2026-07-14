//! Supporting Rust library for the Python bindings.

use pyo3::pymodule;
/// Python module containing the Rust bindings.
///
/// The definitions here should be reflected in the
/// `src/paritea/_bindings/__init__.pyi` type stubs.
#[pymodule]
mod _bindings {
    use num_bigint::BigUint;
    use paritea::diagram::{Diagram, NodeType, Phase};
    use paritea::fault_equivalence::Violation;
    use paritea::fault_equivalence::check_fault_equivalence;
    use paritea::pauli::PauliString;
    use paritea::web::{compute, pauli_webs_through_partitions};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rustworkx_core::petgraph::graph::{EdgeIndex, NodeIndex};
    use std::collections::{BTreeSet, HashMap};

    fn _to_rs_diagram(d: Bound<'_, PyAny>) -> (Diagram, HashMap<EdgeIndex, usize>) {
        let mut nd = Diagram::default();

        if d.call_method0("is_io_virtual")
            .unwrap()
            .extract::<bool>()
            .unwrap()
        {
            panic!("Only IO-realized diagrams are supported");
        }

        let nodes = d
            .call_method0("node_indices")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap();
        let mut py_to_rs_nodes: HashMap<usize, NodeIndex> = HashMap::new();
        for node in nodes {
            let node_type_py = d.call_method1("type", (node,)).unwrap();
            let node_type = match node_type_py.extract::<&str>().unwrap() {
                "Z" => NodeType::Z,
                "X" => NodeType::X,
                "B" => NodeType::B,
                "H" => NodeType::H,
                other => panic!("Unknown node type: {other}"),
            };
            let phase_py = d.call_method1("phase", (node,)).unwrap();
            let phase = Phase::new(
                phase_py
                    .getattr("numerator")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                phase_py
                    .getattr("denominator")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
            );
            py_to_rs_nodes.insert(node, nd.add_node(node_type, Some(phase)));
        }

        let mut rs_to_py_edges = HashMap::new();
        for edge_idx in d
            .call_method0("edge_indices")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap()
        {
            let (source, target) = d
                .call_method1("get_edge_endpoints_by_index", (edge_idx,))
                .unwrap()
                .extract::<(usize, usize)>()
                .unwrap();
            let rs_edge = nd.add_edge(py_to_rs_nodes[&source], py_to_rs_nodes[&target]);
            rs_to_py_edges.insert(rs_edge, edge_idx);
        }

        (nd, rs_to_py_edges)
    }

    fn _to_py_string<'py>(
        py: Python<'py>,
        ps: PauliString,
        rs_to_py_edges: &HashMap<EdgeIndex, usize>,
    ) -> Bound<'py, PyAny> {
        let m = py.import("paritea.pauli").unwrap();
        let string_class = m.getattr("PauliString").unwrap();
        let pauli_class = m.getattr("Pauli").unwrap();
        let dict = PyDict::new(py);
        for (e, p) in ps.0 {
            let p_py_str = match p {
                paritea::pauli::Pauli::I => "I",
                paritea::pauli::Pauli::X => "X",
                paritea::pauli::Pauli::Y => "Y",
                paritea::pauli::Pauli::Z => "Z",
            };

            dict.set_item(rs_to_py_edges[&e], pauli_class.call1((p_py_str,)).unwrap())
                .unwrap();
        }
        string_class.call1((dict,)).unwrap()
    }

    type PyWeb<'py> = Vec<Bound<'py, PyAny>>;

    #[pyfunction]
    #[pyo3(signature = (diagram, *, stabilisers, detecting_regions))]
    fn _compute_pauli_webs<'py>(
        diagram: Bound<'py, PyAny>,
        stabilisers: bool,
        detecting_regions: bool,
    ) -> PyResult<(Option<PyWeb<'py>>, Option<PyWeb<'py>>)> {
        let py = diagram.py();
        let (nd, rs_to_py_edges) = _to_rs_diagram(diagram);
        let (stabs_opt, regions_opt) = compute(&nd, stabilisers, detecting_regions);

        Ok((
            stabs_opt.map(|stabs| {
                stabs
                    .into_iter()
                    .map(|s| _to_py_string(py, s, &rs_to_py_edges))
                    .collect()
            }),
            regions_opt.map(|regions| {
                regions
                    .into_iter()
                    .map(|r| _to_py_string(py, r, &rs_to_py_edges))
                    .collect()
            }),
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (diagram, *, partitions))]
    fn _compute_pauli_webs_through_partitions<'py>(
        diagram: Bound<'py, PyAny>,
        partitions: Vec<Vec<usize>>,
    ) -> PyResult<(PyWeb<'py>, PyWeb<'py>)> {
        let py = diagram.py();
        let (nd, rs_to_py_edges) = _to_rs_diagram(diagram);
        let (stabs, regions) = pauli_webs_through_partitions(
            &nd,
            partitions
                .into_iter()
                .map(|nodes| BTreeSet::from_iter(nodes.into_iter().map(NodeIndex::new)))
                .collect(),
        );

        Ok((
            stabs
                .into_iter()
                .map(|s| _to_py_string(py, s, &rs_to_py_edges))
                .collect(),
            regions
                .into_iter()
                .map(|r| _to_py_string(py, r, &rs_to_py_edges))
                .collect(),
        ))
    }

    #[pyclass]
    #[allow(dead_code)]
    /// A violation of fault equivalence
    struct _Violation {
        #[pyo3(get)]
        /// Whether the violation originates from the first given noise model or the second
        is_nm1: bool,
        #[pyo3(get)]
        /// The combined weight of the violation
        weight: usize,
        #[pyo3(get)]
        /// The indices of the faults that compose to the violating fault
        faults: Option<Vec<usize>>,
    }

    impl From<Violation> for _Violation {
        fn from(violation: Violation) -> Self {
            Self {
                is_nm1: violation.is_nm1,
                weight: violation.weight,
                faults: violation.faults,
            }
        }
    }

    #[pyfunction]
    #[pyo3(signature = (*, nm1_sigs, nm2_sigs, d1_detectors, d2_detectors, until, provenance, quiet))]
    fn _check_fault_equivalence(
        nm1_sigs: Vec<(BigUint, usize)>,
        nm2_sigs: Vec<(BigUint, usize)>,
        d1_detectors: usize,
        d2_detectors: usize,
        until: Option<usize>,
        provenance: bool,
        quiet: bool,
    ) -> Option<_Violation> {
        check_fault_equivalence(
            nm1_sigs,
            nm2_sigs,
            d1_detectors,
            d2_detectors,
            until,
            provenance,
            quiet,
        )
        .map(|v| v.into())
    }
}
