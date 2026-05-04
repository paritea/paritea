use bitgauss::BitMatrix;
use derive_more::From;
use derive_more::with_trait::Index;
use rustc_hash::FxHashMap;
use rustworkx_core::petgraph::graph::EdgeIndex;
use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::ops::Mul;
use std::ops::MulAssign;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    pub fn flip(&self) -> Pauli {
        match self {
            Pauli::I => Pauli::I,
            Pauli::X => Pauli::Z,
            Pauli::Y => Pauli::Y,
            Pauli::Z => Pauli::X,
        }
    }
}

impl Mul for Pauli {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Pauli::I, p) | (p, Pauli::I) => p,
            (Pauli::X, Pauli::X) | (Pauli::Y, Pauli::Y) | (Pauli::Z, Pauli::Z) => Pauli::I,
            (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::Y) => Pauli::X,
            (Pauli::X, Pauli::Z) | (Pauli::Z, Pauli::X) => Pauli::Y,
            (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::X) => Pauli::Z,
        }
    }
}

impl MulAssign for Pauli {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other
    }
}

#[derive(Default, Debug, Clone, From, Index)]
pub struct PauliString(#[from] pub FxHashMap<EdgeIndex, Pauli>);

impl PauliString {
    pub fn is_trivial(&self) -> bool {
        self.0.values().all(|&p| p == Pauli::I)
    }

    pub fn restrict(&self, edges: &BTreeSet<EdgeIndex>) -> Self {
        Self(
            edges
                .iter()
                .copied()
                .filter_map(|e| self.0.get(&e).map(|p| (e, *p)))
                .collect(),
        )
    }

    pub fn compile(&self, idx_map: &FxHashMap<EdgeIndex, usize>) -> BitMatrix {
        let num_indices = idx_map.len();
        let mut matrix = BitMatrix::zeros(num_indices * 2, 1);
        for (e, &p) in &self.0 {
            if p == Pauli::Z || p == Pauli::Y {
                matrix.set_bit(idx_map[e], 0, true);
            }
            if p == Pauli::X || p == Pauli::Y {
                matrix.set_bit(idx_map[e] + num_indices, 0, true);
            }
        }

        matrix
    }
}

impl<S> Mul<S> for &PauliString
where
    S: Borrow<PauliString>,
{
    type Output = PauliString;

    fn mul(self, rhs: S) -> Self::Output {
        let rhs = rhs.borrow();
        let my_keys = self.0.keys().copied().collect::<BTreeSet<_>>();
        let rhs_keys = rhs.0.keys().copied().collect::<BTreeSet<_>>();
        let mut product = FxHashMap::from_iter(
            my_keys
                .symmetric_difference(&rhs_keys)
                .map(|e| (*e, *self.0.get(e).or(rhs.0.get(e)).unwrap())),
        );
        for k in my_keys.intersection(&rhs_keys) {
            let result = self[k] * rhs[k];
            if result != Pauli::I {
                product.insert(*k, result);
            }
        }

        PauliString(product)
    }
}

impl<S> Mul<S> for PauliString
where
    S: Borrow<Self>,
{
    type Output = Self;

    fn mul(self, rhs: S) -> Self::Output {
        &self * rhs
    }
}
