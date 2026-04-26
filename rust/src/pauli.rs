use rustworkx_core::petgraph::graph::EdgeIndex;
use std::collections::HashMap;
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

pub type PauliString = HashMap<EdgeIndex, Pauli>;
