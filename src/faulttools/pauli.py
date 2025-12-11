from enum import StrEnum
from typing import Union, Optional, Iterable, Mapping

import frozendict as fd
from galois import GF2


class Pauli(StrEnum):
    """Represents one of the four Pauli matrices up to a scalar factor."""

    I = "I"
    X = "X"
    Z = "Z"
    Y = "Y"

    def __mul__(self, other: "Pauli") -> "Pauli":
        if self == Pauli.I:
            return other
        elif other == Pauli.I:
            return self
        elif self == other:
            return Pauli.I
        elif self != Pauli.X and other != Pauli.X:
            return Pauli.X
        elif self != Pauli.Y and other != Pauli.Y:
            return Pauli.Y
        elif self != Pauli.Z and other != Pauli.Z:
            return Pauli.Z
        else:
            raise AssertionError("Should never be reached!")

    def __repr__(self):
        return f"Pauli{self.name}"

    def flip(self) -> "Pauli":
        if self == Pauli.X:
            return Pauli.Z
        elif self == Pauli.Z:
            return Pauli.X
        else:
            return self

    def commutes(self, other: "Pauli") -> bool:
        return self == Pauli.I or other == Pauli.I or self == other


class PauliString(fd.frozendict[int, Pauli]):
    """
    A Pauli string representation as a mapping from edge indices to Pauli rotations.
    Identity rotations are not stored in the resulting string.
    """

    @staticmethod
    def unary(edge: int, pauli: Pauli) -> "PauliString":
        return PauliString({edge: pauli})

    def __new__(cls, o: Optional[Union[dict, str]] = None):
        if o is None:
            return super().__new__(cls)
        elif isinstance(o, str):
            return super().__new__(cls, {i: Pauli(c) for i, c in enumerate(o) if c != Pauli.I})
        elif isinstance(o, dict):
            return super().__new__(cls, {e: p for e, p in o.items() if p != Pauli.I})
        else:
            raise ValueError("Unknown source type for PauliString.")

    def __mul__(self, other: "PauliString") -> "PauliString":
        product = {e: self.get(e, other.get(e)) for e in self.keys() ^ other.keys()}
        for k in self.keys() & other.keys():
            result = self[k] * other[k]
            if result != Pauli.I:
                product[k] = result

        return PauliString(product)

    def restrict(self, indices: Iterable[int]) -> "PauliString":
        return PauliString({idx: self[idx] for idx in set(indices).intersection(self.keys())})

    def commutes(self, other: "PauliString") -> bool:
        for k in self.keys() & other.keys():
            if not self[k].commutes(other[k]):
                return False

        return True

    def is_trivial(self) -> bool:
        for p in self:
            if self[p] != Pauli.I:
                return False
        return True

    def compile(self, idx_map: Mapping[int, int]) -> GF2:
        num_indices = len(idx_map)
        compiled = GF2.Zeros(num_indices * 2)
        for idx, pauli in self.items():
            if pauli == Pauli.Z or pauli == Pauli.Y:
                compiled[idx_map[idx]] = 1
            if pauli == Pauli.X or pauli == Pauli.Y:
                compiled[idx_map[idx] + num_indices] = 1

        return compiled
