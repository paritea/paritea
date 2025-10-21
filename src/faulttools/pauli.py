from enum import StrEnum
from typing import Union, Optional

import frozendict as fd


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

    def commutes(self, other: "Pauli") -> bool:
        return self == Pauli.I or other == Pauli.I or self == other


class PauliString(fd.frozendict[int, Pauli]):
    """
    A Pauli string representation as a mapping from edge indices to Pauli rotations.
    Note that identity rotations may be elided by the implementation at will.
    """

    @staticmethod
    def edge_flip(edge: int, pauli: Pauli) -> "PauliString":
        return PauliString({edge: pauli})

    def __new__(cls, o: Optional[Union[dict, str]] = None):
        if o is None:
            return super().__new__(cls)
        elif isinstance(o, str):
            return super().__new__(cls, {i: Pauli(c) for i, c in enumerate(o) if c != Pauli.I})
        elif isinstance(o, dict):
            return super().__new__(cls, {e: p for e, p in o.items() if p != Pauli.I})
        else:
            return super().__new__(cls, o)

    def __mul__(self, other: "PauliString") -> "PauliString":
        product = {e: self.get(e, other.get(e)) for e in self.keys() ^ other.keys()}
        for e in self.keys() & other.keys():
            edge_result = self[e] * other[e]
            if edge_result != Pauli.I:
                product[e] = edge_result

        return PauliString(product)

    def commutes(self, other: "PauliString") -> bool:
        for e in self.keys() & other.keys():
            if not self[e].commutes(other[e]):
                return False

        return True

    def is_trivial(self) -> bool:
        for p in self:
            if self[p] != Pauli.I:
                return False
        return True
