from paritea import Pauli


def test_pauli_mult():
    assert Pauli.I * Pauli.I == Pauli.I
    assert Pauli.I * Pauli.X == Pauli.X
    assert Pauli.I * Pauli.Y == Pauli.Y
    assert Pauli.I * Pauli.Z == Pauli.Z
    assert Pauli.X * Pauli.I == Pauli.X
    assert Pauli.X * Pauli.X == Pauli.I
    assert Pauli.X * Pauli.Y == Pauli.Z
    assert Pauli.X * Pauli.Z == Pauli.Y
    assert Pauli.Y * Pauli.I == Pauli.Y
    assert Pauli.Y * Pauli.X == Pauli.Z
    assert Pauli.Y * Pauli.Y == Pauli.I
    assert Pauli.Y * Pauli.Z == Pauli.X
    assert Pauli.Z * Pauli.I == Pauli.Z
    assert Pauli.Z * Pauli.X == Pauli.Y
    assert Pauli.Z * Pauli.Y == Pauli.X
    assert Pauli.Z * Pauli.Z == Pauli.I


def test_pauli_commutes():
    assert Pauli.I.commutes(Pauli.I)
    assert Pauli.I.commutes(Pauli.X) and Pauli.X.commutes(Pauli.I)
    assert Pauli.I.commutes(Pauli.Y) and Pauli.Y.commutes(Pauli.I)
    assert Pauli.I.commutes(Pauli.Z) and Pauli.Z.commutes(Pauli.I)
    assert Pauli.X.commutes(Pauli.X)
    assert not Pauli.X.commutes(Pauli.Y) and not Pauli.Y.commutes(Pauli.X)
    assert not Pauli.X.commutes(Pauli.Z) and not Pauli.Z.commutes(Pauli.X)
    assert Pauli.Y.commutes(Pauli.Y)
    assert not Pauli.Y.commutes(Pauli.Z) and not Pauli.Z.commutes(Pauli.Y)
    assert Pauli.Z.commutes(Pauli.Z)
