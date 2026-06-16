from paritea import Pauli, PauliString


def test_constructor():
    assert len(PauliString()) == 0
    assert PauliString("XYZ") == {0: Pauli.X, 1: Pauli.Y, 2: Pauli.Z}
    assert PauliString({0: "X", 1: "Y", 2: "Z"}) == {0: Pauli.X, 1: Pauli.Y, 2: Pauli.Z}


def test_edge_flip_factory():
    assert PauliString.unary(4, Pauli.I) == {}
    assert PauliString.unary(4, Pauli.Z) == {4: Pauli.Z}
    assert PauliString.unary(4, Pauli.Z) == {4: Pauli.Z}
    assert PauliString.unary(4, Pauli.Z) == {4: Pauli.Z}


def test_mult():
    assert PauliString() * PauliString() == PauliString()
    assert PauliString("IIIIXXXXYYYYZZZZ") * PauliString("IXYZIXYZIXYZIXYZ") == PauliString("IXYZXIZYYZIXZYXI")


def test_elided_identity():
    assert PauliString("IXIIZYIX") == {1: Pauli.X, 4: Pauli.Z, 5: Pauli.Y, 7: Pauli.X}
    assert PauliString("IXIIZIX") * PauliString("XXZIYI") == {0: Pauli.X, 2: Pauli.Z, 4: Pauli.X, 6: Pauli.X}


def test_commutes():
    assert PauliString().commutes(PauliString())
    assert PauliString("IIII").commutes(PauliString("IXZY"))
    assert PauliString("IZZI").commutes(PauliString("XIZY"))
    assert not PauliString("IZZI").commutes(PauliString("IXII"))
    assert not PauliString("IZZI").commutes(PauliString("IZYI"))
    assert PauliString("IZZI").commutes(PauliString("IYXX"))
    assert not PauliString("YZZI").commutes(PauliString("XXXX"))
