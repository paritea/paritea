import json
from typing import List, Callable, Iterable

import pytest
from pyzx import Graph, VertexType

import generate
from faulttools import PauliString, Pauli
from faulttools.diagram.conversion import from_pyzx
from faulttools.web import compute_pauli_webs


class WebFileIO:
    filename_template: str

    def write_stabilising(self, webs: List[PauliString], file_name_suffix: str = "") -> None:
        self._write(webs, ext="stabilising", file_name_suffix=file_name_suffix)

    def write_detecting(self, webs: List[PauliString], file_name_suffix: str = "") -> None:
        self._write(webs, ext="detecting", file_name_suffix=file_name_suffix)

    def _write(self, webs: List[PauliString], ext: str, file_name_suffix: str) -> None:
        with open(f"{self.filename_template}{file_name_suffix}.{ext}", "w") as f:
            json.dump(webs, f)

    def read_stabilising(self) -> List[PauliString]:
        return self._read(ext="stabilising")

    def read_detecting(self) -> List[PauliString]:
        return self._read(ext="detecting")

    def _read(self, ext: str) -> List[PauliString]:
        with open(f"{self.filename_template}.{ext}", "r") as f:
            return [PauliString({int(e): Pauli(p) for e, p in ps.items()}) for ps in json.load(f)]


@pytest.fixture
def web_io(request) -> WebFileIO:
    io = WebFileIO()
    # Configure test name as file name
    io.filename_template = f"tests/web/{request.node.name.removeprefix('test_')}"

    return io


@pytest.fixture
def assert_pauli_webs(web_io: WebFileIO) -> Callable[[Iterable[PauliString], Iterable[PauliString]], None]:
    def _assert(stabs: Iterable[PauliString], regions: Iterable[PauliString]) -> None:
        exp_stabs = web_io.read_stabilising()
        exp_regions = web_io.read_detecting()

        try:
            assert set(stabs) == set(exp_stabs)
            assert set(exp_regions) == set(exp_regions)
        except AssertionError as e:
            web_io.write_stabilising(list(stabs), file_name_suffix="_actual")
            web_io.write_detecting(list(regions), file_name_suffix="_actual")

            raise e

    return _assert


def test_identity_webs(assert_pauli_webs):
    g = Graph()
    b1 = g.add_vertex(VertexType.BOUNDARY)
    b2 = g.add_vertex(VertexType.BOUNDARY)
    n = g.add_vertex(VertexType.Z)
    g.add_edges([(b1, n), (n, b2)])
    d = from_pyzx(g)

    stabs, regions = compute_pauli_webs(d)
    assert_pauli_webs(stabs, regions)


def test_zweb_webs(assert_pauli_webs):
    d = from_pyzx(generate.zweb(2, 2))
    stabs, regions = compute_pauli_webs(d)

    assert_pauli_webs(stabs, regions)
