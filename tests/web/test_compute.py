import json
from typing import List, Callable, Iterable

import numpy as np
import pytest
from galois import GF2
from pyzx import Graph, VertexType

import generate.diagram.syndrome
import generate.stabilisers
from faulttools import PauliString, Pauli
from faulttools.diagram import Diagram
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
def assert_pauli_webs(web_io: WebFileIO) -> Callable[[Diagram, Iterable[PauliString], Iterable[PauliString]], None]:
    def _assert(d: Diagram, stabs: Iterable[PauliString], regions: Iterable[PauliString]) -> None:
        edge_idx_map = {e: i for i, e in enumerate(d.edge_indices())}

        compiled_stabs = GF2([web.compile(edge_idx_map) for web in stabs])
        compiled_regions = GF2([web.compile(edge_idx_map) for web in regions])
        compiled_exp_stabs = GF2([web.compile(edge_idx_map) for web in web_io.read_stabilising()])
        compiled_exp_regions = GF2([web.compile(edge_idx_map) for web in web_io.read_detecting()])

        try:
            assert len(compiled_regions) == len(compiled_exp_regions)
            if len(compiled_regions) > 0:
                assert np.array_equal(compiled_regions.row_reduce(), compiled_exp_regions.row_reduce()), (
                    "Region spaces are not equal"
                )

            # Stabilising web spaces must only be equal modulo the detecting web spaces. Thus, test the entire Pauli web
            # space for equality, which yields the property under test combined with detecting web space equality.
            if len(compiled_regions) > 0:
                web_space_basis = GF2(np.vstack([compiled_stabs, compiled_regions]))
                exp_web_space_basis = GF2(np.vstack([compiled_exp_stabs, compiled_exp_regions]))
            else:
                web_space_basis = compiled_stabs
                exp_web_space_basis = compiled_exp_stabs

            assert np.array_equal(web_space_basis.row_reduce(), exp_web_space_basis.row_reduce()), (
                "Web spaces are not equal"
            )
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
    assert_pauli_webs(d, stabs, regions)


def test_zweb_webs(assert_pauli_webs):
    d = from_pyzx(generate.diagram.zweb(2, 2))
    stabs, regions = compute_pauli_webs(d)

    assert_pauli_webs(d, stabs, regions)
