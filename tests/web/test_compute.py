import json
from collections.abc import Callable, Mapping

import numpy as np
import pytest
from galois import GF2

from paritea import Pauli, PauliString, generate
from paritea.diagram import Diagram, NodeType
from paritea.glue.pyzx import from_pyzx
from paritea.web import compute_pauli_webs
from paritea.web.partitions import pauli_webs_through_partitions

SerializedPauliString = Mapping[str, Pauli]


def _serialize(webs: list[PauliString], d: Diagram) -> list[SerializedPauliString]:
    return [{d.get_edge_endpoints_by_index(idx): p for idx, p in web.items()} for web in webs]


def _deserialize(webs: list[SerializedPauliString], d: Diagram) -> list[PauliString]:
    return [PauliString({d.edge_indices_from_endpoints(*nodes)[0]: p for nodes, p in web.items()}) for web in webs]


class WebFileIO:
    filename_template: str

    def write_stabilising(self, webs: list[PauliString], d: Diagram, file_name_suffix: str = "") -> None:
        self._write(_serialize(webs, d), ext="stabilising", file_name_suffix=file_name_suffix)

    def write_detecting(self, webs: list[PauliString], d: Diagram, file_name_suffix: str = "") -> None:
        self._write(_serialize(webs, d), ext="detecting", file_name_suffix=file_name_suffix)

    def _write(self, webs: list[SerializedPauliString], ext: str, file_name_suffix: str) -> None:
        with open(f"{self.filename_template}{file_name_suffix}.{ext}", "w") as f:
            json.dump(
                [{f"({e[0]},{e[1]})": p for e, p in web.items()} for web in webs],
                f,
            )

    def read_stabilising(self, d: Diagram) -> list[PauliString]:
        return _deserialize(self._read(ext="stabilising"), d)

    def read_detecting(self, d: Diagram) -> list[PauliString]:
        return _deserialize(self._read(ext="detecting"), d)

    def _read(self, ext: str) -> list[SerializedPauliString]:
        with open(f"{self.filename_template}.{ext}") as f:
            return [
                PauliString({tuple(map(int, str(e)[1:-1].split(","))): Pauli(p) for e, p in ps.items()})
                for ps in json.load(f)
            ]


@pytest.fixture
def web_io(request) -> WebFileIO:
    io = WebFileIO()
    # Configure test name as file name
    io.filename_template = f"tests/web/{request.node.name.removeprefix('test_')}"

    return io


@pytest.fixture
def generate_web_files(web_io: WebFileIO) -> Callable[[Diagram, list[PauliString], list[PauliString]], None]:
    def _generate(d: Diagram, stabs: list[PauliString], regions: list[PauliString]) -> None:
        web_io.write_stabilising(stabs, d)
        web_io.write_detecting(regions, d)

    return _generate


@pytest.fixture
def assert_pauli_webs(web_io: WebFileIO) -> Callable[[Diagram, list[PauliString], list[PauliString]], None]:
    def _assert(d: Diagram, stabs: list[PauliString], regions: list[PauliString]) -> None:
        edge_idx_map = {e: i for i, e in enumerate(d.edge_indices())}

        compiled_stabs = GF2([web.compile(edge_idx_map) for web in stabs])
        compiled_regions = GF2([web.compile(edge_idx_map) for web in regions])
        compiled_exp_stabs = GF2([web.compile(edge_idx_map) for web in web_io.read_stabilising(d)])
        compiled_exp_regions = GF2([web.compile(edge_idx_map) for web in web_io.read_detecting(d)])

        try:
            assert len(compiled_regions) == len(compiled_exp_regions)
            if len(compiled_regions) > 0:
                cr_rref = compiled_regions.row_reduce()
                cer_rref = compiled_exp_regions.row_reduce()

                cr_nonzero = [row for row in cr_rref if any(row)]
                cer_nonzero = [row for row in cer_rref if any(row)]

                assert len(cr_nonzero) == len(cr_rref)
                assert len(cer_nonzero) == len(cer_rref)

                assert np.array_equal(cr_rref, cer_rref), "Region spaces are not equal"

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
        except AssertionError:
            web_io.write_stabilising(stabs, d, file_name_suffix="_actual")
            web_io.write_detecting(regions, d, file_name_suffix="_actual")

            raise

    return _assert


def test_identity_webs(assert_pauli_webs):
    d = Diagram()
    b1 = d.add_node(NodeType.B)
    b2 = d.add_node(NodeType.B)
    n = d.add_node(NodeType.Z)
    d.add_edge(b1, n)
    d.add_edge(n, b2)
    d.infer_io_from_boundaries()

    stabs, regions = compute_pauli_webs(d)
    assert_pauli_webs(d, stabs, regions)
    stabs, regions = pauli_webs_through_partitions(d, partitions=[[n]])
    assert_pauli_webs(d, stabs, regions)


def test_zweb_webs(assert_pauli_webs):
    d = from_pyzx(generate.zweb(2, 2))
    stabs, regions = compute_pauli_webs(d)

    assert_pauli_webs(d, stabs, regions)


@pytest.mark.parametrize("code_size,repeat", [(3, 1), (5, 1), (5, 3)])
def test_rotated_surface_code_shor(code_size, repeat, assert_pauli_webs):
    d, partitions = generate.shor_extraction(
        generate.rotated_planar_surface_code_stabilisers(code_size),
        qubits=code_size**2,
        repeat=repeat,
        partition=True,
    )

    stabs, regions = compute_pauli_webs(d)
    assert_pauli_webs(d, stabs, regions)
    stabs, regions = pauli_webs_through_partitions(d, partitions=partitions)
    assert_pauli_webs(d, stabs, regions)
