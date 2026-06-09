"""Benchmark suite for fault equivalence checking speed.

Measures the time to check fault equivalence between two noise models for:
- Surface code memory experiments at varying distances (weight-limiting check).
- Cat state decompositions at varying sizes.

Usage:
    # Run benchmarks and save results to JSON:
    pytest benchmarks/bench_fault_equivalence.py --benchmark-json=benchmarks/fault_equiv_results.json

    # Plot results from the saved JSON:
    python benchmarks/bench_fault_equivalence.py benchmarks/fault_equiv_results.json
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
import pyzx as zx

from paritea.equivalence import is_fault_equivalence
from paritea.generate import surface_code_memory_experiment
from paritea.noise import NoiseModel

# ---------------------------------------------------------------------------
# Surface code benchmarks
# ---------------------------------------------------------------------------

SURFACE_CODE_DISTANCES = list(range(3, 9, 2))


@dataclass(frozen=True)
class SurfaceCodeEquivInstance:
    distance: int
    nm1: NoiseModel
    nm2: NoiseModel


def _build_surface_code_instance(distance: int) -> SurfaceCodeEquivInstance:
    d, partitions = surface_code_memory_experiment(distance=distance, rounds=2, partition=True)
    [first_part, second_part] = partitions

    def _is_crossing_edge(e: int) -> bool:
        s, t = d.get_edge_endpoints_by_index(e)
        return (s in first_part and t in second_part) or (s in second_part and t in first_part)

    nm1 = NoiseModel.weighted_edge_flip_noise(d, idealised_edges=d.edge_indices())
    nm2 = NoiseModel.weighted_edge_flip_noise(
        d, w_x=1, w_y=2, w_z=1, idealised_edges=[e for e in d.edge_indices() if not _is_crossing_edge(e)]
    )
    return SurfaceCodeEquivInstance(distance=distance, nm1=nm1, nm2=nm2)


_SURFACE_CODE_CACHE: dict[int, SurfaceCodeEquivInstance] = {}


def _get_surface_code_instance(distance: int) -> SurfaceCodeEquivInstance:
    if distance not in _SURFACE_CODE_CACHE:
        _SURFACE_CODE_CACHE[distance] = _build_surface_code_instance(distance)
    return _SURFACE_CODE_CACHE[distance]


_SC_IDS = [f"d{d}" for d in SURFACE_CODE_DISTANCES]


@pytest.mark.parametrize("distance", SURFACE_CODE_DISTANCES, ids=_SC_IDS)
def test_bench_surface_code(benchmark, distance):
    """Check fault equivalence up to weight=distance (should pass)."""
    inst = _get_surface_code_instance(distance)
    benchmark(is_fault_equivalence, inst.nm1, inst.nm2, until=distance, quiet=True)


# ---------------------------------------------------------------------------
# Cat state decomposition benchmarks
# ---------------------------------------------------------------------------

CAT_STATE_SIZES = list(range(2, 8))


def _add_cat_state(g: zx.graph.base.BaseGraph, size: int, qubit: int = 0, row: int = 0) -> tuple[int, list[int]]:
    z = g.add_vertex(zx.VertexType.Z, qubit=qubit, row=row)
    boundaries = [g.add_vertex(zx.VertexType.BOUNDARY, qubit=qubit + i, row=row + 1) for i in range(size)]
    g.add_edges([(z, b) for b in boundaries])
    return z, boundaries


def _add_cz_layer(g: zx.graph.base.BaseGraph, boundaries: list[int]) -> list[int]:
    n = len(boundaries) // 2
    new_bs = [g.add_vertex(zx.VertexType.BOUNDARY, qubit=i, row=2 * (n + 1)) for i in range(2 * n)]
    for i in range(n):
        g.set_type(boundaries[i], zx.VertexType.Z)
        g.set_type(boundaries[i + n], zx.VertexType.Z)
        g.add_edges(
            [
                (boundaries[i], boundaries[i + n]),
                (boundaries[i], new_bs[i]),
                (boundaries[i], new_bs[i + n]),
            ]
        )
    return new_bs


@dataclass(frozen=True)
class CatStateEquivInstance:
    n: int
    g1: zx.Graph
    g2: zx.Graph


def _build_cat_state_instance(n: int) -> CatStateEquivInstance:
    g1 = zx.Graph()
    _add_cat_state(g1, size=2 * n, qubit=0, row=0)

    g2 = zx.Graph()
    _, bs1 = _add_cat_state(g2, size=n, qubit=2, row=0)
    _, bs2 = _add_cat_state(g2, size=n, qubit=6, row=0)
    _add_cz_layer(g2, [*bs1, *bs2])

    return CatStateEquivInstance(n=n, g1=g1, g2=g2)


_CAT_STATE_CACHE: dict[int, CatStateEquivInstance] = {}


def _get_cat_state_instance(n: int) -> CatStateEquivInstance:
    if n not in _CAT_STATE_CACHE:
        _CAT_STATE_CACHE[n] = _build_cat_state_instance(n)
    return _CAT_STATE_CACHE[n]


_CS_IDS = [f"n{n}" for n in CAT_STATE_SIZES]


@pytest.mark.parametrize("n", CAT_STATE_SIZES, ids=_CS_IDS)
def test_bench_cat_state_decomposition(benchmark, n):
    """Check fault equivalence of cat state decomposition (should pass)."""
    inst = _get_cat_state_instance(n)
    benchmark(is_fault_equivalence, inst.g1, inst.g2, quiet=True)


# ---------------------------------------------------------------------------
# Plotting (run this file as a script with the JSON results path)
# ---------------------------------------------------------------------------


def plot_results(json_path: str) -> None:
    import json
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt

    with Path(json_path).open() as f:
        data = json.load(f)

    KIND_LABELS = {
        "test_bench_surface_code": "Surface code (CSS Noise)",
        "test_bench_cat_state_decomposition": "Cat state decomposition",
    }

    # series keyed by method → {param_value: median}
    series: dict[str, dict[int, float]] = defaultdict(dict)
    for bench in data["benchmarks"]:
        full_name: str = bench["name"]
        func_name, param_tag = full_name.split("[")
        param_tag = param_tag.rstrip("]")
        method = KIND_LABELS.get(func_name, func_name)
        # param_tag is like "d3" or "n2"
        param_value = int(param_tag[1:])
        median = bench["stats"]["median"]
        series[method][param_value] = median

    cmap = plt.colormaps["tab10"]
    # Surface code variants share a color, only differ in line style
    STYLES: dict[str, tuple[str, str, int]] = {
        "Cat state decomposition": ("-", "^", 0),
        "Surface code (CSS Noise)": ("-", "o", 1),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, timing in sorted(series.items()):
        ls, marker, color_idx = STYLES.get(method, ("-", "o", 2))
        params_sorted = sorted(timing)
        medians = [timing[p] for p in params_sorted]
        ax.plot(
            params_sorted,
            medians,
            linestyle=ls,
            marker=marker,
            color=cmap(color_idx),
            label=method,
            linewidth=2,
        )

    ax.set_xlabel("Parameter (distance / n)")
    ax.set_ylabel("Median time (s)")
    ax.set_yscale("log")
    ax.set_title("Fault equivalence checking benchmark")
    ax.legend(fontsize="small")
    ax.grid(visible=True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = Path(json_path).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <results.json>")
        sys.exit(1)
    plot_results(sys.argv[1])
