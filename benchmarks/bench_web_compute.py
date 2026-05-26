"""Benchmark suite for Pauli web computation speed.

Measures stabiliser, detecting-region, combined, and partitioned computation
across rotated surface code memory experiments at varying distances and rounds.

Usage:
    # Run benchmarks and save results to JSON:
    pytest benchmarks/bench_web_compute.py --benchmark-json=benchmarks/results.json

    # Plot results from the saved JSON:
    python benchmarks/bench_web_compute.py benchmarks/results.json
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass

import pytest

from paritea.diagram import Diagram
from paritea.generate.diagram.surface_code import surface_code_memory_experiment
from paritea.web.compute import (
    compute_detecting_regions,
    compute_pauli_webs,
    compute_stabilisers,
)
from paritea.web.partitions import pauli_webs_through_partitions

DISTANCES = [3, 5, 7]
ROUNDS = [1, 2, 3, 5, 10, 15, 20]


@dataclass(frozen=True)
class SurfaceCodeInstance:
    distance: int
    rounds: int
    diagram: Diagram
    partitions: list[list[int]] | None


def _build_instance(distance: int, rounds: int) -> SurfaceCodeInstance:
    diagram, partitions = surface_code_memory_experiment(distance=distance, rounds=rounds, partition=True)
    return SurfaceCodeInstance(distance=distance, rounds=rounds, diagram=diagram, partitions=partitions)


# Cache built diagrams across the whole session so each is constructed once.
_INSTANCE_CACHE: dict[tuple[int, int], SurfaceCodeInstance] = {}


def _get_instance(distance: int, rounds: int) -> SurfaceCodeInstance:
    key = (distance, rounds)
    if key not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[key] = _build_instance(distance, rounds)
    return _INSTANCE_CACHE[key]


# ---------------------------------------------------------------------------
# Parametrised benchmarks
# ---------------------------------------------------------------------------

_PARAMS = list(itertools.product(DISTANCES, ROUNDS))
_IDS = [f"d{d}_r{r}" for d, r in _PARAMS]


@pytest.mark.parametrize(("distance", "rounds"), _PARAMS, ids=_IDS)
def test_bench_stabilisers(benchmark, distance, rounds):
    inst = _get_instance(distance, rounds)
    benchmark(compute_stabilisers, inst.diagram)


@pytest.mark.parametrize(("distance", "rounds"), _PARAMS, ids=_IDS)
def test_bench_detecting_regions(benchmark, distance, rounds):
    inst = _get_instance(distance, rounds)
    benchmark(compute_detecting_regions, inst.diagram)


@pytest.mark.parametrize(("distance", "rounds"), _PARAMS, ids=_IDS)
def test_bench_pauli_webs(benchmark, distance, rounds):
    inst = _get_instance(distance, rounds)
    benchmark(compute_pauli_webs, inst.diagram)


@pytest.mark.parametrize(("distance", "rounds"), _PARAMS, ids=_IDS)
def test_bench_partitioned(benchmark, distance, rounds):
    inst = _get_instance(distance, rounds)
    benchmark(pauli_webs_through_partitions, inst.diagram, partitions=inst.partitions)


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

    # Map benchmark names → {(method, distance, rounds): median_seconds}
    KIND_LABELS = {
        "test_bench_stabilisers": "Stabilisers",
        "test_bench_detecting_regions": "Detecting regions",
        "test_bench_pauli_webs": "Combined",
        "test_bench_partitioned": "Partitioned",
    }

    # Keyed by (method, distance) → {rounds: median}
    series: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for bench in data["benchmarks"]:
        full_name: str = bench["name"]
        # full_name looks like "test_bench_stabilisers[d3_r1]"
        func_name, param_tag = full_name.split("[")
        param_tag = param_tag.rstrip("]")
        method = KIND_LABELS.get(func_name, func_name)
        parts = param_tag.replace("d", "").split("_r")
        distance, rounds = int(parts[0]), int(parts[1])
        median = bench["stats"]["median"]
        series[(method, distance)][rounds] = median

    # Line style per method, color per distance
    METHOD_STYLES: dict[str, tuple[str, str]] = {
        "Stabilisers": ("-", "o"),
        "Detecting regions": ("--", "s"),
        "Combined": ("-.", "^"),
        "Partitioned": (":", "D"),
    }
    all_distances = sorted({d for _, d in series})
    cmap = plt.colormaps["tab10"]
    distance_colors = {d: cmap(i) for i, d in enumerate(all_distances)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for (method, distance), timing_by_rounds in sorted(series.items()):
        ls, marker = METHOD_STYLES.get(method, ("-", "o"))
        rounds_sorted = sorted(timing_by_rounds)
        medians = [timing_by_rounds[r] for r in rounds_sorted]
        ax.plot(
            rounds_sorted,
            medians,
            linestyle=ls,
            marker=marker,
            color=distance_colors[distance],
            label=f"{method}, d={distance}",
            linewidth=2,
        )

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Median time (s)")
    ax.set_yscale("log")
    ax.set_xticks(sorted({r for vals in series.values() for r in vals}))
    ax.set_title("Pauli web computation benchmark")
    ax.legend(fontsize="small", ncol=2)
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
