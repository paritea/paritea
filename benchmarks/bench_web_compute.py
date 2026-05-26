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

from paritea.generate.diagram.surface_code import surface_code_memory_experiment
from paritea.web.compute import (
    compute_detecting_regions,
    compute_pauli_webs,
    compute_stabilisers,
)
from paritea.web.partitions import pauli_webs_through_partitions

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

DISTANCES = [3, 5, 7]
ROUNDS = [1, 3, 5]

# ---------------------------------------------------------------------------
# Fixtures: pre-build diagrams so construction cost is excluded from timings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceCodeInstance:
    distance: int
    rounds: int
    diagram: object  # Diagram
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

    # Map benchmark names → {(distance, rounds): median_seconds}
    KIND_LABELS = {
        "test_bench_stabilisers": "Stabilisers",
        "test_bench_detecting_regions": "Detecting regions",
        "test_bench_pauli_webs": "Combined",
        "test_bench_partitioned": "Partitioned",
    }

    series: dict[str, dict[str, float]] = defaultdict(dict)
    for bench in data["benchmarks"]:
        full_name: str = bench["name"]
        # full_name looks like "test_bench_stabilisers[d3_r1]"
        func_name, param_tag = full_name.split("[")
        param_tag = param_tag.rstrip("]")
        label = KIND_LABELS.get(func_name, func_name)
        median = bench["stats"]["median"]
        series[label][param_tag] = median

    # Sort parameter tags by (distance, rounds) for consistent x-axis order
    all_tags = sorted(
        {tag for s in series.values() for tag in s},
        key=lambda t: tuple(int(x) for x in t.replace("d", "").split("_r")),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D"]
    for (label, values), marker in zip(sorted(series.items()), itertools.cycle(markers)):
        x_indices = []
        y_values = []
        for i, tag in enumerate(all_tags):
            if tag in values:
                x_indices.append(i)
                y_values.append(values[tag])
        ax.plot(x_indices, y_values, marker=marker, label=label, linewidth=2)

    ax.set_xticks(range(len(all_tags)))
    ax.set_xticklabels(all_tags, rotation=45, ha="right")
    ax.set_xlabel("Surface code instance (distance, rounds)")
    ax.set_ylabel("Median time (s)")
    ax.set_yscale("log")
    ax.set_title("Pauli web computation benchmark")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
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
