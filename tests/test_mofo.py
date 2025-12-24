import sinter
import stim
from matplotlib import pyplot as plt

from faulttools.glue.stim import export_to_stim_dem, from_stim, wrap_dem_as_sinter_task


def do_(**kwargs) -> tuple[stim.Circuit, stim.DetectorErrorModel]:
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        **kwargs,
    )
    c = c.flattened()

    _, nm, measurement_nodes, observables, detectors = from_stim(c)

    dem, _ = export_to_stim_dem(
        nm, measurement_nodes=measurement_nodes, logicals=list(observables.values()), detectors=detectors
    )

    return c, dem


def test_mofo():
    def to_tasks(d, p: float):
        c, dem = do_(
            rounds=2,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
        print("Done one task")

        return (
            sinter.Task(
                circuit=c,
                detector_error_model=c.detector_error_model(),
                json_metadata={"d": d, "p": p, "name": "stim original"},
            ),
            wrap_dem_as_sinter_task(dem, json_metadata={"d": d, "p": p, "name": "replica"}),
        )

    collected_stats = sinter.collect(
        num_workers=16,
        tasks=[
            t
            for p in [i * 10 ** (-j) for j in range(1, 6) for i in [1, 2, 5]]
            for d in range(3, 11, 2)
            for t in to_tasks(d, p)
        ],
        max_shots=100_000_000,
        max_errors=100_000,
        decoders=["pymatching"],
        print_progress=True,
    )

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stats: stats.json_metadata["p"],
        group_func=lambda stats: {
            "label": f"{stats.json_metadata['name']} d={stats.json_metadata['d']}",
            "linestyle": "loosely dashed" if stats.json_metadata["name"] == "replica" else "dotted",
            "color": f"C{stats.json_metadata['d']}",
        },
    )
    print(f"Stim: {collected_stats[0].errors / collected_stats[0].shots} error rate")
    print(f"NEW: {collected_stats[1].errors / collected_stats[1].shots} error rate")
    ax.set_ylim(auto=True)
    # ax.set_ylim(0.0001, 1)
    ax.set_xlim(auto=True)
    # ax.set_xlim(1e-5, 1e-1)
    ax.loglog()
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)  # Show it bigger
    plt.show()
