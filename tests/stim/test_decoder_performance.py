import sinter
import stim

from paritea.glue.stim import (
    error_rates_equal,
    export_to_stim_dem,
    from_stim,
    push_out_for_measurement_detectors,
    wrap_dem_as_sinter_task,
)


def test_decoder_performance_parity():
    """Asserts that paritea can replicate the performance of stims detectors on the
    surface code, given knowledge about the specific detectors in use."""
    p = 1e-3
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=2,
        distance=7,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    c = c.flattened()
    stim_dem = c.detector_error_model()
    # Paritea translate and dem
    _, nm, measurement_nodes, observables, detectors = from_stim(c)
    pushed_out, logical_regions, detector_regions = push_out_for_measurement_detectors(
        nm,
        measurement_nodes=measurement_nodes,
        logicals=list(observables.values()),
        detectors=detectors,
    )
    pushed_out.compress(lambda x, y: x * (1 - y) + (1 - x) * y)
    paritea_dem = export_to_stim_dem(
        pushed_out,
        logical_regions=logical_regions,
        detector_regions=detector_regions,
    )

    [stim_stats, paritea_stats] = sinter.collect(
        num_workers=8,
        tasks=[
            sinter.Task(circuit=c, detector_error_model=stim_dem, json_metadata={"p": p, "name": "stim"}),
            wrap_dem_as_sinter_task(paritea_dem, json_metadata={"p": p, "name": "paritea"}),
        ],
        max_shots=10_000_000,
        max_errors=1_000,
        decoders=["pymatching"],
    )

    assert error_rates_equal(stim_stats, paritea_stats)
