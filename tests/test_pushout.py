from faulttools import build_flip_operators, generate, push_out
from faulttools.glue.pyzx import from_pyzx
from faulttools.noise import NoiseModel


def test_dagger_form():
    d = from_pyzx(generate.zweb(2, 2))
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.weighted_edge_flip_noise(d)

    dagger_noise_model = push_out(noise_model, flip_ops)

    boundary_edges = d.boundary_edges()
    assert len(noise_model.atomic_faults_with_weight()) == len(dagger_noise_model.atomic_faults_with_weight())
    assert all(set(f.edge_flips.keys()).issubset(boundary_edges) for f in dagger_noise_model.atomic_faults())
