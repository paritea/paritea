import generate.diagram
from faulttools import NoiseModel, build_flip_operators, pushout
from faulttools.diagram.conversion import from_pyzx


def test_dagger_form():
    d = from_pyzx(generate.diagram.zweb(2, 2))
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.edge_flip_noise(d)

    dagger_noise_model = pushout(noise_model, flip_ops)

    boundary_edges = d.boundary_edges()
    assert len(noise_model.atomic_weights()) == len(dagger_noise_model.atomic_weights())
    assert all(set(f.edge_flips.keys()).issubset(boundary_edges) for f in dagger_noise_model.atomic_faults())
