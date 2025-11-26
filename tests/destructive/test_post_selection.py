import time

from faulttools import build_flip_operators, NoiseModel, pushout
from faulttools.destructive import post_select
from faulttools.diagram.conversion import from_pyzx
from generate.diagram import zweb
from generate.diagram.syndrome import generate_shor_extraction
from generate.stabilisers import rotated_planar_surface_code_stabilisers


def test_post_selection_zweb():
    d = from_pyzx(zweb(4, 4))
    flip_ops = build_flip_operators(d)
    noise_model = NoiseModel.edge_flip_noise(d)
    pushed_out_noise_model = pushout(noise_model, flip_ops)

    t = time.time()
    lmao = list(post_select(pushed_out_noise_model, flip_ops.stab_gen_set, len(flip_ops.region_flip_ops)))
    print(f"Generating post selection took {time.time() - t:.2f}s")


def test_post_selection_shor_extraction():
    L = 5
    # Generates a shor extraction of the [7,1,3] Steane code stabilisers
    t = time.time()
    d = generate_shor_extraction(
        rotated_planar_surface_code_stabilisers(L),
        qubits=L**2,
        repeat=1,
    )
    print(f"Generating circuit took {time.time() - t:.2f}s")

    t = time.time()
    flip_ops = build_flip_operators(d)
    print(f"Flip ops took {time.time() - t:.2f}s")

    t = time.time()
    noise_model = NoiseModel.edge_flip_noise(d)
    print(f"Noise model took {time.time() - t:.2f}s")

    t = time.time()
    pushed_out_noise_model = pushout(noise_model, flip_ops)
    print(f"Pushing out took {time.time() - t:.2f}s")

    t = time.time()
    lmao = list(post_select(pushed_out_noise_model, flip_ops.stab_gen_set, len(flip_ops.region_flip_ops)))
    print(f"Generating post selection took {time.time() - t:.2f}s")
    print(f"Found {len(lmao)} undetectable faults under post selection")
