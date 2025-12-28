import dataclasses
from collections.abc import Iterable

from .diagram import Diagram
from .pauli import Pauli, PauliString
from .web import compute_pauli_webs


@dataclasses.dataclass(init=True, frozen=True)
class FlipOperators:
    diagram: Diagram
    stab_flip_ops: list[PauliString]
    stab_gen_set: list[PauliString]
    region_gen_set: list[PauliString]


def _flip_operators(
    web_generating_set: Iterable[PauliString],
    restriction_func=lambda x: x,
) -> tuple[list[PauliString], list[PauliString]]:
    """
    Calculates flip operators for the given collection of webs, which is presumed to be a minimal generating set for
    some space under the given restriction function (defaults to identity).

    Note that this function might change the generating set in use, which will be returned.

    :return: The flip operators and the new generating set, such that the items at the same indices correspond.
    """

    flip_ops = []
    new_gen_set = list(web_generating_set)
    for curr_gen_idx in range(len(new_gen_set)):
        curr_gen = restriction_func(new_gen_set[curr_gen_idx])
        flip_op = None
        for edge, p in curr_gen.items():
            if p != Pauli.I:
                flip_op = PauliString.unary(edge, Pauli.Z if p == Pauli.X else Pauli.X)
                break
        if flip_op is None:
            raise AssertionError(f"No flip operator found for generator {curr_gen}!")

        flip_ops.append(flip_op)
        new_gen_set = [
            new_gen_set[i]
            if i == curr_gen_idx or restriction_func(new_gen_set[i]).commutes(flip_op)
            else new_gen_set[i] * new_gen_set[curr_gen_idx]
            for i in range(len(new_gen_set))
        ]

    return flip_ops, new_gen_set


def build_flip_operators(d: Diagram) -> FlipOperators:
    """
    Builds flip operators, obtaining new generating sets for the stabilising and detecting webs of the diagram.
    """
    if d.is_io_virtual():
        raise ValueError("Diagram must have real IO to build flip operators!")

    stabs, regions = compute_pauli_webs(d)

    stab_flip_ops, stab_gen_set = _flip_operators(stabs, lambda w: w.restrict(d.boundary_edges()))

    return FlipOperators(d, stab_flip_ops, stab_gen_set, regions)
