import time

from tqdm.auto import tqdm


def _format_sig(sig: int, boundaries: int, sinks: int) -> str:
    sig_str = format(sig, "b").zfill(boundaries * 2 + sinks)
    b_str = f"{' '.join(sig_str[:boundaries])} | {' '.join(sig_str[boundaries : boundaries * 2])}"
    if sinks == 0:
        return f"[{b_str}]"

    return f"[{b_str}  ||  {' '.join(sig_str[boundaries * 2 :])}]"


def _next_gen_strategy(
    nm1_sigs: list[tuple[int, int]],
    nm2_sigs: list[tuple[int, int]],
    d1_boundaries: int,
    d1_detectors: int,
    d2_boundaries: int,
    d2_detectors: int,
    *,
    until: int | None = None,
    quiet: bool = True,
) -> int | None:
    """
    Takes weighted fault signatures of nm1,nm2 in normalised form (stabilisers factored out), encoded as integers with
    bits as `<z and x boundary flips><detector flips>`.

    Determines the smallest weight of a combination `comb_sig` from elements of `nm2_sigs` such that

    - `comb_sig` does not flip any detectors is thus not detectable, AND EITHER
    - `comb_sig` does not have an equivalent in `nm1_sigs` OR
    - the equivalent of `comb_sig` in `nm1_sigs` has a greater weight.

    Simultaneously checks the opposite direction with `nm2_sigs` and `nm1_sigs` swapped.

    :returns: the weight of such a combination or `None` if it does not exist.
    """
    # The trivial signature always has weight 0
    nm1_detectable_lookup = {}
    nm1_undetectable_lookup = {0: 0}
    nm2_detectable_lookup = {}
    nm2_undetectable_lookup = {0: 0}

    nm1_atomics = prepare_atomic_faults(nm1_sigs, num_detectors=d1_detectors)
    nm1_pq = prepare_priority_queue(nm1_atomics)

    nm2_atomics = prepare_atomic_faults(nm2_sigs, num_detectors=d2_detectors)
    nm2_pq = prepare_priority_queue(nm2_atomics)

    w = 0
    w_pgb = tqdm(
        desc="Current weight", initial=0, leave=False, disable=quiet, unit="", bar_format="{desc}: {n_fmt}", ncols=0
    )
    while (len(nm1_pq) > 0 or len(nm2_pq) > 0) and (until is None or w < until - 1):
        w += 1
        w_pgb.update()

        nm1_undetectable = _next_gen_unfold(
            w,
            nm1_pq,
            nm1_detectable_lookup,
            nm1_undetectable_lookup,
            nm1_atomics,
            num_detectors=d1_detectors,
            quiet=quiet,
        )
        if not quiet:
            tqdm.write(
                f"Finished unfolding weight {w} in queue 1! Next items remaining: {len(nm1_pq.get(w + 1, []))}..."
            )

        nm2_undetectable = _next_gen_unfold(
            w,
            nm2_pq,
            nm2_detectable_lookup,
            nm2_undetectable_lookup,
            nm2_atomics,
            num_detectors=d2_detectors,
            quiet=quiet,
        )
        if not quiet:
            tqdm.write(
                f"Finished unfolding weight {w} in queue 2! Next items remaining: {len(nm2_pq.get(w + 1, []))}..."
            )

        for sig in nm1_undetectable:
            if sig not in nm2_undetectable_lookup:
                if not quiet:
                    tqdm.write(
                        f"{_format_sig(sig, d1_boundaries, 0)} from nm1 has no equivalent in nm2, or it was not "
                        f"yet generated and thus has higher weight!"
                    )
                return w

        for sig in nm2_undetectable:
            if sig not in nm1_undetectable_lookup:
                if not quiet:
                    tqdm.write(
                        f"{_format_sig(sig, d2_boundaries, 0)} from nm2 has no equivalent in nm1, or it was not "
                        f"yet generated and thus has higher weight!"
                    )
                return w
    w_pgb.close()

    return None
