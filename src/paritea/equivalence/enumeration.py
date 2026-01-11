import itertools
import time

from tqdm.auto import tqdm


def _format_sig(sig: int, boundaries: int, sinks: int) -> str:
    sig_str = format(sig, "b").zfill(boundaries * 2 + sinks)
    b_str = f"{' '.join(sig_str[:boundaries])} | {' '.join(sig_str[boundaries : boundaries * 2])}"
    if sinks == 0:
        return f"[{b_str}]"

    return f"[{b_str}  ||  {' '.join(sig_str[boundaries * 2 :])}]"


def _normal_strategy(
    g1_sigs: list[tuple[int, int]],
    g2_sigs: list[tuple[int, int]],
    g1_sinks: int,
    g2_boundaries: int,
    g2_sinks: int,
    *,
    until: int | None = None,
    quiet: bool = True,
) -> int | None:
    """
    Takes fault signatures of g1,g2 in normal form (stabilisers factored out) where the sink containment information is
    provided in the last `..._sinks` elements of the signature.

    Determines the smallest size of a combination `comb_sig` from elements of `g2_sig_nf` such that

    - `comb_sig` does not enable any sinks and is thus not detectable AND EITHER
    - `comb_sig` does not have an equivalent in `g1_sig_nf` (without sink information) OR
    - the equivalent of `comb_sig` in `g1_sig_nf` has a greater size

    :returns: the size of such a combination or `None` if no such combination exists.
    """
    g1_undetectable_lookup = {0: 0}  # The trivial signature requires zero signatures to generate
    g1_detectable_lookup = {}
    g2_lookup = {}

    g1_unique_sigs = {f for f, w in g1_sigs}
    g2_unique_sigs = {f for f, w in g2_sigs}

    if len(g2_unique_sigs) == 0:
        if not quiet:
            print("No signatures to match for g2!")
        return None
    num_max_signatures = 2 ** (g2_boundaries + g2_sinks)
    num_total_signatures = 0

    if not quiet:
        print(f"Starting iteration until {len(g2_unique_sigs)}!")
    g1_last_new_signatures = [0]
    g2_last_new_detectable = [0]
    g1_sink_mask = (1 << g1_sinks) - 1
    g2_sink_mask = (1 << g2_sinks) - 1
    max_weight = min(len(g2_unique_sigs), until - 1) if until is not None else len(g2_unique_sigs)
    for max_size in tqdm(
        range(1, max_weight + 1), desc="Weight: ", initial=1, total=max_weight, leave=False, disable=quiet
    ):
        if not quiet:
            tqdm.write(f"Starting iteration with combinatory size: {max_size}...")
        # Populate g1_lookup for this weight
        g1_time = time.time()
        g1_new_signatures = []
        for last_it_int, atomic_sig_nf_int in itertools.product(g1_last_new_signatures, g1_unique_sigs):
            combined_sig = last_it_int ^ atomic_sig_nf_int
            if combined_sig & g1_sink_mask > 0:
                if combined_sig not in g1_detectable_lookup:
                    g1_new_signatures.append(combined_sig)
                    g1_detectable_lookup[combined_sig] = max_size
            else:
                combined_sig_no_sinks = combined_sig >> g1_sinks
                if combined_sig_no_sinks not in g1_undetectable_lookup:
                    g1_new_signatures.append(combined_sig)
                    g1_undetectable_lookup[combined_sig_no_sinks] = max_size
        g1_last_new_signatures = g1_new_signatures
        if not quiet:
            tqdm.write(f"Populating g1 lookup took {time.time() - g1_time}s.")

        # Incrementally discover g2 signatures by combining #`max_size` atomic signatures
        g2_time = time.time()
        num_new_signatures = 0
        g2_new_detectable = []
        for last_it_int, atomic_sig_nf_int in tqdm(
            itertools.product(g2_last_new_detectable, g2_unique_sigs),
            desc="Signatures: ",
            total=len(g2_last_new_detectable) * len(g2_unique_sigs),
            leave=False,
            disable=quiet,
        ):
            combined_sig = last_it_int ^ atomic_sig_nf_int
            if combined_sig in g2_lookup:
                continue  # Already discovered

            num_new_signatures += 1
            g2_lookup[combined_sig] = max_size
            if combined_sig & g2_sink_mask > 0:
                g2_new_detectable.append(combined_sig)
                continue  # Detectable

            # Perform search with real output signature
            if combined_sig >> g2_sinks not in g1_undetectable_lookup:
                if not quiet:
                    tqdm.write(
                        f"{_format_sig(combined_sig, g2_boundaries, g2_sinks)} has no equivalent in g1, or it was not "
                        f"yet generated and thus has higher weight!"
                    )
                return max_size  # No equivalent error with equal or lower weight found
        if not quiet:
            tqdm.write(f"Populating g2 lookup took {time.time() - g2_time}s.")

        g2_last_new_detectable = g2_new_detectable
        num_total_signatures += num_new_signatures
        if num_new_signatures == 0:
            if not quiet:
                tqdm.write("No new signatures discovered!")
            break
        else:
            if not quiet:
                tqdm.write(
                    f"Discovered {num_new_signatures} new signatures this iteration (so far: {num_total_signatures}, "
                    f"max signatures: {num_max_signatures})!"
                )

    return None


def _next_gen_unfold(
    w: int,
    pq: dict[int, list[int]],
    detectable_lookup: dict[int, int],
    undetectable_lookup: dict[int, int],
    atomic_lookup: dict[int, int],
    *,
    num_detectors: int,
    quiet: bool = True,
) -> set[int]:
    detector_mask = (1 << num_detectors) - 1
    queue = pq.pop(w, [])
    sigs_pgb = tqdm(
        desc="Sigs remaining: ",
        initial=len(queue),
        total=len(queue),
        leave=False,
        disable=quiet,
        position=1,
        unit="",
    )
    undetectables_generated = []
    while len(queue) > 0:
        new_queue = []
        for sig in queue:
            sigs_pgb.update(n=-1)
            if sig & detector_mask > 0:
                if sig in detectable_lookup and detectable_lookup[sig] <= w:
                    continue  # This signature does not provide a weight improvement
                detectable_lookup[sig] = w
            else:
                sig_no_sinks = sig >> num_detectors
                if sig_no_sinks in undetectable_lookup and undetectable_lookup[sig_no_sinks] <= w:
                    continue  # This signature does not provide a weight improvement
                undetectable_lookup[sig_no_sinks] = w
                undetectables_generated.append(sig_no_sinks)

            if sig in atomic_lookup and atomic_lookup[sig] > w:
                atomic_lookup[sig] = w

            for atomic_sig, atomic_w in atomic_lookup.items():
                comb_w = atomic_w + w
                if comb_w == w:
                    new_queue.append(atomic_sig ^ sig)
                    sigs_pgb.update(n=1)
                else:
                    if comb_w not in pq:
                        pq[comb_w] = []
                    pq[comb_w].append(atomic_sig ^ sig)
        queue = new_queue
    sigs_pgb.close()

    return set(undetectables_generated)


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

    nm1_lowest_atomic_lookup: dict[int, int] = {}
    for sig, v in nm1_sigs:
        if sig in nm1_lowest_atomic_lookup and nm1_lowest_atomic_lookup[sig] <= v:
            continue
        nm1_lowest_atomic_lookup[sig] = v
    nm1_pq: dict[int, list[int]] = {}
    for sig, v in nm1_lowest_atomic_lookup.items():
        if v not in nm1_pq:
            nm1_pq[v] = []
        nm1_pq[v].append(sig)

    nm2_lowest_atomic_lookup: dict[int, int] = {}
    for sig, v in nm2_sigs:
        if sig in nm2_lowest_atomic_lookup and nm2_lowest_atomic_lookup[sig] <= v:
            continue
        nm2_lowest_atomic_lookup[sig] = v
    nm2_pq: dict[int, list[int]] = {}
    for sig, v in nm2_lowest_atomic_lookup.items():
        if v not in nm2_pq:
            nm2_pq[v] = []
        nm2_pq[v].append(sig)

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
            nm1_lowest_atomic_lookup,
            num_detectors=d1_detectors,
            quiet=quiet,
        )
        tqdm.write(f"Finished unfolding weight {w} in queue 1! Next items remaining: {len(nm1_pq.get(w + 1, []))}...")

        nm2_undetectable = _next_gen_unfold(
            w,
            nm2_pq,
            nm2_detectable_lookup,
            nm2_undetectable_lookup,
            nm2_lowest_atomic_lookup,
            num_detectors=d2_detectors,
            quiet=quiet,
        )
        tqdm.write(f"Finished unfolding weight {w} in queue 2! Next items remaining: {len(nm2_pq.get(w + 1, []))}...")

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
