import heapq
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
    g1_lookup = {0: 0}  # The trivial signature requires zero signatures to generate
    g2_lookup = {}

    g1_unique_sigs = {f for f, w in g1_sigs}
    g2_unique_sigs = {f for f, w in g2_sigs}

    if len(g2_unique_sigs) == 0:
        if not quiet:
            print("No signatures to match for g2!")
        return None
    num_max_signatures = 2 ** (g2_boundaries // 2 + g2_sinks)
    num_total_signatures = 0

    if not quiet:
        print(f"Starting iteration until {len(g2_unique_sigs)}!")
    g1_last_new_signatures = [0]
    g2_last_new_detectable = [0]
    g1_sink_mask = (1 << g1_sinks) - 1
    g2_sink_mask = (1 << g2_sinks) - 1
    for max_size in tqdm(
        range(1, len(g2_unique_sigs) + 1),
        desc="Weight: ",
        initial=1,
        total=len(g2_unique_sigs),
        leave=False,
        disable=quiet,
    ):
        if not quiet:
            tqdm.write(f"Starting iteration with combinatory size: {max_size}...")
        # Populate g1_lookup for this weight
        g1_time = time.time()
        g1_new_signatures = []
        for last_it_int, atomic_sig_nf_int in itertools.product(g1_last_new_signatures, g1_unique_sigs):
            combined_sig = last_it_int ^ atomic_sig_nf_int
            if combined_sig & g1_sink_mask > 0:
                continue  # Detectable g1 signatures will never be queried, save space here

            combined_sig_no_sinks = combined_sig >> g1_sinks
            if combined_sig_no_sinks not in g1_lookup:
                g1_new_signatures.append(combined_sig)
                g1_lookup[combined_sig_no_sinks] = max_size
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
            if combined_sig >> g2_sinks not in g1_lookup:
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


def _regular_strategy(
    g1_sigs: list[tuple[int, int]],
    g2_sigs: list[tuple[int, int]],
    g1_sinks: int,
    g2_boundaries: int,
    g2_sinks: int,
    *,
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
    # The trivial signature requires zero signatures to generate
    g1_detectable_lookup = {}
    g1_undetectable_lookup = {0: 0}
    g2_lookup = {0: 0}

    if len(g2_sigs) == 0:
        if not quiet:
            print("No signatures to match for g2!")
        return None

    g1_sink_mask = (1 << g1_sinks) - 1
    g2_sink_mask = (1 << g2_sinks) - 1

    g1_atomic_lookup: dict[int, int] = {}
    for sig, v in g1_sigs:
        if sig in g1_atomic_lookup and g1_atomic_lookup[sig] <= v:
            continue
        g1_atomic_lookup[sig] = v
    g1_pq = [(v, sig) for sig, v in g1_atomic_lookup.items()]
    heapq.heapify(g1_pq)

    g2_atomic_lookup: dict[int, int] = {}
    for sig, v in g2_sigs:
        if sig in g2_atomic_lookup and g2_atomic_lookup[sig] <= v:
            continue
        g2_atomic_lookup[sig] = v
    g2_pq = [(v, sig) for sig, v in g2_atomic_lookup.items()]
    heapq.heapify(g2_pq)

    max_weight = 0
    t = tqdm(desc="Weight: ", initial=0, leave=False, disable=quiet)
    while len(g2_pq) > 0:
        max_weight += 1
        t.update()

        while len(g1_pq) > 0 and g1_pq[0][0] <= max_weight:
            w, sig = heapq.heappop(g1_pq)
            if sig & g1_sink_mask > 0:
                if sig in g1_detectable_lookup and g1_detectable_lookup[sig] <= w:
                    continue  # This signature does not provide a weight improvement
                g1_detectable_lookup[sig] = w
            else:
                sig_no_sinks = sig >> g1_sinks
                if sig_no_sinks in g1_undetectable_lookup and g1_undetectable_lookup[sig_no_sinks] <= w:
                    continue  # This signature does not provide a weight improvement
                g1_undetectable_lookup[sig_no_sinks] = w

            if sig in g1_atomic_lookup and g1_atomic_lookup[sig] > w:
                g1_atomic_lookup[sig] = w

            for atomic_sig, atomic_w in g1_atomic_lookup.items():
                heapq.heappush(g1_pq, (atomic_w + w, atomic_sig ^ sig))
        tqdm.write(f"Finished unfolding weight {max_weight} in queue 1! Items remaining: {len(g1_pq)}...")

        while len(g2_pq) > 0 and g2_pq[0][0] <= max_weight:
            w, sig = heapq.heappop(g2_pq)
            if sig in g2_lookup and g2_lookup[sig] <= w:
                continue  # This signature does not provide a weight improvement
            g2_lookup[sig] = w

            if sig & g2_sink_mask == 0:
                sig_no_sinks = sig >> g2_sinks
                if sig_no_sinks not in g1_undetectable_lookup:
                    if not quiet:
                        tqdm.write(
                            f"{_format_sig(sig, g2_boundaries, g2_sinks)} has no equivalent in g1, or it was not "
                            f"yet generated and thus has higher weight!"
                        )
                    return max_weight

            if sig in g2_atomic_lookup and g2_atomic_lookup[sig] > w:
                g2_atomic_lookup[sig] = w

            for atomic_sig, atomic_w in g2_atomic_lookup.items():
                heapq.heappush(g2_pq, (atomic_w + w, atomic_sig ^ sig))
        tqdm.write(f"Finished unfolding weight {max_weight} in queue 2! Items remaining: {len(g2_pq)}...")

    if len(g1_pq) == 0 and len(g2_pq) != 0:
        return max_weight  # We ran out of signatures to generate for g1

    return None
