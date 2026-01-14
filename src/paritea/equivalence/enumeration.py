import itertools
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

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


@dataclass(init=True)
class AtomicFaults:
    weight_lookup: dict[int, int] = field(default_factory=dict, init=False)
    undetectable: set[int] = field(default_factory=set, init=False)
    detectable_with_detectors: dict[int, int] = field(default_factory=dict, init=False)

    def all_iter(self) -> Iterator[tuple[int, int]]:
        for sig in itertools.chain(self.undetectable, self.detectable_with_detectors.keys()):
            yield sig, self.weight_lookup[sig]

    def detector_overlapping(self, detector_info: int) -> list[int]:
        lowest_weight = math.inf
        lowest_weight_sigs = []
        for sig, sig_info in self.detectable_with_detectors.items():
            if detector_info & sig_info == 0:
                continue
            w = self.weight_lookup[sig]
            if w > lowest_weight:
                continue
            if w == lowest_weight:
                lowest_weight_sigs.append(sig)
            else:
                lowest_weight = w
                lowest_weight_sigs = [sig]

        return lowest_weight_sigs


def prepare_atomic_faults(nm_sigs: list[tuple[int, int]], *, num_detectors: int) -> AtomicFaults:
    atomics: AtomicFaults = AtomicFaults()
    detector_mask = (1 << num_detectors) - 1
    for sig, v in nm_sigs:
        detectable = sig & detector_mask > 0
        if detectable:
            if sig not in atomics.detectable_with_detectors:
                atomics.detectable_with_detectors[sig] = sig & detector_mask
                atomics.weight_lookup[sig] = v
        else:
            if sig not in atomics.undetectable:
                atomics.undetectable.add(sig)
                atomics.weight_lookup[sig] = v

        if atomics.weight_lookup[sig] <= v:
            continue
        atomics.weight_lookup[sig] = v

    return atomics


def prepare_priority_queue(atomics: AtomicFaults) -> dict[int, set[int]]:
    pq: dict[int, set[int]] = {}
    for sig, v in atomics.all_iter():
        if v not in pq:
            pq[v] = set()
        pq[v].add(sig)

    return pq


def _next_gen_unfold(
    w: int,
    pq: dict[int, set[int]],
    detectable_lookup: dict[int, int],
    undetectable_lookup: dict[int, int],
    atomics: AtomicFaults,
    *,
    num_detectors: int,
    quiet: bool = True,
) -> set[int]:
    detector_mask = (1 << num_detectors) - 1
    queue = pq.pop(w, [])
    if len(queue) == 0:
        return set()

    sigs_pgb = tqdm(
        desc="Sigs remaining: ",
        initial=len(queue),
        total=len(queue),
        leave=False,
        disable=quiet,
        position=1,
        unit="",
    )
    undetectables_generated = set()
    items_done, start_time = 0, time.time()
    while len(queue) > 0:
        new_queue = set()
        items_done += len(queue)
        for sig in queue:
            sigs_pgb.update(n=-1)
            detector_info = sig & detector_mask
            detectable = detector_info > 0
            if detectable:
                if sig in detectable_lookup and detectable_lookup[sig] <= w:
                    continue  # This signature does not provide a weight improvement
                detectable_lookup[sig] = w
            else:
                sig_no_sinks = sig >> num_detectors
                if sig_no_sinks in undetectable_lookup and undetectable_lookup[sig_no_sinks] <= w:
                    continue  # This signature does not provide a weight improvement
                undetectable_lookup[sig_no_sinks] = w
                undetectables_generated.add(sig_no_sinks)

            if sig in atomics.weight_lookup and atomics.weight_lookup[sig] > w:
                atomics.weight_lookup[sig] = w

            atomic_faults = atomics.undetectable if not detectable else atomics.detector_overlapping(detector_info)
            for atomic_sig in atomic_faults:
                comb_w = atomics.weight_lookup[atomic_sig] + w
                if comb_w == w:
                    new_queue.add(atomic_sig ^ sig)
                    sigs_pgb.update(n=1)
                else:
                    if comb_w not in pq:
                        pq[comb_w] = set()
                    pq[comb_w].add(atomic_sig ^ sig)
        queue = new_queue
    end_time = time.time()
    sigs_pgb.close()
    if not quiet:
        tqdm.write(
            f"|   w={w} iteration averaged {items_done / (end_time - start_time) / 1000:.2f}k iterations per second ..."
        )

    return undetectables_generated


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
        tqdm.write(f"Finished unfolding weight {w} in queue 1! Next items remaining: {len(nm1_pq.get(w + 1, []))}...")

        nm2_undetectable = _next_gen_unfold(
            w,
            nm2_pq,
            nm2_detectable_lookup,
            nm2_undetectable_lookup,
            nm2_atomics,
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
