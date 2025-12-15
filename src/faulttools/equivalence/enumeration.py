import itertools
import time
from typing import List, Optional

from galois import GF2
from tqdm.auto import tqdm

from faulttools.noise_model import Fault


def _format_sig(sig: int, boundaries: int, sinks: int) -> str:
    sig_str = format(sig, "b").zfill(boundaries * 2 + sinks)
    b_str = f"{' '.join(sig_str[:boundaries])} | {' '.join(sig_str[boundaries : boundaries * 2])}"
    if sinks == 0:
        return f"[{b_str}]"

    return f"[{b_str}  ||  {' '.join(sig_str[boundaries * 2 :])}]"


def _smallest_size_iteration(
    g1_sig_nf: List[GF2],
    g2_sig_nf: List[GF2],
    g1_sinks: int,
    g2_boundaries: int,
    g2_sinks: int,
    quiet: bool = True,
) -> Optional[int]:
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
    g2_lookup = dict()

    if len(g2_sig_nf) == 0:
        if not quiet:
            print("No signatures to match for g2!")
        return None
    num_max_signatures = 2 ** ((len(g2_sig_nf[0]) - g2_sinks) // 2 + g2_sinks)
    num_total_signatures = 0

    if not quiet:
        print(f"Starting iteration until {len(g2_sig_nf)}!")
    g1_last_new_signatures = [0]
    g2_last_new_detectable = [0]
    g1_sig_nf_ints = [Fault.compiled_to_int(sig) for sig in g1_sig_nf]
    g2_sig_nf_ints = [Fault.compiled_to_int(sig) for sig in g2_sig_nf]
    g1_sink_mask = (1 << g1_sinks) - 1
    g2_sink_mask = (1 << g2_sinks) - 1
    for max_size in tqdm(
        range(1, len(g2_sig_nf_ints) + 1),
        desc="Weight: ",
        initial=1,
        total=len(g2_sig_nf_ints),
        leave=False,
        disable=quiet,
    ):
        if not quiet:
            tqdm.write(f"Starting iteration with combinatory size: {max_size}...")
        # Populate g1_lookup for this weight
        g1_time = time.time()
        g1_new_signatures = []
        for last_it_int, atomic_sig_nf_int in itertools.product(g1_last_new_signatures, g1_sig_nf_ints):
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
            itertools.product(g2_last_new_detectable, g2_sig_nf_ints),
            desc="Signatures: ",
            total=len(g2_last_new_detectable) * len(g2_sig_nf_ints),
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
                        f"{_format_sig(combined_sig, g2_boundaries, g2_sinks)} has no equivalent in g1, or it was not yet generated and thus has higher weight!"
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
                    f"Discovered {num_new_signatures} new signatures this iteration (so far: {num_total_signatures}, max signatures: {num_max_signatures})!"
                )

    return None
