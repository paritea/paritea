use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

type Fault = BigUint;
type Weight = usize;

struct AtomicFaults {
    weight_lookup: FxHashMap<Fault, Weight>,
    undetectable: FxHashSet<Fault>,
    detectable_with_detectors: FxHashMap<Fault, BigUint>,
}

impl AtomicFaults {
    fn from_faults(
        atomic_faults: impl IntoIterator<Item = (Fault, Weight)>,
        num_detectors: usize,
    ) -> Self {
        let mut weight_lookup = FxHashMap::default();
        let mut undetectable = FxHashSet::default();
        let mut detectable_with_detectors = FxHashMap::default();

        for (fault, weight) in atomic_faults {
            let detectable =
        }

        Self {
            weight_lookup,
            undetectable,
            detectable_with_detectors,
        }
    }
}

/// Takes weighted fault signatures of nm1,nm2 in normalised form (stabilisers factored out), encoded as integers with
///     bits as `<z and x boundary flips><detector flips>`.
///
///     Determines the smallest weight of a combination `comb_sig` from elements of `nm2_sigs` such that
///
///     - `comb_sig` does not flip any detectors is thus not detectable, AND EITHER
///     - `comb_sig` does not have an equivalent in `nm1_sigs` OR
///     - the equivalent of `comb_sig` in `nm1_sigs` has a greater weight.
///
///     Simultaneously checks the opposite direction with `nm2_sigs` and `nm1_sigs` swapped.
///
///     :returns: the weight of such a combination or `None` if it does not exist.
pub fn check_fault_equivalence(
    nm1_sigs: Vec<(Fault, Weight)>,
    nm2_sigs: Vec<(Fault, Weight)>,
    d1_boundaries: usize,
    d1_detectors: usize,
    d2_boundaries: usize,
    d2_detectors: usize,
    until: Option<Weight>,
    quiet: bool,
) -> Option<usize> {
}
