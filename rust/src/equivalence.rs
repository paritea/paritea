use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;

type Fault = BigUint;
type Weight = usize;

type FaultQueue = FxHashMap<Weight, FxHashSet<Fault>>;

/// Build a bitmask of `num_bits` ones (i.e. `(1 << num_bits) - 1`).
fn ones_mask(num_bits: usize) -> BigUint {
    if num_bits == 0 {
        return BigUint::ZERO;
    }
    (BigUint::from(1u32) << num_bits) - 1u32
}

/// Check whether `sig & mask` is non-zero.
fn has_any_bit(sig: &BigUint, mask: &BigUint) -> bool {
    // BigUint stores data as a Vec<u32>; iterating digits avoids allocation.
    sig.iter_u32_digits()
        .zip(mask.iter_u32_digits())
        .any(|(s, m)| s & m != 0)
}

/// Compute `sig & mask` (allocating a new BigUint).
fn apply_mask(sig: &BigUint, mask: &BigUint) -> BigUint {
    sig & mask
}

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
        let detector_mask = ones_mask(num_detectors);
        let mut weight_lookup = FxHashMap::default();
        let mut undetectable = FxHashSet::default();
        let mut detectable_with_detectors = FxHashMap::default();

        for (sig, v) in atomic_faults {
            let detectable = has_any_bit(&sig, &detector_mask);
            if detectable {
                if let Entry::Vacant(e) = detectable_with_detectors.entry(sig.clone()) {
                    e.insert(apply_mask(&sig, &detector_mask));
                    weight_lookup.insert(sig.clone(), v);
                }
            } else {
                if undetectable.insert(sig.clone()) {
                    weight_lookup.insert(sig.clone(), v);
                }
            }

            if let Some(w) = weight_lookup.get_mut(&sig) {
                if v < *w {
                    *w = v;
                }
            }
        }

        Self {
            weight_lookup,
            undetectable,
            detectable_with_detectors,
        }
    }

    fn all_iter(&self) -> impl Iterator<Item = (&Fault, &Weight)> {
        self.weight_lookup.iter()
    }

    /// Return all detectable sigs whose detector bits overlap with `detector_info`,
    /// filtered to those with the lowest weight among them.
    fn detector_overlapping(&self, detector_info: &BigUint) -> Vec<&Fault> {
        let mut lowest_weight: Option<Weight> = None;
        let mut lowest_weight_sigs = Vec::new();

        for (sig, sig_info) in &self.detectable_with_detectors {
            if !has_any_bit(detector_info, sig_info) {
                continue;
            }
            let w = self.weight_lookup[sig];
            match lowest_weight {
                Some(lw) if w > lw => continue,
                Some(lw) if w == lw => lowest_weight_sigs.push(sig),
                _ => {
                    lowest_weight = Some(w);
                    lowest_weight_sigs.clear();
                    lowest_weight_sigs.push(sig);
                }
            }
        }

        lowest_weight_sigs
    }
}

fn prepare_priority_queue(atomics: &AtomicFaults) -> FaultQueue {
    let mut queue = FaultQueue::default();
    for (sig, &v) in atomics.all_iter() {
        queue.entry(v).or_default().insert(sig.clone());
    }
    queue
}

fn next_gen_unfold(
    w: Weight,
    queue: &mut FaultQueue,
    detectable_lookup: &mut FxHashMap<Fault, Weight>,
    undetectable_lookup: &mut FxHashMap<BigUint, Weight>,
    atomics: &mut AtomicFaults,
    num_detectors: usize,
) -> FxHashSet<BigUint> {
    let detector_mask = ones_mask(num_detectors);

    let mut current_queue = match queue.remove(&w) {
        Some(q) if !q.is_empty() => q,
        _ => return FxHashSet::default(),
    };
    let mut undetectables_generated = FxHashSet::default();

    while !current_queue.is_empty() {
        let mut new_w_queue = FxHashSet::default();

        for sig in &current_queue {
            let detectable = has_any_bit(sig, &detector_mask);
            if detectable {
                if let Some(&existing_w) = detectable_lookup.get(sig) {
                    if existing_w <= w {
                        continue; // This signature does not provide a weight improvement
                    }
                }
                detectable_lookup.insert(sig.clone(), w);
            } else {
                let sig_no_sinks = sig >> num_detectors;
                if let Some(&existing_w) = undetectable_lookup.get(&sig_no_sinks) {
                    if existing_w <= w {
                        continue; // This signature does not provide a weight improvement
                    }
                }
                undetectable_lookup.insert(sig_no_sinks.clone(), w);
                undetectables_generated.insert(sig_no_sinks);
            }

            if let Some(existing_w) = atomics.weight_lookup.get_mut(sig) {
                if *existing_w > w {
                    *existing_w = w; // Improved atomic weight found
                }
            }

            let atomic_sigs: Vec<&Fault> = if !detectable {
                atomics.undetectable.iter().collect()
            } else {
                atomics
                    .detector_overlapping(&apply_mask(sig, &detector_mask))
                    .into_iter()
                    .collect()
            };

            for atomic_sig in atomic_sigs {
                let comb_w = atomics.weight_lookup[atomic_sig] + w;
                let combined = atomic_sig ^ sig;
                if comb_w == w {
                    new_w_queue.insert(combined);
                } else {
                    queue.entry(comb_w).or_default().insert(combined);
                }
            }
        }

        current_queue = new_w_queue;
    }

    undetectables_generated
}

/// Takes weighted fault signatures of nm1,nm2 in normalised form (stabilisers factored out),
/// encoded as integers with bits as `<z and x boundary flips><detector flips>`.
///
/// Determines the smallest weight of a combination `comb_sig` from elements of `nm2_sigs` such
/// that
///
/// - `comb_sig` does not flip any detectors and is thus not detectable, AND EITHER
/// - `comb_sig` does not have an equivalent in `nm1_sigs` OR
/// - the equivalent of `comb_sig` in `nm1_sigs` has a greater weight.
///
/// Simultaneously checks the opposite direction with `nm2_sigs` and `nm1_sigs` swapped.
///
/// Returns the weight of such a combination or `None` if it does not exist.
pub fn check_fault_equivalence(
    nm1_sigs: Vec<(Fault, Weight)>,
    nm2_sigs: Vec<(Fault, Weight)>,
    d1_detectors: usize,
    d2_detectors: usize,
    until: Option<Weight>,
    _quiet: bool,
) -> Option<usize> {
    let mut nm1_detectable_lookup = FxHashMap::default();
    let mut nm1_undetectable_lookup = FxHashMap::default();
    nm1_undetectable_lookup.insert(Fault::ZERO, 0);

    let mut nm2_detectable_lookup = FxHashMap::default();
    let mut nm2_undetectable_lookup = FxHashMap::default();
    nm2_undetectable_lookup.insert(Fault::ZERO, 0);

    let mut nm1_atomics = AtomicFaults::from_faults(nm1_sigs, d1_detectors);
    let mut nm1_pq = prepare_priority_queue(&nm1_atomics);

    let mut nm2_atomics = AtomicFaults::from_faults(nm2_sigs, d2_detectors);
    let mut nm2_pq = prepare_priority_queue(&nm2_atomics);

    let mut w: Weight = 0;

    while (!nm1_pq.is_empty() || !nm2_pq.is_empty()) && until.map_or(true, |u| w < u - 1) {
        w += 1;

        let nm1_undetectable = next_gen_unfold(
            w,
            &mut nm1_pq,
            &mut nm1_detectable_lookup,
            &mut nm1_undetectable_lookup,
            &mut nm1_atomics,
            d1_detectors,
        );

        let nm2_undetectable = next_gen_unfold(
            w,
            &mut nm2_pq,
            &mut nm2_detectable_lookup,
            &mut nm2_undetectable_lookup,
            &mut nm2_atomics,
            d2_detectors,
        );

        for sig in &nm1_undetectable {
            if !nm2_undetectable_lookup.contains_key(sig) {
                return Some(w);
            }
        }

        for sig in &nm2_undetectable {
            if !nm1_undetectable_lookup.contains_key(sig) {
                return Some(w);
            }
        }
    }

    None
}
