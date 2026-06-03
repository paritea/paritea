//! Allows checking fault equivalence.

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

#[derive(Debug, Default)]
struct AtomicFaults {
    undetectable: FxHashMap<Fault, Weight>,
    detectable_with_detectors: FxHashMap<Fault, (Weight, BigUint)>,
}

impl AtomicFaults {
    fn from_faults(
        atomic_faults: impl IntoIterator<Item = (Fault, Weight)>,
        num_detectors: usize,
    ) -> Self {
        let detector_mask = ones_mask(num_detectors);
        let mut atomics = Self::default();

        for (sig, v) in atomic_faults {
            let detectable = has_any_bit(&sig, &detector_mask);
            if detectable {
                match atomics.detectable_with_detectors.entry(sig.clone()) {
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().0 = v;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert((v, apply_mask(&sig, &detector_mask)));
                    }
                }
            } else {
                match atomics.undetectable.entry(sig.clone()) {
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() = v;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(v);
                    }
                }
            }
        }

        atomics
    }

    fn all_iter(&self) -> impl Iterator<Item = (&Fault, &Weight)> {
        self.undetectable.iter().chain(
            self.detectable_with_detectors
                .iter()
                .map(|(sig, (w, _))| (sig, w)),
        )
    }

    fn undetectable_iter(&self) -> Box<dyn Iterator<Item = (&Fault, Weight)> + '_> {
        Box::new(self.undetectable.iter().map(|(s, &w)| (s, w)))
    }

    /// If the fault is an undetectable atomic fault and the given weight is strictly smaller than
    /// the recorded weight for the atomic fault, updates the recorded weight.
    fn check_update_undetectable_weight(&mut self, sig: &Fault, w: Weight) {
        if let Some(existing_w) = self.undetectable.get_mut(sig)
            && w < *existing_w
        {
            *existing_w = w;
        }
    }

    /// If the fault is a detectable atomic fault and the given weight is strictly smaller than the
    /// recorded weight for the atomic fault, updates the recorded weight.
    fn check_update_detectable_weight(&mut self, sig: &Fault, w: Weight) {
        if let Some((existing_w, _)) = self.detectable_with_detectors.get_mut(sig)
            && w < *existing_w
        {
            *existing_w = w;
        }
    }

    /// Return all detectable sigs whose detector bits overlap with `detector_info`,
    /// filtered to those with the lowest weight among them.
    fn detector_overlapping(
        &self,
        detector_info: &BigUint,
    ) -> Box<dyn Iterator<Item = (&Fault, Weight)> + '_> {
        // TODO improvement by pre-sorting by weight (only improves for different weights (non-avg case))
        // TODO improvement by pre-chunking for detector fields (make sure this is an implementation detail)
        let mut lowest_weight: Option<Weight> = None;
        let mut lowest_weight_sigs = Vec::new();

        for (sig, (w, sig_info)) in &self.detectable_with_detectors {
            if !has_any_bit(detector_info, sig_info) {
                continue;
            }
            match lowest_weight {
                Some(lw) if *w > lw => continue,
                Some(lw) if *w == lw => lowest_weight_sigs.push(sig),
                _ => {
                    lowest_weight = Some(*w);
                    lowest_weight_sigs.clear();
                    lowest_weight_sigs.push(sig);
                }
            }
        }

        let lowest_weight = lowest_weight.unwrap_or_else(|| {
            assert!(lowest_weight_sigs.is_empty());
            0
        });

        Box::new(
            lowest_weight_sigs
                .into_iter()
                .map(move |sig| (sig, lowest_weight)),
        )
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
                match detectable_lookup.get(sig) {
                    Some(&existing_w) if existing_w <= w => continue, // No improvement
                    _ => {
                        detectable_lookup.insert(sig.clone(), w);
                    }
                }
                atomics.check_update_detectable_weight(sig, w);
            } else {
                let sig_no_sinks = sig >> num_detectors;

                match undetectable_lookup.get(&sig_no_sinks) {
                    Some(&existing_w) if existing_w <= w => continue, // No improvement
                    _ => {
                        undetectable_lookup.insert(sig_no_sinks.clone(), w);
                        undetectables_generated.insert(sig_no_sinks);
                    }
                }
                atomics.check_update_undetectable_weight(sig, w)
            }

            let atomic_sigs = if !detectable {
                atomics.undetectable_iter()
            } else {
                atomics.detector_overlapping(&apply_mask(sig, &detector_mask))
            };
            for (atomic_sig, atomic_w) in atomic_sigs {
                let combined = atomic_sig ^ sig;
                if atomic_w == 0 {
                    new_w_queue.insert(combined);
                } else {
                    queue.entry(atomic_w + w).or_default().insert(combined);
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

    while (!nm1_pq.is_empty() || !nm2_pq.is_empty()) && until.is_none_or(|u| w < u - 1) {
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
