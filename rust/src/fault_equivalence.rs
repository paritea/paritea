//! Allows checking fault equivalence.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use num_bigint::BigUint;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, btree_map};
use std::time::{Duration, SystemTime};

type Fault = BigUint;
type Weight = usize;

type FaultQueue<P> = FxHashMap<Weight, BTreeMap<Fault, P>>;

/// Provides provenance tracking for faults. Depending on runtime parameters, monomorphisations can
/// choose to utilise [`NoProv`] for zero-overhead tracking.
trait Provenance: Clone {
    fn identity(index: usize) -> Self;
    fn combine(&self, other: &Self) -> Self;
    fn to_fault_indices(&self) -> Option<Vec<usize>>;
}

/// Zero-sized provenance: all operations are no-ops, optimised away entirely.
#[derive(Clone, Debug)]
struct NoProv;

impl Provenance for NoProv {
    #[inline(always)]
    fn identity(_index: usize) -> Self {
        NoProv
    }
    #[inline(always)]
    fn combine(&self, _other: &Self) -> Self {
        NoProv
    }
    #[inline(always)]
    fn to_fault_indices(&self) -> Option<Vec<usize>> {
        None
    }
}

/// Bitmask provenance: tracks contributing fault indices via XOR of one-hot bits.
#[derive(Clone, Debug)]
struct WithProv(BigUint);

impl Provenance for WithProv {
    fn identity(index: usize) -> Self {
        WithProv(BigUint::from(1u32) << index)
    }
    fn combine(&self, other: &Self) -> Self {
        WithProv(&self.0 ^ &other.0)
    }
    fn to_fault_indices(&self) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        for (i, digit) in self.0.iter_u32_digits().enumerate() {
            let mut d = digit;
            let base = i * 32;
            while d != 0 {
                let bit = d.trailing_zeros() as usize;
                indices.push(base + bit);
                d &= d - 1; // clear lowest set bit
            }
        }
        Some(indices)
    }
}

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

#[derive(Debug)]
struct AtomicFaults<P: Provenance> {
    undetectable: BTreeMap<Fault, (Weight, P)>,
    detectable_with_detectors: BTreeMap<Fault, (Weight, BigUint, P)>,
}

impl<P: Provenance> AtomicFaults<P> {
    fn from_faults(
        atomic_faults: impl IntoIterator<Item = (Fault, Weight)>,
        num_detectors: usize,
    ) -> Self {
        let detector_mask = ones_mask(num_detectors);
        let mut undetectable: BTreeMap<Fault, (Weight, P)> = BTreeMap::new();
        let mut detectable_with_detectors: BTreeMap<Fault, (Weight, BigUint, P)> = BTreeMap::new();

        for (index, (sig, v)) in atomic_faults.into_iter().enumerate() {
            let prov = P::identity(index);
            let detectable = has_any_bit(&sig, &detector_mask);
            if detectable {
                match detectable_with_detectors.entry(sig.clone()) {
                    btree_map::Entry::Occupied(mut entry) => {
                        let (existing_w, _, existing_prov) = entry.get_mut();
                        if v < *existing_w {
                            *existing_w = v;
                            *existing_prov = prov;
                        }
                    }
                    btree_map::Entry::Vacant(entry) => {
                        entry.insert((v, apply_mask(&sig, &detector_mask), prov));
                    }
                }
            } else {
                match undetectable.entry(sig.clone()) {
                    btree_map::Entry::Occupied(mut entry) => {
                        let (existing_w, existing_prov) = entry.get_mut();
                        if v < *existing_w {
                            *existing_w = v;
                            *existing_prov = prov;
                        }
                    }
                    btree_map::Entry::Vacant(entry) => {
                        entry.insert((v, prov));
                    }
                }
            }
        }

        Self {
            undetectable,
            detectable_with_detectors,
        }
    }

    fn all_iter(&self) -> impl Iterator<Item = (&Fault, Weight, &P)> {
        self.undetectable
            .iter()
            .map(|(sig, (w, p))| (sig, *w, p))
            .chain(
                self.detectable_with_detectors
                    .iter()
                    .map(|(sig, (w, _, p))| (sig, *w, p)),
            )
    }

    fn undetectable_iter(&self) -> impl Iterator<Item = (&Fault, Weight, &P)> {
        self.undetectable.iter().map(|(s, (w, p))| (s, *w, p))
    }

    /// If the fault is an undetectable atomic fault and the given weight is strictly smaller than
    /// the recorded weight for the atomic fault, updates the recorded weight.
    fn check_update_undetectable_weight(&mut self, sig: &Fault, w: Weight) {
        if let Some((existing_w, _)) = self.undetectable.get_mut(sig)
            && w < *existing_w
        {
            *existing_w = w;
        }
    }

    /// Return all detectable sigs whose detector bits overlap with `detector_info`,
    /// filtered to those with the lowest weight among them.
    fn detector_overlapping(&self, detector_info: &BigUint) -> Vec<(&Fault, Weight, &P)> {
        // TODO improvement by pre-sorting by weight (only improves for different weights (non-avg case))
        // TODO improvement by pre-chunking for detector fields (make sure this is an implementation detail)
        let mut lowest_weight: Option<Weight> = None;
        let mut lowest_weight_entries: Vec<(&Fault, Weight, &P)> = Vec::new();

        for (sig, (w, sig_info, prov)) in &self.detectable_with_detectors {
            match lowest_weight {
                Some(lw) if *w > lw => continue,
                Some(lw) if *w == lw => {
                    if !has_any_bit(detector_info, sig_info) {
                        continue;
                    }
                    lowest_weight_entries.push((sig, *w, prov));
                }
                _ => {
                    if !has_any_bit(detector_info, sig_info) {
                        continue;
                    }
                    lowest_weight = Some(*w);
                    lowest_weight_entries.clear();
                    lowest_weight_entries.push((sig, *w, prov));
                }
            }
        }

        lowest_weight_entries
    }

    /// If the fault is a detectable atomic fault and the given weight is strictly smaller than the
    /// recorded weight for the atomic fault, updates the recorded weight.
    fn check_update_detectable_weight(&mut self, sig: &Fault, w: Weight) {
        if let Some((existing_w, _, _)) = self.detectable_with_detectors.get_mut(sig)
            && w < *existing_w
        {
            *existing_w = w;
        }
    }
}

fn prepare_priority_queue<P: Provenance>(atomics: &AtomicFaults<P>) -> FaultQueue<P> {
    let mut queue: FaultQueue<P> = FxHashMap::default();
    for (sig, v, prov) in atomics.all_iter() {
        queue
            .entry(v)
            .or_default()
            .entry(sig.clone())
            .or_insert_with(|| prov.clone());
    }
    queue
}

const PB_UPDATE_INTERVAL: usize = 1000;

fn next_gen_unfold<P: Provenance>(
    w: Weight,
    queue: &mut FaultQueue<P>,
    detectable_lookup: &mut FxHashMap<Fault, Weight>,
    undetectable_lookup: &mut FxHashMap<BigUint, Weight>,
    atomics: &mut AtomicFaults<P>,
    num_detectors: usize,
    multi_pb: Option<&MultiProgress>,
) -> FxHashMap<BigUint, P> {
    let detector_mask = ones_mask(num_detectors);

    let mut current_queue: BTreeMap<Fault, P> = match queue.remove(&w) {
        Some(q) if !q.is_empty() => q,
        _ => return FxHashMap::default(),
    };
    let mut undetectables_generated: FxHashMap<BigUint, P> = FxHashMap::default();

    let pb = multi_pb.map(|m| {
        let pb = ProgressBar::new(current_queue.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos} remaining",
            )
            .unwrap()
            .progress_chars("#<-"),
        );
        pb.set_position(current_queue.len() as u64);
        pb.enable_steady_tick(Duration::from_millis(50));
        m.add(pb.clone());
        pb
    });
    let (mut items_done, start_time) = (0, SystemTime::now());
    while !current_queue.is_empty() {
        let mut new_w_queue: BTreeMap<Fault, P> = BTreeMap::new();
        let items = current_queue.len();
        items_done += items;

        for (i, (sig, sig_prov)) in current_queue.into_iter().enumerate() {
            let detectable = has_any_bit(&sig, &detector_mask);
            if i % PB_UPDATE_INTERVAL == 0
                && let Some(pb) = &pb
            {
                pb.dec(PB_UPDATE_INTERVAL as u64);
            }

            if detectable {
                match detectable_lookup.get(&sig) {
                    Some(&existing_w) if existing_w <= w => continue, // No improvement
                    _ => {
                        detectable_lookup.insert(sig.clone(), w);
                    }
                }
                atomics.check_update_detectable_weight(&sig, w);
            } else {
                let sig_no_sinks = &sig >> num_detectors;
                match undetectable_lookup.get(&sig_no_sinks) {
                    Some(&existing_w) if existing_w <= w => continue, // No improvement
                    _ => {
                        undetectable_lookup.insert(sig_no_sinks.clone(), w);
                        undetectables_generated
                            .entry(sig_no_sinks)
                            .or_insert_with(|| sig_prov.clone());
                    }
                }
                atomics.check_update_undetectable_weight(&sig, w)
            }

            let atomic_entries: Vec<(&Fault, Weight, &P)> = if !detectable {
                atomics.undetectable_iter().collect()
            } else {
                atomics.detector_overlapping(&apply_mask(&sig, &detector_mask))
            };
            for (atomic_sig, atomic_w, atomic_prov) in atomic_entries {
                let combined = atomic_sig ^ &sig;
                let combined_prov = sig_prov.combine(atomic_prov);
                if atomic_w == 0 {
                    new_w_queue.entry(combined).or_insert(combined_prov);
                    if let Some(pb) = &pb {
                        pb.inc(1);
                    }
                } else {
                    queue
                        .entry(atomic_w + w)
                        .or_default()
                        .entry(combined)
                        .or_insert(combined_prov);
                }
            }
        }
        if let Some(pb) = &pb {
            pb.dec((items % PB_UPDATE_INTERVAL) as u64);
        }
        current_queue = new_w_queue;
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    if let Some(multi_pb) = multi_pb {
        let total_time = SystemTime::now().duration_since(start_time).unwrap();
        multi_pb
            .println(format!(
                "|   w={w} iteration averaged {:.2}k iterations per second ...",
                (items_done as f64 / total_time.as_secs_f64()) / 1000f64
            ))
            .unwrap();
    }

    undetectables_generated
}

#[derive(Clone, Debug)]
/// A violation of fault equivalence
pub struct Violation {
    /// Whether the violation originates from the first given noise model or the second
    pub is_nm1: bool,
    /// The combined weight of the violation
    pub weight: usize,
    /// The indices of the faults that compose to the violating fault (only populated when
    /// provenance tracking is enabled)
    pub faults: Option<Vec<usize>>,
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
    provenance: bool,
    quiet: bool,
) -> Option<Violation> {
    let check_impl = if provenance {
        check_fault_equivalence_impl::<WithProv>
    } else {
        check_fault_equivalence_impl::<NoProv>
    };

    check_impl(nm1_sigs, nm2_sigs, d1_detectors, d2_detectors, until, quiet)
}

fn check_fault_equivalence_impl<P: Provenance>(
    nm1_sigs: Vec<(Fault, Weight)>,
    nm2_sigs: Vec<(Fault, Weight)>,
    d1_detectors: usize,
    d2_detectors: usize,
    until: Option<Weight>,
    quiet: bool,
) -> Option<Violation> {
    let mut nm1_detectable_lookup = FxHashMap::default();
    let mut nm1_undetectable_lookup = FxHashMap::default();
    nm1_undetectable_lookup.insert(Fault::ZERO, 0);

    let mut nm2_detectable_lookup = FxHashMap::default();
    let mut nm2_undetectable_lookup = FxHashMap::default();
    nm2_undetectable_lookup.insert(Fault::ZERO, 0);

    let mut nm1_atomics = AtomicFaults::<P>::from_faults(nm1_sigs, d1_detectors);
    let mut nm1_pq = prepare_priority_queue(&nm1_atomics);

    let mut nm2_atomics = AtomicFaults::<P>::from_faults(nm2_sigs, d2_detectors);
    let mut nm2_pq = prepare_priority_queue(&nm2_atomics);

    let mut w: Weight = 0;

    let multi_pb = MultiProgress::new();
    multi_pb.set_draw_target(ProgressDrawTarget::stdout());
    let pb = if quiet {
        ProgressBar::hidden()
    } else if let Some(until) = until {
        ProgressBar::new(until as u64)
    } else {
        ProgressBar::no_length()
    };
    pb.set_style(ProgressStyle::with_template("{spinner} {msg}: {pos}").unwrap());
    pb.set_message("Current weight");
    pb.enable_steady_tick(Duration::from_millis(100));
    multi_pb.add(pb.clone());
    while (!nm1_pq.is_empty() || !nm2_pq.is_empty()) && until.is_none_or(|u| w < u - 1) {
        w += 1;
        pb.inc(1);

        let nm1_undetectable = next_gen_unfold(
            w,
            &mut nm1_pq,
            &mut nm1_detectable_lookup,
            &mut nm1_undetectable_lookup,
            &mut nm1_atomics,
            d1_detectors,
            (!quiet).then_some(&multi_pb),
        );
        if !quiet {
            multi_pb
                .println(format!(
                    "|   Finished unfolding weight {w} in queue 1! Next items remaining: {}...",
                    nm1_pq.get(&(w + 1)).map(|s| s.len()).unwrap_or(0)
                ))
                .unwrap();
        }

        let nm2_undetectable = next_gen_unfold(
            w,
            &mut nm2_pq,
            &mut nm2_detectable_lookup,
            &mut nm2_undetectable_lookup,
            &mut nm2_atomics,
            d2_detectors,
            (!quiet).then_some(&multi_pb),
        );
        if !quiet {
            multi_pb
                .println(format!(
                    "|   Finished unfolding weight {w} in queue 2! Next items remaining: {}...",
                    nm2_pq.get(&(w + 1)).map(|s| s.len()).unwrap_or(0)
                ))
                .unwrap();
        }

        for (sig, prov) in &nm1_undetectable {
            if !nm2_undetectable_lookup.contains_key(sig) {
                pb.finish_and_clear();
                return Some(Violation {
                    is_nm1: true,
                    weight: w,
                    faults: prov.to_fault_indices(),
                });
            }
        }
        if !quiet {
            multi_pb.println("|   Finished checking new undetectable faults from noise model 1 against noise model 2!").unwrap();
        }

        for (sig, prov) in &nm2_undetectable {
            if !nm1_undetectable_lookup.contains_key(sig) {
                pb.finish_and_clear();
                return Some(Violation {
                    is_nm1: false,
                    weight: w,
                    faults: prov.to_fault_indices(),
                });
            }
        }
        if !quiet {
            multi_pb.println("|   Finished checking new undetectable faults from noise model 2 against noise model 1!").unwrap();
            multi_pb
                .println(format!("Finished checking weight {w}!"))
                .unwrap();
        }
    }
    pb.finish_and_clear();

    None
}
