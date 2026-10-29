//! Conversation segments (v0.0.18, S1): where a conversation changes direction.
//!
//! A conversation is not one block. It is a chain of stretches, each with its own
//! direction, and the direction can change along several independent axes at once:
//! the harness that serves it, the subject, the problem being worked, the way it is
//! being solved, how well it is understood. This module is the PURE MODEL of that:
//! it takes one [`Observation`] per turn and says where a [`Segment`] ends and the
//! next one starts, on each [`Dimension`] separately.
//!
//! Nothing here reads a message, calls a model or touches the network. What a turn
//! looks like on each axis is a [`Fingerprint`] (a small set of labels) that a later
//! slice (S2) extracts from the stored events; this slice only decides, from those
//! fingerprints, when a direction has really changed. Keeping the two apart is what
//! makes the decision testable, and replayable.
//!
//! * **Hysteresis, not a one-turn latch.** A single off-topic turn is not a change of
//!   direction. A boundary needs [`Threshold::confirmations`] consecutive observations
//!   that all sit at least [`Threshold::distance_permille`] away from the current
//!   segment, and it is then placed at the FIRST of them (the segment really changed
//!   there, not when it was confirmed). One observation back within the threshold
//!   cancels the run. The harness axis confirms at once: a switch is explicit.
//! * **No evidence is not a change.** An empty fingerprint means the turn said nothing
//!   on that axis. It neither opens a segment, nor deviates, nor confirms.
//! * **Replayable.** Integer arithmetic only (distances are permille), ordered maps,
//!   and segment ids derived by a fixed hash of `(seed, dimension, start turn)`: the
//!   same seed and the same observations give the same segments and boundaries, byte
//!   for byte, on any machine. Labels are expected to be ASCII (or near-ASCII):
//!   `to_lowercase` follows the toolchain's Unicode tables, so exotic case folding
//!   is not guaranteed bit-identical across toolchains. The caller derives the seed
//!   from the conversation or decision id (constraint: every automatic decision is
//!   attributable and replayable).
//! * **Atomic.** An observation that is refused (turn not strictly increasing) leaves
//!   the state exactly as it was.
use crate::neo4j::traits::*;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Largest distance: nothing in common.
pub const PERMILLE_MAX: u16 = 1_000;
/// Most labels a segment keeps in its fingerprint. A segment that keeps absorbing never
/// grows without bound; labels beyond the cap are not added (the first ones win, in the
/// fingerprint's sorted order, so the result does not depend on arrival order).
pub const MAX_FINGERPRINT_LABELS: usize = 64;

/// An axis along which a conversation can change direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dimension {
    /// The provider/engine serving the conversation.
    Harness,
    /// What the conversation is about.
    Topic,
    /// The problem being worked.
    Problem,
    /// The way that problem is being solved.
    Resolution,
    /// How well the situation is understood.
    Understanding,
}

impl Dimension {
    /// Every dimension, in the fixed order observations are processed in.
    pub const ALL: [Self; 5] = [
        Self::Harness,
        Self::Topic,
        Self::Problem,
        Self::Resolution,
        Self::Understanding,
    ];

    /// Stable name (feeds the segment id: never rename).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Harness => "harness",
            Self::Topic => "topic",
            Self::Problem => "problem",
            Self::Resolution => "resolution",
            Self::Understanding => "understanding",
        }
    }
}

/// What a turn looks like on one axis: a set of normalised labels.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "Vec<String>")]
pub struct Fingerprint(BTreeSet<String>);

impl TryFrom<Vec<String>> for Fingerprint {
    // Normalisation never fails, but `#[serde(try_from)]` needs a `TryFrom`;
    // the error type is nominal and never constructed.
    type Error = String;

    /// Normalises on deserialization, exactly like [`Fingerprint::new`]: a
    /// stored fingerprint that is not trimmed/lower-cased/​capped is brought
    /// back to the canonical form instead of bypassing it.
    fn try_from(labels: Vec<String>) -> Result<Self, Self::Error> {
        Ok(Self::new(labels))
    }
}

impl Fingerprint {
    /// Builds a fingerprint. Labels are trimmed and lower-cased; empty ones are dropped.
    pub fn new<I, S>(labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut fingerprint = Self(
            labels
                .into_iter()
                .map(|label| label.as_ref().trim().to_lowercase())
                .filter(|label| !label.is_empty())
                .collect(),
        );
        fingerprint.truncate_to_cap();
        fingerprint
    }

    /// Whether the turn said nothing on this axis.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The labels, in sorted order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }

    /// Distance of `other` (a turn) from `self` (the accumulated segment), in
    /// permille: the share of the turn's labels the segment does not already know.
    /// `0` = every label already known (no new direction), `1000` = nothing in
    /// common. The share is rounded UP, so a borderline turn counts as farther.
    ///
    /// This is asymmetric on purpose: `self` is the segment (an accumulated union,
    /// up to [`MAX_FINGERPRINT_LABELS`] labels) and `other` is a single turn. A
    /// symmetric measure such as Jaccard would penalise a turn that is a strict
    /// subset of the segment for the segment's size, and over-segment once the
    /// segment has absorbed enough labels. An empty turn is no evidence (distance 0).
    pub fn distance_permille(&self, other: &Self) -> u16 {
        if other.0.is_empty() {
            return 0;
        }
        let new = other.0.difference(&self.0).count();
        let novelty = new * usize::from(PERMILLE_MAX);
        u16::try_from(novelty.div_ceil(other.0.len())).unwrap_or(PERMILLE_MAX)
    }

    /// Adds `other`'s labels, then keeps the lexicographically-first
    /// [`MAX_FINGERPRINT_LABELS`]: the result depends only on the set of labels seen,
    /// never on the order they arrived.
    fn merge(&mut self, other: &Self) {
        for label in &other.0 {
            self.0.insert(label.clone());
        }
        self.truncate_to_cap();
    }

    /// Keeps the lexicographically-first [`MAX_FINGERPRINT_LABELS`] labels.
    fn truncate_to_cap(&mut self) {
        while self.0.len() > MAX_FINGERPRINT_LABELS {
            let Some(last) = self.0.iter().next_back().cloned() else {
                break;
            };
            self.0.remove(&last);
        }
    }
}

/// One turn, seen on every axis it has something to say about.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Observation {
    /// Index of the turn in the conversation; strictly increasing across observations.
    ///
    /// The counter is per-conversation as the CALLER sees it, not a global process
    /// counter. The backend numbers turns per session and a provider move opens a new
    /// session (and a new `Segmenter`), so the caller feeds one segmenter per session
    /// and the counter never resets inside it; a relay that would otherwise send a
    /// non-increasing turn is refused by the segmenter (and tolerated as a no-op by
    /// the wiring, which logs it and continues).
    pub turn: u32,
    /// What the turn looks like per axis. A missing or empty entry is "no evidence".
    pub prints: BTreeMap<Dimension, Fingerprint>,
}

/// How far, and for how long, a direction must move before it counts as changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawThreshold")]
pub struct Threshold {
    /// Distance, in permille, from which an observation deviates (1..=1000).
    pub distance_permille: u16,
    /// Consecutive deviating observations that confirm a boundary (at least 1).
    pub confirmations: u8,
}

/// The wire form of a [`Threshold`], validated on deserialization: a document
/// that says `distance_permille: 0` or `confirmations: 0` is refused instead of
/// silently changing how the segmenter cuts.
#[derive(Deserialize)]
struct RawThreshold {
    distance_permille: u16,
    confirmations: u8,
}

impl TryFrom<RawThreshold> for Threshold {
    type Error = String;

    fn try_from(raw: RawThreshold) -> Result<Self, Self::Error> {
        Self::new(raw.distance_permille, raw.confirmations).map_err(|error| error.to_string())
    }
}

impl Threshold {
    /// A checked threshold.
    pub fn new(distance_permille: u16, confirmations: u8) -> Result<Self, SegmentError> {
        if distance_permille == 0 || distance_permille > PERMILLE_MAX {
            return Err(SegmentError::DistanceOutOfRange(distance_permille));
        }
        if confirmations == 0 {
            return Err(SegmentError::NoConfirmation);
        }
        Ok(Self {
            distance_permille,
            confirmations,
        })
    }
}

/// The thresholds of every axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentPolicy {
    thresholds: BTreeMap<Dimension, Threshold>,
}

impl Default for SegmentPolicy {
    /// The harness changes at once (a switch is explicit); the other axes need two
    /// consecutive turns that sit clearly away from the segment.
    fn default() -> Self {
        let mut thresholds = BTreeMap::new();
        for (dimension, distance, confirmations) in [
            (Dimension::Harness, PERMILLE_MAX, 1),
            (Dimension::Topic, 700, 2),
            (Dimension::Problem, 600, 2),
            (Dimension::Resolution, 600, 2),
            (Dimension::Understanding, 600, 2),
        ] {
            thresholds.insert(
                dimension,
                Threshold {
                    distance_permille: distance,
                    confirmations,
                },
            );
        }
        Self { thresholds }
    }
}

impl SegmentPolicy {
    /// This policy with `dimension` set to `threshold`.
    pub fn with(mut self, dimension: Dimension, threshold: Threshold) -> Self {
        self.thresholds.insert(dimension, threshold);
        self
    }

    /// The threshold of `dimension`.
    pub fn threshold(&self, dimension: Dimension) -> Threshold {
        // The default fills every axis and `with` only replaces: an axis is never absent.
        // The fallback is the strictest threshold, so a hole could only ever under-split.
        self.thresholds
            .get(&dimension)
            .copied()
            .unwrap_or(Threshold {
                distance_permille: PERMILLE_MAX,
                confirmations: u8::MAX,
            })
    }
}

/// Why a boundary was placed.
///
/// An enum, not a bag of numbers: the JSON is stable to grow (a later slice can
/// add a variant without re-reading old documents as a different shape), and a
/// reader can tell a drift from a switch from a compaction at a glance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundaryReason {
    /// The direction drifted: `confirmations` consecutive observations sat at
    /// least `distance_permille` from the segment (at or beyond `threshold_permille`).
    Drift {
        /// Distance of the first deviating observation from the segment it left, in permille.
        distance_permille: u16,
        /// The threshold that was crossed.
        threshold_permille: u16,
        /// Consecutive deviating observations that confirmed it.
        confirmations: u8,
    },
    /// The harness switched (a provider/model change is explicit). Reserved: the
    /// pure segmenter reports a harness change as a [`BoundaryReason::Drift`] at
    /// distance 1000; a later slice that knows the actual `from`/`to` harness can
    /// fill this in.
    Switch {
        /// The harness the conversation left.
        from: String,
        /// The harness it moved to.
        to: String,
    },
    /// The conversation was compacted. Reserved: the pure segmenter does not see
    /// compactions; the wiring slice can open a boundary on one.
    Compaction,
}

/// Stable identifier of a segment: a fixed hash, not a random id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SegmentId(String);

impl SegmentId {
    fn derive(seed: u64, dimension: Dimension, start_turn: u32) -> Self {
        // FNV-1a, 64 bits: fixed by definition, unlike `DefaultHasher`, whose algorithm
        // the standard library does not promise to keep.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            // A separator, so ("ab","c") and ("a","bc") do not collide.
            hash ^= 0xff;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        feed(&seed.to_be_bytes());
        feed(dimension.as_str().as_bytes());
        feed(&start_turn.to_be_bytes());
        Self(format!("seg-{hash:016x}"))
    }

    /// The identifier as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A stretch of the conversation that keeps one direction on one axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    /// Derived from the seed, the axis and the first turn.
    pub id: SegmentId,
    /// The axis.
    pub dimension: Dimension,
    /// First turn of the segment.
    pub start_turn: u32,
    /// Last turn observed in it.
    pub last_turn: u32,
    /// How many observations it absorbed.
    pub observations: u32,
    /// Whether a later segment replaced it.
    pub closed: bool,
    /// What the segment looks like: the labels of the turns it absorbed (capped).
    pub fingerprint: Fingerprint,
    /// Why it started; `None` for the first segment of an axis.
    pub opened_by: Option<BoundaryReason>,
}

impl Segment {
    fn open(
        seed: u64,
        dimension: Dimension,
        start_turn: u32,
        fingerprint: Fingerprint,
        opened_by: Option<BoundaryReason>,
    ) -> Self {
        Self {
            id: SegmentId::derive(seed, dimension, start_turn),
            dimension,
            start_turn,
            last_turn: start_turn,
            observations: 1,
            closed: false,
            fingerprint,
            opened_by,
        }
    }

    fn absorb(&mut self, turn: u32, print: &Fingerprint) {
        self.last_turn = turn;
        self.observations = self.observations.saturating_add(1);
        self.fingerprint.merge(print);
    }
}

/// A change of direction on one axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Boundary {
    /// The axis.
    pub dimension: Dimension,
    /// First turn of the new segment: where the direction really changed.
    pub at_turn: u32,
    /// The segment that ended.
    pub from: SegmentId,
    /// The segment that started.
    pub to: SegmentId,
    /// Why.
    pub reason: BoundaryReason,
}

/// A conversation's segmentation as it was persisted ([`Segmenter::segments`] plus
/// [`Segmenter::boundaries`]). The wiring snapshots it on a confirmed boundary; a
/// conversation reopened later reads it back with [`Segmenter::resume`], so its
/// segmentation survives a restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentsSnapshot {
    /// Every segment of the conversation, closed and open.
    pub segments: Vec<Segment>,
    /// Every boundary, in the order it was confirmed.
    pub boundaries: Vec<Boundary>,
}

impl SegmentsSnapshot {
    /// Nothing to resume.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.boundaries.is_empty()
    }
}

/// A refused input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SegmentError {
    /// The turn does not come after the last one observed.
    #[error("turn {turn} does not follow turn {last}: observations must be strictly increasing")]
    OutOfOrder {
        /// The refused turn.
        turn: u32,
        /// The last turn observed.
        last: u32,
    },
    /// A threshold distance outside 1..=1000.
    #[error("a distance threshold is between 1 and 1000 permille, got {0}")]
    DistanceOutOfRange(u16),
    /// A threshold that no observation can confirm.
    #[error("a boundary needs at least one confirmation")]
    NoConfirmation,
    /// A snapshot that is not a segmentation this model can produce.
    #[error("the {dimension:?} axis carries {count} open segments: a conversation has one open segment per axis")]
    IncoherentSnapshot {
        /// The axis that carries more than one open segment.
        dimension: Dimension,
        /// How many it carries.
        count: usize,
    },
}

/// An observation that deviates and is waiting to be confirmed.
#[derive(Debug, Clone)]
struct Pending {
    turn: u32,
    print: Fingerprint,
    distance: u16,
}

#[derive(Debug, Default)]
struct Track {
    current: Option<Segment>,
    pending: Vec<Pending>,
}

/// Cuts a conversation into segments, one observation at a time.
#[derive(Debug)]
pub struct Segmenter {
    policy: SegmentPolicy,
    seed: u64,
    tracks: BTreeMap<Dimension, Track>,
    closed: Vec<Segment>,
    boundaries: Vec<Boundary>,
    last_turn: Option<u32>,
}

impl Segmenter {
    /// An empty segmenter. `seed` is derived by the caller from the conversation or
    /// decision id; it only feeds the segment ids.
    pub fn new(policy: SegmentPolicy, seed: u64) -> Self {
        Self {
            policy,
            seed,
            tracks: BTreeMap::new(),
            closed: Vec::new(),
            boundaries: Vec::new(),
            last_turn: None,
        }
    }

    /// Replays a recorded conversation from scratch.
    pub fn replay(
        policy: SegmentPolicy,
        seed: u64,
        observations: &[Observation],
    ) -> Result<Self, SegmentError> {
        let mut segmenter = Self::new(policy, seed);
        for observation in observations {
            segmenter.observe(observation)?;
        }
        Ok(segmenter)
    }

    /// Resumes a segmentation from a persisted [`SegmentsSnapshot`], instead of replaying
    /// the observations: the open segment of each axis comes back with the labels it had
    /// absorbed, so the next deviating turn is measured against what the conversation
    /// already knew — not against nothing, which would open a spurious boundary.
    ///
    /// The snapshot is written on a confirmed boundary, so the axis that confirmed has no
    /// pending deviation; a run pending on ANOTHER axis is not part of the snapshot, and
    /// resuming can confirm that axis one turn later than an uninterrupted run would
    /// (`a_pending_run_of_another_axis_is_not_restored`).
    ///
    /// Refused when one axis carries more than one open segment: the model never produces
    /// that, so such a snapshot is corrupt rather than a segmentation.
    pub fn resume(
        policy: SegmentPolicy,
        seed: u64,
        snapshot: &SegmentsSnapshot,
    ) -> Result<Self, SegmentError> {
        let mut open_per_axis: BTreeMap<Dimension, usize> = BTreeMap::new();
        for segment in &snapshot.segments {
            if !segment.closed {
                *open_per_axis.entry(segment.dimension).or_default() += 1;
            }
        }
        if let Some((&dimension, &count)) = open_per_axis.iter().find(|(_, count)| **count > 1) {
            return Err(SegmentError::IncoherentSnapshot { dimension, count });
        }

        let mut segmenter = Self::new(policy, seed);
        for segment in &snapshot.segments {
            if segment.closed {
                segmenter.closed.push(segment.clone());
            } else {
                segmenter
                    .tracks
                    .entry(segment.dimension)
                    .or_default()
                    .current = Some(segment.clone());
            }
        }
        // Closed segments come back in the order `segments()` publishes (axis, then first
        // turn): a reader sorts or scans by axis, so the confirmation order is not needed.
        segmenter
            .closed
            .sort_by(|a, b| (a.dimension, a.start_turn).cmp(&(b.dimension, b.start_turn)));
        segmenter.boundaries = snapshot.boundaries.clone();
        segmenter.last_turn = {
            let segments = segmenter.segments();
            segments
                .iter()
                .map(|segment| segment.last_turn)
                .chain(segmenter.boundaries.iter().map(|boundary| boundary.at_turn))
                .max()
        };
        Ok(segmenter)
    }

    /// Takes one turn in. Returns the boundaries it confirmed (at most one per axis).
    /// A refused turn changes nothing.
    pub fn observe(&mut self, observation: &Observation) -> Result<Vec<Boundary>, SegmentError> {
        if let Some(last) = self.last_turn {
            if observation.turn <= last {
                return Err(SegmentError::OutOfOrder {
                    turn: observation.turn,
                    last,
                });
            }
        }
        self.last_turn = Some(observation.turn);

        let mut confirmed = Vec::new();
        for dimension in Dimension::ALL {
            let Some(print) = observation.prints.get(&dimension) else {
                continue;
            };
            if print.is_empty() {
                continue;
            }
            if let Some(boundary) = self.observe_axis(dimension, observation.turn, print) {
                self.boundaries.push(boundary.clone());
                confirmed.push(boundary);
            }
        }
        Ok(confirmed)
    }

    fn observe_axis(
        &mut self,
        dimension: Dimension,
        turn: u32,
        print: &Fingerprint,
    ) -> Option<Boundary> {
        let threshold = self.policy.threshold(dimension);
        let seed = self.seed;
        let track = self.tracks.entry(dimension).or_default();

        let Some(mut current) = track.current.take() else {
            track.current = Some(Segment::open(seed, dimension, turn, print.clone(), None));
            return None;
        };

        let distance = if dimension == Dimension::Harness {
            // A harness switch is explicit: equality, not a distance, so a change of
            // provider (or model, or a transient two-harness turn) always deviates and
            // a segment can never get stuck by absorbing two harnesses.
            if current.fingerprint == *print {
                0
            } else {
                PERMILLE_MAX
            }
        } else {
            current.fingerprint.distance_permille(print)
        };
        if distance < threshold.distance_permille {
            // Back inside the segment: any run of deviations was a detour, not a change.
            track.pending.clear();
            current.absorb(turn, print);
            track.current = Some(current);
            return None;
        }

        // A deviating turn must head in the SAME new direction as the run already
        // pending: two unrelated off-topic turns are not one change, they restart it.
        if let Some(first) = track.pending.first() {
            let consistent = if dimension == Dimension::Harness {
                first.print == *print
            } else {
                first.print.distance_permille(print) < threshold.distance_permille
            };
            if !consistent {
                track.pending.clear();
            }
        }
        track.pending.push(Pending {
            turn,
            print: print.clone(),
            distance,
        });
        if track.pending.len() < usize::from(threshold.confirmations) {
            track.current = Some(current);
            return None;
        }

        // Confirmed: the new segment starts at the FIRST deviating turn.
        let pending = std::mem::take(&mut track.pending);
        let first = pending.first()?;
        let last = pending.last()?;
        let reason = BoundaryReason::Drift {
            distance_permille: first.distance,
            threshold_permille: threshold.distance_permille,
            confirmations: threshold.confirmations,
        };
        let mut fingerprint = Fingerprint::default();
        for entry in &pending {
            fingerprint.merge(&entry.print);
        }
        let mut next = Segment::open(
            seed,
            dimension,
            first.turn,
            fingerprint,
            Some(reason.clone()),
        );
        next.last_turn = last.turn;
        next.observations = u32::try_from(pending.len()).unwrap_or(u32::MAX);

        current.closed = true;
        let boundary = Boundary {
            dimension,
            at_turn: first.turn,
            from: current.id.clone(),
            to: next.id.clone(),
            reason,
        };
        self.closed.push(current);
        track.current = Some(next);
        Some(boundary)
    }

    /// The open segment of an axis, if the axis has been observed.
    pub fn current(&self, dimension: Dimension) -> Option<&Segment> {
        self.tracks.get(&dimension)?.current.as_ref()
    }

    /// Every segment, closed and open, ordered by axis then first turn.
    pub fn segments(&self) -> Vec<Segment> {
        let mut all: Vec<Segment> = self.closed.clone();
        all.extend(self.tracks.values().filter_map(|t| t.current.clone()));
        all.sort_by(|a, b| (a.dimension, a.start_turn).cmp(&(b.dimension, b.start_turn)));
        all
    }

    /// The segment `turn` belongs to on `dimension`: the latest one that started at or
    /// before it. A turn that deviates but is not yet confirmed still belongs to the
    /// current segment; it is moved when (and if) the boundary is confirmed.
    ///
    /// Segments are scanned in place (closed are already in confirmation order, the
    /// open one is the latest), so this is O(number of segments) with no clone-and-sort,
    /// not O(n²) per turn like the old `segments().rfind(..)`.
    pub fn segment_at(&self, dimension: Dimension, turn: u32) -> Option<Segment> {
        let open = self.tracks.get(&dimension).and_then(|t| t.current.as_ref());
        self.closed
            .iter()
            .chain(open)
            .filter(|s| s.dimension == dimension && s.start_turn <= turn)
            .max_by_key(|s| s.start_turn)
            .cloned()
    }

    /// Every boundary confirmed so far, in the order they were confirmed.
    pub fn boundaries(&self) -> &[Boundary] {
        &self.boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    fn fp(labels: &[&str]) -> Fingerprint {
        Fingerprint::new(labels.iter().copied())
    }

    fn obs(turn: u32, prints: &[(Dimension, &[&str])]) -> Observation {
        Observation {
            turn,
            prints: prints.iter().map(|(d, l)| (*d, fp(l))).collect(),
        }
    }

    fn topic(turn: u32, labels: &[&str]) -> Observation {
        obs(turn, &[(Dimension::Topic, labels)])
    }

    fn harness(turn: u32, id: &str) -> Observation {
        obs(turn, &[(Dimension::Harness, &[id])])
    }

    fn run(observations: &[Observation]) -> Segmenter {
        Segmenter::replay(SegmentPolicy::default(), 7, observations).unwrap()
    }

    fn snapshot_of(segmenter: &Segmenter) -> SegmentsSnapshot {
        SegmentsSnapshot {
            segments: segmenter.segments(),
            boundaries: segmenter.boundaries().to_vec(),
        }
    }

    /// S4: a conversation cut mid-way and resumed segments its FOLLOWING turns exactly as
    /// an uninterrupted run does — same segments, same boundaries. The tail is also run on
    /// a fresh segmenter, which differs: the snapshot really is used.
    #[test]
    fn resuming_from_a_snapshot_segments_the_following_turns_identically() {
        let history = vec![
            topic(0, &["routing"]),
            topic(1, &["routing"]),
            topic(2, &["docker"]),
            topic(3, &["docker"]), // confirms the topic boundary, placed at turn 2
            topic(4, &["docker"]),
            topic(5, &["neo4j"]),
            topic(6, &["neo4j"]), // confirms the second boundary, placed at turn 5
        ];
        let whole = run(&history);
        assert_eq!(whole.boundaries().len(), 2);

        let cut = 4;
        let first_leg = run(&history[..cut]);
        let snapshot = snapshot_of(&first_leg);
        let mut resumed = Segmenter::resume(SegmentPolicy::default(), 7, &snapshot).unwrap();
        for observation in &history[cut..] {
            resumed.observe(observation).unwrap();
        }
        assert_eq!(resumed.segments(), whole.segments());
        assert_eq!(resumed.boundaries(), whole.boundaries());

        // The same tail on a FRESH segmenter is not the same conversation: without the
        // snapshot the segmentation restarts, so this test is red if `resume` ignores it.
        let mut fresh = Segmenter::new(SegmentPolicy::default(), 7);
        for observation in &history[cut..] {
            fresh.observe(observation).unwrap();
        }
        assert_ne!(fresh.segments(), whole.segments());
    }

    /// An empty snapshot behaves as a fresh segmenter.
    #[test]
    fn resuming_an_empty_snapshot_is_a_fresh_segmenter() {
        let snapshot = SegmentsSnapshot {
            segments: Vec::new(),
            boundaries: Vec::new(),
        };
        assert!(snapshot.is_empty());
        let mut resumed = Segmenter::resume(SegmentPolicy::default(), 7, &snapshot).unwrap();
        assert!(resumed.segments().is_empty());
        assert!(resumed.boundaries().is_empty());
        resumed.observe(&topic(0, &["routing"])).unwrap();
        assert_eq!(resumed.segments().len(), 1);
    }

    /// The open segment comes back with the labels it had absorbed: the next turns of the
    /// same subject are not mistaken for a new direction.
    #[test]
    fn the_open_segment_comes_back_with_the_labels_it_had_absorbed() {
        let first = run(&[topic(0, &["routing"]), topic(1, &["routing"])]);
        let mut resumed =
            Segmenter::resume(SegmentPolicy::default(), 7, &snapshot_of(&first)).unwrap();
        assert_eq!(
            resumed.current(Dimension::Topic).map(|s| s.observations),
            Some(2),
            "the absorbed turns are restored"
        );
        assert!(resumed.observe(&topic(2, &["routing"])).unwrap().is_empty());
        assert!(resumed.observe(&topic(3, &["routing"])).unwrap().is_empty());
        assert_eq!(resumed.boundaries().len(), 0);
    }

    /// A snapshot is what the model produced, so an axis with two open segments is refused
    /// rather than silently half-restored.
    #[test]
    fn two_open_segments_on_one_axis_are_refused() {
        let first = run(&[topic(0, &["routing"])]);
        let mut segments = first.segments();
        let mut duplicate = segments[0].clone();
        duplicate.start_turn = 99;
        segments.push(duplicate);
        let snapshot = SegmentsSnapshot {
            segments,
            boundaries: Vec::new(),
        };
        assert!(matches!(
            Segmenter::resume(SegmentPolicy::default(), 7, &snapshot),
            Err(SegmentError::IncoherentSnapshot {
                dimension: Dimension::Topic,
                count: 2
            })
        ));
    }

    /// S4, documented limit. The snapshot is taken ON a boundary, so the confirming axis
    /// carries no pending deviation — but another axis may, and a pending run is not
    /// persisted. Resuming then needs one more deviating turn to confirm that axis. This
    /// test PINS the difference: it is a known property, not a surprise.
    #[test]
    fn a_pending_run_of_another_axis_is_not_restored() {
        let head = vec![
            obs(
                0,
                &[
                    (Dimension::Harness, &["a"]),
                    (Dimension::Topic, &["routing"]),
                ],
            ),
            obs(
                1,
                &[
                    (Dimension::Harness, &["a"]),
                    (Dimension::Topic, &["routing"]),
                ],
            ),
            // Harness switches: confirmed at once (one boundary, snapshot taken here).
            // Topic drifts on the SAME turn: one deviating turn, still pending.
            obs(
                2,
                &[
                    (Dimension::Harness, &["b"]),
                    (Dimension::Topic, &["docker"]),
                ],
            ),
        ];
        let next = obs(
            3,
            &[
                (Dimension::Harness, &["b"]),
                (Dimension::Topic, &["docker"]),
            ],
        );

        let mut uninterrupted = run(&head);
        assert_eq!(uninterrupted.boundaries().len(), 1);
        assert_eq!(uninterrupted.boundaries()[0].dimension, Dimension::Harness);
        uninterrupted.observe(&next).unwrap();
        assert_eq!(
            uninterrupted.boundaries().len(),
            2,
            "uninterrupted: the topic run confirms on its second turn"
        );

        let mut resumed =
            Segmenter::resume(SegmentPolicy::default(), 7, &snapshot_of(&run(&head))).unwrap();
        assert!(
            resumed.observe(&next).unwrap().is_empty(),
            "resumed: the discarded pending run restarts, so the boundary comes one turn later"
        );
        assert_eq!(resumed.boundaries().len(), 1);
    }

    #[test]
    fn fingerprint_normalizes_and_ignores_empty_labels() {
        let print = Fingerprint::new(["  Relay ", "relay", "", "   ", "Routing"]);
        assert_eq!(print.labels().collect::<Vec<_>>(), ["relay", "routing"]);
        assert!(Fingerprint::new(["", " "]).is_empty());
    }

    #[test]
    fn distance_is_the_share_of_new_labels_in_permille() {
        // `self` is the accumulated segment, `other` the turn being placed; the
        // distance is the share of the turn's labels the segment does not know,
        // rounded up (a borderline turn counts as farther).
        assert_eq!(fp(&["a", "b"]).distance_permille(&fp(&["a", "b"])), 0);
        assert_eq!(fp(&["a"]).distance_permille(&fp(&["b"])), 1_000);
        // One new label out of two: 500 — not symmetric Jaccard's 667.
        assert_eq!(fp(&["a", "b"]).distance_permille(&fp(&["b", "c"])), 500);
        // A turn fully contained in the segment adds no new direction.
        assert_eq!(fp(&["a", "b", "c"]).distance_permille(&fp(&["a", "b"])), 0);
        // A turn that is a superset is measured by how much it adds: 2 of 4 new.
        assert_eq!(
            fp(&["a", "b"]).distance_permille(&fp(&["a", "b", "c", "d"])),
            500
        );
        // Rounded up: 2 new labels out of 3 -> 667.
        assert_eq!(fp(&["a"]).distance_permille(&fp(&["a", "b", "c"])), 667);
        // Two empty sets say nothing: not a difference.
        assert_eq!(fp(&[]).distance_permille(&fp(&[])), 0);
        // An empty segment cannot know any turn: fully new.
        assert_eq!(fp(&[]).distance_permille(&fp(&["a"])), 1_000);
    }

    #[test]
    fn an_empty_observation_is_no_evidence() {
        let segmenter = run(&[topic(0, &[]), topic(1, &[]), obs(2, &[])]);
        assert!(segmenter.segments().is_empty());
        assert!(segmenter.boundaries().is_empty());
    }

    #[test]
    fn a_harness_change_is_a_boundary_at_once() {
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        assert!(segmenter
            .observe(&harness(0, "deepseek"))
            .unwrap()
            .is_empty());
        assert!(segmenter
            .observe(&harness(1, "deepseek"))
            .unwrap()
            .is_empty());
        let boundaries = segmenter.observe(&harness(2, "claude-code")).unwrap();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].dimension, Dimension::Harness);
        assert_eq!(boundaries[0].at_turn, 2);
        let BoundaryReason::Drift { confirmations, .. } = &boundaries[0].reason else {
            panic!("a harness switch is a drift");
        };
        assert_eq!(*confirmations, 1);
        let before = segmenter.segment_at(Dimension::Harness, 1).unwrap();
        assert!(before.closed);
        assert_eq!(before.last_turn, 1);
        assert_eq!(before.id, boundaries[0].from);
    }

    #[test]
    fn a_single_off_topic_turn_is_not_a_boundary() {
        let segmenter = run(&[
            topic(0, &["relay", "provider"]),
            topic(1, &["relay", "provider"]),
            topic(2, &["weather"]),
            topic(3, &["relay", "provider"]),
        ]);
        assert!(segmenter.boundaries().is_empty());
        let only = segmenter.current(Dimension::Topic).unwrap();
        assert_eq!(
            (only.start_turn, only.last_turn, only.observations),
            (0, 3, 3)
        );
    }

    #[test]
    fn consecutive_deviations_open_a_segment_at_the_first_deviating_turn() {
        let segmenter = run(&[
            topic(0, &["relay", "provider"]),
            topic(1, &["relay", "provider"]),
            topic(2, &["weather", "antibes"]),
            topic(3, &["weather", "antibes"]),
        ]);
        let boundary = &segmenter.boundaries()[0];
        // Confirmed on turn 3, but the direction changed on turn 2.
        assert_eq!(boundary.at_turn, 2);
        let BoundaryReason::Drift {
            distance_permille,
            threshold_permille,
            ..
        } = &boundary.reason
        else {
            panic!("a topic drift is a drift");
        };
        assert_eq!(*distance_permille, 1_000);
        assert_eq!(*threshold_permille, 700);
        let old = segmenter.segment_at(Dimension::Topic, 1).unwrap();
        assert_eq!((old.start_turn, old.last_turn, old.closed), (0, 1, true));
        let new = segmenter.segment_at(Dimension::Topic, 3).unwrap();
        assert_eq!((new.start_turn, new.last_turn, new.closed), (2, 3, false));
        assert_eq!(new.opened_by.as_ref(), Some(&boundary.reason));
    }

    #[test]
    fn a_return_inside_the_segment_cancels_the_pending_run() {
        let segmenter = run(&[
            topic(0, &["relay"]),
            topic(1, &["weather"]), // deviates (1 of 2 needed)
            topic(2, &["relay"]),   // back: the run is cancelled
            topic(3, &["weather"]), // deviates again, but alone
        ]);
        assert!(segmenter.boundaries().is_empty());
    }

    #[test]
    fn axes_are_independent() {
        let segmenter = run(&[
            obs(
                0,
                &[
                    (Dimension::Harness, &["deepseek"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
            obs(
                1,
                &[
                    (Dimension::Harness, &["claude-code"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
        ]);
        let moved: Vec<_> = segmenter.boundaries().iter().map(|b| b.dimension).collect();
        assert_eq!(moved, [Dimension::Harness]);
        assert_eq!(segmenter.current(Dimension::Topic).unwrap().observations, 2);
    }

    #[test]
    fn a_refused_turn_changes_nothing() {
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        segmenter.observe(&topic(5, &["relay"])).unwrap();
        let before = segmenter.segments();
        for turn in [5, 3, 0] {
            assert_eq!(
                segmenter.observe(&topic(turn, &["weather"])),
                Err(SegmentError::OutOfOrder { turn, last: 5 })
            );
        }
        assert_eq!(segmenter.segments(), before);
        assert!(segmenter.boundaries().is_empty());
        // And the next valid turn is still accepted.
        segmenter.observe(&topic(6, &["relay"])).unwrap();
    }

    #[test]
    fn replay_is_deterministic_and_ids_follow_the_seed() {
        let history = [
            obs(
                0,
                &[
                    (Dimension::Harness, &["deepseek"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
            obs(
                1,
                &[
                    (Dimension::Harness, &["deepseek"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
            obs(
                2,
                &[
                    (Dimension::Harness, &["claude-code"]),
                    (Dimension::Topic, &["tests"]),
                ],
            ),
            obs(
                3,
                &[
                    (Dimension::Harness, &["claude-code"]),
                    (Dimension::Topic, &["tests"]),
                ],
            ),
        ];
        let a = Segmenter::replay(SegmentPolicy::default(), 42, &history).unwrap();
        let b = Segmenter::replay(SegmentPolicy::default(), 42, &history).unwrap();
        assert_eq!(a.segments(), b.segments());
        assert_eq!(a.boundaries(), b.boundaries());
        let other = Segmenter::replay(SegmentPolicy::default(), 43, &history).unwrap();
        assert_ne!(a.segments()[0].id, other.segments()[0].id);
        // Same segments, only the ids differ.
        assert_eq!(a.segments().len(), other.segments().len());
        // The hash itself is pinned in `segment_id_is_pinned`.
        assert!(a.segments()[0].id.as_str().starts_with("seg-"));
        assert_eq!(a.segments()[0].id.as_str().len(), "seg-".len() + 16);
    }

    #[test]
    fn segment_id_is_pinned() {
        // Computed once from the definition (FNV-1a 64 over seed, axis name, first turn,
        // each followed by 0xff). If this changes, stored segment ids stop matching.
        assert_eq!(
            SegmentId::derive(42, Dimension::Harness, 3).as_str(),
            "seg-30a42f3e03fc7385"
        );
        assert_ne!(
            SegmentId::derive(0, Dimension::Harness, 0),
            SegmentId::derive(0, Dimension::Topic, 0)
        );
        assert_ne!(
            SegmentId::derive(0, Dimension::Harness, 0),
            SegmentId::derive(0, Dimension::Harness, 1)
        );
    }

    #[test]
    fn a_fingerprint_stops_growing_at_the_cap() {
        // Every turn shares one label with the segment (so it is never "totally disjoint")
        // and brings ten new ones: only a segment that keeps absorbing can grow past the cap.
        let policy = SegmentPolicy::default()
            .with(Dimension::Understanding, Threshold::new(1_000, 1).unwrap());
        let labels: Vec<String> = (0..200).map(|i| format!("l{i:03}")).collect();
        let mut segmenter = Segmenter::new(policy, 7);
        for (turn, chunk) in labels.chunks(10).enumerate() {
            let mut observation = Observation {
                turn: u32::try_from(turn).unwrap(),
                ..Observation::default()
            };
            let mut print = vec!["core".to_owned()];
            print.extend(chunk.iter().cloned());
            observation
                .prints
                .insert(Dimension::Understanding, Fingerprint::new(print));
            segmenter.observe(&observation).unwrap();
        }
        let segments = segmenter.segments();
        assert_eq!(segments.len(), 1, "one segment absorbed all twenty turns");
        assert_eq!(segments[0].observations, 20);
        assert_eq!(
            segments[0].fingerprint.labels().count(),
            MAX_FINGERPRINT_LABELS
        );
    }

    #[test]
    fn thresholds_are_validated() {
        assert_eq!(
            Threshold::new(0, 1),
            Err(SegmentError::DistanceOutOfRange(0))
        );
        assert_eq!(
            Threshold::new(1_001, 1),
            Err(SegmentError::DistanceOutOfRange(1_001))
        );
        assert_eq!(Threshold::new(500, 0), Err(SegmentError::NoConfirmation));
        assert!(Threshold::new(500, 1).is_ok());
    }

    #[test]
    fn a_custom_policy_changes_where_a_boundary_falls() {
        let strict =
            SegmentPolicy::default().with(Dimension::Topic, Threshold::new(700, 3).unwrap());
        let history = [
            topic(0, &["relay"]),
            topic(1, &["weather"]),
            topic(2, &["weather"]),
        ];
        // Two deviations do not confirm a three-confirmation threshold...
        let segmenter = Segmenter::replay(strict, 7, &history).unwrap();
        assert!(segmenter.boundaries().is_empty());
        // ...they do confirm the default one.
        assert_eq!(run(&history).boundaries().len(), 1);
    }

    #[test]
    fn boundaries_serialize_with_snake_case_axes() {
        let segmenter = run(&[harness(0, "a"), harness(1, "b")]);
        let json = serde_json::to_value(&segmenter.boundaries()[0]).unwrap();
        assert_eq!(json["dimension"], "harness");
        assert_eq!(json["at_turn"], 1);
        let back: Boundary = serde_json::from_value(json).unwrap();
        assert_eq!(&back, &segmenter.boundaries()[0]);
    }

    #[test]
    fn a_fully_contained_turn_is_at_distance_zero() {
        // Metric-level pin of the fix: a turn whose labels are all already in the
        // segment adds no new direction, so its distance is 0 no matter how many
        // labels the segment has absorbed (up to the cap). This is the property
        // that keeps the segmenter from over-segmenting as a segment grows.
        let all: Vec<String> = (0..MAX_FINGERPRINT_LABELS)
            .map(|i| format!("l{i:02}"))
            .collect();
        let turn = Fingerprint::new(["l00", "l01", "l02"]);
        for grown in turn.labels().count()..=MAX_FINGERPRINT_LABELS {
            let segment = Fingerprint::new(all.iter().take(grown).cloned());
            assert_eq!(
                segment.distance_permille(&turn),
                0,
                "a fully contained turn must be at distance 0 (grown={grown})"
            );
        }
    }

    #[test]
    fn a_fully_contained_turn_stays_inside_however_large_the_segment() {
        // Integration pin of the over-segmentation fix: grow a Topic segment well
        // past the 10-label cliff where symmetric Jaccard used to break, then feed
        // two consecutive turns that only repeat labels the segment already knows.
        // Neither may open a boundary — they add no new direction.
        //
        // Each turn brings one new label alongside a shared "core" label, so its
        // novelty is 500 permille (under the Topic threshold of 700) and the
        // segment absorbs it: after 20 turns it holds 21 labels.
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        for turn in 0..20u32 {
            let labels = ["core".to_string(), format!("l{turn}")];
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            let observation = obs(turn, &[(Dimension::Topic, labels.as_slice())]);
            assert!(
                segmenter.observe(&observation).unwrap().is_empty(),
                "growing the segment must not open a boundary (turn={turn})"
            );
        }
        assert_eq!(
            segmenter
                .current(Dimension::Topic)
                .unwrap()
                .fingerprint
                .labels()
                .count(),
            21
        );
        // Two turns that only repeat known labels: no boundary.
        for turn in 20..22u32 {
            assert!(
                segmenter
                    .observe(&topic(turn, &["core", "l0", "l1"]))
                    .unwrap()
                    .is_empty(),
                "a fully contained turn must not open a boundary (turn={turn})"
            );
        }
        assert!(segmenter.boundaries().is_empty());
    }

    #[test]
    fn a_multi_label_harness_change_still_opens_a_boundary() {
        // Review #2: a harness fingerprint that carries more than the provider id
        // (e.g. provider + model) must still fire on ANY change. Strict equality,
        // not a distance, so {a, a-chat} -> {a, a-reasoner} is a switch.
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        segmenter
            .observe(&obs(
                0,
                &[(Dimension::Harness, &["deepseek", "deepseek-chat"])],
            ))
            .unwrap();
        let boundaries = segmenter
            .observe(&obs(
                1,
                &[(Dimension::Harness, &["deepseek", "deepseek-reasoner"])],
            ))
            .unwrap();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].at_turn, 1);
    }

    #[test]
    fn two_unrelated_off_topic_turns_do_not_confirm_a_boundary() {
        // Review #6: two deviating turns that do not head in the same direction are
        // two detours, not one change: the second restarts the pending run.
        let segmenter = run(&[
            topic(0, &["relay", "budget"]),
            topic(1, &["relay", "budget"]),
            topic(2, &["relay", "budget"]),
            topic(3, &["weather"]),
            topic(4, &["football"]),
        ]);
        assert!(segmenter.boundaries().is_empty());
    }

    #[test]
    fn a_fingerprint_keeps_the_sorted_first_64_regardless_of_arrival_order() {
        // Review #4: labels beyond the cap are dropped and the survivors are the
        // lexicographically-first 64, so the result does not depend on arrival order.
        let a: Vec<String> = (0..100).map(|i| format!("l{i:03}")).collect();
        let mut b = a.clone();
        b.reverse();
        let fa = Fingerprint::new(a.iter().map(String::as_str));
        let fb = Fingerprint::new(b.iter().map(String::as_str));
        assert_eq!(fa, fb);
        assert_eq!(fa.labels().count(), MAX_FINGERPRINT_LABELS);
    }

    #[test]
    fn deserializing_an_invalid_threshold_is_refused() {
        // Review #5: the wire form goes through Threshold::new, so out-of-range
        // documents are refused instead of silently changing how the segmenter cuts.
        let zero_distance: Result<Threshold, _> =
            serde_json::from_str(r#"{"distance_permille":0,"confirmations":2}"#);
        assert!(zero_distance.is_err());
        let no_confirmation: Result<Threshold, _> =
            serde_json::from_str(r#"{"distance_permille":700,"confirmations":0}"#);
        assert!(no_confirmation.is_err());
        let ok: Threshold =
            serde_json::from_str(r#"{"distance_permille":700,"confirmations":2}"#).unwrap();
        assert_eq!(ok, Threshold::new(700, 2).unwrap());
    }

    #[test]
    fn deserializing_a_fingerprint_normalizes_it() {
        // Review #5: stored labels are re-normalized (trim, lowercase, drop-empty,
        // cap), not trusted as-is.
        let f: Fingerprint = serde_json::from_str(r#"["  Relay ","RELAY","","Routing"]"#).unwrap();
        assert_eq!(f.labels().collect::<Vec<_>>(), ["relay", "routing"]);
    }

    #[test]
    fn segments_are_contiguous_and_do_not_overlap_on_one_axis() {
        // Review #7: on one axis, a segment ends exactly where the next begins.
        let segmenter = run(&[
            topic(0, &["relay"]),
            topic(1, &["relay"]),
            topic(2, &["weather"]),
            topic(3, &["weather"]),
            topic(4, &["relay"]),
            topic(5, &["relay"]),
        ]);
        let topic_segments: Vec<_> = segmenter
            .segments()
            .into_iter()
            .filter(|s| s.dimension == Dimension::Topic)
            .collect();
        assert_eq!(topic_segments.len(), 3);
        for window in topic_segments.windows(2) {
            assert_eq!(window[0].last_turn + 1, window[1].start_turn);
            assert!(window[0].closed);
        }
        assert!(!topic_segments.last().unwrap().closed);
    }

    #[test]
    fn an_a_to_b_to_a_oscillation_gives_two_boundaries() {
        // Review #7: A -> B -> A' is two boundaries; the ids chain A -> B -> A'.
        let segmenter = run(&[
            topic(0, &["a"]),
            topic(1, &["a"]),
            topic(2, &["b"]),
            topic(3, &["b"]),
            topic(4, &["a"]),
            topic(5, &["a"]),
        ]);
        let boundaries = segmenter.boundaries();
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0].at_turn, 2);
        assert_eq!(boundaries[1].at_turn, 4);
        assert_eq!(
            boundaries[0].to, boundaries[1].from,
            "B ends where A' begins"
        );
        assert_ne!(
            boundaries[0].from, boundaries[1].to,
            "A and A' are distinct segments"
        );
    }

    #[test]
    fn a_change_on_the_first_turn_opens_no_boundary() {
        // Review #7: the first observation of an axis opens a segment, it never deviates.
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        let boundaries = segmenter.observe(&topic(0, &["weather"])).unwrap();
        assert!(boundaries.is_empty());
        assert_eq!(segmenter.current(Dimension::Topic).unwrap().start_turn, 0);
    }

    #[test]
    fn multiple_axes_can_open_boundaries_on_the_same_turn() {
        // Review #7: harness (1 confirmation) and topic (2) can confirm on the same turn.
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        segmenter
            .observe(&obs(
                0,
                &[(Dimension::Harness, &["a"]), (Dimension::Topic, &["x"])],
            ))
            .unwrap();
        segmenter
            .observe(&obs(
                1,
                &[(Dimension::Harness, &["a"]), (Dimension::Topic, &["y"])],
            ))
            .unwrap();
        let boundaries = segmenter
            .observe(&obs(
                2,
                &[(Dimension::Harness, &["b"]), (Dimension::Topic, &["y"])],
            ))
            .unwrap();
        assert_eq!(boundaries.len(), 2);
        let dims: Vec<_> = boundaries.iter().map(|b| b.dimension).collect();
        assert!(dims.contains(&Dimension::Harness) && dims.contains(&Dimension::Topic));
    }

    #[test]
    fn segment_at_is_none_before_the_first_segment_and_in_a_gap() {
        // Review #7: a turn before the first segment, or in a gap with no evidence,
        // belongs to no segment.
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        assert_eq!(segmenter.segment_at(Dimension::Topic, 0), None);
        segmenter
            .observe(&obs(2, &[(Dimension::Topic, &["relay"])]))
            .unwrap();
        assert_eq!(
            segmenter.segment_at(Dimension::Topic, 1),
            None,
            "a turn before the first segment is a gap"
        );
        assert!(segmenter.segment_at(Dimension::Topic, 2).is_some());
    }
}
