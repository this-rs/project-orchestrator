//! Synaptic layer in SHADOW mode (T8 of the plan « Chat sans dépendance au cwd »).
//!
//! A second, separate track of synapse weights (`shadow_weight` on the existing
//! `SYNAPSE` edges) that learns ONLY from signals coming from outside the agent,
//! decays proportionally to elapsed time, and drives a best-first expansion from
//! the session's anchors bounded by a score threshold, a hop limit and a token
//! budget. In shadow mode the expansion is computed and journaled — never
//! injected: the session receives exactly what it received before.
//!
//! # What the layer never does
//!
//! - It never writes `SYNAPSE.weight` (the legacy weight that spreading
//!   activation and every existing reader use), never creates nor deletes an
//!   edge: only the `shadow_*` properties of edges that already exist.
//! - It never reinforces without an [`ExternalSignal`]: there is no other entry
//!   point that raises a shadow weight. Co-activation, retrieval, an agent
//!   confirming a note: none of them is a reward here.
//! - It never rewards a note created during the conversation being judged
//!   (whoever created it): an agent cannot write a note, get it surfaced and
//!   then be rewarded for its own production.
//! - It never crosses a project: edges are read and written only when both ends
//!   belong to the trace's project; global notes (no project) are not seeds.
//!
//! # Pieces
//!
//! - [`SynapticLayerFlags`]: `off` / `shadow` / `on` per project (`on` is
//!   accepted but applied as `shadow`: the switch is gated by the T9 evaluation).
//! - [`ExternalSignal`]: verification of a runner attempt (`verification_json`),
//!   confirmation by a human (a signed-in person, never an agent token), the task
//!   closed without being restarted, or restarted.
//! - [`proportional_decay`]: `w · 2^(−Δt / h)`, by elapsed time, so the real
//!   cadence of the heartbeat does not matter (see [`shadow_half_life_days`]).
//! - [`anchor_affinity`] (top-K = 20) and [`best_first_expand`] (threshold 0.15,
//!   3 hops, tokens first, 10 % deterministic exploration).
//! - [`SynapseShadowTrace`]: the versioned journal entry, with a fingerprint of
//!   every weight the expansion read.
use crate::neo4j::traits::*;

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::chat::anchor::{Anchor, AnchorRole, AnchorState, AnchorTargetType};
use crate::neo4j::GraphStore;

// ============================================================================
// Versions and constants
// ============================================================================

/// Version of the journal entry layout ([`SynapseShadowTrace`]).
pub const TRACE_SCHEMA_VERSION: u32 = 1;
/// Version of the algorithm and its parameters. Bumped whenever a constant
/// below changes, so two traces are only compared within one version.
pub const LAYER_VERSION: &str = "t8-shadow-v1";

/// Score under which the best-first expansion stops.
pub const SCORE_THRESHOLD: f64 = 0.15;
/// Furthest an expansion goes from a seed.
pub const MAX_HOPS: usize = 3;
/// Token budget of what the layer would inject (same order as the focus budget
/// of the anchor resolver).
pub const TOKEN_BUDGET: usize = 1200;
/// At most this many seeds come from the anchors.
pub const ANCHOR_AFFINITY_TOP_K: usize = 20;
/// Probability that one candidate under the threshold is explored.
pub const EXPLORATION_RATE: f64 = 0.10;
/// A shadow weight under this is dormant: not followed (the edge is kept).
pub const DORMANT_WEIGHT: f64 = 0.05;
/// Days for a shadow weight of 1.0 to reach [`DORMANT_WEIGHT`] without reward.
/// Same order as the legacy subtractive decay of a co-activation synapse
/// (0.02 every 6 h: 1.0 → 0.05 in 47.5 cycles, 11.9 days).
pub const DAYS_TO_DORMANT: f64 = 11.0;
/// How long after its attempt closed a trace is judged on the task's closure.
pub const CLOSURE_QUIET_HOURS: i64 = 72;
/// Traces kept per session (older ones are deleted when a new one is written).
pub const TRACES_KEPT_PER_SESSION: usize = 50;
/// A session journals at most one trace per this many seconds.
pub const TURN_RECHECK_SECS: u64 = 60;

/// Neighbours read per expanded node (strongest first), a circuit breaker.
const MAX_EDGES_PER_NODE: usize = 32;
/// Nodes the pre-loaded neighbourhood may hold, a circuit breaker.
const MAX_LOADED_NODES: usize = 400;

/// Half-life (days) that brings 1.0 to [`DORMANT_WEIGHT`] in [`DAYS_TO_DORMANT`]:
/// `h = D / log2(1 / threshold)` — 11 / log2(20) ≈ 2.55 days.
pub fn shadow_half_life_days() -> f64 {
    half_life_days_for(DAYS_TO_DORMANT, DORMANT_WEIGHT)
}

/// Half-life that brings a weight of 1.0 to `threshold` in `days`.
pub fn half_life_days_for(days: f64, threshold: f64) -> f64 {
    days / (1.0 / threshold).log2()
}

/// Proportional decay by elapsed time: `w · 2^(−elapsed / half_life)`.
///
/// Elapsed time, not a number of passes: a pass that runs twice, late or
/// never-then-twice decays the same. A non-positive elapsed time (clock skew)
/// leaves the weight unchanged.
pub fn proportional_decay(weight: f64, elapsed_secs: f64, half_life_days: f64) -> f64 {
    if elapsed_secs <= 0.0 || half_life_days <= 0.0 {
        return weight;
    }
    weight * 0.5_f64.powf(elapsed_secs / (half_life_days * 86_400.0))
}

// ============================================================================
// Flags
// ============================================================================

/// Mode of the synaptic layer for one project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SynapticLayerMode {
    /// Nothing is computed, nothing is written.
    #[default]
    Off,
    /// Computed and journaled, never injected.
    Shadow,
    /// Requested switch. Applied as [`Self::Shadow`] until the T9 evaluation
    /// opens it (14 days and ≥ 100 closed tasks per project, B ≥ A + 3 points).
    On,
}

impl SynapticLayerMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "" => Some(Self::Off),
            "shadow" => Some(Self::Shadow),
            "on" => Some(Self::On),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::On => "on",
        }
    }

    /// What really runs: `on` is not open yet, so it runs as `shadow`.
    pub fn effective(self) -> Self {
        match self {
            Self::On => Self::Shadow,
            other => other,
        }
    }
}

/// Environment variable of the default mode (`off` when absent).
pub const ENV_MODE: &str = "PO_SYNAPTIC_LAYER";
/// Environment variable of the per-project overrides: `<project uuid>=<mode>,…`.
pub const ENV_PROJECTS: &str = "PO_SYNAPTIC_LAYER_PROJECTS";

/// The modes of the layer, per project.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynapticLayerFlags {
    pub default: SynapticLayerMode,
    pub per_project: HashMap<Uuid, SynapticLayerMode>,
}

impl SynapticLayerFlags {
    /// Parse the two variables. An unknown mode or a malformed entry is ignored
    /// with a warning (it never turns the layer on).
    pub fn parse(default: Option<&str>, projects: Option<&str>) -> Self {
        let default = match default {
            None => SynapticLayerMode::Off,
            Some(s) => SynapticLayerMode::parse(s).unwrap_or_else(|| {
                tracing::warn!(target: "synapse_shadow", value = %s, "unknown {ENV_MODE}, layer off");
                SynapticLayerMode::Off
            }),
        };
        let mut per_project = HashMap::new();
        for entry in projects.unwrap_or("").split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let parsed = entry.split_once('=').and_then(|(id, mode)| {
                Some((
                    id.trim().parse::<Uuid>().ok()?,
                    SynapticLayerMode::parse(mode)?,
                ))
            });
            match parsed {
                Some((id, mode)) => {
                    per_project.insert(id, mode);
                }
                None => {
                    tracing::warn!(target: "synapse_shadow", entry = %entry, "ignored {ENV_PROJECTS} entry")
                }
            }
        }
        Self {
            default,
            per_project,
        }
    }

    pub fn from_env() -> Self {
        Self::parse(
            std::env::var(ENV_MODE).ok().as_deref(),
            std::env::var(ENV_PROJECTS).ok().as_deref(),
        )
    }

    /// The flags of this process, read once.
    pub fn global() -> &'static SynapticLayerFlags {
        static FLAGS: LazyLock<SynapticLayerFlags> = LazyLock::new(SynapticLayerFlags::from_env);
        &FLAGS
    }

    /// The mode asked for `project`.
    pub fn requested(&self, project: Uuid) -> SynapticLayerMode {
        self.per_project
            .get(&project)
            .copied()
            .unwrap_or(self.default)
    }

    /// The mode that runs for `project`.
    pub fn effective(&self, project: Uuid) -> SynapticLayerMode {
        self.requested(project).effective()
    }

    /// Whether the layer may run for at least one project.
    pub fn any_enabled(&self) -> bool {
        self.default != SynapticLayerMode::Off
            || self
                .per_project
                .values()
                .any(|m| *m != SynapticLayerMode::Off)
    }

    /// The projects scope of the decay: `(default_enabled, explicitly on, explicitly off)`.
    pub fn decay_scope(&self) -> ShadowDecayScope {
        let mut enabled = Vec::new();
        let mut disabled = Vec::new();
        for (id, mode) in &self.per_project {
            if *mode == SynapticLayerMode::Off {
                disabled.push(*id);
            } else {
                enabled.push(*id);
            }
        }
        enabled.sort();
        disabled.sort();
        ShadowDecayScope {
            default_enabled: self.default != SynapticLayerMode::Off,
            enabled,
            disabled,
        }
    }
}

/// Which projects' shadow weights a decay pass touches.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShadowDecayScope {
    pub default_enabled: bool,
    pub enabled: Vec<Uuid>,
    pub disabled: Vec<Uuid>,
}

impl ShadowDecayScope {
    pub fn includes(&self, project: Uuid) -> bool {
        if self.disabled.contains(&project) {
            return false;
        }
        self.default_enabled || self.enabled.contains(&project)
    }
}

// ============================================================================
// External signals (the only rewards)
// ============================================================================

/// A signal that comes from outside the agent. The ONLY input that changes a
/// shadow weight upwards (and the only one downwards, besides time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "signal")]
pub enum ExternalSignal {
    /// The runner's verifier judged the attempt of the session
    /// (`AgentExecution.verification_json`).
    Verification { passed: bool },
    /// A signed-in person confirmed this note (not an agent token, not the
    /// server's own account; without authentication there is no person to tell
    /// apart, so no such signal).
    HumanConfirmed { note_id: Uuid },
    /// The task of the session's attempt stayed closed (completed, no later
    /// attempt) for [`CLOSURE_QUIET_HOURS`].
    TaskClosedQuietly,
    /// A later attempt of the same task was started: the task was restarted.
    TaskRestarted,
}

impl ExternalSignal {
    /// Change applied to each rewarded shadow weight.
    pub fn delta(&self) -> f64 {
        match self {
            Self::Verification { passed: true } => 0.10,
            Self::Verification { passed: false } => -0.10,
            Self::HumanConfirmed { .. } => 0.15,
            Self::TaskClosedQuietly => 0.05,
            Self::TaskRestarted => -0.10,
        }
    }

    /// Idempotency key: a trace is rewarded at most once per key. Closure is one
    /// key for both outcomes: a trace is judged quiet OR restarted, once.
    pub fn key(&self) -> String {
        match self {
            Self::Verification { .. } => "verification".to_string(),
            Self::HumanConfirmed { note_id } => format!("human_confirmed:{note_id}"),
            Self::TaskClosedQuietly | Self::TaskRestarted => CLOSURE_KEY.to_string(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Verification { passed: true } => "verification_passed",
            Self::Verification { passed: false } => "verification_failed",
            Self::HumanConfirmed { .. } => "human_confirmed",
            Self::TaskClosedQuietly => "task_closed_quietly",
            Self::TaskRestarted => "task_restarted",
        }
    }
}

/// Idempotency key of the closure judgement.
pub const CLOSURE_KEY: &str = "task_closure";

/// How the task of a trace's attempt ended, as the graph shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskClosure {
    /// Completed and no later attempt, long enough ago.
    Quiet,
    /// A later attempt of the same task exists.
    Restarted,
}

impl TaskClosure {
    pub fn signal(self) -> ExternalSignal {
        match self {
            Self::Quiet => ExternalSignal::TaskClosedQuietly,
            Self::Restarted => ExternalSignal::TaskRestarted,
        }
    }
}

// ============================================================================
// Journal entry
// ============================================================================

/// One seed of an expansion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowSeed {
    pub note_id: Uuid,
    pub affinity: f64,
}

/// One note the layer would inject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowSelection {
    pub note_id: Uuid,
    pub score: f64,
    pub hops: usize,
    /// The note it was reached from (`None` for a seed): the edge `via → note`
    /// is what a reward reinforces.
    pub via: Option<Uuid>,
    pub tokens: usize,
    /// Taken by the exploration, under the threshold.
    pub explored: bool,
    /// When the note was created (`None`: unknown).
    pub created_at: Option<DateTime<Utc>>,
}

/// Why an expansion stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Nothing to start from (no anchor with an affine note).
    NoSeed,
    /// The best remaining candidate is under [`SCORE_THRESHOLD`].
    BelowThreshold,
    /// The token budget is spent.
    TokenBudget,
    /// Every reachable candidate was taken.
    Exhausted,
}

/// The versioned journal entry of one shadow turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynapseShadowTrace {
    pub id: Uuid,
    pub schema_version: u32,
    pub layer_version: String,
    pub session_id: Uuid,
    pub project_id: Uuid,
    pub requested_mode: SynapticLayerMode,
    pub effective_mode: SynapticLayerMode,
    /// When the conversation started: a note created at or after it is never
    /// rewarded through this trace.
    pub session_started_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub seeds: Vec<ShadowSeed>,
    pub selected: Vec<ShadowSelection>,
    pub tokens_used: usize,
    pub token_budget: usize,
    pub stopped_by: StopReason,
    /// SHA-256 of every `(from, to, weight)` the expansion read, sorted.
    pub weights_fingerprint: String,
    /// SHA-256 of the parameters ([`ExpansionConfig`] + [`LAYER_VERSION`]).
    pub config_fingerprint: String,
    pub edges_read: usize,
    /// Idempotency keys of the rewards already applied (see [`ExternalSignal::key`]).
    #[serde(default)]
    pub rewarded: Vec<String>,
}

impl SynapseShadowTrace {
    /// Ids of every note the trace selected.
    pub fn note_ids(&self) -> Vec<Uuid> {
        self.selected.iter().map(|s| s.note_id).collect()
    }
}

/// Which traces to read.
#[derive(Debug, Clone, PartialEq)]
pub enum ShadowTraceFilter {
    /// Every trace of one session, oldest first.
    Session(Uuid),
    /// The latest traces (at most `limit`) that selected this note.
    ContainingNote { note_id: Uuid, limit: usize },
}

/// One `SYNAPSE` edge as the shadow layer reads it, with its target's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowEdgeRow {
    pub from: Uuid,
    pub to: Uuid,
    /// `shadow_weight`, or the legacy `weight` while the edge has no shadow weight.
    pub weight: f64,
    pub target: ShadowNode,
}

/// What the expansion needs to know about a note.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowNode {
    pub energy: f64,
    pub scar_intensity: f64,
    /// Status active or needs_review.
    pub reachable: bool,
    /// Estimated tokens of the note's content.
    pub tokens: usize,
    pub created_at: Option<DateTime<Utc>>,
}

/// Rough token estimate of a text (4 characters per token, at least 1).
pub fn estimate_tokens(chars: usize) -> usize {
    chars.div_ceil(4).max(1)
}

// ============================================================================
// Rewards: which edges a signal reaches
// ============================================================================

/// Whether a note may be rewarded through a trace: created strictly before the
/// conversation started. An unknown creation date is not rewarded (unproven).
pub fn rewardable(created_at: Option<DateTime<Utc>>, session_started_at: DateTime<Utc>) -> bool {
    matches!(created_at, Some(at) if at < session_started_at)
}

/// The edges a signal reinforces (or weakens) in a trace: the `via → note`
/// edges the expansion really followed, both ends rewardable. For a human
/// confirmation, only the followed edges that touch the confirmed note.
/// Unordered pairs, deduplicated, sorted.
pub fn rewarded_pairs(trace: &SynapseShadowTrace, signal: &ExternalSignal) -> Vec<(Uuid, Uuid)> {
    let created: HashMap<Uuid, Option<DateTime<Utc>>> = trace
        .selected
        .iter()
        .map(|s| (s.note_id, s.created_at))
        .collect();
    let ok = |id: Uuid| {
        created
            .get(&id)
            .is_some_and(|c| rewardable(*c, trace.session_started_at))
    };
    let mut pairs: Vec<(Uuid, Uuid)> = trace
        .selected
        .iter()
        .filter_map(|s| s.via.map(|via| (via, s.note_id)))
        .filter(|(a, b)| a != b && ok(*a) && ok(*b))
        .filter(|(a, b)| match signal {
            ExternalSignal::HumanConfirmed { note_id } => a == note_id || b == note_id,
            _ => true,
        })
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();
    pairs.sort();
    pairs.dedup();
    pairs
}

// ============================================================================
// Seeds: affinity to the anchors
// ============================================================================

/// Affinity of an anchor's role (the strongest role it holds).
pub fn role_weight(anchor: &Anchor) -> f64 {
    [
        (AnchorRole::Focus, 1.0),
        (AnchorRole::Mention, 0.6),
        (AnchorRole::Origin, 0.3),
    ]
    .iter()
    .filter(|(role, _)| anchor.has_role(*role))
    .map(|(_, w)| *w)
    .fold(0.0, f64::max)
}

/// Whether an anchor seeds the expansion: a live (or moved) target narrower
/// than a project. A project or workspace anchor is the whole scope, not a
/// focus: it seeds nothing.
pub fn seeds_from(anchor: &Anchor) -> bool {
    matches!(anchor.state, AnchorState::Live | AnchorState::Moved)
        && !matches!(
            anchor.target_type,
            AnchorTargetType::Project | AnchorTargetType::Workspace
        )
}

/// Factor of a note reached THROUGH an anchor (linked to its target) rather
/// than being the anchor's target itself.
pub const LINKED_NOTE_FACTOR: f64 = 0.8;

/// The notes an anchor brings, with their affinity.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorNotes {
    pub anchor_weight: f64,
    /// `(note, is the anchor's own target)`.
    pub notes: Vec<(Uuid, bool)>,
}

/// Seeds of the expansion: each note's best affinity over the anchors, bounded
/// to the top `k` (score descending, id ascending on ties).
pub fn anchor_affinity(anchors: &[AnchorNotes], k: usize) -> Vec<ShadowSeed> {
    let mut best: BTreeMap<Uuid, f64> = BTreeMap::new();
    for a in anchors {
        for (note, direct) in &a.notes {
            let affinity =
                (a.anchor_weight * if *direct { 1.0 } else { LINKED_NOTE_FACTOR }).clamp(0.0, 1.0);
            let slot = best.entry(*note).or_insert(0.0);
            if affinity > *slot {
                *slot = affinity;
            }
        }
    }
    let mut seeds: Vec<ShadowSeed> = best
        .into_iter()
        .filter(|(_, a)| *a > 0.0)
        .map(|(note_id, affinity)| ShadowSeed { note_id, affinity })
        .collect();
    seeds.sort_by(|a, b| {
        b.affinity
            .partial_cmp(&a.affinity)
            .unwrap_or(Ordering::Equal)
            .then(a.note_id.cmp(&b.note_id))
    });
    seeds.truncate(k);
    seeds
}

// ============================================================================
// Best-first expansion (pure)
// ============================================================================

/// Parameters of an expansion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ExpansionConfig {
    pub score_threshold: f64,
    pub max_hops: usize,
    pub token_budget: usize,
    pub exploration_rate: f64,
    pub dormant_weight: f64,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            score_threshold: SCORE_THRESHOLD,
            max_hops: MAX_HOPS,
            token_budget: TOKEN_BUDGET,
            exploration_rate: EXPLORATION_RATE,
            dormant_weight: DORMANT_WEIGHT,
        }
    }
}

impl ExpansionConfig {
    pub fn fingerprint(&self) -> String {
        sha256_hex(&format!(
            "{LAYER_VERSION}|{}|{}|{}|{}|{}|{}",
            self.score_threshold,
            self.max_hops,
            self.token_budget,
            self.exploration_rate,
            self.dormant_weight,
            ANCHOR_AFFINITY_TOP_K
        ))
    }
}

/// A resolved neighbourhood: outgoing edges and node metadata.
#[derive(Debug, Clone, Default)]
pub struct ShadowHood {
    pub edges: HashMap<Uuid, Vec<(Uuid, f64)>>,
    pub nodes: HashMap<Uuid, ShadowNode>,
}

impl ShadowHood {
    /// Every weight read, as a fingerprint (sorted, 6 decimals).
    pub fn weights_fingerprint(&self) -> String {
        let mut rows: Vec<(Uuid, Uuid, i64)> = self
            .edges
            .iter()
            .flat_map(|(from, out)| {
                out.iter()
                    .map(move |(to, w)| (*from, *to, (w * 1e6).round() as i64))
            })
            .collect();
        rows.sort();
        let text: String = rows
            .iter()
            .map(|(a, b, w)| format!("{a}>{b}={w};"))
            .collect();
        sha256_hex(&text)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.values().map(Vec::len).sum()
    }
}

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// Result of an expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct Expansion {
    pub selected: Vec<ShadowSelection>,
    pub tokens_used: usize,
    pub stopped_by: StopReason,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    score: f64,
    node: Uuid,
    hops: usize,
    via: Option<Uuid>,
}

impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on score; on ties the LOWEST id first, then the fewest hops.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
            .then_with(|| other.hops.cmp(&self.hops))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Deterministic uniform draw in `[0, 1)` from a seed (SplitMix64).
pub fn unit_draw(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Seed of the exploration draw of one turn: the session and the turn's index.
pub fn exploration_seed(session_id: Uuid, turn: u64) -> u64 {
    let (hi, lo) = session_id.as_u64_pair();
    hi ^ lo.rotate_left(17) ^ turn.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Best-first expansion from `seeds` over `hood`, by score budget.
///
/// The best candidate is taken first. A candidate under the threshold stops the
/// expansion; one that does not fit in the remaining tokens is skipped (a
/// smaller one may still fit: tokens come first, the score only orders). A
/// taken node opens its neighbours: `score · weight · energy · (1 − 0.7·scar)`,
/// dormant edges and unreachable nodes not followed, up to `max_hops`. Then,
/// with probability `exploration_rate` (drawn from `draw`), the best candidate
/// left under the threshold that fits is taken too, marked `explored`.
pub fn best_first_expand(
    seeds: &[ShadowSeed],
    hood: &ShadowHood,
    config: &ExpansionConfig,
    draw: f64,
) -> Expansion {
    if seeds.is_empty() {
        return Expansion {
            selected: Vec::new(),
            tokens_used: 0,
            stopped_by: StopReason::NoSeed,
        };
    }
    let mut heap: BinaryHeap<Candidate> = seeds
        .iter()
        .filter(|s| hood.nodes.get(&s.note_id).is_some_and(|n| n.reachable))
        .map(|s| Candidate {
            score: s.affinity,
            node: s.note_id,
            hops: 0,
            via: None,
        })
        .collect();
    let mut taken: HashSet<Uuid> = HashSet::new();
    let mut selected: Vec<ShadowSelection> = Vec::new();
    let mut rejected: Vec<Candidate> = Vec::new();
    let mut tokens_used = 0usize;
    let mut stopped_by = StopReason::Exhausted;

    while let Some(c) = heap.pop() {
        if taken.contains(&c.node) {
            continue;
        }
        if c.score < config.score_threshold {
            stopped_by = StopReason::BelowThreshold;
            rejected.push(c);
            rejected.extend(heap.drain());
            break;
        }
        let Some(node) = hood.nodes.get(&c.node) else {
            continue;
        };
        if tokens_used + node.tokens > config.token_budget {
            stopped_by = StopReason::TokenBudget;
            continue;
        }
        taken.insert(c.node);
        tokens_used += node.tokens;
        selected.push(ShadowSelection {
            note_id: c.node,
            score: c.score,
            hops: c.hops,
            via: c.via,
            tokens: node.tokens,
            explored: false,
            created_at: node.created_at,
        });
        if c.hops >= config.max_hops {
            continue;
        }
        for (to, weight) in hood.edges.get(&c.node).map(Vec::as_slice).unwrap_or(&[]) {
            if taken.contains(to) || *weight < config.dormant_weight {
                continue;
            }
            let Some(target) = hood.nodes.get(to) else {
                continue;
            };
            if !target.reachable {
                continue;
            }
            let score = c.score
                * weight
                * target.energy.clamp(0.0, 1.0)
                * (1.0 - 0.7 * target.scar_intensity.clamp(0.0, 1.0));
            heap.push(Candidate {
                score,
                node: *to,
                hops: c.hops + 1,
                via: Some(c.node),
            });
        }
    }

    if draw < config.exploration_rate {
        rejected.sort_by(|a, b| b.cmp(a));
        let pick = rejected.iter().find(|c| {
            !taken.contains(&c.node)
                && c.score > 0.0
                && hood
                    .nodes
                    .get(&c.node)
                    .is_some_and(|n| tokens_used + n.tokens <= config.token_budget)
        });
        if let Some(c) = pick.copied() {
            let node = hood.nodes[&c.node];
            tokens_used += node.tokens;
            selected.push(ShadowSelection {
                note_id: c.node,
                score: c.score,
                hops: c.hops,
                via: c.via,
                tokens: node.tokens,
                explored: true,
                created_at: node.created_at,
            });
        }
    }

    Expansion {
        selected,
        tokens_used,
        stopped_by,
    }
}

// ============================================================================
// Driver (graph I/O)
// ============================================================================

/// Load the neighbourhood of `seeds` up to `max_hops` (one read per hop), in
/// the project only. Seeds' own metadata comes with them.
pub async fn load_hood(
    graph: &dyn GraphStore,
    project_id: Uuid,
    seeds: &[(Uuid, ShadowNode)],
    max_hops: usize,
) -> Result<ShadowHood> {
    let mut hood = ShadowHood::default();
    for (id, node) in seeds {
        hood.nodes.insert(*id, *node);
    }
    let mut frontier: Vec<Uuid> = seeds.iter().map(|(id, _)| *id).collect();
    frontier.sort();
    let mut expanded: HashSet<Uuid> = HashSet::new();
    for _ in 0..max_hops {
        frontier.retain(|id| expanded.insert(*id));
        if frontier.is_empty() || hood.nodes.len() >= MAX_LOADED_NODES {
            break;
        }
        let rows = graph.shadow_synapse_edges(project_id, &frontier).await?;
        let mut by_from: BTreeMap<Uuid, Vec<ShadowEdgeRow>> = BTreeMap::new();
        for row in rows {
            by_from.entry(row.from).or_default().push(row);
        }
        let mut next: Vec<Uuid> = Vec::new();
        for (from, mut out) in by_from {
            out.sort_by(|a, b| {
                b.weight
                    .partial_cmp(&a.weight)
                    .unwrap_or(Ordering::Equal)
                    .then(a.to.cmp(&b.to))
            });
            out.truncate(MAX_EDGES_PER_NODE);
            let edges = hood.edges.entry(from).or_default();
            for row in out {
                if hood.nodes.len() >= MAX_LOADED_NODES && !hood.nodes.contains_key(&row.to) {
                    continue;
                }
                hood.nodes.entry(row.to).or_insert(row.target);
                edges.push((row.to, row.weight));
                next.push(row.to);
            }
        }
        next.sort();
        next.dedup();
        frontier = next;
    }
    Ok(hood)
}

fn node_of(note: &crate::notes::Note) -> ShadowNode {
    use crate::notes::NoteStatus;
    ShadowNode {
        energy: note.energy,
        scar_intensity: note.scar_intensity,
        reachable: matches!(note.status, NoteStatus::Active | NoteStatus::NeedsReview),
        tokens: estimate_tokens(note.content.chars().count()),
        created_at: Some(note.created_at),
    }
}

fn entity_of(t: AnchorTargetType) -> Option<crate::notes::EntityType> {
    use crate::notes::EntityType as E;
    Some(match t {
        AnchorTargetType::File => E::File,
        AnchorTargetType::Function => E::Function,
        AnchorTargetType::Component => E::Component,
        AnchorTargetType::Feature => E::FeatureGraph,
        AnchorTargetType::Release => E::Release,
        AnchorTargetType::Plan => E::Plan,
        AnchorTargetType::Task => E::Task,
        AnchorTargetType::Decision => E::Decision,
        AnchorTargetType::Note | AnchorTargetType::Project | AnchorTargetType::Workspace => {
            return None
        }
    })
}

/// The seeds of a session: its anchors' notes in `project_id`, top-K.
pub async fn session_seeds(
    graph: &dyn GraphStore,
    session_id: Uuid,
    project_id: Uuid,
) -> Result<(Vec<ShadowSeed>, HashMap<Uuid, ShadowNode>)> {
    let anchors = graph.list_session_anchors(session_id).await?;
    let mut nodes: HashMap<Uuid, ShadowNode> = HashMap::new();
    let mut per_anchor: Vec<AnchorNotes> = Vec::new();
    for anchor in anchors.iter().filter(|a| seeds_from(a)) {
        let weight = role_weight(anchor) * anchor.confidence.clamp(0.0, 1.0);
        let mut notes = Vec::new();
        if anchor.target_type == AnchorTargetType::Note {
            if let Ok(id) = anchor.target_id.parse::<Uuid>() {
                if let Some(note) = graph.get_note(id).await? {
                    if note.project_id == Some(project_id) {
                        nodes.insert(note.id, node_of(&note));
                        notes.push((note.id, true));
                    }
                }
            }
        } else if let Some(entity) = entity_of(anchor.target_type) {
            for note in graph
                .get_notes_for_entity(&entity, &anchor.target_id)
                .await?
            {
                if note.project_id == Some(project_id) {
                    nodes.insert(note.id, node_of(&note));
                    notes.push((note.id, false));
                }
            }
        }
        per_anchor.push(AnchorNotes {
            anchor_weight: weight,
            notes,
        });
    }
    let seeds = anchor_affinity(&per_anchor, ANCHOR_AFFINITY_TOP_K);
    nodes.retain(|id, _| seeds.iter().any(|s| s.note_id == *id));
    Ok((seeds, nodes))
}

/// What one shadow turn needs to know about its session.
#[derive(Debug, Clone, Copy)]
pub struct TurnInput {
    pub session_id: Uuid,
    pub project_id: Uuid,
    pub session_started_at: DateTime<Utc>,
    pub now: DateTime<Utc>,
    /// Index of the turn, for the exploration draw.
    pub turn: u64,
}

/// Compute, journal and persist the expansion of one turn. Never touches what
/// the session receives. `Ok(None)`: the layer is off for this project.
pub async fn run_turn(
    graph: &dyn GraphStore,
    flags: &SynapticLayerFlags,
    input: TurnInput,
) -> Result<Option<SynapseShadowTrace>> {
    let requested = flags.requested(input.project_id);
    let effective = requested.effective();
    if effective == SynapticLayerMode::Off {
        return Ok(None);
    }
    let config = ExpansionConfig::default();
    let (seeds, seed_nodes) = session_seeds(graph, input.session_id, input.project_id).await?;
    let mut ordered: Vec<(Uuid, ShadowNode)> = seed_nodes.into_iter().collect();
    ordered.sort_by_key(|(id, _)| *id);
    let hood = load_hood(graph, input.project_id, &ordered, config.max_hops).await?;
    let draw = unit_draw(exploration_seed(input.session_id, input.turn));
    let expansion = best_first_expand(&seeds, &hood, &config, draw);
    let trace = SynapseShadowTrace {
        id: Uuid::new_v4(),
        schema_version: TRACE_SCHEMA_VERSION,
        layer_version: LAYER_VERSION.to_string(),
        session_id: input.session_id,
        project_id: input.project_id,
        requested_mode: requested,
        effective_mode: effective,
        session_started_at: input.session_started_at,
        created_at: input.now,
        seeds,
        selected: expansion.selected,
        tokens_used: expansion.tokens_used,
        token_budget: config.token_budget,
        stopped_by: expansion.stopped_by,
        weights_fingerprint: hood.weights_fingerprint(),
        config_fingerprint: config.fingerprint(),
        edges_read: hood.edge_count(),
        rewarded: Vec::new(),
    };
    graph
        .record_synapse_shadow_trace(&trace, TRACES_KEPT_PER_SESSION)
        .await?;
    journal(&trace);
    Ok(Some(trace))
}

/// One JSON line, target `synapse_shadow`.
fn journal(trace: &SynapseShadowTrace) {
    match serde_json::to_string(trace) {
        Ok(json) => tracing::info!(target: "synapse_shadow", "{json}"),
        Err(e) => tracing::warn!(target: "synapse_shadow", error = %e, "trace not serializable"),
    }
}

/// Apply an external signal to some traces. Each trace is rewarded at most once
/// per signal key (claimed atomically before writing), only when the layer is
/// on for its project. Returns the number of shadow edge weights written.
pub async fn apply_signal(
    graph: &dyn GraphStore,
    flags: &SynapticLayerFlags,
    traces: &[SynapseShadowTrace],
    signal: ExternalSignal,
    now: DateTime<Utc>,
) -> Result<usize> {
    let mut written = 0usize;
    for trace in traces {
        if flags.effective(trace.project_id) == SynapticLayerMode::Off {
            continue;
        }
        let key = signal.key();
        if trace.rewarded.contains(&key) {
            continue;
        }
        let pairs = rewarded_pairs(trace, &signal);
        if !graph.claim_shadow_trace_reward(trace.id, &key).await? {
            continue;
        }
        let n = if pairs.is_empty() {
            0
        } else {
            graph
                .reward_shadow_synapses(trace.project_id, &pairs, signal.delta(), now)
                .await?
        };
        tracing::info!(
            target: "synapse_shadow",
            trace_id = %trace.id,
            session_id = %trace.session_id,
            signal = signal.label(),
            delta = signal.delta(),
            pairs = pairs.len(),
            edges = n,
            "shadow reward"
        );
        written += n;
    }
    Ok(written)
}

/// The runner's verifier judged the attempt of `session_id`.
pub async fn reward_verification(
    graph: &dyn GraphStore,
    flags: &SynapticLayerFlags,
    session_id: Uuid,
    passed: bool,
    now: DateTime<Utc>,
) -> Result<usize> {
    let traces = graph
        .list_synapse_shadow_traces(&ShadowTraceFilter::Session(session_id))
        .await?;
    apply_signal(
        graph,
        flags,
        &traces,
        ExternalSignal::Verification { passed },
        now,
    )
    .await
}

/// Most recent traces a human confirmation of a note reaches.
pub const HUMAN_CONFIRMATION_TRACES: usize = 20;

/// A signed-in person confirmed `note_id`.
pub async fn reward_human_confirmation(
    graph: &dyn GraphStore,
    flags: &SynapticLayerFlags,
    note_id: Uuid,
    now: DateTime<Utc>,
) -> Result<usize> {
    let traces = graph
        .list_synapse_shadow_traces(&ShadowTraceFilter::ContainingNote {
            note_id,
            limit: HUMAN_CONFIRMATION_TRACES,
        })
        .await?;
    apply_signal(
        graph,
        flags,
        &traces,
        ExternalSignal::HumanConfirmed { note_id },
        now,
    )
    .await
}

/// Judge the traces whose attempt closed more than [`CLOSURE_QUIET_HOURS`] ago:
/// a later attempt of the task = restarted (negative), else completed = quiet
/// (positive). Returns `(quiet, restarted)` traces judged.
pub async fn settle_closures(
    graph: &dyn GraphStore,
    flags: &SynapticLayerFlags,
    now: DateTime<Utc>,
    limit: usize,
) -> Result<(usize, usize)> {
    let cutoff = now - chrono::Duration::hours(CLOSURE_QUIET_HOURS);
    let closures = graph.shadow_trace_closures(cutoff, limit).await?;
    let (mut quiet, mut restarted) = (0usize, 0usize);
    for (trace_id, closure) in closures {
        let Some(trace) = graph.get_synapse_shadow_trace(trace_id).await? else {
            continue;
        };
        apply_signal(graph, flags, &[trace], closure.signal(), now).await?;
        match closure {
            TaskClosure::Quiet => quiet += 1,
            TaskClosure::Restarted => restarted += 1,
        }
    }
    Ok((quiet, restarted))
}

/// One decay pass over the shadow weights of the projects the flags enable.
pub async fn decay(
    graph: &dyn GraphStore,
    flags: &SynapticLayerFlags,
    now: DateTime<Utc>,
) -> Result<usize> {
    if !flags.any_enabled() {
        return Ok(0);
    }
    graph
        .decay_shadow_synapses(&flags.decay_scope(), shadow_half_life_days(), now)
        .await
}

// ============================================================================
// Detached entry points (production)
// ============================================================================

/// Last turn journaled per session, and the turn counter.
static TURNS: LazyLock<Mutex<HashMap<Uuid, (tokio::time::Instant, u64)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const TURNS_CAPACITY: usize = 1024;

/// Whether a turn of `session_id` is due now (at most one per
/// [`TURN_RECHECK_SECS`]); returns its index. Uses tokio's clock (testable
/// with paused time).
pub fn claim_turn(session_id: Uuid) -> Option<u64> {
    let now = tokio::time::Instant::now();
    let mut turns = TURNS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, n)) = turns.get_mut(&session_id) {
        if now.duration_since(*at) < std::time::Duration::from_secs(TURN_RECHECK_SECS) {
            return None;
        }
        *at = now;
        *n += 1;
        return Some(*n);
    }
    if turns.len() >= TURNS_CAPACITY {
        turns.clear();
    }
    turns.insert(session_id, (now, 0));
    Some(0)
}

/// Run a shadow turn beside the session's turn: detached, never awaited by the
/// turn, errors swallowed (a warning). Nothing is started when the layer is off
/// for the project or the session journaled less than a minute ago.
pub fn spawn_turn(
    graph: Arc<dyn GraphStore>,
    session_id: Uuid,
    project_id: Uuid,
    session_started_at: DateTime<Utc>,
) {
    let flags = SynapticLayerFlags::global();
    if flags.effective(project_id) == SynapticLayerMode::Off {
        return;
    }
    let Some(turn) = claim_turn(session_id) else {
        return;
    };
    tokio::spawn(async move {
        let input = TurnInput {
            session_id,
            project_id,
            session_started_at,
            now: Utc::now(),
            turn,
        };
        let run = run_turn(graph.as_ref(), flags, input);
        match tokio::time::timeout(std::time::Duration::from_secs(3), run).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                tracing::warn!(target: "synapse_shadow", session_id = %session_id, error = %e, "shadow turn failed (ignored)")
            }
            Err(_) => {
                tracing::warn!(target: "synapse_shadow", session_id = %session_id, "shadow turn timed out (ignored)")
            }
        }
    });
}

/// Detached [`reward_verification`] (a no-op when the layer is off everywhere).
pub fn spawn_verification_reward(graph: Arc<dyn GraphStore>, session_id: Uuid, passed: bool) {
    let flags = SynapticLayerFlags::global();
    if !flags.any_enabled() {
        return;
    }
    tokio::spawn(async move {
        if let Err(e) =
            reward_verification(graph.as_ref(), flags, session_id, passed, Utc::now()).await
        {
            tracing::warn!(target: "synapse_shadow", session_id = %session_id, error = %e, "verification reward failed (ignored)");
        }
    });
}

/// Detached [`reward_human_confirmation`] (a no-op when the layer is off everywhere).
pub fn spawn_human_confirmation_reward(graph: Arc<dyn GraphStore>, note_id: Uuid) {
    let flags = SynapticLayerFlags::global();
    if !flags.any_enabled() {
        return;
    }
    tokio::spawn(async move {
        if let Err(e) = reward_human_confirmation(graph.as_ref(), flags, note_id, Utc::now()).await
        {
            tracing::warn!(target: "synapse_shadow", note_id = %note_id, error = %e, "human confirmation reward failed (ignored)");
        }
    });
}

#[cfg(test)]
#[path = "synaptic_shadow_tests.rs"]
mod tests;
