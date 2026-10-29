//! Tests of the synaptic shadow layer. Every clock is injected (`now`
//! parameters, paused tokio time): no test waits on the wall clock.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::*;
use crate::chat::anchor::{AnchorActor, AnchorOp, NewAnchor};
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::TaskStatus;
#[allow(unused_imports)]
use crate::notes::{Note, NoteType};
use chrono::TimeZone;

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 1, 12, 0, 0).unwrap()
}

fn id(n: u128) -> Uuid {
    Uuid::from_u128(n)
}

fn node(tokens: usize) -> ShadowNode {
    ShadowNode {
        energy: 1.0,
        scar_intensity: 0.0,
        reachable: true,
        tokens,
        created_at: Some(t0() - chrono::Duration::days(30)),
    }
}

fn seed(n: u128, affinity: f64) -> ShadowSeed {
    ShadowSeed {
        note_id: id(n),
        affinity,
    }
}

fn no_explore() -> ExpansionConfig {
    ExpansionConfig {
        exploration_rate: 0.0,
        ..ExpansionConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Decay
// ---------------------------------------------------------------------------

#[test]
fn a_weight_of_one_reaches_the_dormant_threshold_in_eleven_days() {
    let h = shadow_half_life_days();
    assert!((h - 11.0 / 20f64.log2()).abs() < 1e-12);
    let w = proportional_decay(1.0, 11.0 * 86_400.0, h);
    assert!((w - DORMANT_WEIGHT).abs() < 1e-9, "{w}");
    // Proportional: a weight of 0.5 loses the same FRACTION, not the same amount.
    let half = proportional_decay(0.5, 11.0 * 86_400.0, h);
    assert!((half - 0.5 * DORMANT_WEIGHT).abs() < 1e-9);
}

#[test]
fn the_decay_depends_on_elapsed_time_not_on_the_number_of_passes() {
    let h = shadow_half_life_days();
    let once = proportional_decay(0.8, 6.0 * 3600.0, h);
    let mut many = 0.8;
    for _ in 0..6 {
        many = proportional_decay(many, 3600.0, h);
    }
    assert!((once - many).abs() < 1e-12);
    // A clock that went backwards changes nothing.
    assert_eq!(proportional_decay(0.8, -10.0, h), 0.8);
    assert_eq!(proportional_decay(0.8, 0.0, h), 0.8);
}

#[test]
fn the_legacy_cadence_puts_a_co_activation_synapse_at_the_threshold_in_about_eleven_days() {
    // Measured from the code: SynapseDecayCheck runs every 6 h and subtracts
    // 0.02 from a co-activation synapse, pruned under 0.05.
    let cycles = (1.0_f64 - 0.05) / 0.02;
    let days = cycles * 6.0 / 24.0;
    assert!((days - 11.875).abs() < 1e-9);
    assert!((days - DAYS_TO_DORMANT).abs() < 1.0);
}

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

#[test]
fn the_flags_default_to_off_and_are_set_per_project() {
    let off = SynapticLayerFlags::parse(None, None);
    assert_eq!(off.effective(id(1)), SynapticLayerMode::Off);
    assert!(!off.any_enabled());

    let p = id(7);
    let q = id(8);
    let flags =
        SynapticLayerFlags::parse(Some("off"), Some(&format!("{p}=shadow, {q}=on, junk, x=y")));
    assert_eq!(flags.effective(p), SynapticLayerMode::Shadow);
    // `on` is accepted, recorded as asked, and runs as shadow.
    assert_eq!(flags.requested(q), SynapticLayerMode::On);
    assert_eq!(flags.effective(q), SynapticLayerMode::Shadow);
    assert_eq!(flags.effective(id(9)), SynapticLayerMode::Off);
    assert!(flags.any_enabled());
    let scope = flags.decay_scope();
    assert!(scope.includes(p) && scope.includes(q) && !scope.includes(id(9)));

    let all_but = SynapticLayerFlags::parse(Some("shadow"), Some(&format!("{p}=off")));
    assert!(!all_but.decay_scope().includes(p));
    assert!(all_but.decay_scope().includes(q));
    // An unknown default never turns the layer on.
    assert!(!SynapticLayerFlags::parse(Some("yes"), None).any_enabled());
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

#[test]
fn every_signal_has_a_sign_and_closure_is_judged_once() {
    assert!(ExternalSignal::Verification { passed: true }.delta() > 0.0);
    assert!(ExternalSignal::Verification { passed: false }.delta() < 0.0);
    assert!(ExternalSignal::HumanConfirmed { note_id: id(1) }.delta() > 0.0);
    assert!(ExternalSignal::TaskClosedQuietly.delta() > 0.0);
    assert!(ExternalSignal::TaskRestarted.delta() < 0.0);
    assert_eq!(
        ExternalSignal::TaskClosedQuietly.key(),
        ExternalSignal::TaskRestarted.key()
    );
    assert_ne!(
        ExternalSignal::HumanConfirmed { note_id: id(1) }.key(),
        ExternalSignal::HumanConfirmed { note_id: id(2) }.key()
    );
}

fn trace_with(selected: Vec<ShadowSelection>, started: DateTime<Utc>) -> SynapseShadowTrace {
    SynapseShadowTrace {
        id: Uuid::new_v4(),
        schema_version: TRACE_SCHEMA_VERSION,
        layer_version: LAYER_VERSION.to_string(),
        session_id: id(100),
        project_id: id(200),
        requested_mode: SynapticLayerMode::Shadow,
        effective_mode: SynapticLayerMode::Shadow,
        session_started_at: started,
        created_at: started,
        seeds: Vec::new(),
        selected,
        tokens_used: 0,
        token_budget: TOKEN_BUDGET,
        stopped_by: StopReason::Exhausted,
        weights_fingerprint: String::new(),
        config_fingerprint: String::new(),
        edges_read: 0,
        rewarded: Vec::new(),
    }
}

fn sel(n: u128, via: Option<u128>, created_at: Option<DateTime<Utc>>) -> ShadowSelection {
    ShadowSelection {
        note_id: id(n),
        score: 0.5,
        hops: usize::from(via.is_some()),
        via: via.map(id),
        tokens: 10,
        explored: false,
        created_at,
    }
}

#[test]
fn only_followed_edges_between_notes_older_than_the_conversation_are_rewarded() {
    let started = t0();
    let old = Some(started - chrono::Duration::days(3));
    let during = Some(started + chrono::Duration::minutes(5));
    let trace = trace_with(
        vec![
            sel(1, None, old),
            sel(2, Some(1), old),
            sel(3, Some(2), during), // created in the conversation
            sel(4, Some(1), None),   // unknown creation date
            sel(5, Some(2), old),
        ],
        started,
    );
    let pairs = rewarded_pairs(&trace, &ExternalSignal::Verification { passed: true });
    assert_eq!(pairs, vec![(id(1), id(2)), (id(2), id(5))]);
    // A human confirmation of 5 only reaches the followed edges that touch 5.
    let human = rewarded_pairs(&trace, &ExternalSignal::HumanConfirmed { note_id: id(5) });
    assert_eq!(human, vec![(id(2), id(5))]);
    // A note created exactly when the session started is not older than it.
    assert!(!rewardable(Some(started), started));
}

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------

#[test]
fn the_anchor_affinity_keeps_the_best_score_per_note_and_the_top_twenty() {
    let focus = AnchorNotes {
        anchor_weight: 1.0,
        notes: (0..30).map(|n| (id(n), n == 0)).collect(),
    };
    let mention = AnchorNotes {
        anchor_weight: 0.6,
        notes: vec![(id(0), false), (id(99), true)],
    };
    let seeds = anchor_affinity(&[focus, mention], ANCHOR_AFFINITY_TOP_K);
    assert_eq!(seeds.len(), 20);
    assert_eq!(seeds[0], seed(0, 1.0)); // direct target of the focus
    assert!((seeds[1].affinity - LINKED_NOTE_FACTOR).abs() < 1e-12);
    // Ties are ordered by id: the same input always gives the same seeds.
    assert_eq!(seeds[1].note_id, id(1));
    // 99 (0.6) loses against the 29 linked notes at 0.8.
    assert!(seeds.iter().all(|s| s.note_id != id(99)));
}

fn anchor(target: AnchorTargetType, roles: &[AnchorRole], state: AnchorState) -> Anchor {
    Anchor {
        id: Uuid::new_v4(),
        session_id: id(1),
        target_type: target,
        target_id: "x".into(),
        roles: roles.iter().copied().collect(),
        state,
        by: AnchorActor::User,
        inferred: false,
        confidence: 1.0,
        snapshot_name: None,
        snapshot_path: None,
        snapshot_type: None,
        rev: None,
        version: 1,
        created_at: t0(),
        updated_at: t0(),
    }
}

#[test]
fn a_project_or_broken_anchor_seeds_nothing_and_focus_outweighs_mention() {
    assert!(!seeds_from(&anchor(
        AnchorTargetType::Project,
        &[AnchorRole::Focus],
        AnchorState::Live
    )));
    assert!(!seeds_from(&anchor(
        AnchorTargetType::File,
        &[AnchorRole::Focus],
        AnchorState::Dangling
    )));
    assert!(seeds_from(&anchor(
        AnchorTargetType::File,
        &[AnchorRole::Mention],
        AnchorState::Moved
    )));
    let both = anchor(
        AnchorTargetType::Note,
        &[AnchorRole::Mention, AnchorRole::Focus],
        AnchorState::Live,
    );
    assert_eq!(role_weight(&both), 1.0);
    assert_eq!(
        role_weight(&anchor(
            AnchorTargetType::Note,
            &[AnchorRole::Origin],
            AnchorState::Live
        )),
        0.3
    );
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

fn chain(len: u128, weight: f64, tokens: usize) -> ShadowHood {
    let mut hood = ShadowHood::default();
    for n in 1..=len {
        hood.nodes.insert(id(n), node(tokens));
        if n < len {
            hood.edges.insert(id(n), vec![(id(n + 1), weight)]);
        }
    }
    hood
}

#[test]
fn the_expansion_stops_at_three_hops() {
    let hood = chain(6, 1.0, 10);
    let out = best_first_expand(&[seed(1, 1.0)], &hood, &no_explore(), 0.99);
    let hops: Vec<usize> = out.selected.iter().map(|s| s.hops).collect();
    assert_eq!(hops, vec![0, 1, 2, 3]);
    assert_eq!(out.selected[3].via, Some(id(3)));
    assert_eq!(out.stopped_by, StopReason::Exhausted);
}

#[test]
fn the_expansion_stops_under_the_score_threshold() {
    // 1.0 → 0.5 → 0.25 → 0.125 (< 0.15).
    let hood = chain(4, 0.5, 10);
    let out = best_first_expand(&[seed(1, 1.0)], &hood, &no_explore(), 0.99);
    assert_eq!(out.selected.len(), 3);
    assert_eq!(out.stopped_by, StopReason::BelowThreshold);
}

#[test]
fn tokens_come_first_a_big_note_is_skipped_and_a_smaller_one_still_fits() {
    let mut hood = ShadowHood::default();
    hood.nodes.insert(id(1), node(700));
    hood.nodes.insert(id(2), node(600)); // best neighbour, does not fit
    hood.nodes.insert(id(3), node(400)); // weaker, fits
    hood.edges.insert(id(1), vec![(id(2), 0.9), (id(3), 0.4)]);
    let config = ExpansionConfig {
        token_budget: 1200,
        ..no_explore()
    };
    let out = best_first_expand(&[seed(1, 1.0)], &hood, &config, 0.99);
    let taken: Vec<Uuid> = out.selected.iter().map(|s| s.note_id).collect();
    assert_eq!(taken, vec![id(1), id(3)]);
    assert_eq!(out.tokens_used, 1100);
    assert!(out.tokens_used <= config.token_budget);
}

#[test]
fn a_dormant_edge_or_an_unreachable_note_is_not_followed() {
    let mut hood = ShadowHood::default();
    hood.nodes.insert(id(1), node(10));
    hood.nodes.insert(id(2), node(10));
    hood.nodes.insert(
        id(3),
        ShadowNode {
            reachable: false,
            ..node(10)
        },
    );
    hood.edges
        .insert(id(1), vec![(id(2), DORMANT_WEIGHT / 2.0), (id(3), 1.0)]);
    let out = best_first_expand(&[seed(1, 1.0)], &hood, &no_explore(), 0.99);
    assert_eq!(out.selected.len(), 1);
}

#[test]
fn the_exploration_takes_the_best_rejected_candidate_only_when_drawn() {
    let mut hood = ShadowHood::default();
    hood.nodes.insert(id(1), node(10));
    hood.nodes.insert(id(2), node(10));
    hood.nodes.insert(id(3), node(10));
    hood.edges.insert(id(1), vec![(id(2), 0.10), (id(3), 0.12)]);
    let config = ExpansionConfig::default();
    let drawn = best_first_expand(&[seed(1, 1.0)], &hood, &config, 0.05);
    assert_eq!(drawn.selected.len(), 2);
    assert!(drawn.selected[1].explored);
    assert_eq!(drawn.selected[1].note_id, id(3));
    let not_drawn = best_first_expand(&[seed(1, 1.0)], &hood, &config, 0.10);
    assert_eq!(not_drawn.selected.len(), 1);
}

#[test]
fn the_exploration_draw_is_deterministic_and_close_to_ten_percent() {
    let session = id(0xABCD);
    assert_eq!(
        unit_draw(exploration_seed(session, 3)),
        unit_draw(exploration_seed(session, 3))
    );
    let hits = (0..10_000u64)
        .filter(|turn| unit_draw(exploration_seed(session, *turn)) < EXPLORATION_RATE)
        .count();
    assert!((800..=1200).contains(&hits), "{hits}");
}

#[test]
fn the_weights_fingerprint_changes_with_a_weight_and_not_with_the_order() {
    let mut a = ShadowHood::default();
    a.edges.insert(id(1), vec![(id(2), 0.4), (id(3), 0.5)]);
    a.edges.insert(id(2), vec![(id(3), 0.6)]);
    let mut b = ShadowHood::default();
    b.edges.insert(id(2), vec![(id(3), 0.6)]);
    b.edges.insert(id(1), vec![(id(3), 0.5), (id(2), 0.4)]);
    assert_eq!(a.weights_fingerprint(), b.weights_fingerprint());
    b.edges.insert(id(2), vec![(id(3), 0.61)]);
    assert_ne!(a.weights_fingerprint(), b.weights_fingerprint());
    assert_eq!(
        ExpansionConfig::default().fingerprint(),
        ExpansionConfig::default().fingerprint()
    );
}

// ---------------------------------------------------------------------------
// Driver on the mock store
// ---------------------------------------------------------------------------

struct Fixture {
    graph: MockGraphStore,
    project: Uuid,
    session: Uuid,
    started: DateTime<Utc>,
    notes: Vec<Uuid>,
}

async fn note_in(graph: &MockGraphStore, project: Uuid, created_at: DateTime<Utc>) -> Uuid {
    let mut n = Note::new(
        Some(project),
        NoteType::Guideline,
        "a short note".into(),
        "api".into(),
    );
    n.created_at = created_at;
    graph.create_note(&n).await.unwrap();
    n.id
}

async fn link(graph: &MockGraphStore, a: Uuid, b: Uuid, w: f64) {
    let mut syn = graph.note_synapses.write().await;
    syn.entry(a).or_default().push((b, w));
    syn.entry(b).or_default().push((a, w));
}

/// Notes 0 → 1 → 2 in the project (0 anchored as focus), 3 in another project
/// linked to 1, and 4 created during the conversation, linked to 0.
async fn fixture() -> Fixture {
    let graph = MockGraphStore::new();
    let project = Uuid::new_v4();
    let other = Uuid::new_v4();
    let started = t0();
    let old = started - chrono::Duration::days(10);
    let mut notes = Vec::new();
    for _ in 0..3 {
        notes.push(note_in(&graph, project, old).await);
    }
    notes.push(note_in(&graph, other, old).await);
    notes.push(note_in(&graph, project, started + chrono::Duration::minutes(1)).await);
    link(&graph, notes[0], notes[1], 0.8).await;
    link(&graph, notes[1], notes[2], 0.6).await;
    link(&graph, notes[1], notes[3], 0.9).await;
    link(&graph, notes[0], notes[4], 0.9).await;

    let mut session = crate::test_helpers::test_chat_session(None);
    session.created_at = started;
    graph.create_chat_session(&session).await.unwrap();
    graph
        .apply_anchor_op(
            session.id,
            AnchorOp::Add(NewAnchor::new(
                AnchorTargetType::Note,
                notes[0].to_string(),
                [AnchorRole::Focus],
                AnchorActor::User,
                "user",
            )),
        )
        .await
        .unwrap();
    Fixture {
        graph,
        project,
        session: session.id,
        started,
        notes,
    }
}

fn shadow_for(project: Uuid) -> SynapticLayerFlags {
    SynapticLayerFlags::parse(Some("off"), Some(&format!("{project}=shadow")))
}

fn input(f: &Fixture, now: DateTime<Utc>) -> TurnInput {
    TurnInput {
        session_id: f.session,
        project_id: f.project,
        session_started_at: f.started,
        now,
        turn: 1,
    }
}

async fn legacy_weights(graph: &MockGraphStore) -> Vec<(Uuid, Uuid, u64)> {
    let syn = graph.note_synapses.read().await;
    let mut out: Vec<_> = syn
        .iter()
        .flat_map(|(a, v)| v.iter().map(move |(b, w)| (*a, *b, w.to_bits())))
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn off_computes_nothing_and_writes_nothing() {
    let f = fixture().await;
    let flags = SynapticLayerFlags::parse(None, None);
    assert!(run_turn(&f.graph, &flags, input(&f, f.started))
        .await
        .unwrap()
        .is_none());
    assert!(f.graph.synapse_shadow_traces.read().await.is_empty());
    assert_eq!(decay(&f.graph, &flags, f.started).await.unwrap(), 0);
    assert!(f.graph.synapse_shadow.read().await.is_empty());
}

#[tokio::test]
async fn a_shadow_turn_journals_the_expansion_and_never_crosses_the_project() {
    let f = fixture().await;
    let flags = shadow_for(f.project);
    let before = legacy_weights(&f.graph).await;
    let trace = run_turn(&f.graph, &flags, input(&f, f.started))
        .await
        .unwrap()
        .expect("shadow trace");
    let taken: HashSet<Uuid> = trace.note_ids().into_iter().collect();
    assert!(taken.contains(&f.notes[0]) && taken.contains(&f.notes[1]));
    assert!(taken.contains(&f.notes[2]));
    assert!(!taken.contains(&f.notes[3]), "other project reached");
    assert_eq!(trace.schema_version, TRACE_SCHEMA_VERSION);
    assert_eq!(trace.layer_version, LAYER_VERSION);
    assert_eq!(trace.weights_fingerprint.len(), 64);
    assert_eq!(trace.seeds, vec![seed_of(f.notes[0])]);
    // Persisted and readable, and the legacy layer is untouched.
    let stored = f
        .graph
        .list_synapse_shadow_traces(&ShadowTraceFilter::Session(f.session))
        .await
        .unwrap();
    assert_eq!(stored, vec![trace.clone()]);
    assert_eq!(legacy_weights(&f.graph).await, before);
    assert!(f.graph.synapse_shadow.read().await.is_empty());
    // Same graph, same weights: same fingerprint.
    let again = run_turn(&f.graph, &flags, input(&f, f.started))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(again.weights_fingerprint, trace.weights_fingerprint);
}

fn seed_of(note_id: Uuid) -> ShadowSeed {
    ShadowSeed {
        note_id,
        affinity: 1.0,
    }
}

async fn shadow_weight(graph: &MockGraphStore, a: Uuid, b: Uuid) -> Option<f64> {
    graph.synapse_shadow.read().await.get(&(a, b)).map(|s| s.0)
}

#[tokio::test]
async fn a_verification_rewards_the_followed_edges_once_and_never_the_legacy_weight() {
    let f = fixture().await;
    let flags = shadow_for(f.project);
    run_turn(&f.graph, &flags, input(&f, f.started))
        .await
        .unwrap();
    let before = legacy_weights(&f.graph).await;
    let now = f.started + chrono::Duration::hours(1);

    let written = reward_verification(&f.graph, &flags, f.session, true, now)
        .await
        .unwrap();
    // 0–1 and 1–2, both directions. 0–4 (4 created during the conversation) is not.
    assert_eq!(written, 4);
    let w01 = shadow_weight(&f.graph, f.notes[0], f.notes[1])
        .await
        .unwrap();
    assert!((w01 - 0.9).abs() < 1e-12, "{w01}");
    assert!(shadow_weight(&f.graph, f.notes[0], f.notes[4])
        .await
        .is_none());
    assert_eq!(legacy_weights(&f.graph).await, before);

    // Idempotent: the same signal on the same trace is applied once.
    assert_eq!(
        reward_verification(&f.graph, &flags, f.session, true, now)
            .await
            .unwrap(),
        0
    );
    // The layer off for the project: nothing, even for a fresh trace.
    let off = SynapticLayerFlags::parse(None, None);
    assert_eq!(
        reward_verification(&f.graph, &off, f.session, false, now)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn a_reward_never_creates_an_edge_and_never_crosses_a_project() {
    let f = fixture().await;
    // 0–2 has no edge; 1–3 crosses projects.
    let n = f
        .graph
        .reward_shadow_synapses(
            f.project,
            &[(f.notes[0], f.notes[2]), (f.notes[1], f.notes[3])],
            0.5,
            f.started,
        )
        .await
        .unwrap();
    assert_eq!(n, 0);
    assert!(f.graph.synapse_shadow.read().await.is_empty());
}

#[tokio::test]
async fn a_human_confirmation_rewards_the_edges_of_the_confirmed_note_only() {
    let f = fixture().await;
    let flags = shadow_for(f.project);
    run_turn(&f.graph, &flags, input(&f, f.started))
        .await
        .unwrap();
    let written = reward_human_confirmation(&f.graph, &flags, f.notes[2], f.started)
        .await
        .unwrap();
    assert_eq!(written, 2);
    assert!(shadow_weight(&f.graph, f.notes[1], f.notes[2])
        .await
        .is_some());
    assert!(shadow_weight(&f.graph, f.notes[0], f.notes[1])
        .await
        .is_none());
}

#[tokio::test]
async fn the_shadow_decay_is_proportional_to_injected_time_and_leaves_the_legacy_weight() {
    let f = fixture().await;
    let flags = shadow_for(f.project);
    let before = legacy_weights(&f.graph).await;
    // First pass: the shadow weight starts from the legacy weight, undecayed.
    decay(&f.graph, &flags, f.started).await.unwrap();
    assert_eq!(
        shadow_weight(&f.graph, f.notes[0], f.notes[1]).await,
        Some(0.8)
    );
    // Eleven days later, in two uneven passes.
    decay(&f.graph, &flags, f.started + chrono::Duration::days(4))
        .await
        .unwrap();
    decay(&f.graph, &flags, f.started + chrono::Duration::days(11))
        .await
        .unwrap();
    let w = shadow_weight(&f.graph, f.notes[0], f.notes[1])
        .await
        .unwrap();
    assert!((w - 0.8 * DORMANT_WEIGHT).abs() < 1e-9, "{w}");
    // The other project is not in scope: never touched.
    assert!(shadow_weight(&f.graph, f.notes[3], f.notes[1])
        .await
        .is_none());
    assert_eq!(legacy_weights(&f.graph).await, before);
}

async fn execution(
    f: &Fixture,
    session: Uuid,
    task: Uuid,
    started: DateTime<Utc>,
    completed: Option<DateTime<Utc>>,
) {
    let ae = crate::neo4j::agent_execution::AgentExecutionNode {
        id: Uuid::new_v4(),
        task_id: task,
        session_id: Some(session),
        started_at: started,
        completed_at: completed,
        ..Default::default()
    };
    f.graph.agent_executions.write().await.insert(ae.id, ae);
}

async fn task_with(f: &Fixture, status: TaskStatus) -> Uuid {
    let mut t = crate::test_helpers::test_task();
    t.status = status;
    let id = t.id;
    f.graph.tasks.write().await.insert(id, t);
    id
}

#[tokio::test]
async fn a_quiet_closure_rewards_and_a_restart_penalises_each_trace_once() {
    let f = fixture().await;
    let flags = shadow_for(f.project);
    run_turn(&f.graph, &flags, input(&f, f.started))
        .await
        .unwrap();
    let task = task_with(&f, TaskStatus::Completed).await;
    let closed = f.started + chrono::Duration::hours(1);
    execution(&f, f.session, task, f.started, Some(closed)).await;

    // Not yet quiet for 72 h: nothing is judged.
    let early = closed + chrono::Duration::hours(CLOSURE_QUIET_HOURS - 1);
    assert_eq!(
        settle_closures(&f.graph, &flags, early, 100).await.unwrap(),
        (0, 0)
    );

    let later = closed + chrono::Duration::hours(CLOSURE_QUIET_HOURS + 1);
    assert_eq!(
        settle_closures(&f.graph, &flags, later, 100).await.unwrap(),
        (1, 0)
    );
    let w = shadow_weight(&f.graph, f.notes[0], f.notes[1])
        .await
        .unwrap();
    assert!((w - 0.85).abs() < 1e-12, "{w}");
    // Judged once.
    assert_eq!(
        settle_closures(&f.graph, &flags, later, 100).await.unwrap(),
        (0, 0)
    );

    // Another session of the same project whose task was attempted again.
    let mut s2 = crate::test_helpers::test_chat_session(None);
    s2.created_at = f.started;
    f.graph.create_chat_session(&s2).await.unwrap();
    f.graph
        .apply_anchor_op(
            s2.id,
            AnchorOp::Add(NewAnchor::new(
                AnchorTargetType::Note,
                f.notes[0].to_string(),
                [AnchorRole::Focus],
                AnchorActor::User,
                "user",
            )),
        )
        .await
        .unwrap();
    let mut i2 = input(&f, f.started);
    i2.session_id = s2.id;
    run_turn(&f.graph, &flags, i2).await.unwrap();
    let task2 = task_with(&f, TaskStatus::Completed).await;
    execution(&f, s2.id, task2, f.started, Some(closed)).await;
    execution(&f, Uuid::new_v4(), task2, closed, None).await;
    assert_eq!(
        settle_closures(&f.graph, &flags, later, 100).await.unwrap(),
        (0, 1)
    );
    let w = shadow_weight(&f.graph, f.notes[0], f.notes[1])
        .await
        .unwrap();
    assert!((w - 0.75).abs() < 1e-12, "{w}");
}

#[tokio::test]
async fn a_session_keeps_its_latest_traces_only() {
    let f = fixture().await;
    let flags = shadow_for(f.project);
    for minute in 0..(TRACES_KEPT_PER_SESSION as i64 + 5) {
        run_turn(
            &f.graph,
            &flags,
            input(&f, f.started + chrono::Duration::minutes(minute)),
        )
        .await
        .unwrap();
    }
    let kept = f
        .graph
        .list_synapse_shadow_traces(&ShadowTraceFilter::Session(f.session))
        .await
        .unwrap();
    assert_eq!(kept.len(), TRACES_KEPT_PER_SESSION);
    assert_eq!(kept[0].created_at, f.started + chrono::Duration::minutes(5));
}

#[tokio::test(start_paused = true)]
async fn a_session_journals_at_most_one_turn_a_minute_on_tokio_time() {
    let session = Uuid::new_v4();
    assert_eq!(claim_turn(session), Some(0));
    assert_eq!(claim_turn(session), None);
    tokio::time::advance(std::time::Duration::from_secs(TURN_RECHECK_SECS - 1)).await;
    assert_eq!(claim_turn(session), None);
    tokio::time::advance(std::time::Duration::from_secs(2)).await;
    assert_eq!(claim_turn(session), Some(1));
}
