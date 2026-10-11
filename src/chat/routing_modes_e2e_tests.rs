//! The three routing modes, end to end (plan « routage cognitif », task INT).
//!
//! Everything here is the production stack except the model endpoints: the REAL cognitive
//! decider over the routing store (`Neo4jRoutingStore` on the mock graph), the REAL pool
//! (`routing_pool_for`, with its health and capability probes), the per-turn router wired
//! the way `wire_learning` does it, and native sessions on two `fake_openai` instances:
//!
//! - `primary` (model `pm`): the instance the global roles name for the pilot and the
//!   executor, i.e. today's behaviour;
//! - `worker` (models `wa` and `wb`): the instance the learned arms prefer.
//!
//! The arms are taught before each test (sixty observations each), so the Thompson draws
//! are decided by the evidence, never by luck: `wa` wins the plain and simple classes,
//! `wb` the debug class and the compaction summary, `pm` loses everywhere. What a test
//! asserts is where the TURNS went (requests seen by each fake; the capability probes the
//! pool sends to every instance are metadata, filtered out) and what the store recorded.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::broadcast;
use uuid::Uuid;

use super::agent_e2e_tests::{
    consent, delta, fake_bin, instance, request, sse_route, store_instance, FakeOpenAi,
};
use super::config::ProviderPath;
use super::manager::ChatManager;
use super::provider::cognitive::decider::CognitiveRouting;
use super::provider::cognitive::decision::{CognitiveDecision, Pick};
use super::provider::cognitive::signature::{ContextHints, TaskClass, TaskSignature};
use super::provider::cognitive::store::{ArmKey, ArmObservation, DecisionFilter, RoutingArmStore};
use super::provider::cognitive::wiring::ManagerPool;
use super::provider::cognitive::ROUTING_KEY;
use super::provider::settings::{GLOBAL, ROLES_KEY};
use super::types::{ChatEvent, ChatRequest, SessionRoutingRequest};
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::routing::Neo4jRoutingStore;
use crate::neo4j::GraphStore;
use crate::test_helpers::mock_app_state;

const SIMPLE: &str = "rename this variable";
const DEBUG: &str = "why does this crash with a stack trace error";
const PROBE: &str = "Call the ping tool now";
/// Text of the native compaction's summarisation prompt (nexus `SUMMARY_SYSTEM_PROMPT`).
const SUMMARY_PROMPT: &str = "You are compacting the history";
/// Turns of the compaction test, after the opening "hi there".
const LONG_TURNS: [&str; 4] = ["turn two", "turn three", "turn four", "turn five"];

fn answer(text: &str, prompt_tokens: u64) -> Vec<Value> {
    vec![
        delta(json!({ "content": text })),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 4, "total_tokens": prompt_tokens + 4}}),
        json!("[DONE]"),
    ]
}

/// One fake's routes. A later turn's request carries the earlier messages too, and the fake
/// answers with the first matching route not used yet (else the last matching one): the
/// later texts are listed first.
///
/// `who` signs the summary this fake writes: the turn after a compaction says which
/// endpoint wrote it. The request log of `fake_openai` can lose a line when two requests
/// are logged at once (the compaction decision reads the catalogues while the summary is
/// asked), so the summary is not looked for there; it answers after 300 ms so that the
/// decision's requests are over before the next turn is sent.
fn script(who: &str, models: &[&str]) -> Value {
    let mut routes = vec![
        sse_route(
            PROBE,
            vec![
                delta(
                    json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]}),
                ),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ],
        ),
        json!({"method": "GET", "path": "/v1/models", "status": 200, "body": {"object": "list",
               "data": models.iter().map(|m| json!({"id": m, "context_length": 32000})).collect::<Vec<_>>()}}),
        {
            let mut summary = sse_route(
                SUMMARY_PROMPT,
                answer(&format!("summary written by {who}"), 50),
            );
            summary["delay_ms"] = json!(300);
            summary
        },
    ];
    // The long conversation reports a prompt close to the 32k window: nexus compacts
    // before a turn once there is enough history to summarise.
    for text in LONG_TURNS.iter().rev() {
        routes.push(sse_route(text, answer("noted", 31_000)));
    }
    routes.push(sse_route(DEBUG, answer("it crashes because", 10)));
    routes.push(sse_route(SIMPLE, answer("renamed", 10)));
    routes.push(sse_route("hi there", answer("hello", 10)));
    Value::Array(routes)
}

struct World {
    primary: FakeOpenAi,
    worker: FakeOpenAi,
    graph: Arc<MockGraphStore>,
    manager: Arc<ChatManager>,
}

fn class_of(message: &str) -> String {
    TaskSignature::from_chat_request(message, false, Some("proj"), ContextHints::default())
        .arm_key()
}

fn executor_class() -> String {
    TaskSignature::from_delegation(
        Some("simple"),
        None,
        1,
        Some("proj"),
        ContextHints::default(),
    )
    .arm_key()
}

/// Sixty observations: `winner` succeeds every time, every other pair fails every time.
async fn teach(store: &Neo4jRoutingStore, class: &str, winner: (&str, &str)) {
    for (provider, model) in [("primary", "pm"), ("worker", "wa"), ("worker", "wb")] {
        let reward = if (provider, model) == winner {
            1.0
        } else {
            0.0
        };
        let key = ArmKey::new(class, provider, model);
        for _ in 0..60 {
            store
                .observe(
                    &key,
                    &ArmObservation {
                        reward,
                        success_threshold: 0.5,
                        cost_usd: None,
                        latency_ms: None,
                    },
                )
                .await
                .unwrap();
        }
    }
}

async fn world(mode: &str, stage: &str) -> World {
    let primary = FakeOpenAi::start(script("primary", &["pm"]));
    let worker = FakeOpenAi::start(script("worker", &["wa", "wb"]));
    let graph = Arc::new(MockGraphStore::new());
    for (fake, id, model) in [(&primary, "primary", "pm"), (&worker, "worker", "wa")] {
        let mut record = instance(fake, "none");
        record.id = id.into();
        record.label = id.into();
        record.default_model = Some(model.into());
        store_instance(&graph, &record).await;
        consent(&graph, "proj", id, &fake.origin()).await;
    }
    // Today's behaviour: the roles name the primary for everyone.
    let roles = json!({"pilot": {"provider": "primary"}, "executor": {"provider": "primary"}});
    graph
        .put_llm_setting(GLOBAL, ROLES_KEY, &roles.to_string())
        .await
        .unwrap();
    graph
        .put_llm_setting(
            GLOBAL,
            ROUTING_KEY,
            &json!({"mode": mode, "stage": stage, "exploration_epsilon": 0.0}).to_string(),
        )
        .await
        .unwrap();

    let store = Neo4jRoutingStore::new(graph.clone());
    // The plain opening and the simple turn may share a class: taught in that order, the
    // debug class and the compaction summary last.
    for class in [class_of("hi there"), class_of(SIMPLE), executor_class()] {
        teach(&store, &class, ("worker", "wa")).await;
    }
    assert_ne!(
        class_of(DEBUG),
        class_of(SIMPLE),
        "two classes to route between"
    );
    teach(&store, &class_of(DEBUG), ("worker", "wb")).await;
    teach(
        &store,
        &TaskSignature::utility(TaskClass::UtilityCompaction, 0, None).arm_key(),
        ("worker", "wb"),
    )
    .await;

    let config = super::config::ChatConfig {
        provider_path: ProviderPath::Agent,
        mcp_server_path: fake_bin("fake_mcp"),
        nexus_tools_path: None,
        nexus_browser_path: None,
        jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
        max_sessions: 10,
        ..Default::default()
    };
    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let manager = Arc::new(
        ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(NoBuiltin))
            .with_cognitive_routing(CognitiveRouting::new(Arc::new(Neo4jRoutingStore::new(
                graph.clone(),
            )))),
    );
    // `wire_learning` without the process-wide runner handle (other tests read that slot).
    let decider = Arc::clone(&manager.cognitive_routing().unwrap().decider);
    manager
        .turn_routing
        .configure(decider, Arc::new(ManagerPool(Arc::downgrade(&manager))));
    World {
        primary,
        worker,
        graph,
        manager,
    }
}

/// No built-in Claude Code: the two fakes are the only instances.
struct NoBuiltin;

impl super::agent_runtime::ProviderSource for NoBuiltin {
    fn get(&self, _provider_id: &str) -> Option<Arc<dyn nexus_claude::agent::AgentProvider>> {
        None
    }
}

fn body_of(request: &Value) -> Value {
    match &request["body"] {
        Value::String(raw) => serde_json::from_str(raw).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

/// The turns a fake answered (not the capability probes), as (model, body).
fn turns(fake: &FakeOpenAi) -> Vec<(String, String)> {
    fake.chat_requests()
        .iter()
        .map(body_of)
        .map(|body| {
            (
                body["model"].as_str().unwrap_or_default().to_owned(),
                body.to_string(),
            )
        })
        .filter(|(_, body)| !body.contains(PROBE))
        .collect()
}

/// The model of the one turn whose request carries `text` and nothing said after it.
fn model_of_turn(fake: &FakeOpenAi, text: &str, later: &[&str]) -> Vec<String> {
    turns(fake)
        .into_iter()
        .filter(|(_, body)| body.contains(text) && !later.iter().any(|l| body.contains(l)))
        .filter(|(_, body)| !body.contains(SUMMARY_PROMPT))
        .map(|(model, _)| model)
        .collect()
}

async fn wait_for(
    rx: &mut broadcast::Receiver<ChatEvent>,
    pred: impl Fn(&ChatEvent) -> bool,
) -> ChatEvent {
    loop {
        let event = tokio::time::timeout(Duration::from_secs(20), rx.recv())
            .await
            .expect("an event within 20 s")
            .expect("channel open");
        if pred(&event) {
            return event;
        }
    }
}

fn is_result(event: &ChatEvent) -> bool {
    matches!(event, ChatEvent::Result { .. })
}

/// Opens a session and waits for the result of its opening turn.
async fn open(w: &World, request: &ChatRequest) -> String {
    let created = w
        .manager
        .create_session(request)
        .await
        .unwrap_or_else(|e| panic!("open failed: {e:#}"));
    let mut rx = w.manager.subscribe(&created.session_id).await.unwrap();
    wait_for(&mut rx, is_result).await;
    idle(w, &created.session_id).await;
    created.session_id
}

async fn idle(w: &World, session_id: &str) {
    for _ in 0..400 {
        if !w.manager.is_session_streaming(session_id).await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("{session_id} never went idle");
}

/// Sends one message to a session that stays where it is, and waits for its result.
async fn turn(w: &World, session_id: &str, text: &str) {
    let mut rx = w.manager.subscribe(session_id).await.unwrap();
    w.manager.send_message(session_id, text).await.unwrap();
    wait_for(&mut rx, is_result).await;
    idle(w, session_id).await;
}

fn pilot() -> ChatRequest {
    request(None, Some("proj"), "default")
}

fn executor() -> ChatRequest {
    let mut r = request(None, Some("proj"), "default");
    r.spawned_by = Some("{}".into());
    r.task_class = Some("simple".into());
    r
}

/// The pool probes only an instance's DEFAULT model: `wb` is a candidate that can call
/// tools once a session has run on it (the session's opening probes it).
async fn probe_wb(w: &World) {
    let mut on_wb = pilot();
    on_wb.provider = Some("worker".into());
    on_wb.model = Some("wb".into());
    open(w, &on_wb).await;
}

async fn node(w: &World, session_id: &str) -> crate::neo4j::models::ChatSessionNode {
    w.graph
        .get_chat_session(Uuid::parse_str(session_id).unwrap())
        .await
        .unwrap()
        .unwrap()
}

async fn decisions(w: &World) -> Vec<CognitiveDecision> {
    Neo4jRoutingStore::new(w.graph.clone())
        .decisions(&DecisionFilter::default())
        .await
        .unwrap()
}

/// `primary`: whatever the arms say and even at the `auto` stage, every turn of the pilot
/// and of the executor goes to the instance the roles name; the decisions are stored,
/// none applied.
#[tokio::test]
async fn primary_sends_every_turn_to_the_primary_and_only_records() {
    let w = world("primary", "auto").await;
    let p = open(&w, &pilot()).await;
    let e = open(&w, &executor()).await;
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("primary"));
    assert_eq!(node(&w, &e).await.provider_id.as_deref(), Some("primary"));
    assert!(turns(&w.worker).is_empty(), "no turn went elsewhere");
    assert_eq!(turns(&w.primary).len(), 2);
    let all = decisions(&w).await;
    assert!(!all.is_empty());
    assert!(all.iter().all(|d| !d.applied), "{all:#?}");
    assert!(
        all.iter()
            .any(|d| d.chosen.as_ref().is_some_and(|c| c.provider_id == "worker")),
        "what the router would have chosen is recorded"
    );
}

/// `mixed`: the executor is routed (to the worker the arms prefer), the pilot is not.
#[tokio::test]
async fn mixed_routes_the_executor_and_leaves_the_pilot_on_the_primary() {
    let w = world("mixed", "auto").await;
    let p = open(&w, &pilot()).await;
    let e = open(&w, &executor()).await;
    let (pilot_node, executor_node) = (node(&w, &p).await, node(&w, &e).await);
    assert_eq!(pilot_node.provider_id.as_deref(), Some("primary"));
    assert_ne!(pilot_node.routed_by.as_deref(), Some("auto"));
    assert_eq!(executor_node.provider_id.as_deref(), Some("worker"));
    assert_eq!(executor_node.model, "wa");
    assert_eq!(executor_node.routed_by.as_deref(), Some("auto"));
    assert_eq!(turns(&w.primary).len(), 1, "the pilot's turn");
    assert_eq!(turns(&w.worker).len(), 1, "the executor's turn");
}

/// `full` + `auto`: the pilot is routed at the opening, and between two turns its model
/// changes with the class of the message (simple → `wa`, debug → `wb`).
#[tokio::test]
async fn full_auto_routes_the_pilot_and_changes_its_model_between_two_turns() {
    let w = world("full", "auto").await;
    probe_wb(&w).await;
    let p = open(&w, &pilot()).await;
    let opened = node(&w, &p).await;
    assert_eq!(opened.provider_id.as_deref(), Some("worker"));
    assert_eq!(opened.routed_by.as_deref(), Some("auto"));
    turn(&w, &p, SIMPLE).await;
    turn(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[DEBUG]), ["wa"]);
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(turns(&w.primary).is_empty());
}

/// A conversation given two models in the menu (`mixed` among them) routes each turn
/// among them: the debug turn runs on `wb`. The per-turn pool carries the PROJECT's
/// consent: without it every model of a stored instance was `not_allowed` and the turn
/// stayed on `wa` (`no_candidate`).
#[tokio::test]
async fn a_conversation_routed_among_two_ticked_models_changes_model_between_turns() {
    use super::provider::cognitive::ProviderRoutingMode;
    use super::types::RoutingPoolEntry;
    let w = world("primary", "auto").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.routing_mode = Some(ProviderRoutingMode::Mixed);
    ticked.routing_pool = Some(
        ["wa", "wb"]
            .iter()
            .map(|m| RoutingPoolEntry {
                provider: "worker".into(),
                model: (*m).into(),
            })
            .collect(),
    );
    let p = open(&w, &ticked).await;
    assert_eq!(node(&w, &p).await.model, "wa", "the plain opening: wa");
    turn(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    let opening_turn = decisions(&w)
        .await
        .into_iter()
        .find(|d| {
            d.session_id.map(|id| id.to_string()) == Some(p.clone()) && d.turn_index == Some(0)
        })
        .expect("the opening turn is decided");
    assert!(opening_turn.chosen.is_some(), "{}", opening_turn.reason);
}

/// `shadow`: even in `full`, nothing is applied (no move, no model change), and every
/// decision is recorded, the per-turn ones included.
#[tokio::test]
async fn the_shadow_stage_applies_nothing_and_records_every_decision() {
    let w = world("full", "shadow").await;
    let p = open(&w, &pilot()).await;
    turn(&w, &p, SIMPLE).await;
    turn(&w, &p, DEBUG).await;
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("primary"));
    assert!(turns(&w.worker).is_empty());
    assert!(turns(&w.primary).iter().all(|(model, _)| model == "pm"));
    let all = decisions(&w).await;
    assert!(all.iter().all(|d| !d.applied), "{all:#?}");
    assert!(
        all.iter().filter(|d| d.turn_index.is_some()).count() >= 2,
        "the turns' decisions are recorded too: {all:#?}"
    );
    assert!(all
        .iter()
        .any(|d| d.chosen.as_ref().is_some_and(|c| c.provider_id == "worker")));
}

/// A provider and a model the request names are never substituted, even in `full` +
/// `auto` with arms that prefer another model for the debug turn.
#[tokio::test]
async fn an_explicit_choice_is_never_substituted() {
    let w = world("full", "auto").await;
    probe_wb(&w).await;
    let mut named = pilot();
    named.provider = Some("worker".into());
    named.model = Some("wa".into());
    let p = open(&w, &named).await;
    turn(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wa"], "not wb");
    assert!(turns(&w.primary).is_empty());
    let opening = decisions(&w)
        .await
        .into_iter()
        .find(|d| d.turn_index.is_none())
        .expect("the opening decision is stored");
    assert!(!opening.applied && opening.chosen.is_none());
}

/// `full` + `auto` may move a conversation to ANOTHER provider (#643): handed back to PO,
/// the next turn runs on the worker the arms prefer, through the relay.
#[tokio::test]
async fn full_auto_moves_a_conversation_handed_back_to_po_to_another_provider() {
    let w = world("full", "auto").await;
    let mut named = pilot();
    named.provider = Some("primary".into());
    let p = open(&w, &named).await;
    w.manager
        .set_session_routing(
            &p,
            &SessionRoutingRequest {
                auto: true,
                routing_pool: Vec::new(),
            },
        )
        .await
        .unwrap();
    let mut rx = w.manager.subscribe(&p).await.unwrap();
    w.manager.send_message(&p, SIMPLE).await.unwrap();
    let relayed = wait_for(&mut rx, |e| {
        matches!(e, ChatEvent::ConversationRelayed { .. })
    })
    .await;
    let ChatEvent::ConversationRelayed {
        to_session_id,
        from_provider,
        to_provider,
        moved_by,
        ..
    } = relayed
    else {
        unreachable!()
    };
    assert_eq!(moved_by, "auto");
    assert_eq!(
        (from_provider.as_str(), to_provider.as_str()),
        ("primary", "worker")
    );
    for _ in 0..400 {
        if !model_of_turn(&w.worker, SIMPLE, &[]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[]), ["wa"]);
    assert!(model_of_turn(&w.primary, SIMPLE, &[]).is_empty());
    assert_eq!(
        node(&w, &to_session_id).await.provider_id.as_deref(),
        Some("worker")
    );
}

/// The compaction summary of a native session goes through a cognitive decision, recorded
/// in shadow: the router names what it would use (`worker/wb`), but the summary is written
/// where nexus writes it, by the session's endpoint and model (`primary/pm`).
#[tokio::test]
async fn a_native_compaction_records_its_decision_in_shadow_and_runs_on_the_session_model() {
    let w = world("full", "auto").await;
    let mut named = pilot();
    named.provider = Some("primary".into());
    named.model = Some("pm".into());
    let p = open(&w, &named).await;
    for text in LONG_TURNS {
        turn(&w, &p, text).await;
    }
    // The last turn runs on the compacted history: the summary the PRIMARY wrote, on pm.
    let after: Vec<(String, String)> = turns(&w.primary)
        .into_iter()
        .filter(|(_, body)| body.contains("turn five"))
        .collect();
    assert_eq!(after.len(), 1, "{after:?}");
    assert_eq!(after[0].0, "pm");
    assert!(
        after[0]
            .1
            .contains("[Summary of the earlier conversation]\\nsummary written by primary"),
        "compacted by the session's endpoint: {}",
        after[0].1
    );
    assert!(
        turns(&w.worker).is_empty(),
        "nothing was sent to the router's pick"
    );
    let mut compaction = None;
    for _ in 0..200 {
        compaction = decisions(&w)
            .await
            .into_iter()
            .find(|d| d.signature.class == TaskClass::UtilityCompaction);
        if compaction.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let decision = compaction.expect("the compaction decision is stored");
    assert!(!decision.applied, "full + auto, yet not applied");
    assert!(
        decision
            .reason
            .starts_with("compaction_model_not_selectable: "),
        "{}",
        decision.reason
    );
    assert_eq!(decision.chosen, Some(Pick::new("worker", "wb")));
    assert_eq!(decision.used, Some(Pick::new("primary", "pm")));
    assert_eq!(decision.session_id.map(|id| id.to_string()), Some(p));
}

/// Sends one message and returns every event of its turn, up to its result.
async fn turn_events(w: &World, session_id: &str, text: &str) -> Vec<ChatEvent> {
    let mut rx = w.manager.subscribe(session_id).await.unwrap();
    w.manager.send_message(session_id, text).await.unwrap();
    let mut events = Vec::new();
    loop {
        let event = wait_for(&mut rx, |_| true).await;
        let done = is_result(&event);
        events.push(event);
        if done {
            break;
        }
    }
    idle(w, session_id).await;
    events
}

/// The decisions stored for one session.
async fn decisions_of(w: &World, session_id: &str) -> Vec<CognitiveDecision> {
    decisions(w)
        .await
        .into_iter()
        .filter(|d| d.session_id.map(|id| id.to_string()).as_deref() == Some(session_id))
        .collect()
}

/// R-S1 (1) and (4): the settings say `primary` + `shadow` (the defaults), the user puts
/// THIS conversation in Auto (`routing_mode: full`): its model changes between a simple
/// and a debug turn, the change is told to the clients, and the stored decisions say what
/// happened (`stage: auto`, the change `applied`, with its reason).
#[tokio::test]
async fn a_conversation_in_auto_routes_its_turns_whatever_the_global_shadow_stage() {
    use super::provider::cognitive::{LearningStage, ProviderRoutingMode};
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut auto = pilot();
    auto.routing_mode = Some(ProviderRoutingMode::Full);
    let p = open(&w, &auto).await;
    turn(&w, &p, SIMPLE).await;
    let events = turn_events(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[DEBUG]), ["wa"]);
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(turns(&w.primary).is_empty(), "the conversation left pm");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ChatEvent::ModelChanged { model } if model == "wb")),
        "{events:#?}"
    );
    let mine = decisions_of(&w, &p).await;
    assert!(!mine.is_empty());
    assert!(
        mine.iter().all(|d| d.stage == LearningStage::Auto),
        "{mine:#?}"
    );
    let change = mine
        .iter()
        .find(|d| d.chosen.as_ref().is_some_and(|c| c.model == "wb"))
        .expect("the debug turn is decided");
    assert!(change.applied, "{change:#?}");
    assert!(!change.reason.is_empty());
}

/// R-S1 (2) and (4): same settings, two models ticked on the conversation: the debug turn
/// runs on `wb`, decided among the pool at `stage: auto`.
#[tokio::test]
async fn a_pool_ticked_on_the_conversation_routes_its_turns_whatever_the_global_shadow_stage() {
    use super::provider::cognitive::{LearningStage, ProviderRoutingMode};
    use super::types::RoutingPoolEntry;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.routing_mode = Some(ProviderRoutingMode::Mixed);
    ticked.routing_pool = Some(
        ["wa", "wb"]
            .iter()
            .map(|m| RoutingPoolEntry {
                provider: "worker".into(),
                model: (*m).into(),
            })
            .collect(),
    );
    let p = open(&w, &ticked).await;
    assert_eq!(node(&w, &p).await.model, "wa");
    let events = turn_events(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(turns(&w.primary).is_empty());
    assert!(events
        .iter()
        .any(|e| matches!(e, ChatEvent::ModelChanged { model } if model == "wb")));
    let change = decisions_of(&w, &p)
        .await
        .into_iter()
        .find(|d| d.turn_index.is_some() && d.chosen.as_ref().is_some_and(|c| c.model == "wb"))
        .expect("the debug turn is decided");
    assert_eq!(change.stage, LearningStage::Auto);
    assert!(change.applied, "{change:#?}");
}

/// R-S1 (3) and (4): `primary` + `shadow` and NO choice on the conversation: strict
/// identity, every turn on `pm`, nothing sent elsewhere, every decision `shadow`, none
/// applied. The global stage still governs the sessions driven by the settings.
#[tokio::test]
async fn without_a_conversation_choice_the_global_shadow_stage_still_governs() {
    use super::provider::cognitive::LearningStage;
    let w = world("primary", "shadow").await;
    let p = open(&w, &pilot()).await;
    turn(&w, &p, SIMPLE).await;
    turn(&w, &p, DEBUG).await;
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("primary"));
    assert!(turns(&w.worker).is_empty(), "no request elsewhere");
    assert_eq!(turns(&w.primary).len(), 3);
    assert!(turns(&w.primary).iter().all(|(model, _)| model == "pm"));
    let all = decisions(&w).await;
    assert!(!all.is_empty());
    assert!(
        all.iter()
            .all(|d| !d.applied && d.stage == LearningStage::Shadow),
        "{all:#?}"
    );
}

/// R-S1 (5): a model the request pins is never substituted, even when the same request
/// puts the conversation in Auto over a `shadow` setting.
#[tokio::test]
async fn a_pinned_model_is_never_substituted_even_in_a_conversation_in_auto() {
    use super::provider::cognitive::ProviderRoutingMode;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut pinned = pilot();
    pinned.provider = Some("worker".into());
    pinned.model = Some("wa".into());
    pinned.routing_mode = Some(ProviderRoutingMode::Full);
    let p = open(&w, &pinned).await;
    let events = turn_events(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wa"], "not wb");
    assert!(turns(&w.primary).is_empty());
    assert!(!events
        .iter()
        .any(|e| matches!(e, ChatEvent::ModelChanged { .. })));
    assert!(decisions_of(&w, &p).await.iter().all(|d| !d.applied
        || d.chosen
            .as_ref()
            .is_none_or(|c| (c.provider_id.as_str(), c.model.as_str()) == ("worker", "wa"))));
}
