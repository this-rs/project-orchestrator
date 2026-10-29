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
#[allow(unused_imports)]
use crate::neo4j::traits::*;

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
#[allow(unused_imports)]
use crate::neo4j::traits::*;
use crate::neo4j::GraphStore;
use crate::test_helpers::mock_app_state;

const SIMPLE: &str = "rename this variable";
const DEBUG: &str = "why does this crash with a stack trace error";
const PROBE: &str = "Call the ping tool now";
/// R-S3: a second simple and a second debug turn, two plan turns (a class where `wa` and `wb`
/// are as good), and an explore turn (a class the PRIMARY wins, `wb` second).
const SIMPLE2: &str = "rename that function";
const DEBUG2: &str = "the build fails with a panic";
const PLAN: &str = "plan and implement the export feature";
const PLAN2: &str = "now create the import feature too";
const EXPLORE: &str = "explain the architecture overview";
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
    script_signing(who, models, &[])
}

/// [`script`], plus a summary route per model of `summary_models`: a summary asked on that
/// model (N-C1, nexus contract v8: the decision names the summary model) signs it
/// (`summary written by <who>/<model>`). The model alone selects that route; the probes of the
/// session's model and of each summary model take the probe routes placed before it.
fn script_signing(who: &str, models: &[&str], summary_models: &[&str]) -> Value {
    let probe = || {
        sse_route(
            PROBE,
            vec![
                delta(
                    json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]}),
                ),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ],
        )
    };
    // A probe takes the first route it matches not used yet, and the probe of a model names
    // it too: one probe route for the session's model and one per summary model come BEFORE
    // that model's summary route, which is left for the summary.
    let mut routes: Vec<Value> = std::iter::once(probe())
        .filter(|_| !summary_models.is_empty())
        .collect();
    routes.extend(summary_models.iter().flat_map(|model| {
        let mut summary = sse_route(
            &format!("\"model\":\"{model}\""),
            answer(&format!("summary written by {who}/{model}"), 50),
        );
        summary["delay_ms"] = json!(300);
        [probe(), summary]
    }));
    routes.extend([
        probe(),
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
    ]);
    // The long conversation reports a prompt close to the 32k window: nexus compacts
    // before a turn once there is enough history to summarise.
    for text in LONG_TURNS.iter().rev() {
        routes.push(sse_route(text, answer("noted", 31_000)));
    }
    for text in [EXPLORE, PLAN2, PLAN, SIMPLE2, DEBUG2] {
        routes.push(sse_route(text, answer("done", 10)));
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
                        decay: 1.0,
                    },
                )
                .await
                .unwrap();
        }
    }
}

/// `successes` then `failures` observations of one pair.
async fn teach_counts(
    store: &Neo4jRoutingStore,
    class: &str,
    (provider, model): (&str, &str),
    successes: usize,
    failures: usize,
) {
    let key = ArmKey::new(class, provider, model);
    for reward in std::iter::repeat_n(1.0, successes).chain(std::iter::repeat_n(0.0, failures)) {
        store
            .observe(
                &key,
                &ArmObservation {
                    reward,
                    success_threshold: 0.5,
                    cost_usd: None,
                    latency_ms: None,
                    decay: 1.0,
                },
            )
            .await
            .unwrap();
    }
}

async fn world(mode: &str, stage: &str) -> World {
    world_with(mode, stage, &["pm"]).await
}

/// The world, with the primary serving `primary_models` (the first is its default; a summary
/// asked on any other signs that model).
async fn world_with(mode: &str, stage: &str, primary_models: &[&str]) -> World {
    let primary = FakeOpenAi::start(script_signing(
        "primary",
        primary_models,
        &primary_models[1..],
    ));
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
    // R-S3: the plan class has two candidates as good (their draws stay within the
    // hysteresis margin); the explore class prefers the primary, then `wb`, then `wa`.
    for (pair, successes, failures) in [
        (("primary", "pm"), 0, 60),
        (("worker", "wa"), 60, 0),
        (("worker", "wb"), 60, 0),
    ] {
        teach_counts(&store, &class_of(PLAN), pair, successes, failures).await;
    }
    for (pair, successes, failures) in [
        (("primary", "pm"), 60, 0),
        (("worker", "wa"), 0, 60),
        (("worker", "wb"), 45, 15),
    ] {
        teach_counts(&store, &class_of(EXPLORE), pair, successes, failures).await;
    }
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
        obscura_path: None,
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

/// Waits for an in-place move to be stated on the SAME thread and returns the
/// `ProviderSwitched` event. The in-place switch replaces the live broadcast channel, so a
/// subscription taken before `send_message` would observe `RecvError::Closed`; the stored
/// thread the session keeps is the reliable view.
async fn switched_in_place(w: &World, session_id: &str) -> ChatEvent {
    let id = Uuid::parse_str(session_id).unwrap();
    for _ in 0..200 {
        let events = w.graph.get_chat_events(id, 0, 500).await.unwrap();
        if let Some(event) = events
            .iter()
            .filter_map(|r| serde_json::from_str::<ChatEvent>(&r.data).ok())
            .find(|e| matches!(e, ChatEvent::ProviderSwitched { .. }))
        {
            return event;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the in-place move was never stated for {session_id}");
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

/// Opens a session on `wb`, which probes it. Under `primary` + `shadow` the pool probes
/// nothing (task 58fa9980): this is how those tests make `wb` a candidate with tools.
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

/// Settings `executors_only` (named `mixed` before R-S4): the executor is routed (to the
/// worker the arms prefer), the pilot is not.
#[tokio::test]
async fn executors_only_routes_the_executor_and_leaves_the_pilot_on_the_primary() {
    let w = world("executors_only", "auto").await;
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
    use super::provider::cognitive::ConversationRoutingMode;
    use super::types::RoutingPoolEntry;
    let w = world("primary", "auto").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
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

/// R-S4, end to end: the menu opens a pool conversation naming its pilot (`worker/wa`) with
/// the pool, as the frontend does. The OPENING model is decided among the pool (the debug
/// message runs on `wb`, `routed_by: auto`), whatever the global `primary` + `shadow`: the
/// pilot is only the fallback. Before R-S4 the pilot opened it (`wa`, `routed_by: request`).
#[tokio::test]
async fn a_pool_conversation_opens_on_the_model_decided_among_the_pool_not_on_the_pilot() {
    use super::provider::cognitive::ConversationRoutingMode;
    use super::types::RoutingPoolEntry;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.message = DEBUG.into();
    ticked.provider = Some("worker".into());
    ticked.model = Some("wa".into());
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
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
    let opened = node(&w, &p).await;
    assert_eq!(opened.provider_id.as_deref(), Some("worker"));
    assert_eq!(opened.model, "wb", "decided among the pool, not the pilot");
    assert_eq!(opened.routed_by.as_deref(), Some("auto"));
    assert_eq!(opened.routing_mode.as_deref(), Some("mixed"));
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(turns(&w.primary).is_empty());
}

/// Review of #713, C1: a `full` conversation with a pool, MOVED by the router, opens the new
/// session on the target the router decided (`worker/wb`), never on a decision taken again
/// among the pool at the opening (the arms prefer `wa` for this simple message). The relayed
/// request names the target with the pool and `full` read from the session: R-S4's pilot
/// rule is for the menu's `mixed` only.
#[tokio::test]
async fn a_full_conversation_with_a_pool_moved_by_the_router_opens_on_its_target() {
    use super::provider::cognitive::ConversationRoutingMode;
    use super::types::RoutingPoolEntry;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut full = pilot();
    full.provider = Some("primary".into());
    full.model = Some("pm".into());
    full.routing_mode = Some(ConversationRoutingMode::Full);
    full.routing_pool = Some(
        [("primary", "pm"), ("worker", "wa"), ("worker", "wb")]
            .iter()
            .map(|(provider, model)| RoutingPoolEntry {
                provider: (*provider).into(),
                model: (*model).into(),
            })
            .collect(),
    );
    let p = open(&w, &full).await;
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("primary"));
    let moved = w
        .manager
        .relay_conversation(
            &p,
            "worker",
            Some("wb"),
            SIMPLE,
            Some(crate::auth::jwt::Claims::service_account("e2e")),
            super::relay::MOVED_BY_AUTO,
        )
        .await
        .unwrap_or_else(|e| panic!("move failed: {e:#}"));
    // The relayed turn may be over before anyone subscribes: wait for its request instead.
    for _ in 0..400 {
        if !model_of_turn(&w.worker, SIMPLE, &[]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    idle(&w, &moved.session_id).await;
    let target = node(&w, &moved.session_id).await;
    assert_eq!(target.provider_id.as_deref(), Some("worker"));
    assert_eq!(
        target.model, "wb",
        "the router's target, not a new decision (wa)"
    );
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[]), ["wb"]);
}

/// Review of #713 (follow-up), C1-bis: a pool sent WITHOUT a mode, routed `full` by the
/// settings, moved by the router: the relayed request says `full`, so the new session opens
/// on the router's target (`worker/wb`), not on a decision taken again among the pool (`wa`
/// for this simple message).
#[tokio::test]
async fn a_pool_without_a_mode_moved_by_the_router_opens_on_its_target() {
    use super::types::RoutingPoolEntry;
    let w = world("full", "auto").await;
    probe_wb(&w).await;
    let mut pooled = pilot();
    // The explore class prefers the primary: the opening, decided among the pool, lands there.
    pooled.message = EXPLORE.into();
    pooled.routing_pool = Some(
        [("primary", "pm"), ("worker", "wa"), ("worker", "wb")]
            .iter()
            .map(|(provider, model)| RoutingPoolEntry {
                provider: (*provider).into(),
                model: (*model).into(),
            })
            .collect(),
    );
    let p = open(&w, &pooled).await;
    let opened = node(&w, &p).await;
    assert_eq!(opened.provider_id.as_deref(), Some("primary"));
    assert_eq!(opened.routing_mode, None, "no mode asked");
    let moved = w
        .manager
        .relay_conversation(
            &p,
            "worker",
            Some("wb"),
            SIMPLE,
            Some(crate::auth::jwt::Claims::service_account("e2e")),
            super::relay::MOVED_BY_AUTO,
        )
        .await
        .unwrap_or_else(|e| panic!("move failed: {e:#}"));
    for _ in 0..400 {
        if !model_of_turn(&w.worker, SIMPLE, &[]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    idle(&w, &moved.session_id).await;
    let target = node(&w, &moved.session_id).await;
    assert_eq!(
        target.model, "wb",
        "the router's target, not a new decision (wa)"
    );
    assert_eq!(target.routing_mode.as_deref(), Some("full"));
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[]), ["wb"]);
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
    w.manager.send_message(&p, SIMPLE).await.unwrap();
    let relayed = switched_in_place(&w, &p).await;
    let ChatEvent::ProviderSwitched {
        session_id,
        from_provider,
        to_provider,
        ..
    } = relayed
    else {
        unreachable!()
    };
    assert_eq!(
        session_id, p,
        "the move re-targets the SAME thread in place"
    );
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
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("worker"));
}

/// Stage `shadow`: the compaction summary of a native session goes through a cognitive
/// decision, recorded and not applied: the router names what it would use (`worker/wb`), but
/// the summary is written by the session's endpoint and model (`primary/pm`).
#[tokio::test]
async fn a_native_compaction_in_stage_shadow_records_its_decision_and_runs_on_the_session_model() {
    let w = world("full", "shadow").await;
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
            .contains("[Summary of the earlier conversation]\\nsummary written by primary")
            && !after[0].1.contains("summary written by primary/"),
        "compacted by the session's endpoint, on its model: {}",
        after[0].1
    );
    assert!(
        turns(&w.worker).is_empty(),
        "nothing was sent to the router's pick"
    );
    let decision = compaction_decision(&w).await;
    assert!(!decision.applied, "stage shadow: not applied");
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

/// N-C1 (nexus contract v8): `full` + `auto` on a native session, whose harness can write
/// the summary with another of its provider's models (`compaction_model`): the decision is
/// taken among the PRIMARY's models (`pm` failed the class sixty times, `ps` succeeded it
/// sixty times), applied, and the summary is written by `primary/ps`; the conversation stays
/// on `pm`.
#[tokio::test]
async fn a_native_compaction_in_full_auto_summarises_with_the_model_of_the_decision() {
    let w = world_with("full", "auto", &["pm", "ps"]).await;
    // `ps` is TAUGHT, not untried: the decider draws each arm from its Beta posterior, seeded
    // by the decision's id (a fresh UUID). An untried `ps` draws from the prior and, a few
    // times in a hundred, not far enough above `pm` to pass the hysteresis margin: the
    // decision then keeps `pm` ("kept because no challenger is clearly better") and the
    // summary is written by the session's model (FLAKY-NC1, 19/10/2026: 6 failures in 60
    // runs untried, 0 in 60 taught; the failing runs sent the fake the same requests, only
    // the draw differed).
    teach_counts(
        &Neo4jRoutingStore::new(w.graph.clone()),
        &TaskSignature::utility(TaskClass::UtilityCompaction, 0, None).arm_key(),
        ("primary", "ps"),
        60,
        0,
    )
    .await;
    let mut named = pilot();
    named.provider = Some("primary".into());
    named.model = Some("pm".into());
    let p = open(&w, &named).await;
    for text in LONG_TURNS {
        turn(&w, &p, text).await;
    }
    let after: Vec<(String, String)> = turns(&w.primary)
        .into_iter()
        .filter(|(_, body)| body.contains("turn five"))
        .collect();
    assert_eq!(after.len(), 1, "{after:?}");
    assert_eq!(after[0].0, "pm", "the conversation keeps its model");
    assert!(
        after[0]
            .1
            .contains("[Summary of the earlier conversation]\\nsummary written by primary/ps"),
        "summarised by the decision's model: {}",
        after[0].1
    );
    assert!(
        turns(&w.worker).is_empty(),
        "the summary stays on the session's endpoint"
    );
    let decision = compaction_decision(&w).await;
    assert!(decision.applied, "{}", decision.reason);
    assert_eq!(decision.chosen, Some(Pick::new("primary", "ps")));
    assert_eq!(decision.session_id.map(|id| id.to_string()), Some(p));
    assert_eq!(
        decisions(&w)
            .await
            .iter()
            .filter(|d| d.signature.class == TaskClass::UtilityCompaction)
            .count(),
        1,
        "one decision for the compaction"
    );
}

/// The stored decision of the compaction class, once it is there.
async fn compaction_decision(w: &World) -> CognitiveDecision {
    for _ in 0..200 {
        if let Some(found) = decisions(w)
            .await
            .into_iter()
            .find(|d| d.signature.class == TaskClass::UtilityCompaction)
        {
            return found;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the compaction decision is stored");
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
    use super::provider::cognitive::{ConversationRoutingMode, LearningStage};
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut auto = pilot();
    auto.routing_mode = Some(ConversationRoutingMode::Full);
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
    use super::provider::cognitive::{ConversationRoutingMode, LearningStage};
    use super::types::RoutingPoolEntry;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
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
    use super::provider::cognitive::ConversationRoutingMode;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut pinned = pilot();
    pinned.provider = Some("worker".into());
    pinned.model = Some("wa".into());
    pinned.routing_mode = Some(ConversationRoutingMode::Full);
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

/// Review of #702 (2): a delegation (`spawned_by`) follows the settings' stage. It cannot
/// put itself in Auto: `routing_mode: full` on its opening request is dropped, so under
/// `primary` + `shadow` it opens on the primary, sends nothing elsewhere, stores no mode,
/// and every decision is `shadow`, none applied.
#[tokio::test]
async fn a_delegation_cannot_put_itself_in_auto() {
    use super::provider::cognitive::{ConversationRoutingMode, LearningStage};
    let w = world("primary", "shadow").await;
    let mut delegated = executor();
    delegated.routing_mode = Some(ConversationRoutingMode::Full);
    let e = open(&w, &delegated).await;
    turn(&w, &e, DEBUG).await;
    let opened = node(&w, &e).await;
    assert_eq!(opened.provider_id.as_deref(), Some("primary"));
    assert_eq!(opened.routing_mode, None, "no mode stored for a resume");
    assert!(turns(&w.worker).is_empty(), "no turn went elsewhere");
    let all = decisions(&w).await;
    assert!(!all.is_empty());
    assert!(
        all.iter()
            .all(|d| !d.applied && d.stage == LearningStage::Shadow),
        "{all:#?}"
    );
}

/// Review of #702 (4): a resume rebuilds the conversation's Auto from what the node stores
/// (the mode as a string): after a close and a resume, under `primary` + `shadow`, the
/// debug turn still changes the model, decided at `stage: auto`.
#[tokio::test]
async fn a_resumed_conversation_in_auto_still_routes_at_the_auto_stage() {
    use super::provider::cognitive::{ConversationRoutingMode, LearningStage};
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut auto = pilot();
    auto.routing_mode = Some(ConversationRoutingMode::Full);
    let p = open(&w, &auto).await;
    assert_eq!(node(&w, &p).await.routing_mode.as_deref(), Some("full"));
    w.manager.close_session(&p).await.unwrap();
    let claims = crate::auth::jwt::Claims::service_account("e2e");
    w.manager
        .resume_session(&p, SIMPLE, Some(&claims))
        .await
        .unwrap_or_else(|e| panic!("resume failed: {e:#}"));
    for _ in 0..400 {
        if !model_of_turn(&w.worker, SIMPLE, &[DEBUG]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[DEBUG]), ["wa"]);
    idle(&w, &p).await;
    let resumed_at = decisions_of(&w, &p).await.len();
    let events = turn_events(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(events
        .iter()
        .any(|e| matches!(e, ChatEvent::ModelChanged { model } if model == "wb")));
    let after = decisions_of(&w, &p).await;
    assert!(after.len() > resumed_at, "the resumed turns are decided");
    let change = after
        .iter()
        .find(|d| d.turn_index.is_some() && d.chosen.as_ref().is_some_and(|c| c.model == "wb"))
        .expect("the debug turn is decided");
    assert_eq!(change.stage, LearningStage::Auto);
    assert!(change.applied, "{change:#?}");
}

/// R-S3: a conversation in Auto (`routing_mode: full` over `primary` + `shadow`), opened on
/// `worker/wa` with `wb` probed.
async fn auto_conversation(w: &World) -> String {
    use super::provider::cognitive::ConversationRoutingMode;
    probe_wb(w).await;
    let mut auto = pilot();
    auto.routing_mode = Some(ConversationRoutingMode::Full);
    let p = open(w, &auto).await;
    assert_eq!(node(w, &p).await.model, "wa", "the plain opening: wa");
    p
}

/// The models a turn's events say the conversation changed to.
fn changes(events: &[ChatEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::ModelChanged { model } => Some(model.clone()),
            _ => None,
        })
        .collect()
}

/// The decisions of a session's turns (not its opening), in turn order.
async fn turn_decisions(w: &World, session_id: &str) -> Vec<CognitiveDecision> {
    let mut mine: Vec<CognitiveDecision> = decisions_of(w, session_id)
        .await
        .into_iter()
        // Only OPEN decisions (outcome None) are the turns' own: a provider move leaves its
        // own decision CLOSED on the same thread, so it is not the decision of the next turn.
        .filter(|d| d.outcome.is_none() && d.turn_index.is_some_and(|i| i > 0))
        .collect();
    mine.sort_by_key(|d| d.turn_index);
    mine
}

/// R-S3 (1): simple and debug turns alternate; each turn runs on the model its class
/// wants, ON THE TURN: `wb`, `wa`, `wb`, `wa`, a `model_changed` every turn, and every
/// turn's decision applied with its reason. Red with the old guard ("never two changes in
/// consecutive turns"): the turn right after a change stayed on the previous model.
#[tokio::test]
async fn alternating_classes_change_the_model_on_every_turn_never_one_turn_late() {
    let w = world("primary", "shadow").await;
    assert_eq!(class_of(SIMPLE2), class_of(SIMPLE));
    assert_eq!(class_of(DEBUG2), class_of(DEBUG));
    let p = auto_conversation(&w).await;
    let mut seen = Vec::new();
    for text in [DEBUG, SIMPLE, DEBUG2, SIMPLE2] {
        seen.push(changes(&turn_events(&w, &p, text).await));
    }
    assert_eq!(
        seen,
        [["wb"], ["wa"], ["wb"], ["wa"]],
        "a model_changed per turn"
    );
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[SIMPLE]), ["wb"]);
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[DEBUG2]), ["wa"]);
    assert_eq!(model_of_turn(&w.worker, DEBUG2, &[SIMPLE2]), ["wb"]);
    assert_eq!(model_of_turn(&w.worker, SIMPLE2, &[]), ["wa"]);
    assert!(turns(&w.primary).is_empty());
    let decided = turn_decisions(&w, &p).await;
    let summary: Vec<(String, bool)> = decided
        .iter()
        .map(|d| {
            (
                d.chosen
                    .as_ref()
                    .map(|c| c.model.clone())
                    .unwrap_or_default(),
                d.applied,
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("wb".to_owned(), true),
            ("wa".to_owned(), true),
            ("wb".to_owned(), true),
            ("wa".to_owned(), true)
        ],
        "{decided:#?}"
    );
    assert!(decided.iter().all(|d| !d.reason.is_empty()), "{decided:#?}");
}

/// The suffix a stored reason carries when a guard narrowed its decision.
fn narrowed(decision: &CognitiveDecision, code: &str) -> bool {
    decision.reason.contains(&format!("; narrowed: {code}"))
}

/// R-S3 (2): after two changes in a row (debug → `wb`, simple → `wa`: red with the old
/// guard), two turns of a class whose two candidates score alike keep the model in force:
/// no flip-flop, the hysteresis margin holds it.
#[tokio::test]
async fn two_turns_of_a_class_with_close_candidates_never_flip_flop() {
    let w = world("primary", "shadow").await;
    assert_eq!(class_of(PLAN2), class_of(PLAN));
    assert_ne!(class_of(PLAN), class_of(DEBUG));
    assert_ne!(class_of(PLAN), class_of(SIMPLE));
    let p = auto_conversation(&w).await;
    let mut seen = Vec::new();
    for text in [DEBUG, SIMPLE, PLAN, PLAN2] {
        seen.push(changes(&turn_events(&w, &p, text).await));
    }
    let none: [&str; 0] = [];
    assert_eq!(seen, [vec!["wb"], vec!["wa"], none.to_vec(), none.to_vec()]);
    assert_eq!(model_of_turn(&w.worker, PLAN, &[PLAN2]), ["wa"]);
    assert_eq!(model_of_turn(&w.worker, PLAN2, &[]), ["wa"]);
    let decided = turn_decisions(&w, &p).await;
    assert!(
        decided[2..]
            .iter()
            .all(|d| d.chosen == Some(Pick::new("worker", "wa"))),
        "kept by the margin: {decided:#?}"
    );
}

/// R-S3 (3): two simple turns in a row right after a change: the first is routed on the
/// turn (`wb` → `wa`, red with the old guard), the second, of the same class, is a STEADY
/// decision (no exploration, margin on the posterior means) that changes nothing, and its
/// stored reason says it was narrowed.
#[tokio::test]
async fn two_simple_turns_in_a_row_change_nothing() {
    let w = world("primary", "shadow").await;
    let p = auto_conversation(&w).await;
    assert_eq!(changes(&turn_events(&w, &p, DEBUG).await), ["wb"]);
    let first = changes(&turn_events(&w, &p, SIMPLE).await);
    let second = changes(&turn_events(&w, &p, SIMPLE2).await);
    assert_eq!(first, ["wa"], "routed on the turn");
    assert!(second.is_empty(), "{second:?}");
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[SIMPLE2]), ["wa"]);
    assert_eq!(model_of_turn(&w.worker, SIMPLE2, &[]), ["wa"]);
    assert!(turns(&w.primary).is_empty());
    let decided = turn_decisions(&w, &p).await;
    let last = decided.last().expect("the second simple turn is decided");
    assert_eq!(last.chosen, Some(Pick::new("worker", "wa")));
    assert!(narrowed(last, "same_class_no_explore"), "{}", last.reason);
    assert!(
        decided[..decided.len() - 1]
            .iter()
            .all(|d| !narrowed(d, "same_class_no_explore")),
        "{decided:#?}"
    );
}

/// R-S3 (4), and the measure behind the decision to KEEP "no two provider moves in a row":
/// the router moves the conversation to the worker (a relay: a new session, the history
/// rendered into its first message); the next turn is of another class, which the PRIMARY
/// wins (`explore`): its model changes on the worker (`wa` → `wb`), its provider does not.
/// Red with the old guard: the turn after a move was only recorded, the model stayed `wa`.
#[tokio::test]
async fn after_a_provider_move_the_next_turn_of_another_class_changes_model_not_provider() {
    let w = world("full", "auto").await;
    probe_wb(&w).await;
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
    let moved_at = std::time::Instant::now();
    w.manager.send_message(&p, SIMPLE).await.unwrap();
    let ChatEvent::ProviderSwitched { session_id, .. } = switched_in_place(&w, &p).await else {
        unreachable!()
    };
    assert_eq!(
        session_id, p,
        "the move re-targets the SAME thread in place"
    );
    for _ in 0..400 {
        if !model_of_turn(&w.worker, SIMPLE, &[]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    idle(&w, &p).await;
    let move_turn = moved_at.elapsed();
    assert_eq!(model_of_turn(&w.worker, SIMPLE, &[]), ["wa"]);

    let changed_at = std::time::Instant::now();
    let events = turn_events(&w, &p, EXPLORE).await;
    let change_turn = changed_at.elapsed();
    assert_eq!(changes(&events), ["wb"], "the model may change");
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ChatEvent::ProviderSwitched { .. })),
        "no second move"
    );
    assert_eq!(model_of_turn(&w.worker, EXPLORE, &[]), ["wb"]);
    assert!(
        model_of_turn(&w.primary, EXPLORE, &[]).is_empty(),
        "the provider stays"
    );
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("worker"));
    let explore = turn_decisions(&w, &p)
        .await
        .into_iter()
        .find(|d| d.signature.arm_key() == class_of(EXPLORE))
        .expect("the explore turn is decided");
    assert!(explore.applied, "{explore:#?}");
    assert_eq!(explore.chosen, Some(Pick::new("worker", "wb")));
    assert!(
        explore
            .alternatives
            .iter()
            .all(|a| a.pick.provider_id == "worker"),
        "decided among the worker's models only: {explore:#?}"
    );
    // Review of #710 (3): the stored reason says why it was narrowed — the move, not the
    // class (explore is not the class of the relayed turn).
    assert!(
        narrowed(&explore, "after_provider_move"),
        "{}",
        explore.reason
    );
    assert!(
        !narrowed(&explore, "same_class_no_explore"),
        "{}",
        explore.reason
    );
    // The measure (decision R-S3): what a move costs next to a change of model.
    let size = |text: &str, fake: &FakeOpenAi| {
        turns(fake)
            .into_iter()
            .filter(|(_, body)| body.contains(text))
            .map(|(_, body)| body.len())
            .min()
            .unwrap_or_default()
    };
    eprintln!(
        "R-S3 measure: move turn {move_turn:?} (relay: new session, first request {} bytes), \
         model-change turn {change_turn:?} (same session, request {} bytes)",
        size(SIMPLE, &w.worker),
        size(EXPLORE, &w.worker),
    );
}

/// Review of #710 (1) and (3), end to end: the router moves the conversation to the worker
/// on a simple turn; the next turn is ANOTHER simple turn. A move is a change of model: that
/// turn is steady (no exploration, margin on the means) and decided on the worker only, it
/// stays on `wa`, and its stored reason says both. Red without `seed_relayed_turn`: no
/// `same_class_no_explore` (the turn was decided with the settings' exploration).
#[tokio::test]
async fn after_a_provider_move_the_next_turn_of_the_same_class_is_steady() {
    assert_eq!(class_of(SIMPLE2), class_of(SIMPLE));
    let w = world("full", "auto").await;
    probe_wb(&w).await;
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
    w.manager.send_message(&p, SIMPLE).await.unwrap();
    let ChatEvent::ProviderSwitched { session_id, .. } = switched_in_place(&w, &p).await else {
        unreachable!()
    };
    assert_eq!(
        session_id, p,
        "the move re-targets the SAME thread in place"
    );
    for _ in 0..400 {
        if !model_of_turn(&w.worker, SIMPLE, &[]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    idle(&w, &p).await;
    let events = turn_events(&w, &p, SIMPLE2).await;
    assert!(changes(&events).is_empty(), "{events:#?}");
    assert_eq!(model_of_turn(&w.worker, SIMPLE2, &[]), ["wa"]);
    let decision = turn_decisions(&w, &p)
        .await
        .pop()
        .expect("the second simple turn is decided");
    assert_eq!(decision.chosen, Some(Pick::new("worker", "wa")));
    assert!(
        narrowed(&decision, "same_class_no_explore"),
        "{}",
        decision.reason
    );
    assert!(
        narrowed(&decision, "after_provider_move"),
        "{}",
        decision.reason
    );
}

/// A pool that ticks models on two providers (the menu's `mixed`): `primary/pm` plus the
/// worker's two models.
fn cross_provider_pool() -> Vec<super::types::RoutingPoolEntry> {
    [("primary", "pm"), ("worker", "wa"), ("worker", "wb")]
        .iter()
        .map(|(provider, model)| super::types::RoutingPoolEntry {
            provider: (*provider).into(),
            model: (*model).into(),
        })
        .collect()
}

/// Waits until the worker served `text` once, then until the session is idle.
async fn served_by_worker(w: &World, session_id: &str, text: &str) {
    for _ in 0..400 {
        if !model_of_turn(&w.worker, text, &[]).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    idle(w, session_id).await;
}

/// R-S2 (1): a pool ticked across two providers is routed like `full` restricted to the
/// pool, whatever the settings say (here `primary` + `shadow`: the conversation's own ticks
/// win, decision R-S1). The opening (EXPLORE, which the arms taught to prefer the primary)
/// lands on `primary/pm`; a DEBUG turn (which prefers `wb`) moves the conversation to the
/// worker, in place (`ProviderSwitched` on the SAME thread). Red before R-S2: a mixed pool
/// kept its provider, so the debug turn stayed on `pm` and no switch happened.
#[tokio::test]
async fn a_pool_spanning_two_providers_moves_to_the_other_provider_on_a_turn() {
    use super::provider::cognitive::ConversationRoutingMode;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.message = EXPLORE.into();
    ticked.provider = Some("primary".into());
    ticked.model = Some("pm".into());
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
    ticked.routing_pool = Some(cross_provider_pool());
    let p = open(&w, &ticked).await;
    let opened = node(&w, &p).await;
    assert_eq!(opened.provider_id.as_deref(), Some("primary"));
    assert_eq!(
        opened.model, "pm",
        "the opening: the explore class prefers pm"
    );

    w.manager.send_message(&p, DEBUG).await.unwrap();
    let ChatEvent::ProviderSwitched {
        session_id,
        from_provider,
        to_provider,
        model,
        ..
    } = switched_in_place(&w, &p).await
    else {
        unreachable!("the pool spans two providers: the debug turn moves")
    };
    assert_eq!(session_id, p, "moved in place, on the same thread");
    assert_eq!(
        (from_provider.as_str(), to_provider.as_str(), model.as_str()),
        ("primary", "worker", "wb")
    );
    served_by_worker(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(model_of_turn(&w.primary, DEBUG, &[]).is_empty());
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("worker"));
}

/// R-S2 (2): a pool on ONE provider keeps today's behaviour: no move, the model is routed
/// inside the provider (debug → `wb`).
#[tokio::test]
async fn a_pool_on_one_provider_never_moves_the_conversation() {
    use super::provider::cognitive::ConversationRoutingMode;
    use super::types::RoutingPoolEntry;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.provider = Some("worker".into());
    ticked.model = Some("wa".into());
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
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
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("worker"));
    let events = turn_events(&w, &p, DEBUG).await;
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ChatEvent::ProviderSwitched { .. })),
        "a pool on one provider never moves: {events:#?}"
    );
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);
    assert!(turns(&w.primary).is_empty());
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("worker"));
}

/// R-S2 (3): ONE model ticked is strict — the request names that pair, `routed_by:
/// request`, and it is never substituted, even when the arms prefer another model.
#[tokio::test]
async fn a_single_tick_is_strict_a_pin_never_substituted() {
    use super::provider::cognitive::ConversationRoutingMode;
    use super::types::RoutingPoolEntry;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.provider = Some("worker".into());
    ticked.model = Some("wb".into());
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
    ticked.routing_pool = Some(vec![RoutingPoolEntry {
        provider: "worker".into(),
        model: "wb".into(),
    }]);
    let p = open(&w, &ticked).await;
    let opened = node(&w, &p).await;
    assert_eq!(opened.provider_id.as_deref(), Some("worker"));
    assert_eq!(opened.model, "wb");
    assert_eq!(
        opened.routed_by.as_deref(),
        Some("request"),
        "one tick = strict"
    );
    turn(&w, &p, DEBUG).await;
    assert_eq!(
        model_of_turn(&w.worker, DEBUG, &[]),
        ["wb"],
        "never substituted"
    );
    assert!(turns(&w.primary).is_empty());
}

/// R-S2 (4): a pool spanning two providers moves the conversation once; the turn right
/// after a move is decided among the new provider's models only (R-S3), even though the
/// EXPLORE class strongly prefers the provider it just left.
#[tokio::test]
async fn a_pool_spanning_two_providers_moves_only_once() {
    use super::provider::cognitive::ConversationRoutingMode;
    let w = world("primary", "shadow").await;
    probe_wb(&w).await;
    let mut ticked = pilot();
    ticked.message = EXPLORE.into();
    ticked.provider = Some("primary".into());
    ticked.model = Some("pm".into());
    ticked.routing_mode = Some(ConversationRoutingMode::Mixed);
    ticked.routing_pool = Some(cross_provider_pool());
    let p = open(&w, &ticked).await;
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("primary"));

    w.manager.send_message(&p, DEBUG).await.unwrap();
    let ChatEvent::ProviderSwitched { to_provider, .. } = switched_in_place(&w, &p).await else {
        unreachable!()
    };
    assert_eq!(to_provider, "worker");
    served_by_worker(&w, &p, DEBUG).await;
    assert_eq!(model_of_turn(&w.worker, DEBUG, &[]), ["wb"]);

    // EXPLORE favours the primary it just left: the turn after a move is decided on the
    // worker only (its model may change, its provider does not).
    let events = turn_events(&w, &p, EXPLORE).await;
    let switches = events
        .iter()
        .filter(|e| matches!(e, ChatEvent::ProviderSwitched { .. }))
        .count();
    assert_eq!(switches, 0, "no second move in a row: {events:#?}");
    // The opening EXPLORE ran on the primary, so look at what came after the move: a
    // request carrying DEBUG (the relayed history) never reached the primary again.
    assert!(
        model_of_turn(&w.primary, DEBUG, &[]).is_empty(),
        "the primary never saw the conversation again"
    );
    assert_eq!(node(&w, &p).await.provider_id.as_deref(), Some("worker"));
}
