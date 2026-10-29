//! T9 — Transverse evaluation, OFFLINE (plan 09c88ffe « Chat sans dépendance au cwd »).
//!
//! A deterministic harness that replays a labeled, synthetic fixture set through
//! the context arms and the posture tracker, and measures three families of
//! metrics. No model call, no database: the graph is the in-process
//! [`MockGraphStore`], filled from `tests/fixtures/eval_transverse/*.json`
//! (versioned, anonymised projects and notes written for this harness — never
//! the content of a live base).
//!
//! # Arms (same [`Arm`] interface)
//!
//! - `none` — reference: no context at all (the « sans contexte » of the task).
//! - `baseline_cwd` — the historical behaviour: the project is the session's
//!   `project_slug`, else the one GUESSED from the cwd (whatever the execution
//!   place), and the context is the project's most salient items, blind to the
//!   anchors (« complet »).
//! - `anchors` — the real resolver of main ([`resolver::resolve_from_anchors`]:
//!   project precedence, consent predicate, per-role budgets, map + live block)
//!   (« borné »).
//! - `anchors_synapses` — the anchors arm plus a [`SynapticLayer`]. The synaptic
//!   layer (T8) lives in an open PR (#856): until it is merged no layer is
//!   plugged, the arm runs as its stand-in (= `anchors`) and the report marks it
//!   `pending #856`. Its code is NOT copied here.
//!
//! # Metrics
//!
//! - Context, per question: precision / recall of the injected items against the
//!   labels, stale items injected (moved / superseded — a false-claim risk),
//!   token cost; means with bootstrap 95 % intervals, paired differences.
//! - Posture (the real [`PostureTracker`] of main): switch rate, switches during
//!   oscillation, hostile turns (a prompt + an agent-injected focus anchor), the
//!   invariant segment byte-identical, prefix cache hits.
//! - Leak: every item an arm injects is checked against an ORACLE written from
//!   the stated rules (not from the resolver): an item of a project other than
//!   the decided one is readable only if it is `explicit_allow`, or if a human
//!   opened its project (live user project / workspace anchor) and it is not
//!   `explicit_deny`. Run on the 50 questions and on a seeded replay of 10 000
//!   random resolutions. ANY leak fails the test, hence the CI (`nextest --lib`).
//!
//! The report is printed as a table and written as JSON to `$PO_EVAL_REPORT`
//! (default `target/eval-transverse/report.json` under the crate).
//! `PO_EVAL_INJECT_LEAK=<item id>` makes the anchors arm inject that item: the
//! red proof of the gate (also held by a test).
use crate::neo4j::traits::*;

use crate::chat::anchor::{Anchor, AnchorActor, AnchorRole, AnchorState, AnchorTargetType};
use crate::chat::anchor_resolver::{
    self as resolver, est_tokens, CwdInference, ProjectInputs, ProjectSource,
};
use crate::chat::posture::{self, Invariants, Posture, PostureTracker, SwitchCause};
use crate::chat::provider::policy::SessionAccess;
use crate::chat::untrusted::{self, Origin};
use crate::episodes::distill_models::SharingConsent;
use crate::graph::neighborhood::RawNode;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::ExecutionPlace;
use crate::neo4j::GraphStore;

use crate::neo4j::traits::*;
use crate::skills::project_resolver::{entries_for_projects, infer_project_slug, ResolvedProject};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use uuid::Uuid;

const WORLD_JSON: &str = include_str!("../../tests/fixtures/eval_transverse/world.json");
const QUESTIONS_JSON: &str = include_str!("../../tests/fixtures/eval_transverse/questions.json");
const POSTURE_JSON: &str = include_str!("../../tests/fixtures/eval_transverse/posture.json");

/// Version of the report format.
const REPORT_VERSION: u32 = 1;
/// Items the baseline injects at most (the legacy injection is not anchor-aware).
const BASELINE_MAX_ITEMS: usize = 20;
/// Token ceiling of the baseline context.
const BASELINE_MAX_TOKENS: usize = 2000;
/// Bootstrap resamples.
const BOOTSTRAP: usize = 2000;
const BOOTSTRAP_SEED: u64 = 0x7439_0001;
/// Thresholds of the plan (go / no-go, reported; only leaks and posture
/// violations fail the run).
const RECALL_GAIN_VS_NONE_PTS: f64 = 15.0;
const SYNAPSE_TOKEN_OVERHEAD_MAX: f64 = 0.05;
const SWITCH_RATE_MAX: f64 = 0.2;
const PREFIX_DROP_MAX_PTS: f64 = 5.0;

// ----------------------------------------------------------------------------
// Fixtures
// ----------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WorldFx {
    version: u32,
    projects: Vec<ProjectFx>,
    workspaces: Vec<WorkspaceFx>,
    items: Vec<ItemFx>,
    edges: Vec<(String, String, f64)>,
    leak_replay: ReplayFx,
}

#[derive(Debug, Deserialize)]
struct ReplayFx {
    seed: u64,
    resolutions: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct ProjectFx {
    slug: String,
    name: String,
    root: String,
}

#[derive(Debug, Deserialize)]
struct WorkspaceFx {
    key: String,
    name: String,
    projects: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct ItemFx {
    id: String,
    #[serde(rename = "type")]
    node_type: String,
    title: String,
    project: String,
    weight: f64,
    #[serde(default)]
    consent: SharingConsent,
    #[serde(default)]
    stale: bool,
}

#[derive(Debug, Deserialize, Clone)]
struct SessionFx {
    #[serde(default)]
    project: Option<String>,
    cwd: String,
    #[serde(default)]
    place: ExecutionPlace,
}

fn d_roles() -> Vec<AnchorRole> {
    vec![AnchorRole::Focus]
}
fn d_user() -> AnchorActor {
    AnchorActor::User
}
fn d_live() -> AnchorState {
    AnchorState::Live
}
fn d_one() -> usize {
    1
}

#[derive(Debug, Deserialize, Clone)]
struct AnchorFx {
    t: AnchorTargetType,
    /// Item id, project slug or workspace key.
    #[serde(default)]
    id: String,
    /// Stable key of the anchor within a conversation (posture set).
    #[serde(default)]
    k: Option<String>,
    #[serde(default = "d_roles")]
    roles: Vec<AnchorRole>,
    #[serde(default = "d_user")]
    by: AnchorActor,
    #[serde(default)]
    inferred: bool,
    #[serde(default = "d_live")]
    state: AnchorState,
    /// Old path of a `moved` anchor.
    #[serde(default)]
    from: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QuestionsFx {
    version: u32,
    default_session: SessionFx,
    questions: Vec<QuestionFx>,
}

#[derive(Debug, Deserialize)]
struct QuestionFx {
    id: String,
    category: String,
    question: String,
    #[serde(default)]
    session: Option<SessionFx>,
    anchors: Vec<AnchorFx>,
    relevant: Vec<String>,
    #[serde(default)]
    stale: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PostureFx {
    version: u32,
    conversations: Vec<ConversationFx>,
    hostile_prompts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ConversationFx {
    id: String,
    access: SessionAccess,
    turns: Vec<TurnBlockFx>,
}

#[derive(Debug, Deserialize)]
struct TurnBlockFx {
    #[serde(default)]
    focus: Vec<AnchorFx>,
    #[serde(default)]
    alternate: Option<Vec<Vec<AnchorFx>>>,
    #[serde(default = "d_one")]
    repeat: usize,
}

fn fixture_digest(s: &str) -> String {
    hex::encode(&Sha256::digest(s.as_bytes())[..8])
}

/// A uuid that only depends on `s` (the opaque ids and journals stay comparable
/// from one run to the next).
fn det_uuid(s: &str) -> Uuid {
    let h = Sha256::digest(s.as_bytes());
    Uuid::from_slice(&h[..16]).expect("16 bytes")
}

// ----------------------------------------------------------------------------
// World
// ----------------------------------------------------------------------------

struct World {
    store: MockGraphStore,
    projects: BTreeMap<String, (Uuid, ProjectFx)>,
    slug_of: HashMap<Uuid, String>,
    workspaces: BTreeMap<String, (Uuid, Vec<String>)>,
    ws_key_of: HashMap<Uuid, String>,
    items: BTreeMap<String, ItemFx>,
    entries: Vec<ResolvedProject>,
}

impl World {
    async fn build(fx: &WorldFx) -> World {
        let store = MockGraphStore::new();
        let mut projects = BTreeMap::new();
        let mut slug_of = HashMap::new();
        let mut nodes = Vec::new();
        for p in &fx.projects {
            let mut n = crate::test_helpers::test_project_named(&p.name);
            n.id = det_uuid(&format!("project:{}", p.slug));
            n.slug = p.slug.clone();
            n.root_path = p.root.clone();
            store.create_project(&n).await.expect("fixture project");
            projects.insert(p.slug.clone(), (n.id, p.clone()));
            slug_of.insert(n.id, p.slug.clone());
            nodes.push(n);
        }
        let mut workspaces = BTreeMap::new();
        let mut ws_key_of = HashMap::new();
        for w in &fx.workspaces {
            let mut ws = crate::test_helpers::test_workspace();
            ws.id = det_uuid(&format!("workspace:{}", w.key));
            ws.name = w.name.clone();
            ws.slug = w.key.clone();
            store
                .create_workspace(&ws)
                .await
                .expect("fixture workspace");
            for slug in &w.projects {
                let (pid, _) = projects
                    .get(slug)
                    .unwrap_or_else(|| panic!("workspace {}: unknown project {slug}", w.key));
                store
                    .add_project_to_workspace(ws.id, *pid)
                    .await
                    .expect("fixture workspace project");
            }
            workspaces.insert(w.key.clone(), (ws.id, w.projects.clone()));
            ws_key_of.insert(ws.id, w.key.clone());
        }
        let mut items = BTreeMap::new();
        {
            let mut g = store.neighborhood_graph.write().await;
            let mut o = store.neighborhood_ownership.write().await;
            for it in &fx.items {
                let (pid, _) = projects
                    .get(&it.project)
                    .unwrap_or_else(|| panic!("item {}: unknown project {}", it.id, it.project));
                g.add_node(RawNode {
                    id: it.id.clone(),
                    node_type: it.node_type.clone(),
                    label: it.title.clone(),
                    subtitle: None,
                    weight: it.weight,
                });
                o.project_of.insert(it.id.clone(), pid.to_string());
                o.consent_of.insert(it.id.clone(), it.consent);
                assert!(
                    items.insert(it.id.clone(), it.clone()).is_none(),
                    "duplicate item {}",
                    it.id
                );
            }
            for (a, b, w) in &fx.edges {
                assert!(
                    items.contains_key(a) && items.contains_key(b),
                    "edge {a} - {b}: unknown endpoint"
                );
                g.add_edge(a, b, "RELATES_TO", *w);
            }
        }
        World {
            store,
            projects,
            slug_of,
            workspaces,
            ws_key_of,
            items,
            entries: entries_for_projects(&nodes),
        }
    }

    fn project_slug(&self, id: &str) -> Option<String> {
        Uuid::parse_str(id)
            .ok()
            .and_then(|u| self.slug_of.get(&u).cloned())
    }

    fn workspace_key(&self, id: &str) -> Option<String> {
        Uuid::parse_str(id)
            .ok()
            .and_then(|u| self.ws_key_of.get(&u).cloned())
    }
}

/// The historical cwd inference, over the fixture roots (pure: no global cache).
struct FxInference<'a>(&'a [ResolvedProject]);

#[async_trait::async_trait]
impl CwdInference for FxInference<'_> {
    async fn infer(&self, cwd: &str) -> Option<String> {
        if cwd.is_empty() {
            return None;
        }
        infer_project_slug(self.0, cwd)
    }
}

fn fixed_time() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::from_timestamp(1_760_000_000, 0).expect("timestamp")
}

fn build_anchor(w: Option<&World>, session_id: Uuid, key: &str, a: &AnchorFx) -> Anchor {
    let resolve = |kind: &str| -> String {
        let Some(w) = w else {
            return a.id.clone();
        };
        let found = match kind {
            "project" => w.projects.get(&a.id).map(|(u, _)| *u),
            _ => w.workspaces.get(&a.id).map(|(u, _)| *u),
        };
        found.map(|u| u.to_string()).unwrap_or_else(|| a.id.clone())
    };
    let target_id = match a.t {
        AnchorTargetType::Project => resolve("project"),
        AnchorTargetType::Workspace => resolve("workspace"),
        _ => a.id.clone(),
    };
    Anchor {
        id: det_uuid(&format!("anchor:{session_id}:{key}")),
        session_id,
        target_type: a.t,
        target_id,
        roles: a.roles.iter().copied().collect(),
        state: a.state,
        by: a.by,
        inferred: a.inferred,
        confidence: 1.0,
        snapshot_name: None,
        snapshot_path: a.from.clone(),
        snapshot_type: None,
        rev: None,
        version: 1,
        created_at: fixed_time(),
        updated_at: fixed_time(),
    }
}

/// One resolution to evaluate.
struct Case {
    key: String,
    session_id: Uuid,
    session: SessionFx,
    anchors: Vec<Anchor>,
    /// Resolve twice and compare the anchor maps (prefix stability).
    check_stability: bool,
}

// ----------------------------------------------------------------------------
// Arms
// ----------------------------------------------------------------------------

/// What an arm puts in front of the model for one turn.
#[derive(Debug, Clone, Default)]
struct ArmOutput {
    /// Project the arm decided (slug).
    project: Option<String>,
    /// Fixture items injected (content), in order, without duplicates.
    items: Vec<String>,
    /// Containers named (`project:<slug>`, `workspace:<key>`).
    containers: Vec<String>,
    /// Everything rendered (scanned for the titles of unreadable items).
    text: String,
    tokens: usize,
    guessed_from_cwd: bool,
    map_stable: Option<bool>,
}

fn push_unique(v: &mut Vec<String>, id: &str) {
    if !v.iter().any(|x| x == id) {
        v.push(id.to_string());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArmStatus {
    /// A reference point, not a candidate.
    Reference,
    /// The arm runs code that is on main.
    OnMain,
    /// The arm needs code of an open PR: a stand-in runs in its place.
    Pending { pr: &'static str },
}

impl ArmStatus {
    fn label(self) -> String {
        match self {
            Self::Reference => "référence".into(),
            Self::OnMain => "main".into(),
            Self::Pending { pr } => format!("pending {pr}"),
        }
    }
}

#[async_trait::async_trait]
trait Arm: Send + Sync {
    fn id(&self) -> &'static str;
    fn describe(&self) -> &'static str;
    fn status(&self) -> ArmStatus;
    async fn context(&self, w: &World, c: &Case) -> anyhow::Result<ArmOutput>;
}

struct NoContext;

#[async_trait::async_trait]
impl Arm for NoContext {
    fn id(&self) -> &'static str {
        "none"
    }
    fn describe(&self) -> &'static str {
        "aucun contexte"
    }
    fn status(&self) -> ArmStatus {
        ArmStatus::Reference
    }
    async fn context(&self, _w: &World, _c: &Case) -> anyhow::Result<ArmOutput> {
        Ok(ArmOutput::default())
    }
}

/// The historical behaviour: project from the slug, else guessed from the cwd;
/// the project's most salient items, whatever the anchors.
struct BaselineCwd;

#[async_trait::async_trait]
impl Arm for BaselineCwd {
    fn id(&self) -> &'static str {
        "baseline_cwd"
    }
    fn describe(&self) -> &'static str {
        "projet du slug ou deviné du cwd, éléments les plus saillants, sans ancres"
    }
    fn status(&self) -> ArmStatus {
        ArmStatus::Reference
    }
    async fn context(&self, w: &World, c: &Case) -> anyhow::Result<ArmOutput> {
        let explicit = c
            .session
            .project
            .as_ref()
            .filter(|s| w.projects.contains_key(*s))
            .cloned();
        let (slug, guessed) = match explicit {
            Some(s) => (Some(s), false),
            None => {
                let s = FxInference(&w.entries).infer(&c.session.cwd).await;
                let g = s.is_some();
                (s, g)
            }
        };
        let Some(slug) = slug else {
            return Ok(ArmOutput::default());
        };
        let name = w
            .projects
            .get(&slug)
            .map(|(_, p)| p.name.clone())
            .unwrap_or_default();
        let mut cands: Vec<&ItemFx> = w.items.values().filter(|i| i.project == slug).collect();
        cands.sort_by(|a, b| b.weight.total_cmp(&a.weight).then_with(|| a.id.cmp(&b.id)));
        let head = format!("projet: {name} ({slug})");
        let mut tokens = est_tokens(&head);
        let mut lines = vec![head];
        let mut items = Vec::new();
        for it in cands.into_iter().take(BASELINE_MAX_ITEMS) {
            let line = format!(
                "- {} «{}» #{}",
                it.node_type,
                it.title,
                resolver::short_id(&it.id)
            );
            let t = est_tokens(&line);
            if tokens + t > BASELINE_MAX_TOKENS {
                break;
            }
            tokens += t;
            lines.push(line);
            items.push(it.id.clone());
        }
        let text = untrusted::wrap(
            &lines.join("\n"),
            Origin::new("knowledge", Some(&slug)),
            &untrusted::nonce_from_seed(&c.key, 0),
            usize::MAX,
        );
        Ok(ArmOutput {
            project: Some(slug),
            items,
            containers: Vec::new(),
            tokens: est_tokens(&text),
            text,
            guessed_from_cwd: guessed,
            map_stable: None,
        })
    }
}

/// The resolver of main: anchor map (cacheable prefix) + live block.
struct AnchorsArm {
    /// Red proof only: an item injected on top of the resolver's output.
    inject_leak: Option<String>,
}

impl AnchorsArm {
    fn from_env() -> Self {
        Self {
            inject_leak: std::env::var("PO_EVAL_INJECT_LEAK")
                .ok()
                .filter(|s| !s.is_empty()),
        }
    }
}

#[async_trait::async_trait]
impl Arm for AnchorsArm {
    fn id(&self) -> &'static str {
        "anchors"
    }
    fn describe(&self) -> &'static str {
        "résolveur d'ancres (carte + bloc vivant, budgets par rôle, consentement)"
    }
    fn status(&self) -> ArmStatus {
        ArmStatus::OnMain
    }
    async fn context(&self, w: &World, c: &Case) -> anyhow::Result<ArmOutput> {
        let inputs = ProjectInputs {
            explicit_slug: c.session.project.as_deref(),
            place: c.session.place,
            cwd: &c.session.cwd,
        };
        let infer = FxInference(&w.entries);
        let r = resolver::resolve_from_anchors(
            &w.store,
            c.session_id,
            &inputs,
            &infer,
            c.anchors.clone(),
        )
        .await?;
        let map = r.map();
        let live = r.live_block();
        let map_stable = if c.check_stability {
            let again = resolver::resolve_from_anchors(
                &w.store,
                c.session_id,
                &inputs,
                &infer,
                c.anchors.clone(),
            )
            .await?;
            Some(again.map() == map)
        } else {
            None
        };
        let mut items = Vec::new();
        let mut containers = Vec::new();
        for a in &r.scope.anchors {
            match a.target_type {
                AnchorTargetType::Project => containers.push(format!(
                    "project:{}",
                    w.project_slug(&a.target_id)
                        .unwrap_or_else(|| a.target_id.clone())
                )),
                AnchorTargetType::Workspace => containers.push(format!(
                    "workspace:{}",
                    w.workspace_key(&a.target_id)
                        .unwrap_or_else(|| a.target_id.clone())
                )),
                _ => push_unique(&mut items, &a.target_id),
            }
            for n in &a.expansion.nodes {
                push_unique(&mut items, &n.id);
            }
        }
        let mut text = format!("{map}\n{live}");
        let tokens = est_tokens(&map) + est_tokens(&live);
        if let Some(id) = &self.inject_leak {
            push_unique(&mut items, id);
            if let Some(it) = w.items.get(id) {
                text.push_str(&format!("\n- {} «{}»", it.node_type, it.title));
            }
        }
        Ok(ArmOutput {
            project: r.decision.project.as_ref().map(|p| p.slug.clone()),
            items,
            containers,
            text,
            tokens,
            guessed_from_cwd: r.decision.source == ProjectSource::InferredFromCwd,
            map_stable,
        })
    }
}

/// The seam of the synaptic layer (T8): given the anchors arm's output, the items
/// a best-first synaptic expansion adds. The implementation of #856 plugs here
/// once merged; none exists on main.
trait SynapticLayer: Send + Sync {
    fn extend(&self, w: &World, c: &Case, base: &ArmOutput) -> Vec<String>;
}

struct AnchorsSynapsesArm {
    anchors: AnchorsArm,
    layer: Option<Box<dyn SynapticLayer>>,
}

#[async_trait::async_trait]
impl Arm for AnchorsSynapsesArm {
    fn id(&self) -> &'static str {
        "anchors_synapses"
    }
    fn describe(&self) -> &'static str {
        "ancres + couche synaptique (sans couche branchée : = ancres)"
    }
    fn status(&self) -> ArmStatus {
        match self.layer {
            Some(_) => ArmStatus::OnMain,
            None => ArmStatus::Pending { pr: "#856" },
        }
    }
    async fn context(&self, w: &World, c: &Case) -> anyhow::Result<ArmOutput> {
        let mut out = self.anchors.context(w, c).await?;
        if let Some(layer) = &self.layer {
            for id in layer.extend(w, c, &out) {
                if let Some(it) = w.items.get(&id) {
                    let line = format!("\n- {} «{}»", it.node_type, it.title);
                    out.tokens += est_tokens(&line);
                    out.text.push_str(&line);
                }
                push_unique(&mut out.items, &id);
            }
        }
        Ok(out)
    }
}

fn arms() -> Vec<Box<dyn Arm>> {
    vec![
        Box::new(NoContext),
        Box::new(BaselineCwd),
        Box::new(AnchorsArm::from_env()),
        Box::new(AnchorsSynapsesArm {
            anchors: AnchorsArm::from_env(),
            layer: None,
        }),
    ]
}

// ----------------------------------------------------------------------------
// Leak oracle (written from the rules, not from the resolver)
// ----------------------------------------------------------------------------

/// Projects a human opened: live user project anchors (existing project) and
/// the projects of live user workspace anchors; the decided project aside.
fn opened_by_human(w: &World, anchors: &[Anchor], decided: Option<&str>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for a in anchors
        .iter()
        .filter(|a| a.state == AnchorState::Live && a.by == AnchorActor::User)
    {
        match a.target_type {
            AnchorTargetType::Project => {
                if let Some(slug) = w.project_slug(&a.target_id) {
                    out.insert(slug);
                }
            }
            AnchorTargetType::Workspace => {
                if let Some(key) = w.workspace_key(&a.target_id) {
                    out.extend(w.workspaces[&key].1.iter().cloned());
                }
            }
            _ => {}
        }
    }
    if let Some(d) = decided {
        out.remove(d);
    }
    out
}

fn item_readable(it: &ItemFx, decided: Option<&str>, opened: &BTreeSet<String>) -> bool {
    if Some(it.project.as_str()) == decided {
        return true;
    }
    match it.consent {
        SharingConsent::ExplicitDeny => false,
        SharingConsent::ExplicitAllow => true,
        SharingConsent::NotSet | SharingConsent::PolicyAuto => opened.contains(&it.project),
    }
}

/// Every injected item / named container / rendered title the oracle refuses.
fn leaks(w: &World, anchors: &[Anchor], out: &ArmOutput) -> Vec<String> {
    let decided = out.project.as_deref();
    let opened = opened_by_human(w, anchors, decided);
    let mut v = Vec::new();
    for id in &out.items {
        match w.items.get(id) {
            Some(it) if item_readable(it, decided, &opened) => {}
            Some(_) => v.push(format!("item:{id}")),
            None => v.push(format!("inconnu:{id}")),
        }
    }
    for c in &out.containers {
        if let Some(slug) = c.strip_prefix("project:") {
            if Some(slug) != decided && !opened.contains(slug) {
                v.push(c.clone());
            }
        }
    }
    // A title rendered without the item being listed: the text path. The targets
    // of the session's own anchors are known to it already (a broken anchor shows
    // its id, which for a file is its path).
    let own: BTreeSet<&str> = anchors.iter().map(|a| a.target_id.as_str()).collect();
    for it in w.items.values() {
        if own.contains(it.id.as_str()) || item_readable(it, decided, &opened) {
            continue;
        }
        let listed = format!("item:{}", it.id);
        if out.text.contains(&it.title) && !v.contains(&listed) {
            v.push(format!("texte:{}", it.id));
        }
    }
    v
}

// ----------------------------------------------------------------------------
// Statistics
// ----------------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pct(&mut self) -> u64 {
        self.next() % 100
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Mean and percentile bootstrap 95 % interval.
fn bootstrap(xs: &[f64], seed: u64) -> (f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = xs.len();
    let mut rng = SplitMix64(seed);
    let mut ms: Vec<f64> = (0..BOOTSTRAP)
        .map(|_| (0..n).map(|_| xs[rng.below(n)]).sum::<f64>() / n as f64)
        .collect();
    ms.sort_by(f64::total_cmp);
    let lo = ms[(BOOTSTRAP as f64 * 0.025) as usize];
    let hi = ms[(BOOTSTRAP as f64 * 0.975) as usize - 1];
    (mean(xs), lo, hi)
}

fn ci_json((m, lo, hi): (f64, f64, f64)) -> Value {
    json!({ "mean": round(m), "ci95": [round(lo), round(hi)] })
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

// ----------------------------------------------------------------------------
// Context evaluation
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QRow {
    id: String,
    category: String,
    injected: usize,
    hits: usize,
    relevant: usize,
    stale: usize,
    tokens: usize,
    recall: f64,
    leaks: Vec<String>,
    guessed: bool,
    map_stable: Option<bool>,
}

fn question_cases<'q>(w: &World, q: &'q QuestionsFx) -> Vec<(Case, &'q QuestionFx)> {
    q.questions
        .iter()
        .map(|qq| {
            let session_id = det_uuid(&format!("question:{}", qq.id));
            let anchors = qq
                .anchors
                .iter()
                .enumerate()
                .map(|(i, a)| build_anchor(Some(w), session_id, &format!("{i}"), a))
                .collect();
            (
                Case {
                    key: qq.id.clone(),
                    session_id,
                    session: qq
                        .session
                        .clone()
                        .unwrap_or_else(|| q.default_session.clone()),
                    anchors,
                    check_stability: true,
                },
                qq,
            )
        })
        .collect()
}

async fn eval_questions(w: &World, q: &QuestionsFx, arm: &dyn Arm) -> Vec<QRow> {
    let mut rows = Vec::new();
    for (case, qq) in question_cases(w, q) {
        let out = arm
            .context(w, &case)
            .await
            .unwrap_or_else(|e| panic!("{} / {}: {e:#}", arm.id(), qq.id));
        let rel: BTreeSet<&str> = qq.relevant.iter().map(String::as_str).collect();
        let stale: BTreeSet<&str> = qq.stale.iter().map(String::as_str).collect();
        let hits = out
            .items
            .iter()
            .filter(|i| rel.contains(i.as_str()))
            .count();
        let stale_hits = out
            .items
            .iter()
            .filter(|i| stale.contains(i.as_str()) || w.items.get(*i).is_some_and(|x| x.stale))
            .count();
        rows.push(QRow {
            id: qq.id.clone(),
            category: qq.category.clone(),
            injected: out.items.len(),
            hits,
            relevant: rel.len(),
            stale: stale_hits,
            tokens: out.tokens,
            recall: if rel.is_empty() {
                1.0
            } else {
                hits as f64 / rel.len() as f64
            },
            leaks: leaks(w, &case.anchors, &out),
            guessed: out.guessed_from_cwd,
            map_stable: out.map_stable,
        });
    }
    rows
}

fn context_summary(rows: &[QRow], seed: u64) -> Value {
    let recalls: Vec<f64> = rows.iter().map(|r| r.recall).collect();
    let tokens: Vec<f64> = rows.iter().map(|r| r.tokens as f64).collect();
    let precs: Vec<f64> = rows
        .iter()
        .filter(|r| r.injected > 0)
        .map(|r| r.hits as f64 / r.injected as f64)
        .collect();
    let injected: usize = rows.iter().map(|r| r.injected).sum();
    let hits: usize = rows.iter().map(|r| r.hits).sum();
    let mut by_cat: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for r in rows {
        by_cat
            .entry(r.category.as_str())
            .or_default()
            .push(r.recall);
    }
    let stable: Vec<bool> = rows.iter().filter_map(|r| r.map_stable).collect();
    json!({
        "questions": rows.len(),
        "recall": ci_json(bootstrap(&recalls, seed)),
        "precision_macro": ci_json(bootstrap(&precs, seed ^ 1)),
        "precision_micro": round(if injected == 0 { 0.0 } else { hits as f64 / injected as f64 }),
        "questions_without_context": rows.iter().filter(|r| r.injected == 0).count(),
        "tokens_per_turn": ci_json(bootstrap(&tokens, seed ^ 2)),
        "stale_injected": rows.iter().map(|r| r.stale).sum::<usize>(),
        "questions_with_stale": rows.iter().filter(|r| r.stale > 0).count(),
        "guessed_from_cwd": rows.iter().filter(|r| r.guessed).count(),
        "map_stable_rate": if stable.is_empty() { Value::Null } else {
            json!(round(stable.iter().filter(|s| **s).count() as f64 / stable.len() as f64))
        },
        "recall_by_category": by_cat.iter().map(|(k, v)| (k.to_string(), json!(round(mean(v))))).collect::<serde_json::Map<_, _>>(),
        "leaks": rows.iter().map(|r| r.leaks.len()).sum::<usize>(),
    })
}

fn paired(a: &[QRow], b: &[QRow], seed: u64) -> Value {
    let d: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x.recall - y.recall) * 100.0)
        .collect();
    ci_json(bootstrap(&d, seed))
}

// ----------------------------------------------------------------------------
// Leak replay
// ----------------------------------------------------------------------------

fn target_type_of(node_type: &str) -> AnchorTargetType {
    use AnchorTargetType as T;
    match node_type {
        "file" => T::File,
        "function" => T::Function,
        "feature_graph" => T::Feature,
        "release" => T::Release,
        "plan" => T::Plan,
        "task" => T::Task,
        "decision" => T::Decision,
        _ => T::Note,
    }
}

fn replay_cases(w: &World, fx: &ReplayFx) -> Vec<Case> {
    let mut rng = SplitMix64(fx.seed);
    let slugs: Vec<&String> = w.projects.keys().collect();
    let keys: Vec<&String> = w.workspaces.keys().collect();
    let items: Vec<&ItemFx> = w.items.values().collect();
    (0..fx.resolutions)
        .map(|i| {
            let session_id = det_uuid(&format!("replay:{}:{i}", fx.seed));
            let r = rng.pct();
            let pick = slugs[rng.below(slugs.len())].clone();
            let root = w.projects[&pick].1.root.clone();
            let session = match r {
                0..=49 => SessionFx {
                    project: Some("atelier".into()),
                    cwd: w.projects["atelier"].1.root.clone(),
                    place: ExecutionPlace::Project,
                },
                50..=69 => SessionFx {
                    project: Some(pick),
                    cwd: root,
                    place: ExecutionPlace::Project,
                },
                // explicit slug, cwd of ANOTHER project
                70..=79 => SessionFx {
                    project: Some("atelier".into()),
                    cwd: root,
                    place: ExecutionPlace::Project,
                },
                // inherited session: no slug, a project's cwd
                80..=91 => SessionFx {
                    project: None,
                    cwd: format!("{root}/src"),
                    place: ExecutionPlace::Project,
                },
                _ => SessionFx {
                    project: None,
                    cwd: format!("/var/fx/neutral/r{i}"),
                    place: ExecutionPlace::Neutral,
                },
            };
            let n = rng.below(6);
            let anchors = (0..n)
                .map(|j| {
                    let (t, id) = match rng.below(12) {
                        0 => (
                            AnchorTargetType::Project,
                            slugs[rng.below(slugs.len())].clone(),
                        ),
                        1 => (
                            AnchorTargetType::Workspace,
                            keys[rng.below(keys.len())].clone(),
                        ),
                        2 => (AnchorTargetType::Component, format!("c-{j}")),
                        3 => (AnchorTargetType::Note, format!("absent-{i}-{j}")),
                        _ => {
                            let it = items[rng.below(items.len())];
                            (target_type_of(&it.node_type), it.id.clone())
                        }
                    };
                    let mut roles: Vec<AnchorRole> = AnchorRole::ALL
                        .iter()
                        .copied()
                        .filter(|_| rng.pct() < 45)
                        .collect();
                    if roles.is_empty() {
                        roles.push(AnchorRole::ALL[rng.below(AnchorRole::ALL.len())]);
                    }
                    let by = match rng.pct() {
                        0..=49 => AnchorActor::User,
                        50..=84 => AnchorActor::Agent,
                        _ => AnchorActor::System,
                    };
                    let state = match rng.pct() {
                        0..=79 => AnchorState::Live,
                        80..=87 => AnchorState::Moved,
                        88..=93 => AnchorState::Dangling,
                        94..=97 => AnchorState::Archived,
                        _ => AnchorState::Unknown,
                    };
                    let afx = AnchorFx {
                        t,
                        id,
                        k: None,
                        roles,
                        by,
                        inferred: by == AnchorActor::Agent,
                        state,
                        from: None,
                    };
                    build_anchor(Some(w), session_id, &format!("{j}"), &afx)
                })
                .collect();
            Case {
                key: format!("replay-{i}"),
                session_id,
                session,
                anchors,
                check_stability: false,
            }
        })
        .collect()
}

struct ReplayRow {
    resolutions: usize,
    leaks: usize,
    examples: Vec<String>,
    guessed_from_cwd: usize,
    with_content: usize,
}

async fn eval_replay(w: &World, cases: &[Case], arm: &dyn Arm) -> ReplayRow {
    let mut row = ReplayRow {
        resolutions: cases.len(),
        leaks: 0,
        examples: Vec::new(),
        guessed_from_cwd: 0,
        with_content: 0,
    };
    for c in cases {
        let out = arm
            .context(w, c)
            .await
            .unwrap_or_else(|e| panic!("{} / {}: {e:#}", arm.id(), c.key));
        let l = leaks(w, &c.anchors, &out);
        if !l.is_empty() {
            row.leaks += l.len();
            if row.examples.len() < 10 {
                row.examples.push(format!("{}: {}", c.key, l.join(", ")));
            }
        }
        row.guessed_from_cwd += usize::from(out.guessed_from_cwd);
        row.with_content += usize::from(!out.items.is_empty());
    }
    row
}

// ----------------------------------------------------------------------------
// Posture evaluation (the real tracker of main)
// ----------------------------------------------------------------------------

struct Turn {
    focus: Vec<AnchorFx>,
    oscillation: bool,
    hostile: Option<String>,
}

struct Conversation {
    id: String,
    access: SessionAccess,
    turns: Vec<Turn>,
}

fn expand_conversations(fx: &PostureFx) -> Vec<Conversation> {
    let mut out: Vec<Conversation> = fx
        .conversations
        .iter()
        .map(|c| {
            let mut turns = Vec::new();
            for b in &c.turns {
                match &b.alternate {
                    Some(alt) if !alt.is_empty() => {
                        for i in 0..b.repeat {
                            turns.push(Turn {
                                focus: alt[i % alt.len()].clone(),
                                oscillation: true,
                                hostile: None,
                            });
                        }
                    }
                    _ => {
                        for _ in 0..b.repeat {
                            turns.push(Turn {
                                focus: b.focus.clone(),
                                oscillation: false,
                                hostile: None,
                            });
                        }
                    }
                }
            }
            Conversation {
                id: c.id.clone(),
                access: c.access,
                turns,
            }
        })
        .collect();
    // hostile conversations: 4 calm turns, 1 hostile, 2 calm
    let all = AnchorTargetType::ALL;
    for (i, prompt) in fx.hostile_prompts.iter().enumerate() {
        let base = AnchorFx {
            t: all[i % all.len()],
            id: String::new(),
            k: Some("base".into()),
            roles: vec![AnchorRole::Focus],
            by: AnchorActor::User,
            inferred: false,
            state: AnchorState::Live,
            from: None,
        };
        let injected = AnchorFx {
            t: all[(i + 5) % all.len()],
            k: Some("injected".into()),
            by: AnchorActor::Agent,
            inferred: true,
            ..base.clone()
        };
        let calm = || Turn {
            focus: vec![base.clone()],
            oscillation: false,
            hostile: None,
        };
        let mut turns: Vec<Turn> = (0..4).map(|_| calm()).collect();
        turns.push(Turn {
            focus: vec![base.clone(), injected],
            oscillation: false,
            hostile: Some(prompt.clone()),
        });
        turns.extend((0..2).map(|_| calm()));
        out.push(Conversation {
            id: format!("hostile-{:02}", i + 1),
            access: SessionAccess::ReadOnly,
            turns,
        });
    }
    out
}

#[derive(Default)]
struct PostureTotals {
    turns: usize,
    switches: usize,
    explicit_switches: usize,
    stable_switches: usize,
    osc_turns: usize,
    osc_switches: usize,
    hostile_turns: usize,
    violations: Vec<String>,
    transitions: usize,
    prefix_hits: usize,
    prefix_hits_without_posture: usize,
    segment_tokens: usize,
    types: BTreeSet<&'static str>,
    calm_turns: usize,
    calm_switches: usize,
}

fn eval_posture(convs: &[Conversation]) -> PostureTotals {
    let mut t = PostureTotals::default();
    for c in convs {
        let session = det_uuid(&format!("posture:{}", c.id));
        let inv = Invariants {
            access: c.access,
            permission_mode: Some("default".into()),
        };
        let reference = inv.render();
        let mut tracker = PostureTracker::default();
        let mut prev_prefix: Option<String> = None;
        let mut prev_eff: Option<Posture> = None;
        let hostile_conv = c.id.starts_with("hostile-");
        for (i, turn) in c.turns.iter().enumerate() {
            let anchors: Vec<Anchor> = turn
                .focus
                .iter()
                .map(|a| {
                    let key = a.k.clone().unwrap_or_else(|| a.t.as_str().to_string());
                    let mut x = build_anchor(None, session, &key, a);
                    x.target_id = format!("{}-{key}", a.t);
                    x
                })
                .collect();
            for a in &anchors {
                t.types.insert(a.target_type.as_str());
            }
            let derived = posture::derive(&anchors, None).within(&inv);
            let explicit = posture::explicit_focus(&anchors, None);
            let switch = tracker.observe(derived.clone(), explicit);
            let eff = tracker.effective().cloned().map(|p| p.within(&inv));
            let prefix = posture::compose(&inv, eff.as_ref());
            t.turns += 1;
            if !hostile_conv {
                t.calm_turns += 1;
            }
            if inv.render() != reference || !prefix.starts_with(&reference) {
                t.violations
                    .push(format!("{}#{i}: segment invariant modifié", c.id));
            }
            for tool in derived
                .tools
                .iter()
                .chain(eff.iter().flat_map(|p| p.tools.iter()))
            {
                if !inv.admits(tool) {
                    t.violations
                        .push(format!("{}#{i}: outil non admis proposé ({tool})", c.id));
                }
            }
            if turn.hostile.is_some() {
                t.hostile_turns += 1;
                if switch.is_some() || eff != prev_eff {
                    t.violations
                        .push(format!("{}#{i}: posture changée par un tour hostile", c.id));
                }
            }
            if let Some(s) = &switch {
                t.switches += 1;
                if !hostile_conv {
                    t.calm_switches += 1;
                }
                match s.cause {
                    SwitchCause::ExplicitAnchor => t.explicit_switches += 1,
                    SwitchCause::Stable => t.stable_switches += 1,
                    SwitchCause::Override => {}
                }
                if turn.oscillation {
                    t.osc_switches += 1;
                }
            }
            if turn.oscillation {
                t.osc_turns += 1;
            }
            if let Some(p) = &prev_prefix {
                t.transitions += 1;
                t.prefix_hits += usize::from(*p == prefix);
                // without any posture the prefix is the invariant segment alone: always a hit
                t.prefix_hits_without_posture += 1;
            }
            t.segment_tokens += eff.as_ref().map_or(0, |p| est_tokens(&p.render()));
            prev_prefix = Some(prefix);
            prev_eff = eff;
        }
    }
    t
}

// ----------------------------------------------------------------------------
// Report
// ----------------------------------------------------------------------------

struct Run {
    report: Value,
    total_leaks: usize,
    posture_violations: Vec<String>,
    leak_examples: Vec<String>,
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

async fn run(arms: &[Box<dyn Arm>], replay: bool) -> Run {
    let world_fx: WorldFx = serde_json::from_str(WORLD_JSON).expect("world.json");
    let q_fx: QuestionsFx = serde_json::from_str(QUESTIONS_JSON).expect("questions.json");
    let p_fx: PostureFx = serde_json::from_str(POSTURE_JSON).expect("posture.json");
    let w = World::build(&world_fx).await;
    for qq in &q_fx.questions {
        for id in qq.relevant.iter().chain(&qq.stale) {
            assert!(w.items.contains_key(id), "{}: unknown label {id}", qq.id);
        }
    }

    let mut total_leaks = 0;
    let mut leak_examples = Vec::new();
    let mut ctx_rows: BTreeMap<&'static str, Vec<QRow>> = BTreeMap::new();
    let mut arm_json = Vec::new();
    let replay_set = replay.then(|| replay_cases(&w, &world_fx.leak_replay));
    for (n, arm) in arms.iter().enumerate() {
        let rows = eval_questions(&w, &q_fx, arm.as_ref()).await;
        let mut summary = context_summary(&rows, BOOTSTRAP_SEED + n as u64);
        let q_leaks: usize = rows.iter().map(|r| r.leaks.len()).sum();
        total_leaks += q_leaks;
        for r in rows.iter().filter(|r| !r.leaks.is_empty()) {
            leak_examples.push(format!("{} / {}: {}", arm.id(), r.id, r.leaks.join(", ")));
        }
        if let Some(cases) = &replay_set {
            let rr = eval_replay(&w, cases, arm.as_ref()).await;
            total_leaks += rr.leaks;
            leak_examples.extend(rr.examples.iter().map(|e| format!("{} / {e}", arm.id())));
            summary["replay"] = json!({
                "resolutions": rr.resolutions,
                "leaks": rr.leaks,
                "with_content": rr.with_content,
                "guessed_from_cwd": rr.guessed_from_cwd,
                "examples": rr.examples,
            });
        }
        arm_json.push(json!({
            "id": arm.id(),
            "describe": arm.describe(),
            "status": arm.status().label(),
            "context": summary,
            "per_question": rows.iter().map(|r| json!({
                "id": r.id, "category": r.category, "injected": r.injected, "hits": r.hits,
                "relevant": r.relevant, "stale": r.stale, "tokens": r.tokens,
                "leaks": r.leaks,
            })).collect::<Vec<_>>(),
        }));
        ctx_rows.insert(arm.id(), rows);
    }

    let pt = eval_posture(&expand_conversations(&p_fx));
    let switch_rate = ratio(pt.switches, pt.turns);
    let calm_rate = ratio(pt.calm_switches, pt.calm_turns);
    let hit = ratio(pt.prefix_hits, pt.transitions);
    let hit_bare = ratio(pt.prefix_hits_without_posture, pt.transitions);
    let posture_json = json!({
        "status": "main (#853)",
        "conversations": p_fx.conversations.len() + p_fx.hostile_prompts.len(),
        "turns": pt.turns,
        "anchor_types_covered": pt.types.iter().collect::<Vec<_>>(),
        "switches": pt.switches,
        "switch_rate": round(switch_rate),
        "switch_rate_calm": round(calm_rate),
        "switches_explicit": pt.explicit_switches,
        "switches_stable": pt.stable_switches,
        "oscillation_turns": pt.osc_turns,
        "oscillation_switches": pt.osc_switches,
        "hostile_prompts": p_fx.hostile_prompts.len(),
        "hostile_turns": pt.hostile_turns,
        "violations": pt.violations,
        "prefix_cache_hit": round(hit),
        "prefix_cache_hit_without_posture": round(hit_bare),
        "segment_tokens_per_turn": round(ratio(pt.segment_tokens, pt.turns)),
    });

    // comparisons and go / no-go
    let get = |id: &str| ctx_rows.get(id).cloned().unwrap_or_default();
    let (none, base, anch, syn) = (
        get("none"),
        get("baseline_cwd"),
        get("anchors"),
        get("anchors_synapses"),
    );
    let mut comparisons = json!({});
    let mut go = json!({});
    if !anch.is_empty() && !none.is_empty() {
        let gain = bootstrap(
            &anch
                .iter()
                .zip(&none)
                .map(|(a, n)| (a.recall - n.recall) * 100.0)
                .collect::<Vec<_>>(),
            BOOTSTRAP_SEED ^ 0xA,
        );
        comparisons["anchors_vs_none_recall_pts"] = ci_json(gain);
        let stable = anch.iter().all(|r| r.map_stable != Some(false));
        go["anchors_shadow_to_on"] = json!({
            "offline_verdict": if gain.1 >= RECALL_GAIN_VS_NONE_PTS && total_leaks == 0 && stable { "go" } else { "no-go" },
            "rule": format!("borne basse du gain de rappel vs aucun contexte ≥ {RECALL_GAIN_VS_NONE_PTS} pts, 0 fuite, carte stable"),
            "not_measured_offline": "justesse des réponses (modèle), latence p95 < 150 ms, cache réel du préfixe",
        });
    }
    if !anch.is_empty() && !base.is_empty() {
        comparisons["anchors_vs_baseline_recall_pts"] = paired(&anch, &base, BOOTSTRAP_SEED ^ 0xB);
        let t = |rows: &[QRow]| mean(&rows.iter().map(|r| r.tokens as f64).collect::<Vec<_>>());
        comparisons["anchors_vs_baseline_tokens_ratio"] = json!(round(ratio_f(t(&anch), t(&base))));
    }
    if !syn.is_empty() && !anch.is_empty() {
        let t = |rows: &[QRow]| mean(&rows.iter().map(|r| r.tokens as f64).collect::<Vec<_>>());
        let overhead = ratio_f(t(&syn), t(&anch)) - 1.0;
        comparisons["synapses_vs_anchors_recall_pts"] = paired(&syn, &anch, BOOTSTRAP_SEED ^ 0xC);
        comparisons["synapses_vs_anchors_token_overhead"] = json!(round(overhead));
        let pending = arms
            .iter()
            .find(|a| a.id() == "anchors_synapses")
            .map(|a| a.status())
            .filter(|s| matches!(s, ArmStatus::Pending { .. }));
        go["synapses_shadow_to_on"] = json!({
            "offline_verdict": match pending {
                Some(s) => s.label(),
                None if overhead <= SYNAPSE_TOKEN_OVERHEAD_MAX && total_leaks == 0 => "go".into(),
                None => "no-go".into(),
            },
            "rule": format!("jetons ≤ +{:.0} % vs ancres, 0 fuite ; en ligne : B ≥ A + 3 pts sur 14 jours", SYNAPSE_TOKEN_OVERHEAD_MAX * 100.0),
        });
    }
    let posture_ok = calm_rate < SWITCH_RATE_MAX
        && pt.violations.is_empty()
        && (hit_bare - hit) * 100.0 <= PREFIX_DROP_MAX_PTS;
    go["posture_shadow_to_on"] = json!({
        "offline_verdict": if posture_ok { "go" } else { "no-go" },
        "rule": format!("bascules < {SWITCH_RATE_MAX} par tour (conversations calmes), 0 violation hostile, cache du préfixe pas pire de {PREFIX_DROP_MAX_PTS} pts"),
        "not_measured_offline": "≥ 500 tours réels en ombre",
    });

    let report = json!({
        "report_version": REPORT_VERSION,
        "task": "T9 — 3547cca0-e653-4a7f-b26b-d3da44f37c9b",
        "fixtures": {
            "world": { "version": world_fx.version, "sha256_8": fixture_digest(WORLD_JSON), "items": w.items.len() },
            "questions": { "version": q_fx.version, "sha256_8": fixture_digest(QUESTIONS_JSON), "count": q_fx.questions.len(),
                "texts": q_fx.questions.iter().map(|q| (q.id.clone(), json!(q.question))).collect::<serde_json::Map<_, _>>() },
            "posture": { "version": p_fx.version, "sha256_8": fixture_digest(POSTURE_JSON) },
            "leak_replay": { "seed": world_fx.leak_replay.seed, "resolutions": if replay { world_fx.leak_replay.resolutions } else { 0 } },
        },
        "arms": arm_json,
        "comparisons": comparisons,
        "posture": posture_json,
        "leak_gate": { "total_leaks": total_leaks, "passed": total_leaks == 0, "examples": leak_examples.iter().take(20).collect::<Vec<_>>() },
        "go_no_go": go,
    });
    Run {
        report,
        total_leaks,
        posture_violations: pt.violations,
        leak_examples,
    }
}

fn ratio_f(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}

fn print_table(r: &Value) {
    println!("\n=== T9 — évaluation transverse (hors ligne) ===");
    println!(
        "{:<17} {:<14} {:>16} {:>9} {:>8} {:>7} {:>6} {:>8} {:>7}",
        "bras", "statut", "rappel [IC95]", "préc.µ", "jetons", "périmé", "cwd?", "rejeu", "fuites"
    );
    for a in r["arms"].as_array().into_iter().flatten() {
        let c = &a["context"];
        let rec = &c["recall"];
        let leaks = c["leaks"].as_u64().unwrap_or(0) + c["replay"]["leaks"].as_u64().unwrap_or(0);
        println!(
            "{:<17} {:<14} {:>5.3} [{:.2},{:.2}] {:>9.3} {:>8.0} {:>7} {:>6} {:>8} {:>7}",
            a["id"].as_str().unwrap_or(""),
            a["status"].as_str().unwrap_or(""),
            rec["mean"].as_f64().unwrap_or(0.0),
            rec["ci95"][0].as_f64().unwrap_or(0.0),
            rec["ci95"][1].as_f64().unwrap_or(0.0),
            c["precision_micro"].as_f64().unwrap_or(0.0),
            c["tokens_per_turn"]["mean"].as_f64().unwrap_or(0.0),
            c["stale_injected"].as_u64().unwrap_or(0),
            c["guessed_from_cwd"].as_u64().unwrap_or(0),
            c["replay"]["resolutions"].as_u64().unwrap_or(0),
            leaks,
        );
    }
    let p = &r["posture"];
    println!(
        "posture : {} tours, bascules {} (taux {} ; calmes {}), oscillation {}/{} tours, hostiles {} tours / {} violations, cache du préfixe {} (sans posture {})",
        p["turns"], p["switches"], p["switch_rate"], p["switch_rate_calm"], p["oscillation_switches"], p["oscillation_turns"],
        p["hostile_turns"], p["violations"].as_array().map_or(0, Vec::len), p["prefix_cache_hit"], p["prefix_cache_hit_without_posture"],
    );
    println!("comparaisons : {}", r["comparisons"]);
    println!("go/no-go : {}", r["go_no_go"]);
    println!("barrière de fuite : {}", r["leak_gate"]);
}

fn report_path() -> std::path::PathBuf {
    std::env::var_os("PO_EVAL_REPORT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("eval-transverse")
                .join("report.json")
        })
}

/// The gate of the CI: no leak, no posture violation.
fn gate(run: &Run) -> Result<(), String> {
    let mut errs = Vec::new();
    if run.total_leaks > 0 {
        errs.push(format!(
            "{} fuite(s) : {}",
            run.total_leaks,
            run.leak_examples
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }
    if !run.posture_violations.is_empty() {
        errs.push(format!(
            "{} violation(s) de posture : {}",
            run.posture_violations.len(),
            run.posture_violations.join(" | ")
        ));
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    /// The run of the CI: the whole fixture set, the 10 000-resolution replay,
    /// the report written, then the gate.
    #[tokio::test]
    async fn eval_transverse_report_and_leak_gate() {
        let run = run(&arms(), true).await;
        print_table(&run.report);
        let path = report_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).expect("report dir");
        }
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&run.report).expect("report json"),
        )
        .expect("report write");
        println!("rapport : {}", path.display());
        // the shape the plan asks for
        assert_eq!(run.report["fixtures"]["questions"]["count"], 50);
        assert!(run.report["posture"]["turns"].as_u64().unwrap_or(0) >= 200);
        assert_eq!(
            run.report["posture"]["anchor_types_covered"]
                .as_array()
                .map(Vec::len),
            Some(AnchorTargetType::ALL.len())
        );
        assert_eq!(run.report["posture"]["hostile_prompts"], 30);
        assert_eq!(
            run.report["arms"][2]["context"]["replay"]["resolutions"],
            10_000
        );
        if let Err(e) = gate(&run) {
            panic!("barrière T9 rouge :\n{e}");
        }
    }

    /// The red proof of the gate, kept in the CI: the anchors arm made to inject
    /// a non-consented item (a sealed note of Granit) is caught.
    #[tokio::test]
    async fn leak_gate_turns_red_on_an_injected_non_consented_item() {
        let leaking: Vec<Box<dyn Arm>> = vec![Box::new(AnchorsArm {
            inject_leak: Some("n-granit-contrat".into()),
        })];
        let run = run(&leaking, false).await;
        // every question but the two asked from Granit itself (q15, q42), where the
        // note is the session project's own
        assert_eq!(run.total_leaks, 48);
        let err = gate(&run).expect_err("an injected leak must fail the gate");
        assert!(err.contains("item:n-granit-contrat"), "{err}");
        // and the text path alone is caught too
        let w = World::build(&serde_json::from_str(WORLD_JSON).unwrap()).await;
        let out = ArmOutput {
            project: Some("atelier".into()),
            text: "- note «Note : contrat client confidentiel»".into(),
            ..Default::default()
        };
        assert_eq!(
            leaks(&w, &[], &out),
            vec!["texte:n-granit-contrat".to_string()]
        );
    }

    /// The oracle follows the stated rules, independently of the resolver.
    #[tokio::test]
    async fn oracle_reads_the_rules() {
        let w = World::build(&serde_json::from_str(WORLD_JSON).unwrap()).await;
        let it = |id: &str| w.items[id].clone();
        let none = BTreeSet::new();
        let brume: BTreeSet<String> = ["brume".to_string()].into();
        assert!(item_readable(&it("pl-migr"), Some("atelier"), &none));
        assert!(!item_readable(&it("pl-capt"), Some("atelier"), &none));
        assert!(item_readable(&it("pl-capt"), Some("atelier"), &brume));
        assert!(item_readable(&it("n-brume-shared"), Some("atelier"), &none));
        assert!(!item_readable(
            &it("n-brume-secret"),
            Some("atelier"),
            &brume
        ));
        assert!(!item_readable(&it("pl-migr"), None, &none));
        // an agent's project anchor opens nothing; a human's does
        let sid = Uuid::nil();
        let fx = |by| AnchorFx {
            t: AnchorTargetType::Project,
            id: "brume".into(),
            k: None,
            roles: vec![AnchorRole::Mention],
            by,
            inferred: false,
            state: AnchorState::Live,
            from: None,
        };
        let agent = build_anchor(Some(&w), sid, "a", &fx(AnchorActor::Agent));
        let user = build_anchor(Some(&w), sid, "u", &fx(AnchorActor::User));
        assert!(opened_by_human(&w, &[agent], Some("atelier")).is_empty());
        assert_eq!(opened_by_human(&w, &[user], Some("atelier")), brume);
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_mean() {
        let xs: Vec<f64> = (0..50).map(|i| (i % 7) as f64 / 6.0).collect();
        let a = bootstrap(&xs, 7);
        assert_eq!(a, bootstrap(&xs, 7));
        assert!(a.1 <= a.0 && a.0 <= a.2, "{a:?}");
    }
}
