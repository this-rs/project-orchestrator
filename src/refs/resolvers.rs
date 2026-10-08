//! One resolver per kind, each delegating to the `get_*` / `list_*` the rest of
//! the platform already uses. Nothing here decides who may see what: a resolver
//! is the *unchecked* read ([`RefSource`]) and only
//! [`AccessPolicy::resolve_checked`](super::access::AccessPolicy::resolve_checked)
//! is meant to call it.
//!
//! What the store does not give us is derived, here, once:
//!
//! * a **note** has no title: its label is its first line (or markdown title);
//! * a **decision** has only a description: that is its label;
//! * a **task**'s title is optional: first line of its description otherwise;
//! * an **RFC** is a note of type `rfc` whose content is JSON
//!   `{"title": ..., "sections": [...]}`; its status is the `rfc-status:` tag.
//!   `note` and `rfc` are disjoint: a note reference never designates an RFC.
//!
//! Text search exists for plans (`list_plans_filtered`) and notes
//! (`NoteFilters::search`) and is used as is. It does **not** exist for tasks
//! (`list_all_tasks_filtered` has no `search`) nor for decisions (their text
//! search lives in Meilisearch, not in the `GraphStore`). Adding a parameter to
//! the trait would mean touching `mock.rs`, `impl_graph_store.rs` and a Cypher
//! query in the coverage-ignored `neo4j/task.rs` for a UI suggestion box; the
//! smaller change is a bounded in-memory filter over the most recent
//! [`SCAN_CAP`] rows. The price is stated, not hidden: a task or decision older
//! than the newest `SCAN_CAP` of its scope is not suggested by *text*; it can
//! still be referenced by id or by dropping it, and it always resolves.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;
use uuid::Uuid;

use super::access::{RefMeta, RefSource};
use super::label;
use super::types::{EntityRef, RefKind};
use super::wire::ScopeLabel;
use crate::neo4j::models::{DecisionNode, DecisionStatus, PlanNode, TaskWithPlan};
use crate::neo4j::GraphStore;
use crate::notes::{Note, NoteFilters, NoteType};

/// Most rows a text filter done in memory looks at, per kind.
pub const SCAN_CAP: usize = 300;

/// The note types a `note` reference may designate: every one but `rfc`.
/// The exhaustive match below makes a new type a compile error until it is
/// placed on one side.
const NOTE_TYPES: [NoteType; 7] = [
    NoteType::Guideline,
    NoteType::Gotcha,
    NoteType::Pattern,
    NoteType::Context,
    NoteType::Tip,
    NoteType::Observation,
    NoteType::Assertion,
];

const _: fn(NoteType) = |t| match t {
    NoteType::Guideline
    | NoteType::Gotcha
    | NoteType::Pattern
    | NoteType::Context
    | NoteType::Tip
    | NoteType::Observation
    | NoteType::Assertion
    | NoteType::Rfc => {}
};

const DECISION_STATUSES: [DecisionStatus; 4] = [
    DecisionStatus::Proposed,
    DecisionStatus::Accepted,
    DecisionStatus::Deprecated,
    DecisionStatus::Superseded,
];

/// What a search asks of one kind. `needle` is already trimmed and lowercased.
#[derive(Debug, Clone)]
pub struct Candidates {
    pub needle: String,
    pub project_id: Option<Uuid>,
    pub workspace_slug: Option<String>,
    /// How many rows to ask the store for (already over-fetched by the caller).
    pub fetch: usize,
}

/// A project and its workspace, as shown next to a result.
type Scope = (Option<ScopeLabel>, Option<ScopeLabel>);

/// Per-request memo: a page of results shares a few projects and plans.
#[derive(Default)]
pub struct Memo {
    scopes: HashMap<Uuid, Scope>,
    plans: HashMap<Uuid, Option<PlanNode>>,
}

/// The resolver of one kind.
#[async_trait]
pub trait KindResolver: Send + Sync {
    fn kind(&self) -> RefKind;

    /// The entity `id`, or `None`. Unchecked.
    async fn load(&self, id: Uuid, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>>;

    /// Recent entities matching the needle, most recent first. Unchecked.
    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<RefMeta>>;

    /// The parts of a result too costly to compute for every candidate.
    async fn finish(&self, _meta: &mut RefMeta) -> anyhow::Result<()> {
        Ok(())
    }
}

/// The wire spelling of an enum (`in_progress`, `active`...).
fn snake<T: Serialize>(v: &T) -> Option<String> {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
}

fn contains(haystack: &str, needle: &str) -> bool {
    needle.is_empty() || haystack.to_lowercase().contains(needle)
}

async fn scope_of(
    graph: &dyn GraphStore,
    memo: &mut Memo,
    project_id: Option<Uuid>,
) -> anyhow::Result<Scope> {
    let Some(pid) = project_id else {
        return Ok((None, None));
    };
    if let Some(s) = memo.scopes.get(&pid) {
        return Ok(s.clone());
    }
    let project = graph.get_project(pid).await?.map(|p| ScopeLabel {
        id: p.id,
        slug: p.slug,
        name: p.name,
    });
    let workspace = if project.is_some() {
        graph.get_project_workspace(pid).await?.map(|w| ScopeLabel {
            id: w.id,
            slug: w.slug,
            name: w.name,
        })
    } else {
        None
    };
    memo.scopes
        .insert(pid, (project.clone(), workspace.clone()));
    Ok((project, workspace))
}

async fn plan_of(
    graph: &dyn GraphStore,
    memo: &mut Memo,
    plan_id: Uuid,
) -> anyhow::Result<Option<PlanNode>> {
    if let Some(p) = memo.plans.get(&plan_id) {
        return Ok(p.clone());
    }
    let plan = graph.get_plan(plan_id).await?;
    memo.plans.insert(plan_id, plan.clone());
    Ok(plan)
}

// ----------------------------------------------------------------------------
// plan
// ----------------------------------------------------------------------------

pub struct PlanResolver {
    graph: Arc<dyn GraphStore>,
}

impl PlanResolver {
    async fn meta(&self, p: &PlanNode, memo: &mut Memo) -> anyhow::Result<RefMeta> {
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, p.project_id).await?;
        Ok(RefMeta {
            kind: RefKind::Plan,
            id: p.id,
            label: label::derive("plan", &p.id, &[&p.title]),
            subtitle: None,
            project,
            workspace,
            entity_status: snake(&p.status),
        })
    }
}

#[async_trait]
impl KindResolver for PlanResolver {
    fn kind(&self) -> RefKind {
        RefKind::Plan
    }

    async fn load(&self, id: Uuid, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        match self.graph.get_plan(id).await? {
            Some(p) => Ok(Some(self.meta(&p, memo).await?)),
            None => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
        let search = (!c.needle.is_empty()).then_some(c.needle.as_str());
        let (plans, _) = self
            .graph
            .list_plans_filtered(
                c.project_id,
                c.workspace_slug.as_deref(),
                None,
                None,
                None,
                search,
                c.fetch,
                0,
                Some("created_at"),
                "desc",
            )
            .await?;
        let mut out = Vec::with_capacity(plans.len());
        for p in &plans {
            out.push(self.meta(p, memo).await?);
        }
        Ok(out)
    }

    async fn finish(&self, meta: &mut RefMeta) -> anyhow::Result<()> {
        let n = self.graph.get_plan_tasks(meta.id).await?.len();
        meta.subtitle = Some(match n {
            1 => "1 tâche".to_string(),
            n => format!("{n} tâches"),
        });
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// task
// ----------------------------------------------------------------------------

pub struct TaskResolver {
    graph: Arc<dyn GraphStore>,
}

impl TaskResolver {
    async fn meta(
        &self,
        t: &crate::neo4j::models::TaskNode,
        plan: Option<&PlanNode>,
        memo: &mut Memo,
    ) -> anyhow::Result<RefMeta> {
        let (project, workspace) =
            scope_of(self.graph.as_ref(), memo, plan.and_then(|p| p.project_id)).await?;
        let title = t.title.as_deref().unwrap_or("");
        Ok(RefMeta {
            kind: RefKind::Task,
            id: t.id,
            label: label::derive("task", &t.id, &[title, &t.description]),
            subtitle: plan.map(|p| format!("Plan : {}", p.title)),
            project,
            workspace,
            entity_status: snake(&t.status),
        })
    }

    fn matches(t: &TaskWithPlan, needle: &str) -> bool {
        contains(t.task.title.as_deref().unwrap_or(""), needle)
            || contains(&t.task.description, needle)
    }
}

#[async_trait]
impl KindResolver for TaskResolver {
    fn kind(&self) -> RefKind {
        RefKind::Task
    }

    async fn load(&self, id: Uuid, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let Some(t) = self.graph.get_task(id).await? else {
            return Ok(None);
        };
        let plan = match self.graph.get_plan_id_for_task(id).await? {
            Some(pid) => plan_of(self.graph.as_ref(), memo, pid).await?,
            None => None,
        };
        Ok(Some(self.meta(&t, plan.as_ref(), memo).await?))
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
        // No text search on tasks in the store: scan the newest SCAN_CAP of the
        // scope and filter here (see the module comment).
        let (rows, _) = self
            .graph
            .list_all_tasks_filtered(
                None,
                c.project_id,
                c.workspace_slug.as_deref(),
                None,
                None,
                None,
                None,
                None,
                SCAN_CAP,
                0,
                Some("created_at"),
                "desc",
            )
            .await?;
        let mut out = Vec::new();
        for row in rows.iter().filter(|r| Self::matches(r, &c.needle)) {
            if out.len() >= c.fetch {
                break;
            }
            let plan = plan_of(self.graph.as_ref(), memo, row.plan_id).await?;
            out.push(self.meta(&row.task, plan.as_ref(), memo).await?);
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// note and rfc
// ----------------------------------------------------------------------------

pub struct NoteResolver {
    graph: Arc<dyn GraphStore>,
}

pub struct RfcResolver {
    graph: Arc<dyn GraphStore>,
}

async fn note_candidates(
    graph: &dyn GraphStore,
    c: &Candidates,
    types: &[NoteType],
) -> anyhow::Result<Vec<Note>> {
    let filters = NoteFilters {
        note_type: Some(types.to_vec()),
        search: (!c.needle.is_empty()).then(|| c.needle.clone()),
        limit: Some(c.fetch as i64),
        offset: Some(0),
        sort_by: Some("created_at".into()),
        sort_order: Some("desc".into()),
        ..Default::default()
    };
    let (notes, _) = graph
        .list_notes(c.project_id, c.workspace_slug.as_deref(), &filters)
        .await?;
    Ok(notes)
}

async fn note_meta(
    graph: &dyn GraphStore,
    kind: RefKind,
    n: &Note,
    memo: &mut Memo,
) -> anyhow::Result<RefMeta> {
    let (project, workspace) = scope_of(graph, memo, n.project_id).await?;
    let (label, subtitle, entity_status) = if kind == RefKind::Rfc {
        let status = n
            .tags
            .iter()
            .find_map(|t| t.strip_prefix("rfc-status:"))
            .map(str::to_string);
        (rfc_label(n), None, status)
    } else {
        (
            label::derive("note", &n.id, &[&n.content]),
            Some(n.note_type.to_string()),
            snake(&n.status),
        )
    };
    Ok(RefMeta {
        kind,
        id: n.id,
        label,
        subtitle,
        project,
        workspace,
        entity_status,
    })
}

/// An RFC's content is JSON with a `title`; anything else reads as text.
fn rfc_label(n: &Note) -> String {
    let title = serde_json::from_str::<serde_json::Value>(&n.content)
        .ok()
        .and_then(|v| v.get("title").and_then(|t| t.as_str()).map(str::to_string))
        .unwrap_or_default();
    label::derive("rfc", &n.id, &[&title, &n.content])
}

#[async_trait]
impl KindResolver for NoteResolver {
    fn kind(&self) -> RefKind {
        RefKind::Note
    }

    async fn load(&self, id: Uuid, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        match self.graph.get_note(id).await? {
            Some(n) if n.note_type != NoteType::Rfc => Ok(Some(
                note_meta(self.graph.as_ref(), RefKind::Note, &n, memo).await?,
            )),
            _ => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
        let notes = note_candidates(self.graph.as_ref(), c, &NOTE_TYPES).await?;
        let mut out = Vec::with_capacity(notes.len());
        for n in &notes {
            out.push(note_meta(self.graph.as_ref(), RefKind::Note, n, memo).await?);
        }
        Ok(out)
    }
}

#[async_trait]
impl KindResolver for RfcResolver {
    fn kind(&self) -> RefKind {
        RefKind::Rfc
    }

    async fn load(&self, id: Uuid, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        match self.graph.get_note(id).await? {
            Some(n) if n.note_type == NoteType::Rfc => Ok(Some(
                note_meta(self.graph.as_ref(), RefKind::Rfc, &n, memo).await?,
            )),
            _ => Ok(None),
        }
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
        let notes = note_candidates(self.graph.as_ref(), c, &[NoteType::Rfc]).await?;
        let mut out = Vec::with_capacity(notes.len());
        for n in &notes {
            out.push(note_meta(self.graph.as_ref(), RefKind::Rfc, n, memo).await?);
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// decision
// ----------------------------------------------------------------------------

pub struct DecisionResolver {
    graph: Arc<dyn GraphStore>,
}

impl DecisionResolver {
    async fn meta(
        &self,
        d: &DecisionNode,
        project_id: Option<Uuid>,
        memo: &mut Memo,
    ) -> anyhow::Result<RefMeta> {
        let (project, workspace) = scope_of(self.graph.as_ref(), memo, project_id).await?;
        Ok(RefMeta {
            kind: RefKind::Decision,
            id: d.id,
            label: label::derive("decision", &d.id, &[&d.description]),
            subtitle: d.chosen_option.as_deref().and_then(label::first_line),
            project,
            workspace,
            entity_status: snake(&d.status),
        })
    }
}

#[async_trait]
impl KindResolver for DecisionResolver {
    fn kind(&self) -> RefKind {
        RefKind::Decision
    }

    async fn load(&self, id: Uuid, memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        let Some(d) = self.graph.get_decision(id).await? else {
            return Ok(None);
        };
        let project_id = self
            .graph
            .get_decision_project_id(id)
            .await?
            .and_then(|s| Uuid::parse_str(&s).ok());
        Ok(Some(self.meta(&d, project_id, memo).await?))
    }

    async fn candidates(&self, c: &Candidates, memo: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
        // Same bounded in-memory filter as tasks: the store lists decisions by
        // status, it does not search their text.
        let mut rows = Vec::new();
        for status in DECISION_STATUSES {
            let (items, _) = self
                .graph
                .list_decisions_by_status(
                    status,
                    c.project_id,
                    c.workspace_slug.as_deref(),
                    SCAN_CAP,
                    0,
                )
                .await?;
            rows.extend(items);
        }
        rows.retain(|i| {
            contains(&i.decision.description, &c.needle)
                || contains(&i.decision.rationale, &c.needle)
        });
        rows.sort_by(|a, b| b.decision.decided_at.cmp(&a.decision.decided_at));
        rows.truncate(c.fetch);
        let mut out = Vec::with_capacity(rows.len());
        for i in &rows {
            out.push(self.meta(&i.decision, i.project_id, memo).await?);
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// the source
// ----------------------------------------------------------------------------

/// The five resolvers over one store. Implements the raw read the access
/// policy wraps.
pub struct GraphRefSource {
    plan: PlanResolver,
    task: TaskResolver,
    note: NoteResolver,
    decision: DecisionResolver,
    rfc: RfcResolver,
}

impl GraphRefSource {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            plan: PlanResolver {
                graph: graph.clone(),
            },
            task: TaskResolver {
                graph: graph.clone(),
            },
            note: NoteResolver {
                graph: graph.clone(),
            },
            decision: DecisionResolver {
                graph: graph.clone(),
            },
            rfc: RfcResolver { graph },
        }
    }

    /// The resolver of `kind`. Exhaustive: a new kind does not compile until it
    /// has one.
    pub fn resolver(&self, kind: RefKind) -> &dyn KindResolver {
        match kind {
            RefKind::Plan => &self.plan,
            RefKind::Task => &self.task,
            RefKind::Note => &self.note,
            RefKind::Decision => &self.decision,
            RefKind::Rfc => &self.rfc,
        }
    }
}

#[async_trait]
impl RefSource for GraphRefSource {
    async fn load_unchecked(&self, r: &EntityRef) -> anyhow::Result<Option<RefMeta>> {
        let resolver = self.resolver(r.kind);
        let Some(mut meta) = resolver.load(r.id, &mut Memo::default()).await? else {
            return Ok(None);
        };
        resolver.finish(&mut meta).await?;
        Ok(Some(meta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refs::access::{AccessPolicy, Disclosure, Resolution};
    use crate::refs::test_support::{note_with, task_titled, user, world, Spy, World};
    use crate::refs::wire::RefStatus;

    fn source(w: &World) -> GraphRefSource {
        GraphRefSource::new(w.graph.clone())
    }

    async fn found(w: &World, kind: RefKind, id: Uuid) -> Option<RefMeta> {
        let r = EntityRef::new(kind, id);
        match AccessPolicy::open_instance()
            .resolve_checked(&user(), &source(w), &r)
            .await
        {
            Resolution::Found(m) => Some(*m),
            _ => None,
        }
    }

    #[tokio::test]
    async fn a_plan_resolves_through_get_plan_with_its_scope_and_task_count() {
        let w = world().await;
        let m = found(&w, RefKind::Plan, w.plan_a.id).await.unwrap();
        assert_eq!(m.label, "Plan alpha refs");
        assert_eq!(m.entity_status.as_deref(), Some("draft"));
        assert_eq!(m.subtitle.as_deref(), Some("2 tâches"));
        assert_eq!(m.project.as_ref().unwrap().slug, "alpha");
        assert_eq!(m.workspace.as_ref().unwrap().slug, "po");
        assert_eq!(m.workspace.as_ref().unwrap().id, w.ws.id);
        let b = found(&w, RefKind::Plan, w.plan_b.id).await.unwrap();
        assert_eq!(b.subtitle.as_deref(), Some("1 tâche"));
        assert!(b.workspace.is_none(), "beta is in no workspace");
        assert_eq!(b.project.unwrap().slug, "beta");
    }

    #[tokio::test]
    async fn a_plan_without_tasks_says_zero() {
        let w = world().await;
        let empty = crate::neo4j::models::PlanNode::new("Vide".into(), "d".into(), "t".into(), 1);
        w.graph.create_plan(&empty).await.unwrap();
        let m = found(&w, RefKind::Plan, empty.id).await.unwrap();
        assert_eq!(m.subtitle.as_deref(), Some("0 tâches"));
        assert!(m.project.is_none() && m.workspace.is_none());
    }

    #[tokio::test]
    async fn a_task_resolves_through_get_task_and_names_its_plan() {
        let w = world().await;
        let m = found(&w, RefKind::Task, w.task_a.id).await.unwrap();
        assert_eq!(m.label, "Tâche alpha refs");
        assert_eq!(m.subtitle.as_deref(), Some("Plan : Plan alpha refs"));
        assert_eq!(m.entity_status.as_deref(), Some("pending"));
        assert_eq!(m.project.unwrap().slug, "alpha");
        assert_eq!(m.workspace.unwrap().slug, "po");
    }

    #[tokio::test]
    async fn an_untitled_task_is_named_by_the_first_line_of_its_description() {
        let w = world().await;
        let m = found(&w, RefKind::Task, w.task_untitled.id).await.unwrap();
        assert_eq!(m.label, "Sans titre explicite");
    }

    #[tokio::test]
    async fn a_task_without_a_plan_has_no_scope_and_no_subtitle() {
        let w = world().await;
        let orphan = task_titled(Some("Orpheline"), "d");
        w.graph
            .tasks
            .write()
            .await
            .insert(orphan.id, orphan.clone());
        let m = found(&w, RefKind::Task, orphan.id).await.unwrap();
        assert!(m.subtitle.is_none() && m.project.is_none() && m.workspace.is_none());
    }

    #[tokio::test]
    async fn a_note_has_no_title_its_label_is_its_first_line_or_markdown_title() {
        let w = world().await;
        let m = found(&w, RefKind::Note, w.note_a.id).await.unwrap();
        assert_eq!(m.label, "Piège des refs");
        assert_eq!(m.subtitle.as_deref(), Some("gotcha"));
        assert_eq!(m.entity_status.as_deref(), Some("active"));
        assert_eq!(m.project.unwrap().slug, "alpha");
        let g = found(&w, RefKind::Note, w.note_global.id).await.unwrap();
        assert_eq!(g.label, "astuce globale");
        assert!(g.project.is_none());
    }

    #[tokio::test]
    async fn a_long_first_line_is_cut_at_eighty_characters() {
        let w = world().await;
        let long = note_with(None, NoteType::Tip, &"x".repeat(300), vec![]);
        w.graph.create_note(&long).await.unwrap();
        let m = found(&w, RefKind::Note, long.id).await.unwrap();
        assert_eq!(m.label.chars().count(), 80);
    }

    #[tokio::test]
    async fn a_decision_is_named_by_its_description_only() {
        let w = world().await;
        let m = found(&w, RefKind::Decision, w.decision_a.id).await.unwrap();
        assert_eq!(m.label, "Utiliser des refs typées");
        assert_eq!(m.subtitle.as_deref(), Some("Refs typées"));
        assert_eq!(m.entity_status.as_deref(), Some("accepted"));
        let b = found(&w, RefKind::Decision, w.decision_b.id).await.unwrap();
        assert!(b.subtitle.is_none());
        assert_eq!(b.entity_status.as_deref(), Some("proposed"));
    }

    #[tokio::test]
    async fn an_rfc_is_a_note_of_type_rfc_titled_from_its_json_and_statused_by_tag() {
        let w = world().await;
        let m = found(&w, RefKind::Rfc, w.rfc.id).await.unwrap();
        assert_eq!(m.kind, RefKind::Rfc);
        assert_eq!(m.label, "RFC refs");
        assert_eq!(m.entity_status.as_deref(), Some("proposed"));
        assert!(m.subtitle.is_none());
    }

    #[tokio::test]
    async fn an_rfc_whose_content_is_not_json_reads_as_text_and_has_no_status_without_a_tag() {
        let w = world().await;
        let rfc = note_with(None, NoteType::Rfc, "## Titre libre\nbla", vec![]);
        w.graph.create_note(&rfc).await.unwrap();
        let m = found(&w, RefKind::Rfc, rfc.id).await.unwrap();
        assert_eq!(m.label, "Titre libre");
        assert!(m.entity_status.is_none());
        let json_no_title = note_with(None, NoteType::Rfc, r#"{"sections":[]}"#, vec![]);
        w.graph.create_note(&json_no_title).await.unwrap();
        let m = found(&w, RefKind::Rfc, json_no_title.id).await.unwrap();
        assert_eq!(m.label, r#"{"sections":[]}"#);
    }

    #[tokio::test]
    async fn note_and_rfc_are_disjoint() {
        let w = world().await;
        assert!(found(&w, RefKind::Note, w.rfc.id).await.is_none());
        assert!(found(&w, RefKind::Rfc, w.note_a.id).await.is_none());
    }

    #[tokio::test]
    async fn an_unknown_id_is_not_found_for_every_kind() {
        let w = world().await;
        let policy = AccessPolicy::open_instance();
        for kind in RefKind::ALL {
            let r = EntityRef::new(kind, Uuid::new_v4());
            assert_eq!(
                policy.resolve_checked(&user(), &source(&w), &r).await,
                Resolution::NotFound,
                "{kind}"
            );
        }
    }

    #[tokio::test]
    async fn the_verdict_on_each_kind_comes_from_the_policy() {
        let w = world().await;
        let cases = [
            (RefKind::Plan, w.plan_a.id),
            (RefKind::Task, w.task_a.id),
            (RefKind::Note, w.note_a.id),
            (RefKind::Decision, w.decision_a.id),
            (RefKind::Rfc, w.rfc.id),
        ];
        for (kind, id) in cases {
            let r = EntityRef::new(kind, id);
            let deny = Spy::new(false);
            let policy = AccessPolicy::new(deny.clone(), Disclosure::Uniform);
            let out = policy.resolve_checked(&user(), &source(&w), &r).await;
            assert_eq!(out, Resolution::Forbidden, "{kind}");
            assert_eq!(deny.asked(), 1, "{kind}: the rule must have spoken");
            assert_eq!(out.status(Disclosure::Uniform), RefStatus::NotFound);

            let allow = Spy::new(true);
            let policy = AccessPolicy::new(allow.clone(), Disclosure::Uniform);
            let out = policy.resolve_checked(&user(), &source(&w), &r).await;
            assert!(matches!(out, Resolution::Found(_)), "{kind}");
            assert_eq!(allow.asked(), 1, "{kind}");
        }
    }

    #[tokio::test]
    async fn every_kind_has_the_resolver_of_its_name() {
        let w = world().await;
        let s = source(&w);
        for kind in RefKind::ALL {
            assert_eq!(s.resolver(kind).kind(), kind);
        }
    }

    #[tokio::test]
    async fn candidates_are_the_store_rows_filtered_by_the_needle() {
        let w = world().await;
        let s = source(&w);
        let all = |needle: &str| Candidates {
            needle: needle.into(),
            project_id: None,
            workspace_slug: None,
            fetch: 50,
        };
        let mut memo = Memo::default();
        let t = s.resolver(RefKind::Task);
        assert_eq!(t.candidates(&all(""), &mut memo).await.unwrap().len(), 3);
        let hit = t.candidates(&all("facturer"), &mut memo).await.unwrap();
        assert_eq!(hit.len(), 1, "description matches, case-insensitively");
        assert_eq!(hit[0].id, w.task_b.id);
        let by_title = t.candidates(&all("alpha"), &mut memo).await.unwrap();
        assert_eq!(by_title.len(), 1);
        let cap = Candidates {
            fetch: 2,
            ..all("")
        };
        assert_eq!(t.candidates(&cap, &mut memo).await.unwrap().len(), 2);

        let d = s.resolver(RefKind::Decision);
        assert_eq!(d.candidates(&all(""), &mut memo).await.unwrap().len(), 2);
        let by_rationale = d.candidates(&all("simplicité"), &mut memo).await.unwrap();
        assert_eq!(by_rationale.len(), 1);
        assert_eq!(by_rationale[0].id, w.decision_b.id);
        let one = Candidates {
            fetch: 1,
            ..all("")
        };
        assert_eq!(d.candidates(&one, &mut memo).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_store_failure_on_a_listing_is_an_error_not_an_empty_list() {
        let w = world().await;
        let s = source(&w);
        let c = Candidates {
            needle: String::new(),
            project_id: None,
            workspace_slug: None,
            fetch: 10,
        };
        for (kind, read) in [
            (RefKind::Plan, "list_plans_filtered"),
            (RefKind::Task, "list_all_tasks_filtered"),
            (RefKind::Note, "list_notes"),
            (RefKind::Rfc, "list_notes"),
            (RefKind::Decision, "list_decisions_by_status"),
        ] {
            w.graph.fail_reads.lock().unwrap().insert(read);
            let out = s.resolver(kind).candidates(&c, &mut Memo::default()).await;
            assert!(out.is_err(), "{kind}");
            w.graph.fail_reads.lock().unwrap().remove(read);
        }
    }
}
