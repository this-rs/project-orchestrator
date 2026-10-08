//! Shared fixtures for the tests of the resolvers, the search and the handler.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use uuid::Uuid;

use super::access::{Principal, RefMeta, ScopeRule};
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::{DecisionNode, DecisionStatus, PlanNode, TaskNode, WorkspaceNode};
use crate::neo4j::GraphStore;
use crate::notes::{Note, NoteImportance, NoteScope, NoteType};
use crate::test_helpers::{test_decision, test_note, test_project, test_task, test_workspace};

pub fn user() -> Principal {
    Principal::User(Uuid::new_v4())
}

/// A rule that records that it was asked and answers a fixed verdict.
pub struct Spy {
    pub verdict: bool,
    asked: AtomicUsize,
}

impl Spy {
    pub fn new(verdict: bool) -> Arc<Self> {
        Arc::new(Self {
            verdict,
            asked: AtomicUsize::new(0),
        })
    }
    pub fn asked(&self) -> usize {
        self.asked.load(Ordering::SeqCst)
    }
}

impl ScopeRule for Spy {
    fn allows(&self, _p: &Principal, _m: &RefMeta) -> bool {
        self.asked.fetch_add(1, Ordering::SeqCst);
        self.verdict
    }
}

/// Two projects: `a` (in workspace `po`) and `b` (in no workspace), each with
/// a plan, a task, a note and a decision, plus a global note and an RFC.
pub struct World {
    pub graph: Arc<MockGraphStore>,
    pub ws: WorkspaceNode,
    pub a: Uuid,
    pub b: Uuid,
    pub plan_a: PlanNode,
    pub plan_b: PlanNode,
    pub task_a: TaskNode,
    pub task_untitled: TaskNode,
    pub task_b: TaskNode,
    pub note_a: Note,
    pub note_global: Note,
    pub rfc: Note,
    pub decision_a: DecisionNode,
    pub decision_b: DecisionNode,
}

pub fn note_with(project: Option<Uuid>, t: NoteType, content: &str, tags: Vec<String>) -> Note {
    match project {
        Some(p) => {
            let mut n = test_note(p, t, content);
            n.tags = tags;
            n
        }
        None => Note::new_full(
            None,
            t,
            NoteImportance::Medium,
            NoteScope::Workspace,
            content.to_string(),
            tags,
            "test-agent".into(),
        ),
    }
}

pub fn task_titled(title: Option<&str>, description: &str) -> TaskNode {
    let mut t = test_task();
    t.title = title.map(str::to_string);
    t.description = description.to_string();
    t
}

pub async fn world() -> World {
    let graph = Arc::new(MockGraphStore::new());
    let g: &dyn GraphStore = graph.as_ref();

    let mut pa = test_project();
    pa.slug = "alpha".into();
    pa.name = "Alpha".into();
    let mut pb = test_project();
    pb.slug = "beta".into();
    pb.name = "Beta".into();
    g.create_project(&pa).await.unwrap();
    g.create_project(&pb).await.unwrap();
    let mut ws = test_workspace();
    ws.slug = "po".into();
    ws.name = "PO".into();
    g.create_workspace(&ws).await.unwrap();
    g.add_project_to_workspace(ws.id, pa.id).await.unwrap();

    let plan_a = PlanNode::new_for_project(
        "Plan alpha refs".into(),
        "Le plan de refs".into(),
        "t".into(),
        5,
        pa.id,
    );
    let plan_b = PlanNode::new_for_project(
        "Plan beta billing".into(),
        "Facturation".into(),
        "t".into(),
        5,
        pb.id,
    );
    g.create_plan(&plan_a).await.unwrap();
    g.create_plan(&plan_b).await.unwrap();

    let task_a = task_titled(Some("Tâche alpha refs"), "Faire les refs\nsuite");
    let task_untitled = task_titled(None, "\n  Sans titre explicite\nsuite du texte");
    let task_b = task_titled(Some("Tâche beta"), "Facturer");
    g.create_task(plan_a.id, &task_a).await.unwrap();
    g.create_task(plan_a.id, &task_untitled).await.unwrap();
    g.create_task(plan_b.id, &task_b).await.unwrap();

    let note_a = note_with(
        Some(pa.id),
        NoteType::Gotcha,
        "# Piège des refs\ncorps de la note",
        vec![],
    );
    let note_global = note_with(None, NoteType::Tip, "astuce globale", vec![]);
    let rfc = note_with(
        None,
        NoteType::Rfc,
        r#"{"title":"RFC refs","sections":[{"title":"s","content":"c"}]}"#,
        vec!["rfc-status:proposed".into()],
    );
    for n in [&note_a, &note_global, &rfc] {
        g.create_note(n).await.unwrap();
    }

    let mut decision_a = test_decision("Utiliser des refs typées\nrationale", "Parce que");
    decision_a.chosen_option = Some("Refs typées".into());
    let mut decision_b = test_decision("Facturation mensuelle", "Simplicité");
    decision_b.status = DecisionStatus::Proposed;
    g.create_decision(task_a.id, &decision_a).await.unwrap();
    g.create_decision(task_b.id, &decision_b).await.unwrap();

    World {
        graph,
        ws,
        a: pa.id,
        b: pb.id,
        plan_a,
        plan_b,
        task_a,
        task_untitled,
        task_b,
        note_a,
        note_global,
        rfc,
        decision_a,
        decision_b,
    }
}
