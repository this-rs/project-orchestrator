//! `AnalyticsStore` on obrain, sub-lot "views": the pure reads that feed the graph
//! visualisation, written on the indexes of the ported domains. The trait `impl` is in
//! `analytics.rs`. Reference: `src/neo4j/decision.rs`, `src/neo4j/constraint.rs`,
//! `src/neo4j/chat.rs`.
//!
//! Four reads serve; each is exact for what a base written by this backend holds, and
//! REFUSES, by name, the one shape of base (a migrated one) where it could not be exact.
//! None answers an empty value in place of a refusal.
//!
//! - **`get_project_decisions_for_graph`**: the decisions attached to the project
//!   (`HAS_DECISION`), each with its `AFFECTS` relations. `AFFECTS` is refused on this
//!   backend (`DecisionStore::add_decision_affects`, its targets are code entities), so
//!   no decision this backend wrote has one: the relations are an exact empty list. A
//!   decision of THIS project that carries an `AFFECTS` edge (migrated), or ANY decision
//!   carrying an `INFORMED_BY` edge (the task chain of the reference's path, which this
//!   backend does not resolve to a project), refuses the call. Order: `(decided_at, id)`
//!   (Neo4j states none). As the reference, `embedding`, `embedding_model` and
//!   `scar_intensity` are not returned; `chosen_option` is `None` for none (the reference
//!   returns `Some("")`, an artefact of its column read, and `get_decision` returns `None`).
//! - **`get_project_constraints`**: the constraints of the plans that reach the project by
//!   the edge `(:Plan)-[:BELONGS_TO_PROJECT]->(:Project)`, which is what the reference
//!   matches. NO writer of this repository draws that edge (a plan reaches its project by
//!   `HAS_PLAN`, `plan.rs`), so on a base written here the answer is empty, as it is on
//!   Neo4j today; the port reproduces the reference, it does not correct it. Whether the
//!   reference meant `HAS_PLAN` is a question for its owner. A plan that carries that edge
//!   and that this backend quarantined refuses. Order: `(constraint id, plan id)`.
//! - **`get_chat_graph_data`**: the sessions that have a `DISCUSSED` relation. `DISCUSSED`
//!   is refused on this backend (`ChatStore::add_discussed`: its targets are code, plan and
//!   task entities) and so is the `HAS_CHAT_SESSION` edge that scopes it, so no session
//!   written here has one: the answer is an exact empty pair. A `ChatSession` carrying a
//!   `DISCUSSED` edge (migrated) refuses the call.
//!
//! - **`get_pm_graph_data`**: plans, tasks, steps, milestones and releases of the project,
//!   and the edges `HAS_TASK`, `HAS_STEP`, `DEPENDS_ON`, `TARGETS_MILESTONE`. Membership is
//!   the one the ported domains use (a plan: its `project_id` or a `HAS_PLAN` edge; a task
//!   and a step: the index of `task.rs`; a milestone and a release: their `project_id`),
//!   the properties are read RAW from the nodes as the Cypher reads them (a status is the
//!   stored string). The commit branches match `(:Commit)-[:LINKED_TO_PLAN|LINKED_TO_TASK]`
//!   edges that no writer of this repository draws (a plan's commits are `RESULTED_IN`
//!   edges, `commit.rs`; `LINKED_TO_PLAN` is drawn from a `ProtocolRun`): they are empty
//!   on a base written here, as on Neo4j, and the call REFUSES if such an edge from a
//!   `Commit` exists (migrated); so do the `TOUCHES` edges they would carry to a `File`
//!   (no `File` is written, `code.rs`). A plan with a part the task index could not read
//!   refuses. Each branch has its own `LIMIT` as the reference, and no order but the plans'
//!   (`priority DESC, id DESC`): the others are served in id order, the edges sorted by
//!   `(type, source, target)`. A date (`target_date`) is decoded by the single decoder and
//!   written RFC 3339 (the reference reads a string that a `datetime` does not give).
//!
//! Not here, refused by name in `analytics.rs`: the two neighbourhoods (they read any
//! entity, `File` and `Function` among them), and `schema_catalog`, `causal_walk`,
//! `typed_path`.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, Result};
use obrain_common::types::NodeId;
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use obrain_common::types::Value;

use super::{NonPorte, ObrainGraphStore};
use crate::neo4j::models::{
    AffectsRelation, ChatGraphDiscussed, ChatGraphSession, ConstraintNode, DecisionNode,
    PmGraphEdge, PmGraphNode,
};

fn refus_contextuel<T>(methode: &'static str, raison: String) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

/// Is a `Project` node of this id on disk? Called for a project the index does not hold:
/// then the node was quarantined at open.
pub(super) fn projet_en_quarantaine(db: &ObrainDB, id: Uuid) -> bool {
    let voulu = id.to_string();
    db.store().nodes_by_label("Project").into_iter().any(|n| {
        db.get_node(n).is_some_and(|nd| {
            nd.get_property("id")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s == voulu)
        })
    })
}

/// The edges of one type leaving (`Outgoing`) or entering (`Incoming`) a node, with the
/// node at the other end.
fn aretes_de_type(
    db: &ObrainDB,
    noeud: NodeId,
    sens: Direction,
    type_: &str,
) -> Vec<(NodeId, obrain_common::types::EdgeId)> {
    db.store()
        .edges_from(noeud, sens)
        .into_iter()
        .filter(|(_, a)| db.get_edge(*a).is_some_and(|e| &*e.edge_type == type_))
        .collect()
}

impl ObrainGraphStore {
    pub(super) fn vue_decisions(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(DecisionNode, Vec<AffectsRelation>)>> {
        const M: &str = "AnalyticsStore::get_project_decisions_for_graph";
        self.lire(move |db, etat| {
            for noeud in etat.decisions.noeuds() {
                if !aretes_de_type(db, noeud, Direction::Incoming, "INFORMED_BY").is_empty() {
                    return refus_contextuel(
                        M,
                        "une décision porte une arête INFORMED_BY (chaîne par les tâches du \
                         chemin de la référence) : son projet n'est pas résolu par ce backend"
                            .to_string(),
                    );
                }
            }
            let mut sortie = Vec::new();
            for (mut d, noeud) in etat.decisions.du_projet(project_id) {
                if !aretes_de_type(db, noeud, Direction::Outgoing, "AFFECTS").is_empty() {
                    return refus_contextuel(
                        M,
                        format!(
                            "la décision {} porte une arête AFFECTS : ses relations visent \
                             des entités de code, domaine non porté",
                            d.id
                        ),
                    );
                }
                // The columns the reference does not return.
                d.embedding = None;
                d.embedding_model = None;
                d.scar_intensity = 0.0;
                sortie.push((d, Vec::new()));
            }
            Ok(sortie)
        })
    }

    pub(super) fn vue_contraintes(&self, project_id: Uuid) -> Result<Vec<(ConstraintNode, Uuid)>> {
        self.lire(move |db, etat| {
            let Some(projet) = etat.projets.noeud_de(project_id) else {
                if projet_en_quarantaine(db, project_id) {
                    bail!(
                        "projet {project_id} : un nœud de cet id est sur disque, illisible à \
                         l'ouverture (quarantaine) ; ses plans ne peuvent pas être lus"
                    );
                }
                return Ok(Vec::new());
            };
            let mut sortie: Vec<(ConstraintNode, Uuid)> = Vec::new();
            for (source, arete) in
                aretes_de_type(db, projet, Direction::Incoming, "BELONGS_TO_PROJECT")
            {
                if !db
                    .get_node_labels(source)
                    .unwrap_or_default()
                    .iter()
                    .any(|l| l == "Plan")
                {
                    continue;
                }
                let plan = db
                    .get_node(source)
                    .and_then(|n| {
                        n.get_property("id")
                            .and_then(|v| v.as_str().and_then(|s| s.parse::<Uuid>().ok()))
                    })
                    .filter(|id| etat.plans.par_id.get(id).is_some_and(|e| e.noeud == source))
                    .ok_or_else(|| {
                        anyhow!(
                            "arête BELONGS_TO_PROJECT {arete:?} : son plan {source:?} n'est pas \
                             adressé par ce backend (quarantaine) ; ses contraintes ne peuvent \
                             pas être lues"
                        )
                    })?;
                for c in etat.contraintes.du_plan(plan) {
                    sortie.push((c, plan));
                }
            }
            sortie.sort_by(|a, b| a.0.id.cmp(&b.0.id).then_with(|| a.1.cmp(&b.1)));
            Ok(sortie)
        })
    }

    pub(super) fn vue_discussions(
        &self,
        _project_id: Uuid,
    ) -> Result<(Vec<ChatGraphSession>, Vec<ChatGraphDiscussed>)> {
        const M: &str = "AnalyticsStore::get_chat_graph_data";
        self.lire(move |db, _| {
            let mut porteurs: BTreeSet<String> = BTreeSet::new();
            for session in db.store().nodes_by_label("ChatSession") {
                if !aretes_de_type(db, session, Direction::Outgoing, "DISCUSSED").is_empty() {
                    porteurs.insert(format!("{session:?}"));
                }
            }
            if !porteurs.is_empty() {
                return refus_contextuel(
                    M,
                    format!(
                        "{} session(s) portent une arête DISCUSSED (migrée) : le domaine \
                         DISCUSSED n'est pas porté, la lecture ne serait pas exacte",
                        porteurs.len()
                    ),
                );
            }
            Ok((Vec::new(), Vec::new()))
        })
    }
}

/// A string property read raw: absent or null is `None`; present with another type is an
/// error naming the node (never read as absent).
fn texte_brut(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<String>> {
    match db
        .get_node(noeud)
        .and_then(|n| n.get_property(cle).cloned())
    {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| anyhow!("nœud {noeud:?} : `{cle}` n'est pas une chaîne")),
    }
}

fn entier_brut(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<i64>> {
    match db
        .get_node(noeud)
        .and_then(|n| n.get_property(cle).cloned())
    {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_int64()
            .map(Some)
            .ok_or_else(|| anyhow!("nœud {noeud:?} : `{cle}` n'est pas un entier")),
    }
}

/// A date property, by the single decoder, written RFC 3339.
fn date_brute(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<String>> {
    match db
        .get_node(noeud)
        .and_then(|n| n.get_property(cle).cloned())
    {
        None => Ok(None),
        Some(v) => Ok(super::date_valeur(&v, cle)
            .map_err(|e| anyhow!("nœud {noeud:?} : {e}"))?
            .map(|d| d.to_rfc3339())),
    }
}

/// The label of a node, cut as the reference cuts it: past 80 characters, 79 and `…`.
fn etiquette_coupee(label: String) -> String {
    if label.chars().count() > 80 {
        let coupe: String = label.chars().take(79).collect();
        format!("{coupe}…")
    } else {
        label
    }
}

/// What a PM node carries beside its id, type and label (the reference's `attributes`).
#[derive(Default)]
struct ChampsPm {
    status: Option<String>,
    priority: Option<i64>,
    target_date: Option<String>,
    version: Option<String>,
    order: Option<i64>,
}

fn noeud_pm(id: Uuid, type_: &str, label: Option<String>, c: ChampsPm) -> PmGraphNode {
    PmGraphNode {
        id: id.to_string(),
        node_type: type_.to_string(),
        label: etiquette_coupee(label.unwrap_or_default()),
        attributes: serde_json::json!({
            "status": c.status,
            "priority": c.priority,
            "target_date": c.target_date,
            "version": c.version,
            "order": c.order,
        }),
    }
}

impl ObrainGraphStore {
    pub(super) fn vue_pm(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<PmGraphNode>, Vec<PmGraphEdge>)> {
        const M: &str = "AnalyticsStore::get_pm_graph_data";
        self.lire(move |db, etat| {
            if !etat.projets.contient(project_id) {
                if projet_en_quarantaine(db, project_id) {
                    bail!(
                        "projet {project_id} : un nœud de cet id est sur disque, illisible à \
                         l'ouverture (quarantaine) ; ses plans ne peuvent pas être lus"
                    );
                }
                return Ok((Vec::new(), Vec::new()));
            }
            let mut plans: Vec<(&super::plan::Entree, i64)> = Vec::new();
            for e in etat
                .plans
                .par_id
                .values()
                .filter(|e| e.appartient_a(project_id))
            {
                let priorite = entier_brut(db, e.noeud, "priority")?
                    .ok_or_else(|| anyhow!("plan {} sans propriété `priority`", e.plan.id))?;
                plans.push((e, priorite));
            }
            for (e, _) in &plans {
                if let Some(raisons) = etat.taches.incomplets.get(&e.plan.id) {
                    bail!(
                        "plan {} : le backend n'a pas pu lire une partie de ses tâches ({}) ; \
                         le graphe serait faux",
                        e.plan.id,
                        raisons.join("; ")
                    );
                }
            }
            let ids_plans: BTreeSet<Uuid> = plans.iter().map(|(e, _)| e.plan.id).collect();
            let mut taches: Vec<&super::task::EntreeTache> = etat
                .taches
                .taches
                .values()
                .filter(|t| ids_plans.contains(&t.plan))
                .collect();
            taches.sort_by_key(|t| t.tache.id);
            let ids_taches: BTreeSet<Uuid> = taches.iter().map(|t| t.tache.id).collect();
            let mut etapes: Vec<&super::task::EntreeEtape> = etat
                .taches
                .etapes
                .values()
                .filter(|s| ids_taches.contains(&s.tache))
                .collect();
            etapes.sort_by_key(|s| s.etape.id);

            // The commit branches: nothing draws these edges here (module doc).
            let from_commit = |cible: NodeId, type_: &str| -> bool {
                aretes_de_type(db, cible, Direction::Incoming, type_)
                    .into_iter()
                    .any(|(source, _)| {
                        db.get_node_labels(source)
                            .unwrap_or_default()
                            .iter()
                            .any(|l| l == "Commit")
                    })
            };
            if plans
                .iter()
                .any(|(e, _)| from_commit(e.noeud, "LINKED_TO_PLAN"))
                || taches
                    .iter()
                    .any(|t| from_commit(t.noeud, "LINKED_TO_TASK"))
            {
                return refus_contextuel(
                    M,
                    "un Commit porte une arête LINKED_TO_PLAN ou LINKED_TO_TASK vers ce \
                     projet (migrée) : les commits et leurs TOUCHES vers des File ne sont pas \
                     lus par ce backend"
                        .to_string(),
                );
            }

            let lim = limit;
            let mut noeuds: Vec<PmGraphNode> = Vec::new();
            // 1. plans: ORDER BY priority DESC, id DESC.
            plans.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.plan.id.cmp(&a.0.plan.id)));
            for (e, priorite) in plans.iter().take(lim) {
                noeuds.push(noeud_pm(
                    e.plan.id,
                    "plan",
                    texte_brut(db, e.noeud, "title")?,
                    ChampsPm {
                        status: texte_brut(db, e.noeud, "status")?,
                        priority: Some(*priorite),
                        ..Default::default()
                    },
                ));
            }
            // 2. tasks.
            for t in taches.iter().take(lim) {
                noeuds.push(noeud_pm(
                    t.tache.id,
                    "task",
                    texte_brut(db, t.noeud, "title")?,
                    ChampsPm {
                        status: texte_brut(db, t.noeud, "status")?,
                        priority: entier_brut(db, t.noeud, "priority")?,
                        ..Default::default()
                    },
                ));
            }
            // 3. steps.
            for s in etapes.iter().take(lim) {
                noeuds.push(noeud_pm(
                    s.etape.id,
                    "step",
                    texte_brut(db, s.noeud, "description")?,
                    ChampsPm {
                        status: texte_brut(db, s.noeud, "status")?,
                        order: entier_brut(db, s.noeud, "order")?,
                        ..Default::default()
                    },
                ));
            }
            // 4. milestones, 5. releases.
            for (id, n) in etat
                .releases
                .jalons_du_projet(project_id)
                .into_iter()
                .take(lim)
            {
                noeuds.push(noeud_pm(
                    id,
                    "milestone",
                    texte_brut(db, n, "title")?,
                    ChampsPm {
                        status: texte_brut(db, n, "status")?,
                        target_date: date_brute(db, n, "target_date")?,
                        ..Default::default()
                    },
                ));
            }
            for (id, n) in etat
                .releases
                .versions_du_projet(project_id)
                .into_iter()
                .take(lim)
            {
                noeuds.push(noeud_pm(
                    id,
                    "release",
                    texte_brut(db, n, "title")?,
                    ChampsPm {
                        status: texte_brut(db, n, "status")?,
                        target_date: date_brute(db, n, "target_date")?,
                        version: texte_brut(db, n, "version")?,
                        ..Default::default()
                    },
                ));
            }

            // The edges, one group per branch of the reference, each sorted.
            let mut aretes: Vec<PmGraphEdge> = Vec::new();
            let mut groupe = |type_: &str, mut paires: Vec<(String, String)>| {
                paires.sort();
                for (source, target) in paires {
                    aretes.push(PmGraphEdge {
                        source,
                        target,
                        rel_type: type_.to_string(),
                        attributes: None,
                    });
                }
            };
            groupe(
                "HAS_TASK",
                taches
                    .iter()
                    .map(|t| (t.plan.to_string(), t.tache.id.to_string()))
                    .collect(),
            );
            groupe(
                "HAS_STEP",
                etapes
                    .iter()
                    .map(|s| (s.tache.to_string(), s.etape.id.to_string()))
                    .collect(),
            );
            let mut dependances: Vec<(String, String)> = Vec::new();
            for t in &taches {
                for (cible, _) in aretes_de_type(db, t.noeud, Direction::Outgoing, "DEPENDS_ON") {
                    if !db
                        .get_node_labels(cible)
                        .unwrap_or_default()
                        .iter()
                        .any(|l| l == "Task")
                    {
                        continue;
                    }
                    let id = texte_brut(db, cible, "id")?.ok_or_else(|| {
                        anyhow!("tâche {} : DEPENDS_ON vers un nœud sans `id`", t.tache.id)
                    })?;
                    dependances.push((t.tache.id.to_string(), id));
                }
            }
            groupe("DEPENDS_ON", dependances);
            let mut jalons: Vec<(String, String)> = Vec::new();
            for (e, _) in &plans {
                for (cible, _) in
                    aretes_de_type(db, e.noeud, Direction::Outgoing, "TARGETS_MILESTONE")
                {
                    if !db
                        .get_node_labels(cible)
                        .unwrap_or_default()
                        .iter()
                        .any(|l| l == "Milestone")
                    {
                        continue;
                    }
                    let id = texte_brut(db, cible, "id")?.ok_or_else(|| {
                        anyhow!(
                            "plan {} : TARGETS_MILESTONE vers un nœud sans `id`",
                            e.plan.id
                        )
                    })?;
                    jalons.push((e.plan.id.to_string(), id));
                }
            }
            groupe("TARGETS_MILESTONE", jalons);
            Ok((noeuds, aretes))
        })
    }
}
