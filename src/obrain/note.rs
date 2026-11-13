//! `NoteStore` on obrain — the note NODE, its edges, and the answers this backend
//! can give about them. A PARTIAL port: each method is either ported here or
//! refused by name (`NonPorte`) with the reason in its context. Nothing is
//! answered with an empty or default value in place of a refusal.
//!
//! A note is one node labelled `Note`, whose properties carry the names and
//! the lossy conventions of the Neo4j adapter (`src/neo4j/note.rs`, the
//! reference of the conformance suite):
//!
//! - `create_note` stores what Neo4j's `CREATE` stores, with the same defaults
//!   (`""` for an absent timestamp of the `human_confirmed_at`/`valid_*` kind,
//!   `last_activated` set to the creation instant, `project_id` absent when
//!   `None`). The other fields (`supersedes`, `superseded_by`, the reactivation
//!   counters…) are only written by their own methods, as in Neo4j.
//! - the edges are the Neo4j ones: `(:Project)-[:HAS_NOTE]->(:Note)`,
//!   `(:Note)-[:LINKED_TO]->(entity)` with `signature_hash`, `body_hash`,
//!   `last_verified` on the edge, `(:Note)-[:SUPERSEDES]->(:Note)`.
//!
//! Differences, all on the strict side or named:
//!
//! - a creation whose id is already taken is refused (Neo4j would create a
//!   second node with the same id);
//! - a note whose `created_by` names an agent (`agent:<uuid>`) is refused: the
//!   Neo4j write also stamps the `Agent` node and its `AUTHORED` edge, and the
//!   agent domain is not ported;
//! - a link to an entity whose domain is not ported (anything but `Project`
//!   and `Note`) is refused: the entity may exist in the Neo4j world, and an
//!   edge to it cannot be written here without losing it silently;
//! - a link to a `Project` or `Note` that this backend does not hold is a
//!   no-op answered `Ok`, as Neo4j's `MATCH … MERGE` does.
//!
//! # Reads are served from memory, a write is visible once durable
//!
//! Same discipline as `project.rs`: the decoded note and its links are held in
//! `IndexNotes`, rebuilt from the base at open, and every write is published to
//! the index only after its durability step. Per-project and global sets bound
//! a project-scoped list to that project's notes. Entity links are indexed by
//! `(entity, id)`, so `get_notes_for_entity` costs the links of that entity.
//!
//! # Atomicity
//!
//! The creation carries the draft marker `_brouillon` (see `project.rs`): the
//! node, its properties and its `HAS_NOTE` edge are written under it, the marker
//! is removed last (the commit point), and opening deletes any draft with its
//! edges.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{project, Ecriture, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::neo4j::document::{Document, DocumentChunk, DocumentChunkHit};
use crate::neo4j::traits::NoteStore;
use crate::notes::provenance;
use crate::notes::{
    EntityType, MemoryHorizon, Note, NoteAnchor, NoteAuthorKind, NoteFilters, NoteImportance,
    NoteScope, NoteStatus, NoteType, PropagatedNote,
};

const LABEL: &str = "Note";
const BROUILLON: &str = "_brouillon";
const LIEN: &str = "LINKED_TO";
const APPARTIENT: &str = "HAS_NOTE";
const SUCCEDE: &str = "SUPERSEDES";

/// The methods of `NoteStore` refused on every call, by name. Listed here so the
/// accounting of `tests/obrain_backend.rs` and `methodes_qui_servent` can tell a
/// refusal from a port. Three more are ported and refuse only in some cases
/// (`create_note` for an agent author, `list_notes` and
/// `get_notes_needing_review` for a `workspace_slug`); they are not listed.
pub const REFUSEES: &[&str] = &[
    "NoteStore::create_document",
    "NoteStore::get_document",
    "NoteStore::find_document_by_sha256",
    "NoteStore::list_project_documents",
    "NoteStore::get_document_chunks",
    "NoteStore::upsert_document_chunks",
    "NoteStore::set_document_chunk_embeddings",
    "NoteStore::vector_search_document_chunks",
    "NoteStore::delete_document",
    "NoteStore::link_document_to_entity",
    "NoteStore::unlink_document_from_entity",
    "NoteStore::get_documents_for_entity",
    "NoteStore::link_note_to_document",
    "NoteStore::propagate_structural_links",
    "NoteStore::propagate_high_level_links",
    "NoteStore::propagate_note_via_feature_graph",
    "NoteStore::propagate_note_via_skill",
    "NoteStore::propagate_note_via_protocol",
    "NoteStore::propagate_semantic_links",
    "NoteStore::semantic_anchor_note",
    "NoteStore::anchor_orphan_notes_to_project",
    "NoteStore::get_propagated_notes",
    "NoteStore::get_workspace_notes_for_project",
    "NoteStore::update_staleness_scores",
    "NoteStore::set_note_embedding",
    "NoteStore::get_note_embedding",
    "NoteStore::vector_search_notes",
    "NoteStore::list_notes_without_embedding",
];

/// Entity types whose domain this backend ports: the only ones it can link.
fn entite_portee(t: &EntityType) -> bool {
    matches!(t, EntityType::Project | EntityType::Note)
}

/// The domain of an entity this backend does not port, as a named refusal.
fn refus_entite<T>(methode: &'static str, t: &EntityType) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(format!(
        "entité `{t}` : son domaine n'est pas porté ; l'arête ne peut pas être écrite ici sans \
         être perdue en silence"
    )))
}

/// An entity id of a ported type must be canonical (UUID, lowercase, hyphenated). Two
/// spellings of one node would otherwise give two links, since the index keys on the
/// string. Refused by name, before any write. Neo4j silently matches nothing for the
/// non-canonical spelling: a named divergence.
fn id_entite_canonique(t: &EntityType, id: &str) -> Result<()> {
    if entite_portee(t) {
        let canonique = id.parse::<Uuid>().is_ok_and(|u| u.to_string() == id);
        if !canonique {
            bail!(
                "identifiant d'entité non canonique « {id} » pour `{t}` (attendu : UUID en \
                 minuscules, avec tirets) : refusé pour qu'un même nœud ne reçoive pas deux liens"
            );
        }
    }
    Ok(())
}

/// A refusal, by name, with the reason it is not ported.
fn refus<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

/// `project_id` of the scope a note is attached to, as Neo4j stores the scope.
fn scope_vers_proprietes(scope: &NoteScope) -> (&'static str, String) {
    match scope {
        NoteScope::Workspace => ("workspace", String::new()),
        NoteScope::Project => ("project", String::new()),
        NoteScope::Module(p) => ("module", p.clone()),
        NoteScope::File(p) => ("file", p.clone()),
        NoteScope::Function(p) => ("function", p.clone()),
        NoteScope::Struct(p) => ("struct", p.clone()),
        NoteScope::Trait(p) => ("trait", p.clone()),
    }
}

fn scope_depuis_proprietes(type_: &str, chemin: String) -> NoteScope {
    match type_ {
        "module" => NoteScope::Module(chemin),
        "file" => NoteScope::File(chemin),
        "function" => NoteScope::Function(chemin),
        "struct" => NoteScope::Struct(chemin),
        "trait" => NoteScope::Trait(chemin),
        // As Neo4j's decoding: anything else, `workspace` included, is `Project`.
        _ => NoteScope::Project,
    }
}

/// The properties a creation writes, in the order it writes them. Mirrors the
/// `CREATE` of `Neo4jClient::create_note`.
fn proprietes_de_creation(note: &Note) -> Result<Vec<(&'static str, Value)>> {
    // The three floats a list sorts on must be finite: NaN has no place in a total order
    // (`partial_cmp` made the order non-transitive). Neo4j would store it; this is a
    // named divergence, refused at the write.
    for (nom, v) in [
        ("staleness_score", note.staleness_score),
        ("energy", note.energy),
        ("scar_intensity", note.scar_intensity),
    ] {
        if !v.is_finite() {
            bail!(
                "`{nom}` non fini ({v}) refusé : un tri total ne peut pas le ranger. Neo4j \
                 l'accepterait ; divergence assumée"
            );
        }
    }
    let (scope_type, scope_path) = scope_vers_proprietes(&note.scope);
    let mut p: Vec<(&'static str, Value)> = vec![
        ("id", Value::from(note.id.to_string())),
        ("note_type", Value::from(note.note_type.to_string())),
        ("status", Value::from(note.status.to_string())),
        ("importance", Value::from(note.importance.to_string())),
        ("scope_type", Value::from(scope_type)),
        ("scope_path", Value::from(scope_path)),
        ("content", Value::from(note.content.clone())),
        (
            "tags",
            Value::List(
                note.tags
                    .iter()
                    .map(|t| Value::from(t.clone()))
                    .collect::<Vec<_>>()
                    .into(),
            ),
        ),
        ("created_at", Value::from(note.created_at.to_rfc3339())),
        ("created_by", Value::from(note.created_by.clone())),
        ("author_kind", Value::from(note.author_kind.as_str())),
        (
            "human_confirmed_at",
            Value::from(
                note.human_confirmed_at
                    .map(|d| d.to_rfc3339())
                    .unwrap_or_default(),
            ),
        ),
        (
            "human_confirmed_by",
            Value::from(note.human_confirmed_by.clone().unwrap_or_default()),
        ),
        (
            "last_confirmed_at",
            Value::from(
                note.last_confirmed_at
                    .unwrap_or(note.created_at)
                    .to_rfc3339(),
            ),
        ),
        (
            "last_confirmed_by",
            Value::from(note.last_confirmed_by.clone().unwrap_or_default()),
        ),
        ("staleness_score", Value::from(note.staleness_score)),
        ("energy", Value::from(note.energy)),
        (
            "last_activated",
            Value::from(note.last_activated.unwrap_or(note.created_at).to_rfc3339()),
        ),
        (
            "changes_json",
            Value::from(serde_json::to_string(&note.changes)?),
        ),
        (
            "assertion_rule_json",
            Value::from(
                note.assertion_rule
                    .as_ref()
                    .map(|r| serde_json::to_string(r).unwrap_or_default())
                    .unwrap_or_default(),
            ),
        ),
        ("scar_intensity", Value::from(note.scar_intensity)),
        (
            "memory_horizon",
            Value::from(note.memory_horizon.to_string()),
        ),
        ("activation_count", Value::from(note.activation_count)),
        (
            "valid_from",
            Value::from(note.valid_from.map(|d| d.to_rfc3339()).unwrap_or_default()),
        ),
        (
            "valid_until",
            Value::from(note.valid_until.map(|d| d.to_rfc3339()).unwrap_or_default()),
        ),
        (
            "revision_json",
            Value::from(
                note.revision
                    .as_ref()
                    .map(|r| serde_json::to_string(r).unwrap_or_default())
                    .unwrap_or_default(),
            ),
        ),
        (
            "sharing_consent",
            Value::from(note.sharing_consent.as_db_str()),
        ),
    ];
    if let Some(c) = note.confidence {
        p.push(("confidence", Value::from(c)));
    }
    if let Some(pid) = note.project_id {
        p.push(("project_id", Value::from(pid.to_string())));
    }
    Ok(p)
}

/// An edge `LINKED_TO` from a note to an entity, as the index holds it.
#[derive(Clone)]
struct Lien {
    entite: EntityType,
    id_entite: String,
    arete: EdgeId,
    signature_hash: Option<String>,
    body_hash: Option<String>,
    last_verified: DateTime<Utc>,
}

/// One note: where it lives in the base, what the base says about it (decoded
/// once), its `updated_at`, and its links.
#[derive(Clone)]
pub(super) struct EntreeNote {
    /// Links on disk that this backend does not resolve (an unported domain, or a
    /// missing target). Their presence makes `get_note_anchors` refuse by name.
    non_resolus: usize,
    noeud: NodeId,
    note: Note,
    mise_a_jour: Option<DateTime<Utc>>,
    liens: Vec<Lien>,
}

impl EntreeNote {
    /// Decode the node (one hydration) into an entry with the given links.
    fn decoder(db: &ObrainDB, noeud: NodeId, liens: Vec<Lien>) -> Result<Self> {
        let get = lecteur(db, noeud)?;
        let (note, mise_a_jour) = lire_note(&get)?;
        Ok(Self {
            non_resolus: 0,
            noeud,
            note,
            mise_a_jour,
            liens,
        })
    }
}

/// A property lookup over one hydration of the node.
fn lecteur(db: &ObrainDB, noeud: NodeId) -> Result<impl Fn(&str) -> Option<Value>> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud Note {noeud:?} indexé mais absent de la base"))?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

/// The domain value of a `Note` node from a property lookup: the counterpart of
/// `Neo4jClient::node_to_note`. Returns the note and its `updated_at`.
fn lire_note(get: &dyn Fn(&str) -> Option<Value>) -> Result<(Note, Option<DateTime<Utc>>)> {
    let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let instant = |cle: &str| chaine(cle).and_then(|s| s.parse::<DateTime<Utc>>().ok());
    let entier = |cle: &str| get(cle).and_then(|v| v.as_int64());
    let reel = |cle: &str| get(cle).and_then(|v| v.as_float64());

    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Note sans propriété `id` lisible"))?;
    let requis = |cle: &str| chaine(cle).ok_or_else(|| anyhow!("note {id} sans propriété `{cle}`"));
    let status_brut = requis("status")?;
    let note_type_brut = requis("note_type")?;
    let created_by = requis("created_by")?;
    let content = requis("content")?;
    let created_at: DateTime<Utc> = requis("created_at")?
        .parse()
        .map_err(|e| anyhow!("note {id} : `created_at` illisible : {e}"))?;
    let type_scope = chaine("scope_type").unwrap_or_else(|| "project".to_string());
    let chemin_scope = chaine("scope_path").unwrap_or_default();
    let tags = get("tags")
        .and_then(|v| {
            v.as_list().map(|l| {
                l.iter()
                    .filter_map(|e| e.as_str().map(str::to_string))
                    .collect()
            })
        })
        .unwrap_or_default();
    let changes_json = chaine("changes_json").unwrap_or_else(|| "[]".to_string());
    let assertion_json = chaine("assertion_rule_json").unwrap_or_default();
    let revision_json = chaine("revision_json").unwrap_or_default();
    let resultat_json = chaine("last_assertion_result_json").unwrap_or_default();

    let note = Note {
        id,
        project_id: chaine("project_id")
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse().ok()),
        note_type: note_type_brut.parse().unwrap_or(NoteType::Observation),
        status: status_brut.parse().unwrap_or(NoteStatus::Active),
        importance: chaine("importance")
            .and_then(|s| s.parse().ok())
            .unwrap_or(NoteImportance::Medium),
        scope: scope_depuis_proprietes(&type_scope, chemin_scope),
        content,
        tags,
        anchors: vec![],
        created_at,
        created_by: created_by.clone(),
        author_kind: chaine("author_kind")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| NoteAuthorKind::infer_from_created_by(&created_by)),
        confidence: reel("confidence"),
        human_confirmed_at: instant("human_confirmed_at"),
        human_confirmed_by: chaine("human_confirmed_by").filter(|s| !s.is_empty()),
        last_confirmed_at: instant("last_confirmed_at"),
        last_confirmed_by: chaine("last_confirmed_by"),
        staleness_score: reel("staleness_score").unwrap_or(0.0),
        energy: reel("energy").unwrap_or(1.0),
        scar_intensity: reel("scar_intensity").unwrap_or(0.0),
        memory_horizon: chaine("memory_horizon")
            .and_then(|s| s.parse().ok())
            .unwrap_or(MemoryHorizon::Consolidated),
        activation_count: entier("activation_count").unwrap_or(0),
        last_activated: instant("last_activated"),
        supersedes: chaine("supersedes").and_then(|s| s.parse().ok()),
        superseded_by: chaine("superseded_by").and_then(|s| s.parse().ok()),
        valid_from: instant("valid_from"),
        valid_until: instant("valid_until"),
        revision: (!revision_json.is_empty())
            .then(|| serde_json::from_str(&revision_json).ok())
            .flatten(),
        changes: serde_json::from_str(&changes_json).unwrap_or_default(),
        assertion_rule: (!assertion_json.is_empty())
            .then(|| serde_json::from_str(&assertion_json).ok())
            .flatten(),
        last_assertion_result: (!resultat_json.is_empty())
            .then(|| serde_json::from_str(&resultat_json).ok())
            .flatten(),
        revision_reason: chaine("revision_reason").filter(|s| !s.is_empty()),
        reactivation_count: entier("reactivation_count").unwrap_or(0),
        last_reactivated: instant("last_reactivated"),
        freshness_pinged_at: instant("freshness_pinged_at"),
        sharing_consent: chaine("sharing_consent")
            .and_then(|s| serde_json::from_str(&format!("\"{s}\"")).ok())
            .unwrap_or_default(),
    };
    Ok((note, instant("updated_at")))
}

/// What an index sorts a note by (the fields of `NOTE_SORT_FIELDS`).
#[derive(PartialEq)]
enum Valeur {
    Instant(DateTime<Utc>),
    Nombre(f64),
    Texte(String),
}

impl Valeur {
    /// The variant's rank: values of one field always share a variant, so this only
    /// orders what could never meet, and keeps the order total.
    fn rang(&self) -> u8 {
        match self {
            Valeur::Instant(_) => 0,
            Valeur::Nombre(_) => 1,
            Valeur::Texte(_) => 2,
        }
    }
}

/// The index of the notes.
#[derive(Default)]
pub(super) struct IndexNotes {
    /// Ids of nodes that are on disk but could not be read at open (quarantined): a new
    /// note may not take one of them, or two nodes would share an id.
    reserves: HashSet<Uuid>,
    par_id: HashMap<Uuid, EntreeNote>,
    /// Notes attached to a project, by project id.
    par_projet: HashMap<Uuid, BTreeSet<Uuid>>,
    /// Notes with no project (`project_id` absent).
    globales: BTreeSet<Uuid>,
    /// Notes linked to an entity, by `(entity type, entity id)`.
    par_entite: HashMap<(String, String), BTreeSet<Uuid>>,
}

impl IndexNotes {
    /// Insert an entry, and its memberships.
    fn inscrire(&mut self, e: EntreeNote) {
        let id = e.note.id;
        match e.note.project_id {
            Some(p) => {
                self.par_projet.entry(p).or_default().insert(id);
            }
            None => {
                self.globales.insert(id);
            }
        }
        for l in &e.liens {
            self.par_entite
                .entry((l.entite.to_string(), l.id_entite.clone()))
                .or_default()
                .insert(id);
        }
        self.par_id.insert(id, e);
    }

    /// Remove an entry and all its memberships.
    fn desinscrire(&mut self, id: Uuid) -> Option<EntreeNote> {
        let e = self.par_id.remove(&id)?;
        match e.note.project_id {
            Some(p) => {
                if let Some(s) = self.par_projet.get_mut(&p) {
                    s.remove(&id);
                    if s.is_empty() {
                        self.par_projet.remove(&p);
                    }
                }
            }
            None => {
                self.globales.remove(&id);
            }
        }
        for l in &e.liens {
            let cle = (l.entite.to_string(), l.id_entite.clone());
            if let Some(s) = self.par_entite.get_mut(&cle) {
                s.remove(&id);
                if s.is_empty() {
                    self.par_entite.remove(&cle);
                }
            }
        }
        Some(e)
    }

    /// Publish a new state of a note (after its durability step).
    fn publier(&mut self, e: EntreeNote) {
        self.desinscrire(e.note.id);
        self.inscrire(e);
    }

    /// Rebuild the index from the persisted `Note` nodes at open: drafts are
    /// deleted (with their edges), what cannot be addressed is quarantined, and
    /// the links are read from the `LINKED_TO` edges. A scan of the notes and of
    /// their out-edges: O(notes + their links), paid at open only.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        projets: &project::IndexProjets,
        rapport: &mut RapportOuverture,
    ) -> Result<Self> {
        let store = db.store();
        let mut index = Self::default();
        let mut vus: Vec<(NodeId, Uuid)> = Vec::new();
        for noeud in store.nodes_by_label(LABEL) {
            let Some(n) = db.get_node(noeud) else {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Note {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                for (_, arete) in store.edges_from(noeud, Direction::Outgoing) {
                    db.delete_edge(arete);
                }
                for (_, arete) in store.edges_from(noeud, Direction::Incoming) {
                    db.delete_edge(arete);
                }
                db.delete_node(noeud);
                rapport.brouillons_supprimes += 1;
                continue;
            }
            let (note, maj) = match lire_note(&get) {
                Ok(v) => v,
                Err(e) => {
                    if let Some(id) =
                        get("id").and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
                    {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Note {noeud:?} : {e}"));
                    continue;
                }
            };
            if index.par_id.contains_key(&note.id) {
                rapport
                    .en_quarantaine
                    .push(format!("nœud Note {noeud:?} : id {} déjà pris", note.id));
                continue;
            }
            let id = note.id;
            index.inscrire(EntreeNote {
                non_resolus: 0,
                noeud,
                note,
                mise_a_jour: maj,
                liens: vec![],
            });
            vus.push((noeud, id));
        }

        // The links: the out-edges `LINKED_TO` of every indexed note.
        let mut liens_par_note: Vec<(Uuid, Vec<Lien>, usize)> = Vec::new();
        for (noeud, id) in &vus {
            let mut liens = Vec::new();
            let mut non_resolus = 0usize;
            for (cible, arete) in store.edges_from(*noeud, Direction::Outgoing) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                if &*e.edge_type != LIEN {
                    continue;
                }
                let etiquette = db
                    .get_node_labels(cible)
                    .and_then(|l| l.into_iter().next())
                    .unwrap_or_default();
                let cible_id = db.get_node(cible).and_then(|n| {
                    n.get_property("id")
                        .and_then(|v| v.as_str().map(str::to_string))
                });
                let entite = match etiquette.as_str() {
                    "Project" => EntityType::Project,
                    "Note" => EntityType::Note,
                    autre => {
                        rapport.en_quarantaine.push(format!(
                            "lien de la note {id} vers un nœud `{autre}` : domaine non porté, ignoré"
                        ));
                        non_resolus += 1;
                        continue;
                    }
                };
                let cible_existe = match &entite {
                    EntityType::Project => cible_id
                        .as_deref()
                        .and_then(|s| s.parse::<Uuid>().ok())
                        .is_some_and(|u| projets.noeud_de(u).is_some()),
                    _ => cible_id
                        .as_deref()
                        .and_then(|s| s.parse::<Uuid>().ok())
                        .is_some_and(|u| index.par_id.contains_key(&u)),
                };
                let Some(id_cible) = cible_id.filter(|_| cible_existe) else {
                    non_resolus += 1;
                    continue;
                };
                let texte = |cle: &str| {
                    e.get_property(cle)
                        .and_then(|v| v.as_str().map(str::to_string))
                };
                liens.push(Lien {
                    entite,
                    id_entite: id_cible,
                    arete,
                    signature_hash: texte("signature_hash").filter(|s| !s.is_empty()),
                    body_hash: texte("body_hash").filter(|s| !s.is_empty()),
                    last_verified: texte("last_verified")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(Utc::now),
                });
            }
            liens_par_note.push((*id, liens, non_resolus));
        }
        for (id, liens, nr) in liens_par_note {
            if let Some(mut e) = index.desinscrire(id) {
                e.non_resolus = nr;
                e.liens = liens;
                index.inscrire(e);
            }
        }
        Ok(index)
    }
}

impl ObrainGraphStore {
    /// The note entry of `id`, from the indexes, as a clone (the write closures
    /// work on a copy and publish a new state).
    fn entree_de(w: &Ecriture<'_>, id: Uuid) -> Result<Option<EntreeNote>> {
        Ok(w.index()?.notes.par_id.get(&id).cloned())
    }

    /// The decoded state of a note, read back from the engine after a write.
    fn relire(w: &Ecriture<'_>, e: &EntreeNote) -> Result<EntreeNote> {
        let mut r = w.apres_ecriture(EntreeNote::decoder(w.db, e.noeud, e.liens.clone()))?;
        r.non_resolus = e.non_resolus;
        Ok(r)
    }

    /// The `Project` node of `id`, if this backend holds that project.
    fn noeud_projet(w: &Ecriture<'_>, id: Uuid) -> Result<Option<NodeId>> {
        Ok(w.index()?.projets.noeud_de(id))
    }
}

/// The ported half of `NoteStore`. The name of the trait method is the name of
/// its refusal, so a refusal is findable by grepping the contract.
#[async_trait]
impl NoteStore for ObrainGraphStore {
    // ------------------------------------------------------------------
    // Documents: not ported (their own domain, and their chunks carry the
    // vector index).
    // ------------------------------------------------------------------

    async fn create_document(&self, _document: &Document, _chunks: &[DocumentChunk]) -> Result<()> {
        refus("NoteStore::create_document", "domaine Document non porté")
    }

    async fn get_document(&self, _id: Uuid) -> Result<Option<Document>> {
        refus("NoteStore::get_document", "domaine Document non porté")
    }

    async fn find_document_by_sha256(
        &self,
        _sha256: &str,
        _project_id: Option<Uuid>,
    ) -> Result<Option<Document>> {
        refus(
            "NoteStore::find_document_by_sha256",
            "domaine Document non porté",
        )
    }

    async fn list_project_documents(&self, _project_id: Uuid) -> Result<Vec<Document>> {
        refus(
            "NoteStore::list_project_documents",
            "domaine Document non porté",
        )
    }

    async fn get_document_chunks(&self, _document_id: Uuid) -> Result<Vec<DocumentChunk>> {
        refus(
            "NoteStore::get_document_chunks",
            "domaine Document non porté",
        )
    }

    async fn upsert_document_chunks(
        &self,
        _document_id: Uuid,
        _chunks: &[DocumentChunk],
    ) -> Result<usize> {
        refus(
            "NoteStore::upsert_document_chunks",
            "domaine Document non porté",
        )
    }

    async fn set_document_chunk_embeddings(
        &self,
        _embeddings: &[(Uuid, Vec<f32>)],
        _model: &str,
    ) -> Result<usize> {
        refus(
            "NoteStore::set_document_chunk_embeddings",
            "vecteurs : index vectoriel hors de ce lot (SearchStore)",
        )
    }

    async fn vector_search_document_chunks(
        &self,
        _embedding: &[f32],
        _limit: usize,
        _project_id: Option<Uuid>,
        _min_similarity: Option<f64>,
    ) -> Result<Vec<DocumentChunkHit>> {
        refus(
            "NoteStore::vector_search_document_chunks",
            "recherche vectorielle : SearchStore, hors de ce lot",
        )
    }

    async fn delete_document(&self, _id: Uuid) -> Result<bool> {
        refus("NoteStore::delete_document", "domaine Document non porté")
    }

    async fn link_document_to_entity(
        &self,
        _document_id: Uuid,
        _entity_type: &EntityType,
        _entity_id: &str,
    ) -> Result<()> {
        refus(
            "NoteStore::link_document_to_entity",
            "domaine Document non porté",
        )
    }

    async fn unlink_document_from_entity(
        &self,
        _document_id: Uuid,
        _entity_type: &EntityType,
        _entity_id: &str,
    ) -> Result<()> {
        refus(
            "NoteStore::unlink_document_from_entity",
            "domaine Document non porté",
        )
    }

    async fn get_documents_for_entity(
        &self,
        _entity_type: &EntityType,
        _entity_id: &str,
    ) -> Result<Vec<Document>> {
        refus(
            "NoteStore::get_documents_for_entity",
            "domaine Document non porté",
        )
    }

    async fn link_note_to_document(&self, _note_id: Uuid, _document_id: Uuid) -> Result<()> {
        refus(
            "NoteStore::link_note_to_document",
            "domaine Document non porté",
        )
    }

    // ------------------------------------------------------------------
    // Knowledge notes: ported.
    // ------------------------------------------------------------------

    async fn create_note(&self, note: &Note) -> Result<()> {
        if provenance::parse_actor(&note.created_by).is_some() {
            return refus(
                "NoteStore::create_note",
                "auteur agent (`agent:<uuid>`) : Neo4j écrit aussi le nœud Agent et l'arête \
                 AUTHORED, domaine non porté",
            );
        }
        let n = note.clone();
        self.ecrire(
            move || n,
            move |n, w: &mut Ecriture<'_>| {
                let (projet_noeud, proprietes) = {
                    let etat = w.index()?;
                    if etat.notes.par_id.contains_key(&n.id) {
                        bail!("une note d'id {} existe déjà", n.id);
                    }
                    if etat.notes.reserves.contains(&n.id) {
                        bail!(
                            "l'id {} est réservé : un nœud Note de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine). Il n'est ni repris ni écrasé",
                            n.id
                        );
                    }
                    let projet = match n.project_id {
                        Some(p) => etat.projets.noeud_de(p),
                        None => None,
                    };
                    (projet, proprietes_de_creation(&n)?)
                };
                let noeud = w.db.create_node(&[LABEL]);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeNoeud)?;
                w.etape()?;
                w.poser(noeud, BROUILLON, Value::from(true));
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeMarqueur)?;
                for (k, (cle, valeur)) in proprietes.into_iter().enumerate() {
                    w.etape()?;
                    w.poser(noeud, cle, valeur);
                    #[cfg(any(test, feature = "mock-store"))]
                    w.point(Interruption::ApresLaPropriete(k + 1))?;
                }
                if let Some(projet) = projet_noeud {
                    let _ = w.db.create_edge(projet, noeud, APPARTIENT);
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::AvantLeRetraitDuMarqueur)?;
                w.etape()?;
                w.db.remove_node_property(noeud, BROUILLON);
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLeRetraitDuMarqueur)?;
                let entree = EntreeNote::decoder(w.db, noeud, vec![]).and_then(|e| {
                    if e.note.id == n.id {
                        Ok(e)
                    } else {
                        bail!("la note relue n'a pas l'id écrit")
                    }
                });
                let entree = w.apres_ecriture(entree)?;
                w.rendre_durable()?;
                w.index_mut()?.notes.publier(entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_note(&self, id: Uuid) -> Result<Option<Note>> {
        self.lire(|_, etat| Ok(etat.notes.par_id.get(&id).map(|e| e.note.clone())))
    }

    async fn update_note(
        &self,
        id: Uuid,
        content: Option<String>,
        importance: Option<NoteImportance>,
        status: Option<NoteStatus>,
        tags: Option<Vec<String>>,
        staleness_score: Option<f64>,
    ) -> Result<Option<Note>> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = Self::entree_de(w, id)? else {
                    return Ok(None);
                };
                let staleness = staleness_score.filter(|v| v.is_finite());
                let noeud = ancienne.noeud;
                let maj = Utc::now();
                w.etape()?;
                w.poser(noeud, "updated_at", Value::from(maj.to_rfc3339()));
                if let Some(c) = &content {
                    w.etape()?;
                    w.poser(noeud, "content", Value::from(c.clone()));
                }
                if let Some(i) = importance {
                    w.etape()?;
                    w.poser(noeud, "importance", Value::from(i.to_string()));
                }
                if let Some(s) = status {
                    w.etape()?;
                    w.poser(noeud, "status", Value::from(s.to_string()));
                }
                if let Some(t) = &tags {
                    w.etape()?;
                    w.poser(
                        noeud,
                        "tags",
                        Value::List(
                            t.iter()
                                .map(|x| Value::from(x.clone()))
                                .collect::<Vec<_>>()
                                .into(),
                        ),
                    );
                }
                if let Some(s) = staleness {
                    w.etape()?;
                    w.poser(noeud, "staleness_score", Value::from(s));
                }
                let entree = Self::relire(w, &ancienne)?;
                w.rendre_durable()?;
                let note = entree.note.clone();
                w.index_mut()?.notes.publier(entree);
                Ok(Some(note))
            },
        )
        .await
    }

    async fn rescope_note(&self, note_id: Uuid, project_id: Option<Uuid>) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = Self::entree_de(w, note_id)? else {
                    return Ok(());
                };
                let noeud = ancienne.noeud;
                let projet_noeud = match project_id {
                    Some(p) => Self::noeud_projet(w, p)?,
                    None => None,
                };
                // The `HAS_NOTE` edge is moved with the property: the old one goes,
                // and a new one is drawn only towards a project this backend holds.
                let anciennes: Vec<EdgeId> =
                    w.db.store()
                        .edges_from(noeud, Direction::Incoming)
                        .into_iter()
                        .filter(|(_, a)| {
                            w.db.get_edge(*a)
                                .is_some_and(|e| &*e.edge_type == APPARTIENT)
                        })
                        .map(|(_, a)| a)
                        .collect();
                for a in anciennes {
                    w.etape()?;
                    w.db.delete_edge(a);
                }
                match project_id {
                    Some(p) => {
                        w.etape()?;
                        w.poser(noeud, "project_id", Value::from(p.to_string()))
                    }
                    None => {
                        w.etape()?;
                        w.db.remove_node_property(noeud, "project_id");
                    }
                }
                if let Some(pn) = projet_noeud {
                    w.etape()?;
                    w.db.create_edge(pn, noeud, APPARTIENT);
                }
                let entree = Self::relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.notes.publier(entree);
                Ok(())
            },
        )
        .await
    }

    async fn delete_note(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(e) = Self::entree_de(w, id)? else {
                    return Ok(false);
                };
                let noeud = e.noeud;
                // DETACH DELETE: every edge of the note goes with it, in both directions.
                let aretes: Vec<EdgeId> =
                    w.db.store()
                        .edges_from(noeud, Direction::Outgoing)
                        .into_iter()
                        .chain(w.db.store().edges_from(noeud, Direction::Incoming))
                        .map(|(_, a)| a)
                        .collect();
                for a in aretes {
                    w.etape()?;
                    w.db.delete_edge(a);
                }
                w.etape()?;
                if !w.db.delete_node(noeud) {
                    // The edges are already gone: what is on disk is no longer the state
                    // before this call. Same contract as `apres_ecriture`: everything is
                    // refused until a restart rebuilds the index from the base.
                    w.interne
                        .indetermine
                        .store(true, std::sync::atomic::Ordering::Release);
                    bail!(
                        "obrain a refusé la suppression du nœud {noeud:?} (note {id}) APRÈS la \
                         suppression de ses arêtes : base indéterminée, redémarrer"
                    );
                }
                #[cfg(any(test, feature = "mock-store"))]
                w.point(Interruption::ApresLaSuppression)?;
                w.rendre_durable()?;
                // The notes that linked TO this one lost their edge with it (DETACH):
                // their links are dropped from the index too.
                let mut etat = w.index_mut()?;
                let idx = &mut etat.notes;
                let sources: Vec<Uuid> = idx
                    .par_entite
                    .get(&(EntityType::Note.to_string(), id.to_string()))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                idx.desinscrire(id);
                for source in sources {
                    if let Some(mut s) = idx.desinscrire(source) {
                        s.liens.retain(|l| {
                            !(l.entite == EntityType::Note && l.id_entite == id.to_string())
                        });
                        idx.inscrire(s);
                    }
                }
                Ok(true)
            },
        )
        .await
    }

    async fn list_notes(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
        filters: &NoteFilters,
    ) -> Result<(Vec<Note>, usize)> {
        if filters.global_only != Some(true) && project_id.is_none() && workspace_slug.is_some() {
            return refus(
                "NoteStore::list_notes",
                "filtre par workspace_slug : le domaine Workspace n'est pas porté",
            );
        }
        let limite = filters.limit.unwrap_or(50);
        let decalage = filters.offset.unwrap_or(0);
        if limite < 0 || decalage < 0 {
            bail!("LIMIT ou SKIP négatif refusé (Neo4j refuse aussi)");
        }
        let tri_champ = note_order_field(filters.sort_by.as_deref());
        let desc = note_ordre_desc(filters.sort_order.as_deref());
        self.lire(|_, etat| {
            let candidats: Vec<&EntreeNote> = if filters.global_only == Some(true) {
                etat.notes
                    .globales
                    .iter()
                    .filter_map(|id| etat.notes.par_id.get(id))
                    .collect()
            } else if let Some(pid) = project_id {
                etat.notes
                    .par_projet
                    .get(&pid)
                    .into_iter()
                    .flatten()
                    .filter_map(|id| etat.notes.par_id.get(id))
                    .collect()
            } else {
                etat.notes.par_id.values().collect()
            };
            let mut notes: Vec<&EntreeNote> = candidats
                .into_iter()
                .filter(|e| filtre_note(&e.note, filters))
                .collect();
            notes.sort_by(|a, b| {
                let ord = cmp_tri(a, b, tri_champ);
                if desc {
                    ord.reverse()
                } else {
                    ord
                }
            });
            let total = notes.len();
            let page = notes
                .into_iter()
                .skip(decalage as usize)
                .take(limite as usize)
                .map(|e| e.note.clone())
                .collect();
            Ok((page, total))
        })
    }

    async fn link_note_to_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
        signature_hash: Option<&str>,
        body_hash: Option<&str>,
    ) -> Result<()> {
        if !entite_portee(entity_type) {
            return refus_entite("NoteStore::link_note_to_entity", entity_type);
        }
        id_entite_canonique(entity_type, entity_id)?;
        let signature = signature_hash.unwrap_or("").to_string();
        let corps = body_hash.unwrap_or("").to_string();
        let entite = entity_type.clone();
        let id_entite = entity_id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(mut e) = Self::entree_de(w, note_id)? else {
                    return Ok(());
                };
                // The target must exist in this backend, else nothing is written (Neo4j's MATCH).
                let cible = {
                    let etat = w.index()?;
                    match (&entite, id_entite.parse::<Uuid>().ok()) {
                        (EntityType::Project, Some(u)) => etat.projets.noeud_de(u),
                        (EntityType::Note, Some(u)) => etat.notes.par_id.get(&u).map(|n| n.noeud),
                        _ => None,
                    }
                };
                let Some(noeud_cible) = cible else {
                    return Ok(());
                };
                let maintenant = Utc::now();
                let arete = match e
                    .liens
                    .iter()
                    .position(|l| l.entite == entite && l.id_entite == id_entite)
                {
                    Some(k) => e.liens[k].arete,
                    None => w.db.create_edge(e.noeud, noeud_cible, LIEN),
                };
                w.etape()?;
                w.poser_arete(arete, "signature_hash", Value::from(signature.clone()));
                w.etape()?;
                w.poser_arete(arete, "body_hash", Value::from(corps.clone()));
                w.etape()?;
                w.poser_arete(arete, "last_verified", Value::from(maintenant.to_rfc3339()));
                let lien = Lien {
                    entite: entite.clone(),
                    id_entite: id_entite.clone(),
                    arete,
                    signature_hash: Some(signature.clone()).filter(|s| !s.is_empty()),
                    body_hash: Some(corps.clone()).filter(|s| !s.is_empty()),
                    last_verified: maintenant,
                };
                match e.liens.iter().position(|l| l.arete == arete) {
                    Some(k) => e.liens[k] = lien,
                    None => e.liens.push(lien),
                }
                w.rendre_durable()?;
                w.index_mut()?.notes.publier(e);
                Ok(())
            },
        )
        .await
    }

    async fn propagate_structural_links(&self, _project_id: Uuid) -> Result<usize> {
        refus(
            "NoteStore::propagate_structural_links",
            "propagation : parcourt IMPORTS/CALLS/CO_CHANGED, domaine code non porté",
        )
    }

    async fn propagate_high_level_links(&self, _project_id: Uuid) -> Result<usize> {
        refus(
            "NoteStore::propagate_high_level_links",
            "propagation : FeatureGraph/Skill/Protocol, domaines non portés",
        )
    }

    async fn propagate_note_via_feature_graph(
        &self,
        _note_id: Uuid,
        _feature_graph_id: &str,
    ) -> Result<usize> {
        refus(
            "NoteStore::propagate_note_via_feature_graph",
            "propagation : domaine FeatureGraph non porté",
        )
    }

    async fn propagate_note_via_skill(&self, _note_id: Uuid, _skill_id: &str) -> Result<usize> {
        refus(
            "NoteStore::propagate_note_via_skill",
            "propagation : domaine Skill non porté",
        )
    }

    async fn propagate_note_via_protocol(
        &self,
        _note_id: Uuid,
        _protocol_id: &str,
    ) -> Result<usize> {
        refus(
            "NoteStore::propagate_note_via_protocol",
            "propagation : domaine Protocol non porté",
        )
    }

    async fn propagate_semantic_links(
        &self,
        _project_id: Uuid,
        _min_similarity: f64,
    ) -> Result<usize> {
        refus(
            "NoteStore::propagate_semantic_links",
            "similarité : recherche vectorielle, SearchStore hors de ce lot",
        )
    }

    async fn semantic_anchor_note(
        &self,
        _note_id: Uuid,
        _project_id: Uuid,
        _min_similarity: f64,
    ) -> Result<usize> {
        refus(
            "NoteStore::semantic_anchor_note",
            "similarité : recherche vectorielle, SearchStore hors de ce lot",
        )
    }

    async fn anchor_orphan_notes_to_project(&self, _project_id: Uuid) -> Result<usize> {
        refus(
            "NoteStore::anchor_orphan_notes_to_project",
            "passe de rattrapage sur les orphelins : hors de ce lot",
        )
    }

    async fn unlink_note_from_entity(
        &self,
        note_id: Uuid,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<()> {
        if !entite_portee(entity_type) {
            return refus_entite("NoteStore::unlink_note_from_entity", entity_type);
        }
        id_entite_canonique(entity_type, entity_id)?;
        let entite = entity_type.clone();
        let id_entite = entity_id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(mut e) = Self::entree_de(w, note_id)? else {
                    return Ok(());
                };
                let Some(k) = e
                    .liens
                    .iter()
                    .position(|l| l.entite == entite && l.id_entite == id_entite)
                else {
                    return Ok(());
                };
                let arete = e.liens.remove(k).arete;
                w.etape()?;
                w.db.delete_edge(arete);
                w.rendre_durable()?;
                w.index_mut()?.notes.publier(e);
                Ok(())
            },
        )
        .await
    }

    async fn get_notes_for_entity(
        &self,
        entity_type: &EntityType,
        entity_id: &str,
    ) -> Result<Vec<Note>> {
        if !entite_portee(entity_type) {
            return refus_entite("NoteStore::get_notes_for_entity", entity_type);
        }
        id_entite_canonique(entity_type, entity_id)?;
        let cle = (entity_type.to_string(), entity_id.to_string());
        self.lire(|_, etat| {
            let mut notes: Vec<&Note> = etat
                .notes
                .par_entite
                .get(&cle)
                .into_iter()
                .flatten()
                .filter_map(|id| etat.notes.par_id.get(id))
                .map(|e| &e.note)
                .filter(|n| matches!(n.status, NoteStatus::Active | NoteStatus::NeedsReview))
                .collect();
            // ORDER BY importance DESC, created_at DESC, id DESC — the importance is
            // compared as its stored string, as Neo4j does.
            notes.sort_by(|a, b| {
                b.importance
                    .to_string()
                    .cmp(&a.importance.to_string())
                    .then_with(|| b.created_at.cmp(&a.created_at))
                    .then_with(|| b.id.cmp(&a.id))
            });
            Ok(notes.into_iter().cloned().collect())
        })
    }

    async fn get_propagated_notes(
        &self,
        _entity_type: &EntityType,
        _entity_id: &str,
        _max_depth: u32,
        _min_score: f64,
        _relation_types: Option<&[String]>,
        _source_project_id: Option<Uuid>,
        _force_cross_project: bool,
    ) -> Result<Vec<PropagatedNote>> {
        refus(
            "NoteStore::get_propagated_notes",
            "propagation : parcours de graphe et couplage inter-projets, hors de ce lot",
        )
    }

    async fn get_workspace_notes_for_project(
        &self,
        _project_id: Uuid,
        _propagation_factor: f64,
    ) -> Result<Vec<PropagatedNote>> {
        refus(
            "NoteStore::get_workspace_notes_for_project",
            "domaine Workspace non porté",
        )
    }

    async fn supersede_note(&self, old_note_id: Uuid, new_note_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (Some(vieille), Some(nouvelle)) = (
                    Self::entree_de(w, old_note_id)?,
                    Self::entree_de(w, new_note_id)?,
                ) else {
                    return Ok(());
                };
                w.etape()?;
                w.poser(vieille.noeud, "status", Value::from("archived"));
                w.etape()?;
                w.poser(
                    vieille.noeud,
                    "superseded_by",
                    Value::from(new_note_id.to_string()),
                );
                w.etape()?;
                w.poser(
                    nouvelle.noeud,
                    "supersedes",
                    Value::from(old_note_id.to_string()),
                );
                // MERGE: the edge is drawn once.
                let deja =
                    w.db.store()
                        .edges_from(nouvelle.noeud, Direction::Outgoing)
                        .into_iter()
                        .any(|(cible, a)| {
                            cible == vieille.noeud
                                && w.db.get_edge(a).is_some_and(|e| &*e.edge_type == SUCCEDE)
                        });
                if !deja {
                    w.etape()?;
                    w.db.create_edge(nouvelle.noeud, vieille.noeud, SUCCEDE);
                }
                let entree_vieille = Self::relire(w, &vieille)?;
                let entree_nouvelle = Self::relire(w, &nouvelle)?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                etat.notes.publier(entree_vieille);
                etat.notes.publier(entree_nouvelle);
                Ok(())
            },
        )
        .await
    }

    async fn confirm_note(&self, note_id: Uuid, confirmed_by: &str) -> Result<Option<Note>> {
        // A confirmation is a SIGNATURE only when a person gives it (as Neo4j).
        let par_humain = !confirmed_by.starts_with("agent:")
            && !confirmed_by.starts_with("system")
            && !confirmed_by.is_empty();
        let par = confirmed_by.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(e) = Self::entree_de(w, note_id)? else {
                    return Ok(None);
                };
                let noeud = e.noeud;
                // Superseded: a note that a newer one replaced is not resurrected.
                let remplacee = e.note.superseded_by.is_some()
                    || w.db
                        .store()
                        .edges_from(noeud, Direction::Incoming)
                        .into_iter()
                        .any(|(_, a)| w.db.get_edge(a).is_some_and(|x| &*x.edge_type == SUCCEDE));
                let maintenant = Utc::now();
                w.etape()?;
                w.poser(
                    noeud,
                    "last_confirmed_at",
                    Value::from(maintenant.to_rfc3339()),
                );
                w.etape()?;
                w.poser(noeud, "last_confirmed_by", Value::from(par.clone()));
                if par_humain {
                    w.etape()?;
                    w.poser(
                        noeud,
                        "human_confirmed_at",
                        Value::from(maintenant.to_rfc3339()),
                    );
                    w.etape()?;
                    w.poser(noeud, "human_confirmed_by", Value::from(par.clone()));
                }
                w.etape()?;
                w.poser(noeud, "staleness_score", Value::from(0.0_f64));
                if !remplacee {
                    w.etape()?;
                    w.poser(noeud, "status", Value::from("active"));
                    let energie = e.note.energy + 0.3;
                    w.etape()?;
                    w.poser(noeud, "energy", Value::from(energie.min(1.0)));
                }
                w.etape()?;
                w.poser(
                    noeud,
                    "last_activated",
                    Value::from(maintenant.to_rfc3339()),
                );
                let entree = Self::relire(w, &e)?;
                w.rendre_durable()?;
                let note = entree.note.clone();
                w.index_mut()?.notes.publier(entree);
                Ok(Some(note))
            },
        )
        .await
    }

    async fn get_notes_needing_review(
        &self,
        project_id: Option<Uuid>,
        workspace_slug: Option<&str>,
    ) -> Result<Vec<Note>> {
        if project_id.is_none() && workspace_slug.is_some() {
            return refus(
                "NoteStore::get_notes_needing_review",
                "filtre par workspace_slug : le domaine Workspace n'est pas porté",
            );
        }
        self.lire(|_, etat| {
            let candidats: Vec<&EntreeNote> = match project_id {
                Some(pid) => etat
                    .notes
                    .par_projet
                    .get(&pid)
                    .into_iter()
                    .flatten()
                    .filter_map(|id| etat.notes.par_id.get(id))
                    .collect(),
                None => etat.notes.par_id.values().collect(),
            };
            let mut notes: Vec<&Note> = candidats
                .into_iter()
                .map(|e| &e.note)
                .filter(|n| matches!(n.status, NoteStatus::NeedsReview | NoteStatus::Stale))
                .collect();
            // ORDER BY staleness_score DESC, importance DESC, id DESC LIMIT 500.
            notes.sort_by(|a, b| {
                b.staleness_score
                    .total_cmp(&a.staleness_score)
                    .then_with(|| b.importance.to_string().cmp(&a.importance.to_string()))
                    .then_with(|| b.id.cmp(&a.id))
            });
            Ok(notes.into_iter().take(500).cloned().collect())
        })
    }

    async fn update_staleness_scores(&self) -> Result<usize> {
        refus(
            "NoteStore::update_staleness_scores",
            "passe sur toutes les notes actives (coût proportionnel au nombre de notes) : hors de ce lot",
        )
    }

    async fn get_note_anchors(&self, note_id: Uuid) -> Result<Vec<NoteAnchor>> {
        self.lire(|_, etat| {
            let Some(e) = etat.notes.par_id.get(&note_id) else {
                return Ok(vec![]);
            };
            // A base migrated from Neo4j can hold links to domains this backend does not
            // port: they are not in `liens`. Answering the partial list would be a silent
            // empty or short result, so the answer is a named refusal instead.
            if e.non_resolus > 0 {
                return refus(
                    "NoteStore::get_note_anchors",
                    "la note a des liens sur disque vers des domaines non portés ou des cibles \
                     absentes : la liste serait partielle",
                );
            }
            Ok(e.liens
                .iter()
                .map(|l| NoteAnchor {
                    entity_type: l.entite.clone(),
                    entity_id: l.id_entite.clone(),
                    signature_hash: l.signature_hash.clone(),
                    body_hash: l.body_hash.clone(),
                    last_verified: l.last_verified,
                    is_valid: true,
                })
                .collect())
        })
    }

    async fn set_note_embedding(
        &self,
        _note_id: Uuid,
        _embedding: &[f32],
        _model: &str,
    ) -> Result<()> {
        refus(
            "NoteStore::set_note_embedding",
            "vecteurs : index vectoriel hors de ce lot (SearchStore)",
        )
    }

    async fn get_note_embedding(&self, _note_id: Uuid) -> Result<Option<Vec<f32>>> {
        refus(
            "NoteStore::get_note_embedding",
            "vecteurs : index vectoriel hors de ce lot (SearchStore)",
        )
    }

    async fn vector_search_notes(
        &self,
        _embedding: &[f32],
        _limit: usize,
        _project_id: Option<Uuid>,
        _workspace_slug: Option<&str>,
        _min_similarity: Option<f64>,
    ) -> Result<Vec<(Note, f64)>> {
        refus(
            "NoteStore::vector_search_notes",
            "recherche vectorielle : SearchStore hors de ce lot",
        )
    }

    async fn list_notes_without_embedding(
        &self,
        _limit: usize,
        _offset: usize,
    ) -> Result<(Vec<Note>, usize)> {
        refus(
            "NoteStore::list_notes_without_embedding",
            "vecteurs : index vectoriel hors de ce lot (SearchStore)",
        )
    }
}

/// Note properties a list may sort on (the same whitelist as Neo4j's).
const NOTE_SORT_FIELDS: &[&str] = &[
    "created_at",
    "updated_at",
    "importance",
    "status",
    "note_type",
    "staleness_score",
    "energy",
    "last_confirmed_at",
    "last_activated",
    "scar_intensity",
];

fn note_order_field(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|r| NOTE_SORT_FIELDS.iter().find(|f| **f == r))
        .copied()
        .unwrap_or("created_at")
}

/// `DESC` unless the request says `asc`, as Neo4j's `note_order_dir`.
fn note_ordre_desc(requested: Option<&str>) -> bool {
    !matches!(requested.map(|d| d.trim().to_ascii_lowercase()), Some(d) if d == "asc")
}

/// The sort value of a note for one field. `None` is a null property.
fn valeur_de_tri(e: &EntreeNote, champ: &str) -> Option<Valeur> {
    let n = &e.note;
    match champ {
        "created_at" => Some(Valeur::Instant(n.created_at)),
        "updated_at" => e.mise_a_jour.map(Valeur::Instant),
        "importance" => Some(Valeur::Texte(n.importance.to_string())),
        "status" => Some(Valeur::Texte(n.status.to_string())),
        "note_type" => Some(Valeur::Texte(n.note_type.to_string())),
        "staleness_score" => Some(Valeur::Nombre(n.staleness_score)),
        "energy" => Some(Valeur::Nombre(n.energy)),
        "last_confirmed_at" => n.last_confirmed_at.map(Valeur::Instant),
        "last_activated" => n.last_activated.map(Valeur::Instant),
        "scar_intensity" => Some(Valeur::Nombre(n.scar_intensity)),
        _ => Some(Valeur::Instant(n.created_at)),
    }
}

/// The ascending order of a list: by the field, nulls LAST (as Cypher ascending),
/// then by id. The caller reverses it for `DESC`, which puts nulls first, as Cypher.
fn cmp_tri(a: &EntreeNote, b: &EntreeNote, champ: &str) -> Ordering {
    let va = valeur_de_tri(a, champ);
    let vb = valeur_de_tri(b, champ);
    let par_champ = match (&va, &vb) {
        (Some(x), Some(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
    };
    par_champ.then_with(|| a.note.id.cmp(&b.note.id))
}

/// The predicates of `list_notes` (`build_note_where`), on the decoded note.
fn filtre_note(n: &Note, f: &NoteFilters) -> bool {
    if let Some(ref statuts) = f.status {
        if !statuts.contains(&n.status) {
            return false;
        }
    }
    if let Some(ref types) = f.note_type {
        if !types.contains(&n.note_type) {
            return false;
        }
    }
    if let Some(ref imp) = f.importance {
        if !imp.contains(&n.importance) {
            return false;
        }
    }
    if let Some(ref tags) = f.tags {
        if !tags.iter().all(|t| n.tags.contains(t)) {
            return false;
        }
    }
    // A NaN is excluded by both bounds, as Neo4j's `NaN >= x` (NULL in WHERE) excludes it.
    if let Some(min) = f.min_staleness.filter(|v| v.is_finite()) {
        if n.staleness_score.is_nan() || n.staleness_score < min {
            return false;
        }
    }
    if let Some(max) = f.max_staleness.filter(|v| v.is_finite()) {
        if n.staleness_score.is_nan() || n.staleness_score > max {
            return false;
        }
    }
    if let Some(ref s) = f.search {
        if !s.trim().is_empty() && !n.content.to_lowercase().contains(&s.to_lowercase()) {
            return false;
        }
    }
    true
}

impl PartialOrd for Valeur {
    /// Total: `f64::total_cmp` for numbers (a NaN is refused at the write, but the order
    /// stays total if one comes from a migrated base).
    fn partial_cmp(&self, autre: &Self) -> Option<Ordering> {
        Some(match (self, autre) {
            (Valeur::Instant(a), Valeur::Instant(b)) => a.cmp(b),
            (Valeur::Nombre(a), Valeur::Nombre(b)) => a.total_cmp(b),
            (Valeur::Texte(a), Valeur::Texte(b)) => a.cmp(b),
            (a, b) => a.rang().cmp(&b.rang()),
        })
    }
}
