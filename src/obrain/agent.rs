//! `AnalyticsStore` on obrain, sub-lot "agents and beliefs": the `Agent` node, its
//! attribution edges (`AUTHORED`, `LIVED`, `KNOWS`, `USES`), the knowledge it holds, and
//! the two writes a belief check needs (`list_assertion_notes`, `record_assertion_check`),
//! plus `project_synced`. The trait `impl` is in `analytics.rs`; each method there calls
//! the function written here. The reference is `src/neo4j/chat.rs` (agents) and
//! `src/neo4j/beliefs.rs` (beliefs).
//!
//! # Layout
//!
//! - `(:Agent {id, kind, created_at})`: `kind = 'self'` is the durable self of the store,
//!   whose id is `agent_identity_namespace` (the `AgentIdentityRoot`, `chat.rs`);
//!   `kind = 'model'` is a dated facet of it (`model`, `last_seen`, edge
//!   `(facet)-[:FACET_OF]->(self)`), one per normalized model name.
//! - `(:Agent)-[:AUTHORED {created_at}]->(:Note)`, `(:Agent)-[:LIVED {created_at}]->(:SessionEpisode)`,
//!   `(:Agent)-[:KNOWS {created_at, rule, measure, at, scope, project_id, score}]->(:Note)`,
//!   `(:Agent)-[:USES {created_at, rule, measure, at}]->(:Persona|:Skill|:Protocol)`.
//!   The edges are NOT indexed: they are read from the engine at each call (an agent
//!   holds tens of them). Only the `Agent` nodes are indexed (`IndexAgents`), rebuilt at
//!   open: a draft is deleted, an unreadable node is quarantined by name and its id
//!   reserved (a write on it refuses).
//!
//! # Deciding, and naming, what the other domains do NOT do
//!
//! - **`save_session_episode` does not make the agent nor the `LIVED` edge.** Neo4j does,
//!   best effort. Here the episode is written by `chat.rs` exactly as before: the agent
//!   domain is a sibling, not a dependency, of the episode write, and an episode is worth
//!   more than the edge that names it. The `LIVED` edge is drawn by `link_agent_lived`
//!   (its caller) and by `backfill_agents`, which rebuilds it from the episodes.
//! - **`create_note` still refuses a note whose author is `agent:<uuid>`** (`note.rs`
//!   documents it): Neo4j stamps the `Agent` node and its `AUTHORED` edge in the same write.
//!   That refusal is kept; `link_agent_authored` / `backfill_agents` draw the edge for the
//!   notes the base holds (a migrated base).
//! - **Deleting a note, a session or an episode takes the agent edges with it.**
//!   `delete_note` and `delete_chat_session` are `DETACH DELETE`s (every edge in both
//!   directions, `tests/obrain_agents.rs` proves it). Deleting a persona, a skill or a
//!   protocol is the engine's `delete_node`, which detaches its edges.
//! - **An `Agent` node makes `delete_project` refuse**, as any node that no domain
//!   cascade reaches (`project.rs`: `LABELS_PAR_DOMAINE` names the domain).
//!
//! # Differences with the Neo4j adapter, all named
//!
//! - **Not atomic.** A write of several engine records is ordered so that every prefix is
//!   valid and a retry converges: the node first (draft marker, `chat.rs` pattern), then
//!   the edge, then its properties. A cut between an edge and its `created_at` leaves an
//!   edge that the next call stamps; `created` then answers `false` (Neo4j's single
//!   statement answers `true` once). `ensure_agent` makes the self, then the facet, then
//!   `FACET_OF`, each durable on its own.
//! - **A target is resolved through the index** (a note, an episode): one the index does
//!   not hold is a no-op `Ok(false)` as Neo4j's `MATCH`; a note QUARANTINED at open refuses
//!   by name (Neo4j would link it). An episode quarantined at open is absent here.
//!   `USES` targets (`Persona`, `Skill`, `Protocol`) are found in the engine by label and
//!   id, as the `MATCH` does; two nodes of one id refuse.
//! - **Two edges of one type between the same two nodes** (a migrated base; Neo4j's
//!   `MERGE` never writes it) refuse by name on a link: updating one would leave the
//!   other stale.
//! - **`agent_knows` reads dates by the single decoder** (`date_valeur`), and orders by
//!   the instant. Neo4j compares the `at` STRINGS; the two agree on the format written
//!   here and differ on a mixed offset, where the instant is right. A present but
//!   unreadable `at`, or a `score` that is not a number, is an error naming the edge; Neo4j
//!   would read `""` / `0.0`. An ABSENT `score` is `0.0` and an ABSENT `at` is the oldest
//!   (`coalesce`, as Neo4j).
//! - **A non-finite `score`** (`NaN`, infinity) on a `KNOWS` edge is refused at the write: a
//!   total order cannot rank it. Neo4j would store it.
//! - **`list_assertion_notes` serves the notes the index holds.** A note quarantined at
//!   open that is a belief is not listed (the same limit as `list_notes`), the count is
//!   logged. The order is by instant (`last_checked_at`, else `created_at`), then id.
//! - **`record_assertion_check` writes `revised_at` and `last_checked_at` as RFC 3339
//!   strings** (Neo4j: `datetime()`), the checked-at LAST: a cut leaves the note at the
//!   head of the next pass, never marked checked without its result. A note quarantined
//!   at open refuses by name.
//! - **`project_synced` of a project the base holds but this backend quarantined refuses**
//!   (Neo4j would answer); an id the base does not hold is `None`.
//!
//! # What this file does NOT serve (refused by name in `analytics.rs`)
//!
//! `entity_exists`, `entities_linked` (they read `File`, `Function`, `Struct`, `Trait`,
//! `Enum`: the code domain is refused whole, answering `false` would REFUTE a belief) and
//! `ranked_foundation` (it counts `LINKED_TO` anchors on `File` and `SYNAPSE` edges).

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::note::{ecrire_proprietes_de_note, supprimer_arete};
use super::trigger::{
    charger, cle_reservee, creer_noeud, indexer, instant, refus_reserve, requise, texte,
    uuid_requis, Lecteur,
};
use super::{Ecriture, ObrainGraphStore, RapportOuverture};
use crate::graph::rules::{AccountableProvenance, AgentKnowledge, KnowledgeScope, USABLE_LABELS};
use crate::neo4j::models::AgentBackfill;
use crate::neo4j::traits::ChatStore;
use crate::notes::models::{AssertionResult, NoteStatus, NoteType};
use crate::notes::{provenance, Note};

const AGENT: &str = "Agent";
const AUTEUR: &str = "AUTHORED";
const VECU: &str = "LIVED";
const SAIT: &str = "KNOWS";
const UTILISE: &str = "USES";
const FACETTE: &str = "FACET_OF";

// ============================================================================
// The index
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Genre {
    Soi,
    Modele,
}

fn lire_agent(get: Lecteur) -> Result<(Uuid, Genre)> {
    let id = uuid_requis(get, "id", "nœud Agent")?;
    let quoi = format!("agent {id}");
    let genre = match requise(get, "kind", &quoi)?.as_str() {
        "self" => Genre::Soi,
        "model" => Genre::Modele,
        autre => bail!("{quoi} : `kind` inconnu « {autre} » (attendu `self` ou `model`)"),
    };
    // Read by the single decoder, and required: an agent with no readable creation date
    // is quarantined by name, not served with an invented one.
    super::date_propriete(get, "created_at")?
        .ok_or_else(|| anyhow!("{quoi} sans propriété `created_at`"))?;
    Ok((id, genre))
}

/// The `Agent` nodes, rebuilt at open.
#[derive(Default)]
pub(super) struct IndexAgents {
    par_id: HashMap<Uuid, (NodeId, Genre)>,
    /// Keys of the quarantined nodes ([`cle_reservee`]): never taken again.
    reserves: HashSet<String>,
}

impl IndexAgents {
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let mut index = Self::default();
        let fiches = charger(db, AGENT, rapport, &mut index.reserves, &lire_agent);
        for (id, f) in indexer(fiches, AGENT, rapport, |v| v.0) {
            index.par_id.insert(id, (f.noeud, f.valeur.1));
        }
        index
    }

    /// The node of an agent this backend holds; a quarantined id refuses by name.
    fn noeud(&self, id: Uuid) -> Result<Option<(NodeId, Genre)>> {
        if self
            .reserves
            .contains(&cle_reservee(AGENT, &[&id.to_string()]))
        {
            return Err(refus_reserve(&format!("l'agent {id}")));
        }
        Ok(self.par_id.get(&id).copied())
    }

    fn modeles(&self) -> usize {
        self.par_id
            .values()
            .filter(|(_, g)| *g == Genre::Modele)
            .count()
    }
}

// ============================================================================
// Edges
// ============================================================================

/// The edges of one type from `de` to `vers`.
fn aretes_vers(db: &ObrainDB, de: NodeId, vers: NodeId, type_: &str) -> Vec<EdgeId> {
    db.store()
        .edges_from(de, Direction::Outgoing)
        .into_iter()
        .filter(|(cible, _)| *cible == vers)
        .filter(|(_, a)| db.get_edge(*a).is_some_and(|e| &*e.edge_type == type_))
        .map(|(_, a)| a)
        .collect()
}

/// `MERGE (de)-[r:type]->(vers) ON CREATE SET r.created_at`: the edge and whether THIS
/// call drew it. Two edges already there refuse (module doc).
fn fusionner(
    w: &mut Ecriture<'_>,
    type_: &str,
    de: NodeId,
    vers: NodeId,
    quoi: &str,
) -> Result<(EdgeId, bool)> {
    let existantes = aretes_vers(w.db, de, vers, type_);
    match existantes.as_slice() {
        [] => {
            w.etape()?;
            let arete = w.db.create_edge(de, vers, type_);
            w.etape()?;
            w.poser_arete(arete, "created_at", instant(Utc::now()));
            Ok((arete, true))
        }
        [arete] => {
            let estampillee =
                w.db.get_edge(*arete)
                    .is_some_and(|e| e.get_property("created_at").is_some());
            if !estampillee {
                // A creation cut between the edge and its stamp: finish it.
                w.etape()?;
                w.poser_arete(*arete, "created_at", instant(Utc::now()));
            }
            Ok((*arete, false))
        }
        n => bail!(
            "{quoi} : la base porte {} arêtes {type_} entre ces deux nœuds (Neo4j n'en écrit \
             jamais deux) ; en mettre à jour une laisserait l'autre périmée",
            n.len()
        ),
    }
}

/// Sets (`Some`) or removes (`None`) a property of an edge, one engine write.
fn poser_ou_retirer(
    w: &mut Ecriture<'_>,
    arete: EdgeId,
    cle: &'static str,
    valeur: Option<Value>,
) -> Result<()> {
    w.etape()?;
    match valeur {
        Some(v) => w.poser_arete(arete, cle, v),
        None => {
            w.db.remove_edge_property(arete, cle);
        }
    }
    Ok(())
}

/// What an attribution edge points to.
#[derive(Clone)]
enum Cible {
    Note(Uuid),
    Episode(Uuid),
    /// A node found in the engine by label and id (`USES`).
    Etiquette(&'static str, Uuid),
}

fn resoudre(w: &Ecriture<'_>, cible: &Cible) -> Result<Option<NodeId>> {
    match cible {
        Cible::Note(id) => {
            let etat = w.index()?;
            match etat.notes.noeud_de(*id) {
                Some(n) => Ok(Some(n)),
                None if etat.notes.est_reservee(*id) => {
                    Err(refus_reserve(&format!("la note {id}")))
                }
                None => Ok(None),
            }
        }
        Cible::Episode(id) => Ok(w.index()?.chat.noeud_d_episode(*id)),
        Cible::Etiquette(label, id) => {
            let voulu = id.to_string();
            let trouves: Vec<NodeId> =
                w.db.store()
                    .nodes_by_label(label)
                    .into_iter()
                    .filter(|n| {
                        w.db.get_node(*n).is_some_and(|nd| {
                            nd.get_property("id")
                                .and_then(|v| v.as_str())
                                .is_some_and(|s| s == voulu)
                        })
                    })
                    .collect();
            match trouves.as_slice() {
                [] => Ok(None),
                [n] => Ok(Some(*n)),
                _ => bail!(
                    "{} nœuds {label} portent l'id {id} : lien refusé",
                    trouves.len()
                ),
            }
        }
    }
}

fn etiquettes(db: &ObrainDB, noeud: NodeId) -> Vec<String> {
    db.get_node_labels(noeud).unwrap_or_default()
}

fn propriete_texte(db: &ObrainDB, noeud: NodeId, cle: &str) -> Option<String> {
    db.get_node(noeud).and_then(|n| {
        n.get_property(cle)
            .and_then(|v| v.as_str().map(str::to_string))
    })
}

// ============================================================================
// The agent
// ============================================================================

/// `ensure_agent`'s write: the self, then the facet and its `FACET_OF`. Each node is made
/// durable and published on its own, so a cut leaves nodes the index knows.
fn assurer(
    w: &mut Ecriture<'_>,
    racine: Uuid,
    facette: Option<Uuid>,
    modele: &str,
) -> Result<bool> {
    let soi = w.index()?.agents.noeud(racine)?;
    let facette_existante = match facette {
        Some(f) => Some(w.index()?.agents.noeud(f)?),
        None => None,
    };
    let (soi, cree) = match soi {
        Some((n, _)) => (n, false),
        None => {
            let n = creer_noeud(
                w,
                AGENT,
                vec![
                    ("id", texte(racine.to_string())),
                    ("kind", texte("self")),
                    ("created_at", instant(Utc::now())),
                ],
            )?;
            w.rendre_durable()?;
            w.index_mut()?.agents.par_id.insert(racine, (n, Genre::Soi));
            (n, true)
        }
    };
    if let Some(f) = facette {
        let maintenant = instant(Utc::now());
        let noeud = match facette_existante.flatten() {
            Some((n, _)) => n,
            None => {
                let n = creer_noeud(
                    w,
                    AGENT,
                    vec![
                        ("id", texte(f.to_string())),
                        ("kind", texte("model")),
                        ("created_at", maintenant.clone()),
                    ],
                )?;
                w.rendre_durable()?;
                w.index_mut()?.agents.par_id.insert(f, (n, Genre::Modele));
                n
            }
        };
        w.etape()?;
        w.poser(noeud, "model", texte(modele));
        w.etape()?;
        w.poser(noeud, "last_seen", maintenant);
        if aretes_vers(w.db, noeud, soi, FACETTE).is_empty() {
            w.etape()?;
            w.db.create_edge(noeud, soi, FACETTE);
        }
        w.rendre_durable()?;
    }
    Ok(cree)
}

/// The durable self an agent id stands for: the root itself, or the root a model facet of
/// this store hangs under. `None` for any other id.
fn soi_de(w: &Ecriture<'_>, id: Uuid, racine: Uuid) -> Result<Option<Uuid>> {
    if id == racine {
        return Ok(Some(racine));
    }
    let etat = w.index()?;
    let Some((facette, Genre::Modele)) = etat.agents.noeud(id)? else {
        return Ok(None);
    };
    let Some((soi, _)) = etat.agents.noeud(racine)? else {
        return Ok(None);
    };
    Ok((!aretes_vers(w.db, facette, soi, FACETTE).is_empty()).then_some(racine))
}

/// A plain attribution edge (`AUTHORED`, `LIVED`): `true` when THIS call drew it.
fn lier_simple(
    w: &mut Ecriture<'_>,
    type_: &'static str,
    agent: Uuid,
    cible: &Cible,
) -> Result<bool> {
    let Some((a, _)) = w.index()?.agents.noeud(agent)? else {
        return Ok(false);
    };
    let Some(t) = resoudre(w, cible)? else {
        return Ok(false);
    };
    let (_, cree) = fusionner(w, type_, a, t, &format!("{type_} de l'agent {agent}"))?;
    w.rendre_durable()?;
    Ok(cree)
}

impl ObrainGraphStore {
    pub(super) async fn agent_assurer(&self, agent_id: Uuid, model: Option<&str>) -> Result<bool> {
        let racine = self.agent_identity_namespace().await?;
        if agent_id != racine {
            // Neo4j's own refusal: the only agent of a store is its durable self.
            bail!("{agent_id} is not the durable self of this store");
        }
        let facette = model.and_then(|m| provenance::model_facet_id(racine, m));
        let nom = model
            .and_then(provenance::normalized_model)
            .unwrap_or_default();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| assurer(w, racine, facette, &nom),
        )
        .await
    }

    pub(super) async fn agent_reprise(&self) -> Result<AgentBackfill> {
        let racine = self.agent_identity_namespace().await?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let avant = w.index()?.agents.modeles();
                // What is already written, read from the indexes (Neo4j reads the nodes).
                let notes: Vec<(Uuid, Uuid)> = w
                    .index()?
                    .notes
                    .avec_noeuds()
                    .into_iter()
                    .filter_map(|(n, _)| provenance::parse_actor(&n.created_by).map(|a| (a, n.id)))
                    .collect();
                let episodes: Vec<(Uuid, Uuid)> = w
                    .index()?
                    .chat
                    .episodes_et_agents()
                    .into_iter()
                    .filter_map(|(e, _, a)| a.map(|a| (a, e)))
                    .collect();
                let des_notes: HashSet<Uuid> = notes.iter().map(|(a, _)| *a).collect();
                let des_episodes: HashSet<Uuid> = episodes.iter().map(|(a, _)| *a).collect();
                // An actor resolves to the self through its `FACET_OF`; an id this store
                // does not know resolves to nothing and draws nothing.
                let mut resolus: HashMap<Uuid, Uuid> = HashMap::new();
                for brut in des_notes.union(&des_episodes) {
                    if let Some(s) = soi_de(w, *brut, racine)? {
                        resolus.insert(*brut, s);
                    }
                }
                if !resolus.is_empty() {
                    assurer(w, racine, None, "")?;
                }
                // The same edges the write path draws: one path, not two that could diverge.
                let mut ecrits = 0;
                for (agent, note) in &notes {
                    if let Some(s) = resolus.get(agent) {
                        if lier_simple(w, AUTEUR, *s, &Cible::Note(*note))? {
                            ecrits += 1;
                        }
                    }
                }
                let mut vecus = 0;
                for (agent, episode) in &episodes {
                    if let Some(s) = resolus.get(agent) {
                        if lier_simple(w, VECU, *s, &Cible::Episode(*episode))? {
                            vecus += 1;
                        }
                    }
                }
                let apres = w.index()?.agents.modeles();
                Ok(AgentBackfill {
                    before: avant,
                    after: apres,
                    from_notes: des_notes.len(),
                    from_episodes: des_episodes.len(),
                    authored: ecrits,
                    lived: vecus,
                })
            },
        )
        .await
    }

    pub(super) async fn agent_a_ecrit(&self, agent_id: Uuid, note_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| lier_simple(w, AUTEUR, agent_id, &Cible::Note(note_id)),
        )
        .await
    }

    pub(super) async fn agent_a_vecu(&self, agent_id: Uuid, episode_id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                lier_simple(w, VECU, agent_id, &Cible::Episode(episode_id))
            },
        )
        .await
    }

    pub(super) async fn agent_sait(
        &self,
        agent_id: Uuid,
        note_id: Uuid,
        scope: KnowledgeScope,
        project: Option<Uuid>,
        provenance: &AccountableProvenance,
    ) -> Result<bool> {
        let p = provenance.as_provenance().clone();
        if let Some(s) = p.score {
            if !s.is_finite() {
                bail!("`score` non fini ({s}) refusé : un tri total ne peut pas le ranger");
            }
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some((a, _)) = w.index()?.agents.noeud(agent_id)? else {
                    return Ok(false);
                };
                let Some(t) = resoudre(w, &Cible::Note(note_id))? else {
                    return Ok(false);
                };
                let (arete, cree) =
                    fusionner(w, SAIT, a, t, &format!("KNOWS de l'agent {agent_id}"))?;
                // `SET` every time: the last measure wins, as Neo4j's `SET` after `MERGE`.
                poser_ou_retirer(w, arete, "rule", Some(texte(p.rule.clone())))?;
                poser_ou_retirer(w, arete, "measure", Some(texte(p.measure.clone())))?;
                poser_ou_retirer(w, arete, "at", Some(texte(p.at.clone())))?;
                poser_ou_retirer(w, arete, "scope", Some(texte(scope.as_str())))?;
                poser_ou_retirer(
                    w,
                    arete,
                    "project_id",
                    project.map(|u| texte(u.to_string())),
                )?;
                poser_ou_retirer(w, arete, "score", p.score.map(Value::from))?;
                w.rendre_durable()?;
                Ok(cree)
            },
        )
        .await
    }

    pub(super) async fn agent_utilise(
        &self,
        agent_id: Uuid,
        label: &str,
        target_id: Uuid,
        provenance: &AccountableProvenance,
    ) -> Result<bool> {
        // Neo4j validates the label first, before it looks at anything.
        let Some(etiquette) = USABLE_LABELS.iter().copied().find(|l| *l == label) else {
            bail!(
                "{label} is not a usable label: {}",
                USABLE_LABELS.join(", ")
            );
        };
        let p = provenance.as_provenance().clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some((a, _)) = w.index()?.agents.noeud(agent_id)? else {
                    return Ok(false);
                };
                let Some(t) = resoudre(w, &Cible::Etiquette(etiquette, target_id))? else {
                    return Ok(false);
                };
                let (arete, cree) =
                    fusionner(w, UTILISE, a, t, &format!("USES de l'agent {agent_id}"))?;
                poser_ou_retirer(w, arete, "rule", Some(texte(p.rule.clone())))?;
                poser_ou_retirer(w, arete, "measure", Some(texte(p.measure.clone())))?;
                poser_ou_retirer(w, arete, "at", Some(texte(p.at.clone())))?;
                w.rendre_durable()?;
                Ok(cree)
            },
        )
        .await
    }

    pub(super) async fn agent_relacher(
        &self,
        agent_id: Uuid,
        rule: &str,
        keep: &[Uuid],
    ) -> Result<usize> {
        let regle = rule.to_string();
        let garder: HashSet<String> = keep.iter().map(|k| k.to_string()).collect();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some((a, _)) = w.index()?.agents.noeud(agent_id)? else {
                    return Ok(0);
                };
                let mut a_retirer: Vec<EdgeId> = Vec::new();
                for (cible, arete) in w.db.store().edges_from(a, Direction::Outgoing) {
                    let Some(e) = w.db.get_edge(arete) else {
                        continue;
                    };
                    if &*e.edge_type != SAIT
                        || e.get_property("rule").and_then(|v| v.as_str()) != Some(regle.as_str())
                        || !etiquettes(w.db, cible).iter().any(|l| l == "Note")
                    {
                        continue;
                    }
                    // `NOT (n.id IN $keep)` of a note with no `id` is null: not released.
                    let Some(id) = propriete_texte(w.db, cible, "id") else {
                        continue;
                    };
                    if !garder.contains(&id) {
                        a_retirer.push(arete);
                    }
                }
                let n = a_retirer.len();
                for arete in a_retirer {
                    w.etape()?;
                    supprimer_arete(w, arete, "release_agent_knows")?;
                }
                w.rendre_durable()?;
                Ok(n)
            },
        )
        .await
    }

    pub(super) fn agent_connait(
        &self,
        agent_id: Uuid,
        project: Option<Uuid>,
        limit: usize,
    ) -> Result<AgentKnowledge> {
        self.lire(move |db, etat| {
            let Some((a, _)) = etat.agents.noeud(agent_id)? else {
                return Ok(AgentKnowledge {
                    notes: vec![],
                    available: 0,
                });
            };
            let voulu = project.map(|p| p.to_string());
            // (id, score, instant of `at`)
            let mut tous: Vec<(Uuid, f64, Option<DateTime<Utc>>)> = Vec::new();
            for (cible, arete) in db.store().edges_from(a, Direction::Outgoing) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                if &*e.edge_type != SAIT || !etiquettes(db, cible).iter().any(|l| l == "Note") {
                    continue;
                }
                let quoi = format!("arête KNOWS de l'agent {agent_id}");
                let chaine = |cle: &str| -> Result<Option<String>> {
                    match e.get_property(cle) {
                        None | Some(Value::Null) => Ok(None),
                        Some(v) => v
                            .as_str()
                            .map(|s| Some(s.to_string()))
                            .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas une chaîne")),
                    }
                };
                // `r.scope = 'global' OR ($project IS NOT NULL AND r.project_id = $project)`
                let global = chaine("scope")?.as_deref() == Some("global");
                let du_projet = match (&voulu, chaine("project_id")?) {
                    (Some(v), Some(p)) => *v == p,
                    _ => false,
                };
                if !(global || du_projet) {
                    continue;
                }
                let score = match e.get_property("score") {
                    None | Some(Value::Null) => 0.0,
                    Some(v) => v
                        .as_float64()
                        .ok_or_else(|| anyhow!("{quoi} : `score` n'est pas un nombre"))?,
                };
                let at = match e.get_property("at") {
                    None => None,
                    Some(v) => {
                        super::date_valeur(v, "at").map_err(|err| anyhow!("{quoi} : {err}"))?
                    }
                };
                let id = propriete_texte(db, cible, "id")
                    .and_then(|s| s.parse::<Uuid>().ok())
                    .ok_or_else(|| anyhow!("{quoi} : la note cible n'a pas d'`id` lisible"))?;
                tous.push((id, score, at));
            }
            // Strongest first; the instant breaks ties (newest first), then the id.
            tous.sort_by(|x, y| {
                y.1.partial_cmp(&x.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| y.2.cmp(&x.2))
                    .then_with(|| x.0.cmp(&y.0))
            });
            Ok(AgentKnowledge {
                available: tous.len(),
                notes: tous.into_iter().take(limit).map(|(id, _, _)| id).collect(),
            })
        })
    }

    pub(super) fn projet_synchronise(&self, project_id: Uuid) -> Result<Option<bool>> {
        self.lire(move |db, etat| {
            if let Some(p) = etat.projets.projet(project_id) {
                return Ok(Some(p.last_synced.is_some()));
            }
            if super::vues::projet_en_quarantaine(db, project_id) {
                bail!(
                    "projet {project_id} : un nœud de cet id est sur disque, illisible à \
                     l'ouverture (quarantaine) ; la réponse de Neo4j ne peut pas être donnée"
                );
            }
            Ok(None)
        })
    }

    pub(super) fn croyances(&self, limit: usize) -> Result<Vec<Note>> {
        self.lire(move |db, etat| {
            let mut v: Vec<(DateTime<Utc>, Uuid, Note)> = Vec::new();
            for (note, noeud) in etat.notes.avec_noeuds() {
                // The raw property, as Neo4j's predicate reads it: the decoded note has
                // already turned an unreadable rule into `None`.
                let brute = match db
                    .get_node(noeud)
                    .and_then(|n| n.get_property("assertion_rule_json").cloned())
                {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(v.as_str().map(str::to_string).ok_or_else(|| {
                        anyhow!(
                            "note {} : `assertion_rule_json` n'est pas une chaîne",
                            note.id
                        )
                    })?),
                };
                let a_une_regle = brute
                    .as_deref()
                    .is_some_and(|s| !s.is_empty() && s != "null");
                if !(a_une_regle || note.note_type == NoteType::Assertion) {
                    continue;
                }
                let verifiee = match db
                    .get_node(noeud)
                    .and_then(|n| n.get_property("last_checked_at").cloned())
                {
                    Some(v) => super::date_valeur(&v, "last_checked_at")
                        .map_err(|e| anyhow!("note {} : {e}", note.id))?,
                    None => None,
                };
                v.push((verifiee.unwrap_or(note.created_at), note.id, note.clone()));
            }
            if etat.notes.en_quarantaine() > 0 {
                tracing::warn!(
                    quarantaine = etat.notes.en_quarantaine(),
                    "list_assertion_notes : des notes en quarantaine ne sont pas listées"
                );
            }
            v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            v.truncate(limit);
            Ok(v.into_iter().map(|(_, _, n)| n).collect())
        })
    }

    pub(super) async fn croyance_verifiee(
        &self,
        note_id: Uuid,
        result: &AssertionResult,
        status: Option<NoteStatus>,
        reason: Option<&str>,
    ) -> Result<()> {
        let json = serde_json::to_string(result)?;
        let raison = reason.map(str::to_string);
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                if w.index()?.notes.noeud_de(note_id).is_none() {
                    if w.index()?.notes.est_reservee(note_id) {
                        return Err(refus_reserve(&format!("la note {note_id}")));
                    }
                    // A note gone between the read and the write: nothing to do, as Neo4j.
                    return Ok(());
                }
                let maintenant = instant(Utc::now());
                let mut p: Vec<(&'static str, Option<Value>)> = Vec::new();
                // `status` absent writes nothing: a `Warn` traces without breaking the note.
                // The revision first, the result and the check date LAST (module doc).
                if let Some(s) = status {
                    p.push(("revision_reason", raison.clone().map(texte)));
                    p.push(("revised_at", Some(maintenant.clone())));
                    p.push(("status", Some(texte(s.to_string()))));
                }
                p.push(("last_assertion_result_json", Some(texte(json.clone()))));
                p.push(("last_checked_at", Some(maintenant)));
                ecrire_proprietes_de_note(w, note_id, p)?;
                Ok(())
            },
        )
        .await
    }
}
