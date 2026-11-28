//! `AnalyticsStore` on obrain, in four sub-lots: A1 ("profiles, rules, routing, settings"),
//! A2 delegated to `synapse.rs` ("the neural layer of the notes": synapses, energy, scars,
//! the deep-maintenance clock), the memory-and-maintenance lot (`memoire.rs`), and the
//! "agents and views" lot (`agent.rs`, `vues.rs`, 15 methods that delegate there): 79 of the
//! 145 methods serve, `run_data_migrations` answers its documented "no migration", and the
//! 66 others are refused by name in this very `impl` ([`REFUSEES`]). Nothing here answers an
//! empty or default value in place of a
//! refusal. The classification of the 145 methods into sub-lots is in
//! `CONSOLIDATION.md` ("Portage d'`AnalyticsStore` : sous-lots").
//!
//! # What serves
//!
//! - analysis profiles (`AnalysisProfile`, 4 methods) and topology rules
//!   (`TopologyRule`, create / list / delete);
//! - user profiles (`UserProfile`) and their `WORKS_ON` edge to a project (5);
//! - the routing arms (`RoutingArm`) and decisions (`RoutingDecision`) of the
//!   cognitive router (7);
//! - `health_check`, `llm_settings_by_key` (reads the `LlmSetting` index of
//!   `chat.rs`), `get_step_parent_task_id`, `get_frustration`,
//!   `increment_frustration`, `decrement_frustration` (read and write the task
//!   index of `task.rs`).
//!
//! No Cypher or GQL: everything goes through the Rust API, one index per label
//! rebuilt at open (as `trigger.rs`, whose decoders and `charger` are reused).
//!
//! # Atomicity
//!
//! A creation is a draft until its last property (`_brouillon`, `trigger::creer`).
//! A multi-property update (an arm's seven counters, a profile's six fields, the
//! rewrite of a decision by id) is not
//! atomic in the engine: it is made so by the update marker `_maj` of
//! `protocolrun.rs` (the whole new state on the node first, then the properties,
//! the marker removed last; opening finishes a cut update from it and counts it in
//! `RapportOuverture::mises_a_jour_achevees`). The `WORKS_ON` edge carries two
//! properties updated together: an update CREATES the new edge (frequency and
//! date in one record) then deletes the old one; a cut leaves two edges, and
//! opening keeps the one with the highest frequency and deletes the other.
//!
//! # Differences with the Neo4j adapter, all named
//!
//! - **Absent targets refuse, except for `upsert_works_on`.** Neo4j's `MATCH … MERGE`
//!   creates nothing and answers `Ok` when the project of `create_analysis_profile`
//!   is absent, and `update_user_profile` of an unknown user writes nothing. Here
//!   those two refuse with an error. `upsert_works_on` of an absent user or project
//!   keeps Neo4j's no-op (logged): its one production caller, the profile
//!   aggregator, upserts every project of a flush with `?`, and a refusal would drop
//!   the relations of the projects after it.
//! - (Not a difference: a repeated `TopologyRule.id` refuses, as Neo4j's uniqueness
//!   constraint `topology_rule_id` makes its `CREATE` fail.)
//! - **`threshold` keeps its absence.** Neo4j writes `0` for a rule without
//!   threshold and reads `Some(0)` back; here an absent threshold stays absent.
//! - **No `HAS_PROFILE` edge, no `ROUTED_FOR` edge.** The membership of a profile is
//!   its `project_id` property; a decision keeps its `session_id`, `task_id` and
//!   `run_id` as properties. Nothing in the contract reads those edges, and writing
//!   them would attach foreign edges to tasks and sessions (which refuse deletion
//!   when they carry an edge of a domain they do not know).
//! - **Strict decoding.** Neo4j logs and defaults an unreadable JSON, a missing
//!   `created_at`; here the node is quarantined by name at open (its key reserved)
//!   and a write that would take its key refuses. A list that could miss a
//!   quarantined node refuses, as in the other ported domains: the profiles visible
//!   to a project, the rules of a project, the arms of a class; the `WORKS_ON` of a
//!   user with a quarantined edge (read and upsert: rewriting it would lose its
//!   frequency).
//! - **Decisions follow Neo4j's skip.** A decision whose document does not decode is
//!   quarantined by name at open (Neo4j skips it at each read and logs) and is not
//!   in `list_routing_decisions`: Neo4j's own contract for that list is "an
//!   unreadable document is skipped", and its callers (demotion, report, routes) need
//!   the rest of the log. `get_routing_decision` of it refuses.
//! - **`list_routing_decisions` is O(decisions)** over a light in-memory entry
//!   (id, instant, project, session); only the page is decoded from its JSON.
//! - **`get_frustration` / `increment_frustration` / `decrement_frustration` of an
//!   unknown task answer `0.0`**, as Neo4j's `MATCH` with no row.
//! - `run_data_migrations` returns an empty list: the one-time migrations are
//!   written in Cypher for a Neo4j base (`neo4j::data_migrations`) and none is
//!   ported. Their effect on migrated data is NOT guaranteed by this backend: the
//!   migrator (outside this repository) must have run them on the source before the
//!   copy; the list is in `CONSOLIDATION.md`.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::task::ecritures;
use super::trigger::{
    booleen, charger, cle_reservee, creer, entier, facultative, flottant, indexer, json_facultatif,
    refus_reserve, relire, requise, supprimer, texte, uuid_requis, Fiche, Lecteur,
};
use super::{Ecriture, ObrainGraphStore, RapportOuverture};
use crate::chat::provider::cognitive::decision::{CognitiveDecision, DecisionOutcome};
use crate::chat::provider::cognitive::store::{ArmKey, ArmObservation, ArmStats, DecisionFilter};
use crate::graph::models::{
    AnalysisProfile, FabricFileAnalyticsUpdate, FileAnalyticsUpdate, FunctionAnalyticsUpdate,
    TopologyRule, TopologyRuleType, TopologySeverity, TopologyViolation,
};
use crate::neo4j::models::*;
use crate::neo4j::traits::AnalyticsStore;
use crate::profile::{UserProfile, WorksOnRelation};

const PROFIL: &str = "AnalysisProfile";
const REGLE: &str = "TopologyRule";
const UTILISATEUR: &str = "UserProfile";
const BRAS: &str = "RoutingArm";
const DECISION: &str = "RoutingDecision";
const TRAVAILLE_SUR: &str = "WORKS_ON";
const MAJ: &str = "_maj";

// ============================================================================
// Encoding and decoding
// ============================================================================

/// An arm's or a decision's instant: fixed width, so string order is time order.
fn stamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn instant(d: DateTime<Utc>) -> Value {
    texte(d.to_rfc3339_opts(SecondsFormat::Nanos, true))
}

fn date_requise(get: Lecteur, cle: &str, quoi: &str) -> Result<DateTime<Utc>> {
    super::date_propriete(get, cle)?.ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}`"))
}

fn proprietes_profil(p: &AnalysisProfile) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(p.id.clone())),
        ("name", texte(p.name.clone())),
        (
            "description",
            texte(p.description.clone().unwrap_or_default()),
        ),
        (
            "project_id",
            texte(p.project_id.clone().unwrap_or_default()),
        ),
        (
            "edge_weights_json",
            texte(serde_json::to_string(&p.edge_weights)?),
        ),
        (
            "fusion_weights_json",
            texte(serde_json::to_string(&p.fusion_weights)?),
        ),
        ("is_builtin", Value::from(p.is_builtin)),
    ])
}

fn lire_profil(get: Lecteur) -> Result<AnalysisProfile> {
    let id = requise(get, "id", "nœud AnalysisProfile")?;
    let quoi = format!("profil d'analyse {id}");
    let quoi = quoi.as_str();
    Ok(AnalysisProfile {
        name: requise(get, "name", quoi)?,
        description: facultative(get, "description", quoi)?,
        project_id: facultative(get, "project_id", quoi)?,
        edge_weights: json_facultatif(get, "edge_weights_json", quoi)?.unwrap_or_default(),
        fusion_weights: json_facultatif(get, "fusion_weights_json", quoi)?.unwrap_or_default(),
        is_builtin: booleen(get, "is_builtin", quoi)?.unwrap_or(false),
        id,
    })
}

fn proprietes_regle(r: &TopologyRule) -> Vec<(&'static str, Value)> {
    let mut v = vec![
        ("id", texte(r.id.clone())),
        ("project_id", texte(r.project_id.clone())),
        ("rule_type", texte(r.rule_type.to_string())),
        ("source_pattern", texte(r.source_pattern.clone())),
        (
            "target_pattern",
            texte(r.target_pattern.clone().unwrap_or_default()),
        ),
        ("severity", texte(r.severity.to_string())),
        ("description", texte(r.description.clone())),
    ];
    if let Some(t) = r.threshold {
        v.push(("threshold", Value::from(i64::from(t))));
    }
    v
}

fn lire_regle(get: Lecteur) -> Result<TopologyRule> {
    let id = requise(get, "id", "nœud TopologyRule")?;
    let quoi = format!("règle de topologie {id}");
    let quoi = quoi.as_str();
    let type_brut = requise(get, "rule_type", quoi)?;
    let severite_brute = requise(get, "severity", quoi)?;
    Ok(TopologyRule {
        project_id: requise(get, "project_id", quoi)?,
        rule_type: TopologyRuleType::from_str_loose(&type_brut)
            .ok_or_else(|| anyhow!("{quoi} : `rule_type` inconnu « {type_brut} »"))?,
        source_pattern: requise(get, "source_pattern", quoi)?,
        target_pattern: facultative(get, "target_pattern", quoi)?,
        threshold: entier(get, "threshold", quoi)?
            .map(|t| {
                u32::try_from(t).map_err(|_| anyhow!("{quoi} : `threshold` hors bornes ({t})"))
            })
            .transpose()?,
        severity: TopologySeverity::from_str_loose(&severite_brute)
            .ok_or_else(|| anyhow!("{quoi} : `severity` inconnue « {severite_brute} »"))?,
        description: requise(get, "description", quoi)?,
        id,
    })
}

fn proprietes_utilisateur(u: &UserProfile) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(u.id.to_string())),
        ("user_id", texte(u.user_id.clone())),
        ("verbosity", Value::from(u.verbosity)),
        ("commit_style", Value::from(u.commit_style)),
        ("language", texte(u.language.clone())),
        ("expertise_level", Value::from(u.expertise_level)),
        (
            "interaction_count",
            Value::from(i64::try_from(u.interaction_count).map_err(|_| {
                anyhow!(
                    "`interaction_count` {} dépasse i64 : refusé",
                    u.interaction_count
                )
            })?),
        ),
        ("created_at", instant(u.created_at)),
        ("updated_at", instant(u.updated_at)),
    ])
}

fn lire_utilisateur(get: Lecteur) -> Result<UserProfile> {
    let user_id = requise(get, "user_id", "nœud UserProfile")?;
    let quoi = format!("profil utilisateur {user_id}");
    let quoi = quoi.as_str();
    // The defaults are Neo4j's own for a node written before a field existed.
    let nombre = |cle: &str| flottant(get, cle, quoi).map(|v| v.unwrap_or(0.5));
    Ok(UserProfile {
        id: uuid_requis(get, "id", quoi)?,
        verbosity: nombre("verbosity")?,
        commit_style: nombre("commit_style")?,
        language: facultative(get, "language", quoi)?.unwrap_or_else(|| "en".to_string()),
        expertise_level: nombre("expertise_level")?,
        interaction_count: match entier(get, "interaction_count", quoi)? {
            Some(n) => u64::try_from(n.max(0)).unwrap_or(u64::MAX),
            None => 0,
        },
        created_at: date_requise(get, "created_at", quoi)?,
        updated_at: date_requise(get, "updated_at", quoi)?,
        user_id,
    })
}

fn proprietes_bras(a: &ArmStats) -> Result<Vec<(&'static str, Value)>> {
    let i = |n: u64, cle: &str| {
        i64::try_from(n).map_err(|_| anyhow!("bras : `{cle}` {n} dépasse i64 : refusé"))
    };
    Ok(vec![
        ("class", texte(a.key.class.clone())),
        ("provider_id", texte(a.key.provider_id.clone())),
        ("model", texte(a.key.model.clone())),
        ("alpha", Value::from(a.alpha)),
        ("beta", Value::from(a.beta)),
        ("n", Value::from(i(a.n, "n")?)),
        ("cost_sum_usd", Value::from(a.cost_sum_usd)),
        ("cost_n", Value::from(i(a.cost_n, "cost_n")?)),
        (
            "latency_sum_ms",
            Value::from(i(a.latency_sum_ms, "latency_sum_ms")?),
        ),
        ("latency_n", Value::from(i(a.latency_n, "latency_n")?)),
        ("updated_at", texte(stamp(a.updated_at))),
    ])
}

fn lire_bras(get: Lecteur) -> Result<ArmStats> {
    let classe = requise(get, "class", "nœud RoutingArm")?;
    let fournisseur = requise(get, "provider_id", "nœud RoutingArm")?;
    let modele = requise(get, "model", "nœud RoutingArm")?;
    let quoi = format!("bras {classe}|{fournisseur}|{modele}");
    let quoi = quoi.as_str();
    let compte = |cle: &str| -> Result<u64> {
        let n = entier(get, cle, quoi)?.ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}`"))?;
        u64::try_from(n).map_err(|_| anyhow!("{quoi} : `{cle}` négatif ({n})"))
    };
    let reel = |cle: &str| -> Result<f64> {
        flottant(get, cle, quoi)?.ok_or_else(|| anyhow!("{quoi} sans propriété `{cle}`"))
    };
    Ok(ArmStats {
        alpha: reel("alpha")?,
        beta: reel("beta")?,
        n: compte("n")?,
        cost_sum_usd: reel("cost_sum_usd")?,
        cost_n: compte("cost_n")?,
        latency_sum_ms: compte("latency_sum_ms")?,
        latency_n: compte("latency_n")?,
        updated_at: date_requise(get, "updated_at", quoi)?,
        key: ArmKey::new(classe, &fournisseur, &modele),
    })
}

/// What the index keeps of a decision: enough to filter, order and find it. The
/// document (`json`) is DECODED at open (a node whose document does not decode is
/// quarantined by name there, never served), but only the page is decoded again to
/// be served.
#[derive(Clone)]
pub(super) struct DecisionLegere {
    id: Uuid,
    at: String,
    projet: String,
    session: String,
}

fn lire_decision_legere(get: Lecteur) -> Result<DecisionLegere> {
    let id = uuid_requis(get, "id", "nœud RoutingDecision")?;
    let quoi = format!("décision de routage {id}");
    let quoi = quoi.as_str();
    // Decoded here, once: an unreadable document is a quarantine at open (Neo4j skips
    // it at each read and logs), never a page that fails as a whole.
    let brut = requise(get, "json", quoi)?;
    let doc: CognitiveDecision = serde_json::from_str(&brut)
        .map_err(|e| anyhow!("{quoi} : document `json` illisible : {e}"))?;
    if doc.id != id {
        bail!("{quoi} : le document porte l'id {}", doc.id);
    }
    Ok(DecisionLegere {
        at: requise(get, "at", quoi)?,
        projet: chaine_ou_vide(get, "project_slug", quoi)?,
        session: chaine_ou_vide(get, "session_id", quoi)?,
        id,
    })
}

fn chaine_ou_vide(get: Lecteur, cle: &str, quoi: &str) -> Result<String> {
    Ok(super::trigger::chaine(get, cle, quoi)?.unwrap_or_default())
}

fn proprietes_decision(d: &CognitiveDecision) -> Result<Vec<(&'static str, Value)>> {
    let id_ou_vide = |v: Option<Uuid>| texte(v.map(|u| u.to_string()).unwrap_or_default());
    let (fournisseur, modele) = d
        .chosen
        .as_ref()
        .map(|p| (p.provider_id.clone(), p.model.clone()))
        .unwrap_or_default();
    Ok(vec![
        ("id", texte(d.id.to_string())),
        ("at", texte(stamp(d.at))),
        ("json", texte(serde_json::to_string(d)?)),
        (
            "project_slug",
            texte(d.signature.project_slug.clone().unwrap_or_default()),
        ),
        ("session_id", id_ou_vide(d.session_id)),
        ("task_id", id_ou_vide(d.task_id)),
        ("run_id", id_ou_vide(d.run_id)),
        ("chosen_provider", texte(fournisseur)),
        ("chosen_model", texte(modele)),
        ("applied", Value::from(d.applied)),
    ])
}

// ----- the update marker -----------------------------------------------------

fn valeur_en_json(v: &Value) -> Result<serde_json::Value> {
    Ok(match v {
        Value::String(s) => serde_json::json!({"s": s.to_string()}),
        Value::Int64(i) => serde_json::json!({"i": i}),
        // The bits: a float must come back identical, not as printed.
        Value::Float64(f) => serde_json::json!({"f": f.to_bits().to_string()}),
        Value::Bool(b) => serde_json::json!({"b": b}),
        autre => bail!("marqueur de mise à jour : type non pris en charge ({autre:?})"),
    })
}

fn json_en_valeur(j: &serde_json::Value) -> Result<Value> {
    let objet = j
        .as_object()
        .filter(|o| o.len() == 1)
        .ok_or_else(|| anyhow!("marqueur de mise à jour : valeur mal formée"))?;
    let (tag, v) = objet.iter().next().expect("one entry");
    Ok(match (tag.as_str(), v) {
        ("s", serde_json::Value::String(s)) => Value::from(s.clone()),
        ("i", serde_json::Value::Number(n)) => Value::from(
            n.as_i64()
                .ok_or_else(|| anyhow!("marqueur de mise à jour : entier illisible"))?,
        ),
        ("f", serde_json::Value::String(bits)) => {
            Value::from(f64::from_bits(bits.parse::<u64>().map_err(|e| {
                anyhow!("marqueur de mise à jour : flottant illisible : {e}")
            })?))
        }
        ("b", serde_json::Value::Bool(b)) => Value::from(*b),
        _ => bail!("marqueur de mise à jour : valeur de type inconnu"),
    })
}

fn encoder_maj(champs: &[(&'static str, Value)]) -> Result<String> {
    let mut objet = serde_json::Map::new();
    for (cle, v) in champs {
        objet.insert((*cle).to_string(), valeur_en_json(v)?);
    }
    Ok(serde_json::Value::Object(objet).to_string())
}

fn decoder_maj(brut: &str) -> Result<Vec<(String, Value)>> {
    let objet: serde_json::Map<String, serde_json::Value> = serde_json::from_str(brut)
        .map_err(|e| anyhow!("marqueur de mise à jour illisible : {e}"))?;
    objet
        .into_iter()
        .map(|(cle, j)| Ok((cle, json_en_valeur(&j)?)))
        .collect()
}

/// Rewrites a node's properties atomically in effect: the whole new state on the
/// node first, the properties, the marker removed last. Inside [`ecritures`].
fn poser_par_marqueur(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    champs: Vec<(&'static str, Value)>,
) -> Result<()> {
    let marqueur = encoder_maj(&champs)?;
    w.etape()?;
    w.poser(noeud, MAJ, texte(marqueur));
    for (cle, v) in champs {
        w.etape()?;
        w.poser(noeud, cle, v);
    }
    w.etape()?;
    w.db.remove_node_property(noeud, MAJ);
    Ok(())
}

/// Opening: finishes every update cut after its marker. A marker that does not
/// read is left in place and its node stays unreadable (`lire_*` refuse a node that
/// still carries one).
fn achever_les_mises_a_jour(db: &ObrainDB, label: &str) -> usize {
    let mut achevees = 0;
    for noeud in db.store().nodes_by_label(label) {
        let Some(n) = db.get_node(noeud) else {
            continue;
        };
        let Some(brut) = n
            .get_property(MAJ)
            .and_then(|v| v.as_str().map(str::to_string))
        else {
            continue;
        };
        let Ok(valeurs) = decoder_maj(&brut) else {
            continue;
        };
        for (cle, v) in valeurs {
            // hors-poser: recovery at open, outside a write closure (no `Ecriture`); open flushes after
            db.set_node_property(noeud, &cle, v);
        }
        db.remove_node_property(noeud, MAJ);
        achevees += 1;
    }
    achevees
}

/// A node still carrying an unreadable marker is not served.
fn sans_marqueur<T>(
    get: Lecteur,
    quoi: &str,
    lire: impl FnOnce(Lecteur) -> Result<T>,
) -> Result<T> {
    if get(MAJ).is_some() {
        bail!("{quoi} : mise à jour interrompue dont le marqueur `{MAJ}` est illisible");
    }
    lire(get)
}

// ============================================================================
// The index
// ============================================================================

/// A `WORKS_ON` edge: its engine id, and what it says.
#[derive(Clone)]
struct Relation {
    arete: EdgeId,
    frequence: u64,
    derniere: DateTime<Utc>,
}

/// Profiles, rules, routing, decoded by key, rebuilt at open.
#[derive(Default)]
pub(super) struct IndexAnalyse {
    profils: HashMap<String, Fiche<AnalysisProfile>>,
    regles: HashMap<String, Fiche<TopologyRule>>,
    utilisateurs: HashMap<String, Fiche<UserProfile>>,
    /// user → (project → edge).
    relations: HashMap<String, HashMap<Uuid, Relation>>,
    bras: HashMap<ArmKey, Fiche<ArmStats>>,
    decisions: HashMap<Uuid, Fiche<DecisionLegere>>,
    /// Keys of the quarantined nodes ([`cle_reservee`]): never taken again.
    reserves: HashSet<String>,
    /// label → the scope of each quarantined node (`None`: unreadable scope). A list
    /// that could miss one of them refuses, as the other ported domains do
    /// (`sharing.rs`'s `list_tombstones`, `task.rs`'s partial plans): the project of
    /// a profile (`""` = global), the project of a rule, the class of an arm.
    quarantaines: HashMap<&'static str, Vec<Option<String>>>,
    /// (user, project) of the `WORKS_ON` edges quarantined at open (`None`: the
    /// target project is unreadable, every relation of the user is in doubt).
    relations_reservees: HashSet<(String, Option<Uuid>)>,
}

/// Records the scope of a node that does not decode, then gives the error back.
fn noter_quarantaine<T>(
    notes: &std::cell::RefCell<Vec<Option<String>>>,
    get: Lecteur,
    portee: &str,
    lu: Result<T>,
) -> Result<T> {
    if lu.is_err() {
        notes
            .borrow_mut()
            .push(get(portee).and_then(|v| v.as_str().map(str::to_string)));
    }
    lu
}

impl IndexAnalyse {
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let mut index = Self::default();
        for label in [PROFIL, UTILISATEUR, BRAS, DECISION] {
            rapport.mises_a_jour_achevees += achever_les_mises_a_jour(db, label);
        }
        // The scopes of the nodes each label quarantines (filled by `noter_quarantaine`).
        let q_profils = std::cell::RefCell::<Vec<Option<String>>>::default();
        let q_regles = std::cell::RefCell::<Vec<Option<String>>>::default();
        let q_bras = std::cell::RefCell::<Vec<Option<String>>>::default();
        index.profils = indexer(
            charger(db, PROFIL, rapport, &mut index.reserves, &|g| {
                noter_quarantaine(
                    &q_profils,
                    g,
                    "project_id",
                    sans_marqueur(g, PROFIL, lire_profil),
                )
            }),
            PROFIL,
            rapport,
            |p| p.id.clone(),
        );
        index.regles = indexer(
            charger(db, REGLE, rapport, &mut index.reserves, &|g| {
                noter_quarantaine(&q_regles, g, "project_id", lire_regle(g))
            }),
            REGLE,
            rapport,
            |r| r.id.clone(),
        );
        index.utilisateurs = indexer(
            charger(db, UTILISATEUR, rapport, &mut index.reserves, &|g| {
                sans_marqueur(g, UTILISATEUR, lire_utilisateur)
            }),
            UTILISATEUR,
            rapport,
            |u| u.user_id.clone(),
        );
        // Keyed by the ArmKey itself (a joined string could collide on a `|` in a name).
        for f in charger(db, BRAS, rapport, &mut index.reserves, &|g| {
            noter_quarantaine(&q_bras, g, "class", sans_marqueur(g, BRAS, lire_bras))
        }) {
            match index.bras.entry(f.valeur.key.clone()) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud {BRAS} {:?} : clé {} déjà prise",
                        f.noeud,
                        f.valeur.key.scheduler_key()
                    ))
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(f);
                }
            }
        }
        index.decisions = indexer(
            charger(db, DECISION, rapport, &mut index.reserves, &|g| {
                sans_marqueur(g, DECISION, lire_decision_legere)
            }),
            DECISION,
            rapport,
            |d| d.id,
        );
        index.quarantaines.insert(PROFIL, q_profils.into_inner());
        index.quarantaines.insert(REGLE, q_regles.into_inner());
        index.quarantaines.insert(BRAS, q_bras.into_inner());
        index.relier_les_utilisateurs(db, rapport);
        index
    }

    /// Refuses a list of `label` that could miss a quarantined node: one whose scope
    /// is in `portees`, or unreadable.
    fn garde_liste(&self, label: &str, portees: &[&str]) -> Result<()> {
        let suspects = self
            .quarantaines
            .get(label)
            .into_iter()
            .flatten()
            .filter(|p| p.as_deref().is_none_or(|p| portees.contains(&p)))
            .count();
        if suspects > 0 {
            bail!(
                "{suspects} nœud(s) {label} de cette portée sont en quarantaine (illisibles à \
                 l'ouverture) : la liste serait partielle, elle est refusée"
            );
        }
        Ok(())
    }

    /// Refuses the `WORKS_ON` of a user when one of its edges was quarantined.
    fn garde_relation(&self, user: &str, projet: Option<Uuid>) -> Result<()> {
        let douteuse = self.relations_reservees.contains(&(user.to_string(), None))
            || projet.is_some_and(|p| {
                self.relations_reservees
                    .contains(&(user.to_string(), Some(p)))
            })
            || (projet.is_none() && self.relations_reservees.iter().any(|(u, _)| u == user));
        if douteuse {
            bail!(
                "une arête WORKS_ON de « {user} » est en quarantaine (illisible à l'ouverture) : \
                 elle n'est ni lue, ni réécrite (sa fréquence serait perdue)"
            );
        }
        Ok(())
    }

    /// The `WORKS_ON` edges of the user profiles that were read. Two edges for one
    /// (user, project) are an update cut between its creation and its deletion:
    /// the one with the highest frequency (then the latest date) is kept.
    fn relier_les_utilisateurs(&mut self, db: &ObrainDB, rapport: &mut RapportOuverture) {
        let store = db.store();
        let mut reservees = Vec::new();
        for f in self.utilisateurs.values() {
            let mut par_projet: HashMap<Uuid, Vec<Relation>> = HashMap::new();
            for (cible, arete) in store.edges_from(f.noeud, Direction::Outgoing) {
                let Some(e) = db.get_edge(arete) else {
                    continue;
                };
                if &*e.edge_type != TRAVAILLE_SUR {
                    continue;
                }
                let Some(projet) = db
                    .get_node(cible)
                    .and_then(|n| {
                        n.get_property("id")
                            .and_then(|v| v.as_str().map(str::to_string))
                    })
                    .and_then(|s| s.parse::<Uuid>().ok())
                else {
                    rapport.en_quarantaine.push(format!(
                        "arête WORKS_ON {arete:?} de {} : cible sans id lisible",
                        f.valeur.user_id
                    ));
                    reservees.push((f.valeur.user_id.clone(), None));
                    continue;
                };
                let get = |cle: &str| {
                    e.get_property(cle)
                        .filter(|v| !matches!(v, Value::Null))
                        .cloned()
                };
                // Neo4j reads a missing frequency as 1; an unreadable date is a quarantine.
                let lue = (|| -> Result<Relation> {
                    Ok(Relation {
                        arete,
                        frequence: match entier(&get, "frequency", "arête WORKS_ON")? {
                            Some(n) => u64::try_from(n.max(0)).unwrap_or(u64::MAX),
                            None => 1,
                        },
                        derniere: date_requise(&get, "last_active", "arête WORKS_ON")?,
                    })
                })();
                match lue {
                    Ok(r) => par_projet.entry(projet).or_default().push(r),
                    Err(err) => {
                        rapport.en_quarantaine.push(format!(
                            "arête WORKS_ON {arete:?} de {} : {err}",
                            f.valeur.user_id
                        ));
                        reservees.push((f.valeur.user_id.clone(), Some(projet)));
                    }
                }
            }
            for (projet, mut rels) in par_projet {
                rels.sort_by(|a, b| {
                    (b.frequence, b.derniere, b.arete).cmp(&(a.frequence, a.derniere, a.arete))
                });
                let garde = rels.remove(0);
                for autre in rels {
                    db.delete_edge(autre.arete);
                    rapport.mises_a_jour_achevees += 1;
                }
                self.relations
                    .entry(f.valeur.user_id.clone())
                    .or_default()
                    .insert(projet, garde);
            }
        }
        self.relations_reservees.extend(reservees);
    }

    /// Refuses, by name, a key a quarantined node holds.
    fn garde(&self, label: &str, cle: &[&str]) -> Result<()> {
        if self.reserves.contains(&cle_reservee(label, cle)) {
            return Err(refus_reserve(&format!(
                "la clé {label} « {} »",
                cle.join("|")
            )));
        }
        Ok(())
    }
}

// ============================================================================
// The writes and reads
// ============================================================================

/// Rewrites properties by marker, reads the node back, makes it durable.
fn reecrire_par_marqueur<T>(
    w: &mut Ecriture<'_>,
    noeud: NodeId,
    champs: Vec<(&'static str, Value)>,
    decoder: &dyn Fn(Lecteur) -> Result<T>,
) -> Result<Fiche<T>> {
    ecritures(w, |w| poser_par_marqueur(w, noeud, champs))?;
    let fiche = w.apres_ecriture(relire(w.db, noeud, decoder))?;
    w.rendre_durable()?;
    Ok(fiche)
}

fn lire_json_decision(db: &ObrainDB, noeud: NodeId, id: Uuid) -> Result<CognitiveDecision> {
    let brut = db
        .get_node(noeud)
        .and_then(|n| {
            n.get_property("json")
                .and_then(|v| v.as_str().map(str::to_string))
        })
        .ok_or_else(|| anyhow!("décision de routage {id} : document absent"))?;
    serde_json::from_str(&brut)
        .with_context(|| format!("décision de routage {id} : document illisible"))
}

impl ObrainGraphStore {
    async fn profil_creer(&self, profile: &AnalysisProfile) -> Result<()> {
        let p = profile.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    etat.analyse.garde(PROFIL, &[&p.id])?;
                    if let Some(pid) = p.project_id.as_deref().filter(|s| !s.is_empty()) {
                        let existe = pid
                            .parse::<Uuid>()
                            .ok()
                            .is_some_and(|u| etat.projets.noeud_de(u).is_some());
                        if !existe {
                            bail!(
                                "le projet {pid} n'existe pas : le profil {} n'est pas créé (Neo4j \
                                 ne créerait rien et répondrait Ok ; le backend obrain le dit)",
                                p.id
                            );
                        }
                    }
                }
                let proprietes = proprietes_profil(&p)?;
                let existant = w.index()?.analyse.profils.get(&p.id).map(|f| f.noeud);
                let fiche = match existant {
                    Some(n) => reecrire_par_marqueur(w, n, proprietes, &lire_profil)?,
                    None => ecritures(w, |w| creer(w, PROFIL, proprietes, &lire_profil))?,
                };
                w.index_mut()?.analyse.profils.insert(p.id.clone(), fiche);
                Ok(())
            },
        )
        .await
    }

    async fn profil_supprimer(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some((noeud, builtin, nom)) = w
                    .index()?
                    .analyse
                    .profils
                    .get(&id)
                    .map(|f| (f.noeud, f.valeur.is_builtin, f.valeur.name.clone()))
                else {
                    return Ok(());
                };
                if builtin {
                    bail!("Cannot delete built-in profile '{nom}'");
                }
                ecritures(w, |w| supprimer(w, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.analyse.profils.remove(&id);
                Ok(())
            },
        )
        .await
    }

    async fn regle_creer(&self, rule: &TopologyRule) -> Result<()> {
        let r = rule.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    etat.analyse.garde(REGLE, &[&r.id])?;
                    if etat.analyse.regles.contains_key(&r.id) {
                        bail!(
                            "la règle de topologie {} existe déjà (unicité de TopologyRule.id, comme \
                             la contrainte Neo4j `topology_rule_id`)",
                            r.id
                        );
                    }
                }
                let fiche = ecritures(w, |w| creer(w, REGLE, proprietes_regle(&r), &lire_regle))?;
                w.index_mut()?.analyse.regles.insert(r.id.clone(), fiche);
                Ok(())
            },
        )
        .await
    }

    async fn regle_supprimer(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(noeud) = w.index()?.analyse.regles.get(&id).map(|f| f.noeud) else {
                    return Ok(());
                };
                ecritures(w, |w| supprimer(w, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.analyse.regles.remove(&id);
                Ok(())
            },
        )
        .await
    }

    async fn utilisateur_creer_ou_lire(&self, user_id: &str) -> Result<UserProfile> {
        let uid = user_id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                {
                    let etat = w.index()?;
                    etat.analyse.garde(UTILISATEUR, &["user_id", &uid])?;
                    if let Some(f) = etat.analyse.utilisateurs.get(&uid) {
                        return Ok(f.valeur.clone());
                    }
                }
                let profil = UserProfile::new(uid.clone());
                let fiche = ecritures(w, |w| {
                    creer(w, UTILISATEUR, proprietes_utilisateur(&profil)?, &|g| {
                        sans_marqueur(g, UTILISATEUR, lire_utilisateur)
                    })
                })?;
                let rendu = fiche.valeur.clone();
                w.index_mut()?.analyse.utilisateurs.insert(uid, fiche);
                Ok(rendu)
            },
        )
        .await
    }

    async fn utilisateur_maj(&self, profile: &UserProfile) -> Result<()> {
        let p = profile.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    etat.analyse.garde(UTILISATEUR, &["user_id", &p.user_id])?;
                    match etat.analyse.utilisateurs.get(&p.user_id) {
                        Some(f) => f.noeud,
                        None => bail!(
                            "le profil utilisateur « {} » n'existe pas : rien n'est écrit (Neo4j \
                             ne ferait rien et répondrait Ok ; le backend obrain le dit)",
                            p.user_id
                        ),
                    }
                };
                let champs = vec![
                    ("verbosity", Value::from(p.verbosity)),
                    ("commit_style", Value::from(p.commit_style)),
                    ("language", texte(p.language.clone())),
                    ("expertise_level", Value::from(p.expertise_level)),
                    (
                        "interaction_count",
                        Value::from(i64::try_from(p.interaction_count).map_err(|_| {
                            anyhow!(
                                "`interaction_count` {} dépasse i64 : refusé",
                                p.interaction_count
                            )
                        })?),
                    ),
                    ("updated_at", instant(Utc::now())),
                ];
                let fiche = reecrire_par_marqueur(w, noeud, champs, &|g| {
                    sans_marqueur(g, UTILISATEUR, lire_utilisateur)
                })?;
                w.index_mut()?
                    .analyse
                    .utilisateurs
                    .insert(p.user_id.clone(), fiche);
                Ok(())
            },
        )
        .await
    }

    async fn travaille_sur(&self, user_id: &str, project_id: Uuid) -> Result<()> {
        let uid = user_id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (utilisateur, projet, ancienne) = {
                    let etat = w.index()?;
                    etat.analyse.garde(UTILISATEUR, &["user_id", &uid])?;
                    etat.analyse.garde_relation(&uid, Some(project_id))?;
                    // Neo4j's `MATCH … MATCH … MERGE`: an absent user or project writes
                    // nothing and answers Ok. Kept, and logged: the profile aggregator
                    // (`profile/aggregator.rs`) upserts every project of a flush with `?`,
                    // so a refusal here would drop the relations of the projects after it.
                    let Some(u) = etat.analyse.utilisateurs.get(&uid) else {
                        tracing::warn!(user = %uid, "upsert_works_on : profil utilisateur absent, rien n'est écrit (comme Neo4j)");
                        return Ok(());
                    };
                    let Some(p) = etat.projets.noeud_de(project_id) else {
                        tracing::warn!(user = %uid, project = %project_id, "upsert_works_on : projet absent, rien n'est écrit (comme Neo4j)");
                        return Ok(());
                    };
                    let ancienne = etat
                        .analyse
                        .relations
                        .get(&uid)
                        .and_then(|m| m.get(&project_id))
                        .cloned();
                    (u.noeud, p, ancienne)
                };
                let maintenant = Utc::now();
                let frequence = ancienne
                    .as_ref()
                    .map_or(1, |r| r.frequence.saturating_add(1));
                let valeur = i64::try_from(frequence).unwrap_or(i64::MAX);
                // The new edge, frequency and date in ONE record, then the old one goes.
                let arete = ecritures(w, |w| {
                    w.etape()?;
                    // hors-poser: an edge with a small integer and a date under 256 bytes: never out of the WAL
                    let nouvelle = w.db.create_edge_with_props(
                        utilisateur,
                        projet,
                        TRAVAILLE_SUR,
                        [
                            ("frequency", Value::from(valeur)),
                            ("last_active", instant(maintenant)),
                        ],
                    );
                    if let Some(a) = &ancienne {
                        w.etape()?;
                        w.db.delete_edge(a.arete);
                    }
                    Ok(nouvelle)
                })?;
                w.rendre_durable()?;
                w.index_mut()?
                    .analyse
                    .relations
                    .entry(uid)
                    .or_default()
                    .insert(
                        project_id,
                        Relation {
                            arete,
                            frequence,
                            derniere: maintenant,
                        },
                    );
                Ok(())
            },
        )
        .await
    }

    async fn bras_observer(&self, key: &ArmKey, observation: &ArmObservation) -> Result<ArmStats> {
        let (k, o) = (key.clone(), observation.clone());
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let existant = {
                    let etat = w.index()?;
                    etat.analyse
                        .garde(BRAS, &["bras", &k.class, &k.provider_id, &k.model])?;
                    etat.analyse
                        .bras
                        .get(&k)
                        .map(|f| (f.noeud, f.valeur.clone()))
                };
                // Neo4j's statement, term for term: fractional reward, decay before the counts.
                let reward = o.reward.clamp(0.0, 1.0);
                let decay = o.decay.clamp(0.0, 1.0);
                let base = existant
                    .as_ref()
                    .map(|(_, a)| a.clone())
                    .unwrap_or_else(|| ArmStats::fresh(k.clone(), 1.0, 1.0));
                let suivant = ArmStats {
                    key: k.clone(),
                    alpha: base.alpha * decay + reward,
                    beta: base.beta * decay + (1.0 - reward),
                    n: base.n.saturating_add(1),
                    cost_sum_usd: base.cost_sum_usd + o.cost_usd.unwrap_or(0.0),
                    cost_n: base.cost_n + u64::from(o.cost_usd.is_some()),
                    latency_sum_ms: base
                        .latency_sum_ms
                        .saturating_add(o.latency_ms.unwrap_or(0)),
                    latency_n: base.latency_n + u64::from(o.latency_ms.is_some()),
                    updated_at: Utc::now(),
                };
                if !suivant.alpha.is_finite() || !suivant.beta.is_finite() {
                    bail!(
                        "observation non finie (reward {}, decay {}) : refusée",
                        o.reward,
                        o.decay
                    );
                }
                let proprietes = proprietes_bras(&suivant)?;
                let fiche = match existant {
                    Some((noeud, _)) => {
                        // Only the counters move: the key never changes.
                        let champs = proprietes.into_iter().skip(3).collect();
                        reecrire_par_marqueur(w, noeud, champs, &|g| {
                            sans_marqueur(g, BRAS, lire_bras)
                        })?
                    }
                    None => ecritures(w, |w| {
                        creer(w, BRAS, proprietes, &|g| sans_marqueur(g, BRAS, lire_bras))
                    })?,
                };
                let rendu = fiche.valeur.clone();
                w.index_mut()?.analyse.bras.insert(k, fiche);
                Ok(rendu)
            },
        )
        .await
    }

    async fn decision_poser(&self, decision: &CognitiveDecision) -> Result<()> {
        let d = decision.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let existant = {
                    let etat = w.index()?;
                    etat.analyse.garde(DECISION, &[&d.id.to_string()])?;
                    etat.analyse.decisions.get(&d.id).map(|f| f.noeud)
                };
                let proprietes = proprietes_decision(&d)?;
                let fiche = match existant {
                    // `id` is the key: it does not move.
                    Some(n) => reecrire_par_marqueur(
                        w,
                        n,
                        proprietes.into_iter().skip(1).collect(),
                        &lire_decision_legere,
                    )?,
                    None => {
                        ecritures(w, |w| creer(w, DECISION, proprietes, &lire_decision_legere))?
                    }
                };
                w.index_mut()?.analyse.decisions.insert(d.id, fiche);
                Ok(())
            },
        )
        .await
    }

    async fn decision_issue(&self, id: Uuid, outcome: DecisionOutcome) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    etat.analyse.garde(DECISION, &[&id.to_string()])?;
                    match etat.analyse.decisions.get(&id) {
                        Some(f) => f.noeud,
                        None => return Ok(false),
                    }
                };
                let mut decision = lire_json_decision(w.db, noeud, id)?;
                decision.outcome = Some(outcome);
                let json = serde_json::to_string(&decision)?;
                // ONE property: atomic in itself.
                ecritures(w, |w| {
                    w.etape()?;
                    w.poser(noeud, "json", texte(json));
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(true)
            },
        )
        .await
    }

    async fn frustration_ecrire(&self, task_id: Uuid, delta: f64, hausse: bool) -> Result<f64> {
        let resultat = std::sync::Arc::new(std::sync::Mutex::new(0.0f64));
        let sortie = resultat.clone();
        self.tache_ecrire(task_id, move |t| {
            // Neo4j: the increment is capped at 1.0, the decrement floored at 0.0, and
            // nothing else bounds either.
            let nouveau = if hausse {
                (t.frustration_score + delta).min(1.0)
            } else {
                (t.frustration_score - delta).max(0.0)
            };
            if nouveau.is_nan() {
                bail!("frustration non numérique (delta {delta}) : refusée");
            }
            *sortie.lock().expect("sortie") = nouveau;
            Ok(Some(vec![(
                "frustration_score",
                Some(Value::from(nouveau)),
            )]))
        })
        .await?;
        let v = *resultat.lock().expect("sortie");
        Ok(v)
    }
}

pub(super) const REFUSEES: &[&str] = &[
    "AnalyticsStore::set_file_embedding",
    "AnalyticsStore::set_function_embedding",
    "AnalyticsStore::vector_search_files",
    "AnalyticsStore::vector_search_functions",
    "AnalyticsStore::shadow_trace_closures",
    "AnalyticsStore::compute_homeostasis",
    "AnalyticsStore::compute_structural_drift",
    "AnalyticsStore::get_project_import_edges",
    "AnalyticsStore::get_project_call_edges",
    "AnalyticsStore::get_project_extends_edges",
    "AnalyticsStore::get_project_implements_edges",
    "AnalyticsStore::batch_update_file_analytics",
    "AnalyticsStore::batch_update_function_analytics",
    "AnalyticsStore::batch_update_fabric_file_analytics",
    "AnalyticsStore::batch_update_structural_dna",
    "AnalyticsStore::write_predicted_links",
    "AnalyticsStore::get_project_structural_dna",
    "AnalyticsStore::batch_update_structural_fingerprints",
    "AnalyticsStore::get_project_structural_fingerprints",
    "AnalyticsStore::get_project_file_signals",
    "AnalyticsStore::get_file_structural_dna",
    "AnalyticsStore::get_import_edges_among",
    "AnalyticsStore::get_call_edges_among",
    "AnalyticsStore::get_project_synapse_edges",
    "AnalyticsStore::get_neural_metrics",
    "AnalyticsStore::compute_churn_scores",
    "AnalyticsStore::batch_update_churn_scores",
    "AnalyticsStore::get_top_hotspots",
    "AnalyticsStore::compute_knowledge_density",
    "AnalyticsStore::batch_update_knowledge_density",
    "AnalyticsStore::get_top_knowledge_gaps",
    "AnalyticsStore::compute_risk_scores",
    "AnalyticsStore::batch_update_risk_scores",
    "AnalyticsStore::get_risk_summary",
    "AnalyticsStore::batch_upsert_processes",
    "AnalyticsStore::batch_create_step_relationships",
    "AnalyticsStore::delete_project_processes",
    "AnalyticsStore::auto_scope_to_feature_graphs",
    "AnalyticsStore::compute_adaptive_thresholds",
    "AnalyticsStore::auto_grow_file_knows",
    "AnalyticsStore::propagate_knows_via_co_change",
    "AnalyticsStore::get_learning_health",
    "AnalyticsStore::find_bridge_subgraph",
    "AnalyticsStore::get_knowledge_density",
    "AnalyticsStore::get_node_pagerank",
    "AnalyticsStore::get_bridge_proximity",
    "AnalyticsStore::get_avg_multi_signal_score",
    "AnalyticsStore::check_topology_rules",
    "AnalyticsStore::check_file_topology",
    "AnalyticsStore::batch_save_context_cards",
    "AnalyticsStore::invalidate_context_cards",
    "AnalyticsStore::get_context_card",
    "AnalyticsStore::get_context_cards_batch",
    "AnalyticsStore::find_isomorphic_groups",
    "AnalyticsStore::has_context_cards",
    "AnalyticsStore::get_note_embeddings_for_project",
    "AnalyticsStore::list_project_symbols",
    "AnalyticsStore::get_project_inheritance_edges",
    "AnalyticsStore::get_entity_neighborhood",
    "AnalyticsStore::get_scoped_entity_neighborhood",
    "AnalyticsStore::entity_exists",
    "AnalyticsStore::entities_linked",
    "AnalyticsStore::ranked_foundation",
    "AnalyticsStore::schema_catalog",
    "AnalyticsStore::causal_walk",
    "AnalyticsStore::typed_path",
];

#[async_trait]
#[allow(unused_variables, clippy::too_many_arguments)]
impl AnalyticsStore for ObrainGraphStore {
    // ---- sub-lot A1: what serves --------------------------------------------

    async fn create_analysis_profile(&self, profile: &AnalysisProfile) -> Result<()> {
        self.profil_creer(profile).await
    }

    async fn list_analysis_profiles(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<AnalysisProfile>> {
        let pid = project_id.map(str::to_string);
        self.lire(move |_, etat| {
            let mut portees = vec![""];
            if let Some(p) = pid.as_deref() {
                portees.push(p);
            }
            etat.analyse.garde_liste(PROFIL, &portees)?;
            let mut v: Vec<&AnalysisProfile> = etat
                .analyse
                .profils
                .values()
                .map(|f| &f.valeur)
                .filter(|p| {
                    let propre = p.project_id.as_deref().is_none_or(str::is_empty);
                    propre || (pid.is_some() && p.project_id == pid)
                })
                .collect();
            // ORDER BY is_builtin DESC, name ASC, id ASC.
            v.sort_by(|a, b| {
                b.is_builtin
                    .cmp(&a.is_builtin)
                    .then_with(|| a.name.cmp(&b.name))
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(v.into_iter().cloned().collect())
        })
    }

    async fn get_analysis_profile(&self, id: &str) -> Result<Option<AnalysisProfile>> {
        let id = id.to_string();
        self.lire(move |_, etat| {
            etat.analyse.garde(PROFIL, &[&id])?;
            Ok(etat.analyse.profils.get(&id).map(|f| f.valeur.clone()))
        })
    }

    async fn delete_analysis_profile(&self, id: &str) -> Result<()> {
        self.profil_supprimer(id).await
    }

    async fn create_topology_rule(&self, rule: &TopologyRule) -> Result<()> {
        self.regle_creer(rule).await
    }

    async fn list_topology_rules(&self, project_id: &str) -> Result<Vec<TopologyRule>> {
        let pid = project_id.to_string();
        self.lire(move |_, etat| {
            etat.analyse.garde_liste(REGLE, &[&pid])?;
            let mut v: Vec<&TopologyRule> = etat
                .analyse
                .regles
                .values()
                .map(|f| &f.valeur)
                .filter(|r| r.project_id == pid)
                .collect();
            // ORDER BY rule_type, source_pattern, id.
            v.sort_by(|a, b| {
                a.rule_type
                    .to_string()
                    .cmp(&b.rule_type.to_string())
                    .then_with(|| a.source_pattern.cmp(&b.source_pattern))
                    .then_with(|| a.id.cmp(&b.id))
            });
            Ok(v.into_iter().cloned().collect())
        })
    }

    async fn delete_topology_rule(&self, rule_id: &str) -> Result<()> {
        self.regle_supprimer(rule_id).await
    }

    async fn create_or_get_user_profile(&self, user_id: &str) -> Result<UserProfile> {
        self.utilisateur_creer_ou_lire(user_id).await
    }

    async fn update_user_profile(&self, profile: &UserProfile) -> Result<()> {
        self.utilisateur_maj(profile).await
    }

    async fn get_user_profile(&self, user_id: &str) -> Result<Option<UserProfile>> {
        let uid = user_id.to_string();
        self.lire(move |_, etat| {
            etat.analyse.garde(UTILISATEUR, &["user_id", &uid])?;
            Ok(etat
                .analyse
                .utilisateurs
                .get(&uid)
                .map(|f| f.valeur.clone()))
        })
    }

    async fn upsert_works_on(&self, user_id: &str, project_id: Uuid) -> Result<()> {
        self.travaille_sur(user_id, project_id).await
    }

    async fn get_works_on(&self, user_id: &str) -> Result<Vec<WorksOnRelation>> {
        let uid = user_id.to_string();
        self.lire(move |_, etat| {
            etat.analyse.garde(UTILISATEUR, &["user_id", &uid])?;
            etat.analyse.garde_relation(&uid, None)?;
            if !etat.analyse.utilisateurs.contains_key(&uid) {
                return Ok(Vec::new());
            }
            let mut v: Vec<WorksOnRelation> = etat
                .analyse
                .relations
                .get(&uid)
                .into_iter()
                .flatten()
                .map(|(p, r)| WorksOnRelation {
                    user_id: uid.clone(),
                    project_id: *p,
                    frequency: r.frequence,
                    last_active: r.derniere,
                })
                .collect();
            // ORDER BY r.last_active DESC, p.id DESC.
            v.sort_by(|a, b| {
                b.last_active
                    .cmp(&a.last_active)
                    .then_with(|| b.project_id.cmp(&a.project_id))
            });
            Ok(v)
        })
    }

    async fn get_routing_arm(&self, key: &ArmKey) -> Result<Option<ArmStats>> {
        let k = key.clone();
        self.lire(move |_, etat| {
            etat.analyse
                .garde(BRAS, &["bras", &k.class, &k.provider_id, &k.model])?;
            Ok(etat.analyse.bras.get(&k).map(|f| f.valeur.clone()))
        })
    }

    async fn list_routing_arms(&self, class: &str) -> Result<Vec<ArmStats>> {
        let classe = class.to_string();
        self.lire(move |_, etat| {
            // A quarantined arm of this class: the list refuses; `observe_routing_arm`
            // and `get_routing_arm` on that arm refuse by its reserved key.
            etat.analyse.garde_liste(BRAS, &[&classe])?;
            let mut v: Vec<ArmStats> = etat
                .analyse
                .bras
                .values()
                .filter(|f| f.valeur.key.class == classe)
                .map(|f| f.valeur.clone())
                .collect();
            v.sort_by(|a, b| {
                (&a.key.provider_id, &a.key.model).cmp(&(&b.key.provider_id, &b.key.model))
            });
            Ok(v)
        })
    }

    async fn observe_routing_arm(
        &self,
        key: &ArmKey,
        observation: &ArmObservation,
    ) -> Result<ArmStats> {
        self.bras_observer(key, observation).await
    }

    async fn put_routing_decision(&self, decision: &CognitiveDecision) -> Result<()> {
        self.decision_poser(decision).await
    }

    async fn get_routing_decision(&self, id: Uuid) -> Result<Option<CognitiveDecision>> {
        self.lire(move |db, etat| {
            etat.analyse.garde(DECISION, &[&id.to_string()])?;
            etat.analyse
                .decisions
                .get(&id)
                .map(|f| lire_json_decision(db, f.noeud, id))
                .transpose()
        })
    }

    async fn list_routing_decisions(
        &self,
        filter: &DecisionFilter,
    ) -> Result<Vec<CognitiveDecision>> {
        let f = filter.clone();
        self.lire(move |db, etat| {
            let projet = f.project_slug.clone().unwrap_or_default();
            let session = f.session_id.map(|s| s.to_string()).unwrap_or_default();
            let depuis = f.since.map(stamp).unwrap_or_default();
            let mut v: Vec<&Fiche<DecisionLegere>> = etat
                .analyse
                .decisions
                .values()
                .filter(|d| {
                    (projet.is_empty() || d.valeur.projet == projet)
                        && (session.is_empty() || d.valeur.session == session)
                        && (depuis.is_empty() || d.valeur.at >= depuis)
                })
                .collect();
            // ORDER BY at DESC, id DESC (the id as its string, as Neo4j).
            v.sort_by(|a, b| {
                b.valeur
                    .at
                    .cmp(&a.valeur.at)
                    .then_with(|| b.valeur.id.to_string().cmp(&a.valeur.id.to_string()))
            });
            v.into_iter()
                .skip(f.offset)
                .take(f.limit.unwrap_or(50))
                .map(|d| lire_json_decision(db, d.noeud, d.valeur.id))
                .collect()
        })
    }

    async fn set_routing_decision_outcome(
        &self,
        id: Uuid,
        outcome: DecisionOutcome,
    ) -> Result<bool> {
        self.decision_issue(id, outcome).await
    }

    /// Whether the base answers: not closed, not in an indeterminate state.
    async fn health_check(&self) -> Result<bool> {
        Ok(self.garde().is_ok())
    }

    async fn llm_settings_by_key(&self, key: &str) -> Result<Vec<(String, String)>> {
        let k = key.to_string();
        self.lire(move |_, etat| Ok(etat.chat.reglages_de_cle(&k)))
    }

    async fn get_step_parent_task_id(&self, step_id: Uuid) -> Result<Option<Uuid>> {
        self.lire(move |_, etat| {
            let idx = &etat.taches;
            if idx.reserves_etapes.contains(&step_id) {
                bail!("l'étape {step_id} est en quarantaine : le backend obrain ne répond pas pour elle");
            }
            Ok(idx.etapes.get(&step_id).map(|e| e.tache))
        })
    }

    async fn get_frustration(&self, task_id: Uuid) -> Result<f64> {
        self.lire(move |_, etat| {
            Ok(etat
                .taches
                .tache(task_id)?
                .map_or(0.0, |e| e.tache.frustration_score))
        })
    }

    async fn increment_frustration(&self, task_id: Uuid, delta: f64) -> Result<f64> {
        self.frustration_ecrire(task_id, delta, true).await
    }

    async fn decrement_frustration(&self, task_id: Uuid, delta: f64) -> Result<f64> {
        self.frustration_ecrire(task_id, delta, false).await
    }

    /// The one-time data migrations are Cypher for a Neo4j base
    /// (`neo4j::data_migrations`); none is ported to obrain, so none runs here. This
    /// is NOT a proof that migrated data is repaired: the migrator (outside this
    /// repository) must run them on the Neo4j source before the copy — the list is in
    /// `CONSOLIDATION.md`. Written by hand (it cannot carry a refusal: it returns no
    /// `Result`) so that the contract's default is not inherited by omission.
    async fn run_data_migrations(&self) -> Vec<crate::neo4j::data_migrations::MigrationOutcome> {
        Vec::new()
    }

    // ---- refused by name: sub-lots A2 to A12 (`CONSOLIDATION.md`) -----------

    // ---- sub-lot A10 ----
    async fn find_bridge_subgraph(
        &self,
        source: &str,
        target: &str,
        max_hops: u32,
        relation_types: &[String],
        project_id: &str,
    ) -> Result<(
        Vec<crate::graph::models::BridgeRawNode>,
        Vec<crate::graph::models::BridgeRawEdge>,
    )> {
        Err(crate::obrain::NonPorte("AnalyticsStore::find_bridge_subgraph").into())
    }
    async fn check_topology_rules(&self, project_id: &str) -> Result<Vec<TopologyViolation>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::check_topology_rules").into())
    }
    async fn check_file_topology(
        &self,
        project_id: &str,
        file_path: &str,
        new_imports: &[String],
    ) -> Result<Vec<TopologyViolation>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::check_file_topology").into())
    }
    // REFUSED: walks the neighbours of ANY entity, `File` and `Function` among them; no traversal was run against obrain.
    async fn get_entity_neighborhood(
        &self,
        center_type: &str,
        center_id: &str,
        params: &crate::graph::neighborhood::NeighborhoodParams,
    ) -> Result<Option<crate::graph::neighborhood::RawNeighborhood>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_entity_neighborhood").into())
    }
    // REFUSED: same as `get_entity_neighborhood`, scoped.
    async fn get_scoped_entity_neighborhood(
        &self,
        center_type: &str,
        center_id: &str,
        params: &crate::graph::neighborhood::NeighborhoodParams,
        filter: &crate::graph::neighborhood::ProjectFilter,
    ) -> Result<Option<crate::graph::neighborhood::ScopedNeighborhood>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_scoped_entity_neighborhood").into())
    }
    // REFUSED: `apoc` / `CALL` over every label and relationship of the base; not a query this backend runs.
    async fn schema_catalog(&self) -> Result<crate::graph::schema::SchemaCatalog> {
        Err(crate::obrain::NonPorte("AnalyticsStore::schema_catalog").into())
    }
    // REFUSED: a variable-length walk over every edge type; no GQL against obrain without having been run (`CONSOLIDATION.md`).
    async fn causal_walk(
        &self,
        label: &str,
        id: &str,
        max_depth: u32,
    ) -> Result<crate::graph::schema::CausalWalk> {
        Err(crate::obrain::NonPorte("AnalyticsStore::causal_walk").into())
    }
    // REFUSED: a variable-length path between two entities; same reason as `causal_walk`.
    async fn typed_path(
        &self,
        from_label: &str,
        from_id: &str,
        to_label: &str,
        to_id: &str,
        max_depth: u32,
    ) -> Result<crate::graph::schema::TypedPath> {
        Err(crate::obrain::NonPorte("AnalyticsStore::typed_path").into())
    }

    // ---- sub-lot A11 ----
    async fn auto_scope_to_feature_graphs(&self, project_id: Uuid) -> Result<usize> {
        Err(crate::obrain::NonPorte("AnalyticsStore::auto_scope_to_feature_graphs").into())
    }
    async fn compute_adaptive_thresholds(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::persona::AdaptivePersonaThresholds> {
        Err(crate::obrain::NonPorte("AnalyticsStore::compute_adaptive_thresholds").into())
    }
    async fn auto_grow_file_knows(
        &self,
        persona_id: Uuid,
        file_path: &str,
        weight: f64,
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::auto_grow_file_knows").into())
    }
    async fn propagate_knows_via_co_change(
        &self,
        persona_id: Uuid,
        file_path: &str,
        base_weight: f64,
    ) -> Result<usize> {
        Err(crate::obrain::NonPorte("AnalyticsStore::propagate_knows_via_co_change").into())
    }
    async fn rate_limited_energy_boost(
        &self,
        persona_id: Uuid,
        boost: f64,
        max_per_cycle: f64,
    ) -> Result<bool> {
        self.persona_energie_limitee(persona_id, boost, max_per_cycle)
            .await
    }
    async fn ensure_agent(&self, agent_id: Uuid, model: Option<&str>) -> Result<bool> {
        self.agent_assurer(agent_id, model).await
    }
    async fn backfill_agents(&self) -> Result<crate::neo4j::models::AgentBackfill> {
        self.agent_reprise().await
    }
    async fn link_agent_authored(&self, agent_id: Uuid, note_id: Uuid) -> Result<bool> {
        self.agent_a_ecrit(agent_id, note_id).await
    }
    async fn link_agent_lived(&self, agent_id: Uuid, episode_id: Uuid) -> Result<bool> {
        self.agent_a_vecu(agent_id, episode_id).await
    }
    async fn link_agent_knows(
        &self,
        agent_id: Uuid,
        note_id: Uuid,
        scope: crate::graph::rules::KnowledgeScope,
        project: Option<Uuid>,
        provenance: &crate::graph::rules::AccountableProvenance,
    ) -> Result<bool> {
        self.agent_sait(agent_id, note_id, scope, project, provenance)
            .await
    }
    async fn link_agent_uses(
        &self,
        agent_id: Uuid,
        label: &str,
        target_id: Uuid,
        provenance: &crate::graph::rules::AccountableProvenance,
    ) -> Result<bool> {
        self.agent_utilise(agent_id, label, target_id, provenance)
            .await
    }
    async fn release_agent_knows(
        &self,
        agent_id: Uuid,
        rule: &str,
        keep: &[Uuid],
    ) -> Result<usize> {
        self.agent_relacher(agent_id, rule, keep).await
    }
    // REFUSED: reads `File` / `Function` / `Struct` / `Trait` / `Enum` by name; the code domain is refused whole (`code.rs`), so `false` would REFUTE a belief that Neo4j can confirm.
    async fn entity_exists(
        &self,
        kind: &str,
        name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Option<bool>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::entity_exists").into())
    }
    // REFUSED: same as `entity_exists` (an edge between two code entities).
    async fn entities_linked(
        &self,
        from_kind: &str,
        from_name: &str,
        edge: &str,
        to_kind: &str,
        to_name: &str,
        project_id: Option<Uuid>,
    ) -> Result<Option<bool>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::entities_linked").into())
    }
    async fn project_synced(&self, project_id: Uuid) -> Result<Option<bool>> {
        self.projet_synchronise(project_id)
    }
    async fn list_assertion_notes(&self, limit: usize) -> Result<Vec<crate::notes::Note>> {
        self.croyances(limit)
    }
    async fn record_assertion_check(
        &self,
        note_id: Uuid,
        result: &crate::notes::models::AssertionResult,
        status: Option<crate::notes::models::NoteStatus>,
        reason: Option<&str>,
    ) -> Result<()> {
        self.croyance_verifiee(note_id, result, status, reason)
            .await
    }
    async fn agent_knows(
        &self,
        agent_id: Uuid,
        project: Option<Uuid>,
        limit: usize,
    ) -> Result<crate::graph::rules::AgentKnowledge> {
        self.agent_connait(agent_id, project, limit)
    }
    // REFUSED: counts `LINKED_TO` anchors on `File` and `SYNAPSE` edges (sub-lot A2), neither written here.
    async fn ranked_foundation(
        &self,
        project: Option<Uuid>,
        limit: usize,
    ) -> Result<Vec<crate::graph::rules::FoundationCandidate>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::ranked_foundation").into())
    }

    // ---- sub-lot A12 ----
    async fn auto_observed_report(
        &self,
        project_id: Option<Uuid>,
    ) -> Result<Vec<serde_json::Value>> {
        self.memoire_observees_rapport(project_id).await
    }
    async fn archive_unconfirmed_auto_observed(&self, project_id: Option<Uuid>) -> Result<usize> {
        self.memoire_observees_archiver(project_id).await
    }
    async fn rescope_protocol(&self, protocol_id: Uuid, project_id: Option<Uuid>) -> Result<()> {
        self.protocole_rescoper(protocol_id, project_id).await
    }
    async fn ensure_project_canonical_paths(&self) -> Result<(usize, usize)> {
        self.chemins_canoniques_assurer().await
    }
    async fn get_deep_maintenance_times(
        &self,
    ) -> Result<Vec<(Uuid, chrono::DateTime<chrono::Utc>)>> {
        self.maintenance_profonde_lire().await
    }
    async fn mark_deep_maintenance(&self, project_id: Uuid) -> Result<()> {
        self.maintenance_profonde_marquer(project_id).await
    }

    // ---- sub-lot A2 ----
    async fn create_synapses(&self, note_id: Uuid, neighbors: &[(Uuid, f64)]) -> Result<usize> {
        self.synapses_creer(note_id, neighbors).await
    }
    async fn get_synapses(&self, note_id: Uuid) -> Result<Vec<(Uuid, f64)>> {
        self.synapses_lire(note_id).await
    }
    async fn delete_synapses(&self, note_id: Uuid) -> Result<usize> {
        self.synapses_supprimer(note_id).await
    }
    async fn get_project_note_entity_links(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, String)>> {
        self.liens_de_notes_du_projet(project_id).await
    }
    async fn get_project_note_synapses(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.synapses_du_projet(project_id, min_weight).await
    }
    async fn update_energy_scores(&self, half_life_days: f64) -> Result<usize> {
        self.energie_mettre_a_jour(half_life_days).await
    }
    async fn boost_energy(&self, note_id: Uuid, amount: f64) -> Result<()> {
        self.energie_renforcer(note_id, amount).await
    }
    async fn track_reactivation(&self, note_ids: &[Uuid]) -> Result<usize> {
        self.reactivations_suivre(note_ids).await
    }
    async fn reinforce_synapses(&self, note_ids: &[Uuid], boost: f64) -> Result<usize> {
        self.synapses_renforcer(note_ids, boost).await
    }
    async fn decay_synapses(
        &self,
        policy: &crate::neurons::forgetting::DecayPolicy,
    ) -> Result<crate::neurons::forgetting::PassReport> {
        self.synapses_oublier(policy, None).await
    }
    async fn apply_scars(&self, node_ids: &[Uuid], increment: f64) -> Result<usize> {
        self.cicatrices_appliquer(node_ids, increment).await
    }
    async fn heal_scars(&self, node_id: Uuid) -> Result<bool> {
        self.cicatrices_guerir(node_id).await
    }
    async fn consolidate_memory(&self) -> Result<(usize, usize)> {
        self.memoire_consolider().await
    }
    async fn init_note_energy(&self, project_id: Option<Uuid>) -> Result<usize> {
        self.energie_initialiser(project_id).await
    }
    async fn list_notes_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
        project_id: Option<Uuid>,
    ) -> Result<(Vec<crate::notes::Note>, usize)> {
        self.notes_sans_synapse(limit, offset, project_id).await
    }
    async fn create_cross_entity_synapses(
        &self,
        source_id: Uuid,
        neighbors: &[(Uuid, f64)],
    ) -> Result<usize> {
        self.synapses_creer_entre(source_id, neighbors, true).await
    }
    async fn get_cross_entity_synapses(&self, node_id: Uuid) -> Result<Vec<(Uuid, f64, String)>> {
        self.synapses_croisees(node_id).await
    }
    async fn list_decisions_needing_synapses(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DecisionNode>, usize)> {
        self.decisions_sans_synapse(limit, offset).await
    }
    async fn get_all_synapse_weights(&self, project_id: Option<Uuid>) -> Result<Vec<f64>> {
        self.synapses_poids(project_id).await
    }
    async fn get_project_synapse_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, f64)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_synapse_edges").into())
    }
    async fn decay_project_synapses(
        &self,
        _project_id: Uuid,
        policy: &crate::neurons::forgetting::DecayPolicy,
    ) -> Result<crate::neurons::forgetting::PassReport> {
        self.synapses_oublier(policy, Some(_project_id)).await
    }
    async fn weaken_node_synapses(
        &self,
        node_id: Uuid,
        amount: f64,
        prune_threshold: f64,
    ) -> Result<usize> {
        self.synapses_affaiblir(node_id, amount, prune_threshold)
            .await
    }
    async fn get_synapse_graph(
        &self,
        project_id: Uuid,
        min_weight: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.synapses_graphe(project_id, min_weight).await
    }
    async fn coactivate_session(
        &self,
        project: Uuid,
        notes: &[Uuid],
        session: Uuid,
        provenance: &crate::graph::rules::AccountableProvenance,
    ) -> Result<usize> {
        self.synapses_coactiver(
            project,
            notes,
            session,
            provenance.rule.clone(),
            provenance.measure.clone(),
            provenance.at.clone(),
        )
        .await
    }

    // ---- sub-lot A3 ----
    async fn shadow_synapse_edges(
        &self,
        project_id: Uuid,
        from: &[Uuid],
    ) -> Result<Vec<crate::neurons::synaptic_shadow::ShadowEdgeRow>> {
        self.ombre_aretes(project_id, from).await
    }
    async fn reward_shadow_synapses(
        &self,
        project_id: Uuid,
        pairs: &[(Uuid, Uuid)],
        delta: f64,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        self.ombre_recompenser(project_id, pairs, delta, now).await
    }
    async fn decay_shadow_synapses(
        &self,
        scope: &crate::neurons::synaptic_shadow::ShadowDecayScope,
        half_life_days: f64,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        self.ombre_oublier(scope, half_life_days, now).await
    }
    async fn record_synapse_shadow_trace(
        &self,
        trace: &crate::neurons::synaptic_shadow::SynapseShadowTrace,
        keep: usize,
    ) -> Result<()> {
        self.ombre_enregistrer(trace, keep).await
    }
    async fn list_synapse_shadow_traces(
        &self,
        filter: &crate::neurons::synaptic_shadow::ShadowTraceFilter,
    ) -> Result<Vec<crate::neurons::synaptic_shadow::SynapseShadowTrace>> {
        self.ombre_lister(filter).await
    }
    async fn get_synapse_shadow_trace(
        &self,
        id: Uuid,
    ) -> Result<Option<crate::neurons::synaptic_shadow::SynapseShadowTrace>> {
        self.ombre_lire(id).await
    }
    async fn claim_shadow_trace_reward(&self, trace_id: Uuid, key: &str) -> Result<bool> {
        self.ombre_reclamer(trace_id, key).await
    }
    async fn shadow_trace_closures(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<(Uuid, crate::neurons::synaptic_shadow::TaskClosure)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::shadow_trace_closures").into())
    }

    // ---- sub-lot A4 ----
    async fn set_file_embedding(
        &self,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::set_file_embedding").into())
    }
    async fn set_function_embedding(
        &self,
        function_name: &str,
        file_path: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::set_function_embedding").into())
    }
    async fn vector_search_files(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, f64)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::vector_search_files").into())
    }
    async fn vector_search_functions(
        &self,
        embedding: &[f32],
        limit: usize,
        project_id: Option<Uuid>,
    ) -> Result<Vec<(String, String, f64)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::vector_search_functions").into())
    }
    async fn get_note_embeddings_for_project(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<NoteEmbeddingPoint>> {
        // Refused, with its reason: a note's embedding can be neither written nor read back on
        // this backend (`NoteStore::set_note_embedding` / `get_note_embedding` refuse: obrain does
        // not journal a vector), so an empty list could not be told from lost embeddings.
        Err(crate::obrain::NonPorte("AnalyticsStore::get_note_embeddings_for_project").into())
    }

    // ---- sub-lot A5 ----
    async fn get_project_import_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_import_edges").into())
    }
    async fn get_project_call_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_call_edges").into())
    }
    async fn get_project_extends_edges(&self, project_id: Uuid) -> Result<Vec<(String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_extends_edges").into())
    }
    async fn get_project_implements_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_implements_edges").into())
    }
    async fn get_project_structural_dna(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_structural_dna").into())
    }
    async fn get_project_structural_fingerprints(
        &self,
        project_id: &str,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_structural_fingerprints").into())
    }
    async fn get_project_file_signals(
        &self,
        project_id: &str,
    ) -> Result<Vec<crate::graph::models::FileSignalRecord>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_file_signals").into())
    }
    async fn get_file_structural_dna(
        &self,
        project_id: &str,
        path: &str,
    ) -> Result<Option<Vec<f64>>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_file_structural_dna").into())
    }
    async fn get_import_edges_among(
        &self,
        project_id: Uuid,
        paths: &[String],
    ) -> Result<Vec<(String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_import_edges_among").into())
    }
    async fn get_call_edges_among(
        &self,
        project_id: Uuid,
        ids: &[String],
    ) -> Result<Vec<(String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_call_edges_among").into())
    }
    async fn get_top_hotspots(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<FileChurnScore>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_top_hotspots").into())
    }
    async fn get_top_knowledge_gaps(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<FileKnowledgeDensity>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_top_knowledge_gaps").into())
    }
    async fn get_node_pagerank(&self, file_path: &str, project_id: &str) -> Result<f64> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_node_pagerank").into())
    }
    async fn get_bridge_proximity(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> Result<Vec<(String, f64)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_bridge_proximity").into())
    }
    async fn list_project_symbols(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<(String, String, String, String, Option<String>, Option<i64>)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::list_project_symbols").into())
    }
    async fn get_project_inheritance_edges(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(String, String, String)>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_project_inheritance_edges").into())
    }

    // ---- sub-lot A6 ----
    async fn batch_update_file_analytics(&self, updates: &[FileAnalyticsUpdate]) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_file_analytics").into())
    }
    async fn batch_update_function_analytics(
        &self,
        updates: &[FunctionAnalyticsUpdate],
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_function_analytics").into())
    }
    async fn batch_update_fabric_file_analytics(
        &self,
        updates: &[FabricFileAnalyticsUpdate],
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_fabric_file_analytics").into())
    }
    async fn batch_update_structural_dna(
        &self,
        updates: &[crate::graph::models::StructuralDnaUpdate],
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_structural_dna").into())
    }
    async fn write_predicted_links(
        &self,
        project_id: &str,
        links: &[crate::graph::models::LinkPrediction],
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::write_predicted_links").into())
    }
    async fn batch_update_structural_fingerprints(
        &self,
        updates: &[crate::graph::models::StructuralFingerprintUpdate],
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_structural_fingerprints").into())
    }
    async fn batch_update_churn_scores(&self, updates: &[FileChurnScore]) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_churn_scores").into())
    }
    async fn batch_update_knowledge_density(&self, updates: &[FileKnowledgeDensity]) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_knowledge_density").into())
    }
    async fn batch_update_risk_scores(&self, updates: &[FileRiskScore]) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_update_risk_scores").into())
    }
    async fn batch_upsert_processes(&self, processes: &[ProcessNode]) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_upsert_processes").into())
    }
    async fn batch_create_step_relationships(&self, steps: &[(String, String, u32)]) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_create_step_relationships").into())
    }
    async fn delete_project_processes(&self, project_id: Uuid) -> Result<u64> {
        Err(crate::obrain::NonPorte("AnalyticsStore::delete_project_processes").into())
    }

    // ---- sub-lot A7 ----
    async fn compute_homeostasis(
        &self,
        project_id: Uuid,
        custom_ranges: Option<&[(String, f64, f64)]>,
    ) -> Result<crate::neo4j::models::HomeostasisReport> {
        Err(crate::obrain::NonPorte("AnalyticsStore::compute_homeostasis").into())
    }
    async fn compute_structural_drift(
        &self,
        project_id: Uuid,
        warning_threshold: Option<f64>,
        critical_threshold: Option<f64>,
    ) -> Result<crate::neo4j::models::StructuralDriftReport> {
        Err(crate::obrain::NonPorte("AnalyticsStore::compute_structural_drift").into())
    }
    async fn get_neural_metrics(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::models::NeuralMetrics> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_neural_metrics").into())
    }
    async fn compute_churn_scores(&self, project_id: Uuid) -> Result<Vec<FileChurnScore>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::compute_churn_scores").into())
    }
    async fn compute_knowledge_density(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<FileKnowledgeDensity>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::compute_knowledge_density").into())
    }
    async fn compute_risk_scores(&self, project_id: Uuid) -> Result<Vec<FileRiskScore>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::compute_risk_scores").into())
    }
    async fn get_risk_summary(&self, project_id: Uuid) -> Result<serde_json::Value> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_risk_summary").into())
    }
    async fn get_learning_health(
        &self,
        project_id: Uuid,
    ) -> Result<crate::neo4j::analytics::LearningHealthReport> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_learning_health").into())
    }
    async fn get_knowledge_density(&self, file_path: &str, project_id: &str) -> Result<f64> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_knowledge_density").into())
    }
    async fn get_avg_multi_signal_score(&self, project_id: Uuid) -> Result<f64> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_avg_multi_signal_score").into())
    }
    async fn find_isomorphic_groups(
        &self,
        project_id: &str,
        min_group_size: usize,
    ) -> Result<Vec<crate::graph::models::IsomorphicGroup>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::find_isomorphic_groups").into())
    }
    async fn audit_knowledge_gaps(&self, project_id: Uuid) -> Result<AuditGapsReport> {
        self.memoire_lacunes(project_id).await
    }

    // ---- sub-lot A8 ----
    async fn batch_save_context_cards(
        &self,
        cards: &[crate::graph::models::ContextCard],
    ) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::batch_save_context_cards").into())
    }
    async fn invalidate_context_cards(&self, paths: &[String], project_id: &str) -> Result<()> {
        Err(crate::obrain::NonPorte("AnalyticsStore::invalidate_context_cards").into())
    }
    async fn get_context_card(
        &self,
        path: &str,
        project_id: &str,
    ) -> Result<Option<crate::graph::models::ContextCard>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_context_card").into())
    }
    async fn get_context_cards_batch(
        &self,
        paths: &[String],
        project_id: &str,
    ) -> Result<Vec<crate::graph::models::ContextCard>> {
        Err(crate::obrain::NonPorte("AnalyticsStore::get_context_cards_batch").into())
    }
    async fn has_context_cards(&self, project_id: &str) -> Result<bool> {
        Err(crate::obrain::NonPorte("AnalyticsStore::has_context_cards").into())
    }

    // ---- sub-lot A9 ----
    async fn get_project_decisions_for_graph(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(DecisionNode, Vec<AffectsRelation>)>> {
        self.vue_decisions(project_id)
    }
    async fn get_pm_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<PmGraphNode>, Vec<PmGraphEdge>)> {
        self.vue_pm(project_id, limit)
    }
    async fn get_chat_graph_data(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<(Vec<ChatGraphSession>, Vec<ChatGraphDiscussed>)> {
        self.vue_discussions(project_id)
    }
    async fn get_project_constraints(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<(crate::neo4j::models::ConstraintNode, Uuid)>> {
        self.vue_contraintes(project_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal list names exactly the methods this file refuses, and every one
    /// of them answers its own name.
    #[test]
    fn the_registry_matches_the_written_refusals() {
        let source = include_str!("analytics.rs");
        let ecrit = source
            .split("Err(crate::obrain::NonPorte(\"")
            .skip(1)
            .map(|s| s.split('"').next().unwrap())
            .filter(|n| n.starts_with("AnalyticsStore::"))
            .collect::<Vec<_>>();
        let mut a: Vec<&str> = ecrit;
        let mut b: Vec<&str> = REFUSEES.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(REFUSEES.len(), 66);
    }

    /// The reserved keys of `trigger::cle_reservee` never collide on a value that
    /// contains what used to be the separator.
    #[test]
    fn reserved_keys_are_unambiguous() {
        assert_ne!(
            cle_reservee(BRAS, &["bras", "a|b", "c", "m"]),
            cle_reservee(BRAS, &["bras", "a", "b|c", "m"])
        );
        assert_ne!(
            cle_reservee(UTILISATEUR, &["alice"]),
            cle_reservee(UTILISATEUR, &["user_id", "alice"])
        );
        assert_ne!(
            cle_reservee(REGLE, &["ab"]),
            cle_reservee(REGLE, &["a", "b"])
        );
    }
}
