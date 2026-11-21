//! `AnalyticsStore` on obrain, the memory layer of the notes: the forgetting passes over the
//! `SYNAPSE` edges, the lifecycle sweep of the notes, the notes the heuristic authored, and the
//! knowledge-gap audit. The methods are declared in `analytics.rs`, which delegates here. In Rust
//! over the indexes and the edges of the engine, no query language.
//!
//! # What serves (6)
//!
//! `decay_synapses`, `decay_project_synapses`, `consolidate_memory`, `auto_observed_report`,
//! `archive_unconfirmed_auto_observed`, `audit_knowledge_gaps`.
//!
//! Refused by name, here and in `analytics.rs`: `get_note_embeddings_for_project` (a note's
//! embedding can be neither written nor read back on this backend — `NoteStore::set_note_embedding`
//! and `get_note_embedding` refuse for the reason `decision.rs` gives: obrain does not journal a
//! vector, so an answer "no embedding" could not be told from a lost one).
//!
//! # The forgetting passes (`decay_synapses`, `decay_project_synapses`)
//!
//! The model is `crate::neurons::forgetting`; the three steps are those of
//! `Neo4jClient::forgetting_pass`:
//!
//! 1. proportional decay of every `SYNAPSE` edge in scope, `w · 2^(−Δt / h)`, `Δt` read from the
//!    edge's own `decayed_at` (an edge with none is stamped, not decayed), `h` by provenance,
//!    floored by the protection of an edge reinforced `protected_after` times;
//! 2. archive, never destroy: an edge below the forget threshold, never proven by use, untouched
//!    for `forget_after_days`, becomes a `FORGOTTEN` edge carrying all its properties, with
//!    `forgotten_at` and `forgotten_reason = 'decay'`;
//! 3. the scars of EVERY note and decision heal by `0.02 · 0.05`, whatever the scope (Neo4j's
//!    query has no scope on this step: a project pass heals the scars of all projects; reproduced).
//!
//! A project pass of a dormant project does nothing (not even step 3). The global pass leaves the
//! edges leaving a note of a dormant project alone.
//!
//! ## The ORDER of the archive, named
//!
//! Neo4j's archive is `… WITH a, b, s LIMIT $cap` with no `ORDER BY`: WHICH edges the cap lets
//! through is whatever the engine produces first, and the degree floor is evaluated on the state
//! BEFORE the pass (so a node of degree 4 could lose three edges in one pass, below
//! `min_degree`). Neither can be reproduced, and neither is a reference. This backend defines a
//! TOTAL order and a stricter floor, both named:
//!
//! - the candidates are ranked by weight ascending (the weakest first), then by the id of the
//!   source note, then by the id of the target (`f64::total_cmp`, then the string order of the
//!   ids);
//! - the degree floor is checked on the LIVE degrees: an edge is archived only while both its ends
//!   keep strictly more than `min_degree` outgoing `SYNAPSE` edges after the edges the pass has
//!   already archived. A node is therefore never taken below `min_degree` by one pass, which is
//!   what the documentation of the policy promises.
//!
//! ## Other differences with the Neo4j adapter
//!
//! - **A weight that is absent or not a number refuses the pass** (Neo4j writes a null weight,
//!   i.e. removes it), as `synapse.rs`. Non-finite policy values refuse.
//! - **A scan refuses when a note or a decision was quarantined at open.**
//! - **Elapsed time is a whole number of seconds** truncated toward zero, as
//!   `duration.inSeconds(…).seconds`.
//! - **An archived edge replaces the `FORGOTTEN` edge the pair already had** (one can exist: a
//!   cosine `create_synapses` does not remove it). Neo4j would write a second one, which
//!   `reinforce_synapses` then refuses to revive (`synapse.rs`). More than one already there
//!   refuses the pass.
//! - **Nothing is swallowed**: the scar step and the counts, which Neo4j wraps in `if let Ok`,
//!   propagate their errors.
//!
//! ## Atomicity
//!
//! An archived edge is created (`FORGOTTEN`, every property in ONE engine record) BEFORE the
//! `SYNAPSE` edge is deleted: a cut leaves both, readable, never a lost edge; the next pass finds
//! the `FORGOTTEN` edge of the pair and replaces it. Step 1 writes `decayed_at` BEFORE `weight`: a
//! cut loses the decay of one interval, it never charges it twice. Every multi-write is under
//! `ecritures`: a cut makes the store refuse everything until reopened (`indetermine`).
//!
//! # The lifecycle sweep (`consolidate_memory`)
//!
//! The notes whose `status` is `active`, in the order `(energy, id)`, each judged by
//! `NoteLifecycleManager` exactly as `Neo4jClient::consolidate_memory` does (auto-archive,
//! ephemeral expiry, promotion). The usage signal is the greatest of `activation_count`,
//! `reactivation_count` and the number of `SYNAPSE` edges touching the note. It REFUSES when a
//! note was quarantined at open (the sweep would skip it; Neo4j logs and skips it, here that is
//! a refusal, as `synapse.rs`), and when the history of a note to change (`changes_json`) is
//! present and unreadable (Neo4j would overwrite it with an empty one plus the new entry). The
//! two properties of a change are written history FIRST, meaning (`status`, `memory_horizon`)
//! LAST: a cut leaves a history entry without its effect, never an effect without its trace.
//!
//! # The auto-observed notes
//!
//! `auto_observed_report` groups by the RAW `(status, note_type, importance)`, ordered by count
//! descending then the three strings; `archive_unconfirmed_auto_observed` sets `status` to
//! `archived`, one property per note (each is atomic), for the active notes of the scope that no
//! person confirmed. Both refuse when a note was quarantined.
//!
//! # The audit (`audit_knowledge_gaps`)
//!
//! Notes without an outgoing `LINKED_TO`, decisions without an outgoing `AFFECTS`, commits without
//! an outgoing `TOUCHES`, skills with no incoming `MEMBER_OF` from a note nor `MEMBER_OF_SKILL`
//! from a decision, and the census of every relationship type of the base. Read on the engine's
//! nodes and edges, not on the indexes (it is what Neo4j's `MATCH` sees, quarantined nodes
//! included): the true totals, the samples (100, 50 for the skills) sorted by id, since Cypher
//! gives no order. A node whose key is unreadable refuses (Neo4j prints an empty one).
//! This backend cannot create an `AFFECTS` edge (the decision domain refuses it): every decision
//! of a base made here is a gap, which is what the base holds, and a migrated base brings its own
//! edges.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use chrono::{Duration, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::synapse::{
    aretes_de, entre, fini, genre_de, id_du_noeud, incidentes, instant, nombre_du_noeud,
    projets_endormis, propriete, refus_si_notes_en_quarantaine, statut_brut_actif, supprimer_arete,
    Arete, Genre, OUBLIE, SYNAPSE,
};
use super::task::ecritures;
use super::{decision, note, Ecriture, ObrainGraphStore};
use crate::neo4j::models::{AuditGapsReport, RelTypeCount};
use crate::neurons::forgetting::{DecayPolicy, PassReport};
use crate::notes::lifecycle::NoteLifecycleManager;
use crate::notes::{ChangeType, MemoryHorizon, NoteChange};

/// What a pass heals of a scar: `0.02 · 0.05` (`SCAR_DECAY_PER_PASS` in `neo4j/note.rs`).
const CICATRISATION_PAR_PASSE: f64 = 0.02 * 0.05;
const BROUILLON: &str = "_brouillon";

// ============================================================================
// The forgetting passes
// ============================================================================

/// A node that carries `SYNAPSE` edges: a note or a decision.
struct Source {
    noeud: NodeId,
    id: Uuid,
    /// The raw `project_id` (`None` when absent).
    projet: Option<String>,
    aretes: Vec<Arete>,
}

/// One edge of step 1, with what the pass computed.
struct Ligne {
    de: usize,
    arete: Arete,
    nouveau: f64,
}

fn politique_finie(p: &DecayPolicy) -> Result<()> {
    for (nom, v) in [
        ("half_life_days", p.half_life_days),
        ("similarity_half_life_days", p.similarity_half_life_days),
    ] {
        if !(v.is_finite() && v > 0.0) {
            bail!("DecayPolicy : `{nom}` doit être un nombre strictement positif (reçu {v})");
        }
    }
    for (nom, v) in [
        ("dormancy_threshold", p.dormancy_threshold),
        ("forget_threshold", p.forget_threshold),
        ("protection_floor", p.protection_floor),
        ("max_archive_fraction", p.max_archive_fraction),
    ] {
        fini(v, nom)?;
    }
    Ok(())
}

impl ObrainGraphStore {
    /// `decay_synapses` (`portee: None`) and `decay_project_synapses` (`Some(projet)`).
    pub(super) async fn synapses_oublier(
        &self,
        politique: &DecayPolicy,
        portee: Option<Uuid>,
    ) -> Result<PassReport> {
        politique_finie(politique)?;
        let p = politique.clone();
        let methode = if portee.is_some() {
            "decay_project_synapses"
        } else {
            "decay_synapses"
        };
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let now = Utc::now();
                let portee_s = portee.map(|u| u.to_string());
                let mut sources: Vec<Source> = Vec::new();
                let mut notes_a_cicatriser: Vec<(Uuid, NodeId)> = Vec::new();
                let mut decisions_a_cicatriser: Vec<(Uuid, NodeId)> = Vec::new();
                let endormis: HashSet<String>;
                {
                    let etat = w.index()?;
                    refus_si_notes_en_quarantaine(&etat, methode)?;
                    let n = etat.decisions.en_quarantaine();
                    if n > 0 {
                        bail!(
                            "{methode} : {n} nœud(s) Decision en quarantaine (illisibles à \
                             l'ouverture) : la passe les manquerait"
                        );
                    }
                    endormis = projets_endormis(&etat, now)?
                        .into_iter()
                        .map(|u| u.to_string())
                        .collect();
                    if let Some(p) = &portee_s {
                        if endormis.contains(p) {
                            // A dormant project is frozen: its knowledge does not decay.
                            return Ok(PassReport::default());
                        }
                    }
                    let mut noeuds: Vec<(Uuid, NodeId)> = etat
                        .notes
                        .toutes_avec_noeud()
                        .map(|(n, x)| (n.id, x))
                        .collect();
                    notes_a_cicatriser.extend(noeuds.iter().copied());
                    let decisions: Vec<(Uuid, NodeId)> = etat
                        .decisions
                        .toutes_avec_noeud()
                        .map(|(d, x)| (d.id, x))
                        .collect();
                    decisions_a_cicatriser.extend(decisions.iter().copied());
                    noeuds.extend(decisions);
                    for (id, noeud) in noeuds {
                        let projet = match propriete(w.db, noeud, "project_id")? {
                            None => None,
                            Some(v) => Some(v.as_str().map(str::to_string).ok_or_else(|| {
                                anyhow!("nœud {noeud:?} : `project_id` n'est pas une chaîne")
                            })?),
                        };
                        sources.push(Source {
                            noeud,
                            id,
                            projet,
                            aretes: aretes_de(w.db, noeud, Direction::Outgoing, SYNAPSE)?,
                        });
                    }
                }
                // The outgoing degree of every node, before the pass (the floor's start).
                let mut degre: HashMap<NodeId, usize> =
                    sources.iter().map(|s| (s.noeud, s.aretes.len())).collect();

                // ---- 1. proportional decay ---------------------------------------------
                let mut lignes: Vec<Ligne> = Vec::new();
                let mut rapport = PassReport::default();
                for (i, s) in sources.iter().enumerate() {
                    let dans_la_portee = match &portee_s {
                        Some(cible) => s.projet.as_deref() == Some(cible.as_str()),
                        None => s.projet.as_ref().is_none_or(|pid| !endormis.contains(pid)),
                    };
                    if !dans_la_portee {
                        continue;
                    }
                    for a in &s.aretes {
                        let poids = a.poids()?;
                        let rc = a.nombre("reinforcement_count")?.unwrap_or(0.0);
                        let h = if a.texte("source") == Some("coactivation") {
                            p.half_life_days
                        } else {
                            p.similarity_half_life_days
                        };
                        // The weight is valid AS OF its own `decayed_at`; none: stamped now.
                        let ecoule = match a.get("decayed_at") {
                            None => 0,
                            Some(v) => {
                                let d = super::date_valeur(v, "decayed_at")
                                    .map_err(|e| anyhow!("{methode} : arête {:?} : {e}", a.id))?;
                                d.map_or(0, |d| (now - d).num_seconds().max(0))
                            }
                        };
                        let brut = poids * 2f64.powf(-(ecoule as f64) / (h * 86_400.0));
                        let w1 = if rc >= f64::from(p.protected_after) && brut < p.protection_floor
                        {
                            p.protection_floor
                        } else {
                            brut
                        };
                        rapport.decayed += 1;
                        if w1 < p.dormancy_threshold {
                            rapport.dormant += 1;
                        }
                        if rc >= f64::from(p.protected_after) {
                            rapport.protected += 1;
                        }
                        lignes.push(Ligne {
                            de: i,
                            arete: a.clone(),
                            nouveau: w1,
                        });
                    }
                }

                // ---- 2. archive, never destroy ----------------------------------------
                let cap = p.archive_cap(rapport.decayed);
                let limite = now - Duration::days(p.forget_after_days);
                let mut candidats: Vec<(usize, Uuid, Uuid)> = Vec::new();
                for (k, l) in lignes.iter().enumerate() {
                    let a = &l.arete;
                    let rc = a.nombre("reinforcement_count")?.unwrap_or(0.0);
                    if !(l.nouveau < p.forget_threshold && rc < f64::from(p.protected_after)) {
                        continue;
                    }
                    // `coalesce(last_reinforced_at, created_at) < now - forget_after_days`:
                    // no date at all is NULL, so the edge is left alone.
                    let (cle, v) = match (a.get("last_reinforced_at"), a.get("created_at")) {
                        (Some(v), _) => ("last_reinforced_at", v),
                        (None, Some(v)) => ("created_at", v),
                        (None, None) => continue,
                    };
                    let touche = super::date_valeur(v, cle)
                        .map_err(|e| anyhow!("{methode} : arête {:?} : {e}", a.id))?;
                    if !touche.is_some_and(|t| t < limite) {
                        continue;
                    }
                    candidats.push((k, sources[l.de].id, id_du_noeud(w.db, a.vers)?));
                }
                // The total order, named in the module doc: weakest first, then the ids.
                candidats.sort_by(|x, y| {
                    lignes[x.0]
                        .nouveau
                        .total_cmp(&lignes[y.0].nouveau)
                        .then_with(|| x.1.to_string().cmp(&y.1.to_string()))
                        .then_with(|| x.2.to_string().cmp(&y.2.to_string()))
                        .then_with(|| {
                            format!("{:?}", lignes[x.0].arete.id)
                                .cmp(&format!("{:?}", lignes[y.0].arete.id))
                        })
                });
                let mut archives: Vec<usize> = Vec::new();
                for (k, _, _) in &candidats {
                    if archives.len() >= cap {
                        break;
                    }
                    let a = &lignes[*k].arete;
                    let (da, db_) = (
                        degre.get(&a.de).copied().unwrap_or(0),
                        degre.get(&a.vers).copied().unwrap_or(0),
                    );
                    if da > p.min_degree && db_ > p.min_degree {
                        // The live floor: this edge leaves its source.
                        *degre.entry(a.de).or_insert(0) -= 1;
                        archives.push(*k);
                    }
                }
                rapport.archived = archives.len();
                rapport.capped = rapport.archived >= cap && rapport.archived > 0;
                // The FORGOTTEN edges the pairs already have, read before any write.
                let mut anciens: Vec<Vec<Arete>> = Vec::new();
                for k in &archives {
                    let a = &lignes[*k].arete;
                    let deja = entre(w.db, a.de, a.vers, OUBLIE)?;
                    if deja.len() > 1 {
                        bail!(
                            "{methode} : {} arêtes {OUBLIE} parallèles de {:?} vers {:?} : la \
                             passe ne devine pas laquelle remplacer",
                            deja.len(),
                            a.de,
                            a.vers
                        );
                    }
                    anciens.push(deja);
                }

                // ---- 3. the scars heal, every note and decision --------------------------
                let mut cicatrises_notes: Vec<(Uuid, f64)> = Vec::new();
                let mut cicatrises_decisions: Vec<(Uuid, f64)> = Vec::new();
                for (cible, ids) in [
                    (&mut cicatrises_notes, &notes_a_cicatriser),
                    (&mut cicatrises_decisions, &decisions_a_cicatriser),
                ] {
                    for (id, noeud) in ids {
                        let Some(s) = nombre_du_noeud(w.db, *noeud, "scar_intensity")? else {
                            continue;
                        };
                        if s > 0.0 {
                            let n = if s - CICATRISATION_PAR_PASSE < 0.001 {
                                0.0
                            } else {
                                s - CICATRISATION_PAR_PASSE
                            };
                            cible.push((*id, n));
                        }
                    }
                }

                // ---- the writes ----------------------------------------------------------
                let maintenant = instant(now);
                let mut pub_notes = Vec::new();
                let mut pub_decisions = Vec::new();
                ecritures(w, |w| {
                    let archive_set: HashSet<usize> = archives.iter().copied().collect();
                    for (k, l) in lignes.iter().enumerate() {
                        if archive_set.contains(&k) {
                            continue;
                        }
                        // `decayed_at` first: a cut then loses an interval of decay, never
                        // charges it twice.
                        w.etape()?;
                        w.poser_arete(l.arete.id, "decayed_at", maintenant.clone());
                        w.etape()?;
                        w.poser_arete(l.arete.id, "weight", Value::from(l.nouveau));
                    }
                    for (n, k) in archives.iter().enumerate() {
                        let l = &lignes[*k];
                        let a = &l.arete;
                        let mut props: BTreeMap<String, Value> =
                            a.p.iter()
                                .filter(|(_, v)| !matches!(v, Value::Null))
                                .map(|(c, v)| (c.clone(), v.clone()))
                                .collect();
                        props.insert("weight".into(), Value::from(l.nouveau));
                        props.insert("decayed_at".into(), maintenant.clone());
                        props.insert("forgotten_at".into(), maintenant.clone());
                        props.insert("forgotten_reason".into(), Value::from("decay"));
                        // Created whole BEFORE the synapse goes: a cut leaves both.
                        w.etape()?;
                        w.db.create_edge_with_props(a.de, a.vers, OUBLIE, props);
                        for vieux in &anciens[n] {
                            supprimer_arete(w, vieux.id, methode)?;
                        }
                        supprimer_arete(w, a.id, methode)?;
                    }
                    for (id, n) in &cicatrises_notes {
                        pub_notes.extend(note::poser_proprietes_de_note(
                            w,
                            *id,
                            vec![("scar_intensity", Value::from(*n))],
                        )?);
                    }
                    for (id, n) in &cicatrises_decisions {
                        pub_decisions.extend(decision::poser_proprietes_de_decision(
                            w,
                            *id,
                            vec![("scar_intensity", Value::from(*n))],
                        )?);
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for e in pub_notes {
                    etat.notes.publier_entree(e);
                }
                for e in pub_decisions {
                    etat.decisions.publier_entree(e);
                }
                Ok(rapport)
            },
        )
        .await
    }
}

// ============================================================================
// The lifecycle sweep, and the notes the heuristic wrote
// ============================================================================

/// The authors the heuristic detectors sign with.
const DETECTEURS: [&str; 2] = ["system/observation-detector", "system/rfc-detector"];

fn auto_observee(n: &crate::notes::Note) -> bool {
    DETECTEURS.contains(&n.created_by.as_str()) || n.tags.iter().any(|t| t == "auto-observed")
}

/// The history of a note as the base holds it: an absent `changes_json` is empty, a present one
/// that does not read is an error (never overwritten by an empty history).
fn historique_brut(db: &ObrainDB, noeud: NodeId, id: Uuid) -> Result<Vec<NoteChange>> {
    match propriete(db, noeud, "changes_json")? {
        None => Ok(Vec::new()),
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| anyhow!("note {id} : `changes_json` n'est pas une chaîne"))?;
            if s.trim().is_empty() {
                return Ok(Vec::new());
            }
            serde_json::from_str(s)
                .map_err(|e| anyhow!("note {id} : `changes_json` illisible : {e}"))
        }
    }
}

impl ObrainGraphStore {
    pub(super) async fn memoire_consolider(&self) -> Result<(usize, usize)> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let lifecycle = NoteLifecycleManager::new();
                let now = Utc::now();
                // (sweep energy, note, node, synapse count), the active notes only.
                let mut notes: Vec<(f64, crate::notes::Note, NodeId, i64)> = Vec::new();
                {
                    let etat = w.index()?;
                    refus_si_notes_en_quarantaine(&etat, "consolidate_memory")?;
                    for (n, noeud) in etat.notes.toutes_avec_noeud() {
                        if !statut_brut_actif(w.db, noeud)? {
                            continue;
                        }
                        let energie = nombre_du_noeud(w.db, noeud, "energy")?.unwrap_or(1.0);
                        let synapses = incidentes(w.db, noeud)?.len() as i64;
                        notes.push((energie, n.clone(), noeud, synapses));
                    }
                }
                // `ORDER BY coalesce(energy, 1.0) ASC, id ASC`.
                notes.sort_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then_with(|| a.1.id.to_string().cmp(&b.1.id.to_string()))
                });

                enum Decision {
                    Archiver(String, serde_json::Value),
                    Promouvoir(String, serde_json::Value),
                }
                let mut a_faire: Vec<(Uuid, NodeId, Decision)> = Vec::new();
                for (_, note, noeud, synapses) in &notes {
                    // `reactivation_count` is the counter the activation path maintains.
                    let activations = note
                        .activation_count
                        .max(note.reactivation_count)
                        .max(*synapses);
                    if lifecycle.should_auto_archive(note, activations, now) {
                        let age = now.signed_duration_since(note.created_at).num_days();
                        let raison = if activations == 0 && age > 90 {
                            "dead_note_no_activations_90d"
                        } else {
                            "low_energy_60d"
                        };
                        a_faire.push((
                            note.id,
                            *noeud,
                            Decision::Archiver(
                                "archived".into(),
                                serde_json::json!({"reason": raison, "activation_count": activations, "energy": note.energy, "age_days": age}),
                            ),
                        ));
                        continue;
                    }
                    if lifecycle.should_archive_ephemeral(note, now) {
                        a_faire.push((
                            note.id,
                            *noeud,
                            Decision::Archiver(
                                "archived".into(),
                                serde_json::json!({"reason": "ephemeral_expired", "idle_hours": 48}),
                            ),
                        ));
                        continue;
                    }
                    if note.memory_horizon == MemoryHorizon::Consolidated {
                        continue;
                    }
                    if lifecycle.should_promote_to_consolidated(note, activations, now) {
                        a_faire.push((
                            note.id,
                            *noeud,
                            Decision::Promouvoir(
                                "consolidated".into(),
                                serde_json::json!({
                                    "from": "ephemeral",
                                    "to": "consolidated",
                                    "reason": format!("auto-promote: activations={}, age>14d, energy={:.2}", activations, note.energy)
                                }),
                            ),
                        ));
                        continue;
                    }
                    let promo = lifecycle.evaluate_promotion(note, activations as u64);
                    if let Some(h) = promo.new_horizon {
                        a_faire.push((
                            note.id,
                            *noeud,
                            Decision::Promouvoir(
                                h.to_string(),
                                serde_json::json!({
                                    "from": promo.current_horizon.to_string(),
                                    "to": h.to_string(),
                                    "reason": promo.reason
                                }),
                            ),
                        ));
                    }
                }
                // The histories, read and extended BEFORE the first write: an unreadable
                // one refuses the sweep with nothing written.
                let mut ecrits: Vec<(Uuid, String, &'static str, String)> = Vec::new();
                let (mut promus, mut archives) = (0usize, 0usize);
                for (id, noeud, d) in &a_faire {
                    let mut histoire = historique_brut(w.db, *noeud, *id)?;
                    let (valeur, cle, type_, details) = match d {
                        Decision::Archiver(v, det) => {
                            archives += 1;
                            (v, "status", ChangeType::StatusChanged, det)
                        }
                        Decision::Promouvoir(v, det) => {
                            promus += 1;
                            (v, "memory_horizon", ChangeType::Promoted, det)
                        }
                    };
                    histoire.push(NoteChange::with_details(
                        type_,
                        "consolidate_memory".to_string(),
                        details.clone(),
                    ));
                    ecrits.push((*id, valeur.clone(), cle, serde_json::to_string(&histoire)?));
                }
                let mut publiees = Vec::new();
                ecritures(w, |w| {
                    for (id, valeur, cle, histoire) in &ecrits {
                        // The history first, the effect last.
                        if let Some(e) = note::poser_proprietes_de_note(
                            w,
                            *id,
                            vec![
                                ("changes_json", Value::from(histoire.clone())),
                                (*cle, Value::from(valeur.clone())),
                            ],
                        )? {
                            publiees.push(e);
                        }
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for e in publiees {
                    etat.notes.publier_entree(e);
                }
                Ok((promus, archives))
            },
        )
        .await
    }

    pub(super) async fn memoire_observees_rapport(
        &self,
        projet: Option<Uuid>,
    ) -> Result<Vec<serde_json::Value>> {
        self.lire(|db, etat| {
            refus_si_notes_en_quarantaine(etat, "auto_observed_report")?;
            // (status, note_type, importance) → (notes, total size)
            let mut groupes: BTreeMap<(String, String, String), (i64, i64)> = BTreeMap::new();
            for (n, noeud) in etat.notes.toutes_avec_noeud() {
                if projet.is_some() && n.project_id != projet {
                    continue;
                }
                if !auto_observee(n) {
                    continue;
                }
                let brut = |cle: &str| -> Result<String> {
                    match propriete(db, noeud, cle)? {
                        None => Ok(String::new()),
                        Some(v) => v
                            .as_str()
                            .map(str::to_string)
                            .ok_or_else(|| anyhow!("note {} : `{cle}` n'est pas une chaîne", n.id)),
                    }
                };
                let e = groupes
                    .entry((brut("status")?, brut("note_type")?, brut("importance")?))
                    .or_insert((0, 0));
                e.0 += 1;
                e.1 += n.content.chars().count() as i64;
            }
            let mut lignes: Vec<((String, String, String), (i64, i64))> =
                groupes.into_iter().collect();
            // `ORDER BY notes DESC, status, note_type, importance`: BTreeMap already gave the
            // three strings ascending; the sort is stable.
            lignes.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
            Ok(lignes
                .into_iter()
                .map(|((status, note_type, importance), (notes, total))| {
                    serde_json::json!({
                        "status": status,
                        "note_type": note_type,
                        "importance": importance,
                        "notes": notes,
                        // `toInteger(avg(size(content)))`: truncated.
                        "avg_size": if notes > 0 { total / notes } else { 0 },
                    })
                })
                .collect())
        })
    }

    pub(super) async fn memoire_observees_archiver(&self, projet: Option<Uuid>) -> Result<usize> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut ids: Vec<Uuid> = Vec::new();
                {
                    let etat = w.index()?;
                    refus_si_notes_en_quarantaine(&etat, "archive_unconfirmed_auto_observed")?;
                    for (n, noeud) in etat.notes.toutes_avec_noeud() {
                        if projet.is_some() && n.project_id != projet {
                            continue;
                        }
                        if statut_brut_actif(w.db, noeud)?
                            && n.human_confirmed_at.is_none()
                            && auto_observee(n)
                        {
                            ids.push(n.id);
                        }
                    }
                }
                ids.sort();
                let mut publiees = Vec::new();
                ecritures(w, |w| {
                    for id in &ids {
                        publiees.extend(note::poser_proprietes_de_note(
                            w,
                            *id,
                            vec![("status", Value::from("archived"))],
                        )?);
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for e in publiees {
                    etat.notes.publier_entree(e);
                }
                Ok(ids.len())
            },
        )
        .await
    }
}

// ============================================================================
// The audit
// ============================================================================

/// `left(s, n)`: the first `n` characters.
fn gauche(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `"{id[..8]} — {preview}"`.
fn ligne_d_audit(cle: &str, apercu: &str) -> String {
    format!("{} — {}", gauche(cle, 8), apercu)
}

/// A string property of a node, absent as `None`, a non-string as an error.
fn chaine_brute(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<String>> {
    match propriete(db, noeud, cle)? {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| anyhow!("nœud {noeud:?} : `{cle}` n'est pas une chaîne ({v:?})")),
    }
}

/// The key of a node a gap names: required (Neo4j prints an empty one).
fn cle_requise(db: &ObrainDB, noeud: NodeId, cle: &str, label: &str) -> Result<String> {
    chaine_brute(db, noeud, cle)?
        .ok_or_else(|| anyhow!("audit_knowledge_gaps : nœud {label} {noeud:?} sans `{cle}`"))
}

/// The live nodes of a label of a project (raw `project_id`), drafts left out.
fn noeuds_du_projet(db: &ObrainDB, label: &str, pid: &str) -> Result<Vec<NodeId>> {
    let mut v = Vec::new();
    for noeud in db.store().nodes_by_label(label) {
        if propriete(db, noeud, BROUILLON)?.is_some() {
            continue;
        }
        if chaine_brute(db, noeud, "project_id")?.as_deref() == Some(pid) {
            v.push(noeud);
        }
    }
    Ok(v)
}

/// Whether `noeud` has an edge of `type_` in the direction, and (when asked) from/to a node
/// of one of the `genres`.
fn a_une_arete(
    db: &ObrainDB,
    noeud: NodeId,
    sens: Direction,
    type_: &str,
    genres: &[Genre],
) -> Result<bool> {
    for (voisin, id) in db.store().edges_from(noeud, sens) {
        let e = db
            .get_edge(id)
            .ok_or_else(|| anyhow!("arête {id:?} listée par le moteur mais illisible"))?;
        if &*e.edge_type != type_ {
            continue;
        }
        if genres.is_empty() || genre_de(db, voisin).is_some_and(|g| genres.contains(&g)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `MATCH ()-[r]->() RETURN type(r), count(r)`: every edge by type, from the live nodes.
fn recensement_des_relations(db: &ObrainDB) -> Result<BTreeMap<String, i64>> {
    let store = db.store();
    let mut m: BTreeMap<String, i64> = BTreeMap::new();
    for noeud in store.node_ids() {
        for (_, id) in store.edges_from(noeud, Direction::Outgoing) {
            let e = db
                .get_edge(id)
                .ok_or_else(|| anyhow!("arête {id:?} listée par le moteur mais illisible"))?;
            *m.entry(e.edge_type.to_string()).or_insert(0) += 1;
        }
    }
    Ok(m)
}

impl ObrainGraphStore {
    pub(super) async fn memoire_lacunes(&self, projet: Uuid) -> Result<AuditGapsReport> {
        let pid = projet.to_string();
        self.lire(|db, _| {
            // (sort key, line)
            let mut orphelines: Vec<(String, String)> = Vec::new();
            for noeud in noeuds_du_projet(db, "Note", &pid)? {
                if !a_une_arete(db, noeud, Direction::Outgoing, "LINKED_TO", &[])? {
                    let id = cle_requise(db, noeud, "id", "Note")?;
                    let contenu = chaine_brute(db, noeud, "content")?.unwrap_or_default();
                    orphelines.push((id.clone(), ligne_d_audit(&id, &gauche(&contenu, 80))));
                }
            }
            let mut decisions: Vec<(String, String)> = Vec::new();
            for noeud in noeuds_du_projet(db, "Decision", &pid)? {
                if !a_une_arete(db, noeud, Direction::Outgoing, "AFFECTS", &[])? {
                    let id = cle_requise(db, noeud, "id", "Decision")?;
                    let d = chaine_brute(db, noeud, "description")?.unwrap_or_default();
                    decisions.push((id.clone(), ligne_d_audit(&id, &gauche(&d, 80))));
                }
            }
            let mut commits: Vec<(String, String)> = Vec::new();
            for noeud in noeuds_du_projet(db, "Commit", &pid)? {
                if !a_une_arete(db, noeud, Direction::Outgoing, "TOUCHES", &[])? {
                    let hash = cle_requise(db, noeud, "hash", "Commit")?;
                    let m = chaine_brute(db, noeud, "message")?.unwrap_or_default();
                    commits.push((hash.clone(), ligne_d_audit(&hash, &gauche(&m, 80))));
                }
            }
            let mut skills: Vec<(String, String)> = Vec::new();
            for noeud in noeuds_du_projet(db, "Skill", &pid)? {
                let membre_note =
                    a_une_arete(db, noeud, Direction::Incoming, "MEMBER_OF", &[Genre::Note])?;
                let membre_decision = a_une_arete(
                    db,
                    noeud,
                    Direction::Incoming,
                    "MEMBER_OF_SKILL",
                    &[Genre::Decision],
                )?;
                if !membre_note && !membre_decision {
                    let id = cle_requise(db, noeud, "id", "Skill")?;
                    let nom = chaine_brute(db, noeud, "name")?.unwrap_or_default();
                    skills.push((id.clone(), ligne_d_audit(&id, &nom)));
                }
            }
            let (no, nd, nc, ns) = (
                orphelines.len(),
                decisions.len(),
                commits.len(),
                skills.len(),
            );
            let echantillon = |mut v: Vec<(String, String)>, max: usize| -> Vec<String> {
                v.sort();
                v.into_iter().take(max).map(|(_, l)| l).collect()
            };
            // Prefer the engine's own census when it can walk the edges as `MATCH ()-[r]->()`
            // does; otherwise walk them here.
            let types = match db.store().walked_edge_counts_by_type() {
                Some(m) => m
                    .into_iter()
                    .map(|(k, v)| {
                        i64::try_from(v)
                            .map(|v| (k.clone(), v))
                            .map_err(|_| anyhow!("compte de `{k}` hors i64"))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?,
                None => recensement_des_relations(db)?,
            };
            Ok(AuditGapsReport {
                total_gaps: no + nd + nc + ns,
                orphan_notes: echantillon(orphelines, 100),
                orphan_notes_total: no,
                decisions_without_affects: echantillon(decisions, 100),
                decisions_without_affects_total: nd,
                commits_without_touches: echantillon(commits, 100),
                commits_without_touches_total: nc,
                skills_without_members: echantillon(skills, 50),
                skills_without_members_total: ns,
                relationship_type_counts: types
                    .into_iter()
                    .map(|(rel_type, count)| RelTypeCount { rel_type, count })
                    .collect(),
            })
        })
    }
}
