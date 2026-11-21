//! `AnalyticsStore` on obrain, sub-lot A2 ("the neural layer of the notes"): the
//! `SYNAPSE` edges between notes (and decisions), the energy and the scars of a
//! note, the deep-maintenance clock of a project. The methods are declared in
//! `analytics.rs`, which delegates here.
//!
//! # What serves (23)
//!
//! - synapses: `create_synapses`, `get_synapses`, `delete_synapses`,
//!   `get_all_synapse_weights`, `get_project_note_synapses`, `get_synapse_graph`,
//!   `reinforce_synapses`, `coactivate_session`, `weaken_node_synapses`,
//!   `create_cross_entity_synapses`, `get_cross_entity_synapses`;
//! - what a note or a decision has not got yet: `list_notes_needing_synapses`,
//!   `list_decisions_needing_synapses`, `get_project_note_entity_links`;
//! - energy and scars: `update_energy_scores`, `boost_energy`, `track_reactivation`,
//!   `init_note_energy`, `apply_scars`, `heal_scars`, `rate_limited_energy_boost`
//!   (a persona's);
//! - `get_deep_maintenance_times`, `mark_deep_maintenance`.
//!
//! # Why this batch
//!
//! One storage mechanism: a `SYNAPSE` edge between two `Note` (or `Decision`) nodes,
//! with its properties on the edge, and numeric/date properties on the node of a note.
//! One dependency set: the note, decision, persona, project and chat indexes, all
//! ported. No Cypher or GQL, no aggregate pushed to the engine: the edges of a node
//! are read from the engine (`edges_from`), so there is no second index to keep in step
//! with the base, and `delete_note` (a `DETACH DELETE`) already takes its `SYNAPSE`
//! edges with it.
//!
//! # Dates
//!
//! Every date read goes through `date_propriete` / `date_valeur` (the three stored
//! forms: ISO string, `Timestamp`, `ZonedDatetime`); a present but unreadable date
//! refuses the call, it is never read as "no date". Dates are WRITTEN as RFC 3339
//! strings (as `note.rs` does), which the decoder reads back.
//!
//! # Differences with the Neo4j adapter, all named
//!
//! - **Reserved ids refuse.** A note or a decision whose node was quarantined at open
//!   (unreadable, or an id taken twice) is neither read nor written by id: the call
//!   refuses by name, as `note.rs` does. A scan of every note (`update_energy_scores`,
//!   `init_note_energy`, the two `*_needing_synapses`, the per-project reads) refuses
//!   when a note was quarantined, as `count_notes_by_consent` does: it would miss it.
//! - **A weight that is absent or not a number refuses the call.** Neo4j skips the row
//!   (`get_all_synapse_weights`, `get_project_note_synapses` with a null weight) or
//!   writes a null (`reinforce`, `weaken`).
//! - **A write that would touch two parallel edges refuses.** Neo4j's `MERGE` never
//!   writes two `SYNAPSE` edges in one direction between two nodes; a migrated base
//!   could carry them, and the count a `MERGE` would then answer is not reproduced.
//!   The same for a doubled `FORGOTTEN` edge on a pair that `reinforce_synapses`
//!   revives. The reads answer what is there.
//! - **A `source` that is not a string, and a `path` that is not a string, refuse.**
//!   `reinforce_synapses` / `coactivate_session` would otherwise read the `source` as
//!   absent and rewrite it `coactivation`; `get_project_note_entity_links` would otherwise
//!   answer the id in place of an unreadable `path`. Both are refused by name, before any
//!   write.
//! - **A coactivation's provenance is bounded** (`rule`, `measure`, `at`: 4096 bytes each):
//!   `AccountableProvenance` bounds nothing, and the edge is created in one small record.
//! - **An unreadable chat session blocks `update_energy_scores`, as a CHOICE.** The dormant
//!   projects (energy frozen) are computed from the last chat activity of each project; a
//!   session that cannot be read could belong to any project and make it look dormant, so
//!   it would be frozen without a word. The whole pass refuses until the session is fixed,
//!   rather than guess a project for it. A decay pass is a maintenance that can wait; a
//!   silently frozen project cannot be told from a healthy one.
//! - **Non-finite numbers refuse** (`boost`, `increment`, `half_life`, a weight):
//!   Neo4j would store them. `update_energy_scores` refuses a `half_life` that is not
//!   strictly positive.
//! - **Unordered Cypher results are sorted** (`get_all_synapse_weights`,
//!   `get_project_note_entity_links`, `get_synapse_graph`): by weight then ids, so two
//!   reads agree. `get_synapse_graph` keeps Neo4j's `LIMIT 10000` after that order.
//! - **Not ported, refused by name** (`analytics.rs`): the forgetting passes
//!   (`decay_synapses`, `decay_project_synapses`, which archive an edge as
//!   `FORGOTTEN` under a cap and a degree floor, and for which the reference store
//!   has no comparable model) and `consolidate_memory` (the note lifecycle sweep).
//!   `reinforce_synapses` and `coactivate_session` do revive a `FORGOTTEN` edge a
//!   migrated base carries, at the weight it had.
//!
//! # Atomicity
//!
//! An edge is created with all its properties in ONE engine record
//! (`create_edge_with_props`). A multi-property update of an edge or of a note is a
//! sequence of engine writes under `ecritures`: a cut leaves a prefix of them, the
//! store refuses everything until reopened (`indetermine`), and each property of the
//! prefix is valid on its own (a weight stays in range, a counter only grows). The
//! property a reinforcement carries the meaning of (the weight) is written last.
//!
//! ONE rule for what is counted: a cut may LOSE an effect, never apply it twice or past its
//! bound. So the stamp or the counter that bounds an effect is written BEFORE the value it
//! guards: `energy_updated_at` before `energy` (a cut loses one decay, the next pass starts
//! from the new stamp; the reverse would decay the same days twice, and the floor could
//! fix 0.0 for good), `energy_boost_accumulated` before `energy` (a cut loses one boost,
//! never exceeds the cap), the counters of an edge before its weight. Archived `FORGOTTEN`
//! edges are deleted AFTER the live edges exist: a cut keeps the archive.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Duration, SecondsFormat, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::task::ecritures;
use super::{decision, note, persona, project, Ecriture, Etat, ObrainGraphStore};
use crate::neo4j::models::DecisionNode;
use crate::notes::{Note, PROJECT_DORMANT_AFTER_DAYS};

pub(super) const SYNAPSE: &str = "SYNAPSE";
pub(super) const OUBLIE: &str = "FORGOTTEN";
/// The longest `rule`, `measure` or `at` of a coactivation's provenance, in bytes.
const BORNE_PROVENANCE: usize = 4096;
/// What a revived archived edge never starts under: `DecayPolicy::default().forget_threshold`.
const PLANCHER_REVIVAL: f64 = 0.02;
/// The property of a `Project` node that holds when its deep maintenance last ran.
const PROJET_DEEP: &str = "last_deep_maintenance_at";

// ============================================================================
// The ends of an edge, and the edges themselves
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Genre {
    Note,
    Decision,
}

impl Genre {
    fn nom(self) -> &'static str {
        match self {
            Genre::Note => "Note",
            Genre::Decision => "Decision",
        }
    }
}

/// A `Note` or `Decision` this backend holds.
#[derive(Clone, Copy, Debug)]
struct Bout {
    noeud: NodeId,
}

/// A note this backend holds: refused when its node was quarantined at open.
pub(super) fn noeud_de_note(etat: &Etat, id: Uuid, methode: &str) -> Result<Option<NodeId>> {
    if etat.notes.est_reservee(id) {
        bail!(
            "{methode} de la note {id} refusé : un nœud Note de cet id est sur disque, illisible \
             ou en double à l'ouverture (quarantaine). Il n'est ni lu ni modifié"
        );
    }
    Ok(etat.notes.noeud_de(id))
}

pub(super) fn noeud_de_decision(etat: &Etat, id: Uuid, methode: &str) -> Result<Option<NodeId>> {
    if etat.decisions.est_reservee(id) {
        bail!(
            "{methode} de la décision {id} refusé : un nœud Decision de cet id est sur disque, \
             illisible ou en double à l'ouverture (quarantaine). Il n'est ni lu ni modifié"
        );
    }
    Ok(etat.decisions.noeud_de(id))
}

/// A note or a decision by id (`MATCH (n {id: $id}) WHERE n:Note OR n:Decision`).
fn bout(etat: &Etat, id: Uuid, methode: &str) -> Result<Option<Bout>> {
    let n = noeud_de_note(etat, id, methode)?;
    let d = noeud_de_decision(etat, id, methode)?;
    match (n, d) {
        (Some(_), Some(_)) => bail!(
            "{methode} : l'id {id} est celui d'une note ET d'une décision ; Neo4j visiterait les \
             deux, ce backend ne devine pas laquelle"
        ),
        (Some(noeud), None) | (None, Some(noeud)) => Ok(Some(Bout { noeud })),
        (None, None) => Ok(None),
    }
}

/// One edge as the engine holds it: its ends and a copy of its properties.
#[derive(Clone, Debug)]
pub(super) struct Arete {
    pub(super) id: EdgeId,
    pub(super) de: NodeId,
    pub(super) vers: NodeId,
    pub(super) p: HashMap<String, Value>,
}

impl Arete {
    pub(super) fn get(&self, cle: &str) -> Option<&Value> {
        self.p.get(cle).filter(|v| !matches!(v, Value::Null))
    }

    /// The weight: a number, or an error (never a default).
    pub(super) fn poids(&self) -> Result<f64> {
        match self.get("weight") {
            None => bail!(
                "arête {SYNAPSE} {:?} → {:?} sans propriété `weight`",
                self.de,
                self.vers
            ),
            Some(v) => v
                .as_float64()
                .or_else(|| v.as_int64().map(|i| i as f64))
                .ok_or_else(|| {
                    anyhow!(
                        "arête {SYNAPSE} {:?} → {:?} : `weight` n'est pas un nombre ({v:?})",
                        self.de,
                        self.vers
                    )
                }),
        }
    }

    /// An optional number property: absent is `None`, a non-number is an error.
    pub(super) fn nombre(&self, cle: &str) -> Result<Option<f64>> {
        match self.get(cle) {
            None => Ok(None),
            Some(v) => v
                .as_float64()
                .or_else(|| v.as_int64().map(|i| i as f64))
                .map(Some)
                .ok_or_else(|| {
                    anyhow!("arête {:?} : `{cle}` n'est pas un nombre ({v:?})", self.id)
                }),
        }
    }

    pub(super) fn texte(&self, cle: &str) -> Option<&str> {
        self.get(cle).and_then(Value::as_str)
    }
}

pub(super) fn lire_arete(db: &ObrainDB, id: EdgeId) -> Result<Arete> {
    let e = db
        .get_edge(id)
        .ok_or_else(|| anyhow!("arête {id:?} listée par le moteur mais illisible"))?;
    Ok(Arete {
        id,
        de: e.src,
        vers: e.dst,
        p: e.properties
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    })
}

/// The edges of `type_` leaving `noeud` (`Outgoing`) or arriving at it (`Incoming`).
pub(super) fn aretes_de(
    db: &ObrainDB,
    noeud: NodeId,
    sens: Direction,
    type_: &str,
) -> Result<Vec<Arete>> {
    let mut v = Vec::new();
    for (_, id) in db.store().edges_from(noeud, sens) {
        let Some(e) = db.get_edge(id) else {
            bail!("arête {id:?} listée par le moteur mais illisible");
        };
        if &*e.edge_type == type_ {
            v.push(lire_arete(db, id)?);
        }
    }
    Ok(v)
}

/// Every `SYNAPSE` edge touching `noeud`, in both directions, each once.
pub(super) fn incidentes(db: &ObrainDB, noeud: NodeId) -> Result<Vec<Arete>> {
    let mut vues: HashSet<EdgeId> = HashSet::new();
    let mut v = Vec::new();
    for sens in [Direction::Outgoing, Direction::Incoming] {
        for a in aretes_de(db, noeud, sens, SYNAPSE)? {
            if vues.insert(a.id) {
                v.push(a);
            }
        }
    }
    Ok(v)
}

/// The edges of `type_` from `de` to `vers`.
pub(super) fn entre(db: &ObrainDB, de: NodeId, vers: NodeId, type_: &str) -> Result<Vec<Arete>> {
    Ok(aretes_de(db, de, Direction::Outgoing, type_)?
        .into_iter()
        .filter(|a| a.vers == vers)
        .collect())
}

/// Refuse, before any write, a pair that carries parallel edges of `type_` in one direction.
pub(super) fn sans_doublon(db: &ObrainDB, de: NodeId, vers: NodeId, type_: &str) -> Result<()> {
    let n = entre(db, de, vers, type_)?.len();
    if n > 1 {
        bail!(
            "{n} arêtes {type_} parallèles de {de:?} vers {vers:?} : le `MERGE` de Neo4j n'en \
             écrit jamais qu'une, ce backend ne devine pas laquelle réécrire"
        );
    }
    Ok(())
}

/// The kind of a node: `None` for a node that is neither a note nor a decision. A node the
/// engine cannot give the labels of (a dangling edge) is an error, never a skipped neighbour.
pub(super) fn genre_de(db: &ObrainDB, noeud: NodeId) -> Result<Option<Genre>> {
    let l = db.get_node_labels(noeud).ok_or_else(|| {
        anyhow!("nœud {noeud:?} voisin d'une arête SYNAPSE mais illisible (arête pendante)")
    })?;
    Ok(if l.iter().any(|e| &**e == "Note") {
        Some(Genre::Note)
    } else if l.iter().any(|e| &**e == "Decision") {
        Some(Genre::Decision)
    } else {
        None
    })
}

/// The `id` property of a node, as a UUID: an error when it is absent or unreadable.
pub(super) fn id_du_noeud(db: &ObrainDB, noeud: NodeId) -> Result<Uuid> {
    db.get_node(noeud)
        .and_then(|n| {
            n.get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
        })
        .ok_or_else(|| anyhow!("nœud {noeud:?} sans propriété `id` lisible"))?
        .parse()
        .map_err(|e| anyhow!("nœud {noeud:?} : `id` n'est pas un UUID : {e}"))
}

/// A property of a node, as the engine holds it (never from an index).
pub(super) fn propriete(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<Value>> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} indexé mais absent de la base"))?;
    Ok(n.get_property(cle)
        .filter(|v| !matches!(v, Value::Null))
        .cloned())
}

pub(super) fn nombre_du_noeud(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<f64>> {
    match propriete(db, noeud, cle)? {
        None => Ok(None),
        Some(v) => v
            .as_float64()
            .or_else(|| v.as_int64().map(|i| i as f64))
            .map(Some)
            .ok_or_else(|| anyhow!("nœud {noeud:?} : `{cle}` n'est pas un nombre ({v:?})")),
    }
}

/// The `project_id` property of a node, `None` when absent or empty.
fn projet_du_noeud(db: &ObrainDB, noeud: NodeId) -> Result<Option<String>> {
    match propriete(db, noeud, "project_id")? {
        None => Ok(None),
        Some(v) => match v.as_str() {
            Some("") => Ok(None),
            Some(s) => Ok(Some(s.to_string())),
            None => bail!("nœud {noeud:?} : `project_id` n'est pas une chaîne ({v:?})"),
        },
    }
}

pub(super) fn maintenant() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true)
}

pub(super) fn instant(d: DateTime<Utc>) -> Value {
    Value::from(d.to_rfc3339_opts(SecondsFormat::Nanos, true))
}

pub(super) fn fini(valeur: f64, nom: &str) -> Result<()> {
    if !valeur.is_finite() {
        bail!("`{nom}` non fini ({valeur}) refusé : Neo4j le stockerait ; divergence assumée");
    }
    Ok(())
}

/// Create the edge `de → vers` with `creation` in one engine record, or update the one that
/// exists with what `mise_a_jour` says of it (one engine write per property, in order).
/// Neo4j's `MERGE … ON CREATE SET … ON MATCH SET`.
fn fusionner(
    w: &mut Ecriture<'_>,
    de: NodeId,
    vers: NodeId,
    creation: Vec<(&'static str, Value)>,
    mise_a_jour: impl FnOnce(&Arete) -> Result<Vec<(&'static str, Value)>>,
) -> Result<()> {
    let existantes = entre(w.db, de, vers, SYNAPSE)?;
    match existantes.as_slice() {
        [] => {
            w.etape()?;
            // hors-poser: an edge whose properties are a few numbers and short strings: never out of the WAL
            w.db.create_edge_with_props(de, vers, SYNAPSE, creation);
        }
        [a] => {
            for (cle, valeur) in mise_a_jour(a)? {
                w.etape()?;
                w.poser_arete(a.id, cle, valeur);
            }
        }
        _ => bail!("arêtes {SYNAPSE} parallèles de {de:?} vers {vers:?} : refusé"),
    }
    Ok(())
}

pub(super) fn supprimer_arete(w: &mut Ecriture<'_>, id: EdgeId, methode: &str) -> Result<()> {
    w.etape()?;
    let supprimee = !w.suppression_d_arete_refusee() && w.db.delete_edge(id);
    if !supprimee {
        w.interne
            .indetermine
            .store(true, std::sync::atomic::Ordering::Release);
        bail!(
            "obrain a refusé la suppression de l'arête {id:?} ({methode}) : base indéterminée, \
             redémarrer"
        );
    }
    Ok(())
}

/// What reviving a pair reads, BEFORE any write: the weight the pair starts from (the one an
/// archived edge had, never under the forget threshold) and the archive rows to delete once
/// the live edges exist (`FOREACH (x IN stale | DELETE x)`).
fn lire_reveil(db: &ObrainDB, a: NodeId, b: NodeId) -> Result<(f64, Vec<EdgeId>)> {
    let f1 = entre(db, a, b, OUBLIE)?;
    let f2 = entre(db, b, a, OUBLIE)?;
    let archive = match (f1.first(), f2.first()) {
        (Some(x), _) if x.get("weight").is_some() => x.nombre("weight")?,
        (_, Some(y)) => y.nombre("weight")?,
        _ => None,
    }
    .unwrap_or(0.5);
    let depart = if archive < PLANCHER_REVIVAL {
        PLANCHER_REVIVAL
    } else {
        archive
    };
    Ok((depart, f1.iter().chain(f2.iter()).map(|x| x.id).collect()))
}

/// Delete the archive rows of a revived pair: LAST, after the live edges, so that a cut
/// leaves the archive (and a retry revives it) rather than neither.
fn oublier_archives(w: &mut Ecriture<'_>, stale: &[EdgeId], methode: &str) -> Result<()> {
    for id in stale {
        supprimer_arete(w, *id, methode)?;
    }
    Ok(())
}

/// Everything a write on the edges of a pair reads, validated BEFORE the first engine write
/// (an error inside `ecritures` would make the whole store indeterminate): every weight
/// readable, the counter a number, `source` a string, the archived weight a number.
fn valider_pair(db: &ObrainDB, a: NodeId, b: NodeId) -> Result<()> {
    for (x, y) in [(a, b), (b, a)] {
        for e in entre(db, x, y, SYNAPSE)? {
            e.poids()?;
            e.nombre("reinforcement_count")?;
            if let Some(v) = e.get("source") {
                if v.as_str().is_none() {
                    bail!(
                        "arête {SYNAPSE} {:?} → {:?} : `source` n'est pas une chaîne ({v:?}) ; \
                         refusé plutôt que réécrit",
                        e.de,
                        e.vers
                    );
                }
            }
        }
    }
    lire_reveil(db, a, b)?;
    Ok(())
}

/// The per-pair checks done before any write: no parallel `SYNAPSE` or `FORGOTTEN` edge.
/// `avec_oubli`: also the archived `FORGOTTEN` edges, for the writers that read them
/// (a revival); `create_synapses` never reads them.
fn pair_sans_doublon(db: &ObrainDB, a: NodeId, b: NodeId, avec_oubli: bool) -> Result<()> {
    for (x, y) in [(a, b), (b, a)] {
        sans_doublon(db, x, y, SYNAPSE)?;
        if avec_oubli {
            sans_doublon(db, x, y, OUBLIE)?;
        }
    }
    if avec_oubli {
        valider_pair(db, a, b)?;
    }
    Ok(())
}

// ============================================================================
// Dormant projects
// ============================================================================

/// The projects with no human activity (code sync nor chat session) for
/// [`PROJECT_DORMANT_AFTER_DAYS`] days: their energy is frozen. As Neo4j's
/// `dormant_project_ids`, from the project and chat indexes.
pub(super) fn projets_endormis(etat: &Etat, now: DateTime<Utc>) -> Result<HashSet<Uuid>> {
    let n = etat.chat.sessions_en_quarantaine();
    if n > 0 {
        bail!(
            "{n} session(s) de chat en quarantaine (illisibles à l'ouverture) : l'activité d'un \
             projet pourrait être sous-estimée, et son énergie gelée à tort"
        );
    }
    let chat = etat.chat.derniere_activite_par_projet();
    let limite = now - Duration::days(PROJECT_DORMANT_AFTER_DAYS);
    let mut v = HashSet::new();
    for p in etat.projets.tous() {
        let active = match (chat.get(&p.slug).copied(), p.last_synced) {
            (None, s) => s,
            (Some(c), None) => Some(c),
            (Some(c), Some(s)) => Some(c.max(s)),
        };
        if active.is_none_or(|a| a < limite) {
            v.insert(p.id);
        }
    }
    Ok(v)
}

pub(super) fn refus_si_notes_en_quarantaine(etat: &Etat, methode: &str) -> Result<()> {
    let n = etat.notes.en_quarantaine();
    if n > 0 {
        bail!(
            "{methode} : {n} nœud(s) Note en quarantaine (illisibles à l'ouverture) : le résultat \
             les manquerait. Neo4j les traite"
        );
    }
    Ok(())
}

pub(super) fn statut_brut_actif(db: &ObrainDB, noeud: NodeId) -> Result<bool> {
    Ok(propriete(db, noeud, "status")?
        .as_ref()
        .and_then(Value::as_str)
        .is_some_and(|s| s == "active"))
}

// ============================================================================
// The methods
// ============================================================================

impl ObrainGraphStore {
    // ---- synapses ----------------------------------------------------------

    pub(super) async fn synapses_creer(
        &self,
        note_id: Uuid,
        voisins: &[(Uuid, f64)],
    ) -> Result<usize> {
        self.synapses_creer_entre(note_id, voisins, false).await
    }

    /// `create_synapses` (`croise: false`, two `Note`s) and `create_cross_entity_synapses`
    /// (`croise: true`, a `Note` or a `Decision` at each end).
    pub(super) async fn synapses_creer_entre(
        &self,
        source: Uuid,
        voisins: &[(Uuid, f64)],
        croise: bool,
    ) -> Result<usize> {
        if voisins.is_empty() {
            return Ok(0);
        }
        if let Some((nid, p)) = voisins.iter().find(|(_, p)| !p.is_finite()) {
            bail!("non-finite synapse weight {p} for neighbor {nid}");
        }
        let voisins = voisins.to_vec();
        let methode = if croise {
            "create_cross_entity_synapses"
        } else {
            "create_synapses"
        };
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (src, cibles) = {
                    let etat = w.index()?;
                    let trouver = |id: Uuid| -> Result<Option<NodeId>> {
                        if croise {
                            Ok(bout(&etat, id, methode)?.map(|b| b.noeud))
                        } else {
                            noeud_de_note(&etat, id, methode)
                        }
                    };
                    let Some(src) = trouver(source)? else {
                        return Ok(0);
                    };
                    let mut cibles = Vec::new();
                    for (id, p) in &voisins {
                        if let Some(n) = trouver(*id)? {
                            cibles.push((n, *p));
                        }
                    }
                    (src, cibles)
                };
                for (n, _) in &cibles {
                    pair_sans_doublon(w.db, src, *n, false)?;
                }
                let mut lignes = 0usize;
                ecritures(w, |w| {
                    for (cible, p) in &cibles {
                        for (de, vers) in [(src, *cible), (*cible, src)] {
                            fusionner(
                                w,
                                de,
                                vers,
                                vec![
                                    ("weight", Value::from(*p)),
                                    ("created_at", Value::from(maintenant())),
                                    ("source", Value::from("cosine")),
                                ],
                                |_| Ok(vec![("weight", Value::from(*p))]),
                            )?;
                            lignes += 1;
                        }
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(lignes)
            },
        )
        .await
    }

    pub(super) async fn synapses_lire(&self, note_id: Uuid) -> Result<Vec<(Uuid, f64)>> {
        self.lire(move |db, etat| {
            let Some(n) = noeud_de_note(etat, note_id, "get_synapses")? else {
                return Ok(Vec::new());
            };
            // `DISTINCT neighbor.id, s.weight`: a neighbour linked in both directions with
            // two different weights answers twice.
            let mut lignes: BTreeSet<(Uuid, u64)> = BTreeSet::new();
            let mut poids: BTreeMap<u64, f64> = BTreeMap::new();
            for a in incidentes(db, n)? {
                let autre = if a.de == n { a.vers } else { a.de };
                if genre_de(db, autre)? != Some(Genre::Note) {
                    continue;
                }
                let p = a.poids()?;
                lignes.insert((id_du_noeud(db, autre)?, p.to_bits()));
                poids.insert(p.to_bits(), p);
            }
            let mut v: Vec<(Uuid, f64)> =
                lignes.into_iter().map(|(id, b)| (id, poids[&b])).collect();
            // ORDER BY weight DESC, neighbor_id DESC.
            v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
            Ok(v)
        })
    }

    pub(super) async fn synapses_supprimer(&self, note_id: Uuid) -> Result<usize> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let aretes = {
                    let etat = w.index()?;
                    let Some(n) = noeud_de_note(&etat, note_id, "delete_synapses")? else {
                        return Ok(0);
                    };
                    incidentes(w.db, n)?
                };
                let total = aretes.len();
                ecritures(w, |w| {
                    for a in &aretes {
                        supprimer_arete(w, a.id, "delete_synapses")?;
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(total)
            },
        )
        .await
    }

    pub(super) async fn synapses_poids(&self, projet: Option<Uuid>) -> Result<Vec<f64>> {
        self.lire(move |db, etat| {
            let mut v = Vec::new();
            match projet {
                Some(pid) => {
                    refus_si_notes_en_quarantaine(etat, "get_all_synapse_weights")?;
                    if etat.projets.noeud_de(pid).is_none() {
                        return Ok(v);
                    }
                    // (p:Project)-[:HAS_NOTE]->(n1:Note)-[s:SYNAPSE]->(n2:Note)
                    for (_, noeud) in etat.notes.du_projet(pid) {
                        for a in aretes_de(db, noeud, Direction::Outgoing, SYNAPSE)? {
                            if genre_de(db, a.vers)? == Some(Genre::Note) {
                                v.push(a.poids()?);
                            }
                        }
                    }
                }
                None => {
                    // MATCH ()-[s:SYNAPSE]->(): every SYNAPSE edge. Only a note or a
                    // decision carries one.
                    for etiquette in ["Note", "Decision"] {
                        for noeud in db.store().nodes_by_label(etiquette) {
                            for a in aretes_de(db, noeud, Direction::Outgoing, SYNAPSE)? {
                                v.push(a.poids()?);
                            }
                        }
                    }
                }
            }
            v.sort_by(|a, b| a.total_cmp(b));
            Ok(v)
        })
    }

    pub(super) async fn synapses_du_projet(
        &self,
        projet: Uuid,
        poids_min: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.lire(move |db, etat| {
            refus_si_notes_en_quarantaine(etat, "get_project_note_synapses")?;
            let notes = etat.notes.du_projet(projet);
            let dans: HashMap<NodeId, Uuid> = notes.iter().map(|(id, n)| (*n, *id)).collect();
            let mut v: Vec<(String, String, f64)> = Vec::new();
            for (a_id, noeud) in &notes {
                for s in aretes_de(db, *noeud, Direction::Outgoing, SYNAPSE)? {
                    let Some(b_id) = dans.get(&s.vers) else {
                        continue;
                    };
                    let p = s.poids()?;
                    // `s.weight >= $min_weight AND a.id < b.id`
                    if p >= poids_min && a_id.to_string() < b_id.to_string() {
                        v.push((a_id.to_string(), b_id.to_string(), p));
                    }
                }
            }
            // ORDER BY s.weight DESC, source_id DESC, target_id DESC.
            v.sort_by(|a, b| {
                b.2.total_cmp(&a.2)
                    .then_with(|| b.0.cmp(&a.0))
                    .then_with(|| b.1.cmp(&a.1))
            });
            Ok(v)
        })
    }

    /// The `SYNAPSE` graph skill detection reads: current notes of one project, from the
    /// edge to the edge, weight over `poids_min`, the audit-trail notes of skill evolution
    /// left out.
    pub(super) async fn synapses_graphe(
        &self,
        projet: Uuid,
        poids_min: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.lire(move |db, etat| {
            refus_si_notes_en_quarantaine(etat, "get_synapse_graph")?;
            let notes = etat.notes.du_projet(projet);
            let mut admis: HashMap<NodeId, Uuid> = HashMap::new();
            for (id, noeud) in &notes {
                let Some((n, _, _)) = etat.notes.consulter(*id) else {
                    continue;
                };
                // The raw status and `superseded_by`, as the Cypher reads them.
                let courant = propriete(db, *noeud, "status")?
                    .as_ref()
                    .and_then(Value::as_str)
                    .is_some_and(|s| s == "active" || s == "needs_review");
                let remplacee = propriete(db, *noeud, "superseded_by")?.is_some();
                if courant && !remplacee && n.created_by != "skill-evolution" {
                    admis.insert(*noeud, *id);
                }
            }
            let mut v: Vec<(String, String, f64)> = Vec::new();
            for (a_id, noeud) in admis.iter().map(|(n, i)| (*i, *n)) {
                for s in aretes_de(db, noeud, Direction::Outgoing, SYNAPSE)? {
                    let Some(b_id) = admis.get(&s.vers) else {
                        continue;
                    };
                    let p = s.poids()?;
                    if p > poids_min {
                        v.push((a_id.to_string(), b_id.to_string(), p));
                    }
                }
            }
            v.sort_by(|a, b| {
                b.2.total_cmp(&a.2)
                    .then_with(|| a.0.cmp(&b.0))
                    .then_with(|| a.1.cmp(&b.1))
            });
            v.truncate(10_000);
            Ok(v)
        })
    }

    pub(super) async fn synapses_renforcer(&self, ids: &[Uuid], boost: f64) -> Result<usize> {
        if ids.len() < 2 {
            bail!(
                "reinforce_synapses needs at least 2 note ids (got {})",
                ids.len()
            );
        }
        fini(boost, "boost")?;
        let ids = ids.to_vec();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let paires = Self::paires_de_notes(w, &ids, "reinforce_synapses", |a, b| {
                    // `a.project_id IS NOT NULL AND a.project_id <> '' AND
                    // a.project_id = b.project_id AND a.id <> b.id`
                    a.0 != b.0 && a.1.is_some() && a.1 == b.1
                })?;
                let n = paires.len();
                ecritures(w, |w| {
                    for (a, b) in &paires {
                        let (depart, stale) = lire_reveil(w.db, *a, *b)?;
                        let now = maintenant();
                        for (de, vers) in [(*a, *b), (*b, *a)] {
                            fusionner(
                                w,
                                de,
                                vers,
                                vec![
                                    ("weight", Value::from(depart)),
                                    ("created_at", Value::from(now.clone())),
                                    ("last_reinforced_at", Value::from(now.clone())),
                                    ("reinforcement_count", Value::from(1_i64)),
                                    ("source", Value::from("coactivation")),
                                ],
                                |s| {
                                    let p = s.poids()?;
                                    let compte = s
                                        .nombre("reinforcement_count")?
                                        .map_or(0, |c| c as i64)
                                        .saturating_add(1);
                                    let source = match s.texte("source") {
                                        None | Some("cosine") => "coactivation".to_string(),
                                        Some(autre) => autre.to_string(),
                                    };
                                    Ok(vec![
                                        ("last_reinforced_at", Value::from(now.clone())),
                                        ("decayed_at", Value::from(now.clone())),
                                        ("reinforcement_count", Value::from(compte)),
                                        ("source", Value::from(source)),
                                        (
                                            "weight",
                                            Value::from(if p + boost > 1.0 {
                                                1.0
                                            } else {
                                                p + boost
                                            }),
                                        ),
                                    ])
                                },
                            )?;
                        }
                        oublier_archives(w, &stale, "reinforce_synapses")?;
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                // Two directed synapses per pair that was really reinforced.
                Ok(n * 2)
            },
        )
        .await
    }

    /// `coactivate_session`: the pair logic of `reinforce_synapses`, scoped to `projet`, with a
    /// fixed `+0.2` and the provenance on each edge.
    pub(super) async fn synapses_coactiver(
        &self,
        projet: Uuid,
        notes: &[Uuid],
        session: Uuid,
        regle: String,
        mesure: String,
        date: String,
    ) -> Result<usize> {
        if notes.len() < 2 {
            return Ok(0);
        }
        // `AccountableProvenance` proves the three are non-empty and bounds nothing; they are
        // written in the one record that creates an edge, which must stay small.
        for (nom, v) in [("rule", &regle), ("measure", &mesure), ("at", &date)] {
            if v.len() > BORNE_PROVENANCE {
                bail!(
                    "coactivate_session : `{nom}` de la provenance fait {} octets, au plus \
                     {BORNE_PROVENANCE} sont admis (la provenance est une courte déclaration)",
                    v.len()
                );
            }
        }
        let notes = notes.to_vec();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let paires = Self::paires_de_notes(w, &notes, "coactivate_session", |a, b| {
                    // `a.project_id = $project AND b.project_id = $project AND a.id <> b.id`
                    a.0 != b.0 && a.1 == Some(projet) && b.1 == Some(projet)
                })?;
                let n = paires.len();
                ecritures(w, |w| {
                    for (a, b) in &paires {
                        let (depart, stale) = lire_reveil(w.db, *a, *b)?;
                        let now = maintenant();
                        for (k, (de, vers)) in [(*a, *b), (*b, *a)].into_iter().enumerate() {
                            let provenance = |mut v: Vec<(&'static str, Value)>| {
                                v.push(("rule", Value::from(regle.clone())));
                                v.push(("measure", Value::from(mesure.clone())));
                                v.push(("at", Value::from(date.clone())));
                                v.push(("last_session", Value::from(session.to_string())));
                                v
                            };
                            // The first direction also stamps `decayed_at` (created or matched).
                            let mut creation = vec![
                                ("weight", Value::from(depart)),
                                ("created_at", Value::from(now.clone())),
                                ("reinforcement_count", Value::from(1_i64)),
                                ("source", Value::from("coactivation")),
                                ("last_reinforced_at", Value::from(now.clone())),
                            ];
                            if k == 0 {
                                creation.push(("decayed_at", Value::from(now.clone())));
                            }
                            fusionner(w, de, vers, provenance(creation), |s| {
                                let p = s.poids()?;
                                let compte = s
                                    .nombre("reinforcement_count")?
                                    .map_or(0, |c| c as i64)
                                    .saturating_add(1);
                                let mut v = vec![
                                    ("reinforcement_count", Value::from(compte)),
                                    ("last_reinforced_at", Value::from(now.clone())),
                                ];
                                if k == 0 {
                                    v.push(("decayed_at", Value::from(now.clone())));
                                }
                                let mut v = provenance(v);
                                v.push((
                                    "weight",
                                    Value::from(if p + 0.2 > 1.0 { 1.0 } else { p + 0.2 }),
                                ));
                                Ok(v)
                            })?;
                        }
                        oublier_archives(w, &stale, "coactivate_session")?;
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(n * 2)
            },
        )
        .await
    }

    /// The `(a, b)` node pairs `i < j` of `ids` that both exist as notes and satisfy `garde`
    /// (given each side's `(id, project)`), after the checks that must precede any write.
    #[allow(clippy::type_complexity)]
    fn paires_de_notes(
        w: &Ecriture<'_>,
        ids: &[Uuid],
        methode: &str,
        garde: impl Fn((Uuid, Option<Uuid>), (Uuid, Option<Uuid>)) -> bool,
    ) -> Result<Vec<(NodeId, NodeId)>> {
        let etat = w.index()?;
        let mut tenues: HashMap<Uuid, (NodeId, Option<Uuid>)> = HashMap::new();
        for id in ids {
            if let Some(n) = noeud_de_note(&etat, *id, methode)? {
                let projet = etat.notes.consulter(*id).and_then(|(n, _, _)| n.project_id);
                tenues.insert(*id, (n, projet));
            }
        }
        let mut paires = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (Some(a), Some(b)) = (tenues.get(&ids[i]), tenues.get(&ids[j])) else {
                    continue;
                };
                if garde((ids[i], a.1), (ids[j], b.1)) {
                    paires.push((a.0, b.0));
                }
            }
        }
        for (a, b) in &paires {
            pair_sans_doublon(w.db, *a, *b, true)?;
        }
        Ok(paires)
    }

    pub(super) async fn synapses_affaiblir(
        &self,
        id: Uuid,
        quantite: f64,
        seuil: f64,
    ) -> Result<usize> {
        fini(quantite, "amount")?;
        fini(seuil, "prune_threshold")?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, aretes) = {
                    let etat = w.index()?;
                    let Some(b) = bout(&etat, id, "weaken_node_synapses")? else {
                        return Ok(0);
                    };
                    (b.noeud, incidentes(w.db, b.noeud)?)
                };
                let mien = projet_du_noeud(w.db, noeud)?;
                // A synapse that crosses two known projects is left alone.
                let mut retenues: Vec<(Arete, f64)> = Vec::new();
                for a in aretes {
                    let autre = if a.de == noeud { a.vers } else { a.de };
                    let leur = projet_du_noeud(w.db, autre)?;
                    let permis = match (&mien, &leur) {
                        (Some(x), Some(y)) => x == y,
                        _ => true,
                    };
                    if permis {
                        let p = a.poids()?;
                        retenues.push((a, p));
                    }
                }
                let n = retenues.len();
                ecritures(w, |w| {
                    for (a, p) in &retenues {
                        let nouveau = p - quantite;
                        if nouveau < seuil {
                            supprimer_arete(w, a.id, "weaken_node_synapses")?;
                        } else {
                            w.etape()?;
                            w.poser_arete(a.id, "weight", Value::from(nouveau));
                        }
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(n)
            },
        )
        .await
    }

    pub(super) async fn synapses_croisees(&self, id: Uuid) -> Result<Vec<(Uuid, f64, String)>> {
        self.lire(move |db, etat| {
            let Some(b) = bout(etat, id, "get_cross_entity_synapses")? else {
                return Ok(Vec::new());
            };
            let mut lignes: BTreeSet<(Uuid, u64, &'static str)> = BTreeSet::new();
            for a in incidentes(db, b.noeud)? {
                let autre = if a.de == b.noeud { a.vers } else { a.de };
                let Some(genre) = genre_de(db, autre)? else {
                    continue;
                };
                lignes.insert((id_du_noeud(db, autre)?, a.poids()?.to_bits(), genre.nom()));
            }
            let mut v: Vec<(Uuid, f64, String)> = lignes
                .into_iter()
                .map(|(i, p, g)| (i, f64::from_bits(p), g.to_string()))
                .collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
            Ok(v)
        })
    }

    pub(super) async fn notes_sans_synapse(
        &self,
        limite: usize,
        decalage: usize,
        projet: Option<Uuid>,
    ) -> Result<(Vec<Note>, usize)> {
        self.lire(move |db, etat| {
            refus_si_notes_en_quarantaine(etat, "list_notes_needing_synapses")?;
            if projet.is_some_and(|p| etat.projets.noeud_de(p).is_none()) {
                return Ok((Vec::new(), 0));
            }
            let mut v: Vec<&Note> = Vec::new();
            for (n, noeud) in etat.notes.toutes_avec_noeud() {
                if projet.is_some() && n.project_id != projet {
                    continue;
                }
                let courant = propriete(db, noeud, "status")?
                    .as_ref()
                    .and_then(Value::as_str)
                    .is_some_and(|s| s == "active" || s == "needs_review");
                if !courant {
                    continue;
                }
                // `n.embedding IS NOT NULL AND NOT (n)-[:SYNAPSE]->()`
                if propriete(db, noeud, "embedding")?.is_none() {
                    continue;
                }
                if !aretes_de(db, noeud, Direction::Outgoing, SYNAPSE)?.is_empty() {
                    continue;
                }
                v.push(n);
            }
            v.sort_by(|a, b| {
                a.created_at
                    .cmp(&b.created_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
            let total = v.len();
            if total == 0 || limite == 0 {
                return Ok((Vec::new(), total));
            }
            Ok((
                v.into_iter().skip(decalage).take(limite).cloned().collect(),
                total,
            ))
        })
    }

    pub(super) async fn decisions_sans_synapse(
        &self,
        limite: usize,
        decalage: usize,
    ) -> Result<(Vec<DecisionNode>, usize)> {
        self.lire(move |db, etat| {
            let n = etat.decisions.en_quarantaine();
            if n > 0 {
                bail!(
                    "list_decisions_needing_synapses : {n} nœud(s) Decision en quarantaine \
                     (illisibles à l'ouverture) : le résultat les manquerait"
                );
            }
            let mut v: Vec<&DecisionNode> = Vec::new();
            for (d, noeud) in etat.decisions.toutes_avec_noeud() {
                if propriete(db, noeud, "embedding")?.is_none() {
                    continue;
                }
                if !aretes_de(db, noeud, Direction::Outgoing, SYNAPSE)?.is_empty() {
                    continue;
                }
                v.push(d);
            }
            v.sort_by(|a, b| {
                a.decided_at
                    .cmp(&b.decided_at)
                    .then_with(|| a.id.cmp(&b.id))
            });
            let total = v.len();
            if total == 0 || limite == 0 {
                return Ok((Vec::new(), total));
            }
            Ok((
                v.into_iter().skip(decalage).take(limite).cloned().collect(),
                total,
            ))
        })
    }

    pub(super) async fn liens_de_notes_du_projet(
        &self,
        projet: Uuid,
    ) -> Result<Vec<(String, String, String)>> {
        self.lire(move |db, etat| {
            refus_si_notes_en_quarantaine(etat, "get_project_note_entity_links")?;
            let mut v: Vec<(String, String, String)> = Vec::new();
            for (note, entite) in etat.notes.liens_du_projet(projet)? {
                // `COALESCE(e.path, e.id, ...)`: the target's `path` property when it has
                // one, else its id. The type is `unknown`: a `Project` or a `Note` is none of
                // the code entities (File, Function, Struct, Trait, Enum) the Cypher names.
                let cible = entite
                    .parse::<Uuid>()
                    .ok()
                    .and_then(|u| etat.notes.noeud_de(u).or_else(|| etat.projets.noeud_de(u)));
                let id = match cible {
                    Some(n) => match propriete(db, n, "path")? {
                        Some(p) => p
                            .as_str()
                            .map(str::to_string)
                            .ok_or_else(|| anyhow!("nœud {n:?} : `path` n'est pas une chaîne"))?,
                        None => entite,
                    },
                    None => entite,
                };
                // Neo4j drops a row with an empty id or entity id.
                if !id.is_empty() {
                    v.push((note.to_string(), "unknown".to_string(), id));
                }
            }
            v.sort();
            Ok(v)
        })
    }

    // ---- energy and scars --------------------------------------------------

    pub(super) async fn energie_mettre_a_jour(&self, demi_vie: f64) -> Result<usize> {
        if !(demi_vie.is_finite() && demi_vie > 0.0) {
            bail!(
                "update_energy_scores : `half_life_days` doit être un nombre strictement positif \
                 (reçu {demi_vie})"
            );
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let now = Utc::now();
                // What is read, from the indexes and the engine, before the first write.
                let mut a_faire: Vec<(Uuid, f64, f64)> = Vec::new();
                {
                    let etat = w.index()?;
                    refus_si_notes_en_quarantaine(&etat, "update_energy_scores")?;
                    let endormis = projets_endormis(&etat, now)?;
                    for (n, noeud) in etat.notes.toutes_avec_noeud() {
                        // `n.status = 'active' AND n.energy > 0.0` on the raw properties.
                        if !statut_brut_actif(w.db, noeud)? {
                            continue;
                        }
                        let Some(energie) = nombre_du_noeud(w.db, noeud, "energy")? else {
                            continue;
                        };
                        if energie.is_nan() || energie <= 0.0 {
                            continue;
                        }
                        let get = |cle: &str| {
                            w.db.get_node(noeud).and_then(|x| {
                                x.get_property(cle)
                                    .filter(|v| !matches!(v, Value::Null))
                                    .cloned()
                            })
                        };
                        let stampe = super::date_propriete(&get, "energy_updated_at")?;
                        let dort = n.project_id.is_some_and(|p| endormis.contains(&p));
                        let jours = match stampe {
                            Some(s) if !dort => (now - s).num_milliseconds() as f64 / 86_400_000.0,
                            _ => 0.0,
                        };
                        let nouvelle = energie * (-jours / demi_vie).exp();
                        let ramenee = if nouvelle < 0.05 { 0.0 } else { nouvelle };
                        a_faire.push((n.id, energie, ramenee));
                    }
                }
                let mut publiees = Vec::new();
                let mut mises_a_jour = 0usize;
                ecritures(w, |w| {
                    for (id, avant, apres) in &a_faire {
                        // The stamp first, the energy last (the one rule, see the header): a
                        // cut between the two loses this decay, the next pass takes it from
                        // the new stamp; it can never decay the same days twice.
                        let mut p = vec![("energy_updated_at", Value::from(now.to_rfc3339()))];
                        if (avant - apres).abs() > 0.001 {
                            p.push(("energy", Value::from(*apres)));
                            mises_a_jour += 1;
                        }
                        if let Some(e) = note::poser_proprietes_de_note(w, *id, p)? {
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
                Ok(mises_a_jour)
            },
        )
        .await
    }

    pub(super) async fn energie_renforcer(&self, id: Uuid, quantite: f64) -> Result<()> {
        fini(quantite, "amount")?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let noeud = {
                    let etat = w.index()?;
                    match noeud_de_note(&etat, id, "boost_energy")? {
                        Some(n) => n,
                        None => return Ok(()),
                    }
                };
                // `coalesce(n.energy, 1.0)`: the raw property, not the index.
                let courante = nombre_du_noeud(w.db, noeud, "energy")?.unwrap_or(1.0);
                let nouvelle = if courante + quantite > 1.0 {
                    1.0
                } else {
                    courante + quantite
                };
                let now = Utc::now().to_rfc3339();
                let entree = ecritures(w, |w| {
                    note::poser_proprietes_de_note(
                        w,
                        id,
                        vec![
                            ("last_activated", Value::from(now.clone())),
                            ("energy_updated_at", Value::from(now.clone())),
                            ("energy", Value::from(nouvelle)),
                        ],
                    )
                })?;
                w.rendre_durable()?;
                if let Some(e) = entree {
                    w.index_mut()?.notes.publier_entree(e);
                }
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn reactivations_suivre(&self, ids: &[Uuid]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let mut fois: BTreeMap<Uuid, i64> = BTreeMap::new();
        for id in ids {
            *fois.entry(*id).or_insert(0) += 1;
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut a_faire: Vec<(Uuid, i64, i64)> = Vec::new();
                {
                    let etat = w.index()?;
                    for (id, n) in &fois {
                        if noeud_de_note(&etat, *id, "track_reactivation")?.is_some() {
                            let courant = etat
                                .notes
                                .consulter(*id)
                                .map_or(0, |(note, _, _)| note.reactivation_count);
                            a_faire.push((*id, *n, courant));
                        }
                    }
                }
                let now = Utc::now().to_rfc3339();
                let mut publiees = Vec::new();
                let mut lignes = 0usize;
                ecritures(w, |w| {
                    for (id, n, courant) in &a_faire {
                        // One row per id of the list: a repeated id counts, and adds, each time.
                        let e = note::poser_proprietes_de_note(
                            w,
                            *id,
                            vec![
                                ("last_reactivated", Value::from(now.clone())),
                                ("reactivation_count", Value::from(courant + n)),
                            ],
                        )?;
                        if let Some(e) = e {
                            publiees.push(e);
                            lignes += *n as usize;
                        }
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for e in publiees {
                    etat.notes.publier_entree(e);
                }
                Ok(lignes)
            },
        )
        .await
    }

    pub(super) async fn energie_initialiser(&self, projet: Option<Uuid>) -> Result<usize> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut a_faire: Vec<(Uuid, Value)> = Vec::new();
                {
                    let etat = w.index()?;
                    refus_si_notes_en_quarantaine(&etat, "init_note_energy")?;
                    for (n, noeud) in etat.notes.toutes_avec_noeud() {
                        if projet.is_some() && n.project_id != projet {
                            continue;
                        }
                        // `n.energy IS NULL`: absent on the node, which the decoded note
                        // (energy defaults to 1.0) cannot tell.
                        if propriete(w.db, noeud, "energy")?.is_some() {
                            continue;
                        }
                        // `coalesce(n.last_confirmed_at, n.created_at, datetime())`, as stored.
                        let depart = match propriete(w.db, noeud, "last_confirmed_at")? {
                            Some(v) => v,
                            None => match propriete(w.db, noeud, "created_at")? {
                                Some(v) => v,
                                None => instant(Utc::now()),
                            },
                        };
                        a_faire.push((n.id, depart));
                    }
                }
                let n = a_faire.len();
                let mut publiees = Vec::new();
                ecritures(w, |w| {
                    for (id, depart) in &a_faire {
                        // The energy last: a cut leaves a note that the next call finishes.
                        let e = note::poser_proprietes_de_note(
                            w,
                            *id,
                            vec![
                                ("last_activated", depart.clone()),
                                ("energy", Value::from(1.0_f64)),
                            ],
                        )?;
                        publiees.extend(e);
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for e in publiees {
                    etat.notes.publier_entree(e);
                }
                Ok(n)
            },
        )
        .await
    }

    pub(super) async fn cicatrices_appliquer(&self, ids: &[Uuid], increment: f64) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        fini(increment, "increment")?;
        let mut fois: BTreeMap<Uuid, usize> = BTreeMap::new();
        for id in ids {
            *fois.entry(*id).or_insert(0) += 1;
        }
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let mut notes: Vec<(Uuid, f64, usize)> = Vec::new();
                let mut decisions: Vec<(Uuid, f64, usize)> = Vec::new();
                {
                    let etat = w.index()?;
                    for (id, n) in &fois {
                        if noeud_de_note(&etat, *id, "apply_scars")?.is_some() {
                            let s = etat
                                .notes
                                .consulter(*id)
                                .map_or(0.0, |(x, _, _)| x.scar_intensity);
                            notes.push((*id, s, *n));
                        }
                        if noeud_de_decision(&etat, *id, "apply_scars")?.is_some() {
                            let s = etat
                                .decisions
                                .decision(*id)
                                .map_or(0.0, |d| d.scar_intensity);
                            decisions.push((*id, s, *n));
                        }
                    }
                }
                // One row per id of the list: a repeated id is scarred each time, capped at 1.0.
                let monter = |depart: f64, n: usize| {
                    let mut s = depart;
                    for _ in 0..n {
                        s = if s + increment > 1.0 {
                            1.0
                        } else {
                            s + increment
                        };
                    }
                    s
                };
                let total = notes.iter().chain(decisions.iter()).map(|x| x.2).sum();
                let mut pub_notes = Vec::new();
                let mut pub_decisions = Vec::new();
                ecritures(w, |w| {
                    for (id, s, n) in &notes {
                        let e = note::poser_proprietes_de_note(
                            w,
                            *id,
                            vec![("scar_intensity", Value::from(monter(*s, *n)))],
                        )?;
                        pub_notes.extend(e);
                    }
                    for (id, s, n) in &decisions {
                        let e = decision::poser_proprietes_de_decision(
                            w,
                            *id,
                            vec![("scar_intensity", Value::from(monter(*s, *n)))],
                        )?;
                        pub_decisions.extend(e);
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
                Ok(total)
            },
        )
        .await
    }

    pub(super) async fn cicatrices_guerir(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (note_tenue, decision_tenue) = {
                    let etat = w.index()?;
                    (
                        noeud_de_note(&etat, id, "heal_scars")?.is_some(),
                        noeud_de_decision(&etat, id, "heal_scars")?.is_some(),
                    )
                };
                // Note first; the decision only when it is not a note.
                if note_tenue {
                    let e = ecritures(w, |w| {
                        note::poser_proprietes_de_note(
                            w,
                            id,
                            vec![("scar_intensity", Value::from(0.0_f64))],
                        )
                    })?;
                    w.rendre_durable()?;
                    if let Some(e) = e {
                        w.index_mut()?.notes.publier_entree(e);
                    }
                    return Ok(true);
                }
                if decision_tenue {
                    let e = ecritures(w, |w| {
                        decision::poser_proprietes_de_decision(
                            w,
                            id,
                            vec![("scar_intensity", Value::from(0.0_f64))],
                        )
                    })?;
                    w.rendre_durable()?;
                    if let Some(e) = e {
                        w.index_mut()?.decisions.publier_entree(e);
                    }
                    return Ok(true);
                }
                Ok(false)
            },
        )
        .await
    }

    pub(super) async fn persona_energie_limitee(
        &self,
        persona_id: Uuid,
        boost: f64,
        plafond: f64,
    ) -> Result<bool> {
        fini(boost, "boost")?;
        fini(plafond, "max_per_cycle")?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(e) = Self::entree_persona(w, persona_id)? else {
                    return Ok(false);
                };
                let accumule = e.persona.energy_boost_accumulated;
                // `COALESCE(acc, 0.0) + $boost <= $max`
                if accumule + boost > plafond {
                    return Ok(false);
                }
                let energie = e.persona.energy;
                let proprietes = vec![
                    ("updated_at", Value::from(Utc::now().to_rfc3339())),
                    // The accumulator BEFORE the energy: a cut charges a boost that was not
                    // given (a boost lost), and can never give a boost that is not charged
                    // (the cap exceeded).
                    ("energy_boost_accumulated", Value::from(accumule + boost)),
                    (
                        "energy",
                        Value::from(if energie + boost > 1.0 {
                            1.0
                        } else {
                            energie + boost
                        }),
                    ),
                ];
                ecritures(w, |w| {
                    persona::ecrire_proprietes_de_persona(w, persona_id, proprietes)
                })?;
                Ok(true)
            },
        )
        .await
    }

    // ---- deep maintenance ----------------------------------------------------

    pub(super) async fn maintenance_profonde_lire(&self) -> Result<Vec<(Uuid, DateTime<Utc>)>> {
        self.lire(|db, _| {
            // Every `Project` node of the base, readable or not: an unreadable one refuses.
            let mut v = Vec::new();
            for noeud in db.store().nodes_by_label("Project") {
                let get = |cle: &str| {
                    db.get_node(noeud).and_then(|n| {
                        n.get_property(cle)
                            .filter(|v| !matches!(v, Value::Null))
                            .cloned()
                    })
                };
                let Some(at) = super::date_propriete(&get, PROJET_DEEP)? else {
                    continue;
                };
                v.push((id_du_noeud(db, noeud)?, at));
            }
            v.sort();
            Ok(v)
        })
    }

    pub(super) async fn maintenance_profonde_marquer(&self, projet: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                project::ecrire_propriete_de_projet(w, projet, PROJET_DEEP, instant(Utc::now()))?;
                Ok(())
            },
        )
        .await
    }
}
