//! `AnalyticsStore` on obrain, the synaptic layer in SHADOW mode (`src/neo4j/synapse_shadow.rs`,
//! `crate::neurons::synaptic_shadow`): the `shadow_*` properties of the `SYNAPSE` edges between two
//! notes of one project, and the `SynapseShadowTrace` nodes. The methods are declared in
//! `analytics.rs`, which delegates here.
//!
//! # What serves (7)
//!
//! `shadow_synapse_edges`, `reward_shadow_synapses`, `decay_shadow_synapses`,
//! `record_synapse_shadow_trace`, `list_synapse_shadow_traces`, `get_synapse_shadow_trace`,
//! `claim_shadow_trace_reward`. `shadow_trace_closures` is NOT here: it reads `AgentExecution`
//! nodes and the `Task` they point to, which cannot be created without a `Task`
//! (`planrun.rs`); it stays refused by name.
//!
//! # Layout
//!
//! - The shadow state of an edge is five properties of the `SYNAPSE` edge itself
//!   (`shadow_weight`, `shadow_rewarded_at`, `shadow_reward_count`, `shadow_decayed_at`), as
//!   in Neo4j. `weight` is read (the starting value of a shadow weight) and NEVER written; no
//!   edge is created or deleted.
//! - A trace is one node `SynapseShadowTrace` with `id`, `session_id`, `project_id`,
//!   `created_at`, `schema_version`, `layer_version`, `note_ids`, `rewarded` and `payload` (the
//!   JSON of the trace). The index holds every trace decoded, rebuilt at open. The rewards
//!   already claimed live on the node (`rewarded`), written after the payload, and override
//!   the payload's own list, as in Neo4j.
//!
//! # Dates
//!
//! Every stored date goes through `date_propriete` / `date_valeur` (ISO string, `Timestamp`,
//! `ZonedDatetime`); a present but unreadable date refuses the call, it is never read as "no
//! date" (Neo4j's `parse_at` turned it into `None`). Dates are written as RFC 3339 strings.
//!
//! # Differences with the Neo4j adapter, all named
//!
//! - **A weight that is absent or not a number refuses the call** (`coalesce(…, 0.0)` in
//!   Neo4j): the same stance as `synapse.rs`.
//! - **Parallel edges refuse a write** (`reward_shadow_synapses`): Neo4j writes each; this
//!   backend does not guess (as `synapse.rs`).
//! - **A scan refuses when a note was quarantined at open** (it would miss it); so does
//!   every list of traces when a `SynapseShadowTrace` node was quarantined, and
//!   `record_synapse_shadow_trace` (it could miss a trace of the session it trims).
//! - **`record_synapse_shadow_trace` of an id already held refuses** (Neo4j's `CREATE` would
//!   write a second node answering the same id).
//! - **Elapsed time is a whole number of seconds**, truncated toward zero, as
//!   `duration.inSeconds(…).seconds` is; the mock's milliseconds are not the reference.
//! - **The dormant projects** (`decay_shadow_synapses`) are evaluated at the `now` the caller
//!   gives (Neo4j reads its own clock); the projects are those of the project and chat
//!   indexes, as `synapse.rs`.
//! - **Unordered Cypher results are sorted**: `shadow_synapse_edges` answers per requested id
//!   (repeats repeated, as `UNWIND`), targets by id.
//! - A project id compared as the RAW stored string, as Cypher does (`''` is a project id
//!   like another, never "no project").
//!
//! # Atomicity
//!
//! A multi-property update of an edge is a sequence of engine writes under `ecritures`: a cut
//! leaves a prefix, the store refuses everything until reopened, and each property of the
//! prefix is valid on its own. The order is chosen so that a cut can only UNDER-decay or
//! UNDER-count, never over-decay: `shadow_decayed_at` is written BEFORE `shadow_weight` (a
//! cut after it loses the decay of one interval; the reverse order would charge that interval
//! twice), and the property that carries the meaning, `shadow_weight`, last. A trace is
//! created behind a draft marker (a creation that never reached its commit point is deleted at
//! open); the trimming of the older traces of the session comes after, so a cut leaves more
//! than `keep` traces, which the next record trims. A claim is ONE write.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::synapse::{
    aretes_de, entre, fini, genre_de, id_du_noeud, instant, noeud_de_note, nombre_du_noeud,
    projets_endormis, propriete, refus_si_notes_en_quarantaine, sans_doublon, Arete, Genre,
    SYNAPSE,
};
use super::task::ecritures;
use super::trigger::{
    charger, cle_reservee, creer_noeud, refus_reserve, relire, requise, supprimer, texte, Fiche,
    Lecteur,
};
use super::{Ecriture, ObrainGraphStore, RapportOuverture};
use crate::neurons::synaptic_shadow::{
    estimate_tokens, ShadowDecayScope, ShadowEdgeRow, ShadowNode, ShadowTraceFilter,
    SynapseShadowTrace,
};
use obrain_core::graph::Direction;

const OMBRE: &str = "SynapseShadowTrace";

// ============================================================================
// Raw reads
// ============================================================================

/// The `project_id` of a node as the engine holds it: `None` when absent, the raw string
/// otherwise (`''` included: Cypher compares it like any other value).
fn projet_brut(db: &ObrainDB, noeud: NodeId) -> Result<Option<String>> {
    match propriete(db, noeud, "project_id")? {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| anyhow!("nœud {noeud:?} : `project_id` n'est pas une chaîne ({v:?})")),
    }
}

/// The weight an edge starts a shadow weight from: `coalesce(shadow_weight, weight)`.
fn poids_d_ombre(a: &Arete) -> Result<f64> {
    match a.nombre("shadow_weight")? {
        Some(w) => Ok(w),
        None => a.poids(),
    }
}

/// A date property of an edge, through the one decoder.
fn date_d_arete(a: &Arete, cle: &str) -> Result<Option<DateTime<Utc>>> {
    match a.get(cle) {
        None => Ok(None),
        Some(v) => super::date_valeur(v, cle)
            .map_err(|e| anyhow!("arête {SYNAPSE} {:?} → {:?} : {e}", a.de, a.vers)),
    }
}

/// A property of a node, as `Some(&str)` (a non-string is an error).
fn texte_du_noeud(db: &ObrainDB, noeud: NodeId, cle: &str) -> Result<Option<String>> {
    match propriete(db, noeud, cle)? {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| anyhow!("nœud {noeud:?} : `{cle}` n'est pas une chaîne ({v:?})")),
    }
}

// ============================================================================
// Traces: encoding, decoding, the index
// ============================================================================

/// A trace as the index holds it.
struct Ombre {
    trace: SynapseShadowTrace,
    /// The `note_ids` property of the node (what `ContainingNote` reads).
    note_ids: Vec<Uuid>,
    /// The node's `created_at`: what the trimming and the listings order on.
    cree: DateTime<Utc>,
}

fn liste_de_textes(get: Lecteur, cle: &str, quoi: &str) -> Result<Vec<String>> {
    match get(cle) {
        None => Ok(Vec::new()),
        Some(v) => {
            let l = v
                .as_list()
                .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas une liste ({v:?})"))?;
            l.iter()
                .map(|e| {
                    e.as_str().map(str::to_string).ok_or_else(|| {
                        anyhow!("{quoi} : un élément de `{cle}` n'est pas une chaîne ({e:?})")
                    })
                })
                .collect()
        }
    }
}

fn liste(v: Vec<String>) -> Value {
    Value::List(v.into_iter().map(Value::from).collect::<Vec<_>>().into())
}

fn proprietes_ombre(t: &SynapseShadowTrace) -> Result<Vec<(&'static str, Value)>> {
    Ok(vec![
        ("id", texte(t.id.to_string())),
        ("session_id", texte(t.session_id.to_string())),
        ("project_id", texte(t.project_id.to_string())),
        ("created_at", instant(t.created_at)),
        ("schema_version", Value::from(i64::from(t.schema_version))),
        ("layer_version", texte(t.layer_version.clone())),
        (
            "note_ids",
            liste(t.note_ids().iter().map(Uuid::to_string).collect()),
        ),
        ("rewarded", liste(Vec::new())),
        // Last: the property that carries the trace, so a node that reads whole is complete.
        ("payload", texte(serde_json::to_string(t)?)),
    ])
}

fn lire_ombre(get: Lecteur) -> Result<Ombre> {
    let id = requise(get, "id", "nœud SynapseShadowTrace")?;
    let quoi = format!("trace d'ombre {id}");
    let quoi = quoi.as_str();
    let payload = requise(get, "payload", quoi)?;
    let mut trace: SynapseShadowTrace = serde_json::from_str(&payload)
        .map_err(|e| anyhow!("{quoi} : la charge utile n'est pas une trace : {e}"))?;
    if trace.id.to_string() != id {
        bail!(
            "{quoi} : la charge utile est celle de la trace {} : refusée",
            trace.id
        );
    }
    // The claimed rewards live on the node, written after the payload.
    trace.rewarded = liste_de_textes(get, "rewarded", quoi)?;
    let note_ids = liste_de_textes(get, "note_ids", quoi)?
        .into_iter()
        .map(|s| {
            s.parse::<Uuid>()
                .map_err(|e| anyhow!("{quoi} : `note_ids` : « {s} » n'est pas un UUID : {e}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let cree = super::date_propriete(get, "created_at")?
        .ok_or_else(|| anyhow!("{quoi} sans propriété `created_at`"))?;
    Ok(Ombre {
        trace,
        note_ids,
        cree,
    })
}

/// The traces, decoded by id, rebuilt at every open.
#[derive(Default)]
pub(super) struct IndexOmbres {
    par_id: HashMap<Uuid, Fiche<Ombre>>,
    /// The ids a quarantined node still names: never taken again, never claimed.
    reserves: HashSet<String>,
    /// How many `SynapseShadowTrace` nodes are on disk and not served.
    en_quarantaine: usize,
}

impl IndexOmbres {
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let mut index = Self::default();
        for f in charger(db, OMBRE, rapport, &mut index.reserves, &lire_ombre) {
            match index.par_id.entry(f.valeur.trace.id) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud {OMBRE} {:?} : id {} déjà pris",
                        f.noeud, f.valeur.trace.id
                    ))
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(f);
                }
            }
        }
        index.en_quarantaine = db
            .store()
            .nodes_by_label(OMBRE)
            .len()
            .saturating_sub(index.par_id.len());
        index
    }

    fn garde_id(&self, id: Uuid, methode: &str) -> Result<()> {
        if self
            .reserves
            .contains(&cle_reservee(OMBRE, &[&id.to_string()]))
        {
            return Err(refus_reserve(&format!("{methode} de la trace {id}")));
        }
        Ok(())
    }

    fn garde_liste(&self, methode: &str) -> Result<()> {
        if self.en_quarantaine > 0 {
            bail!(
                "{methode} : {} nœud(s) {OMBRE} en quarantaine (illisibles à l'ouverture) : le \
                 résultat les manquerait",
                self.en_quarantaine
            );
        }
        Ok(())
    }
}

// ============================================================================
// The methods
// ============================================================================

impl ObrainGraphStore {
    /// `shadow_synapse_edges`: the outgoing `SYNAPSE` edges of `from` towards the notes of the
    /// project (both ends in it), with the shadow weight and the target's metadata.
    pub(super) async fn ombre_aretes(
        &self,
        projet: Uuid,
        depart: &[Uuid],
    ) -> Result<Vec<ShadowEdgeRow>> {
        if depart.is_empty() {
            return Ok(Vec::new());
        }
        let pid = projet.to_string();
        self.lire(|db, etat| {
            let mut lignes = Vec::new();
            for a in depart {
                let Some(na) = noeud_de_note(etat, *a, "shadow_synapse_edges")? else {
                    continue;
                };
                if projet_brut(db, na)?.as_deref() != Some(pid.as_str()) {
                    continue;
                }
                let mut vues = Vec::new();
                for s in aretes_de(db, na, Direction::Outgoing, SYNAPSE)? {
                    if genre_de(db, s.vers)? != Some(Genre::Note) {
                        continue;
                    }
                    if projet_brut(db, s.vers)?.as_deref() != Some(pid.as_str()) {
                        continue;
                    }
                    let to = id_du_noeud(db, s.vers)?;
                    if to == *a {
                        continue;
                    }
                    vues.push((to, s));
                }
                // Neo4j gives no order; the targets are sorted by id.
                vues.sort_by_key(|(to, _)| *to);
                for (to, s) in vues {
                    let b = s.vers;
                    let statut = texte_du_noeud(db, b, "status")?;
                    let contenu = texte_du_noeud(db, b, "content")?;
                    let cree = match propriete(db, b, "created_at")? {
                        None => None,
                        Some(v) => super::date_valeur(&v, "created_at")
                            .map_err(|e| anyhow!("shadow_synapse_edges : note {to} : {e}"))?,
                    };
                    lignes.push(ShadowEdgeRow {
                        from: *a,
                        to,
                        weight: poids_d_ombre(&s)?,
                        target: ShadowNode {
                            energy: nombre_du_noeud(db, b, "energy")?.unwrap_or(1.0),
                            scar_intensity: nombre_du_noeud(db, b, "scar_intensity")?
                                .unwrap_or(0.0),
                            reachable: matches!(
                                statut.as_deref().unwrap_or("active"),
                                "active" | "needs_review"
                            ),
                            tokens: estimate_tokens(
                                contenu.as_deref().unwrap_or("").chars().count(),
                            ),
                            created_at: cree,
                        },
                    });
                }
            }
            Ok(lignes)
        })
    }

    /// `reward_shadow_synapses`: `delta` added to the shadow weight of the existing edges (both
    /// directions) of each unordered pair of the project, clamped to `[0, 1]`.
    pub(super) async fn ombre_recompenser(
        &self,
        projet: Uuid,
        paires: &[(Uuid, Uuid)],
        delta: f64,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        fini(delta, "delta")?;
        let mut uniques: Vec<(Uuid, Uuid)> = paires
            .iter()
            .filter(|(a, b)| a != b)
            .map(|(a, b)| if a < b { (*a, *b) } else { (*b, *a) })
            .collect();
        uniques.sort();
        uniques.dedup();
        if uniques.is_empty() {
            return Ok(0);
        }
        let pid = projet.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // What is read, before the first write: one edge, its new values.
                let mut a_faire: Vec<(Arete, f64)> = Vec::new();
                {
                    let etat = w.index()?;
                    for (a, b) in &uniques {
                        let (Some(na), Some(nb)) = (
                            noeud_de_note(&etat, *a, "reward_shadow_synapses")?,
                            noeud_de_note(&etat, *b, "reward_shadow_synapses")?,
                        ) else {
                            continue;
                        };
                        if projet_brut(w.db, na)?.as_deref() != Some(pid.as_str())
                            || projet_brut(w.db, nb)?.as_deref() != Some(pid.as_str())
                        {
                            continue;
                        }
                        for (x, y) in [(na, nb), (nb, na)] {
                            sans_doublon(w.db, x, y, SYNAPSE)?;
                            if let Some(s) = entre(w.db, x, y, SYNAPSE)?.into_iter().next() {
                                let p = poids_d_ombre(&s)? + delta;
                                let borne = p.clamp(0.0, 1.0);
                                a_faire.push((s, borne));
                            }
                        }
                    }
                }
                let ecrites = a_faire.len();
                let maintenant = instant(now);
                ecritures(w, |w| {
                    for (s, nouveau) in &a_faire {
                        let compte = s
                            .nombre("shadow_reward_count")?
                            .map_or(0, |c| c as i64)
                            .saturating_add(1);
                        w.etape()?;
                        w.poser_arete(s.id, "shadow_rewarded_at", maintenant.clone());
                        w.etape()?;
                        w.poser_arete(s.id, "shadow_reward_count", Value::from(compte));
                        if s.get("shadow_decayed_at").is_none() {
                            w.etape()?;
                            w.poser_arete(s.id, "shadow_decayed_at", maintenant.clone());
                        }
                        // The meaning, last.
                        w.etape()?;
                        w.poser_arete(s.id, "shadow_weight", Value::from(*nouveau));
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(ecrites)
            },
        )
        .await
    }

    /// `decay_shadow_synapses`: `w · 2^(−Δt / half_life)` on the shadow weights of the projects
    /// in `scope`; a dormant project's edges are frozen (their clock advances, their weight
    /// does not). Returns the number of edges visited.
    pub(super) async fn ombre_oublier(
        &self,
        portee: &ShadowDecayScope,
        demi_vie_jours: f64,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        if !(demi_vie_jours.is_finite() && demi_vie_jours > 0.0) {
            bail!(
                "decay_shadow_synapses : `half_life_days` doit être un nombre strictement positif \
                 (reçu {demi_vie_jours})"
            );
        }
        if !portee.default_enabled && portee.enabled.is_empty() {
            return Ok(0);
        }
        let desactives: HashSet<String> = portee.disabled.iter().map(Uuid::to_string).collect();
        let actives: HashSet<String> = portee.enabled.iter().map(Uuid::to_string).collect();
        let defaut = portee.default_enabled;
        let secondes_de_demi_vie = demi_vie_jours * 86_400.0;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // (edge, weight now, new clock or None to keep, frozen)
                let mut a_faire: Vec<(Arete, f64, Option<DateTime<Utc>>)> = Vec::new();
                {
                    let etat = w.index()?;
                    refus_si_notes_en_quarantaine(&etat, "decay_shadow_synapses")?;
                    let endormis: HashSet<String> = projets_endormis(&etat, now)?
                        .into_iter()
                        .map(|u| u.to_string())
                        .collect();
                    for (_, noeud) in etat.notes.toutes_avec_noeud() {
                        let Some(pa) = projet_brut(w.db, noeud)? else {
                            continue;
                        };
                        if desactives.contains(&pa) || !(defaut || actives.contains(&pa)) {
                            continue;
                        }
                        for s in aretes_de(w.db, noeud, Direction::Outgoing, SYNAPSE)? {
                            if genre_de(w.db, s.vers)? != Some(Genre::Note)
                                || projet_brut(w.db, s.vers)?.as_deref() != Some(pa.as_str())
                            {
                                continue;
                            }
                            let w0 = poids_d_ombre(&s)?;
                            let ecoule = match date_d_arete(&s, "shadow_decayed_at")? {
                                None => 0.0,
                                Some(d) => (now - d).num_seconds() as f64,
                            };
                            let gele = endormis.contains(&pa);
                            let poids = if gele || ecoule <= 0.0 {
                                w0
                            } else {
                                w0 * 0.5_f64.powf(ecoule / secondes_de_demi_vie)
                            };
                            let horloge = if ecoule < 0.0 { None } else { Some(now) };
                            a_faire.push((s, poids, horloge));
                        }
                    }
                }
                let visitees = a_faire.len();
                ecritures(w, |w| {
                    for (s, poids, horloge) in &a_faire {
                        // The clock first: a cut can then only lose one interval of decay.
                        if let Some(h) = horloge {
                            w.etape()?;
                            w.poser_arete(s.id, "shadow_decayed_at", instant(*h));
                        }
                        w.etape()?;
                        w.poser_arete(s.id, "shadow_weight", Value::from(*poids));
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                Ok(visitees)
            },
        )
        .await
    }

    /// `record_synapse_shadow_trace`: persist the trace, keep the latest `garder` of its session.
    pub(super) async fn ombre_enregistrer(
        &self,
        trace: &SynapseShadowTrace,
        garder: usize,
    ) -> Result<()> {
        let t = trace.clone();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let proprietes = proprietes_ombre(&t)?;
                // (created_at, id, node) of every trace of the session, the new one included.
                let mut session: Vec<(DateTime<Utc>, Uuid, NodeId)> = Vec::new();
                {
                    let etat = w.index()?;
                    etat.ombres.garde_liste("record_synapse_shadow_trace")?;
                    etat.ombres.garde_id(t.id, "record_synapse_shadow_trace")?;
                    if etat.ombres.par_id.contains_key(&t.id) {
                        bail!(
                            "record_synapse_shadow_trace : la trace {} existe déjà ; Neo4j en \
                             écrirait un second nœud, ce backend refuse",
                            t.id
                        );
                    }
                    for f in etat.ombres.par_id.values() {
                        if f.valeur.trace.session_id == t.session_id {
                            session.push((f.valeur.cree, f.valeur.trace.id, f.noeud));
                        }
                    }
                }
                let mut fiche = None;
                let mut supprimees: Vec<Uuid> = Vec::new();
                ecritures(w, |w| {
                    let noeud = creer_noeud(w, OMBRE, proprietes)?;
                    let f = w.apres_ecriture(relire(w.db, noeud, &lire_ombre))?;
                    session.push((f.valeur.cree, t.id, noeud));
                    fiche = Some(f);
                    // `ORDER BY created_at DESC, id DESC SKIP keep`.
                    session.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
                    for (_, id, n) in session.iter().skip(garder) {
                        supprimer(w, *n)?;
                        supprimees.push(*id);
                    }
                    Ok(())
                })?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                if let Some(f) = fiche {
                    etat.ombres.par_id.insert(t.id, f);
                }
                for id in supprimees {
                    etat.ombres.par_id.remove(&id);
                }
                Ok(())
            },
        )
        .await
    }

    pub(super) async fn ombre_lister(
        &self,
        filtre: &ShadowTraceFilter,
    ) -> Result<Vec<SynapseShadowTrace>> {
        self.lire(|_, etat| {
            etat.ombres.garde_liste("list_synapse_shadow_traces")?;
            Ok(match filtre {
                ShadowTraceFilter::Session(sid) => {
                    let mut v: Vec<&Fiche<Ombre>> = etat
                        .ombres
                        .par_id
                        .values()
                        .filter(|f| f.valeur.trace.session_id == *sid)
                        .collect();
                    v.sort_by_key(|f| (f.valeur.cree, f.valeur.trace.id));
                    v.into_iter().map(|f| f.valeur.trace.clone()).collect()
                }
                ShadowTraceFilter::ContainingNote { note_id, limit } => {
                    let mut v: Vec<&Fiche<Ombre>> = etat
                        .ombres
                        .par_id
                        .values()
                        .filter(|f| f.valeur.note_ids.contains(note_id))
                        .collect();
                    v.sort_by_key(|f| std::cmp::Reverse((f.valeur.cree, f.valeur.trace.id)));
                    v.into_iter()
                        .take(*limit)
                        .map(|f| f.valeur.trace.clone())
                        .collect()
                }
            })
        })
    }

    pub(super) async fn ombre_lire(&self, id: Uuid) -> Result<Option<SynapseShadowTrace>> {
        self.lire(|_, etat| {
            etat.ombres.garde_id(id, "get_synapse_shadow_trace")?;
            Ok(etat.ombres.par_id.get(&id).map(|f| f.valeur.trace.clone()))
        })
    }

    /// `claim_shadow_trace_reward`: `true` the first time only. ONE engine write, under the
    /// single writer: two claims cannot both win.
    pub(super) async fn ombre_reclamer(&self, id: Uuid, cle: &str) -> Result<bool> {
        let cle = cle.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, deja) = {
                    let etat = w.index()?;
                    etat.ombres.garde_id(id, "claim_shadow_trace_reward")?;
                    let Some(f) = etat.ombres.par_id.get(&id) else {
                        return Ok(false);
                    };
                    (f.noeud, f.valeur.trace.rewarded.clone())
                };
                if deja.contains(&cle) {
                    return Ok(false);
                }
                let mut nouvelle = deja;
                nouvelle.push(cle);
                w.etape()?;
                w.poser(noeud, "rewarded", liste(nouvelle));
                let f = w.apres_ecriture(relire(w.db, noeud, &lire_ombre))?;
                w.rendre_durable()?;
                w.index_mut()?.ombres.par_id.insert(id, f);
                Ok(true)
            },
        )
        .await
    }
}
