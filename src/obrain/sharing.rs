//! `SharingStore` on obrain — the sharing policy of a project, the consent of a
//! note, the audit trail of sharing events, and the signed tombstones. Every
//! method of the contract is written here: it serves, or it refuses by name
//! (`NonPorte`). Nothing is answered with an empty or default value in place of
//! a refusal.
//!
//! Where it lives, as in the Neo4j adapter (`src/neo4j/sharing.rs`, the
//! reference):
//!
//! - the policy is the JSON property `sharing_policy` of the `Project` node
//!   (`project.rs` decodes it into `ProjectNode`; the write here re-publishes
//!   the project);
//! - the consent of a note is the property `sharing_consent` of the `Note` node
//!   (`note.rs` decodes it; the write here re-publishes the note);
//! - an event is a `SharingEvent` node, a tombstone a `Tombstone` node keyed by
//!   `content_hash` (Neo4j's `MERGE`).
//!
//! # Refused by name
//!
//! Listed in [`REFUSEES`]:
//!
//! - `update_decision_sharing_consent`: the Decision domain is not ported;
//! - `list_sharing_events`: an event carries no link to its project, so the
//!   events "of a project" cannot be told apart. Neo4j ignores `project_id` and
//!   lists every project's events, and the history handler checks no owner:
//!   reproducing it would leak one project's audit trail into another's
//!   (reasoning taken from an earlier, uncommitted port, branch
//!   `sauvegarde/sharing-session-af008d00`). Events are WRITTEN (and read back
//!   at each write and at open), so nothing is lost until they can be served.
//!
//! # Differences with Neo4j, all on the strict side or named
//!
//! - an event whose id is taken is refused (Neo4j `CREATE`s a second node);
//! - timestamps are RFC 3339 strings, read back as written (Neo4j writes the
//!   event `timestamp` as a `datetime` and reads it as a string, which fails and
//!   falls back to "now"). An event without a readable `timestamp` or `id` is
//!   quarantined at open;
//! - an event `consent` / `privacy_mode` that is not one of the known values is
//!   quarantined (Neo4j: the default, silently);
//! - `count_notes_by_consent` refuses (an error, not `NonPorte`) when notes were
//!   quarantined at open: Neo4j counts every `Note` node, and this backend
//!   cannot place an unreadable one;
//! - `get_sharing_consent` / `update_sharing_consent` of a note that is on disk
//!   but quarantined refuse (Neo4j reads / writes it);
//! - two `Tombstone` nodes for one hash (an imported base) reserve that hash:
//!   `is_tombstoned` of it answers an error (Neo4j counts every node; serving the
//!   first alone could hide the real revocation behind a placeholder), and
//!   `list_tombstones` refuses while any tombstone is quarantined;
//! - a note whose `sharing_consent` is present but unreadable is quarantined by
//!   `note.rs` (Neo4j reads it `NotSet`, which lets it be shared);
//! - `list_tombstones` comes in `content_hash` order (Neo4j: no order).
//!
//! Kept from Neo4j, and named here: `update_sharing_policy` / `update_sharing_consent`
//! of an unknown project or note write nothing and answer `Ok(())` (Neo4j's
//! `MATCH … SET`). And a consequence of the strict delete of `project.rs`: a
//! `SharingEvent`, `Tombstone` (or `Skill`, `PublishedSkill`) node in the base makes
//! `delete_project` refuse by name (`ProjectStore::delete_project (cascade)`), as any
//! node outside the projects and plans already does.
//!
//! Kept from Neo4j, because they are its stated contract: a tombstone whose
//! `issued_at` does not parse reads as the Unix epoch, deterministic and
//! unverifiable (`tombstone_from_stored`), so that a revocation is never lost;
//! `get_sharing_consent` of an unknown note is `NotSet`.
//!
//! # Atomicity
//!
//! An event and a new tombstone are created under the draft marker
//! `_brouillon`, removed last; opening deletes any draft. Rewriting an existing
//! tombstone sets its properties one by one (Neo4j's `MERGE … SET`).
//!
//! A value above `SEUIL_COLONNE_BLOB` (an event `reason`, a policy JSON…), or the
//! replacement of one, pays the full flush, before the marker for a creation — the
//! motif of `note.rs` / `plan.rs` (`flush_complet`), until obrain journals those
//! values (#1190). The policy and consent writes go through the helpers of
//! `project.rs` / `note.rs`, which apply the same rule.

use std::collections::{BTreeMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::skill::{ancienne_en_colonne_blob, en_colonne_blob};
use super::{note, project, Ecriture, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::episodes::distill_models::{
    ConsentCounts, PrivacyMode, SharingConsent, SharingEvent, SharingPolicy,
};
use crate::neo4j::traits::SharingStore;
use crate::reception::anchor::SignedTombstone;

const EVENEMENT: &str = "SharingEvent";
const TOMBE: &str = "Tombstone";
const BROUILLON: &str = "_brouillon";

/// The methods of `SharingStore` refused on every call, by name.
pub const REFUSEES: &[&str] = &[
    "SharingStore::update_decision_sharing_consent",
    // An event carries no project: see the method.
    "SharingStore::list_sharing_events",
];

// ============================================================================
// Encoding and decoding
// ============================================================================

/// A string property: absent is `None`, present with another type is an error.
fn chaine(get: &dyn Fn(&str) -> Option<Value>, quoi: &str, cle: &str) -> Result<Option<String>> {
    match get(cle) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| anyhow!("{quoi} : `{cle}` n'est pas une chaîne")),
    }
}

/// A serde enum written as its bare string (`"explicit_allow"`): absent is the
/// default, an unknown value is an error.
fn enumere<T: serde::de::DeserializeOwned + Default>(
    get: &dyn Fn(&str) -> Option<Value>,
    quoi: &str,
    cle: &str,
) -> Result<T> {
    match chaine(get, quoi, cle)? {
        None => Ok(T::default()),
        Some(s) => serde_json::from_value(serde_json::Value::String(s.clone()))
            .map_err(|_| anyhow!("{quoi} : `{cle}` inconnu « {s} »")),
    }
}

/// The bare serde string of an enum (`PrivacyMode::Standard` → `standard`).
fn chaine_serde<T: serde::Serialize>(v: &T) -> Result<String> {
    match serde_json::to_value(v)? {
        serde_json::Value::String(s) => Ok(s),
        autre => bail!("valeur sérialisée inattendue : {autre}"),
    }
}

fn proprietes_evenement(ev: &SharingEvent) -> Result<Vec<(&'static str, Value)>> {
    if ev.id.is_empty() {
        bail!("événement de partage sans id refusé : il ne pourrait pas être adressé");
    }
    Ok(vec![
        ("id", Value::from(ev.id.clone())),
        ("content_hash", Value::from(ev.content_hash.clone())),
        ("artifact_type", Value::from(ev.artifact_type.clone())),
        ("action", Value::from(ev.action.clone())),
        ("source_did", Value::from(ev.source_did.clone())),
        ("target_did", Value::from(ev.target_did.clone())),
        (
            "timestamp",
            Value::from(ev.timestamp.to_rfc3339_opts(SecondsFormat::Nanos, true)),
        ),
        ("consent", Value::from(ev.consent.as_db_str())),
        ("privacy_mode", Value::from(chaine_serde(&ev.privacy_mode)?)),
        ("reason", Value::from(ev.reason.clone().unwrap_or_default())),
    ])
}

fn lire_evenement(get: &dyn Fn(&str) -> Option<Value>) -> Result<SharingEvent> {
    let id = chaine(get, "nœud SharingEvent", "id")?
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("nœud SharingEvent sans `id`"))?;
    let quoi = format!("événement {id}");
    let texte = |cle: &str| -> Result<String> { Ok(chaine(get, &quoi, cle)?.unwrap_or_default()) };
    let brut =
        chaine(get, &quoi, "timestamp")?.ok_or_else(|| anyhow!("{quoi} sans `timestamp`"))?;
    let timestamp = brut
        .parse::<DateTime<Utc>>()
        .map_err(|e| anyhow!("{quoi} : `timestamp` illisible ({brut}) : {e}"))?;
    let consent: SharingConsent = enumere(get, &quoi, "consent")?;
    let privacy_mode: PrivacyMode = enumere(get, &quoi, "privacy_mode")?;
    Ok(SharingEvent {
        content_hash: texte("content_hash")?,
        artifact_type: texte("artifact_type")?,
        action: texte("action")?,
        source_did: texte("source_did")?,
        target_did: texte("target_did")?,
        timestamp,
        consent,
        privacy_mode,
        reason: Some(texte("reason")?).filter(|r| !r.is_empty()),
        id,
    })
}

fn proprietes_tombe(t: &SignedTombstone) -> Vec<(&'static str, Value)> {
    vec![
        ("content_hash", Value::from(t.content_hash.clone())),
        ("issuer_did", Value::from(t.issuer_did.clone())),
        ("signature_hex", Value::from(t.signature_hex.clone())),
        // To the nanosecond: `issued_at` is covered by the signature.
        (
            "issued_at",
            Value::from(t.issued_at.to_rfc3339_opts(SecondsFormat::Nanos, true)),
        ),
        ("reason", Value::from(t.reason.clone().unwrap_or_default())),
    ]
}

fn lire_tombe(get: &dyn Fn(&str) -> Option<Value>) -> Result<SignedTombstone> {
    let hash = chaine(get, "nœud Tombstone", "content_hash")?
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("nœud Tombstone sans `content_hash`"))?;
    let quoi = format!("tombe {hash}");
    let texte = |cle: &str| -> Result<String> { Ok(chaine(get, &quoi, cle)?.unwrap_or_default()) };
    Ok(SignedTombstone {
        issuer_did: texte("issuer_did")?,
        signature_hex: texte("signature_hex")?,
        // Neo4j's stated rule (`tombstone_from_stored`): the epoch, never "now".
        issued_at: texte("issued_at")?
            .parse::<DateTime<Utc>>()
            .unwrap_or(DateTime::UNIX_EPOCH),
        reason: Some(texte("reason")?).filter(|r| !r.is_empty()),
        content_hash: hash,
    })
}

/// `is_tombstoned`'s filter: a real signature (not empty, not all zeros) by a
/// named issuer. Legacy placeholders never count.
fn tombe_effective(t: &SignedTombstone) -> bool {
    !t.signature_hex.chars().all(|c| c == '0')
        && !t.issuer_did.is_empty()
        && t.issuer_did != "did:local:unknown"
}

// ============================================================================
// Index
// ============================================================================

struct EntreeTombe {
    noeud: NodeId,
    tombe: SignedTombstone,
}

/// The index of the sharing domain (events and tombstones; the policy and the
/// consent live in the project and note indexes).
#[derive(Default)]
pub(super) struct IndexPartage {
    /// Ids of the events this backend holds (decoded at open and after each write:
    /// what is written is readable). The events themselves are not served — see
    /// `list_sharing_events`.
    evenements: HashSet<String>,
    /// Event ids of nodes quarantined at open: not reusable.
    evenements_reserves: HashSet<String>,
    tombes: BTreeMap<String, EntreeTombe>,
    /// Hashes of tombstone nodes quarantined at open: a MERGE would hit them.
    tombes_reservees: HashSet<String>,
}

/// At open: the property lookup of a node, or `None` when it is a draft
/// (deleted, with its edges) or unreadable (quarantined).
fn lecture_ou_brouillon(
    db: &ObrainDB,
    noeud: NodeId,
    etiquette: &str,
    rapport: &mut RapportOuverture,
) -> Option<impl Fn(&str) -> Option<Value>> {
    let store = db.store();
    let Some(n) = db.get_node(noeud) else {
        rapport
            .en_quarantaine
            .push(format!("nœud {etiquette} {noeud:?} : illisible"));
        return None;
    };
    let brouillon = n
        .get_property(BROUILLON)
        .is_some_and(|v| !matches!(v, Value::Null))
        || (n.properties.is_empty() && store.out_degree(noeud) == 0);
    if brouillon {
        for (_, arete) in store.edges_from(noeud, Direction::Outgoing) {
            db.delete_edge(arete);
        }
        for (_, arete) in store.edges_from(noeud, Direction::Incoming) {
            db.delete_edge(arete);
        }
        db.delete_node(noeud);
        rapport.brouillons_supprimes += 1;
        return None;
    }
    Some(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

fn lecteur(db: &ObrainDB, noeud: NodeId) -> Result<impl Fn(&str) -> Option<Value>> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} indexé mais absent de la base"))?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

impl IndexPartage {
    /// Rebuild from the persisted nodes at open.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Result<Self> {
        let mut index = Self::default();
        for noeud in db.store().nodes_by_label(EVENEMENT) {
            let Some(get) = lecture_ou_brouillon(db, noeud, EVENEMENT, rapport) else {
                continue;
            };
            match lire_evenement(&get) {
                Ok(ev) if index.evenements.contains(&ev.id) => {
                    index.evenements_reserves.insert(ev.id.clone());
                    rapport.en_quarantaine.push(format!(
                        "nœud SharingEvent {noeud:?} : id {} déjà pris",
                        ev.id
                    ));
                }
                Ok(ev) => {
                    index.evenements.insert(ev.id);
                }
                Err(e) => {
                    if let Some(id) = get("id").and_then(|v| v.as_str().map(str::to_string)) {
                        index.evenements_reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud SharingEvent {noeud:?} : {e}"));
                }
            }
        }
        for noeud in db.store().nodes_by_label(TOMBE) {
            let Some(get) = lecture_ou_brouillon(db, noeud, TOMBE, rapport) else {
                continue;
            };
            match lire_tombe(&get) {
                // Two nodes for one hash (an imported base: Neo4j's MERGE has no
                // uniqueness constraint). Neo4j's `is_tombstoned` counts every node;
                // serving the first alone could hide the real revocation behind a
                // placeholder. The hash is reserved: its reads answer an error.
                Ok(t) if index.tombes.contains_key(&t.content_hash) => {
                    index.tombes_reservees.insert(t.content_hash.clone());
                    rapport.en_quarantaine.push(format!(
                        "nœud Tombstone {noeud:?} : content_hash {} déjà pris",
                        t.content_hash
                    ));
                }
                Ok(t) => {
                    index
                        .tombes
                        .insert(t.content_hash.clone(), EntreeTombe { noeud, tombe: t });
                }
                Err(e) => {
                    if let Some(h) =
                        get("content_hash").and_then(|v| v.as_str().map(str::to_string))
                    {
                        index.tombes_reservees.insert(h);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Tombstone {noeud:?} : {e}"));
                }
            }
        }
        Ok(index)
    }
}

/// A note that is on disk but quarantined: neither read nor written here.
fn note_en_quarantaine(notes: &note::IndexNotes, id: Uuid) -> Result<()> {
    if notes.est_reservee(id) {
        bail!(
            "la note {id} est sur disque mais en quarantaine (illisible à l'ouverture) : son \
             consentement n'est ni lu ni écrit ici"
        );
    }
    Ok(())
}

// ============================================================================
// SharingStore
// ============================================================================

#[async_trait]
impl SharingStore for ObrainGraphStore {
    async fn get_sharing_policy(&self, project_id: Uuid) -> Result<Option<SharingPolicy>> {
        self.lire(|db, etat| {
            let Some(noeud) = etat.projets.noeud_de(project_id) else {
                return Ok(None);
            };
            // Read raw, and strictly: a policy that does not parse is an error, as
            // in Neo4j (the project decoding itself reads it leniently).
            let get = lecteur(db, noeud)?;
            let Some(json) = chaine(&get, "projet", "sharing_policy")?.filter(|s| !s.is_empty())
            else {
                return Ok(None);
            };
            let politique = serde_json::from_str(&json).map_err(|e| {
                anyhow!("projet {project_id} : politique de partage illisible : {e}")
            })?;
            Ok(Some(politique))
        })
    }

    async fn update_sharing_policy(&self, project_id: Uuid, policy: &SharingPolicy) -> Result<()> {
        let json = serde_json::to_string(policy)?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                project::ecrire_propriete_de_projet(
                    w,
                    project_id,
                    "sharing_policy",
                    Value::from(json),
                )?;
                Ok(())
            },
        )
        .await
    }

    async fn get_sharing_consent(&self, note_id: Uuid) -> Result<SharingConsent> {
        self.lire(|_, etat| {
            note_en_quarantaine(&etat.notes, note_id)?;
            Ok(etat
                .notes
                .consulter(note_id)
                .map(|(n, _, _)| n.sharing_consent)
                .unwrap_or_default())
        })
    }

    async fn update_sharing_consent(&self, note_id: Uuid, consent: &SharingConsent) -> Result<()> {
        let valeur = consent.as_db_str();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                note_en_quarantaine(&w.index()?.notes, note_id)?;
                note::ecrire_propriete_de_note(w, note_id, "sharing_consent", Value::from(valeur))?;
                Ok(())
            },
        )
        .await
    }

    async fn update_decision_sharing_consent(
        &self,
        _decision_id: Uuid,
        _consent: &SharingConsent,
    ) -> Result<()> {
        Err(
            anyhow::Error::new(NonPorte("SharingStore::update_decision_sharing_consent"))
                .context("le consentement est une propriété du nœud Decision : domaine non porté"),
        )
    }

    async fn count_notes_by_consent(&self, project_id: Option<Uuid>) -> Result<ConsentCounts> {
        self.lire(|_, etat| {
            let illisibles = etat.notes.en_quarantaine();
            if illisibles > 0 {
                bail!(
                    "{illisibles} nœud(s) Note en quarantaine (illisibles à l'ouverture) : le \
                     compte par consentement les manquerait. Neo4j les compte"
                );
            }
            let mut compte = ConsentCounts::default();
            for n in etat.notes.toutes() {
                if project_id.is_none() || n.project_id == project_id {
                    compte.add(n.sharing_consent, 1);
                }
            }
            Ok(compte)
        })
    }

    async fn create_sharing_event(&self, event: &SharingEvent) -> Result<()> {
        let ev = event.clone();
        self.ecrire(
            move || ev,
            move |ev, w: &mut Ecriture<'_>| {
                let proprietes = proprietes_evenement(&ev)?;
                {
                    let etat = w.index()?;
                    if etat.partage.evenements.contains(&ev.id) {
                        bail!("un événement de partage d'id {} existe déjà", ev.id);
                    }
                    if etat.partage.evenements_reserves.contains(&ev.id) {
                        bail!(
                            "l'id {} est réservé : un nœud SharingEvent de cet id est sur \
                             disque, illisible à l'ouverture (quarantaine)",
                            ev.id
                        );
                    }
                }
                w.flush_complet = proprietes.iter().any(|(_, v)| en_colonne_blob(v));
                let noeud = w.db.create_node(&[EVENEMENT]);
                w.etape()?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                for (cle, valeur) in proprietes {
                    w.etape()?;
                    w.db.set_node_property(noeud, cle, valeur);
                }
                w.rendre_durable_avant_le_marqueur()?;
                w.etape()?;
                w.db.remove_node_property(noeud, BROUILLON);
                let lu = lecteur(w.db, noeud).and_then(|get| lire_evenement(&get));
                let evenement = w.apres_ecriture(lu)?;
                w.rendre_durable()?;
                w.index_mut()?.partage.evenements.insert(evenement.id);
                Ok(())
            },
        )
        .await
    }

    /// Refused by name: an event carries no project, so "the events of this
    /// project" cannot be answered. Neo4j ignores `project_id` and returns the
    /// events of EVERY project, and `get_sharing_history` checks no ownership:
    /// reproducing it would show one project the audit trail of the others.
    async fn list_sharing_events(
        &self,
        _project_id: Uuid,
        _limit: i64,
        _offset: i64,
    ) -> Result<Vec<SharingEvent>> {
        Err(anyhow::Error::new(NonPorte("SharingStore::list_sharing_events")).context(
            "un SharingEvent ne porte pas son projet : la liste « du projet » serait celle de              tous les projets (ce que fait Neo4j, sans contrôle de propriétaire). Refusé jusqu'à              ce que l'événement soit rattaché à son projet",
        ))
    }

    async fn persist_tombstone(&self, tombstone: &SignedTombstone) -> Result<()> {
        let t = tombstone.clone();
        self.ecrire(
            move || t,
            move |t, w: &mut Ecriture<'_>| {
                if t.content_hash.is_empty() {
                    bail!("tombe sans content_hash refusée : elle ne pourrait pas être adressée");
                }
                let existante = {
                    let etat = w.index()?;
                    if etat.partage.tombes_reservees.contains(&t.content_hash) {
                        bail!(
                            "une tombe de content_hash {} est sur disque, illisible à \
                             l'ouverture (quarantaine) : elle n'est ni reprise ni doublée",
                            t.content_hash
                        );
                    }
                    etat.partage.tombes.get(&t.content_hash).map(|e| e.noeud)
                };
                let proprietes = proprietes_tombe(&t);
                w.flush_complet = proprietes.iter().any(|(_, v)| en_colonne_blob(v))
                    || existante.is_some_and(|n| {
                        ancienne_en_colonne_blob(w.db, n, proprietes.iter().map(|(k, _)| *k))
                    });
                let noeud = match existante {
                    // MERGE on the hash, then SET: `content_hash` itself stays.
                    Some(n) => {
                        for (cle, valeur) in proprietes.into_iter().skip(1) {
                            w.etape()?;
                            w.db.set_node_property(n, cle, valeur);
                        }
                        n
                    }
                    None => {
                        let n = w.db.create_node(&[TOMBE]);
                        w.etape()?;
                        w.db.set_node_property(n, BROUILLON, Value::from(true));
                        for (cle, valeur) in proprietes {
                            w.etape()?;
                            w.db.set_node_property(n, cle, valeur);
                        }
                        w.rendre_durable_avant_le_marqueur()?;
                        w.etape()?;
                        w.db.remove_node_property(n, BROUILLON);
                        n
                    }
                };
                let lu = lecteur(w.db, noeud).and_then(|get| lire_tombe(&get));
                let tombe = w.apres_ecriture(lu)?;
                w.rendre_durable()?;
                w.index_mut()?
                    .partage
                    .tombes
                    .insert(tombe.content_hash.clone(), EntreeTombe { noeud, tombe });
                Ok(())
            },
        )
        .await
    }

    async fn list_tombstones(&self) -> Result<Vec<SignedTombstone>> {
        self.lire(|_, etat| {
            let illisibles = etat.partage.tombes_reservees.len();
            if illisibles > 0 {
                bail!(
                    "{illisibles} tombe(s) en quarantaine (illisibles ou en double à \
                     l'ouverture) : la liste les manquerait"
                );
            }
            Ok(etat
                .partage
                .tombes
                .values()
                .map(|e| e.tombe.clone())
                .collect())
        })
    }

    async fn is_tombstoned(&self, content_hash: &str) -> Result<bool> {
        self.lire(|_, etat| {
            if etat.partage.tombes_reservees.contains(content_hash) {
                bail!(
                    "la tombe de {content_hash} est sur disque mais illisible ou en double \
                     (quarantaine) : répondre « non révoqué » pourrait être faux"
                );
            }
            Ok(etat
                .partage
                .tombes
                .get(content_hash)
                .is_some_and(|e| tombe_effective(&e.tombe)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn depuis(props: Vec<(&'static str, Value)>) -> impl Fn(&str) -> Option<Value> {
        move |cle: &str| {
            props
                .iter()
                .find(|(k, _)| *k == cle)
                .map(|(_, v)| v.clone())
        }
    }

    fn evenement() -> SharingEvent {
        SharingEvent {
            id: "ev-1".into(),
            content_hash: "h".into(),
            artifact_type: "note".into(),
            action: "shared".into(),
            source_did: "did:a".into(),
            target_did: "broadcast".into(),
            timestamp: Utc::now(),
            consent: SharingConsent::ExplicitAllow,
            privacy_mode: PrivacyMode::Strict,
            reason: Some("pourquoi".into()),
        }
    }

    #[test]
    fn un_evenement_ecrit_se_relit_a_l_identique() {
        let ev = evenement();
        let relu = lire_evenement(&depuis(proprietes_evenement(&ev).unwrap())).unwrap();
        assert_eq!(
            serde_json::to_value(&relu).unwrap(),
            serde_json::to_value(&ev).unwrap()
        );
    }

    #[test]
    fn un_evenement_illisible_n_a_pas_de_valeur_par_defaut() {
        let base = proprietes_evenement(&evenement()).unwrap();
        for (cle, mauvaise) in [
            ("timestamp", Some(Value::from("hier"))),
            ("timestamp", None),
            ("consent", Some(Value::from("peut-etre"))),
            ("privacy_mode", Some(Value::from("paranoia"))),
            ("id", None),
        ] {
            let mut p: Vec<(&'static str, Value)> =
                base.iter().filter(|(k, _)| *k != cle).cloned().collect();
            if let Some(v) = mauvaise {
                p.push((cle, v));
            }
            assert!(lire_evenement(&depuis(p)).is_err(), "{cle} accepté");
        }
    }

    #[test]
    fn une_tombe_garde_sa_signature_et_les_fausses_ne_comptent_pas() {
        let id = crate::identity::InstanceIdentity::generate();
        let t = crate::sharing::tombstone::sign_tombstone(
            &id,
            "hash".into(),
            Utc::now(),
            Some("rgpd".into()),
        );
        let relue = lire_tombe(&depuis(proprietes_tombe(&t))).unwrap();
        assert_eq!(relue.issued_at, t.issued_at);
        assert!(crate::sharing::tombstone::verify_tombstone(&relue));
        assert!(tombe_effective(&relue));

        let mut fausse = t.clone();
        fausse.signature_hex = "0000".into();
        assert!(!tombe_effective(&fausse));
        fausse.signature_hex = String::new();
        assert!(!tombe_effective(&fausse));
        let mut inconnue = t.clone();
        inconnue.issuer_did = "did:local:unknown".into();
        assert!(!tombe_effective(&inconnue));

        // An unreadable `issued_at` is the epoch (deterministic, unverifiable).
        let mut p = proprietes_tombe(&t);
        p.retain(|(k, _)| *k != "issued_at");
        p.push(("issued_at", Value::from("jamais")));
        let epoque = lire_tombe(&depuis(p)).unwrap();
        assert_eq!(epoque.issued_at, DateTime::UNIX_EPOCH);
        assert!(!crate::sharing::tombstone::verify_tombstone(&epoque));
    }
}
