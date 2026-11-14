//! `AuthStore` on obrain — the auth domain: users, refresh tokens, MCP tokens.
//!
//! Three kinds of node, with the labels and property names of the Neo4j adapter
//! (`src/neo4j/user.rs`, the reference of the conformance suite):
//!
//! - `User`: `id`, `email`, `name`, `picture_url`, `auth_provider`,
//!   `external_id`, `password_hash`, `created_at`, `last_login_at`;
//! - `RefreshToken`: `token_hash`, `user_id`, `expires_at`, `created_at`, `revoked`;
//! - `McpToken`: `jti`, `user_id`, `label`, `scope`, `expires_at`, `created_at`,
//!   `revoked`.
//!
//! An absent optional string is stored as `""` and read back as `None`, as Neo4j
//! does. Timestamps are RFC 3339 strings (as in `project.rs`). The MERGE keys are
//! the Neo4j ones: `(auth_provider, external_id)` for OIDC users,
//! `(auth_provider, email)` for password users. An ON MATCH updates what the
//! Neo4j query updates and nothing else (the password hash is never rewritten by
//! an upsert; an OIDC email follows the provider).
//!
//! # Deliberate differences, on the strict side
//!
//! - A creation that would duplicate a `User.id`, a `RefreshToken.token_hash` or
//!   a `McpToken.jti` is REFUSED by name. Neo4j accepts the duplicate and then
//!   reads an arbitrary one.
//! - Where Neo4j returns an arbitrary row among several matches (`get_user_by_email`,
//!   `get_user_by_email_and_provider`, `get_user_by_provider_id` on foreign data),
//!   this backend returns the earliest-created match (`created_at`, then `id`).
//!   Only a base holding duplicates can tell the difference.
//! - `list_mcp_tokens` breaks ties on `jti` (descending). Neo4j orders on `mt.id`,
//!   a property this adapter never writes, so its order among equal `created_at`
//!   is undefined.
//! - A stored timestamp that does not parse is an error at read, and quarantines
//!   the node at open. Neo4j substitutes "now". No timestamp is invented here.
//! - `upsert_user` on an OIDC email change that another user of the same
//!   provider already holds is not refused: the pair `(provider, email)` is not a
//!   key (two OIDC users may share an email), exactly as in Neo4j.
//! - A `User` node quarantined at open keeps its keys (id, `(provider,
//!   external_id)`, `(provider, email)`, email): an `upsert_user` that merges on one
//!   of them, and a `get_user_by_*` that reads one, is REFUSED by name (an error,
//!   never `None`). Neo4j would MATCH that node; answering `None` here would turn a
//!   `register` on its email into a second account (Neo4j: Conflict), and an OIDC
//!   login into a new id without its tokens. `consume_refresh_token` of a token
//!   whose user is reserved is refused too (not "no user row", which would send the
//!   handler to the root-account fallback). A key that a HEALTHY indexed node also
//!   holds (a quarantined duplicate of a readable user) is NOT reserved: the MERGE
//!   and the lookups find the healthy node, as Neo4j's MATCH would, and the healthy
//!   account is not refused because of its broken twin.
//! - A `RefreshToken` or `McpToken` quarantined at open (undecodable, or a
//!   duplicate hash / jti) reserves its hash / jti, EVEN when a twin is indexed:
//!   validate, consume, revoke, create, and `is_mcp_token_active` (an error, which
//!   the middleware turns into Unauthorized) refuse it, and `purge_mcp_tokens`
//!   keeps the indexed twin. Neo4j revokes every twin; here the quarantined one is
//!   never written, so without the reservation it would be served, with its own
//!   `revoked`, the day its indexed twin became unreadable or was purged.
//! - `revoked` of a `RefreshToken` or a `McpToken` must be a boolean. Absent or of
//!   another type, the node is quarantined at open and its hash / jti reserved
//!   (see below): refused by name, never valid, active or consumable, not listed.
//!   Neo4j reads such an MCP token
//!   as revoked in `is_mcp_token_active` (`unwrap_or(true)`) but as not revoked in
//!   `list_mcp_tokens`, and such a refresh token as not revoked (`unwrap_or(false)`:
//!   valid and consumable). In particular `revoke_all_user_tokens` never meets a
//!   token without `revoked` (Neo4j's `{revoked: false}` pattern skips it and
//!   leaves it valid; here it is not valid at all).
//! - Where several `User` nodes share a MERGE key (foreign data only: this backend
//!   refuses the duplicate it would create), ON MATCH updates the earliest-created
//!   one, the one the `get_*` return. Neo4j's MERGE applies ON MATCH to every
//!   matching node.
//!
//! # Durability and atomicity
//!
//! Same discipline as `project.rs`. Writes are serialized by the writer lock, the
//! decision (MERGE match, expiry, revoked state) is taken UNDER that lock, and the
//! index is published only after the durability step. A creation writes a draft
//! marker first and removes it last (the commit point); a draft found at open is
//! deleted. A multi-property update is not atomic: after a crash it can show some
//! of its properties, each state a valid user or token. A blob-sized value (above
//! [`SEUIL_COLONNE_BLOB`]) pays the full flush, as in `project.rs`.
//!
//! `purge_mcp_tokens` and `revoke_all_user_tokens` write one property (or delete one
//! node) per row and make the set durable once. A crash in the middle leaves some
//! rows done: each row's own state is valid, and the returned count is the count of
//! the call that completed.
//!
//! # Secrets
//!
//! `password_hash`, the token hashes and the JWT ids never reach an error message,
//! a quarantine line or a log. Quarantine lines name the node id only.
//!
//! # Reads
//!
//! Served from the index (hash lookups, and a sort for the listings), without
//! engine calls. The index is rebuilt from the base at every open.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::atomic::Ordering;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{Ecriture, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::neo4j::models::{AuthProvider, McpTokenNode, RefreshTokenNode, UserNode};
use crate::neo4j::traits::AuthStore;

const UTILISATEUR: &str = "User";
const JETON_RAFRAICHISSEMENT: &str = "RefreshToken";
const JETON_MCP: &str = "McpToken";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";

// ============================================================================
// Reading a node
// ============================================================================

/// A property lookup over ONE hydration of the node.
fn hydrater(db: &ObrainDB, noeud: NodeId) -> Result<impl Fn(&str) -> Option<Value>> {
    let n = db.get_node(noeud).ok_or_else(|| {
        anyhow!("nœud d'authentification {noeud:?} indexé mais absent de la base")
    })?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

fn texte(get: &dyn Fn(&str) -> Option<Value>, cle: &str) -> Option<String> {
    get(cle).and_then(|v| v.as_str().map(str::to_string))
}

fn requis(get: &dyn Fn(&str) -> Option<Value>, cle: &str) -> Result<String> {
    texte(get, cle).ok_or_else(|| anyhow!("nœud d'authentification sans propriété `{cle}` lisible"))
}

fn instant(get: &dyn Fn(&str) -> Option<Value>, cle: &str) -> Result<DateTime<Utc>> {
    let s = requis(get, cle)?;
    s.parse::<DateTime<Utc>>()
        .map_err(|e| anyhow!("`{cle}` illisible : {e}"))
}

/// A boolean that must be one. Absent or of another type (`"true"`, `1`) is an
/// error, so the node is quarantined at open: a token whose revocation state cannot
/// be read is never served as "not revoked" (see the module doc).
fn booleen(get: &dyn Fn(&str) -> Option<Value>, cle: &str) -> Result<bool> {
    get(cle)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| anyhow!("nœud d'authentification sans booléen `{cle}` lisible"))
}

fn non_vide(s: Option<String>) -> Option<String> {
    s.filter(|s| !s.is_empty())
}

/// The domain value of a `User` node: the counterpart of Neo4j's `node_to_user`,
/// with its legacy fallbacks (a node without `auth_provider` is OIDC when it has a
/// `google_id`; `external_id` falls back to `google_id`).
fn lire_utilisateur(get: &dyn Fn(&str) -> Option<Value>) -> Result<UserNode> {
    let auth_provider = texte(get, "auth_provider")
        .and_then(|s| s.parse::<AuthProvider>().ok())
        .unwrap_or_else(|| {
            if texte(get, "google_id").is_some() {
                AuthProvider::Oidc
            } else {
                AuthProvider::Password
            }
        });
    let external_id =
        non_vide(texte(get, "external_id")).or_else(|| non_vide(texte(get, "google_id")));
    Ok(UserNode {
        id: requis(get, "id")?
            .parse::<Uuid>()
            .map_err(|e| anyhow!("`id` illisible : {e}"))?,
        email: requis(get, "email")?,
        name: requis(get, "name")?,
        picture_url: non_vide(texte(get, "picture_url")),
        auth_provider,
        external_id,
        password_hash: non_vide(texte(get, "password_hash")),
        created_at: instant(get, "created_at")?,
        last_login_at: instant(get, "last_login_at")?,
    })
}

fn lire_jeton(get: &dyn Fn(&str) -> Option<Value>) -> Result<RefreshTokenNode> {
    Ok(RefreshTokenNode {
        token_hash: requis(get, "token_hash")?,
        user_id: requis(get, "user_id")?
            .parse::<Uuid>()
            .map_err(|e| anyhow!("`user_id` illisible : {e}"))?,
        expires_at: instant(get, "expires_at")?,
        created_at: instant(get, "created_at")?,
        revoked: booleen(get, "revoked")?,
    })
}

fn lire_mcp(get: &dyn Fn(&str) -> Option<Value>) -> Result<McpTokenNode> {
    Ok(McpTokenNode {
        jti: requis(get, "jti")?,
        user_id: requis(get, "user_id")?
            .parse::<Uuid>()
            .map_err(|e| anyhow!("`user_id` illisible : {e}"))?,
        label: texte(get, "label").unwrap_or_default(),
        scope: texte(get, "scope").unwrap_or_default(),
        expires_at: instant(get, "expires_at")?,
        created_at: instant(get, "created_at")?,
        revoked: booleen(get, "revoked")?,
    })
}

/// A decoded `User` with the RAW keys its MERGE and lookups match on. The decoded
/// fields apply the fallbacks; the raw ones are what the engine compares.
struct EntreeUtilisateur {
    noeud: NodeId,
    utilisateur: UserNode,
    fournisseur: Option<String>,
    externe: Option<String>,
}

struct EntreeJeton {
    noeud: NodeId,
    jeton: RefreshTokenNode,
}

struct EntreeMcp {
    noeud: NodeId,
    jeton: McpTokenNode,
}

fn entree_utilisateur(db: &ObrainDB, noeud: NodeId) -> Result<EntreeUtilisateur> {
    let get = hydrater(db, noeud)?;
    Ok(EntreeUtilisateur {
        noeud,
        utilisateur: lire_utilisateur(&get)?,
        fournisseur: texte(&get, "auth_provider"),
        externe: texte(&get, "external_id"),
    })
}

fn entree_jeton(db: &ObrainDB, noeud: NodeId) -> Result<EntreeJeton> {
    let get = hydrater(db, noeud)?;
    Ok(EntreeJeton {
        noeud,
        jeton: lire_jeton(&get)?,
    })
}

fn entree_mcp(db: &ObrainDB, noeud: NodeId) -> Result<EntreeMcp> {
    let get = hydrater(db, noeud)?;
    Ok(EntreeMcp {
        noeud,
        jeton: lire_mcp(&get)?,
    })
}

// ============================================================================
// The index
// ============================================================================

/// The auth indexes, rebuilt from the base at open and published after each
/// durable write.
#[derive(Default)]
pub(super) struct IndexAuth {
    utilisateurs: HashMap<Uuid, EntreeUtilisateur>,
    /// (raw `auth_provider`, raw `external_id`) → users: the OIDC MERGE key.
    par_externe: HashMap<(String, String), Vec<Uuid>>,
    /// (raw `auth_provider`, email) → users: the password MERGE key.
    par_fournisseur_email: HashMap<(String, String), Vec<Uuid>>,
    /// email → users, any provider.
    par_email: HashMap<String, Vec<Uuid>>,
    /// token hash → refresh token.
    jetons: HashMap<String, EntreeJeton>,
    jetons_par_utilisateur: HashMap<Uuid, Vec<String>>,
    /// jti → MCP token.
    mcp: HashMap<String, EntreeMcp>,
    mcp_par_utilisateur: HashMap<Uuid, Vec<String>>,
    /// The raw keys of the `User` nodes quarantined at open (see [`Reserves`]).
    reserves: Reserves,
}

/// The keys of the `User` nodes that are on disk but quarantined at open (undecodable,
/// or a duplicate id): what Neo4j's MATCH and MERGE would still find. A quarantined
/// node is in no index, so without this a MERGE would fall through to ON CREATE and
/// make a SECOND account under the key (a `register` on the email of a quarantined
/// password account would succeed, and hand its email to whoever registers; Neo4j
/// answers Conflict). A key listed here refuses, by name, the upsert that merges on
/// it and the lookups that read it; nothing is ever written over the node.
#[derive(Default)]
struct Reserves {
    ids: HashSet<Uuid>,
    /// (raw `auth_provider`, raw `external_id`), as `par_externe`.
    externes: HashSet<(String, String)>,
    /// (raw `auth_provider`, raw `email`), as `par_fournisseur_email`.
    fournisseur_email: HashSet<(String, String)>,
    emails: HashSet<String>,
    /// `token_hash` of the `RefreshToken` nodes quarantined at open.
    jetons: HashSet<String>,
    /// `jti` of the `McpToken` nodes quarantined at open.
    mcp: HashSet<String>,
}

impl Reserves {
    /// Reserve what can be read of a quarantined `User` node's keys. The keys are
    /// the raw strings the index would have used; an unreadable key reserves nothing
    /// (no lookup can produce it).
    fn reserver(&mut self, get: &dyn Fn(&str) -> Option<Value>) {
        if let Some(id) = texte(get, "id").and_then(|s| s.parse::<Uuid>().ok()) {
            self.ids.insert(id);
        }
        let email = texte(get, "email");
        if let Some(f) = texte(get, "auth_provider") {
            if let Some(x) = texte(get, "external_id") {
                self.externes.insert((f.clone(), x));
            }
            if let Some(e) = &email {
                self.fournisseur_email.insert((f, e.clone()));
            }
        }
        if let Some(e) = email {
            self.emails.insert(e);
        }
    }

    fn refus(quoi: &str) -> anyhow::Error {
        anyhow!(
            "{quoi} est réservé : un nœud User de cette clé est sur disque, illisible ou en \
             double à l'ouverture (quarantaine). Il n'est ni lu, ni repris, ni doublé ; \
             réparer la base puis redémarrer"
        )
    }

    fn id(&self, id: Uuid) -> Result<()> {
        if self.ids.contains(&id) {
            return Err(Self::refus(&format!("l'id utilisateur {id}")));
        }
        Ok(())
    }

    // The key values (an email, an external id) are not repeated in the message.
    fn externe(&self, fournisseur: &str, externe: &str) -> Result<()> {
        if self
            .externes
            .contains(&(fournisseur.to_string(), externe.to_string()))
        {
            return Err(Self::refus(&format!(
                "la clé ({fournisseur}, external_id) demandée"
            )));
        }
        Ok(())
    }

    fn fournisseur_email(&self, fournisseur: &str, email: &str) -> Result<()> {
        if self
            .fournisseur_email
            .contains(&(fournisseur.to_string(), email.to_string()))
        {
            return Err(Self::refus(&format!(
                "la clé ({fournisseur}, email) demandée"
            )));
        }
        Ok(())
    }

    fn email(&self, email: &str) -> Result<()> {
        if self.emails.contains(email) {
            return Err(Self::refus("l'email demandé"));
        }
        Ok(())
    }

    /// A refresh token whose hash a quarantined node also holds. The hash is a
    /// secret: never in the message.
    fn jeton(&self, hash: &str) -> Result<()> {
        if self.jetons.contains(hash) {
            bail!(
                "ce jeton de rafraîchissement est réservé : un nœud RefreshToken de même \
                 empreinte est sur disque, illisible ou en double à l'ouverture (quarantaine). \
                 Refusé jusqu'à réparation de la base"
            );
        }
        Ok(())
    }

    /// An MCP token whose jti a quarantined node also holds (the jti is not repeated).
    fn jeton_mcp(&self, jti: &str) -> Result<()> {
        if self.mcp.contains(jti) {
            bail!(
                "ce jeton MCP est réservé : un nœud McpToken de même jti est sur disque, \
                 illisible ou en double à l'ouverture (quarantaine). Refusé jusqu'à réparation \
                 de la base"
            );
        }
        Ok(())
    }
}

/// Remove one value from the list of a key, and the key when its list is empty.
fn retirer_de<K: Hash + Eq, V: PartialEq>(m: &mut HashMap<K, Vec<V>>, k: &K, valeur: &V) {
    if let Some(valeurs) = m.get_mut(k) {
        valeurs.retain(|x| x != valeur);
        if valeurs.is_empty() {
            m.remove(k);
        }
    }
}

impl IndexAuth {
    /// The earliest-created of the users listed under a key (a deterministic pick
    /// where Neo4j returns an arbitrary row; see the module doc).
    fn choisir(&self, ids: Option<&Vec<Uuid>>) -> Option<&EntreeUtilisateur> {
        ids?.iter()
            .filter_map(|id| self.utilisateurs.get(id))
            .min_by_key(|e| (e.utilisateur.created_at, e.utilisateur.id))
    }

    fn externe(&self, fournisseur: &str, externe: &str) -> Option<&EntreeUtilisateur> {
        self.choisir(
            self.par_externe
                .get(&(fournisseur.to_string(), externe.to_string())),
        )
    }

    fn fournisseur_email(&self, fournisseur: &str, email: &str) -> Option<&EntreeUtilisateur> {
        self.choisir(
            self.par_fournisseur_email
                .get(&(fournisseur.to_string(), email.to_string())),
        )
    }

    fn email(&self, email: &str) -> Option<&EntreeUtilisateur> {
        self.choisir(self.par_email.get(email))
    }

    fn inserer_utilisateur(&mut self, e: EntreeUtilisateur) {
        let id = e.utilisateur.id;
        if let Some(f) = &e.fournisseur {
            if let Some(x) = &e.externe {
                self.par_externe
                    .entry((f.clone(), x.clone()))
                    .or_default()
                    .push(id);
            }
            self.par_fournisseur_email
                .entry((f.clone(), e.utilisateur.email.clone()))
                .or_default()
                .push(id);
        }
        self.par_email
            .entry(e.utilisateur.email.clone())
            .or_default()
            .push(id);
        self.utilisateurs.insert(id, e);
    }

    /// Drop a user from every index it is listed in. The entry itself is replaced
    /// by the caller (an email change re-inserts it).
    fn retirer_utilisateur(&mut self, id: Uuid) {
        let Some(e) = self.utilisateurs.remove(&id) else {
            return;
        };
        if let Some(f) = &e.fournisseur {
            if let Some(x) = &e.externe {
                retirer_de(&mut self.par_externe, &(f.clone(), x.clone()), &id);
            }
            retirer_de(
                &mut self.par_fournisseur_email,
                &(f.clone(), e.utilisateur.email.clone()),
                &id,
            );
        }
        retirer_de(&mut self.par_email, &e.utilisateur.email, &id);
    }

    fn inserer_jeton(&mut self, e: EntreeJeton) {
        let hash = e.jeton.token_hash.clone();
        self.jetons_par_utilisateur
            .entry(e.jeton.user_id)
            .or_default()
            .push(hash.clone());
        self.jetons.insert(hash, e);
    }

    fn inserer_mcp(&mut self, e: EntreeMcp) {
        let jti = e.jeton.jti.clone();
        self.mcp_par_utilisateur
            .entry(e.jeton.user_id)
            .or_default()
            .push(jti.clone());
        self.mcp.insert(jti, e);
    }

    fn retirer_mcp(&mut self, jti: &str) {
        if let Some(e) = self.mcp.remove(jti) {
            retirer_de(
                &mut self.mcp_par_utilisateur,
                &e.jeton.user_id,
                &e.jeton.jti,
            );
        }
    }

    /// Rebuild the indexes from the persisted nodes. Drafts (a creation that never
    /// reached its commit point) are deleted; a node this backend cannot address
    /// (undecodable, or a duplicate id / hash / jti) is left untouched and named in
    /// the report.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let store = db.store();
        let mut index = Self::default();
        for etiquette in [UTILISATEUR, JETON_RAFRAICHISSEMENT, JETON_MCP] {
            for noeud in store.nodes_by_label(etiquette) {
                let Some(n) = db.get_node(noeud) else {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud {etiquette} {noeud:?} : illisible"));
                    continue;
                };
                let brouillon = n
                    .get_property(BROUILLON)
                    .is_some_and(|v| !matches!(v, Value::Null));
                if brouillon || (n.properties.is_empty() && store.out_degree(noeud) == 0) {
                    db.delete_node(noeud);
                    rapport.brouillons_supprimes += 1;
                    continue;
                }
                match etiquette {
                    UTILISATEUR => {
                        let motif = match entree_utilisateur(db, noeud) {
                            Ok(e) if !index.utilisateurs.contains_key(&e.utilisateur.id) => {
                                index.inserer_utilisateur(e);
                                continue;
                            }
                            Ok(e) => format!("id {} déjà pris", e.utilisateur.id),
                            Err(err) => err.to_string(),
                        };
                        // Its keys stay taken: see `Reserves`.
                        if let Ok(get) = hydrater(db, noeud) {
                            index.reserves.reserver(&get);
                        }
                        rapport
                            .en_quarantaine
                            .push(format!("nœud {etiquette} {noeud:?} : {motif}"));
                    }
                    JETON_RAFRAICHISSEMENT => {
                        let motif = match entree_jeton(db, noeud) {
                            Ok(e) if !index.jetons.contains_key(&e.jeton.token_hash) => {
                                index.inserer_jeton(e);
                                continue;
                            }
                            Ok(_) => "empreinte déjà prise".to_string(),
                            Err(err) => err.to_string(),
                        };
                        // Reserved even when a twin is indexed: the twin on disk keeps its
                        // own `revoked`, and would be served in place of the indexed one
                        // the day that one is unreadable.
                        if let Some(h) = hydrater(db, noeud)
                            .ok()
                            .and_then(|get| texte(&get, "token_hash"))
                        {
                            index.reserves.jetons.insert(h);
                        }
                        rapport
                            .en_quarantaine
                            .push(format!("nœud {etiquette} {noeud:?} : {motif}"));
                    }
                    _ => {
                        let motif = match entree_mcp(db, noeud) {
                            Ok(e) if !index.mcp.contains_key(&e.jeton.jti) => {
                                index.inserer_mcp(e);
                                continue;
                            }
                            Ok(_) => "jti déjà pris".to_string(),
                            Err(err) => err.to_string(),
                        };
                        if let Some(j) = hydrater(db, noeud).ok().and_then(|get| texte(&get, "jti"))
                        {
                            index.reserves.mcp.insert(j);
                        }
                        rapport
                            .en_quarantaine
                            .push(format!("nœud {etiquette} {noeud:?} : {motif}"));
                    }
                }
            }
        }
        // A user key also held by a healthy indexed node is not reserved: Neo4j's
        // MATCH would find that node too, and so does the index (a MERGE updates it,
        // never creates under the key; a lookup returns it). Reserving it would refuse
        // a healthy account because of its broken twin. Only the keys that nothing
        // healthy holds stay taken: those are the ones a MERGE would wrongly create
        // under.
        let IndexAuth {
            utilisateurs,
            par_externe,
            par_fournisseur_email,
            par_email,
            reserves,
            ..
        } = &mut index;
        reserves.ids.retain(|id| !utilisateurs.contains_key(id));
        reserves.externes.retain(|k| !par_externe.contains_key(k));
        reserves
            .fournisseur_email
            .retain(|k| !par_fournisseur_email.contains_key(k));
        reserves.emails.retain(|e| !par_email.contains_key(e));
        index
    }
}

// ============================================================================
// Writing a node
// ============================================================================

/// Does this value go to a blob column (a string above the threshold)?
fn en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

/// A write that changed nothing: the durability step is not needed.
fn sans_ecriture(w: &mut Ecriture<'_>) {
    w.rendu_durable = true;
}

/// Create a node: draft marker first, properties, marker removed last (the commit
/// point). A value in a blob column is only recovered by the full flush, which
/// runs BEFORE the marker is removed (see `project.rs`).
fn creer_noeud(
    w: &mut Ecriture<'_>,
    etiquette: &str,
    proprietes: Vec<(&'static str, Value)>,
) -> Result<NodeId> {
    w.flush_complet = proprietes.iter().any(|(_, v)| en_colonne_blob(v));
    let noeud = w.db.create_node(&[etiquette]);
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeNoeud)?;
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeMarqueur)?;
    for (_k, (cle, valeur)) in proprietes.into_iter().enumerate() {
        w.db.set_node_property(noeud, cle, valeur);
        #[cfg(any(test, feature = "mock-store"))]
        w.point(Interruption::ApresLaPropriete(_k + 1))?;
    }
    w.rendre_durable_avant_le_marqueur()?;
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::AvantLeRetraitDuMarqueur)?;
    w.db.remove_node_property(noeud, BROUILLON);
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeRetraitDuMarqueur)?;
    Ok(noeud)
}

/// Set one property of an existing node. `longue` is true when the value it
/// replaces was blob-sized: clearing that slot is not WAL-recovered either.
fn poser(w: &mut Ecriture<'_>, noeud: NodeId, cle: &'static str, valeur: Value, longue: bool) {
    if longue || en_colonne_blob(&valeur) {
        w.flush_complet = true;
    }
    w.db.set_node_property(noeud, cle, valeur);
}

fn longue(u: &UserNode) -> bool {
    let seuil = SEUIL_COLONNE_BLOB;
    u.email.len() > seuil
        || u.name.len() > seuil
        || u.picture_url.as_ref().is_some_and(|p| p.len() > seuil)
}

/// Upsert under the writer lock: MERGE on the provider key, ON MATCH or ON CREATE.
fn fusionner_utilisateur(w: &mut Ecriture<'_>, u: UserNode) -> Result<UserNode> {
    let fournisseur = u.auth_provider.to_string();
    let oidc = u.auth_provider == AuthProvider::Oidc;
    let existant = {
        let etat = w.index()?;
        // A quarantined node holds the MERGE key: Neo4j would MATCH it. Neither
        // updated (it is unreadable) nor doubled (a second account under its key).
        if oidc {
            etat.auth
                .reserves
                .externe(&fournisseur, u.external_id.as_deref().unwrap_or_default())?;
        } else {
            etat.auth
                .reserves
                .fournisseur_email(&fournisseur, &u.email)?;
        }
        let entree = if oidc {
            etat.auth
                .externe(&fournisseur, u.external_id.as_deref().unwrap_or_default())
        } else {
            etat.auth.fournisseur_email(&fournisseur, &u.email)
        };
        entree.map(|e| (e.utilisateur.id, e.noeud, longue(&e.utilisateur)))
    };

    match existant {
        // ON MATCH: the email of an OIDC user follows the provider; name, picture
        // and last login are refreshed. The password hash is never rewritten here.
        Some((id, noeud, ancienne_longue)) => {
            if oidc {
                poser(
                    w,
                    noeud,
                    "email",
                    Value::from(u.email.clone()),
                    ancienne_longue,
                );
            }
            poser(
                w,
                noeud,
                "name",
                Value::from(u.name.clone()),
                ancienne_longue,
            );
            poser(
                w,
                noeud,
                "picture_url",
                Value::from(u.picture_url.clone().unwrap_or_default()),
                ancienne_longue,
            );
            poser(
                w,
                noeud,
                "last_login_at",
                Value::from(u.last_login_at.to_rfc3339()),
                ancienne_longue,
            );
            let entree = w.apres_ecriture(entree_utilisateur(w.db, noeud))?;
            w.rendre_durable()?;
            let utilisateur = entree.utilisateur.clone();
            let mut etat = w.index_mut()?;
            etat.auth.retirer_utilisateur(id);
            etat.auth.inserer_utilisateur(entree);
            Ok(utilisateur)
        }
        // ON CREATE: the MERGE key and the ON CREATE properties of the Neo4j query.
        None => {
            let deja_pris = {
                let etat = w.index()?;
                etat.auth.reserves.id(u.id)?;
                etat.auth.utilisateurs.contains_key(&u.id)
            };
            if deja_pris {
                bail!(
                    "un utilisateur d'id {} existe déjà (unicité de User.id)",
                    u.id
                );
            }
            let noeud = creer_noeud(
                w,
                UTILISATEUR,
                vec![
                    ("id", Value::from(u.id.to_string())),
                    ("auth_provider", Value::from(fournisseur.clone())),
                    ("email", Value::from(u.email.clone())),
                    ("name", Value::from(u.name.clone())),
                    (
                        "picture_url",
                        Value::from(u.picture_url.clone().unwrap_or_default()),
                    ),
                    (
                        "external_id",
                        Value::from(u.external_id.clone().unwrap_or_default()),
                    ),
                    (
                        "password_hash",
                        Value::from(u.password_hash.clone().unwrap_or_default()),
                    ),
                    ("created_at", Value::from(u.created_at.to_rfc3339())),
                    ("last_login_at", Value::from(u.last_login_at.to_rfc3339())),
                ],
            )?;
            let entree = w.apres_ecriture(entree_utilisateur(w.db, noeud))?;
            w.rendre_durable()?;
            let utilisateur = entree.utilisateur.clone();
            w.index_mut()?.auth.inserer_utilisateur(entree);
            Ok(utilisateur)
        }
    }
}

/// Set `revoked = true` on a refresh token, make it durable, publish it.
fn revoquer_jeton(w: &mut Ecriture<'_>, hash: &str, noeud: NodeId) -> Result<()> {
    w.db.set_node_property(noeud, "revoked", Value::from(true));
    let entree = w.apres_ecriture(entree_jeton(w.db, noeud))?;
    w.rendre_durable()?;
    w.index_mut()?.auth.jetons.insert(hash.to_string(), entree);
    Ok(())
}

/// Set `revoked = true` on an MCP token, make it durable, publish it.
fn revoquer_mcp(w: &mut Ecriture<'_>, jti: &str, noeud: NodeId) -> Result<()> {
    w.db.set_node_property(noeud, "revoked", Value::from(true));
    let entree = w.apres_ecriture(entree_mcp(w.db, noeud))?;
    w.rendre_durable()?;
    w.index_mut()?.auth.mcp.insert(jti.to_string(), entree);
    Ok(())
}

/// The rotation: validate and revoke under the writer lock, so two concurrent
/// consumptions of one token cannot both succeed. A token is revoked whatever its
/// expiry (as the Neo4j statement does); the answer is `None` for unknown, expired
/// and already-used tokens alike.
fn consommer_jeton(
    w: &mut Ecriture<'_>,
    hash: &str,
    maintenant: DateTime<Utc>,
) -> Result<Option<(RefreshTokenNode, Option<UserNode>)>> {
    let existant = {
        let etat = w.index()?;
        etat.auth.reserves.jeton(hash)?;
        let existant = etat
            .auth
            .jetons
            .get(hash)
            .map(|e| (e.noeud, e.jeton.clone()));
        // A user in quarantine is not "no user row" (the handler would fall back to
        // the root account): refused, before anything is written.
        if let Some((_, j)) = &existant {
            etat.auth.reserves.id(j.user_id)?;
        }
        existant
    };
    let Some((noeud, jeton)) = existant else {
        sans_ecriture(w);
        return Ok(None);
    };
    // The state observed BEFORE the revoke, as the Neo4j statement returns it.
    let etait_revoque = jeton.revoked;
    if etait_revoque {
        sans_ecriture(w);
    } else {
        revoquer_jeton(w, hash, noeud)?;
    }
    if etait_revoque || jeton.expires_at < maintenant {
        return Ok(None);
    }
    let utilisateur = {
        let etat = w.index()?;
        etat.auth
            .utilisateurs
            .get(&jeton.user_id)
            .map(|e| e.utilisateur.clone())
    };
    Ok(Some((jeton, utilisateur)))
}

// ============================================================================
// AuthStore
// ============================================================================

#[async_trait]
impl AuthStore for ObrainGraphStore {
    async fn upsert_user(&self, user: &UserNode) -> Result<UserNode> {
        if user.auth_provider == AuthProvider::Oidc && user.external_id.is_none() {
            bail!("OIDC user must have external_id");
        }
        let user = user.clone();
        self.ecrire(|| (), move |(), w| fusionner_utilisateur(w, user))
            .await
    }

    async fn get_user_by_id(&self, id: Uuid) -> Result<Option<UserNode>> {
        self.lire(|_, etat| {
            etat.auth.reserves.id(id)?;
            Ok(etat
                .auth
                .utilisateurs
                .get(&id)
                .map(|e| e.utilisateur.clone()))
        })
    }

    async fn get_user_by_provider_id(
        &self,
        provider: &str,
        external_id: &str,
    ) -> Result<Option<UserNode>> {
        self.lire(|_, etat| {
            etat.auth.reserves.externe(provider, external_id)?;
            Ok(etat
                .auth
                .externe(provider, external_id)
                .map(|e| e.utilisateur.clone()))
        })
    }

    async fn get_user_by_email_and_provider(
        &self,
        email: &str,
        provider: &str,
    ) -> Result<Option<UserNode>> {
        self.lire(|_, etat| {
            etat.auth.reserves.fournisseur_email(provider, email)?;
            Ok(etat
                .auth
                .fournisseur_email(provider, email)
                .map(|e| e.utilisateur.clone()))
        })
    }

    async fn get_user_by_email(&self, email: &str) -> Result<Option<UserNode>> {
        self.lire(|_, etat| {
            etat.auth.reserves.email(email)?;
            Ok(etat.auth.email(email).map(|e| e.utilisateur.clone()))
        })
    }

    async fn create_password_user(
        &self,
        email: &str,
        name: &str,
        password_hash: &str,
    ) -> Result<UserNode> {
        let now = Utc::now();
        let user = UserNode {
            id: Uuid::new_v4(),
            email: email.to_string(),
            name: name.to_string(),
            picture_url: None,
            auth_provider: AuthProvider::Password,
            external_id: None,
            password_hash: Some(password_hash.to_string()),
            created_at: now,
            last_login_at: now,
        };
        self.upsert_user(&user).await
    }

    async fn list_users(&self) -> Result<Vec<UserNode>> {
        self.lire(|_, etat| {
            let mut users: Vec<UserNode> = etat
                .auth
                .utilisateurs
                .values()
                .map(|e| e.utilisateur.clone())
                .collect();
            // ORDER BY u.created_at DESC, u.id DESC
            users.sort_by(|a, b| {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| b.id.cmp(&a.id))
            });
            Ok(users)
        })
    }

    async fn create_refresh_token(
        &self,
        user_id: Uuid,
        token_hash: &str,
        expires_at: DateTime<Utc>,
    ) -> Result<()> {
        let hash = token_hash.to_string();
        self.ecrire(
            || (),
            move |(), w| {
                let deja_pris = {
                    let etat = w.index()?;
                    etat.auth.reserves.jeton(&hash)?;
                    etat.auth.jetons.contains_key(&hash)
                };
                if deja_pris {
                    bail!("un jeton de rafraîchissement de même empreinte existe déjà");
                }
                let noeud = creer_noeud(
                    w,
                    JETON_RAFRAICHISSEMENT,
                    vec![
                        ("token_hash", Value::from(hash.clone())),
                        ("user_id", Value::from(user_id.to_string())),
                        ("expires_at", Value::from(expires_at.to_rfc3339())),
                        ("created_at", Value::from(Utc::now().to_rfc3339())),
                        ("revoked", Value::from(false)),
                    ],
                )?;
                let entree = w.apres_ecriture(entree_jeton(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.auth.inserer_jeton(entree);
                Ok(())
            },
        )
        .await
    }

    async fn validate_refresh_token(&self, token_hash: &str) -> Result<Option<RefreshTokenNode>> {
        let maintenant = Utc::now();
        self.lire(|_, etat| {
            etat.auth.reserves.jeton(token_hash)?;
            Ok(etat
                .auth
                .jetons
                .get(token_hash)
                .map(|e| &e.jeton)
                .filter(|j| !j.revoked && j.expires_at >= maintenant)
                .cloned())
        })
    }

    async fn revoke_refresh_token(&self, token_hash: &str) -> Result<bool> {
        let hash = token_hash.to_string();
        self.ecrire(
            || (),
            move |(), w| {
                let existant = {
                    let etat = w.index()?;
                    etat.auth.reserves.jeton(&hash)?;
                    etat.auth
                        .jetons
                        .get(&hash)
                        .map(|e| (e.noeud, e.jeton.revoked))
                };
                let Some((noeud, deja)) = existant else {
                    sans_ecriture(w);
                    return Ok(false);
                };
                if deja {
                    sans_ecriture(w);
                } else {
                    revoquer_jeton(w, &hash, noeud)?;
                }
                Ok(true)
            },
        )
        .await
    }

    async fn consume_refresh_token(
        &self,
        token_hash: &str,
    ) -> Result<Option<(RefreshTokenNode, Option<UserNode>)>> {
        let hash = token_hash.to_string();
        // The instant is taken UNDER the writer lock: a token that expires while the
        // call waits for the lock is expired (as Neo4j evaluates it after the MATCH).
        self.ecrire(|| (), move |(), w| consommer_jeton(w, &hash, Utc::now()))
            .await
    }

    async fn revoke_all_user_tokens(&self, user_id: Uuid) -> Result<u64> {
        self.ecrire(
            || (),
            move |(), w| {
                let actifs: Vec<(String, NodeId)> = {
                    let etat = w.index()?;
                    etat.auth
                        .jetons_par_utilisateur
                        .get(&user_id)
                        .map(|hashes| {
                            hashes
                                .iter()
                                .filter_map(|h| {
                                    etat.auth
                                        .jetons
                                        .get(h)
                                        .filter(|e| !e.jeton.revoked)
                                        .map(|e| (h.clone(), e.noeud))
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                };
                if actifs.is_empty() {
                    sans_ecriture(w);
                    return Ok(0);
                }
                for (_, noeud) in &actifs {
                    w.db.set_node_property(*noeud, "revoked", Value::from(true));
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (hash, _) in &actifs {
                    if let Some(e) = etat.auth.jetons.get_mut(hash) {
                        e.jeton.revoked = true;
                    }
                }
                Ok(actifs.len() as u64)
            },
        )
        .await
    }

    async fn create_mcp_token(
        &self,
        user_id: Uuid,
        jti: &str,
        label: &str,
        scope: &str,
        expires_at: DateTime<Utc>,
    ) -> Result<()> {
        let jti = jti.to_string();
        let label = label.to_string();
        let scope = scope.to_string();
        self.ecrire(
            || (),
            move |(), w| {
                let deja_pris = {
                    let etat = w.index()?;
                    etat.auth.reserves.jeton_mcp(&jti)?;
                    etat.auth.mcp.contains_key(&jti)
                };
                if deja_pris {
                    bail!("un jeton MCP de même identifiant (jti) existe déjà");
                }
                let noeud = creer_noeud(
                    w,
                    JETON_MCP,
                    vec![
                        ("jti", Value::from(jti.clone())),
                        ("user_id", Value::from(user_id.to_string())),
                        ("label", Value::from(label.clone())),
                        ("scope", Value::from(scope.clone())),
                        ("expires_at", Value::from(expires_at.to_rfc3339())),
                        ("created_at", Value::from(Utc::now().to_rfc3339())),
                        ("revoked", Value::from(false)),
                    ],
                )?;
                let entree = w.apres_ecriture(entree_mcp(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.auth.inserer_mcp(entree);
                Ok(())
            },
        )
        .await
    }

    async fn is_mcp_token_active(&self, jti: &str) -> Result<bool> {
        let maintenant = Utc::now();
        self.lire(|_, etat| {
            // An error, not `false`: the middleware refuses it all the same
            // (Unauthorized) and logs why.
            etat.auth.reserves.jeton_mcp(jti)?;
            Ok(etat
                .auth
                .mcp
                .get(jti)
                .is_some_and(|e| !e.jeton.revoked && e.jeton.expires_at > maintenant))
        })
    }

    async fn revoke_mcp_token(&self, user_id: Uuid, jti: &str) -> Result<bool> {
        let jti = jti.to_string();
        self.ecrire(
            || (),
            move |(), w| {
                // Matched on the jti AND the owner: another user's token is not found.
                let existant = {
                    let etat = w.index()?;
                    etat.auth.reserves.jeton_mcp(&jti)?;
                    etat.auth
                        .mcp
                        .get(&jti)
                        .filter(|e| e.jeton.user_id == user_id)
                        .map(|e| (e.noeud, e.jeton.revoked))
                };
                let Some((noeud, deja)) = existant else {
                    sans_ecriture(w);
                    return Ok(false);
                };
                if deja {
                    sans_ecriture(w);
                } else {
                    revoquer_mcp(w, &jti, noeud)?;
                }
                Ok(true)
            },
        )
        .await
    }

    async fn list_mcp_tokens(&self, user_id: Uuid) -> Result<Vec<McpTokenNode>> {
        self.lire(|_, etat| {
            let mut tokens: Vec<McpTokenNode> = etat
                .auth
                .mcp_par_utilisateur
                .get(&user_id)
                .map(|jtis| {
                    jtis.iter()
                        .filter_map(|j| etat.auth.mcp.get(j))
                        .map(|e| e.jeton.clone())
                        .collect()
                })
                .unwrap_or_default();
            // ORDER BY mt.created_at DESC, then jti DESC (see the module doc on ties).
            tokens.sort_by(|a, b| {
                b.created_at
                    .cmp(&a.created_at)
                    .then_with(|| b.jti.cmp(&a.jti))
            });
            Ok(tokens)
        })
    }

    async fn purge_mcp_tokens(&self, user_id: Uuid) -> Result<u64> {
        self.ecrire(
            || (),
            move |(), w| {
                let maintenant = Utc::now();
                // Only dead records: revoked, or expired. An active token is never listed.
                // A reserved jti is kept: deleting its indexed twin would let the
                // quarantined one (with its own `revoked`) be indexed at the next open.
                let morts: Vec<(String, NodeId)> = {
                    let etat = w.index()?;
                    etat.auth
                        .mcp_par_utilisateur
                        .get(&user_id)
                        .map(|jtis| {
                            jtis.iter()
                                .filter_map(|j| etat.auth.mcp.get(j))
                                .filter(|e| e.jeton.revoked || e.jeton.expires_at < maintenant)
                                .filter(|e| !etat.auth.reserves.mcp.contains(&e.jeton.jti))
                                .map(|e| (e.jeton.jti.clone(), e.noeud))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                };
                if morts.is_empty() {
                    sans_ecriture(w);
                    return Ok(0);
                }
                for (_, noeud) in &morts {
                    if !w.db.delete_node(*noeud) {
                        // Some rows may already be gone: what the base holds is not
                        // what the index says. Refuse everything until a restart.
                        w.interne.indetermine.store(true, Ordering::Release);
                        bail!(
                            "obrain a refusé la suppression d'un jeton MCP : état indéterminé, \
                         redémarrer"
                        );
                    }
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (jti, _) in &morts {
                    etat.auth.retirer_mcp(jti);
                }
                Ok(morts.len() as u64)
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obrain::Creation;

    fn oidc(externe: &str, email: &str) -> UserNode {
        let now = Utc::now();
        UserNode {
            id: Uuid::new_v4(),
            email: email.to_string(),
            name: "Nom".to_string(),
            picture_url: None,
            auth_provider: AuthProvider::Oidc,
            external_id: Some(externe.to_string()),
            password_hash: None,
            created_at: now,
            last_login_at: now,
        }
    }

    #[tokio::test]
    async fn upsert_oidc_fusionne_sur_la_cle_externe() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let premier = store
            .upsert_user(&oidc("sub-1", "a@ex.test"))
            .await
            .unwrap();
        let mut second = oidc("sub-1", "b@ex.test");
        second.name = "Autre".to_string();
        let fusionne = store.upsert_user(&second).await.unwrap();
        // The MERGE matched: same id, the email and name follow the provider.
        assert_eq!(fusionne.id, premier.id);
        assert_eq!(fusionne.email, "b@ex.test");
        assert_eq!(fusionne.name, "Autre");
        assert_eq!(store.list_users().await.unwrap().len(), 1);
        let lu = store
            .get_user_by_provider_id("oidc", "sub-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lu.id, premier.id);
        assert!(store
            .get_user_by_email("a@ex.test")
            .await
            .unwrap()
            .is_none());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn mot_de_passe_fusionne_sur_l_email_sans_reecrire_le_hash() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let premier = store
            .create_password_user("p@ex.test", "P", "hash-1")
            .await
            .unwrap();
        let second = store
            .create_password_user("p@ex.test", "Q", "hash-2")
            .await
            .unwrap();
        assert_eq!(second.id, premier.id);
        assert_eq!(second.name, "Q");
        // ON MATCH never rewrites the password hash.
        assert_eq!(second.password_hash.as_deref(), Some("hash-1"));
        let par_email = store
            .get_user_by_email_and_provider("p@ex.test", "password")
            .await
            .unwrap();
        assert_eq!(par_email.map(|u| u.id), Some(premier.id));
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn jeton_de_rafraichissement_consomme_une_seule_fois() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let user = store
            .create_password_user("r@ex.test", "R", "h")
            .await
            .unwrap();
        let expire = Utc::now() + chrono::Duration::hours(1);
        store
            .create_refresh_token(user.id, "empreinte-1", expire)
            .await
            .unwrap();
        assert!(store
            .validate_refresh_token("empreinte-1")
            .await
            .unwrap()
            .is_some());
        let consomme = store.consume_refresh_token("empreinte-1").await.unwrap();
        let (jeton, utilisateur) = consomme.expect("premier usage valide");
        assert!(!jeton.revoked);
        assert_eq!(utilisateur.map(|u| u.id), Some(user.id));
        // Second use: already revoked, nothing returned.
        assert!(store
            .consume_refresh_token("empreinte-1")
            .await
            .unwrap()
            .is_none());
        assert!(store
            .validate_refresh_token("empreinte-1")
            .await
            .unwrap()
            .is_none());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn jeton_mcp_revoque_par_son_proprietaire_puis_purge() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let proprietaire = Uuid::new_v4();
        let autre = Uuid::new_v4();
        let expire = Utc::now() + chrono::Duration::days(1);
        store
            .create_mcp_token(proprietaire, "jti-1", "portable", "mcp:read", expire)
            .await
            .unwrap();
        assert!(store.is_mcp_token_active("jti-1").await.unwrap());
        assert!(!store.revoke_mcp_token(autre, "jti-1").await.unwrap());
        assert!(store.revoke_mcp_token(proprietaire, "jti-1").await.unwrap());
        assert!(!store.is_mcp_token_active("jti-1").await.unwrap());
        assert_eq!(store.purge_mcp_tokens(proprietaire).await.unwrap(), 1);
        assert!(store
            .list_mcp_tokens(proprietaire)
            .await
            .unwrap()
            .is_empty());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn etat_d_authentification_survit_a_la_reouverture() {
        let dossier = tempfile::tempdir().unwrap();
        let chemin = dossier.path().join("base");
        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        let user = store
            .upsert_user(&oidc("sub-9", "d@ex.test"))
            .await
            .unwrap();
        let expire = Utc::now() + chrono::Duration::hours(1);
        store
            .create_refresh_token(user.id, "empreinte-9", expire)
            .await
            .unwrap();
        store.close().await.unwrap();

        let rouverte = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let lu = rouverte.get_user_by_id(user.id).await.unwrap().unwrap();
        assert_eq!(lu.email, "d@ex.test");
        assert!(rouverte
            .validate_refresh_token("empreinte-9")
            .await
            .unwrap()
            .is_some());
        rouverte.close().await.unwrap();
    }

    /// The node of a user, as the index holds it.
    fn noeud_utilisateur(store: &ObrainGraphStore, id: Uuid) -> NodeId {
        store.interne.etat.read().unwrap().auth.utilisateurs[&id].noeud
    }

    /// A node written straight into the engine (another writer, a migration, a
    /// damaged import), made durable by the close that follows.
    fn noeud_etranger(store: &ObrainGraphStore, etiquette: &str, proprietes: &[(&str, Value)]) {
        let db = &store.interne.db;
        let n = db.create_node(&[etiquette]);
        for (cle, valeur) in proprietes {
            db.set_node_property(n, cle, valeur.clone());
        }
    }

    async fn rouvrir(chemin: &std::path::Path) -> ObrainGraphStore {
        let chemin = chemin.to_path_buf();
        tokio::task::spawn_blocking(move || ObrainGraphStore::open(chemin, Creation::Interdite))
            .await
            .unwrap()
            .unwrap()
    }

    fn refuse<T: std::fmt::Debug>(r: Result<T>, quoi: &str) {
        let e = r.expect_err(quoi);
        assert!(format!("{e:#}").contains("réservé"), "{quoi}: {e:#}");
    }

    /// A `User` quarantined at open keeps its keys: no second account is created
    /// under them, and no lookup answers `None` for them (Neo4j would MATCH it).
    #[tokio::test]
    async fn un_utilisateur_en_quarantaine_garde_ses_cles() {
        let dossier = tempfile::tempdir().unwrap();
        let chemin = dossier.path().join("base");
        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        let victime = store
            .create_password_user("p@ex.test", "P", "hash-victime")
            .await
            .unwrap();
        let oidc_victime = store
            .upsert_user(&oidc("sub-q", "q@ex.test"))
            .await
            .unwrap();
        let sain = store
            .create_password_user("s@ex.test", "S", "hash-s")
            .await
            .unwrap();
        // Damaged: `created_at` unreadable (no timestamp is invented), `name` gone.
        let db = &store.interne.db;
        db.set_node_property(
            noeud_utilisateur(&store, victime.id),
            "created_at",
            Value::from("pas une date"),
        );
        db.remove_node_property(noeud_utilisateur(&store, oidc_victime.id), "name");
        store.close().await.unwrap();
        drop(store);

        let store = rouvrir(&chemin).await;
        assert_eq!(
            store.rapport_d_ouverture().en_quarantaine.len(),
            2,
            "{:?}",
            store.rapport_d_ouverture()
        );
        // Password: the `register` sequence (lookup, then create) is refused at both
        // steps, never a second account with the newcomer's hash.
        refuse(
            store
                .get_user_by_email_and_provider("p@ex.test", "password")
                .await,
            "lookup of a quarantined (provider, email)",
        );
        refuse(
            store
                .create_password_user("p@ex.test", "Intrus", "hash-intrus")
                .await,
            "a creation on a quarantined (provider, email)",
        );
        refuse(store.get_user_by_email("p@ex.test").await, "by email");
        refuse(store.get_user_by_id(victime.id).await, "by id");
        let mut meme_id = oidc("sub-autre", "autre@ex.test");
        meme_id.id = victime.id;
        refuse(
            store.upsert_user(&meme_id).await,
            "a creation on a quarantined id",
        );
        // OIDC: the login of the quarantined subject is refused, not a new id.
        refuse(
            store.upsert_user(&oidc("sub-q", "q2@ex.test")).await,
            "an OIDC upsert on a quarantined (provider, external_id)",
        );
        refuse(
            store.get_user_by_provider_id("oidc", "sub-q").await,
            "lookup of a quarantined (provider, external_id)",
        );
        // Nothing was created; the healthy user and other keys are served.
        let ids: Vec<Uuid> = store
            .list_users()
            .await
            .unwrap()
            .iter()
            .map(|u| u.id)
            .collect();
        assert_eq!(ids, vec![sain.id]);
        assert_eq!(
            store
                .get_user_by_email_and_provider("s@ex.test", "password")
                .await
                .unwrap()
                .map(|u| u.id),
            Some(sain.id)
        );
        store
            .create_password_user("n@ex.test", "N", "hash-n")
            .await
            .unwrap();
        store.close().await.unwrap();
    }

    /// A token whose `revoked` is absent or not a boolean is never served as
    /// active (Neo4j's `is_mcp_token_active` reads it as revoked): quarantined.
    #[tokio::test]
    async fn un_jeton_sans_booleen_revoked_n_est_jamais_actif() {
        let dossier = tempfile::tempdir().unwrap();
        let chemin = dossier.path().join("base");
        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        let user = store
            .create_password_user("t@ex.test", "T", "h")
            .await
            .unwrap();
        let futur = (Utc::now() + chrono::Duration::days(1)).to_rfc3339();
        let maintenant = Utc::now().to_rfc3339();
        store
            .create_mcp_token(
                user.id,
                "jti-sain",
                "l",
                "s",
                Utc::now() + chrono::Duration::days(1),
            )
            .await
            .unwrap();
        store
            .create_refresh_token(
                user.id,
                "empreinte-saine",
                Utc::now() + chrono::Duration::days(1),
            )
            .await
            .unwrap();
        let mcp = |jti: &'static str, revoked: Option<Value>| {
            let mut p = vec![
                ("jti", Value::from(jti)),
                ("user_id", Value::from(user.id.to_string())),
                ("label", Value::from("l")),
                ("scope", Value::from("s")),
                ("expires_at", Value::from(futur.clone())),
                ("created_at", Value::from(maintenant.clone())),
            ];
            if let Some(r) = revoked {
                p.push(("revoked", r));
            }
            p
        };
        noeud_etranger(&store, JETON_MCP, &mcp("jti-absent", None));
        noeud_etranger(
            &store,
            JETON_MCP,
            &mcp("jti-texte", Some(Value::from("true"))),
        );
        noeud_etranger(
            &store,
            JETON_RAFRAICHISSEMENT,
            &[
                ("token_hash", Value::from("empreinte-absente")),
                ("user_id", Value::from(user.id.to_string())),
                ("expires_at", Value::from(futur.clone())),
                ("created_at", Value::from(maintenant.clone())),
            ],
        );
        store.close().await.unwrap();
        drop(store);

        let store = rouvrir(&chemin).await;
        assert_eq!(
            store.rapport_d_ouverture().en_quarantaine.len(),
            3,
            "{:?}",
            store.rapport_d_ouverture()
        );
        // Never active, never valid: refused by name (reserved), never `Ok(true)`.
        assert!(store.is_mcp_token_active("jti-absent").await.is_err());
        assert!(store.is_mcp_token_active("jti-texte").await.is_err());
        assert!(store.is_mcp_token_active("jti-sain").await.unwrap());
        assert!(store
            .validate_refresh_token("empreinte-absente")
            .await
            .is_err());
        assert!(store
            .consume_refresh_token("empreinte-absente")
            .await
            .is_err());
        assert!(store
            .validate_refresh_token("empreinte-saine")
            .await
            .unwrap()
            .is_some());
        // Only the healthy one is revoked: the quarantined one is not a token here.
        assert_eq!(store.revoke_all_user_tokens(user.id).await.unwrap(), 1);
        store.close().await.unwrap();
    }

    /// The rotation's guarantee under real concurrency: N consumptions of one
    /// token at once, exactly one succeeds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn rotation_concurrente_un_seul_succes() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        let user = store
            .create_password_user("c@ex.test", "C", "h")
            .await
            .unwrap();
        for tour in 0..5 {
            let hash = format!("empreinte-concurrente-{tour}");
            store
                .create_refresh_token(user.id, &hash, Utc::now() + chrono::Duration::hours(1))
                .await
                .unwrap();
            let reponses =
                futures::future::join_all((0..16).map(|_| store.consume_refresh_token(&hash)))
                    .await;
            let succes = reponses
                .into_iter()
                .map(|r| r.unwrap())
                .filter(Option::is_some)
                .count();
            assert_eq!(succes, 1, "tour {tour}: {succes} consumptions succeeded");
        }
        store.close().await.unwrap();
    }

    /// One interrupted creation, judged at reopen. `false` when the point is not on
    /// the creation's path (the creation then completed).
    async fn creation_interrompue(chemin: &std::path::Path, point: Interruption) -> bool {
        let store = ObrainGraphStore::open(chemin, Creation::Permise).unwrap();
        store.interrompre_au_prochain(point);
        let Err(err) = store.create_password_user("i@ex.test", "I", "h").await else {
            store.close().await.unwrap();
            return false;
        };
        assert!(
            format!("{err:#}").contains("interruption injectée"),
            "{point:?}"
        );
        assert!(store
            .get_user_by_email("i@ex.test")
            .await
            .unwrap()
            .is_none());
        store.close().await.unwrap();
        drop(store);

        let store = rouvrir(chemin).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert!(rapport.en_quarantaine.is_empty(), "{point:?}: {rapport:?}");
        let lu = store
            .get_user_by_email_and_provider("i@ex.test", "password")
            .await
            .unwrap();
        if point == Interruption::ApresLeRetraitDuMarqueur {
            assert_eq!(rapport.brouillons_supprimes, 0, "{point:?}");
            assert_eq!(lu.and_then(|u| u.password_hash).as_deref(), Some("h"));
        } else {
            assert_eq!(rapport.brouillons_supprimes, 1, "{point:?}: {rapport:?}");
            assert!(lu.is_none(), "{point:?}");
        }
        let u = store
            .create_password_user("i@ex.test", "I", "h")
            .await
            .unwrap();
        assert_eq!(store.list_users().await.unwrap().len(), 1, "{point:?}");
        assert_eq!(u.email, "i@ex.test");
        store.close().await.unwrap();
        true
    }

    /// A creation stopped at any point before its commit point reopens as a
    /// deleted draft (the key is free, a retry succeeds); stopped after it, as the
    /// whole record (a retry merges on it). The property points are counted, not
    /// written down: they run until one is past the last property, so a property
    /// added to the creation is covered without touching this test.
    #[tokio::test]
    async fn une_creation_interrompue_reouvre_coherente() {
        let dossier = tempfile::tempdir().unwrap();
        let mut n = 0;
        let mut chemin = || {
            n += 1;
            dossier.path().join(format!("base-{n}"))
        };
        for point in [
            Interruption::ApresLeNoeud,
            Interruption::ApresLeMarqueur,
            Interruption::AvantLeRetraitDuMarqueur,
            Interruption::ApresLeRetraitDuMarqueur,
        ] {
            assert!(creation_interrompue(&chemin(), point).await, "{point:?}");
        }
        let mut proprietes = 0;
        while creation_interrompue(&chemin(), Interruption::ApresLaPropriete(proprietes + 1)).await
        {
            proprietes += 1;
        }
        // The user node carries the nine properties of the Neo4j ON CREATE; at least
        // those are each a point (more is fine: the loop covered them).
        assert!(proprietes >= 9, "only {proprietes} property points reached");
    }

    /// The instant of a consumption is taken under the writer lock: a token that
    /// expires while the call waits behind a slow write is expired, not consumed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn la_consommation_prend_l_heure_sous_le_verrou() {
        use crate::obrain::Durabilite;
        let store =
            ObrainGraphStore::open_temporary_avec(Durabilite::LenteAPourTest(2000)).unwrap();
        let user_id = Uuid::new_v4();
        let t0 = Utc::now();
        // Created in ~2 s (the slow durability step): ~1 s of life left after.
        store
            .create_refresh_token(
                user_id,
                "empreinte-lente",
                t0 + chrono::Duration::seconds(3),
            )
            .await
            .unwrap();
        assert!(
            Utc::now() < t0 + chrono::Duration::milliseconds(2700),
            "setup too slow"
        );
        // A slow write holds the writer lock ~2 s; the consumption is called during
        // it, while the token is still valid, and runs after it, once it has expired.
        let (_, consomme) = tokio::join!(
            store.create_mcp_token(
                user_id,
                "jti-lent",
                "l",
                "s",
                t0 + chrono::Duration::days(1)
            ),
            async {
                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                assert!(
                    Utc::now() < t0 + chrono::Duration::seconds(3),
                    "called too late"
                );
                store.consume_refresh_token("empreinte-lente").await
            }
        );
        assert!(Utc::now() > t0 + chrono::Duration::seconds(3));
        assert!(
            consomme.unwrap().is_none(),
            "expired while waiting for the lock: not a valid rotation"
        );
        store.close().await.unwrap();
    }

    /// A quarantined duplicate of a readable user's id does not lock out the
    /// healthy account; its OWN keys (the ones no healthy node holds) stay taken.
    #[tokio::test]
    async fn un_doublon_d_id_ne_bloque_pas_le_compte_sain() {
        let dossier = tempfile::tempdir().unwrap();
        let chemin = dossier.path().join("base");
        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        let sain = store
            .create_password_user("sain@ex.test", "S", "hash-sain")
            .await
            .unwrap();
        let maintenant = Utc::now().to_rfc3339();
        // Twin of the same id, written after (a later slot: the healthy one is indexed
        // first), with a key of its own.
        noeud_etranger(
            &store,
            UTILISATEUR,
            &[
                ("id", Value::from(sain.id.to_string())),
                ("auth_provider", Value::from("password")),
                ("email", Value::from("jumeau@ex.test")),
                ("name", Value::from("J")),
                ("password_hash", Value::from("hash-jumeau")),
                ("created_at", Value::from(maintenant.clone())),
                ("last_login_at", Value::from(maintenant.clone())),
            ],
        );
        store.close().await.unwrap();
        drop(store);

        let store = rouvrir(&chemin).await;
        assert_eq!(
            store.rapport_d_ouverture().en_quarantaine.len(),
            1,
            "{:?}",
            store.rapport_d_ouverture()
        );
        // The healthy account is served by every lookup, and its login merges.
        assert_eq!(
            store
                .get_user_by_id(sain.id)
                .await
                .unwrap()
                .map(|u| u.email),
            Some("sain@ex.test".to_string())
        );
        assert_eq!(
            store
                .get_user_by_email_and_provider("sain@ex.test", "password")
                .await
                .unwrap()
                .map(|u| u.id),
            Some(sain.id)
        );
        let encore = store
            .create_password_user("sain@ex.test", "S2", "autre")
            .await
            .unwrap();
        assert_eq!(encore.id, sain.id);
        // The twin's own key: no second account under it, no `None` for it.
        refuse(
            store
                .get_user_by_email_and_provider("jumeau@ex.test", "password")
                .await,
            "the twin's (provider, email)",
        );
        refuse(
            store
                .create_password_user("jumeau@ex.test", "Intrus", "h")
                .await,
            "a creation on the twin's key",
        );
        store.close().await.unwrap();
    }

    /// Consuming a refresh token whose user is quarantined is refused (not "no user
    /// row", which sends the handler to the root fallback), and writes nothing.
    #[tokio::test]
    async fn la_consommation_pour_un_utilisateur_en_quarantaine_est_refusee() {
        let dossier = tempfile::tempdir().unwrap();
        let chemin = dossier.path().join("base");
        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        let u = store
            .create_password_user("qr@ex.test", "Q", "h")
            .await
            .unwrap();
        store
            .create_refresh_token(u.id, "empreinte-q", Utc::now() + chrono::Duration::hours(1))
            .await
            .unwrap();
        store.interne.db.set_node_property(
            noeud_utilisateur(&store, u.id),
            "created_at",
            Value::from("pas une date"),
        );
        store.close().await.unwrap();
        drop(store);

        let store = rouvrir(&chemin).await;
        refuse(
            store.consume_refresh_token("empreinte-q").await,
            "consume for a quarantined user",
        );
        // Refused before any write: the token is still valid for after the repair.
        assert!(store
            .validate_refresh_token("empreinte-q")
            .await
            .unwrap()
            .is_some());
        store.close().await.unwrap();
    }

    /// A token twin quarantined at open keeps its hash / jti reserved: the day the
    /// indexed twin is unreadable (or purged), the unrevoked one is not served.
    #[tokio::test]
    async fn un_jumeau_de_jeton_en_quarantaine_ne_revit_pas() {
        let dossier = tempfile::tempdir().unwrap();
        let chemin = dossier.path().join("base");
        let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
        let user = Uuid::new_v4();
        let futur = Utc::now() + chrono::Duration::days(1);
        // Used (revoked) refresh token, and a revoked MCP token.
        store
            .create_refresh_token(user, "empreinte-j", futur)
            .await
            .unwrap();
        assert!(store
            .consume_refresh_token("empreinte-j")
            .await
            .unwrap()
            .is_some());
        store
            .create_mcp_token(user, "jti-j", "l", "s", futur)
            .await
            .unwrap();
        assert!(store.revoke_mcp_token(user, "jti-j").await.unwrap());
        // Their twins, NOT revoked (foreign data), in later slots.
        let maintenant = Utc::now().to_rfc3339();
        let commun = |cle: &'static str, v: &'static str| {
            vec![
                (cle, Value::from(v)),
                ("user_id", Value::from(user.to_string())),
                ("expires_at", Value::from(futur.to_rfc3339())),
                ("created_at", Value::from(maintenant.clone())),
                ("revoked", Value::from(false)),
            ]
        };
        noeud_etranger(
            &store,
            JETON_RAFRAICHISSEMENT,
            &commun("token_hash", "empreinte-j"),
        );
        noeud_etranger(&store, JETON_MCP, &commun("jti", "jti-j"));
        store.close().await.unwrap();
        drop(store);

        // Open 1: the twins are quarantined; their keys are refused by name.
        let store = rouvrir(&chemin).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 2);
        assert!(store.validate_refresh_token("empreinte-j").await.is_err());
        assert!(store.consume_refresh_token("empreinte-j").await.is_err());
        assert!(store.is_mcp_token_active("jti-j").await.is_err());
        // The purge keeps the indexed (revoked) MCP twin: deleting it would let the
        // unrevoked one be indexed at the next open.
        assert_eq!(store.purge_mcp_tokens(user).await.unwrap(), 0);
        // The indexed refresh twin becomes unreadable.
        let noeud = store.interne.etat.read().unwrap().auth.jetons["empreinte-j"].noeud;
        store
            .interne
            .db
            .set_node_property(noeud, "revoked", Value::from("true"));
        store.close().await.unwrap();
        drop(store);

        // Open 2: the unrevoked twins are what is readable; still not served.
        let store = rouvrir(&chemin).await;
        assert!(
            !matches!(
                store.validate_refresh_token("empreinte-j").await,
                Ok(Some(_))
            ),
            "an unrevoked twin of a used refresh token is valid again"
        );
        assert!(
            !matches!(
                store.consume_refresh_token("empreinte-j").await,
                Ok(Some(_))
            ),
            "an unrevoked twin of a used refresh token is consumable again"
        );
        assert!(
            !matches!(store.is_mcp_token_active("jti-j").await, Ok(true)),
            "an unrevoked twin of a revoked MCP token is active again"
        );
        store.close().await.unwrap();
    }
}
