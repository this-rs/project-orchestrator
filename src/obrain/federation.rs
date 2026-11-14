//! `FederationStore` on obrain — the federation domain: the MCP servers a
//! project federates, and the tools they expose.
//!
//! Written the way the Neo4j adapter (`src/neo4j/mcp_federation.rs`) writes and
//! reads them, which is the reference of the conformance suite:
//!
//! - two labels, `McpServer` and `McpTool`, whose properties carry the names of
//!   the Neo4j nodes. An absent optional value is stored as `""` and reads back
//!   as `None`, as the Neo4j reader does;
//! - timestamps are RFC 3339 strings, as in `project.rs`;
//! - ordering is TOTAL: servers by `server_id`, then the MOST RECENT `created_at`
//!   first, then `id`; tools by `(name, id)`. Neo4j orders the servers by
//!   `server_id` alone. The tie-break is not arbitrary: startup
//!   (`list_all_mcp_servers`, then the first server of each `server_id`) restores
//!   the latest configuration of a server that was connected again after a failed
//!   reconnection (`connect_server` creates a node and keeps the old one). `Uuid`
//!   order is the order of its canonical string.
//!
//! Deliberate differences with the Neo4j adapter:
//!
//! - a duplicate `id` is REFUSED. Neo4j has no uniqueness constraint on these
//!   labels and would store the second node; a read by id must not pick one of
//!   two;
//! - `created_at` and `updated_at`. The Neo4j reader decodes them as strings, but
//!   they are written as native `datetime()` values, which that decoding does not
//!   read: as read by the code, `created_at` comes back as the current instant and
//!   `updated_at` as `None`. Here both read back as stored. NOT checked against a
//!   live Neo4j;
//! - the server-to-tool link is the `server_id` property. The `EXPOSES` edge is not
//!   written: nothing here reads it, and the deletions go by `server_id`. The Neo4j
//!   deletions go through `EXPOSES`, which Neo4j writes only when the server exists
//!   at the tool's creation: `delete_mcp_server` here also deletes the tools of
//!   that `server_id` created BEFORE their server, which Neo4j leaves (no production
//!   caller creates tools, `create_mcp_tool` has none);
//! - the project-to-server edge `HAS_MCP_SERVER` (Neo4j's `MERGE (p)-[:HAS_MCP_SERVER]->(s)`)
//!   is not written either: nothing reads it, the project is the `project_id`
//!   property, which is `Uuid::nil()` in production;
//! - an optional timestamp (`updated_at`, `last_connected_at`) that does not parse
//!   reads as absent, as on Neo4j; a `created_at` that does not parse quarantines
//!   the node;
//! - two nodes with the same `id` (a corrupted base: creation refuses a duplicate)
//!   are resolved at open in the engine's slot order (`nodes_by_label` is sorted
//!   by node id): the first created is kept, the others are quarantined and named.
//!
//! Refused by name ([`NonPorte`]):
//!
//! - `create_mcp_similar_to`, `create_mcp_co_activated`, `create_mcp_often_follows`:
//!   a MERGE of a typed edge between two tools found by FQN. The edge write needs
//!   an existing-edge lookup the index does not hold, so it is not written at all;
//! - `backfill_co_activated_with`, `backfill_often_follows`: they read the
//!   `ChatEventRecord` nodes of the chat domain, which this backend has not ported.
//!
//! Atomicity is built by ordering, as in `project.rs`: a creation sets the draft
//! marker `_brouillon` FIRST and removes it LAST, and opening deletes a draft.
//! A deletion removes the tools BEFORE the server: a cut between the two leaves a
//! server with no tools, which startup (`list_all_mcp_servers`) reconnects.
//!
//! Reads are served from the in-memory index, rebuilt at open. A write is
//! published to it only after its durability step.

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::Ordering;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

#[cfg(any(test, feature = "mock-store"))]
use super::Interruption;
use super::{Ecriture, NonPorte, ObrainGraphStore, SEUIL_COLONNE_BLOB};
use crate::neo4j::models::{McpServerNode, McpToolNode};
use crate::neo4j::traits::FederationStore;

const ETIQUETTE_SERVEUR: &str = "McpServer";
const ETIQUETTE_OUTIL: &str = "McpTool";
/// The draft marker: present from the first record of a creation to the last.
const BROUILLON: &str = "_brouillon";

/// `(name, id)` for a tool: the total order.
type Cle = (String, Uuid);

/// `(server_id, most recent created_at first, id)` for a server: the total order
/// (module doc).
type CleServeur = (String, Reverse<DateTime<Utc>>, Uuid);

fn cle_serveur(s: &McpServerNode) -> CleServeur {
    (s.server_id.clone(), Reverse(s.created_at), s.id)
}

struct EntreeServeur {
    noeud: NodeId,
    serveur: McpServerNode,
}

struct EntreeOutil {
    noeud: NodeId,
    outil: McpToolNode,
}

/// The decoded federation nodes, with the orders the reads need.
#[derive(Default)]
pub(super) struct IndexFederation {
    serveurs: HashMap<Uuid, EntreeServeur>,
    /// Every server in the order of `CleServeur`: `list_all_mcp_servers`.
    ordre_serveurs: BTreeSet<CleServeur>,
    /// The same keys by project: `list_mcp_servers`.
    serveurs_par_projet: HashMap<Uuid, BTreeSet<CleServeur>>,
    outils: HashMap<Uuid, EntreeOutil>,
    /// The tools of each `server_id` by `(name, id)`: `list_mcp_tools_for_server`.
    outils_par_serveur: HashMap<String, BTreeSet<Cle>>,
}

impl IndexFederation {
    fn inserer_serveur(&mut self, e: EntreeServeur) {
        let id = e.serveur.id;
        self.retirer_serveur(id);
        let cle = cle_serveur(&e.serveur);
        self.ordre_serveurs.insert(cle.clone());
        self.serveurs_par_projet
            .entry(e.serveur.project_id)
            .or_default()
            .insert(cle);
        self.serveurs.insert(id, e);
    }

    fn retirer_serveur(&mut self, id: Uuid) -> Option<EntreeServeur> {
        let e = self.serveurs.remove(&id)?;
        let cle = cle_serveur(&e.serveur);
        self.ordre_serveurs.remove(&cle);
        let projet = e.serveur.project_id;
        let vide = self
            .serveurs_par_projet
            .get_mut(&projet)
            .is_some_and(|ens| {
                ens.remove(&cle);
                ens.is_empty()
            });
        if vide {
            self.serveurs_par_projet.remove(&projet);
        }
        Some(e)
    }

    fn inserer_outil(&mut self, e: EntreeOutil) {
        let id = e.outil.id;
        self.retirer_outil(id);
        let cle: Cle = (e.outil.name.clone(), id);
        self.outils_par_serveur
            .entry(e.outil.server_id.clone())
            .or_default()
            .insert(cle);
        self.outils.insert(id, e);
    }

    fn retirer_outil(&mut self, id: Uuid) -> Option<EntreeOutil> {
        let e = self.outils.remove(&id)?;
        let cle: Cle = (e.outil.name.clone(), id);
        let serveur = e.outil.server_id.clone();
        let vide = self
            .outils_par_serveur
            .get_mut(&serveur)
            .is_some_and(|ens| {
                ens.remove(&cle);
                ens.is_empty()
            });
        if vide {
            self.outils_par_serveur.remove(&serveur);
        }
        Some(e)
    }

    /// The `(id, node)` of every tool of a server, in the order of the reads.
    fn outils_du_serveur(&self, server_id: &str) -> Vec<(Uuid, NodeId)> {
        self.outils_par_serveur
            .get(server_id)
            .into_iter()
            .flatten()
            .filter_map(|(_, id)| self.outils.get(id).map(|e| (*id, e.noeud)))
            .collect()
    }

    /// Rebuild the index from the persisted `McpServer` and `McpTool` nodes, and
    /// delete the drafts. Returns the index, the number of drafts deleted, and
    /// what could not be addressed (named, left untouched).
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<(Self, usize, Vec<String>)> {
        let store = db.store();
        let mut index = Self::default();
        let mut brouillons = 0usize;
        let mut quarantaine = Vec::new();

        let serveurs: Vec<NodeId> = store
            .nodes_by_label(ETIQUETTE_SERVEUR)
            .into_iter()
            .collect();
        for noeud in serveurs {
            let Some(n) = db.get_node(noeud) else {
                quarantaine.push(format!("nœud McpServer {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            // A draft is a creation that never reached its commit point: deleted.
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                db.delete_node(noeud);
                brouillons += 1;
                continue;
            }
            match lire_serveur(&get) {
                Ok(serveur) if index.serveurs.contains_key(&serveur.id) => {
                    quarantaine.push(format!(
                        "nœud McpServer {noeud:?} : id {} déjà pris",
                        serveur.id
                    ));
                }
                Ok(serveur) => index.inserer_serveur(EntreeServeur { noeud, serveur }),
                Err(e) => quarantaine.push(format!("nœud McpServer {noeud:?} : {e:#}")),
            }
        }

        let outils: Vec<NodeId> = store.nodes_by_label(ETIQUETTE_OUTIL).into_iter().collect();
        for noeud in outils {
            let Some(n) = db.get_node(noeud) else {
                quarantaine.push(format!("nœud McpTool {noeud:?} : illisible"));
                continue;
            };
            let get = |cle: &str| {
                n.get_property(cle)
                    .filter(|v| !matches!(v, Value::Null))
                    .cloned()
            };
            if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0)
            {
                db.delete_node(noeud);
                brouillons += 1;
                continue;
            }
            match lire_outil(&get) {
                Ok(outil) if index.outils.contains_key(&outil.id) => {
                    quarantaine.push(format!(
                        "nœud McpTool {noeud:?} : id {} déjà pris",
                        outil.id
                    ));
                }
                Ok(outil) => index.inserer_outil(EntreeOutil { noeud, outil }),
                Err(e) => quarantaine.push(format!("nœud McpTool {noeud:?} : {e:#}")),
            }
        }
        Ok((index, brouillons, quarantaine))
    }
}

/// Does this value go to a blob column (a string above the threshold)?
fn en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
}

fn instant(brut: Option<String>, cle: &str) -> Result<Option<DateTime<Utc>>> {
    brut.filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<DateTime<Utc>>()
                .map_err(|e| anyhow!("`{cle}` illisible (« {s} ») : {e}"))
        })
        .transpose()
}

/// A `McpServer` from a property lookup: the counterpart of
/// `Neo4jClient::parse_mcp_server_node`, except that an unreadable timestamp is
/// an error, not "now" (see the module doc). The errors name keys and ids only:
/// `transport_secrets` is never printed.
fn lire_serveur(get: &dyn Fn(&str) -> Option<Value>) -> Result<McpServerNode> {
    let texte = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
    let non_vide = |cle: &str| texte(cle).filter(|s| !s.is_empty());

    let id_brut = texte("id").ok_or_else(|| anyhow!("serveur MCP sans propriété `id` lisible"))?;
    let id: Uuid = id_brut
        .parse()
        .map_err(|e| anyhow!("serveur MCP : `id` illisible (« {id_brut} ») : {e}"))?;
    let projet = texte("project_id")
        .ok_or_else(|| anyhow!("serveur MCP {id} sans propriété `project_id`"))?;
    let project_id: Uuid = projet
        .parse()
        .map_err(|e| anyhow!("serveur MCP {id} : `project_id` illisible : {e}"))?;
    let server_id =
        texte("server_id").ok_or_else(|| anyhow!("serveur MCP {id} sans `server_id`"))?;
    let created_at = instant(texte("created_at"), "created_at")?
        .ok_or_else(|| anyhow!("serveur MCP {id} sans `created_at`"))?;

    Ok(McpServerNode {
        id,
        project_id,
        server_id,
        display_name: texte("display_name").unwrap_or_default(),
        transport_type: texte("transport_type").unwrap_or_default(),
        transport_url: non_vide("transport_url"),
        transport_command: non_vide("transport_command"),
        transport_args: non_vide("transport_args"),
        transport_secrets: non_vide("transport_secrets"),
        status: texte("status").unwrap_or_else(|| "disconnected".to_string()),
        protocol_version: non_vide("protocol_version"),
        server_name: non_vide("server_name"),
        tool_count: get("tool_count")
            .and_then(|v| v.as_int64())
            .unwrap_or(0)
            .max(0) as usize,
        created_at,
        // Optional: unreadable reads as absent, as Neo4j's `.ok()` (module doc).
        updated_at: instant(texte("updated_at"), "updated_at").ok().flatten(),
        last_connected_at: instant(texte("last_connected_at"), "last_connected_at")
            .ok()
            .flatten(),
    })
}

/// A `McpTool` from a property lookup (`Neo4jClient::parse_mcp_tool_node`).
fn lire_outil(get: &dyn Fn(&str) -> Option<Value>) -> Result<McpToolNode> {
    let texte = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));

    let id_brut = texte("id").ok_or_else(|| anyhow!("outil MCP sans propriété `id` lisible"))?;
    let id: Uuid = id_brut
        .parse()
        .map_err(|e| anyhow!("outil MCP : `id` illisible (« {id_brut} ») : {e}"))?;
    let server_id = texte("server_id").ok_or_else(|| anyhow!("outil MCP {id} sans `server_id`"))?;
    let name = texte("name").ok_or_else(|| anyhow!("outil MCP {id} sans `name`"))?;
    let fqn = texte("fqn").ok_or_else(|| anyhow!("outil MCP {id} sans `fqn`"))?;
    let created_at = instant(texte("created_at"), "created_at")?
        .ok_or_else(|| anyhow!("outil MCP {id} sans `created_at`"))?;

    Ok(McpToolNode {
        id,
        server_id,
        name,
        fqn,
        description: texte("description").unwrap_or_default(),
        input_schema: texte("input_schema").unwrap_or_else(|| "{}".to_string()),
        category: texte("category").unwrap_or_else(|| "unknown".to_string()),
        embedding: texte("embedding").filter(|s| !s.is_empty()),
        created_at,
        updated_at: instant(texte("updated_at"), "updated_at").ok().flatten(),
    })
}

/// The properties of a server, as a creation writes them.
fn proprietes_serveur(s: &McpServerNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(s.id.to_string())),
        ("project_id", Value::from(s.project_id.to_string())),
        ("server_id", Value::from(s.server_id.clone())),
        ("display_name", Value::from(s.display_name.clone())),
        ("transport_type", Value::from(s.transport_type.clone())),
        (
            "transport_url",
            Value::from(s.transport_url.clone().unwrap_or_default()),
        ),
        (
            "transport_command",
            Value::from(s.transport_command.clone().unwrap_or_default()),
        ),
        (
            "transport_args",
            Value::from(s.transport_args.clone().unwrap_or_default()),
        ),
        (
            "transport_secrets",
            Value::from(s.transport_secrets.clone().unwrap_or_default()),
        ),
        ("status", Value::from(s.status.clone())),
        (
            "protocol_version",
            Value::from(s.protocol_version.clone().unwrap_or_default()),
        ),
        (
            "server_name",
            Value::from(s.server_name.clone().unwrap_or_default()),
        ),
        ("tool_count", Value::from(s.tool_count as i64)),
        ("created_at", Value::from(s.created_at.to_rfc3339())),
        (
            "last_connected_at",
            Value::from(
                s.last_connected_at
                    .map(|t| t.to_rfc3339())
                    .unwrap_or_default(),
            ),
        ),
    ]
}

/// The properties of a tool, as a creation writes them.
fn proprietes_outil(t: &McpToolNode) -> Vec<(&'static str, Value)> {
    vec![
        ("id", Value::from(t.id.to_string())),
        ("server_id", Value::from(t.server_id.clone())),
        ("name", Value::from(t.name.clone())),
        ("fqn", Value::from(t.fqn.clone())),
        ("description", Value::from(t.description.clone())),
        ("input_schema", Value::from(t.input_schema.clone())),
        ("category", Value::from(t.category.clone())),
        (
            "embedding",
            Value::from(t.embedding.clone().unwrap_or_default()),
        ),
        ("created_at", Value::from(t.created_at.to_rfc3339())),
    ]
}

/// Create a node the way a creation must: the draft marker first, the
/// properties, the marker removed last (the commit point). Returns the node.
/// The caller validated the write BEFORE calling this.
fn creer_noeud(
    w: &mut Ecriture<'_>,
    etiquette: &'static str,
    proprietes: Vec<(&'static str, Value)>,
) -> Result<NodeId> {
    // A value in a blob column is only recovered by the full flush: it must be on
    // disk BEFORE the marker says the node is complete (see `project.rs`).
    w.flush_complet = proprietes.iter().any(|(_, v)| en_colonne_blob(v));
    let noeud = w.db.create_node(&[etiquette]);
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeNoeud)?;
    w.db.set_node_property(noeud, BROUILLON, Value::from(true));
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeMarqueur)?;
    for (n, (cle, valeur)) in proprietes.into_iter().enumerate() {
        w.db.set_node_property(noeud, cle, valeur);
        #[cfg(any(test, feature = "mock-store"))]
        w.point(Interruption::ApresLaPropriete(n + 1))?;
        // Only the test hook reads the rank.
        #[cfg(not(any(test, feature = "mock-store")))]
        let _ = n;
    }
    w.rendre_durable_avant_le_marqueur()?;
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::AvantLeRetraitDuMarqueur)?;
    w.db.remove_node_property(noeud, BROUILLON);
    #[cfg(any(test, feature = "mock-store"))]
    w.point(Interruption::ApresLeRetraitDuMarqueur)?;
    Ok(noeud)
}

/// Delete nodes, failing (and marking the store indeterminate) if the engine
/// does not find one the index says exists: the index and the base disagree.
fn supprimer_noeuds(w: &mut Ecriture<'_>, noeuds: impl IntoIterator<Item = NodeId>) -> Result<()> {
    for noeud in noeuds {
        if !w.db.delete_node(noeud) {
            w.interne.indetermine.store(true, Ordering::Release);
            bail!(
                "obrain a refusé la suppression du nœud {noeud:?} (indexé) : index et base divergent"
            );
        }
    }
    Ok(())
}

/// Read a node back into the index entry, as the readers will be served it.
fn decoder_serveur(db: &ObrainDB, noeud: NodeId) -> Result<EntreeServeur> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud McpServer {noeud:?} indexé mais absent de la base"))?;
    let get = |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    Ok(EntreeServeur {
        noeud,
        serveur: lire_serveur(&get)?,
    })
}

fn decoder_outil(db: &ObrainDB, noeud: NodeId) -> Result<EntreeOutil> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud McpTool {noeud:?} indexé mais absent de la base"))?;
    let get = |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    Ok(EntreeOutil {
        noeud,
        outil: lire_outil(&get)?,
    })
}

#[async_trait]
impl FederationStore for ObrainGraphStore {
    async fn create_mcp_server(&self, server: &McpServerNode) -> Result<()> {
        let s = server.clone();
        self.ecrire(
            move || s,
            move |s: McpServerNode, w: &mut Ecriture<'_>| {
                // Validate first: nothing is written when this refuses.
                if w.index()?.federation.serveurs.contains_key(&s.id) {
                    bail!(
                        "un serveur MCP d'id {} existe déjà (unicité de McpServer.id)",
                        s.id
                    );
                }
                let noeud = creer_noeud(w, ETIQUETTE_SERVEUR, proprietes_serveur(&s))?;
                let entree = w.apres_ecriture(decoder_serveur(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.federation.inserer_serveur(entree);
                Ok(())
            },
        )
        .await
    }

    async fn get_mcp_server(&self, id: Uuid) -> Result<Option<McpServerNode>> {
        self.lire(|_, etat| Ok(etat.federation.serveurs.get(&id).map(|e| e.serveur.clone())))
    }

    async fn list_mcp_servers(&self, project_id: Uuid) -> Result<Vec<McpServerNode>> {
        self.lire(|_, etat| {
            let f = &etat.federation;
            let servers: Vec<McpServerNode> = f
                .serveurs_par_projet
                .get(&project_id)
                .into_iter()
                .flatten()
                .filter_map(|(_, _, id)| f.serveurs.get(id).map(|e| e.serveur.clone()))
                .collect();
            Ok(servers)
        })
    }

    async fn list_all_mcp_servers(&self) -> Result<Vec<McpServerNode>> {
        self.lire(|_, etat| {
            let f = &etat.federation;
            let servers: Vec<McpServerNode> = f
                .ordre_serveurs
                .iter()
                .filter_map(|(_, _, id)| f.serveurs.get(id).map(|e| e.serveur.clone()))
                .collect();
            Ok(servers)
        })
    }

    async fn delete_mcp_server(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // Tools first, then the server (module doc).
                let (noeud, outils) = {
                    let etat = w.index()?;
                    let Some(e) = etat.federation.serveurs.get(&id) else {
                        return Ok(());
                    };
                    (
                        e.noeud,
                        etat.federation.outils_du_serveur(&e.serveur.server_id),
                    )
                };
                supprimer_noeuds(w, outils.iter().map(|(_, n)| *n))?;
                supprimer_noeuds(w, [noeud])?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (oid, _) in &outils {
                    etat.federation.retirer_outil(*oid);
                }
                etat.federation.retirer_serveur(id);
                Ok(())
            },
        )
        .await
    }

    async fn delete_mcp_server_by_server_id(&self, server_id: &str) -> Result<()> {
        let sid = server_id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                // Every server with this server_id, in any project. No server, no
                // deletion of tools either (as the Neo4j statement, which starts
                // from the servers).
                let (serveurs, outils) = {
                    let etat = w.index()?;
                    let f = &etat.federation;
                    let serveurs: Vec<(Uuid, NodeId)> = f
                        .ordre_serveurs
                        .range((sid.clone(), Reverse(DateTime::<Utc>::MAX_UTC), Uuid::nil())..)
                        .take_while(|(s, _, _)| s == &sid)
                        .filter_map(|(_, _, id)| f.serveurs.get(id).map(|e| (*id, e.noeud)))
                        .collect();
                    (serveurs, f.outils_du_serveur(&sid))
                };
                if serveurs.is_empty() {
                    return Ok(());
                }
                supprimer_noeuds(w, outils.iter().map(|(_, n)| *n))?;
                supprimer_noeuds(w, serveurs.iter().map(|(_, n)| *n))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (oid, _) in &outils {
                    etat.federation.retirer_outil(*oid);
                }
                for (id, _) in &serveurs {
                    etat.federation.retirer_serveur(*id);
                }
                Ok(())
            },
        )
        .await
    }

    async fn update_mcp_server_status(&self, id: Uuid, status: &str) -> Result<()> {
        let status = status.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (noeud, ancien) = {
                    let etat = w.index()?;
                    let Some(e) = etat.federation.serveurs.get(&id) else {
                        return Ok(());
                    };
                    (e.noeud, e.serveur.status.clone())
                };
                // A replaced value that WAS in a blob column is not in the WAL either.
                w.flush_complet =
                    status.len() > SEUIL_COLONNE_BLOB || ancien.len() > SEUIL_COLONNE_BLOB;
                w.db.set_node_property(noeud, "status", Value::from(status));
                w.db.set_node_property(noeud, "updated_at", Value::from(Utc::now().to_rfc3339()));
                let entree = w.apres_ecriture(decoder_serveur(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.federation.inserer_serveur(entree);
                Ok(())
            },
        )
        .await
    }

    async fn create_mcp_tool(&self, tool: &McpToolNode) -> Result<()> {
        let t = tool.clone();
        self.ecrire(
            move || t,
            move |t: McpToolNode, w: &mut Ecriture<'_>| {
                if w.index()?.federation.outils.contains_key(&t.id) {
                    bail!(
                        "un outil MCP d'id {} existe déjà (unicité de McpTool.id)",
                        t.id
                    );
                }
                let noeud = creer_noeud(w, ETIQUETTE_OUTIL, proprietes_outil(&t))?;
                let entree = w.apres_ecriture(decoder_outil(w.db, noeud))?;
                w.rendre_durable()?;
                w.index_mut()?.federation.inserer_outil(entree);
                Ok(())
            },
        )
        .await
    }

    async fn list_mcp_tools_for_server(&self, server_id: &str) -> Result<Vec<McpToolNode>> {
        self.lire(|_, etat| {
            let f = &etat.federation;
            let tools: Vec<McpToolNode> = f
                .outils_par_serveur
                .get(server_id)
                .into_iter()
                .flatten()
                .filter_map(|(_, id)| f.outils.get(id).map(|e| e.outil.clone()))
                .collect();
            Ok(tools)
        })
    }

    async fn delete_mcp_tools_for_server(&self, server_id: &str) -> Result<()> {
        let sid = server_id.to_string();
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let outils = w.index()?.federation.outils_du_serveur(&sid);
                if outils.is_empty() {
                    return Ok(());
                }
                supprimer_noeuds(w, outils.iter().map(|(_, n)| *n))?;
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for (oid, _) in &outils {
                    etat.federation.retirer_outil(*oid);
                }
                Ok(())
            },
        )
        .await
    }

    // ------------------------------------------------------------------
    // Refused by name: see the module doc.
    // ------------------------------------------------------------------

    async fn create_mcp_similar_to(
        &self,
        _tool_fqn: &str,
        _similar_to_fqn: &str,
        _score: f64,
    ) -> Result<()> {
        Err(NonPorte("FederationStore::create_mcp_similar_to").into())
    }

    async fn create_mcp_co_activated(
        &self,
        _tool_fqn_a: &str,
        _tool_fqn_b: &str,
        _count: i64,
    ) -> Result<()> {
        Err(NonPorte("FederationStore::create_mcp_co_activated").into())
    }

    async fn create_mcp_often_follows(
        &self,
        _tool_fqn_a: &str,
        _tool_fqn_b: &str,
        _count: i64,
        _avg_delta_ms: f64,
    ) -> Result<()> {
        Err(NonPorte("FederationStore::create_mcp_often_follows").into())
    }

    async fn backfill_co_activated_with(&self) -> Result<usize> {
        Err(NonPorte("FederationStore::backfill_co_activated_with").into())
    }

    async fn backfill_often_follows(&self) -> Result<usize> {
        Err(NonPorte("FederationStore::backfill_often_follows").into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obrain::Creation;

    fn serveur(project_id: Uuid, server_id: &str) -> McpServerNode {
        McpServerNode {
            id: Uuid::new_v4(),
            project_id,
            server_id: server_id.to_string(),
            display_name: format!("serveur {server_id}"),
            transport_type: "stdio".to_string(),
            transport_url: None,
            transport_command: Some("npx".to_string()),
            transport_args: Some(r#"["-y","pkg"]"#.to_string()),
            transport_secrets: Some(r#"{"env":{"TOKEN":"secret"}}"#.to_string()),
            status: "connected".to_string(),
            protocol_version: Some("2024-11-05".to_string()),
            server_name: None,
            tool_count: 2,
            created_at: Utc::now(),
            updated_at: None,
            last_connected_at: None,
        }
    }

    /// The values of a server and of a tool (a description above the blob
    /// threshold, a null embedding, a secret kept as stored) survive a close and a
    /// reopen of the base.
    #[tokio::test]
    async fn servers_and_tools_survive_a_reopen_with_their_values() {
        let dir = tempfile::Builder::new()
            .prefix("po-federation-")
            .tempdir()
            .unwrap();
        let chemin = dir.path().to_path_buf();
        let projet = Uuid::new_v4();
        let s = serveur(projet, "reopen");
        let t = McpToolNode {
            id: Uuid::new_v4(),
            server_id: "reopen".to_string(),
            name: "read".to_string(),
            fqn: "reopen::read".to_string(),
            description: "lit".repeat(100),
            input_schema: "{}".to_string(),
            category: "query".to_string(),
            embedding: None,
            created_at: Utc::now(),
            updated_at: None,
        };
        {
            let store = ObrainGraphStore::open(&chemin, Creation::Permise).unwrap();
            store.create_mcp_server(&s).await.unwrap();
            store.create_mcp_tool(&t).await.unwrap();
            store.update_mcp_server_status(s.id, "error").await.unwrap();
            store.close().await.unwrap();
        }
        let store = ObrainGraphStore::open(&chemin, Creation::Interdite).unwrap();
        let lu = store.get_mcp_server(s.id).await.unwrap().unwrap();
        assert_eq!(lu.status, "error");
        assert_eq!(lu.transport_secrets, s.transport_secrets);
        assert_eq!(lu.transport_url, None);
        assert_eq!(lu.created_at, s.created_at);
        assert!(lu.updated_at.is_some());
        let tools = store.list_mcp_tools_for_server("reopen").await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].description, t.description);
        assert_eq!(tools[0].embedding, None);
        store.close().await.unwrap();
    }

    /// The five relation writes and backfills are refused by name, never answered
    /// as written.
    #[tokio::test]
    async fn relations_and_backfills_refuse_by_name() {
        let store = ObrainGraphStore::open_temporary().unwrap();
        fn nom<T>(r: Result<T>) -> Option<NonPorte> {
            r.err().and_then(|e| e.downcast_ref::<NonPorte>().copied())
        }
        let vus = [
            nom(store.create_mcp_similar_to("a::x", "a::y", 0.5).await),
            nom(store.create_mcp_co_activated("a::x", "a::y", 3).await),
            nom(store.create_mcp_often_follows("a::x", "a::y", 3, 1.0).await),
            nom(store.backfill_co_activated_with().await),
            nom(store.backfill_often_follows().await),
        ];
        let attendus = [
            "FederationStore::create_mcp_similar_to",
            "FederationStore::create_mcp_co_activated",
            "FederationStore::create_mcp_often_follows",
            "FederationStore::backfill_co_activated_with",
            "FederationStore::backfill_often_follows",
        ]
        .map(|m| Some(NonPorte(m)));
        assert_eq!(vus, attendus);
    }
}
