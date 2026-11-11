//! `ProjectStore` on obrain — the first ported domain.
//!
//! A project is one node labelled `Project` whose properties carry the SAME
//! names and the same lossy conventions as the Neo4j adapter
//! (`src/neo4j/project.rs`), which is the reference of the conformance suite:
//!
//! - creation stores `id`, `name`, `slug`, `root_path`, `canonical_path`,
//!   `declared_path`, `paths_declared = true`, `description`, `created_at`,
//!   `watch_enabled`, `profile` — and nothing else: the other fields of
//!   `ProjectNode` are only ever written by their own setters;
//! - an absent description is stored as `""` and reads back as `Some("")`;
//! - an update or a timestamp on an unknown project is a silent no-op.
//!
//! Two deliberate differences, both on the strict side:
//!
//! - timestamps are stored as RFC 3339 **strings** (obrain has temporal types,
//!   but the decoding stays a plain string, as the Cypher/GQL mapping
//!   recommends). A `created_at` that does not parse is an error naming the
//!   project; the Neo4j adapter replaces it with "now";
//! - `delete_project` deletes the node only if it carries no relationship.
//!   The Neo4j cascade walks plans, tasks, notes, files… — domains this
//!   backend has not ported and therefore cannot have written. A relationship
//!   on a project means data from elsewhere (a migrated base): the delete is
//!   refused by name rather than leaving orphans.
//!
//! Order: `list_projects` sorts by `(name, id)` ascending, `id` compared as
//! its canonical string — the total order `ORDER BY p.name, p.id ASC` states.
//! Names compare by Unicode code point.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{NodeId, Value};
use obrain_engine::ObrainDB;
use uuid::Uuid;

use super::{NonPorte, ObrainGraphStore};
use crate::neo4j::file_uid::{canonical_root, declared_root_of, PathPolicy};
use crate::neo4j::models::{ProjectNode, ProjectProfile};
use crate::neo4j::traits::ProjectStore;

const LABEL: &str = "Project";

/// `Uuid → NodeId` and `slug → Uuid` for the `Project` nodes.
#[derive(Default)]
pub(super) struct IndexProjets {
    par_id: HashMap<Uuid, NodeId>,
    par_slug: HashMap<String, Uuid>,
}

impl IndexProjets {
    /// Rebuild the index from the persisted `Project` nodes.
    ///
    /// A `Project` node this backend could not address (no readable `id`), or
    /// two nodes claiming one id or one slug, fail the open: skipping them
    /// would hide data, and both are states this adapter never writes.
    pub(super) fn reconstruire(db: &ObrainDB) -> Result<Self> {
        let mut index = Self::default();
        for noeud in db.store().nodes_by_label(LABEL) {
            let id = texte(db, noeud, "id")
                .and_then(|s| s.parse::<Uuid>().ok())
                .ok_or_else(|| anyhow!("nœud Project {noeud:?} sans propriété `id` lisible"))?;
            let slug = texte(db, noeud, "slug")
                .ok_or_else(|| anyhow!("projet {id} sans propriété `slug` lisible"))?;
            if index.par_id.insert(id, noeud).is_some() {
                bail!("deux nœuds Project portent l'id {id}");
            }
            if let Some(autre) = index.par_slug.insert(slug.clone(), id) {
                bail!("les projets {autre} et {id} portent le même slug « {slug} »");
            }
        }
        Ok(index)
    }
}

/// A property of a node, `None` when absent or null.
fn valeur(db: &ObrainDB, noeud: NodeId, cle: &str) -> Option<Value> {
    db.get_node(noeud)?
        .get_property(cle)
        .filter(|v| !matches!(v, Value::Null))
        .cloned()
}

fn texte(db: &ObrainDB, noeud: NodeId, cle: &str) -> Option<String> {
    valeur(db, noeud, cle)?.as_str().map(str::to_string)
}

/// The domain value of a `Project` node — the counterpart of
/// `Neo4jClient::node_to_project`, field for field.
fn lire_projet(db: &ObrainDB, noeud: NodeId) -> Result<ProjectNode> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud Project {noeud:?} indexé mais absent de la base"))?;
    let prop = |cle: &str| n.get_property(cle).filter(|v| !matches!(v, Value::Null));
    let chaine = |cle: &str| prop(cle).and_then(Value::as_str).map(str::to_string);
    let instant = |cle: &str| chaine(cle).and_then(|s| s.parse::<DateTime<Utc>>().ok());

    let id: Uuid = chaine("id")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("nœud Project {noeud:?} sans propriété `id` lisible"))?;
    let created_at =
        chaine("created_at").ok_or_else(|| anyhow!("projet {id} sans propriété `created_at`"))?;
    let created_at: DateTime<Utc> = created_at
        .parse()
        .map_err(|e| anyhow!("projet {id} : `created_at` illisible (« {created_at} ») : {e}"))?;

    Ok(ProjectNode {
        id,
        name: chaine("name").ok_or_else(|| anyhow!("projet {id} sans propriété `name`"))?,
        slug: chaine("slug").ok_or_else(|| anyhow!("projet {id} sans propriété `slug`"))?,
        // Absent or empty for a project with no codebase.
        root_path: chaine("root_path").unwrap_or_default(),
        description: chaine("description"),
        created_at,
        last_synced: instant("last_synced"),
        analytics_computed_at: instant("analytics_computed_at"),
        last_co_change_computed_at: instant("last_co_change_computed_at"),
        // None means default (1.0).
        default_note_energy: prop("default_note_energy")
            .and_then(Value::as_float64)
            .filter(|&v| v < 1.0),
        scaffolding_override: prop("scaffolding_override")
            .and_then(Value::as_int64)
            .map(|v| v as u8),
        sharing_policy: chaine("sharing_policy").and_then(|s| serde_json::from_str(&s).ok()),
        watch_enabled: prop("watch_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        // A node without the property predates profiles: it is software.
        profile: chaine("profile")
            .and_then(|s| s.parse::<ProjectProfile>().ok())
            .unwrap_or_default(),
    })
}

#[async_trait]
impl ProjectStore for ObrainGraphStore {
    async fn create_project(&self, project: &ProjectNode) -> Result<()> {
        let p = project.clone();
        self.ecrire(move |db, etat| {
            // Validate first: nothing is written when one of these refuses.
            if etat.projets.par_id.contains_key(&p.id) {
                bail!(
                    "un projet d'id {} existe déjà (unicité de Project.id)",
                    p.id
                );
            }
            if let Some(autre) = etat.projets.par_slug.get(&p.slug) {
                bail!(
                    "le slug « {} » est déjà celui du projet {autre} (unicité de Project.slug)",
                    p.slug
                );
            }
            let proprietes: Vec<(&str, Value)> = vec![
                ("id", Value::from(p.id.to_string())),
                ("name", Value::from(p.name.clone())),
                ("slug", Value::from(p.slug.clone())),
                ("root_path", Value::from(p.root_path.clone())),
                (
                    "canonical_path",
                    Value::from(canonical_root(Some(&p.root_path))),
                ),
                ("declared_path", Value::from(declared_root_of(&p.root_path))),
                ("paths_declared", Value::from(true)),
                (
                    "description",
                    Value::from(p.description.clone().unwrap_or_default()),
                ),
                ("created_at", Value::from(p.created_at.to_rfc3339())),
                ("watch_enabled", Value::from(p.watch_enabled)),
                ("profile", Value::from(p.profile.as_str())),
            ];
            let noeud = db.create_node_with_props(&[LABEL], proprietes);
            etat.projets.par_id.insert(p.id, noeud);
            etat.projets.par_slug.insert(p.slug, p.id);
            Ok(())
        })
        .await
    }

    async fn get_project(&self, id: Uuid) -> Result<Option<ProjectNode>> {
        self.lire(move |db, etat| {
            etat.projets
                .par_id
                .get(&id)
                .map(|&noeud| lire_projet(db, noeud))
                .transpose()
        })
        .await
    }

    async fn path_policy(&self, id: Uuid) -> Result<PathPolicy> {
        self.lire(move |db, etat| {
            let Some(&noeud) = etat.projets.par_id.get(&id) else {
                // A project that does not exist has no vocabulary.
                return Ok(PathPolicy::none());
            };
            let racine = texte(db, noeud, "declared_path")
                .or_else(|| texte(db, noeud, "root_path"))
                .unwrap_or_default();
            let canonique = texte(db, noeud, "canonical_path").unwrap_or_default();
            let declare = valeur(db, noeud, "paths_declared")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            Ok(PathPolicy::new(&racine, &canonique, declare))
        })
        .await
    }

    async fn get_project_by_slug(&self, slug: &str) -> Result<Option<ProjectNode>> {
        let slug = slug.to_string();
        self.lire(move |db, etat| {
            etat.projets
                .par_slug
                .get(&slug)
                .and_then(|id| etat.projets.par_id.get(id))
                .map(|&noeud| lire_projet(db, noeud))
                .transpose()
        })
        .await
    }

    async fn list_projects(&self) -> Result<Vec<ProjectNode>> {
        self.lire(move |db, etat| {
            let mut projets = etat
                .projets
                .par_id
                .values()
                .map(|&noeud| lire_projet(db, noeud))
                .collect::<Result<Vec<_>>>()?;
            // ORDER BY p.name, p.id ASC — total: the id breaks every tie.
            projets.sort_by(|a, b| {
                a.name
                    .cmp(&b.name)
                    .then_with(|| a.id.to_string().cmp(&b.id.to_string()))
            });
            Ok(projets)
        })
        .await
    }

    async fn update_project(
        &self,
        id: Uuid,
        name: Option<String>,
        description: Option<Option<String>>,
        root_path: Option<String>,
        profile: Option<ProjectProfile>,
    ) -> Result<()> {
        self.ecrire(move |db, etat| {
            let Some(&noeud) = etat.projets.par_id.get(&id) else {
                return Ok(());
            };
            if let Some(name) = name {
                db.set_node_property(noeud, "name", Value::from(name));
            }
            if let Some(description) = description {
                db.set_node_property(
                    noeud,
                    "description",
                    Value::from(description.unwrap_or_default()),
                );
            }
            if let Some(root_path) = root_path {
                // The declared root changed: so do its canonical and
                // normalized forms, or file identities would stay computed
                // against the old one.
                db.set_node_property(
                    noeud,
                    "canonical_path",
                    Value::from(canonical_root(Some(&root_path))),
                );
                db.set_node_property(
                    noeud,
                    "declared_path",
                    Value::from(declared_root_of(&root_path)),
                );
                db.set_node_property(noeud, "root_path", Value::from(root_path));
            }
            if let Some(profile) = profile {
                db.set_node_property(noeud, "profile", Value::from(profile.as_str()));
            }
            Ok(())
        })
        .await
    }

    async fn update_project_synced(&self, id: Uuid) -> Result<()> {
        self.horodater(id, "last_synced").await
    }

    async fn update_project_analytics_timestamp(&self, id: Uuid) -> Result<()> {
        self.horodater(id, "analytics_computed_at").await
    }

    async fn delete_project(&self, id: Uuid, _project_name: &str) -> Result<()> {
        self.ecrire(move |db, etat| {
            let Some(&noeud) = etat.projets.par_id.get(&id) else {
                return Ok(());
            };
            let relations = db.store().out_degree(noeud) + db.store().in_degree(noeud);
            if relations > 0 {
                // `project_name` only tags the notes and decisions the cascade
                // archives: part of what is refused here.
                return Err(
                    anyhow::Error::new(NonPorte("ProjectStore::delete_project (cascade)")).context(
                        format!(
                    "le projet {id} porte {relations} relation(s) écrites hors de ce backend ; \
                     la suppression en cascade attend le portage des domaines concernés"
                ),
                    ),
                );
            }
            let slug = texte(db, noeud, "slug");
            if !db.delete_node(noeud) {
                bail!("obrain a refusé la suppression du nœud {noeud:?} (projet {id})");
            }
            etat.projets.par_id.remove(&id);
            if let Some(slug) = slug {
                etat.projets.par_slug.remove(&slug);
            }
            Ok(())
        })
        .await
    }
}

impl ObrainGraphStore {
    /// Set `cle` to now on the project, if it exists.
    async fn horodater(&self, id: Uuid, cle: &'static str) -> Result<()> {
        self.ecrire(move |db, etat| {
            if let Some(&noeud) = etat.projets.par_id.get(&id) {
                db.set_node_property(noeud, cle, Value::from(Utc::now().to_rfc3339()));
            }
            Ok(())
        })
        .await
    }
}
