//! `SkillStore` on obrain — the skills, their note members, their activation,
//! and the registry of published skills. Every method of the contract is
//! written here: it serves, or it refuses by name (`NonPorte`) with the reason
//! in its context. Nothing is answered with an empty or default value in place
//! of a refusal.
//!
//! A skill is one node labelled `Skill`, a published skill one node labelled
//! `PublishedSkill`, with the property names of the Neo4j adapter
//! (`src/neo4j/skill.rs`, `src/neo4j/registry.rs`, the reference). Edges, as in
//! Neo4j:
//!
//! - `(:Skill)-[:BELONGS_TO]->(:Project)`, written at creation and moved by
//!   `rescope_skill`, toward a project this backend holds;
//! - `(:Note)-[:MEMBER_OF]->(:Skill)`: the note members (the note domain is
//!   ported, `note.rs`). `note_count` is recounted from the edges at every
//!   membership write, as Neo4j's `size(...)` does.
//!
//! # Refused by name
//!
//! - always: `create_used_skill_relation` (the `AgentExecution` domain is not
//!   ported) — listed in [`REFUSEES`];
//! - in some cases only (not listed in `REFUSEES`, they serve otherwise):
//!   `add_skill_member` / `remove_skill_member` for a `"decision"` (the Decision
//!   domain is not ported); `get_skill_members` and `activate_skill` for a skill
//!   whose base holds a membership this backend does not resolve (a
//!   `MEMBER_OF_SKILL` from a Decision, a `MEMBER_OF` from a note it does not
//!   hold), and `get_skill_members` when a member note has links it does not
//!   resolve (Neo4j would return them as anchors); `search_published_skills`
//!   with a non-empty `tags` filter (see below).
//!
//! # Differences with Neo4j, all on the strict side or named
//!
//! - a creation whose id is taken, or reserved by a quarantined node, is
//!   refused (Neo4j would create a second node);
//! - `create_skill` toward a project this backend does not hold is refused
//!   (Neo4j's `OPTIONAL MATCH` lets the `CREATE` run and the skill names a
//!   project it is not attached to; the mock refuses too);
//! - a non-finite `energy`, `cohesion`, `hit_rate` or `trust_score` is refused
//!   at the write: the lists sort on them;
//! - a node whose trigger JSON, status, timestamps or scope cannot be read is
//!   quarantined at open (Neo4j substitutes `[]`, `Emerging` or "now"); so is a
//!   skill whose `project_id` is the legacy empty string (Neo4j serves it to no
//!   project, its filters testing `IS NULL`; read as transverse it would be served
//!   to every project);
//! - `activate_skill` retains a member note on its RAW properties, as Neo4j's
//!   query (`energy` present, `status` active or needs_review, not superseded),
//!   not on the decoded note (which reads an absent energy as 1.0 and an unknown
//!   status as `Active`);
//! - `purge_archived_empty_skills` keeps a skill that has ANY incoming edge
//!   (Neo4j: only the listed reference types), and deletes all candidates in one
//!   pass (Neo4j: in batches);
//! - `increment_skill_activation` and the counter bump of `activate_skill`
//!   report a storage failure (Neo4j swallows it, "best-effort");
//! - `upsert_published_skill` without `source_project_id` is refused: Neo4j
//!   writes that node and then fails to read it back (`source_project_id` is
//!   required by its decoding);
//! - the tags of a published skill are written under `tags` (JSON) as Neo4j does,
//!   and READ back from there (`tags_json` only when `tags` is absent). Neo4j reads `tags_json`, which it never writes, so
//!   its tags always read `[]`; its `tags` filter tests `tags_json` too and so
//!   matches nothing, where the mock tests membership. The reference contradicts
//!   itself: a non-empty `tags` filter is refused by name rather than answered
//!   one way or the other. Reading back the written tags is the one difference
//!   here that is not on the strict side: it is named, for review;
//! - candidates of `match_skills_by_trigger` are taken in a fixed order
//!   (energy, name, id) before its cap of 500 (Neo4j: no order before `LIMIT`).
//!
//! # Reads are served from memory, a write is visible once durable
//!
//! Same discipline as `note.rs`: decoded skills, their member notes and the
//! published skills are held in `IndexSkills`, rebuilt at open (after the note
//! index, which memberships resolve against); every write is published after its
//! durability step. A note deleted by `note.rs` loses its `MEMBER_OF` edges with
//! it (DETACH): a membership is served only while the note index holds that note
//! on the same node.
//!
//! # Atomicity
//!
//! A creation (skill, published skill) carries the draft marker `_brouillon`,
//! removed last; opening deletes any draft. An update writes its properties one
//! by one, without a marker: after a crash the node holds some new and some old
//! values, each a valid skill (as in `note.rs`).
//!
//! A write that carries a value above [`SEUIL_COLONNE_BLOB`] (a description, a
//! context template, the trigger JSON, a package JSON…) — or that REPLACES or
//! removes one — pays the full flush (`flush_complet`), and a creation pays it
//! BEFORE the marker is removed (`rendre_durable_avant_le_marqueur`): until obrain
//! journals blob-column values (#1190), the WAL alone would not bring them back
//! after a crash. Same motif as `note.rs` and `plan.rs`.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use obrain_common::types::{EdgeId, NodeId, Value};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use regex::RegexBuilder;
use uuid::Uuid;

use super::{note, Ecriture, NonPorte, ObrainGraphStore, RapportOuverture, SEUIL_COLONNE_BLOB};
use crate::analytics::distribution::adaptive_threshold;
use crate::neo4j::models::DecisionNode;
use crate::neo4j::traits::SkillStore;
use crate::neurons::activation::{ActivatedNote, ActivationSource};
use crate::notes::Note;
use crate::skills::registry::PublishedSkill;
use crate::skills::trust::{TrustComponents, TrustLevel};
use crate::skills::{
    ActivatedSkillContext, SkillNode, SkillStatus, SkillTrigger, TriggerType,
    MAX_TRIGGER_PATTERN_LEN, REGEX_DFA_SIZE_LIMIT, REGEX_SIZE_LIMIT,
};

const LABEL: &str = "Skill";
const PUBLIEE: &str = "PublishedSkill";
const BROUILLON: &str = "_brouillon";
const MEMBRE: &str = "MEMBER_OF";
const MEMBRE_DECISION: &str = "MEMBER_OF_SKILL";
const APPARTIENT: &str = "BELONGS_TO";

/// The methods of `SkillStore` refused on EVERY call, by name. The conditional
/// refusals (see the module doc) serve otherwise and are not listed.
pub const REFUSEES: &[&str] = &["SkillStore::create_used_skill_relation"];

/// A refusal, by name, with the reason.
fn refus<T>(methode: &'static str, raison: impl Into<String>) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison.into()))
}

// ============================================================================
// Encoding and decoding
// ============================================================================

/// A property lookup over one hydration of a node.
fn lecteur(db: &ObrainDB, noeud: NodeId) -> Result<impl Fn(&str) -> Option<Value> + '_> {
    let n = db
        .get_node(noeud)
        .ok_or_else(|| anyhow!("nœud {noeud:?} indexé mais absent de la base"))?;
    Ok(move |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    })
}

/// Does this value go to a blob column (a string above [`SEUIL_COLONNE_BLOB`], or a
/// list holding one)? Until obrain journals those values (#1190), a write that
/// carries one — or REPLACES one — pays the full flush (`flush_complet`), exactly
/// as in `note.rs` / `plan.rs`.
pub(super) fn en_colonne_blob(v: &Value) -> bool {
    v.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB)
        || v.as_list().is_some_and(|l| {
            l.iter()
                .any(|e| e.as_str().is_some_and(|s| s.len() > SEUIL_COLONNE_BLOB))
        })
}

/// Does one of these properties of `noeud` hold, NOW, a blob-sized value? Replacing
/// (or removing) such a value clears its slot in the blob column, which the WAL
/// does not recover either.
pub(super) fn ancienne_en_colonne_blob<'a>(
    db: &ObrainDB,
    noeud: NodeId,
    cles: impl IntoIterator<Item = &'a str>,
) -> bool {
    let Some(n) = db.get_node(noeud) else {
        return false;
    };
    cles.into_iter()
        .any(|k| n.get_property(k).is_some_and(en_colonne_blob))
}

/// Typed reads of a property: ABSENT is `None`, PRESENT with another type is an
/// error (the node is then quarantined, never read as a default).
struct Lecture<'a> {
    get: &'a dyn Fn(&str) -> Option<Value>,
    quoi: String,
}

impl Lecture<'_> {
    fn chaine(&self, cle: &str) -> Result<Option<String>> {
        match (self.get)(cle) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(|s| Some(s.to_string()))
                .ok_or_else(|| anyhow!("{} : `{cle}` n'est pas une chaîne", self.quoi)),
        }
    }

    fn requise(&self, cle: &str) -> Result<String> {
        self.chaine(cle)?
            .ok_or_else(|| anyhow!("{} sans propriété `{cle}`", self.quoi))
    }

    fn instant(&self, cle: &str) -> Result<Option<DateTime<Utc>>> {
        self.chaine(cle)?
            .map(|s| {
                s.parse::<DateTime<Utc>>()
                    .map_err(|e| anyhow!("{} : `{cle}` illisible ({s}) : {e}", self.quoi))
            })
            .transpose()
    }

    fn entier(&self, cle: &str) -> Result<Option<i64>> {
        match (self.get)(cle) {
            None => Ok(None),
            Some(v) => v
                .as_int64()
                .map(Some)
                .ok_or_else(|| anyhow!("{} : `{cle}` n'est pas un entier", self.quoi)),
        }
    }

    fn reel(&self, cle: &str) -> Result<Option<f64>> {
        match (self.get)(cle) {
            None => Ok(None),
            Some(v) => {
                let x = v
                    .as_float64()
                    .or_else(|| v.as_int64().map(|i| i as f64))
                    .ok_or_else(|| anyhow!("{} : `{cle}` n'est pas un nombre", self.quoi))?;
                if !x.is_finite() {
                    bail!("{} : `{cle}` non fini ({x})", self.quoi);
                }
                Ok(Some(x))
            }
        }
    }

    fn booleen(&self, cle: &str) -> Result<Option<bool>> {
        match (self.get)(cle) {
            None => Ok(None),
            Some(v) => v
                .as_bool()
                .map(Some)
                .ok_or_else(|| anyhow!("{} : `{cle}` n'est pas un booléen", self.quoi)),
        }
    }

    fn liste_de_chaines(&self, cle: &str) -> Result<Vec<String>> {
        match (self.get)(cle) {
            None => Ok(vec![]),
            Some(v) => {
                let l = v
                    .as_list()
                    .ok_or_else(|| anyhow!("{} : `{cle}` n'est pas une liste", self.quoi))?;
                l.iter()
                    .map(|e| {
                        e.as_str().map(str::to_string).ok_or_else(|| {
                            anyhow!("{} : `{cle}` contient autre chose qu'une chaîne", self.quoi)
                        })
                    })
                    .collect()
            }
        }
    }
}

fn liste(v: &[String]) -> Value {
    Value::List(
        v.iter()
            .map(|x| Value::from(x.clone()))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// The floats a list sorts on must be finite (a total order cannot place NaN).
fn finis(skill: &SkillNode) -> Result<()> {
    for (nom, v) in [
        ("energy", skill.energy),
        ("cohesion", skill.cohesion),
        ("hit_rate", skill.hit_rate),
    ] {
        if !v.is_finite() {
            bail!(
                "`{nom}` non fini ({v}) refusé : un tri total ne peut pas le ranger. Neo4j \
                 l'accepterait ; divergence assumée"
            );
        }
    }
    Ok(())
}

/// The trigger JSON the store keeps: unreliable triggers are never persisted
/// (the gate of `Neo4jClient::create_skill`).
fn json_des_declencheurs(skill: &SkillNode) -> Result<String> {
    let fiables: Vec<&SkillTrigger> = skill
        .trigger_patterns
        .iter()
        .filter(|t| t.is_reliable())
        .collect();
    Ok(serde_json::to_string(&fiables)?)
}

/// The mutable properties (what `update_skill` sets), in write order. `None`
/// for a property that is absent (Neo4j's `SET x = null`).
fn proprietes_mutables(skill: &SkillNode) -> Result<Vec<(&'static str, Option<Value>)>> {
    finis(skill)?;
    Ok(vec![
        ("name", Some(Value::from(skill.name.clone()))),
        ("description", Some(Value::from(skill.description.clone()))),
        ("status", Some(Value::from(skill.status.to_string()))),
        (
            "trigger_patterns_json",
            Some(Value::from(json_des_declencheurs(skill)?)),
        ),
        (
            "context_template",
            skill.context_template.clone().map(Value::from),
        ),
        ("energy", Some(Value::from(skill.energy))),
        ("cohesion", Some(Value::from(skill.cohesion))),
        ("coverage", Some(Value::from(skill.coverage))),
        ("note_count", Some(Value::from(skill.note_count))),
        ("decision_count", Some(Value::from(skill.decision_count))),
        (
            "activation_count",
            Some(Value::from(skill.activation_count)),
        ),
        ("hit_rate", Some(Value::from(skill.hit_rate))),
        (
            "last_activated",
            skill.last_activated.map(|d| Value::from(d.to_rfc3339())),
        ),
        ("version", Some(Value::from(skill.version))),
        ("fingerprint", skill.fingerprint.clone().map(Value::from)),
        (
            "imported_at",
            skill.imported_at.map(|d| Value::from(d.to_rfc3339())),
        ),
        ("is_validated", Some(Value::from(skill.is_validated))),
        ("tags", Some(liste(&skill.tags))),
        (
            "updated_at",
            Some(Value::from(skill.updated_at.to_rfc3339())),
        ),
    ])
}

/// What a creation writes: the id and scope first, then the mutable fields,
/// then `created_at`. Absent optionals are not written.
fn proprietes_de_creation(skill: &SkillNode) -> Result<Vec<(&'static str, Value)>> {
    let mut p: Vec<(&'static str, Value)> = vec![("id", Value::from(skill.id.to_string()))];
    if let Some(pid) = skill.project_id {
        p.push(("project_id", Value::from(pid.to_string())));
    }
    p.extend(
        proprietes_mutables(skill)?
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v))),
    );
    p.push(("created_at", Value::from(skill.created_at.to_rfc3339())));
    Ok(p)
}

/// The domain value of a `Skill` node: the counterpart of
/// `Neo4jClient::node_to_skill`, strict where Neo4j substitutes a default for a
/// value it cannot read.
fn lire_skill(get: &dyn Fn(&str) -> Option<Value>) -> Result<SkillNode> {
    let l = Lecture {
        get,
        quoi: "nœud Skill".to_string(),
    };
    let id: Uuid = l
        .requise("id")?
        .parse()
        .map_err(|e| anyhow!("nœud Skill : `id` illisible : {e}"))?;
    let l = Lecture {
        get,
        quoi: format!("skill {id}"),
    };
    // The scope: absent is transverse; anything else must be a UUID. The legacy empty
    // string is NOT read as transverse: Neo4j decodes it so but its listings filter
    // `project_id IS NULL`, which misses it — it is served to no project there, and
    // read as transverse here it would be served to EVERY project. Quarantined.
    let project_id = match l.chaine("project_id")? {
        None => None,
        Some(s) if s.is_empty() => bail!(
            "skill {id} : `project_id` vide (forme héritée) : ni transverse ni rattaché ; mis \
             en quarantaine plutôt que servi à tous les projets"
        ),
        Some(s) => Some(
            s.parse::<Uuid>()
                .map_err(|e| anyhow!("skill {id} : `project_id` illisible ({s}) : {e}"))?,
        ),
    };
    let status = match l.chaine("status")? {
        None => SkillStatus::default(),
        Some(s) => s
            .parse::<SkillStatus>()
            .map_err(|e| anyhow!("skill {id} : {e}"))?,
    };
    let trigger_patterns: Vec<SkillTrigger> = match l.chaine("trigger_patterns_json")? {
        None => vec![],
        Some(j) => serde_json::from_str(&j)
            .map_err(|e| anyhow!("skill {id} : `trigger_patterns_json` illisible : {e}"))?,
    };
    Ok(SkillNode {
        id,
        project_id,
        name: l.requise("name")?,
        description: l.chaine("description")?.unwrap_or_default(),
        status,
        trigger_patterns,
        context_template: l.chaine("context_template")?.filter(|s| !s.is_empty()),
        energy: l.reel("energy")?.unwrap_or(0.0),
        cohesion: l.reel("cohesion")?.unwrap_or(0.0),
        coverage: l.entier("coverage")?.unwrap_or(0),
        note_count: l.entier("note_count")?.unwrap_or(0),
        decision_count: l.entier("decision_count")?.unwrap_or(0),
        activation_count: l.entier("activation_count")?.unwrap_or(0),
        hit_rate: l.reel("hit_rate")?.unwrap_or(0.0),
        last_activated: l.instant("last_activated")?,
        version: l.entier("version")?.unwrap_or(1),
        fingerprint: l.chaine("fingerprint")?.filter(|s| !s.is_empty()),
        imported_at: l.instant("imported_at")?,
        is_validated: l.booleen("is_validated")?.unwrap_or(false),
        tags: l.liste_de_chaines("tags")?,
        created_at: l
            .instant("created_at")?
            .ok_or_else(|| anyhow!("skill {id} sans propriété `created_at`"))?,
        updated_at: l
            .instant("updated_at")?
            .ok_or_else(|| anyhow!("skill {id} sans propriété `updated_at`"))?,
    })
}

/// What `upsert_published_skill` writes, in order. Mirrors the `SET` of
/// `Neo4jClient::upsert_published_skill`.
fn proprietes_publiee(p: &PublishedSkill) -> Result<Vec<(&'static str, Value)>> {
    let Some(source) = p.source_project_id else {
        bail!(
            "skill publié {} sans `source_project_id` refusé : Neo4j l'écrirait et ne pourrait \
             plus le relire (sa lecture l'exige). Divergence du côté strict",
            p.id
        );
    };
    if !p.trust_score.is_finite() {
        bail!(
            "`trust_score` non fini ({}) refusé : la recherche trie dessus",
            p.trust_score
        );
    }
    Ok(vec![
        ("id", Value::from(p.id.to_string())),
        ("name", Value::from(p.name.clone())),
        ("description", Value::from(p.description.clone())),
        ("tags", Value::from(serde_json::to_string(&p.tags)?)),
        (
            "package_json",
            Value::from(serde_json::to_string(&p.package)?),
        ),
        ("trust_score", Value::from(p.trust_score)),
        (
            "trust_components_json",
            Value::from(serde_json::to_string(&p.trust_components)?),
        ),
        ("trust_level", Value::from(p.trust_level.to_string())),
        ("source_project_id", Value::from(source.to_string())),
        (
            "source_project_name",
            Value::from(p.source_project_name.clone()),
        ),
        ("published_by", Value::from(p.published_by.clone())),
        ("published_at", Value::from(p.published_at.to_rfc3339())),
        ("import_count", Value::from(p.import_count)),
    ])
}

/// The counterpart of `parse_published_skill_node`, strict where it substitutes.
fn lire_publiee(get: &dyn Fn(&str) -> Option<Value>) -> Result<PublishedSkill> {
    let l = Lecture {
        get,
        quoi: "nœud PublishedSkill".to_string(),
    };
    let id: Uuid = l
        .requise("id")?
        .parse()
        .map_err(|e| anyhow!("nœud PublishedSkill : `id` illisible : {e}"))?;
    let l = Lecture {
        get,
        quoi: format!("skill publié {id}"),
    };
    // Written under `tags` (as Neo4j writes), so read from there first; `tags_json`
    // (what Neo4j reads) only when `tags` is absent.
    let tags_json = match l.chaine("tags")? {
        Some(t) => Some(t),
        None => l.chaine("tags_json")?,
    };
    let tags: Vec<String> = match tags_json {
        None => vec![],
        Some(j) => serde_json::from_str(&j)
            .map_err(|e| anyhow!("skill publié {id} : tags illisibles : {e}"))?,
    };
    let package = serde_json::from_str(&l.requise("package_json")?)
        .map_err(|e| anyhow!("skill publié {id} : `package_json` illisible : {e}"))?;
    let trust_components: TrustComponents = match l.chaine("trust_components_json")? {
        None => TrustComponents {
            energy: 0.0,
            cohesion: 0.0,
            activation: 0.0,
            success_rate: 0.0,
            source_projects: 0.0,
        },
        Some(j) => serde_json::from_str(&j)
            .map_err(|e| anyhow!("skill publié {id} : `trust_components_json` illisible : {e}"))?,
    };
    let trust_level = match l.chaine("trust_level")?.as_deref() {
        None | Some("untrusted") => TrustLevel::Untrusted,
        Some("high") => TrustLevel::High,
        Some("medium") => TrustLevel::Medium,
        Some("low") => TrustLevel::Low,
        Some(autre) => bail!("skill publié {id} : `trust_level` inconnu « {autre} »"),
    };
    let source: Uuid = l
        .requise("source_project_id")?
        .parse()
        .map_err(|e| anyhow!("skill publié {id} : `source_project_id` illisible : {e}"))?;
    Ok(PublishedSkill {
        id,
        name: l.requise("name")?,
        description: l.chaine("description")?.unwrap_or_default(),
        tags,
        package,
        trust_score: l.reel("trust_score")?.unwrap_or(0.0),
        trust_components,
        trust_level,
        source_project_id: Some(source),
        source_project_name: l.chaine("source_project_name")?.unwrap_or_default(),
        published_by: l
            .chaine("published_by")?
            .unwrap_or_else(|| "unknown".to_string()),
        published_at: l
            .instant("published_at")?
            .ok_or_else(|| anyhow!("skill publié {id} sans propriété `published_at`"))?,
        import_count: l.entier("import_count")?.unwrap_or(0),
    })
}

// ============================================================================
// Trigger evaluation (the code of `Neo4jClient::match_skills_by_trigger`)
// ============================================================================

/// The confidence of a skill for `input`: 1.0 if one of its reliable triggers
/// fires above its threshold, 0.0 otherwise; × 0.8 for an Emerging skill.
fn confiance(skill: &SkillNode, input: &str) -> f64 {
    let mut meilleure = 0.0f64;
    for t in &skill.trigger_patterns {
        if !t.is_reliable() {
            continue;
        }
        let c = match t.pattern_type {
            TriggerType::Regex => {
                if t.pattern_value.len() > MAX_TRIGGER_PATTERN_LEN {
                    0.0
                } else {
                    match RegexBuilder::new(&t.pattern_value)
                        .case_insensitive(true)
                        .size_limit(REGEX_SIZE_LIMIT)
                        .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
                        .build()
                    {
                        Ok(re) if re.is_match(input) => 1.0,
                        _ => 0.0,
                    }
                }
            }
            TriggerType::FileGlob => match glob::Pattern::new(&t.pattern_value) {
                Ok(p) if p.matches(input) => 1.0,
                _ => 0.0,
            },
            TriggerType::McpAction => {
                if crate::skills::activation::match_mcp_action_trigger(&t.pattern_value, input) {
                    1.0
                } else {
                    0.0
                }
            }
            // Needs an embedding provider: handled above the store, as in Neo4j.
            TriggerType::Semantic => 0.0,
        };
        if c >= t.confidence_threshold {
            meilleure = meilleure.max(c);
        }
    }
    if meilleure > 0.0 && skill.status == SkillStatus::Emerging {
        meilleure * 0.8
    } else {
        meilleure
    }
}

/// `ORDER BY s.energy DESC, s.name ASC, s.id ASC`.
fn ordre_energie_nom_id(a: &SkillNode, b: &SkillNode) -> Ordering {
    b.energy
        .total_cmp(&a.energy)
        .then_with(|| a.name.cmp(&b.name))
        .then_with(|| a.id.cmp(&b.id))
}

/// The scope rule of the transverse library: `Some(pid)` sees that project's
/// skills AND the transverse ones, `None` the transverse ones alone.
fn dans_la_portee(s: &SkillNode, project_id: Option<Uuid>) -> bool {
    match project_id {
        Some(pid) => s.project_id == Some(pid) || s.project_id.is_none(),
        None => s.project_id.is_none(),
    }
}

// ============================================================================
// Index
// ============================================================================

#[derive(Clone)]
struct EntreeSkill {
    noeud: NodeId,
    skill: SkillNode,
    /// Note members: note id → the note's node when the edge was read. Served only
    /// while the note index holds that note on that node.
    notes: BTreeMap<Uuid, NodeId>,
    /// Incoming membership edges this backend does not resolve: a
    /// `MEMBER_OF_SKILL` (Decision), a `MEMBER_OF` from a node it does not hold.
    non_resolus: usize,
}

#[derive(Clone)]
struct EntreePubliee {
    noeud: NodeId,
    publiee: PublishedSkill,
}

/// The index of the skill domain.
#[derive(Default)]
pub(super) struct IndexSkills {
    /// Ids of `Skill` nodes on disk, quarantined at open: not reusable.
    reserves: HashSet<Uuid>,
    par_id: HashMap<Uuid, EntreeSkill>,
    /// Skills by member note.
    par_note: HashMap<Uuid, BTreeSet<Uuid>>,
    publiees: HashMap<Uuid, EntreePubliee>,
    publiees_reservees: HashSet<Uuid>,
}

impl IndexSkills {
    fn inscrire(&mut self, e: EntreeSkill) {
        let id = e.skill.id;
        for n in e.notes.keys() {
            self.par_note.entry(*n).or_default().insert(id);
        }
        self.par_id.insert(id, e);
    }

    fn desinscrire(&mut self, id: Uuid) -> Option<EntreeSkill> {
        let e = self.par_id.remove(&id)?;
        for n in e.notes.keys() {
            if let Some(s) = self.par_note.get_mut(n) {
                s.remove(&id);
                if s.is_empty() {
                    self.par_note.remove(n);
                }
            }
        }
        Some(e)
    }

    fn publier(&mut self, e: EntreeSkill) {
        self.desinscrire(e.skill.id);
        self.inscrire(e);
    }

    /// Rebuild from the persisted nodes at open, AFTER the note index: drafts are
    /// deleted with their edges, what cannot be read is quarantined, memberships
    /// are read from the incoming edges.
    pub(super) fn reconstruire(
        db: &ObrainDB,
        notes: &note::IndexNotes,
        rapport: &mut RapportOuverture,
    ) -> Result<Self> {
        let mut index = Self::default();
        for noeud in db.store().nodes_by_label(LABEL) {
            let Some(get) = lecture_ou_brouillon(db, noeud, "Skill", rapport) else {
                continue;
            };
            match lire_skill(&get) {
                Ok(skill) if index.par_id.contains_key(&skill.id) => {
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Skill {noeud:?} : id {} déjà pris", skill.id));
                }
                Ok(skill) => {
                    let (membres, non_resolus) = membres(db, noeud, notes);
                    if non_resolus > 0 {
                        rapport.en_quarantaine.push(format!(
                            "skill {} : {non_resolus} appartenance(s) non résolue(s) (Decision ou \
                             note non tenue) ; ses membres sont refusés par leur nom",
                            skill.id
                        ));
                    }
                    index.inscrire(EntreeSkill {
                        noeud,
                        skill,
                        notes: membres,
                        non_resolus,
                    });
                }
                Err(e) => {
                    if let Some(id) = id_brut(&get) {
                        index.reserves.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud Skill {noeud:?} : {e}"));
                }
            }
        }
        for noeud in db.store().nodes_by_label(PUBLIEE) {
            let Some(get) = lecture_ou_brouillon(db, noeud, "PublishedSkill", rapport) else {
                continue;
            };
            match lire_publiee(&get) {
                Ok(p) if index.publiees.contains_key(&p.id) => {
                    rapport.en_quarantaine.push(format!(
                        "nœud PublishedSkill {noeud:?} : id {} déjà pris",
                        p.id
                    ));
                }
                Ok(p) => {
                    index
                        .publiees
                        .insert(p.id, EntreePubliee { noeud, publiee: p });
                }
                Err(e) => {
                    if let Some(id) = id_brut(&get) {
                        index.publiees_reservees.insert(id);
                    }
                    rapport
                        .en_quarantaine
                        .push(format!("nœud PublishedSkill {noeud:?} : {e}"));
                }
            }
        }
        Ok(index)
    }
}

/// The raw `id` of a node, when it parses: what a quarantine reserves.
fn id_brut(get: &dyn Fn(&str) -> Option<Value>) -> Option<Uuid> {
    get("id").and_then(|v| v.as_str().and_then(|t| t.parse::<Uuid>().ok()))
}

/// At open: the property lookup of a node, or `None` when the node is a draft
/// (deleted here, with its edges) or unreadable (quarantined).
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

/// The memberships of a skill node, read from its incoming edges: the note
/// members this backend holds, and how many it does not resolve.
fn membres(
    db: &ObrainDB,
    skill: NodeId,
    notes: &note::IndexNotes,
) -> (BTreeMap<Uuid, NodeId>, usize) {
    let mut resolus = BTreeMap::new();
    let mut non_resolus = 0usize;
    for (_, arete) in db.store().edges_from(skill, Direction::Incoming) {
        let Some(e) = db.get_edge(arete) else {
            continue;
        };
        match &*e.edge_type {
            MEMBRE => {
                let note_id = db.get_node(e.src).and_then(|n| {
                    n.get_property("id")
                        .and_then(|v| v.as_str().and_then(|s| s.parse::<Uuid>().ok()))
                });
                match note_id {
                    Some(id) if notes.consulter(id).is_some_and(|(_, n, _)| n == e.src) => {
                        resolus.insert(id, e.src);
                    }
                    _ => non_resolus += 1,
                }
            }
            MEMBRE_DECISION => non_resolus += 1,
            _ => {}
        }
    }
    (resolus, non_resolus)
}

/// Incoming edges of `type_` on `skill` whose source carries `etiquette`.
fn aretes_entrantes(
    db: &ObrainDB,
    skill: NodeId,
    type_: &str,
    etiquette: &str,
) -> Vec<(NodeId, EdgeId)> {
    db.store()
        .edges_from(skill, Direction::Incoming)
        .into_iter()
        .filter_map(|(_, a)| {
            let e = db.get_edge(a)?;
            let porte = db
                .get_node_labels(e.src)
                .is_some_and(|l| l.iter().any(|x| x == etiquette));
            (&*e.edge_type == type_ && porte).then_some((e.src, a))
        })
        .collect()
}

/// DETACH DELETE of a node: its edges first, then the node.
fn detacher(w: &mut Ecriture<'_>, noeud: NodeId) -> Result<()> {
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
        w.interne
            .indetermine
            .store(true, std::sync::atomic::Ordering::Release);
        bail!(
            "obrain a refusé la suppression du nœud {noeud:?} APRÈS la suppression de ses \
             arêtes : base indéterminée, redémarrer"
        );
    }
    Ok(())
}

/// The skill entry of `id`, cloned (a write works on a copy and publishes it).
fn entree(w: &Ecriture<'_>, id: Uuid) -> Result<Option<EntreeSkill>> {
    Ok(w.index()?.skills.par_id.get(&id).cloned())
}

/// Re-decode a skill after a write, and re-read its memberships from the edges.
fn relire(w: &Ecriture<'_>, e: &EntreeSkill) -> Result<EntreeSkill> {
    let decode = lecteur(w.db, e.noeud).and_then(|get| lire_skill(&get));
    let skill = w.apres_ecriture(decode)?;
    let (notes, non_resolus) = membres(w.db, e.noeud, &w.index()?.notes);
    Ok(EntreeSkill {
        noeud: e.noeud,
        skill,
        notes,
        non_resolus,
    })
}

/// The raw `energy` of a member note node when the note is CURRENT in Neo4j's
/// terms: `energy` a number, `status` `active` or `needs_review`, no
/// `superseded_by`. `None` otherwise (the note is not part of the activation).
fn energie_si_courante(db: &ObrainDB, noeud: NodeId) -> Option<f64> {
    let n = db.get_node(noeud)?;
    let present = |cle: &str| {
        n.get_property(cle)
            .filter(|v| !matches!(v, Value::Null))
            .cloned()
    };
    let energie = present("energy")?;
    let energie = energie
        .as_float64()
        .or_else(|| energie.as_int64().map(|i| i as f64))?;
    let statut = present("status")?;
    let courant = matches!(statut.as_str(), Some("active") | Some("needs_review"))
        && present("superseded_by").is_none();
    courant.then_some(energie)
}

/// Is this note member still the note the index holds (same node)?
fn membre_vivant(notes: &note::IndexNotes, id: Uuid, noeud: NodeId) -> bool {
    notes.consulter(id).is_some_and(|(_, n, _)| n == noeud)
}

/// Only a `"note"` member is served; a `"decision"` is refused by name.
fn membre_note(methode: &'static str, entity_type: &str) -> Result<()> {
    match entity_type {
        "note" => Ok(()),
        "decision" => refus(
            methode,
            "membre `decision` : le domaine Decision n'est pas porté ; l'arête MEMBER_OF_SKILL \
             ne peut pas être écrite ici",
        ),
        autre => bail!(
            "Unknown entity type for skill membership: '{autre}'. Expected 'note' or 'decision'."
        ),
    }
}

// ============================================================================
// SkillStore
// ============================================================================

#[async_trait]
impl SkillStore for ObrainGraphStore {
    async fn create_skill(&self, skill: &SkillNode) -> Result<()> {
        let s = skill.clone();
        self.ecrire(
            move || s,
            move |s, w: &mut Ecriture<'_>| {
                let proprietes = proprietes_de_creation(&s)?;
                let projet = {
                    let etat = w.index()?;
                    if etat.skills.par_id.contains_key(&s.id) {
                        bail!("un skill d'id {} existe déjà", s.id);
                    }
                    if etat.skills.reserves.contains(&s.id) {
                        bail!(
                            "l'id {} est réservé : un nœud Skill de cet id est sur disque, \
                             illisible à l'ouverture (quarantaine). Il n'est ni repris ni écrasé",
                            s.id
                        );
                    }
                    match s.project_id {
                        Some(p) => Some(etat.projets.noeud_de(p).ok_or_else(|| {
                            anyhow!("Cannot create skill: project {p} does not exist")
                        })?),
                        None => None,
                    }
                };
                w.flush_complet = proprietes.iter().any(|(_, v)| en_colonne_blob(v));
                let noeud = w.db.create_node(&[LABEL]);
                w.etape()?;
                w.db.set_node_property(noeud, BROUILLON, Value::from(true));
                for (cle, valeur) in proprietes {
                    w.etape()?;
                    w.db.set_node_property(noeud, cle, valeur);
                }
                if let Some(p) = projet {
                    w.etape()?;
                    w.db.create_edge(noeud, p, APPARTIENT);
                }
                // A blob-sized value is made durable BEFORE the commit point.
                w.rendre_durable_avant_le_marqueur()?;
                // The commit point.
                w.etape()?;
                w.db.remove_node_property(noeud, BROUILLON);
                let lu = lecteur(w.db, noeud)
                    .and_then(|get| lire_skill(&get))
                    .and_then(|k| {
                        if k.id == s.id {
                            Ok(k)
                        } else {
                            bail!("le skill relu n'a pas l'id écrit")
                        }
                    });
                let skill = w.apres_ecriture(lu)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(EntreeSkill {
                    noeud,
                    skill,
                    notes: BTreeMap::new(),
                    non_resolus: 0,
                });
                Ok(())
            },
        )
        .await
    }

    async fn get_skill(&self, id: Uuid) -> Result<Option<SkillNode>> {
        self.lire(|_, etat| Ok(etat.skills.par_id.get(&id).map(|e| e.skill.clone())))
    }

    async fn update_skill(&self, skill: &SkillNode) -> Result<()> {
        let s = skill.clone();
        self.ecrire(
            move || s,
            move |s, w: &mut Ecriture<'_>| {
                let proprietes = proprietes_mutables(&s)?;
                let Some(ancienne) = entree(w, s.id)? else {
                    bail!("Skill not found: {}", s.id);
                };
                // New values, and the old ones they replace or remove.
                w.flush_complet = proprietes
                    .iter()
                    .any(|(_, v)| v.as_ref().is_some_and(en_colonne_blob))
                    || ancienne_en_colonne_blob(
                        w.db,
                        ancienne.noeud,
                        proprietes.iter().map(|(k, _)| *k),
                    );
                for (cle, valeur) in proprietes {
                    w.etape()?;
                    match valeur {
                        Some(v) => {
                            w.db.set_node_property(ancienne.noeud, cle, v);
                        }
                        None => {
                            w.db.remove_node_property(ancienne.noeud, cle);
                        }
                    }
                }
                let nouvelle = relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(nouvelle);
                Ok(())
            },
        )
        .await
    }

    async fn rescope_skill(&self, skill_id: Uuid, project_id: Option<Uuid>) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = entree(w, skill_id)? else {
                    return Ok(());
                };
                let noeud = ancienne.noeud;
                let cible = match project_id {
                    Some(p) => w.index()?.projets.noeud_de(p),
                    None => None,
                };
                // `(s)-[old:BELONGS_TO]->(:Project)`: every such edge goes.
                let anciennes: Vec<EdgeId> =
                    w.db.store()
                        .edges_from(noeud, Direction::Outgoing)
                        .into_iter()
                        .filter(|(dst, a)| {
                            w.db.get_edge(*a)
                                .is_some_and(|e| &*e.edge_type == APPARTIENT)
                                && w.db
                                    .get_node_labels(*dst)
                                    .is_some_and(|l| l.iter().any(|x| x == "Project"))
                        })
                        .map(|(_, a)| a)
                        .collect();
                for a in anciennes {
                    w.etape()?;
                    w.db.delete_edge(a);
                }
                if let Some(p) = cible {
                    w.etape()?;
                    w.db.create_edge(noeud, p, APPARTIENT);
                }
                w.etape()?;
                match project_id {
                    Some(p) => {
                        w.db.set_node_property(noeud, "project_id", Value::from(p.to_string()));
                    }
                    None => {
                        w.db.remove_node_property(noeud, "project_id");
                    }
                }
                let nouvelle = relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(nouvelle);
                Ok(())
            },
        )
        .await
    }

    async fn delete_skill(&self, id: Uuid) -> Result<bool> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(e) = entree(w, id)? else {
                    if w.index()?.skills.reserves.contains(&id) {
                        bail!(
                            "le skill {id} est sur disque mais en quarantaine (illisible à \
                             l'ouverture) : il n'est pas supprimé à l'aveugle"
                        );
                    }
                    return Ok(false);
                };
                detacher(w, e.noeud)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.desinscrire(id);
                Ok(true)
            },
        )
        .await
    }

    async fn list_skills(
        &self,
        project_id: Option<Uuid>,
        status: Option<SkillStatus>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<SkillNode>, usize)> {
        self.lire(|_, etat| {
            let mut skills: Vec<&SkillNode> = etat
                .skills
                .par_id
                .values()
                .map(|e| &e.skill)
                .filter(|s| dans_la_portee(s, project_id))
                .filter(|s| status.is_none_or(|st| s.status == st))
                .collect();
            skills.sort_by(|a, b| ordre_energie_nom_id(a, b));
            let total = skills.len();
            let page = skills
                .into_iter()
                .skip(offset)
                .take(limit)
                .cloned()
                .collect();
            Ok((page, total))
        })
    }

    async fn get_skill_members(&self, skill_id: Uuid) -> Result<(Vec<Note>, Vec<DecisionNode>)> {
        self.lire(|_, etat| {
            let Some(e) = etat.skills.par_id.get(&skill_id) else {
                return Ok((vec![], vec![]));
            };
            if e.non_resolus > 0 {
                return refus(
                    "SkillStore::get_skill_members",
                    format!(
                        "skill {skill_id} : {} appartenance(s) que ce backend ne résout pas \
                         (décisions — domaine non porté — ou notes qu'il ne tient pas) ; la liste \
                         serait incomplète",
                        e.non_resolus
                    ),
                );
            }
            let mut notes: Vec<&Note> = Vec::new();
            for (id, noeud) in &e.notes {
                let Some((n, nn, non_resolus)) = etat.notes.consulter(*id) else {
                    continue;
                };
                if nn != *noeud {
                    continue;
                }
                if non_resolus > 0 {
                    return refus(
                        "SkillStore::get_skill_members",
                        format!(
                            "la note membre {id} a {non_resolus} lien(s) que ce backend ne résout \
                             pas : Neo4j les rendrait en ancres"
                        ),
                    );
                }
                notes.push(n);
            }
            // `ORDER BY n.energy DESC, n.id DESC LIMIT 500`.
            notes.sort_by(|a, b| b.energy.total_cmp(&a.energy).then_with(|| b.id.cmp(&a.id)));
            Ok((notes.into_iter().take(500).cloned().collect(), vec![]))
        })
    }

    async fn add_skill_member(
        &self,
        skill_id: Uuid,
        entity_type: &str,
        entity_id: Uuid,
    ) -> Result<()> {
        membre_note("SkillStore::add_skill_member", entity_type)?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let (ancienne, note_noeud) = {
                    let etat = w.index()?;
                    let s = etat.skills.par_id.get(&skill_id).cloned();
                    let n = etat.notes.consulter(entity_id).map(|(_, n, _)| n);
                    match (s, n) {
                        (Some(s), Some(n)) => (s, n),
                        _ => bail!(
                            "Cannot add member: skill {skill_id} or note {entity_id} not found"
                        ),
                    }
                };
                // MERGE: the edge is drawn only if this note is not already a member.
                let deja = aretes_entrantes(w.db, ancienne.noeud, MEMBRE, "Note")
                    .iter()
                    .any(|(src, _)| *src == note_noeud);
                if !deja {
                    w.etape()?;
                    w.db.create_edge(note_noeud, ancienne.noeud, MEMBRE);
                }
                let compte = aretes_entrantes(w.db, ancienne.noeud, MEMBRE, "Note").len() as i64;
                w.etape()?;
                w.db.set_node_property(ancienne.noeud, "note_count", Value::from(compte));
                w.etape()?;
                w.db.set_node_property(
                    ancienne.noeud,
                    "updated_at",
                    Value::from(Utc::now().to_rfc3339()),
                );
                let nouvelle = relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(nouvelle);
                Ok(())
            },
        )
        .await
    }

    async fn remove_skill_member(
        &self,
        skill_id: Uuid,
        entity_type: &str,
        entity_id: Uuid,
    ) -> Result<bool> {
        membre_note("SkillStore::remove_skill_member", entity_type)?;
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = entree(w, skill_id)? else {
                    return Ok(false);
                };
                let note_noeud = w.index()?.notes.consulter(entity_id).map(|(_, n, _)| n);
                let aretes: Vec<EdgeId> = match note_noeud {
                    Some(nn) => aretes_entrantes(w.db, ancienne.noeud, MEMBRE, "Note")
                        .into_iter()
                        .filter(|(src, _)| *src == nn)
                        .map(|(_, a)| a)
                        .collect(),
                    None => vec![],
                };
                let existait = !aretes.is_empty();
                for a in aretes {
                    w.etape()?;
                    w.db.delete_edge(a);
                }
                // As Neo4j: the count and `updated_at` are set even when nothing was removed.
                let compte = aretes_entrantes(w.db, ancienne.noeud, MEMBRE, "Note").len() as i64;
                w.etape()?;
                w.db.set_node_property(ancienne.noeud, "note_count", Value::from(compte));
                w.etape()?;
                w.db.set_node_property(
                    ancienne.noeud,
                    "updated_at",
                    Value::from(Utc::now().to_rfc3339()),
                );
                let nouvelle = relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(nouvelle);
                Ok(existait)
            },
        )
        .await
    }

    async fn remove_all_skill_members(&self, skill_id: Uuid) -> Result<i64> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = entree(w, skill_id)? else {
                    return Ok(0);
                };
                // Every `MEMBER_OF|MEMBER_OF_SKILL` into the skill, whatever its
                // source: an edge is deleted without reading the Decision node.
                let aretes: Vec<EdgeId> =
                    w.db.store()
                        .edges_from(ancienne.noeud, Direction::Incoming)
                        .into_iter()
                        .filter(|(_, a)| {
                            w.db.get_edge(*a).is_some_and(|e| {
                                &*e.edge_type == MEMBRE || &*e.edge_type == MEMBRE_DECISION
                            })
                        })
                        .map(|(_, a)| a)
                        .collect();
                let n = aretes.len() as i64;
                for a in aretes {
                    w.etape()?;
                    w.db.delete_edge(a);
                }
                for (cle, v) in [
                    ("note_count", Value::from(0i64)),
                    ("decision_count", Value::from(0i64)),
                    ("updated_at", Value::from(Utc::now().to_rfc3339())),
                ] {
                    w.etape()?;
                    w.db.set_node_property(ancienne.noeud, cle, v);
                }
                let nouvelle = relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(nouvelle);
                Ok(n)
            },
        )
        .await
    }

    async fn get_skills_for_note(&self, note_id: Uuid) -> Result<Vec<SkillNode>> {
        self.lire(|_, etat| {
            if etat.notes.est_reservee(note_id) {
                bail!(
                    "la note {note_id} est sur disque mais en quarantaine : ses appartenances ne \
                     sont pas lisibles ici"
                );
            }
            let mut skills: Vec<&SkillNode> = etat
                .skills
                .par_note
                .get(&note_id)
                .into_iter()
                .flatten()
                .filter_map(|sid| etat.skills.par_id.get(sid))
                .filter(|e| {
                    e.notes
                        .get(&note_id)
                        .is_some_and(|nn| membre_vivant(&etat.notes, note_id, *nn))
                })
                .map(|e| &e.skill)
                .collect();
            // `ORDER BY s.energy DESC, s.id DESC LIMIT 100`.
            skills.sort_by(|a, b| b.energy.total_cmp(&a.energy).then_with(|| b.id.cmp(&a.id)));
            Ok(skills.into_iter().take(100).cloned().collect())
        })
    }

    async fn get_skills_for_project(&self, project_id: Uuid) -> Result<Vec<SkillNode>> {
        self.lire(|_, etat| {
            let mut skills: Vec<&SkillNode> = etat
                .skills
                .par_id
                .values()
                .map(|e| &e.skill)
                .filter(|s| s.project_id == Some(project_id))
                .collect();
            skills.sort_by(|a, b| ordre_energie_nom_id(a, b));
            Ok(skills.into_iter().take(1000).cloned().collect())
        })
    }

    async fn purge_archived_empty_skills(
        &self,
        project_id: Uuid,
        archived_before: DateTime<Utc>,
    ) -> Result<i64> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let candidats: Vec<(Uuid, NodeId)> = w
                    .index()?
                    .skills
                    .par_id
                    .values()
                    .filter(|e| {
                        e.skill.project_id == Some(project_id)
                            && e.skill.status == SkillStatus::Archived
                            && e.skill.updated_at < archived_before
                    })
                    .map(|e| (e.skill.id, e.noeud))
                    .collect();
                if candidats.is_empty() {
                    return Ok(0);
                }
                // `NOT EXISTS { (pr:Protocol) WHERE pr.skill_id = s.id }`: read on the
                // Protocol nodes themselves, whatever domain owns them.
                let par_protocole: HashSet<String> =
                    w.db.store()
                        .nodes_by_label("Protocol")
                        .into_iter()
                        .filter_map(|p| {
                            w.db.get_node(p).and_then(|n| {
                                n.get_property("skill_id")
                                    .and_then(|v| v.as_str().map(str::to_string))
                            })
                        })
                        .collect();
                let mut supprimes = Vec::new();
                for (id, noeud) in candidats {
                    // Strict: ANY incoming edge keeps the skill (Neo4j: the member and
                    // reference types only).
                    let reference = !w
                        .db
                        .store()
                        .edges_from(noeud, Direction::Incoming)
                        .is_empty();
                    if reference || par_protocole.contains(&id.to_string()) {
                        continue;
                    }
                    detacher(w, noeud)?;
                    supprimes.push(id);
                }
                w.rendre_durable()?;
                let mut etat = w.index_mut()?;
                for id in &supprimes {
                    etat.skills.desinscrire(*id);
                }
                Ok(supprimes.len() as i64)
            },
        )
        .await
    }

    async fn get_live_skills_for_project(&self, project_id: Uuid) -> Result<Vec<SkillNode>> {
        self.lire(|_, etat| {
            let mut skills: Vec<&SkillNode> = etat
                .skills
                .par_id
                .values()
                .map(|e| &e.skill)
                .filter(|s| s.project_id == Some(project_id) && s.status != SkillStatus::Archived)
                .collect();
            skills.sort_by(|a, b| ordre_energie_nom_id(a, b));
            Ok(skills.into_iter().cloned().collect())
        })
    }

    async fn activate_skill(&self, skill_id: Uuid, _query: &str) -> Result<ActivatedSkillContext> {
        let (skill, activated_notes) = self.lire(|db, etat| {
            let e = etat
                .skills
                .par_id
                .get(&skill_id)
                .ok_or_else(|| anyhow!("Skill not found: {skill_id}"))?;
            if e.non_resolus > 0 {
                return refus(
                    "SkillStore::activate_skill",
                    format!(
                        "skill {skill_id} : {} appartenance(s) que ce backend ne résout pas \
                         (décisions — domaine non porté — ou notes qu'il ne tient pas) ; le \
                         contexte serait incomplet",
                        e.non_resolus
                    ),
                );
            }
            // Current knowledge only, read on the RAW node as Neo4j's query does
            // (`n.energy IS NOT NULL AND n.status IN ['active', 'needs_review'] AND
            // n.superseded_by IS NULL`): the note decoding reads an absent energy as
            // 1.0 and an unknown status as `Active`, which would let them in.
            let courantes: Vec<(&Note, f64)> = e
                .notes
                .iter()
                .filter_map(|(id, nn)| {
                    etat.notes
                        .consulter(*id)
                        .filter(|(_, n, _)| n == nn)
                        .and_then(|(note, n, _)| energie_si_courante(db, n).map(|x| (note, x)))
                })
                .collect();
            let energies: Vec<f64> = courantes.iter().map(|(_, x)| *x).collect();
            let seuil = adaptive_threshold(&energies, 0.05, 0.05);
            let mut retenues: Vec<&Note> = courantes
                .into_iter()
                .filter(|(_, x)| *x >= seuil)
                .map(|(n, _)| n)
                .collect();
            retenues.sort_by(|a, b| b.energy.total_cmp(&a.energy).then_with(|| b.id.cmp(&a.id)));
            let notes = retenues
                .into_iter()
                .take(500)
                .map(|n| ActivatedNote {
                    note: n.clone(),
                    activation_score: n.energy,
                    source: ActivationSource::Direct,
                    entity_type: "note".to_string(),
                })
                .collect::<Vec<_>>();
            Ok((e.skill.clone(), notes))
        })?;

        // The context text, as `Neo4jClient::activate_skill` assembles it (no
        // decision: a skill with one was refused above).
        let mut parties = vec![format!("## Skill: {}", skill.name)];
        if !skill.description.is_empty() {
            parties.push(skill.description.clone());
        }
        parties.push(String::new());
        if !activated_notes.is_empty() {
            parties.push("### Notes".to_string());
            for an in &activated_notes {
                parties.push(format!(
                    "- **[{}]** {}: {}",
                    an.note.importance,
                    an.note.note_type,
                    an.note.content.chars().take(200).collect::<String>()
                ));
            }
            parties.push(String::new());
        }
        let context_text = parties.join("\n");

        self.increment_skill_activation(skill_id).await?;
        Ok(ActivatedSkillContext {
            skill,
            activated_notes,
            relevant_decisions: vec![],
            context_text,
            confidence: 1.0,
        })
    }

    async fn increment_skill_activation(&self, skill_id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some(ancienne) = entree(w, skill_id)? else {
                    return Ok(());
                };
                let maintenant = Value::from(Utc::now().to_rfc3339());
                for (cle, v) in [
                    (
                        "activation_count",
                        Value::from(ancienne.skill.activation_count + 1),
                    ),
                    ("last_activated", maintenant.clone()),
                    ("updated_at", maintenant),
                ] {
                    w.etape()?;
                    w.db.set_node_property(ancienne.noeud, cle, v);
                }
                let nouvelle = relire(w, &ancienne)?;
                w.rendre_durable()?;
                w.index_mut()?.skills.publier(nouvelle);
                Ok(())
            },
        )
        .await
    }

    /// The contract's composition, written out (the build requires every method
    /// to be written): one ranking, the one of `rank_skills`.
    async fn search_skills(
        &self,
        project_id: Option<Uuid>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<crate::skills::activation::SkillMatch>> {
        self.search_skills_with_affinity(project_id, query, limit, &HashMap::new())
            .await
    }

    async fn search_skills_with_affinity(
        &self,
        project_id: Option<Uuid>,
        query: &str,
        limit: usize,
        affinity: &HashMap<Uuid, f64>,
    ) -> Result<Vec<crate::skills::activation::SkillMatch>> {
        let (candidats, _) = self
            .list_skills(project_id, Some(SkillStatus::Active), 500, 0)
            .await?;
        let declenches = self.match_skills_by_trigger(project_id, query).await?;
        Ok(crate::skills::activation::rank_skills(
            &candidats,
            &declenches,
            affinity,
            limit,
        ))
    }

    async fn match_skills_by_trigger(
        &self,
        project_id: Option<Uuid>,
        input: &str,
    ) -> Result<Vec<(SkillNode, f64)>> {
        let mut candidats: Vec<SkillNode> = self.lire(|_, etat| {
            let mut c: Vec<&SkillNode> = etat
                .skills
                .par_id
                .values()
                .map(|e| &e.skill)
                .filter(|s| matches!(s.status, SkillStatus::Active | SkillStatus::Emerging))
                .filter(|s| dans_la_portee(s, project_id))
                .collect();
            c.sort_by(|a, b| ordre_energie_nom_id(a, b));
            Ok(c.into_iter().take(500).cloned().collect())
        })?;
        // The regexes are compiled outside the read lock.
        let mut resultats: Vec<(SkillNode, f64)> = candidats
            .drain(..)
            .filter_map(|s| {
                let c = confiance(&s, input);
                (c > 0.0).then_some((s, c))
            })
            .collect();
        // Stable: equal confidences keep the candidate order.
        resultats.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(resultats)
    }

    async fn upsert_published_skill(&self, published: &PublishedSkill) -> Result<()> {
        let p = published.clone();
        self.ecrire(
            move || p,
            move |p, w: &mut Ecriture<'_>| {
                let proprietes = proprietes_publiee(&p)?;
                let existante = {
                    let etat = w.index()?;
                    if etat.skills.publiees_reservees.contains(&p.id) {
                        bail!(
                            "l'id {} est réservé : un nœud PublishedSkill de cet id est sur \
                             disque, illisible à l'ouverture (quarantaine)",
                            p.id
                        );
                    }
                    etat.skills.publiees.get(&p.id).map(|e| e.noeud)
                };
                w.flush_complet = proprietes.iter().any(|(_, v)| en_colonne_blob(v))
                    || existante.is_some_and(|n| {
                        ancienne_en_colonne_blob(w.db, n, proprietes.iter().map(|(k, _)| *k))
                    });
                let noeud = match existante {
                    // MERGE on an existing node: the SET, property by property.
                    Some(n) => {
                        for (cle, valeur) in proprietes {
                            w.etape()?;
                            w.db.set_node_property(n, cle, valeur);
                        }
                        n
                    }
                    None => {
                        let n = w.db.create_node(&[PUBLIEE]);
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
                let lu = lecteur(w.db, noeud).and_then(|get| lire_publiee(&get));
                let publiee = w.apres_ecriture(lu)?;
                w.rendre_durable()?;
                w.index_mut()?
                    .skills
                    .publiees
                    .insert(publiee.id, EntreePubliee { noeud, publiee });
                Ok(())
            },
        )
        .await
    }

    async fn get_published_skill(&self, id: Uuid) -> Result<Option<PublishedSkill>> {
        self.lire(|_, etat| Ok(etat.skills.publiees.get(&id).map(|e| e.publiee.clone())))
    }

    async fn search_published_skills(
        &self,
        query: Option<&str>,
        min_trust: Option<f64>,
        tags: Option<&[String]>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<PublishedSkill>, usize)> {
        if tags.is_some_and(|t| !t.is_empty()) {
            return refus(
                "SkillStore::search_published_skills",
                "filtre `tags` : la référence se contredit (Neo4j teste `tags_json`, qu'il \
                 n'écrit jamais, et ne trouve rien ; le mock teste l'appartenance). Refusé plutôt \
                 que tranché en silence",
            );
        }
        let q = query.map(str::to_lowercase);
        self.lire(|_, etat| {
            let mut liste: Vec<&PublishedSkill> = etat
                .skills
                .publiees
                .values()
                .map(|e| &e.publiee)
                .filter(|p| {
                    q.as_ref().is_none_or(|q| {
                        p.name.to_lowercase().contains(q.as_str())
                            || p.description.to_lowercase().contains(q.as_str())
                    })
                })
                .filter(|p| min_trust.is_none_or(|m| p.trust_score >= m))
                .collect();
            // `ORDER BY ps.trust_score DESC, ps.published_at DESC, ps.id DESC`.
            liste.sort_by(|a, b| {
                b.trust_score
                    .total_cmp(&a.trust_score)
                    .then_with(|| b.published_at.cmp(&a.published_at))
                    .then_with(|| b.id.cmp(&a.id))
            });
            let total = liste.len();
            let page = liste
                .into_iter()
                .skip(offset)
                .take(limit)
                .cloned()
                .collect();
            Ok((page, total))
        })
    }

    async fn increment_published_skill_imports(&self, id: Uuid) -> Result<()> {
        self.ecrire(
            || (),
            move |(), w: &mut Ecriture<'_>| {
                let Some((noeud, n)) = w
                    .index()?
                    .skills
                    .publiees
                    .get(&id)
                    .map(|e| (e.noeud, e.publiee.import_count))
                else {
                    return Ok(());
                };
                w.etape()?;
                w.db.set_node_property(noeud, "import_count", Value::from(n + 1));
                let lu = lecteur(w.db, noeud).and_then(|get| lire_publiee(&get));
                let publiee = w.apres_ecriture(lu)?;
                w.rendre_durable()?;
                w.index_mut()?
                    .skills
                    .publiees
                    .insert(id, EntreePubliee { noeud, publiee });
                Ok(())
            },
        )
        .await
    }

    async fn create_used_skill_relation(
        &self,
        _agent_execution_id: Uuid,
        _skill_id: Uuid,
        _result: &str,
    ) -> Result<()> {
        refus(
            "SkillStore::create_used_skill_relation",
            "l'arête USED_SKILL part d'un AgentExecution : domaine non porté",
        )
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

    #[test]
    fn un_skill_ecrit_se_relit_a_l_identique() {
        let mut s = SkillNode::new(Some(Uuid::new_v4()), "relu");
        s.description = "desc".into();
        s.status = SkillStatus::Active;
        s.trigger_patterns = vec![SkillTrigger::regex("neo4j|cypher", 0.7)];
        s.context_template = Some("tpl".into());
        s.energy = 0.4;
        s.last_activated = Some(Utc::now());
        s.fingerprint = Some("f".into());
        s.tags = vec!["a".into(), "b".into()];
        let props = proprietes_de_creation(&s).unwrap();
        let relu = lire_skill(&depuis(props)).unwrap();
        assert_eq!(
            serde_json::to_value(&relu).unwrap(),
            serde_json::to_value(&s).unwrap()
        );
    }

    #[test]
    fn les_declencheurs_non_fiables_ne_sont_pas_ecrits() {
        let mut s = SkillNode::new(None, "x");
        let mut faible = SkillTrigger::regex("x", 0.5);
        faible.quality_score = Some(0.1);
        s.trigger_patterns = vec![SkillTrigger::regex("y", 0.5), faible];
        let relu = lire_skill(&depuis(proprietes_de_creation(&s).unwrap())).unwrap();
        assert_eq!(relu.trigger_patterns.len(), 1);
        assert_eq!(relu.trigger_patterns[0].pattern_value, "y");
    }

    #[test]
    fn une_valeur_illisible_n_est_jamais_un_defaut() {
        let s = SkillNode::new(None, "x");
        let base = proprietes_de_creation(&s).unwrap();
        for (cle, mauvaise) in [
            ("status", Value::from("zombie")),
            ("trigger_patterns_json", Value::from("pas du json")),
            ("created_at", Value::from("hier")),
            ("energy", Value::from("beaucoup")),
            ("project_id", Value::from("pas-un-uuid")),
            ("tags", Value::from("pas une liste")),
            // The legacy empty scope: served to no project by Neo4j, so not transverse.
            ("project_id", Value::from("")),
        ] {
            let mut p: Vec<(&'static str, Value)> =
                base.iter().filter(|(k, _)| *k != cle).cloned().collect();
            p.push((cle, mauvaise));
            assert!(lire_skill(&depuis(p)).is_err(), "{cle} accepté");
        }
        let sans_nom: Vec<_> = base.into_iter().filter(|(k, _)| *k != "name").collect();
        assert!(lire_skill(&depuis(sans_nom)).is_err());
    }

    #[test]
    fn un_nombre_non_fini_est_refuse_a_l_ecriture() {
        let mut s = SkillNode::new(None, "x");
        s.energy = f64::NAN;
        assert!(proprietes_de_creation(&s).is_err());
    }

    #[test]
    fn la_confiance_suit_le_code_de_neo4j() {
        let mut s = SkillNode::new(None, "x");
        s.status = SkillStatus::Active;
        s.trigger_patterns = vec![SkillTrigger::regex("NEO4J", 0.7)];
        assert_eq!(confiance(&s, "un client neo4j"), 1.0);
        assert_eq!(confiance(&s, "python"), 0.0);
        s.status = SkillStatus::Emerging;
        assert!((confiance(&s, "neo4j") - 0.8).abs() < 1e-12);
        // An unreliable trigger never fires.
        s.trigger_patterns[0].quality_score = Some(0.1);
        assert_eq!(confiance(&s, "neo4j"), 0.0);
        // A glob.
        s.trigger_patterns = vec![SkillTrigger::file_glob("src/neo4j/**", 0.8)];
        assert!(confiance(&s, "src/neo4j/skill.rs") > 0.0);
    }

    #[test]
    fn un_skill_publie_sans_projet_source_est_refuse() {
        let p: PublishedSkill = serde_json::from_value(serde_json::json!({
            "id": Uuid::new_v4(),
            "name": "p",
            "description": "",
            "tags": [],
            "package": {
                "schema_version": 3,
                "metadata": {"format": "po-skill/v3", "exported_at": "2026-01-01T00:00:00Z",
                    "source_project": null,
                    "stats": {"note_count": 0, "decision_count": 0, "trigger_count": 0, "activation_count": 0}},
                "skill": {"name": "p", "description": "", "trigger_patterns": [],
                    "context_template": null, "tags": [], "cohesion": 0.0},
                "notes": [], "decisions": []
            },
            "trust_score": 0.5,
            "trust_components": {"energy": 0.0, "cohesion": 0.0, "activation": 0.0,
                "success_rate": 0.0, "source_projects": 0.0},
            "trust_level": "medium",
            "source_project_id": null,
            "source_project_name": "",
            "published_by": "x",
            "published_at": "2026-01-01T00:00:00Z",
            "import_count": 0
        }))
        .unwrap();
        assert!(proprietes_publiee(&p).is_err());
        let mut avec = p.clone();
        avec.source_project_id = Some(Uuid::new_v4());
        let relu = lire_publiee(&depuis(proprietes_publiee(&avec).unwrap())).unwrap();
        assert_eq!(
            serde_json::to_value(&relu).unwrap(),
            serde_json::to_value(&avec).unwrap()
        );
    }

    /// `tags` is what this backend writes: it wins over a stale `tags_json` left
    /// in an imported base.
    #[test]
    fn les_tags_ecrits_l_emportent_sur_un_tags_json_ancien() {
        let mut p = lire_publiee(&depuis(vec![
            ("id", Value::from(Uuid::new_v4().to_string())),
            ("name", Value::from("p")),
            ("package_json", Value::from(r#"{"schema_version":3,"metadata":{"format":"po-skill/v3","exported_at":"2026-01-01T00:00:00Z","stats":{"note_count":0,"decision_count":0,"trigger_count":0,"activation_count":0}},"skill":{"name":"p"},"notes":[],"decisions":[]}"#)),
            ("source_project_id", Value::from(Uuid::new_v4().to_string())),
            ("published_at", Value::from("2026-01-01T00:00:00Z")),
        ]))
        .unwrap();
        p.tags = vec!["neuf".into()];
        let mut props = proprietes_publiee(&p).unwrap();
        props.push(("tags_json", Value::from(r#"["ancien"]"#)));
        assert_eq!(lire_publiee(&depuis(props)).unwrap().tags, ["neuf"]);
    }
}
