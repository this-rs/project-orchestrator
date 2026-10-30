//! Le CATALOGUE du schéma du graphe, et la marche dans une chaîne de causalité.
//!
//! # Le problème que ce module résout
//!
//! Le graphe porte 77 types de relations et 70 labels, mais il n'existait aucun point de
//! lecture du *schéma* : quels types relient quels labels, avec quels volumes. Un agent
//! qui veut écrire une chaîne de causalité devait donc deviner les types — et devinait
//! faux, ou écrivait la causalité en texte libre, non requêtable.
//!
//! Le catalogue est **mesuré dans la base**, jamais écrit à la main : il décrit ce qui
//! existe, pas ce qu'on croit exister. Un type qui disparaît disparaît du catalogue ; un
//! type nouveau y entre tout seul.
//!
//! # Deux mesures, et pourquoi elles sont séparées
//!
//! La FORME (quel type relie quel label) se lit vite, mais la source qui la donne
//! (`apoc.meta.schema`) **compte faux** : mesuré le 30/10, elle annonçait 0 `INFORMED_BY`
//! là où le graphe en porte 833, et 13 898 `LINKED_TO` là où il en porte 305 706. Ses
//! volumes sont donc jetés.
//!
//! Le VOLUME par type se lit exactement, par le count-store (`MATCH ()-[r:T]->()
//! RETURN count(r)`), un appel par type. Le catalogue porte donc les deux : la forme
//! d'une source approximative, les volumes d'une source exacte, et **chaque nombre sait
//! d'où il vient** — plutôt qu'un seul nombre dont on ignore s'il est vrai.
//!
//! # Ce que « causalité » veut dire ici
//!
//! Le graphe ne porte pas une relation `CAUSES`. Il porte une **famille** de relations
//! typées, chacune à un étage : une tâche est `INFORMED_BY` une décision, elle est
//! `RESOLVED_BY` un commit ; un plan `RESULTED_IN` un commit ; une note `SUPERSEDES` une
//! autre. Une chaîne de causalité est donc une **composition** de ces types, pas un type
//! unique — c'est ce que [`CAUSAL_TYPES`] déclare et que [`CausalWalk`] traverse.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// La famille des relations qui portent une raison, mesurée dans la base avant d'être
/// déclarée ici : `RESOLVED_BY` (Task→Commit), `RESULTED_IN` (Plan→Commit),
/// `INFORMED_BY` (Task→Decision), `NEXT_DECISION`, `SUPERSEDES`, `PRECEDES`.
pub const CAUSAL_TYPES: &[&str] = &[
    "INFORMED_BY",
    "NEXT_DECISION",
    "PRECEDES",
    "RESOLVED_BY",
    "RESULTED_IN",
    "SUPERSEDES",
];

/// The labels a causal walk may start from. A label is interpolated into a query, so it is
/// matched against this closed list first: an unknown label is an error, never a query
/// built from caller input.
pub const CAUSAL_ANCHORS: &[&str] = &[
    "Commit",
    "ConversationSegment",
    "Decision",
    "Note",
    "Plan",
    "Task",
    "TrajectoryNode",
];

/// One way a relationship is used: which label it leaves and which it reaches. The volume
/// lives in [`SchemaCatalog::volumes`], per TYPE, because per-pair counts are not exact in
/// any source the graph offers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RelationShape {
    pub edge: String,
    pub from: String,
    pub to: String,
}

/// What the graph can say about itself: its labels, the shapes of its relationships, and
/// how many edges of each type it carries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaCatalog {
    pub labels: Vec<String>,
    /// The relationship types the graph declares, from `db.relationshipTypes()` — **exact**,
    /// unlike the shapes: `apoc.meta.schema` reads statistics and does not see an edge
    /// written a second ago (measured: a fresh `KNOWS` was absent from its report).
    pub types: Vec<String>,
    pub edges: Vec<RelationShape>,
    /// Edges per type, from the count store. A type absent from this map has an unknown
    /// volume — which is not the same as zero, and is never published as zero.
    pub volumes: BTreeMap<String, u64>,
}

impl SchemaCatalog {
    /// The label pairs this type connects.
    pub fn shapes_of(&self, edge: &str) -> Vec<&RelationShape> {
        self.edges.iter().filter(|s| s.edge == edge).collect()
    }

    /// Whether the graph actually carries this type. An agent that asks before it writes
    /// needs this answer, so it comes from the ONE source that is exact: the declared types.
    ///
    /// The shapes are deliberately NOT consulted. Measured on CI: `apoc.meta.schema` keeps
    /// describing a type after its last edge is deleted, while `db.relationshipTypes()` stops
    /// listing it. Trusting the union made `has_edge` answer « yes » for a type the graph no
    /// longer had — the inverse of the freshness problem, and the same lie.
    ///
    /// `types` empty means the catalog was built without that read; only then do the shapes
    /// answer, and they do so as witnesses, not as authorities.
    pub fn has_edge(&self, edge: &str) -> bool {
        if self.types.is_empty() {
            return self.edges.iter().any(|s| s.edge == edge);
        }
        self.types.iter().any(|t| t == edge)
    }

    /// The label pairs measured for a type. Empty does NOT mean the type is absent — it may
    /// simply be younger than the statistics. [`Self::has_edge`] is the answer to existence.
    pub fn shapes_of_measured(&self, edge: &str) -> Vec<&RelationShape> {
        self.shapes_of(edge)
    }

    /// The measured volume of a type, when it is known.
    pub fn volume(&self, edge: &str) -> Option<u64> {
        self.volumes.get(edge).copied()
    }

    /// The types that carry a reason, and only those the graph really has — the declared
    /// family filtered through the measurement.
    pub fn causal_edges(&self) -> Vec<&str> {
        let mut out: Vec<&str> = CAUSAL_TYPES
            .iter()
            .copied()
            .filter(|t| self.has_edge(t))
            .collect();
        out.sort_unstable();
        out
    }

    /// The distinct relationship types described.
    pub fn edge_types(&self) -> usize {
        let mut seen: Vec<&str> = self.edges.iter().map(|s| s.edge.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }
}

/// One hop of a causal chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalStep {
    /// Depth from the anchor, 1 being the immediate neighbour.
    pub depth: u32,
    pub edge: String,
    pub label: String,
    pub id: String,
    /// The label's display name when it has one (`title`, `name`, `slug`).
    pub title: Option<String>,
}

/// The result of walking forward from an anchor.
///
/// `truncated` is published with `steps` because « la part lue est dite » : a walk stopped
/// by its own depth limit must not read as « the chain ends here ».
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalWalk {
    pub anchor_label: String,
    pub anchor_id: String,
    pub max_depth: u32,
    pub edges_followed: Vec<String>,
    pub steps: Vec<CausalStep>,
    pub truncated: bool,
}

impl CausalWalk {
    /// The chain exists when the walk reached anything at all. An empty walk is the honest
    /// answer to « this chain does not exist »; « I could not look » is an `Err`, and the
    /// two must never be confused.
    pub fn exists(&self) -> bool {
        !self.steps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(edge: &str, from: &str, to: &str) -> RelationShape {
        RelationShape {
            edge: edge.into(),
            from: from.into(),
            to: to.into(),
        }
    }

    fn catalog(edges: Vec<RelationShape>, volumes: &[(&str, u64)]) -> SchemaCatalog {
        SchemaCatalog {
            labels: vec![],
            types: edges.iter().map(|s| s.edge.clone()).collect(),
            edges,
            volumes: volumes
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        }
    }

    #[test]
    fn the_causal_family_is_measured_not_declared() {
        // A catalog carrying only one of the declared types must report one: the declared
        // list is a question put to the graph, not an answer.
        let c = catalog(
            vec![
                shape("RESOLVED_BY", "Task", "Commit"),
                shape("CONTAINS", "File", "Function"),
            ],
            &[("RESOLVED_BY", 3679), ("CONTAINS", 1_016_375)],
        );
        assert_eq!(c.causal_edges(), vec!["RESOLVED_BY"]);
        assert!(c.has_edge("CONTAINS"));
        assert!(!c.has_edge("CAUSES"), "an invented type is absent");
    }

    #[test]
    fn a_type_can_connect_several_label_pairs() {
        // KNOWS is a live example: Persona→File today, Agent→Note tomorrow. A catalog that
        // assumed one pair per type would hide the second — and the two meanings would
        // contaminate each other silently.
        let c = catalog(
            vec![
                shape("KNOWS", "Persona", "File"),
                shape("KNOWS", "Agent", "Note"),
            ],
            &[("KNOWS", 792)],
        );
        assert_eq!(c.shapes_of("KNOWS").len(), 2);
        assert_eq!(c.edge_types(), 1, "one type, two shapes");
        assert_eq!(c.volume("KNOWS"), Some(792));
    }

    #[test]
    fn a_stale_shape_does_not_make_a_type_exist() {
        // Measured on CI: apoc kept describing KNOWS after its last edge was deleted, while
        // `db.relationshipTypes()` no longer listed it. The exact source decides; a shape is
        // a witness, never an authority.
        let c = SchemaCatalog {
            labels: vec!["Agent".into()],
            types: vec!["CONTAINS".into()],
            edges: vec![shape("KNOWS", "Agent", "Note")],
            volumes: [("CONTAINS".to_string(), 1u64)].into_iter().collect(),
        };
        assert!(
            !c.has_edge("KNOWS"),
            "a shape alone must not answer « this type exists »"
        );
        assert!(c.has_edge("CONTAINS"));

        // Without the exact read at all, the shapes are all we have — and they answer.
        let fallback = SchemaCatalog {
            labels: vec![],
            types: vec![],
            edges: vec![shape("KNOWS", "Agent", "Note")],
            volumes: Default::default(),
        };
        assert!(fallback.has_edge("KNOWS"));
    }

    #[test]
    fn an_unknown_volume_is_not_zero() {
        // The distinction the graph API keeps getting wrong: absent is not empty.
        let c = catalog(vec![shape("KNOWS", "Agent", "Note")], &[]);
        assert_eq!(c.volume("KNOWS"), None);
        assert_ne!(c.volume("KNOWS"), Some(0));
    }

    #[test]
    fn an_empty_walk_says_the_chain_does_not_exist() {
        let walk = CausalWalk::default();
        assert!(!walk.exists());
    }

    #[test]
    fn the_causal_anchors_are_a_closed_list() {
        // The label is interpolated into a query: this list is the guard, so it must stay
        // explicit and hold nothing a caller could reach the graph with.
        for label in CAUSAL_ANCHORS {
            assert!(
                label.chars().all(|c| c.is_ascii_alphanumeric()),
                "{label} is not a bare label"
            );
        }
        assert!(CAUSAL_ANCHORS.contains(&"Task"));
        assert!(CAUSAL_ANCHORS.contains(&"Decision"));
    }

    #[test]
    fn the_causal_family_is_free_of_injection() {
        for edge in CAUSAL_TYPES {
            assert!(
                edge.chars().all(|c| c == '_' || c.is_ascii_uppercase()),
                "{edge} is not a bare relationship type"
            );
        }
    }
}
