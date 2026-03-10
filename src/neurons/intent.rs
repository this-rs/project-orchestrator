//! Intent Detection for adaptive memory retrieval.
//!
//! Detects query intent using lightweight heuristics (< 1ms, no LLM)
//! to dynamically weight the spreading activation algorithm.
//! Bilingual FR/EN support.

use serde::{Deserialize, Serialize};

/// The detected intent mode for a query, used to weight spreading activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntentMode {
    /// Debugging intent — highest priority. Triggered by error/failure keywords.
    Debug,
    /// Exploration intent — understanding how something works.
    Explore,
    /// Impact analysis intent — assessing consequences of a change.
    Impact,
    /// Planning intent — creating or implementing something new.
    Plan,
    /// Fallback when no specific intent is detected.
    Default,
}

impl std::fmt::Display for QueryIntentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            QueryIntentMode::Debug => "debug",
            QueryIntentMode::Explore => "explore",
            QueryIntentMode::Impact => "impact",
            QueryIntentMode::Plan => "plan",
            QueryIntentMode::Default => "default",
        };
        write!(f, "{}", name)
    }
}

const DEBUG_KEYWORDS: &[&str] = &[
    "pourquoi",
    "échoue",
    "bug",
    "erreur",
    "crash",
    "plantage",
    "why",
    "fail",
    "broken",
    "error",
    "panic",
    "trace",
];

const EXPLORE_KEYWORDS: &[&str] = &[
    "comment",
    "fonctionne",
    "architecture",
    "structure",
    "how does",
    "explain",
    "what is",
    "overview",
    "comprendre",
    "understand",
];

const IMPACT_KEYWORDS: &[&str] = &[
    "impact",
    "modifier",
    "changer",
    "refactor",
    "change",
    "modify",
    "rename",
    "move",
    "delete",
    "supprim",
    "déplacer",
];

const PLAN_KEYWORDS: &[&str] = &[
    "planifie",
    "ajouter",
    "implémenter",
    "plan",
    "implement",
    "create",
    "build",
    "add feature",
    "nouvelle",
    "design",
];

/// Lightweight intent detector using keyword heuristics.
///
/// Priority order: Debug > Impact > Plan > Explore > Default.
/// Debug has highest priority because debugging is urgent.
pub struct IntentDetector;

impl IntentDetector {
    /// Detect the query intent from a raw query string.
    ///
    /// This is a pure function with no async or state — just keyword matching
    /// on the lowercased query. Designed to complete in < 1ms.
    pub fn detect(query: &str) -> QueryIntentMode {
        let lower = query.to_lowercase();

        if Self::matches_any(&lower, DEBUG_KEYWORDS) {
            return QueryIntentMode::Debug;
        }
        if Self::matches_any(&lower, IMPACT_KEYWORDS) {
            return QueryIntentMode::Impact;
        }
        if Self::matches_any(&lower, PLAN_KEYWORDS) {
            return QueryIntentMode::Plan;
        }
        if Self::matches_any(&lower, EXPLORE_KEYWORDS) {
            return QueryIntentMode::Explore;
        }

        QueryIntentMode::Default
    }

    fn matches_any(query: &str, keywords: &[&str]) -> bool {
        keywords.iter().any(|kw| query.contains(kw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Debug queries (FR) ──────────────────────────────────────────

    #[test]
    fn debug_fr_pourquoi_paiement() {
        assert_eq!(
            IntentDetector::detect("pourquoi le paiement échoue"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn debug_fr_bug_dans_le_cache() {
        assert_eq!(
            IntentDetector::detect("il y a un bug dans le cache"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn debug_fr_erreur_connexion() {
        assert_eq!(
            IntentDetector::detect("erreur de connexion à la base"),
            QueryIntentMode::Debug
        );
    }

    // ── Debug queries (EN) ──────────────────────────────────────────

    #[test]
    fn debug_en_why_auth_fail() {
        assert_eq!(
            IntentDetector::detect("why does auth fail"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn debug_en_broken_pipeline() {
        assert_eq!(
            IntentDetector::detect("the CI pipeline is broken"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn debug_en_panic_handler() {
        assert_eq!(
            IntentDetector::detect("panic in the request handler"),
            QueryIntentMode::Debug
        );
    }

    // ── Explore queries (FR) ────────────────────────────────────────

    #[test]
    fn explore_fr_comment_cache() {
        assert_eq!(
            IntentDetector::detect("comment fonctionne le cache"),
            QueryIntentMode::Explore
        );
    }

    #[test]
    fn explore_fr_architecture_api() {
        assert_eq!(
            IntentDetector::detect("quelle est l'architecture de l'API"),
            QueryIntentMode::Explore
        );
    }

    #[test]
    fn explore_fr_comprendre_module() {
        assert_eq!(
            IntentDetector::detect("je veux comprendre ce module"),
            QueryIntentMode::Explore
        );
    }

    // ── Explore queries (EN) ────────────────────────────────────────

    #[test]
    fn explore_en_how_does_api() {
        assert_eq!(
            IntentDetector::detect("how does the API work"),
            QueryIntentMode::Explore
        );
    }

    #[test]
    fn explore_en_explain_caching() {
        assert_eq!(
            IntentDetector::detect("explain the caching strategy"),
            QueryIntentMode::Explore
        );
    }

    #[test]
    fn explore_en_what_is_synapse() {
        assert_eq!(
            IntentDetector::detect("what is a synapse relation"),
            QueryIntentMode::Explore
        );
    }

    // ── Impact queries (FR) ─────────────────────────────────────────

    #[test]
    fn impact_fr_modifier_auth() {
        assert_eq!(
            IntentDetector::detect("impact si je modifie auth.rs"),
            QueryIntentMode::Impact
        );
    }

    #[test]
    fn impact_fr_supprimer_table() {
        assert_eq!(
            IntentDetector::detect("que se passe-t-il si je supprime cette table"),
            QueryIntentMode::Impact
        );
    }

    #[test]
    fn impact_fr_changer_schema() {
        assert_eq!(
            IntentDetector::detect("je veux changer le schéma de données"),
            QueryIntentMode::Impact
        );
    }

    // ── Impact queries (EN) ─────────────────────────────────────────

    #[test]
    fn impact_en_change_schema() {
        assert_eq!(
            IntentDetector::detect("what if I change the schema"),
            QueryIntentMode::Impact
        );
    }

    #[test]
    fn impact_en_rename_module() {
        assert_eq!(
            IntentDetector::detect("rename the auth module"),
            QueryIntentMode::Impact
        );
    }

    #[test]
    fn impact_en_refactor_service() {
        assert_eq!(
            IntentDetector::detect("refactor the payment service"),
            QueryIntentMode::Impact
        );
    }

    // ── Plan queries (FR) ───────────────────────────────────────────

    #[test]
    fn plan_fr_planifie_endpoint() {
        assert_eq!(
            IntentDetector::detect("planifie l'ajout d'un endpoint"),
            QueryIntentMode::Plan
        );
    }

    #[test]
    fn plan_fr_ajouter_middleware() {
        assert_eq!(
            IntentDetector::detect("ajouter un middleware de rate limiting"),
            QueryIntentMode::Plan
        );
    }

    #[test]
    fn plan_fr_nouvelle_feature() {
        assert_eq!(
            IntentDetector::detect("nouvelle fonctionnalité de notifications"),
            QueryIntentMode::Plan
        );
    }

    // ── Plan queries (EN) ───────────────────────────────────────────

    #[test]
    fn plan_en_implement_registration() {
        assert_eq!(
            IntentDetector::detect("implement user registration"),
            QueryIntentMode::Plan
        );
    }

    #[test]
    fn plan_en_build_dashboard() {
        assert_eq!(
            IntentDetector::detect("build an admin dashboard"),
            QueryIntentMode::Plan
        );
    }

    #[test]
    fn plan_en_add_feature_oauth() {
        assert_eq!(
            IntentDetector::detect("add feature for OAuth integration"),
            QueryIntentMode::Plan
        );
    }

    // ── Default queries (no specific intent) ────────────────────────

    #[test]
    fn default_payment_service() {
        assert_eq!(
            IntentDetector::detect("payment service"),
            QueryIntentMode::Default
        );
    }

    #[test]
    fn default_auth_handler() {
        assert_eq!(
            IntentDetector::detect("auth handler"),
            QueryIntentMode::Default
        );
    }

    #[test]
    fn default_neo4j_client() {
        assert_eq!(
            IntentDetector::detect("Neo4j client"),
            QueryIntentMode::Default
        );
    }

    #[test]
    fn default_middleware_stack() {
        assert_eq!(
            IntentDetector::detect("middleware stack"),
            QueryIntentMode::Default
        );
    }

    #[test]
    fn default_database_pool() {
        assert_eq!(
            IntentDetector::detect("database connection pool"),
            QueryIntentMode::Default
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn edge_empty_query() {
        assert_eq!(IntentDetector::detect(""), QueryIntentMode::Default);
    }

    #[test]
    fn edge_mixed_debug_wins_over_explore() {
        // "why" (Debug) + "how does" (Explore) → Debug wins
        assert_eq!(
            IntentDetector::detect("why and how does this work"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn edge_mixed_impact_wins_over_plan() {
        // "change" (Impact) + "implement" (Plan) → Impact wins
        assert_eq!(
            IntentDetector::detect("change and implement the auth flow"),
            QueryIntentMode::Impact
        );
    }

    #[test]
    fn edge_debug_wins_over_impact() {
        // "error" (Debug) + "change" (Impact) → Debug wins
        assert_eq!(
            IntentDetector::detect("error when I change config"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn edge_case_insensitive() {
        assert_eq!(
            IntentDetector::detect("WHY DOES THIS CRASH"),
            QueryIntentMode::Debug
        );
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", QueryIntentMode::Debug), "debug");
        assert_eq!(format!("{}", QueryIntentMode::Explore), "explore");
        assert_eq!(format!("{}", QueryIntentMode::Impact), "impact");
        assert_eq!(format!("{}", QueryIntentMode::Plan), "plan");
        assert_eq!(format!("{}", QueryIntentMode::Default), "default");
    }
}
