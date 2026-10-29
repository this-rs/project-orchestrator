//! Topic extraction for conversation segments.
//!
//! Deterministic, bilingual (FR/EN) subject labelling of a raw message, used by
//! the segment model to build the `Topic` dimension. Pure keyword matching: a
//! label is emitted iff one of its curated keywords appears in the message.
//! No network, no LLM, no NLP, no hashing of arbitrary words.
//!
//! Matching is case-insensitive. Long words and multi-word phrases match as
//! plain substrings (so `"review"` still fires on `"reviewed"`, `"test"` on
//! `"testing"`); short single words (three characters or fewer) match only as
//! whole words, so abbreviations like `"ci"`, `"api"` or `"fix"` do not fire
//! inside unrelated words (`"merci"`, `"rapide"`, `"prefix"`).
#[allow(unused_imports)]
use crate::neo4j::traits::*;

/// A single-word keyword this short (or shorter) is matched against whole words
/// rather than as a substring. Longer words are matched as substrings so their
/// inflections stay covered without enumerating every form.
const SHORT_KEYWORD_LEN: usize = 3;

/// One normalized subject label and the FR/EN keywords that trigger it.
///
/// Labels are already normalized in the table (lowercase, trimmed, non-empty),
/// and each label appears at most once, so the output of [`topic_labels`] is
/// inherently deduplicated and in a stable order (table order).
struct TopicRule {
    /// Canonical label, already lowercase.
    label: &'static str,
    /// Curated keywords; matched case-insensitively. Short single words match
    /// as whole words, longer words and phrases as substrings (see
    /// [`keyword_matches`]).
    keywords: &'static [&'static str],
}

const TOPIC_RULES: &[TopicRule] = &[
    TopicRule {
        label: "routing",
        keywords: &["routing", "routage", "route", "router"],
    },
    TopicRule {
        label: "provider",
        keywords: &["provider", "fournisseur"],
    },
    TopicRule {
        label: "model",
        keywords: &["model", "modèle", "llm", "gpt", "claude", "gemini"],
    },
    TopicRule {
        label: "auth",
        keywords: &[
            "auth",
            "authentication",
            "authentification",
            "oauth",
            "jwt",
            "login",
            "sso",
            "token",
            "jeton",
        ],
    },
    TopicRule {
        label: "neo4j",
        keywords: &["neo4j", "cypher"],
    },
    TopicRule {
        label: "graph",
        keywords: &["graph", "graphe"],
    },
    TopicRule {
        label: "review",
        keywords: &[
            "review",
            "reviewed",
            "reviewing",
            "reviews",
            "pr",
            "pull request",
            "merge request",
            "revue",
            "relire",
            "relecture",
        ],
    },
    TopicRule {
        label: "frontend",
        keywords: &[
            "frontend",
            "front-end",
            "front end",
            "ui/ux",
            "react",
            "angular",
            "svelte",
            "css",
            "html",
            "javascript",
            "typescript",
        ],
    },
    TopicRule {
        label: "packaging",
        keywords: &[
            "packaging",
            "package",
            "empaquetage",
            "paquet",
            "crate",
            "cargo",
        ],
    },
    TopicRule {
        label: "docker",
        keywords: &["docker", "container", "conteneur"],
    },
    TopicRule {
        label: "deployment",
        keywords: &[
            "deployment",
            "deploy",
            "déploiement",
            "déployer",
            "release",
            "production",
            "mise en production",
        ],
    },
    TopicRule {
        label: "ci",
        keywords: &[
            "ci",
            "ci/cd",
            "continuous integration",
            "intégration continue",
            "pipeline",
            "github actions",
            "gitlab",
            "jenkins",
            "build",
        ],
    },
    TopicRule {
        label: "tests",
        keywords: &["test", "testing", "assert"],
    },
    TopicRule {
        label: "debug",
        keywords: &[
            "debug",
            "debugging",
            "bug",
            "bugs",
            "buggy",
            "fix",
            "fixed",
            "fixes",
            "fixing",
            "crash",
            "fail",
            "failed",
            "failure",
            "stack trace",
            "stacktrace",
            "traceback",
            "error",
            "exception",
            "segfault",
            "bogue",
            "débogage",
            "corriger",
            "corrige",
            "corrigé",
            "plantage",
            "erreur",
            "trace de pile",
        ],
    },
    TopicRule {
        label: "docs",
        keywords: &["docs", "documentation", "readme", "document"],
    },
    TopicRule {
        label: "chat",
        keywords: &["chat", "conversation", "websocket", "message", "session"],
    },
    TopicRule {
        label: "mcp",
        keywords: &["mcp", "model context protocol"],
    },
    TopicRule {
        label: "vault",
        keywords: &["vault", "secret", "coffre", "credential"],
    },
    TopicRule {
        label: "database",
        keywords: &[
            "database",
            "base de données",
            "postgres",
            "sql",
            "sqlite",
            "mysql",
            "query",
            "requête",
            "migration",
            "schema",
            "schéma",
        ],
    },
    TopicRule {
        label: "api",
        keywords: &[
            "api", "endpoint", "http", "graphql", "rest api", "restful", "openapi", "swagger",
        ],
    },
    TopicRule {
        label: "cache",
        keywords: &["cache", "redis", "memoization", "mémoisation"],
    },
    TopicRule {
        label: "config",
        keywords: &[
            "config",
            "configuration",
            "settings",
            "yaml",
            "toml",
            "json",
            "env",
            "environment",
            "environnement",
            "réglage",
            "paramètre",
            "variable d'environnement",
        ],
    },
    TopicRule {
        label: "refactor",
        keywords: &[
            "refactor",
            "refactoring",
            "refactored",
            "refactorisation",
            "refactoriser",
            "cleanup",
            "clean up",
            "restructure",
            "restructurer",
        ],
    },
    TopicRule {
        label: "performance",
        keywords: &[
            "performance",
            "optimize",
            "optimization",
            "optimisation",
            "optimiser",
            "benchmark",
            "latency",
            "latence",
            "throughput",
            "débit",
            "perf",
            "slow",
            "lenteur",
        ],
    },
];

/// Whether `keyword` matches `message` (both already lower-cased).
///
/// A keyword that is a single word of at most [`SHORT_KEYWORD_LEN`] characters
/// matches only when it is not part of a longer word: the characters before and
/// after it (if any) must not be alphanumeric. Anything else — a longer word or
/// a phrase carrying its own separators (spaces, hyphens, slashes) — matches as
/// a plain substring.
fn keyword_matches(message: &str, keyword: &str) -> bool {
    let single_word = keyword.chars().all(|c| c.is_alphanumeric());
    let short = keyword.chars().count() <= SHORT_KEYWORD_LEN;
    if !single_word || !short {
        return message.contains(keyword);
    }
    message.match_indices(keyword).any(|(start, _)| {
        let before_ok = start == 0
            || message[..start]
                .chars()
                .next_back()
                .is_some_and(|c| !c.is_alphanumeric());
        let end = start + keyword.len();
        let after_ok = end >= message.len()
            || message[end..]
                .chars()
                .next()
                .is_some_and(|c| !c.is_alphanumeric());
        before_ok && after_ok
    })
}

/// Returns the normalized subject labels present in `message`.
///
/// A label is emitted iff at least one of its curated keywords matches the
/// message (case-insensitive; see [`keyword_matches`]). The returned labels are
/// always lowercase, trimmed, non-empty, deduplicated, and in a stable order
/// (the order of the table above, never the order in which keywords appear in
/// the message). A message with no known keyword returns an empty vec: no
/// evidence, no invented topic.
pub fn topic_labels(message: &str) -> Vec<String> {
    let lower = message.to_lowercase();

    TOPIC_RULES
        .iter()
        .filter(|rule| rule.keywords.iter().any(|kw| keyword_matches(&lower, kw)))
        .map(|rule| rule.label.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn french_and_english_map_to_the_same_label() {
        let fr = topic_labels("le routage du provider et l'authentification");
        let en = topic_labels("the routing of the provider and the authentication");
        assert_eq!(
            fr,
            vec![
                "routing".to_string(),
                "provider".to_string(),
                "auth".to_string()
            ]
        );
        assert_eq!(fr, en);
    }

    #[test]
    fn an_unknown_message_is_empty() {
        assert!(topic_labels("the weather in antibes is lovely").is_empty());
        assert!(topic_labels("").is_empty());
    }

    #[test]
    fn normalization_case_and_trim() {
        // Leading/trailing whitespace and casing must not matter; labels come
        // back lowercase.
        assert_eq!(
            topic_labels("  DOCKER and Neo4J  "),
            vec!["neo4j".to_string(), "docker".to_string()]
        );
    }

    #[test]
    fn dedup_and_stable_order() {
        // Repeated keywords collapse to one label, and the order follows the
        // table (neo4j, docker, tests), not the message order.
        assert_eq!(
            topic_labels("docker docker container and tests tests and neo4j"),
            vec![
                "neo4j".to_string(),
                "docker".to_string(),
                "tests".to_string()
            ]
        );
    }

    #[test]
    fn french_accents_are_handled() {
        assert_eq!(
            topic_labels("le DÉPLOIEMENT docker"),
            vec!["docker".to_string(), "deployment".to_string()]
        );
    }

    #[test]
    fn review_pr_and_french_revue_map_to_review() {
        assert_eq!(topic_labels("review PR #758"), vec!["review".to_string()]);
        assert_eq!(
            topic_labels("passer à la revue de la PR"),
            vec!["review".to_string()]
        );
        assert_eq!(
            topic_labels("I reviewed the pull request"),
            vec!["review".to_string()]
        );
    }

    #[test]
    fn a_topic_transition_between_subjects_is_visible() {
        // The phrase the review flagged: "I finished debugging the router, now
        // let me review PR #758". It carries three subjects at once, in table
        // order (routing, review, debug), not message order.
        assert_eq!(
            topic_labels("I finished debugging the router, now let me review PR #758"),
            vec![
                "routing".to_string(),
                "review".to_string(),
                "debug".to_string()
            ]
        );
    }

    #[test]
    fn debug_and_refactor_labels_are_detected() {
        assert_eq!(
            topic_labels("fix the crash caused by the bug"),
            vec!["debug".to_string()]
        );
        assert_eq!(
            topic_labels("il faut corriger le plantage"),
            vec!["debug".to_string()]
        );
        assert_eq!(
            topic_labels("refactor the module and clean up the code"),
            vec!["refactor".to_string()]
        );
    }

    #[test]
    fn ci_and_database_sql_labels_are_detected() {
        assert_eq!(
            topic_labels("run the CI pipeline build"),
            vec!["ci".to_string()]
        );
        assert_eq!(
            topic_labels("the ci/cd and continuous integration"),
            vec!["ci".to_string()]
        );
        assert_eq!(
            topic_labels("run this SQL query against the database"),
            vec!["database".to_string()]
        );
    }

    #[test]
    fn short_words_do_not_fire_inside_longer_words() {
        // `ci` in `merci`/`ici`, `api` in `rapide`, `fix` in `prefix`/`suffix`
        // must not match: these abbreviations are matched as whole words.
        assert!(topic_labels("merci beaucoup").is_empty());
        assert!(topic_labels("c'est ici").is_empty());
        assert!(topic_labels("c'est trop rapide").is_empty());
        assert!(topic_labels("prefix and suffix").is_empty());
        // The same abbreviations, as whole words, do match.
        assert_eq!(
            topic_labels("fix the CI"),
            vec!["ci".to_string(), "debug".to_string()]
        );
    }
}
