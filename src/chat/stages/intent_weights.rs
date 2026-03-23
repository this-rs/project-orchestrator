//! Intent-aware weight map for knowledge note reweighting.
//!
//! Maps detected user intents to per-note-type score multipliers.
//! Used by `KnowledgeInjectionStage` to prioritize notes that are most
//! relevant to what the user is doing (debugging, planning, coding, etc.).
//!
//! Weight semantics:
//! - `1.0` = neutral (no change to BM25/entity score)
//! - `>1.0` = boost (this note type is more relevant for this intent)
//! - `<1.0` = dampen (this note type is less relevant for this intent)
//!
//! Unknown note types always get weight `1.0` (no penalty).

use std::collections::HashMap;

/// Per-note-type score multipliers for a given intent.
#[derive(Debug, Clone)]
pub struct IntentWeightMap {
    weights: HashMap<String, f64>,
}

impl IntentWeightMap {
    /// Build the weight map for a detected intent.
    ///
    /// Recognized intents: `debug`, `planning`/`plan`, `code`, `review`, `explore`, `general`.
    /// Unrecognized intents fall back to uniform weights (all 1.0).
    pub fn for_intent(intent: &str) -> Self {
        // Check env override first
        if let Some(map) = Self::from_env_override(intent) {
            return map;
        }

        let weights = match intent {
            "debug" => HashMap::from([
                ("gotcha".into(), 1.5),
                ("observation".into(), 1.3),
                ("guideline".into(), 0.7),
                ("tip".into(), 1.2),
                ("pattern".into(), 1.0),
                ("context".into(), 0.8),
                ("assertion".into(), 1.0),
                ("rfc".into(), 0.5),
            ]),
            "planning" | "plan" => HashMap::from([
                ("guideline".into(), 1.5),
                ("pattern".into(), 1.3),
                ("gotcha".into(), 0.8),
                ("rfc".into(), 1.3),
                ("context".into(), 1.2),
                ("tip".into(), 0.8),
                ("observation".into(), 0.7),
                ("assertion".into(), 1.0),
            ]),
            "code" => HashMap::from([
                ("pattern".into(), 1.5),
                ("tip".into(), 1.2),
                ("gotcha".into(), 1.0),
                ("guideline".into(), 0.8),
                ("context".into(), 0.7),
                ("observation".into(), 0.8),
                ("assertion".into(), 0.8),
                ("rfc".into(), 0.5),
            ]),
            "review" => HashMap::from([
                ("assertion".into(), 1.5),
                ("gotcha".into(), 1.3),
                ("guideline".into(), 1.2),
                ("pattern".into(), 1.0),
                ("context".into(), 0.7),
                ("tip".into(), 0.8),
                ("observation".into(), 1.0),
                ("rfc".into(), 1.0),
            ]),
            "explore" => HashMap::from([
                ("context".into(), 1.3),
                ("pattern".into(), 1.2),
                ("observation".into(), 1.2),
                ("tip".into(), 1.0),
                ("gotcha".into(), 1.0),
                ("guideline".into(), 1.0),
                ("assertion".into(), 0.8),
                ("rfc".into(), 1.0),
            ]),
            // "general" and any unrecognized intent → uniform weights
            _ => HashMap::new(),
        };

        Self { weights }
    }

    /// Get the weight multiplier for a note type.
    ///
    /// Returns `1.0` for unknown note types (no penalty).
    pub fn get(&self, note_type: &str) -> f64 {
        let key = note_type.to_lowercase();
        self.weights.get(&key).copied().unwrap_or(1.0)
    }

    /// Try to load intent weights from the `ENRICHMENT_INTENT_WEIGHTS_JSON` env var.
    ///
    /// Expected format: `{"debug": {"gotcha": 1.5, "tip": 1.2}, "planning": {...}}`.
    /// Returns `None` if the env var is not set or if parsing fails for the requested intent.
    #[doc(hidden)]
    pub fn from_env_override(intent: &str) -> Option<Self> {
        let json_str = std::env::var("ENRICHMENT_INTENT_WEIGHTS_JSON").ok()?;
        let parsed: HashMap<String, HashMap<String, f64>> =
            serde_json::from_str(&json_str).ok()?;
        let weights = parsed.get(intent)?;
        Some(Self {
            weights: weights.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_intent_boosts_gotcha() {
        let map = IntentWeightMap::for_intent("debug");
        assert!((map.get("gotcha") - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn debug_intent_dampens_guideline() {
        let map = IntentWeightMap::for_intent("debug");
        assert!((map.get("guideline") - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn planning_intent_boosts_guideline() {
        let map = IntentWeightMap::for_intent("planning");
        assert!((map.get("guideline") - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn plan_alias_same_as_planning() {
        let map = IntentWeightMap::for_intent("plan");
        assert!((map.get("guideline") - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn code_intent_boosts_pattern() {
        let map = IntentWeightMap::for_intent("code");
        assert!((map.get("pattern") - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn review_intent_boosts_assertion() {
        let map = IntentWeightMap::for_intent("review");
        assert!((map.get("assertion") - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn explore_intent_boosts_context() {
        let map = IntentWeightMap::for_intent("explore");
        assert!((map.get("context") - 1.3).abs() < f64::EPSILON);
    }

    #[test]
    fn general_intent_returns_uniform_weights() {
        let map = IntentWeightMap::for_intent("general");
        assert!((map.get("gotcha") - 1.0).abs() < f64::EPSILON);
        assert!((map.get("guideline") - 1.0).abs() < f64::EPSILON);
        assert!((map.get("pattern") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_intent_returns_uniform_weights() {
        let map = IntentWeightMap::for_intent("something_unexpected");
        assert!((map.get("gotcha") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_note_type_returns_one() {
        let map = IntentWeightMap::for_intent("debug");
        assert!((map.get("nonexistent_type") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn case_insensitive_note_type() {
        let map = IntentWeightMap::for_intent("debug");
        assert!((map.get("Gotcha") - 1.5).abs() < f64::EPSILON);
        assert!((map.get("GOTCHA") - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn env_override_parse_and_fallback() {
        // Test env override by calling from_env_override directly to avoid
        // parallel test races with for_intent (which also checks the env var).
        let json = r#"{"debug": {"gotcha": 2.0, "tip": 0.5}}"#;
        std::env::set_var("ENRICHMENT_INTENT_WEIGHTS_JSON", json);

        // Override is active for "debug"
        let map = IntentWeightMap::from_env_override("debug").expect("should parse debug");
        assert!((map.get("gotcha") - 2.0).abs() < f64::EPSILON);
        assert!((map.get("tip") - 0.5).abs() < f64::EPSILON);
        assert!((map.get("pattern") - 1.0).abs() < f64::EPSILON); // unknown → 1.0

        // "planning" not in the override → None (caller falls back to built-in)
        assert!(
            IntentWeightMap::from_env_override("planning").is_none(),
            "Missing intent should return None"
        );

        std::env::remove_var("ENRICHMENT_INTENT_WEIGHTS_JSON");

        // Without env var, from_env_override returns None
        assert!(IntentWeightMap::from_env_override("debug").is_none());
    }
}
