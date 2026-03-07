//! Observation Detector — detects implicit insights in LLM responses.
//!
//! Analyzes response text for patterns indicating knowledge worth capturing:
//! - **Bug resolution**: "the problem was", "root cause", "le problème était"
//! - **Conventions**: "always use", "never do", "il faut toujours"
//! - **Gotchas**: "watch out", "be careful", "attention à", "piège"
//! - **Patterns**: "this pattern", "recurring pattern", "on retrouve ce pattern"
//!
//! When a pattern matches with sufficient confidence, produces a `suggestion_card`
//! VizBlock with pre-filled note content for user approval.
//!
//! **Constraint**: Maximum 1 suggestion per response, confidence > 0.8.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

use crate::chat::viz::{VizBlock, VizType};

// ============================================================================
// Types
// ============================================================================

/// A detected observation that could become a knowledge note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObservation {
    /// Suggested note type (gotcha, pattern, guideline, tip)
    pub note_type: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// The matched pattern that triggered detection
    pub trigger_pattern: String,
    /// Surrounding context extracted from the response
    pub context_excerpt: String,
    /// Pre-filled note content suggestion
    pub suggested_content: String,
    /// Suggested importance level
    pub importance: String,
}

/// The category of observation pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ObservationCategory {
    BugResolution,
    Convention,
    Gotcha,
    Pattern,
}

impl ObservationCategory {
    fn note_type(&self) -> &str {
        match self {
            Self::BugResolution => "gotcha",
            Self::Convention => "guideline",
            Self::Gotcha => "gotcha",
            Self::Pattern => "pattern",
        }
    }

    fn importance(&self) -> &str {
        match self {
            Self::BugResolution => "high",
            Self::Convention => "high",
            Self::Gotcha => "critical",
            Self::Pattern => "medium",
        }
    }

    fn base_confidence(&self) -> f64 {
        match self {
            Self::BugResolution => 0.85,
            Self::Convention => 0.80,
            Self::Gotcha => 0.90,
            Self::Pattern => 0.82,
        }
    }
}

/// A compiled pattern rule.
struct PatternRule {
    regex: Regex,
    category: ObservationCategory,
    /// Confidence modifier: multiplied with base confidence
    weight: f64,
}

// ============================================================================
// Pattern Rules
// ============================================================================

static OBSERVATION_PATTERNS: LazyLock<Vec<PatternRule>> = LazyLock::new(|| {
    vec![
        // === Bug Resolution ===
        PatternRule {
            regex: Regex::new(r"(?i)(?:the\s+)?(?:root\s+cause|problem|issue|bug)\s+(?:was|is|turned out)").unwrap(),
            category: ObservationCategory::BugResolution,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:le\s+)?(?:problème|bug|cause)\s+(?:était|est|venait)").unwrap(),
            category: ObservationCategory::BugResolution,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)fixed\s+by\s+(?:changing|adding|removing|updating)").unwrap(),
            category: ObservationCategory::BugResolution,
            weight: 0.95,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:corrigé|résolu)\s+en\s+(?:changeant|ajoutant|supprimant)").unwrap(),
            category: ObservationCategory::BugResolution,
            weight: 0.95,
        },
        // === Conventions ===
        PatternRule {
            regex: Regex::new(r"(?i)(?:always|must|should\s+always)\s+(?:use|call|check|add|include)").unwrap(),
            category: ObservationCategory::Convention,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:never|must\s+not|should\s+never)\s+(?:use|call|skip|forget)").unwrap(),
            category: ObservationCategory::Convention,
            weight: 1.05,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:il\s+faut\s+(?:toujours|systématiquement)|on\s+doit\s+(?:toujours|systématiquement))").unwrap(),
            category: ObservationCategory::Convention,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:ne\s+jamais|il\s+ne\s+faut\s+(?:jamais|pas))").unwrap(),
            category: ObservationCategory::Convention,
            weight: 1.05,
        },
        // === Gotchas ===
        PatternRule {
            regex: Regex::new(r"(?i)(?:watch\s+out|be\s+careful|careful\s+with|beware\s+of|trap|pitfall)").unwrap(),
            category: ObservationCategory::Gotcha,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:attention\s+à|piège|méfie[zr]?\s*-?\s*(?:toi|vous)|prenez?\s+garde)").unwrap(),
            category: ObservationCategory::Gotcha,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)gotcha\s*[:\-—]").unwrap(),
            category: ObservationCategory::Gotcha,
            weight: 1.1,
        },
        // === Patterns ===
        PatternRule {
            regex: Regex::new(r"(?i)(?:this|the|a\s+common)\s+pattern\s+(?:is|works\s+\w+|applies|shows)").unwrap(),
            category: ObservationCategory::Pattern,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:on\s+retrouve|ce\s+pattern|le\s+même\s+pattern|motif\s+récurrent)").unwrap(),
            category: ObservationCategory::Pattern,
            weight: 1.0,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:recurring|repeated)\s+(?:pattern|issue|problem)").unwrap(),
            category: ObservationCategory::Pattern,
            weight: 0.95,
        },
    ]
});

// Minimum confidence threshold for suggestion
const CONFIDENCE_THRESHOLD: f64 = 0.80;

// ============================================================================
// Detection
// ============================================================================

/// Detect observations in an LLM response text.
///
/// Returns at most ONE observation (the highest confidence match),
/// as per the constraint of max 1 suggestion per response.
pub fn detect_observations(response_text: &str) -> Option<DetectedObservation> {
    // Skip very short responses (unlikely to contain insights)
    if response_text.len() < 50 {
        return None;
    }

    let mut best_match: Option<(f64, &PatternRule, regex::Match)> = None;

    for rule in OBSERVATION_PATTERNS.iter() {
        if let Some(m) = rule.regex.find(response_text) {
            let confidence = rule.category.base_confidence() * rule.weight;
            if confidence >= CONFIDENCE_THRESHOLD
                && best_match
                    .as_ref()
                    .is_none_or(|(best_conf, _, _)| confidence > *best_conf)
            {
                best_match = Some((confidence, rule, m));
            }
        }
    }

    let (confidence, rule, matched) = best_match?;

    // Extract surrounding context (up to 300 chars around match)
    let context_excerpt = extract_context(response_text, matched.start(), matched.end(), 300);

    // Build suggested note content
    let suggested_content = build_suggestion(&context_excerpt, rule.category);

    Some(DetectedObservation {
        note_type: rule.category.note_type().to_string(),
        confidence,
        trigger_pattern: matched.as_str().to_string(),
        context_excerpt,
        suggested_content,
        importance: rule.category.importance().to_string(),
    })
}

/// Extract context around a match position, expanding to sentence boundaries.
fn extract_context(text: &str, start: usize, end: usize, max_len: usize) -> String {
    let half = max_len / 2;

    // Expand backwards to sentence boundary
    let ctx_start = if start > half {
        let search_from = start - half;
        text[search_from..start]
            .rfind(|c: char| ['.', '\n'].contains(&c))
            .map(|pos| search_from + pos + 1)
            .unwrap_or(search_from)
    } else {
        0
    };

    // Expand forwards to sentence boundary
    let ctx_end = {
        let search_to = (end + half).min(text.len());
        text[end..search_to]
            .find(|c: char| ['.', '\n'].contains(&c))
            .map(|pos| end + pos + 1)
            .unwrap_or(search_to)
    };

    text[ctx_start..ctx_end].trim().to_string()
}

/// Build a suggested note content from the extracted context.
fn build_suggestion(context: &str, category: ObservationCategory) -> String {
    let prefix = match category {
        ObservationCategory::BugResolution => "**Bug Fix**: ",
        ObservationCategory::Convention => "**Convention**: ",
        ObservationCategory::Gotcha => "**Gotcha**: ",
        ObservationCategory::Pattern => "**Pattern**: ",
    };

    format!("{}{}", prefix, context)
}

// ============================================================================
// VizBlock Builder
// ============================================================================

/// Build a `suggestion_card` VizBlock from a detected observation.
///
/// The VizBlock uses `Custom("suggestion_card")` type and contains:
/// - `note_type`: suggested note type
/// - `content`: pre-filled note content
/// - `importance`: suggested importance level
/// - `confidence`: detection confidence score
/// - `trigger`: the matched pattern
pub fn build_suggestion_vizblock(obs: &DetectedObservation) -> VizBlock {
    let data = serde_json::json!({
        "note_type": obs.note_type,
        "content": obs.suggested_content,
        "importance": obs.importance,
        "confidence": obs.confidence,
        "trigger": obs.trigger_pattern,
    });

    let fallback = format!(
        "💡 Suggestion: create a {} note (confidence: {:.0}%)\n{}",
        obs.note_type,
        obs.confidence * 100.0,
        obs.suggested_content,
    );

    VizBlock {
        viz_type: VizType::Custom("suggestion_card".to_string()),
        data,
        interactive: true,
        fallback_text: fallback,
        title: Some(format!("💡 {} suggestion", obs.note_type)),
        max_height: 200,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_bug_resolution_en() {
        let text = "After investigation, the root cause was a missing ?Sized bound on the generic parameter. Adding + ?Sized to validate_entities fixed the compilation error.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "gotcha");
        assert!(obs.confidence >= CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_detect_bug_resolution_fr() {
        let text = "Après analyse, le problème était que la signature attendait &[(String, String)] et non &[(&str, &str)]. Il fallait cloner les strings.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "gotcha");
    }

    #[test]
    fn test_detect_gotcha() {
        let text = "Be careful with the pre-push hook in this repo — it runs cargo fmt AND clippy. Both must pass or the push will be rejected.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "gotcha");
        assert_eq!(obs.importance, "critical");
    }

    #[test]
    fn test_detect_convention() {
        let text = "You should always use Arc<dyn GraphStore> when passing the graph store to async functions. Never use the concrete type directly.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "guideline");
    }

    #[test]
    fn test_detect_pattern() {
        let text = "This pattern works well for all enrichment stages: implement the EnrichmentStage trait, add a config struct, and register in the pipeline.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "pattern");
    }

    #[test]
    fn test_no_detection_normal_text() {
        let text = "I've updated the function to handle the edge case properly. The tests are passing now.";
        let obs = detect_observations(text);
        assert!(obs.is_none());
    }

    #[test]
    fn test_no_detection_short_text() {
        let text = "Done.";
        let obs = detect_observations(text);
        assert!(obs.is_none());
    }

    #[test]
    fn test_vizblock_builder() {
        let obs = DetectedObservation {
            note_type: "gotcha".to_string(),
            confidence: 0.90,
            trigger_pattern: "root cause was".to_string(),
            context_excerpt: "The root cause was a missing bound".to_string(),
            suggested_content: "**Bug Fix**: The root cause was a missing bound".to_string(),
            importance: "high".to_string(),
        };

        let viz = build_suggestion_vizblock(&obs);
        assert!(matches!(viz.viz_type, VizType::Custom(ref s) if s == "suggestion_card"));
        assert!(viz.interactive);
        assert!(!viz.fallback_text.is_empty());
        assert!(viz.data.get("note_type").is_some());
    }

    #[test]
    fn test_max_one_observation() {
        // Text with multiple patterns — should only return the highest confidence one
        let text = "Watch out for this gotcha: the root cause was that you should always use ?Sized bounds on generics when using dyn traits.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        // Should return exactly one observation (the highest confidence)
    }

    #[test]
    fn test_extract_context_boundaries() {
        let text = "First sentence. The root cause was a bug. Last sentence here.";
        let context = extract_context(text, 16, 40, 300);
        assert!(context.contains("root cause"));
    }
}
