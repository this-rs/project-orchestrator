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
#[allow(unused_imports)]
use crate::neo4j::traits::*;

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
    /// RFC proposal — architectural discussion worth formalizing
    Rfc,
}

impl ObservationCategory {
    fn note_type(&self) -> &str {
        match self {
            Self::BugResolution => "gotcha",
            Self::Convention => "guideline",
            Self::Gotcha => "gotcha",
            Self::Pattern => "pattern",
            Self::Rfc => "rfc",
        }
    }

    fn importance(&self) -> &str {
        match self {
            Self::BugResolution => "high",
            Self::Convention => "high",
            Self::Gotcha => "critical",
            Self::Pattern => "medium",
            Self::Rfc => "high",
        }
    }

    fn base_confidence(&self) -> f64 {
        match self {
            Self::BugResolution => 0.85,
            Self::Convention => 0.80,
            Self::Gotcha => 0.90,
            Self::Pattern => 0.82,
            Self::Rfc => 0.75,
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
        // === RFC Proposals (direct — high confidence) ===
        PatternRule {
            regex: Regex::new(r"(?i)\bRFC\b\s*[:\-—]").unwrap(),
            category: ObservationCategory::Rfc,
            weight: 1.15, // 0.75 * 1.15 ≈ 0.86
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:I\s+propose|we\s+should\s+consider|architecture\s+question|architectural\s+proposal)").unwrap(),
            category: ObservationCategory::Rfc,
            weight: 1.10, // 0.75 * 1.10 ≈ 0.83
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:proposition\s+(?:architecturale|technique)|proposition\s*[:\-—])").unwrap(),
            category: ObservationCategory::Rfc,
            weight: 1.10,
        },
        // === RFC Proposals (indirect — lower confidence) ===
        PatternRule {
            regex: Regex::new(r"(?i)(?:on\s+devrait|il\s+faudrait|il\s+manque\s+un)").unwrap(),
            category: ObservationCategory::Rfc,
            weight: 1.07, // 0.75 * 1.07 ≈ 0.80 (just at threshold)
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:we\s+should|we\s+need\s+to\s+(?:rethink|redesign|refactor)|how\s+would\s+you\s+structure)").unwrap(),
            category: ObservationCategory::Rfc,
            weight: 1.07,
        },
        PatternRule {
            regex: Regex::new(r"(?i)(?:comment\s+tu\s+structurerais|il\s+faudrait\s+repenser|on\s+pourrait\s+(?:envisager|concevoir))").unwrap(),
            category: ObservationCategory::Rfc,
            weight: 1.07,
        },
    ]
});

// Minimum confidence threshold for suggestion
const CONFIDENCE_THRESHOLD: f64 = 0.80;

/// Shortest excerpt that can carry a claim.
///
/// Measured on one project before this existed: 70 of 406 detected notes were
/// under 80 characters, 36 ended without a terminal mark, 81 opened on a torn
/// list item. None of those is a statement.
const MIN_EXCERPT_CHARS: usize = 120;

/// Whether an excerpt has the SHAPE of a statement — the write-side guard.
///
/// A claim has the shape of at least two finished sentences. Everything else
/// this returns `false` on was actually observed in the wild: a bullet line
/// (`- Piège déjoué …`), a table row (`| 3 | Staging … |`), a bolded heading
/// (`## Le piège …`), a question (`ne jamais supprimer de lien laisse-t-il … ?`),
/// a quotation, and a sentence cut mid-word by `extract_context`.
///
/// This is a filter, not a fix: a complete, grammatical, worthless sentence
/// passes it. It exists so that the shape is decided here rather than inherited
/// by every reader downstream.
fn has_statement_shape(excerpt: &str) -> bool {
    let text = excerpt.trim();

    if text.chars().count() < MIN_EXCERPT_CHARS {
        return false;
    }

    // Fragments that open on a marker rather than on a sentence.
    let first = text.chars().next().unwrap_or(' ');
    if matches!(first, '-' | '|' | '*' | '#' | '>' | '«' | '`' | '…') || first.is_ascii_digit() {
        return false;
    }

    // A question asks; it does not state.
    if text.ends_with('?') {
        return false;
    }

    // The excerpt opens on the sentence holding the match: if THAT one asks, the
    // observation is a question, whatever follows it.
    if first_sentence_is_a_question(text) {
        return false;
    }

    // A table row is a fragment of a table, not of a reasoning.
    if text.contains('|') || text.contains("```") {
        return false;
    }

    // A claim ends. A cut extract does not.
    let terminal = ['.', '!', ':'];
    if !text.chars().last().is_some_and(|c| terminal.contains(&c)) {
        return false;
    }

    // Two sentences: a terminal mark must exist BEFORE the closing one.
    let body = &text[..text.len() - text.chars().last().map(|c| c.len_utf8()).unwrap_or(1)];
    if !body.contains(['.', '!', '?', ':']) {
        return false;
    }

    true
}

/// Whether the first sentence of `text` is a question.
///
/// A `?` only ends a sentence when it is followed by a space or the end of the
/// text: `?Sized` is a bound, not an interrogation.
fn first_sentence_is_a_question(text: &str) -> bool {
    let bytes = text.as_bytes();
    for (i, byte) in bytes.iter().enumerate() {
        match byte {
            b'?' if bytes.get(i + 1).is_none_or(|c| *c == b' ' || *c == b'\n') => return true,
            b'.' | b'!' | b'\n' => return false,
            _ => {}
        }
    }
    false
}

// ============================================================================
// Detection
// ============================================================================

/// Detect observations in an LLM response text.
///
/// Returns at most ONE observation (the highest confidence match),
/// as per the constraint of max 1 suggestion per response. The excerpt must
/// also have the shape of a statement — see [`has_statement_shape`].
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

    // The shape gate: an excerpt that is not a statement cannot become a claim.
    if !has_statement_shape(&context_excerpt) {
        return None;
    }

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

/// Extract the STATEMENT around a match: the sentence that holds it, plus the
/// following sentences, up to `max_len`.
///
/// This used to take `max_len / 2` characters on each side and stop at the
/// nearest `.` or `\n` — which, by construction, could only ever return ONE
/// sentence, and often a bullet line, because `\n` counted as a boundary. The
/// result was a fragment of a conversation, presented as a piece of knowledge.
///
/// A claim needs at least two sentences: the observation itself and what
/// surrounds it. When the sentence holding the match is shorter than
/// [`MIN_EXCERPT_CHARS`], the following sentences are appended until the excerpt
/// is long enough to say something or the budget is spent.
fn extract_context(text: &str, start: usize, end: usize, max_len: usize) -> String {
    let boundary = |c: char| ['.', '\n', '!', '?', ':'].contains(&c);

    // The sentence holding the match.
    let mut ctx_start = text[..start].rfind(boundary).map(|i| i + 1).unwrap_or(0);

    // Its end.
    let mut ctx_end = text[end..]
        .find(boundary)
        .map(|i| end + i + 1)
        .unwrap_or(text.len());

    // Then one sentence after another, while the excerpt is still too thin.
    while (ctx_end - ctx_start) < MIN_EXCERPT_CHARS && ctx_end < text.len() {
        let next = text[ctx_end..]
            .find(boundary)
            .map(|i| ctx_end + i + 1)
            .unwrap_or(text.len());
        if next <= ctx_end {
            break;
        }
        ctx_end = next;
    }

    // Then the preceding one: "Never use X." is a maxim only together with the
    // sentence that says what it is about.
    let mut guard = ctx_start;
    while (ctx_end - ctx_start) < MIN_EXCERPT_CHARS && ctx_start > 0 {
        let before = &text[..ctx_start.saturating_sub(1)];
        ctx_start = match before.rfind(boundary) {
            Some(i) => i + 1,
            None => 0,
        };
        if ctx_start == guard {
            // No sentence boundary before: take the whole beginning, once.
            ctx_start = 0;
            break;
        }
        guard = ctx_start;
    }

    // Never past the budget, and never on half a character.
    let ctx_end = ctx_end.min(ctx_start + max_len);
    let ctx_end = floor_char_boundary(text, ctx_end);

    text[ctx_start..ctx_end].trim().to_string()
}

/// The largest index `<= i` that is a character boundary.
fn floor_char_boundary(text: &str, i: usize) -> usize {
    let mut idx = i.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Build a suggested note content from the extracted context.
fn build_suggestion(context: &str, category: ObservationCategory) -> String {
    let prefix = match category {
        ObservationCategory::BugResolution => "**Bug Fix**: ",
        ObservationCategory::Convention => "**Convention**: ",
        ObservationCategory::Gotcha => "**Gotcha**: ",
        ObservationCategory::Pattern => "**Pattern**: ",
        ObservationCategory::Rfc => "**RFC Proposal**: ",
    };

    format!("{}{}", prefix, context)
}

// ============================================================================
// RFC Accumulator
// ============================================================================

/// Tracks consecutive RFC-related observations to avoid creating notes
/// from a single offhand mention. Only suggests note creation when
/// 2+ responses in a conversation contain RFC-qualifying patterns.
#[derive(Debug)]
pub struct RfcAccumulator {
    /// Number of consecutive responses with RFC observations
    pub consecutive_count: u32,
    /// Accumulated context from multiple responses
    pub accumulated_context: Vec<String>,
    /// The threshold for triggering RFC creation (default: 2)
    pub threshold: u32,
}

impl Default for RfcAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl RfcAccumulator {
    pub fn new() -> Self {
        Self {
            consecutive_count: 0,
            accumulated_context: Vec::new(),
            threshold: 2,
        }
    }

    /// Feed a response observation result.
    /// Returns `Some(accumulated_content)` if the threshold is reached.
    pub fn feed(&mut self, observation: Option<&DetectedObservation>) -> Option<String> {
        match observation {
            Some(obs) if obs.note_type == "rfc" => {
                self.consecutive_count += 1;
                self.accumulated_context.push(obs.context_excerpt.clone());

                if self.consecutive_count >= self.threshold {
                    // Build accumulated RFC content
                    let content = self.build_rfc_content();
                    Some(content)
                } else {
                    None
                }
            }
            _ => {
                // Non-RFC response breaks the streak
                self.reset();
                None
            }
        }
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.consecutive_count = 0;
        self.accumulated_context.clear();
    }

    /// Build RFC note content from accumulated context
    fn build_rfc_content(&self) -> String {
        let contexts = self.accumulated_context.join("\n\n---\n\n");
        format!(
            "## Problem\n\n{}\n\n## Proposed Solution\n\n_To be filled based on discussion above._\n\n## Context\n\nAuto-detected from {} consecutive architectural discussions.",
            contexts, self.consecutive_count
        )
    }
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

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

    // === RFC detection tests ===

    #[test]
    fn test_detect_rfc_direct_en() {
        let text = "RFC: We need a centralized event bus for inter-service communication. Currently each service has its own ad-hoc notification mechanism which creates tight coupling.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "rfc");
        assert_eq!(obs.importance, "high");
        assert!(obs.confidence >= CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_detect_rfc_proposal_en() {
        let text = "I propose we restructure the protocol module to separate the engine from the hooks. This would make testing much easier and reduce coupling between components.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "rfc");
    }

    #[test]
    fn test_detect_rfc_direct_fr() {
        let text = "Proposition architecturale: on devrait séparer le graph store en deux traits distincts — un pour les lectures et un pour les écritures, comme le pattern CQRS.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "rfc");
    }

    #[test]
    fn test_detect_rfc_indirect_fr() {
        let text = "Il faudrait repenser la façon dont les events sont propagés. Le système actuel de broadcast est trop simple pour gérer les cas de retry et les dead letters.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "rfc");
    }

    #[test]
    fn test_detect_rfc_indirect_en() {
        let text = "We should rethink how the knowledge fabric handles cross-project notes. The current approach doesn't scale well when there are more than 10 projects in a workspace.";
        let obs = detect_observations(text);
        assert!(obs.is_some());
        let obs = obs.unwrap();
        assert_eq!(obs.note_type, "rfc");
    }

    #[test]
    fn test_no_rfc_false_positive() {
        // Normal code discussion, not an architectural proposal
        let text = "I've added the missing import and fixed the test. The function now returns the correct value when called with empty input.";
        let obs = detect_observations(text);
        // Should not detect an RFC
        assert!(obs.is_none() || obs.unwrap().note_type != "rfc");
    }

    // === Shape gate: the shapes that were actually persisted, measured on one
    // project before the gate existed (70 of 406 notes under 80 characters, 81
    // opening on a torn list item, 36 without final punctuation).

    #[test]
    fn rejects_a_torn_list_item() {
        // Sample taken verbatim from the audit of issue #1138.
        let text = "## Le piège qui valait le détour\n\n- Piège déjoué : git rerere a rejoué une résolution périmée qui mettait .\n";
        assert!(detect_observations(text).is_none());
    }

    #[test]
    fn rejects_a_table_row() {
        let text = "| 3 | Staging : rattrapage repair-withdrawn-dates | |\n\nWatch out: the repair job ran twice on staging and duplicated two rows.";
        assert!(detect_observations(text).is_none());
    }

    #[test]
    fn rejects_a_question() {
        let text = "Be careful, ne jamais supprimer de lien laisse-t-il de mauvais lieux ? C'est la question qu'on se pose encore après trois jours de debug.";
        assert!(detect_observations(text).is_none());
    }

    #[test]
    fn rejects_a_short_excerpt() {
        let text = "Attention à ce piège. Rien de plus.";
        assert!(detect_observations(text).is_none());
    }

    #[test]
    fn rejects_an_excerpt_cut_mid_word() {
        // One long sentence, no terminal mark inside the budget: the excerpt
        // would end mid-word, which is what a torn note looked like.
        let mut text = String::from("Watch out for the pre-push hook because ");
        text.push_str(&"it runs a very long list of checks ".repeat(12));
        assert!(detect_observations(&text).is_none());
    }

    #[test]
    fn accepts_a_two_sentence_statement() {
        let text = "The root cause was a missing ?Sized bound on the generic parameter. Adding + ?Sized to validate_entities fixed the compilation error.";
        assert!(
            has_statement_shape(text),
            "a real two-sentence claim passes"
        );
        assert!(detect_observations(text).is_some());
    }

    #[test]
    fn rejects_an_excerpt_that_ends_on_a_question() {
        // The statement opens fine, then turns into an interrogation: a claim
        // does not end on a question mark.
        let text = "The root cause was a missing bound in the parser, and adding it fixed the build. Is that clear enough for you to move on?";
        assert!(!has_statement_shape(text));
    }

    #[test]
    fn rejects_a_pipe_or_a_code_fence_inside_a_statement() {
        let with_pipe = "The root cause was a missing | separator in the parser, and adding it back fixed the whole build again.";
        assert!(!has_statement_shape(with_pipe));

        let with_fence = "The root cause was a missing bound, as this snippet shows. ```rust let x = 1; ``` and the build passes again.";
        assert!(!has_statement_shape(with_fence));
    }

    #[test]
    fn rejects_a_long_single_sentence() {
        // Long enough, ends with a full stop — but it is ONE sentence: a remark,
        // not a reasoning that can stand on its own.
        let text = "Watch out for the pre-push hook because it runs a very long list of checks that must all pass before anything at all can be pushed anywhere.";
        assert!(text.chars().count() >= MIN_EXCERPT_CHARS);
        assert!(!has_statement_shape(text));
    }

    #[test]
    fn an_excerpt_after_a_short_opening_sentence_takes_the_next_one() {
        // The sentence holding the match is too short on its own: the extractor
        // appends the following ones instead of returning a fragment.
        let text = "Be careful with hooks. The pre-push hook runs fmt and clippy, and both must pass before anything can be pushed to the remote repository.";
        let obs = detect_observations(text).expect("a two-sentence statement is captured");
        assert!(obs.context_excerpt.contains("The pre-push hook runs fmt"));
    }

    #[test]
    fn an_excerpt_never_cuts_through_a_multi_byte_character() {
        // 299 ASCII bytes then 'é' (two bytes): the budget of 300 lands inside
        // the character, and the extractor must step back instead of panicking.
        let mut text = String::from("Be careful ");
        text.push_str(&"a".repeat(284));
        text.push('é');
        text.push_str(" et la suite du raisonnement, qui continue encore un peu.");

        let excerpt = extract_context(&text, 0, 11, 300);
        assert!(!excerpt.is_empty());
        assert!(excerpt.is_char_boundary(excerpt.len()));
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

    // === RFC Accumulator tests ===

    #[test]
    fn test_rfc_accumulator_threshold() {
        let mut acc = RfcAccumulator::new();

        let rfc_obs = DetectedObservation {
            note_type: "rfc".to_string(),
            confidence: 0.85,
            trigger_pattern: "I propose".to_string(),
            context_excerpt: "I propose we restructure the module".to_string(),
            suggested_content: "**RFC Proposal**: I propose we restructure the module".to_string(),
            importance: "high".to_string(),
        };

        // First RFC observation — not enough
        assert!(acc.feed(Some(&rfc_obs)).is_none());
        assert_eq!(acc.consecutive_count, 1);

        // Second RFC observation — threshold reached
        let result = acc.feed(Some(&rfc_obs));
        assert!(result.is_some());
        let content = result.unwrap();
        assert!(content.contains("## Problem"));
        assert!(content.contains("## Proposed Solution"));
    }

    #[test]
    fn test_rfc_accumulator_reset_on_non_rfc() {
        let mut acc = RfcAccumulator::new();

        let rfc_obs = DetectedObservation {
            note_type: "rfc".to_string(),
            confidence: 0.85,
            trigger_pattern: "we should".to_string(),
            context_excerpt: "We should consider a new approach".to_string(),
            suggested_content: "content".to_string(),
            importance: "high".to_string(),
        };

        let non_rfc_obs = DetectedObservation {
            note_type: "gotcha".to_string(),
            confidence: 0.90,
            trigger_pattern: "watch out".to_string(),
            context_excerpt: "Watch out for this".to_string(),
            suggested_content: "content".to_string(),
            importance: "critical".to_string(),
        };

        // First RFC
        assert!(acc.feed(Some(&rfc_obs)).is_none());
        assert_eq!(acc.consecutive_count, 1);

        // Non-RFC breaks streak
        assert!(acc.feed(Some(&non_rfc_obs)).is_none());
        assert_eq!(acc.consecutive_count, 0);

        // RFC again — count starts from 1
        assert!(acc.feed(Some(&rfc_obs)).is_none());
        assert_eq!(acc.consecutive_count, 1);
    }

    #[test]
    fn test_rfc_accumulator_none_resets() {
        let mut acc = RfcAccumulator::new();

        let rfc_obs = DetectedObservation {
            note_type: "rfc".to_string(),
            confidence: 0.85,
            trigger_pattern: "RFC:".to_string(),
            context_excerpt: "RFC: new event bus".to_string(),
            suggested_content: "content".to_string(),
            importance: "high".to_string(),
        };

        acc.feed(Some(&rfc_obs));
        assert_eq!(acc.consecutive_count, 1);

        // None observation (no match) resets
        acc.feed(None);
        assert_eq!(acc.consecutive_count, 0);
    }
}
