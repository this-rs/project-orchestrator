//! Trigger Pattern Generation — Auto-generate SkillTrigger patterns from note analysis.
//!
//! Three types of triggers are generated:
//! - **FileGlob**: From file paths linked to member notes (via LINKED_TO anchors)
//! - **Regex**: From frequent tags and content keywords
//! - **Semantic**: From embedding centroid of member notes (if available)

use crate::notes::{EntityType, Note};
use crate::skills::{SkillTrigger, TriggerType};
use std::collections::HashMap;

// ============================================================================
// FileGlob Triggers
// ============================================================================

/// Generate FileGlob triggers from file paths anchored to member notes.
///
/// Algorithm:
/// 1. Extract all file paths from note anchors (entity_type == File)
/// 2. Group paths by directory prefix
/// 3. Find the longest common prefix(es) with sufficient coverage
/// 4. Generate glob patterns (e.g., `src/api/**`)
///
/// Returns empty vec if no file anchors found.
pub fn generate_file_glob_triggers(notes: &[Note]) -> Vec<SkillTrigger> {
    // Extract all file paths from anchors
    let file_paths: Vec<&str> = notes
        .iter()
        .flat_map(|n| n.anchors.iter())
        .filter(|a| a.entity_type == EntityType::File)
        .map(|a| a.entity_id.as_str())
        .collect();

    if file_paths.is_empty() {
        return Vec::new();
    }

    // Group by directory prefixes and find common patterns
    let globs = find_common_glob_patterns(&file_paths);

    globs
        .into_iter()
        .map(|(pattern, coverage)| {
            // Confidence based on what fraction of file paths this glob covers
            let confidence = (0.5 + coverage * 0.4).min(0.95);
            SkillTrigger::file_glob(pattern, confidence)
        })
        .collect()
}

/// Find common glob patterns from a set of file paths.
///
/// Returns (glob_pattern, coverage_ratio) pairs where coverage_ratio
/// is the fraction of input paths matched by the glob.
fn find_common_glob_patterns(paths: &[&str]) -> Vec<(String, f64)> {
    if paths.is_empty() {
        return Vec::new();
    }

    let total = paths.len() as f64;

    // Count files per directory
    let mut dir_counts: HashMap<&str, usize> = HashMap::new();
    for path in paths {
        // Extract directory part
        if let Some(last_slash) = path.rfind('/') {
            let dir = &path[..last_slash];
            *dir_counts.entry(dir).or_insert(0) += 1;
        }
    }

    if dir_counts.is_empty() {
        return Vec::new();
    }

    // Find the most specific common prefix(es) with good coverage
    // Strategy: merge child dirs into parent when parent has higher total coverage
    let mut prefix_counts: HashMap<String, usize> = HashMap::new();
    for (dir, count) in &dir_counts {
        // Accumulate counts up the directory tree
        let parts: Vec<&str> = dir.split('/').collect();
        for depth in 1..=parts.len() {
            let prefix = parts[..depth].join("/");
            *prefix_counts.entry(prefix).or_insert(0) += count;
        }
    }

    // Select prefixes where:
    // - Coverage >= 50% of paths (this prefix covers enough files)
    // - It's the most specific prefix that still has good coverage
    let mut results: Vec<(String, f64)> = Vec::new();

    // Sort by depth (most specific first), then by coverage descending
    let mut candidates: Vec<(String, f64)> = prefix_counts
        .iter()
        .map(|(prefix, count)| (prefix.clone(), *count as f64 / total))
        .filter(|(_, coverage)| *coverage >= 0.4) // At least 40% coverage
        .collect();

    candidates.sort_by(|a, b| {
        let depth_a = a.0.matches('/').count();
        let depth_b = b.0.matches('/').count();
        depth_b
            .cmp(&depth_a) // deeper first
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Take the most specific prefix that covers the most paths
    let mut covered = std::collections::HashSet::new();
    for (prefix, coverage) in candidates {
        // Skip if a more specific prefix already covers these paths
        let dominated = results
            .iter()
            .any(|(existing, _)| existing.starts_with(&prefix));
        if dominated {
            continue;
        }

        // Check this prefix isn't redundant
        if !covered.contains(&prefix) {
            results.push((format!("{}/**", prefix), coverage));
            covered.insert(prefix);
        }

        // Max 3 glob triggers
        if results.len() >= 3 {
            break;
        }
    }

    results
}

// ============================================================================
// Regex Triggers
// ============================================================================

/// Common English stop words to filter out from content analysis.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over", "under", "and", "but",
    "or", "nor", "not", "so", "yet", "both", "either", "neither", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your", "he", "she", "him", "her",
    "his", "i", "me", "my", "if", "then", "else", "when", "where", "how", "what", "which", "who",
    "whom", "all", "each", "every", "any", "some", "no", "more", "most", "other", "such", "only",
    "very", "just", "also", "than", "too", "here", "there", "now",
    // Common technical stop words
    "use", "used", "using", "new", "get", "set", "add", "update", "delete", "create", "file",
    "function", "method", "class", "type", "value", "data", "note", "notes",
];

/// Generate Regex triggers from tag frequencies and content keywords.
///
/// Algorithm:
/// 1. Count tag frequencies across all member notes
/// 2. Select tags appearing in > 30% of notes
/// 3. Extract frequent content keywords (simple TF-IDF)
/// 4. Build alternation pattern: `tag1|tag2|keyword1`
pub fn generate_regex_triggers(notes: &[Note]) -> Vec<SkillTrigger> {
    if notes.is_empty() {
        return Vec::new();
    }

    let total_notes = notes.len() as f64;
    let threshold = 0.3; // Must appear in at least 30% of notes

    // 1. Tag frequency analysis
    let mut tag_freq: HashMap<&str, usize> = HashMap::new();
    for note in notes {
        // Count each tag once per note (not per occurrence)
        let mut seen_tags = std::collections::HashSet::new();
        for tag in &note.tags {
            if seen_tags.insert(tag.as_str()) {
                *tag_freq.entry(tag.as_str()).or_insert(0) += 1;
            }
        }
    }

    // Select tags with frequency > threshold
    let mut frequent_tags: Vec<(&str, f64)> = tag_freq
        .iter()
        .map(|(tag, count)| (*tag, *count as f64 / total_notes))
        .filter(|(_, freq)| *freq >= threshold)
        .collect();

    // Sort by frequency descending
    frequent_tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top 5 tags
    let top_tags: Vec<&str> = frequent_tags.iter().take(5).map(|(tag, _)| *tag).collect();

    // 2. Content keyword extraction (simplified TF-IDF)
    let content_keywords = extract_content_keywords(notes, 5);

    // 3. Merge tags and keywords (tags first, they're more reliable)
    let mut all_terms: Vec<String> = top_tags.iter().map(|t| t.to_string()).collect();
    for kw in &content_keywords {
        if !all_terms.iter().any(|t| t == kw) {
            all_terms.push(kw.clone());
        }
    }

    // Cap at 8 terms total
    all_terms.truncate(8);

    if all_terms.is_empty() {
        return Vec::new();
    }

    // Build regex pattern with case-insensitive alternation
    // Escape special regex characters in terms
    let escaped: Vec<String> = all_terms.iter().map(|t| regex_escape(t)).collect();
    let pattern = escaped.join("|");

    // Confidence based on how many tags passed the threshold
    let tag_coverage = if !frequent_tags.is_empty() {
        frequent_tags[0].1 // coverage of the most common tag
    } else {
        0.3
    };
    let confidence = (0.5 + tag_coverage * 0.3).min(0.9);

    vec![SkillTrigger::regex(pattern, confidence)]
}

/// Extract top-N content keywords using simplified TF-IDF.
///
/// Tokenizes note content, filters stop words and short tokens,
/// counts document frequency, and returns the most distinctive terms.
fn extract_content_keywords(notes: &[Note], top_n: usize) -> Vec<String> {
    let total_docs = notes.len() as f64;
    if total_docs == 0.0 {
        return Vec::new();
    }

    // Document frequency: how many notes contain each token
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for note in notes {
        let mut seen = std::collections::HashSet::new();
        for token in tokenize(&note.content) {
            if seen.insert(token.clone()) {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }
    }

    // Score: prefer terms that appear in many notes (high DF) but not ALL notes
    // Simple scoring: df * (1 - df/total) to avoid terms that appear in every note
    let mut scored: Vec<(String, f64)> = doc_freq
        .into_iter()
        .filter(|(_, count)| *count >= 2) // Must appear in at least 2 notes
        .map(|(term, count)| {
            let df_ratio = count as f64 / total_docs;
            let score = df_ratio * (1.0 - df_ratio * 0.5); // Penalize ubiquitous terms slightly
            (term, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_n)
        .map(|(term, _)| term)
        .collect()
}

/// Tokenize text into lowercase words, filtering stop words and short tokens.
fn tokenize(text: &str) -> Vec<String> {
    let stop_set: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();

    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 3 && !stop_set.contains(w.as_str()))
        .collect()
}

/// Escape special regex characters in a string.
fn regex_escape(s: &str) -> String {
    let special = [
        '\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '|', '^', '$',
    ];
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        if special.contains(&c) {
            result.push('\\');
        }
        result.push(c);
    }
    result
}

// ============================================================================
// Semantic Triggers
// ============================================================================

/// Generate a Semantic trigger from note embeddings.
///
/// If embeddings are available, computes the centroid (mean vector)
/// and creates a trigger with it as a JSON-encoded vector.
///
/// `embeddings` is a map of note_id → embedding vector.
/// Returns None if no embeddings are available.
pub fn generate_semantic_trigger(
    embeddings: &HashMap<String, Vec<f64>>,
    confidence_threshold: f64,
) -> Option<SkillTrigger> {
    if embeddings.is_empty() {
        return None;
    }

    let dim = embeddings.values().next()?.len();
    if dim == 0 {
        return None;
    }

    // Compute centroid: mean of dimension-matched embedding vectors
    let mut centroid = vec![0.0_f64; dim];
    let mut matched_count = 0_usize;

    for embedding in embeddings.values() {
        if embedding.len() != dim {
            continue; // Skip mismatched dimensions
        }
        matched_count += 1;
        for (i, val) in embedding.iter().enumerate() {
            centroid[i] += val;
        }
    }

    if matched_count == 0 {
        return None;
    }

    let divisor = matched_count as f64;
    for val in &mut centroid {
        *val /= divisor;
    }

    // Normalize the centroid to unit length
    let magnitude: f64 = centroid.iter().map(|v| v * v).sum::<f64>().sqrt();
    if magnitude > 0.0 {
        for val in &mut centroid {
            *val /= magnitude;
        }
    }

    // Serialize as compact JSON
    let json = serde_json::to_string(&centroid).ok()?;

    Some(SkillTrigger::semantic(json, confidence_threshold))
}

// ============================================================================
// Quality Evaluation
// ============================================================================

/// Evaluate the quality (F1 score) of a trigger pattern.
///
/// - **Precision** = skill_notes_matching / all_notes_matching
///   (Low precision means the trigger is too broad — matches many non-member notes)
/// - **Recall** = skill_notes_matching / total_skill_notes
///   (Low recall means the trigger is too narrow — misses many member notes)
/// - **F1** = 2 × (P × R) / (P + R)
///
/// For FileGlob triggers, matching is done against note anchors' file paths.
/// For Regex triggers, matching is done against note tags and content.
/// Semantic triggers always get quality_score = None (evaluated at activation time).
pub fn evaluate_trigger_quality(
    trigger: &SkillTrigger,
    skill_notes: &[Note],
    all_project_notes: &[Note],
) -> Option<f64> {
    match trigger.pattern_type {
        TriggerType::Regex => {
            evaluate_regex_quality(&trigger.pattern_value, skill_notes, all_project_notes)
        }
        TriggerType::FileGlob => {
            evaluate_file_glob_quality(&trigger.pattern_value, skill_notes, all_project_notes)
        }
        TriggerType::Semantic => None, // Semantic triggers are evaluated at activation time
    }
}

/// Evaluate regex trigger quality against note tags and content.
fn evaluate_regex_quality(
    pattern: &str,
    skill_notes: &[Note],
    all_project_notes: &[Note],
) -> Option<f64> {
    // Reject overly long patterns and limit regex compilation size
    if pattern.len() > 500 {
        return Some(0.0);
    }
    let regex = regex::RegexBuilder::new(&format!("(?i){}", pattern))
        .size_limit(10_000)
        .dfa_size_limit(10_000)
        .build()
        .ok()?;

    let total_skill = skill_notes.len();
    if total_skill == 0 {
        return Some(0.0);
    }

    // Count how many skill notes match
    let skill_matches = skill_notes
        .iter()
        .filter(|n| note_matches_regex(n, &regex))
        .count();

    // Count how many project notes match (including skill notes)
    let all_matches = all_project_notes
        .iter()
        .filter(|n| note_matches_regex(n, &regex))
        .count();

    compute_f1(skill_matches, all_matches, total_skill)
}

/// Check if a note matches a regex pattern (checks tags and content).
fn note_matches_regex(note: &Note, regex: &regex::Regex) -> bool {
    // Check tags
    for tag in &note.tags {
        if regex.is_match(tag) {
            return true;
        }
    }
    // Check content (first 500 chars for performance)
    // Use char_indices to avoid panicking on multi-byte UTF-8 boundaries
    let content_prefix = if note.content.chars().count() > 500 {
        let end = note
            .content
            .char_indices()
            .nth(500)
            .map(|(i, _)| i)
            .unwrap_or(note.content.len());
        &note.content[..end]
    } else {
        &note.content
    };
    regex.is_match(content_prefix)
}

/// Evaluate file glob trigger quality against note file anchors.
fn evaluate_file_glob_quality(
    pattern: &str,
    skill_notes: &[Note],
    all_project_notes: &[Note],
) -> Option<f64> {
    let glob = glob::Pattern::new(pattern).ok()?;

    let total_skill = skill_notes.len();
    if total_skill == 0 {
        return Some(0.0);
    }

    // Count skill notes with at least one matching file anchor
    let skill_matches = skill_notes
        .iter()
        .filter(|n| note_matches_glob(n, &glob))
        .count();

    // Count all project notes with at least one matching file anchor
    let all_matches = all_project_notes
        .iter()
        .filter(|n| note_matches_glob(n, &glob))
        .count();

    compute_f1(skill_matches, all_matches, total_skill)
}

/// Check if a note has any file anchor matching a glob pattern.
fn note_matches_glob(note: &Note, glob: &glob::Pattern) -> bool {
    note.anchors
        .iter()
        .any(|a| a.entity_type == crate::notes::EntityType::File && glob.matches(&a.entity_id))
}

/// Compute F1 score from match counts.
/// precision = true_positives / all_positives
/// recall = true_positives / total_relevant
fn compute_f1(true_positives: usize, all_positives: usize, total_relevant: usize) -> Option<f64> {
    if all_positives == 0 || total_relevant == 0 {
        return Some(0.0);
    }

    let precision = true_positives as f64 / all_positives as f64;
    let recall = true_positives as f64 / total_relevant as f64;

    if precision + recall == 0.0 {
        Some(0.0)
    } else {
        Some(2.0 * precision * recall / (precision + recall))
    }
}

// ============================================================================
// Orchestrator
// ============================================================================

/// Result of trigger generation for a skill.
#[derive(Debug, Clone)]
pub struct TriggerGenerationResult {
    /// Generated triggers (FileGlob, Regex, Semantic)
    pub triggers: Vec<SkillTrigger>,
    /// Summary statistics
    pub file_glob_count: usize,
    pub regex_count: usize,
    pub semantic_count: usize,
}

/// Generate all trigger patterns for a skill from its member notes.
///
/// Combines FileGlob, Regex, and optionally Semantic triggers.
/// Evaluates quality (F1) for each trigger against the full project notes.
/// `embeddings` can be empty if no embedding provider is available.
/// `all_project_notes` is needed for quality evaluation (precision/recall).
pub fn generate_all_triggers(
    skill_notes: &[Note],
    all_project_notes: &[Note],
    embeddings: &HashMap<String, Vec<f64>>,
) -> TriggerGenerationResult {
    let mut all_triggers = Vec::new();

    // 1. FileGlob from file anchors
    let mut file_globs = generate_file_glob_triggers(skill_notes);
    for trigger in &mut file_globs {
        trigger.quality_score = evaluate_trigger_quality(trigger, skill_notes, all_project_notes);
    }
    let file_glob_count = file_globs.len();
    all_triggers.extend(file_globs);

    // 2. Regex from tags and content
    let mut regex_triggers = generate_regex_triggers(skill_notes);
    for trigger in &mut regex_triggers {
        trigger.quality_score = evaluate_trigger_quality(trigger, skill_notes, all_project_notes);
    }
    let regex_count = regex_triggers.len();
    all_triggers.extend(regex_triggers);

    // 3. Semantic from embeddings (optional) — quality is None (evaluated at activation)
    let semantic_count;
    if let Some(semantic) = generate_semantic_trigger(embeddings, 0.75) {
        semantic_count = 1;
        all_triggers.push(semantic);
    } else {
        semantic_count = 0;
    }

    TriggerGenerationResult {
        triggers: all_triggers,
        file_glob_count,
        regex_count,
        semantic_count,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::{NoteAnchor, NoteImportance, NoteScope, NoteStatus, NoteType};
    use chrono::Utc;

    fn make_note_with_anchors(tags: Vec<&str>, content: &str, anchors: Vec<NoteAnchor>) -> Note {
        Note {
            id: uuid::Uuid::new_v4(),
            project_id: Some(uuid::Uuid::nil()),
            note_type: NoteType::Observation,
            status: NoteStatus::Active,
            importance: NoteImportance::Medium,
            scope: NoteScope::Project,
            content: content.to_string(),
            tags: tags.into_iter().map(|t| t.to_string()).collect(),
            anchors,
            created_at: Utc::now(),
            created_by: "test".to_string(),
            last_confirmed_at: None,
            last_confirmed_by: None,
            staleness_score: 0.0,
            energy: 0.8,
            last_activated: None,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
        }
    }

    fn file_anchor(path: &str) -> NoteAnchor {
        NoteAnchor::new(EntityType::File, path.to_string())
    }

    // ================================================================
    // FileGlob tests
    // ================================================================

    #[test]
    fn test_file_glob_common_directory() {
        let notes = vec![
            make_note_with_anchors(
                vec!["api"],
                "Auth handler",
                vec![file_anchor("src/api/auth.rs")],
            ),
            make_note_with_anchors(
                vec!["api"],
                "Route handler",
                vec![file_anchor("src/api/handlers.rs")],
            ),
            make_note_with_anchors(
                vec!["api"],
                "Route definitions",
                vec![file_anchor("src/api/routes.rs")],
            ),
        ];

        let triggers = generate_file_glob_triggers(&notes);
        assert!(!triggers.is_empty(), "Should generate at least one glob");

        let glob_values: Vec<&str> = triggers.iter().map(|t| t.pattern_value.as_str()).collect();
        assert!(
            glob_values.iter().any(|g| g.contains("src/api")),
            "Expected glob containing 'src/api', got {:?}",
            glob_values
        );
        assert!(triggers
            .iter()
            .all(|t| t.pattern_type == TriggerType::FileGlob));
    }

    #[test]
    fn test_file_glob_no_anchors() {
        let notes = vec![make_note_with_anchors(
            vec!["test"],
            "No file anchors",
            vec![],
        )];

        let triggers = generate_file_glob_triggers(&notes);
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_file_glob_non_file_anchors_ignored() {
        let notes = vec![make_note_with_anchors(
            vec!["test"],
            "Function anchor only",
            vec![NoteAnchor::new(
                EntityType::Function,
                "my_function".to_string(),
            )],
        )];

        let triggers = generate_file_glob_triggers(&notes);
        assert!(triggers.is_empty());
    }

    // ================================================================
    // Regex tests
    // ================================================================

    #[test]
    fn test_regex_from_tags() {
        let notes = vec![
            make_note_with_anchors(
                vec!["neo4j", "cypher", "query"],
                "Neo4j query patterns",
                vec![],
            ),
            make_note_with_anchors(vec!["neo4j", "index"], "Neo4j indexing tips", vec![]),
            make_note_with_anchors(vec!["cypher", "perf"], "Cypher performance", vec![]),
        ];

        let triggers = generate_regex_triggers(&notes);
        assert!(!triggers.is_empty(), "Should generate regex trigger");

        let pattern = &triggers[0].pattern_value;
        assert!(
            pattern.contains("neo4j"),
            "Expected 'neo4j' in pattern '{}'",
            pattern
        );
        assert!(
            pattern.contains("cypher"),
            "Expected 'cypher' in pattern '{}'",
            pattern
        );
        assert_eq!(triggers[0].pattern_type, TriggerType::Regex);
    }

    #[test]
    fn test_regex_empty_notes() {
        let triggers = generate_regex_triggers(&[]);
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_regex_no_common_tags() {
        // Each tag appears in only 1/4 notes = 25% < 30% threshold
        let notes = vec![
            make_note_with_anchors(vec!["alpha"], "A", vec![]),
            make_note_with_anchors(vec!["beta"], "B", vec![]),
            make_note_with_anchors(vec!["gamma"], "C", vec![]),
            make_note_with_anchors(vec!["delta"], "D", vec![]),
        ];

        let triggers = generate_regex_triggers(&notes);
        // May still generate from content keywords, but tags won't contribute
        if !triggers.is_empty() {
            let pattern = &triggers[0].pattern_value;
            // Should not contain any of the rare tags
            assert!(
                !pattern.contains("alpha")
                    || !pattern.contains("beta")
                    || !pattern.contains("gamma")
                    || !pattern.contains("delta"),
                "Rare tags should not all appear in pattern",
            );
        }
    }

    #[test]
    fn test_regex_escapes_special_chars() {
        let escaped = regex_escape("neo4j.driver");
        assert_eq!(escaped, "neo4j\\.driver");

        let escaped2 = regex_escape("C++");
        assert_eq!(escaped2, "C\\+\\+");
    }

    // ================================================================
    // Semantic tests
    // ================================================================

    #[test]
    fn test_semantic_trigger_centroid() {
        let mut embeddings = HashMap::new();
        embeddings.insert("note-1".to_string(), vec![1.0, 0.0, 0.0]);
        embeddings.insert("note-2".to_string(), vec![0.0, 1.0, 0.0]);

        let trigger = generate_semantic_trigger(&embeddings, 0.75);
        assert!(trigger.is_some());

        let t = trigger.unwrap();
        assert_eq!(t.pattern_type, TriggerType::Semantic);
        assert_eq!(t.confidence_threshold, 0.75);

        // Centroid should be [0.5, 0.5, 0.0] normalized
        let centroid: Vec<f64> = serde_json::from_str(&t.pattern_value).unwrap();
        assert_eq!(centroid.len(), 3);
        // After normalization: [0.5, 0.5, 0.0] / sqrt(0.5) ≈ [0.707, 0.707, 0.0]
        assert!(
            (centroid[0] - centroid[1]).abs() < 0.01,
            "x and y should be equal"
        );
        assert!(centroid[2].abs() < 0.01, "z should be ~0");
    }

    #[test]
    fn test_semantic_trigger_empty_embeddings() {
        let embeddings: HashMap<String, Vec<f64>> = HashMap::new();
        assert!(generate_semantic_trigger(&embeddings, 0.75).is_none());
    }

    #[test]
    fn test_semantic_trigger_zero_dim() {
        let mut embeddings = HashMap::new();
        embeddings.insert("note-1".to_string(), vec![]);
        assert!(generate_semantic_trigger(&embeddings, 0.75).is_none());
    }

    // ================================================================
    // Orchestrator tests
    // ================================================================

    #[test]
    fn test_generate_all_triggers_with_file_and_tags() {
        let notes = vec![
            make_note_with_anchors(
                vec!["api", "auth"],
                "Authentication handler for the API",
                vec![file_anchor("src/api/auth.rs")],
            ),
            make_note_with_anchors(
                vec!["api", "jwt"],
                "JWT validation in API layer",
                vec![file_anchor("src/api/jwt.rs")],
            ),
            make_note_with_anchors(
                vec!["auth", "security"],
                "Security considerations for auth",
                vec![file_anchor("src/api/security.rs")],
            ),
        ];

        let embeddings = HashMap::new(); // No embeddings
        let result = generate_all_triggers(&notes, &notes, &embeddings);

        assert!(
            result.file_glob_count >= 1,
            "Expected at least 1 FileGlob trigger"
        );
        assert!(result.regex_count >= 1, "Expected at least 1 Regex trigger");
        assert_eq!(result.semantic_count, 0, "No embeddings → no semantic");
        assert!(
            result.triggers.len() >= 2,
            "Expected at least 2 triggers total, got {}",
            result.triggers.len()
        );
    }

    #[test]
    fn test_generate_all_triggers_with_embeddings() {
        let notes = vec![
            make_note_with_anchors(vec!["test"], "Test note", vec![]),
            make_note_with_anchors(vec!["test"], "Another test", vec![]),
        ];

        let mut embeddings = HashMap::new();
        embeddings.insert("note-1".to_string(), vec![1.0, 0.0]);
        embeddings.insert("note-2".to_string(), vec![0.0, 1.0]);

        let result = generate_all_triggers(&notes, &notes, &embeddings);
        assert_eq!(result.semantic_count, 1);
    }

    // ================================================================
    // Quality evaluation tests
    // ================================================================

    #[test]
    fn test_regex_quality_high_precision_recall() {
        // Skill notes all have "neo4j" tag, no other project notes match
        let skill_notes = vec![
            make_note_with_anchors(vec!["neo4j"], "Neo4j query", vec![]),
            make_note_with_anchors(vec!["neo4j"], "Neo4j index", vec![]),
            make_note_with_anchors(vec!["neo4j"], "Neo4j driver", vec![]),
        ];
        // All project notes = skill notes only
        let all_notes = skill_notes.clone();

        let trigger = SkillTrigger::regex("neo4j", 0.7);
        let quality = evaluate_trigger_quality(&trigger, &skill_notes, &all_notes);

        // P=3/3=1.0, R=3/3=1.0, F1=1.0
        assert!(quality.is_some());
        assert!(
            quality.unwrap() > 0.9,
            "Expected high F1, got {:?}",
            quality
        );
    }

    #[test]
    fn test_regex_quality_too_broad() {
        // Trigger "api" matches 80% of all project notes → low precision
        let skill_notes = vec![
            make_note_with_anchors(vec!["api"], "Skill API note", vec![]),
            make_note_with_anchors(vec!["api"], "Skill API auth", vec![]),
        ];
        let mut all_notes = skill_notes.clone();
        // Add 8 more non-skill notes that also match "api"
        for i in 0..8 {
            all_notes.push(make_note_with_anchors(
                vec!["api"],
                &format!("Other API note {}", i),
                vec![],
            ));
        }

        let trigger = SkillTrigger::regex("api", 0.7);
        let quality = evaluate_trigger_quality(&trigger, &skill_notes, &all_notes);

        // P=2/10=0.2, R=2/2=1.0, F1=2*(0.2*1.0)/(0.2+1.0)=0.333
        assert!(quality.is_some());
        assert!(
            quality.unwrap() < 0.4,
            "Expected low F1 for broad trigger, got {:?}",
            quality
        );
    }

    #[test]
    fn test_regex_quality_too_narrow() {
        // Trigger "obscure_term" matches only 1 of 5 skill notes → low recall
        let skill_notes = vec![
            make_note_with_anchors(vec!["obscure_term"], "Matches", vec![]),
            make_note_with_anchors(vec!["other"], "No match", vec![]),
            make_note_with_anchors(vec!["other"], "No match 2", vec![]),
            make_note_with_anchors(vec!["other"], "No match 3", vec![]),
            make_note_with_anchors(vec!["other"], "No match 4", vec![]),
        ];
        let all_notes = skill_notes.clone();

        let trigger = SkillTrigger::regex("obscure_term", 0.7);
        let quality = evaluate_trigger_quality(&trigger, &skill_notes, &all_notes);

        // P=1/1=1.0, R=1/5=0.2, F1=2*(1.0*0.2)/(1.0+0.2)=0.333
        assert!(quality.is_some());
        assert!(
            quality.unwrap() < 0.4,
            "Expected low F1 for narrow trigger, got {:?}",
            quality
        );
    }

    #[test]
    fn test_file_glob_quality() {
        let skill_notes = vec![
            make_note_with_anchors(vec![], "A", vec![file_anchor("src/api/auth.rs")]),
            make_note_with_anchors(vec![], "B", vec![file_anchor("src/api/routes.rs")]),
        ];
        let mut all_notes = skill_notes.clone();
        // Non-skill note in different directory
        all_notes.push(make_note_with_anchors(
            vec![],
            "C",
            vec![file_anchor("src/graph/algo.rs")],
        ));

        let trigger = SkillTrigger::file_glob("src/api/**", 0.7);
        let quality = evaluate_trigger_quality(&trigger, &skill_notes, &all_notes);

        // P=2/2=1.0, R=2/2=1.0, F1=1.0
        assert!(quality.is_some());
        assert!(
            quality.unwrap() > 0.9,
            "Expected high F1, got {:?}",
            quality
        );
    }

    #[test]
    fn test_semantic_trigger_no_quality() {
        let trigger = SkillTrigger::semantic("[0.1, 0.2]", 0.75);
        let quality = evaluate_trigger_quality(&trigger, &[], &[]);
        assert!(
            quality.is_none(),
            "Semantic triggers should have no quality score"
        );
    }

    #[test]
    fn test_generate_all_triggers_sets_quality() {
        let skill_notes = vec![
            make_note_with_anchors(vec!["neo4j", "cypher"], "Neo4j queries", vec![]),
            make_note_with_anchors(vec!["neo4j"], "Neo4j driver", vec![]),
            make_note_with_anchors(vec!["neo4j", "index"], "Neo4j indexing", vec![]),
        ];
        let all_notes = skill_notes.clone();
        let embeddings = HashMap::new();

        let result = generate_all_triggers(&skill_notes, &all_notes, &embeddings);

        // Regex triggers should have quality scores set
        for trigger in &result.triggers {
            if trigger.pattern_type == TriggerType::Regex {
                assert!(
                    trigger.quality_score.is_some(),
                    "Regex trigger should have quality_score set"
                );
            }
        }
    }

    // ================================================================
    // Tokenizer tests
    // ================================================================

    #[test]
    fn test_tokenize_filters_stop_words() {
        let tokens = tokenize("the quick brown fox is a test");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn test_tokenize_filters_short_tokens() {
        let tokens = tokenize("a be do it go neo4j");
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"be".to_string()));
        assert!(tokens.contains(&"neo4j".to_string()));
    }

    // ================================================================
    // Common prefix tests
    // ================================================================

    #[test]
    fn test_find_common_glob_single_dir() {
        let paths = vec![
            "src/api/auth.rs",
            "src/api/handlers.rs",
            "src/api/routes.rs",
        ];
        let globs = find_common_glob_patterns(&paths);
        assert!(!globs.is_empty());
        assert!(
            globs.iter().any(|(g, _)| g.contains("src/api")),
            "Expected 'src/api' glob, got {:?}",
            globs
        );
    }

    #[test]
    fn test_find_common_glob_empty() {
        let paths: Vec<&str> = vec![];
        let globs = find_common_glob_patterns(&paths);
        assert!(globs.is_empty());
    }
}
