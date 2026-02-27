//! Context Template Generation — Build structured Markdown templates from skill members.
//!
//! Generates a `context_template` for a skill by grouping member notes by type,
//! sorting by importance, and assembling into a Markdown document with placeholders
//! for dynamic content injected at activation time.

use crate::notes::{Note, NoteImportance, NoteType};

/// Approximate token budget (in characters). ~4 chars per token → 2000 tokens ≈ 8000 chars.
const MAX_TEMPLATE_CHARS: usize = 8000;

/// Sections that are always preserved during truncation (even at budget limit).
const PROTECTED_TYPES: &[NoteType] = &[NoteType::Gotcha, NoteType::Guideline];

// ============================================================================
// Template Generation
// ============================================================================

/// Generate a context template from a skill's member notes.
///
/// The template is structured Markdown with:
/// 1. Header: skill name + description
/// 2. Static sections grouped by NoteType (sorted by importance)
/// 3. Dynamic placeholders for activation-time content
///
/// Notes of Critical/High importance are always included in static sections.
/// Lower-importance notes may be truncated to respect the ~2000 token budget.
pub fn generate_context_template(
    skill_name: &str,
    skill_description: &str,
    notes: &[Note],
) -> String {
    let mut sections = Vec::new();

    // Header
    sections.push(format!("# {}\n\n{}", skill_name, skill_description));

    // Group notes by type
    let guidelines = filter_sort(notes, NoteType::Guideline);
    let patterns = filter_sort(notes, NoteType::Pattern);
    let gotchas = filter_sort(notes, NoteType::Gotcha);
    let tips = filter_sort(notes, NoteType::Tip);
    let observations = filter_sort(notes, NoteType::Observation);
    let contexts = filter_sort(notes, NoteType::Context);
    let assertions = filter_sort(notes, NoteType::Assertion);

    // Build sections in priority order
    if !guidelines.is_empty() {
        sections.push(format_section("## Guidelines", &guidelines));
    }
    if !gotchas.is_empty() {
        sections.push(format_section("## ⚠️ Gotchas", &gotchas));
    }
    if !patterns.is_empty() {
        sections.push(format_section("## Patterns", &patterns));
    }
    if !assertions.is_empty() {
        sections.push(format_section("## Assertions", &assertions));
    }
    if !tips.is_empty() {
        sections.push(format_section("## Tips", &tips));
    }
    if !contexts.is_empty() {
        sections.push(format_section("## Context", &contexts));
    }
    if !observations.is_empty() {
        sections.push(format_section("## Observations", &observations));
    }

    // Dynamic placeholders
    sections.push("## Activated Notes\n\n{{activated_notes}}".to_string());
    sections.push("## Relevant Decisions\n\n{{relevant_decisions}}".to_string());

    // Join and apply token budget
    let full_template = sections.join("\n\n");
    apply_token_budget(full_template, notes)
}

/// Filter notes by type and sort by importance descending, then energy descending.
fn filter_sort(notes: &[Note], note_type: NoteType) -> Vec<&Note> {
    let mut filtered: Vec<&Note> = notes.iter().filter(|n| n.note_type == note_type).collect();
    filtered.sort_by(|a, b| {
        importance_rank(&b.importance)
            .cmp(&importance_rank(&a.importance))
            .then(
                b.energy
                    .partial_cmp(&a.energy)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    filtered
}

/// Numeric rank for sorting (higher = more important).
fn importance_rank(imp: &NoteImportance) -> u8 {
    match imp {
        NoteImportance::Critical => 4,
        NoteImportance::High => 3,
        NoteImportance::Medium => 2,
        NoteImportance::Low => 1,
    }
}

/// Format a section with a heading and bullet-pointed notes.
fn format_section(heading: &str, notes: &[&Note]) -> String {
    let mut lines = vec![heading.to_string()];
    lines.push(String::new()); // blank line after heading
    for note in notes {
        let importance_badge = match note.importance {
            NoteImportance::Critical => " 🔴",
            NoteImportance::High => " 🟠",
            _ => "",
        };
        // Format: bullet with importance badge and content (first 200 chars)
        let content = truncate_content(&note.content, 200);
        lines.push(format!("-{} {}", importance_badge, content));
    }
    lines.join("\n")
}

/// Truncate content to max_chars, appending "..." if truncated.
/// Uses char_indices to avoid panicking on multi-byte UTF-8 boundaries.
fn truncate_content(content: &str, max_chars: usize) -> String {
    // Take the first line or max_chars, whichever is shorter
    let first_line = content.lines().next().unwrap_or(content);
    if first_line.chars().count() <= max_chars {
        first_line.to_string()
    } else {
        let end = first_line
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(first_line.len());
        format!("{}...", &first_line[..end])
    }
}

/// Apply token budget by removing low-importance notes from dispensable sections.
///
/// Strategy:
/// 1. If under budget, return as-is
/// 2. Remove Observations first (lowest priority)
/// 3. Then Tips
/// 4. Then Context
/// 5. Never remove Gotchas or Guidelines with Critical/High importance
fn apply_token_budget(template: String, notes: &[Note]) -> String {
    if template.len() <= MAX_TEMPLATE_CHARS {
        return template;
    }

    // Rebuild with truncation
    // Count how many notes were removed
    let mut omitted = 0usize;

    // Categorize notes into protected vs dispensable
    let protected: Vec<&Note> = notes
        .iter()
        .filter(|n| {
            PROTECTED_TYPES.contains(&n.note_type)
                || matches!(
                    n.importance,
                    NoteImportance::Critical | NoteImportance::High
                )
        })
        .collect();

    let dispensable: Vec<&Note> = notes
        .iter()
        .filter(|n| {
            !PROTECTED_TYPES.contains(&n.note_type)
                && !matches!(
                    n.importance,
                    NoteImportance::Critical | NoteImportance::High
                )
        })
        .collect();

    // Sort dispensable by priority: Observations < Tips < Context < Patterns < Assertions
    let dispensable_priority = |n: &Note| -> u8 {
        match n.note_type {
            NoteType::Observation => 0,
            NoteType::Tip => 1,
            NoteType::Context => 2,
            NoteType::Pattern => 3,
            NoteType::Assertion => 4,
            _ => 5,
        }
    };

    let mut sorted_dispensable: Vec<&Note> = dispensable;
    sorted_dispensable.sort_by_key(|n| dispensable_priority(n));

    // Rebuild: start with header and protected notes, then add dispensable until budget
    let mut parts = Vec::new();

    // Determine skill name/desc from the template header
    let header_end = template.find("\n\n## ").unwrap_or(template.len());
    let header = &template[..header_end];
    parts.push(header.to_string());

    // Add protected notes grouped by type
    let protected_guidelines: Vec<&&Note> = protected
        .iter()
        .filter(|n| n.note_type == NoteType::Guideline)
        .collect();
    let protected_gotchas: Vec<&&Note> = protected
        .iter()
        .filter(|n| n.note_type == NoteType::Gotcha)
        .collect();
    let protected_other: Vec<&&Note> = protected
        .iter()
        .filter(|n| n.note_type != NoteType::Guideline && n.note_type != NoteType::Gotcha)
        .collect();

    if !protected_guidelines.is_empty() {
        let notes_ref: Vec<&Note> = protected_guidelines.iter().map(|n| **n).collect();
        parts.push(format_section("## Guidelines", &notes_ref));
    }
    if !protected_gotchas.is_empty() {
        let notes_ref: Vec<&Note> = protected_gotchas.iter().map(|n| **n).collect();
        parts.push(format_section("## ⚠️ Gotchas", &notes_ref));
    }
    if !protected_other.is_empty() {
        let notes_ref: Vec<&Note> = protected_other.iter().map(|n| **n).collect();
        parts.push(format_section("## Key Notes", &notes_ref));
    }

    let mut current = parts.join("\n\n");

    // Add dispensable notes one by one until budget
    let mut dispensable_by_type: std::collections::BTreeMap<&str, Vec<&Note>> =
        std::collections::BTreeMap::new();
    for note in &sorted_dispensable {
        let section = match note.note_type {
            NoteType::Pattern => "## Patterns",
            NoteType::Assertion => "## Assertions",
            NoteType::Tip => "## Tips",
            NoteType::Context => "## Context",
            NoteType::Observation => "## Observations",
            _ => "## Other",
        };
        dispensable_by_type.entry(section).or_default().push(note);
    }

    for (section_name, section_notes) in &dispensable_by_type {
        let section_text = format_section(section_name, section_notes);
        if current.len() + section_text.len() + 4 <= MAX_TEMPLATE_CHARS - 200 {
            // Leave room for placeholders
            current.push_str("\n\n");
            current.push_str(&section_text);
        } else {
            omitted += section_notes.len();
        }
    }

    // Add placeholders (always present)
    current.push_str("\n\n## Activated Notes\n\n{{activated_notes}}");
    current.push_str("\n\n## Relevant Decisions\n\n{{relevant_decisions}}");

    if omitted > 0 {
        current.push_str(&format!(
            "\n\n<!-- {} notes omitted due to token budget -->",
            omitted
        ));
    }

    current
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::{NoteScope, NoteStatus};
    use chrono::Utc;

    fn make_note(
        note_type: NoteType,
        importance: NoteImportance,
        content: &str,
        tags: Vec<&str>,
    ) -> Note {
        Note {
            id: uuid::Uuid::new_v4(),
            project_id: Some(uuid::Uuid::nil()),
            note_type,
            status: NoteStatus::Active,
            importance,
            scope: NoteScope::Project,
            content: content.to_string(),
            tags: tags.into_iter().map(|t| t.to_string()).collect(),
            anchors: vec![],
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

    #[test]
    fn test_generate_template_basic_sections() {
        let notes = vec![
            make_note(
                NoteType::Guideline,
                NoteImportance::High,
                "Always use UNWIND for batch operations",
                vec!["neo4j"],
            ),
            make_note(
                NoteType::Gotcha,
                NoteImportance::Critical,
                "Neo4j driver pool exhaustion under load",
                vec!["neo4j"],
            ),
            make_note(
                NoteType::Pattern,
                NoteImportance::Medium,
                "Repository pattern for data access",
                vec!["pattern"],
            ),
            make_note(
                NoteType::Tip,
                NoteImportance::Low,
                "Use EXPLAIN to debug queries",
                vec!["neo4j"],
            ),
        ];

        let template =
            generate_context_template("Neo4j Skills", "Knowledge about Neo4j usage", &notes);

        assert!(template.contains("# Neo4j Skills"), "Missing header");
        assert!(
            template.contains("## Guidelines"),
            "Missing Guidelines section"
        );
        assert!(
            template.contains("## ⚠️ Gotchas"),
            "Missing Gotchas section"
        );
        assert!(template.contains("## Patterns"), "Missing Patterns section");
        assert!(template.contains("## Tips"), "Missing Tips section");
        assert!(
            template.contains("{{activated_notes}}"),
            "Missing activated_notes placeholder"
        );
        assert!(
            template.contains("{{relevant_decisions}}"),
            "Missing relevant_decisions placeholder"
        );
    }

    #[test]
    fn test_generate_template_sorted_by_importance() {
        let notes = vec![
            make_note(
                NoteType::Guideline,
                NoteImportance::Low,
                "Low importance guideline",
                vec![],
            ),
            make_note(
                NoteType::Guideline,
                NoteImportance::Critical,
                "Critical guideline",
                vec![],
            ),
            make_note(
                NoteType::Guideline,
                NoteImportance::High,
                "High importance guideline",
                vec![],
            ),
        ];

        let template = generate_context_template("Test", "Test skill", &notes);

        // Critical should appear before High, High before Low
        let critical_pos = template.find("Critical guideline").unwrap();
        let high_pos = template.find("High importance guideline").unwrap();
        let low_pos = template.find("Low importance guideline").unwrap();

        assert!(critical_pos < high_pos, "Critical should be before High");
        assert!(high_pos < low_pos, "High should be before Low");
    }

    #[test]
    fn test_generate_template_empty_notes() {
        let template = generate_context_template("Empty Skill", "No notes yet", &[]);

        assert!(template.contains("# Empty Skill"));
        assert!(template.contains("{{activated_notes}}"));
        assert!(template.contains("{{relevant_decisions}}"));
        // Should not contain any section headers for note types
        assert!(!template.contains("## Guidelines"));
        assert!(!template.contains("## Gotchas"));
    }

    #[test]
    fn test_generate_template_importance_badges() {
        let notes = vec![
            make_note(
                NoteType::Guideline,
                NoteImportance::Critical,
                "Critical note",
                vec![],
            ),
            make_note(
                NoteType::Guideline,
                NoteImportance::High,
                "High note",
                vec![],
            ),
            make_note(
                NoteType::Guideline,
                NoteImportance::Medium,
                "Medium note",
                vec![],
            ),
        ];

        let template = generate_context_template("Test", "Test", &notes);

        assert!(template.contains("🔴"), "Critical badge missing");
        assert!(template.contains("🟠"), "High badge missing");
    }

    #[test]
    fn test_generate_template_token_budget() {
        // Create 30 notes to exceed the token budget
        let mut notes = Vec::new();

        // 5 critical guidelines (always preserved)
        for i in 0..5 {
            notes.push(make_note(
                NoteType::Guideline,
                NoteImportance::Critical,
                &format!("Critical guideline number {} with some detailed explanation about important patterns", i),
                vec!["critical"],
            ));
        }

        // 25 low-importance observations (candidates for truncation)
        for i in 0..25 {
            let long_content = format!(
                "Observation {} — {} — detailed analysis of behavior and edge cases",
                i,
                "x".repeat(200)
            );
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &long_content,
                vec!["observation"],
            ));
        }

        let template = generate_context_template("Big Skill", "Skill with many notes", &notes);

        // Should be within budget
        assert!(
            template.len() <= MAX_TEMPLATE_CHARS + 100, // small margin for truncation comment
            "Template too long: {} chars (max {})",
            template.len(),
            MAX_TEMPLATE_CHARS
        );

        // Critical notes should still be present
        assert!(template.contains("Critical guideline number 0"));
        assert!(template.contains("Critical guideline number 4"));

        // Placeholders always present
        assert!(template.contains("{{activated_notes}}"));
        assert!(template.contains("{{relevant_decisions}}"));
    }

    #[test]
    fn test_generate_template_preserves_gotchas() {
        let mut notes = Vec::new();

        // Gotchas should always be preserved
        notes.push(make_note(
            NoteType::Gotcha,
            NoteImportance::Medium,
            "Important gotcha about edge case",
            vec![],
        ));

        // Add many observations to trigger truncation
        for i in 0..30 {
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &format!("Observation {} — {}", i, "padding".repeat(50)),
                vec![],
            ));
        }

        let template = generate_context_template("Test", "Test", &notes);

        // Gotcha should be present even after truncation
        assert!(
            template.contains("Important gotcha about edge case"),
            "Gotcha should be preserved during truncation"
        );
    }

    #[test]
    fn test_truncate_content() {
        assert_eq!(truncate_content("short", 200), "short");
        assert_eq!(
            truncate_content(&"x".repeat(300), 200),
            format!("{}...", "x".repeat(200))
        );
        // Multi-line: only takes first line
        assert_eq!(
            truncate_content("first line\nsecond line", 200),
            "first line"
        );
    }

    #[test]
    fn test_generate_template_varied_types() {
        let notes = vec![
            make_note(
                NoteType::Guideline,
                NoteImportance::High,
                "Use batch operations",
                vec![],
            ),
            make_note(
                NoteType::Gotcha,
                NoteImportance::Critical,
                "Connection pool limit",
                vec![],
            ),
            make_note(
                NoteType::Pattern,
                NoteImportance::Medium,
                "Repository pattern",
                vec![],
            ),
            make_note(NoteType::Tip, NoteImportance::Low, "Use EXPLAIN", vec![]),
            make_note(
                NoteType::Observation,
                NoteImportance::Low,
                "Performance observation",
                vec![],
            ),
            make_note(
                NoteType::Context,
                NoteImportance::Medium,
                "Current refactoring context",
                vec![],
            ),
            make_note(
                NoteType::Assertion,
                NoteImportance::High,
                "All queries must use parameters",
                vec![],
            ),
            make_note(
                NoteType::Guideline,
                NoteImportance::Medium,
                "Prefer MERGE over CREATE",
                vec![],
            ),
        ];

        let template = generate_context_template("Neo4j", "Neo4j knowledge", &notes);

        // All sections with notes should appear
        assert!(template.contains("## Guidelines"));
        assert!(template.contains("## ⚠️ Gotchas"));
        assert!(template.contains("## Patterns"));
        assert!(template.contains("## Tips"));
        assert!(template.contains("## Observations"));
        assert!(template.contains("## Context"));
        assert!(template.contains("## Assertions"));
    }
}
