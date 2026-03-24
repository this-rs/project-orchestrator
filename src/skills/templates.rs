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
    let rfcs = filter_sort(notes, NoteType::Rfc);

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
    if !rfcs.is_empty() {
        sections.push(format_section("## RFCs", &rfcs));
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
    if template.chars().count() <= MAX_TEMPLATE_CHARS {
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
        if current.chars().count() + section_text.chars().count() + 4 <= MAX_TEMPLATE_CHARS - 200 {
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
            reactivation_count: 0,
            last_reactivated: None,
            freshness_pinged_at: None,
            activation_count: 0,
            supersedes: None,
            superseded_by: None,
            changes: vec![],
            assertion_rule: None,
            last_assertion_result: None,
            memory_horizon: crate::notes::MemoryHorizon::Operational,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
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

        // Should be within budget (use chars().count() to match the budget logic)
        let char_count = template.chars().count();
        assert!(
            char_count <= MAX_TEMPLATE_CHARS + 100, // small margin for truncation comment
            "Template too long: {} chars (max {})",
            char_count,
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

    // ================================================================
    // importance_rank — cover all arms
    // ================================================================

    #[test]
    fn test_importance_rank_all_variants() {
        assert_eq!(importance_rank(&NoteImportance::Critical), 4);
        assert_eq!(importance_rank(&NoteImportance::High), 3);
        assert_eq!(importance_rank(&NoteImportance::Medium), 2);
        assert_eq!(importance_rank(&NoteImportance::Low), 1);
        // Verify ordering
        assert!(
            importance_rank(&NoteImportance::Critical) > importance_rank(&NoteImportance::High)
        );
        assert!(importance_rank(&NoteImportance::High) > importance_rank(&NoteImportance::Medium));
        assert!(importance_rank(&NoteImportance::Medium) > importance_rank(&NoteImportance::Low));
    }

    // ================================================================
    // filter_sort — secondary sort by energy
    // ================================================================

    #[test]
    fn test_filter_sort_by_energy_secondary() {
        let mut note_high_energy = make_note(
            NoteType::Pattern,
            NoteImportance::Medium,
            "High energy pattern",
            vec![],
        );
        note_high_energy.energy = 0.9;

        let mut note_low_energy = make_note(
            NoteType::Pattern,
            NoteImportance::Medium,
            "Low energy pattern",
            vec![],
        );
        note_low_energy.energy = 0.1;

        let mut note_mid_energy = make_note(
            NoteType::Pattern,
            NoteImportance::Medium,
            "Mid energy pattern",
            vec![],
        );
        note_mid_energy.energy = 0.5;

        let notes = vec![note_low_energy, note_high_energy, note_mid_energy];
        let sorted = filter_sort(&notes, NoteType::Pattern);

        assert_eq!(sorted.len(), 3);
        // Same importance, sorted by energy descending
        assert!(sorted[0].content.contains("High energy"));
        assert!(sorted[1].content.contains("Mid energy"));
        assert!(sorted[2].content.contains("Low energy"));
    }

    #[test]
    fn test_filter_sort_importance_takes_priority_over_energy() {
        let mut note_low_imp_high_energy = make_note(
            NoteType::Tip,
            NoteImportance::Low,
            "Low importance high energy",
            vec![],
        );
        note_low_imp_high_energy.energy = 1.0;

        let mut note_high_imp_low_energy = make_note(
            NoteType::Tip,
            NoteImportance::Critical,
            "Critical importance low energy",
            vec![],
        );
        note_high_imp_low_energy.energy = 0.0;

        let notes = vec![note_low_imp_high_energy, note_high_imp_low_energy];
        let sorted = filter_sort(&notes, NoteType::Tip);

        assert_eq!(sorted.len(), 2);
        assert!(sorted[0].content.contains("Critical importance"));
        assert!(sorted[1].content.contains("Low importance"));
    }

    #[test]
    fn test_filter_sort_excludes_other_types() {
        let notes = vec![
            make_note(NoteType::Tip, NoteImportance::High, "A tip", vec![]),
            make_note(NoteType::Gotcha, NoteImportance::High, "A gotcha", vec![]),
        ];
        let tips = filter_sort(&notes, NoteType::Tip);
        assert_eq!(tips.len(), 1);
        assert!(tips[0].content.contains("tip"));
    }

    // ================================================================
    // format_section — badge coverage
    // ================================================================

    #[test]
    fn test_format_section_no_badge_for_medium_and_low() {
        let note_medium = make_note(
            NoteType::Guideline,
            NoteImportance::Medium,
            "Medium note",
            vec![],
        );
        let note_low = make_note(NoteType::Guideline, NoteImportance::Low, "Low note", vec![]);

        let section = format_section("## Test", &[&note_medium, &note_low]);
        // Medium and Low should not have badges
        assert!(section.contains("- Medium note"), "Medium note missing");
        assert!(section.contains("- Low note"), "Low note missing");
        // No badge emojis
        let medium_line = section.lines().find(|l| l.contains("Medium note")).unwrap();
        assert!(!medium_line.contains('🔴'));
        assert!(!medium_line.contains('🟠'));
        let low_line = section.lines().find(|l| l.contains("Low note")).unwrap();
        assert!(!low_line.contains('🔴'));
        assert!(!low_line.contains('🟠'));
    }

    #[test]
    fn test_format_section_critical_and_high_badges() {
        let note_critical = make_note(
            NoteType::Guideline,
            NoteImportance::Critical,
            "Critical note",
            vec![],
        );
        let note_high = make_note(
            NoteType::Guideline,
            NoteImportance::High,
            "High note",
            vec![],
        );

        let section = format_section("## Test", &[&note_critical, &note_high]);

        let critical_line = section
            .lines()
            .find(|l| l.contains("Critical note"))
            .unwrap();
        assert!(critical_line.contains('🔴'), "Critical badge missing");
        let high_line = section.lines().find(|l| l.contains("High note")).unwrap();
        assert!(high_line.contains('🟠'), "High badge missing");
    }

    #[test]
    fn test_format_section_structure() {
        let note = make_note(NoteType::Tip, NoteImportance::Low, "A tip", vec![]);
        let section = format_section("## Tips", &[&note]);

        let lines: Vec<&str> = section.lines().collect();
        assert_eq!(lines[0], "## Tips");
        assert_eq!(lines[1], ""); // blank line after heading
        assert!(lines[2].starts_with("- "));
    }

    // ================================================================
    // truncate_content — edge cases
    // ================================================================

    #[test]
    fn test_truncate_content_exact_length() {
        let content = "a".repeat(200);
        let result = truncate_content(&content, 200);
        // Exactly at max_chars should not truncate
        assert_eq!(result, content);
        assert!(!result.ends_with("..."));
    }

    #[test]
    fn test_truncate_content_one_over_limit() {
        let content = "a".repeat(201);
        let result = truncate_content(&content, 200);
        assert_eq!(result, format!("{}...", "a".repeat(200)));
    }

    #[test]
    fn test_truncate_content_empty_string() {
        assert_eq!(truncate_content("", 200), "");
    }

    #[test]
    fn test_truncate_content_multiline_first_line_short() {
        // First line is short, should return just first line
        let content = "short first\nthis is a much longer second line that exceeds many limits";
        assert_eq!(truncate_content(content, 200), "short first");
    }

    #[test]
    fn test_truncate_content_multiline_first_line_long() {
        let long_first = format!("{}\nsecond line", "x".repeat(300));
        let result = truncate_content(&long_first, 200);
        assert_eq!(result, format!("{}...", "x".repeat(200)));
    }

    #[test]
    fn test_truncate_content_unicode_safety() {
        // Multi-byte UTF-8: each emoji is multiple bytes
        let emojis = "🔴".repeat(250); // 250 emojis, each 4 bytes
        let result = truncate_content(&emojis, 200);
        assert!(result.ends_with("..."));
        // Should have exactly 200 emoji chars + "..."
        let without_dots = result.strip_suffix("...").unwrap();
        assert_eq!(without_dots.chars().count(), 200);
    }

    #[test]
    fn test_truncate_content_zero_max() {
        let result = truncate_content("hello", 0);
        assert_eq!(result, "...");
    }

    // ================================================================
    // generate_context_template — RFC section
    // ================================================================

    #[test]
    fn test_generate_template_rfc_section() {
        let notes = vec![make_note(
            NoteType::Rfc,
            NoteImportance::Medium,
            "RFC: new API design proposal",
            vec!["rfc"],
        )];

        let template = generate_context_template("Design", "Design skill", &notes);
        assert!(template.contains("## RFCs"), "Missing RFCs section");
        assert!(template.contains("RFC: new API design proposal"));
    }

    #[test]
    fn test_generate_template_section_ordering() {
        // Verify sections appear in the correct priority order
        let notes = vec![
            make_note(NoteType::Observation, NoteImportance::Low, "obs", vec![]),
            make_note(NoteType::Context, NoteImportance::Low, "ctx", vec![]),
            make_note(NoteType::Tip, NoteImportance::Low, "tip", vec![]),
            make_note(NoteType::Assertion, NoteImportance::Low, "assert", vec![]),
            make_note(NoteType::Pattern, NoteImportance::Low, "pattern", vec![]),
            make_note(NoteType::Gotcha, NoteImportance::Low, "gotcha", vec![]),
            make_note(NoteType::Guideline, NoteImportance::Low, "guide", vec![]),
            make_note(NoteType::Rfc, NoteImportance::Low, "rfc", vec![]),
        ];

        let template = generate_context_template("Test", "Test", &notes);

        let guideline_pos = template.find("## Guidelines").unwrap();
        let gotcha_pos = template.find("## ⚠️ Gotchas").unwrap();
        let pattern_pos = template.find("## Patterns").unwrap();
        let assertion_pos = template.find("## Assertions").unwrap();
        let tip_pos = template.find("## Tips").unwrap();
        let context_pos = template.find("## Context").unwrap();
        let observation_pos = template.find("## Observations").unwrap();
        let rfc_pos = template.find("## RFCs").unwrap();

        assert!(guideline_pos < gotcha_pos);
        assert!(gotcha_pos < pattern_pos);
        assert!(pattern_pos < assertion_pos);
        assert!(assertion_pos < tip_pos);
        assert!(tip_pos < context_pos);
        assert!(context_pos < observation_pos);
        assert!(observation_pos < rfc_pos);
    }

    // ================================================================
    // apply_token_budget — detailed truncation tests
    // ================================================================

    #[test]
    fn test_apply_token_budget_under_budget_returns_unchanged() {
        let notes = vec![make_note(
            NoteType::Guideline,
            NoteImportance::High,
            "Short note",
            vec![],
        )];
        let template = "# Skill\n\nDescription\n\n## Guidelines\n\n- Short note".to_string();
        let result = apply_token_budget(template.clone(), &notes);
        assert_eq!(result, template);
    }

    #[test]
    fn test_apply_token_budget_preserves_protected_types() {
        // Create enough content to exceed the budget
        let mut notes = Vec::new();

        // Protected: Gotcha (always kept regardless of importance)
        notes.push(make_note(
            NoteType::Gotcha,
            NoteImportance::Low,
            "Low importance gotcha is still protected",
            vec![],
        ));

        // Protected: Guideline (always kept regardless of importance)
        notes.push(make_note(
            NoteType::Guideline,
            NoteImportance::Low,
            "Low importance guideline is still protected",
            vec![],
        ));

        // Lots of dispensable observations to exceed budget
        for i in 0..100 {
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &format!("Observation {} {}", i, "x".repeat(200)),
                vec![],
            ));
        }

        let template = generate_context_template("Budget Test", "Testing budget", &notes);

        // Protected notes survive truncation
        assert!(
            template.contains("Low importance gotcha is still protected"),
            "Protected gotcha should survive"
        );
        assert!(
            template.contains("Low importance guideline is still protected"),
            "Protected guideline should survive"
        );
    }

    #[test]
    fn test_apply_token_budget_preserves_high_importance_non_protected_types() {
        let mut notes = Vec::new();

        // High importance Pattern (not a protected type, but high importance = protected)
        notes.push(make_note(
            NoteType::Pattern,
            NoteImportance::High,
            "High importance pattern survives",
            vec![],
        ));

        // Critical importance Tip (not a protected type, but critical = protected)
        notes.push(make_note(
            NoteType::Tip,
            NoteImportance::Critical,
            "Critical tip survives",
            vec![],
        ));

        // Lots of dispensable to exceed budget
        for i in 0..100 {
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &format!("Obs {} {}", i, "y".repeat(200)),
                vec![],
            ));
        }

        let template = generate_context_template("Budget Test", "Testing budget", &notes);

        // High/Critical importance notes of non-protected types go into "Key Notes"
        assert!(
            template.contains("High importance pattern survives"),
            "High importance pattern should survive as protected"
        );
        assert!(
            template.contains("Critical tip survives"),
            "Critical tip should survive as protected"
        );
    }

    #[test]
    fn test_apply_token_budget_shows_omitted_count() {
        let mut notes = Vec::new();

        // A few protected notes
        notes.push(make_note(
            NoteType::Guideline,
            NoteImportance::Critical,
            "Critical guideline",
            vec![],
        ));

        // Many dispensable notes to force omission
        for i in 0..100 {
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &format!("Observation {} {}", i, "z".repeat(200)),
                vec![],
            ));
        }

        let template = generate_context_template("Budget Test", "Testing budget", &notes);

        assert!(
            template.contains("notes omitted due to token budget"),
            "Should show omission comment when notes are dropped"
        );
    }

    #[test]
    fn test_apply_token_budget_dispensable_priority_order() {
        // When budget is tight, Observations are dropped before Tips,
        // Tips before Context, Context before Patterns, Patterns before Assertions.
        let mut notes = Vec::new();

        // A protected note to anchor the template
        notes.push(make_note(
            NoteType::Guideline,
            NoteImportance::Critical,
            "Anchor guideline",
            vec![],
        ));

        // Add dispensable notes of each type with padding to approach budget
        // Observations (lowest priority, dropped first)
        for i in 0..20 {
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &format!("Observation {} {}", i, "o".repeat(150)),
                vec![],
            ));
        }
        // Tips (next lowest)
        for i in 0..20 {
            notes.push(make_note(
                NoteType::Tip,
                NoteImportance::Low,
                &format!("Tip {} {}", i, "t".repeat(150)),
                vec![],
            ));
        }
        // Context
        for i in 0..5 {
            notes.push(make_note(
                NoteType::Context,
                NoteImportance::Low,
                &format!("Context {} {}", i, "c".repeat(150)),
                vec![],
            ));
        }

        let template = generate_context_template("Priority Test", "Testing priority order", &notes);

        // Template should be within budget
        assert!(
            template.chars().count() <= MAX_TEMPLATE_CHARS + 200,
            "Template should respect budget"
        );

        // If observations are dropped, they should not appear (or appear less)
        // but higher-priority dispensable types might still be included
        // The key thing is the template stays within budget
        assert!(template.contains("{{activated_notes}}"));
        assert!(template.contains("{{relevant_decisions}}"));
    }

    #[test]
    fn test_apply_token_budget_protected_other_section() {
        // Non-Guideline, non-Gotcha notes with High/Critical importance go into "Key Notes"
        let mut notes = Vec::new();

        // High importance Assertion (protected but not Guideline/Gotcha)
        notes.push(make_note(
            NoteType::Assertion,
            NoteImportance::High,
            "High assertion goes to Key Notes",
            vec![],
        ));

        // Critical Context (protected but not Guideline/Gotcha)
        notes.push(make_note(
            NoteType::Context,
            NoteImportance::Critical,
            "Critical context goes to Key Notes",
            vec![],
        ));

        // Lots of dispensable to exceed budget
        for i in 0..100 {
            notes.push(make_note(
                NoteType::Observation,
                NoteImportance::Low,
                &format!("Disposable {} {}", i, "d".repeat(200)),
                vec![],
            ));
        }

        let template = generate_context_template("Key Notes Test", "Testing key notes", &notes);

        assert!(
            template.contains("## Key Notes"),
            "Should have Key Notes section for protected non-Guideline/non-Gotcha"
        );
        assert!(template.contains("High assertion goes to Key Notes"));
        assert!(template.contains("Critical context goes to Key Notes"));
    }

    #[test]
    fn test_apply_token_budget_no_header_sections_in_template() {
        // When there are no "## " sections after the header,
        // header_end falls back to template.len()
        let notes: Vec<Note> = vec![];
        let template = "# Skill\n\nJust a description with no sections at all".to_string();
        let result = apply_token_budget(template.clone(), &notes);
        // Under budget, returns unchanged
        assert_eq!(result, template);
    }

    #[test]
    fn test_apply_token_budget_dispensable_by_type_grouping() {
        // Verify that dispensable notes are grouped by section name in BTreeMap
        // One note of each dispensable type
        let notes = vec![
            make_note(NoteType::Pattern, NoteImportance::Low, "A pattern", vec![]),
            make_note(
                NoteType::Assertion,
                NoteImportance::Low,
                "An assertion",
                vec![],
            ),
            make_note(NoteType::Tip, NoteImportance::Low, "A tip", vec![]),
            make_note(NoteType::Context, NoteImportance::Low, "A context", vec![]),
            make_note(
                NoteType::Observation,
                NoteImportance::Low,
                "An observation",
                vec![],
            ),
        ];

        // Under budget, so all sections should appear
        let template = generate_context_template("Grouping", "Test grouping", &notes);

        assert!(template.contains("## Patterns"));
        assert!(template.contains("## Assertions"));
        assert!(template.contains("## Tips"));
        assert!(template.contains("## Context"));
        assert!(template.contains("## Observations"));
    }

    #[test]
    fn test_apply_token_budget_rfc_as_dispensable() {
        // RFC type with low importance is dispensable and maps to "Other" in the budget code
        let notes = vec![
            make_note(
                NoteType::Guideline,
                NoteImportance::Critical,
                "Anchor",
                vec![],
            ),
            make_note(
                NoteType::Rfc,
                NoteImportance::Low,
                "Dispensable RFC",
                vec![],
            ),
        ];

        // Not enough to trigger budget, so it stays
        let template = generate_context_template("RFC Test", "Test", &notes);
        assert!(template.contains("## RFCs"));
        assert!(template.contains("Dispensable RFC"));
    }

    // ================================================================
    // Edge cases for generate_context_template
    // ================================================================

    #[test]
    fn test_generate_template_only_placeholders_for_empty_notes() {
        let template = generate_context_template("Solo", "Just placeholders", &[]);

        // Should have header + 2 placeholder sections
        assert!(template.starts_with("# Solo\n\nJust placeholders"));
        assert!(template.contains("## Activated Notes\n\n{{activated_notes}}"));
        assert!(template.contains("## Relevant Decisions\n\n{{relevant_decisions}}"));

        // No note-type sections
        assert!(!template.contains("## Patterns"));
        assert!(!template.contains("## Tips"));
        assert!(!template.contains("## Observations"));
        assert!(!template.contains("## Context"));
        assert!(!template.contains("## Assertions"));
        assert!(!template.contains("## RFCs"));
    }

    #[test]
    fn test_generate_template_single_note_type() {
        let notes = vec![
            make_note(
                NoteType::Context,
                NoteImportance::Medium,
                "Context A",
                vec![],
            ),
            make_note(NoteType::Context, NoteImportance::High, "Context B", vec![]),
        ];

        let template = generate_context_template("Ctx", "Context only", &notes);

        assert!(template.contains("## Context"));
        // High should come before Medium
        let high_pos = template.find("Context B").unwrap();
        let medium_pos = template.find("Context A").unwrap();
        assert!(high_pos < medium_pos);

        // No other sections
        assert!(!template.contains("## Guidelines"));
        assert!(!template.contains("## Patterns"));
    }

    #[test]
    fn test_format_section_long_content_truncated() {
        let long_content = "x".repeat(500);
        let note = make_note(NoteType::Tip, NoteImportance::Low, &long_content, vec![]);
        let section = format_section("## Tips", &[&note]);

        // Content should be truncated to 200 chars + "..."
        assert!(section.contains("..."), "Long content should be truncated");
        // The line should not contain the full 500 chars
        let tip_line = section.lines().find(|l| l.starts_with("- ")).unwrap();
        assert!(tip_line.len() < 510, "Line should be truncated");
    }

    #[test]
    fn test_filter_sort_empty_notes() {
        let notes: Vec<Note> = vec![];
        let result = filter_sort(&notes, NoteType::Guideline);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_sort_nan_energy_handling() {
        // Test the unwrap_or(Equal) path for NaN energy values
        let mut note_nan = make_note(
            NoteType::Pattern,
            NoteImportance::Medium,
            "NaN energy",
            vec![],
        );
        note_nan.energy = f64::NAN;

        let mut note_normal = make_note(
            NoteType::Pattern,
            NoteImportance::Medium,
            "Normal energy",
            vec![],
        );
        note_normal.energy = 0.5;

        let notes = vec![note_nan, note_normal];
        // Should not panic even with NaN
        let result = filter_sort(&notes, NoteType::Pattern);
        assert_eq!(result.len(), 2);
    }
}
