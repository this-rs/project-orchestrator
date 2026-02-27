//! Skill naming — Generate descriptive names from note tags and types.
//!
//! Uses statistical analysis of note tags to compose human-readable skill names.
//! No LLM needed — purely deterministic heuristic.

use std::collections::{HashMap, HashSet};

/// Generate a skill name from the tags of its member notes.
///
/// Algorithm:
/// 1. Count tag frequencies across all member notes
/// 2. Select top 2-3 most frequent tags
/// 3. Title-case and join → e.g. "Api Authentication"
/// 4. Fallback to "Cluster-{id}" if no tags available
/// 5. If `existing_names` is provided and the name collides, append "-{fallback_id}"
pub fn generate_skill_name(
    tags_per_note: &[Vec<String>],
    fallback_id: u32,
    existing_names: Option<&HashSet<String>>,
) -> String {
    // Count tag frequencies
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for tags in tags_per_note {
        for tag in tags {
            *freq.entry(tag.as_str()).or_insert(0) += 1;
        }
    }

    if freq.is_empty() {
        return format!("Cluster-{}", fallback_id);
    }

    // Sort by frequency descending, then alphabetically for stability
    let mut sorted: Vec<(&&str, &usize)> = freq.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));

    // Take top 3 tags
    let top_tags: Vec<String> = sorted
        .iter()
        .take(3)
        .map(|(tag, _)| title_case(tag))
        .collect();

    let base_name = if top_tags.is_empty() {
        format!("Cluster-{}", fallback_id)
    } else {
        top_tags.join(" ")
    };

    // Disambiguate if name already exists
    if let Some(names) = existing_names {
        if names.contains(&base_name) {
            format!("{}-{}", base_name, fallback_id)
        } else {
            base_name
        }
    } else {
        base_name
    }
}

/// Convert a tag string to title case: "api_auth" → "Api Auth", "neo4j" → "Neo4j"
fn title_case(s: &str) -> String {
    s.split(['_', '-', ' '])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => {
                    let mut result = c.to_uppercase().to_string();
                    result.extend(chars);
                    result
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_skill_name_basic() {
        let tags = vec![
            vec!["api".to_string(), "auth".to_string()],
            vec!["api".to_string(), "jwt".to_string()],
            vec!["auth".to_string(), "security".to_string()],
        ];
        let name = generate_skill_name(&tags, 0, None);
        // "api" appears 2x, "auth" appears 2x, then "jwt" or "security" 1x each
        assert!(name.contains("Api"), "Expected 'Api' in '{}'", name);
        assert!(name.contains("Auth"), "Expected 'Auth' in '{}'", name);
    }

    #[test]
    fn test_generate_skill_name_empty_tags() {
        let tags: Vec<Vec<String>> = vec![vec![], vec![]];
        let name = generate_skill_name(&tags, 42, None);
        assert_eq!(name, "Cluster-42");
    }

    #[test]
    fn test_generate_skill_name_no_notes() {
        let tags: Vec<Vec<String>> = vec![];
        let name = generate_skill_name(&tags, 7, None);
        assert_eq!(name, "Cluster-7");
    }

    #[test]
    fn test_generate_skill_name_single_dominant_tag() {
        let tags = vec![
            vec!["neo4j".to_string()],
            vec!["neo4j".to_string()],
            vec!["neo4j".to_string()],
            vec!["cypher".to_string()],
        ];
        let name = generate_skill_name(&tags, 0, None);
        assert!(
            name.starts_with("Neo4j"),
            "Expected 'Neo4j' first in '{}'",
            name
        );
    }

    #[test]
    fn test_generate_skill_name_collision() {
        let tags = vec![
            vec!["api".to_string(), "auth".to_string()],
            vec!["api".to_string()],
        ];
        let base = generate_skill_name(&tags, 5, None);
        let mut existing = HashSet::new();
        existing.insert(base.clone());
        let deduped = generate_skill_name(&tags, 5, Some(&existing));
        assert_ne!(base, deduped);
        assert!(deduped.ends_with("-5"), "Expected '-5' suffix in '{}'", deduped);
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("api"), "Api");
        assert_eq!(title_case("api_auth"), "Api Auth");
        assert_eq!(title_case("neo4j"), "Neo4j");
        assert_eq!(title_case("knowledge-fabric"), "Knowledge Fabric");
    }

    #[test]
    fn test_generate_skill_name_max_3_tags() {
        let tags = vec![
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
        ];
        let name = generate_skill_name(&tags, 0, None);
        // Should only use top 3 tags
        let word_count = name.split_whitespace().count();
        assert!(
            word_count <= 3,
            "Expected max 3 words, got {} in '{}'",
            word_count,
            name
        );
    }
}
