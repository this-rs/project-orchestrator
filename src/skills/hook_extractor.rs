//! Hook Pattern Extraction
//!
//! Extracts search patterns and file context from Claude Code tool inputs.
//! These extracted patterns are evaluated against skill trigger_patterns
//! to determine which skills should be activated during hook processing.
//!
//! Inspired by the `extractPattern` function in gitnexus-hook.cjs.
//!
//! # Supported tools
//!
//! - **Grep**: extracts `pattern` field (regex search term)
//! - **Glob**: extracts `pattern` field (file glob pattern)
//! - **Read**: extracts `file_path` field (file being read)
//! - **Bash**: extracts search terms from `rg`, `grep`, `find` commands
//! - **Edit**: extracts `file_path` field
//! - **Write**: extracts `file_path` field
//! - Other tools: returns `None` (skip)

use regex::Regex;
use std::sync::LazyLock;

// ============================================================================
// Compiled regex patterns for Bash command parsing
// ============================================================================

// Note: rg and grep parsing uses token-based extraction (see extract_rg_grep_pattern)
// because single-pass regex can't handle flags with values (e.g., --type rust).

/// Matches: find ... -name "pattern" or find ... -name pattern
static FIND_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"-name\s+['""]?([^'""\s]+)['""]?"#).unwrap());

/// Matches: cargo test [flags] pattern (last non-flag argument)
static CARGO_TEST_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?:^|\s)cargo\s+test\s+(?:-\S+\s+)*(\S+)"#).unwrap());

// ============================================================================
// Public API
// ============================================================================

/// Extract a search pattern from a Claude Code tool input.
///
/// Returns the pattern string that should be matched against skill triggers.
/// Returns `None` if no meaningful pattern can be extracted (tool not supported
/// or required fields missing).
///
/// # Arguments
///
/// * `tool_name` - Name of the Claude Code tool (e.g., "Grep", "Bash", "Read")
/// * `tool_input` - Raw JSON input of the tool call
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use crate::skills::hook_extractor::extract_pattern;
///
/// // Grep tool → extract pattern field
/// let input = json!({"pattern": "reinforce_synapses", "path": "src/"});
/// assert_eq!(extract_pattern("Grep", &input), Some("reinforce_synapses".to_string()));
///
/// // Bash tool → extract rg search term
/// let input = json!({"command": "rg 'TODO' src/"});
/// assert_eq!(extract_pattern("Bash", &input), Some("TODO".to_string()));
/// ```
pub fn extract_pattern(tool_name: &str, tool_input: &serde_json::Value) -> Option<String> {
    match tool_name {
        "Grep" => tool_input
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Glob" => tool_input
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Read" => tool_input
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Edit" => tool_input
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Write" => tool_input
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Bash" => extract_bash_pattern(tool_input),

        _ => None,
    }
}

/// Extract a file path context from a Claude Code tool input.
///
/// This provides a secondary signal (file path) for FileGlob trigger matching,
/// complementing the primary pattern signal. For tools that directly operate
/// on files, this returns the target file path.
///
/// # Arguments
///
/// * `tool_name` - Name of the Claude Code tool
/// * `tool_input` - Raw JSON input of the tool call
///
/// # Returns
///
/// The file path being operated on, or `None` if not applicable.
pub fn extract_file_context(tool_name: &str, tool_input: &serde_json::Value) -> Option<String> {
    match tool_name {
        "Read" | "Edit" | "Write" => tool_input
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Grep" => tool_input
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Glob" => tool_input
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Bash" => extract_bash_file_context(tool_input),

        _ => None,
    }
}

// ============================================================================
// Bash command parsing (internal)
// ============================================================================

/// Known rg flags that take a value argument.
/// When we see `--type rust`, `rust` is a flag value, not the pattern.
const RG_FLAGS_WITH_VALUES: &[&str] = &[
    "--type",
    "-t",
    "--glob",
    "-g",
    "--max-count",
    "-m",
    "--max-depth",
    "--threads",
    "-j",
    "--color",
    "--colors",
    "--encoding",
    "-E",
    "--replace",
    "-r",
    "--context",
    "-C",
    "--after-context",
    "-A",
    "--before-context",
    "-B",
    "--max-filesize",
    "--type-add",
    "--type-not",
    "-T",
    "--file",
    "-f",
];

/// Known grep flags that take a value argument.
/// Note: `-r` in grep means --recursive (no value), unlike rg where it means --replace.
/// Note: `-e`/`--regexp` are NOT here — they are pattern-specifying flags handled specially.
const GREP_FLAGS_WITH_VALUES: &[&str] = &[
    "--include",
    "--exclude",
    "--exclude-dir",
    "--label",
    "--max-count",
    "-m",
    "--context",
    "-C",
    "--after-context",
    "-A",
    "--before-context",
    "-B",
    "--color",
    "--file",
    "-f",
];

/// Flags that specify the search pattern explicitly.
/// When encountered, the next token IS the pattern (e.g., `grep -e "pattern" file`).
/// Both rg and grep support these.
const PATTERN_FLAGS: &[&str] = &["-e", "--regexp"];

/// Extract a search pattern from a Bash command.
///
/// Recognizes common search tools: `rg`, `grep`, `find`, `cargo test`.
/// Returns `None` for commands without recognizable search patterns.
fn extract_bash_pattern(tool_input: &serde_json::Value) -> Option<String> {
    let command = tool_input.get("command").and_then(|v| v.as_str())?;

    // Try rg/grep first (token-based parsing)
    if command.contains("rg ") || command.contains("rg\t") {
        if let Some(pattern) = extract_rg_grep_pattern(command, "rg", RG_FLAGS_WITH_VALUES) {
            return Some(pattern);
        }
    }

    if command.contains("grep ") || command.contains("grep\t") {
        if let Some(pattern) = extract_rg_grep_pattern(command, "grep", GREP_FLAGS_WITH_VALUES) {
            return Some(pattern);
        }
    }

    // find -name pattern
    if let Some(caps) = FIND_PATTERN.captures(command) {
        return caps.get(1).map(|m| m.as_str().to_string());
    }

    // cargo test pattern
    if let Some(caps) = CARGO_TEST_PATTERN.captures(command) {
        return caps.get(1).map(|m| m.as_str().to_string());
    }

    None
}

/// Token-based pattern extraction for rg/grep commands.
///
/// Handles flags with values correctly (e.g., `--type rust` skips `rust`).
/// Returns the first non-flag, non-flag-value token as the pattern.
fn extract_rg_grep_pattern(
    command: &str,
    tool: &str,
    flags_with_values: &[&str],
) -> Option<String> {
    let tokens = tokenize_command(command);

    // Find the tool position
    let tool_pos = tokens.iter().position(|t| t == tool)?;

    let mut skip_next = false;
    let mut return_next_as_pattern = false;
    for token in &tokens[tool_pos + 1..] {
        if skip_next {
            skip_next = false;
            continue;
        }

        // Previous token was -e/--regexp → this token IS the pattern
        if return_next_as_pattern {
            let pattern = token.trim_matches(|c| c == '\'' || c == '"');
            if !pattern.is_empty() {
                return Some(pattern.to_string());
            }
            return_next_as_pattern = false;
            continue;
        }

        // Long flag with = (e.g., --type=rust) — skip entirely
        // But handle --regexp=pattern specially
        if token.starts_with("--") && token.contains('=') {
            if token.starts_with("--regexp=") {
                let pattern = token
                    .trim_start_matches("--regexp=")
                    .trim_matches(|c| c == '\'' || c == '"');
                if !pattern.is_empty() {
                    return Some(pattern.to_string());
                }
            }
            continue;
        }

        // -e/--regexp: next token is the search pattern
        if PATTERN_FLAGS.contains(&token.as_str()) {
            return_next_as_pattern = true;
            continue;
        }

        // Flag that takes a value — skip this flag AND the next token
        if flags_with_values.contains(&token.as_str()) {
            skip_next = true;
            continue;
        }

        // Any other flag (e.g., -r, -i, --recursive) — skip
        if token.starts_with('-') {
            continue;
        }

        // This is the pattern (first non-flag argument)
        let pattern = token.trim_matches(|c| c == '\'' || c == '"');
        if !pattern.is_empty() {
            return Some(pattern.to_string());
        }
    }

    None
}

/// Simple command tokenizer that respects quoted strings.
fn tokenize_command(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for ch in command.chars() {
        match ch {
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                // Include quote in token so we can strip later
                current.push(ch);
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                current.push(ch);
            }
            ' ' | '\t' if !in_single_quote && !in_double_quote => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Extract file context from a Bash command.
///
/// Tries to find the directory/file being operated on in search commands.
/// Uses `tokenize_command` for proper handling of quoted paths.
fn extract_bash_file_context(tool_input: &serde_json::Value) -> Option<String> {
    let command = tool_input.get("command").and_then(|v| v.as_str())?;

    // For rg/grep commands, try to extract the path argument (last arg that looks like a path)
    // rg pattern path/ or grep -r pattern path/
    if command.contains("rg ") || command.contains("grep ") {
        let tokens = tokenize_command(command);
        for token in tokens.iter().rev() {
            let clean = token.trim_matches(|c| c == '\'' || c == '"');
            // Skip flags
            if clean.starts_with('-') {
                continue;
            }
            // If it looks like a path (contains / or starts with src, lib, etc.)
            if clean.contains('/') || clean.starts_with("src") || clean.starts_with("lib") {
                return Some(clean.to_string());
            }
        }
    }

    // For `cat`, `head`, `tail` commands, extract the file path
    if command.starts_with("cat ") || command.starts_with("head ") || command.starts_with("tail ")
    {
        let tokens = tokenize_command(command);
        // Last non-flag argument
        for token in tokens.iter().rev() {
            let clean = token.trim_matches(|c| c == '\'' || c == '"');
            if !clean.starts_with('-')
                && clean != "cat"
                && clean != "head"
                && clean != "tail"
            {
                return Some(clean.to_string());
            }
        }
    }

    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- extract_pattern: Grep ---

    #[test]
    fn test_extract_pattern_grep() {
        let input = json!({"pattern": "reinforce_synapses", "path": "src/"});
        assert_eq!(
            extract_pattern("Grep", &input),
            Some("reinforce_synapses".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_grep_missing_pattern() {
        let input = json!({"path": "src/"});
        assert_eq!(extract_pattern("Grep", &input), None);
    }

    #[test]
    fn test_extract_pattern_grep_empty() {
        let input = json!({});
        assert_eq!(extract_pattern("Grep", &input), None);
    }

    // --- extract_pattern: Glob ---

    #[test]
    fn test_extract_pattern_glob() {
        let input = json!({"pattern": "**/*.rs", "path": "src/"});
        assert_eq!(extract_pattern("Glob", &input), Some("**/*.rs".to_string()));
    }

    #[test]
    fn test_extract_pattern_glob_missing() {
        let input = json!({"path": "src/"});
        assert_eq!(extract_pattern("Glob", &input), None);
    }

    // --- extract_pattern: Read ---

    #[test]
    fn test_extract_pattern_read() {
        let input = json!({"file_path": "src/api/auth.rs"});
        assert_eq!(
            extract_pattern("Read", &input),
            Some("src/api/auth.rs".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_read_missing() {
        let input = json!({});
        assert_eq!(extract_pattern("Read", &input), None);
    }

    // --- extract_pattern: Edit ---

    #[test]
    fn test_extract_pattern_edit() {
        let input = json!({"file_path": "src/skills/detection.rs", "old_string": "foo", "new_string": "bar"});
        assert_eq!(
            extract_pattern("Edit", &input),
            Some("src/skills/detection.rs".to_string())
        );
    }

    // --- extract_pattern: Write ---

    #[test]
    fn test_extract_pattern_write() {
        let input = json!({"file_path": "src/new_file.rs", "content": "fn main() {}"});
        assert_eq!(
            extract_pattern("Write", &input),
            Some("src/new_file.rs".to_string())
        );
    }

    // --- extract_pattern: Bash ---

    #[test]
    fn test_extract_pattern_bash_rg() {
        let input = json!({"command": "rg TODO src/"});
        assert_eq!(extract_pattern("Bash", &input), Some("TODO".to_string()));
    }

    #[test]
    fn test_extract_pattern_bash_rg_with_flags() {
        let input = json!({"command": "rg -i --type rust pattern_name src/"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("pattern_name".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_rg_quoted() {
        let input = json!({"command": "rg 'reinforce_synapses' src/"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("reinforce_synapses".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_grep() {
        let input = json!({"command": "grep -r 'pattern' ."});
        assert_eq!(extract_pattern("Bash", &input), Some("pattern".to_string()));
    }

    #[test]
    fn test_extract_pattern_bash_grep_with_flags() {
        let input = json!({"command": "grep -rn --include='*.rs' ActivatedNote src/"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("ActivatedNote".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_find() {
        let input = json!({"command": "find . -name '*.test.rs'"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("*.test.rs".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_cargo_test() {
        let input = json!({"command": "cargo test --lib skills::detection"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("skills::detection".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_grep_with_e_flag() {
        // grep -e "pattern" means pattern IS the search regex
        let input = json!({"command": "grep -r -e 'my_pattern' src/"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("my_pattern".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_grep_with_regexp_flag() {
        let input = json!({"command": "grep --regexp 'search_term' file.rs"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("search_term".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_grep_with_regexp_equals() {
        let input = json!({"command": "grep --regexp=search_term file.rs"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("search_term".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_rg_with_e_flag() {
        let input = json!({"command": "rg -e 'pattern' src/"});
        assert_eq!(
            extract_pattern("Bash", &input),
            Some("pattern".to_string())
        );
    }

    #[test]
    fn test_extract_pattern_bash_no_pattern() {
        let input = json!({"command": "ls -la"});
        assert_eq!(extract_pattern("Bash", &input), None);
    }

    #[test]
    fn test_extract_pattern_bash_cd() {
        let input = json!({"command": "cd /tmp && pwd"});
        assert_eq!(extract_pattern("Bash", &input), None);
    }

    #[test]
    fn test_extract_pattern_bash_missing_command() {
        let input = json!({});
        assert_eq!(extract_pattern("Bash", &input), None);
    }

    // --- extract_pattern: Unknown tool ---

    #[test]
    fn test_extract_pattern_unknown_tool() {
        let input = json!({"something": "value"});
        assert_eq!(extract_pattern("WebSearch", &input), None);
    }

    #[test]
    fn test_extract_pattern_task_tool() {
        let input = json!({"prompt": "explore the codebase"});
        assert_eq!(extract_pattern("Task", &input), None);
    }

    // --- extract_file_context ---

    #[test]
    fn test_extract_file_context_read() {
        let input = json!({"file_path": "src/api/auth.rs"});
        assert_eq!(
            extract_file_context("Read", &input),
            Some("src/api/auth.rs".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_edit() {
        let input =
            json!({"file_path": "src/neo4j/client.rs", "old_string": "a", "new_string": "b"});
        assert_eq!(
            extract_file_context("Edit", &input),
            Some("src/neo4j/client.rs".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_write() {
        let input = json!({"file_path": "src/new_module.rs", "content": ""});
        assert_eq!(
            extract_file_context("Write", &input),
            Some("src/new_module.rs".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_grep() {
        let input = json!({"pattern": "foo", "path": "src/api/"});
        assert_eq!(
            extract_file_context("Grep", &input),
            Some("src/api/".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_grep_no_path() {
        let input = json!({"pattern": "foo"});
        assert_eq!(extract_file_context("Grep", &input), None);
    }

    #[test]
    fn test_extract_file_context_glob() {
        let input = json!({"pattern": "**/*.rs", "path": "src/skills"});
        assert_eq!(
            extract_file_context("Glob", &input),
            Some("src/skills".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_bash_rg_with_path() {
        let input = json!({"command": "rg pattern src/neo4j/"});
        assert_eq!(
            extract_file_context("Bash", &input),
            Some("src/neo4j/".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_bash_grep_with_path() {
        let input = json!({"command": "grep -r pattern src/api/"});
        assert_eq!(
            extract_file_context("Bash", &input),
            Some("src/api/".to_string())
        );
    }

    #[test]
    fn test_extract_file_context_bash_no_path() {
        let input = json!({"command": "ls -la"});
        assert_eq!(extract_file_context("Bash", &input), None);
    }

    #[test]
    fn test_extract_file_context_unknown_tool() {
        let input = json!({"something": "value"});
        assert_eq!(extract_file_context("WebSearch", &input), None);
    }

    // --- Edge cases ---

    #[test]
    fn test_extract_pattern_null_value() {
        let input = json!({"pattern": null});
        assert_eq!(extract_pattern("Grep", &input), None);
    }

    #[test]
    fn test_extract_pattern_numeric_value() {
        let input = json!({"pattern": 42});
        assert_eq!(extract_pattern("Grep", &input), None);
    }

    #[test]
    fn test_extract_file_context_null_path() {
        let input = json!({"file_path": null});
        assert_eq!(extract_file_context("Read", &input), None);
    }
}
