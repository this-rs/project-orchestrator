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
//! - **MCP mega-tools** (`mcp__*`): extracts `action` + key domain params
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
/// use project_orchestrator::skills::hook_extractor::extract_pattern;
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

        "NotebookEdit" => tool_input
            .get("notebook_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),

        "Bash" => extract_bash_pattern(tool_input),

        // MCP mega-tools: mcp__project-orchestrator__task, etc.
        name if name.starts_with("mcp__") => extract_mcp_pattern(name, tool_input),

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

        "NotebookEdit" => tool_input
            .get("notebook_path")
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

        // MCP mega-tools: extract file_path from code actions
        name if name.starts_with("mcp__") => extract_mcp_file_context(tool_input),

        _ => None,
    }
}

// ============================================================================
// MCP mega-tool pattern extraction (internal)
// ============================================================================

/// Known MCP prefix for Project Orchestrator mega-tools.
const MCP_PO_PREFIX: &str = "mcp__project-orchestrator__";

/// Key fields to extract per mega-tool, in priority order.
/// The first matching field becomes part of the pattern string.
///
/// Fields are chosen to maximize trigger matching relevance:
/// - Search/query fields carry the user's intent
/// - Entity identifiers help match file-based triggers
/// - Titles/descriptions carry domain keywords
const MCP_KEY_FIELDS: &[(&str, &[&str])] = &[
    // Code exploration — richest search signals
    (
        "code",
        &[
            "query",
            "symbol",
            "function",
            "file_path",
            "target",
            "type_name",
            "trait_name",
            "class_name",
            "interface_name",
            "node_path",
            "code_snippet",
            "description",
        ],
    ),
    // Notes — semantic search and content
    (
        "note",
        &[
            "query",
            "content",
            "entity_type",
            "entity_id",
            "file_path",
            "note_type",
        ],
    ),
    // Decisions — architectural context
    (
        "decision",
        &[
            "query",
            "description",
            "rationale",
            "chosen_option",
            "entity_type",
            "entity_id",
        ],
    ),
    // Skills — activation and management
    (
        "skill",
        &["query", "name", "description", "context_template"],
    ),
    // Tasks — work units
    (
        "task",
        &["title", "description", "search", "plan_id", "task_id"],
    ),
    // Plans — objectives
    ("plan", &["title", "description", "search", "plan_id"]),
    // Steps — sub-tasks
    ("step", &["description", "verification", "task_id"]),
    // Projects — codebase tracking
    ("project", &["slug", "name", "search", "root_path"]),
    // Admin — system operations
    ("admin", &["path", "query", "project_id"]),
    // Chat — conversations
    ("chat", &["message", "project_slug"]),
    // Commits — git tracking
    ("commit", &["message", "sha", "file_path", "task_id"]),
    // Feature graphs
    ("feature_graph", &["name", "description", "entry_function"]),
    // Milestones
    ("milestone", &["title", "description"]),
    // Releases
    ("release", &["title", "version", "description"]),
    // Constraints
    ("constraint", &["description", "constraint_type"]),
    // Workspaces
    ("workspace", &["slug", "name"]),
    // Workspace milestones
    ("workspace_milestone", &["title", "description"]),
    // Resources
    ("resource", &["name", "description", "file_path"]),
    // Components
    ("component", &["name", "description"]),
];

/// Extract a search pattern from an MCP mega-tool call.
///
/// Parses the tool name to identify the mega-tool, extracts the `action`
/// parameter, and appends relevant key fields to build a rich pattern string.
///
/// # Pattern format
///
/// `"{mega_tool} {action} {key_field_values...}"`
///
/// Examples:
/// - `"task create Implement auth middleware"`  (from title)
/// - `"code search neo4j batch UNWIND"`  (from query)
/// - `"note search_semantic authentication login"`  (from query)
/// - `"admin sync_directory"`  (action only, no extra fields)
///
/// # Arguments
///
/// * `tool_name` - Full MCP tool name (e.g., `"mcp__project-orchestrator__task"`)
/// * `tool_input` - JSON object with `action` and tool-specific params
fn extract_mcp_pattern(tool_name: &str, tool_input: &serde_json::Value) -> Option<String> {
    // Extract the mega-tool short name from the MCP prefix
    let mega_tool = if let Some(suffix) = tool_name.strip_prefix(MCP_PO_PREFIX) {
        suffix
    } else {
        // Non-PO MCP tool — try generic extraction with just the last segment
        tool_name.rsplit("__").next().unwrap_or(tool_name)
    };

    // Action is mandatory for all mega-tools
    let action = tool_input
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    if action.is_empty() && mega_tool.is_empty() {
        return None;
    }

    // Build pattern: start with "{mega_tool} {action}"
    let mut parts: Vec<&str> = Vec::with_capacity(4);
    parts.push(mega_tool);
    if !action.is_empty() {
        parts.push(action);
    }

    // Find key fields for this mega-tool and append their values
    let key_fields = MCP_KEY_FIELDS
        .iter()
        .find(|(name, _)| *name == mega_tool)
        .map(|(_, fields)| *fields)
        .unwrap_or(&["query", "description", "name", "title"]); // generic fallback

    for field in key_fields {
        if let Some(value) = tool_input.get(*field).and_then(|v| v.as_str()) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                // Truncate long values to keep pattern manageable for trigger matching
                let capped = if trimmed.len() > 200 {
                    &trimmed[..trimmed.floor_char_boundary(200)]
                } else {
                    trimmed
                };
                parts.push(capped);
                // At most 2 key fields to keep patterns focused
                if parts.len() >= 4 {
                    break;
                }
            }
        }
    }

    Some(parts.join(" "))
}

/// Extract a file context from an MCP mega-tool call.
///
/// Returns a file path if the tool input contains file-related fields,
/// enabling FileGlob trigger matching for MCP tool calls.
fn extract_mcp_file_context(tool_input: &serde_json::Value) -> Option<String> {
    // Try common file-related fields in priority order
    for field in &[
        "file_path",
        "path",
        "target",
        "node_path",
        "root_path",
        "cwd",
    ] {
        if let Some(value) = tool_input.get(*field).and_then(|v| v.as_str()) {
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
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
    let mut escaped = false;

    for ch in command.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if !in_single_quote => {
                // Backslash escapes next char in double quotes or unquoted context
                // (In single quotes, backslash is literal per POSIX)
                escaped = true;
            }
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
    if command.starts_with("cat ") || command.starts_with("head ") || command.starts_with("tail ") {
        let tokens = tokenize_command(command);
        // Last non-flag argument
        for token in tokens.iter().rev() {
            let clean = token.trim_matches(|c| c == '\'' || c == '"');
            if !clean.starts_with('-') && clean != "cat" && clean != "head" && clean != "tail" {
                return Some(clean.to_string());
            }
        }
    }

    None
}

// ============================================================================
// Redirect suggestion — MCP-first guidance
// ============================================================================

/// A suggestion to redirect from a raw tool (Grep/Bash) to an MCP code tool.
#[derive(Debug, Clone)]
pub struct RedirectSuggestion {
    /// The MCP tool action to suggest (e.g., "code(action: \"find_references\")")
    pub mcp_tool: String,
    /// Parameters to pass
    pub mcp_params: String,
    /// Human-readable reason for the redirect
    pub reason: String,
}

/// Enriched redirect suggestion with ContextCard intelligence.
#[derive(Debug, Clone)]
pub struct EnrichedRedirectSuggestion {
    /// Base redirect suggestion
    pub suggestion: RedirectSuggestion,
    /// Additional warnings/context from the ContextCard
    pub context_warnings: Vec<String>,
}

impl std::fmt::Display for EnrichedRedirectSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "## 🔄 MCP Redirect Suggestion\n**Instead of** this tool, use:\n`{}`\nwith params: `{}`\n**Why**: {}",
            self.suggestion.mcp_tool, self.suggestion.mcp_params, self.suggestion.reason
        )?;
        for warning in &self.context_warnings {
            write!(f, "\n{}", warning)?;
        }
        Ok(())
    }
}

/// Check if a pattern looks like a code symbol (CamelCase, snake_case, or module::path).
fn is_symbol_like(pattern: &str) -> bool {
    if pattern.is_empty() || pattern.len() < 3 {
        return false;
    }
    // Reject URLs, file paths, regex-heavy patterns, and natural language (has spaces)
    if pattern.contains("://")
        || pattern.starts_with('/')
        || pattern.starts_with('.')
        || pattern.contains('*')
        || pattern.contains('[')
        || pattern.contains('{')
        || pattern.contains('|')
        || pattern.contains(' ')
    {
        return false;
    }
    // CamelCase: starts with uppercase, has mixed case, no spaces
    let has_camel = pattern.chars().next().is_some_and(|c| c.is_uppercase())
        && pattern.chars().any(|c| c.is_lowercase());
    // snake_case: contains underscore with alphanumeric only
    let has_snake =
        pattern.contains('_') && pattern.chars().all(|c| c.is_alphanumeric() || c == '_');
    // Rust path: contains ::
    let has_path = pattern.contains("::");

    has_camel || has_snake || has_path
}

/// Generate a redirect suggestion when a raw tool (Grep/Bash) is used for something
/// that an MCP code tool can handle better.
///
/// Returns `None` if no redirect is applicable (e.g., the pattern is a URL or regex).
pub fn generate_redirect_suggestion(
    tool_name: &str,
    tool_input: &serde_json::Value,
) -> Option<RedirectSuggestion> {
    let pattern = extract_pattern(tool_name, tool_input)?;

    match tool_name {
        "Grep" | "Bash" => {
            if is_symbol_like(&pattern) {
                Some(RedirectSuggestion {
                    mcp_tool: "code(action: \"find_references\")".to_string(),
                    mcp_params: format!("symbol: \"{}\"", pattern),
                    reason: format!(
                        "\"{}\" looks like a code symbol — find_references traverses the graph (imports, calls) and is more precise than text search",
                        pattern
                    ),
                })
            } else if !pattern.contains('/') && !pattern.contains('\\') && pattern.len() > 2 {
                // Free-text content search → semantic search
                Some(RedirectSuggestion {
                    mcp_tool: "code(action: \"search_project\")".to_string(),
                    mcp_params: format!("query: \"{}\"", pattern),
                    reason: format!(
                        "\"{}\" — semantic search ranks results by relevance across all project files",
                        pattern
                    ),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Enrich a redirect suggestion with intelligence from a ContextCard.
///
/// Adds warnings about bridge files, high-risk zones, and co-changers.
pub fn enrich_redirect_with_context_card(
    suggestion: RedirectSuggestion,
    card: &crate::graph::models::ContextCard,
) -> EnrichedRedirectSuggestion {
    let mut warnings = Vec::new();

    // Bridge warning: file with high betweenness
    if card.cc_betweenness > 0.5 {
        warnings.push(format!(
            "🌉 **Bridge file** (betweenness: {:.2}) — this file is a critical bottleneck between clusters. Use `code(action: \"analyze_impact\")` before modifying.",
            card.cc_betweenness
        ));
    }

    // High-risk warning based on centrality metrics
    let risk_score = card.cc_pagerank * 0.4
        + card.cc_betweenness * 0.3
        + (card.cc_imports_in as f64 / (card.cc_imports_in as f64 + 1.0).max(1.0)) * 0.3;
    if risk_score > 0.5 {
        warnings.push(format!(
            "⚠️ **High-risk file** (pagerank: {:.2}, betweenness: {:.2}, imports_in: {}) — changes here may cascade widely.",
            card.cc_pagerank, card.cc_betweenness, card.cc_imports_in
        ));
    }

    // Co-changers alert
    if !card.cc_co_changers_top5.is_empty() {
        let changers: Vec<&str> = card
            .cc_co_changers_top5
            .iter()
            .take(3)
            .map(|s| s.as_str())
            .collect();
        warnings.push(format!(
            "🔗 **Co-changers**: {} — these files often change together, check them too.",
            changers.join(", ")
        ));
    }

    // Community context
    if !card.cc_community_label.is_empty() {
        warnings.push(format!(
            "🏛️ **Community**: \"{}\" (id: {})",
            card.cc_community_label, card.cc_community_id
        ));
    }

    EnrichedRedirectSuggestion {
        suggestion,
        context_warnings: warnings,
    }
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
        assert_eq!(extract_pattern("Bash", &input), Some("pattern".to_string()));
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

    // ========================================================================
    // MCP mega-tool extraction tests
    // ========================================================================

    // --- extract_pattern: MCP task ---

    #[test]
    fn test_mcp_task_create() {
        let input =
            json!({"action": "create", "plan_id": "abc-123", "title": "Implement auth middleware"});
        // title (1st key) + plan_id (2nd key) both extracted
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__task", &input),
            Some("task create Implement auth middleware abc-123".to_string())
        );
    }

    #[test]
    fn test_mcp_task_get_next() {
        let input = json!({"action": "get_next", "plan_id": "abc-123"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__task", &input),
            Some("task get_next abc-123".to_string())
        );
    }

    #[test]
    fn test_mcp_task_missing_action() {
        let input = json!({"plan_id": "abc-123"});
        // Still returns pattern with empty action — mega_tool name is enough
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__task", &input),
            Some("task abc-123".to_string())
        );
    }

    #[test]
    fn test_mcp_task_update_status() {
        let input = json!({"action": "update", "task_id": "t-123", "status": "completed"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__task", &input),
            Some("task update t-123".to_string())
        );
    }

    // --- extract_pattern: MCP plan ---

    #[test]
    fn test_mcp_plan_create() {
        let input =
            json!({"action": "create", "title": "Hook Augmentation for MCP", "priority": 85});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__plan", &input),
            Some("plan create Hook Augmentation for MCP".to_string())
        );
    }

    #[test]
    fn test_mcp_plan_get_dependency_graph() {
        let input = json!({"action": "get_dependency_graph", "plan_id": "plan-uuid"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__plan", &input),
            Some("plan get_dependency_graph plan-uuid".to_string())
        );
    }

    // --- extract_pattern: MCP note ---

    #[test]
    fn test_mcp_note_search_semantic() {
        let input = json!({"action": "search_semantic", "query": "neo4j batch performance"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__note", &input),
            Some("note search_semantic neo4j batch performance".to_string())
        );
    }

    #[test]
    fn test_mcp_note_create() {
        let input = json!({"action": "create", "content": "Always use parameterized queries", "note_type": "guideline"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__note", &input),
            Some("note create Always use parameterized queries guideline".to_string())
        );
    }

    #[test]
    fn test_mcp_note_link_to_entity() {
        let input = json!({"action": "link_to_entity", "entity_type": "file", "entity_id": "src/neo4j/skill.rs"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__note", &input),
            Some("note link_to_entity file src/neo4j/skill.rs".to_string())
        );
    }

    // --- extract_pattern: MCP code ---

    #[test]
    fn test_mcp_code_search() {
        let input = json!({"action": "search", "query": "neo4j batch UNWIND"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__code", &input),
            Some("code search neo4j batch UNWIND".to_string())
        );
    }

    #[test]
    fn test_mcp_code_find_references() {
        let input = json!({"action": "find_references", "symbol": "activate_for_hook"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__code", &input),
            Some("code find_references activate_for_hook".to_string())
        );
    }

    #[test]
    fn test_mcp_code_analyze_impact() {
        let input = json!({"action": "analyze_impact", "target": "/Users/foo/src/neo4j/client.rs"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__code", &input),
            Some("code analyze_impact /Users/foo/src/neo4j/client.rs".to_string())
        );
    }

    #[test]
    fn test_mcp_code_get_call_graph() {
        let input = json!({"action": "get_call_graph", "function": "stream_response"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__code", &input),
            Some("code get_call_graph stream_response".to_string())
        );
    }

    #[test]
    fn test_mcp_code_get_class_hierarchy() {
        let input = json!({"action": "get_class_hierarchy", "type_name": "GraphStore"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__code", &input),
            Some("code get_class_hierarchy GraphStore".to_string())
        );
    }

    // --- extract_pattern: MCP skill ---

    #[test]
    fn test_mcp_skill_activate() {
        let input = json!({"action": "activate", "query": "Tauri auth cookies"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__skill", &input),
            Some("skill activate Tauri auth cookies".to_string())
        );
    }

    #[test]
    fn test_mcp_skill_list() {
        let input = json!({"action": "list", "project_id": "uuid-here"});
        // project_id is not in skill key fields — just action
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__skill", &input),
            Some("skill list".to_string())
        );
    }

    // --- extract_pattern: MCP admin ---

    #[test]
    fn test_mcp_admin_detect_skills() {
        let input = json!({"action": "detect_skills", "project_id": "uuid-here"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__admin", &input),
            Some("admin detect_skills uuid-here".to_string())
        );
    }

    #[test]
    fn test_mcp_admin_sync_directory() {
        let input = json!({"action": "sync_directory", "path": "/Users/foo/project"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__admin", &input),
            Some("admin sync_directory /Users/foo/project".to_string())
        );
    }

    // --- extract_pattern: MCP step ---

    #[test]
    fn test_mcp_step_create() {
        let input = json!({"action": "create", "task_id": "t-uuid", "description": "Add batch UNWIND to neo4j queries"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__step", &input),
            Some("step create Add batch UNWIND to neo4j queries t-uuid".to_string())
        );
    }

    // --- extract_pattern: MCP decision ---

    #[test]
    fn test_mcp_decision_search_semantic() {
        let input = json!({"action": "search_semantic", "query": "authentication strategy"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__decision", &input),
            Some("decision search_semantic authentication strategy".to_string())
        );
    }

    #[test]
    fn test_mcp_decision_add() {
        let input = json!({"action": "add", "description": "Use JWT over session cookies", "rationale": "Stateless"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__decision", &input),
            Some("decision add Use JWT over session cookies Stateless".to_string())
        );
    }

    // --- extract_pattern: MCP project ---

    #[test]
    fn test_mcp_project_sync() {
        let input = json!({"action": "sync", "slug": "project-orchestrator-backend"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__project", &input),
            Some("project sync project-orchestrator-backend".to_string())
        );
    }

    // --- extract_pattern: MCP commit ---

    #[test]
    fn test_mcp_commit_create() {
        let input =
            json!({"action": "create", "sha": "abc123", "message": "feat: add MCP hook support"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__commit", &input),
            Some("commit create feat: add MCP hook support abc123".to_string())
        );
    }

    // --- extract_pattern: MCP chat ---

    #[test]
    fn test_mcp_chat_send_message() {
        let input = json!({"action": "send_message", "message": "Explain the auth flow"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__chat", &input),
            Some("chat send_message Explain the auth flow".to_string())
        );
    }

    // --- extract_pattern: MCP feature_graph ---

    #[test]
    fn test_mcp_feature_graph_auto_build() {
        let input =
            json!({"action": "auto_build", "name": "Auth Flow", "entry_function": "login_handler"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__feature_graph", &input),
            Some("feature_graph auto_build Auth Flow login_handler".to_string())
        );
    }

    // --- extract_pattern: MCP milestone ---

    #[test]
    fn test_mcp_milestone_create() {
        let input = json!({"action": "create", "title": "Neural Skills v2", "description": "MCP augmentation"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__milestone", &input),
            Some("milestone create Neural Skills v2 MCP augmentation".to_string())
        );
    }

    // --- extract_pattern: MCP release ---

    #[test]
    fn test_mcp_release_create() {
        let input = json!({"action": "create", "version": "0.2.0", "title": "Hook Augmentation"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__release", &input),
            Some("release create Hook Augmentation 0.2.0".to_string())
        );
    }

    // --- extract_pattern: MCP constraint ---

    #[test]
    fn test_mcp_constraint_add() {
        let input = json!({"action": "add", "description": "P99 < 50ms for activation", "constraint_type": "performance"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__constraint", &input),
            Some("constraint add P99 < 50ms for activation performance".to_string())
        );
    }

    // --- extract_pattern: MCP workspace ---

    #[test]
    fn test_mcp_workspace_get_overview() {
        let input = json!({"action": "get_overview", "slug": "my-workspace"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__workspace", &input),
            Some("workspace get_overview my-workspace".to_string())
        );
    }

    // --- extract_pattern: MCP workspace_milestone ---

    #[test]
    fn test_mcp_workspace_milestone_create() {
        let input = json!({"action": "create", "title": "Cross-project delivery"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__workspace_milestone", &input),
            Some("workspace_milestone create Cross-project delivery".to_string())
        );
    }

    // --- extract_pattern: MCP resource ---

    #[test]
    fn test_mcp_resource_create() {
        let input =
            json!({"action": "create", "name": "OpenAPI schema", "file_path": "/api/openapi.yaml"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__resource", &input),
            Some("resource create OpenAPI schema /api/openapi.yaml".to_string())
        );
    }

    // --- extract_pattern: MCP component ---

    #[test]
    fn test_mcp_component_create() {
        let input =
            json!({"action": "create", "name": "Auth Service", "description": "JWT + OIDC"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__component", &input),
            Some("component create Auth Service JWT + OIDC".to_string())
        );
    }

    // --- extract_pattern: MCP edge cases ---

    #[test]
    fn test_mcp_empty_action() {
        let input = json!({"action": "", "plan_id": "abc"});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__plan", &input),
            Some("plan abc".to_string())
        );
    }

    #[test]
    fn test_mcp_no_action_no_fields() {
        let input = json!({});
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__admin", &input),
            Some("admin".to_string())
        );
    }

    #[test]
    fn test_mcp_unknown_mega_tool_generic_fallback() {
        let input = json!({"action": "list", "query": "something"});
        // Unknown mega-tool uses generic fallback fields
        assert_eq!(
            extract_pattern("mcp__project-orchestrator__unknown_tool", &input),
            Some("unknown_tool list something".to_string())
        );
    }

    #[test]
    fn test_mcp_non_po_prefix() {
        // Non-PO MCP tool — extracts last segment as mega-tool name
        let input = json!({"action": "search", "query": "test"});
        assert_eq!(
            extract_pattern("mcp__other-server__some_tool", &input),
            Some("some_tool search test".to_string())
        );
    }

    #[test]
    fn test_mcp_long_value_truncated() {
        let long_value = "a".repeat(300);
        let input = json!({"action": "search", "query": long_value});
        let result = extract_pattern("mcp__project-orchestrator__code", &input).unwrap();
        // Pattern should be capped: "code search " + 200 chars max
        assert!(result.len() <= "code search ".len() + 200);
        assert!(result.starts_with("code search "));
    }

    // --- extract_file_context: MCP tools ---

    #[test]
    fn test_mcp_file_context_code_file_path() {
        let input = json!({"action": "get_file_symbols", "file_path": "/Users/foo/src/main.rs"});
        assert_eq!(
            extract_file_context("mcp__project-orchestrator__code", &input),
            Some("/Users/foo/src/main.rs".to_string())
        );
    }

    #[test]
    fn test_mcp_file_context_admin_path() {
        let input = json!({"action": "sync_directory", "path": "/Users/foo/project"});
        assert_eq!(
            extract_file_context("mcp__project-orchestrator__admin", &input),
            Some("/Users/foo/project".to_string())
        );
    }

    #[test]
    fn test_mcp_file_context_code_target() {
        let input = json!({"action": "analyze_impact", "target": "/Users/foo/src/neo4j/client.rs"});
        assert_eq!(
            extract_file_context("mcp__project-orchestrator__code", &input),
            Some("/Users/foo/src/neo4j/client.rs".to_string())
        );
    }

    #[test]
    fn test_mcp_file_context_no_file_field() {
        let input = json!({"action": "list", "project_id": "uuid"});
        assert_eq!(
            extract_file_context("mcp__project-orchestrator__skill", &input),
            None
        );
    }

    #[test]
    fn test_mcp_file_context_resource_file_path() {
        let input = json!({"action": "create", "name": "Schema", "file_path": "/api/schema.json"});
        assert_eq!(
            extract_file_context("mcp__project-orchestrator__resource", &input),
            Some("/api/schema.json".to_string())
        );
    }

    // --- is_symbol_like ---

    #[test]
    fn test_is_symbol_like_camel_case() {
        assert!(is_symbol_like("MyStruct"));
        assert!(is_symbol_like("RunnerState"));
        assert!(is_symbol_like("GraphStore"));
    }

    #[test]
    fn test_is_symbol_like_snake_case() {
        assert!(is_symbol_like("parse_uuid"));
        assert!(is_symbol_like("get_context_card"));
        assert!(is_symbol_like("extract_pattern"));
    }

    #[test]
    fn test_is_symbol_like_rust_path() {
        assert!(is_symbol_like("neo4j::client"));
        assert!(is_symbol_like("crate::graph::models"));
    }

    #[test]
    fn test_is_symbol_like_rejects_non_symbols() {
        assert!(!is_symbol_like(""));
        assert!(!is_symbol_like("ab")); // too short
        assert!(!is_symbol_like("http://localhost"));
        assert!(!is_symbol_like("/usr/bin/cargo"));
        assert!(!is_symbol_like("*.rs"));
        assert!(!is_symbol_like("[a-z]+"));
        assert!(!is_symbol_like("foo|bar"));
        assert!(!is_symbol_like("./relative"));
    }

    // --- generate_redirect_suggestion ---

    #[test]
    fn test_redirect_grep_symbol() {
        let input = json!({"pattern": "RunnerState", "path": "src/"});
        let suggestion = generate_redirect_suggestion("Grep", &input).unwrap();
        assert!(suggestion.mcp_tool.contains("find_references"));
        assert!(suggestion.mcp_params.contains("RunnerState"));
    }

    #[test]
    fn test_redirect_grep_free_text() {
        let input = json!({"pattern": "TODO fix later", "path": "src/"});
        let suggestion = generate_redirect_suggestion("Grep", &input).unwrap();
        assert!(suggestion.mcp_tool.contains("search_project"));
    }

    #[test]
    fn test_redirect_grep_url_no_redirect() {
        let input = json!({"pattern": "http://localhost:8080"});
        let suggestion = generate_redirect_suggestion("Grep", &input);
        assert!(suggestion.is_none());
    }

    #[test]
    fn test_redirect_bash_rg_symbol() {
        let input = json!({"command": "rg parse_uuid src/"});
        let suggestion = generate_redirect_suggestion("Bash", &input).unwrap();
        assert!(suggestion.mcp_tool.contains("find_references"));
        assert!(suggestion.mcp_params.contains("parse_uuid"));
    }

    #[test]
    fn test_redirect_read_no_redirect() {
        let input = json!({"file_path": "/src/main.rs"});
        let suggestion = generate_redirect_suggestion("Read", &input);
        assert!(suggestion.is_none());
    }

    // --- enrich_redirect_with_context_card ---

    #[test]
    fn test_enrich_bridge_warning() {
        let suggestion = RedirectSuggestion {
            mcp_tool: "code(action: \"find_references\")".to_string(),
            mcp_params: "symbol: \"MyStruct\"".to_string(),
            reason: "test".to_string(),
        };
        let card = crate::graph::models::ContextCard {
            cc_betweenness: 0.8,
            cc_pagerank: 0.5,
            ..Default::default()
        };
        let enriched = enrich_redirect_with_context_card(suggestion, &card);
        assert!(enriched
            .context_warnings
            .iter()
            .any(|w| w.contains("Bridge")));
    }

    #[test]
    fn test_enrich_co_changers() {
        let suggestion = RedirectSuggestion {
            mcp_tool: "code(action: \"find_references\")".to_string(),
            mcp_params: "symbol: \"X\"".to_string(),
            reason: "test".to_string(),
        };
        let card = crate::graph::models::ContextCard {
            cc_co_changers_top5: vec!["enrichment.rs".to_string(), "hook_handlers.rs".to_string()],
            ..Default::default()
        };
        let enriched = enrich_redirect_with_context_card(suggestion, &card);
        assert!(enriched
            .context_warnings
            .iter()
            .any(|w| w.contains("Co-changers")));
        assert!(enriched
            .context_warnings
            .iter()
            .any(|w| w.contains("enrichment.rs")));
    }

    #[test]
    fn test_enrich_low_risk_no_warning() {
        let suggestion = RedirectSuggestion {
            mcp_tool: "test".to_string(),
            mcp_params: "test".to_string(),
            reason: "test".to_string(),
        };
        let card = crate::graph::models::ContextCard::default();
        let enriched = enrich_redirect_with_context_card(suggestion, &card);
        // Default card has 0 for all metrics → no risk or bridge warnings
        assert!(enriched
            .context_warnings
            .iter()
            .all(|w| !w.contains("Bridge")));
        assert!(enriched
            .context_warnings
            .iter()
            .all(|w| !w.contains("High-risk")));
    }
}
