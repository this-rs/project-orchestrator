//! Chat entity extraction — extract code entity references from chat messages.
//!
//! Extracts file paths, backtick identifiers, and code patterns from user messages,
//! then validates them against the Neo4j knowledge graph to reject false positives.
//!
//! **Design constraints:**
//! - Lightweight: regex/string matching only, no LLM calls
//! - Non-blocking: designed to run in `tokio::spawn` background tasks
//! - Security: only extracts structured metadata, never stores raw message content

use std::collections::HashSet;

use tracing::debug;
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;

// ─── Types ──────────────────────────────────────────────────────────────────

/// The type of code entity extracted from chat text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// A file path (e.g. `src/main.rs`, `Cargo.toml`)
    File,
    /// A function name (e.g. `build_prompt`, `main`)
    Function,
    /// A struct/type name (e.g. `ChatManager`, `FileNode`)
    Struct,
    /// A trait name (e.g. `GraphStore`, `Iterator`)
    Trait,
    /// An enum name (e.g. `ChatEvent`, `NoteType`)
    Enum,
    /// A generic symbol — type unknown until validated
    Symbol,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::File => write!(f, "file"),
            EntityType::Function => write!(f, "function"),
            EntityType::Struct => write!(f, "struct"),
            EntityType::Trait => write!(f, "trait"),
            EntityType::Enum => write!(f, "enum"),
            EntityType::Symbol => write!(f, "symbol"),
        }
    }
}

/// An entity extracted from chat text (before validation).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtractedEntity {
    /// The type of entity (may be `Symbol` if unknown)
    pub entity_type: EntityType,
    /// The identifier or path as extracted from text
    pub identifier: String,
    /// How the entity was extracted
    pub source: ExtractionSource,
}

/// How an entity was extracted from chat text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionSource {
    /// Matched a file path pattern (e.g. `src/foo/bar.rs`)
    FilePath,
    /// Extracted from backticks (e.g. `` `MyStruct` ``)
    Backtick,
    /// Matched a code keyword pattern (e.g. `fn build_prompt`, `struct FileNode`)
    CodePattern,
}

/// A validated entity — confirmed to exist in the Neo4j knowledge graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidatedEntity {
    /// The resolved entity type (no longer `Symbol`)
    pub entity_type: EntityType,
    /// The canonical identifier from the graph
    pub identifier: String,
    /// How the entity was extracted
    pub source: ExtractionSource,
    /// The Neo4j node type label (e.g. "File", "Function", "Struct")
    pub node_label: String,
    /// The file this entity belongs to (if applicable)
    pub file_path: Option<String>,
}

// ─── Extraction ─────────────────────────────────────────────────────────────

/// Known source code file extensions for path detection.
const CODE_EXTENSIONS: &[&str] = &[
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".py",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".kt",
    ".swift",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".md",
    ".txt",
    ".sql",
    ".graphql",
    ".proto",
    ".dockerfile",
];

/// Known config/root file names (no directory prefix needed).
const ROOT_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "go.sum",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".env",
    ".env.example",
    "README.md",
];

/// Words that look like identifiers but are common English / code noise.
const NOISE_WORDS: &[&str] = &[
    "true", "false", "null", "None", "Some", "Ok", "Err", "self", "Self", "super", "crate", "pub",
    "fn", "let", "mut", "if", "else", "match", "for", "while", "loop", "return", "break",
    "continue", "use", "mod", "struct", "enum", "trait", "impl", "type", "where", "async", "await",
    "const", "static", "ref", "move", "dyn", "box", "in", "as", "is", "TODO", "FIXME", "NOTE",
    "HACK", "XXX", "SAFETY", "String", "Vec", "Option", "Result", "Box", "Arc", "Rc", "HashMap",
    "HashSet", "BTreeMap", "BTreeSet", "Uuid", "DateTime", "i8", "i16", "i32", "i64", "i128",
    "isize", "u8", "u16", "u32", "u64", "u128", "usize", "f32", "f64", "bool", "char", "str",
];

/// Extract code entity references from a chat message.
///
/// Uses three strategies:
/// 1. **File path patterns** — paths like `src/main.rs`, `Cargo.toml`
/// 2. **Backtick identifiers** — code in backticks like `` `MyStruct` ``
/// 3. **Code keyword patterns** — `fn build_prompt`, `struct FileNode`, etc.
///
/// Returns deduplicated entities. No Neo4j calls — pure text processing.
pub fn extract_entities(text: &str) -> Vec<ExtractedEntity> {
    let mut seen = HashSet::new();
    let mut entities = Vec::new();

    // Strategy 1: File paths
    for entity in extract_file_paths(text) {
        if seen.insert((entity.entity_type.clone(), entity.identifier.clone())) {
            entities.push(entity);
        }
    }

    // Strategy 2: Backtick identifiers
    for entity in extract_backtick_identifiers(text) {
        if seen.insert((entity.entity_type.clone(), entity.identifier.clone())) {
            entities.push(entity);
        }
    }

    // Strategy 3: Code keyword patterns (fn, struct, trait, enum, impl)
    for entity in extract_code_patterns(text) {
        if seen.insert((entity.entity_type.clone(), entity.identifier.clone())) {
            entities.push(entity);
        }
    }

    debug!(
        count = entities.len(),
        "Extracted entities from chat message"
    );
    entities
}

/// Strategy 1: Extract file paths from text.
///
/// Matches patterns like:
/// - `src/main.rs`, `src/neo4j/client.rs`
/// - `Cargo.toml`, `package.json`
/// - Paths with or without leading `/` or `./`
fn extract_file_paths(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();

    // Split text into tokens (whitespace, backticks, quotes, parens)
    for token in tokenize(text) {
        let cleaned = token.trim_matches(|c: char| {
            c == '`'
                || c == '\''
                || c == '"'
                || c == '('
                || c == ')'
                || c == '['
                || c == ']'
                || c == '{'
                || c == '}'
                || c == ','
                || c == ';'
                || c == ':'
        });

        if cleaned.is_empty() {
            continue;
        }

        // Check if token looks like a file path
        if is_file_path(cleaned) {
            // Normalize: strip leading ./
            let normalized = cleaned.strip_prefix("./").unwrap_or(cleaned);
            results.push(ExtractedEntity {
                entity_type: EntityType::File,
                identifier: normalized.to_string(),
                source: ExtractionSource::FilePath,
            });
        }
    }

    results
}

/// Check if a token looks like a file path.
fn is_file_path(token: &str) -> bool {
    // Must not be too short or too long
    if token.len() < 3 || token.len() > 200 {
        return false;
    }

    // Must not contain spaces (file paths in chat are usually space-free)
    if token.contains(' ') {
        return false;
    }

    // Check for known root files (exact match)
    if ROOT_FILES.contains(&token) {
        return true;
    }

    // Must contain a `/` or `\` (directory separator) or have a code extension
    let has_separator = token.contains('/') || token.contains('\\');
    let has_extension = CODE_EXTENSIONS.iter().any(|ext| token.ends_with(ext));

    if !has_separator && !has_extension {
        return false;
    }

    // If it has a separator, verify it looks like a path (not a URL protocol, etc.)
    if has_separator {
        // Reject URLs
        if token.starts_with("http://")
            || token.starts_with("https://")
            || token.starts_with("ftp://")
        {
            return false;
        }

        // Should have at least one valid path component
        let parts: Vec<&str> = token.split('/').collect();
        if parts.iter().all(|p| p.is_empty()) {
            return false;
        }

        return true;
    }

    // Has extension but no separator — only accept if it looks like a filename
    // (not a domain like "example.com")
    has_extension
}

/// Strategy 2: Extract identifiers from backticks.
///
/// Matches patterns like `` `MyStruct` ``, `` `build_prompt` ``, `` `src/main.rs` ``
fn extract_backtick_identifiers(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();

    // Find single-backtick spans (not triple-backtick code blocks)
    let mut chars = text.char_indices().peekable();
    while let Some((start_idx, ch)) = chars.next() {
        if ch != '`' {
            continue;
        }

        // Check for triple backtick (code block delimiter) — skip
        let next1 = chars.peek().map(|&(_, c)| c);
        if next1 == Some('`') {
            // Skip all consecutive backticks
            while chars.peek().map(|&(_, c)| c) == Some('`') {
                chars.next();
            }
            continue;
        }

        // Find the closing backtick
        let content_start = start_idx + 1;
        let mut content_end = None;
        for (idx, c) in chars.by_ref() {
            if c == '`' {
                content_end = Some(idx);
                break;
            }
            if c == '\n' {
                // Backtick span doesn't cross newlines
                break;
            }
        }

        if let Some(end) = content_end {
            let content = &text[content_start..end];
            let trimmed = content.trim();

            if trimmed.is_empty() || trimmed.len() > 200 {
                continue;
            }

            // If it looks like a file path, classify it as File
            if is_file_path(trimmed) {
                let normalized = trimmed.strip_prefix("./").unwrap_or(trimmed);
                results.push(ExtractedEntity {
                    entity_type: EntityType::File,
                    identifier: normalized.to_string(),
                    source: ExtractionSource::Backtick,
                });
                continue;
            }

            // If it contains spaces or special chars (not ::), skip
            if trimmed.contains(' ') || trimmed.contains('(') || trimmed.contains(')') {
                continue;
            }

            // If it's a noise word, skip
            if NOISE_WORDS.contains(&trimmed) {
                continue;
            }

            // Check if it's a valid Rust/code identifier
            if is_valid_identifier(trimmed) {
                let entity_type = classify_identifier(trimmed);
                results.push(ExtractedEntity {
                    entity_type,
                    identifier: trimmed.to_string(),
                    source: ExtractionSource::Backtick,
                });
            }
        }
    }

    results
}

/// Strategy 3: Extract code keyword patterns.
///
/// Matches patterns like `fn build_prompt`, `struct FileNode`, `trait GraphStore`,
/// `enum ChatEvent`, `impl ChatManager`.
fn extract_code_patterns(text: &str) -> Vec<ExtractedEntity> {
    let mut results = Vec::new();

    let patterns: &[(&str, EntityType)] = &[
        ("fn ", EntityType::Function),
        ("struct ", EntityType::Struct),
        ("trait ", EntityType::Trait),
        ("enum ", EntityType::Enum),
        ("impl ", EntityType::Struct), // impl blocks are for structs
    ];

    for line in text.lines() {
        let trimmed = line.trim();
        for &(keyword, ref entity_type) in patterns {
            // Find all occurrences of the keyword in the line
            let mut search_from = 0;
            while let Some(pos) = trimmed[search_from..].find(keyword) {
                let abs_pos = search_from + pos;

                // Ensure keyword is at word boundary (start of line or preceded by space/punct)
                let at_boundary =
                    abs_pos == 0 || !trimmed.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();

                if at_boundary {
                    let after_keyword = &trimmed[abs_pos + keyword.len()..];
                    // Extract the identifier after the keyword
                    let ident: String = after_keyword
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();

                    if !ident.is_empty()
                        && ident.len() >= 2
                        && !NOISE_WORDS.contains(&ident.as_str())
                    {
                        results.push(ExtractedEntity {
                            entity_type: entity_type.clone(),
                            identifier: ident,
                            source: ExtractionSource::CodePattern,
                        });
                    }
                }

                search_from = abs_pos + keyword.len();
            }
        }
    }

    results
}

/// Check if a string is a valid code identifier.
///
/// Supports:
/// - Rust identifiers: `snake_case`, `CamelCase`, `SCREAMING_CASE`
/// - Qualified paths: `module::Type`, `crate::foo::bar`
fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // Allow qualified paths (split on ::)
    let parts: Vec<&str> = s.split("::").collect();
    for part in &parts {
        if part.is_empty() {
            return false;
        }
        let first = part.chars().next().unwrap();
        if !first.is_ascii_alphabetic() && first != '_' {
            return false;
        }
        if !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return false;
        }
    }

    true
}

/// Classify an identifier based on naming convention.
///
/// - `CamelCase` → Struct (or Trait/Enum — validated later)
/// - `snake_case` → Function
/// - `SCREAMING_CASE` → Symbol (constant)
/// - `module::Type` → Symbol (qualified, validated later)
fn classify_identifier(s: &str) -> EntityType {
    // If it contains ::, it's a qualified path — generic symbol
    if s.contains("::") {
        return EntityType::Symbol;
    }

    let first = s.chars().next().unwrap_or('_');

    // If starts with uppercase → likely Struct/Trait/Enum
    if first.is_ascii_uppercase() {
        // Check if ALL uppercase + underscores → constant, not a type
        if s.chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        {
            return EntityType::Symbol;
        }
        return EntityType::Struct; // Will be refined during validation
    }

    // Starts with lowercase or _ → likely function
    EntityType::Function
}

/// Tokenize text by splitting on whitespace and common delimiters.
fn tokenize(text: &str) -> Vec<&str> {
    text.split(|c: char| c.is_ascii_whitespace() || c == '`' || c == '"' || c == '\'' || c == '|')
        .filter(|s| !s.is_empty())
        .collect()
}

// ─── Validation ─────────────────────────────────────────────────────────────

/// Validate extracted entities against the Neo4j knowledge graph.
///
/// For each entity:
/// - **File**: checks if the path exists as a `File` node
/// - **Function/Struct/Trait/Enum/Symbol**: checks via `find_symbol_references`
///
/// Entities that don't exist in the graph are filtered out (false positives).
/// Validated entities get their `node_label` and `file_path` populated.
pub async fn validate_entities<G: GraphStore>(
    entities: Vec<ExtractedEntity>,
    graph: &G,
    project_id: Option<Uuid>,
) -> Vec<ValidatedEntity> {
    let mut validated = Vec::new();

    for entity in &entities {
        match entity.entity_type {
            EntityType::File => {
                // Check if file exists in graph
                match graph.get_file(&entity.identifier).await {
                    Ok(Some(_file)) => {
                        validated.push(ValidatedEntity {
                            entity_type: EntityType::File,
                            identifier: entity.identifier.clone(),
                            source: entity.source.clone(),
                            node_label: "File".to_string(),
                            file_path: Some(entity.identifier.clone()),
                        });
                    }
                    Ok(None) => {
                        debug!(
                            path = %entity.identifier,
                            "File not found in graph, skipping"
                        );
                    }
                    Err(e) => {
                        debug!(
                            path = %entity.identifier,
                            error = %e,
                            "Error validating file, skipping"
                        );
                    }
                }
            }
            _ => {
                // For symbols, try find_symbol_references
                match graph
                    .find_symbol_references(&entity.identifier, 1, project_id)
                    .await
                {
                    Ok(refs) if !refs.is_empty() => {
                        // Determine the actual node type from the reference
                        let ref_type = &refs[0].reference_type;
                        let (resolved_type, node_label) =
                            resolve_node_type(ref_type, &entity.identifier);
                        validated.push(ValidatedEntity {
                            entity_type: resolved_type,
                            identifier: entity.identifier.clone(),
                            source: entity.source.clone(),
                            node_label,
                            file_path: Some(refs[0].file_path.clone()),
                        });
                    }
                    Ok(_) => {
                        debug!(
                            symbol = %entity.identifier,
                            "Symbol not found in graph, skipping"
                        );
                    }
                    Err(e) => {
                        debug!(
                            symbol = %entity.identifier,
                            error = %e,
                            "Error validating symbol, skipping"
                        );
                    }
                }
            }
        }
    }

    debug!(
        input = entities.len(),
        validated = validated.len(),
        "Validated entities against graph"
    );
    validated
}

/// Resolve the Neo4j node type from a symbol reference type string.
fn resolve_node_type(ref_type: &str, identifier: &str) -> (EntityType, String) {
    match ref_type {
        "call" | "function" => (EntityType::Function, "Function".to_string()),
        "struct" | "type" => (EntityType::Struct, "Struct".to_string()),
        "trait" => (EntityType::Trait, "Trait".to_string()),
        "enum" => (EntityType::Enum, "Enum".to_string()),
        _ => {
            // Fallback: use naming convention
            let entity_type = classify_identifier(identifier);
            let label = match entity_type {
                EntityType::Function => "Function",
                EntityType::Struct => "Struct",
                EntityType::Trait => "Trait",
                EntityType::Enum => "Enum",
                _ => "Function", // default
            };
            (entity_type, label.to_string())
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── File path extraction ────────────────────────────────────────────

    #[test]
    fn test_extract_file_path_simple() {
        let text = "Modifie le fichier src/main.rs pour ajouter le handler";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::File && e.identifier == "src/main.rs"),
            "Should extract src/main.rs, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_file_path_nested() {
        let text = "Le bug est dans src/neo4j/client.rs ligne 42";
        let entities = extract_entities(text);
        assert!(
            entities.iter().any(|e| e.entity_type == EntityType::File && e.identifier == "src/neo4j/client.rs"),
            "Should extract src/neo4j/client.rs, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_root_file() {
        let text = "Ajoute une dep dans Cargo.toml";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::File && e.identifier == "Cargo.toml"),
            "Should extract Cargo.toml, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_file_path_with_dot_prefix() {
        let text = "Regarde ./src/chat/mod.rs";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::File && e.identifier == "src/chat/mod.rs"),
            "Should normalize ./src/ to src/, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_no_extract_url() {
        let text = "Regarde https://github.com/foo/bar";
        let entities = extract_entities(text);
        assert!(
            !entities
                .iter()
                .any(|e| e.entity_type == EntityType::File && e.identifier.contains("github.com")),
            "Should not extract URLs as file paths, got: {:?}",
            entities
        );
    }

    // ── Backtick extraction ─────────────────────────────────────────────

    #[test]
    fn test_extract_backtick_function() {
        let text = "La fonction `build_prompt` doit retourner un String";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "build_prompt" && e.source == ExtractionSource::Backtick),
            "Should extract build_prompt from backticks, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_backtick_struct() {
        let text = "Le `ChatManager` gere les sessions";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "ChatManager" && e.entity_type == EntityType::Struct),
            "Should extract ChatManager as Struct, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_backtick_file_path() {
        let text = "Regarde `src/chat/mod.rs` pour voir les modules";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::File && e.identifier == "src/chat/mod.rs"),
            "Should extract file path from backticks, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_skip_noise_in_backticks() {
        let text = "Utilise `Option` et `Result` pour le retour";
        let entities = extract_entities(text);
        assert!(
            !entities
                .iter()
                .any(|e| e.identifier == "Option" || e.identifier == "Result"),
            "Should skip standard library types, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_skip_triple_backtick_code_blocks() {
        let text = "Voici le code:\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\nQu'en penses-tu?";
        let entities = extract_entities(text);
        // Should not extract `rust` as an identifier from ```rust
        assert!(
            !entities
                .iter()
                .any(|e| e.identifier == "rust" && e.source == ExtractionSource::Backtick),
            "Should not extract language tag from code block, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_qualified_path() {
        let text = "Appelle `crate::chat::manager::send_message`";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "crate::chat::manager::send_message"),
            "Should extract qualified paths, got: {:?}",
            entities
        );
    }

    // ── Code pattern extraction ─────────────────────────────────────────

    #[test]
    fn test_extract_fn_pattern() {
        let text = "Il faut modifier fn build_system_prompt dans le module chat";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "build_system_prompt"
                    && e.entity_type == EntityType::Function),
            "Should extract function from 'fn' pattern, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_struct_pattern() {
        let text = "Ajoute un champ a struct EntityExtractor";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "EntityExtractor" && e.entity_type == EntityType::Struct),
            "Should extract struct from 'struct' pattern, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_trait_pattern() {
        let text = "Implemente trait GraphStore pour MockGraphStore";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "GraphStore" && e.entity_type == EntityType::Trait),
            "Should extract trait from 'trait' pattern, got: {:?}",
            entities
        );
    }

    #[test]
    fn test_extract_enum_pattern() {
        let text = "Ajoute une variante a enum ChatEvent";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.identifier == "ChatEvent" && e.entity_type == EntityType::Enum),
            "Should extract enum from 'enum' pattern, got: {:?}",
            entities
        );
    }

    // ── Combined / realistic messages ───────────────────────────────────

    #[test]
    fn test_realistic_message_1() {
        let text =
            "Modifie src/main.rs et la fonction `build_prompt` pour supporter le nouveau format";
        let entities = extract_entities(text);
        assert!(
            entities
                .iter()
                .any(|e| e.entity_type == EntityType::File && e.identifier == "src/main.rs"),
            "Should extract file"
        );
        assert!(
            entities.iter().any(|e| e.identifier == "build_prompt"),
            "Should extract function"
        );
    }

    #[test]
    fn test_realistic_message_2() {
        let text = "Le `ChatManager` dans src/chat/manager.rs appelle `send_message` qui est defini dans le trait `GraphStore`";
        let entities = extract_entities(text);
        assert!(entities.iter().any(|e| e.identifier == "ChatManager"));
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::File && e.identifier == "src/chat/manager.rs"));
        assert!(entities.iter().any(|e| e.identifier == "send_message"));
        assert!(entities.iter().any(|e| e.identifier == "GraphStore"));
    }

    #[test]
    fn test_realistic_message_3() {
        let text = "Il faut ajouter un champ `energy` a struct NoteNode dans src/neo4j/models.rs et mettre a jour impl GraphStore dans src/neo4j/impl_graph_store.rs";
        let entities = extract_entities(text);
        assert!(entities.iter().any(|e| e.identifier == "energy"));
        assert!(entities
            .iter()
            .any(|e| e.identifier == "NoteNode" || e.identifier == "GraphStore"));
        assert!(entities
            .iter()
            .any(|e| e.identifier == "src/neo4j/models.rs"));
        assert!(entities
            .iter()
            .any(|e| e.identifier == "src/neo4j/impl_graph_store.rs"));
    }

    #[test]
    fn test_deduplication() {
        let text = "Modifie `src/main.rs`, regarde src/main.rs et le fichier src/main.rs";
        let entities = extract_entities(text);
        let file_count = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::File && e.identifier == "src/main.rs")
            .count();
        assert_eq!(
            file_count, 1,
            "Should deduplicate identical entities, got {} occurrences",
            file_count
        );
    }

    #[test]
    fn test_empty_message() {
        let entities = extract_entities("");
        assert!(
            entities.is_empty(),
            "Empty message should yield no entities"
        );
    }

    #[test]
    fn test_no_entities() {
        let text = "Bonjour, comment ca va aujourd'hui?";
        let entities = extract_entities(text);
        assert!(
            entities.is_empty(),
            "Casual message should yield no entities, got: {:?}",
            entities
        );
    }

    // ── Helper function tests ───────────────────────────────────────────

    #[test]
    fn test_is_valid_identifier() {
        assert!(is_valid_identifier("foo_bar"));
        assert!(is_valid_identifier("FooBar"));
        assert!(is_valid_identifier("_private"));
        assert!(is_valid_identifier("crate::foo::bar"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("123abc"));
        assert!(!is_valid_identifier("foo bar"));
        assert!(!is_valid_identifier("foo::"));
    }

    #[test]
    fn test_classify_identifier() {
        assert_eq!(classify_identifier("build_prompt"), EntityType::Function);
        assert_eq!(classify_identifier("ChatManager"), EntityType::Struct);
        assert_eq!(classify_identifier("MAX_RETRIES"), EntityType::Symbol);
        assert_eq!(classify_identifier("crate::foo"), EntityType::Symbol);
        assert_eq!(classify_identifier("_internal"), EntityType::Function);
    }

    #[test]
    fn test_is_file_path() {
        assert!(is_file_path("src/main.rs"));
        assert!(is_file_path("Cargo.toml"));
        assert!(is_file_path("src/neo4j/client.rs"));
        assert!(is_file_path("./src/main.rs"));
        assert!(is_file_path("tests/integration.rs"));
        assert!(!is_file_path("hello"));
        assert!(!is_file_path("https://github.com/foo"));
        assert!(!is_file_path("ab"));
        assert!(!is_file_path(""));
    }

    // ── Validation tests (with mock) ────────────────────────────────────

    #[tokio::test]
    async fn test_validate_filters_unknown_entities() {
        use crate::neo4j::mock::MockGraphStore;
        let mock = MockGraphStore::new();

        let entities = vec![
            ExtractedEntity {
                entity_type: EntityType::File,
                identifier: "src/nonexistent.rs".to_string(),
                source: ExtractionSource::FilePath,
            },
            ExtractedEntity {
                entity_type: EntityType::Function,
                identifier: "nonexistent_function".to_string(),
                source: ExtractionSource::Backtick,
            },
        ];

        let validated = validate_entities(entities, &mock, None).await;
        assert!(
            validated.is_empty(),
            "Non-existent entities should be filtered out, got: {:?}",
            validated
        );
    }
}
