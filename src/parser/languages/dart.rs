//! Dart language extractor — regex-first approach
//!
//! The tree-sitter-dart grammar (nielsenko v0.1.0) is fundamentally broken:
//! - Imports → ERROR nodes that corrupt the entire AST
//! - `extends` + constructor calls in method bodies → complete parse failure
//! - Doc comments break name extraction
//!
//! This extractor uses regex for reliable structure extraction (classes, functions,
//! enums, mixins, imports) with brace-matching for scope detection.
//!
//! TODO: Upgrade to nielsenko/tree-sitter-dart latest (supports Dart 3.11) and
//! switch back to AST-based extraction.

use crate::neo4j::models::*;
use crate::parser::ParsedFile;
use anyhow::Result;
use regex::Regex;
use std::sync::LazyLock;

// ─── Regex patterns ───────────────────────────────────────────────────────────

static RE_CLASS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^[ \t]*(?:abstract\s+|sealed\s+|final\s+|base\s+|interface\s+|mixin\s+)*class\s+(\w+)(?:<[^>]*>)?\s*(?:extends\s+([\w<>, ]+?))?\s*(?:with\s+([\w<>, ]+?))?\s*(?:implements\s+([\w<>, ]+?))?\s*\{"
    ).unwrap()
});

static RE_MIXIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*mixin\s+(\w+)(?:<[^>]*>)?\s*(?:on\s+([\w<>, ]+?))?\s*\{").unwrap()
});

static RE_ENUM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*enum\s+(\w+)\s*\{").unwrap());

static RE_EXTENSION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*extension\s+(\w+)?(?:<[^>]*>)?\s+on\s+([\w<>, ]+?)\s*\{").unwrap()
});

/// extension type Foo(Type value) implements Bar { ... }
static RE_EXTENSION_TYPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^[ \t]*extension\s+type\s+(\w+)(?:<[^>]*>)?\s*\([^)]*\)\s*(?:implements\s+([\w<>, ]+?))?\s*\{"
    ).unwrap()
});

/// Matches top-level and class-level function/method declarations.
/// Captures: optional return type, name, parameters
static RE_FUNCTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^[ \t]*(?:static\s+)?(?:abstract\s+)?(?:external\s+)?([\w<>?,\s]+?\s+)?(\w+)\s*\(([^)]*)\)\s*(?:async\s*\*?\s*)?(?:\{|=>|;)"
    ).unwrap()
});

/// Matches getter declarations
static RE_GETTER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*(?:static\s+)?([\w<>?]+\s+)?get\s+(\w+)\s*(?:\{|=>)").unwrap()
});

/// Matches setter declarations
static RE_SETTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*(?:static\s+)?set\s+(\w+)\s*\(").unwrap());

// ─── Dart keywords that should NOT be treated as function names ───────────────
const DART_KEYWORDS: &[&str] = &[
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "continue",
    "return",
    "try",
    "catch",
    "finally",
    "throw",
    "rethrow",
    "new",
    "const",
    "var",
    "final",
    "late",
    "class",
    "enum",
    "mixin",
    "extension",
    "typedef",
    "import",
    "export",
    "part",
    "library",
    "show",
    "hide",
    "as",
    "is",
    "in",
    "await",
    "yield",
    "async",
    "sync",
    "super",
    "this",
    "true",
    "false",
    "null",
    "void",
    "assert",
    "with",
    "implements",
    "extends",
    "abstract",
    "sealed",
    "base",
    "interface",
    "required",
    "covariant",
    "factory",
    "operator",
];

// ─── Public API ───────────────────────────────────────────────────────────────

/// Extract Dart code structure using regex-first approach.
/// This is the primary entry point — does NOT depend on tree-sitter AST.
pub fn extract(
    _root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    extract_all_regex(source, file_path, parsed)
}

/// Extract only the AST-based symbols (legacy, kept for API compatibility).
pub fn extract_ast(
    _root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    extract_all_regex(source, file_path, parsed)
}

/// Replace import/export lines with empty lines to preserve line numbers.
/// Kept for API compatibility but no longer needed by the regex extractor.
pub fn blank_import_lines(source: &str) -> String {
    source
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("import ")
                || trimmed.starts_with("export ")
                || trimmed.starts_with("part ")
                || trimmed.starts_with("library ")
            {
                ""
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ─── Core extraction ──────────────────────────────────────────────────────────

/// Full regex-based extraction for Dart files.
fn extract_all_regex(source: &str, file_path: &str, parsed: &mut ParsedFile) -> Result<()> {
    // 1. Imports
    extract_imports_regex(source, file_path, parsed);

    // 2. Build line → byte offset mapping
    let line_offsets = build_line_offsets(source);

    // 3. Find all class/mixin/enum/extension declarations with their brace ranges
    let class_ranges = extract_classes_regex(source, file_path, parsed, &line_offsets);
    extract_mixins_regex(source, file_path, parsed, &line_offsets);
    extract_enums_regex(source, file_path, parsed, &line_offsets);
    extract_extensions_regex(source, file_path, parsed, &line_offsets);
    extract_extension_types_regex(source, file_path, parsed, &line_offsets);

    // 4. Extract functions — both top-level and inside class bodies
    extract_functions_regex(source, file_path, parsed, &line_offsets, &class_ranges);

    Ok(())
}

/// Build a map from line number (0-based) to byte offset in source.
fn build_line_offsets(source: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (i, ch) in source.char_indices() {
        if ch == '\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

/// Convert byte offset to 1-based line number.
fn offset_to_line(offset: usize, line_offsets: &[usize]) -> u32 {
    match line_offsets.binary_search(&offset) {
        Ok(i) => i as u32 + 1,
        Err(i) => i as u32, // offset is in the middle of a line
    }
}

/// Find the matching closing brace for an opening brace at `open_pos`.
/// Handles nested braces, strings (single/double/triple quotes), and comments.
fn find_matching_brace(source: &str, open_pos: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut depth = 0i32;
    let mut i = open_pos;

    while i < len {
        match bytes[i] {
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
                i += 1;
            }
            // Skip single-line comments
            b'/' if i + 1 < len && bytes[i + 1] == b'/' => {
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            // Skip block comments
            b'/' if i + 1 < len && bytes[i + 1] == b'*' => {
                i += 2;
                let mut comment_depth = 1;
                while i + 1 < len && comment_depth > 0 {
                    if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        comment_depth += 1;
                        i += 2;
                    } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        comment_depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            // Skip triple-quoted strings
            b'\'' if i + 2 < len && bytes[i + 1] == b'\'' && bytes[i + 2] == b'\'' => {
                i += 3;
                while i + 2 < len {
                    if bytes[i] == b'\'' && bytes[i + 1] == b'\'' && bytes[i + 2] == b'\'' {
                        i += 3;
                        break;
                    }
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'"' if i + 2 < len && bytes[i + 1] == b'"' && bytes[i + 2] == b'"' => {
                i += 3;
                while i + 2 < len {
                    if bytes[i] == b'"' && bytes[i + 1] == b'"' && bytes[i + 2] == b'"' {
                        i += 3;
                        break;
                    }
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            // Skip single-quoted strings
            b'\'' => {
                i += 1;
                while i < len && bytes[i] != b'\'' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
            }
            // Skip double-quoted strings
            b'"' => {
                i += 1;
                while i < len && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    None
}

// ─── Class extraction ─────────────────────────────────────────────────────────

/// Returns (start_offset, end_offset) pairs for each class body.
fn extract_classes_regex(
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    line_offsets: &[usize],
) -> Vec<(usize, usize)> {
    let mut class_ranges = Vec::new();

    for cap in RE_CLASS.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name = cap[1].to_string();
        let line_start = offset_to_line(full_match.start(), line_offsets);

        // Find the opening brace
        let brace_pos = full_match.end() - 1; // regex ends with {
        let line_end = if let Some(close) = find_matching_brace(source, brace_pos) {
            class_ranges.push((brace_pos, close));
            offset_to_line(close, line_offsets)
        } else {
            line_start
        };

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        let parent_class = cap.get(2).map(|m| {
            m.as_str()
                .split('<')
                .next()
                .unwrap_or(m.as_str())
                .trim()
                .to_string()
        });

        let mut interfaces: Vec<String> = Vec::new();

        // with clause → mixins
        if let Some(m) = cap.get(3) {
            for mixin_name in m.as_str().split(',') {
                let clean = mixin_name
                    .split('<')
                    .next()
                    .unwrap_or(mixin_name)
                    .trim()
                    .to_string();
                if !clean.is_empty() {
                    interfaces.push(clean);
                }
            }
        }

        // implements clause
        if let Some(m) = cap.get(4) {
            for iface_name in m.as_str().split(',') {
                let clean = iface_name
                    .split('<')
                    .next()
                    .unwrap_or(iface_name)
                    .trim()
                    .to_string();
                if !clean.is_empty() {
                    interfaces.push(clean);
                }
            }
        }

        // Extract generics from the class name line
        let generics = extract_generics_from_match(full_match.as_str());

        let docstring = extract_doc_comment(source, full_match.start(), line_offsets);

        parsed.symbols.push(name.clone());
        parsed.structs.push(StructNode {
            name,
            visibility,
            generics,
            file_path: file_path.to_string(),
            line_start,
            line_end,
            docstring,
            parent_class,
            interfaces,
        });
    }

    class_ranges
}

// ─── Mixin extraction ─────────────────────────────────────────────────────────

fn extract_mixins_regex(
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    line_offsets: &[usize],
) {
    for cap in RE_MIXIN.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name = cap[1].to_string();

        // Skip if this was already captured as "mixin class" by RE_CLASS
        if parsed.structs.iter().any(|s| s.name == name) {
            continue;
        }

        let line_start = offset_to_line(full_match.start(), line_offsets);
        let brace_pos = full_match.end() - 1;
        let line_end = find_matching_brace(source, brace_pos)
            .map(|close| offset_to_line(close, line_offsets))
            .unwrap_or(line_start);

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        let generics = extract_generics_from_match(full_match.as_str());
        let docstring = extract_doc_comment(source, full_match.start(), line_offsets);

        parsed.symbols.push(name.clone());
        parsed.traits.push(TraitNode {
            name,
            visibility,
            generics,
            file_path: file_path.to_string(),
            line_start,
            line_end,
            docstring,
            is_external: false,
            source: None,
        });
    }
}

// ─── Enum extraction ──────────────────────────────────────────────────────────

fn extract_enums_regex(
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    line_offsets: &[usize],
) {
    for cap in RE_ENUM.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name = cap[1].to_string();
        let line_start = offset_to_line(full_match.start(), line_offsets);

        let brace_pos = full_match.end() - 1;
        let (line_end, variants) = if let Some(close) = find_matching_brace(source, brace_pos) {
            let body = &source[brace_pos + 1..close];
            let variants = extract_enum_variants(body);
            (offset_to_line(close, line_offsets), variants)
        } else {
            (line_start, vec![])
        };

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        let docstring = extract_doc_comment(source, full_match.start(), line_offsets);

        parsed.symbols.push(name.clone());
        parsed.enums.push(EnumNode {
            name,
            visibility,
            variants,
            file_path: file_path.to_string(),
            line_start,
            line_end,
            docstring,
        });
    }
}

/// Extract enum variant names from the body between braces.
/// Handles both single-line `{ a, b, c }` and multi-line enums with methods.
fn extract_enum_variants(body: &str) -> Vec<String> {
    let mut variants = Vec::new();

    // Split by commas and semicolons to handle both simple and enhanced enums
    for token in body.split([',', ';']) {
        let trimmed = token.trim();
        // Take first word (the variant name), ignoring arguments like `value(1)`
        let name = trimmed
            .split([' ', '(', '{', '\n'])
            .next()
            .unwrap_or("")
            .trim();

        if name.is_empty() || DART_KEYWORDS.contains(&name) {
            continue;
        }

        // Variant names start with lowercase in Dart convention
        // Skip method declarations (have return types before name)
        if name
            .chars()
            .next()
            .is_some_and(|c| c.is_lowercase() || c == '_')
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            variants.push(name.to_string());
        }
    }
    variants
}

// ─── Extension extraction ─────────────────────────────────────────────────────

fn extract_extensions_regex(
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    line_offsets: &[usize],
) {
    for cap in RE_EXTENSION.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let line_start = offset_to_line(full_match.start(), line_offsets);

        let brace_pos = full_match.end() - 1;
        let line_end = find_matching_brace(source, brace_pos)
            .map(|close| offset_to_line(close, line_offsets))
            .unwrap_or(line_start);

        let for_type = cap
            .get(2)
            .map(|m| {
                m.as_str()
                    .split('<')
                    .next()
                    .unwrap_or(m.as_str())
                    .trim()
                    .to_string()
            })
            .unwrap_or_default();

        if !for_type.is_empty() {
            parsed.impl_blocks.push(ImplNode {
                for_type,
                trait_name: None,
                generics: vec![],
                where_clause: None,
                file_path: file_path.to_string(),
                line_start,
                line_end,
            });
        }
    }
}

// ─── Extension type extraction ────────────────────────────────────────────────

fn extract_extension_types_regex(
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    line_offsets: &[usize],
) {
    for cap in RE_EXTENSION_TYPE.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name = cap[1].to_string();
        let line_start = offset_to_line(full_match.start(), line_offsets);

        let brace_pos = full_match.end() - 1;
        let line_end = find_matching_brace(source, brace_pos)
            .map(|close| offset_to_line(close, line_offsets))
            .unwrap_or(line_start);

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        let interfaces: Vec<String> = cap
            .get(2)
            .map(|m| {
                m.as_str()
                    .split(',')
                    .filter_map(|s| {
                        let clean = s.split('<').next().unwrap_or(s).trim().to_string();
                        if clean.is_empty() {
                            None
                        } else {
                            Some(clean)
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        parsed.symbols.push(name.clone());
        parsed.structs.push(StructNode {
            name,
            visibility,
            generics: vec![],
            file_path: file_path.to_string(),
            line_start,
            line_end,
            docstring: None,
            parent_class: None,
            interfaces,
        });
    }
}

// ─── Function extraction ──────────────────────────────────────────────────────

fn extract_functions_regex(
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    line_offsets: &[usize],
    class_ranges: &[(usize, usize)],
) {
    // Extract regular functions/methods
    for cap in RE_FUNCTION.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name_match = cap.get(2).unwrap();
        let name = name_match.as_str().to_string();

        // Skip keywords, constructors (capitalized name matching class name), etc.
        if DART_KEYWORDS.contains(&name.as_str()) {
            continue;
        }

        // Skip constructors (ClassName(...) or ClassName.named(...))
        if parsed.structs.iter().any(|s| s.name == name) {
            continue;
        }

        let offset = full_match.start();
        let line_start = offset_to_line(offset, line_offsets);

        // Determine if inside a class body
        let _in_class = class_ranges
            .iter()
            .any(|(start, end)| offset > *start && offset < *end);

        // Find the function body end
        let line_end = if full_match.as_str().ends_with('{') {
            let brace_pos = full_match.end() - 1;
            find_matching_brace(source, brace_pos)
                .map(|close| offset_to_line(close, line_offsets))
                .unwrap_or(line_start)
        } else if full_match.as_str().ends_with("=>") {
            // Arrow function — find the ; ending
            let rest = &source[full_match.end()..];
            rest.find(';')
                .map(|pos| offset_to_line(full_match.end() + pos, line_offsets))
                .unwrap_or(line_start)
        } else {
            // Abstract method ending with ;
            line_start
        };

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        let return_type = cap
            .get(1)
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| !s.is_empty() && s != "@override");

        let params_str = cap.get(3).map(|m| m.as_str()).unwrap_or("");
        let params = parse_param_string(params_str);

        let is_async = full_match.as_str().contains("async");
        let docstring = extract_doc_comment(source, offset, line_offsets);

        // Estimate complexity from body
        let complexity = if full_match.as_str().ends_with('{') {
            let brace_pos = full_match.end() - 1;
            find_matching_brace(source, brace_pos)
                .map(|close| estimate_complexity(&source[brace_pos..=close]))
                .unwrap_or(1)
        } else {
            1
        };

        parsed.symbols.push(name.clone());
        parsed.functions.push(FunctionNode {
            name,
            visibility,
            params,
            return_type,
            generics: vec![],
            is_async,
            is_unsafe: false,
            complexity,
            file_path: file_path.to_string(),
            line_start,
            line_end,
            docstring,
        });
    }

    // Extract getters
    for cap in RE_GETTER.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name = cap[2].to_string();

        // Skip if already found
        if parsed.functions.iter().any(|f| f.name == name) {
            continue;
        }

        let offset = full_match.start();
        let line_start = offset_to_line(offset, line_offsets);
        let line_end = if full_match.as_str().ends_with('{') {
            let brace_pos = full_match.end() - 1;
            find_matching_brace(source, brace_pos)
                .map(|close| offset_to_line(close, line_offsets))
                .unwrap_or(line_start)
        } else {
            line_start
        };

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        let return_type = cap.get(1).map(|m| m.as_str().trim().to_string());

        parsed.symbols.push(name.clone());
        parsed.functions.push(FunctionNode {
            name,
            visibility,
            params: vec![],
            return_type,
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: file_path.to_string(),
            line_start,
            line_end,
            docstring: None,
        });
    }

    // Extract setters
    for cap in RE_SETTER.captures_iter(source) {
        let full_match = cap.get(0).unwrap();
        let name = cap[1].to_string();

        if parsed.functions.iter().any(|f| f.name == name) {
            continue;
        }

        let offset = full_match.start();
        let line_start = offset_to_line(offset, line_offsets);

        let visibility = if name.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        parsed.symbols.push(name.clone());
        parsed.functions.push(FunctionNode {
            name,
            visibility,
            params: vec![Parameter {
                name: "value".to_string(),
                type_name: None,
            }],
            return_type: Some("void".to_string()),
            generics: vec![],
            is_async: false,
            is_unsafe: false,
            complexity: 1,
            file_path: file_path.to_string(),
            line_start,
            line_end: line_start,
            docstring: None,
        });
    }
}

// ─── Import extraction ────────────────────────────────────────────────────────

/// Extract imports via regex since tree-sitter-dart parses them as ERROR nodes.
pub fn extract_imports_regex(source: &str, file_path: &str, parsed: &mut ParsedFile) {
    for (line_num, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("import ") && !trimmed.starts_with("export ") {
            continue;
        }

        // Extract the URI between quotes
        let uri = if let Some(start) = trimmed.find('\'') {
            let rest = &trimmed[start + 1..];
            rest.find('\'').map(|end| &rest[..end])
        } else if let Some(start) = trimmed.find('"') {
            let rest = &trimmed[start + 1..];
            rest.find('"').map(|end| &rest[..end])
        } else {
            None
        };

        if let Some(uri) = uri {
            let alias = if let Some(as_pos) = trimmed.find(" as ") {
                let after_as = &trimmed[as_pos + 4..];
                let end = after_as.find([';', ' ']).unwrap_or(after_as.len());
                Some(after_as[..end].to_string())
            } else {
                None
            };

            parsed.imports.push(ImportNode {
                path: uri.to_string(),
                alias,
                items: vec![],
                file_path: file_path.to_string(),
                line: line_num as u32 + 1,
            });
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Parse parameter string like "int a, String b, {required int c}" into Parameter list.
fn parse_param_string(params_str: &str) -> Vec<Parameter> {
    if params_str.trim().is_empty() {
        return vec![];
    }

    let mut params = Vec::new();
    // Simple split by comma — doesn't handle nested generics perfectly but good enough
    for part in params_str.split(',') {
        let part = part
            .trim()
            .trim_start_matches('{')
            .trim_start_matches('[')
            .trim_end_matches('}')
            .trim_end_matches(']')
            .trim_start_matches("required ")
            .trim_start_matches("covariant ")
            .trim();

        if part.is_empty() {
            continue;
        }

        // Split "Type name" or just "name"
        let tokens: Vec<&str> = part.split_whitespace().collect();
        match tokens.len() {
            0 => {}
            1 => {
                params.push(Parameter {
                    name: tokens[0].to_string(),
                    type_name: None,
                });
            }
            _ => {
                // Last token is name, everything before is type
                let name = tokens.last().unwrap().to_string();
                let type_name = tokens[..tokens.len() - 1].join(" ");
                params.push(Parameter {
                    name,
                    type_name: Some(type_name),
                });
            }
        }
    }
    params
}

/// Extract generic type parameters from a match string like "class Foo<T, U> extends..."
fn extract_generics_from_match(text: &str) -> Vec<String> {
    if let Some(start) = text.find('<') {
        if let Some(end) = text.find('>') {
            return text[start + 1..end]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    vec![]
}

/// Extract doc comment (/// or /** */) immediately before a given offset.
fn extract_doc_comment(source: &str, decl_offset: usize, line_offsets: &[usize]) -> Option<String> {
    let decl_line = offset_to_line(decl_offset, line_offsets) as usize;
    if decl_line <= 1 {
        return None;
    }

    let lines: Vec<&str> = source.lines().collect();
    let mut doc_lines = Vec::new();
    let mut line_idx = decl_line - 2; // 0-based, line before declaration

    loop {
        if line_idx >= lines.len() {
            break;
        }
        let trimmed = lines[line_idx].trim();
        if trimmed.starts_with("///") {
            doc_lines.push(trimmed.trim_start_matches('/').trim().to_string());
        } else if trimmed.starts_with("/**") || trimmed.starts_with("* ") || trimmed == "*/" {
            // Part of block doc comment — simplified handling
            if trimmed.starts_with("/**") {
                let content = trimmed
                    .trim_start_matches("/**")
                    .trim_end_matches("*/")
                    .trim();
                if !content.is_empty() {
                    doc_lines.push(content.to_string());
                }
                break;
            }
            if trimmed.starts_with("* ") {
                doc_lines.push(trimmed.trim_start_matches("* ").to_string());
            }
        } else if trimmed.starts_with('@') {
            // Skip annotations like @override, @deprecated — continue looking above
        } else {
            break;
        }

        if line_idx == 0 {
            break;
        }
        line_idx -= 1;
    }

    if doc_lines.is_empty() {
        None
    } else {
        doc_lines.reverse();
        Some(doc_lines.join("\n"))
    }
}

/// Estimate cyclomatic complexity from a function body string.
fn estimate_complexity(body: &str) -> u32 {
    let mut complexity = 1u32;
    // Count branching keywords
    for keyword in &[
        "if ", "else ", "for ", "while ", "case ", "catch ", "?", "&&", "||",
    ] {
        complexity += body.matches(keyword).count() as u32;
    }
    complexity
}
