//! Bash/Shell language extractor
//!
//! Extractor for Bash/Shell scripts including:
//! - Functions
//! - Variables (exported and local)
//! - Source/include statements
//! - Aliases

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Bash code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        match node.kind() {
            "function_definition" => {
                if let Some(func) = extract_function(&node, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&node, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "command" => {
                // Check for source/. commands (includes)
                if let Some(import) = extract_source(&node, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "variable_assignment" => {
                // Track exported variables as "structs"
                if let Some(var) = extract_variable(&node, source, file_path) {
                    parsed.symbols.push(var.name.clone());
                    parsed.structs.push(var);
                }
            }
            "declaration_command" => {
                // Handle export/declare/local
                extract_declaration(&node, source, file_path, parsed)?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_function(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_bash_doc(node, source);

    Some(FunctionNode {
        name,
        visibility: Visibility::Public, // Shell functions are always "public"
        params: vec![], // Bash uses $1, $2, etc.
        return_type: None,
        generics: vec![],
        is_async: false,
        is_unsafe: false,
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_source(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    // Get the command name
    let command_name = node
        .child_by_field_name("name")
        .and_then(|n| get_text(&n, source))?;

    // Check if it's source or .
    if command_name != "source" && command_name != "." {
        return None;
    }

    // Get the file being sourced
    let args = node.child_by_field_name("argument");
    let path = args
        .and_then(|a| get_text(&a, source))
        .map(|s| s.trim_matches('"').trim_matches('\'').to_string())?;

    Some(ImportNode {
        path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_variable(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_bash_doc(node, source);

    Some(StructNode {
        name,
        visibility: Visibility::Private, // Local by default
        generics: vec![],
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_declaration(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let text = get_text(node, source).unwrap_or_default();
    let is_export = text.starts_with("export");

    // Find variable assignments in the declaration
    for child in node.children(&mut node.walk()) {
        if child.kind() == "variable_assignment" {
            if let Some(name) = get_field_text(&child, "name", source) {
                let visibility = if is_export {
                    Visibility::Public
                } else {
                    Visibility::Private
                };

                parsed.symbols.push(name.clone());
                parsed.structs.push(StructNode {
                    name,
                    visibility,
                    generics: vec![],
                    file_path: file_path.to_string(),
                    line_start: child.start_position().row as u32 + 1,
                    line_end: child.end_position().row as u32 + 1,
                    docstring: get_bash_doc(node, source),
                });
            }
        }
    }

    Ok(())
}

fn get_bash_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = get_text(&sibling, source)?;
            // Bash comments start with #
            if text.starts_with('#') && !text.starts_with("#!") {
                doc_lines.push(text.trim_start_matches('#').trim().to_string());
            }
        } else {
            break;
        }
        prev = sibling.prev_sibling();
    }

    if doc_lines.is_empty() {
        None
    } else {
        doc_lines.reverse();
        Some(doc_lines.join("\n"))
    }
}
