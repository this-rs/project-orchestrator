//! Zig language extractor
//!
//! Extractor for Zig code including:
//! - Functions (fn, pub fn)
//! - Structs, enums, unions
//! - @import directives
//! - Test blocks
//! - Comptime blocks

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Zig code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    extract_recursive(root, source, file_path, parsed)
}

fn extract_recursive(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            // tree-sitter-zig 1.1 uses snake_case node kinds
            "function_declaration" => {
                if let Some(func) = extract_function(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "test_declaration" => {
                // Zig test blocks: test "name" { ... }
                if let Some(func) = extract_test(&child, source, file_path) {
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "variable_declaration" => {
                // const/var declarations — may contain @import, struct, or enum
                if let Some(import) = extract_import_from_decl(&child, source, file_path) {
                    parsed.imports.push(import);
                } else if let Some(s) = extract_type_decl(&child, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                } else if let Some(e) = extract_enum_decl(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            _ => {
                extract_recursive(&child, source, file_path, parsed)?;
            }
        }
    }

    Ok(())
}

fn extract_function(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<FunctionNode> {
    // Look for fn keyword and function name
    let text = get_text(node, source)?;

    // Find "fn" keyword and extract name
    let fn_idx = text.find("fn ")?;
    let after_fn = &text[fn_idx + 3..];
    let name_end = after_fn.find('(').unwrap_or(after_fn.len());
    let name = after_fn[..name_end].trim().to_string();

    if name.is_empty() || name == "(" {
        return None;
    }

    let visibility = if text.contains("pub ") {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let docstring = get_zig_doc(node, source);

    Some(FunctionNode {
        name,
        visibility,
        params: vec![], // Zig params require deeper parsing
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

fn extract_test(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let text = get_text(node, source)?;
    // test "test name" { ... }
    let name = if let Some(after) = text.strip_prefix("test ") {
        after
            .find('"')
            .and_then(|start| {
                after[start + 1..]
                    .find('"')
                    .map(|end| after[start + 1..start + 1 + end].to_string())
            })
            .unwrap_or_else(|| "test".to_string())
    } else {
        "test".to_string()
    };

    Some(FunctionNode {
        name: format!("test_{}", name.replace(' ', "_")),
        visibility: Visibility::Private,
        params: vec![],
        return_type: None,
        generics: vec![],
        is_async: false,
        is_unsafe: false,
        complexity: 1,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring: None,
    })
}

fn extract_import_from_decl(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<ImportNode> {
    let text = get_text(node, source)?;

    // Look for @import("...") pattern
    let import_idx = text.find("@import(")?;
    let after = &text[import_idx + 8..];
    let quote_start = after.find('"')?;
    let path_start = quote_start + 1;
    let path_end = after[path_start..].find('"')? + path_start;
    let import_path = after[path_start..path_end].to_string();

    if import_path.is_empty() {
        return None;
    }

    Some(ImportNode {
        path: import_path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_type_decl(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<StructNode> {
    let text = get_text(node, source)?;

    // Match patterns like: const Foo = struct { ... } or pub const Foo = struct { ... }
    if !text.contains("= struct") && !text.contains("= packed struct") {
        return None;
    }

    // Extract name: const NAME = struct
    let const_idx = text.find("const ")?;
    let after = &text[const_idx + 6..];
    let eq_idx = after.find('=')?;
    let name = after[..eq_idx].trim().to_string();

    if name.is_empty() {
        return None;
    }

    let visibility = if text.starts_with("pub ") {
        Visibility::Public
    } else {
        Visibility::Private
    };

    Some(StructNode {
        name,
        visibility,
        generics: vec![],
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring: get_zig_doc(node, source),
        parent_class: None,
        interfaces: vec![],
    })
}

fn extract_enum_decl(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let text = get_text(node, source)?;

    if !text.contains("= enum") {
        return None;
    }

    let const_idx = text.find("const ")?;
    let after = &text[const_idx + 6..];
    let eq_idx = after.find('=')?;
    let name = after[..eq_idx].trim().to_string();

    if name.is_empty() {
        return None;
    }

    let visibility = if text.starts_with("pub ") {
        Visibility::Public
    } else {
        Visibility::Private
    };

    Some(EnumNode {
        name,
        visibility,
        variants: vec![], // Would need deeper parsing
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring: get_zig_doc(node, source),
    })
}

fn get_zig_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "doc_comment" || sibling.kind() == "line_comment" {
            if let Some(text) = get_text(&sibling, source) {
                if text.starts_with("///") {
                    doc_lines.push(text.trim_start_matches("///").trim().to_string());
                } else {
                    break;
                }
            }
        } else {
            break;
        }
        prev = sibling.prev_sibling();
    }

    if doc_lines.is_empty() {
        return None;
    }

    doc_lines.reverse();
    Some(doc_lines.join("\n"))
}
