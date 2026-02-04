//! Swift language extractor
//!
//! Extractor for Swift code including:
//! - Classes and structs
//! - Functions and methods
//! - Protocols
//! - Enums
//! - Extensions
//! - Async/await

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Swift code structure
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
            "function_declaration" => {
                if let Some(func) = extract_function(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_declaration" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "struct_declaration" => {
                if let Some(s) = extract_struct(&child, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "protocol_declaration" => {
                if let Some(proto) = extract_protocol(&child, source, file_path) {
                    parsed.symbols.push(proto.name.clone());
                    parsed.traits.push(proto);
                }
            }
            "enum_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "extension_declaration" => {
                // Extensions add methods to existing types
                extract_extension(&child, source, file_path, parsed)?;
            }
            "import_declaration" => {
                if let Some(import) = extract_import(&child, source, file_path) {
                    parsed.imports.push(import);
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
    let name = get_field_text(node, "name", source)?;
    let visibility = get_swift_visibility(node, source);
    let docstring = get_swift_doc(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_swift_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches("->").trim().to_string());

    let generics = extract_swift_generics(node, source);
    let is_async = has_swift_modifier(node, source, "async");

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async,
        is_unsafe: false,
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_swift_visibility(node, source);
    let docstring = get_swift_doc(node, source);
    let generics = extract_swift_generics(node, source);

    Some(StructNode {
        name,
        visibility,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_struct(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_swift_visibility(node, source);
    let docstring = get_swift_doc(node, source);
    let generics = extract_swift_generics(node, source);

    Some(StructNode {
        name,
        visibility,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_protocol(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_swift_visibility(node, source);
    let docstring = get_swift_doc(node, source);
    let generics = extract_swift_generics(node, source);

    Some(TraitNode {
        name,
        visibility,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
        is_external: false,
        source: None,
    })
}

fn extract_enum(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_swift_visibility(node, source);
    let docstring = get_swift_doc(node, source);

    let mut variants = Vec::new();
    if let Some(body) = node.child_by_field_name("body") {
        for case in body.children(&mut body.walk()) {
            if case.kind() == "enum_case" {
                for pattern in case.children(&mut case.walk()) {
                    if pattern.kind() == "enum_case_pattern" {
                        if let Some(name) = get_field_text(&pattern, "name", source) {
                            variants.push(name);
                        }
                    }
                }
            }
        }
    }

    Some(EnumNode {
        name,
        visibility,
        variants,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_extension(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    // Get the type being extended
    let for_type = get_field_text(node, "name", source);

    if let Some(for_type) = for_type {
        // Create an impl block for the extension
        parsed.impl_blocks.push(ImplNode {
            for_type,
            trait_name: None,
            generics: vec![],
            where_clause: None,
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
        });
    }

    // Extract methods
    if let Some(body) = node.child_by_field_name("body") {
        extract_class_body(&body, source, file_path, parsed)?;
    }

    Ok(())
}

fn extract_import(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let text = get_text(node, source)?;
    let path = text
        .trim_start_matches("import ")
        .trim()
        .to_string();

    Some(ImportNode {
        path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_class_body(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in body.children(&mut body.walk()) {
        match child.kind() {
            "function_declaration" => {
                if let Some(func) = extract_function(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_declaration" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
            }
            "struct_declaration" => {
                if let Some(s) = extract_struct(&child, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
            }
            "enum_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn extract_swift_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "parameter" {
            let name = child
                .child_by_field_name("name")
                .or_else(|| child.child_by_field_name("internal_name"))
                .and_then(|n| get_text(&n, source))
                .unwrap_or("_")
                .to_string();

            let type_name = child
                .child_by_field_name("type")
                .and_then(|t| get_text(&t, source))
                .map(|s| s.to_string());

            params.push(Parameter { name, type_name });
        }
    }

    params
}

fn extract_swift_generics(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();

    let type_params = node
        .child_by_field_name("type_parameters")
        .or_else(|| find_child_by_kind(node, "type_parameters"));

    if let Some(params) = type_params {
        for param in params.children(&mut params.walk()) {
            if param.kind() == "type_parameter" {
                if let Some(text) = get_text(&param, source) {
                    generics.push(text.to_string());
                }
            }
        }
    }

    generics
}

fn get_swift_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "modifiers" {
            if let Some(text) = get_text(&child, source) {
                if text.contains("public") || text.contains("open") {
                    return Visibility::Public;
                } else if text.contains("internal") {
                    return Visibility::Crate;
                } else if text.contains("fileprivate") {
                    return Visibility::Super;
                } else if text.contains("private") {
                    return Visibility::Private;
                }
            }
        }
    }
    Visibility::Crate // internal by default
}

fn has_swift_modifier(node: &tree_sitter::Node, source: &str, modifier: &str) -> bool {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "modifiers" {
            if let Some(text) = get_text(&child, source) {
                if text.contains(modifier) {
                    return true;
                }
            }
        }
    }
    false
}

fn get_swift_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" || sibling.kind() == "multiline_comment" {
            let text = get_text(&sibling, source)?;
            // Swift doc comments: /// or /** */
            if text.starts_with("///") {
                doc_lines.push(text.trim_start_matches('/').trim().to_string());
            } else if text.starts_with("/**") {
                return Some(
                    text.trim_start_matches("/**")
                        .trim_end_matches("*/")
                        .lines()
                        .map(|l| l.trim().trim_start_matches('*').trim())
                        .filter(|l| !l.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            } else {
                break;
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
