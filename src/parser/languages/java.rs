//! Java language extractor
//!
//! Extractor for Java code including:
//! - Classes and interfaces
//! - Methods and constructors
//! - Enums
//! - Annotations
//! - Generics
//! - JavaDoc comments

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Java code structure
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
            "class_declaration" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                // Extract class members
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "interface_declaration" => {
                if let Some(iface) = extract_interface(&child, source, file_path) {
                    parsed.symbols.push(iface.name.clone());
                    parsed.traits.push(iface);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_interface_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "import_declaration" => {
                if let Some(import) = extract_import(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "method_declaration" | "constructor_declaration" => {
                if let Some(func) = extract_method(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            _ => {
                // Recurse into other structures
                extract_recursive(&child, source, file_path, parsed)?;
            }
        }
    }

    Ok(())
}

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_java_visibility(node, source);
    let docstring = get_javadoc(node, source);
    let generics = extract_java_type_params(node, source);

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

fn extract_interface(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_java_visibility(node, source);
    let docstring = get_javadoc(node, source);
    let generics = extract_java_type_params(node, source);

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
    let visibility = get_java_visibility(node, source);
    let docstring = get_javadoc(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_constant")
                .filter_map(|v| get_field_text(&v, "name", source))
                .collect()
        })
        .unwrap_or_default();

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

fn extract_method(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_java_visibility(node, source);
    let docstring = get_javadoc(node, source);
    let generics = extract_java_type_params(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_java_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("type")
        .and_then(|t| get_text(&t, source))
        .map(|s| s.to_string());

    // Check modifiers for static, abstract, etc.
    let is_async = has_java_modifier(node, source, "synchronized");

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async, // Using for synchronized in Java
        is_unsafe: has_java_modifier(node, source, "native"),
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_import(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let text = get_text(node, source)?;
    let path = text
        .trim_start_matches("import ")
        .trim_start_matches("static ")
        .trim_end_matches(';')
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
            "method_declaration" | "constructor_declaration" => {
                if let Some(func) = extract_method(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_declaration" => {
                // Inner class
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                if let Some(inner_body) = child.child_by_field_name("body") {
                    extract_class_body(&inner_body, source, file_path, parsed)?;
                }
            }
            "interface_declaration" => {
                if let Some(iface) = extract_interface(&child, source, file_path) {
                    parsed.symbols.push(iface.name.clone());
                    parsed.traits.push(iface);
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

fn extract_interface_body(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in body.children(&mut body.walk()) {
        if child.kind() == "method_declaration" {
            if let Some(func) = extract_method(&child, source, file_path) {
                parsed.symbols.push(func.name.clone());
                parsed.functions.push(func);
            }
        }
    }
    Ok(())
}

fn extract_java_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "formal_parameter" || child.kind() == "spread_parameter" {
            let name = child
                .child_by_field_name("name")
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

fn extract_java_type_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
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

fn get_java_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    // Check modifiers
    for child in node.children(&mut node.walk()) {
        if child.kind() == "modifiers" {
            let text = get_text(&child, source).unwrap_or_default();
            if text.contains("public") {
                return Visibility::Public;
            } else if text.contains("protected") {
                return Visibility::Crate;
            } else if text.contains("private") {
                return Visibility::Private;
            }
        }
    }

    // Package-private by default
    Visibility::Crate
}

fn has_java_modifier(node: &tree_sitter::Node, source: &str, modifier: &str) -> bool {
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

fn get_javadoc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "block_comment" | "comment" => {
                let text = get_text(&sibling, source)?;
                if text.starts_with("/**") {
                    return Some(
                        text.trim_start_matches("/**")
                            .trim_end_matches("*/")
                            .lines()
                            .map(|l| l.trim().trim_start_matches('*').trim())
                            .filter(|l| !l.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                }
            }
            "marker_annotation" | "annotation" => {
                // Skip annotations
            }
            _ => break,
        }
        prev = sibling.prev_sibling();
    }

    None
}
