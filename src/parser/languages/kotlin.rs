//! Kotlin language extractor
//!
//! Extractor for Kotlin code including:
//! - Classes and data classes
//! - Functions (including extension functions)
//! - Interfaces
//! - Objects and companions
//! - Coroutines (suspend functions)

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Kotlin code structure
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
                // Extract members
                if let Some(body) = child.child_by_field_name("class_body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "interface_declaration" => {
                if let Some(iface) = extract_interface(&child, source, file_path) {
                    parsed.symbols.push(iface.name.clone());
                    parsed.traits.push(iface);
                }
            }
            "object_declaration" => {
                if let Some(obj) = extract_object(&child, source, file_path) {
                    parsed.symbols.push(obj.name.clone());
                    parsed.structs.push(obj);
                }
                if let Some(body) = child.child_by_field_name("class_body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_class_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "import_header" => {
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
    let visibility = get_kotlin_visibility(node, source);
    let docstring = get_kdoc(node, source);

    let params = node
        .child_by_field_name("function_value_parameters")
        .or_else(|| find_child_by_kind(node, "function_value_parameters"))
        .map(|p| extract_kotlin_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches(':').trim().to_string());

    let generics = extract_kotlin_type_params(node, source);
    let is_suspend = has_kotlin_modifier(node, source, "suspend");

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async: is_suspend,
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
    let visibility = get_kotlin_visibility(node, source);
    let docstring = get_kdoc(node, source);
    let generics = extract_kotlin_type_params(node, source);

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
    let visibility = get_kotlin_visibility(node, source);
    let docstring = get_kdoc(node, source);
    let generics = extract_kotlin_type_params(node, source);

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

fn extract_object(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_kotlin_visibility(node, source);
    let docstring = get_kdoc(node, source);

    Some(StructNode {
        name,
        visibility,
        generics: vec![],
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_enum(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_kotlin_visibility(node, source);
    let docstring = get_kdoc(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("class_body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_entry")
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
                // Nested class
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
            }
            "companion_object" => {
                if let Some(body) = child.child_by_field_name("class_body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn extract_kotlin_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "function_value_parameter" {
            let name = child
                .child_by_field_name("name")
                .and_then(|n| get_text(&n, source))
                .unwrap_or("_")
                .to_string();

            let type_name = child
                .child_by_field_name("type")
                .and_then(|t| get_text(&t, source))
                .map(|s| s.trim_start_matches(':').trim().to_string());

            params.push(Parameter { name, type_name });
        }
    }

    params
}

fn extract_kotlin_type_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
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

fn get_kotlin_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "modifiers" {
            for modifier in child.children(&mut child.walk()) {
                if modifier.kind() == "visibility_modifier" {
                    if let Some(text) = get_text(&modifier, source) {
                        return match text {
                            "public" => Visibility::Public,
                            "internal" => Visibility::Crate,
                            "protected" => Visibility::Crate,
                            "private" => Visibility::Private,
                            _ => Visibility::Public,
                        };
                    }
                }
            }
        }
    }
    Visibility::Public // Public by default in Kotlin
}

fn has_kotlin_modifier(node: &tree_sitter::Node, source: &str, modifier: &str) -> bool {
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

fn get_kdoc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if sibling.kind() == "multiline_comment" {
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
        } else {
            break;
        }
        prev = sibling.prev_sibling();
    }

    None
}
