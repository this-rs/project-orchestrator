//! PHP language extractor
//!
//! Extractor for PHP code including:
//! - Classes and interfaces
//! - Functions and methods
//! - Traits
//! - Namespaces
//! - Type hints

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract PHP code structure
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
            "function_definition" => {
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
                // Extract methods
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "interface_declaration" => {
                if let Some(iface) = extract_interface(&child, source, file_path) {
                    parsed.symbols.push(iface.name.clone());
                    parsed.traits.push(iface);
                }
            }
            "trait_declaration" => {
                if let Some(trait_node) = extract_trait(&child, source, file_path) {
                    parsed.symbols.push(trait_node.name.clone());
                    parsed.traits.push(trait_node);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "namespace_definition" => {
                // Recurse into namespace
                if let Some(body) = child.child_by_field_name("body") {
                    extract_recursive(&body, source, file_path, parsed)?;
                }
            }
            "namespace_use_declaration" => {
                extract_use(&child, source, file_path, parsed)?;
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
    let docstring = get_php_doc(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_php_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches(':').trim().to_string());

    Some(FunctionNode {
        name,
        visibility: Visibility::Public,
        params,
        return_type,
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

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_php_doc(node, source);
    let visibility = get_php_visibility(node, source);

    // Extract base class and interfaces as generics
    let mut generics = Vec::new();
    if let Some(base) = node.child_by_field_name("base_clause") {
        if let Some(text) = get_text(&base, source) {
            generics.push(text.trim_start_matches("extends").trim().to_string());
        }
    }

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
    let docstring = get_php_doc(node, source);

    Some(TraitNode {
        name,
        visibility: Visibility::Public,
        generics: vec![],
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
        is_external: false,
        source: None,
    })
}

fn extract_trait(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_php_doc(node, source);

    Some(TraitNode {
        name,
        visibility: Visibility::Public,
        generics: vec![],
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
    let docstring = get_php_doc(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_case")
                .filter_map(|v| get_field_text(&v, "name", source))
                .collect()
        })
        .unwrap_or_default();

    Some(EnumNode {
        name,
        visibility: Visibility::Public,
        variants,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_class_body(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in body.children(&mut body.walk()) {
        if child.kind() == "method_declaration" {
            if let Some(func) = extract_method(&child, source, file_path) {
                let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                let calls = extract_calls_from_node(&child, source, &func_id);
                parsed.function_calls.extend(calls);
                parsed.symbols.push(func.name.clone());
                parsed.functions.push(func);
            }
        }
    }
    Ok(())
}

fn extract_method(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_php_visibility(node, source);
    let docstring = get_php_doc(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_php_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches(':').trim().to_string());

    let is_async = has_child_kind(node, "static_modifier");

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics: vec![],
        is_async, // Using for static
        is_unsafe: false,
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_use(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "namespace_use_clause" {
            let path = get_text(&child, source)
                .map(|s| s.to_string())
                .unwrap_or_default();

            parsed.imports.push(ImportNode {
                path,
                alias: None,
                items: vec![],
                file_path: file_path.to_string(),
                line: child.start_position().row as u32 + 1,
            });
        }
    }
    Ok(())
}

fn extract_php_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "simple_parameter" || child.kind() == "variadic_parameter" {
            let name = child
                .child_by_field_name("name")
                .and_then(|n| get_text(&n, source))
                .unwrap_or("_")
                .trim_start_matches('$')
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

fn get_php_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "visibility_modifier" {
            if let Some(text) = get_text(&child, source) {
                return match text {
                    "public" => Visibility::Public,
                    "protected" => Visibility::Crate,
                    "private" => Visibility::Private,
                    _ => Visibility::Public,
                };
            }
        }
    }
    Visibility::Public
}

fn get_php_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
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
