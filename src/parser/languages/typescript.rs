//! TypeScript/JavaScript language extractor
//!
//! Enriched extractor for TypeScript/JavaScript code including:
//! - Functions (regular, arrow, async, generator)
//! - Classes and interfaces
//! - Type aliases and enums
//! - Parameters with types
//! - JSDoc comments
//! - Decorators

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract TypeScript/JavaScript code structure
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
            "generator_function_declaration" => {
                if let Some(mut func) = extract_function(&child, source, file_path) {
                    func.is_async = true; // Mark generators as async-like
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
                // Extract methods from class body
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_members(&body, source, file_path, parsed)?;
                }
            }
            "interface_declaration" => {
                if let Some(iface) = extract_interface(&child, source, file_path) {
                    parsed.symbols.push(iface.name.clone());
                    parsed.traits.push(iface);
                }
            }
            "type_alias_declaration" => {
                if let Some(s) = extract_type_alias(&child, source, file_path) {
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
            "import_statement" => {
                if let Some(import) = extract_import(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "export_statement" => {
                // Handle exported declarations
                extract_recursive(&child, source, file_path, parsed)?;
            }
            "lexical_declaration" | "variable_declaration" => {
                // Handle arrow functions assigned to variables
                extract_variable_functions(&child, source, file_path, parsed)?;
            }
            _ => {
                // Recurse into other nodes
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
    let is_async = has_child_kind(node, "async");
    let docstring = get_jsdoc(node, source);
    let generics = extract_ts_type_parameters(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_ts_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches(':').trim().to_string());

    let visibility = get_ts_visibility(node, source);

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
    let docstring = get_jsdoc(node, source);
    let generics = extract_ts_type_parameters(node, source);
    let visibility = get_ts_visibility(node, source);

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
    let docstring = get_jsdoc(node, source);
    let generics = extract_ts_type_parameters(node, source);
    let visibility = get_ts_visibility(node, source);

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

fn extract_type_alias(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_jsdoc(node, source);
    let generics = extract_ts_type_parameters(node, source);
    let visibility = get_ts_visibility(node, source);

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

fn extract_enum(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_jsdoc(node, source);
    let visibility = get_ts_visibility(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_assignment" || c.kind() == "property_identifier")
                .filter_map(|v| {
                    v.child_by_field_name("name")
                        .or(Some(v))
                        .and_then(|n| get_text(&n, source))
                        .map(|s| s.to_string())
                })
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

    // Extract the module path from the import
    let path = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "string")
        .and_then(|s| get_text(&s, source))
        .map(|s| s.trim_matches('"').trim_matches('\'').to_string())
        .unwrap_or_else(|| text.to_string());

    // Extract imported items
    let items: Vec<String> = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "import_clause")
        .map(|clause| {
            clause
                .children(&mut clause.walk())
                .filter_map(|c| {
                    if c.kind() == "identifier" || c.kind() == "import_specifier" {
                        get_text(&c, source).map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    Some(ImportNode {
        path,
        alias: None,
        items,
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_class_members(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in body.children(&mut body.walk()) {
        match child.kind() {
            "method_definition" => {
                if let Some(func) = extract_method(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "public_field_definition" | "field_definition" => {
                // Could extract fields as well if needed
            }
            _ => {}
        }
    }
    Ok(())
}

fn extract_method(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;
    let is_async = has_child_kind(node, "async");
    let docstring = get_jsdoc(node, source);
    let generics = extract_ts_type_parameters(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_ts_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches(':').trim().to_string());

    let visibility = get_ts_visibility(node, source);

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

fn extract_variable_functions(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "variable_declarator" {
            let name = get_field_text(&child, "name", source);
            let value = child.child_by_field_name("value");

            if let (Some(name), Some(value)) = (name, value) {
                if value.kind() == "arrow_function" || value.kind() == "function" {
                    let is_async = has_child_kind(&value, "async");
                    let docstring = get_jsdoc(node, source);
                    let generics = extract_ts_type_parameters(&value, source);

                    let params = value
                        .child_by_field_name("parameters")
                        .map(|p| extract_ts_params(&p, source))
                        .unwrap_or_default();

                    let return_type = value
                        .child_by_field_name("return_type")
                        .and_then(|r| get_text(&r, source))
                        .map(|s| s.trim_start_matches(':').trim().to_string());

                    let func = FunctionNode {
                        name: name.clone(),
                        visibility: Visibility::Private,
                        params,
                        return_type,
                        generics,
                        is_async,
                        is_unsafe: false,
                        complexity: calculate_complexity(&value),
                        file_path: file_path.to_string(),
                        line_start: node.start_position().row as u32 + 1,
                        line_end: node.end_position().row as u32 + 1,
                        docstring,
                    };

                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&value, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(name);
                    parsed.functions.push(func);
                }
            }
        }
    }
    Ok(())
}

fn extract_ts_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "required_parameter" | "optional_parameter" | "rest_parameter" => {
                let name = child
                    .child_by_field_name("pattern")
                    .or_else(|| find_child_by_kind(&child, "identifier"))
                    .and_then(|n| get_text(&n, source))
                    .unwrap_or("_")
                    .to_string();

                let type_name = child
                    .child_by_field_name("type")
                    .and_then(|t| get_text(&t, source))
                    .map(|s| s.trim_start_matches(':').trim().to_string());

                params.push(Parameter { name, type_name });
            }
            _ => {}
        }
    }

    params
}

fn extract_ts_type_parameters(node: &tree_sitter::Node, source: &str) -> Vec<String> {
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

fn get_ts_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    // Check for TypeScript access modifiers
    for child in node.children(&mut node.walk()) {
        if child.kind() == "accessibility_modifier" {
            if let Some(text) = get_text(&child, source) {
                return match text {
                    "public" => Visibility::Public,
                    "private" => Visibility::Private,
                    "protected" => Visibility::Crate, // Map protected to Crate
                    _ => Visibility::Public,
                };
            }
        }
    }

    // Check if exported
    if let Some(parent) = node.parent() {
        if parent.kind() == "export_statement" {
            return Visibility::Public;
        }
    }

    Visibility::Private
}

fn get_jsdoc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" => {
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
            _ => break,
        }
        prev = sibling.prev_sibling();
    }

    None
}
