//! Go language extractor
//!
//! Enriched extractor for Go code including:
//! - Functions and methods
//! - Structs and interfaces
//! - Type definitions
//! - Package imports
//! - Doc comments

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Go code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        match node.kind() {
            "function_declaration" => {
                if let Some(func) = extract_function(&node, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&node, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "method_declaration" => {
                if let Some(func) = extract_method(&node, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&node, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "type_declaration" => {
                extract_type_declaration(&node, source, file_path, parsed)?;
            }
            "import_declaration" => {
                extract_imports(&node, source, file_path, parsed)?;
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

    // Go visibility: uppercase first letter = public
    let visibility = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_go_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("result")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.to_string());

    let docstring = get_go_doc(node, source);
    let generics = extract_go_type_params(node, source);

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async: false, // Go uses goroutines, not async/await
        is_unsafe: false,
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_method(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;

    let visibility = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_go_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("result")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.to_string());

    let docstring = get_go_doc(node, source);
    let generics = extract_go_type_params(node, source);

    // Extract receiver type for context
    let _receiver = node
        .child_by_field_name("receiver")
        .and_then(|r| get_text(&r, source));

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async: false,
        is_unsafe: false,
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_type_declaration(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "type_spec" {
            let name = get_field_text(&child, "name", source);

            if let Some(name) = name {
                let visibility =
                    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Visibility::Public
                    } else {
                        Visibility::Private
                    };

                let docstring = get_go_doc(node, source);
                let generics = extract_go_type_params(&child, source);

                // Determine if it's a struct, interface, or type alias
                let type_def = child.child_by_field_name("type");

                if let Some(type_node) = type_def {
                    match type_node.kind() {
                        "struct_type" => {
                            parsed.symbols.push(name.clone());
                            parsed.structs.push(StructNode {
                                name,
                                visibility,
                                generics,
                                file_path: file_path.to_string(),
                                line_start: child.start_position().row as u32 + 1,
                                line_end: child.end_position().row as u32 + 1,
                                docstring,
                            });
                        }
                        "interface_type" => {
                            parsed.symbols.push(name.clone());
                            parsed.traits.push(TraitNode {
                                name,
                                visibility,
                                generics,
                                file_path: file_path.to_string(),
                                line_start: child.start_position().row as u32 + 1,
                                line_end: child.end_position().row as u32 + 1,
                                docstring,
                                is_external: false,
                                source: None,
                            });
                        }
                        _ => {
                            // Type alias - treat as struct
                            parsed.symbols.push(name.clone());
                            parsed.structs.push(StructNode {
                                name,
                                visibility,
                                generics,
                                file_path: file_path.to_string(),
                                line_start: child.start_position().row as u32 + 1,
                                line_end: child.end_position().row as u32 + 1,
                                docstring,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn extract_imports(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "import_spec" => {
                let path = child
                    .child_by_field_name("path")
                    .and_then(|p| get_text(&p, source))
                    .map(|s| s.trim_matches('"').to_string());

                let alias = child
                    .child_by_field_name("name")
                    .and_then(|n| get_text(&n, source))
                    .map(|s| s.to_string());

                if let Some(path) = path {
                    parsed.imports.push(ImportNode {
                        path,
                        alias,
                        items: vec![],
                        file_path: file_path.to_string(),
                        line: child.start_position().row as u32 + 1,
                    });
                }
            }
            "import_spec_list" => {
                // Multiple imports in parentheses
                for spec in child.children(&mut child.walk()) {
                    if spec.kind() == "import_spec" {
                        let path = spec
                            .child_by_field_name("path")
                            .and_then(|p| get_text(&p, source))
                            .map(|s| s.trim_matches('"').to_string());

                        let alias = spec
                            .child_by_field_name("name")
                            .and_then(|n| get_text(&n, source))
                            .map(|s| s.to_string());

                        if let Some(path) = path {
                            parsed.imports.push(ImportNode {
                                path,
                                alias,
                                items: vec![],
                                file_path: file_path.to_string(),
                                line: spec.start_position().row as u32 + 1,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_go_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "parameter_declaration" {
            // Go can have multiple names with same type: a, b int
            let type_node = child
                .children(&mut child.walk())
                .find(|c| {
                    c.kind() == "type_identifier"
                        || c.kind() == "pointer_type"
                        || c.kind() == "slice_type"
                        || c.kind() == "map_type"
                        || c.kind() == "channel_type"
                        || c.kind() == "qualified_type"
                        || c.kind() == "array_type"
                        || c.kind() == "function_type"
                        || c.kind() == "struct_type"
                        || c.kind() == "interface_type"
                });

            let type_name = type_node
                .and_then(|t| get_text(&t, source))
                .map(|s| s.to_string());

            // Get all identifier names before the type
            for ident in child.children(&mut child.walk()) {
                if ident.kind() == "identifier" {
                    let name = get_text(&ident, source).unwrap_or("_").to_string();
                    params.push(Parameter {
                        name,
                        type_name: type_name.clone(),
                    });
                }
            }
        }
    }

    params
}

fn extract_go_type_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();

    let type_params = node
        .child_by_field_name("type_parameters")
        .or_else(|| find_child_by_kind(node, "type_parameter_list"));

    if let Some(params) = type_params {
        for param in params.children(&mut params.walk()) {
            if param.kind() == "type_parameter_declaration" {
                // Get the type parameter name
                for child in param.children(&mut param.walk()) {
                    if child.kind() == "identifier" {
                        if let Some(text) = get_text(&child, source) {
                            generics.push(text.to_string());
                        }
                    }
                }
            }
        }
    }

    generics
}

fn get_go_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = get_text(&sibling, source)?;
            // Go doc comments start with //
            if text.starts_with("//") {
                doc_lines.push(text.trim_start_matches('/').trim().to_string());
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
