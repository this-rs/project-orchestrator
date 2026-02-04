//! Ruby language extractor
//!
//! Extractor for Ruby code including:
//! - Classes and modules
//! - Methods (instance and class)
//! - Blocks and procs
//! - Mixins (include/extend)

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Ruby code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    extract_recursive(root, source, file_path, parsed, false)
}

fn extract_recursive(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    in_class: bool,
) -> Result<()> {
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            "method" => {
                if let Some(func) = extract_method(&child, source, file_path, in_class) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "singleton_method" => {
                // Class method (def self.method_name)
                if let Some(func) = extract_singleton_method(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                // Recurse into class body
                if let Some(body) = child.child_by_field_name("body") {
                    extract_recursive(&body, source, file_path, parsed, true)?;
                }
            }
            "module" => {
                if let Some(module) = extract_module(&child, source, file_path) {
                    parsed.symbols.push(module.name.clone());
                    parsed.traits.push(module);
                }
                // Recurse into module body
                if let Some(body) = child.child_by_field_name("body") {
                    extract_recursive(&body, source, file_path, parsed, true)?;
                }
            }
            "call" => {
                // Check for require/require_relative
                if let Some(import) = extract_require(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            _ => {
                // Recurse into other nodes
                extract_recursive(&child, source, file_path, parsed, in_class)?;
            }
        }
    }

    Ok(())
}

fn extract_method(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    _in_class: bool,
) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;

    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_ruby_params(&p, source))
        .unwrap_or_default();

    let docstring = get_ruby_doc(node, source);

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type: None, // Ruby is dynamically typed
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

fn extract_singleton_method(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_ruby_doc(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_ruby_params(&p, source))
        .unwrap_or_default();

    Some(FunctionNode {
        name: format!("self.{}", name),
        visibility: Visibility::Public,
        params,
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

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_ruby_doc(node, source);

    // Extract superclass as generic
    let generics: Vec<String> = node
        .child_by_field_name("superclass")
        .and_then(|s| get_text(&s, source))
        .map(|s| vec![s.to_string()])
        .unwrap_or_default();

    Some(StructNode {
        name,
        visibility: Visibility::Public,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_module(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_ruby_doc(node, source);

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

fn extract_require(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    // Check if this is a require call
    let method = get_field_text(node, "method", source)?;

    if method != "require" && method != "require_relative" {
        return None;
    }

    // Get the argument (path)
    let args = node.child_by_field_name("arguments")?;
    let path = args
        .children(&mut args.walk())
        .find(|c| c.kind() == "string")
        .and_then(|s| get_text(&s, source))
        .map(|s| s.trim_matches('"').trim_matches('\'').to_string())?;

    Some(ImportNode {
        path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_ruby_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "identifier" => {
                let name = get_text(&child, source).unwrap_or("_").to_string();
                params.push(Parameter {
                    name,
                    type_name: None,
                });
            }
            "optional_parameter" => {
                let name = get_field_text(&child, "name", source).unwrap_or("_".to_string());
                params.push(Parameter {
                    name,
                    type_name: None,
                });
            }
            "splat_parameter" | "hash_splat_parameter" | "block_parameter" => {
                let name = get_text(&child, source).unwrap_or("_").to_string();
                params.push(Parameter {
                    name,
                    type_name: None,
                });
            }
            "keyword_parameter" => {
                let name = get_field_text(&child, "name", source).unwrap_or("_".to_string());
                params.push(Parameter {
                    name: format!("{}:", name),
                    type_name: None,
                });
            }
            _ => {}
        }
    }

    params
}

fn get_ruby_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = get_text(&sibling, source)?;
            // Ruby doc comments typically start with #
            if text.starts_with('#') {
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
