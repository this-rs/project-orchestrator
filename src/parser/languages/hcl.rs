//! HCL/Terraform language extractor
//!
//! Extractor for HCL (HashiCorp Configuration Language) code including:
//! - Resources and data sources → StructNode
//! - Variables, outputs, locals → symbols
//! - Module blocks → ImportNode
//! - Provider blocks → symbols
//! - Inter-resource references → FunctionCall
//! - Implicit same-directory scope → ImportNode("__same_dir__")

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::{FunctionCall, ParsedFile};
use anyhow::Result;

/// Extract HCL/Terraform code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    extract_blocks(root, source, file_path, parsed)?;

    // Add implicit same-directory import (Terraform scope: all .tf in same dir share scope)
    parsed.imports.push(ImportNode {
        path: "__same_dir__".to_string(),
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: 0,
    });

    Ok(())
}

/// Extract all top-level blocks from the HCL config
///
/// Structure: config_file → body → block[]
fn extract_blocks(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    // config_file has a single "body" child which contains all blocks
    let body = {
        let mut cursor = node.walk();
        let mut found = None;
        for child in node.children(&mut cursor) {
            if child.kind() == "body" {
                found = Some(child);
                break;
            }
        }
        found
    };

    if let Some(body) = body {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "block" {
                extract_block(&child, source, file_path, parsed);
            }
        }
    }

    Ok(())
}

/// Extract a single HCL block based on its type identifier
fn extract_block(node: &tree_sitter::Node, source: &str, file_path: &str, parsed: &mut ParsedFile) {
    // First child should be the block type identifier
    let block_type = match get_block_type(node, source) {
        Some(t) => t,
        None => return,
    };

    match block_type.as_str() {
        "resource" => extract_resource(node, source, file_path, parsed),
        "data" => extract_data_source(node, source, file_path, parsed),
        "variable" => extract_variable(node, source, file_path, parsed),
        "output" => extract_output(node, source, file_path, parsed),
        "locals" => extract_locals(node, source, file_path, parsed),
        "module" => extract_module(node, source, file_path, parsed),
        "provider" => extract_provider(node, source, file_path, parsed),
        _ => {} // terraform, required_providers, etc. — skip
    }
}

/// Get the block type identifier (first child that is an identifier)
fn get_block_type(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            return get_text(&child, source).map(|s| s.to_string());
        }
    }
    None
}

/// Get block labels (string_lit children between the identifier and body)
fn get_block_labels(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut labels = Vec::new();
    let mut cursor = node.walk();
    let mut past_identifier = false;

    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            past_identifier = true;
            continue;
        }
        if !past_identifier {
            continue;
        }
        // Labels are string_lit nodes
        if child.kind() == "string_lit" {
            if let Some(text) = get_text(&child, source) {
                // Strip surrounding quotes
                let clean = text.trim_matches('"').to_string();
                if !clean.is_empty() {
                    labels.push(clean);
                }
            }
        }
        // Stop at block body
        if child.kind() == "block" || child.kind() == "object" {
            break;
        }
    }

    labels
}

/// Get the block body node (the { ... } part)
///
/// In tree-sitter-hcl, a block's children are:
///   identifier + string_lit[] + block_start + body + block_end
#[allow(clippy::manual_find)] // cursor lifetime prevents .find()
fn get_block_body<'a>(node: &'a tree_sitter::Node<'a>) -> Option<tree_sitter::Node<'a>> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "body" {
            return Some(child);
        }
    }
    None
}

/// Extract `resource "type" "name" { ... }` → StructNode
fn extract_resource(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) {
    let labels = get_block_labels(node, source);
    if labels.len() >= 2 {
        let resource_type = &labels[0];
        let resource_name = &labels[1];
        let name = format!("{}.{}", resource_type, resource_name);

        parsed.symbols.push(name.clone());
        parsed.structs.push(StructNode {
            name,
            visibility: Visibility::Public,
            generics: vec![],
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
            docstring: None,
            parent_class: None,
            interfaces: vec![],
        });

        // Extract inter-resource references from the block body
        if let Some(body) = get_block_body(node) {
            let caller_id = format!(
                "{}:{}.{}:{}",
                file_path,
                resource_type,
                resource_name,
                node.start_position().row + 1
            );
            extract_references(&body, source, &caller_id, parsed);
        }
    }
}

/// Extract `data "type" "name" { ... }` → StructNode with "data." prefix
fn extract_data_source(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) {
    let labels = get_block_labels(node, source);
    if labels.len() >= 2 {
        let data_type = &labels[0];
        let data_name = &labels[1];
        let name = format!("data.{}.{}", data_type, data_name);

        parsed.symbols.push(name.clone());
        parsed.structs.push(StructNode {
            name,
            visibility: Visibility::Public,
            generics: vec![],
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
            docstring: None,
            parent_class: None,
            interfaces: vec![],
        });
    }
}

/// Extract `variable "name" { ... }` → symbol "var.name"
fn extract_variable(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) {
    let labels = get_block_labels(node, source);
    if let Some(var_name) = labels.first() {
        let symbol = format!("var.{}", var_name);
        parsed.symbols.push(symbol);
    }
    let _ = file_path; // used for consistency
}

/// Extract `output "name" { ... }` → symbol "output.name"
fn extract_output(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) {
    let labels = get_block_labels(node, source);
    if let Some(out_name) = labels.first() {
        let symbol = format!("output.{}", out_name);
        parsed.symbols.push(symbol);
    }

    // Extract references from the output value expression
    if let Some(body) = get_block_body(node) {
        let caller_id = format!(
            "{}:output.{}:{}",
            file_path,
            labels.first().map(|s| s.as_str()).unwrap_or("unknown"),
            node.start_position().row + 1
        );
        extract_references(&body, source, &caller_id, parsed);
    }
}

/// Extract `locals { key = value ... }` → symbols "local.key" for each attribute
///
/// Attribute children (no named fields): identifier(key), "=", expression(value)
fn extract_locals(
    node: &tree_sitter::Node,
    source: &str,
    _file_path: &str,
    parsed: &mut ParsedFile,
) {
    if let Some(body) = get_block_body(node) {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "attribute" {
                // named_child(0) = identifier (the key)
                if let Some(key) = child.named_child(0) {
                    if key.kind() == "identifier" {
                        if let Some(key_text) = get_text(&key, source) {
                            let symbol = format!("local.{}", key_text);
                            parsed.symbols.push(symbol);
                        }
                    }
                }
            }
        }
    }
}

/// Extract `module "name" { source = "..." }` → ImportNode
fn extract_module(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) {
    let labels = get_block_labels(node, source);
    let module_name = labels.first().cloned();

    // Find the "source" attribute inside the block body
    if let Some(body) = get_block_body(node) {
        if let Some(source_val) = find_attribute_value(&body, "source", source) {
            parsed.imports.push(ImportNode {
                path: source_val,
                alias: module_name.clone(),
                items: vec![],
                file_path: file_path.to_string(),
                line: node.start_position().row as u32 + 1,
            });
        }
    }

    // Also add module name as a symbol
    if let Some(name) = module_name {
        parsed.symbols.push(format!("module.{}", name));
    }
}

/// Extract `provider "name" { ... }` → symbol "provider.name"
fn extract_provider(
    node: &tree_sitter::Node,
    source: &str,
    _file_path: &str,
    parsed: &mut ParsedFile,
) {
    let labels = get_block_labels(node, source);
    if let Some(provider_name) = labels.first() {
        let symbol = format!("provider.{}", provider_name);
        parsed.symbols.push(symbol);
    }
}

/// Find the string value of a named attribute inside a block body.
/// e.g., find_attribute_value(body, "source", ...) extracts the value of `source = "..."`.
///
/// Attribute children (no named fields): named_child(0)=identifier, named_child(1)=expression
fn find_attribute_value(body: &tree_sitter::Node, attr_name: &str, source: &str) -> Option<String> {
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "attribute" {
            // named_child(0) = identifier (key), named_child(1) = expression (value)
            if let Some(key) = child.named_child(0) {
                if key.kind() == "identifier" && get_text(&key, source) == Some(attr_name) {
                    if let Some(val) = child.named_child(1) {
                        return extract_string_value(&val, source);
                    }
                }
            }
        }
    }
    None
}

/// Extract a string value from an expression node (handles string_lit, template_literal)
fn extract_string_value(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let text = get_text(node, source)?;
    // Strip surrounding quotes
    let clean = text.trim_matches('"').to_string();
    if clean.is_empty() {
        None
    } else {
        Some(clean)
    }
}

/// Extract inter-resource references from expressions within a block body.
///
/// Looks for patterns like `aws_vpc.main.id` which reference other resources.
/// These become FunctionCall entries in the parsed output.
///
/// Ignores internal scope prefixes: var., local., data., module., self., each., count., path.
fn extract_references(
    node: &tree_sitter::Node,
    source: &str,
    caller_id: &str,
    parsed: &mut ParsedFile,
) {
    let mut cursor = node.walk();
    extract_references_recursive(&mut cursor, source, caller_id, parsed);
}

fn extract_references_recursive(
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    caller_id: &str,
    parsed: &mut ParsedFile,
) {
    loop {
        let node = cursor.node();

        // Look for variable_expr nodes which represent references
        if node.kind() == "variable_expr" {
            if let Some(text) = get_text(&node, source) {
                // Check if this variable is followed by get_attr to form a resource reference
                // Pattern: <identifier> in variable_expr, possibly followed by get_attr siblings
                if let Some(parent) = node.parent() {
                    let full_ref = build_reference_chain(&parent, source);
                    if let Some(ref_parts) = full_ref {
                        // Must have at least 2 parts: type.name
                        if ref_parts.len() >= 2 {
                            let prefix = &ref_parts[0];
                            // Skip internal scopes
                            if !is_internal_scope(prefix) {
                                let callee_name = format!("{}.{}", ref_parts[0], ref_parts[1]);
                                // Avoid duplicate references
                                if !parsed.function_calls.iter().any(|c| {
                                    c.caller_id == caller_id
                                        && c.callee_name == callee_name
                                        && c.line == node.start_position().row as u32 + 1
                                }) {
                                    parsed.function_calls.push(FunctionCall {
                                        caller_id: caller_id.to_string(),
                                        callee_name,
                                        line: node.start_position().row as u32 + 1,
                                        confidence: 0.70,
                                        reason: "hcl-reference".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                let _ = text;
            }
        }

        // Recurse into children
        if cursor.goto_first_child() {
            extract_references_recursive(cursor, source, caller_id, parsed);
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

/// Build a reference chain from a traversal expression.
/// e.g., `aws_vpc.main.id` → ["aws_vpc", "main", "id"]
fn build_reference_chain(node: &tree_sitter::Node, source: &str) -> Option<Vec<String>> {
    let text = get_text(node, source)?;
    let parts: Vec<String> = text.split('.').map(|s| s.trim().to_string()).collect();
    if parts.len() >= 2 {
        Some(parts)
    } else {
        None
    }
}

/// Check if a reference prefix is an internal Terraform scope (not a cross-resource reference)
fn is_internal_scope(prefix: &str) -> bool {
    matches!(
        prefix,
        "var" | "local" | "data" | "module" | "self" | "each" | "count" | "path" | "terraform"
    )
}
