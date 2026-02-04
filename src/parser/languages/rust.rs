//! Rust language extractor
//!
//! Full-featured extractor for Rust code including:
//! - Functions (with async/unsafe modifiers)
//! - Structs, Traits, Enums
//! - Impl blocks
//! - Generics and lifetimes
//! - Derive macros

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::{FunctionCall, ParsedFile};
use anyhow::Result;

/// Extract Rust code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        match node.kind() {
            "function_item" => {
                if let Some(func) = extract_function(&node, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_function_calls(&node, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "struct_item" => {
                if let Some(s) = extract_struct(&node, source, file_path) {
                    // Extract derive traits
                    let derives = extract_derive_traits(&node, source);
                    for trait_name in derives {
                        parsed.impl_blocks.push(ImplNode {
                            for_type: s.name.clone(),
                            trait_name: Some(trait_name),
                            generics: s.generics.clone(),
                            where_clause: None,
                            file_path: file_path.to_string(),
                            line_start: s.line_start,
                            line_end: s.line_start,
                        });
                    }
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
            }
            "trait_item" => {
                if let Some(t) = extract_trait(&node, source, file_path) {
                    parsed.symbols.push(t.name.clone());
                    parsed.traits.push(t);
                }
            }
            "enum_item" => {
                if let Some(e) = extract_enum(&node, source, file_path) {
                    let derives = extract_derive_traits(&node, source);
                    for trait_name in derives {
                        parsed.impl_blocks.push(ImplNode {
                            for_type: e.name.clone(),
                            trait_name: Some(trait_name),
                            generics: vec![],
                            where_clause: None,
                            file_path: file_path.to_string(),
                            line_start: e.line_start,
                            line_end: e.line_start,
                        });
                    }
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "use_declaration" => {
                if let Some(import) = extract_import(&node, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "impl_item" => {
                extract_impl(&node, source, file_path, parsed)?;
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
    let visibility = get_visibility(node, source);
    let is_async = has_modifier(node, source, "async");
    let is_unsafe = has_modifier(node, source, "unsafe");

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches("->").trim().to_string());

    let docstring = get_rust_docstring(node, source);
    let generics = extract_rust_type_parameters(node, source);

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async,
        is_unsafe,
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_struct(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_visibility(node, source);
    let docstring = get_rust_docstring(node, source);
    let generics = extract_rust_type_parameters(node, source);

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

fn extract_trait(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_visibility(node, source);
    let docstring = get_rust_docstring(node, source);
    let generics = extract_rust_type_parameters(node, source);

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
    let visibility = get_visibility(node, source);
    let docstring = get_rust_docstring(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_variant")
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
    let path = get_text(node, source)?;
    let path = path
        .trim_start_matches("use ")
        .trim_end_matches(';')
        .to_string();

    Some(ImportNode {
        path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_impl(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let for_type = get_impl_type(node, source);
    let trait_name = get_impl_trait(node, source);

    if let Some(for_type) = for_type {
        let generics = extract_rust_type_parameters(node, source);

        parsed.impl_blocks.push(ImplNode {
            for_type: for_type.clone(),
            trait_name: trait_name.clone(),
            generics,
            where_clause: None,
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
        });
    }

    // Extract methods from the body
    if let Some(body) = node.child_by_field_name("body") {
        for child in body.children(&mut body.walk()) {
            if child.kind() == "function_item" {
                if let Some(func) = extract_function(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_function_calls(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
        }
    }

    Ok(())
}

/// Get the type being implemented
fn get_impl_type(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut found_for = false;
    let mut first_type: Option<String> = None;

    for child in node.children(&mut node.walk()) {
        if child.kind() == "for" {
            found_for = true;
            continue;
        }

        if child.kind() != "type_identifier" && child.kind() != "generic_type" {
            continue;
        }

        let type_name = match child.kind() {
            "type_identifier" => get_text(&child, source).map(|s| s.to_string()),
            "generic_type" => child
                .child_by_field_name("type")
                .and_then(|n| get_text(&n, source))
                .map(|s| s.to_string()),
            _ => None,
        };

        if let Some(name) = type_name {
            if found_for {
                return Some(name);
            } else if first_type.is_none() {
                first_type = Some(name);
            }
        }
    }

    first_type
}

/// Get the trait being implemented
fn get_impl_trait(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let has_for = node.children(&mut node.walk()).any(|c| c.kind() == "for");
    if !has_for {
        return None;
    }

    for child in node.children(&mut node.walk()) {
        if child.kind() == "for" {
            break;
        }
        if child.kind() == "type_identifier" {
            return get_text(&child, source).map(|s| s.to_string());
        }
        if child.kind() == "generic_type" {
            return child
                .child_by_field_name("type")
                .and_then(|n| get_text(&n, source))
                .map(|s| s.to_string());
        }
    }

    None
}

fn extract_function_calls(
    node: &tree_sitter::Node,
    source: &str,
    caller_id: &str,
) -> Vec<FunctionCall> {
    let mut calls = Vec::new();
    let mut cursor = node.walk();
    extract_calls_recursive(&mut cursor, source, caller_id, &mut calls);
    calls
}

fn extract_calls_recursive(
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    caller_id: &str,
    calls: &mut Vec<FunctionCall>,
) {
    loop {
        let node = cursor.node();

        if node.kind() == "call_expression" {
            if let Some(func) = node.child_by_field_name("function") {
                let callee_name = match func.kind() {
                    "identifier" => get_text(&func, source).map(|s| s.to_string()),
                    "field_expression" => func
                        .child_by_field_name("field")
                        .and_then(|f| get_text(&f, source))
                        .map(|s| s.to_string()),
                    "scoped_identifier" => get_text(&func, source).map(|s| s.to_string()),
                    _ => None,
                };

                if let Some(callee) = callee_name {
                    calls.push(FunctionCall {
                        caller_id: caller_id.to_string(),
                        callee_name: callee,
                        line: node.start_position().row as u32 + 1,
                    });
                }
            }
        }

        if cursor.goto_first_child() {
            extract_calls_recursive(cursor, source, caller_id, calls);
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

fn extract_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "parameter" {
            let name = child
                .child_by_field_name("pattern")
                .and_then(|p| get_text(&p, source))
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

fn get_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "visibility_modifier" {
            let text = get_text(&child, source).unwrap_or_default();
            return match text {
                "pub" => Visibility::Public,
                s if s.starts_with("pub(crate)") => Visibility::Crate,
                s if s.starts_with("pub(super)") => Visibility::Super,
                s if s.starts_with("pub(in") => Visibility::InPath(
                    s.trim_start_matches("pub(in ")
                        .trim_end_matches(')')
                        .to_string(),
                ),
                _ => Visibility::Private,
            };
        }
    }
    Visibility::Private
}

fn has_modifier(node: &tree_sitter::Node, source: &str, modifier: &str) -> bool {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "function_modifiers" {
            for modifier_child in child.children(&mut child.walk()) {
                if modifier_child.kind() == modifier {
                    return true;
                }
                if let Some(text) = get_text(&modifier_child, source) {
                    if text == modifier {
                        return true;
                    }
                }
            }
            if let Some(text) = get_text(&child, source) {
                if text.contains(modifier) {
                    return true;
                }
            }
        }
        if child.kind() == modifier {
            return true;
        }
    }
    false
}

fn get_rust_docstring(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "line_comment" => {
                let text = get_text(&sibling, source)?;
                if text.starts_with("///") || text.starts_with("//!") {
                    doc_lines.push(text.trim_start_matches('/').trim().to_string());
                } else {
                    break;
                }
            }
            "block_comment" => {
                let text = get_text(&sibling, source)?;
                if text.starts_with("/**") || text.starts_with("/*!") {
                    doc_lines.push(
                        text.trim_start_matches("/**")
                            .trim_start_matches("/*!")
                            .trim_end_matches("*/")
                            .trim()
                            .to_string(),
                    );
                }
                break;
            }
            _ => break,
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

fn extract_derive_traits(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut traits = Vec::new();
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if sibling.kind() == "attribute_item" {
            if let Some(attr_text) = get_text(&sibling, source) {
                if attr_text.starts_with("#[derive(") {
                    if let Some(start) = attr_text.find('(') {
                        if let Some(end) = attr_text.rfind(')') {
                            let traits_str = &attr_text[start + 1..end];
                            for trait_name in traits_str.split(',') {
                                let name = trait_name.trim();
                                if !name.is_empty() {
                                    traits.push(name.to_string());
                                }
                            }
                        }
                    }
                }
            }
        } else if sibling.kind() != "line_comment" && sibling.kind() != "block_comment" {
            break;
        }
        prev = sibling.prev_sibling();
    }

    traits
}

fn extract_rust_type_parameters(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();

    let type_params_node = node
        .child_by_field_name("type_parameters")
        .or_else(|| find_child_by_kind(node, "type_parameters"));

    if let Some(type_params) = type_params_node {
        for param in type_params.children(&mut type_params.walk()) {
            match param.kind() {
                "type_parameter"
                | "constrained_type_parameter"
                | "lifetime_parameter"
                | "const_parameter" => {
                    if let Some(text) = get_text(&param, source) {
                        generics.push(text.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    generics
}
