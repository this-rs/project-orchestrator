//! C language extractor
//!
//! Extractor for C code including:
//! - Functions
//! - Structs, unions, enums
//! - Typedefs
//! - Macros/defines
//! - Header includes

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract C code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        match node.kind() {
            "function_definition" => {
                if let Some(func) = extract_function(&node, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&node, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "declaration" => {
                // Function declarations/prototypes
                if let Some(func) = extract_function_declaration(&node, source, file_path) {
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "struct_specifier" | "union_specifier" => {
                if let Some(s) = extract_struct(&node, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
            }
            "enum_specifier" => {
                if let Some(e) = extract_enum(&node, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "type_definition" => {
                extract_typedef(&node, source, file_path, parsed)?;
            }
            "preproc_include" => {
                if let Some(import) = extract_include(&node, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "preproc_def" | "preproc_function_def" => {
                // Macros - treat as functions
                if let Some(func) = extract_macro(&node, source, file_path) {
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
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
    let declarator = node.child_by_field_name("declarator")?;
    let name = extract_function_name(&declarator, source)?;

    let docstring = get_c_doc(node, source);

    let params = extract_c_params(&declarator, source);

    let return_type = node
        .child_by_field_name("type")
        .and_then(|t| get_text(&t, source))
        .map(|s| s.to_string());

    // Check for static (file-local)
    let visibility = if has_storage_class(node, source, "static") {
        Visibility::Private
    } else {
        Visibility::Public
    };

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics: vec![],
        is_async: false,
        is_unsafe: true, // C is inherently unsafe
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_function_declaration(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<FunctionNode> {
    // Look for function declarator in the declaration
    let declarator = node.child_by_field_name("declarator")?;

    // Check if this is actually a function declaration
    if declarator.kind() != "function_declarator"
        && !has_child_kind(&declarator, "function_declarator")
    {
        return None;
    }

    let name = extract_function_name(&declarator, source)?;
    let docstring = get_c_doc(node, source);
    let params = extract_c_params(&declarator, source);

    let return_type = node
        .child_by_field_name("type")
        .and_then(|t| get_text(&t, source))
        .map(|s| s.to_string());

    let visibility = if has_storage_class(node, source, "static") {
        Visibility::Private
    } else {
        // In C, non-static functions are effectively public (extern is default)
        Visibility::Public
    };

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics: vec![],
        is_async: false,
        is_unsafe: true,
        complexity: 1, // Just a declaration
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_function_name(declarator: &tree_sitter::Node, source: &str) -> Option<String> {
    // Handle nested declarators (e.g., pointer to function)
    let mut current = *declarator;

    loop {
        match current.kind() {
            "function_declarator" => {
                if let Some(inner) = current.child_by_field_name("declarator") {
                    current = inner;
                } else {
                    break;
                }
            }
            "pointer_declarator" => {
                if let Some(inner) = current.child_by_field_name("declarator") {
                    current = inner;
                } else {
                    break;
                }
            }
            "identifier" => {
                return get_text(&current, source).map(|s| s.to_string());
            }
            "parenthesized_declarator" => {
                if let Some(inner) = current.child(1) {
                    current = inner;
                } else {
                    break;
                }
            }
            _ => {
                // Try to find identifier directly
                if let Some(ident) = find_child_by_kind(&current, "identifier") {
                    return get_text(&ident, source).map(|s| s.to_string());
                }
                break;
            }
        }
    }

    None
}

fn extract_struct(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_c_doc(node, source);

    Some(StructNode {
        name,
        visibility: Visibility::Public,
        generics: vec![],
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_enum(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_c_doc(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enumerator")
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

fn extract_typedef(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    // Typedef can define struct, enum, or type alias
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "struct_specifier" | "union_specifier" => {
                // typedef struct { ... } Name;
                if let Some(s) = extract_struct(&child, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
            }
            "enum_specifier" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "type_identifier" => {
                // The typedef name
                if let Some(name) = get_text(&child, source) {
                    parsed.symbols.push(name.to_string());
                    parsed.structs.push(StructNode {
                        name: name.to_string(),
                        visibility: Visibility::Public,
                        generics: vec![],
                        file_path: file_path.to_string(),
                        line_start: node.start_position().row as u32 + 1,
                        line_end: node.end_position().row as u32 + 1,
                        docstring: get_c_doc(node, source),
                    });
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_include(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let path = node
        .child_by_field_name("path")
        .and_then(|p| get_text(&p, source))
        .map(|s| {
            s.trim_matches('"')
                .trim_matches('<')
                .trim_matches('>')
                .to_string()
        })?;

    Some(ImportNode {
        path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_macro(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;

    Some(FunctionNode {
        name,
        visibility: Visibility::Public,
        params: vec![],
        return_type: None,
        generics: vec![],
        is_async: false,
        is_unsafe: true,
        complexity: 1,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring: get_c_doc(node, source),
    })
}

fn extract_c_params(declarator: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    // Find the function_declarator
    let func_decl = if declarator.kind() == "function_declarator" {
        Some(*declarator)
    } else {
        find_child_by_kind(declarator, "function_declarator")
    };

    if let Some(func_decl) = func_decl {
        if let Some(param_list) = func_decl.child_by_field_name("parameters") {
            for child in param_list.children(&mut param_list.walk()) {
                if child.kind() == "parameter_declaration" {
                    let type_name = child
                        .child_by_field_name("type")
                        .and_then(|t| get_text(&t, source))
                        .map(|s| s.to_string());

                    let name = child
                        .child_by_field_name("declarator")
                        .and_then(|d| {
                            if d.kind() == "identifier" {
                                get_text(&d, source)
                            } else {
                                find_child_by_kind(&d, "identifier")
                                    .and_then(|i| get_text(&i, source))
                            }
                        })
                        .unwrap_or("_")
                        .to_string();

                    params.push(Parameter { name, type_name });
                }
            }
        }
    }

    params
}

fn has_storage_class(node: &tree_sitter::Node, source: &str, class: &str) -> bool {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "storage_class_specifier" {
            if let Some(text) = get_text(&child, source) {
                if text == class {
                    return true;
                }
            }
        }
    }
    false
}

fn get_c_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = get_text(&sibling, source)?;
            // Doxygen-style comments
            if text.starts_with("/**") || text.starts_with("/*!") || text.starts_with("///") {
                if text.starts_with("/**") || text.starts_with("/*!") {
                    doc_lines.push(
                        text.trim_start_matches("/**")
                            .trim_start_matches("/*!")
                            .trim_end_matches("*/")
                            .lines()
                            .map(|l| l.trim().trim_start_matches('*').trim())
                            .filter(|l| !l.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                    break;
                } else {
                    doc_lines.push(text.trim_start_matches('/').trim().to_string());
                }
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
