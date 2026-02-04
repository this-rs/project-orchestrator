//! C++ language extractor
//!
//! Extractor for C++ code including:
//! - Classes and structs
//! - Functions and methods
//! - Templates
//! - Namespaces
//! - Enums (including enum class)

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract C++ code structure
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
            "declaration" => {
                // Function declarations
                if let Some(func) = extract_function_declaration(&child, source, file_path) {
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_specifier" | "struct_specifier" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                // Extract members
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_specifier" => {
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
            "template_declaration" => {
                // Template function/class
                extract_template(&child, source, file_path, parsed)?;
            }
            "preproc_include" => {
                if let Some(import) = extract_include(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "using_declaration" => {
                if let Some(import) = extract_using(&child, source, file_path) {
                    parsed.imports.push(import);
                }
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
    let declarator = node.child_by_field_name("declarator")?;
    let name = extract_function_name(&declarator, source)?;

    let docstring = get_cpp_doc(node, source);
    let params = extract_cpp_params(&declarator, source);
    let generics = extract_template_params_from_parent(node, source);

    let return_type = node
        .child_by_field_name("type")
        .and_then(|t| get_text(&t, source))
        .map(|s| s.to_string());

    let visibility = if has_specifier(node, source, "static") {
        Visibility::Private
    } else {
        Visibility::Public
    };

    let is_async = has_specifier(node, source, "virtual");

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async, // Using for virtual
        is_unsafe: false,
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
    let declarator = node.child_by_field_name("declarator")?;

    // Check if this is a function declaration
    if declarator.kind() != "function_declarator"
        && !has_child_kind(&declarator, "function_declarator")
    {
        return None;
    }

    let name = extract_function_name(&declarator, source)?;
    let docstring = get_cpp_doc(node, source);
    let params = extract_cpp_params(&declarator, source);

    let return_type = node
        .child_by_field_name("type")
        .and_then(|t| get_text(&t, source))
        .map(|s| s.to_string());

    Some(FunctionNode {
        name,
        visibility: Visibility::Public,
        params,
        return_type,
        generics: vec![],
        is_async: false,
        is_unsafe: false,
        complexity: 1,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_function_name(declarator: &tree_sitter::Node, source: &str) -> Option<String> {
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
            "pointer_declarator" | "reference_declarator" => {
                if let Some(inner) = current.child_by_field_name("declarator") {
                    current = inner;
                } else {
                    break;
                }
            }
            "identifier" => {
                return get_text(&current, source).map(|s| s.to_string());
            }
            "qualified_identifier" | "scoped_identifier" => {
                // Get the full qualified name
                return get_text(&current, source).map(|s| s.to_string());
            }
            "destructor_name" => {
                return get_text(&current, source).map(|s| s.to_string());
            }
            "operator_name" => {
                return get_text(&current, source).map(|s| format!("operator{}", s));
            }
            "parenthesized_declarator" => {
                if let Some(inner) = current.child(1) {
                    current = inner;
                } else {
                    break;
                }
            }
            "field_identifier" => {
                return get_text(&current, source).map(|s| s.to_string());
            }
            _ => {
                if let Some(ident) = find_child_by_kind(&current, "identifier") {
                    return get_text(&ident, source).map(|s| s.to_string());
                }
                if let Some(ident) = find_child_by_kind(&current, "field_identifier") {
                    return get_text(&ident, source).map(|s| s.to_string());
                }
                break;
            }
        }
    }

    None
}

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_cpp_doc(node, source);
    let generics = extract_template_params_from_parent(node, source);

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

fn extract_enum(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let name = get_field_text(node, "name", source)?;
    let docstring = get_cpp_doc(node, source);

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

fn extract_class_body(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let mut current_visibility = Visibility::Private; // Default for class

    for child in body.children(&mut body.walk()) {
        match child.kind() {
            "access_specifier" => {
                if let Some(text) = get_text(&child, source) {
                    current_visibility = match text.trim_end_matches(':') {
                        "public" => Visibility::Public,
                        "protected" => Visibility::Crate,
                        "private" => Visibility::Private,
                        _ => Visibility::Private,
                    };
                }
            }
            "function_definition" => {
                if let Some(mut func) = extract_function(&child, source, file_path) {
                    func.visibility = current_visibility.clone();
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "declaration" => {
                if let Some(mut func) = extract_function_declaration(&child, source, file_path) {
                    func.visibility = current_visibility.clone();
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_specifier" | "struct_specifier" => {
                // Nested class
                if let Some(mut class) = extract_class(&child, source, file_path) {
                    class.visibility = current_visibility.clone();
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
            }
            "enum_specifier" => {
                if let Some(mut e) = extract_enum(&child, source, file_path) {
                    e.visibility = current_visibility.clone();
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_template(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let template_params = extract_template_params(node, source);

    // Find the templated declaration
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "function_definition" => {
                if let Some(mut func) = extract_function(&child, source, file_path) {
                    func.generics = template_params.clone();
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_specifier" | "struct_specifier" => {
                if let Some(mut class) = extract_class(&child, source, file_path) {
                    class.generics = template_params.clone();
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "declaration" => {
                if let Some(mut func) = extract_function_declaration(&child, source, file_path) {
                    func.generics = template_params.clone();
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
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

fn extract_using(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let text = get_text(node, source)?;
    let path = text
        .trim_start_matches("using ")
        .trim_start_matches("namespace ")
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

fn extract_cpp_params(declarator: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    let func_decl = if declarator.kind() == "function_declarator" {
        Some(*declarator)
    } else {
        find_child_by_kind(declarator, "function_declarator")
    };

    if let Some(func_decl) = func_decl {
        if let Some(param_list) = func_decl.child_by_field_name("parameters") {
            for child in param_list.children(&mut param_list.walk()) {
                if child.kind() == "parameter_declaration"
                    || child.kind() == "optional_parameter_declaration"
                {
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

fn extract_template_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut params = Vec::new();

    if let Some(param_list) = node.child_by_field_name("parameters") {
        for child in param_list.children(&mut param_list.walk()) {
            match child.kind() {
                "type_parameter_declaration" | "template_type_parameter" => {
                    if let Some(name) = get_field_text(&child, "name", source) {
                        params.push(name);
                    } else if let Some(text) = get_text(&child, source) {
                        // Extract just the parameter name
                        let name = text.split_whitespace().last().unwrap_or(text).to_string();
                        params.push(name);
                    }
                }
                "parameter_declaration" => {
                    // Non-type template parameter
                    if let Some(text) = get_text(&child, source) {
                        params.push(text.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    params
}

fn extract_template_params_from_parent(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    // Check if parent is a template_declaration
    if let Some(parent) = node.parent() {
        if parent.kind() == "template_declaration" {
            return extract_template_params(&parent, source);
        }
    }
    vec![]
}

fn has_specifier(node: &tree_sitter::Node, source: &str, specifier: &str) -> bool {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "storage_class_specifier"
            || child.kind() == "type_qualifier"
            || child.kind() == "virtual_function_specifier"
        {
            if let Some(text) = get_text(&child, source) {
                if text == specifier {
                    return true;
                }
            }
        }
    }
    false
}

fn get_cpp_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = get_text(&sibling, source)?;
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
