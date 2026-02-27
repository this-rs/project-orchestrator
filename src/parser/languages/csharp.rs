//! C# language extractor
//!
//! Extractor for C# code including:
//! - Classes, interfaces, structs, records
//! - Methods and constructors
//! - Enums
//! - Using directives (imports)
//! - Namespace declarations
//! - XML doc comments

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract C# code structure
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
            "class_declaration" | "record_declaration" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                // Extract class members
                if let Some(body) = child.child_by_field_name("body") {
                    extract_type_body(&body, source, file_path, parsed)?;
                }
            }
            "struct_declaration" => {
                if let Some(s) = extract_struct(&child, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_type_body(&body, source, file_path, parsed)?;
                }
            }
            "interface_declaration" => {
                if let Some(iface) = extract_interface(&child, source, file_path) {
                    parsed.symbols.push(iface.name.clone());
                    parsed.traits.push(iface);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_type_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "using_directive" => {
                if let Some(import) = extract_using(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "namespace_declaration" | "file_scoped_namespace_declaration" => {
                // Recurse into namespace body
                extract_recursive(&child, source, file_path, parsed)?;
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
            "global_statement" => {
                // Top-level statements in C# 9+
                extract_recursive(&child, source, file_path, parsed)?;
            }
            _ => {
                // Recurse into other structures (declaration_list, etc.)
                extract_recursive(&child, source, file_path, parsed)?;
            }
        }
    }

    Ok(())
}

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_csharp_visibility(node, source);
    let docstring = get_xml_doc(node, source);
    let generics = extract_csharp_type_params(node, source);

    // Extract base class from base_list
    let (parent_class, interfaces) = extract_base_list(node, source);

    Some(StructNode {
        name,
        visibility,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
        parent_class,
        interfaces,
    })
}

fn extract_struct(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_csharp_visibility(node, source);
    let docstring = get_xml_doc(node, source);
    let generics = extract_csharp_type_params(node, source);
    let (_, interfaces) = extract_base_list(node, source);

    Some(StructNode {
        name,
        visibility,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
        parent_class: None, // C# structs can't inherit
        interfaces,
    })
}

fn extract_interface(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_csharp_visibility(node, source);
    let docstring = get_xml_doc(node, source);
    let generics = extract_csharp_type_params(node, source);

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
    let visibility = get_csharp_visibility(node, source);
    let docstring = get_xml_doc(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_member_declaration")
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
    let visibility = get_csharp_visibility(node, source);
    let docstring = get_xml_doc(node, source);
    let generics = extract_csharp_type_params(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_csharp_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("type")
        .or_else(|| node.child_by_field_name("returns"))
        .and_then(|t| get_text(&t, source))
        .map(|s| s.to_string());

    let is_async = has_csharp_modifier(node, source, "async");

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics,
        is_async,
        is_unsafe: has_csharp_modifier(node, source, "unsafe"),
        complexity: calculate_complexity(node),
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

fn extract_using(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let text = get_text(node, source)?;
    // Parse: using Namespace.Sub; | using static Namespace.Class; | using Alias = Namespace.Type;
    let path = text
        .trim_start_matches("using ")
        .trim_end_matches(';')
        .trim()
        .to_string();

    if path.is_empty() {
        return None;
    }

    Some(ImportNode {
        path,
        alias: None,
        items: vec![],
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_type_body(
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
            "class_declaration" | "record_declaration" => {
                // Nested class
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                if let Some(inner_body) = child.child_by_field_name("body") {
                    extract_type_body(&inner_body, source, file_path, parsed)?;
                }
            }
            "struct_declaration" => {
                if let Some(s) = extract_struct(&child, source, file_path) {
                    parsed.symbols.push(s.name.clone());
                    parsed.structs.push(s);
                }
                if let Some(inner_body) = child.child_by_field_name("body") {
                    extract_type_body(&inner_body, source, file_path, parsed)?;
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
            "property_declaration" | "field_declaration" | "event_declaration" => {
                // Skip fields/properties for now — could be added later
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract base types from the base_list (C# inheritance syntax)
///
/// In C#, `class Foo : Bar, IDisposable, IComparable` —
/// the first type is the base class (if it's not an interface),
/// and the rest are interfaces. Since we can't reliably distinguish
/// classes from interfaces at the AST level, we use a heuristic:
/// names starting with 'I' followed by uppercase = interface.
fn extract_base_list(node: &tree_sitter::Node, source: &str) -> (Option<String>, Vec<String>) {
    let mut parent_class = None;
    let mut interfaces = Vec::new();

    let base_list = node.child_by_field_name("bases").or_else(|| {
        node.children(&mut node.walk())
            .find(|c| c.kind() == "base_list")
    });

    if let Some(bases) = base_list {
        let mut first = true;
        for child in bases.children(&mut bases.walk()) {
            let kind = child.kind();
            if kind == "," || kind == ":" {
                continue;
            }
            if let Some(name) = get_text(&child, source) {
                let name = name.trim().to_string();
                if name.is_empty() {
                    continue;
                }
                // Heuristic: interface names start with 'I' + uppercase
                let is_interface = name.starts_with('I')
                    && name.len() > 1
                    && name.chars().nth(1).map_or(false, |c| c.is_uppercase());

                if first && !is_interface {
                    parent_class = Some(name);
                    first = false;
                } else {
                    interfaces.push(name);
                    first = false;
                }
            }
        }
    }

    (parent_class, interfaces)
}

fn extract_csharp_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "parameter" {
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

fn extract_csharp_type_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();

    let type_params = node
        .child_by_field_name("type_parameters")
        .or_else(|| find_child_by_kind(node, "type_parameter_list"));

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

fn get_csharp_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "modifier" {
            if let Some(text) = get_text(&child, source) {
                match text {
                    "public" => return Visibility::Public,
                    "protected" => return Visibility::Crate,
                    "private" => return Visibility::Private,
                    "internal" => return Visibility::Crate,
                    _ => {}
                }
            }
        }
    }

    // C# default is private for class members, internal for types
    Visibility::Private
}

fn has_csharp_modifier(node: &tree_sitter::Node, source: &str, modifier: &str) -> bool {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "modifier" {
            if let Some(text) = get_text(&child, source) {
                if text == modifier {
                    return true;
                }
            }
        }
    }
    false
}

fn get_xml_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    // Collect consecutive XML doc comment lines (///) above the node
    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" => {
                if let Some(text) = get_text(&sibling, source) {
                    if text.starts_with("///") {
                        let content = text.trim_start_matches("///").trim();
                        // Strip XML tags for clean docstring
                        let clean = content
                            .replace("<summary>", "")
                            .replace("</summary>", "")
                            .replace("<param name=", "param: ")
                            .replace("</param>", "")
                            .replace("<returns>", "Returns: ")
                            .replace("</returns>", "")
                            .replace("<remarks>", "")
                            .replace("</remarks>", "");
                        let clean = clean.trim().to_string();
                        if !clean.is_empty() && !clean.starts_with('>') {
                            doc_lines.push(clean);
                        }
                    } else {
                        break; // Not an XML doc comment, stop
                    }
                }
            }
            "attribute_list" => {
                // Skip [Attributes] above declarations
            }
            _ => break,
        }
        prev = sibling.prev_sibling();
    }

    if doc_lines.is_empty() {
        return None;
    }

    doc_lines.reverse();
    Some(doc_lines.join("\n"))
}
