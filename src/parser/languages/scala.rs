//! Scala language extractor
//!
//! Extractor for Scala code including:
//! - Classes, case classes, traits, objects
//! - Methods (def) and vals
//! - Enums (Scala 3)
//! - Import declarations (standard, wildcard, selective, rename)
//! - Package declarations

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Scala code structure
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
            "class_definition" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                // Extract class body
                if let Some(body) = child.child_by_field_name("body") {
                    extract_body(&body, source, file_path, parsed)?;
                }
            }
            "trait_definition" => {
                if let Some(t) = extract_trait(&child, source, file_path) {
                    parsed.symbols.push(t.name.clone());
                    parsed.traits.push(t);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_body(&body, source, file_path, parsed)?;
                }
            }
            "object_definition" => {
                // Objects are treated as structs (singleton)
                if let Some(obj) = extract_object(&child, source, file_path) {
                    parsed.symbols.push(obj.name.clone());
                    parsed.structs.push(obj);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_definition" => {
                // Scala 3 enums
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "import_declaration" => {
                if let Some(import) = extract_import(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "function_definition" | "val_definition" | "var_definition" => {
                if let Some(func) = extract_method(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "package_clause" | "package_object" => {
                // Recurse into package body
                extract_recursive(&child, source, file_path, parsed)?;
            }
            _ => {
                extract_recursive(&child, source, file_path, parsed)?;
            }
        }
    }

    Ok(())
}

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_scala_visibility(node, source);
    let docstring = get_scaladoc(node, source);
    let generics = extract_scala_type_params(node, source);

    // Check for "case" keyword (case class)
    let text = get_text(node, source).unwrap_or_default();
    let is_case = text.starts_with("case ");

    // Extract extends clause
    let (parent_class, interfaces) = extract_extends(node, source);

    Some(StructNode {
        name: if is_case {
            name // case classes are still named normally
        } else {
            name
        },
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

fn extract_trait(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_scala_visibility(node, source);
    let docstring = get_scaladoc(node, source);
    let generics = extract_scala_type_params(node, source);

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
    let visibility = get_scala_visibility(node, source);
    let docstring = get_scaladoc(node, source);

    let (parent_class, interfaces) = extract_extends(node, source);

    Some(StructNode {
        name,
        visibility,
        generics: vec![],
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
        parent_class,
        interfaces,
    })
}

fn extract_enum(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<EnumNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_scala_visibility(node, source);
    let docstring = get_scaladoc(node, source);

    // Extract enum cases from body
    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_case_definitions" || c.kind() == "simple_enum_case")
                .flat_map(|c| {
                    // Recurse into case definitions
                    c.children(&mut c.walk())
                        .filter_map(|v| get_field_text(&v, "name", source))
                        .collect::<Vec<_>>()
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

fn extract_method(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = get_scala_visibility(node, source);
    let docstring = get_scaladoc(node, source);
    let generics = extract_scala_type_params(node, source);

    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_scala_params(&p, source))
        .unwrap_or_default();

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|t| get_text(&t, source))
        .map(|s| s.trim_start_matches(':').trim().to_string());

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

fn extract_import(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let text = get_text(node, source)?;
    let path = text.trim_start_matches("import ").trim().to_string();

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

fn extract_body(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in body.children(&mut body.walk()) {
        match child.kind() {
            "function_definition" | "val_definition" | "var_definition" => {
                if let Some(func) = extract_method(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    let calls = extract_calls_from_node(&child, source, &func_id);
                    parsed.function_calls.extend(calls);
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
            "class_definition" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                if let Some(inner) = child.child_by_field_name("body") {
                    extract_body(&inner, source, file_path, parsed)?;
                }
            }
            "trait_definition" => {
                if let Some(t) = extract_trait(&child, source, file_path) {
                    parsed.symbols.push(t.name.clone());
                    parsed.traits.push(t);
                }
            }
            "object_definition" => {
                if let Some(obj) = extract_object(&child, source, file_path) {
                    parsed.symbols.push(obj.name.clone());
                    parsed.structs.push(obj);
                }
                if let Some(inner) = child.child_by_field_name("body") {
                    extract_body(&inner, source, file_path, parsed)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract extends/with clause for Scala classes and objects.
///
/// `class Foo extends Bar with Baz with Qux` →
/// parent_class = Some("Bar"), interfaces = ["Baz", "Qux"]
fn extract_extends(node: &tree_sitter::Node, source: &str) -> (Option<String>, Vec<String>) {
    let mut parent_class = None;
    let mut interfaces = Vec::new();

    let extends_clause = node.child_by_field_name("extend").or_else(|| {
        node.children(&mut node.walk()).find(|c| {
            c.kind() == "extends_clause"
                || c.kind() == "template_body"
                || c.kind() == "class_parents"
        })
    });

    if let Some(clause) = extends_clause {
        let text = get_text(&clause, source).unwrap_or_default();
        let text = text.trim_start_matches("extends").trim();

        // Split by " with " (Scala 2) or ", " (Scala 3)
        let parts: Vec<&str> = text
            .split(" with ")
            .flat_map(|p| p.split(','))
            .map(|p| p.trim())
            .filter(|p| !p.is_empty() && !p.starts_with('{'))
            .collect();

        for (i, part) in parts.iter().enumerate() {
            // Extract just the type name (strip constructor args etc.)
            let type_name = part
                .split('(')
                .next()
                .unwrap_or(part)
                .split('[')
                .next()
                .unwrap_or(part)
                .trim()
                .to_string();

            if type_name.is_empty() {
                continue;
            }

            if i == 0 {
                parent_class = Some(type_name);
            } else {
                interfaces.push(type_name);
            }
        }
    }

    (parent_class, interfaces)
}

fn extract_scala_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in node.children(&mut node.walk()) {
        if child.kind() == "parameter" || child.kind() == "class_parameter" {
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

fn extract_scala_type_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();

    let type_params = node
        .child_by_field_name("type_parameters")
        .or_else(|| find_child_by_kind(node, "type_parameters"));

    if let Some(params) = type_params {
        for param in params.children(&mut params.walk()) {
            if param.kind() == "type_parameter" || param.kind() == "identifier" {
                if let Some(text) = get_text(&param, source) {
                    generics.push(text.to_string());
                }
            }
        }
    }

    generics
}

fn get_scala_visibility(node: &tree_sitter::Node, source: &str) -> Visibility {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "access_modifier" || child.kind() == "modifiers" {
            if let Some(text) = get_text(&child, source) {
                if text.contains("private") {
                    return Visibility::Private;
                } else if text.contains("protected") {
                    return Visibility::Crate;
                }
            }
        }
    }
    // Scala default is public
    Visibility::Public
}

fn get_scaladoc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" | "block_comment" => {
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
            "annotation" => {
                // Skip annotations
            }
            _ => break,
        }
        prev = sibling.prev_sibling();
    }

    None
}
