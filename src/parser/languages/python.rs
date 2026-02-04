//! Python language extractor
//!
//! Enriched extractor for Python code including:
//! - Functions (regular, async, generators)
//! - Classes with inheritance
//! - Decorators
//! - Type hints
//! - Docstrings (triple-quoted)

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Python code structure
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
    inside_class: bool,
) -> Result<()> {
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_definition" => {
                if let Some(func) = extract_function(&child, source, file_path, inside_class) {
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
                // Extract methods from class body
                if let Some(body) = child.child_by_field_name("body") {
                    extract_recursive(&body, source, file_path, parsed, true)?;
                }
            }
            "decorated_definition" => {
                // Handle decorated functions/classes
                extract_decorated(&child, source, file_path, parsed, inside_class)?;
            }
            "import_statement" => {
                if let Some(import) = extract_import(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            "import_from_statement" => {
                if let Some(import) = extract_import_from(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            _ => {
                // Recurse into blocks
                if child.kind() == "block" || child.kind() == "module" {
                    extract_recursive(&child, source, file_path, parsed, inside_class)?;
                }
            }
        }
    }

    Ok(())
}

fn extract_function(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    inside_class: bool,
) -> Option<FunctionNode> {
    let name = get_field_text(node, "name", source)?;

    // Determine visibility from name
    let visibility = if name.starts_with("__") && !name.ends_with("__") {
        Visibility::Private // Name mangled
    } else if name.starts_with('_') {
        Visibility::Crate // Protected/internal
    } else {
        Visibility::Public
    };

    // Check for async
    let is_async = node.children(&mut node.walk()).any(|c| c.kind() == "async");

    // Extract parameters
    let params = node
        .child_by_field_name("parameters")
        .map(|p| extract_py_params(&p, source, inside_class))
        .unwrap_or_default();

    // Extract return type annotation
    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.trim_start_matches("->").trim().to_string());

    // Extract docstring
    let docstring = get_py_docstring(node, source);

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics: vec![],
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

    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };

    // Extract base classes as "generics" (representing inheritance)
    let generics: Vec<String> = node
        .child_by_field_name("superclasses")
        .or_else(|| find_child_by_kind(node, "argument_list"))
        .map(|args| {
            args.children(&mut args.walk())
                .filter(|c| c.kind() == "identifier" || c.kind() == "attribute")
                .filter_map(|c| get_text(&c, source).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let docstring = get_py_docstring(node, source);

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

fn extract_decorated(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
    inside_class: bool,
) -> Result<()> {
    // Extract decorators
    let decorators: Vec<String> = node
        .children(&mut node.walk())
        .filter(|c| c.kind() == "decorator")
        .filter_map(|d| get_text(&d, source).map(|s| s.to_string()))
        .collect();

    // Find the definition
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "function_definition" => {
                if let Some(func) = extract_function(&child, source, file_path, inside_class) {
                    // Check decorators for special meanings
                    for decorator in &decorators {
                        if decorator.contains("staticmethod") || decorator.contains("classmethod") {
                            // Keep visibility
                        } else if decorator.contains("property") {
                            // Could mark as property
                        }
                    }

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
                if let Some(body) = child.child_by_field_name("body") {
                    extract_recursive(&body, source, file_path, parsed, true)?;
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn extract_import(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let text = get_text(node, source)?;

    // Extract module names
    let items: Vec<String> = node
        .children(&mut node.walk())
        .filter(|c| c.kind() == "dotted_name" || c.kind() == "aliased_import")
        .filter_map(|c| get_text(&c, source).map(|s| s.to_string()))
        .collect();

    Some(ImportNode {
        path: text.trim_start_matches("import ").trim().to_string(),
        alias: None,
        items,
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_import_from(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<ImportNode> {
    let module = node
        .child_by_field_name("module_name")
        .and_then(|m| get_text(&m, source))
        .map(|s| s.to_string())
        .unwrap_or_default();

    // Extract imported items
    let items: Vec<String> = node
        .children(&mut node.walk())
        .filter(|c| {
            c.kind() == "dotted_name"
                || c.kind() == "aliased_import"
                || c.kind() == "import_from_specifier"
        })
        .filter_map(|c| {
            if c.kind() == "import_from_specifier" {
                c.child_by_field_name("name")
                    .and_then(|n| get_text(&n, source))
                    .map(|s| s.to_string())
            } else {
                get_text(&c, source).map(|s| s.to_string())
            }
        })
        .filter(|s| s != &module)
        .collect();

    Some(ImportNode {
        path: module,
        alias: None,
        items,
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_py_params(
    node: &tree_sitter::Node,
    source: &str,
    inside_class: bool,
) -> Vec<Parameter> {
    let mut params = Vec::new();
    let mut is_first = true;

    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "identifier" => {
                let name = get_text(&child, source).unwrap_or("_").to_string();

                // Skip 'self' and 'cls' in methods
                if inside_class && is_first && (name == "self" || name == "cls") {
                    is_first = false;
                    continue;
                }
                is_first = false;

                params.push(Parameter {
                    name,
                    type_name: None,
                });
            }
            "typed_parameter" | "default_parameter" | "typed_default_parameter" => {
                let name = child
                    .child_by_field_name("name")
                    .or_else(|| find_child_by_kind(&child, "identifier"))
                    .and_then(|n| get_text(&n, source))
                    .unwrap_or("_")
                    .to_string();

                // Skip 'self' and 'cls' in methods
                if inside_class && is_first && (name == "self" || name == "cls") {
                    is_first = false;
                    continue;
                }
                is_first = false;

                let type_name = child
                    .child_by_field_name("type")
                    .and_then(|t| get_text(&t, source))
                    .map(|s| s.to_string());

                params.push(Parameter { name, type_name });
            }
            "list_splat_pattern" | "dictionary_splat_pattern" => {
                // *args, **kwargs
                let name = get_text(&child, source).unwrap_or("_").to_string();
                params.push(Parameter {
                    name,
                    type_name: None,
                });
            }
            _ => {}
        }
    }

    params
}

fn get_py_docstring(node: &tree_sitter::Node, source: &str) -> Option<String> {
    // Python docstrings are the first statement in a function/class body
    let body = node.child_by_field_name("body")?;

    // Get the first child of the block
    let first_stmt = body.child(0)?;

    // Check if it's an expression statement containing a string
    if first_stmt.kind() == "expression_statement" {
        if let Some(string_node) = first_stmt.child(0) {
            if string_node.kind() == "string" {
                let text = get_text(&string_node, source)?;
                // Remove triple quotes
                return Some(
                    text.trim_start_matches("\"\"\"")
                        .trim_start_matches("'''")
                        .trim_end_matches("\"\"\"")
                        .trim_end_matches("'''")
                        .trim()
                        .to_string(),
                );
            }
        }
    }

    None
}
