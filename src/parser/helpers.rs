//! Common helper functions for code parsing across all languages

use crate::neo4j::models::*;

/// Get the text content of a node
pub fn get_text<'a>(node: &tree_sitter::Node<'a>, source: &'a str) -> Option<&'a str> {
    node.utf8_text(source.as_bytes()).ok()
}

/// Get text from a named field in a node
pub fn get_field_text<'a>(
    node: &tree_sitter::Node<'a>,
    field: &str,
    source: &'a str,
) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|n| get_text(&n, source))
        .map(|s| s.to_string())
}

/// Calculate cyclomatic complexity for a function body
pub fn calculate_complexity(node: &tree_sitter::Node) -> u32 {
    let mut complexity = 1u32;
    let mut cursor = node.walk();

    fn count_branches(cursor: &mut tree_sitter::TreeCursor, complexity: &mut u32) {
        loop {
            let node = cursor.node();
            match node.kind() {
                // Control flow
                "if_expression"
                | "if_statement"
                | "while_expression"
                | "while_statement"
                | "for_expression"
                | "for_statement"
                | "for_in_statement"
                | "match_arm"
                | "loop_expression"
                | "case_clause"
                | "switch_case"
                | "catch_clause"
                | "except_clause"
                | "elif_clause"
                | "else_clause"
                | "?"
                | "ternary_expression"
                | "conditional_expression" => {
                    *complexity += 1;
                }
                // Logical operators
                "binary_expression" | "boolean_operator" => {
                    if let Some(op) = node.child_by_field_name("operator") {
                        let kind = op.kind();
                        if kind == "&&" || kind == "||" || kind == "and" || kind == "or" {
                            *complexity += 1;
                        }
                    }
                }
                _ => {}
            }

            if cursor.goto_first_child() {
                count_branches(cursor, complexity);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    count_branches(&mut cursor, &mut complexity);
    complexity
}

/// Extract preceding docstring/comment for a node
pub fn get_docstring(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        match sibling.kind() {
            // Rust-style line comments
            "line_comment" => {
                let text = get_text(&sibling, source)?;
                if text.starts_with("///") || text.starts_with("//!") {
                    doc_lines.push(text.trim_start_matches('/').trim().to_string());
                } else {
                    break;
                }
            }
            // Python-style
            "expression_statement" => {
                // Check if it's a string (docstring)
                if let Some(child) = sibling.child(0) {
                    if child.kind() == "string" {
                        let text = get_text(&child, source)?;
                        let trimmed = text
                            .trim_start_matches("\"\"\"")
                            .trim_start_matches("'''")
                            .trim_end_matches("\"\"\"")
                            .trim_end_matches("'''")
                            .trim();
                        doc_lines.push(trimmed.to_string());
                        break;
                    }
                }
                break;
            }
            // Block comments (Rust, Java, C++, etc.)
            "block_comment" | "comment" => {
                let text = get_text(&sibling, source)?;
                // JavaDoc/JSDoc style
                if text.starts_with("/**") {
                    doc_lines.push(
                        text.trim_start_matches("/**")
                            .trim_end_matches("*/")
                            .lines()
                            .map(|l| l.trim().trim_start_matches('*').trim())
                            .filter(|l| !l.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                    break;
                }
                // Rust-style block
                if text.starts_with("/*!") {
                    doc_lines.push(
                        text.trim_start_matches("/*!")
                            .trim_end_matches("*/")
                            .trim()
                            .to_string(),
                    );
                    break;
                }
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

/// Common visibility determination based on naming conventions
pub fn visibility_from_name(name: &str) -> Visibility {
    // Python/JavaScript convention: underscore prefix = private
    if name.starts_with('_') {
        Visibility::Private
    } else {
        // Go convention: uppercase first letter = public, but we default to Public
        Visibility::Public
    }
}

/// Extract function calls from a node recursively
pub fn extract_calls_from_node(
    node: &tree_sitter::Node,
    source: &str,
    caller_id: &str,
) -> Vec<super::FunctionCall> {
    let mut calls = Vec::new();
    let mut cursor = node.walk();
    extract_calls_recursive(&mut cursor, source, caller_id, &mut calls);
    calls
}

fn extract_calls_recursive(
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    caller_id: &str,
    calls: &mut Vec<super::FunctionCall>,
) {
    loop {
        let node = cursor.node();

        // Handle different call expression types across languages
        match node.kind() {
            "call_expression" | "call" | "function_call_expression" | "method_call_expression" => {
                if let Some(callee_name) = extract_callee_name(&node, source) {
                    calls.push(super::FunctionCall {
                        caller_id: caller_id.to_string(),
                        callee_name,
                        line: node.start_position().row as u32 + 1,
                    });
                }
            }
            _ => {}
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

/// Extract the callee name from a call expression
fn extract_callee_name(node: &tree_sitter::Node, source: &str) -> Option<String> {
    // Try different field names for the function being called
    let func = node.child_by_field_name("function").or_else(|| {
        node.child_by_field_name("method")
            .or_else(|| node.child_by_field_name("name"))
    })?;

    match func.kind() {
        "identifier" | "name" => get_text(&func, source).map(|s| s.to_string()),
        "field_expression" | "attribute" | "member_expression" | "selector_expression" => {
            // e.g., obj.method()
            func.child_by_field_name("field")
                .or_else(|| func.child_by_field_name("attribute"))
                .or_else(|| func.child_by_field_name("property"))
                .and_then(|f| get_text(&f, source))
                .map(|s| s.to_string())
        }
        "scoped_identifier" | "qualified_identifier" => {
            // e.g., Module::function()
            get_text(&func, source).map(|s| s.to_string())
        }
        _ => get_text(&func, source).map(|s| s.to_string()),
    }
}

/// Extract parameters in a generic way
pub fn extract_params_generic(params_node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    for child in params_node.children(&mut params_node.walk()) {
        match child.kind() {
            "parameter"
            | "formal_parameter"
            | "required_parameter"
            | "optional_parameter"
            | "rest_parameter"
            | "parameter_declaration"
            | "simple_parameter"
            | "typed_parameter"
            | "default_parameter" => {
                let name = child
                    .child_by_field_name("name")
                    .or_else(|| child.child_by_field_name("pattern"))
                    .and_then(|n| get_text(&n, source))
                    .unwrap_or("_")
                    .to_string();

                let type_name = child
                    .child_by_field_name("type")
                    .and_then(|t| get_text(&t, source))
                    .map(|s| s.to_string());

                params.push(Parameter { name, type_name });
            }
            // Skip commas, parentheses, etc.
            _ => {}
        }
    }

    params
}

/// Find a child node by kind
pub fn find_child_by_kind<'a>(
    node: &tree_sitter::Node<'a>,
    kind: &str,
) -> Option<tree_sitter::Node<'a>> {
    node.children(&mut node.walk()).find(|c| c.kind() == kind)
}

/// Check if a node has a child of a specific kind
pub fn has_child_kind(node: &tree_sitter::Node, kind: &str) -> bool {
    node.children(&mut node.walk()).any(|c| c.kind() == kind)
}

/// Extract generics/type parameters in a generic way
pub fn extract_type_parameters(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();

    // Try different field names for type parameters
    let type_params = node
        .child_by_field_name("type_parameters")
        .or_else(|| node.child_by_field_name("type_params"))
        .or_else(|| find_child_by_kind(node, "type_parameters"))
        .or_else(|| find_child_by_kind(node, "type_parameter_list"));

    if let Some(params) = type_params {
        for param in params.children(&mut params.walk()) {
            match param.kind() {
                "type_parameter" | "type_identifier" | "identifier" => {
                    if let Some(text) = get_text(&param, source) {
                        generics.push(text.to_string());
                    }
                }
                "constrained_type_parameter" | "type_bound" => {
                    if let Some(text) = get_text(&param, source) {
                        generics.push(text.to_string());
                    }
                }
                "lifetime_parameter" | "lifetime" => {
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
