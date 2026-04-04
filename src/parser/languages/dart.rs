//! Dart language extractor
//!
//! Extractor for Dart code including:
//! - Classes (with inheritance, mixins, interfaces)
//! - Functions and methods (including async/await, getters/setters)
//! - Mixins
//! - Enums
//! - Extensions
//! - Imports (via regex fallback — tree-sitter-dart parses them as ERROR nodes)
//!
//! ## AST structure notes (tree-sitter-dart ABI 14):
//! - Top-level functions: `function_signature` + `function_body` as **siblings** under `source_file`
//! - Class methods: `class_body` → `class_member` → (`method_signature` | `declaration`) + `function_body`
//! - Imports: parsed as ERROR nodes by this grammar version — extracted via regex fallback

use crate::neo4j::models::*;
use crate::parser::helpers::*;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Extract Dart code structure
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    // 1. Extract imports via regex (tree-sitter-dart parses them as ERROR)
    extract_imports_regex(source, file_path, parsed);

    // 2. Walk the AST for everything else
    extract_toplevel(root, source, file_path, parsed)
}

/// Extract top-level declarations from source_file.
/// Top-level functions appear as sibling `function_signature` + `function_body` nodes.
fn extract_toplevel(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let child_count = root.child_count();
    let mut i = 0;

    while i < child_count {
        let child = root.child(i).unwrap();

        match child.kind() {
            "class_declaration" => {
                if let Some(class) = extract_class(&child, source, file_path) {
                    parsed.symbols.push(class.name.clone());
                    parsed.structs.push(class);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "mixin_declaration" => {
                if let Some(mixin) = extract_mixin(&child, source, file_path) {
                    parsed.symbols.push(mixin.name.clone());
                    parsed.traits.push(mixin);
                }
                if let Some(body) = child.child_by_field_name("body") {
                    extract_class_body(&body, source, file_path, parsed)?;
                }
            }
            "enum_declaration" => {
                if let Some(e) = extract_enum(&child, source, file_path) {
                    parsed.symbols.push(e.name.clone());
                    parsed.enums.push(e);
                }
            }
            "extension_declaration" => {
                extract_extension(&child, source, file_path, parsed)?;
            }
            "extension_type_declaration" => {
                // extension type Foo(...) implements Bar { ... }
                if let Some(name) = get_field_text(&child, "name", source) {
                    parsed.symbols.push(name.clone());
                    parsed.structs.push(StructNode {
                        name,
                        visibility: Visibility::Public,
                        generics: vec![],
                        file_path: file_path.to_string(),
                        line_start: child.start_position().row as u32 + 1,
                        line_end: child.end_position().row as u32 + 1,
                        docstring: None,
                        parent_class: None,
                        interfaces: vec![],
                    });
                }
            }
            // Top-level function: function_signature followed by function_body sibling
            "function_signature" => {
                // Look ahead for function_body
                let body_node = if i + 1 < child_count {
                    root.child(i + 1).filter(|n| n.kind() == "function_body")
                } else {
                    None
                };

                if let Some(func) =
                    extract_function_from_signature(&child, body_node.as_ref(), source, file_path)
                {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                    if let Some(ref body) = body_node {
                        let calls = extract_calls_from_node(body, source, &func_id);
                        parsed.function_calls.extend(calls);
                    }
                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }

                // Skip the function_body we consumed
                if body_node.is_some() {
                    i += 1;
                }
            }
            "library_import" => {
                if let Some(import) = extract_import(&child, source, file_path) {
                    parsed.imports.push(import);
                }
            }
            _ => {
                // Recurse into unknown containers (e.g., conditional compilation)
                if child.named_child_count() > 0 && child.kind() != "ERROR" {
                    extract_toplevel(&child, source, file_path, parsed)?;
                }
            }
        }

        i += 1;
    }

    Ok(())
}

/// Extract a function from a `function_signature` node (top-level or inside declaration)
fn extract_function_from_signature(
    sig: &tree_sitter::Node,
    body: Option<&tree_sitter::Node>,
    source: &str,
    file_path: &str,
) -> Option<FunctionNode> {
    let name = get_field_text(sig, "name", source)?;
    if name == "new" {
        return None;
    }

    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };

    let docstring = get_dart_doc(sig, source);

    let params = sig
        .child_by_field_name("parameters")
        .filter(|p| p.kind() == "formal_parameter_list")
        .map(|p| extract_dart_params(&p, source))
        .unwrap_or_default();

    let return_type = sig
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.to_string());

    let line_end = body
        .map(|b| b.end_position().row as u32 + 1)
        .unwrap_or(sig.end_position().row as u32 + 1);

    let is_async =
        body.is_some_and(|b| get_text(b, source).is_some_and(|t| t.starts_with("async")));

    let complexity = body.map(|b| calculate_complexity(b)).unwrap_or(1);

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics: vec![],
        is_async,
        is_unsafe: false,
        complexity,
        file_path: file_path.to_string(),
        line_start: sig.start_position().row as u32 + 1,
        line_end,
        docstring,
    })
}

/// Extract a function from a class_member node.
/// Structure: class_member → (method_signature | declaration) → function_signature + function_body
fn extract_function_from_member(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
) -> Option<FunctionNode> {
    // Find the signature (may be nested in method_signature or declaration)
    let func_sig = find_function_signature(node)?;

    let name = get_field_text(&func_sig, "name", source)?;
    if name == "new" {
        return None;
    }

    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };

    let docstring = get_dart_doc(node, source);

    let params = func_sig
        .child_by_field_name("parameters")
        .filter(|p| p.kind() == "formal_parameter_list")
        .map(|p| extract_dart_params(&p, source))
        .unwrap_or_default();

    let return_type = func_sig
        .child_by_field_name("return_type")
        .and_then(|r| get_text(&r, source))
        .map(|s| s.to_string());

    // Check for async in function_body sibling
    let is_async = find_child_by_kind(node, "function_body")
        .and_then(|b| get_text(&b, source))
        .is_some_and(|t| t.starts_with("async"));

    let complexity = find_child_by_kind(node, "function_body")
        .map(|b| calculate_complexity(&b))
        .unwrap_or(1);

    Some(FunctionNode {
        name,
        visibility,
        params,
        return_type,
        generics: vec![],
        is_async,
        is_unsafe: false,
        complexity,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
    })
}

/// Find the function_signature node, which may be nested inside method_signature or declaration
fn find_function_signature<'a>(node: &tree_sitter::Node<'a>) -> Option<tree_sitter::Node<'a>> {
    // Direct function_signature child
    if let Some(fs) = find_child_by_kind(node, "function_signature") {
        return Some(fs);
    }

    // Inside a method_signature
    if let Some(ms) = find_child_by_kind(node, "method_signature") {
        if let Some(fs) = find_child_by_kind(&ms, "function_signature") {
            return Some(fs);
        }
        if let Some(gs) = find_child_by_kind(&ms, "getter_signature") {
            return Some(gs);
        }
        if let Some(ss) = find_child_by_kind(&ms, "setter_signature") {
            return Some(ss);
        }
    }

    // Inside a declaration wrapper
    if let Some(decl) = find_child_by_kind(node, "declaration") {
        if let Some(fs) = find_child_by_kind(&decl, "function_signature") {
            return Some(fs);
        }
    }

    None
}

fn extract_class(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<StructNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };
    let docstring = get_dart_doc(node, source);
    let generics = node
        .child_by_field_name("type_parameters")
        .map(|tp| extract_dart_type_params(&tp, source))
        .unwrap_or_default();

    let parent_class = node
        .child_by_field_name("superclass")
        .and_then(|sc| sc.child_by_field_name("type"))
        .and_then(|t| get_text(&t, source))
        .map(|s| s.split('<').next().unwrap_or(s).trim().to_string());

    let mut ifaces: Vec<String> = node
        .child_by_field_name("interfaces")
        .map(|i| extract_type_list(&i, source))
        .unwrap_or_default();

    let mixins_from_superclass: Vec<String> = node
        .child_by_field_name("superclass")
        .and_then(|sc| find_child_by_kind(&sc, "mixins"))
        .map(|m| extract_type_list(&m, source))
        .unwrap_or_default();

    ifaces.extend(mixins_from_superclass);

    Some(StructNode {
        name,
        visibility,
        generics,
        file_path: file_path.to_string(),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        docstring,
        parent_class,
        interfaces: ifaces,
    })
}

fn extract_mixin(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<TraitNode> {
    let name = get_field_text(node, "name", source)?;
    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };
    let docstring = get_dart_doc(node, source);
    let generics = node
        .child_by_field_name("type_parameters")
        .map(|tp| extract_dart_type_params(&tp, source))
        .unwrap_or_default();

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
    let visibility = if name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    };
    let docstring = get_dart_doc(node, source);

    let variants: Vec<String> = node
        .child_by_field_name("body")
        .map(|body| {
            body.children(&mut body.walk())
                .filter(|c| c.kind() == "enum_constant")
                .filter_map(|v| {
                    get_field_text(&v, "name", source).or_else(|| {
                        find_child_by_kind(&v, "identifier")
                            .and_then(|id| get_text(&id, source).map(|s| s.to_string()))
                    })
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

fn extract_extension(
    node: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    let for_type = node
        .child_by_field_name("class")
        .and_then(|c| get_text(&c, source))
        .map(|s| s.to_string())
        .unwrap_or_default();

    if !for_type.is_empty() {
        parsed.impl_blocks.push(ImplNode {
            for_type,
            trait_name: None,
            generics: vec![],
            where_clause: None,
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
        });
    }

    if let Some(body) = node.child_by_field_name("body") {
        extract_class_body(&body, source, file_path, parsed)?;
    }

    Ok(())
}

/// Extract imports via regex since tree-sitter-dart parses them as ERROR nodes
fn extract_imports_regex(source: &str, file_path: &str, parsed: &mut ParsedFile) {
    for (line_num, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("import ") && !trimmed.starts_with("export ") {
            continue;
        }

        // Extract the URI between quotes
        let uri = if let Some(start) = trimmed.find('\'') {
            let rest = &trimmed[start + 1..];
            rest.find('\'').map(|end| &rest[..end])
        } else if let Some(start) = trimmed.find('"') {
            let rest = &trimmed[start + 1..];
            rest.find('"').map(|end| &rest[..end])
        } else {
            None
        };

        if let Some(uri) = uri {
            let alias = if let Some(as_pos) = trimmed.find(" as ") {
                let after_as = &trimmed[as_pos + 4..];
                let end = after_as.find([';', ' ']).unwrap_or(after_as.len());
                Some(after_as[..end].to_string())
            } else {
                None
            };

            parsed.imports.push(ImportNode {
                path: uri.to_string(),
                alias,
                items: vec![],
                file_path: file_path.to_string(),
                line: line_num as u32 + 1,
            });
        }
    }
}

fn extract_import(node: &tree_sitter::Node, source: &str, file_path: &str) -> Option<ImportNode> {
    let spec = find_child_by_kind(node, "import_specification")?;

    let uri = spec
        .child_by_field_name("uri")
        .and_then(|u| get_text(&u, source))
        .map(|s| s.trim_matches('\'').trim_matches('"').to_string())?;

    let alias = spec
        .child_by_field_name("alias")
        .and_then(|a| get_text(&a, source))
        .map(|s| s.to_string());

    let mut items = Vec::new();
    for c in spec.children(&mut spec.walk()) {
        if c.kind() == "combinator" {
            for id in c.children(&mut c.walk()) {
                if id.kind() == "identifier" {
                    if let Some(s) = get_text(&id, source) {
                        items.push(s.to_string());
                    }
                }
            }
        }
    }

    Some(ImportNode {
        path: uri,
        alias,
        items,
        file_path: file_path.to_string(),
        line: node.start_position().row as u32 + 1,
    })
}

fn extract_class_body(
    body: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    for child in body.children(&mut body.walk()) {
        if child.kind() == "class_member" {
            if let Some(func) = extract_function_from_member(&child, source, file_path) {
                let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                let calls = extract_calls_from_node(&child, source, &func_id);
                parsed.function_calls.extend(calls);
                parsed.symbols.push(func.name.clone());
                parsed.functions.push(func);
            }
        }
    }
    Ok(())
}

fn extract_dart_params(node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
    let mut params = Vec::new();

    fn collect_params(node: &tree_sitter::Node, source: &str, params: &mut Vec<Parameter>) {
        for child in node.children(&mut node.walk()) {
            match child.kind() {
                "formal_parameter" => {
                    let name = child
                        .child_by_field_name("name")
                        .and_then(|n| get_text(&n, source))
                        .unwrap_or("_")
                        .to_string();

                    let type_name = child
                        .children(&mut child.walk())
                        .find(|c| {
                            matches!(
                                c.kind(),
                                "type_identifier"
                                    | "function_type"
                                    | "void_type"
                                    | "type_arguments"
                            )
                        })
                        .and_then(|t| get_text(&t, source))
                        .map(|s| s.to_string());

                    params.push(Parameter { name, type_name });
                }
                "optional_formal_parameters" => {
                    collect_params(&child, source, params);
                }
                _ => {}
            }
        }
    }

    collect_params(node, source, &mut params);
    params
}

fn extract_dart_type_params(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut generics = Vec::new();
    for child in node.children(&mut node.walk()) {
        if child.kind() == "type_parameter" {
            if let Some(text) = get_text(&child, source) {
                generics.push(text.to_string());
            }
        }
    }
    generics
}

fn extract_type_list(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    node.children(&mut node.walk())
        .filter(|c| c.kind() == "type_identifier" || c.kind() == "type_arguments")
        .filter_map(|c| {
            get_text(&c, source).map(|s| s.split('<').next().unwrap_or(s).trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Extract Dart doc comments (/// style)
fn get_dart_doc(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines = Vec::new();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" => {
                let text = get_text(&sibling, source)?;
                if text.starts_with("///") {
                    doc_lines.push(text.trim_start_matches('/').trim().to_string());
                } else {
                    break;
                }
            }
            "documentation_comment" => {
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
                doc_lines.push(
                    text.lines()
                        .map(|l| l.trim_start_matches('/').trim())
                        .filter(|l| !l.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
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
