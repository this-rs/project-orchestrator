//! Canonical intermediate representation for the AST → scene projection.
//!
//! Folds the 18 tree-sitter languages onto seven mechanical primitives
//! (`assembly`, `mechanism`, `loop`, `branch`, `call`, `assign`, `data`), as specified
//! in `frontend/docs/architecture/3d-general/04-ast-projection.md` (§1 pipeline, §2 table).
//!
//! Two layers:
//! - [`extract_flow_nodes`] extends the entity-level AST with the control-flow / assignment
//!   nodes the parser does not emit today (`loop`, `branch`, `assign`), reusing the same
//!   `TreeCursor` walk pattern as [`crate::parser::helpers::calculate_complexity`].
//! - [`build_code_ir`] normalizes [`ParsedFile`] + flow nodes into [`CodeIr`] (7 primitives),
//!   folding the language-specific tree-sitter node kinds onto one canonical vocabulary.

use crate::neo4j::models::FunctionNode;
use crate::parser::ParsedFile;

/// Maximum bounded size of a primitive (spec §2: `size = clamp(complexity, 1, K_max)`).
pub const SIZE_MAX: f64 = 8.0;

/// The seven canonical mechanical primitives (spec §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveKind {
    /// Module / class / file.
    Assembly,
    /// Function / method.
    Mechanism,
    /// Loop (`for` / `while`).
    Loop,
    /// Conditional (`if` / `match` / `switch`).
    Branch,
    /// Function call.
    Call,
    /// Assignment / binding.
    Assign,
    /// Parameter / return / field (data reservoir).
    Data,
}

impl PrimitiveKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PrimitiveKind::Assembly => "assembly",
            PrimitiveKind::Mechanism => "mechanism",
            PrimitiveKind::Loop => "loop",
            PrimitiveKind::Branch => "branch",
            PrimitiveKind::Call => "call",
            PrimitiveKind::Assign => "assign",
            PrimitiveKind::Data => "data",
        }
    }
}

/// Typed edge between primitives (spec §2 : appels → tuyaux, contrôle → liaisons, données → tuyaux).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// Function call edge (caller mechanism → call → callee mechanism).
    Call,
    /// Control-flow containment (mechanism → loop / branch).
    Control,
    /// Data flow (mechanism → data / assign).
    Data,
    /// Containment (file/module assembly → member assembly/mechanism).
    Member,
    /// External dependency (file assembly → external module).
    Import,
}

/// One primitive node of the canonical IR.
#[derive(Debug, Clone)]
pub struct CodeIrNode {
    /// Stable id (derived from file path / symbol name / line, deterministic for a given source).
    pub id: String,
    pub kind: PrimitiveKind,
    /// Symbol name (function name, variable name, type name…).
    pub label: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    /// Bounded size metric (cyclomatic complexity clamped to `SIZE_MAX`; call = confidence 0..1).
    pub size: f64,
    /// Optional discriminator: loop kind (`for`/`while`/…), branch kind (`if`/`match`/…),
    /// data role (`param`/`return`), assembly kind (`file`/`struct`/`trait`/`enum`).
    pub detail: Option<String>,
}

/// A typed edge between two primitive ids.
#[derive(Debug, Clone)]
pub struct CodeIrLink {
    pub from: String,
    pub to: String,
    pub kind: LinkKind,
}

/// Canonical IR of a parsed file: nodes + links, ready for the semantic-mapping stage.
#[derive(Debug, Clone, Default)]
pub struct CodeIr {
    pub nodes: Vec<CodeIrNode>,
    pub links: Vec<CodeIrLink>,
}

impl CodeIr {
    /// Nodes of a given primitive kind.
    pub fn primitives_of(&self, kind: PrimitiveKind) -> impl Iterator<Item = &CodeIrNode> {
        self.nodes.iter().filter(move |n| n.kind == kind)
    }
}

/// A raw control/assignment flow node before IR folding (deliverable 1 : parser extension).
#[derive(Debug, Clone)]
pub struct FlowNode {
    /// Always `Loop`, `Branch` or `Assign`.
    pub kind: PrimitiveKind,
    /// Discriminator (`for`, `while`, `if`, `match`, `assign`…).
    pub detail: String,
    /// Symbol/condition text.
    pub label: String,
    pub line: u32,
    /// Id of the enclosing container (innermost function, else the file assembly).
    pub parent_id: String,
}

/// Stable function id — matches the `caller_id` emitted by every language extractor
/// (`format!("{}:{}:{}", file_path, name, line_start)`).
pub fn function_id(file_path: &str, name: &str, line_start: u32) -> String {
    format!("{}:{}:{}", file_path, name, line_start)
}

fn file_assembly_id(path: &str) -> String {
    format!("file:{}", path)
}

fn type_assembly_id(path: &str, kind: &str, name: &str, line: u32) -> String {
    format!("{}:{}:{}:{}", path, kind, name, line)
}

fn file_label(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn bounded_size(v: f64) -> f64 {
    v.clamp(1.0, SIZE_MAX)
}

fn row(node: &tree_sitter::Node) -> u32 {
    node.start_position().row as u32 + 1
}

/// Innermost function whose line span contains `line`.
fn innermost_function(functions: &[FunctionNode], line: u32) -> Option<&FunctionNode> {
    functions
        .iter()
        .filter(|f| f.line_start <= line && line <= f.line_end)
        .min_by_key(|f| f.line_end - f.line_start)
}

// ── Flow extraction (deliverable 1) ────────────────────────────────────────────────────────────

/// tree-sitter node kinds that fold to a `loop` primitive, with their discriminator.
fn is_loop_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "for_statement"
        | "for_expression"
        | "for_in_statement"
        | "enhanced_for_statement"
        | "for_each" => Some("for"),
        "while_statement" | "while_expression" => Some("while"),
        "loop_expression" => Some("loop"),
        "do_statement" | "do_while_statement" => Some("do"),
        "repeat_statement" => Some("repeat"),
        _ => None,
    }
}

/// tree-sitter node kinds that fold to a `branch` primitive, with their discriminator.
fn is_branch_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "if_statement" | "if_expression" => Some("if"),
        "match_expression" | "match_statement" | "case_statement" => Some("match"),
        "switch_statement" | "switch_expression" => Some("switch"),
        "ternary_expression" | "conditional_expression" | "conditional" => Some("ternary"),
        "unless_statement" | "unless_expression" => Some("unless"),
        _ => None,
    }
}

/// tree-sitter node kinds that fold to an `assign` primitive.
fn is_assign_kind(kind: &str) -> bool {
    matches!(
        kind,
        "let_declaration"
            | "assignment_expression"
            | "assignment"
            | "augmented_assignment"
            | "compound_assignment"
            | "assignment_statement"
            | "short_variable_declaration"
    )
}

/// True when this `if` is the `if` of an `else if` (already covered by the enclosing branch),
/// so that `else if` normalizes to a single `branch` (spec §2 « règle d'or »).
fn is_else_if(node: &tree_sitter::Node) -> bool {
    node.parent()
        .map(|p| {
            matches!(
                p.kind(),
                "else_clause" | "else" | "alternative" | "otherwise"
            )
        })
        .unwrap_or(false)
}

fn short_text(node: &tree_sitter::Node, source: &str, max: usize) -> Option<String> {
    let text = node.utf8_text(source.as_bytes()).ok()?.trim().to_string();
    if text.chars().count() <= max {
        Some(text)
    } else {
        Some(format!("{}…", text.chars().take(max).collect::<String>()))
    }
}

/// Left-hand side of an assignment (the written target), for the `assign` label.
fn assign_label(node: &tree_sitter::Node, source: &str) -> String {
    for field in ["left", "pattern", "name", "target"] {
        if let Some(child) = node.child_by_field_name(field) {
            if let Some(text) = short_text(&child, source, 64) {
                if !text.is_empty() {
                    return text;
                }
            }
        }
    }
    "assign".to_string()
}

/// Extends the entity-level AST with control-flow / assignment nodes.
///
/// Walks the whole tree (same pattern as `calculate_complexity`) and emits a `loop`/`branch`/
/// `assign` for the corresponding tree-sitter node kinds, attributed to the innermost enclosing
/// function (falling back to the file assembly for top-level statements).
pub fn extract_flow_nodes(
    root: &tree_sitter::Node,
    source: &str,
    parsed: &ParsedFile,
) -> Vec<FlowNode> {
    let mut out = Vec::new();
    let mut cursor = root.walk();
    walk_flow(&mut cursor, source, parsed, &mut out);
    out
}

fn walk_flow(
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    parsed: &ParsedFile,
    out: &mut Vec<FlowNode>,
) {
    loop {
        let node = cursor.node();
        let kind = node.kind();
        let line = row(&node);

        let primitive: Option<(PrimitiveKind, String, String)> = if let Some(d) = is_loop_kind(kind)
        {
            Some((PrimitiveKind::Loop, d.to_string(), d.to_string()))
        } else if let Some(d) = is_branch_kind(kind) {
            if is_else_if(&node) {
                None
            } else {
                let label = node
                    .child_by_field_name("condition")
                    .and_then(|c| short_text(&c, source, 40))
                    .unwrap_or_else(|| d.to_string());
                Some((PrimitiveKind::Branch, d.to_string(), label))
            }
        } else if is_assign_kind(kind) {
            Some((
                PrimitiveKind::Assign,
                "assign".to_string(),
                assign_label(&node, source),
            ))
        } else {
            None
        };

        if let Some((primitive, detail, label)) = primitive {
            let parent_id = innermost_function(&parsed.functions, line)
                .map(|f| function_id(&f.file_path, &f.name, f.line_start))
                .unwrap_or_else(|| file_assembly_id(&parsed.path));
            out.push(FlowNode {
                kind: primitive,
                detail,
                label,
                line,
                parent_id,
            });
        }

        if cursor.goto_first_child() {
            walk_flow(cursor, source, parsed, out);
            cursor.goto_parent();
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

// ── Normalization into CodeIR (deliverable 2) ─────────────────────────────────────────────────

fn assembly_node(
    id: String,
    kind: &str,
    name: &str,
    file_path: &str,
    line_start: u32,
    line_end: u32,
    size: f64,
) -> CodeIrNode {
    CodeIrNode {
        id,
        kind: PrimitiveKind::Assembly,
        label: name.to_string(),
        file_path: file_path.to_string(),
        line_start,
        line_end,
        size: bounded_size(size),
        detail: Some(kind.to_string()),
    }
}

/// Normalizes a parsed file into the canonical [`CodeIr`].
pub fn build_code_ir(root: &tree_sitter::Node, source: &str, parsed: &ParsedFile) -> CodeIr {
    let mut ir = CodeIr::default();
    let mut counter = IdCounter::default();

    // 1. File assembly (the root container).
    let file_id = file_assembly_id(&parsed.path);
    let max_line = parsed
        .functions
        .iter()
        .map(|f| f.line_end)
        .max()
        .unwrap_or(1);
    ir.nodes.push(CodeIrNode {
        id: file_id.clone(),
        kind: PrimitiveKind::Assembly,
        label: file_label(&parsed.path),
        file_path: parsed.path.clone(),
        line_start: 1,
        line_end: max_line,
        size: bounded_size(parsed.functions.len() as f64),
        detail: Some("file".to_string()),
    });

    // 2. Structs / traits / enums → nested assemblies.
    for s in &parsed.structs {
        let id = type_assembly_id(&parsed.path, "struct", &s.name, s.line_start);
        ir.nodes.push(assembly_node(
            id.clone(),
            "struct",
            &s.name,
            &parsed.path,
            s.line_start,
            s.line_end,
            1.0,
        ));
        ir.links.push(CodeIrLink {
            from: file_id.clone(),
            to: id,
            kind: LinkKind::Member,
        });
    }
    for t in &parsed.traits {
        let id = type_assembly_id(&parsed.path, "trait", &t.name, t.line_start);
        ir.nodes.push(assembly_node(
            id.clone(),
            "trait",
            &t.name,
            &parsed.path,
            t.line_start,
            t.line_end,
            1.0,
        ));
        ir.links.push(CodeIrLink {
            from: file_id.clone(),
            to: id,
            kind: LinkKind::Member,
        });
    }
    for e in &parsed.enums {
        let id = type_assembly_id(&parsed.path, "enum", &e.name, e.line_start);
        ir.nodes.push(assembly_node(
            id.clone(),
            "enum",
            &e.name,
            &parsed.path,
            e.line_start,
            e.line_end,
            1.0,
        ));
        ir.links.push(CodeIrLink {
            from: file_id.clone(),
            to: id,
            kind: LinkKind::Member,
        });
    }

    // 3. Functions → mechanisms, plus their `data` (params + return).
    for f in &parsed.functions {
        let id = function_id(&f.file_path, &f.name, f.line_start);
        ir.nodes.push(CodeIrNode {
            id: id.clone(),
            kind: PrimitiveKind::Mechanism,
            label: f.name.clone(),
            file_path: f.file_path.clone(),
            line_start: f.line_start,
            line_end: f.line_end,
            size: bounded_size(f.complexity as f64),
            detail: None,
        });
        ir.links.push(CodeIrLink {
            from: file_id.clone(),
            to: id.clone(),
            kind: LinkKind::Member,
        });

        for p in &f.params {
            let did = format!("{}:param:{}", id, p.name);
            ir.nodes.push(CodeIrNode {
                id: did.clone(),
                kind: PrimitiveKind::Data,
                label: p.name.clone(),
                file_path: f.file_path.clone(),
                line_start: f.line_start,
                line_end: f.line_start,
                size: 1.0,
                detail: Some("param".to_string()),
            });
            ir.links.push(CodeIrLink {
                from: id.clone(),
                to: did,
                kind: LinkKind::Data,
            });
        }
        if let Some(ret) = &f.return_type {
            let did = format!("{}:return", id);
            ir.nodes.push(CodeIrNode {
                id: did.clone(),
                kind: PrimitiveKind::Data,
                label: ret.clone(),
                file_path: f.file_path.clone(),
                line_start: f.line_start,
                line_end: f.line_start,
                size: 1.0,
                detail: Some("return".to_string()),
            });
            ir.links.push(CodeIrLink {
                from: id.clone(),
                to: did,
                kind: LinkKind::Data,
            });
        }
    }

    // 4. Function calls → `call` nodes + `Call` links.
    for call in &parsed.function_calls {
        let caller_id = call.caller_id.clone();
        let callee_id = parsed
            .functions
            .iter()
            .find(|f| f.name == call.callee_name)
            .map(|f| function_id(&f.file_path, &f.name, f.line_start))
            .unwrap_or_else(|| format!("external:{}", call.callee_name));
        let call_id = format!("{}:call:{}:{}", caller_id, call.line, call.callee_name);
        ir.nodes.push(CodeIrNode {
            id: call_id.clone(),
            kind: PrimitiveKind::Call,
            label: call.callee_name.clone(),
            file_path: parsed.path.clone(),
            line_start: call.line,
            line_end: call.line,
            size: call.confidence.clamp(0.0, 1.0),
            detail: Some(call.reason.clone()),
        });
        ir.links.push(CodeIrLink {
            from: caller_id,
            to: call_id.clone(),
            kind: LinkKind::Call,
        });
        ir.links.push(CodeIrLink {
            from: call_id,
            to: callee_id,
            kind: LinkKind::Call,
        });
    }

    // 5. Flow nodes → `loop` / `branch` / `assign`, linked to their container.
    for flow in extract_flow_nodes(root, source, parsed) {
        let id = format!(
            "{}@{}:{}:{}",
            flow.parent_id,
            flow.kind.as_str(),
            flow.line,
            counter.next()
        );
        ir.nodes.push(CodeIrNode {
            id: id.clone(),
            kind: flow.kind,
            label: flow.label,
            file_path: parsed.path.clone(),
            line_start: flow.line,
            line_end: flow.line,
            size: 1.0,
            detail: Some(flow.detail),
        });
        ir.links.push(CodeIrLink {
            from: flow.parent_id,
            to: id,
            kind: LinkKind::Control,
        });
    }

    // 6. Imports → external dependency links.
    for imp in &parsed.imports {
        ir.links.push(CodeIrLink {
            from: file_id.clone(),
            to: format!("external:{}", imp.path),
            kind: LinkKind::Import,
        });
    }

    ir
}

/// Monotonic counter for deterministic node-id disambiguation.
#[derive(Default)]
struct IdCounter(u64);

impl IdCounter {
    fn next(&mut self) -> u64 {
        let v = self.0;
        self.0 += 1;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::CodeParser;

    #[test]
    fn test_code_ir_contains_mechanism_branch_loop() {
        let src = "fn process(a: i32) -> i32 {\n  let mut x = a;\n  if x > 0 { x += 1; }\n  while x < 10 { x += 2; }\n  x\n}\n";
        let mut parser = CodeParser::new().unwrap();
        let (parsed, ir) = parser
            .parse_file_with_ir(std::path::Path::new("sample.rs"), src)
            .unwrap();

        // The entity-level AST still has the function.
        assert_eq!(parsed.functions.len(), 1);

        // A `mechanism` named `process`.
        let mechanisms: Vec<&CodeIrNode> = ir.primitives_of(PrimitiveKind::Mechanism).collect();
        assert!(mechanisms.iter().any(|m| m.label == "process"));

        // A `branch` (the `if`) and a `loop` (the `while`).
        assert!(ir
            .primitives_of(PrimitiveKind::Branch)
            .any(|b| b.detail.as_deref() == Some("if")));
        assert!(ir
            .primitives_of(PrimitiveKind::Loop)
            .any(|l| l.detail.as_deref() == Some("while")));

        // Assignments (the `let` and the two `+=`).
        assert!(ir.primitives_of(PrimitiveKind::Assign).count() >= 1);

        // Data: the parameter `a` and the return type.
        assert!(ir
            .primitives_of(PrimitiveKind::Data)
            .any(|d| d.label == "a" && d.detail.as_deref() == Some("param")));
        assert!(ir
            .primitives_of(PrimitiveKind::Data)
            .any(|d| d.detail.as_deref() == Some("return")));

        // The branch/loop are linked to the `process` mechanism (control flow).
        let process_id = mechanisms
            .iter()
            .find(|m| m.label == "process")
            .unwrap()
            .id
            .clone();
        assert!(ir
            .links
            .iter()
            .any(|l| l.kind == LinkKind::Control && l.from == process_id));
    }

    #[test]
    fn test_else_if_is_a_single_branch() {
        let src = "fn f(a: i32) -> i32 {\n  if a > 0 { 1 } else if a < 0 { -1 } else { 0 }\n}\n";
        let mut parser = CodeParser::new().unwrap();
        let (_, ir) = parser
            .parse_file_with_ir(std::path::Path::new("sample.rs"), src)
            .unwrap();
        let branches: Vec<&CodeIrNode> = ir.primitives_of(PrimitiveKind::Branch).collect();
        assert_eq!(
            branches.len(),
            1,
            "else-if should fold to a single branch, got: {:?}",
            branches.iter().map(|b| &b.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn primitive_kind_names_and_function_ids_are_stable() {
        assert_eq!(PrimitiveKind::Assembly.as_str(), "assembly");
        assert_eq!(PrimitiveKind::Mechanism.as_str(), "mechanism");
        assert_eq!(PrimitiveKind::Loop.as_str(), "loop");
        assert_eq!(PrimitiveKind::Branch.as_str(), "branch");
        assert_eq!(PrimitiveKind::Call.as_str(), "call");
        assert_eq!(PrimitiveKind::Assign.as_str(), "assign");
        assert_eq!(PrimitiveKind::Data.as_str(), "data");
        assert_eq!(function_id("a/b.rs", "f", 7), "a/b.rs:f:7");
        assert_eq!(bounded_size(0.0), 1.0, "sizes are floored at 1");
        assert_eq!(bounded_size(1e9), SIZE_MAX, "sizes are capped at SIZE_MAX");
        assert_eq!(file_label("src/a/b.rs"), "b.rs");
    }

    #[test]
    fn code_ir_folds_types_calls_imports_and_every_flow_kind() {
        // A struct, a trait and an enum (nested assemblies), a `for`, a `loop` and a
        // `match` (the other loop/branch kinds), a local and an external call, an import,
        // and an `if` whose condition is longer than 40 characters (truncated label).
        let src = "use std::collections::HashMap;\n\
                   struct Point { x: i32 }\n\
                   trait Shape { fn area(&self) -> i32; }\n\
                   enum Color { Red, Blue }\n\
                   fn helper(v: i32) -> i32 { v + 1 }\n\
                   fn main_flow(n: i32) -> i32 {\n\
                   \x20 let mut total = 0;\n\
                   \x20 for i in 0..n { total += helper(i); }\n\
                   \x20 loop { break; }\n\
                   \x20 match n { 0 => total = 1, _ => total = 2 }\n\
                   \x20 if total_is_very_large_and_this_condition_is_long(total) && n > 0 { total = 3; }\n\
                   \x20 external_function(total);\n\
                   \x20 total\n\
                   }\n";
        let mut parser = CodeParser::new().unwrap();
        let (parsed, ir) = parser
            .parse_file_with_ir(std::path::Path::new("sample.rs"), src)
            .unwrap();
        assert!(parsed.functions.len() >= 2, "{:?}", parsed.functions.len());

        // Nested assemblies for the struct, trait and enum, each a Member of the file.
        let file_id = file_assembly_id("sample.rs");
        for (detail, name) in [("struct", "Point"), ("trait", "Shape"), ("enum", "Color")] {
            let node = ir
                .primitives_of(PrimitiveKind::Assembly)
                .find(|a| a.detail.as_deref() == Some(detail) && a.label == name)
                .unwrap_or_else(|| panic!("missing {detail} assembly {name}"));
            assert!(
                ir.links
                    .iter()
                    .any(|l| l.kind == LinkKind::Member && l.from == file_id && l.to == node.id),
                "{name} must be a Member of the file"
            );
            assert_eq!(
                node.id,
                type_assembly_id("sample.rs", detail, name, node.line_start)
            );
        }
        // The file assembly itself, labelled by file name.
        assert!(ir
            .primitives_of(PrimitiveKind::Assembly)
            .any(|a| a.id == file_id && a.label == "b.rs".replace("b.rs", "sample.rs")));

        // Every loop and branch kind of the snippet.
        let details = |k: PrimitiveKind| -> Vec<String> {
            ir.primitives_of(k)
                .filter_map(|n| n.detail.clone())
                .collect()
        };
        let loops = details(PrimitiveKind::Loop);
        assert!(loops.iter().any(|d| d == "for"), "{loops:?}");
        assert!(loops.iter().any(|d| d == "loop"), "{loops:?}");
        let branches = details(PrimitiveKind::Branch);
        assert!(branches.iter().any(|d| d == "match"), "{branches:?}");
        assert!(branches.iter().any(|d| d == "if"), "{branches:?}");

        // The long `if` condition is truncated to 40 characters plus an ellipsis.
        let truncated = ir
            .primitives_of(PrimitiveKind::Branch)
            .find(|b| b.label.ends_with('…'))
            .expect("a truncated branch label");
        assert_eq!(truncated.label.chars().count(), 41, "{}", truncated.label);

        // Calls: a `call` node per call site, linked Call-wise; an unknown callee is external.
        let calls: Vec<&CodeIrNode> = ir.primitives_of(PrimitiveKind::Call).collect();
        assert!(
            !calls.is_empty(),
            "the snippet calls helper() and external_function()"
        );
        assert!(calls.iter().all(|c| (0.0..=1.0).contains(&c.size)));
        assert!(ir.links.iter().any(|l| l.kind == LinkKind::Call));
        if calls.iter().any(|c| c.label == "external_function") {
            assert!(ir
                .links
                .iter()
                .any(|l| l.kind == LinkKind::Call && l.to == "external:external_function"));
        }

        // The import becomes an Import link from the file to an external id.
        assert!(ir.links.iter().any(|l| l.kind == LinkKind::Import
            && l.from == file_id
            && l.to.starts_with("external:")));

        // Flow nodes hang off the innermost function (Control links), never the file.
        let main_id = parsed
            .functions
            .iter()
            .find(|f| f.name == "main_flow")
            .map(|f| function_id(&f.file_path, &f.name, f.line_start))
            .expect("main_flow parsed");
        assert!(ir
            .links
            .iter()
            .any(|l| l.kind == LinkKind::Control && l.from == main_id));
        for n in ir.nodes.iter() {
            assert!(n.size >= 1.0 || n.kind == PrimitiveKind::Call, "{:?}", n);
        }
    }

    #[test]
    fn a_very_complex_function_is_capped_at_size_max() {
        let mut body = String::from("fn busy(a: i32) -> i32 {\n  let mut x = a;\n");
        for i in 0..12 {
            body.push_str(&format!("  if x > {i} {{ x += 1; }}\n"));
        }
        body.push_str("  x\n}\n");
        let mut parser = CodeParser::new().unwrap();
        let (_, ir) = parser
            .parse_file_with_ir(std::path::Path::new("sample.rs"), &body)
            .unwrap();
        let busy = ir
            .primitives_of(PrimitiveKind::Mechanism)
            .find(|m| m.label == "busy")
            .expect("busy mechanism");
        assert_eq!(busy.size, SIZE_MAX, "complexity beyond SIZE_MAX is clamped");
        assert_eq!(ir.primitives_of(PrimitiveKind::Branch).count(), 12);
    }
}
