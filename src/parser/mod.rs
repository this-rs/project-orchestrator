//! Tree-sitter based code parser

pub mod languages;

use crate::meilisearch::indexes::CodeDocument;
use crate::neo4j::models::*;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Language, Parser};

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedLanguage {
    Rust,
    TypeScript,
    Python,
    Go,
}

impl SupportedLanguage {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "ts" | "tsx" => Some(Self::TypeScript),
            "js" | "jsx" => Some(Self::TypeScript), // Use TS parser for JS
            "py" => Some(Self::Python),
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    /// Get the tree-sitter language
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
        }
    }

    /// Get the language name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Python => "python",
            Self::Go => "go",
        }
    }
}

/// Code parser using tree-sitter
pub struct CodeParser {
    parsers: HashMap<SupportedLanguage, Parser>,
}

impl CodeParser {
    /// Create a new code parser
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();

        for lang in [
            SupportedLanguage::Rust,
            SupportedLanguage::TypeScript,
            SupportedLanguage::Python,
            SupportedLanguage::Go,
        ] {
            let mut parser = Parser::new();
            parser
                .set_language(&lang.tree_sitter_language())
                .context(format!("Failed to set language for {:?}", lang))?;
            parsers.insert(lang, parser);
        }

        Ok(Self { parsers })
    }

    /// Parse a file and extract code structure
    pub fn parse_file(&mut self, path: &Path, content: &str) -> Result<ParsedFile> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();

        let language = SupportedLanguage::from_extension(ext)
            .context(format!("Unsupported file extension: {}", ext))?;

        let parser = self
            .parsers
            .get_mut(&language)
            .context("Parser not found")?;

        let tree = parser
            .parse(content, None)
            .context("Failed to parse file")?;

        let root = tree.root_node();
        let path_str = path.to_string_lossy().to_string();

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = hex::encode(hasher.finalize());

        let mut parsed = ParsedFile {
            path: path_str.clone(),
            language: language.as_str().to_string(),
            hash,
            functions: Vec::new(),
            structs: Vec::new(),
            traits: Vec::new(),
            enums: Vec::new(),
            imports: Vec::new(),
            impl_blocks: Vec::new(),
            function_calls: Vec::new(),
            symbols: Vec::new(),
        };

        // Extract based on language
        match language {
            SupportedLanguage::Rust => {
                self.extract_rust(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::TypeScript => {
                self.extract_typescript(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Python => {
                self.extract_python(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Go => {
                self.extract_go(&root, content, &path_str, &mut parsed)?;
            }
        }

        Ok(parsed)
    }

    /// Extract Rust code structure
    fn extract_rust(
        &self,
        root: &tree_sitter::Node,
        source: &str,
        file_path: &str,
        parsed: &mut ParsedFile,
    ) -> Result<()> {
        let mut cursor = root.walk();

        for node in root.children(&mut cursor) {
            match node.kind() {
                "function_item" => {
                    if let Some(func) = self.extract_rust_function(&node, source, file_path) {
                        let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);

                        // Extract function calls within this function
                        self.extract_function_calls(&node, source, &func_id, parsed);

                        parsed.symbols.push(func.name.clone());
                        parsed.functions.push(func);
                    }
                }
                "struct_item" => {
                    if let Some(s) = self.extract_rust_struct(&node, source, file_path) {
                        parsed.symbols.push(s.name.clone());
                        parsed.structs.push(s);
                    }
                }
                "trait_item" => {
                    if let Some(t) = self.extract_rust_trait(&node, source, file_path) {
                        parsed.symbols.push(t.name.clone());
                        parsed.traits.push(t);
                    }
                }
                "enum_item" => {
                    if let Some(e) = self.extract_rust_enum(&node, source, file_path) {
                        parsed.symbols.push(e.name.clone());
                        parsed.enums.push(e);
                    }
                }
                "use_declaration" => {
                    if let Some(import) = self.extract_rust_import(&node, source, file_path) {
                        parsed.imports.push(import);
                    }
                }
                "impl_item" => {
                    // Extract impl block and its methods
                    self.extract_rust_impl(&node, source, file_path, parsed)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn extract_rust_function(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
    ) -> Option<FunctionNode> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

        let visibility = self.get_rust_visibility(node, source);

        // Check for async/unsafe in function_modifiers or as direct text
        let is_async = self.has_rust_modifier(node, source, "async");
        let is_unsafe = self.has_rust_modifier(node, source, "unsafe");

        // Extract parameters
        let params = node
            .child_by_field_name("parameters")
            .map(|p| self.extract_rust_params(&p, source))
            .unwrap_or_default();

        // Extract return type
        let return_type = node
            .child_by_field_name("return_type")
            .and_then(|r| r.utf8_text(source.as_bytes()).ok())
            .map(|s| s.trim_start_matches("->").trim().to_string());

        // Extract docstring (look for preceding doc comments)
        let docstring = self.get_preceding_docstring(node, source);

        Some(FunctionNode {
            name,
            visibility,
            params,
            return_type,
            generics: vec![], // TODO: extract generics
            is_async,
            is_unsafe,
            complexity: self.calculate_complexity(node),
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
            docstring,
        })
    }

    fn extract_rust_struct(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
    ) -> Option<StructNode> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

        let visibility = self.get_rust_visibility(node, source);
        let docstring = self.get_preceding_docstring(node, source);

        Some(StructNode {
            name,
            visibility,
            generics: vec![], // TODO: extract generics
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
            docstring,
        })
    }

    fn extract_rust_trait(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
    ) -> Option<TraitNode> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

        let visibility = self.get_rust_visibility(node, source);
        let docstring = self.get_preceding_docstring(node, source);

        Some(TraitNode {
            name,
            visibility,
            generics: vec![],
            file_path: file_path.to_string(),
            line_start: node.start_position().row as u32 + 1,
            line_end: node.end_position().row as u32 + 1,
            docstring,
        })
    }

    fn extract_rust_enum(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
    ) -> Option<EnumNode> {
        let name_node = node.child_by_field_name("name")?;
        let name = name_node.utf8_text(source.as_bytes()).ok()?.to_string();

        let visibility = self.get_rust_visibility(node, source);
        let docstring = self.get_preceding_docstring(node, source);

        // Extract variants
        let variants: Vec<String> = node
            .child_by_field_name("body")
            .map(|body| {
                body.children(&mut body.walk())
                    .filter(|c| c.kind() == "enum_variant")
                    .filter_map(|v| {
                        v.child_by_field_name("name")
                            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                            .map(|s| s.to_string())
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

    fn extract_rust_import(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
    ) -> Option<ImportNode> {
        let path = node.utf8_text(source.as_bytes()).ok()?;
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

    fn extract_rust_impl(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        file_path: &str,
        parsed: &mut ParsedFile,
    ) -> Result<()> {
        // Extract the type being implemented for
        let for_type = self.get_rust_impl_type(node, source);
        let trait_name = self.get_rust_impl_trait(node, source);

        if let Some(for_type) = for_type {
            // Create ImplNode
            let impl_node = ImplNode {
                for_type: for_type.clone(),
                trait_name: trait_name.clone(),
                generics: vec![], // TODO: extract generics
                where_clause: None, // TODO: extract where clause
                file_path: file_path.to_string(),
                line_start: node.start_position().row as u32 + 1,
                line_end: node.end_position().row as u32 + 1,
            };
            parsed.impl_blocks.push(impl_node);
        }

        // Extract methods from the body
        let body = match node.child_by_field_name("body") {
            Some(b) => b,
            None => return Ok(()),
        };

        for child in body.children(&mut body.walk()) {
            if child.kind() == "function_item" {
                if let Some(func) = self.extract_rust_function(&child, source, file_path) {
                    let func_id = format!("{}:{}:{}", file_path, func.name, func.line_start);

                    // Extract function calls within this method
                    self.extract_function_calls(&child, source, &func_id, parsed);

                    parsed.symbols.push(func.name.clone());
                    parsed.functions.push(func);
                }
            }
        }

        Ok(())
    }

    /// Extract the type being implemented in an impl block
    fn get_rust_impl_type(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        // Look for type_identifier in the impl item
        for child in node.children(&mut node.walk()) {
            match child.kind() {
                "type_identifier" => {
                    return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
                }
                "generic_type" => {
                    // e.g., Vec<T> - get the base type
                    if let Some(name) = child.child_by_field_name("type") {
                        return name.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Extract the trait being implemented in an impl block
    fn get_rust_impl_trait(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        // Look for "impl Trait for Type" pattern
        let mut found_trait = None;
        let mut found_for = false;

        for child in node.children(&mut node.walk()) {
            if child.kind() == "for" {
                found_for = true;
                break;
            }
            if child.kind() == "type_identifier" || child.kind() == "generic_type" {
                if let Ok(text) = child.utf8_text(source.as_bytes()) {
                    found_trait = Some(text.to_string());
                }
            }
        }

        // If we found "for", the first type was the trait
        if found_for {
            found_trait
        } else {
            None
        }
    }

    /// Extract function calls from a function body
    fn extract_function_calls(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        caller_id: &str,
        parsed: &mut ParsedFile,
    ) {
        let mut cursor = node.walk();
        self.extract_calls_recursive(&mut cursor, source, caller_id, parsed);
    }

    fn extract_calls_recursive(
        &self,
        cursor: &mut tree_sitter::TreeCursor,
        source: &str,
        caller_id: &str,
        parsed: &mut ParsedFile,
    ) {
        loop {
            let node = cursor.node();

            if node.kind() == "call_expression" {
                // Get the function being called
                if let Some(func) = node.child_by_field_name("function") {
                    let callee_name = match func.kind() {
                        "identifier" => func.utf8_text(source.as_bytes()).ok().map(|s| s.to_string()),
                        "field_expression" => {
                            // e.g., self.method() or obj.method()
                            func.child_by_field_name("field")
                                .and_then(|f| f.utf8_text(source.as_bytes()).ok())
                                .map(|s| s.to_string())
                        }
                        "scoped_identifier" => {
                            // e.g., Module::function()
                            func.utf8_text(source.as_bytes()).ok().map(|s| s.to_string())
                        }
                        _ => None,
                    };

                    if let Some(callee) = callee_name {
                        parsed.function_calls.push(FunctionCall {
                            caller_id: caller_id.to_string(),
                            callee_name: callee,
                            line: node.start_position().row as u32 + 1,
                        });
                    }
                }
            }

            if cursor.goto_first_child() {
                self.extract_calls_recursive(cursor, source, caller_id, parsed);
                cursor.goto_parent();
            }

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    fn extract_rust_params(&self, node: &tree_sitter::Node, source: &str) -> Vec<Parameter> {
        let mut params = Vec::new();

        for child in node.children(&mut node.walk()) {
            if child.kind() == "parameter" {
                let name = child
                    .child_by_field_name("pattern")
                    .and_then(|p| p.utf8_text(source.as_bytes()).ok())
                    .unwrap_or("_")
                    .to_string();

                let type_name = child
                    .child_by_field_name("type")
                    .and_then(|t| t.utf8_text(source.as_bytes()).ok())
                    .map(|s| s.to_string());

                params.push(Parameter { name, type_name });
            }
        }

        params
    }

    fn get_rust_visibility(&self, node: &tree_sitter::Node, source: &str) -> Visibility {
        for child in node.children(&mut node.walk()) {
            if child.kind() == "visibility_modifier" {
                let text = child.utf8_text(source.as_bytes()).unwrap_or_default();
                return match text {
                    "pub" => Visibility::Public,
                    s if s.starts_with("pub(crate)") => Visibility::Crate,
                    s if s.starts_with("pub(super)") => Visibility::Super,
                    s if s.starts_with("pub(in") => {
                        Visibility::InPath(s.trim_start_matches("pub(in ").trim_end_matches(')').to_string())
                    }
                    _ => Visibility::Private,
                };
            }
        }
        Visibility::Private
    }

    /// Check if a function has a specific modifier (async, unsafe, etc.)
    fn has_rust_modifier(&self, node: &tree_sitter::Node, source: &str, modifier: &str) -> bool {
        // Check in function_modifiers child (tree-sitter-rust wraps modifiers there)
        for child in node.children(&mut node.walk()) {
            if child.kind() == "function_modifiers" {
                // Check children of function_modifiers for the modifier keyword
                for modifier_child in child.children(&mut child.walk()) {
                    if modifier_child.kind() == modifier {
                        return true;
                    }
                    // Also check the text content
                    if let Ok(text) = modifier_child.utf8_text(source.as_bytes()) {
                        if text == modifier {
                            return true;
                        }
                    }
                }
                // Also check the full text of function_modifiers
                if let Ok(text) = child.utf8_text(source.as_bytes()) {
                    if text.contains(modifier) {
                        return true;
                    }
                }
            }
            // Direct check for modifier keyword as child
            if child.kind() == modifier {
                return true;
            }
        }
        false
    }

    fn get_preceding_docstring(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut doc_lines = Vec::new();

        while let Some(sibling) = prev {
            match sibling.kind() {
                "line_comment" => {
                    let text = sibling.utf8_text(source.as_bytes()).ok()?;
                    if text.starts_with("///") || text.starts_with("//!") {
                        doc_lines.push(text.trim_start_matches('/').trim().to_string());
                    } else {
                        break;
                    }
                }
                "block_comment" => {
                    let text = sibling.utf8_text(source.as_bytes()).ok()?;
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

    fn calculate_complexity(&self, node: &tree_sitter::Node) -> u32 {
        // Simple cyclomatic complexity approximation
        let mut complexity = 1u32;
        let mut cursor = node.walk();

        fn count_branches(cursor: &mut tree_sitter::TreeCursor, complexity: &mut u32) {
            loop {
                let node = cursor.node();
                match node.kind() {
                    "if_expression" | "while_expression" | "for_expression" | "match_arm"
                    | "loop_expression" | "?" => {
                        *complexity += 1;
                    }
                    "binary_expression" => {
                        // Count && and || as branches
                        if let Some(op) = node.child_by_field_name("operator") {
                            let kind = op.kind();
                            if kind == "&&" || kind == "||" {
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

    /// Extract TypeScript code structure (simplified)
    fn extract_typescript(
        &self,
        root: &tree_sitter::Node,
        source: &str,
        file_path: &str,
        parsed: &mut ParsedFile,
    ) -> Result<()> {
        let mut cursor = root.walk();

        for node in root.children(&mut cursor) {
            match node.kind() {
                "function_declaration" | "method_definition" | "arrow_function" => {
                    if let Some(name) = self.get_ts_function_name(&node, source) {
                        parsed.symbols.push(name.clone());
                        parsed.functions.push(FunctionNode {
                            name,
                            visibility: Visibility::Public,
                            params: vec![],
                            return_type: None,
                            generics: vec![],
                            is_async: node
                                .children(&mut node.walk())
                                .any(|c| c.kind() == "async"),
                            is_unsafe: false,
                            complexity: self.calculate_complexity(&node),
                            file_path: file_path.to_string(),
                            line_start: node.start_position().row as u32 + 1,
                            line_end: node.end_position().row as u32 + 1,
                            docstring: None,
                        });
                    }
                }
                "class_declaration" | "interface_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                            parsed.symbols.push(name.to_string());
                            parsed.structs.push(StructNode {
                                name: name.to_string(),
                                visibility: Visibility::Public,
                                generics: vec![],
                                file_path: file_path.to_string(),
                                line_start: node.start_position().row as u32 + 1,
                                line_end: node.end_position().row as u32 + 1,
                                docstring: None,
                            });
                        }
                    }
                }
                "import_statement" => {
                    if let Ok(text) = node.utf8_text(source.as_bytes()) {
                        parsed.imports.push(ImportNode {
                            path: text.to_string(),
                            alias: None,
                            items: vec![],
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn get_ts_function_name(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        node.child_by_field_name("name")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .map(|s| s.to_string())
    }

    /// Extract Python code structure (simplified)
    fn extract_python(
        &self,
        root: &tree_sitter::Node,
        source: &str,
        file_path: &str,
        parsed: &mut ParsedFile,
    ) -> Result<()> {
        let mut cursor = root.walk();

        for node in root.children(&mut cursor) {
            match node.kind() {
                "function_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                            let is_async = node
                                .children(&mut node.walk())
                                .any(|c| c.kind() == "async");
                            parsed.symbols.push(name.to_string());
                            parsed.functions.push(FunctionNode {
                                name: name.to_string(),
                                visibility: if name.starts_with('_') {
                                    Visibility::Private
                                } else {
                                    Visibility::Public
                                },
                                params: vec![],
                                return_type: None,
                                generics: vec![],
                                is_async,
                                is_unsafe: false,
                                complexity: self.calculate_complexity(&node),
                                file_path: file_path.to_string(),
                                line_start: node.start_position().row as u32 + 1,
                                line_end: node.end_position().row as u32 + 1,
                                docstring: None,
                            });
                        }
                    }
                }
                "class_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                            parsed.symbols.push(name.to_string());
                            parsed.structs.push(StructNode {
                                name: name.to_string(),
                                visibility: Visibility::Public,
                                generics: vec![],
                                file_path: file_path.to_string(),
                                line_start: node.start_position().row as u32 + 1,
                                line_end: node.end_position().row as u32 + 1,
                                docstring: None,
                            });
                        }
                    }
                }
                "import_statement" | "import_from_statement" => {
                    if let Ok(text) = node.utf8_text(source.as_bytes()) {
                        parsed.imports.push(ImportNode {
                            path: text.to_string(),
                            alias: None,
                            items: vec![],
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Extract Go code structure (simplified)
    fn extract_go(
        &self,
        root: &tree_sitter::Node,
        source: &str,
        file_path: &str,
        parsed: &mut ParsedFile,
    ) -> Result<()> {
        let mut cursor = root.walk();

        for node in root.children(&mut cursor) {
            match node.kind() {
                "function_declaration" | "method_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                            let visibility = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                Visibility::Public
                            } else {
                                Visibility::Private
                            };
                            parsed.symbols.push(name.to_string());
                            parsed.functions.push(FunctionNode {
                                name: name.to_string(),
                                visibility,
                                params: vec![],
                                return_type: None,
                                generics: vec![],
                                is_async: false,
                                is_unsafe: false,
                                complexity: self.calculate_complexity(&node),
                                file_path: file_path.to_string(),
                                line_start: node.start_position().row as u32 + 1,
                                line_end: node.end_position().row as u32 + 1,
                                docstring: None,
                            });
                        }
                    }
                }
                "type_declaration" => {
                    // Extract struct definitions
                    for child in node.children(&mut node.walk()) {
                        if child.kind() == "type_spec" {
                            if let Some(name_node) = child.child_by_field_name("name") {
                                if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                                    parsed.symbols.push(name.to_string());
                                    parsed.structs.push(StructNode {
                                        name: name.to_string(),
                                        visibility: Visibility::Public,
                                        generics: vec![],
                                        file_path: file_path.to_string(),
                                        line_start: child.start_position().row as u32 + 1,
                                        line_end: child.end_position().row as u32 + 1,
                                        docstring: None,
                                    });
                                }
                            }
                        }
                    }
                }
                "import_declaration" => {
                    if let Ok(text) = node.utf8_text(source.as_bytes()) {
                        parsed.imports.push(ImportNode {
                            path: text.to_string(),
                            alias: None,
                            items: vec![],
                            file_path: file_path.to_string(),
                            line: node.start_position().row as u32 + 1,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Convert parsed file to a code document for Meilisearch
    pub fn to_code_document(parsed: &ParsedFile, content: &str) -> CodeDocument {
        CodeDocument {
            id: crate::meilisearch::client::MeiliClient::path_to_id(&parsed.path),
            path: parsed.path.clone(),
            language: parsed.language.clone(),
            content: content.to_string(),
            symbols: parsed.symbols.clone(),
            imports: parsed
                .imports
                .iter()
                .map(|i| i.path.clone())
                .collect(),
            project_id: None,
            project_slug: None,
        }
    }
}

/// Result of parsing a file
#[derive(Debug, Clone)]
pub struct ParsedFile {
    pub path: String,
    pub language: String,
    pub hash: String,
    pub functions: Vec<FunctionNode>,
    pub structs: Vec<StructNode>,
    pub traits: Vec<TraitNode>,
    pub enums: Vec<EnumNode>,
    pub imports: Vec<ImportNode>,
    pub impl_blocks: Vec<ImplNode>,
    pub function_calls: Vec<FunctionCall>,
    pub symbols: Vec<String>,
}

/// Represents a function call found in code
#[derive(Debug, Clone)]
pub struct FunctionCall {
    /// The function making the call
    pub caller_id: String,
    /// The name of the function being called
    pub callee_name: String,
    /// Line where the call occurs
    pub line: u32,
}
