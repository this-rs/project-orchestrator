//! Tree-sitter based code parser
//!
//! Supports multiple programming languages with full AST extraction.

pub mod helpers;
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
    Java,
    C,
    Cpp,
    Ruby,
    Php,
    Kotlin,
    Swift,
    Bash,
}

impl SupportedLanguage {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "ts" | "tsx" => Some(Self::TypeScript),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::TypeScript), // Use TS parser for JS
            "py" | "pyi" => Some(Self::Python),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(Self::Cpp),
            "rb" | "rake" | "gemspec" => Some(Self::Ruby),
            "php" | "phtml" | "php5" | "php7" => Some(Self::Php),
            "kt" | "kts" => Some(Self::Kotlin),
            "swift" => Some(Self::Swift),
            "sh" | "bash" | "zsh" => Some(Self::Bash),
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
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Self::Php => tree_sitter_php::LANGUAGE_PHP.into(),
            Self::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Self::Swift => tree_sitter_swift::LANGUAGE.into(),
            Self::Bash => tree_sitter_bash::LANGUAGE.into(),
        }
    }

    /// Get the language name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Python => "python",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Ruby => "ruby",
            Self::Php => "php",
            Self::Kotlin => "kotlin",
            Self::Swift => "swift",
            Self::Bash => "bash",
        }
    }

    /// Get all supported languages
    pub fn all() -> &'static [Self] {
        &[
            Self::Rust,
            Self::TypeScript,
            Self::Python,
            Self::Go,
            Self::Java,
            Self::C,
            Self::Cpp,
            Self::Ruby,
            Self::Php,
            Self::Kotlin,
            Self::Swift,
            Self::Bash,
        ]
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

        for lang in SupportedLanguage::all() {
            let mut parser = Parser::new();
            parser
                .set_language(&lang.tree_sitter_language())
                .context(format!("Failed to set language for {:?}", lang))?;
            parsers.insert(*lang, parser);
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
                languages::rust::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::TypeScript => {
                languages::typescript::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Python => {
                languages::python::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Go => {
                languages::go::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Java => {
                languages::java::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::C => {
                languages::c::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Cpp => {
                languages::cpp::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Ruby => {
                languages::ruby::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Php => {
                languages::php::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Kotlin => {
                languages::kotlin::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Swift => {
                languages::swift::extract(&root, content, &path_str, &mut parsed)?;
            }
            SupportedLanguage::Bash => {
                languages::bash::extract(&root, content, &path_str, &mut parsed)?;
            }
        }

        Ok(parsed)
    }

    /// Convert parsed file to a code document for Meilisearch
    ///
    /// Creates a lightweight document for semantic search containing:
    /// - Symbol names for keyword search
    /// - Docstrings for semantic/natural language search
    /// - Function signatures for quick reference
    pub fn to_code_document(
        parsed: &ParsedFile,
        project_id: &str,
        project_slug: &str,
    ) -> CodeDocument {
        // Collect all docstrings
        let mut docstrings = Vec::new();

        for func in &parsed.functions {
            if let Some(ref doc) = func.docstring {
                docstrings.push(doc.clone());
            }
        }
        for s in &parsed.structs {
            if let Some(ref doc) = s.docstring {
                docstrings.push(doc.clone());
            }
        }
        for t in &parsed.traits {
            if let Some(ref doc) = t.docstring {
                docstrings.push(doc.clone());
            }
        }
        for e in &parsed.enums {
            if let Some(ref doc) = e.docstring {
                docstrings.push(doc.clone());
            }
        }

        // Build function signatures
        let signatures: Vec<String> = parsed
            .functions
            .iter()
            .map(|f| {
                let params = f
                    .params
                    .iter()
                    .map(|p| {
                        if let Some(ref t) = p.type_name {
                            format!("{}: {}", p.name, t)
                        } else {
                            p.name.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let ret = f
                    .return_type
                    .as_ref()
                    .map(|r| format!(" -> {}", r))
                    .unwrap_or_default();
                let async_kw = if f.is_async { "async " } else { "" };
                format!("{}fn {}({}){}", async_kw, f.name, params, ret)
            })
            .collect();

        CodeDocument {
            id: crate::meilisearch::client::MeiliClient::path_to_id(&parsed.path),
            path: parsed.path.clone(),
            language: parsed.language.clone(),
            symbols: parsed.symbols.clone(),
            docstrings: docstrings.join("\n\n"),
            signatures,
            imports: parsed.imports.iter().map(|i| i.path.clone()).collect(),
            project_id: project_id.to_string(),
            project_slug: project_slug.to_string(),
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
