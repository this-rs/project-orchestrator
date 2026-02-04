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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // =========================================================================
    // SupportedLanguage Tests
    // =========================================================================

    #[test]
    fn test_from_extension_rust() {
        assert_eq!(
            SupportedLanguage::from_extension("rs"),
            Some(SupportedLanguage::Rust)
        );
    }

    #[test]
    fn test_from_extension_typescript() {
        assert_eq!(
            SupportedLanguage::from_extension("ts"),
            Some(SupportedLanguage::TypeScript)
        );
        assert_eq!(
            SupportedLanguage::from_extension("tsx"),
            Some(SupportedLanguage::TypeScript)
        );
    }

    #[test]
    fn test_from_extension_javascript_uses_ts_parser() {
        // JavaScript uses TypeScript parser
        assert_eq!(
            SupportedLanguage::from_extension("js"),
            Some(SupportedLanguage::TypeScript)
        );
        assert_eq!(
            SupportedLanguage::from_extension("jsx"),
            Some(SupportedLanguage::TypeScript)
        );
        assert_eq!(
            SupportedLanguage::from_extension("mjs"),
            Some(SupportedLanguage::TypeScript)
        );
        assert_eq!(
            SupportedLanguage::from_extension("cjs"),
            Some(SupportedLanguage::TypeScript)
        );
    }

    #[test]
    fn test_from_extension_python() {
        assert_eq!(
            SupportedLanguage::from_extension("py"),
            Some(SupportedLanguage::Python)
        );
        assert_eq!(
            SupportedLanguage::from_extension("pyi"),
            Some(SupportedLanguage::Python)
        );
    }

    #[test]
    fn test_from_extension_go() {
        assert_eq!(
            SupportedLanguage::from_extension("go"),
            Some(SupportedLanguage::Go)
        );
    }

    #[test]
    fn test_from_extension_java() {
        assert_eq!(
            SupportedLanguage::from_extension("java"),
            Some(SupportedLanguage::Java)
        );
    }

    #[test]
    fn test_from_extension_c() {
        assert_eq!(
            SupportedLanguage::from_extension("c"),
            Some(SupportedLanguage::C)
        );
        assert_eq!(
            SupportedLanguage::from_extension("h"),
            Some(SupportedLanguage::C)
        );
    }

    #[test]
    fn test_from_extension_cpp() {
        assert_eq!(
            SupportedLanguage::from_extension("cpp"),
            Some(SupportedLanguage::Cpp)
        );
        assert_eq!(
            SupportedLanguage::from_extension("cc"),
            Some(SupportedLanguage::Cpp)
        );
        assert_eq!(
            SupportedLanguage::from_extension("cxx"),
            Some(SupportedLanguage::Cpp)
        );
        assert_eq!(
            SupportedLanguage::from_extension("hpp"),
            Some(SupportedLanguage::Cpp)
        );
        assert_eq!(
            SupportedLanguage::from_extension("hxx"),
            Some(SupportedLanguage::Cpp)
        );
        assert_eq!(
            SupportedLanguage::from_extension("hh"),
            Some(SupportedLanguage::Cpp)
        );
    }

    #[test]
    fn test_from_extension_ruby() {
        assert_eq!(
            SupportedLanguage::from_extension("rb"),
            Some(SupportedLanguage::Ruby)
        );
        assert_eq!(
            SupportedLanguage::from_extension("rake"),
            Some(SupportedLanguage::Ruby)
        );
        assert_eq!(
            SupportedLanguage::from_extension("gemspec"),
            Some(SupportedLanguage::Ruby)
        );
    }

    #[test]
    fn test_from_extension_php() {
        assert_eq!(
            SupportedLanguage::from_extension("php"),
            Some(SupportedLanguage::Php)
        );
        assert_eq!(
            SupportedLanguage::from_extension("phtml"),
            Some(SupportedLanguage::Php)
        );
        assert_eq!(
            SupportedLanguage::from_extension("php5"),
            Some(SupportedLanguage::Php)
        );
        assert_eq!(
            SupportedLanguage::from_extension("php7"),
            Some(SupportedLanguage::Php)
        );
    }

    #[test]
    fn test_from_extension_kotlin() {
        assert_eq!(
            SupportedLanguage::from_extension("kt"),
            Some(SupportedLanguage::Kotlin)
        );
        assert_eq!(
            SupportedLanguage::from_extension("kts"),
            Some(SupportedLanguage::Kotlin)
        );
    }

    #[test]
    fn test_from_extension_swift() {
        assert_eq!(
            SupportedLanguage::from_extension("swift"),
            Some(SupportedLanguage::Swift)
        );
    }

    #[test]
    fn test_from_extension_bash() {
        assert_eq!(
            SupportedLanguage::from_extension("sh"),
            Some(SupportedLanguage::Bash)
        );
        assert_eq!(
            SupportedLanguage::from_extension("bash"),
            Some(SupportedLanguage::Bash)
        );
        assert_eq!(
            SupportedLanguage::from_extension("zsh"),
            Some(SupportedLanguage::Bash)
        );
    }

    #[test]
    fn test_from_extension_case_insensitive() {
        assert_eq!(
            SupportedLanguage::from_extension("RS"),
            Some(SupportedLanguage::Rust)
        );
        assert_eq!(
            SupportedLanguage::from_extension("Py"),
            Some(SupportedLanguage::Python)
        );
        assert_eq!(
            SupportedLanguage::from_extension("JAVA"),
            Some(SupportedLanguage::Java)
        );
    }

    #[test]
    fn test_from_extension_unsupported() {
        assert_eq!(SupportedLanguage::from_extension("txt"), None);
        assert_eq!(SupportedLanguage::from_extension("md"), None);
        assert_eq!(SupportedLanguage::from_extension("json"), None);
        assert_eq!(SupportedLanguage::from_extension("yaml"), None);
        assert_eq!(SupportedLanguage::from_extension(""), None);
    }

    #[test]
    fn test_as_str_all_languages() {
        assert_eq!(SupportedLanguage::Rust.as_str(), "rust");
        assert_eq!(SupportedLanguage::TypeScript.as_str(), "typescript");
        assert_eq!(SupportedLanguage::Python.as_str(), "python");
        assert_eq!(SupportedLanguage::Go.as_str(), "go");
        assert_eq!(SupportedLanguage::Java.as_str(), "java");
        assert_eq!(SupportedLanguage::C.as_str(), "c");
        assert_eq!(SupportedLanguage::Cpp.as_str(), "cpp");
        assert_eq!(SupportedLanguage::Ruby.as_str(), "ruby");
        assert_eq!(SupportedLanguage::Php.as_str(), "php");
        assert_eq!(SupportedLanguage::Kotlin.as_str(), "kotlin");
        assert_eq!(SupportedLanguage::Swift.as_str(), "swift");
        assert_eq!(SupportedLanguage::Bash.as_str(), "bash");
    }

    #[test]
    fn test_all_returns_12_languages() {
        let all = SupportedLanguage::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_all_contains_all_variants() {
        let all = SupportedLanguage::all();
        assert!(all.contains(&SupportedLanguage::Rust));
        assert!(all.contains(&SupportedLanguage::TypeScript));
        assert!(all.contains(&SupportedLanguage::Python));
        assert!(all.contains(&SupportedLanguage::Go));
        assert!(all.contains(&SupportedLanguage::Java));
        assert!(all.contains(&SupportedLanguage::C));
        assert!(all.contains(&SupportedLanguage::Cpp));
        assert!(all.contains(&SupportedLanguage::Ruby));
        assert!(all.contains(&SupportedLanguage::Php));
        assert!(all.contains(&SupportedLanguage::Kotlin));
        assert!(all.contains(&SupportedLanguage::Swift));
        assert!(all.contains(&SupportedLanguage::Bash));
    }

    // =========================================================================
    // CodeParser Tests
    // =========================================================================

    #[test]
    fn test_code_parser_creation() {
        let parser = CodeParser::new();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_parse_rust_file() {
        let mut parser = CodeParser::new().unwrap();
        let content = r#"
/// A simple function
fn hello() -> String {
    "Hello".to_string()
}

pub struct Config {
    pub name: String,
}
"#;
        let path = PathBuf::from("test.rs");
        let result = parser.parse_file(&path, content);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.language, "rust");
        assert!(!parsed.functions.is_empty());
        assert!(!parsed.structs.is_empty());
    }

    #[test]
    fn test_parse_python_file() {
        let mut parser = CodeParser::new().unwrap();
        let content = r#"
def hello():
    """A simple function"""
    return "Hello"

class Config:
    def __init__(self, name):
        self.name = name
"#;
        let path = PathBuf::from("test.py");
        let result = parser.parse_file(&path, content);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.language, "python");
    }

    #[test]
    fn test_parse_typescript_file() {
        let mut parser = CodeParser::new().unwrap();
        let content = r#"
function hello(): string {
    return "Hello";
}

interface Config {
    name: string;
}
"#;
        let path = PathBuf::from("test.ts");
        let result = parser.parse_file(&path, content);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.language, "typescript");
    }

    #[test]
    fn test_parse_go_file() {
        let mut parser = CodeParser::new().unwrap();
        let content = r#"
package main

func hello() string {
    return "Hello"
}

type Config struct {
    Name string
}
"#;
        let path = PathBuf::from("test.go");
        let result = parser.parse_file(&path, content);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.language, "go");
    }

    #[test]
    fn test_parse_unsupported_extension() {
        let mut parser = CodeParser::new().unwrap();
        let content = "some content";
        let path = PathBuf::from("test.txt");
        let result = parser.parse_file(&path, content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parsed_file_hash_changes_with_content() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.rs");

        let parsed1 = parser.parse_file(&path, "fn a() {}").unwrap();
        let parsed2 = parser.parse_file(&path, "fn b() {}").unwrap();

        assert_ne!(parsed1.hash, parsed2.hash);
    }

    #[test]
    fn test_parsed_file_hash_consistent() {
        let mut parser = CodeParser::new().unwrap();
        let path = PathBuf::from("test.rs");
        let content = "fn hello() {}";

        let parsed1 = parser.parse_file(&path, content).unwrap();
        let parsed2 = parser.parse_file(&path, content).unwrap();

        assert_eq!(parsed1.hash, parsed2.hash);
    }

    // =========================================================================
    // to_code_document Tests
    // =========================================================================

    #[test]
    fn test_to_code_document() {
        let mut parser = CodeParser::new().unwrap();
        let content = r#"
/// Says hello
fn hello() -> String {
    "Hello".to_string()
}
"#;
        let path = PathBuf::from("src/lib.rs");
        let parsed = parser.parse_file(&path, content).unwrap();

        let doc = CodeParser::to_code_document(&parsed, "project-123", "my-project");

        assert_eq!(doc.path, "src/lib.rs");
        assert_eq!(doc.language, "rust");
        assert_eq!(doc.project_id, "project-123");
        assert_eq!(doc.project_slug, "my-project");
    }

    #[test]
    fn test_to_code_document_collects_docstrings() {
        let mut parser = CodeParser::new().unwrap();
        let content = r#"
/// Function doc
fn foo() {}

/// Struct doc
struct Bar {}
"#;
        let path = PathBuf::from("test.rs");
        let parsed = parser.parse_file(&path, content).unwrap();

        let doc = CodeParser::to_code_document(&parsed, "id", "slug");

        // Should have collected docstrings
        assert!(doc.docstrings.contains("Function doc") || doc.docstrings.contains("Struct doc"));
    }

    // =========================================================================
    // FunctionCall Tests
    // =========================================================================

    #[test]
    fn test_function_call_struct() {
        let call = FunctionCall {
            caller_id: "main".to_string(),
            callee_name: "helper".to_string(),
            line: 42,
        };

        assert_eq!(call.caller_id, "main");
        assert_eq!(call.callee_name, "helper");
        assert_eq!(call.line, 42);
    }

    // =========================================================================
    // ParsedFile Tests
    // =========================================================================

    #[test]
    fn test_parsed_file_default_empty() {
        let parsed = ParsedFile {
            path: "test.rs".to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            functions: vec![],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            imports: vec![],
            impl_blocks: vec![],
            function_calls: vec![],
            symbols: vec![],
        };

        assert!(parsed.functions.is_empty());
        assert!(parsed.structs.is_empty());
        assert!(parsed.imports.is_empty());
    }
}
