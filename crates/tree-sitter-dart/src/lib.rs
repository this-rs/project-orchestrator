//! Dart grammar for tree-sitter.
//!
//! Built from nielsenko/tree-sitter-dart v0.1.0 (grammar ABI patched to 14).
//! Compatible with tree-sitter 0.24.x.
//!
//! ```
//! let code = r#"class Foo { void bar() {} }"#;
//! let mut parser = tree_sitter::Parser::new();
//! parser
//!     .set_language(&tree_sitter_dart::LANGUAGE.into())
//!     .expect("Error loading Dart parser");
//! let tree = parser.parse(code, None).unwrap();
//! assert!(!tree.root_node().has_error());
//! ```

use tree_sitter_language::LanguageFn;

extern "C" {
    fn tree_sitter_dart() -> *const ();
}

/// The tree-sitter [`LanguageFn`] for the Dart grammar.
pub const LANGUAGE: LanguageFn = unsafe { LanguageFn::from_raw(tree_sitter_dart) };

/// The content of the [`node-types.json`] file for this grammar.
pub const NODE_TYPES: &str = include_str!("node-types.json");
