//! HCL/Terraform grammar for tree-sitter.
//!
//! Built from tree-sitter-grammars/tree-sitter-hcl v1.1.0 (grammar ABI 14).
//! Compatible with tree-sitter 0.24.x.
//!
//! ```
//! let code = r#"resource "aws_instance" "web" { ami = "ami-123" }"#;
//! let mut parser = tree_sitter::Parser::new();
//! parser
//!     .set_language(&tree_sitter_hcl::LANGUAGE.into())
//!     .expect("Error loading HCL parser");
//! let tree = parser.parse(code, None).unwrap();
//! assert!(!tree.root_node().has_error());
//! ```

use tree_sitter_language::LanguageFn;

extern "C" {
    fn tree_sitter_hcl() -> *const ();
}

/// The tree-sitter [`LanguageFn`] for the HCL grammar.
pub const LANGUAGE: LanguageFn = unsafe { LanguageFn::from_raw(tree_sitter_hcl) };

/// The content of the [`node-types.json`] file for this grammar.
pub const NODE_TYPES: &str = include_str!("node-types.json");
