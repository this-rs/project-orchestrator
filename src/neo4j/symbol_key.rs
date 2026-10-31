//! Stable identity for code symbols, independent of source position.
//!
//! The legacy identity of a function was `file_path:name:line_start`
//! (`code.rs`), so any edit that moved a function created a new node and
//! abandoned the old one — the code graph became a journal of versions
//! instead of a photograph of the code. `stable_symbol_key` removes the
//! line from the identity and appends a parameter-type signature only when
//! needed to keep overloads distinct.

use super::models::Parameter;

/// Compute the stable key of a function or method.
///
/// Grammar: `<file_path>::<name>` when there are no parameters, or
/// `<file_path>::<name>(<type>,...)` when there are, so that two overloads
/// in the same file remain distinct. Parameter *names* are not part of the
/// identity (renaming a parameter is not a new symbol); only the types are.
pub fn stable_symbol_key(file_path: &str, name: &str, params: &[Parameter]) -> String {
    let types: Vec<&str> = params
        .iter()
        .map(|p| p.type_name.as_deref().unwrap_or("?"))
        .collect();
    if types.is_empty() {
        format!("{}::{}", file_path, name)
    } else {
        format!("{}::{}({})", file_path, name, types.join(","))
    }
}

/// Stable key of a type declaration — struct, trait or enum.
///
/// Grammar: `<file_path>::<name>`. A type cannot be overloaded in the same file,
/// so no signature suffix is needed; a rename is a new type, and that is correct.
pub fn stable_type_key(file_path: &str, name: &str) -> String {
    format!("{}::{}", file_path, name)
}

/// Stable key of an `impl` block.
///
/// Grammar: `<file_path>::impl <Type>` or `<file_path>::impl <Trait> for <Type>`.
/// The line is not part of it: moving an impl must not create a second impl of
/// the same type. Two impls of the same trait for the same type in one file are
/// already a compile error in Rust, so no disambiguating suffix is needed.
pub fn stable_impl_key(file_path: &str, for_type: &str, trait_name: Option<&str>) -> String {
    match trait_name {
        Some(t) if !t.trim().is_empty() => format!("{}::impl {} for {}", file_path, t, for_type),
        _ => format!("{}::impl {}", file_path, for_type),
    }
}

/// Stable key of an import.
///
/// Grammar: `<file_path>::import:<line>:<path>`. Unlike symbols, the **line stays**
/// here: two identical `use` statements on two lines are two distinct facts
/// (a duplicate import is a finding, not a duplicate node), and the line is what
/// tells them apart.
pub fn stable_import_key(file_path: &str, line: u32, path: &str) -> String {
    format!("{}::import:{}:{}", file_path, line, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str, ty: Option<&str>) -> Parameter {
        Parameter {
            name: name.to_string(),
            type_name: ty.map(|s| s.to_string()),
        }
    }

    #[test]
    fn a_type_key_does_not_depend_on_its_position() {
        assert_eq!(stable_type_key("src/a.rs", "Parser"), "src/a.rs::Parser");
        // The same type seen through two mount points is two files, not one:
        // that is T7's job, and the key must not paper over it.
        assert_ne!(
            stable_type_key("src/a.rs", "Parser"),
            stable_type_key("src/b.rs", "Parser")
        );
    }

    #[test]
    fn an_impl_key_names_the_trait_and_the_type() {
        assert_eq!(
            stable_impl_key("src/a.rs", "Parser", Some("Debug")),
            "src/a.rs::impl Debug for Parser"
        );
        assert_eq!(
            stable_impl_key("src/a.rs", "Parser", None),
            "src/a.rs::impl Parser"
        );
        // An empty trait name is no trait at all, not "impl  for Parser".
        assert_eq!(
            stable_impl_key("src/a.rs", "Parser", Some("  ")),
            "src/a.rs::impl Parser"
        );
    }

    #[test]
    fn an_import_key_keeps_the_line_on_purpose() {
        assert_eq!(
            stable_import_key("src/a.rs", 12, "std::fmt"),
            "src/a.rs::import:12:std::fmt"
        );
        assert_ne!(
            stable_import_key("src/a.rs", 12, "std::fmt"),
            stable_import_key("src/a.rs", 13, "std::fmt"),
            "two identical imports on two lines are two facts"
        );
    }

    #[test]
    fn key_without_params_matches_public_contract() {
        assert_eq!(
            stable_symbol_key("src/a.rs", "build", &[]),
            "src/a.rs::build"
        );
    }

    #[test]
    fn key_uses_param_types_not_names() {
        let k1 = stable_symbol_key("src/a.rs", "foo", &[p("x", Some("i32"))]);
        let k2 = stable_symbol_key("src/a.rs", "foo", &[p("renamed", Some("i32"))]);
        assert_eq!(k1, k2, "a parameter rename must not change identity");
    }

    #[test]
    fn overloads_stay_distinct() {
        let k1 = stable_symbol_key("src/a.rs", "foo", &[p("x", Some("i32"))]);
        let k2 = stable_symbol_key("src/a.rs", "foo", &[p("x", Some("bool"))]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn unknown_type_is_a_placeholder() {
        let k = stable_symbol_key("src/a.rs", "foo", &[p("x", None)]);
        assert_eq!(k, "src/a.rs::foo(?)");
    }

    // The regression this module exists for: a function parsed at two
    // different lines (an edit above it) is the SAME symbol. If someone ever
    // puts the line back into the identity, this test fails.
    #[test]
    fn moving_a_function_does_not_change_its_identity() {
        use crate::neo4j::models::{FunctionNode, Visibility};

        fn node(line: u32) -> FunctionNode {
            FunctionNode {
                name: "build".to_string(),
                visibility: Visibility::default(),
                params: vec![p("x", Some("i32"))],
                return_type: None,
                generics: vec![],
                is_async: false,
                is_unsafe: false,
                complexity: 1,
                file_path: "src/a.rs".to_string(),
                line_start: line,
                line_end: line + 5,
                docstring: None,
            }
        }

        let at_line_10 = node(10);
        let at_line_12 = node(12);
        assert_eq!(
            stable_symbol_key(&at_line_10.file_path, &at_line_10.name, &at_line_10.params),
            stable_symbol_key(&at_line_12.file_path, &at_line_12.name, &at_line_12.params),
            "an edit above the function must not change its identity"
        );
    }
}
