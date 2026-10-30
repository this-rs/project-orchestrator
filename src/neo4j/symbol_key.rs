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
}
