//! Stable identity for code symbols, independent of source position.
//!
//! The legacy identity of a function was `file_path:name:line_start`
//! (`code.rs`), so any edit that moved a function created a new node and
//! abandoned the old one — the code graph became a journal of versions
//! instead of a photograph of the code. `stable_symbol_key` removes the
//! line from the identity, qualifies a method by the block that owns it and
//! appends a parameter-type signature only when needed to keep overloads
//! distinct.
//!
//! This module is the ONE grammar: the write paths of `code.rs` and the
//! backfill of `data_migrations.rs` both call these functions, so a key
//! written at parse time and a key backfilled on an existing node cannot
//! drift apart.

use super::models::{Parameter, SymbolOwner};

/// Compute the stable key of a function or method.
///
/// Grammar: `<file_path>::<qualifier><name>[(<type>,...)]` where
/// - `<qualifier>` is empty for a free function, `Type::` for an inherent
///   method and `<Type as Trait>::` for a trait method (Rust's fully
///   qualified syntax — `fmt` of `Display for A` and `fmt` of `Debug for A`
///   are two methods, and `Type::method` alone would merge them);
/// - the parameter suffix is present only when there are parameters, so that
///   two overloads in the same file remain distinct. Parameter *names* are
///   not part of the identity (renaming a parameter is not a new symbol);
///   only the types are.
pub fn stable_symbol_key(
    file_path: &str,
    owner: Option<&SymbolOwner>,
    name: &str,
    params: &[Parameter],
) -> String {
    let qualifier = owner
        .map(|o| format!("{}::", o.qualifier()))
        .unwrap_or_default();
    let types: Vec<&str> = params
        .iter()
        .map(|p| p.type_name.as_deref().unwrap_or("?"))
        .collect();
    if types.is_empty() {
        format!("{}::{}{}", file_path, qualifier, name)
    } else {
        format!("{}::{}{}({})", file_path, qualifier, name, types.join(","))
    }
}

/// The same key, computed from the properties of a `Function` node as the
/// graph stores them (`params` is the JSON of `Vec<Parameter>`, written by
/// `code.rs`; the owner comes from `owner_type`/`owner_trait`).
///
/// Measured on the production graph (31/10/2026): 490 672 `Function` nodes,
/// every one of them with a `params` string starting with `[`, none NULL or
/// empty. The lenient fallback below (an unreadable `params` counts as no
/// parameters) therefore changes nothing today; it exists so that one
/// corrupted node cannot stall the whole backfill.
pub fn stable_symbol_key_from_graph(
    file_path: &str,
    owner: Option<&SymbolOwner>,
    name: &str,
    params_json: Option<&str>,
) -> String {
    let params: Vec<Parameter> = params_json
        .filter(|s| !s.trim().is_empty())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    stable_symbol_key(file_path, owner, name, &params)
}

/// Stable key of a type declaration — struct, trait or enum.
///
/// Grammar: `<file_path>::<name>`. A type cannot be overloaded in the same file,
/// so no signature suffix is needed; a rename is a new type, and that is correct.
pub fn stable_type_key(file_path: &str, name: &str) -> String {
    format!("{}::{}", file_path, name)
}

/// Stable key of the placeholder node of a trait the project does not
/// declare (`std::fmt::Debug`, `serde::Serialize`...), created by
/// `upsert_impl` so that `IMPLEMENTS_TRAIT` has a target.
///
/// Grammar: `external:trait:<full_path>`. It has no file and no line, so its
/// legacy id was already stable; the key repeats it so that every `Trait`
/// node carries a key — a key `::Debug` built from its empty `file_path`
/// would merge two external traits with one simple name. Measured on the
/// production graph (31/10/2026): 227 of the 4 387 `Trait` nodes are external.
pub fn stable_external_trait_key(full_path: &str) -> String {
    format!("external:trait:{}", full_path)
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
/// Grammar: `<file_path>::import:<path>` for the first import of a path in a
/// file, `<file_path>::import:<path>#<n>` (n ≥ 2) for a genuine duplicate in
/// the same file (`use std::fmt;` at the top and again in `mod tests`).
///
/// The line is NOT part of it. Measured on the production graph
/// (31/10/2026): 289 491 `Import` nodes for 250 936 distinct `(file, path)`
/// pairs — 38 555 nodes (13 %) exist only because the line moved; on `.rs`
/// files alone, 9 348 pairs sit on several lines although Rust rejects the
/// same `use` twice in one scope. A key carrying the line is the legacy id
/// under another name, and would keep that journal of versions.
pub fn stable_import_key(file_path: &str, path: &str, occurrence: u32) -> String {
    if occurrence == 0 {
        format!("{}::import:{}", file_path, path)
    } else {
        format!("{}::import:{}#{}", file_path, path, occurrence + 1)
    }
}

/// The owner of a function, among the candidate blocks of the same file:
/// the blocks whose lines enclose the function's, chained from the outermost
/// to the innermost (`Foo::Config` for `class Foo { class Config { load } }`
/// — the innermost alone would merge `Foo::Config::load` with
/// `Bar::Config::load`). The trait, if any, is the innermost block's. `None`
/// when no block encloses it — a free function.
///
/// This is the ONE rule, used at parse time for every language
/// (`parser::assign_owners`) and by the backfill of existing nodes
/// (`data_migrations.rs`), which deduces the owner of a legacy `Function`
/// node by line inclusion in the `Impl` and `Struct` nodes of its file — the
/// graph has no `DEFINED_IN` edge yet (measured: 0 on 31/10/2026).
pub fn enclosing_owner<I>(line_start: u32, line_end: u32, candidates: I) -> Option<SymbolOwner>
where
    I: IntoIterator<Item = (u32, u32, SymbolOwner)>,
{
    let mut enclosing: Vec<(u32, u32, SymbolOwner)> = candidates
        .into_iter()
        .filter(|(start, end, _)| *start <= line_start && line_end <= *end)
        .collect();
    // Outer to inner: earliest start first; on a tie, the widest first.
    enclosing.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let (_, _, innermost) = enclosing.last()?.clone();
    let chain: Vec<&str> = enclosing
        .iter()
        .map(|(_, _, o)| o.type_name.as_str())
        .collect();
    Some(SymbolOwner {
        type_name: chain.join("::"),
        trait_name: innermost.trait_name,
    })
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

    fn inherent(ty: &str) -> SymbolOwner {
        SymbolOwner::new(ty, None::<&str>)
    }

    fn of_trait(ty: &str, tr: &str) -> SymbolOwner {
        SymbolOwner::new(ty, Some(tr))
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
        assert_eq!(
            stable_impl_key("src/a.rs", "Parser", Some("")),
            "src/a.rs::impl Parser"
        );
    }

    #[test]
    fn an_import_key_drops_the_line_and_ranks_true_duplicates() {
        assert_eq!(
            stable_import_key("src/a.rs", "std::fmt", 0),
            "src/a.rs::import:std::fmt"
        );
        // The import moved by one line (an edit above it): same import.
        assert_eq!(
            stable_import_key("src/a.rs", "std::fmt", 0),
            stable_import_key("src/a.rs", "std::fmt", 0)
        );
        // The same path imported twice in one file (top + `mod tests`): two facts.
        assert_eq!(
            stable_import_key("src/a.rs", "std::fmt", 1),
            "src/a.rs::import:std::fmt#2"
        );
    }

    #[test]
    fn key_without_params_matches_public_contract() {
        assert_eq!(
            stable_symbol_key("src/a.rs", None, "build", &[]),
            "src/a.rs::build"
        );
    }

    #[test]
    fn key_uses_param_types_not_names() {
        let k1 = stable_symbol_key("src/a.rs", None, "foo", &[p("x", Some("i32"))]);
        let k2 = stable_symbol_key("src/a.rs", None, "foo", &[p("renamed", Some("i32"))]);
        assert_eq!(k1, k2, "a parameter rename must not change identity");
    }

    #[test]
    fn overloads_stay_distinct() {
        let k1 = stable_symbol_key("src/a.rs", None, "foo", &[p("x", Some("i32"))]);
        let k2 = stable_symbol_key("src/a.rs", None, "foo", &[p("x", Some("bool"))]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn unknown_type_is_a_placeholder() {
        let k = stable_symbol_key("src/a.rs", None, "foo", &[p("x", None)]);
        assert_eq!(k, "src/a.rs::foo(?)");
    }

    // The defect the review found: `impl A { fn new() }` and `impl B { fn new() }`
    // in one file shared the key `src/a.rs::new`. Measured on the production
    // graph (31/10/2026): 7 168 (file, signature) groups mixed functions whose
    // enclosing Impl differs — 19 885 keys needed where one would have been set.
    #[test]
    fn a_method_is_qualified_by_its_owner() {
        let a = stable_symbol_key("src/a.rs", Some(&inherent("A")), "new", &[]);
        let b = stable_symbol_key("src/a.rs", Some(&inherent("B")), "new", &[]);
        let free = stable_symbol_key("src/a.rs", None, "new", &[]);
        assert_eq!(a, "src/a.rs::A::new");
        assert_eq!(b, "src/a.rs::B::new");
        assert_eq!(free, "src/a.rs::new");
        assert_ne!(a, b);
        assert_ne!(a, free);
    }

    // `Type::method` is not enough for trait methods: `Display for A` and
    // `Debug for A` both define `fmt(&self, &mut Formatter)` in the same file.
    #[test]
    fn two_trait_impls_of_one_type_keep_their_methods_apart() {
        let params = [p("self", Some("&self")), p("f", Some("&mut Formatter"))];
        let display =
            stable_symbol_key("src/a.rs", Some(&of_trait("A", "Display")), "fmt", &params);
        let debug = stable_symbol_key("src/a.rs", Some(&of_trait("A", "Debug")), "fmt", &params);
        assert_eq!(
            display,
            "src/a.rs::<A as Display>::fmt(&self,&mut Formatter)"
        );
        assert_eq!(debug, "src/a.rs::<A as Debug>::fmt(&self,&mut Formatter)");
        assert_ne!(display, debug);
    }

    // The graph stores `''` for "no trait" on Impl nodes; the owner built from
    // such a node must be the inherent owner, not `<A as >`.
    #[test]
    fn a_blank_trait_is_an_inherent_owner() {
        assert_eq!(SymbolOwner::new("A", Some("")), inherent("A"));
        assert_eq!(SymbolOwner::new("A", Some("  ")), inherent("A"));
        assert_eq!(inherent("A").qualifier(), "A");
        assert_eq!(of_trait("A", "Debug").qualifier(), "<A as Debug>");
    }

    // The backfill reads `params` as the JSON `code.rs` wrote; the result must
    // be byte-identical to the key the write path computes from the struct.
    #[test]
    fn the_key_from_graph_properties_equals_the_key_from_the_struct() {
        let params = [p("x", Some("i32")), p("y", None)];
        let json = serde_json::to_string(&params).unwrap();
        let owner = of_trait("A", "Debug");
        assert_eq!(
            stable_symbol_key_from_graph("src/a.rs", Some(&owner), "foo", Some(&json)),
            stable_symbol_key("src/a.rs", Some(&owner), "foo", &params)
        );
        assert_eq!(
            stable_symbol_key_from_graph("src/a.rs", None, "foo", Some("[]")),
            "src/a.rs::foo"
        );
        // Lenient on a corrupted or missing `params`: no suffix, never a stall.
        assert_eq!(
            stable_symbol_key_from_graph("src/a.rs", None, "foo", None),
            "src/a.rs::foo"
        );
        assert_eq!(
            stable_symbol_key_from_graph("src/a.rs", None, "foo", Some("not json")),
            "src/a.rs::foo"
        );
    }

    #[test]
    fn the_owner_is_the_chain_of_enclosing_blocks() {
        let candidates = vec![
            (1, 100, inherent("Outer")),
            (10, 50, inherent("Inner")),
            (60, 90, of_trait("Other", "Debug")),
        ];
        // Nested blocks chain outer to inner (review 2, point 5: two nested
        // `Config` classes in two outer classes must not share a key).
        assert_eq!(
            enclosing_owner(20, 30, candidates.clone()),
            Some(inherent("Outer::Inner"))
        );
        assert_eq!(
            enclosing_owner(70, 80, candidates.clone()),
            Some(of_trait("Outer::Other", "Debug"))
        );
        assert_eq!(
            enclosing_owner(55, 58, candidates.clone()),
            Some(inherent("Outer"))
        );
        assert_eq!(enclosing_owner(200, 210, candidates), None);
        // A function that straddles the block's end is not inside it.
        assert_eq!(enclosing_owner(45, 55, vec![(10, 50, inherent("X"))]), None);
        // Same start: the wider block is the outer one of the chain.
        assert_eq!(
            enclosing_owner(
                5,
                5,
                vec![(5, 5, inherent("Tight")), (5, 9, inherent("Wide"))]
            ),
            Some(inherent("Wide::Tight"))
        );
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
                owner: Some(SymbolOwner::new("Builder", None::<&str>)),
            }
        }

        let at_line_10 = node(10);
        let at_line_12 = node(12);
        assert_eq!(
            stable_symbol_key(
                &at_line_10.file_path,
                at_line_10.owner.as_ref(),
                &at_line_10.name,
                &at_line_10.params
            ),
            stable_symbol_key(
                &at_line_12.file_path,
                at_line_12.owner.as_ref(),
                &at_line_12.name,
                &at_line_12.params
            ),
            "an edit above the function must not change its identity"
        );
    }
}
