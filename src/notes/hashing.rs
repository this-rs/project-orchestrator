//! Semantic hashing for note anchoring
//!
//! Generates stable hashes that detect meaningful code changes while
//! being tolerant to formatting changes (whitespace, comments).

use sha2::{Digest, Sha256};

/// Hash a function signature (name + parameters + return type)
///
/// Ignores: body, comments, attributes
/// Includes: function name, parameter names and types, return type
pub fn hash_function_signature(
    name: &str,
    params: &[(String, Option<String>)], // (param_name, param_type)
    return_type: Option<&str>,
    is_async: bool,
    is_unsafe: bool,
) -> String {
    let mut hasher = Sha256::new();

    // Prefix markers for function properties
    if is_async {
        hasher.update(b"async:");
    }
    if is_unsafe {
        hasher.update(b"unsafe:");
    }

    // Function name
    hasher.update(name.as_bytes());
    hasher.update(b"(");

    // Parameters (sorted by name for stability)
    let mut sorted_params: Vec<_> = params.iter().collect();
    sorted_params.sort_by(|a, b| a.0.cmp(&b.0));

    for (i, (param_name, param_type)) in sorted_params.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(param_name.as_bytes());
        if let Some(ty) = param_type {
            hasher.update(b":");
            hasher.update(normalize_type(ty).as_bytes());
        }
    }

    hasher.update(b")");

    // Return type
    if let Some(ret) = return_type {
        hasher.update(b"->");
        hasher.update(normalize_type(ret).as_bytes());
    }

    hex::encode(hasher.finalize())
}

/// Hash a function body (normalized)
///
/// Ignores: whitespace, comments, string contents (replaced with placeholder)
/// Includes: structure, variable names, calls, control flow
pub fn hash_function_body(body: &str) -> String {
    let normalized = normalize_code(body);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    hex::encode(hasher.finalize())
}

/// Hash a struct signature (name + fields + types)
pub fn hash_struct_signature(
    name: &str,
    fields: &[(String, String, bool)], // (field_name, field_type, is_public)
    generics: &[String],
) -> String {
    let mut hasher = Sha256::new();

    // Struct name
    hasher.update(b"struct:");
    hasher.update(name.as_bytes());

    // Generics (sorted)
    if !generics.is_empty() {
        let mut sorted_generics = generics.to_vec();
        sorted_generics.sort();
        hasher.update(b"<");
        for (i, g) in sorted_generics.iter().enumerate() {
            if i > 0 {
                hasher.update(b",");
            }
            hasher.update(g.as_bytes());
        }
        hasher.update(b">");
    }

    hasher.update(b"{");

    // Fields (sorted by name for stability)
    let mut sorted_fields: Vec<_> = fields.iter().collect();
    sorted_fields.sort_by(|a, b| a.0.cmp(&b.0));

    for (i, (field_name, field_type, is_public)) in sorted_fields.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        if *is_public {
            hasher.update(b"pub:");
        }
        hasher.update(field_name.as_bytes());
        hasher.update(b":");
        hasher.update(normalize_type(field_type).as_bytes());
    }

    hasher.update(b"}");

    hex::encode(hasher.finalize())
}

/// Hash a file structure (list of symbols + imports)
pub fn hash_file_structure(
    symbols: &[String],
    imports: &[String],
) -> String {
    let mut hasher = Sha256::new();

    // Sorted symbols
    let mut sorted_symbols = symbols.to_vec();
    sorted_symbols.sort();
    hasher.update(b"symbols:");
    for sym in &sorted_symbols {
        hasher.update(sym.as_bytes());
        hasher.update(b";");
    }

    // Sorted imports
    let mut sorted_imports = imports.to_vec();
    sorted_imports.sort();
    hasher.update(b"imports:");
    for imp in &sorted_imports {
        hasher.update(imp.as_bytes());
        hasher.update(b";");
    }

    hex::encode(hasher.finalize())
}

/// Hash a module structure
pub fn hash_module_structure(
    module_path: &str,
    file_hashes: &[(String, String)], // (file_path, file_hash)
) -> String {
    let mut hasher = Sha256::new();

    hasher.update(b"module:");
    hasher.update(module_path.as_bytes());
    hasher.update(b"{");

    // Sort by file path for stability
    let mut sorted: Vec<_> = file_hashes.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    for (path, hash) in sorted {
        hasher.update(path.as_bytes());
        hasher.update(b"=");
        hasher.update(hash.as_bytes());
        hasher.update(b";");
    }

    hasher.update(b"}");

    hex::encode(hasher.finalize())
}

/// Calculate similarity between two hashes
///
/// Returns a value between 0.0 (completely different) and 1.0 (identical).
/// Uses character-level comparison for simple similarity detection.
pub fn similarity_score(hash1: &str, hash2: &str) -> f64 {
    if hash1 == hash2 {
        return 1.0;
    }

    if hash1.is_empty() || hash2.is_empty() {
        return 0.0;
    }

    // Compare character by character
    let chars1: Vec<char> = hash1.chars().collect();
    let chars2: Vec<char> = hash2.chars().collect();
    let max_len = chars1.len().max(chars2.len());
    let min_len = chars1.len().min(chars2.len());

    let mut matching = 0;
    for i in 0..min_len {
        if chars1[i] == chars2[i] {
            matching += 1;
        }
    }

    matching as f64 / max_len as f64
}

/// Normalize a type string for consistent hashing
///
/// Removes unnecessary whitespace, normalizes generic bounds, etc.
fn normalize_type(ty: &str) -> String {
    let mut result = String::new();
    let mut last_was_space = false;

    for c in ty.chars() {
        if c.is_whitespace() {
            if !last_was_space && !result.is_empty() {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(c);
            last_was_space = false;
        }
    }

    result.trim().to_string()
}

/// Normalize code for consistent hashing
///
/// - Removes comments (// and /* */)
/// - Normalizes whitespace
/// - Replaces string literals with placeholders
fn normalize_code(code: &str) -> String {
    let mut result = String::new();
    let mut chars = code.chars().peekable();
    let mut in_string = false;
    let mut in_char = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut last_was_space = false;

    while let Some(c) = chars.next() {
        // Handle line comments
        if !in_string && !in_char && !in_block_comment && c == '/' {
            if chars.peek() == Some(&'/') {
                chars.next();
                in_line_comment = true;
                continue;
            } else if chars.peek() == Some(&'*') {
                chars.next();
                in_block_comment = true;
                continue;
            }
        }

        // Handle block comment end
        if in_block_comment && c == '*' && chars.peek() == Some(&'/') {
            chars.next();
            in_block_comment = false;
            continue;
        }

        // Handle line comment end
        if in_line_comment && c == '\n' {
            in_line_comment = false;
            // Add a space to preserve some structure
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
            continue;
        }

        // Skip comment content
        if in_line_comment || in_block_comment {
            continue;
        }

        // Handle string literals
        if !in_char && c == '"' {
            in_string = !in_string;
            if !in_string {
                result.push_str("\"STR\"");
            }
            continue;
        }

        // Handle char literals
        if !in_string && c == '\'' {
            in_char = !in_char;
            if !in_char {
                result.push_str("'C'");
            }
            continue;
        }

        // Skip string/char content (replaced with placeholder)
        if in_string || in_char {
            continue;
        }

        // Normalize whitespace
        if c.is_whitespace() {
            if !last_was_space && !result.is_empty() {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(c);
            last_was_space = false;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_signature_hash_same_code() {
        let hash1 = hash_function_signature(
            "process",
            &[
                ("input".to_string(), Some("&str".to_string())),
                ("count".to_string(), Some("usize".to_string())),
            ],
            Some("Result<String>"),
            false,
            false,
        );

        let hash2 = hash_function_signature(
            "process",
            &[
                ("input".to_string(), Some("&str".to_string())),
                ("count".to_string(), Some("usize".to_string())),
            ],
            Some("Result<String>"),
            false,
            false,
        );

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_function_signature_hash_different_params() {
        let hash1 = hash_function_signature(
            "process",
            &[("input".to_string(), Some("&str".to_string()))],
            Some("Result<String>"),
            false,
            false,
        );

        let hash2 = hash_function_signature(
            "process",
            &[("input".to_string(), Some("String".to_string()))],
            Some("Result<String>"),
            false,
            false,
        );

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_function_signature_hash_param_order_independent() {
        let hash1 = hash_function_signature(
            "process",
            &[
                ("a".to_string(), Some("i32".to_string())),
                ("b".to_string(), Some("i32".to_string())),
            ],
            None,
            false,
            false,
        );

        let hash2 = hash_function_signature(
            "process",
            &[
                ("b".to_string(), Some("i32".to_string())),
                ("a".to_string(), Some("i32".to_string())),
            ],
            None,
            false,
            false,
        );

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_function_body_hash_whitespace_independent() {
        let code1 = r#"
            let x = 1;
            let y = 2;
            x + y
        "#;

        let code2 = "let x = 1; let y = 2; x + y";

        let hash1 = hash_function_body(code1);
        let hash2 = hash_function_body(code2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_function_body_hash_comment_independent() {
        let code1 = r#"
            // This is a comment
            let x = 1;
            /* Another comment */
            x + 1
        "#;

        let code2 = "let x = 1; x + 1";

        let hash1 = hash_function_body(code1);
        let hash2 = hash_function_body(code2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_function_body_hash_string_independent() {
        let code1 = r#"println!("Hello, World!");"#;
        let code2 = r#"println!("Goodbye, World!");"#;

        let hash1 = hash_function_body(code1);
        let hash2 = hash_function_body(code2);

        // String contents are replaced with placeholder, so hashes should match
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_function_body_hash_different_logic() {
        let code1 = "let x = 1; x + 1";
        let code2 = "let x = 1; x * 2";

        let hash1 = hash_function_body(code1);
        let hash2 = hash_function_body(code2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_struct_signature_hash() {
        let hash1 = hash_struct_signature(
            "User",
            &[
                ("name".to_string(), "String".to_string(), true),
                ("age".to_string(), "u32".to_string(), false),
            ],
            &["T".to_string()],
        );

        let hash2 = hash_struct_signature(
            "User",
            &[
                ("age".to_string(), "u32".to_string(), false),
                ("name".to_string(), "String".to_string(), true),
            ],
            &["T".to_string()],
        );

        // Field order shouldn't matter
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_file_structure_hash() {
        let hash1 = hash_file_structure(
            &["func1".to_string(), "func2".to_string()],
            &["std::io".to_string()],
        );

        let hash2 = hash_file_structure(
            &["func2".to_string(), "func1".to_string()],
            &["std::io".to_string()],
        );

        // Order shouldn't matter
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_similarity_identical() {
        let hash = "abc123def456";
        assert_eq!(similarity_score(hash, hash), 1.0);
    }

    #[test]
    fn test_similarity_different() {
        let hash1 = "aaaaaaaaaaaaaaaa";
        let hash2 = "bbbbbbbbbbbbbbbb";
        assert_eq!(similarity_score(hash1, hash2), 0.0);
    }

    #[test]
    fn test_similarity_partial() {
        let hash1 = "aaaaabbbbb";
        let hash2 = "aaaaaccccc";
        let score = similarity_score(hash1, hash2);
        assert!(score > 0.0 && score < 1.0);
        assert_eq!(score, 0.5); // First 5 chars match
    }

    #[test]
    fn test_normalize_type() {
        assert_eq!(normalize_type("  String  "), "String");
        assert_eq!(normalize_type("Vec< String >"), "Vec< String >");
        assert_eq!(normalize_type("Result<T,   E>"), "Result<T, E>");
    }

    #[test]
    fn test_normalize_code_removes_comments() {
        let code = "let x = 1; // comment\nlet y = 2;";
        let normalized = normalize_code(code);
        assert!(!normalized.contains("comment"));
        assert!(normalized.contains("let x = 1"));
        assert!(normalized.contains("let y = 2"));
    }

    #[test]
    fn test_normalize_code_handles_block_comments() {
        let code = "let x = /* inline comment */ 1;";
        let normalized = normalize_code(code);
        assert!(!normalized.contains("inline comment"));
        assert!(normalized.contains("let x ="));
        assert!(normalized.contains("1"));
    }
}
