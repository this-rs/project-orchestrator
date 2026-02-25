//! Compact formatter for MCP tool responses.
//!
//! Converts JSON `Value` into a YAML-like compact text format to minimize
//! token usage when the LLM reads tool results. Savings are ~40-50% compared
//! to `serde_json::to_string_pretty`.
//!
//! Rules:
//! - No quotes around keys or simple string values
//! - Null fields are omitted entirely
//! - Empty strings are omitted
//! - Simple arrays are inlined: `tags: [a, b, c]`
//! - Object arrays use `- ` prefix (YAML-style)
//! - Nested objects are indented by 2 spaces

use serde_json::Value;

/// Convert a JSON value to a compact, token-efficient text representation.
///
/// This is the single entry point — called from `McpServer::handle_tools_call`
/// instead of `serde_json::to_string_pretty`.
pub fn json_to_compact(value: &Value) -> String {
    let mut out = String::with_capacity(256);
    write_value(&mut out, value, 0, false);
    // Trim trailing newline if any
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Write a value at the given indentation level.
/// `inline` means we're writing inside an already-started line (e.g. after `- `).
fn write_value(out: &mut String, value: &Value, indent: usize, inline: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_string(out, s),
        Value::Array(arr) => write_array(out, arr, indent, inline),
        Value::Object(map) => write_object(out, map, indent, inline),
    }
}

/// Write a string value. No quotes unless the string contains characters
/// that would be ambiguous in our format.
fn write_string(out: &mut String, s: &str) {
    if s.is_empty() {
        out.push_str("\"\"");
        return;
    }

    // Needs quoting if: contains newlines, starts/ends with whitespace,
    // looks like it could be confused with other types, or contains `: `
    let needs_quoting = s.contains('\n')
        || s.starts_with(' ')
        || s.ends_with(' ')
        || s.starts_with('[')
        || s.starts_with('{')
        || s.starts_with("- ")
        || s == "true"
        || s == "false"
        || s == "null";

    if needs_quoting {
        // Use JSON-style quoting for safety
        out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                _ => out.push(ch),
            }
        }
        out.push('"');
    } else {
        out.push_str(s);
    }
}

/// Write an array value.
fn write_array(out: &mut String, arr: &[Value], indent: usize, inline: bool) {
    if arr.is_empty() {
        out.push_str("[]");
        return;
    }

    // Check if all items are scalars (strings, numbers, bools) — inline them
    let all_scalar = arr
        .iter()
        .all(|v| matches!(v, Value::String(_) | Value::Number(_) | Value::Bool(_)));

    if all_scalar {
        out.push('[');
        for (i, v) in arr.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_value(out, v, 0, true);
        }
        out.push(']');
        return;
    }

    // Complex array: each item on its own line with `- ` prefix
    let item_indent = indent;

    for (i, v) in arr.iter().enumerate() {
        if i > 0 || !inline {
            if i > 0 {
                out.push('\n');
            }
            push_indent(out, item_indent);
        }
        out.push_str("- ");

        match v {
            Value::Object(map) => {
                write_object(out, map, item_indent + 2, true);
            }
            _ => {
                write_value(out, v, item_indent + 2, true);
            }
        }
    }
}

/// Write an object value.
/// `inline` means the first key should be written on the current line.
fn write_object(
    out: &mut String,
    map: &serde_json::Map<String, Value>,
    indent: usize,
    inline: bool,
) {
    if map.is_empty() {
        out.push_str("{}");
        return;
    }

    let mut first = true;

    for (key, value) in map {
        // Skip null and empty string values
        if value.is_null() {
            continue;
        }
        if let Value::String(s) = value {
            if s.is_empty() {
                continue;
            }
        }

        if first && inline {
            // Write on the current line (after `- ` for example)
            first = false;
        } else {
            if !first {
                out.push('\n');
            }
            push_indent(out, indent);
            first = false;
        }

        out.push_str(key);

        match value {
            Value::Object(_) | Value::Array(_) if is_complex(value) => {
                // Complex nested value: start on next line, indented
                out.push_str(":\n");
                write_value(out, value, indent + 2, false);
            }
            _ => {
                out.push_str(": ");
                write_value(out, value, indent + 2, true);
            }
        }
    }
}

/// Check if a value is "complex" (needs its own block).
/// Objects and arrays of objects are complex. Simple arrays are not.
fn is_complex(value: &Value) -> bool {
    match value {
        Value::Object(map) => !map.is_empty(),
        Value::Array(arr) => {
            !arr.is_empty()
                && !arr
                    .iter()
                    .all(|v| matches!(v, Value::String(_) | Value::Number(_) | Value::Bool(_)))
        }
        _ => false,
    }
}

/// Push `n` spaces for indentation.
fn push_indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_scalars() {
        assert_eq!(json_to_compact(&json!(42)), "42");
        assert_eq!(json_to_compact(&json!(true)), "true");
        assert_eq!(json_to_compact(&json!(2.71)), "2.71");
        assert_eq!(json_to_compact(&json!("hello")), "hello");
    }

    #[test]
    fn test_null_standalone() {
        assert_eq!(json_to_compact(&json!(null)), "null");
    }

    #[test]
    fn test_simple_object() {
        let val = json!({
            "id": "abc-123",
            "title": "My Task",
            "status": "completed",
            "priority": 3
        });
        let out = json_to_compact(&val);
        assert!(out.contains("id: abc-123"));
        assert!(out.contains("title: My Task"));
        assert!(out.contains("status: completed"));
        assert!(out.contains("priority: 3"));
        // No JSON artifacts
        assert!(!out.contains('"'));
        assert!(!out.contains('{'));
        assert!(!out.contains('}'));
    }

    #[test]
    fn test_null_fields_omitted() {
        let val = json!({
            "id": "abc",
            "completed_at": null,
            "closed_at": null,
            "title": "test"
        });
        let out = json_to_compact(&val);
        assert!(out.contains("id: abc"));
        assert!(out.contains("title: test"));
        assert!(!out.contains("completed_at"));
        assert!(!out.contains("closed_at"));
    }

    #[test]
    fn test_empty_string_fields_omitted() {
        let val = json!({
            "id": "abc",
            "description": "",
            "title": "test"
        });
        let out = json_to_compact(&val);
        assert!(!out.contains("description"));
    }

    #[test]
    fn test_scalar_array_inline() {
        let val = json!({
            "tags": ["rust", "api", "mcp"]
        });
        let out = json_to_compact(&val);
        assert_eq!(out, "tags: [rust, api, mcp]");
    }

    #[test]
    fn test_empty_array() {
        let val = json!({ "items": [] });
        let out = json_to_compact(&val);
        assert_eq!(out, "items: []");
    }

    #[test]
    fn test_object_array() {
        let val = json!({
            "tasks": [
                {"id": "1", "title": "First"},
                {"id": "2", "title": "Second"}
            ]
        });
        let out = json_to_compact(&val);
        assert!(out.contains("tasks:"));
        assert!(out.contains("- id: 1"));
        assert!(out.contains("  title: First"));
        assert!(out.contains("- id: 2"));
        assert!(out.contains("  title: Second"));
    }

    #[test]
    fn test_nested_object() {
        let val = json!({
            "milestone": {
                "id": "abc",
                "title": "v1.0"
            },
            "progress": {
                "total": 5,
                "completed": 3
            }
        });
        let out = json_to_compact(&val);
        assert!(out.contains("milestone:"));
        assert!(out.contains("  id: abc"));
        assert!(out.contains("  title: v1.0"));
        assert!(out.contains("progress:"));
        assert!(out.contains("  total: 5"));
        assert!(out.contains("  completed: 3"));
    }

    #[test]
    fn test_string_needing_quotes() {
        let val = json!({
            "content": "line1\nline2",
            "flag": "true",
            "normal": "hello world"
        });
        let out = json_to_compact(&val);
        // "true" as a string needs quoting to not be confused with bool
        assert!(out.contains("flag: \"true\""));
        // Newlines are escaped
        assert!(out.contains("content: \"line1\\nline2\""));
        // Normal string has no quotes
        assert!(out.contains("normal: hello world"));
    }

    #[test]
    fn test_top_level_array_of_objects() {
        let val = json!([
            {"id": "1", "name": "Project A"},
            {"id": "2", "name": "Project B"}
        ]);
        let out = json_to_compact(&val);
        assert!(out.contains("- id: 1"));
        assert!(out.contains("  name: Project A"));
        assert!(out.contains("- id: 2"));
        assert!(out.contains("  name: Project B"));
    }

    #[test]
    fn test_top_level_scalar_array() {
        let val = json!(["a", "b", "c"]);
        let out = json_to_compact(&val);
        assert_eq!(out, "[a, b, c]");
    }

    #[test]
    fn test_empty_object() {
        assert_eq!(json_to_compact(&json!({})), "{}");
    }

    #[test]
    fn test_boolean_fields() {
        let val = json!({
            "active": true,
            "archived": false
        });
        let out = json_to_compact(&val);
        assert!(out.contains("active: true"));
        assert!(out.contains("archived: false"));
    }

    #[test]
    fn test_realistic_task_response() {
        let val = json!({
            "id": "4ee35887-fe28-4536-9c55-411c3559dbb6",
            "title": "Implement dual-mode MCP",
            "description": "Support both direct and HTTP modes",
            "status": "in_progress",
            "priority": 3,
            "tags": ["mcp", "architecture"],
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-16T10:00:00Z",
            "completed_at": null
        });
        let compact = json_to_compact(&val);
        let json_pretty = serde_json::to_string_pretty(&val).unwrap();

        // Compact should be significantly shorter
        assert!(
            compact.len() < json_pretty.len(),
            "compact ({}) should be shorter than pretty JSON ({})",
            compact.len(),
            json_pretty.len()
        );

        // No JSON structural characters (except for inline arrays)
        assert!(!compact.contains('{'));
        assert!(!compact.contains('}'));

        // Null field should be omitted
        assert!(!compact.contains("completed_at"));

        // Content preserved
        assert!(compact.contains("id: 4ee35887-fe28-4536-9c55-411c3559dbb6"));
        assert!(compact.contains("tags: [mcp, architecture]"));
    }

    #[test]
    fn test_realistic_list_response() {
        let val = json!({
            "items": [
                {
                    "id": "abc",
                    "title": "Task 1",
                    "status": "completed",
                    "tags": ["backend"],
                    "completed_at": null
                },
                {
                    "id": "def",
                    "title": "Task 2",
                    "status": "pending",
                    "tags": [],
                    "description": null
                }
            ],
            "total": 2,
            "limit": 50,
            "offset": 0
        });
        let compact = json_to_compact(&val);

        // Null fields should be omitted in nested objects
        assert!(!compact.contains("completed_at"));
        assert!(!compact.contains("description"));

        // Structure preserved
        assert!(compact.contains("total: 2"));
        assert!(compact.contains("- id: abc"));
        assert!(compact.contains("  title: Task 1"));
    }

    #[test]
    fn test_deeply_nested() {
        let val = json!({
            "plan": {
                "tasks": [
                    {
                        "id": "t1",
                        "steps": [
                            {"order": 1, "desc": "Step A"},
                            {"order": 2, "desc": "Step B"}
                        ]
                    }
                ]
            }
        });
        let out = json_to_compact(&val);
        assert!(out.contains("plan:"), "missing plan: in:\n{out}");
        assert!(out.contains("  tasks:"), "missing tasks: in:\n{out}");
        assert!(out.contains("    - id: t1"), "missing - id: in:\n{out}");
        assert!(out.contains("      steps:"), "missing steps: in:\n{out}");
        // Keys are BTreeMap-ordered (alphabetical): desc before order
        assert!(
            out.contains("        - desc: Step A"),
            "missing - desc: Step A in:\n{out}"
        );
        assert!(
            out.contains("          order: 1"),
            "missing order: 1 in:\n{out}"
        );
    }

    #[test]
    fn test_success_json_value() {
        // The { "success": true } pattern from empty HTTP responses
        let val = json!({"success": true});
        let out = json_to_compact(&val);
        assert_eq!(out, "success: true");
    }

    #[test]
    fn test_token_savings_estimate() {
        // Simulate a typical API response with 5 tasks
        let tasks: Vec<Value> = (0..5)
            .map(|i| {
                json!({
                    "id": format!("id-{}", i),
                    "title": format!("Task number {}", i),
                    "status": "pending",
                    "priority": i,
                    "tags": ["backend", "api"],
                    "created_at": "2024-01-15T10:00:00Z",
                    "completed_at": null,
                    "description": null
                })
            })
            .collect();
        let val = json!({ "items": tasks, "total": 5 });

        let compact = json_to_compact(&val);
        let pretty = serde_json::to_string_pretty(&val).unwrap();

        let savings = 100.0 - (compact.len() as f64 / pretty.len() as f64 * 100.0);
        // We expect at least 30% savings
        assert!(
            savings > 30.0,
            "Expected >30% savings, got {:.1}% (compact: {} chars, pretty: {} chars)",
            savings,
            compact.len(),
            pretty.len()
        );
    }
}
