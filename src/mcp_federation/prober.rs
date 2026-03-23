//! Tool Prober — safely probe read-only external tools to learn their behavior.
//!
//! After introspection classifies tools, the prober makes minimal test calls
//! to **read-only** tools (Query/Search) to gather runtime information:
//!
//! - Response latency
//! - Response shape (Object, Array, Scalar)
//! - Pagination support
//! - Error format
//!
//! ## Safety
//!
//! **The prober NEVER calls tools classified as Create, Mutation, Delete, or Unknown.**
//! This is enforced at the type level (check before call) and runtime (guard assertion).
//!
//! ## Architecture
//!
//! ```text
//! DiscoveredTool (category = Query/Search)
//!   │
//!   ▼
//! ToolProber.probe()
//!   ├── generate_probe_args(input_schema)  → minimal JSON args
//!   ├── client.call_tool(name, args)       → timed call
//!   └── analyze_response(value)            → ToolProfile
//!   │
//!   ▼
//! ToolProfile { latency_ms, response_shape, pagination, ... }
//! ```

use chrono::Utc;
use serde_json::Value;
use std::time::Instant;
use tracing::{debug, warn};

use super::client::McpClient;
use super::discovery::{DiscoveredTool, ResponseShape, ToolProfile};

// ─────────────────────────────────────────────────────────────────────────────
// Probe argument generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate minimal probe arguments from a JSON Schema.
///
/// For each **required** parameter, generates the smallest valid value:
/// - `string` → `""`
/// - `integer` / `number` → `0`
/// - `boolean` → `false`
/// - `object` → `{}`
/// - `array` → `[]`
/// - `null` / unknown → `null`
///
/// Optional parameters are omitted entirely.
pub fn generate_probe_args(input_schema: &Value) -> Option<Value> {
    let properties = input_schema.get("properties")?.as_object()?;
    let required = input_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<&str>>())
        .unwrap_or_default();

    if required.is_empty() && properties.is_empty() {
        // No params at all — call with empty object
        return Some(serde_json::json!({}));
    }

    let mut args = serde_json::Map::new();

    for param_name in &required {
        let param_def = match properties.get(*param_name) {
            Some(def) => def,
            None => continue,
        };

        let value = generate_minimal_value(param_def);
        args.insert(param_name.to_string(), value);
    }

    Some(Value::Object(args))
}

/// Generate a minimal value for a JSON Schema property definition.
fn generate_minimal_value(schema: &Value) -> Value {
    // Check for enum first — use the first value
    if let Some(enum_values) = schema.get("enum").and_then(|e| e.as_array()) {
        if let Some(first) = enum_values.first() {
            return first.clone();
        }
    }

    // Check for const
    if let Some(const_val) = schema.get("const") {
        return const_val.clone();
    }

    // Check for default
    if let Some(default) = schema.get("default") {
        return default.clone();
    }

    let type_str = schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("string");

    match type_str {
        "string" => {
            // Check for minLength
            let min_len = schema
                .get("minLength")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            if min_len > 0 {
                Value::String("a".repeat(min_len))
            } else {
                Value::String(String::new())
            }
        }
        "integer" | "number" => {
            // Use minimum if specified
            if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64()) {
                serde_json::json!(min)
            } else {
                serde_json::json!(0)
            }
        }
        "boolean" => Value::Bool(false),
        "object" => Value::Object(serde_json::Map::new()),
        "array" => Value::Array(vec![]),
        "null" => Value::Null,
        _ => Value::Null,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Analyze a probe response to determine its shape.
fn analyze_response_shape(value: &Value) -> ResponseShape {
    match value {
        Value::Object(_) => ResponseShape::Object,
        Value::Array(_) => ResponseShape::Array,
        _ => ResponseShape::Scalar,
    }
}

/// Detect pagination support from a response.
///
/// Looks for common pagination indicators:
/// - `next_cursor`, `nextCursor`, `cursor`, `next_page`
/// - `has_more`, `hasMore`, `has_next`
/// - `offset`, `page`, `total`, `count` + data array
fn detect_pagination(value: &Value) -> bool {
    if let Some(obj) = value.as_object() {
        let pagination_keys = [
            "next_cursor",
            "nextCursor",
            "cursor",
            "next_page",
            "nextPage",
            "has_more",
            "hasMore",
            "has_next",
            "hasNext",
            "next_token",
            "nextToken",
        ];
        for key in &pagination_keys {
            if obj.contains_key(*key) {
                return true;
            }
        }

        // Also detect: has both a data array + offset/total/page
        let has_data_array = obj.values().any(|v| v.is_array());
        let has_pagination_meta = obj.contains_key("offset")
            || obj.contains_key("page")
            || obj.contains_key("total")
            || obj.contains_key("total_count")
            || obj.contains_key("totalCount");

        if has_data_array && has_pagination_meta {
            return true;
        }
    }
    false
}

/// Detect the error format from an error response.
fn detect_error_format(value: &Value) -> Option<String> {
    if let Some(obj) = value.as_object() {
        if obj.contains_key("jsonrpc") || obj.contains_key("error") {
            return Some("json_rpc".to_string());
        }
        if obj.contains_key("message") && (obj.contains_key("code") || obj.contains_key("status")) {
            return Some("structured".to_string());
        }
    }
    if value.is_string() {
        return Some("plain_text".to_string());
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Prober
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the tool prober.
#[derive(Debug, Clone)]
pub struct ProberConfig {
    /// Timeout for each probe call in milliseconds.
    pub probe_timeout_ms: u64,
    /// Whether probing is enabled at all.
    pub enabled: bool,
}

impl Default for ProberConfig {
    fn default() -> Self {
        Self {
            probe_timeout_ms: 10_000, // 10 seconds per probe
            enabled: true,
        }
    }
}

/// Probes read-only external tools to gather runtime information.
pub struct ToolProber {
    config: ProberConfig,
}

impl ToolProber {
    pub fn new(config: ProberConfig) -> Self {
        Self { config }
    }

    /// Probe a single tool.
    ///
    /// **Safety**: Returns `ToolProfile::not_probed()` if the tool is NOT
    /// classified as Query or Search. This is the critical safety guard.
    pub async fn probe(&self, client: &dyn McpClient, tool: &DiscoveredTool) -> ToolProfile {
        // SAFETY GUARD: NEVER probe mutations
        if !tool.category.is_safe_to_probe() {
            debug!(
                tool = %tool.name,
                category = ?tool.category,
                "Skipping probe — unsafe category"
            );
            return ToolProfile::not_probed();
        }

        if !self.config.enabled {
            debug!(tool = %tool.name, "Probing disabled");
            return ToolProfile::not_probed();
        }

        // Generate minimal probe arguments
        let args = generate_probe_args(&tool.input_schema);

        debug!(
            tool = %tool.name,
            args = ?args,
            "Probing tool"
        );

        // Timed call
        let start = Instant::now();
        let timeout = std::time::Duration::from_millis(self.config.probe_timeout_ms);

        let result = tokio::time::timeout(timeout, client.call_tool(&tool.name, args)).await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(Ok(response)) => {
                let response_shape = analyze_response_shape(&response);
                let pagination = detect_pagination(&response);

                debug!(
                    tool = %tool.name,
                    latency_ms = latency_ms,
                    shape = ?response_shape,
                    pagination = pagination,
                    "Probe succeeded"
                );

                ToolProfile {
                    latency_ms,
                    response_shape,
                    pagination,
                    error_format: None,
                    probed_at: Utc::now(),
                }
            }
            Ok(Err(e)) => {
                warn!(
                    tool = %tool.name,
                    error = %e,
                    latency_ms = latency_ms,
                    "Probe returned error"
                );

                // Try to extract error format from the error message
                let error_format =
                    detect_error_format(&serde_json::json!({"message": e.to_string()}));

                ToolProfile {
                    latency_ms,
                    response_shape: ResponseShape::Error,
                    pagination: false,
                    error_format,
                    probed_at: Utc::now(),
                }
            }
            Err(_) => {
                warn!(
                    tool = %tool.name,
                    timeout_ms = self.config.probe_timeout_ms,
                    "Probe timed out"
                );

                ToolProfile {
                    latency_ms: self.config.probe_timeout_ms,
                    response_shape: ResponseShape::Error,
                    pagination: false,
                    error_format: None,
                    probed_at: Utc::now(),
                }
            }
        }
    }

    /// Probe multiple tools in parallel.
    ///
    /// Only probes tools with safe categories (Query/Search).
    /// Returns a map of tool name → ToolProfile.
    pub async fn probe_batch(&self, client: &dyn McpClient, tools: &mut [DiscoveredTool]) {
        let safe_indices: Vec<usize> = tools
            .iter()
            .enumerate()
            .filter(|(_, t)| t.category.is_safe_to_probe() && self.config.enabled)
            .map(|(i, _)| i)
            .collect();

        if safe_indices.is_empty() {
            debug!("No safe tools to probe");
            return;
        }

        debug!(count = safe_indices.len(), "Probing safe tools");

        // Probe sequentially to avoid overwhelming the server
        // (parallel probing can be enabled later with semaphore)
        for idx in safe_indices {
            let profile = self.probe(client, &tools[idx]).await;
            tools[idx].profile = Some(profile);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::discovery::InferredCategory;
    use super::*;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use serde_json::json;

    // ── Mock MCP client for testing ───────────────────────────────────────

    #[derive(Debug)]
    struct MockProbeClient {
        /// Response to return for any call_tool.
        response: Value,
        /// Simulated latency in ms.
        latency_ms: u64,
    }

    #[async_trait]
    impl McpClient for MockProbeClient {
        async fn initialize(&self) -> Result<super::super::client::InitializeResult> {
            unimplemented!()
        }
        async fn initialized_notification(&self) -> Result<()> {
            unimplemented!()
        }
        async fn tools_list(&self) -> Result<Vec<super::super::client::McpToolDef>> {
            unimplemented!()
        }
        async fn call_tool(&self, _name: &str, _arguments: Option<Value>) -> Result<Value> {
            if self.latency_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
            }
            Ok(self.response.clone())
        }
        async fn ping(&self) -> Result<()> {
            Ok(())
        }
        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
        fn transport_name(&self) -> &'static str {
            "mock"
        }
    }

    #[derive(Debug)]
    struct ErrorProbeClient;

    #[async_trait]
    impl McpClient for ErrorProbeClient {
        async fn initialize(&self) -> Result<super::super::client::InitializeResult> {
            unimplemented!()
        }
        async fn initialized_notification(&self) -> Result<()> {
            unimplemented!()
        }
        async fn tools_list(&self) -> Result<Vec<super::super::client::McpToolDef>> {
            unimplemented!()
        }
        async fn call_tool(&self, _name: &str, _arguments: Option<Value>) -> Result<Value> {
            Err(anyhow!("permission denied"))
        }
        async fn ping(&self) -> Result<()> {
            Ok(())
        }
        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
        fn transport_name(&self) -> &'static str {
            "mock"
        }
    }

    fn make_discovered_tool(name: &str, category: InferredCategory) -> DiscoveredTool {
        DiscoveredTool {
            name: name.to_string(),
            fqn: format!("test::{}", name),
            description: format!("{} description", name),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
            category,
            embedding: None,
            similar_internal: vec![],
            profile: None,
        }
    }

    // ── generate_probe_args tests ─────────────────────────────────────────

    #[test]
    fn test_probe_args_required_string() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "limit": { "type": "integer" }
            },
            "required": ["query"]
        });
        let args = generate_probe_args(&schema).unwrap();
        assert_eq!(args.get("query").unwrap(), "");
        assert!(args.get("limit").is_none()); // Optional, omitted
    }

    #[test]
    fn test_probe_args_all_types() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" },
                "score": { "type": "number" },
                "active": { "type": "boolean" },
                "data": { "type": "object" },
                "items": { "type": "array" }
            },
            "required": ["name", "count", "score", "active", "data", "items"]
        });
        let args = generate_probe_args(&schema).unwrap();
        assert_eq!(args.get("name").unwrap(), "");
        assert_eq!(args.get("count").unwrap(), 0);
        assert_eq!(args.get("score").unwrap(), 0);
        assert_eq!(args.get("active").unwrap(), false);
        assert!(args.get("data").unwrap().is_object());
        assert!(args.get("items").unwrap().is_array());
    }

    #[test]
    fn test_probe_args_with_enum() {
        let schema = json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["list", "create", "get"] }
            },
            "required": ["action"]
        });
        let args = generate_probe_args(&schema).unwrap();
        assert_eq!(args.get("action").unwrap(), "list"); // First enum value
    }

    #[test]
    fn test_probe_args_with_default() {
        let schema = json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer", "default": 10 }
            },
            "required": ["limit"]
        });
        let args = generate_probe_args(&schema).unwrap();
        assert_eq!(args.get("limit").unwrap(), 10);
    }

    #[test]
    fn test_probe_args_no_properties() {
        let schema = json!({ "type": "object" });
        // No properties key at all
        assert!(generate_probe_args(&schema).is_none());
    }

    #[test]
    fn test_probe_args_empty_schema() {
        let schema = json!({
            "type": "object",
            "properties": {}
        });
        let args = generate_probe_args(&schema).unwrap();
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_probe_args_min_length_string() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "minLength": 3 }
            },
            "required": ["query"]
        });
        let args = generate_probe_args(&schema).unwrap();
        assert_eq!(args.get("query").unwrap(), "aaa");
    }

    // ── Safety guard tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_probe_mutation_returns_not_probed() {
        let client = MockProbeClient {
            response: json!({"data": "should never see this"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("update_user", InferredCategory::Mutation);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::NotProbed);
        assert_eq!(profile.latency_ms, 0);
    }

    #[tokio::test]
    async fn test_probe_delete_returns_not_probed() {
        let client = MockProbeClient {
            response: json!({"data": "should never see this"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("delete_user", InferredCategory::Delete);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::NotProbed);
    }

    #[tokio::test]
    async fn test_probe_create_returns_not_probed() {
        let client = MockProbeClient {
            response: json!({"data": "should never see this"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("create_issue", InferredCategory::Create);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::NotProbed);
    }

    #[tokio::test]
    async fn test_probe_unknown_returns_not_probed() {
        let client = MockProbeClient {
            response: json!({"data": "should never see this"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("execute", InferredCategory::Unknown);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::NotProbed);
    }

    // ── Successful probe tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_probe_query_object_response() {
        let client = MockProbeClient {
            response: json!({"id": "123", "name": "test"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("get_user", InferredCategory::Query);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::Object);
        assert!(!profile.pagination);
    }

    #[tokio::test]
    async fn test_probe_query_array_response() {
        let client = MockProbeClient {
            response: json!([{"id": 1}, {"id": 2}]),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("list_items", InferredCategory::Query);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::Array);
    }

    #[tokio::test]
    async fn test_probe_search_tool() {
        let client = MockProbeClient {
            response: json!({"results": [], "total": 0}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("search_docs", InferredCategory::Search);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::Object);
        // Has data array + total → pagination detected
        assert!(profile.pagination);
    }

    #[tokio::test]
    async fn test_probe_detects_cursor_pagination() {
        let client = MockProbeClient {
            response: json!({"items": [1, 2, 3], "next_cursor": "abc123"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("list_all", InferredCategory::Query);

        let profile = prober.probe(&client, &tool).await;
        assert!(profile.pagination);
    }

    // ── Error handling tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_probe_error_response() {
        let client = ErrorProbeClient;
        let prober = ToolProber::new(ProberConfig::default());
        let tool = make_discovered_tool("get_data", InferredCategory::Query);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::Error);
    }

    #[tokio::test]
    async fn test_probe_disabled() {
        let client = MockProbeClient {
            response: json!({"data": "test"}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig {
            enabled: false,
            ..Default::default()
        });
        let tool = make_discovered_tool("get_data", InferredCategory::Query);

        let profile = prober.probe(&client, &tool).await;
        assert_eq!(profile.response_shape, ResponseShape::NotProbed);
    }

    // ── Batch probe tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_probe_batch_only_safe_tools() {
        let client = MockProbeClient {
            response: json!({"ok": true}),
            latency_ms: 0,
        };
        let prober = ToolProber::new(ProberConfig::default());

        let mut tools = vec![
            make_discovered_tool("get_data", InferredCategory::Query),
            make_discovered_tool("create_item", InferredCategory::Create),
            make_discovered_tool("search_docs", InferredCategory::Search),
            make_discovered_tool("delete_stuff", InferredCategory::Delete),
        ];

        prober.probe_batch(&client, &mut tools).await;

        // Only Query and Search should have profiles
        assert!(tools[0].profile.is_some());
        assert!(tools[1].profile.is_none()); // Create — not probed
        assert!(tools[2].profile.is_some());
        assert!(tools[3].profile.is_none()); // Delete — not probed

        assert_ne!(
            tools[0].profile.as_ref().unwrap().response_shape,
            ResponseShape::NotProbed
        );
    }

    // ── Pagination detection tests ────────────────────────────────────────

    #[test]
    fn test_detect_pagination_cursor() {
        assert!(detect_pagination(
            &json!({"data": [], "next_cursor": "abc"})
        ));
        assert!(detect_pagination(
            &json!({"items": [], "nextCursor": "xyz"})
        ));
        assert!(detect_pagination(&json!({"has_more": true})));
        assert!(detect_pagination(&json!({"hasNext": false})));
    }

    #[test]
    fn test_detect_pagination_offset() {
        assert!(detect_pagination(&json!({"items": [1, 2], "total": 100})));
        assert!(detect_pagination(&json!({"data": [], "offset": 0})));
    }

    #[test]
    fn test_no_pagination() {
        assert!(!detect_pagination(&json!({"id": 1, "name": "test"})));
        assert!(!detect_pagination(&json!("just a string")));
        assert!(!detect_pagination(&json!([1, 2, 3])));
    }

    // ── Response shape tests ──────────────────────────────────────────────

    #[test]
    fn test_response_shape_detection() {
        assert_eq!(analyze_response_shape(&json!({})), ResponseShape::Object);
        assert_eq!(analyze_response_shape(&json!([])), ResponseShape::Array);
        assert_eq!(
            analyze_response_shape(&json!("hello")),
            ResponseShape::Scalar
        );
        assert_eq!(analyze_response_shape(&json!(42)), ResponseShape::Scalar);
        assert_eq!(analyze_response_shape(&json!(true)), ResponseShape::Scalar);
        assert_eq!(analyze_response_shape(&json!(null)), ResponseShape::Scalar);
    }

    // ── Error format detection ────────────────────────────────────────────

    #[test]
    fn test_error_format_detection() {
        assert_eq!(
            detect_error_format(&json!({"jsonrpc": "2.0", "error": {"code": -1}})),
            Some("json_rpc".to_string())
        );
        assert_eq!(
            detect_error_format(&json!({"message": "not found", "code": 404})),
            Some("structured".to_string())
        );
        assert_eq!(
            detect_error_format(&json!("plain error message")),
            Some("plain_text".to_string())
        );
        assert_eq!(detect_error_format(&json!(42)), None);
    }
}
