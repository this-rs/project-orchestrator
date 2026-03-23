//! MCP Federation Security Policy
//!
//! Enforces security constraints on external MCP tool calls:
//! - Mutation blocking (allow_mutations = false by default)
//! - Server allowlist (allowed_servers)
//! - Rate limiting (max_calls_per_minute)
//! - Credential isolation (prevent forwarding PO auth headers)

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::discovery::InferredCategory;

/// Security policy violation — returned when a call is blocked.
#[derive(Debug, Clone, Serialize)]
pub struct PolicyViolation {
    pub rule: String,
    pub message: String,
    pub server_id: String,
    pub tool_name: String,
}

impl std::fmt::Display for PolicyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.rule, self.server_id, self.message)
    }
}

impl std::error::Error for PolicyViolation {}

/// Security policy for MCP federation.
///
/// Controls what external MCP servers and tools can do.
/// Default: conservative (no mutations, no credential forwarding).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSecurityPolicy {
    /// Allow mutation tools (Create, Mutation, Delete, Unknown).
    /// Default: false — only Query and Search are allowed.
    pub allow_mutations: bool,

    /// Allowlist of server IDs. None = allow all connected servers.
    /// When set, only listed servers can be called.
    pub allowed_servers: Option<Vec<String>>,

    /// Maximum external tool calls per minute (per server).
    /// Default: 60.
    pub max_calls_per_minute: u32,

    /// Prevent forwarding PO's own authorization headers to external servers.
    /// Default: true (always isolate credentials).
    pub credential_isolation: bool,
}

impl Default for McpSecurityPolicy {
    fn default() -> Self {
        Self {
            allow_mutations: false,
            allowed_servers: None,
            max_calls_per_minute: 60,
            credential_isolation: true,
        }
    }
}

impl McpSecurityPolicy {
    /// Create a permissive policy (for trusted servers).
    pub fn permissive() -> Self {
        Self {
            allow_mutations: true,
            allowed_servers: None,
            max_calls_per_minute: 120,
            credential_isolation: true, // always isolate creds even in permissive mode
        }
    }

    /// Check if a tool call is allowed by this policy.
    ///
    /// Returns `Ok(())` if allowed, `Err(PolicyViolation)` if blocked.
    pub fn enforce(
        &self,
        server_id: &str,
        tool_name: &str,
        category: &InferredCategory,
    ) -> std::result::Result<(), PolicyViolation> {
        // 1. Server allowlist check
        if let Some(ref allowed) = self.allowed_servers {
            if !allowed.iter().any(|s| s == server_id) {
                return Err(PolicyViolation {
                    rule: "allowed_servers".to_string(),
                    message: format!("Server '{}' is not in the allowed servers list", server_id),
                    server_id: server_id.to_string(),
                    tool_name: tool_name.to_string(),
                });
            }
        }

        // 2. Mutation check
        if !self.allow_mutations && category.is_mutating() {
            return Err(PolicyViolation {
                rule: "allow_mutations".to_string(),
                message: format!(
                    "Tool '{}' is classified as {:?} but mutations are disabled. \
                     Set allow_mutations=true in the security policy to allow this.",
                    tool_name, category
                ),
                server_id: server_id.to_string(),
                tool_name: tool_name.to_string(),
            });
        }

        Ok(())
    }
}

/// Per-server rate limiter using a sliding window counter.
#[derive(Debug)]
pub struct RateLimiter {
    /// Per-server call timestamps within the current window.
    windows: HashMap<String, Vec<Instant>>,
    /// Max calls per minute.
    max_per_minute: u32,
}

impl RateLimiter {
    pub fn new(max_per_minute: u32) -> Self {
        Self {
            windows: HashMap::new(),
            max_per_minute,
        }
    }

    /// Check if a call is allowed for a given server. If allowed, records the call.
    /// Returns `Ok(())` if under the limit, `Err(PolicyViolation)` if rate-limited.
    pub fn check_and_record(
        &mut self,
        server_id: &str,
        tool_name: &str,
    ) -> std::result::Result<(), PolicyViolation> {
        let now = Instant::now();
        let window = Duration::from_secs(60);

        let timestamps = self.windows.entry(server_id.to_string()).or_default();

        // Prune old entries outside the window
        timestamps.retain(|t| now.duration_since(*t) < window);

        if timestamps.len() >= self.max_per_minute as usize {
            return Err(PolicyViolation {
                rule: "rate_limit".to_string(),
                message: format!(
                    "Rate limit exceeded for server '{}': {} calls/min (max: {})",
                    server_id,
                    timestamps.len(),
                    self.max_per_minute
                ),
                server_id: server_id.to_string(),
                tool_name: tool_name.to_string(),
            });
        }

        timestamps.push(now);
        Ok(())
    }

    /// Reset the rate limiter for a specific server.
    pub fn reset(&mut self, server_id: &str) {
        self.windows.remove(server_id);
    }
}

/// Combined security enforcer: policy + rate limiter.
#[derive(Debug)]
pub struct SecurityEnforcer {
    pub policy: McpSecurityPolicy,
    pub rate_limiter: RateLimiter,
}

impl SecurityEnforcer {
    pub fn new(policy: McpSecurityPolicy) -> Self {
        let rate_limiter = RateLimiter::new(policy.max_calls_per_minute);
        Self {
            policy,
            rate_limiter,
        }
    }

    /// Enforce all security checks before an external tool call.
    ///
    /// Checks in order:
    /// 1. Server allowlist
    /// 2. Mutation policy
    /// 3. Rate limiting
    pub fn enforce(
        &mut self,
        server_id: &str,
        tool_name: &str,
        category: &InferredCategory,
    ) -> std::result::Result<(), PolicyViolation> {
        // Static policy checks
        self.policy.enforce(server_id, tool_name, category)?;

        // Rate limiting (mutable — records the call)
        self.rate_limiter.check_and_record(server_id, tool_name)?;

        Ok(())
    }
}

impl Default for SecurityEnforcer {
    fn default() -> Self {
        Self::new(McpSecurityPolicy::default())
    }
}

// ---------------------------------------------------------------------------
// InferredCategory extension
// ---------------------------------------------------------------------------

impl InferredCategory {
    /// Whether this category represents a mutating operation.
    pub fn is_mutating(&self) -> bool {
        matches!(
            self,
            InferredCategory::Create
                | InferredCategory::Mutation
                | InferredCategory::Delete
                | InferredCategory::Unknown
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy_blocks_mutations() {
        let policy = McpSecurityPolicy::default();
        assert!(!policy.allow_mutations);

        // Query should pass
        let result = policy.enforce("test-server", "list_items", &InferredCategory::Query);
        assert!(result.is_ok());

        // Search should pass
        let result = policy.enforce("test-server", "find_items", &InferredCategory::Search);
        assert!(result.is_ok());

        // Mutation should be blocked
        let result = policy.enforce("test-server", "update_item", &InferredCategory::Mutation);
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.rule, "allow_mutations");

        // Create should be blocked
        let result = policy.enforce("test-server", "create_item", &InferredCategory::Create);
        assert!(result.is_err());

        // Delete should be blocked
        let result = policy.enforce("test-server", "delete_item", &InferredCategory::Delete);
        assert!(result.is_err());

        // Unknown should be blocked (conservative)
        let result = policy.enforce("test-server", "do_thing", &InferredCategory::Unknown);
        assert!(result.is_err());
    }

    #[test]
    fn test_permissive_policy_allows_mutations() {
        let policy = McpSecurityPolicy::permissive();
        assert!(policy.allow_mutations);

        let result = policy.enforce("test-server", "update_item", &InferredCategory::Mutation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_server_allowlist() {
        let policy = McpSecurityPolicy {
            allowed_servers: Some(vec!["allowed-server".to_string()]),
            ..Default::default()
        };

        // Allowed server
        let result = policy.enforce("allowed-server", "list_items", &InferredCategory::Query);
        assert!(result.is_ok());

        // Blocked server
        let result = policy.enforce("blocked-server", "list_items", &InferredCategory::Query);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().rule, "allowed_servers");
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(3);

        // First 3 calls succeed
        assert!(limiter.check_and_record("srv", "tool").is_ok());
        assert!(limiter.check_and_record("srv", "tool").is_ok());
        assert!(limiter.check_and_record("srv", "tool").is_ok());

        // 4th call is rate-limited
        let result = limiter.check_and_record("srv", "tool");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().rule, "rate_limit");

        // Different server is not affected
        assert!(limiter.check_and_record("other", "tool").is_ok());
    }

    #[test]
    fn test_rate_limiter_reset() {
        let mut limiter = RateLimiter::new(2);

        assert!(limiter.check_and_record("srv", "tool").is_ok());
        assert!(limiter.check_and_record("srv", "tool").is_ok());
        assert!(limiter.check_and_record("srv", "tool").is_err());

        limiter.reset("srv");
        assert!(limiter.check_and_record("srv", "tool").is_ok());
    }

    #[test]
    fn test_security_enforcer_combined() {
        let policy = McpSecurityPolicy {
            allow_mutations: false,
            max_calls_per_minute: 2,
            ..Default::default()
        };
        let mut enforcer = SecurityEnforcer::new(policy);

        // First query passes
        assert!(enforcer
            .enforce("srv", "list", &InferredCategory::Query)
            .is_ok());

        // Mutation blocked (before rate limit)
        let result = enforcer.enforce("srv", "update", &InferredCategory::Mutation);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().rule, "allow_mutations");

        // Second query passes
        assert!(enforcer
            .enforce("srv", "list", &InferredCategory::Query)
            .is_ok());

        // Third query rate-limited
        let result = enforcer.enforce("srv", "search", &InferredCategory::Search);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().rule, "rate_limit");
    }

    #[test]
    fn test_credential_isolation_default() {
        let policy = McpSecurityPolicy::default();
        assert!(policy.credential_isolation);

        // Even permissive mode keeps credential isolation
        let permissive = McpSecurityPolicy::permissive();
        assert!(permissive.credential_isolation);
    }

    #[test]
    fn test_is_mutating() {
        assert!(!InferredCategory::Query.is_mutating());
        assert!(!InferredCategory::Search.is_mutating());
        assert!(InferredCategory::Create.is_mutating());
        assert!(InferredCategory::Mutation.is_mutating());
        assert!(InferredCategory::Delete.is_mutating());
        assert!(InferredCategory::Unknown.is_mutating());
    }
}
