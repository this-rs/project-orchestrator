//! Privacy gate — anonymization of episode content before export.
//!
//! Implements 13 regex heuristics split into two sensitivity levels:
//!
//! - **L3-FORBIDDEN** (H1–H7): secrets that must never leave the instance.
//!   Detection blocks export entirely.
//! - **L2-CONFIDENTIAL** (H8–H13): identifiers that can be replaced with
//!   deterministic placeholders.

use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

use super::distill_models::AnonymizationReport;

/// Error returned when L3-FORBIDDEN content is detected.
#[derive(Debug, Clone)]
pub struct L3BlockedError {
    /// Which L3 heuristic(s) fired.
    pub patterns: Vec<String>,
}

impl std::fmt::Display for L3BlockedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "L3-FORBIDDEN content detected ({}). Export blocked.",
            self.patterns.join(", ")
        )
    }
}

impl std::error::Error for L3BlockedError {}

// ============================================================================
// Compiled regexes (compiled once via LazyLock)
// ============================================================================

// --- L3-FORBIDDEN (H1-H7) ---

/// H1: Generic high-entropy API keys (20+ alphanumeric chars that look random).
static RE_H1_API_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token)\s*[=:]\s*['"]?([A-Za-z0-9/+=]{20,})['"]?"#).unwrap()
});

/// H2: Prefixed tokens (sk-ant-, ghp_, AKIA, gho_, xoxb-, glpat-).
static RE_H2_PREFIXED_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:sk-ant-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36,}|AKIA[A-Z0-9]{16}|gho_[A-Za-z0-9]{36,}|xoxb-[A-Za-z0-9\-]+|glpat-[A-Za-z0-9_-]{20,})").unwrap()
});

/// H3: Private key blocks.
static RE_H3_PRIVATE_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-----BEGIN[A-Z ]*PRIVATE KEY-----").unwrap()
});

/// H4: Connection strings with credentials (user:pass@host).
static RE_H4_CONN_STRING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^:]+:[^@]+@[^\s]+").unwrap()
});

/// H5: JWT tokens (three base64url segments separated by dots).
static RE_H5_JWT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}").unwrap()
});

/// H6: Password hashes ($2b$, $argon2id$, etc.).
static RE_H6_PASS_HASH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\$(?:2[aby]\$|argon2id\$)[^\s]+").unwrap()
});

/// H7: .env-style secret assignments.
static RE_H7_ENV_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:PASSWORD|SECRET|PRIVATE_KEY|API_KEY|AUTH_TOKEN)\s*=\s*\S+").unwrap()
});

// --- L2-CONFIDENTIAL (H8-H13) ---

/// H8: Absolute paths (/Users/xxx/ or /home/xxx/).
static RE_H8_ABS_PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"/(?:Users|home)/[A-Za-z0-9._-]+/[^\s]*").unwrap()
});

/// H9: RFC1918 private IPs.
static RE_H9_PRIVATE_IP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3})\b").unwrap()
});

/// H10: Email addresses.
static RE_H10_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap()
});

/// H11: Corporate/internal domains.
static RE_H11_CORP_DOMAIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[a-zA-Z0-9-]+\.(?:holdings|internal|corp|local)\b").unwrap()
});

/// H12: UUIDs (8-4-4-4-12 hex).
static RE_H12_UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b").unwrap()
});

/// H13: Git SHAs (40-character hex strings).
static RE_H13_GIT_SHA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[0-9a-fA-F]{40}\b").unwrap()
});

// ============================================================================
// Anonymization entry point
// ============================================================================

/// Anonymize content by detecting and replacing sensitive patterns.
///
/// # Returns
/// - `Ok((anonymized_content, report))` if content is safe (possibly after redaction).
/// - `Err(L3BlockedError)` if L3-FORBIDDEN content is detected.
pub fn anonymize(content: &str) -> Result<(String, AnonymizationReport), L3BlockedError> {
    let mut patterns_applied = Vec::new();
    let mut blocked_l3_patterns = Vec::new();

    // --- Phase 1: Check L3-FORBIDDEN patterns (block export) ---
    if RE_H1_API_KEY.is_match(content) {
        blocked_l3_patterns.push("H1-api-key".to_string());
    }
    if RE_H2_PREFIXED_TOKEN.is_match(content) {
        blocked_l3_patterns.push("H2-prefixed-token".to_string());
    }
    if RE_H3_PRIVATE_KEY.is_match(content) {
        blocked_l3_patterns.push("H3-private-key".to_string());
    }
    if RE_H4_CONN_STRING.is_match(content) {
        blocked_l3_patterns.push("H4-conn-string".to_string());
    }
    if RE_H5_JWT.is_match(content) {
        blocked_l3_patterns.push("H5-jwt".to_string());
    }
    if RE_H6_PASS_HASH.is_match(content) {
        blocked_l3_patterns.push("H6-pass-hash".to_string());
    }
    if RE_H7_ENV_SECRET.is_match(content) {
        blocked_l3_patterns.push("H7-env-secret".to_string());
    }

    if !blocked_l3_patterns.is_empty() {
        return Err(L3BlockedError {
            patterns: blocked_l3_patterns,
        });
    }

    // --- Phase 2: Replace L2-CONFIDENTIAL patterns ---
    let mut result = content.to_string();
    let mut redacted_count: u32 = 0;

    // Deterministic counters for placeholders.
    let mut uuid_map: HashMap<String, usize> = HashMap::new();
    let mut ip_map: HashMap<String, usize> = HashMap::new();
    let mut sha_map: HashMap<String, usize> = HashMap::new();

    // H8: Absolute paths → <PROJECT_ROOT>/...
    if RE_H8_ABS_PATH.is_match(&result) {
        patterns_applied.push("H8-abs-path".to_string());
        let replaced = RE_H8_ABS_PATH.replace_all(&result, "<PROJECT_ROOT>/");
        redacted_count += count_matches(&RE_H8_ABS_PATH, &result);
        result = replaced.into_owned();
    }

    // H9: Private IPs → <ip:N> (deterministic)
    if RE_H9_PRIVATE_IP.is_match(&result) {
        patterns_applied.push("H9-private-ip".to_string());
        result = replace_deterministic(&RE_H9_PRIVATE_IP, &result, "ip", &mut ip_map, &mut redacted_count);
    }

    // H10: Emails → <author>
    if RE_H10_EMAIL.is_match(&result) {
        patterns_applied.push("H10-email".to_string());
        redacted_count += count_matches(&RE_H10_EMAIL, &result);
        result = RE_H10_EMAIL.replace_all(&result, "<author>").into_owned();
    }

    // H11: Corporate domains → <corp-domain>
    if RE_H11_CORP_DOMAIN.is_match(&result) {
        patterns_applied.push("H11-corp-domain".to_string());
        redacted_count += count_matches(&RE_H11_CORP_DOMAIN, &result);
        result = RE_H11_CORP_DOMAIN
            .replace_all(&result, "<corp-domain>")
            .into_owned();
    }

    // H12: UUIDs → <uuid:N> (deterministic)
    if RE_H12_UUID.is_match(&result) {
        patterns_applied.push("H12-uuid".to_string());
        result = replace_deterministic(&RE_H12_UUID, &result, "uuid", &mut uuid_map, &mut redacted_count);
    }

    // H13: Git SHAs → <commit:N> (deterministic)
    if RE_H13_GIT_SHA.is_match(&result) {
        patterns_applied.push("H13-git-sha".to_string());
        result = replace_deterministic(&RE_H13_GIT_SHA, &result, "commit", &mut sha_map, &mut redacted_count);
    }

    let report = AnonymizationReport {
        redacted_count,
        patterns_applied,
        blocked_l3: false,
    };

    Ok((result, report))
}

// ============================================================================
// Helpers
// ============================================================================

/// Count regex matches in text.
fn count_matches(re: &Regex, text: &str) -> u32 {
    re.find_iter(text).count() as u32
}

/// Replace matches deterministically: same input value always maps to the same
/// `<prefix:N>` placeholder.
fn replace_deterministic(
    re: &Regex,
    text: &str,
    prefix: &str,
    map: &mut HashMap<String, usize>,
    count: &mut u32,
) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for m in re.find_iter(text) {
        result.push_str(&text[last_end..m.start()]);
        let matched = m.as_str().to_string();
        let next_id = map.len();
        let id = *map.entry(matched).or_insert(next_id);
        result.push_str(&format!("<{}:{}>", prefix, id));
        *count += 1;
        last_end = m.end();
    }
    result.push_str(&text[last_end..]);
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- L3-FORBIDDEN tests (should block) ---

    #[test]
    fn test_h1_api_key_blocked() {
        let content = "api_key = ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H1-api-key".to_string()));
    }

    #[test]
    fn test_h2_prefixed_token_sk_ant() {
        let content = "token: sk-ant-abcdefghij1234567890abcd";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H2-prefixed-token".to_string()));
    }

    #[test]
    fn test_h2_prefixed_token_ghp() {
        let content = "GITHUB_TOKEN=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345678901";
        // This also matches H7 (env secret)
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H2-prefixed-token".to_string()));
    }

    #[test]
    fn test_h2_prefixed_token_akia() {
        let content = "aws key: AKIAIOSFODNN7EXAMPLE";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H2-prefixed-token".to_string()));
    }

    #[test]
    fn test_h2_prefixed_token_gho() {
        let content = "gho_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345678901";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H2-prefixed-token".to_string()));
    }

    #[test]
    fn test_h2_prefixed_token_xoxb() {
        let content = "slack: xoxb-123456789012-abcdefg";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H2-prefixed-token".to_string()));
    }

    #[test]
    fn test_h2_prefixed_token_glpat() {
        let content = "glpat-aBcDeFgHiJkLmNoPqRsTu";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H2-prefixed-token".to_string()));
    }

    #[test]
    fn test_h3_private_key_blocked() {
        let content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ...";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H3-private-key".to_string()));
    }

    #[test]
    fn test_h4_conn_string_blocked() {
        let content = "postgres://admin:s3cret@db.example.com:5432/mydb";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H4-conn-string".to_string()));
    }

    #[test]
    fn test_h5_jwt_blocked() {
        let content = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H5-jwt".to_string()));
    }

    #[test]
    fn test_h6_bcrypt_hash_blocked() {
        let content = "hash = $2b$12$LJ3m4ys3Lz0YOV5v1X2Q3eRSDF98765432abcdefghijklmnop";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H6-pass-hash".to_string()));
    }

    #[test]
    fn test_h6_argon2_hash_blocked() {
        let content = "hash = $argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$RdescudvJCsgt3ub+b+daw";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H6-pass-hash".to_string()));
    }

    #[test]
    fn test_h7_env_secret_blocked() {
        let content = "PASSWORD=my_super_secret_password";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H7-env-secret".to_string()));
    }

    // --- L2-CONFIDENTIAL tests (should redact, not block) ---

    #[test]
    fn test_h8_absolute_path_redacted() {
        let content = "file at /Users/johndoe/projects/my-app/src/main.rs";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<PROJECT_ROOT>/"));
        assert!(!result.contains("/Users/johndoe"));
        assert!(report.patterns_applied.contains(&"H8-abs-path".to_string()));
        assert!(report.redacted_count >= 1);
    }

    #[test]
    fn test_h8_home_path_redacted() {
        let content = "config at /home/deploy/.config/app.toml";
        let (result, _report) = anonymize(content).unwrap();
        assert!(result.contains("<PROJECT_ROOT>/"));
        assert!(!result.contains("/home/deploy"));
    }

    #[test]
    fn test_h9_private_ip_redacted() {
        let content = "server at 192.168.1.100 and also 10.0.0.1";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<ip:0>"));
        assert!(result.contains("<ip:1>"));
        assert!(!result.contains("192.168.1.100"));
        assert!(report.patterns_applied.contains(&"H9-private-ip".to_string()));
    }

    #[test]
    fn test_h9_private_ip_deterministic() {
        let content = "host 192.168.1.100 and again 192.168.1.100";
        let (result, _report) = anonymize(content).unwrap();
        // Same IP should get same placeholder
        assert_eq!(result.matches("<ip:0>").count(), 2);
    }

    #[test]
    fn test_h10_email_redacted() {
        let content = "contact alice@example.com for details";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<author>"));
        assert!(!result.contains("alice@example.com"));
        assert!(report.patterns_applied.contains(&"H10-email".to_string()));
    }

    #[test]
    fn test_h11_corp_domain_redacted() {
        let content = "deployed to app.internal and api.corp";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<corp-domain>"));
        assert!(!result.contains("app.internal"));
        assert!(report.patterns_applied.contains(&"H11-corp-domain".to_string()));
    }

    #[test]
    fn test_h12_uuid_redacted() {
        let content = "note 550e8400-e29b-41d4-a716-446655440000 and 6ba7b810-9dad-11d1-80b4-00c04fd430c8";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<uuid:0>"));
        assert!(result.contains("<uuid:1>"));
        assert!(!result.contains("550e8400"));
        assert!(report.patterns_applied.contains(&"H12-uuid".to_string()));
    }

    #[test]
    fn test_h12_uuid_deterministic() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let content = format!("first {} then {}", uuid, uuid);
        let (result, _report) = anonymize(&content).unwrap();
        assert_eq!(result.matches("<uuid:0>").count(), 2);
    }

    #[test]
    fn test_h13_git_sha_redacted() {
        let content = "commit abc1234567890abc1234567890abc123456789ab";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<commit:0>"));
        assert!(report.patterns_applied.contains(&"H13-git-sha".to_string()));
    }

    // --- Combined tests ---

    #[test]
    fn test_clean_content_passes() {
        let content = "Implement the PRODUCED_DURING relation for episode tracking.";
        let (result, report) = anonymize(content).unwrap();
        assert_eq!(result, content);
        assert_eq!(report.redacted_count, 0);
        assert!(report.patterns_applied.is_empty());
        assert!(!report.blocked_l3);
    }

    #[test]
    fn test_multiple_l2_patterns() {
        let content = "path /Users/dev/app with uuid 550e8400-e29b-41d4-a716-446655440000 at 192.168.1.1";
        let (result, report) = anonymize(content).unwrap();
        assert!(result.contains("<PROJECT_ROOT>/"));
        assert!(result.contains("<uuid:0>"));
        assert!(result.contains("<ip:0>"));
        assert!(report.redacted_count >= 3);
    }

    #[test]
    fn test_l3_blocks_before_l2_redaction() {
        // Content with both L3 and L2 patterns — should block
        let content = "path /Users/dev/app with PASSWORD=secret123";
        let err = anonymize(content).unwrap_err();
        assert!(err.patterns.contains(&"H7-env-secret".to_string()));
    }
}
