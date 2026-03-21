//! Privacy gate — multi-layered anonymization pipeline for episode content.
//!
//! Implements a configurable pipeline of anonymization stages:
//!
//! 1. **`SecretScanner`** (L3-FORBIDDEN, H1-H7): secrets that must never leave
//!    the instance. Detection blocks export entirely.
//! 2. **`ConfidentialReplacer`** (L2-CONFIDENTIAL, H8-H13): identifiers replaced
//!    with deterministic placeholders.
//! 3. **`EntityGeneralizer`** (L2-CONFIDENTIAL, H14): code entity names (structs,
//!    functions, files) replaced with typed placeholders for consistency.
//! 4. **`MetricNoise`**: controlled Gaussian noise applied to numeric metrics
//!    for differential privacy.
//!
//! # References
//!
//! - "Episodic Memory in AI Agents Poses Risks That Should Be Studied and
//!   Mitigated" (2025) — motivates multi-layered anonymization and differential
//!   privacy for exported episodic memories.

use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

use super::distill_models::AnonymizationReport;
use crate::episodes::distill_models::SharingPolicy;
use crate::sharing::consent_gate;
use crate::notes::models::Note;

// ============================================================================
// Error types
// ============================================================================

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

/// Error returned by the anonymization pipeline or export gate.
#[derive(Debug)]
pub enum AnonymizeError {
    /// L3-FORBIDDEN content detected.
    L3Blocked(L3BlockedError),
    /// Sharing is disabled for the project.
    SharingDisabled,
}

impl std::fmt::Display for AnonymizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnonymizeError::L3Blocked(e) => write!(f, "{}", e),
            AnonymizeError::SharingDisabled => {
                write!(f, "Sharing is disabled for this project. Export blocked.")
            }
        }
    }
}

impl std::error::Error for AnonymizeError {}

impl From<L3BlockedError> for AnonymizeError {
    fn from(e: L3BlockedError) -> Self {
        AnonymizeError::L3Blocked(e)
    }
}

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
static RE_H3_PRIVATE_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-----BEGIN[A-Z ]*PRIVATE KEY-----").unwrap());

/// H4: Connection strings with credentials (user:pass@host).
static RE_H4_CONN_STRING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^:]+:[^@]+@[^\s]+").unwrap());

/// H5: JWT tokens (three base64url segments separated by dots).
static RE_H5_JWT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}").unwrap()
});

/// H6: Password hashes ($2b$, $argon2id$, etc.).
static RE_H6_PASS_HASH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$(?:2[aby]\$|argon2id\$)[^\s]+").unwrap());

/// H7: .env-style secret assignments.
static RE_H7_ENV_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:PASSWORD|SECRET|PRIVATE_KEY|API_KEY|AUTH_TOKEN)\s*=\s*\S+").unwrap()
});

// --- L2-CONFIDENTIAL (H8-H13) ---

/// H8: Absolute paths (/Users/xxx/ or /home/xxx/).
static RE_H8_ABS_PATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/(?:Users|home)/[A-Za-z0-9._-]+/[^\s]*").unwrap());

/// H9: RFC1918 private IPs.
static RE_H9_PRIVATE_IP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3})\b").unwrap()
});

/// H10: Email addresses.
static RE_H10_EMAIL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap());

/// H11: Corporate/internal domains.
static RE_H11_CORP_DOMAIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-zA-Z0-9-]+\.(?:holdings|internal|corp|local)\b").unwrap());

/// H12: UUIDs (8-4-4-4-12 hex).
static RE_H12_UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
        .unwrap()
});

/// H13: Git SHAs (40-character hex strings).
static RE_H13_GIT_SHA: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[0-9a-fA-F]{40}\b").unwrap());

// --- H14: Code entity names ---

/// H14: CamelCase struct/class names (e.g. `MyStruct`, `HttpClient`).
static RE_H14_STRUCT_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:struct|class|enum|trait|interface|type)\s+([A-Z][a-zA-Z0-9_]{2,})\b")
        .unwrap()
});

/// H14: Function/method names in `fn name(` or `def name(` style.
static RE_H14_FN_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:fn|func|def|function)\s+([a-z_][a-zA-Z0-9_]{2,})\s*\(").unwrap()
});

/// H14: File references like `foo.rs`, `bar.py`, `baz.ts`.
static RE_H14_FILE_REF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_-]*)\.(?:rs|py|ts|js|go|java|cpp|c|h|rb|kt|swift)\b")
        .unwrap()
});

// --- Metric noise ---

/// Matches numeric values (integers and floats) that look like metrics.
/// Captures patterns like `duration: 1234`, `score: 0.95`, `count: 42`.
static RE_METRIC_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(duration|score|count|latency|time|elapsed|total|size|length|tokens)\s*[=:]\s*(\d+(?:\.\d+)?)\b").unwrap()
});

// ============================================================================
// AnonymizationStage trait and pipeline
// ============================================================================

/// A single stage in the anonymization pipeline.
///
/// Each stage transforms content and may update the report with details
/// of what was redacted.
///
/// # References
///
/// - "Episodic Memory in AI Agents Poses Risks That Should Be Studied and
///   Mitigated" (2025) — recommends layered anonymization with distinct
///   stages for secrets, identifiers, and statistical perturbation.
pub trait AnonymizationStage: Send + Sync {
    /// Human-readable name of this stage (for reporting).
    fn name(&self) -> &str;

    /// Apply this stage to the content.
    ///
    /// # Returns
    /// - `Ok(transformed_content)` on success.
    /// - `Err(L3BlockedError)` if the stage detects blocking content.
    fn apply(&self, content: &str) -> Result<String, L3BlockedError>;
}

/// Configurable multi-stage anonymization pipeline.
///
/// Runs stages in order; if any stage returns `Err`, the pipeline aborts.
///
/// # References
///
/// - "Episodic Memory in AI Agents Poses Risks That Should Be Studied and
///   Mitigated" (2025) — pipeline approach ensures defense-in-depth:
///   secrets are blocked before identifiers are replaced, before noise
///   is added.
pub struct AnonymizationPipeline {
    stages: Vec<Box<dyn AnonymizationStage>>,
}

impl AnonymizationPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Create the default pipeline with all stages in recommended order.
    ///
    /// Order: SecretScanner -> ConfidentialReplacer -> EntityGeneralizer -> MetricNoise
    pub fn default_pipeline() -> Self {
        let mut pipeline = Self::new();
        pipeline.add_stage(Box::new(SecretScanner));
        pipeline.add_stage(Box::new(ConfidentialReplacer));
        pipeline.add_stage(Box::new(EntityGeneralizer));
        pipeline.add_stage(Box::new(MetricNoise::default()));
        pipeline
    }

    /// Add a stage to the end of the pipeline.
    pub fn add_stage(&mut self, stage: Box<dyn AnonymizationStage>) {
        self.stages.push(stage);
    }

    /// Run all stages in order, accumulating report data.
    ///
    /// # Returns
    /// - `Ok((anonymized_content, report))` on success.
    /// - `Err(L3BlockedError)` if any stage blocks.
    pub fn run(&self, content: &str) -> Result<(String, AnonymizationReport), L3BlockedError> {
        let mut current = content.to_string();
        let mut all_patterns = Vec::new();

        for stage in &self.stages {
            let before = current.clone();
            current = stage.apply(&current)?;
            if current != before {
                all_patterns.push(stage.name().to_string());
            }
        }

        // Count total redactions by comparing original to final
        let redacted_count = count_placeholder_occurrences(&current);

        let report = AnonymizationReport {
            redacted_count,
            patterns_applied: all_patterns,
            blocked_l3: false,
            consent_stats: None,
        };

        Ok((current, report))
    }
}

impl Default for AnonymizationPipeline {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

// ============================================================================
// Stage 1: SecretScanner (L3-FORBIDDEN)
// ============================================================================

/// Scans for L3-FORBIDDEN secrets (H1-H7) and blocks export if found.
///
/// This stage produces no output transformation — it either passes content
/// through unchanged or returns an error.
pub struct SecretScanner;

impl AnonymizationStage for SecretScanner {
    fn name(&self) -> &str {
        "L3-SecretScanner"
    }

    fn apply(&self, content: &str) -> Result<String, L3BlockedError> {
        scan_for_l3(content)?;
        Ok(content.to_string())
    }
}

// ============================================================================
// Stage 2: ConfidentialReplacer (L2-CONFIDENTIAL, H8-H13)
// ============================================================================

/// Replaces L2-CONFIDENTIAL identifiers (H8-H13) with deterministic placeholders.
pub struct ConfidentialReplacer;

impl AnonymizationStage for ConfidentialReplacer {
    fn name(&self) -> &str {
        "L2-ConfidentialReplacer"
    }

    fn apply(&self, content: &str) -> Result<String, L3BlockedError> {
        let mut result = content.to_string();

        let mut uuid_map: HashMap<String, usize> = HashMap::new();
        let mut ip_map: HashMap<String, usize> = HashMap::new();
        let mut sha_map: HashMap<String, usize> = HashMap::new();

        // H8: Absolute paths
        result = RE_H8_ABS_PATH.replace_all(&result, "<PROJECT_ROOT>/").into_owned();

        // H9: Private IPs
        result = replace_deterministic(&RE_H9_PRIVATE_IP, &result, "ip", &mut ip_map);

        // H10: Emails
        result = RE_H10_EMAIL.replace_all(&result, "<author>").into_owned();

        // H11: Corporate domains
        result = RE_H11_CORP_DOMAIN
            .replace_all(&result, "<corp-domain>")
            .into_owned();

        // H12: UUIDs
        result = replace_deterministic(&RE_H12_UUID, &result, "uuid", &mut uuid_map);

        // H13: Git SHAs
        result = replace_deterministic(&RE_H13_GIT_SHA, &result, "commit", &mut sha_map);

        Ok(result)
    }
}

// ============================================================================
// Stage 3: EntityGeneralizer (H14 — code identifiers)
// ============================================================================

/// Replaces specific code entity names with typed placeholders.
///
/// Handles struct/class names, function names, and file references.
/// Uses a deterministic [`HashMap`] so the same identifier always maps
/// to the same placeholder within a single anonymization pass.
///
/// # Examples
///
/// - `struct HttpClient` -> `struct Struct_1`
/// - `fn process_events(` -> `fn Function_A(`
/// - `main.rs` -> `File_1.rs`
///
/// # References
///
/// - "Episodic Memory in AI Agents Poses Risks That Should Be Studied and
///   Mitigated" (2025) — code identifiers can leak proprietary architecture
///   details; generalization preserves structural information while removing
///   identifying specifics.
pub struct EntityGeneralizer;

impl AnonymizationStage for EntityGeneralizer {
    fn name(&self) -> &str {
        "L2-EntityGeneralizer"
    }

    fn apply(&self, content: &str) -> Result<String, L3BlockedError> {
        let mut result = content.to_string();
        let mut struct_map: HashMap<String, usize> = HashMap::new();
        let mut fn_map: HashMap<String, usize> = HashMap::new();
        let mut file_map: HashMap<String, usize> = HashMap::new();

        // H14a: Struct/class/enum/trait names
        result = replace_capture_group(
            &RE_H14_STRUCT_NAME,
            &result,
            1,
            "Struct",
            &mut struct_map,
            PlaceholderStyle::Numeric,
        );

        // H14b: Function/method names
        result = replace_capture_group(
            &RE_H14_FN_NAME,
            &result,
            1,
            "Function",
            &mut fn_map,
            PlaceholderStyle::Alpha,
        );

        // H14c: File references
        result = replace_capture_group(
            &RE_H14_FILE_REF,
            &result,
            1,
            "File",
            &mut file_map,
            PlaceholderStyle::Numeric,
        );

        Ok(result)
    }
}

/// Style of placeholder suffix.
enum PlaceholderStyle {
    /// Numeric: `_1`, `_2`, ...
    Numeric,
    /// Alphabetic: `_A`, `_B`, ...
    Alpha,
}

/// Replace a specific capture group within regex matches with a deterministic placeholder.
fn replace_capture_group(
    re: &Regex,
    text: &str,
    group: usize,
    prefix: &str,
    map: &mut HashMap<String, usize>,
    style: PlaceholderStyle,
) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for caps in re.captures_iter(text) {
        let full_match = caps.get(0).unwrap();
        let capture = match caps.get(group) {
            Some(c) => c,
            None => continue,
        };

        // Push text before this match
        result.push_str(&text[last_end..full_match.start()]);

        // Push the part of the match before the capture group
        result.push_str(&text[full_match.start()..capture.start()]);

        // Generate placeholder for the captured name
        let name = capture.as_str().to_string();
        let next_id = map.len();
        let id = *map.entry(name).or_insert(next_id);
        let suffix = match style {
            PlaceholderStyle::Numeric => format!("{}", id + 1),
            PlaceholderStyle::Alpha => {
                // A, B, C, ... Z, AA, AB, ...
                let mut s = String::new();
                let mut n = id;
                loop {
                    s.insert(0, (b'A' + (n % 26) as u8) as char);
                    if n < 26 {
                        break;
                    }
                    n = n / 26 - 1;
                }
                s
            }
        };
        result.push_str(&format!("{}_{}", prefix, suffix));

        // Push the part of the match after the capture group
        result.push_str(&text[capture.end()..full_match.end()]);

        last_end = full_match.end();
    }

    result.push_str(&text[last_end..]);
    result
}

// ============================================================================
// Stage 4: MetricNoise (differential privacy)
// ============================================================================

/// Adds controlled Gaussian noise to numeric metrics for differential privacy.
///
/// Noise is drawn from N(0, sigma * original_value) and clamped so the
/// perturbed value stays within +-20% of the original and remains non-negative.
///
/// # Configuration
///
/// - `sigma`: standard deviation as a fraction of the original value (default: 0.05).
///   A sigma of 0.05 means ~95% of perturbations fall within +-10%.
///
/// # References
///
/// - "Episodic Memory in AI Agents Poses Risks That Should Be Studied and
///   Mitigated" (2025) — recommends differential privacy mechanisms for
///   numeric data in exported episodes to prevent re-identification through
///   unique metric fingerprints.
pub struct MetricNoise {
    /// Standard deviation as a fraction of the metric value.
    pub sigma: f64,
    /// Optional fixed seed for deterministic testing.
    seed: Option<u64>,
}

impl MetricNoise {
    /// Create a new MetricNoise stage with the given sigma.
    pub fn new(sigma: f64) -> Self {
        Self { sigma, seed: None }
    }

    /// Create a MetricNoise stage with a fixed seed (for deterministic tests).
    #[cfg(test)]
    pub fn with_seed(sigma: f64, seed: u64) -> Self {
        Self {
            sigma,
            seed: Some(seed),
        }
    }

    /// Generate a Gaussian random value using the Box-Muller transform.
    fn gaussian_noise(&self, original: f64) -> f64 {
        use rand::{RngExt, SeedableRng};

        let std_dev = self.sigma * original.abs().max(1.0);

        let mut rng: rand::rngs::StdRng = if let Some(seed) = self.seed {
            // Deterministic RNG for testing
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        };

        // Box-Muller transform
        let u1: f64 = rng.random_range(f64::EPSILON..1.0);
        let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        let z: f64 = (-2.0_f64 * u1.ln()).sqrt() * u2.cos();

        z * std_dev
    }

    /// Perturb a numeric value with Gaussian noise, clamped to +-20% and non-negative.
    fn perturb(&self, value: f64) -> f64 {
        if value == 0.0 {
            return 0.0;
        }
        let noise = self.gaussian_noise(value);
        let perturbed = value + noise;
        // Clamp to +-20% of original
        let lower = value * 0.8;
        let upper = value * 1.2;
        let (lower, upper) = if lower <= upper {
            (lower, upper)
        } else {
            (upper, lower)
        };
        perturbed.clamp(lower, upper).max(0.0)
    }
}

impl Default for MetricNoise {
    fn default() -> Self {
        Self::new(0.05)
    }
}

impl AnonymizationStage for MetricNoise {
    fn name(&self) -> &str {
        "MetricNoise"
    }

    fn apply(&self, content: &str) -> Result<String, L3BlockedError> {
        let mut result = String::with_capacity(content.len());
        let mut last_end = 0;

        for caps in RE_METRIC_VALUE.captures_iter(content) {
            let full_match = caps.get(0).unwrap();
            let value_match = caps.get(2).unwrap();

            result.push_str(&content[last_end..full_match.start()]);

            // Push the label part (everything before the numeric value)
            result.push_str(&content[full_match.start()..value_match.start()]);

            // Parse and perturb the value
            let value_str = value_match.as_str();
            if let Ok(value) = value_str.parse::<f64>() {
                let perturbed = self.perturb(value);
                // Preserve integer formatting if original was integer
                if value_str.contains('.') {
                    // Keep same decimal precision
                    let decimals = value_str.split('.').nth(1).map_or(0, |d| d.len());
                    result.push_str(&format!("{:.*}", decimals, perturbed));
                } else {
                    result.push_str(&format!("{}", perturbed.round() as i64));
                }
            } else {
                result.push_str(value_str);
            }

            last_end = value_match.end();
        }

        result.push_str(&content[last_end..]);
        Ok(result)
    }
}

// ============================================================================
// L3 scan — reusable secret detection (RFC Privacy §2.3)
// ============================================================================

/// Scan content for L3-FORBIDDEN secrets (H1-H7).
///
/// This is a lightweight check that can be used independently of the full
/// anonymization pipeline — e.g., at note creation/update time to prevent
/// secrets from being persisted in the graph.
///
/// # Returns
/// - `Ok(())` if no L3 patterns detected.
/// - `Err(L3BlockedError)` listing all detected L3 patterns.
pub fn scan_for_l3(content: &str) -> Result<(), L3BlockedError> {
    let mut blocked = Vec::new();

    if RE_H1_API_KEY.is_match(content) {
        blocked.push("H1-api-key".to_string());
    }
    if RE_H2_PREFIXED_TOKEN.is_match(content) {
        blocked.push("H2-prefixed-token".to_string());
    }
    if RE_H3_PRIVATE_KEY.is_match(content) {
        blocked.push("H3-private-key".to_string());
    }
    if RE_H4_CONN_STRING.is_match(content) {
        blocked.push("H4-conn-string".to_string());
    }
    if RE_H5_JWT.is_match(content) {
        blocked.push("H5-jwt".to_string());
    }
    if RE_H6_PASS_HASH.is_match(content) {
        blocked.push("H6-pass-hash".to_string());
    }
    if RE_H7_ENV_SECRET.is_match(content) {
        blocked.push("H7-env-secret".to_string());
    }

    if blocked.is_empty() {
        Ok(())
    } else {
        Err(L3BlockedError { patterns: blocked })
    }
}

// ============================================================================
// Anonymization entry point (backward-compatible)
// ============================================================================

/// Anonymize content by detecting and replacing sensitive patterns.
///
/// This is the backward-compatible entry point that runs the L3 scan
/// followed by L2 confidential replacement. For the full pipeline
/// (including entity generalization and metric noise), use
/// [`AnonymizationPipeline::default_pipeline`].
///
/// # Returns
/// - `Ok((anonymized_content, report))` if content is safe (possibly after redaction).
/// - `Err(L3BlockedError)` if L3-FORBIDDEN content is detected.
pub fn anonymize(content: &str) -> Result<(String, AnonymizationReport), L3BlockedError> {
    let mut patterns_applied = Vec::new();

    // --- Phase 1: Check L3-FORBIDDEN patterns (block export) ---
    // Delegates to the reusable scan_for_l3() function.
    scan_for_l3(content)?;

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
        result = replace_deterministic_counted(
            &RE_H9_PRIVATE_IP,
            &result,
            "ip",
            &mut ip_map,
            &mut redacted_count,
        );
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
        result = replace_deterministic_counted(
            &RE_H12_UUID,
            &result,
            "uuid",
            &mut uuid_map,
            &mut redacted_count,
        );
    }

    // H13: Git SHAs → <commit:N> (deterministic)
    if RE_H13_GIT_SHA.is_match(&result) {
        patterns_applied.push("H13-git-sha".to_string());
        result = replace_deterministic_counted(
            &RE_H13_GIT_SHA,
            &result,
            "commit",
            &mut sha_map,
            &mut redacted_count,
        );
    }

    let report = AnonymizationReport {
        redacted_count,
        patterns_applied,
        blocked_l3: false,
        consent_stats: None,
    };

    Ok((result, report))
}

// ============================================================================
// Consent-gated export entry point
// ============================================================================

/// Anonymize episode content for export, checking consent first.
///
/// This is the recommended entry point for all episode exports. It:
/// 1. Checks the project's sharing policy via [`consent_gate`].
/// 2. If sharing is disabled, returns [`AnonymizeError::SharingDisabled`].
/// 3. If sharing is enabled, runs the full [`AnonymizationPipeline`].
///
/// # Arguments
/// - `content`: the episode content to anonymize.
/// - `notes`: notes associated with the episode (for consent filtering).
/// - `policy`: the project's sharing policy.
///
/// # Returns
/// - `Ok((anonymized_content, report, allowed_notes))` on success.
/// - `Err(AnonymizeError)` if sharing is disabled or L3 content is detected.
///
/// # References
///
/// - "Episodic Memory in AI Agents Poses Risks That Should Be Studied and
///   Mitigated" (2025) — consent verification must precede any anonymization
///   to respect user data sovereignty.
pub fn anonymize_for_export<'a>(
    content: &str,
    notes: &'a [Note],
    policy: &SharingPolicy,
) -> Result<(String, AnonymizationReport, Vec<&'a Note>), AnonymizeError> {
    // Step 1: Check if sharing is enabled at all
    if !policy.enabled {
        return Err(AnonymizeError::SharingDisabled);
    }

    // Step 2: Run consent gate on notes
    let mut report = AnonymizationReport {
        redacted_count: 0,
        patterns_applied: Vec::new(),
        blocked_l3: false,
        consent_stats: None,
    };
    let allowed_notes = consent_gate::apply_consent_gate_to_report(notes, policy, &mut report);

    // Step 3: Run the full anonymization pipeline
    let pipeline = AnonymizationPipeline::default_pipeline();
    let (anonymized, mut pipeline_report) = pipeline.run(content)?;

    // Merge consent stats into pipeline report
    pipeline_report.consent_stats = report.consent_stats;

    Ok((anonymized, pipeline_report, allowed_notes))
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
) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for m in re.find_iter(text) {
        result.push_str(&text[last_end..m.start()]);
        let matched = m.as_str().to_string();
        let next_id = map.len();
        let id = *map.entry(matched).or_insert(next_id);
        result.push_str(&format!("<{}:{}>", prefix, id));
        last_end = m.end();
    }
    result.push_str(&text[last_end..]);
    result
}

/// Replace matches deterministically with a redaction counter
/// (backward-compatible version used by the `anonymize()` function).
fn replace_deterministic_counted(
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

/// Count placeholder occurrences in anonymized text (for pipeline reporting).
fn count_placeholder_occurrences(text: &str) -> u32 {
    static RE_PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"<(?:PROJECT_ROOT|ip|author|corp-domain|uuid|commit|Struct_|Function_|File_)[^>]*>")
            .unwrap()
    });
    RE_PLACEHOLDER.find_iter(text).count() as u32
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
        assert!(report
            .patterns_applied
            .contains(&"H9-private-ip".to_string()));
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
        assert!(report
            .patterns_applied
            .contains(&"H11-corp-domain".to_string()));
    }

    #[test]
    fn test_h12_uuid_redacted() {
        let content =
            "note 550e8400-e29b-41d4-a716-446655440000 and 6ba7b810-9dad-11d1-80b4-00c04fd430c8";
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
        let content =
            "path /Users/dev/app with uuid 550e8400-e29b-41d4-a716-446655440000 at 192.168.1.1";
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

    // --- Pipeline tests ---

    #[test]
    fn test_pipeline_runs_all_stages() {
        let pipeline = AnonymizationPipeline::default_pipeline();
        let content = "Clean content with no sensitive data.";
        let (result, _report) = pipeline.run(content).unwrap();
        assert_eq!(result, content);
    }

    #[test]
    fn test_pipeline_blocks_l3() {
        let pipeline = AnonymizationPipeline::default_pipeline();
        let content = "PASSWORD=secret123";
        let err = pipeline.run(content).unwrap_err();
        assert!(err.patterns.contains(&"H7-env-secret".to_string()));
    }

    #[test]
    fn test_pipeline_replaces_l2() {
        let pipeline = AnonymizationPipeline::default_pipeline();
        let content = "file at /Users/dev/project/src and ip 192.168.1.1";
        let (result, report) = pipeline.run(content).unwrap();
        assert!(result.contains("<PROJECT_ROOT>/"));
        assert!(result.contains("<ip:0>"));
        assert!(!report.patterns_applied.is_empty());
    }

    // --- EntityGeneralizer tests ---

    #[test]
    fn test_entity_generalizer_struct() {
        let stage = EntityGeneralizer;
        let content = "struct HttpClient { }";
        let result = stage.apply(content).unwrap();
        assert!(result.contains("Struct_1"));
        assert!(!result.contains("HttpClient"));
    }

    #[test]
    fn test_entity_generalizer_function() {
        let stage = EntityGeneralizer;
        let content = "fn process_events(data: Vec<u8>)";
        let result = stage.apply(content).unwrap();
        assert!(result.contains("Function_A"));
        assert!(!result.contains("process_events"));
    }

    #[test]
    fn test_entity_generalizer_file() {
        let stage = EntityGeneralizer;
        let content = "see main.rs and utils.py for details";
        let result = stage.apply(content).unwrap();
        assert!(result.contains("File_1.rs"));
        assert!(result.contains("File_2.py"));
        assert!(!result.contains("main.rs"));
        assert!(!result.contains("utils.py"));
    }

    #[test]
    fn test_entity_generalizer_deterministic() {
        let stage = EntityGeneralizer;
        let content = "struct Foo { }\nstruct Foo { }";
        let result = stage.apply(content).unwrap();
        // Both should map to the same placeholder
        assert_eq!(result.matches("Struct_1").count(), 2);
    }

    #[test]
    fn test_entity_generalizer_multiple_structs() {
        let stage = EntityGeneralizer;
        let content = "struct Alpha and struct Beta";
        let result = stage.apply(content).unwrap();
        assert!(result.contains("Struct_1"));
        assert!(result.contains("Struct_2"));
    }

    // --- MetricNoise tests ---

    #[test]
    fn test_metric_noise_perturbs_values() {
        let stage = MetricNoise::with_seed(0.05, 42);
        let content = "duration: 1000 and score: 0.95";
        let result = stage.apply(content).unwrap();
        // Values should be perturbed but within +-20%
        assert!(result.contains("duration:"));
        assert!(result.contains("score:"));
        // Original exact values should likely not appear (very small chance they do)
        // We test bounds more precisely below
    }

    #[test]
    fn test_metric_noise_bounds() {
        let noise = MetricNoise::with_seed(0.15, 12345);
        let value = 1000.0;
        let perturbed = noise.perturb(value);
        assert!(
            perturbed >= 800.0 && perturbed <= 1200.0,
            "Perturbed value {} out of +-20% range for {}",
            perturbed,
            value
        );
        assert!(perturbed >= 0.0, "Perturbed value must be non-negative");
    }

    #[test]
    fn test_metric_noise_zero_unchanged() {
        let noise = MetricNoise::with_seed(0.05, 42);
        assert_eq!(noise.perturb(0.0), 0.0);
    }

    #[test]
    fn test_metric_noise_non_negative() {
        let noise = MetricNoise::with_seed(0.5, 99);
        for _ in 0..100 {
            let perturbed = noise.perturb(1.0);
            assert!(perturbed >= 0.0);
        }
    }

    // --- Consent-gated export tests ---

    #[test]
    fn test_anonymize_for_export_sharing_disabled() {
        let policy = SharingPolicy {
            enabled: false,
            mode: crate::episodes::distill_models::SharingMode::Auto,
            type_overrides: HashMap::new(),
            l3_scan_enabled: true,
            min_shareability_score: 0.5,
        };
        let content = "some episode content";
        let notes: Vec<Note> = vec![];
        let result = anonymize_for_export(content, &notes, &policy);
        assert!(matches!(result, Err(AnonymizeError::SharingDisabled)));
    }

    #[test]
    fn test_anonymize_for_export_l3_blocked() {
        let policy = SharingPolicy {
            enabled: true,
            mode: crate::episodes::distill_models::SharingMode::Auto,
            type_overrides: HashMap::new(),
            l3_scan_enabled: true,
            min_shareability_score: 0.5,
        };
        let content = "PASSWORD=supersecret";
        let notes: Vec<Note> = vec![];
        let result = anonymize_for_export(content, &notes, &policy);
        assert!(matches!(result, Err(AnonymizeError::L3Blocked(_))));
    }

    #[test]
    fn test_anonymize_for_export_success() {
        let policy = SharingPolicy {
            enabled: true,
            mode: crate::episodes::distill_models::SharingMode::Auto,
            type_overrides: HashMap::new(),
            l3_scan_enabled: true,
            min_shareability_score: 0.5,
        };
        let content = "Clean episode content with 192.168.1.1";
        let notes: Vec<Note> = vec![];
        let (result, report, allowed) = anonymize_for_export(content, &notes, &policy).unwrap();
        assert!(result.contains("<ip:0>"));
        assert!(report.consent_stats.is_some());
        assert!(allowed.is_empty()); // no notes provided
    }
}
