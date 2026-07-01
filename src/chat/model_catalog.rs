//! Live Claude model catalog — merges Anthropic's Models API with local UI curation.
//!
//! Historically, the list of selectable Claude models was hardcoded in two
//! independent places (this backend's model list and the frontend's model
//! selector), which meant every model rename or addition required a manual
//! code change + release in both. This module fetches
//! `GET https://api.anthropic.com/v1/models` (when an API key is configured)
//! and caches the result in memory with a long TTL, so new/renamed models
//! show up automatically.
//!
//! Design goals:
//! - **Never block a request on network I/O.** Reads always return whatever
//!   is currently cached; a stale cache triggers a background refresh
//!   (stale-while-revalidate) rather than making the caller wait.
//! - **Never look broken.** If no API key is configured, or the Anthropic API
//!   is unreachable, we fall back to a small static list — the model
//!   selector always has *something* sensible to show.
//! - **Don't hammer the API.** The cache TTL is deliberately long (hours, not
//!   minutes) — a new model appearing a few hours late is fine.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// How long a cached catalog is considered fresh before a background refresh
/// is triggered.
const CACHE_TTL: Duration = Duration::from_secs(12 * 60 * 60); // 12h

/// HTTP timeout for the Anthropic Models API call itself.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

const ANTHROPIC_MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Defensive cap on pagination — the live catalog is small (dozens of
/// models at most); this just prevents a buggy/malicious `has_more` loop.
const MAX_PAGES: u8 = 5;

/// A single model entry, shaped for direct consumption by the frontend's
/// model selector (mirrors `ModelDefinition` in `frontend/src/constants/models.ts`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelDefinition {
    /// Official Anthropic API model ID (e.g. "claude-sonnet-4-6")
    pub id: String,
    /// Short display label for compact UI (e.g. "Sonnet 4.6")
    pub short_label: String,
    /// Full marketing name (e.g. "Claude Sonnet 4.6")
    pub full_label: String,
    /// Tailwind dot color class (e.g. "bg-blue-400")
    pub dot_color: String,
    /// One-line description for selection cards
    pub description: String,
}

/// Manual curation for known model families, in the preferred display order.
/// `(id, short_label, dot_color, description)`. `full_label` is always
/// derived as `"Claude {short_label}"` — every known model follows that
/// pattern, so it isn't worth a fifth tuple field.
///
/// Models not listed here (including brand-new ones the live API returns)
/// still show up — see `derive_fallback` — just without a curated color,
/// abbreviation, or description until someone adds an entry here.
const CURATED_ORDER: &[(&str, &str, &str, &str)] = &[
    (
        "claude-sonnet-5",
        "Sonnet 5",
        "bg-rose-500",
        "Most capable — demanding reasoning & long-horizon agentic work",
    ),
    (
        "claude-opus-4-8",
        "Opus 4.8",
        "bg-violet-500",
        "Most intelligent — complex reasoning",
    ),
    (
        "claude-opus-4-7",
        "Opus 4.7",
        "bg-violet-400",
        "Previous Opus — complex reasoning",
    ),
    (
        "claude-opus-4-6",
        "Opus 4.6",
        "bg-violet-300",
        "Older Opus — complex reasoning",
    ),
    (
        "claude-sonnet-4-6",
        "Sonnet 4.6",
        "bg-blue-400",
        "Fast & capable — best for most tasks",
    ),
    (
        "claude-haiku-4-5",
        "Haiku 4.5",
        "bg-emerald-400",
        "Fastest — lightweight tasks",
    ),
];

fn curated_lookup(id: &str) -> Option<ModelDefinition> {
    CURATED_ORDER.iter().find(|(cid, ..)| *cid == id).map(
        |(id, short_label, dot_color, description)| ModelDefinition {
            id: id.to_string(),
            short_label: short_label.to_string(),
            full_label: format!("Claude {short_label}"),
            dot_color: dot_color.to_string(),
            description: description.to_string(),
        },
    )
}

/// Derive a readable short label from an unknown model ID.
///
/// `"claude-foo-bar-7"` → `"Foo Bar 7"`; trailing numeric segments are
/// grouped with dots (`"claude-sonnet-4-5"` → `"Sonnet 4.5"`). Mirrors the
/// frontend's `getModelShortLabel` fallback so an unrecognized ID from the
/// live API never renders as a raw slug.
fn derive_short_label(id: &str) -> String {
    let without_prefix = id.strip_prefix("claude-").unwrap_or(id);
    let parts: Vec<&str> = without_prefix.split('-').collect();

    let mut text_parts: Vec<String> = Vec::new();
    let mut num_parts: Vec<&str> = Vec::new();

    for part in parts {
        if !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()) {
            num_parts.push(part);
        } else {
            if !num_parts.is_empty() {
                text_parts.push(num_parts.join("."));
                num_parts.clear();
            }
            let mut chars = part.chars();
            let capitalized = match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            };
            text_parts.push(capitalized);
        }
    }
    if !num_parts.is_empty() {
        text_parts.push(num_parts.join("."));
    }

    if text_parts.is_empty() {
        id.to_string()
    } else {
        text_parts.join(" ")
    }
}

/// Family-based color fallback for an unrecognized model ID (mirrors the
/// frontend's `getModelDotColor` fallback).
fn derive_dot_color(id: &str) -> String {
    if id.contains("opus") {
        "bg-violet-400".to_string()
    } else if id.contains("haiku") {
        "bg-emerald-400".to_string()
    } else {
        "bg-blue-400".to_string() // sonnet / default
    }
}

/// Build a full `ModelDefinition` for a model ID the live API returned,
/// preferring curated data and falling back to derived heuristics.
fn resolve_model(id: &str, api_display_name: Option<&str>) -> ModelDefinition {
    if let Some(curated) = curated_lookup(id) {
        return curated;
    }
    let short_label = derive_short_label(id);
    ModelDefinition {
        id: id.to_string(),
        full_label: api_display_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Claude {short_label}")),
        dot_color: derive_dot_color(id),
        short_label,
        description: String::new(),
    }
}

/// The static list used when no API key is configured, or the live fetch
/// fails and nothing has ever been cached yet. Order matches `CURATED_ORDER`.
fn static_fallback_models() -> Vec<ModelDefinition> {
    CURATED_ORDER
        .iter()
        .map(|(id, ..)| curated_lookup(id).expect("id comes from CURATED_ORDER itself"))
        .collect()
}

#[derive(Debug, Deserialize)]
struct AnthropicModelsResponse {
    data: Vec<AnthropicModelEntry>,
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicModelEntry {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
}

struct CacheState {
    models: Vec<ModelDefinition>,
    fetched_at: Instant,
    refreshing: bool,
}

/// Shared, lazily-refreshed cache of the Claude model catalog.
pub struct ModelCatalogCache {
    inner: RwLock<CacheState>,
    http: reqwest::Client,
    api_key: Option<String>,
}

impl ModelCatalogCache {
    /// Construct a new cache. Does **not** make any network calls — the
    /// first call to `get_models()` seeds it with the static fallback and
    /// kicks off a background refresh if an API key is configured.
    pub fn new(api_key: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(CacheState {
                models: static_fallback_models(),
                // Force the first `get_models()` call to treat this as stale
                // so a real fetch is scheduled immediately when a key exists.
                fetched_at: Instant::now() - CACHE_TTL - Duration::from_secs(1),
                refreshing: false,
            }),
            http: reqwest::Client::builder()
                .timeout(FETCH_TIMEOUT)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            api_key,
        })
    }

    /// Return the current catalog, triggering a background refresh if the
    /// cache is stale (or empty) and no refresh is already in flight. Never
    /// blocks on network I/O — always returns immediately with whatever is
    /// cached (which is at minimum the static fallback list).
    pub async fn get_models(self: &Arc<Self>) -> Vec<ModelDefinition> {
        let needs_refresh = {
            let state = self.inner.read().await;
            !state.refreshing && state.fetched_at.elapsed() >= CACHE_TTL
        };

        if needs_refresh && self.api_key.is_some() {
            let mut state = self.inner.write().await;
            // Re-check under the write lock — another task may have started
            // the refresh between our read and this write.
            if !state.refreshing && state.fetched_at.elapsed() >= CACHE_TTL {
                state.refreshing = true;
                let this = Arc::clone(self);
                tokio::spawn(async move {
                    this.refresh().await;
                });
            }
        }

        self.inner.read().await.models.clone()
    }

    /// Force an immediate synchronous refresh attempt (used by tests and by
    /// an optional "refresh now" admin action). Falls back silently to the
    /// existing cache on any failure.
    async fn refresh(self: &Arc<Self>) {
        let result = self.fetch_live_catalog().await;

        let mut state = self.inner.write().await;
        match result {
            Ok(models) if !models.is_empty() => {
                tracing::info!(
                    count = models.len(),
                    "Refreshed Claude model catalog from Anthropic Models API"
                );
                state.models = models;
                state.fetched_at = Instant::now();
            }
            Ok(_) => {
                tracing::warn!(
                    "Anthropic Models API returned an empty catalog — keeping previous list"
                );
                // Still bump fetched_at so we don't hammer the API every request.
                state.fetched_at = Instant::now();
            }
            Err(err) => {
                tracing::warn!(error = %err, "Failed to refresh Claude model catalog — keeping previous list");
                // Bump fetched_at anyway (with a shorter effective backoff isn't
                // worth the complexity here — 12h between attempts on a broken
                // key/network is an acceptable ceiling on wasted calls).
                state.fetched_at = Instant::now();
            }
        }
        state.refreshing = false;
    }

    async fn fetch_live_catalog(&self) -> anyhow::Result<Vec<ModelDefinition>> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no Anthropic API key configured"))?;

        let mut entries: Vec<AnthropicModelEntry> = Vec::new();
        let mut after_id: Option<String> = None;

        for _ in 0..MAX_PAGES {
            let mut req = self
                .http
                .get(ANTHROPIC_MODELS_URL)
                .header("x-api-key", api_key)
                .header("anthropic-version", ANTHROPIC_VERSION);
            if let Some(cursor) = &after_id {
                req = req.query(&[("after_id", cursor.as_str())]);
            }

            let resp = req.send().await?.error_for_status()?;
            let page: AnthropicModelsResponse = resp.json().await?;
            let has_more = page.has_more;
            let last_id = page.last_id.clone();
            entries.extend(page.data);

            if !has_more || last_id.is_none() {
                break;
            }
            after_id = last_id;
        }

        // Curated models first (in our preferred order), then anything the
        // live API knows about that we haven't curated yet, newest-looking
        // (API order) last.
        let mut seen = std::collections::HashSet::new();
        let mut models: Vec<ModelDefinition> = Vec::new();

        for (id, ..) in CURATED_ORDER {
            if let Some(entry) = entries.iter().find(|e| e.id == *id) {
                models.push(resolve_model(&entry.id, entry.display_name.as_deref()));
                seen.insert(entry.id.clone());
            }
        }
        for entry in &entries {
            if seen.contains(&entry.id) {
                continue;
            }
            models.push(resolve_model(&entry.id, entry.display_name.as_deref()));
            seen.insert(entry.id.clone());
        }

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_short_label_known_pattern() {
        assert_eq!(derive_short_label("claude-sonnet-4-5"), "Sonnet 4.5");
        assert_eq!(derive_short_label("claude-opus-4-8"), "Opus 4.8");
        assert_eq!(derive_short_label("claude-haiku-4-5"), "Haiku 4.5");
    }

    #[test]
    fn test_derive_short_label_unknown_family() {
        assert_eq!(derive_short_label("claude-foo-bar-7"), "Foo Bar 7");
    }

    #[test]
    fn test_derive_dot_color_families() {
        assert_eq!(derive_dot_color("claude-opus-4-9"), "bg-violet-400");
        assert_eq!(derive_dot_color("claude-haiku-5"), "bg-emerald-400");
        assert_eq!(derive_dot_color("claude-sonnet-5"), "bg-blue-400");
    }

    #[test]
    fn test_curated_lookup_no_fable() {
        // The whole point of this module's introduction: Fable 5 was renamed
        // to Sonnet 5 — it must not resurface via curation.
        assert!(curated_lookup("claude-fable-5").is_none());
        assert!(curated_lookup("claude-sonnet-5").is_some());
    }

    #[test]
    fn test_static_fallback_models_nonempty_and_ordered() {
        let models = static_fallback_models();
        assert!(!models.is_empty());
        assert_eq!(models[0].id, "claude-sonnet-5");
        assert!(models.iter().all(|m| m.id != "claude-fable-5"));
    }

    #[test]
    fn test_resolve_model_prefers_curated_over_api_display_name() {
        let m = resolve_model("claude-sonnet-5", Some("Some Other Name"));
        assert_eq!(m.full_label, "Claude Sonnet 5");
        assert_eq!(m.short_label, "Sonnet 5");
    }

    #[test]
    fn test_resolve_model_falls_back_for_unknown_id() {
        let m = resolve_model("claude-new-hotness-9", Some("Claude New Hotness 9"));
        assert_eq!(m.full_label, "Claude New Hotness 9");
        assert_eq!(m.short_label, "New Hotness 9");
        assert_eq!(m.dot_color, "bg-blue-400");
        assert_eq!(m.description, "");
    }

    #[tokio::test]
    async fn test_get_models_without_api_key_returns_static_fallback() {
        let cache = ModelCatalogCache::new(None);
        let models = cache.get_models().await;
        assert_eq!(models, static_fallback_models());
        // No API key — must never attempt a refresh (would flip `refreshing`
        // to true and stay there since fetch_live_catalog short-circuits an
        // error, but we assert the simpler, load-bearing invariant: no crash,
        // stable fallback content).
        let models_again = cache.get_models().await;
        assert_eq!(models_again, models);
    }
}
