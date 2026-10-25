//! Quota of a provider ACCOUNT: until when it refuses us, and what the provider (or a person)
//! said about its limits (QUOTA).
//!
//! The registry answers one question for the router and for the send path: "is this instance
//! exhausted right now, and until when?". Three things can say so:
//!
//! * the provider itself ([`nexus_claude::agent::AgentEvent::Quota`]: Claude Code's
//!   `rate_limit_event`, Codex's `account/rateLimits/updated`) — `source: live`;
//! * a refusal that names its reset (`ProviderError::RateLimited { resets_at_ms }`) or not
//!   (an HTTP 402 "out of credit": [`FALLBACK_COOLDOWN`] applies, and only then);
//! * a person, for a provider that exposes nothing — `source: manual`, a human action.
//!
//! The newest observation wins, whoever made it: a live snapshot observed after a manual entry
//! replaces it, and a manual entry made after the last snapshot replaces that.
//!
//! This is NOT health. An exhausted account is healthy: it answers, and says no. It goes through
//! [`RejectReason::QuotaExhausted`](super::cognitive::candidates::RejectReason), never through
//! `healthy` nor through the breaker (which may read this registry, not the other way round).
//!
//! The clock is injected ([`Clock`]): the exclusion and the re-admission at the reset are
//! asserted with a clock a test drives by hand.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use nexus_claude::agent::{QuotaSnapshot, QuotaSource, QuotaStatus, QuotaWindow};
use serde::{Deserialize, Serialize};

use super::limits::{Clock, SystemClock};
use super::settings::GLOBAL;
use crate::neo4j::GraphStore;

/// How long an instance is held out when it refuses and says nothing about when it will serve
/// again (an HTTP 402, or a 429 without a date). Used ONLY when no reset is known.
pub const FALLBACK_COOLDOWN: Duration = Duration::from_secs(300);

/// Key prefix of a stored entry (`LlmSetting`, scope `global`).
pub const QUOTA_PREFIX: &str = "quota:";

/// Share of a window above which an account is "near its limit": still serving, but the router
/// should prefer another one when it has the choice (QUOTA). A window at or below this is not
/// penalised at all. The STRENGTH of that preference is the routing setting `quota_weight`
/// (0 disables it); this is only where the slope starts.
pub const QUOTA_NEAR_LIMIT: f64 = 0.9;

/// Why an instance is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExhaustionCause {
    /// A window of the provider's own report is full.
    Window,
    /// The provider answered 429.
    RateLimited,
    /// The account has no credit left (HTTP 402).
    OutOfCredit,
    /// A person said so.
    Manual,
}

/// An exhaustion in force: the instance refuses until `resets_at_ms`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exhaustion {
    /// The instant the instance serves again (epoch ms).
    pub resets_at_ms: u64,
    /// Whether the provider (or the person) gave that instant; `false` is the fallback cooldown.
    pub reset_known: bool,
    /// The window that is full, when one is named (`seven_day`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    /// What made the instance exhausted.
    pub cause: ExhaustionCause,
}

/// What is stored for one instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaEntry {
    /// The windows last reported (live), or typed (manual).
    #[serde(default)]
    pub windows: Vec<QuotaWindow>,
    /// Where the entry comes from.
    pub source: QuotaSource,
    /// When it was observed or typed (epoch ms, the registry's clock).
    pub observed_at_ms: u64,
    /// Who typed it, for a manual entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set_by: Option<String>,
    /// The exhaustion this entry carries, if any (may be in the past: see [`QuotaRegistry::exhausted`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exhaustion: Option<Exhaustion>,
}

/// What a person types for a provider that exposes nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManualQuota {
    /// The account refuses until this instant (epoch ms).
    #[serde(default)]
    pub exhausted_until_ms: Option<u64>,
    /// Share of the window used (`0.0..=1.0`), for the gauge.
    #[serde(default)]
    pub utilization: Option<f64>,
    /// When the window resets (epoch ms), for the gauge.
    #[serde(default)]
    pub resets_at_ms: Option<u64>,
    /// Name of the window (`seven_day`, `five_hour`, `monthly`…); `manual` when absent.
    #[serde(default)]
    pub window: Option<String>,
}

/// An entry as the API shows it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuotaView {
    /// Instance identifier.
    pub provider_id: String,
    /// The windows (gauge).
    pub windows: Vec<QuotaWindow>,
    /// `live` or `manual`.
    pub source: QuotaSource,
    /// When it was observed (epoch ms).
    pub observed_at_ms: u64,
    /// The exhaustion in force NOW, if any.
    pub exhausted: Option<Exhaustion>,
}

/// Where the entries survive a restart.
#[async_trait::async_trait]
pub trait QuotaStore: Send + Sync {
    /// Every stored entry.
    async fn load_all(&self) -> Result<Vec<(String, QuotaEntry)>>;
    /// Stores one entry.
    async fn save(&self, instance: &str, entry: &QuotaEntry) -> Result<()>;
    /// Forgets one entry.
    async fn delete(&self, instance: &str) -> Result<()>;
}

/// The graph's `LlmSetting` documents (`global`, `quota:<instance>`).
pub struct GraphQuotaStore {
    graph: Arc<dyn GraphStore>,
}

impl GraphQuotaStore {
    /// A store over `graph`.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

#[async_trait::async_trait]
impl QuotaStore for GraphQuotaStore {
    async fn load_all(&self) -> Result<Vec<(String, QuotaEntry)>> {
        Ok(self
            .graph
            .list_llm_settings(GLOBAL, QUOTA_PREFIX)
            .await?
            .into_iter()
            .filter_map(|(key, raw)| {
                let instance = key.strip_prefix(QUOTA_PREFIX).unwrap_or(&key).to_owned();
                // A document that no longer decodes is skipped, never guessed.
                serde_json::from_str(&raw)
                    .ok()
                    .map(|entry| (instance, entry))
            })
            .collect())
    }

    async fn save(&self, instance: &str, entry: &QuotaEntry) -> Result<()> {
        self.graph
            .put_llm_setting(
                GLOBAL,
                &format!("{QUOTA_PREFIX}{instance}"),
                &serde_json::to_string(entry)?,
            )
            .await
    }

    async fn delete(&self, instance: &str) -> Result<()> {
        self.graph
            .delete_llm_setting(GLOBAL, &format!("{QUOTA_PREFIX}{instance}"))
            .await
            .map(|_| ())
    }
}

/// The quota of every instance.
pub struct QuotaRegistry {
    clock: Arc<dyn Clock>,
    entries: Mutex<HashMap<String, QuotaEntry>>,
    store: Option<Arc<dyn QuotaStore>>,
}

impl Default for QuotaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for QuotaRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotaRegistry").finish_non_exhaustive()
    }
}

/// Epoch ms of the clock.
fn ms(clock: &dyn Clock) -> u64 {
    u64::try_from(clock.now().timestamp_millis()).unwrap_or(0)
}

impl QuotaRegistry {
    /// An in-memory registry on the real clock.
    pub fn new() -> Self {
        Self {
            clock: Arc::new(SystemClock),
            entries: Mutex::new(HashMap::new()),
            store: None,
        }
    }

    /// This registry reading `clock` (tests drive it by hand).
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// This registry persisting its entries in `store`.
    pub fn with_store(mut self, store: Arc<dyn QuotaStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// The registry's now (epoch ms).
    pub fn now_ms(&self) -> u64 {
        ms(self.clock.as_ref())
    }

    /// Reads the stored entries (at startup). A failing store leaves the registry empty and says
    /// why: quota is a hint the next refusal re-creates, never a reason not to boot.
    pub async fn load(&self) -> Result<usize> {
        let Some(store) = &self.store else {
            return Ok(0);
        };
        let loaded = store.load_all().await?;
        let count = loaded.len();
        let mut entries = self.lock();
        for (instance, entry) in loaded {
            entries.insert(instance, entry);
        }
        Ok(count)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, QuotaEntry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes `instance`'s entry to the store, without waiting (no store, or no runtime, nothing).
    fn persist_in_background(&self, instance: &str) {
        let (Some(store), Ok(runtime)) = (&self.store, tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let store = Arc::clone(store);
        let instance = instance.to_owned();
        let entry = self.lock().get(&instance).cloned();
        runtime.spawn(async move {
            let result = match entry {
                Some(entry) => store.save(&instance, &entry).await,
                None => store.delete(&instance).await,
            };
            if let Err(error) = result {
                tracing::warn!(%instance, %error, "storing the quota failed");
            }
        });
    }

    /// Writes `instance`'s entry to the store and waits for it (a person's action: the answer
    /// says it is kept).
    pub async fn persist(&self, instance: &str) -> Result<()> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        let entry = self.lock().get(instance).cloned();
        match entry {
            Some(entry) => store.save(instance, &entry).await,
            None => store.delete(instance).await,
        }
    }

    /// A live snapshot of `instance`'s account: the newest observation, it replaces whatever was
    /// there (a manual entry included).
    pub fn observe(&self, instance: &str, snapshot: &QuotaSnapshot) {
        let now = self.now_ms();
        let exhaustion = snapshot.is_exhausted().then(|| {
            let until = snapshot.exhausted_until_ms();
            Exhaustion {
                resets_at_ms: until.unwrap_or(now + FALLBACK_COOLDOWN.as_millis() as u64),
                reset_known: until.is_some(),
                window: snapshot
                    .windows
                    .iter()
                    .filter(|w| w.status == QuotaStatus::Exhausted)
                    .max_by_key(|w| w.resets_at_ms)
                    .map(|w| w.kind.clone()),
                cause: ExhaustionCause::Window,
            }
        });
        self.lock().insert(
            instance.to_owned(),
            QuotaEntry {
                windows: snapshot.windows.clone(),
                source: QuotaSource::Live,
                observed_at_ms: now,
                set_by: None,
                exhaustion,
            },
        );
        self.persist_in_background(instance);
    }

    /// The instance refused with a rate limit. `resets_at_ms` is the provider's date when it gave
    /// one, else `retry_after_ms` from now. A 429 that names neither is a transient failure (a
    /// burst), not an exhaustion: nothing is recorded (the turn is retried as before). A later date
    /// already known for the instance is kept: a 429 never shortens what a window report said.
    pub fn mark_rate_limited(
        &self,
        instance: &str,
        resets_at_ms: Option<u64>,
        retry_after_ms: Option<u64>,
    ) {
        let now = self.now_ms();
        let resets = resets_at_ms.or_else(|| retry_after_ms.map(|d| now + d));
        if resets.is_some() {
            self.mark(instance, resets, ExhaustionCause::RateLimited);
        }
    }

    /// The account has no credit left. No date exists for a top-up: the fallback cooldown holds
    /// it out, unless a reset is already known.
    pub fn mark_out_of_credit(&self, instance: &str) {
        self.mark(instance, None, ExhaustionCause::OutOfCredit);
    }

    fn mark(&self, instance: &str, resets_at_ms: Option<u64>, cause: ExhaustionCause) {
        let now = self.now_ms();
        let (until, known) = match resets_at_ms {
            Some(at) => (at, true),
            None => (now + FALLBACK_COOLDOWN.as_millis() as u64, false),
        };
        {
            let mut entries = self.lock();
            let entry = entries
                .entry(instance.to_owned())
                .or_insert_with(|| QuotaEntry {
                    windows: Vec::new(),
                    source: QuotaSource::Live,
                    observed_at_ms: now,
                    set_by: None,
                    exhaustion: None,
                });
            // A reset already known and still ahead is never shortened by a refusal.
            let keep = entry
                .exhaustion
                .as_ref()
                .filter(|e| e.reset_known && e.resets_at_ms > now && e.resets_at_ms >= until);
            if keep.is_none() {
                entry.exhaustion = Some(Exhaustion {
                    resets_at_ms: until,
                    reset_known: known,
                    window: None,
                    cause,
                });
                entry.observed_at_ms = now;
            }
        }
        self.persist_in_background(instance);
    }

    /// A person's entry for a provider that exposes nothing: replaces what was there (it is the
    /// newest word). `set_by` is who typed it.
    pub fn set_manual(&self, instance: &str, manual: &ManualQuota, set_by: &str) {
        let now = self.now_ms();
        let kind = manual.window.clone().unwrap_or_else(|| "manual".to_owned());
        let exhausted = manual.exhausted_until_ms.filter(|until| *until > now);
        let windows =
            if manual.utilization.is_some() || manual.resets_at_ms.is_some() || exhausted.is_some()
            {
                vec![QuotaWindow {
                    kind: kind.clone(),
                    utilization: manual.utilization,
                    resets_at_ms: manual.resets_at_ms.or(manual.exhausted_until_ms),
                    status: if exhausted.is_some() {
                        QuotaStatus::Exhausted
                    } else {
                        QuotaStatus::Allowed
                    },
                }]
            } else {
                Vec::new()
            };
        self.lock().insert(
            instance.to_owned(),
            QuotaEntry {
                windows,
                source: QuotaSource::Manual,
                observed_at_ms: now,
                set_by: Some(set_by.to_owned()),
                exhaustion: exhausted.map(|until| Exhaustion {
                    resets_at_ms: until,
                    reset_known: true,
                    window: Some(kind),
                    cause: ExhaustionCause::Manual,
                }),
            },
        );
        // The caller awaits `persist` (a person's action); the background write is for events.
    }

    /// Forgets what is known of `instance`.
    pub fn clear(&self, instance: &str) {
        self.lock().remove(instance);
    }

    /// The exhaustion of `instance` in force NOW, `None` once its reset has passed: the instance
    /// is back without anyone doing anything.
    pub fn exhausted(&self, instance: &str) -> Option<Exhaustion> {
        let now = self.now_ms();
        self.lock()
            .get(instance)?
            .exhaustion
            .clone()
            .filter(|e| e.resets_at_ms > now)
    }

    /// How full the FULLEST window of `instance` is (`0.0..=1.0`), whatever its reset: what the
    /// scorer turns into a soft preference. `None` when nothing is known, which is NOT 0 — an
    /// account nobody has measured must not look emptier than one reported at 10 %.
    ///
    /// Unlike [`Self::exhausted`] this ignores the reset: a window reported at 97 % still says 97 %
    /// until a newer observation says otherwise.
    pub fn pressure(&self, instance: &str) -> Option<f64> {
        self.lock()
            .get(instance)?
            .windows
            .iter()
            .filter_map(|w| w.utilization)
            .filter(|u| u.is_finite())
            .fold(None, |acc: Option<f64>, u| {
                Some(acc.map_or(u, |a| a.max(u)))
            })
    }

    /// Every instance with something known, for the API and the gauge.
    pub fn views(&self) -> Vec<QuotaView> {
        let now = self.now_ms();
        let mut views: Vec<QuotaView> = self
            .lock()
            .iter()
            .map(|(instance, entry)| QuotaView {
                provider_id: instance.clone(),
                windows: entry.windows.clone(),
                source: entry.source,
                observed_at_ms: entry.observed_at_ms,
                exhausted: entry.exhaustion.clone().filter(|e| e.resets_at_ms > now),
            })
            .collect();
        views.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));
        views
    }

    /// The view of one instance.
    pub fn view(&self, instance: &str) -> Option<QuotaView> {
        self.views().into_iter().find(|v| v.provider_id == instance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::limits::ManualClock;
    use nexus_claude::agent::QuotaWindow;

    const T0: &str = "2026-10-21T10:00:00Z";

    fn registry() -> (Arc<ManualClock>, QuotaRegistry) {
        let clock = Arc::new(ManualClock::at(T0));
        (Arc::clone(&clock), QuotaRegistry::new().with_clock(clock))
    }

    fn ms_of(instant: &str) -> u64 {
        chrono::DateTime::parse_from_rfc3339(instant)
            .unwrap()
            .timestamp_millis() as u64
    }

    fn window(kind: &str, status: QuotaStatus, resets: Option<u64>) -> QuotaWindow {
        QuotaWindow {
            kind: kind.into(),
            utilization: Some(1.0),
            resets_at_ms: resets,
            status,
        }
    }

    #[test]
    fn an_exhausted_window_holds_the_instance_out_until_its_reset_then_lets_it_back() {
        let (clock, registry) = registry();
        let reset = ms_of("2026-10-24T10:00:00Z");
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![
                window(
                    "five_hour",
                    QuotaStatus::Allowed,
                    Some(ms_of("2026-10-21T15:00:00Z")),
                ),
                window("seven_day", QuotaStatus::Exhausted, Some(reset)),
            ]),
        );
        let held = registry.exhausted("claude-code").expect("exhausted");
        assert_eq!(held.resets_at_ms, reset);
        assert_eq!(held.window.as_deref(), Some("seven_day"));
        assert!(held.reset_known);
        // Three days later, a second before the reset: still out.
        clock.advance(Duration::from_secs(3 * 86_400 - 1));
        assert!(registry.exhausted("claude-code").is_some());
        // At the reset: back, nobody did anything.
        clock.advance(Duration::from_secs(1));
        assert_eq!(registry.exhausted("claude-code"), None);
    }

    #[test]
    fn a_refusal_without_a_date_holds_the_fallback_cooldown_and_a_dated_one_holds_its_date() {
        let (clock, registry) = registry();
        registry.mark_out_of_credit("deepseek");
        let held = registry.exhausted("deepseek").unwrap();
        assert!(!held.reset_known);
        assert_eq!(held.cause, ExhaustionCause::OutOfCredit);
        clock.advance(FALLBACK_COOLDOWN - Duration::from_secs(1));
        assert!(registry.exhausted("deepseek").is_some());
        clock.advance(Duration::from_secs(2));
        assert!(registry.exhausted("deepseek").is_none());

        let dated = clock.now().timestamp_millis() as u64 + 3_600_000;
        registry.mark_rate_limited("codex", Some(dated), None);
        let held = registry.exhausted("codex").unwrap();
        assert!(held.reset_known);
        assert_eq!(held.resets_at_ms, dated, "the date, not the 300 s");
        // A 429 with only a delay is dated from the clock.
        registry.mark_rate_limited("other", None, Some(90_000));
        assert_eq!(
            registry.exhausted("other").unwrap().resets_at_ms,
            clock.now().timestamp_millis() as u64 + 90_000
        );
    }

    #[test]
    fn a_refusal_never_shortens_a_reset_already_known() {
        let (clock, registry) = registry();
        let weekly = ms_of("2026-10-24T10:00:00Z");
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![window(
                "seven_day",
                QuotaStatus::Exhausted,
                Some(weekly),
            )]),
        );
        registry.mark_out_of_credit("claude-code");
        assert_eq!(
            registry.exhausted("claude-code").unwrap().resets_at_ms,
            weekly
        );
        clock.advance(Duration::from_secs(60));
        registry.mark_rate_limited("claude-code", None, Some(1000));
        assert_eq!(
            registry.exhausted("claude-code").unwrap().resets_at_ms,
            weekly
        );
    }

    #[test]
    fn the_newest_word_wins_between_a_person_and_the_provider() {
        let (clock, registry) = registry();
        let until = ms_of("2026-10-22T10:00:00Z");
        registry.set_manual(
            "deepseek",
            &ManualQuota {
                exhausted_until_ms: Some(until),
                utilization: None,
                resets_at_ms: None,
                window: Some("monthly".into()),
            },
            "me@example.com",
        );
        let held = registry.exhausted("deepseek").unwrap();
        assert_eq!(held.cause, ExhaustionCause::Manual);
        assert_eq!(
            registry.view("deepseek").unwrap().source,
            QuotaSource::Manual
        );
        // The provider speaks later: its word replaces the person's.
        clock.advance(Duration::from_secs(60));
        registry.observe(
            "deepseek",
            &QuotaSnapshot::live(vec![window("requests", QuotaStatus::Allowed, None)]),
        );
        assert_eq!(registry.exhausted("deepseek"), None);
        assert_eq!(registry.view("deepseek").unwrap().source, QuotaSource::Live);
        // The person speaks after: theirs replaces the snapshot.
        clock.advance(Duration::from_secs(60));
        registry.set_manual(
            "deepseek",
            &ManualQuota {
                exhausted_until_ms: Some(until),
                utilization: None,
                resets_at_ms: None,
                window: None,
            },
            "me@example.com",
        );
        assert!(registry.exhausted("deepseek").is_some());
    }

    #[test]
    fn a_manual_date_in_the_past_exhausts_nothing() {
        let (_clock, registry) = registry();
        registry.set_manual(
            "deepseek",
            &ManualQuota {
                exhausted_until_ms: Some(ms_of("2026-10-20T10:00:00Z")),
                utilization: Some(0.4),
                resets_at_ms: Some(ms_of("2026-10-31T00:00:00Z")),
                window: None,
            },
            "me",
        );
        assert_eq!(registry.exhausted("deepseek"), None);
        let view = registry.view("deepseek").unwrap();
        assert_eq!(view.windows[0].utilization, Some(0.4));
    }

    /// A store in memory, standing for the graph across a restart.
    #[derive(Default)]
    struct MemStore(Mutex<HashMap<String, QuotaEntry>>);

    #[async_trait::async_trait]
    impl QuotaStore for MemStore {
        async fn load_all(&self) -> Result<Vec<(String, QuotaEntry)>> {
            Ok(self.0.lock().unwrap().clone().into_iter().collect())
        }
        async fn save(&self, instance: &str, entry: &QuotaEntry) -> Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(instance.to_owned(), entry.clone());
            Ok(())
        }
        async fn delete(&self, instance: &str) -> Result<()> {
            self.0.lock().unwrap().remove(instance);
            Ok(())
        }
    }

    /// The RECORDED `rate_limit_event` of Claude Code 2.1.296 (copied verbatim from the parity
    /// fixture of nexus #137) goes through nexus's OWN mapper and lands in the registry.
    ///
    /// The one line done here by hand — taking `rate_limit_info` out of the envelope — is exactly
    /// what nexus's `parse_message` does; that function is private to the SDK, and nexus tests it
    /// (`a_rate_limit_event_is_kept_as_a_system_message_carrying_its_rate_limit_info`). Everything
    /// that reads the payload (window names, utilisations, epoch SECONDS to ms, status) is nexus's
    /// code, not a second parser written here: the day the CLI changes its shape, this test fails
    /// with the provider rather than agreeing with a copy of it.
    #[test]
    fn the_recorded_rate_limit_event_of_the_cli_lands_in_the_registry_through_nexus() {
        use nexus_claude::agent::AgentEvent;
        use nexus_claude::providers::claude_code::{map_message, MapState};
        use nexus_claude::Message;

        const RECORDED: &str =
            include_str!("../../../tests/fixtures/quota/rate_limit_event.ndjson");
        let envelope: serde_json::Value = serde_json::from_str(RECORDED.trim()).unwrap();
        assert_eq!(envelope["type"], "rate_limit_event", "the recorded line");

        let events = map_message(
            &Message::System {
                subtype: "rate_limit_event".to_string(),
                data: envelope["rate_limit_info"].clone(),
            },
            &mut MapState::new(),
        );
        let [AgentEvent::Quota { quota }] = events.as_slice() else {
            panic!("nexus maps a rate_limit_event to exactly one Quota event, got {events:?}");
        };

        let (_clock, registry) = registry();
        registry.observe("claude-code", quota);

        // Both windows of the real payload, with their own resets, in epoch MILLIseconds.
        let view = registry.view("claude-code").expect("the instance is known");
        assert_eq!(view.source, QuotaSource::Live);
        let window = |kind: &str| {
            view.windows
                .iter()
                .find(|w| w.kind == kind)
                .unwrap_or_else(|| panic!("the payload names {kind}: {:?}", view.windows))
                .clone()
        };
        assert_eq!(window("seven_day").utilization, Some(0.97));
        assert_eq!(window("seven_day").resets_at_ms, Some(1_792_828_800_000));
        assert_eq!(window("five_hour").utilization, Some(0.09));
        assert_eq!(window("five_hour").resets_at_ms, Some(1_792_597_800_000));

        // `allowed_warning` is NOT a refusal: the account still serves, so nothing is held out.
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "a warning must never hold an account that still answers"
        );
        // What the scorer reads: the FULLEST window, 97 %, which is past QUOTA_NEAR_LIMIT.
        assert_eq!(registry.pressure("claude-code"), Some(0.97));
        assert!(registry.pressure("claude-code").unwrap() > QUOTA_NEAR_LIMIT);
    }

    /// The pressure is the FULLEST window, and an account nobody measured stays unknown (`None`),
    /// which the scorer must not read as "empty".
    #[test]
    fn the_pressure_is_the_fullest_window_and_an_unmeasured_account_stays_unknown() {
        let (_clock, registry) = registry();
        assert_eq!(registry.pressure("nobody"), None, "never measured");

        registry.observe(
            "both",
            &QuotaSnapshot::live(vec![
                QuotaWindow {
                    kind: "five_hour".into(),
                    utilization: Some(0.10),
                    resets_at_ms: None,
                    status: QuotaStatus::Allowed,
                },
                QuotaWindow {
                    kind: "seven_day".into(),
                    utilization: Some(0.93),
                    resets_at_ms: None,
                    status: QuotaStatus::Allowed,
                },
            ]),
        );
        assert_eq!(registry.pressure("both"), Some(0.93), "the fullest window");

        // A report that carries no utilisation at all says nothing, it does not say zero.
        registry.observe(
            "silent",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "seven_day".into(),
                utilization: None,
                resets_at_ms: None,
                status: QuotaStatus::Allowed,
            }]),
        );
        assert_eq!(registry.pressure("silent"), None);
    }

    #[tokio::test]
    async fn the_entries_survive_a_restart() {
        let store = Arc::new(MemStore::default());
        let clock = Arc::new(ManualClock::at(T0));
        let before = QuotaRegistry::new()
            .with_clock(clock.clone())
            .with_store(store.clone());
        let until = ms_of("2026-10-24T10:00:00Z");
        before.set_manual(
            "deepseek",
            &ManualQuota {
                exhausted_until_ms: Some(until),
                utilization: None,
                resets_at_ms: None,
                window: None,
            },
            "me",
        );
        before.persist("deepseek").await.unwrap();

        let after = QuotaRegistry::new().with_clock(clock).with_store(store);
        assert_eq!(after.exhausted("deepseek"), None, "nothing before the load");
        assert_eq!(after.load().await.unwrap(), 1);
        assert_eq!(after.exhausted("deepseek").unwrap().resets_at_ms, until);
    }

    #[tokio::test]
    async fn the_graph_store_round_trips_through_llm_settings() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let store = GraphQuotaStore::new(graph);
        let entry = QuotaEntry {
            windows: vec![window("seven_day", QuotaStatus::Exhausted, Some(42))],
            source: QuotaSource::Live,
            observed_at_ms: 7,
            set_by: None,
            exhaustion: Some(Exhaustion {
                resets_at_ms: 42,
                reset_known: true,
                window: Some("seven_day".into()),
                cause: ExhaustionCause::Window,
            }),
        };
        store.save("claude-code", &entry).await.unwrap();
        assert_eq!(
            store.load_all().await.unwrap(),
            vec![("claude-code".to_owned(), entry)]
        );
        store.delete("claude-code").await.unwrap();
        assert!(store.load_all().await.unwrap().is_empty());
    }
}
