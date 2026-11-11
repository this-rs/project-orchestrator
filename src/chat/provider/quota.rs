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
//! NOT EVERY FULL WINDOW REFUSES THE ACCOUNT. A provider also reports the allowance it sells ON TOP
//! of the plan (extra usage: Claude Code's `seven_day_overage_included`). Such a window being full
//! says what the ADD-ON has left, not what the account has left; condemning the instance for it
//! takes a serving account out of routing for as long as that window's date. Measured twice
//! (10/11 and 11/11/2026, see [`window_condemns_account`]): the account answered on another model
//! minutes after the overage window was reported `exhausted`. The window is kept — it feeds
//! [`QuotaRegistry::pressure`] and [`QuotaView`] — and it never closes the door.
//!
//! A word from the provider is a HYPOTHESIS; a served turn is PROOF. Every hold is therefore
//! re-observed after [`DATED_EXHAUSTION_STALENESS`] without a human, and a refusal that comes back
//! re-arms it: nobody has to clear an entry after resetting an account by hand.
//!
//! The clock is injected ([`Clock`]): the exclusion and the re-admission at the reset are
//! asserted with a clock a test drives by hand.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

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

/// How long a DATED exhaustion (the provider named its reset) is trusted before it is re-observed.
///
/// A `rate_limit_event` names the instant its window resets, and that date is reliable for an
/// account that is genuinely out. But it cannot see a person resetting it by hand — a top-up, a
/// plan change, a console refresh — which makes the dated event stale while it still claims the
/// account is out for days. This is the 10/11/2026 case: `seven_day` served at 69 % while
/// `seven_day_overage_included` was dated `exhausted` until 2026-11-14, and the exclusion held for
/// 3.7 days with no turn ever allowed to re-observe it.
///
/// Once an observation is this old, [`QuotaRegistry::exhausted`] drops it so the next turn is
/// admitted as a SINGLE re-observation probe. If the account truly still refuses, the fresh
/// refusal re-arms the entry with a new observation time, so a genuinely-exhausted account is
/// probed at most once per window — never hammered — and the exclusion stays effective until the
/// provider proves otherwise.
pub const DATED_EXHAUSTION_STALENESS: Duration = Duration::from_secs(15 * 60);

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
    /// When this hold was last CONFIRMED (epoch ms, the registry's clock), so that its age is its
    /// own and not the entry's: an unrelated snapshot arriving must not make a refusal from three
    /// days ago look fresh. `None` on an entry stored before this field existed — the entry's own
    /// observation time decides then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at_ms: Option<u64>,
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
    /// Per-model holds: a refusal named ONE model (`model:<name>` in the `window`), so the other
    /// models of the instance stay eligible. Keyed by model name; expiry is the same clock rule
    /// as [`Self::exhaustion`] (a hold whose reset has passed filters nothing).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub model_holds: HashMap<String, Exhaustion>,
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
    /// Tracks the `observed_at_ms` of the last write we DISPATCHED to the store for each
    /// instance. A write whose `observed_at_ms` is not strictly newer is skipped (ordering +
    /// dedup). A deletion records the deletion time, preventing in-flight older writes from
    /// resurrecting the entry.
    store_watermark: Arc<Mutex<HashMap<String, u64>>>,
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

/// Whether a held name (`"Fable"`, `"opus"`, `"claude-opus-4-1"`) covers `model`.
///
/// Providers refuse a FAMILY, and name it in prose while the host asks for an id: the
/// Claude Code refusal "You've reached your Fable limit" is about `claude-fable-5-1`,
/// and its windows are called `seven_day_opus`. A held name of four characters or
/// more, found in the model id, is that family. An id held exactly is that model.
/// Shorter names are refused as a match on purpose: `"max"`, `"1m"` would take out
/// half a catalogue.
fn same_family(held: &str, model: &str) -> bool {
    let held = held.trim().to_ascii_lowercase();
    let model = model.trim().to_ascii_lowercase();
    if held.is_empty() || model.is_empty() {
        return false;
    }
    if held == model {
        return true;
    }
    held.len() >= 4 && model.contains(&held)
}

/// What a rate-limit window of the provider, being `Exhausted`, actually CLOSES. The provider
/// names its windows, so the answer is read from the name — and there are exactly three:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowScope<'a> {
    /// Every model of the account: `five_hour`, `seven_day`, `requests`, `credits`… This is the
    /// "100 % of all models" limit, and it is the ONLY one that excludes the instance.
    Account,
    /// ONE family: `seven_day_fable`, `seven_day_opus`. The account serves on the other families,
    /// and the provider says so itself ("switch to another model") — the FAMILY comes back here so
    /// the hold lands on it (`model:<family>`), never on the account.
    Model(&'a str),
    /// An allowance bought ON TOP of the plan: `seven_day_overage_included`. Full, it says what
    /// the add-on has left, not what the account has left.
    AddOn,
}

/// Which scope an exhausted window of that name closes (see [`WindowScope`]).
///
/// The three-way answer is the whole rule, and each arm rests on something measured:
///
/// * **AddOn** — 10/11/2026, found again 11/11/2026: `seven_day` at 0,69→0,72 `allowed` beside
///   `seven_day_overage_included` at 1,0→1,01 `exhausted` until 14/11. The registry held the whole
///   instance out for 3,7 days, and the same account answered `claude-haiku` at 14:45:27 and
///   14:58:17 with that entry in place. The overage window excludes nothing by itself.
/// * **Model** — the CLI's refusal "You've reached your Fable limit. Switch to another model" is a
///   family limit; confusing it with the account's is what made a served account look dead.
/// * **Account** — `five_hour`/`seven_day` full is the account refusing every model.
pub fn window_scope(kind: &str) -> WindowScope<'_> {
    if kind.to_ascii_lowercase().contains("overage") {
        return WindowScope::AddOn;
    }
    match model_of_window(kind) {
        Some(family) => WindowScope::Model(family),
        None => WindowScope::Account,
    }
}

/// The family a per-model window belongs to, when it belongs to one: `seven_day_opus` → `opus`,
/// `seven_day_fable` → `fable`.
///
/// The same five names nexus uses to tell a host which windows it can move away from
/// (`claude-code/map_events.rs::model_of_window`): the two lists must agree, and this one is here
/// because the DECISION — hold a family or hold the account — is the registry's, not the SDK's.
/// A window whose suffix is not a family is the account's own (`seven_day`, `five_hour`).
fn model_of_window(kind: &str) -> Option<&str> {
    const FAMILIES: &[&str] = &["opus", "sonnet", "haiku", "fable", "mythos"];
    let family = kind.rsplit('_').next()?;
    FAMILIES
        .iter()
        .find(|known| family.eq_ignore_ascii_case(known))
        .map(|_| family)
}

/// Whether a stored hold can never exclude the account: a LIVE window report about an add-on
/// allowance ([`WindowScope::AddOn`]). The 10/11 and 11/11 shapes were both this, and the entry
/// survived a restart: dropping it on load repairs the store instead of waiting for a refusal that
/// will never come. A hold a PERSON typed is left alone: that is a different axis.
fn condemned_by_an_add_on(exhaustion: &Exhaustion) -> bool {
    exhaustion.cause == ExhaustionCause::Window
        && exhaustion
            .window
            .as_deref()
            .is_some_and(|kind| window_scope(kind) == WindowScope::AddOn)
}

/// The account-wide hold of `entry` that is in force at `now`, if any — the ONE rule the router
/// ([`QuotaRegistry::exhausted`]) and the interface ([`QuotaRegistry::views`]) both obey.
///
/// Two things can void a hold here:
///
/// * a full ADD-ON window never closes the door: the plan still serves (see [`window_scope`]). The
///   entry may predate that rule — `load` repairs the stored one — so the verdict is given here too;
/// * a DATED hold older than [`DATED_EXHAUSTION_STALENESS`] is stale: the reset the provider named
///   may have been invalidated by a person resetting the account, and no `rate_limit_event`
///   reflects that. Dropping it admits the next turn as a SINGLE re-observation probe, and a
///   refusal the provider then answers re-arms the clock, so a genuinely-exhausted account is
///   probed at most once per window, never hammered. The age is the HOLD's, not the entry's: a
///   snapshot about another window must not make a dated refusal from last week look fresh.
fn hold_in_force(entry: &QuotaEntry, now: u64) -> Option<Exhaustion> {
    let exhaustion = entry.exhaustion.clone().filter(|e| e.resets_at_ms > now)?;
    if condemned_by_an_add_on(&exhaustion) {
        return None;
    }
    let observed = exhaustion.observed_at_ms.unwrap_or(entry.observed_at_ms);
    if exhaustion.reset_known
        && observed.saturating_add(DATED_EXHAUSTION_STALENESS.as_millis() as u64) <= now
    {
        return None;
    }
    Some(exhaustion)
}

/// The key under which the hold of `family` is (or will be) stored: one entry per family, whatever
/// the caller called it. `observe` reads the suffix of a window (`fable`), an adapter can name the
/// family as the provider worded it (`Fable`), and [`same_family`] already treats them as one — so
/// the map must too, or a fresh hold can hide behind a stale one.
fn family_key<'a>(holds: &'a HashMap<String, Exhaustion>, family: &'a str) -> &'a str {
    holds
        .keys()
        .find(|held| same_family(held, family))
        .map_or(family, String::as_str)
}

impl QuotaRegistry {
    /// An in-memory registry on the real clock.
    pub fn new() -> Self {
        Self {
            clock: Arc::new(SystemClock),
            entries: Mutex::new(HashMap::new()),
            store: None,
            store_watermark: Arc::new(Mutex::new(HashMap::new())),
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
        let mut repaired: Vec<String> = Vec::new();
        {
            let mut entries = self.lock();
            for (instance, mut entry) in loaded {
                // A hold that could never exclude the account is dropped, and the store is
                // rewritten: the alternative is a dated exclusion on the screen for days, which
                // nobody can clear and no turn can disprove (10/11 and 11/11/2026).
                if entry
                    .exhaustion
                    .as_ref()
                    .is_some_and(condemned_by_an_add_on)
                {
                    tracing::info!(
                        %instance,
                        "dropping a stored exhaustion on an add-on window: it never excluded the account"
                    );
                    entry.exhaustion = None;
                    repaired.push(instance.clone());
                }
                entries.insert(instance, entry);
            }
        }
        for instance in repaired {
            // Awaited: the repair must be in the store before the first turn can re-observe it,
            // and a test must be able to see it without racing a background task.
            if let Err(error) = self.persist(&instance).await {
                tracing::warn!(%instance, %error, "repairing a stored quota failed");
            }
        }
        Ok(count)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, QuotaEntry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes `instance`'s entry to the store, without waiting (no store, or no runtime, nothing).
    ///
    /// Ordering: a write whose `observed_at_ms` is not strictly newer than the last one we
    /// dispatched for this instance is skipped. A deletion uses the clock's now, which is
    /// always ≥ every preceding observation. Dedup: identical timestamps are skipped, so two
    /// observations that land within the same millisecond produce at most one write.
    fn persist_in_background(&self, instance: &str) {
        let (Some(store), Ok(runtime)) = (&self.store, tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let store = Arc::clone(store);
        let instance = instance.to_owned();
        let entry = self.lock().get(&instance).cloned();
        // The write's timestamp: the entry's observation time, or the clock's now for a deletion.
        let write_at = entry
            .as_ref()
            .map_or_else(|| self.now_ms(), |e| e.observed_at_ms);
        let watermark = Arc::clone(&self.store_watermark);
        {
            let mut wm = watermark.lock().unwrap_or_else(|e| e.into_inner());
            if write_at <= wm.get(&instance).copied().unwrap_or(0) {
                return; // Not strictly newer: skip (ordering + dedup).
            }
            wm.insert(instance.clone(), write_at);
        }
        runtime.spawn(async move {
            // Re-check: a newer write may have been dispatched while this task waited to run.
            {
                let wm = watermark.lock().unwrap_or_else(|e| e.into_inner());
                if wm.get(&instance).copied().unwrap_or(0) > write_at {
                    return;
                }
            }
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
    /// there (a manual entry included). An observation whose `observed_at_ms` is not strictly
    /// newer than the current entry is ignored: an old event that arrives late (or a restored
    /// entry with a past timestamp) cannot overwrite a newer manual or live entry.
    pub fn observe(&self, instance: &str, snapshot: &QuotaSnapshot) {
        let now = self.now_ms();
        // Each exhausted window is read at ITS scope ([`window_scope`]): the account's own windows
        // condemn the instance, a family's window holds that family, an add-on window does
        // neither. Fable's limit is not the account's 100 %, and neither is a full overage.
        let mut holds: Vec<(String, Exhaustion)> = Vec::new();
        let mut condemning: Vec<&QuotaWindow> = Vec::new();
        for window in &snapshot.windows {
            if window.status != QuotaStatus::Exhausted {
                continue;
            }
            match window_scope(&window.kind) {
                WindowScope::AddOn => tracing::info!(
                    %instance,
                    kind = %window.kind,
                    "a full add-on window is not a refusal: the plan still serves"
                ),
                WindowScope::Model(family) => holds.push((
                    family.to_owned(),
                    Exhaustion {
                        resets_at_ms: window
                            .resets_at_ms
                            .unwrap_or(now + FALLBACK_COOLDOWN.as_millis() as u64),
                        reset_known: window.resets_at_ms.is_some(),
                        window: Some(format!("model:{family}")),
                        cause: ExhaustionCause::Window,
                        observed_at_ms: Some(now),
                    },
                )),
                WindowScope::Account => condemning.push(window),
            }
        }
        let exhaustion = (!condemning.is_empty()).then(|| {
            let until = condemning.iter().filter_map(|w| w.resets_at_ms).max();
            Exhaustion {
                resets_at_ms: until.unwrap_or(now + FALLBACK_COOLDOWN.as_millis() as u64),
                reset_known: until.is_some(),
                window: condemning
                    .iter()
                    .max_by_key(|w| w.resets_at_ms)
                    .map(|w| w.kind.clone()),
                cause: ExhaustionCause::Window,
                observed_at_ms: Some(now),
            }
        });
        let mut new_entry = QuotaEntry {
            windows: snapshot.windows.clone(),
            source: QuotaSource::Live,
            observed_at_ms: now,
            set_by: None,
            exhaustion,
            model_holds: HashMap::new(),
        };
        {
            let mut entries = self.lock();
            if let Some(existing) = entries.get(instance) {
                if existing.observed_at_ms > now {
                    return; // The current entry is newer: keep it.
                }
                // A live account snapshot says nothing about a single model: carry the holds.
                new_entry.model_holds = existing.model_holds.clone();
            }
            // A family the provider now reports full is held — and a reset already known and still
            // ahead is never shortened by it, the same rule as [`Self::mark`].
            for (family, hold) in holds {
                // A live window report is the provider's NEWEST word on that family, so it
                // replaces whatever was held — including a date a refusal had set (a refusal never
                // shortens a known window, but a window report is not a refusal). The key is the
                // family's, so one family never carries two holds.
                let key = family_key(&new_entry.model_holds, &family).to_owned();
                new_entry.model_holds.insert(key, hold);
            }
            entries.insert(instance.to_owned(), new_entry);
        }
        self.persist_in_background(instance);
    }

    /// The instance refused with a rate limit. `resets_at_ms` is the provider's date when it gave
    /// one, else `retry_after_ms` from now. A 429 that names NEITHER falls back to
    /// [`FALLBACK_COOLDOWN`], exactly like a 402: the account did refuse, and a refusal we cannot
    /// date is still a refusal — sending the next turn to the same place would meet it again.
    /// `reset_known` stays `false`, so the interface says "about 5 minutes", never an hour it
    /// invented. A later date already known for the instance is kept: a 429 never shortens what a
    /// window report said.
    pub fn mark_rate_limited(
        &self,
        instance: &str,
        resets_at_ms: Option<u64>,
        retry_after_ms: Option<u64>,
    ) {
        let now = self.now_ms();
        let resets = resets_at_ms.or_else(|| retry_after_ms.map(|d| now.saturating_add(d)));
        self.mark(instance, resets, ExhaustionCause::RateLimited);
    }

    /// The account has no credit left. No date exists for a top-up: the fallback cooldown holds
    /// it out, unless a reset is already known.
    pub fn mark_out_of_credit(&self, instance: &str) {
        self.mark(instance, None, ExhaustionCause::OutOfCredit);
    }

    /// A turn of `instance` SUCCEEDED: the account answered, so an exhaustion we only GUESSED is
    /// over. Only a fallback one is lifted — one whose provider named no date
    /// (`reset_known == false`, the [`FALLBACK_COOLDOWN`] that a 402 or an undated 429 sets).
    ///
    /// A DATED exhaustion is left alone. The provider said when its window comes back; a success
    /// before that date means another window opened (a per-model or per-scope one), which is what
    /// [`Self::observe`] records from a live snapshot — never evidence that the account as a whole
    /// is back. Lifting it here would contradict "the newest observation wins".
    ///
    /// WHY THIS EXISTS. A 402 (top-up) and an undated 429 held the instance for five fixed minutes
    /// and NOTHING lifted it: a top-up, or the provider's own window reset, left the conversation
    /// blocked until the timer ran out. That is exactly what "it did not take my provider reset
    /// into account" looks like from the screen.
    ///
    /// Model-scoped holds are untouched: a success on one model says nothing about another.
    pub fn confirm_available(&self, instance: &str) {
        let lifted = {
            let mut entries = self.lock();
            match entries.get_mut(instance) {
                Some(entry) => match entry.exhaustion.take() {
                    Some(exhaustion) if !exhaustion.reset_known => {
                        entry.observed_at_ms = self.now_ms();
                        true
                    }
                    taken => {
                        entry.exhaustion = taken;
                        false
                    }
                },
                None => false,
            }
        };
        if lifted {
            self.persist_in_background(instance);
        }
    }

    /// ONE model of `provider_id` refused with a rate limit (e.g. "you've reached your limit for
    /// this model, switch to another"). Same reset/cooldown arithmetic as
    /// [`Self::mark_rate_limited`], but scoped to that model only: the instance's other models
    /// stay eligible. The hold is recorded under `window: "model:<name>"` so the filter can tell
    /// it apart from an account-wide exhaustion.
    ///
    /// WIRED IN R4: nothing calls this yet — the producer is a later slice, when the backend bumps
    /// its nexus pin and a typed provider refusal names a model. Keep it: it is the read side's
    /// (`model_exhausted`, `apply_quota`) reason to exist, not dead code.
    pub fn mark_model_rate_limited(
        &self,
        provider_id: &str,
        model: &str,
        resets_at_ms: Option<u64>,
        retry_after_ms: Option<u64>,
    ) {
        let now = self.now_ms();
        let resets = resets_at_ms.or_else(|| retry_after_ms.map(|d| now.saturating_add(d)));
        self.mark_model(provider_id, model, resets, ExhaustionCause::RateLimited);
    }

    fn mark(&self, instance: &str, resets_at_ms: Option<u64>, cause: ExhaustionCause) {
        let now = self.now_ms();
        let (until, known) = match resets_at_ms {
            Some(at) => (at, true),
            None => (
                now.saturating_add(FALLBACK_COOLDOWN.as_millis() as u64),
                false,
            ),
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
                    model_holds: HashMap::new(),
                });
            // A reset already known and still ahead is never shortened by a refusal.
            let keep = entry
                .exhaustion
                .as_ref()
                .filter(|e| e.reset_known && e.resets_at_ms > now && e.resets_at_ms >= until);
            if let Some(kept) = keep
                .is_some()
                .then_some(entry.exhaustion.as_mut())
                .flatten()
            {
                // A refusal is a fresh observation EVEN when it repeats the same date: re-arm the
                // hold's own clock, or a genuinely exhausted account would be re-probed at every
                // turn for ever (the entry's clock no longer decides — the hold's does).
                kept.observed_at_ms = Some(now);
            } else {
                entry.exhaustion = Some(Exhaustion {
                    resets_at_ms: until,
                    reset_known: known,
                    window: None,
                    cause,
                    observed_at_ms: Some(now),
                });
            }
            entry.observed_at_ms = now;
        }
        self.persist_in_background(instance);
    }

    /// The per-model counterpart of [`Self::mark`]: records a hold on `(instance, model)` without
    /// touching the instance's account-wide exhaustion, and keeps a reset already known and still
    /// ahead from being shortened by a later refusal (the same rule as the instance-level one).
    fn mark_model(
        &self,
        instance: &str,
        model: &str,
        resets_at_ms: Option<u64>,
        cause: ExhaustionCause,
    ) {
        let now = self.now_ms();
        let (until, known) = match resets_at_ms {
            Some(at) => (at, true),
            None => (
                now.saturating_add(FALLBACK_COOLDOWN.as_millis() as u64),
                false,
            ),
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
                    model_holds: HashMap::new(),
                });
            use std::collections::hash_map::Entry;
            let key = family_key(&entry.model_holds, model).to_owned();
            match entry.model_holds.entry(key) {
                Entry::Occupied(mut slot) => {
                    // A reset already known and still ahead is never shortened by a refusal — but a
                    // refusal IS a fresh observation, so it re-arms the staleness clock. Without
                    // this, a hold the provider keeps confirming (the same date every time) would
                    // look abandoned and be re-probed every window.
                    let keep = slot.get().reset_known
                        && slot.get().resets_at_ms > now
                        && slot.get().resets_at_ms >= until;
                    if keep {
                        slot.get_mut().observed_at_ms = Some(now);
                    } else {
                        slot.insert(Exhaustion {
                            resets_at_ms: until,
                            reset_known: known,
                            window: Some(format!("model:{model}")),
                            cause,
                            observed_at_ms: Some(now),
                        });
                    }
                    entry.observed_at_ms = now;
                }
                Entry::Vacant(slot) => {
                    slot.insert(Exhaustion {
                        resets_at_ms: until,
                        reset_known: known,
                        window: Some(format!("model:{model}")),
                        cause,
                        observed_at_ms: Some(now),
                    });
                    entry.observed_at_ms = now;
                }
            }
        }
        self.persist_in_background(instance);
    }

    /// A person's entry for a provider that exposes nothing: replaces what was there (it is the
    /// newest word). `set_by` is who typed it. An entry whose `observed_at_ms` is not strictly
    /// newer than the current one is ignored (same guard as [`Self::observe`]).
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
        let mut new_entry = QuotaEntry {
            windows,
            source: QuotaSource::Manual,
            observed_at_ms: now,
            set_by: Some(set_by.to_owned()),
            exhaustion: exhausted.map(|until| Exhaustion {
                resets_at_ms: until,
                reset_known: true,
                window: Some(kind),
                cause: ExhaustionCause::Manual,
                observed_at_ms: Some(now),
            }),
            model_holds: HashMap::new(),
        };
        {
            let mut entries = self.lock();
            if let Some(existing) = entries.get(instance) {
                if existing.observed_at_ms > now {
                    return; // The current entry is newer: keep it.
                }
                // A person's entry names the ACCOUNT: single-model holds are a different axis.
                new_entry.model_holds = existing.model_holds.clone();
            }
            entries.insert(instance.to_owned(), new_entry);
        }
        // The caller awaits `persist` (a person's action); the background write is for events.
    }

    /// Forgets what is known of `instance`. The watermark is advanced so that an in-flight
    /// background write for an older observation cannot resurrect the entry.
    pub fn clear(&self, instance: &str) {
        self.lock().remove(instance);
        self.store_watermark
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(instance.to_owned(), self.now_ms());
    }

    /// The exhaustion of `instance` in force NOW, `None` once its reset has passed: the instance
    /// is back without anyone doing anything.
    pub fn exhausted(&self, instance: &str) -> Option<Exhaustion> {
        let now = self.now_ms();
        let entries = self.lock();
        let entry = entries.get(instance)?;
        hold_in_force(entry, now)
    }

    /// The hold on ONE model of `instance` in force NOW, `None` once its reset has passed or when
    /// the model was never held. `window` carries `model:<name>` so the filter can tell a
    /// model-scoped hold from an account-wide exhaustion ([`Self::exhausted`]).
    pub fn model_exhausted(&self, instance: &str, model: &str) -> Option<Exhaustion> {
        let now = self.now_ms();
        let entries = self.lock();
        let entry = entries.get(instance)?;
        // Exact first, then by FAMILY: a provider refuses a family, not an id. Claude
        // Code says "You've reached your Fable limit" and windows that quota as
        // `seven_day_opus`, while the model in force is `claude-fable-5-1`. Holding the
        // family the provider named must therefore take that family's models out.
        entry
            .model_holds
            .get(model)
            .or_else(|| {
                entry
                    .model_holds
                    .iter()
                    .find(|(held, _)| same_family(held, model))
                    .map(|(_, held)| held)
            })
            .cloned()
            .filter(|e| e.resets_at_ms > now)
            // A hold nobody re-confirmed for DATED_EXHAUSTION_STALENESS is re-observed instead of
            // lasting for the window's date: the same rule as the account-wide one, and the only
            // way a person's provider reset reaches a model that routing keeps out (measured
            // 11/11/2026: the Fable hold was dated 3.7 days out and no turn could try it again
            // to prove the reset had landed).
            .filter(|e| {
                !e.reset_known
                    || e.observed_at_ms
                        .unwrap_or(entry.observed_at_ms)
                        .saturating_add(DATED_EXHAUSTION_STALENESS.as_millis() as u64)
                        > now
            })
    }

    /// How full the FULLEST window of `instance` is (`0.0..=1.0`): what the scorer turns into a
    /// soft preference. `None` when nothing is known, which is NOT 0 — an account nobody has
    /// measured must not look emptier than one reported at 10 %.
    ///
    /// A window whose reset has passed (`resets_at_ms <= now`) no longer counts: the provider
    /// has reset its counter, and carrying the old utilisation penalises an instance that is
    /// now free. A window without a reset time is always counted (the provider did not date it).
    pub fn pressure(&self, instance: &str) -> Option<f64> {
        let now = self.now_ms();
        self.lock()
            .get(instance)?
            .windows
            .iter()
            .filter(|w| w.resets_at_ms.is_none_or(|r| r > now))
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
                // The SAME rule as the router: an interface must never show an exclusion that
                // the next turn would not apply (a stale dated hold, or an add-on window).
                exhausted: hold_in_force(entry, now),
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
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
    fn a_dated_exhaustion_is_trusted_until_it_goes_stale_then_reobserved() {
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
        // Within the trust window the dated word holds the instance out.
        clock.advance(DATED_EXHAUSTION_STALENESS - Duration::from_secs(1));
        assert!(registry.exhausted("claude-code").is_some());
        // Past the trust window, still before the reset: the observation is stale, so the next
        // turn is admitted as a re-observation probe instead of being held out for days.
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "a stale dated exhaustion is dropped, nobody had to do anything"
        );
    }

    /// The 10/11/2026 case, replayed — and found again on 11/11/2026. `seven_day` served at 69 %
    /// while `seven_day_overage_included` was dated `exhausted` until 14/11. The registry held the
    /// WHOLE instance out for 3.7 days, and no turn was allowed to disprove it: the account
    /// answered `claude-haiku` at 14:45:27 and 14:58:17 with that entry in place. An add-on window
    /// never condemns the account: it is kept for the gauge and for `pressure`, the door stays
    /// open.
    #[test]
    fn the_10_11_case_keeps_the_account_routable() {
        let (_clock, registry) = registry();
        let reset = ms_of("2026-11-14T08:00:00Z");
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![
                QuotaWindow {
                    kind: "seven_day".into(),
                    utilization: Some(0.69),
                    resets_at_ms: Some(reset),
                    status: QuotaStatus::Allowed,
                },
                QuotaWindow {
                    kind: "seven_day_overage_included".into(),
                    utilization: Some(1.0),
                    resets_at_ms: Some(reset),
                    status: QuotaStatus::Exhausted,
                },
            ]),
        );
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "the plan serves: a full add-on window is not a refusal of the account"
        );
        // The fact is not lost: the gauge keeps both windows, and the account still presses.
        let view = registry.view("claude-code").expect("observed");
        assert_eq!(view.windows.len(), 2);
        assert_eq!(view.windows[1].status, QuotaStatus::Exhausted);
        assert_eq!(registry.pressure("claude-code"), Some(1.0));
    }

    /// The distinction the whole rule rests on, asserted straight on the helper: a window's NAME
    /// says what it closes. Fable's window is not the account's 100 %, and neither is an overage.
    #[test]
    fn a_windows_name_says_what_it_closes() {
        assert_eq!(window_scope("five_hour"), WindowScope::Account);
        assert_eq!(window_scope("seven_day"), WindowScope::Account);
        assert_eq!(window_scope("requests"), WindowScope::Account);
        assert_eq!(window_scope("credits"), WindowScope::Account);
        assert_eq!(window_scope("seven_day_fable"), WindowScope::Model("fable"));
        assert_eq!(window_scope("seven_day_opus"), WindowScope::Model("opus"));
        // Order matters: `overage_included` is not a family, and it is not the account either.
        assert_eq!(
            window_scope("seven_day_overage_included"),
            WindowScope::AddOn
        );
    }

    /// A FAMILY limit is not the account's 100 %. The provider reports it on its own window
    /// (`seven_day_fable`) or in its own words ("You've reached your Fable limit — switch to
    /// another model"): the family is held, every other model keeps serving, and the instance is
    /// never excluded. Measured 11/11/2026, when the two were conflated.
    #[test]
    fn a_family_window_holds_that_family_and_the_account_keeps_serving() {
        let (_clock, registry) = registry();
        let reset = ms_of("2026-11-14T08:00:00Z");
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![
                QuotaWindow {
                    kind: "seven_day".into(),
                    utilization: Some(0.72),
                    resets_at_ms: Some(reset),
                    status: QuotaStatus::Allowed,
                },
                QuotaWindow {
                    kind: "seven_day_fable".into(),
                    utilization: Some(1.0),
                    resets_at_ms: Some(reset),
                    status: QuotaStatus::Exhausted,
                },
            ]),
        );
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "a family limit is not the account's 100 %"
        );
        let held = registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .expect("Fable is out until its reset");
        assert_eq!(held.window.as_deref(), Some("model:fable"));
        assert_eq!(held.resets_at_ms, reset);
        assert_eq!(
            registry.model_exhausted("claude-code", "claude-opus-5-5"),
            None,
            "the other families carry no hold"
        );
    }

    /// The other half of the rule: when the window that GOVERNS the account is full, the instance
    /// is excluded as before. The add-on exception must never disarm a real refusal.
    #[test]
    fn a_full_serving_window_still_condemns_the_account() {
        let (_clock, registry) = registry();
        let reset = ms_of("2026-11-14T08:00:00Z");
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![
                QuotaWindow {
                    kind: "seven_day".into(),
                    utilization: Some(1.0),
                    resets_at_ms: Some(reset),
                    status: QuotaStatus::Exhausted,
                },
                QuotaWindow {
                    kind: "seven_day_overage_included".into(),
                    utilization: Some(1.0),
                    resets_at_ms: Some(reset),
                    status: QuotaStatus::Exhausted,
                },
            ]),
        );
        let held = registry.exhausted("claude-code").expect("excluded");
        assert_eq!(held.window.as_deref(), Some("seven_day"));
        assert_eq!(held.resets_at_ms, reset);
    }

    /// A model hold lives on the same clock as the account-wide one. A person's provider reset
    /// cannot reach a held model by itself: routing keeps the model out, so no turn can prove
    /// anything — that is how the Fable hold was dated 3.7 days out on 11/11/2026. Past the trust
    /// window the hold is re-observed ONCE, and a refusal that comes back re-arms it.
    #[test]
    fn a_model_hold_is_reobserved_once_stale() {
        let (clock, registry) = registry();
        let reset = ms_of("2026-11-14T08:00:00Z");
        registry.mark_model_rate_limited("claude-code", "Fable", Some(reset), None);
        assert!(registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .is_some());
        clock.advance(DATED_EXHAUSTION_STALENESS - Duration::from_secs(1));
        assert!(
            registry
                .model_exhausted("claude-code", "claude-fable-5-1")
                .is_some(),
            "trusted for the whole window"
        );
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            registry.model_exhausted("claude-code", "claude-fable-5-1"),
            None,
            "a hold nobody re-confirmed is re-observed, not kept for three days"
        );
        // The probe meets the same refusal: a fresh observation re-arms the hold.
        registry.mark_model_rate_limited("claude-code", "Fable", Some(reset), None);
        assert!(registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .is_some());
        clock.advance(DATED_EXHAUSTION_STALENESS - Duration::from_secs(1));
        assert!(registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .is_some());
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            registry.model_exhausted("claude-code", "claude-fable-5-1"),
            None
        );
    }

    /// The age of a hold is its OWN: a live snapshot about another window arrives all the time and
    /// must not make a dated refusal look fresh — otherwise the re-observation above never fires.
    #[test]
    fn an_unrelated_snapshot_does_not_make_a_hold_fresh() {
        let (clock, registry) = registry();
        let reset = ms_of("2026-11-14T08:00:00Z");
        registry.mark_model_rate_limited("claude-code", "Fable", Some(reset), None);
        clock.advance(DATED_EXHAUSTION_STALENESS / 2);
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![window(
                "five_hour",
                QuotaStatus::Allowed,
                Some(ms_of("2026-10-21T15:00:00Z")),
            )]),
        );
        clock.advance(DATED_EXHAUSTION_STALENESS / 2 + Duration::from_secs(1));
        assert_eq!(
            registry.model_exhausted("claude-code", "claude-fable-5-1"),
            None,
            "the hold's age is its own, not the entry's"
        );
    }

    /// A refusal that repeats the SAME date is a fresh observation: the hold's own clock must be
    /// re-armed, or a genuinely exhausted account is re-probed at every single turn for ever. Found
    /// by the independent review of this change — `mark_model` already did it, `mark` did not, and
    /// the entry's clock no longer decides once the hold carries its own.
    #[test]
    fn a_repeated_refusal_rearms_the_exhaustions_staleness_clock() {
        let (clock, registry) = registry();
        let reset = ms_of("2026-10-24T10:00:00Z");
        registry.mark_rate_limited("claude-code", Some(reset), None);
        clock.advance(DATED_EXHAUSTION_STALENESS + Duration::from_secs(1));
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "stale: the next turn is admitted as a probe"
        );
        // The interface obeys the same rule: no badge for a hold the router would not apply.
        assert_eq!(registry.view("claude-code").unwrap().exhausted, None);
        // The probe meets the same refusal, with the same date.
        registry.mark_rate_limited("claude-code", Some(reset), None);
        assert!(
            registry.exhausted("claude-code").is_some(),
            "re-armed by a refusal that repeats the date"
        );
        assert!(registry.view("claude-code").unwrap().exhausted.is_some());
        clock.advance(DATED_EXHAUSTION_STALENESS - Duration::from_secs(1));
        assert!(registry.exhausted("claude-code").is_some());
        clock.advance(Duration::from_secs(2));
        assert_eq!(registry.exhausted("claude-code"), None);
    }

    /// One family, one hold: the window suffix (`fable`) and the provider's own word (`Fable`) name
    /// the same family, so the map must not carry two — a fresh hold hiding behind a stale one under
    /// another key would make the choice depend on the hash order.
    #[test]
    fn one_family_one_key_whatever_its_case() {
        let holds = HashMap::from([(
            "Fable".to_owned(),
            Exhaustion {
                resets_at_ms: 1,
                reset_known: true,
                window: Some("model:Fable".into()),
                cause: ExhaustionCause::RateLimited,
                observed_at_ms: None,
            },
        )]);
        assert_eq!(
            family_key(&holds, "fable"),
            "Fable",
            "the existing key is reused"
        );
        assert_eq!(
            family_key(&holds, "opus"),
            "opus",
            "a new family keeps its name"
        );
    }

    /// A genuinely-exhausted account is not hammered: each re-observation probe that gets refused
    /// re-arms the staleness clock, so the next probe is a full window away.
    #[test]
    fn a_fresh_refusal_rearms_the_staleness_clock() {
        let (clock, registry) = registry();
        let reset = ms_of("2026-10-24T10:00:00Z");
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![window(
                "seven_day",
                QuotaStatus::Exhausted,
                Some(reset),
            )]),
        );
        // Go stale: the next turn is admitted as a probe.
        clock.advance(DATED_EXHAUSTION_STALENESS + Duration::from_secs(1));
        assert_eq!(registry.exhausted("claude-code"), None);
        // The probe meets the same refusal: a fresh observation re-arms the exclusion.
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![window(
                "seven_day",
                QuotaStatus::Exhausted,
                Some(reset),
            )]),
        );
        assert!(registry.exhausted("claude-code").is_some(), "re-armed");
        // The fresh observation is trusted for a full window: not stale again until then.
        clock.advance(DATED_EXHAUSTION_STALENESS - Duration::from_secs(1));
        assert!(registry.exhausted("claude-code").is_some());
        clock.advance(Duration::from_secs(2));
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

        // A 429 that names NEITHER a date nor a delay: the account refused, so it is held out on
        // the fallback cooldown like a 402 — and `reset_known` is false, so nothing invents an
        // hour. This is what docs/guides/providers.md promises; the code used to record nothing.
        let bare = clock.now().timestamp_millis() as u64;
        registry.mark_rate_limited("bare", None, None);
        let held = registry
            .exhausted("bare")
            .expect("a bare 429 still holds it out");
        assert!(!held.reset_known, "no date was given, none is invented");
        assert_eq!(held.cause, ExhaustionCause::RateLimited);
        assert_eq!(
            held.resets_at_ms,
            bare + FALLBACK_COOLDOWN.as_millis() as u64
        );
        clock.advance(FALLBACK_COOLDOWN + Duration::from_secs(1));
        assert!(
            registry.exhausted("bare").is_none(),
            "and it comes back by itself"
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

    /// A 402 (top-up) and an undated 429 hold the instance on a GUESSED five minutes. A turn that
    /// goes through proves the account answers: the guess must not outlive it, or a top-up or the
    /// provider's own window reset leaves the conversation blocked until the timer runs out.
    #[test]
    fn a_successful_turn_lifts_an_exhaustion_whose_reset_was_never_given() {
        let (_clock, registry) = registry();
        registry.mark_out_of_credit("claude-code");
        let held = registry
            .exhausted("claude-code")
            .expect("a 402 holds the instance out");
        assert!(!held.reset_known, "no date was given: a guessed cooldown");
        assert_eq!(held.cause, ExhaustionCause::OutOfCredit);

        registry.confirm_available("claude-code");
        assert!(
            registry.exhausted("claude-code").is_none(),
            "the account answered: the guessed cooldown is over"
        );

        // An instance nobody held stays free, and the call is idempotent.
        registry.confirm_available("other");
        assert!(registry.exhausted("other").is_none());
        registry.confirm_available("claude-code");
        assert!(registry.exhausted("claude-code").is_none());
    }

    /// The mirror rule: a reset the PROVIDER dated is never shortened by a success. A success
    /// before that date means another window opened, which only a live snapshot may say
    /// (`observe`, "the newest observation wins").
    #[test]
    fn a_successful_turn_never_shortens_a_dated_reset() {
        let (_clock, registry) = registry();
        let reset = ms_of("2026-10-21T15:00:00Z");
        registry.mark_rate_limited("claude-code", Some(reset), None);
        assert!(registry.exhausted("claude-code").is_some());

        registry.confirm_available("claude-code");
        let kept = registry
            .exhausted("claude-code")
            .expect("a dated reset survives a successful turn");
        assert!(kept.reset_known);
        assert_eq!(kept.resets_at_ms, reset);
    }

    /// A model-scoped hold is about ONE model: a success elsewhere says nothing about it.
    #[test]
    fn a_successful_turn_leaves_a_model_hold_alone() {
        let (_clock, registry) = registry();
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", None, None);
        registry.mark_out_of_credit("claude-code");

        registry.confirm_available("claude-code");
        assert!(
            registry.exhausted("claude-code").is_none(),
            "the account-wide guess is gone"
        );
        assert!(
            registry
                .model_exhausted("claude-code", "claude-fable-5-1")
                .is_some(),
            "the model the provider refused stays held"
        );
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

    /// The stored shape of 11/11/2026, exactly as it was found in the graph: an account-wide hold
    /// on `seven_day_overage_included` until 14/11, written by a build that did not know an add-on
    /// window is not a refusal. The account answered `claude-haiku` at 14:45:27 and 14:58:17 with
    /// that entry in place. A restart repairs the entry AND the store, instead of resurrecting a
    /// lock nobody can clear from the screen.
    #[tokio::test]
    async fn a_stored_add_on_exhaustion_is_repaired_on_load() {
        let store = Arc::new(MemStore::default());
        let until = ms_of("2026-11-14T08:00:00Z");
        store.0.lock().unwrap().insert(
            "claude-code".into(),
            QuotaEntry {
                windows: vec![QuotaWindow {
                    kind: "seven_day_overage_included".into(),
                    utilization: Some(1.01),
                    resets_at_ms: Some(until),
                    status: QuotaStatus::Exhausted,
                }],
                source: QuotaSource::Live,
                observed_at_ms: ms_of("2026-11-11T14:46:16Z"),
                set_by: None,
                exhaustion: Some(Exhaustion {
                    resets_at_ms: until,
                    reset_known: true,
                    window: Some("seven_day_overage_included".into()),
                    cause: ExhaustionCause::Window,
                    observed_at_ms: None,
                }),
                model_holds: HashMap::new(),
            },
        );
        let clock = Arc::new(ManualClock::at("2026-11-11T15:00:00Z"));
        let registry = QuotaRegistry::new()
            .with_clock(clock)
            .with_store(store.clone());
        assert_eq!(registry.load().await.unwrap(), 1);
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "the stored lock cannot outlive the rule that wrote it"
        );
        // The repair reached the store: a second boot finds nothing either.
        let stored = store.0.lock().unwrap().get("claude-code").cloned().unwrap();
        assert_eq!(stored.exhaustion, None);
        assert_eq!(stored.windows.len(), 1, "the gauge keeps its window");
    }

    /// A person's reset survives a restart: after `clear` + `persist`, a registry reloaded from
    /// the same store finds nothing, so the instance is a candidate again and no stale exclusion
    /// resurrects it.
    #[tokio::test]
    async fn clearing_a_quota_survives_a_restart() {
        let store = Arc::new(MemStore::default());
        let clock = Arc::new(ManualClock::at(T0));
        let before = QuotaRegistry::new()
            .with_clock(clock.clone())
            .with_store(store.clone());
        let until = ms_of("2026-11-14T08:00:00Z");
        before.set_manual(
            "claude-code",
            &ManualQuota {
                exhausted_until_ms: Some(until),
                utilization: Some(1.0),
                resets_at_ms: Some(until),
                window: Some("seven_day_overage_included".into()),
            },
            "me",
        );
        before.persist("claude-code").await.unwrap();
        // The person clears it: the entry is forgotten and the deletion is persisted.
        before.clear("claude-code");
        before.persist("claude-code").await.unwrap();

        let after = QuotaRegistry::new().with_clock(clock).with_store(store);
        assert_eq!(after.load().await.unwrap(), 0, "nothing reloaded");
        assert_eq!(
            after.exhausted("claude-code"),
            None,
            "the instance is a candidate again"
        );
        assert_eq!(after.view("claude-code"), None);
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
                observed_at_ms: None,
            }),
            // A per-model hold rides the SAME document and round-trips with it.
            model_holds: HashMap::from([(
                "claude-fable-5-1".to_owned(),
                Exhaustion {
                    resets_at_ms: 43,
                    reset_known: true,
                    window: Some("model:claude-fable-5-1".into()),
                    cause: ExhaustionCause::RateLimited,
                    observed_at_ms: None,
                },
            )]),
        };
        store.save("claude-code", &entry).await.unwrap();
        assert_eq!(
            store.load_all().await.unwrap(),
            vec![("claude-code".to_owned(), entry)]
        );
        store.delete("claude-code").await.unwrap();
        assert!(store.load_all().await.unwrap().is_empty());
    }

    // ── Fix 1: pressure ignores windows whose reset has passed ───────────────

    /// A window at 97 % whose reset is in the past no longer counts: the provider has reset
    /// its counter, and carrying the old utilisation would penalise an instance that is now free.
    #[test]
    fn pressure_drops_a_window_whose_reset_has_passed() {
        let (clock, registry) = registry();
        let five_hour_reset = ms_of("2026-10-21T15:00:00Z"); // 5 h after T0
        let seven_day_reset = ms_of("2026-10-28T10:00:00Z"); // 7 d after T0
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![
                QuotaWindow {
                    kind: "five_hour".into(),
                    utilization: Some(0.97),
                    resets_at_ms: Some(five_hour_reset),
                    status: QuotaStatus::Allowed,
                },
                QuotaWindow {
                    kind: "seven_day".into(),
                    utilization: Some(0.50),
                    resets_at_ms: Some(seven_day_reset),
                    status: QuotaStatus::Allowed,
                },
            ]),
        );
        // Before any reset: pressure is the max utilisation (0.97 from five_hour).
        assert_eq!(registry.pressure("claude-code"), Some(0.97));

        // Advance past the five_hour reset: only seven_day remains.
        clock.advance(Duration::from_secs(5 * 3600 + 1));
        assert_eq!(
            registry.pressure("claude-code"),
            Some(0.50),
            "five_hour expired, seven_day is the only window left"
        );

        // Advance past the seven_day reset too: all windows expired → None.
        clock.advance(Duration::from_secs(7 * 86_400));
        assert_eq!(
            registry.pressure("claude-code"),
            None,
            "all windows have expired"
        );
    }

    /// A window WITHOUT a reset time is always counted (the provider did not date it).
    #[test]
    fn pressure_keeps_a_window_without_a_reset_time() {
        let (_clock, registry) = registry();
        registry.observe(
            "deepseek",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "requests".into(),
                utilization: Some(0.80),
                resets_at_ms: None,
                status: QuotaStatus::Allowed,
            }]),
        );
        assert_eq!(
            registry.pressure("deepseek"),
            Some(0.80),
            "no reset → always counted"
        );
    }

    // ── Fix 2: persist_in_background ordering + dedup ────────────────────────

    /// Two observations in order leave the MOST RECENT state in the store.
    /// Then a delete prevents a late in-flight write from resurrecting the entry.
    #[tokio::test]
    async fn persist_ordering_keeps_the_newest_and_delete_blocks_resurrection() {
        let store = Arc::new(MemStore::default());
        let clock = Arc::new(ManualClock::at(T0));
        let registry = Arc::new(
            QuotaRegistry::new()
                .with_clock(clock.clone())
                .with_store(store.clone()),
        );

        // T=0: first observation.
        registry.observe(
            "a",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "five_hour".into(),
                utilization: Some(0.10),
                resets_at_ms: None,
                status: QuotaStatus::Allowed,
            }]),
        );
        // Let the background write land.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        let stored = store.load_all().await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].1.windows[0].utilization, Some(0.10));

        // T=1s: second observation (newer) overwrites.
        clock.advance(Duration::from_secs(1));
        registry.observe(
            "a",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "five_hour".into(),
                utilization: Some(0.90),
                resets_at_ms: None,
                status: QuotaStatus::Allowed,
            }]),
        );
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        let stored = store.load_all().await.unwrap();
        assert_eq!(
            stored[0].1.windows[0].utilization,
            Some(0.90),
            "the newer observation overwrote the older one"
        );

        // T=2s: delete via clear() + persist → the entry is gone.
        clock.advance(Duration::from_secs(1));
        registry.clear("a");
        // persist sends the deletion to the store.
        registry.persist("a").await.unwrap();
        assert!(
            store.load_all().await.unwrap().is_empty(),
            "the entry is deleted"
        );

        // Verify the watermark blocks a stale write: if we tried to re-persist at an old
        // timestamp the watermark would reject it. The watermark was advanced by clear().
    }

    // ── Fix 5: observe/set_manual reject an older observed_at ────────────────

    /// A live observation that arrives late (its `now_ms()` < the current entry's
    /// `observed_at_ms`) does not overwrite a newer manual entry.
    #[test]
    fn a_live_observation_older_than_the_manual_entry_is_rejected() {
        let clock = Arc::new(ManualClock::at("2026-10-21T12:00:00Z"));
        let registry = QuotaRegistry::new().with_clock(Arc::clone(&clock) as Arc<dyn Clock>);

        // T=12:00: set manual entry with exhaustion.
        let manual_at = ms_of("2026-10-21T12:00:00Z");
        registry.set_manual(
            "deepseek",
            &ManualQuota {
                exhausted_until_ms: Some(ms_of("2026-10-22T00:00:00Z")),
                utilization: Some(1.0),
                resets_at_ms: None,
                window: Some("monthly".into()),
            },
            "me",
        );
        assert_eq!(
            registry.view("deepseek").unwrap().source,
            QuotaSource::Manual
        );
        assert_eq!(registry.view("deepseek").unwrap().observed_at_ms, manual_at);

        // Simulate a late-arriving live event: the clock goes backwards.
        clock.set("2026-10-21T11:00:00Z");
        registry.observe(
            "deepseek",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "requests".into(),
                utilization: Some(0.10),
                resets_at_ms: None,
                status: QuotaStatus::Allowed,
            }]),
        );
        assert_eq!(
            registry.view("deepseek").unwrap().source,
            QuotaSource::Manual,
            "the manual entry is newer and must not be overwritten by an older live observation"
        );
        assert_eq!(registry.view("deepseek").unwrap().observed_at_ms, manual_at);
    }

    /// Symmetry: a manual entry older than a live observation is also rejected.
    #[test]
    fn a_manual_entry_older_than_the_live_observation_is_rejected() {
        let clock = Arc::new(ManualClock::at("2026-10-21T12:00:00Z"));
        let registry = QuotaRegistry::new().with_clock(Arc::clone(&clock) as Arc<dyn Clock>);

        // T=12:00: live observation.
        registry.observe(
            "deepseek",
            &QuotaSnapshot::live(vec![QuotaWindow {
                kind: "requests".into(),
                utilization: Some(0.50),
                resets_at_ms: None,
                status: QuotaStatus::Allowed,
            }]),
        );
        assert_eq!(registry.view("deepseek").unwrap().source, QuotaSource::Live);

        // Clock goes backwards: the manual entry is rejected.
        clock.set("2026-10-21T11:00:00Z");
        registry.set_manual(
            "deepseek",
            &ManualQuota {
                exhausted_until_ms: Some(ms_of("2026-10-22T00:00:00Z")),
                utilization: None,
                resets_at_ms: None,
                window: None,
            },
            "me",
        );
        assert_eq!(
            registry.view("deepseek").unwrap().source,
            QuotaSource::Live,
            "the live entry is newer and must not be overwritten by an older manual entry"
        );
    }

    // ── Per-model holds ─────────────────────────────────────────────────

    #[test]
    fn a_model_hold_is_scoped_to_that_model_and_never_reads_as_the_account() {
        let (_clock, registry) = registry();
        let until = ms_of("2026-10-22T10:00:00Z");
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", Some(until), None);

        let held = registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .expect("the model is held");
        assert_eq!(held.resets_at_ms, until);
        assert_eq!(held.window.as_deref(), Some("model:claude-fable-5-1"));
        assert!(held.reset_known);
        // The instance itself is NOT exhausted: its other models stay eligible.
        assert_eq!(registry.exhausted("claude-code"), None);
        assert_eq!(
            registry.model_exhausted("claude-code", "claude-sonnet-4-5"),
            None
        );
    }

    #[test]
    fn a_hold_named_by_family_takes_that_family_and_nothing_else() {
        // Measured 2026-11-04: Claude Code refuses with "You've reached your Fable
        // limit", a FAMILY name, while the host asks for `claude-fable-5-1`. A hold
        // keyed on the refusal's own words must still take that model out — and only
        // that family, never the catalogue.
        let (_clock, registry) = registry();
        let until = ms_of("2026-10-22T10:00:00Z");
        registry.mark_model_rate_limited("claude-code", "Fable", Some(until), None);

        assert!(
            registry
                .model_exhausted("claude-code", "claude-fable-5-1")
                .is_some(),
            "the family the provider named covers its models"
        );
        assert!(
            registry
                .model_exhausted("claude-code", "claude-sonnet-4-5")
                .is_none(),
            "another family is untouched"
        );
        assert_eq!(
            registry.exhausted("claude-code"),
            None,
            "the account is not out"
        );
    }

    #[test]
    fn a_short_held_name_never_matches_half_the_catalogue() {
        // A two- or three-letter name would match too much; the match is by family
        // only from four characters.
        let (_clock, registry) = registry();
        registry.mark_model_rate_limited("claude-code", "max", None, None);
        assert!(registry
            .model_exhausted("claude-code", "claude-sonnet-4-5-max")
            .is_none());
    }

    #[test]
    fn a_family_match_survives_case_and_never_fires_on_an_empty_name() {
        // Same name, other case: only the FAMILY path can see it (the exact lookup
        // is case-sensitive), so this is the `held == model` branch of `same_family`.
        let (_clock, registry) = registry();
        let until = ms_of("2026-10-22T10:00:00Z");
        registry.mark_model_rate_limited("claude-code", "Fable", Some(until), None);
        assert!(
            registry.model_exhausted("claude-code", "fable").is_some(),
            "the same family, written another way"
        );
        // An empty model name never matches a hold: the early return, not a panic.
        assert!(registry.model_exhausted("claude-code", "   ").is_none());
        // And an instance nobody held has nothing to match against.
        assert!(registry.model_exhausted("other", "fable").is_none());
    }

    #[test]
    fn a_refusal_never_shortens_a_model_reset_already_known() {
        // Measured rule, same as the instance-level one: a refusal dated EARLIER than
        // the reset we already know must not make the model eligible sooner.
        let (_clock, registry) = registry();
        let far = ms_of("2026-10-24T10:00:00Z");
        let near = ms_of("2026-10-22T10:00:00Z");
        registry.mark_model_rate_limited("claude-code", "Fable", Some(far), None);
        registry.mark_model_rate_limited("claude-code", "Fable", Some(near), None);
        let held = registry.model_exhausted("claude-code", "Fable").unwrap();
        assert_eq!(
            held.resets_at_ms, far,
            "the reset already known and still ahead is kept"
        );
    }

    #[test]
    fn a_model_hold_without_a_date_uses_the_fallback_and_expires_by_itself() {
        let (clock, registry) = registry();
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", None, None);
        let held = registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .unwrap();
        assert!(!held.reset_known);
        assert_eq!(held.cause, ExhaustionCause::RateLimited);
        clock.advance(FALLBACK_COOLDOWN - Duration::from_secs(1));
        assert!(registry
            .model_exhausted("claude-code", "claude-fable-5-1")
            .is_some());
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            registry.model_exhausted("claude-code", "claude-fable-5-1"),
            None
        );
    }

    #[test]
    fn a_retry_after_delay_dates_the_model_hold_from_the_clock() {
        let (_clock, registry) = registry();
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", None, Some(90_000));
        assert_eq!(
            registry
                .model_exhausted("claude-code", "claude-fable-5-1")
                .unwrap()
                .resets_at_ms,
            registry.now_ms() + 90_000
        );
    }

    #[test]
    fn a_model_refusal_never_shortens_a_model_reset_already_known() {
        let (clock, registry) = registry();
        let until = ms_of("2026-10-24T10:00:00Z");
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", Some(until), None);
        clock.advance(Duration::from_secs(60));
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", None, Some(1000));
        assert_eq!(
            registry
                .model_exhausted("claude-code", "claude-fable-5-1")
                .unwrap()
                .resets_at_ms,
            until
        );
    }

    #[test]
    fn a_live_snapshot_does_not_wipe_a_model_hold() {
        let (clock, registry) = registry();
        let until = ms_of("2026-10-22T10:00:00Z");
        registry.mark_model_rate_limited("claude-code", "claude-fable-5-1", Some(until), None);
        // A later account snapshot replaces windows/exhaustion but keeps the model hold.
        clock.advance(Duration::from_secs(1));
        registry.observe(
            "claude-code",
            &QuotaSnapshot::live(vec![window("requests", QuotaStatus::Allowed, None)]),
        );
        assert_eq!(registry.exhausted("claude-code"), None);
        assert_eq!(
            registry
                .model_exhausted("claude-code", "claude-fable-5-1")
                .unwrap()
                .resets_at_ms,
            until
        );
    }
}
