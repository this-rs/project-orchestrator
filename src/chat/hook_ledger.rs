//! Per-session memory of what the PreToolUse hook has already injected.
//!
//! ## Why
//!
//! `additionalContext` is appended to the conversation at **every** tool call,
//! and it stays there: the agent does not forget a block because the next tool
//! call did not repeat it. Re-injecting the same skill, the same persona and the
//! same notes on each of 30 consecutive tool calls therefore adds nothing but
//! tokens — measured on a real session, the same ~2 KB persona block came back
//! more than ten times in a row.
//!
//! The ledger makes each piece of knowledge enter the context **once**, and
//! come back only when it may genuinely have left it:
//!
//! - after [`REINJECT_AFTER_CALLS`] tool calls, a coarse proxy for "far enough
//!   back that the agent's attention has moved on";
//! - after a compaction ([`HookLedger::reset`]), which does remove it — the
//!   summary keeps the gist of the conversation, not injected notes verbatim.
//!
//! One ledger per hook instance, and the hook is instantiated per CLI session
//! (`create_session` / `resume_session`), so the ledger's scope is the session
//! without needing a session id.
//!
//! ## Whatever the provider (B14)
//!
//! The ledger knows nothing of the provider. A provider that runs the hooks in its
//! loop resets it from its `PreCompact` callback ([`HookLedgerReset`]); a provider
//! that runs none has it reset by the host when the session's compaction signal is
//! seen (`provider::hooks::HookFallback`). Such a provider also loses the per-tool
//! injection itself — the skill context then arrives once per turn, by the
//! enrichment — and that loss is written here ([`HookLedger::record_lost`]), per
//! hook point and tool, never silent. A reset does not forget it: it says what the
//! session MISSED, not what it has in context.

use std::collections::HashMap;
use std::sync::Mutex;
use uuid::Uuid;

/// Tool calls after which an already-injected item is considered stale enough
/// to be injected again.
pub(crate) const REINJECT_AFTER_CALLS: u64 = 30;

/// Hard ceiling on the context a single hook call may add, in characters.
///
/// The per-part budgets (skill 3200, persona 2000) were each reasonable but
/// summed without a cap, so one call could add over 5 KB. With de-duplication
/// most calls add nothing; this bounds the ones that do.
pub(crate) const MAX_HOOK_CONTEXT_CHARS: usize = 2500;

/// What was injected. Distinct variants so a note shown by a persona is also
/// recognised as seen when a skill would show it again.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum InjectionKey {
    Skill(Uuid),
    Persona(Uuid),
    Note(Uuid),
    /// A redirect suggestion: the advice "use find_references instead of grep"
    /// does not improve by repetition. Keyed by the MCP tool it recommends, or
    /// (`PostToolUseRedirectHook::with_ledger`) by the tool and the file advised on.
    Redirect(String),
}

/// Where a hook would have run, for an injection the provider could not take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum HookPoint {
    /// Before a tool (`PreToolUse`: skill activation). The advice AFTER a tool is
    /// not lost on a provider without hooks, only deferred to the end of the turn.
    BeforeTool,
    /// Before a compaction (`PreCompact`: the compaction guidance the provider could
    /// not be given). Its "tool" is the compaction's trigger.
    BeforeCompaction,
}

#[derive(Default)]
struct LedgerState {
    /// Monotonic count of hook calls in this session.
    call: u64,
    /// Call number at which each key was last injected.
    last_injected: HashMap<InjectionKey, u64>,
    /// Injections the provider could not take, per hook point and tool: how many.
    /// Bounded by the number of distinct tools of the session.
    lost: HashMap<(HookPoint, String), u64>,
}

#[derive(Default)]
pub(crate) struct HookLedger {
    // std Mutex: every critical section is a few HashMap operations with no
    // await inside, so an async lock would only add overhead.
    state: Mutex<LedgerState>,
}

impl HookLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark the start of a hook call. Must be called once per PreToolUse event,
    /// before any [`is_fresh`](Self::is_fresh) check for that event.
    pub fn begin_call(&self) {
        let mut s = self.lock();
        s.call += 1;
    }

    /// Would injecting `key` now tell the agent something it does not already
    /// have in context?
    pub fn is_fresh(&self, key: &InjectionKey) -> bool {
        let s = self.lock();
        match s.last_injected.get(key) {
            None => true,
            Some(&at) => s.call.saturating_sub(at) >= REINJECT_AFTER_CALLS,
        }
    }

    /// Record that `key` was injected during the current call.
    pub fn record(&self, key: InjectionKey) {
        let mut s = self.lock();
        let call = s.call;
        s.last_injected.insert(key, call);
    }

    /// Forget everything — the context was compacted, so nothing injected
    /// earlier can be assumed to still be visible to the agent. What was lost
    /// ([`Self::record_lost`]) is kept: it is not context.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.last_injected.clear();
    }

    /// Record that the hook of `point` could not inject anything for a call of
    /// `tool`: the provider runs no hook at that point. Returns how many
    /// injections of the session were lost so far, all points and tools together.
    pub fn record_lost(&self, point: HookPoint, tool: &str) -> u64 {
        let mut s = self.lock();
        *s.lost.entry((point, tool.to_string())).or_default() += 1;
        s.lost.values().sum()
    }

    /// The injections lost so far: `(point, tool, count)`, in a stable order.
    pub fn lost(&self) -> Vec<(HookPoint, String, u64)> {
        let s = self.lock();
        let mut lost: Vec<_> = s
            .lost
            .iter()
            .map(|((point, tool), n)| (*point, tool.clone(), *n))
            .collect();
        lost.sort();
        lost
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LedgerState> {
        // A poisoned lock only means a panic happened mid-update of a cache.
        // The worst outcome of using it anyway is one redundant injection,
        // which is better than disabling the hook for the rest of the session.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Truncate `text` to at most `max` characters, on a char boundary, marking
/// the cut. Char-based, not byte-based: the context is full of accents and
/// emoji, and a byte cut would panic.
pub(crate) fn cap_chars(text: &mut String, max: usize) {
    if text.chars().count() <= max {
        return;
    }
    let keep = max.saturating_sub(2);
    *text = text.chars().take(keep).collect();
    text.push_str("…\n");
}

/// Hook registered on PreCompact that clears the ledger it shares with the
/// PreToolUse hook. Kept separate from `CompactionNotifier` so the ledger does
/// not leak into the compaction pipeline's constructor.
pub(crate) struct HookLedgerReset {
    ledger: std::sync::Arc<HookLedger>,
}

impl HookLedgerReset {
    pub fn new(ledger: std::sync::Arc<HookLedger>) -> Self {
        Self { ledger }
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for HookLedgerReset {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        if matches!(input, nexus_claude::HookInput::PreCompact(_)) {
            self.ledger.reset();
        }
        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                ..Default::default()
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> InjectionKey {
        InjectionKey::Persona(Uuid::from_u128(1))
    }

    #[test]
    fn new_key_is_fresh() {
        let l = HookLedger::new();
        l.begin_call();
        assert!(l.is_fresh(&key()));
    }

    #[test]
    fn injected_key_is_not_fresh_on_following_calls() {
        let l = HookLedger::new();
        l.begin_call();
        l.record(key());
        for _ in 1..REINJECT_AFTER_CALLS {
            l.begin_call();
            assert!(!l.is_fresh(&key()), "re-injected before the window elapsed");
        }
    }

    #[test]
    fn key_becomes_fresh_again_after_the_window() {
        let l = HookLedger::new();
        l.begin_call();
        l.record(key());
        for _ in 0..REINJECT_AFTER_CALLS {
            l.begin_call();
        }
        assert!(l.is_fresh(&key()));
    }

    #[test]
    fn reset_makes_everything_fresh() {
        // Compaction removed the injected blocks from context.
        let l = HookLedger::new();
        l.begin_call();
        l.record(key());
        l.begin_call();
        l.reset();
        assert!(l.is_fresh(&key()));
    }

    #[test]
    fn keys_of_different_kinds_do_not_collide() {
        let id = Uuid::from_u128(7);
        let l = HookLedger::new();
        l.begin_call();
        l.record(InjectionKey::Persona(id));
        assert!(l.is_fresh(&InjectionKey::Note(id)));
        assert!(l.is_fresh(&InjectionKey::Skill(id)));
    }

    #[test]
    fn redirect_keyed_by_tool() {
        let l = HookLedger::new();
        l.begin_call();
        l.record(InjectionKey::Redirect("find_references".into()));
        assert!(!l.is_fresh(&InjectionKey::Redirect("find_references".into())));
        assert!(l.is_fresh(&InjectionKey::Redirect("search_project".into())));
    }

    #[test]
    fn a_lost_injection_is_counted_per_point_and_tool_and_survives_a_reset() {
        let l = HookLedger::new();
        assert!(l.lost().is_empty());
        assert_eq!(l.record_lost(HookPoint::BeforeTool, "Read"), 1);
        assert_eq!(l.record_lost(HookPoint::BeforeTool, "Read"), 2);
        assert_eq!(l.record_lost(HookPoint::BeforeTool, "Grep"), 3);
        l.reset();
        assert_eq!(
            l.lost(),
            vec![
                (HookPoint::BeforeTool, "Grep".to_string(), 1),
                (HookPoint::BeforeTool, "Read".to_string(), 2),
            ]
        );
    }

    #[test]
    fn cap_chars_is_char_safe_and_bounded() {
        let mut s = "é🧠".repeat(2000);
        cap_chars(&mut s, 100);
        assert!(s.chars().count() <= 100);
        assert!(s.ends_with("…\n"));

        let mut short = "ok".to_string();
        cap_chars(&mut short, 100);
        assert_eq!(short, "ok");
    }
}
