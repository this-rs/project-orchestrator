//! Live `agent_session` tokens.
//!
//! A chat session's MCP subprocess authenticates with a JWT that sits in the
//! agent's environment — and, for a third-party CLI, in that CLI's config file.
//! A signature and an expiry are not enough for such a token: it must stop
//! working when the session it was minted for is closed.
//!
//! This is an ALLOW list, not a revocation list: a `jti` is valid only while it
//! is registered here. The registry lives in memory, so a server restart
//! invalidates every agent token — which is correct, since a restart also kills
//! every agent subprocess that held one (they are children of the server and
//! each resume mints a fresh token).

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, RwLock};

/// `jti` → session id (`None` for a token minted without a session).
static LIVE: LazyLock<RwLock<HashMap<String, Option<String>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Sessions taken out of `trust` after their token was minted (H6). A third
/// party's `full` profile is what trust grants it: the token, minted once, keeps
/// saying `full`, so the REST boundary reads this set to apply the restricted
/// profile while the session is out of trust.
static OUT_OF_TRUST: LazyLock<RwLock<HashSet<String>>> =
    LazyLock::new(|| RwLock::new(HashSet::new()));

/// Register a freshly minted token. Minting a token for a session supersedes
/// the tokens that session held before: a new token is minted only when the
/// agent process is (re)spawned, so the previous holder is gone. The new token
/// is minted for the session's current mode: it is no longer out of trust.
pub fn register(jti: &str, session_id: Option<&str>) {
    let mut live = LIVE.write().unwrap_or_else(|e| e.into_inner());
    if let Some(sid) = session_id {
        live.retain(|_, s| s.as_deref() != Some(sid));
        set_out_of_trust(sid, false);
    }
    live.insert(jti.to_string(), session_id.map(str::to_string));
}

/// Record that a live session left `trust` (`true`) or came back to it.
pub fn set_out_of_trust(session_id: &str, out: bool) {
    let mut set = OUT_OF_TRUST.write().unwrap_or_else(|e| e.into_inner());
    if out {
        set.insert(session_id.to_string());
    } else {
        set.remove(session_id);
    }
}

/// Whether the session left `trust` after its token was minted.
pub fn is_out_of_trust(session_id: &str) -> bool {
    OUT_OF_TRUST
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(session_id)
}

/// Whether a token is still usable.
pub fn is_live(jti: &str) -> bool {
    LIVE.read()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(jti)
}

/// Sessions whose `nexus-tools` profile token (B40) is live. Kept apart from [`LIVE`]: that map
/// is keyed by `jti` and a new entry there supersedes the session's other tokens, which would
/// revoke the project-orchestrator token of the very same session.
static TOOLS_LIVE: LazyLock<RwLock<std::collections::HashSet<String>>> =
    LazyLock::new(|| RwLock::new(std::collections::HashSet::new()));

/// Registers the `nexus-tools` profile token minted for a session. A new one (the session is
/// resumed, the process restarted) replaces the previous: the key is per process.
pub fn register_tools(session_id: &str) {
    TOOLS_LIVE
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(session_id.to_string());
}

/// Whether the session's `nexus-tools` token is still usable.
pub fn tools_live(session_id: &str) -> bool {
    TOOLS_LIVE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(session_id)
}

/// Revoke every token minted for a session: the `agent_session` JWTs and the `nexus-tools`
/// profile token. Returns how many were revoked.
pub fn revoke_session(session_id: &str) -> usize {
    set_out_of_trust(session_id, false);
    let tools = usize::from(
        TOOLS_LIVE
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(session_id),
    );
    let mut live = LIVE.write().unwrap_or_else(|e| e.into_inner());
    let before = live.len();
    live.retain(|_, s| s.as_deref() != Some(session_id));
    before - live.len() + tools
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_registered_token_is_live_until_its_session_is_revoked() {
        register("jti-a", Some("sess-a"));
        register("jti-b", Some("sess-b"));
        assert!(is_live("jti-a"));
        assert_eq!(revoke_session("sess-a"), 1);
        assert!(!is_live("jti-a"));
        assert!(is_live("jti-b"), "another session's token is untouched");
        revoke_session("sess-b");
    }

    #[test]
    fn an_unknown_token_is_not_live() {
        assert!(!is_live("never-registered"));
    }

    #[test]
    fn a_new_token_for_a_session_supersedes_the_previous_one() {
        register("jti-old", Some("sess-respawn"));
        register("jti-new", Some("sess-respawn"));
        assert!(!is_live("jti-old"));
        assert!(is_live("jti-new"));
        revoke_session("sess-respawn");
    }

    #[test]
    fn a_token_without_a_session_is_not_swept_by_a_session_revocation() {
        register("jti-free", None);
        revoke_session("anything");
        assert!(is_live("jti-free"));
    }
}
