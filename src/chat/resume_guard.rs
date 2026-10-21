//! PYSR-2: a resume must not lose the tools the conversation already used.
//!
//! The fingerprint recorded at each opening holds the NAMES of the MCP servers the
//! session ran with (never their env or headers: they carry credentials). A plain
//! resume rebuilds the spec and compares: a server that was there and is gone is
//! refused, naming the field; a server that is new is accepted (a backend upgrade
//! that adds one must not strand every session). The model and the prompt are not
//! compared here: the model is rewritten by every legitimate switch (Auto changes it
//! from one turn to the next) and the prompt embeds the date and the project's live
//! instructions, so a difference there is the normal state of a resumed session.

use nexus_claude::agent::{ResumeFingerprint, ResumeIncompatible};
use tracing::warn;

/// The fingerprint of a session opened with `mcp_servers`, as JSON.
pub(crate) fn fingerprint_json(mcp_servers: &[String]) -> String {
    let fingerprint = ResumeFingerprint::new(None, mcp_servers.iter().cloned(), None);
    serde_json::to_string(&fingerprint).unwrap_or_default()
}

/// Refuses a resume that would lose an MCP server the session was opened with.
/// `stored`: the recorded fingerprint (`None` = a session opened before PYSR-2,
/// never refused); a fingerprint that does not parse is treated as absent, so a bad
/// record cannot strand a session.
pub(crate) fn check_servers_kept(
    stored: Option<&str>,
    now: &[String],
) -> Result<(), ResumeIncompatible> {
    let Some(raw) = stored.filter(|raw| !raw.is_empty()) else {
        return Ok(());
    };
    let Ok(stored) = serde_json::from_str::<ResumeFingerprint>(raw) else {
        warn!("an unreadable resume fingerprint is ignored");
        return Ok(());
    };
    if stored.mcp_servers.iter().all(|name| now.contains(name)) {
        Ok(())
    } else {
        Err(ResumeIncompatible {
            field: "mcp_servers",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_resume_that_loses_a_server_is_refused_and_names_the_field() {
        let stored = fingerprint_json(&names(&["project-orchestrator", "nexus"]));
        let error = check_servers_kept(Some(&stored), &names(&["nexus"])).unwrap_err();
        assert_eq!(error.field, "mcp_servers");
        assert!(error.to_string().contains("mcp_servers"));
    }

    #[test]
    fn the_same_servers_are_accepted_in_any_order() {
        let stored = fingerprint_json(&names(&["b", "a"]));
        assert!(check_servers_kept(Some(&stored), &names(&["a", "b"])).is_ok());
    }

    #[test]
    fn a_new_server_does_not_strand_the_session() {
        let stored = fingerprint_json(&names(&["a"]));
        assert!(check_servers_kept(Some(&stored), &names(&["a", "added-by-upgrade"])).is_ok());
    }

    #[test]
    fn a_session_without_fingerprint_is_never_refused() {
        assert!(check_servers_kept(None, &names(&[])).is_ok());
        assert!(check_servers_kept(Some(""), &names(&[])).is_ok());
    }

    #[test]
    fn an_unreadable_fingerprint_is_ignored() {
        assert!(check_servers_kept(Some("{not json"), &names(&[])).is_ok());
    }

    #[test]
    fn the_recorded_fingerprint_carries_names_only() {
        let json = fingerprint_json(&names(&["nexus", "a"]));
        assert_eq!(json, r#"{"mcp_servers":["a","nexus"]}"#);
    }
}
