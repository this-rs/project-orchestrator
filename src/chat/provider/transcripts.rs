//! Where the conversations of the native sessions are kept on disk (P14).
//!
//! A native session (nexus `NativeProvider`) IS its transcript: the list of
//! messages it replays to the endpoint. Its resume token, persisted on the
//! `ChatSession` node, only names it (`{"transcript_id": "<id>"}`). nexus keeps
//! transcripts in memory by default, so after a restart the token named nothing
//! and every resume failed ("unknown transcript: nothing to resume"). The host
//! gives each stored native instance a nexus `FileTranscriptStore`:
//!
//! ```text
//! <app dir>/native-transcripts/            0700
//!     <instance id>/                       0700 (nexus)
//!         <transcript id>.json             0600 (nexus), credential-shaped text masked
//! ```
//!
//! Lifetime: a transcript lives as long as its session. A CLOSED session stays
//! resumable, so its transcript stays; it goes when the session is deleted
//! ([`remove_for_token`], called by `ChatManager::delete_session`).
//!
//! What is written is the conversation itself, which the graph already holds as
//! chat events: no new class of data, but owner-only files, and nexus masks the
//! credential-shaped fragments before writing.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nexus_claude::agent::{ProviderKind, ResumeToken};
use nexus_claude::providers::native::FileTranscriptStore;

const DIR_NAME: &str = "native-transcripts";

/// The default root, under the application directory where the vault, the
/// identity key and the neutral directories already live.
pub fn default_root() -> PathBuf {
    let app = match dirs::config_dir() {
        #[cfg(target_os = "windows")]
        Some(c) => c.join("ProjectOrchestrator"),
        #[cfg(not(target_os = "windows"))]
        Some(c) => c.join("project-orchestrator"),
        None => dirs::home_dir()
            .map(|h| h.join(".project-orchestrator"))
            .unwrap_or_else(|| PathBuf::from(".project-orchestrator")),
    };
    app.join(DIR_NAME)
}

/// A name that is safe as ONE path component: letters, digits, `-`, `_`.
fn safe_component(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

/// The directory of an instance's transcripts; `None` for an id that is not a
/// plain path component (a stored id always is: `settings::valid_id`).
pub fn dir_for(root: &Path, instance_id: &str) -> Option<PathBuf> {
    safe_component(instance_id).then(|| root.join(instance_id))
}

/// The store of an instance. The root is made owner-only here (it lists the
/// instances); nexus makes the instance directory `0700` and each file `0600`.
pub fn store_for(root: &Path, instance_id: &str) -> std::io::Result<Arc<FileTranscriptStore>> {
    let dir = dir_for(root, instance_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "instance id is not a path component",
        )
    })?;
    create_private_dir(root)?;
    Ok(Arc::new(FileTranscriptStore::new(dir)))
}

fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// The transcript a persisted resume token names, when it is a native one.
pub fn transcript_id_of(wire: &str) -> Option<String> {
    let token = ResumeToken::from_wire(wire).ok()?;
    if token.kind() != ProviderKind::Native {
        return None;
    }
    token
        .data()
        .get("transcript_id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| safe_component(id))
        .map(str::to_string)
}

/// Removes the transcript a session's token names (and a leftover temporary
/// file of an interrupted write). `Ok(false)`: nothing to remove — not a native
/// token, or no file. nexus' `TranscriptStore` has no deletion: the host owns
/// the layout it chose (`<dir>/<id>.json`, `.<id>.tmp`).
pub fn remove_for_token(root: &Path, instance_id: &str, wire: &str) -> std::io::Result<bool> {
    let (Some(dir), Some(id)) = (dir_for(root, instance_id), transcript_id_of(wire)) else {
        return Ok(false);
    };
    let _ = std::fs::remove_file(dir.join(format!(".{id}.tmp")));
    match std::fs::remove_file(dir.join(format!("{id}.json"))) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_claude::model::ChatMessage;
    use nexus_claude::providers::native::TranscriptStore;

    fn native_token(id: &str) -> String {
        ResumeToken::new(
            ProviderKind::Native,
            1,
            serde_json::json!({ "transcript_id": id }),
        )
        .to_wire()
    }

    #[test]
    fn an_instance_directory_is_one_plain_component() {
        let root = Path::new("/r");
        assert_eq!(dir_for(root, "local-1"), Some(root.join("local-1")));
        for bad in ["", "..", "a/b", "a\\b", ".hidden", "é"] {
            assert_eq!(dir_for(root, bad), None, "{bad:?}");
        }
    }

    #[test]
    fn only_a_native_token_names_a_transcript() {
        assert_eq!(
            transcript_id_of(&native_token("abc123")),
            Some("abc123".into())
        );
        let cli = ResumeToken::claude_code_session("abc123").to_wire();
        assert_eq!(transcript_id_of(&cli), None);
        assert_eq!(transcript_id_of(&native_token("../x")), None);
        assert_eq!(transcript_id_of("not json"), None);
    }

    #[cfg(unix)]
    #[test]
    fn the_store_writes_owner_only_files_and_removal_deletes_them() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("native-transcripts");
        let store = store_for(&root, "local").unwrap();
        store.save("t1", &[ChatMessage::user("hello")]).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&root), 0o700);
        assert_eq!(mode(&root.join("local")), 0o700);
        let file = root.join("local").join("t1.json");
        assert_eq!(mode(&file), 0o600);

        assert!(remove_for_token(&root, "local", &native_token("t1")).unwrap());
        assert!(!file.exists());
        assert!(!remove_for_token(&root, "local", &native_token("t1")).unwrap());
        assert!(!remove_for_token(&root, "local", "garbage").unwrap());
    }
}
