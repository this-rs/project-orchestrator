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
//!         <transcript id>.json             0600 (nexus), the conversation byte for byte
//! ```
//!
//! Lifetime: a transcript lives as long as its session. A CLOSED session stays
//! resumable, so its transcript stays; it goes when the session is deleted
//! ([`remove_for_token`], called by `ChatManager::delete_session`).
//!
//! What is written is the conversation itself, which the graph already holds as
//! chat events: no new class of data, in owner-only files. It is written BYTE FOR
//! BYTE (decision CTX6): a resume replays the request the live session sent, so the
//! endpoint's prefix cache still serves it and every tool-call argument is the JSON
//! the model wrote. Masking by shape (a field named `max_tokens`, a long word)
//! corrupted that history and is gone. The only masking is by VALUE: the keys the
//! instance's resolver handed out ([`store_for`]'s `known`, fed by
//! `KnownSecrets::recording` in `native_factory`) are replaced in contents and
//! reasoning, never inside a tool-call argument. A native resume reloads THIS
//! file (the token names it); it is never rebuilt from the graph's chat events.
//!
//! SECURITY, what the masking does NOT cover: a known key that the model itself wrote
//! into a tool-call ARGUMENT (`curl -H "Authorization: Bearer <key>"`) stays in clear in
//! the file — masking it would change the argument the endpoint's cache and the replay
//! depend on. The file is owner-only (`0600`), like the vault's; it is still the
//! conversation, and whoever reads the server's application directory reads it. The
//! prompt tells agents to use secrets only through `orchestrator secret exec`, which keeps
//! the value out of every argument.
//!
//! Where: the server's application directory (beside the vault), NEVER a project
//! directory or a git work tree: the files hold the conversation in clear, and a
//! `git add -A` there would publish it. [`store_for`] refuses a root inside a git work
//! tree ([`inside_git_worktree`]); `native_factory` then keeps the transcripts in
//! memory (the session does not resume after a restart), says so in a warning, and the
//! session itself declares it: `resume_after_restart` among its `degraded_features`
//! (`ChatManager::transcripts_kept_in_memory`), so a person sees it where they work. A
//! versioned HOME (a dotfiles repository with its `.git` in `~`) is the common case: the
//! walk is NOT bounded at HOME on purpose — such a repository may well track the
//! application directory, and a `git add -A` in `~` would then publish every conversation.
//!
//! On Windows the `0600`/`0700` modes do not exist: the protection is the ACL of
//! the user profile. The root is under `%LOCALAPPDATA%` (`data_local_dir`), which
//! does not roam: the earlier `%APPDATA%` location roamed with the profile, and a
//! roaming profile synced to a share carried the conversations with it.
//! [`default_root`] moves the transcripts of that earlier location
//! ([`migrate_legacy_root`]) on first use. A profile whose ACLs were widened still
//! exposes the transcripts.
use crate::neo4j::traits::*;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nexus_claude::agent::{KnownSecrets, ProviderKind, ResumeToken};
use nexus_claude::providers::native::FileTranscriptStore;

const DIR_NAME: &str = "native-transcripts";

/// The default root, under the application directory where the vault, the
/// identity key and the neutral directories already live.
pub fn default_root() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        // %LOCALAPPDATA%, not %APPDATA%: the latter roams with the profile and
        // would carry the conversations to a synced share.
        if let Some(local) = dirs::data_local_dir() {
            let current = local.join("ProjectOrchestrator").join(DIR_NAME);
            if let Some(roaming) = dirs::config_dir() {
                let legacy = roaming.join("ProjectOrchestrator").join(DIR_NAME);
                match migrate_legacy_root(&legacy, &current) {
                    Ok(0) => {}
                    Ok(n) => {
                        tracing::info!(moved = n, "native transcripts moved to the local profile")
                    }
                    Err(e) => tracing::warn!(
                        error = %e,
                        "native transcripts not migrated; they stay in the roaming profile"
                    ),
                }
            }
            return current;
        }
    }
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

/// Whether `path` (absolute, or taken from the current directory) lies in a git
/// work tree: one of its ancestors, or itself, holds a `.git` entry (a directory, or
/// the file of a linked worktree or a submodule).
pub fn inside_git_worktree(path: &Path) -> bool {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(path),
            // Unknown place: fail closed.
            Err(_) => return true,
        }
    };
    let has_git = |path: &Path| {
        path.ancestors()
            .any(|dir| std::fs::symlink_metadata(dir.join(".git")).is_ok())
    };
    // Through a symbolic link too: the deepest existing ancestor, resolved.
    let resolved = absolute
        .ancestors()
        .find_map(|dir| std::fs::canonicalize(dir).ok());
    has_git(&absolute) || resolved.is_some_and(|real| has_git(&real))
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
/// `known`: the secret values to mask (shared with the instance's resolver).
pub fn store_for(
    root: &Path,
    instance_id: &str,
    known: KnownSecrets,
) -> std::io::Result<Arc<FileTranscriptStore>> {
    if inside_git_worktree(root) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the transcript root is inside a git work tree",
        ));
    }
    let dir = dir_for(root, instance_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "instance id is not a path component",
        )
    })?;
    create_private_dir(root)?;
    Ok(Arc::new(
        FileTranscriptStore::new(dir).with_known_secrets(known),
    ))
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

/// Moves the transcripts of a previous root (`legacy`, on Windows the roaming
/// `%APPDATA%` copy) into `current`, one instance directory at a time. An instance
/// directory is renamed only when `current` has no directory of that name: an
/// existing one is never overwritten, and its legacy copy stays where it is.
/// Returns the number of instance directories moved. Idempotent: a second run moves
/// nothing. The legacy root is removed once it is empty.
pub fn migrate_legacy_root(legacy: &Path, current: &Path) -> std::io::Result<usize> {
    let entries = match std::fs::read_dir(legacy) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    std::fs::create_dir_all(current)?;
    let mut moved = 0;
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let target = current.join(entry.file_name());
        // Never overwrite: an instance already present under `current` wins, and
        // its legacy copy stays in place for a person to look at.
        if std::fs::symlink_metadata(&target).is_ok() {
            continue;
        }
        // One rename per instance directory: atomic on the same volume. Across
        // volumes it fails and the error reaches the caller; nothing is half-moved.
        std::fs::rename(entry.path(), &target)?;
        moved += 1;
    }
    // Removes the legacy root only when every instance has left it.
    let _ = std::fs::remove_dir(legacy);
    Ok(moved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;
    use nexus_claude::model::ChatMessage;
    use nexus_claude::providers::native::TranscriptStore;

    /// Writes `<root>/<instance>/<file>` with `body`, for the migration tests.
    fn put(root: &Path, instance: &str, file: &str, body: &str) {
        let dir = root.join(instance);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), body).unwrap();
    }

    #[test]
    fn migration_moves_each_instance_directory_once() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp
            .path()
            .join("roaming/ProjectOrchestrator/native-transcripts");
        let current = tmp
            .path()
            .join("local/ProjectOrchestrator/native-transcripts");
        put(&legacy, "a", "t1.json", "one");
        put(&legacy, "b", "t2.json", "two");

        assert_eq!(migrate_legacy_root(&legacy, &current).unwrap(), 2);
        assert_eq!(
            std::fs::read_to_string(current.join("a/t1.json")).unwrap(),
            "one"
        );
        assert_eq!(
            std::fs::read_to_string(current.join("b/t2.json")).unwrap(),
            "two"
        );
        assert!(!legacy.join("a").exists(), "moved, not copied");
        assert!(!legacy.exists(), "an emptied legacy root is removed");
    }

    #[test]
    fn migration_never_overwrites_an_existing_instance() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join("roaming");
        let current = tmp.path().join("local");
        put(&legacy, "a", "t1.json", "old");
        put(&current, "a", "t1.json", "new");

        assert_eq!(migrate_legacy_root(&legacy, &current).unwrap(), 0);
        assert_eq!(
            std::fs::read_to_string(current.join("a/t1.json")).unwrap(),
            "new"
        );
        assert_eq!(
            std::fs::read_to_string(legacy.join("a/t1.json")).unwrap(),
            "old"
        );
    }

    #[test]
    fn migration_is_idempotent_and_a_missing_legacy_root_is_a_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join("roaming");
        let current = tmp.path().join("local");
        assert_eq!(migrate_legacy_root(&legacy, &current).unwrap(), 0);
        assert!(
            !current.exists(),
            "nothing created when there is nothing to move"
        );

        put(&legacy, "a", "t1.json", "one");
        assert_eq!(migrate_legacy_root(&legacy, &current).unwrap(), 1);
        assert_eq!(migrate_legacy_root(&legacy, &current).unwrap(), 0);
        assert_eq!(
            std::fs::read_to_string(current.join("a/t1.json")).unwrap(),
            "one"
        );
    }

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
        let store = store_for(&root, "local", KnownSecrets::new()).unwrap();
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

    /// CTX6: the conversation is in clear on disk, so its root is the server's own
    /// application directory (beside the vault), absolute, never a relative path that
    /// would land in whatever directory the server was started from.
    #[test]
    fn the_default_root_is_the_servers_application_directory() {
        let root = default_root();
        // Windows: the local profile (%LOCALAPPDATA%), never the roaming one.
        let base = if cfg!(target_os = "windows") {
            dirs::data_local_dir()
        } else {
            dirs::config_dir()
        };
        if let Some(app) = base {
            assert!(root.starts_with(&app), "{}", root.display());
            assert!(root.is_absolute());
        }
        assert!(root.ends_with(DIR_NAME));
    }

    /// CTX6: a root inside a git work tree (a project, a worktree, a submodule, or
    /// reached through a symbolic link) is refused: `git add -A` would publish it.
    #[test]
    fn a_root_inside_a_git_work_tree_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tmp.path().join("app").join(DIR_NAME);
        assert!(!inside_git_worktree(&outside));
        assert!(store_for(&outside, "local", KnownSecrets::new()).is_ok());

        let repo = tmp.path().join("project");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let nested = repo.join("data").join(DIR_NAME);
        assert!(inside_git_worktree(&nested));
        let refused = store_for(&nested, "local", KnownSecrets::new()).unwrap_err();
        assert_eq!(refused.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!nested.exists(), "nothing is created there");

        // A linked worktree or a submodule has a `.git` FILE.
        let linked = tmp.path().join("wt");
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(linked.join(".git"), "gitdir: /elsewhere").unwrap();
        assert!(inside_git_worktree(&linked.join(DIR_NAME)));

        #[cfg(unix)]
        {
            let link = tmp.path().join("innocent");
            std::os::unix::fs::symlink(&repo, &link).unwrap();
            assert!(inside_git_worktree(&link.join(DIR_NAME)));
        }
    }

    /// CTX6: the file is the conversation (a field named like a secret, a long
    /// argument, all byte for byte); only a key the resolver handed out is masked,
    /// and never inside a tool-call argument.
    #[test]
    fn the_store_keeps_the_conversation_and_masks_only_a_known_value() {
        use nexus_claude::agent::Secret;
        use nexus_claude::model::ToolCallChunk;
        let tmp = tempfile::tempdir().unwrap();
        let known = KnownSecrets::new();
        let store = store_for(tmp.path(), "local", known.clone()).unwrap();
        known.register(&Secret::new("vault-key-77e1b0"));
        let long =
            serde_json::json!({ "max_tokens": 4096, "path": "segment/".repeat(80) }).to_string();
        let conversation = vec![
            ChatMessage::user(r#"{"input_tokens":12} and vault-key-77e1b0"#),
            ChatMessage::assistant_tool_calls(vec![ToolCallChunk {
                id: "c1".into(),
                name: "mcp__po__note".into(),
                arguments: long.clone(),
            }]),
            ChatMessage::tool("c1", "ok"),
        ];
        store.save("t", &conversation).unwrap();
        let back = store.load("t").unwrap().unwrap();
        assert_eq!(
            back[0].content.as_deref(),
            Some(r#"{"input_tokens":12} and <redacted>"#)
        );
        assert_eq!(back[1].tool_calls[0].arguments, long);
        assert_eq!(back[1..], conversation[1..]);
    }

    /// The provider-facing prefix is the system message + tools + replayed history.
    /// This store holds the history; a resume replays it byte for byte, and a later
    /// turn APPENDS to it — never rewrites the part the endpoint already cached.
    /// This proves the append-only contract over a full save → load (resume) →
    /// append (turn) → save cycle: the cached prefix stays byte-identical.
    #[test]
    fn a_turn_appends_to_the_replayed_history_without_touching_its_prefix() {
        use nexus_claude::model::ToolCallChunk;
        let tmp = tempfile::tempdir().unwrap();
        let store = store_for(tmp.path(), "local", KnownSecrets::new()).unwrap();
        let prefix = vec![
            ChatMessage::user("plan the auth module"),
            ChatMessage::assistant("Sure, here is the plan."),
            ChatMessage::assistant_tool_calls(vec![ToolCallChunk {
                id: "c1".into(),
                name: "mcp__po__plan".into(),
                arguments: r#"{"title":"auth"}"#.into(),
            }]),
            ChatMessage::tool("c1", "created plan 42"),
        ];
        store.save("t", &prefix).unwrap();

        // Resume: the transcript replays byte for byte.
        let resumed = store.load("t").unwrap().unwrap();
        assert_eq!(resumed, prefix, "resume replays the history verbatim");

        // A turn appends a new exchange; the already-sent prefix is never rewritten.
        let mut next = resumed;
        next.push(ChatMessage::user("continue with the tests"));
        next.push(ChatMessage::assistant("Wrote the tests."));
        assert_eq!(
            &next[..prefix.len()],
            prefix.as_slice(),
            "the appended turn leaves the cached prefix byte-identical"
        );
        store.save("t", &next).unwrap();
        let after = store.load("t").unwrap().unwrap();
        assert_eq!(after, next, "the turn is saved append-only, byte for byte");
        assert_eq!(&after[..prefix.len()], prefix.as_slice());
    }

    /// CACHE-HIT ACCOUNTING — the nexus boundary, documented precisely.
    ///
    /// The wire parse of a cache hit lives in NEXUS, not here:
    /// `claude-code-sdk-rs/src/model/wire.rs::parse_usage` reads an
    /// OpenAI-compatible `usage` object and splits the cached part OUT of the
    /// prompt: `prompt_tokens_details.cached_tokens` (OpenAI form) or
    /// `prompt_cache_hit_tokens` (DeepSeek form) becomes `Usage.cache_read_tokens`,
    /// and `Usage.input_tokens` is set to `prompt_tokens - cached` — what was NOT
    /// served from the cache, i.e. what the price table bills. That function is
    /// `pub(crate)` in the `nexus-claude` git dependency, so it is not reachable
    /// from this crate's tests.
    ///
    /// The highest layer of THIS crate that surfaces a turn's usage is
    /// `chat::cost::session_usage`: it reads the `result` event payloads that
    /// `chat::provider::event_map::usage_value` filled from the nexus `Usage`
    /// (the same `cache_read_tokens` field as Claude Code's), sums the cache reads
    /// apart from the billed input and computes the cache-hit ratio. This test
    /// proves that accounting at that boundary with a fake provider usage (no
    /// network): the cache hit is reported, and the billed input does not count it
    /// twice.
    #[test]
    fn cache_reads_flow_into_the_turn_usage_without_double_counting_input() {
        use serde_json::json;
        // What nexus `parse_usage` yields for a DeepSeek turn that reported
        // `prompt_cache_hit_tokens: 3800` and `prompt_cache_miss_tokens: 200`
        // (`prompt_tokens: 4000`): the miss is the billed input, the hit the cache
        // read — the total prompt is their sum, never the input twice.
        let first = json!({"type": "result", "model": "deepseek-chat",
            "usage": {"input_tokens": 200, "output_tokens": 60, "cache_read_tokens": 3800}});
        let second = json!({"type": "result", "model": "deepseek-chat",
            "usage": {"input_tokens": 150, "output_tokens": 40, "cache_read_tokens": 0}});
        let usage = crate::chat::cost::session_usage([(1, &first), (2, &second)]);

        assert_eq!((usage.turns, usage.turns_with_usage), (2, 2));
        // The cache hit is surfaced, and it is > 0 (what the DSH e2e asserts).
        assert_eq!(usage.cache_read_tokens, Some(3800));
        assert!(usage.by_turn[0].cache_read_tokens.unwrap() > 0);
        // The billed input does NOT double-count the cached part: 200 + 150, not
        // (200 + 3800) + 150.
        assert_eq!(usage.input_tokens, Some(350));
        assert_eq!(usage.output_tokens, Some(100));
        // cache_hit_ratio = cache_read / (input + cache_read).
        assert_eq!(usage.by_turn[0].cache_hit_ratio, Some(3800.0 / 4000.0));
        assert_eq!(usage.by_turn[1].cache_hit_ratio, Some(0.0));
        assert!((usage.cache_hit_ratio.unwrap() - 3800.0 / 4150.0).abs() < 1e-12);
    }
}
