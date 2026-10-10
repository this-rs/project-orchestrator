//! What a permission granted "for the session" covers (P11a), decided by the backend.
//!
//! A `session` approval is NEVER handed to a provider's allow list (a nexus `allow`
//! pattern, the CLI's `updatedPermissions`): those match on loose forms (a command split
//! on `; & |` only, redirections ignored, a whole program as a prefix) and the security
//! review of P11 showed an approval of `cat README.md` turning into `cat x >> ~/.zshrc`.
//! The provider is answered `once`; the backend keeps the grant in the session's memory
//! and answers ITSELF the later requests of the SAME session it covers:
//!
//! - a read-only tool (`Read`, `Glob`, `Grep`, `LS`, `NotebookRead`: no side effect) is
//!   covered whole: any later call of that tool;
//! - any other tool only for an IDENTICAL call: the same tool, the same input
//!   (canonical JSON, a command's surrounding blanks trimmed). `cat README.md` covers
//!   `cat README.md`, never `cat x >> ~/.zshrc`; `ls` never `ls > ~/.ssh/authorized_keys`;
//! - a command that runs ANOTHER command (`env`, `sudo`, `bash -c`, `xargs`, `timeout`,
//!   `ssh`, `docker exec`..., by basename: `/usr/bin/env` is `env`) gets no session
//!   grant at all: the answer is refused, typed.
//!
//! Order: the backend only sees the requests the provider ASKED, i.e. after the
//! provider's own policy (read-only access, explicit denies, trust, plan mode) and the
//! project's consent decided to ask; a grant can therefore never allow what the policy
//! refuses, and never answers a request that was not asked. Grants live in the session's
//! memory: another session (or the same one after a restart) asks again.
//!
//! Lasting approvals (`always`) are deliberately absent from this lot (P11b).

use serde_json::Value;

/// Tools with no side effect: a session grant covers any later call of them.
const READ_ONLY: &[&str] = &["Read", "Glob", "Grep", "LS", "NotebookRead"];

/// Programs that run another command, a script or a remote command: never granted for
/// the session (compared by basename).
const RUNS_ANOTHER: &[&str] = &[
    "env", "sudo", "doas", "su", "exec", "command", "builtin", "nohup", "nice", "ionice", "setsid",
    "time", "timeout", "watch", "stdbuf", "flock", "chroot", "strace", "ltrace", "xargs",
    "parallel", "eval", "source", ".", "sh", "bash", "zsh", "dash", "ksh", "ash", "fish", "csh",
    "tcsh", "npx", "uvx", "bunx", "pnpx", "ssh", "docker", "podman", "kubectl", "script", "expect",
    "busybox",
];

/// A permission request, as the backend saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct AskedCall {
    /// The tool's name as the provider gave it (`Bash`, `mcp__nexus__Bash`).
    pub tool: String,
    /// Its stable alias (`Bash`), when the adapter gave one.
    pub canonical: Option<String>,
    pub input: Value,
}

impl AskedCall {
    /// The neutral name of the tool: the alias, else the part after the last `__`.
    fn base(&self) -> &str {
        self.canonical
            .as_deref()
            .unwrap_or_else(|| self.tool.rsplit("__").next().unwrap_or(&self.tool))
    }

    fn is_command(&self) -> bool {
        matches!(self.base(), "Bash" | "Monitor" | "shell" | "exec_command")
    }
}

/// What one session grant covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionGrant {
    /// Any later call of this read-only tool.
    WholeTool { tool: String },
    /// Exactly this call again.
    Exact { tool: String, input: String },
}

impl SessionGrant {
    /// The grant as the user is shown it (`permission_decision.rule`).
    pub fn describe(&self) -> String {
        match self {
            Self::WholeTool { tool } => format!("{tool} (read-only)"),
            Self::Exact { tool, input } => {
                // A command reads as the command line; anything else as its input.
                let command = serde_json::from_str::<Value>(input)
                    .ok()
                    .and_then(|v| v.get("command").and_then(Value::as_str).map(str::to_string));
                match command {
                    Some(command) => format!("{tool}: {command}"),
                    None => format!("{tool} {input}"),
                }
            }
        }
    }

    fn covers(&self, call: &AskedCall) -> bool {
        match self {
            Self::WholeTool { tool } => *tool == call.tool,
            Self::Exact { tool, input } => {
                *tool == call.tool && canonical_input(call).as_deref() == Some(input.as_str())
            }
        }
    }
}

/// `value` as JSON with its object keys sorted (so two spellings of one input compare
/// equal whatever the key order).
fn canonical_json(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = serde_json::Map::new();
                for k in keys {
                    out.insert(k.clone(), sorted(&map[k]));
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(value).to_string()
}

/// The input an exact grant compares; a command's surrounding blanks trimmed. `None`:
/// a command with no command text.
fn canonical_input(call: &AskedCall) -> Option<String> {
    if call.is_command() {
        let mut input = call.input.clone();
        let command = input.get("command")?;
        let trimmed = match command {
            Value::String(s) => Value::String(s.trim().to_string()),
            // Codex's argv form.
            Value::Array(_) => command.clone(),
            _ => return None,
        };
        input["command"] = trimmed;
        return Some(canonical_json(&input));
    }
    Some(canonical_json(&call.input))
}

/// The program a command runs first: the first word of the line (or of argv), its
/// basename; `None` when there is none.
fn program(call: &AskedCall) -> Option<String> {
    let first = match call.input.get("command")? {
        Value::String(s) => s.split_whitespace().next()?.to_string(),
        Value::Array(argv) => argv.first()?.as_str()?.to_string(),
        _ => return None,
    };
    // `FOO=1 cmd`: an assignment in front hides the program.
    if first.contains('=') {
        return Some("env".to_string());
    }
    let base = first
        .trim_matches(|c| c == '"' || c == '\'')
        .rsplit('/')
        .next()
        .unwrap_or(&first)
        .to_string();
    Some(base)
}

/// The grant an approval "for the session" of this call gives, or `None` when the call
/// cannot be granted for the session (a command that runs another command, a command
/// with no text): the answer `session` is then refused, typed.
pub fn grant_for(call: &AskedCall) -> Option<SessionGrant> {
    if call.tool.is_empty() || call.tool == "unknown" {
        return None;
    }
    if READ_ONLY.contains(&call.base()) {
        return Some(SessionGrant::WholeTool {
            tool: call.tool.clone(),
        });
    }
    if call.is_command() {
        let program = program(call)?;
        if RUNS_ANOTHER.contains(&program.as_str()) {
            return None;
        }
    }
    Some(SessionGrant::Exact {
        tool: call.tool.clone(),
        input: canonical_input(call)?,
    })
}

/// The grants of one session.
#[derive(Debug, Default)]
pub struct SessionGrants {
    grants: Vec<SessionGrant>,
}

impl SessionGrants {
    /// Keeps a grant for the rest of the session.
    pub fn add(&mut self, grant: SessionGrant) {
        if !self.grants.contains(&grant) {
            self.grants.push(grant);
        }
    }

    /// The grant that covers this call, if any.
    pub fn covering(&self, call: &AskedCall) -> Option<&SessionGrant> {
        self.grants.iter().find(|g| g.covers(call))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bash(command: &str) -> AskedCall {
        AskedCall {
            tool: "Bash".into(),
            canonical: None,
            input: json!({ "command": command, "description": "d" }),
        }
    }

    fn granted(call: &AskedCall) -> SessionGrants {
        let mut grants = SessionGrants::default();
        grants.add(grant_for(call).expect("grantable"));
        grants
    }

    #[test]
    fn cat_readme_granted_for_the_session_does_not_allow_appending_to_zshrc() {
        let grants = granted(&bash("cat README.md"));
        assert!(grants.covering(&bash("cat README.md")).is_some());
        assert!(grants.covering(&bash("  cat README.md ")).is_some());
        assert!(grants.covering(&bash("cat x >> ~/.zshrc")).is_none());
        assert!(grants
            .covering(&bash("cat README.md >> ~/.zshrc"))
            .is_none());
        assert!(grants.covering(&bash("cat README.md; rm -rf ~")).is_none());
    }

    #[test]
    fn ls_granted_for_the_session_does_not_allow_a_redirection() {
        let grants = granted(&bash("ls"));
        assert!(grants
            .covering(&bash("ls > ~/.ssh/authorized_keys"))
            .is_none());
        assert!(grants.covering(&bash("ls -la")).is_none());
        assert!(grants.covering(&bash("lsof")).is_none());
    }

    #[test]
    fn a_command_that_runs_another_command_is_never_granted_for_the_session() {
        for command in [
            "/usr/bin/env -i ls",
            "env ls",
            "FOO=1 ls",
            "sudo ls",
            "/usr/bin/sudo -n true",
            "bash -c 'ls'",
            "sh -c ls",
            "timeout 5 ls",
            "xargs rm",
            "npx some-package",
            "ssh host ls",
            "docker exec box ls",
        ] {
            assert!(grant_for(&bash(command)).is_none(), "{command:?}");
        }
        // ...nor covered by a grant of the command they wrap.
        let grants = granted(&bash("ls"));
        for wrapped in ["/usr/bin/env ls", "sudo ls", "bash -c ls"] {
            assert!(grants.covering(&bash(wrapped)).is_none(), "{wrapped:?}");
        }
    }

    #[test]
    fn a_read_only_tool_is_granted_whole_any_other_only_for_the_identical_call() {
        let read = |path: &str| AskedCall {
            tool: "mcp__nexus__Read".into(),
            canonical: Some("Read".into()),
            input: json!({ "file_path": path }),
        };
        let grants = granted(&read("a.rs"));
        assert!(grants.covering(&read("b.rs")).is_some());
        assert_eq!(
            grant_for(&read("a.rs")).unwrap().describe(),
            "mcp__nexus__Read (read-only)"
        );

        let write = |path: &str, content: &str| AskedCall {
            tool: "mcp__nexus__Write".into(),
            canonical: Some("Write".into()),
            input: json!({ "file_path": path, "content": content }),
        };
        let grants = granted(&write("a.rs", "x"));
        // Same input, other key order: covered.
        let reordered = AskedCall {
            input: json!({ "content": "x", "file_path": "a.rs" }),
            ..write("a.rs", "x")
        };
        assert!(grants.covering(&reordered).is_some());
        assert!(grants.covering(&write("a.rs", "y")).is_none());
        assert!(grants.covering(&write("b.rs", "x")).is_none());
        // Another tool of the same name on another server is another tool.
        let other = AskedCall {
            tool: "mcp__other__Write".into(),
            ..write("a.rs", "x")
        };
        assert!(grants.covering(&other).is_none());
    }

    #[test]
    fn a_grant_says_exactly_what_it_covers() {
        assert_eq!(
            grant_for(&bash("git status")).unwrap().describe(),
            "Bash: git status"
        );
    }

    #[test]
    fn a_request_without_a_tool_or_a_command_is_not_grantable() {
        let unknown = AskedCall {
            tool: "unknown".into(),
            canonical: None,
            input: json!({}),
        };
        assert!(grant_for(&unknown).is_none());
        let no_command = AskedCall {
            tool: "Bash".into(),
            canonical: None,
            input: json!({}),
        };
        assert!(grant_for(&no_command).is_none());
    }
}
