//! Lasting permission rules of the native harness: the scope `always` (P11).
//!
//! Claude Code keeps an approval "always" itself (`updatedPermissions` towards its
//! `localSettings`, `<project>/.claude/settings.local.json`), with the rule the CLI
//! suggests: `Bash(npm test:*)`, a path, a domain. The native harness of nexus has no
//! persistent rule: it declares `permission_scopes = [once, session]`. The backend
//! gives it the missing scope, with the SAME granularity as the nexus policy patterns:
//!
//! - answering `always` derives the rule from the actual call ([`rules_for_call`]):
//!   a command → its program (and subcommand) as a prefix, `Bash(git status)` and
//!   `Bash(git status *)`; a file tool → that path (normalised as nexus does); WebFetch
//!   → `WebFetch(domain:<host>)`; any other tool → its exact argument. A call no rule
//!   can name safely (a compound command, a substitution, a wrapper such as `sudo` or
//!   `sh -c`, a glob character in the argument) gets no `always`: the answer is refused,
//!   typed, never widened to the whole tool;
//! - the rule is recorded for the project (its directory), the harness is answered
//!   `once`, and the backend answers itself the next calls of the live session that the
//!   project's rules cover ([`matches_call`], `AgentSessionHandle::emit`);
//! - every native session opened or resumed in that project gets the project's rules
//!   in its `allow` list (`ChatManager::build_agent_spec_with_access`): nexus matches
//!   them as any `allow` pattern, after a restart too.
//!
//! ```text
//! <app dir>/permission-rules.json      0600
//! { "projects": { "<canonical project directory>": ["mcp__nexus__Bash(ls)", "mcp__nexus__Bash(ls *)"] } }
//! ```
//!
//! The file lives in the application directory, NOT in the project: a model that
//! can write the project (`auto_edits`) must not be able to grant itself a tool.
//! `deny` rules and the read-only access still win (`deny` is checked before
//! `allow`). Removing a rule: edit or delete the file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use nexus_claude::agent::ToolPattern;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const FILE_NAME: &str = "permission-rules.json";

/// Words that run ANOTHER command after them (nexus `policy_args::WRAPPERS`, plus the
/// shells): a prefix rule on them would allow anything.
const WRAPPERS: &[&str] = &[
    "then", "do", "else", "elif", "if", "while", "until", "!", "time", "nohup", "exec", "command",
    "builtin", "env", "nice", "sudo", "doas", "setsid", "xargs", "ionice", "sh", "bash", "zsh",
    "dash", "ksh", "ash", "eval", "source", ".",
];

/// The default file, under the application directory where the vault, the
/// identity key and the native transcripts already live.
pub fn default_path() -> PathBuf {
    let app = match dirs::config_dir() {
        #[cfg(target_os = "windows")]
        Some(c) => c.join("ProjectOrchestrator"),
        #[cfg(not(target_os = "windows"))]
        Some(c) => c.join("project-orchestrator"),
        None => dirs::home_dir()
            .map(|h| h.join(".project-orchestrator"))
            .unwrap_or_else(|| PathBuf::from(".project-orchestrator")),
    };
    app.join(FILE_NAME)
}

/// The key of a project: its directory, canonical when it exists (a symlink and
/// its target are one project), as given otherwise.
pub fn project_key(cwd: &str) -> String {
    std::fs::canonicalize(cwd)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| cwd.to_string())
}

/// A permission request of a native session, as the backend saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct AskedCall {
    /// The tool's full name (`mcp__nexus__Bash`): what nexus matches patterns on.
    pub tool: String,
    /// Its stable alias (`Bash`), when the adapter gave one.
    pub canonical: Option<String>,
    pub input: Value,
}

/// Whether `pattern` can be a rule: a pattern the nexus policy parses, with a tool
/// name that is not a glob.
pub fn valid_rule(pattern: &str) -> bool {
    pattern
        .parse::<ToolPattern>()
        .is_ok_and(|p| !p.tool.contains('*') && !p.tool.is_empty())
        && super::provider::policy::tool_policy("ask", &[pattern.to_string()], &[]).is_some()
}

/// `path` lexically normalised as the nexus policy does (`policy_args::normalise`):
/// relative to `cwd` when inside it, `.` and `..` resolved.
fn normalise(path: &str, cwd: &Path) -> String {
    let absolute = if Path::new(path).is_absolute() || path.starts_with('/') {
        PathBuf::from(path)
    } else {
        cwd.join(path)
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    let slashed = |p: &Path| {
        let shown = p.display().to_string();
        if cfg!(windows) {
            shown.replace('\\', "/")
        } else {
            shown
        }
    };
    match out.strip_prefix(cwd) {
        Ok(relative) if relative.as_os_str().is_empty() => ".".to_owned(),
        Ok(relative) => slashed(relative),
        Err(_) => slashed(&out),
    }
}

/// A command line simple enough to be named by a prefix: one command, no operator,
/// redirection, substitution, quote or escape.
fn simple_command(command: &str) -> Option<&str> {
    let command = command.trim();
    let unsafe_char = |c: char| {
        matches!(
            c,
            ';' | '&'
                | '|'
                | '`'
                | '$'
                | '<'
                | '>'
                | '('
                | ')'
                | '\n'
                | '\r'
                | '\\'
                | '\''
                | '"'
                | '*'
                | '?'
                | '['
                | ']'
                | '{'
                | '}'
                | '~'
                | '#'
        )
    };
    (!command.is_empty() && !command.chars().any(unsafe_char)).then_some(command)
}

/// The argument nexus matches an `allow` pattern against, for this call (`None`: the
/// call has no argument a rule can name).
fn call_argument(call: &AskedCall, cwd: &Path) -> Option<String> {
    let field = |name: &str| call.input.get(name).and_then(Value::as_str);
    match call.canonical.as_deref() {
        // A tool of another server: nexus matches its whole input.
        None => Some(call.input.to_string()),
        Some("Bash" | "Monitor") => simple_command(field("command")?).map(str::to_string),
        Some("Write" | "Edit" | "Read") => Some(normalise(field("file_path")?, cwd)),
        Some("NotebookEdit") => Some(normalise(field("notebook_path")?, cwd)),
        Some("WebFetch") => {
            let host = url::Url::parse(field("url")?.trim())
                .ok()?
                .host_str()?
                .trim_end_matches('.')
                .to_lowercase();
            Some(format!("domain:{host}"))
        }
        Some("Glob" | "Grep") => field("pattern").map(str::to_string),
        Some("WebSearch") => field("query").map(str::to_string),
        Some("TaskStop") => field("task_id").or(field("shell_id")).map(str::to_string),
        // A canonical tool nexus does not profile: no argument to name.
        Some(_) => None,
    }
}

/// The rules an approval `always` of this call records, scoped as the nexus policy
/// patterns are: a command's program (and its subcommand) as a prefix, a file's path,
/// a domain, or the exact argument. `None`: no rule can name this call without
/// allowing more than it (the scope `always` is then refused).
pub fn rules_for_call(call: &AskedCall, cwd: &Path) -> Option<Vec<String>> {
    let argument = call_argument(call, cwd)?;
    // A glob character would widen the rule; a bracket would break its text form.
    if argument.contains('*') || argument.is_empty() {
        return None;
    }
    let rules = match call.canonical.as_deref() {
        Some("Bash" | "Monitor") => {
            let words: Vec<&str> = argument.split_whitespace().collect();
            let program = *words.first()?;
            if WRAPPERS.contains(&program) || program.contains('=') {
                return None;
            }
            // `git status`, `npm test`, `cargo build`: the subcommand is part of the
            // prefix; a flag or a path is not.
            let prefix = match words.get(1) {
                Some(sub)
                    if !sub.starts_with('-')
                        && !sub.contains('/')
                        && !sub.contains('.')
                        && sub
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':')) =>
                {
                    format!("{program} {sub}")
                }
                _ => program.to_string(),
            };
            vec![
                format!("{}({prefix})", call.tool),
                format!("{}({prefix} *)", call.tool),
            ]
        }
        _ => vec![format!("{}({argument})", call.tool)],
    };
    rules.iter().all(|r| valid_rule(r)).then_some(rules)
}

/// Whether one of `rules` covers this call, as nexus would match it.
pub fn matches_call(rules: &[String], call: &AskedCall, cwd: &Path) -> bool {
    let Some(argument) = call_argument(call, cwd) else {
        return false;
    };
    rules
        .iter()
        .filter_map(|r| r.parse::<ToolPattern>().ok())
        .any(|p| p.matches(&call.tool, Some(&argument)))
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RulesFile {
    #[serde(default)]
    projects: BTreeMap<String, BTreeSet<String>>,
}

/// The store of the lasting rules (one file, read at each use: two servers on one
/// machine see each other's rules; writes are serialized in this process).
#[derive(Debug)]
pub struct LastingRules {
    path: PathBuf,
    write: std::sync::Mutex<()>,
}

impl LastingRules {
    /// A store kept in `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            write: std::sync::Mutex::new(()),
        }
    }

    /// Where the rules are kept.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read(&self) -> RulesFile {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!(path = %self.path.display(), error = %e, "unreadable permission rules: none applied");
                RulesFile::default()
            }),
            Err(_) => RulesFile::default(),
        }
    }

    /// The rules of this project (the `allow` entries a native session opened there
    /// gets). A hand-edited entry that is not a valid scoped rule is skipped.
    pub fn allowed_for(&self, project: &str) -> Vec<String> {
        self.read()
            .projects
            .get(project)
            .map(|rules| rules.iter().filter(|r| valid_rule(r)).cloned().collect())
            .unwrap_or_default()
    }

    /// Records the `rules` for `project`. Refuses any text that is not a valid rule.
    /// Written to a temporary file then renamed (0600).
    pub fn allow(&self, project: &str, rules: &[String]) -> Result<()> {
        if let Some(bad) = rules.iter().find(|r| !valid_rule(r)) {
            return Err(anyhow!("{bad:?} is not a rule the policy can hold"));
        }
        let _guard = self.write.lock().unwrap_or_else(|p| p.into_inner());
        let mut file = self.read();
        let entry = file.projects.entry(project.to_string()).or_default();
        let before = entry.len();
        entry.extend(rules.iter().cloned());
        if entry.len() == before {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_vec_pretty(&file)?;
        write_private(&tmp, &body).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("replacing {}", self.path.display()))?;
        Ok(())
    }
}

fn write_private(path: &Path, body: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut f = options.open(path)?;
    f.write_all(body)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bash(command: &str) -> AskedCall {
        AskedCall {
            tool: "mcp__nexus__Bash".into(),
            canonical: Some("Bash".into()),
            input: json!({ "command": command }),
        }
    }

    fn tool(name: &str, canonical: &str, input: Value) -> AskedCall {
        AskedCall {
            tool: format!("mcp__nexus__{name}"),
            canonical: Some(canonical.into()),
            input,
        }
    }

    fn policy_allows(rules: &[String], tool: &str, arg: &str) -> bool {
        use nexus_claude::agent::{PolicyDecision, ToolCategory};
        let policy = super::super::provider::policy::tool_policy("ask", rules, &[]).unwrap();
        policy.decide(tool, Some(arg), ToolCategory::Command) == PolicyDecision::Allow
    }

    #[test]
    fn an_always_on_bash_ls_does_not_allow_rm() {
        let cwd = Path::new("/work");
        let rules = rules_for_call(&bash("ls"), cwd).unwrap();
        assert_eq!(
            rules,
            vec!["mcp__nexus__Bash(ls)", "mcp__nexus__Bash(ls *)"]
        );
        // The live session (the backend's own matching)...
        assert!(matches_call(&rules, &bash("ls"), cwd));
        assert!(matches_call(&rules, &bash("ls -la src"), cwd));
        assert!(!matches_call(&rules, &bash("rm -rf x"), cwd));
        assert!(!matches_call(&rules, &bash("lsof"), cwd));
        assert!(!matches_call(&rules, &bash("ls; rm -rf x"), cwd));
        // ...and the nexus policy of a session opened later.
        assert!(policy_allows(&rules, "mcp__nexus__Bash", "ls -la"));
        assert!(!policy_allows(&rules, "mcp__nexus__Bash", "rm -rf x"));
    }

    #[test]
    fn a_command_rule_keeps_the_subcommand_not_the_arguments() {
        let cwd = Path::new("/work");
        let rules = rules_for_call(&bash("git status --short"), cwd).unwrap();
        assert_eq!(
            rules,
            vec![
                "mcp__nexus__Bash(git status)",
                "mcp__nexus__Bash(git status *)"
            ]
        );
        assert!(!matches_call(&rules, &bash("git push --force"), cwd));
        let rules = rules_for_call(&bash("cargo test -p foo"), cwd).unwrap();
        assert!(matches_call(&rules, &bash("cargo test"), cwd));
        assert!(!matches_call(&rules, &bash("cargo publish"), cwd));
    }

    #[test]
    fn a_call_no_rule_can_name_safely_gets_no_always() {
        let cwd = Path::new("/work");
        for command in [
            "ls && rm -rf x",
            "echo $(rm -rf x)",
            "sudo rm -rf x",
            "sh -c 'rm -rf x'",
            "env rm x",
            "FOO=1 rm x",
            "rm *",
            "cat a > b",
            "",
        ] {
            assert!(rules_for_call(&bash(command), cwd).is_none(), "{command:?}");
        }
        let glob = tool(
            "Write",
            "Write",
            json!({"file_path": "src/*.rs", "content": ""}),
        );
        assert!(rules_for_call(&glob, cwd).is_none());
    }

    #[test]
    fn a_file_rule_is_that_path_and_a_fetch_rule_that_domain() {
        let cwd = Path::new("/work");
        let edit = |path: &str| tool("Edit", "Edit", json!({"file_path": path}));
        let rules = rules_for_call(&edit("/work/src/a.rs"), cwd).unwrap();
        assert_eq!(rules, vec!["mcp__nexus__Edit(src/a.rs)"]);
        assert!(matches_call(&rules, &edit("src/./a.rs"), cwd));
        assert!(!matches_call(&rules, &edit("src/b.rs"), cwd));
        assert!(!matches_call(&rules, &edit(".env"), cwd));

        let fetch = |url: &str| tool("WebFetch", "WebFetch", json!({"url": url}));
        let rules = rules_for_call(&fetch("https://Docs.rs/serde/latest"), cwd).unwrap();
        assert_eq!(rules, vec!["mcp__nexus__WebFetch(domain:docs.rs)"]);
        assert!(matches_call(&rules, &fetch("https://docs.rs/tokio"), cwd));
        assert!(!matches_call(&rules, &fetch("https://evil.test/x"), cwd));
    }

    #[test]
    fn a_tool_of_another_server_is_allowed_for_its_exact_input_only() {
        let cwd = Path::new("/work");
        let call = |text: &str| AskedCall {
            tool: "mcp__po__write".into(),
            canonical: None,
            input: json!({ "text": text }),
        };
        let rules = rules_for_call(&call("one"), cwd).unwrap();
        assert!(matches_call(&rules, &call("one"), cwd));
        assert!(!matches_call(&rules, &call("two"), cwd));
    }

    #[test]
    fn rules_are_kept_per_project_and_survive_a_new_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("app").join(FILE_NAME);
        let rules = LastingRules::new(&path);
        let ls = rules_for_call(&bash("ls"), Path::new("/work/a")).unwrap();
        rules.allow("/work/a", &ls).unwrap();
        rules.allow("/work/a", &ls).unwrap();
        // A restart: a new store on the same file.
        let again = LastingRules::new(&path);
        let mut sorted = ls.clone();
        sorted.sort();
        assert_eq!(again.allowed_for("/work/a"), sorted);
        assert!(again.allowed_for("/work/b").is_empty());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn only_a_valid_scoped_rule_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let rules = LastingRules::new(dir.path().join(FILE_NAME));
        for bad in ["", "Bash(", "*", "mcp__*__Bash", "Bash()"] {
            assert!(rules.allow("/work", &[bad.to_string()]).is_err(), "{bad:?}");
        }
        assert!(rules.allowed_for("/work").is_empty());
        // A hand-edited file with a glob tool name: not applied.
        std::fs::write(
            rules.path(),
            r#"{"projects":{"/work":["Read(a.rs)","mcp__*"]}}"#,
        )
        .unwrap();
        assert_eq!(rules.allowed_for("/work"), vec!["Read(a.rs)"]);
    }
}
