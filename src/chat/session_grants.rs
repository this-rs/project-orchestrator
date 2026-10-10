//! What a permission granted "for the session" covers (P11a, P11c), decided by the backend.
//!
//! A `session` approval is NEVER handed to a provider's allow list (a nexus `allow`
//! pattern, the CLI's `updatedPermissions`): those match on loose forms (a command split
//! on `; & |` only, redirections ignored, a whole program as a prefix) and the security
//! review of P11 showed an approval of `cat README.md` turning into `cat x >> ~/.zshrc`.
//! The provider is answered `once`; the backend keeps the grant in the session's memory
//! and answers ITSELF the later requests of the SAME session it covers.
//!
//! A session grant may only ever cover exactly what the user saw and approved, and only
//! for an action that cannot run code the model can change. Everything is an ALLOWLIST:
//! what is not listed below is refused (`session` answered `ScopeUnsupported`, typed; the
//! user answers `once`).
//!
//! 1. The request must carry the WHOLE call ([`Asker`]): the Claude Code CLI and the
//!    native engine hand the model's own tool input; Codex only for an MCP tool call
//!    whose elicitation carries its `tool_params`. Never Codex's `apply_patch` (its input
//!    is `{reason}`: no path, no diff), its `shell` (an argv joined by spaces: `["rm",
//!    "a b"]` and `["rm", "a", "b"]` read the same) or `request_permissions`; never ACP.
//! 2. The tool must be a known one whose identical call does the same thing:
//!    - a read-only built-in tool of `nexus-tools` (`Read`, `Glob`, `Grep`, `LS`,
//!      `NotebookRead`), identified FOR SURE (the adapter's `canonical` name AND the
//!      `mcp__nexus__` name it belongs to), is covered whole: any later call of it;
//!    - the file tools (`Write`, `Edit`, `MultiEdit`, `NotebookEdit`), the reads of the
//!      CLI (it only asks a read OUTSIDE the working directory, so never whole), the web
//!      tools (`WebFetch`, `WebSearch`) and a third party's MCP tool: the identical call;
//!    - a command (`Bash`, `Monitor`): the identical line, and only when EVERY simple
//!      command of it is a program of [`SAFE_PROGRAMS`] (programs that execute nothing
//!      from the project), none of its options able to run something, the line readable
//!      for sure ([`simple_commands`]);
//!    - anything else (`SlashCommand`, `Skill`, `Task`, `Agent`, `TaskStop`, a tool the
//!      backend does not know) is refused.
//!
//! "Identical": the same tool, the same input (canonical JSON; for a command, the line with
//! its surrounding blanks trimmed and every field but the cosmetic `description`). The
//! working directory is NOT part of it: both engines keep the `cd` of one call for the
//! next, so a grant of `ls` lists whatever directory the shell is in, and a grant of
//! `echo x > out` writes `out` there. The programs of the allowlist only read, list or
//! print; what a redirection writes is in the line the user approved.
//!
//! The programs are found through the server's `PATH`, never the model's: neither engine
//! keeps an environment change from one call to the next, and a line that changes it
//! (`export`, `hash`, `FOO=1 cmd`...) is never granted.
//!
//! Order: the backend only sees the requests the provider ASKED, i.e. after the
//! provider's own policy (read-only access, explicit denies, trust, plan mode) and the
//! project's consent decided to ask; a grant can therefore never allow what the policy
//! refuses, and never answers a request that was not asked. Grants live in the session's
//! memory: another session (or the same one after a restart) asks again.
//!
//! Lasting approvals (`always`) are deliberately absent from this lot (P11b).

use serde_json::Value;

/// The MCP server under which the native engine serves its built-in tools
/// (`mcp__nexus__Read`).
const NEXUS_TOOLS_SERVER: &str = nexus_claude::providers::native::NEXUS_TOOLS_SERVER;

/// Read-only built-in tools of `nexus-tools` (no side effect): a session grant covers any
/// later call of them on the native engine, once identified for sure
/// ([`AskedCall::nexus_builtin`]).
const READ_ONLY: &[&str] = &["Read", "Glob", "Grep", "LS", "NotebookRead"];

/// Built-in tools (Claude Code CLI names, `nexus-tools` canonical names) whose input is the
/// whole call and whose identical call does the same thing: granted for the identical call.
const EXACT_TOOLS: &[&str] = &[
    "Read",
    "Glob",
    "Grep",
    "LS",
    "NotebookRead",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
];

/// Built-in tools that run a command line (`command`): granted for the identical line when
/// it only runs [`SAFE_PROGRAMS`].
const COMMAND_BUILTINS: &[&str] = &["Bash", "Monitor"];

/// Tool names (alias, exact name, or the last segment of an MCP name) that run a command
/// line: a third party's tool of that name goes through the command checks too. Matching
/// more names here only ever refuses more.
const COMMAND_TOOLS: &[&str] = &["Bash", "Monitor", "shell", "exec_command"];

/// Who asked: decides whether a request's input is the WHOLE call (`grant_for`, rule 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asker {
    /// The Claude Code CLI (legacy engine, or the `claude_code` provider of the agent
    /// engine): the input is the model's `tool_use` input.
    ClaudeCode,
    /// The native engine of nexus (`nexus-tools`, MCP servers): the model's tool input.
    Native,
    /// The Codex app-server: only an MCP elicitation carrying its `tool_params` is whole.
    Codex,
    /// Any other provider (ACP, a test kit...): nothing is granted for the session.
    Other,
}

impl Asker {
    /// From a provider kind's wire name (`nexus_claude::agent::ProviderKind::as_str`).
    pub fn from_provider_kind(kind: &str) -> Self {
        match kind {
            "claude_code" => Self::ClaudeCode,
            "native" => Self::Native,
            "codex" => Self::Codex,
            _ => Self::Other,
        }
    }
}

/// A program a session grant may cover in a command line: it executes nothing from the
/// project (no script, no plugin, no configuration file of the working tree), whatever the
/// files it reads say. Its options that can run something are refused.
struct SafeProgram {
    name: &'static str,
    /// Long options that run a program (`--pre`): refused, also abbreviated (GNU
    /// `getopt_long` takes any unambiguous prefix: `--compress` for `--compress-program`)
    /// and in their `--opt=value` form.
    long: &'static [&'static str],
    /// Single-dash words that run a program (`find -exec`).
    single: &'static [&'static str],
    /// Short options that run a program, also inside a cluster (`fd -Hx`).
    short: &'static [char],
}

impl SafeProgram {
    const fn plain(name: &'static str) -> Self {
        Self {
            name,
            long: &[],
            single: &[],
            short: &[],
        }
    }

    /// Whether some option of it can run a program: a glob in its arguments could then
    /// expand to that option (a file named `--pre=./x.sh`), so none is allowed.
    fn has_dangerous_options(&self) -> bool {
        !(self.long.is_empty() && self.single.is_empty() && self.short.is_empty())
    }

    fn forbids(&self, word: &str) -> bool {
        if let Some(long) = word.strip_prefix("--") {
            let name = long.split('=').next().unwrap_or(long);
            return !name.is_empty()
                && self
                    .long
                    .iter()
                    .any(|option| option.trim_start_matches('-').starts_with(name));
        }
        if self.single.contains(&word) {
            return true;
        }
        word.strip_prefix('-')
            .is_some_and(|cluster| cluster.chars().any(|c| self.short.contains(&c)))
    }
}

/// The programs a `session` grant of a command line may run (P11c): they only read, list,
/// compare or print. Everything else is refused: interpreters, shells, task runners, build
/// tools, linters and servers that load project files (`eslint`, `vite`, `mypy`...),
/// `git` (hooks, `core.fsmonitor`, diff drivers from the repository's configuration),
/// `sed` (`-f`, the `e` command), `awk`, `jq`, `tar`, `sqlite3`, `xargs`, `env`, any
/// builtin that changes how a name resolves (`export`, `hash`, `enable`, `alias`, `cd`...).
const SAFE_PROGRAMS: &[SafeProgram] = &[
    SafeProgram::plain("ls"),
    SafeProgram::plain("cat"),
    SafeProgram::plain("head"),
    SafeProgram::plain("tail"),
    SafeProgram::plain("wc"),
    SafeProgram::plain("grep"),
    SafeProgram::plain("egrep"),
    SafeProgram::plain("fgrep"),
    SafeProgram {
        name: "rg",
        long: &["--pre", "--pre-glob", "--hostname-bin"],
        single: &[],
        short: &[],
    },
    SafeProgram {
        name: "find",
        long: &[],
        single: &["-exec", "-execdir", "-ok", "-okdir"],
        short: &[],
    },
    SafeProgram {
        name: "fd",
        long: &["--exec", "--exec-batch"],
        single: &[],
        short: &['x', 'X'],
    },
    SafeProgram {
        name: "fdfind",
        long: &["--exec", "--exec-batch"],
        single: &[],
        short: &['x', 'X'],
    },
    SafeProgram {
        name: "sort",
        long: &["--compress-program"],
        single: &[],
        short: &[],
    },
    SafeProgram::plain("uniq"),
    SafeProgram::plain("cut"),
    SafeProgram::plain("tr"),
    SafeProgram::plain("nl"),
    SafeProgram::plain("diff"),
    SafeProgram::plain("cmp"),
    SafeProgram::plain("stat"),
    SafeProgram::plain("file"),
    SafeProgram::plain("du"),
    SafeProgram::plain("df"),
    SafeProgram::plain("tree"),
    SafeProgram::plain("pwd"),
    SafeProgram::plain("echo"),
    // `printf -v VAR` assigns a variable of the shell (`printf -v PATH ./bin; ls`).
    SafeProgram {
        name: "printf",
        long: &[],
        single: &[],
        short: &['v'],
    },
    SafeProgram::plain("which"),
    SafeProgram::plain("basename"),
    SafeProgram::plain("dirname"),
    SafeProgram::plain("realpath"),
    SafeProgram::plain("readlink"),
    SafeProgram::plain("whoami"),
    SafeProgram::plain("uname"),
    SafeProgram::plain("id"),
    SafeProgram::plain("true"),
    SafeProgram::plain("false"),
];

/// The names of [`SAFE_PROGRAMS`], in order (documentation, tests).
pub fn safe_program_names() -> Vec<&'static str> {
    SAFE_PROGRAMS.iter().map(|p| p.name).collect()
}

/// A permission request, as the backend saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct AskedCall {
    /// Who asked (the session's provider kind).
    pub asker: Asker,
    /// The tool's name as the provider gave it (`Bash`, `mcp__nexus__Bash`).
    pub tool: String,
    /// Its stable alias (`Bash`), when the adapter gave one. The Claude Code CLI path
    /// gives none.
    pub canonical: Option<String>,
    pub input: Value,
}

/// What a session grant of a call would be, before the input is looked at.
enum Kind {
    WholeTool,
    Exact,
    Command,
}

impl AskedCall {
    /// The canonical name of a built-in tool of `nexus-tools`, identified for sure: the
    /// adapter named it (`canonical`) AND the call is that tool of the `nexus` server.
    /// Never the suffix of a name: `mcp__acme__Read` is a third party's tool.
    fn nexus_builtin(&self) -> Option<&str> {
        let canonical = self.canonical.as_deref()?;
        (self.tool == format!("mcp__{NEXUS_TOOLS_SERVER}__{canonical}")).then_some(canonical)
    }

    /// `mcp__<server>__<tool>`, both parts non-empty: an MCP server's tool.
    fn mcp_server(&self) -> Option<&str> {
        let rest = self.tool.strip_prefix("mcp__")?;
        let (server, tool) = rest.split_once("__")?;
        (!server.is_empty() && !tool.is_empty()).then_some(server)
    }

    /// A tool that runs a command line. Broad on purpose (alias, name or MCP suffix):
    /// being one only adds checks.
    fn is_command(&self) -> bool {
        let suffix = self.tool.rsplit("__").next().unwrap_or(&self.tool);
        [
            self.canonical.as_deref(),
            Some(self.tool.as_str()),
            Some(suffix),
        ]
        .into_iter()
        .flatten()
        .any(|name| COMMAND_TOOLS.contains(&name))
    }

    /// A third party's MCP tool: the identical call (the command checks when it is named
    /// like a command tool).
    fn third_party(&self) -> Kind {
        if self.is_command() {
            Kind::Command
        } else {
            Kind::Exact
        }
    }

    /// Rules 1 and 2 of the module: what a grant of this call may be, `None` when it
    /// cannot be granted for the session at all.
    fn kind(&self) -> Option<Kind> {
        match self.asker {
            Asker::Native => {
                if let Some(name) = self.nexus_builtin() {
                    return if READ_ONLY.contains(&name) {
                        Some(Kind::WholeTool)
                    } else if EXACT_TOOLS.contains(&name) {
                        Some(Kind::Exact)
                    } else if COMMAND_BUILTINS.contains(&name) {
                        Some(Kind::Command)
                    } else {
                        None
                    };
                }
                match self.mcp_server() {
                    // A `nexus` tool the adapter did not name for sure: unknown.
                    Some(NEXUS_TOOLS_SERVER) | None => None,
                    Some(_) => Some(self.third_party()),
                }
            }
            Asker::ClaudeCode => {
                if EXACT_TOOLS.contains(&self.tool.as_str()) {
                    Some(Kind::Exact)
                } else if self.tool == "Bash" {
                    Some(Kind::Command)
                } else if self.mcp_server().is_some() {
                    Some(self.third_party())
                } else {
                    None
                }
            }
            // Only an MCP tool call whose elicitation carries the call's arguments
            // (`tool_params`): nexus names it `mcp__<server>__<tool>` with that same name
            // as `canonical`, and gives `{"message": ...}` when the arguments are missing.
            Asker::Codex => {
                let whole = self.mcp_server().is_some()
                    && self.canonical.as_deref() == Some(self.tool.as_str())
                    && self.input.as_object().is_some_and(|fields| {
                        !(fields.len() == 1 && fields.contains_key("message"))
                    });
                whole.then(|| self.third_party())
            }
            Asker::Other => None,
        }
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

/// The input an exact grant compares. A command: its line with the surrounding blanks
/// trimmed, and every other field (`timeout`, `run_in_background`, a sandbox switch...)
/// but the cosmetic `description`, which the model rewrites freely. `None`: a command
/// with no command line (an argv is not one: its words joined lose where each begins).
fn canonical_input(call: &AskedCall) -> Option<String> {
    if call.is_command() {
        let mut input = call.input.clone();
        let Value::String(line) = input.get("command")? else {
            return None;
        };
        input["command"] = Value::String(line.trim().to_string());
        if let Some(fields) = input.as_object_mut() {
            fields.remove("description");
        }
        return Some(canonical_json(&input));
    }
    Some(canonical_json(&call.input))
}

/// One word of a shell line, its quotes removed.
#[derive(Debug, Default, Clone, PartialEq)]
struct Word {
    text: String,
    /// It holds an unquoted glob character (`*`, `?`, `[`): the shell may expand it to
    /// file names the model chose (a file named `--pre=./x.sh`).
    glob: bool,
}

/// The simple commands of a shell line, each as its words (quotes removed), split on
/// `; & | ( )` and new lines outside quotes. `None` when the line cannot be read for sure,
/// i.e. when the words the shell would run might not be the ones read here:
///
/// - an expansion: `$` (a variable, `${x:=...}`, `$(...)`, `$((...))`) outside single
///   quotes, a backquote anywhere, a process substitution (`<(...)`, `>(...)`);
/// - a brace outside quotes: `{bash,x.sh}` runs `bash x.sh`, `{ cmd; }` is a group;
/// - a backslash outside single quotes: `ba\⏎sh x.sh` (a line continuation) runs
///   `bash x.sh`;
/// - an unterminated quote, a control character (only blanks, tabs and new lines are
///   read: a non-breaking space is not a separator for the shell).
fn simple_commands(line: &str) -> Option<Vec<Vec<Word>>> {
    if ["<(", ">("].iter().any(|s| line.contains(s)) {
        return None;
    }
    let mut commands = Vec::new();
    let mut words: Vec<Word> = Vec::new();
    let mut word = Word::default();
    let mut in_word = false;
    // The word so far ends with an unquoted `<` / `>`: a `&` after it is part of the
    // redirection (`2>&1`), not a separator.
    let mut redirect_tail = false;
    let mut quote: Option<char> = None;
    let mut chars = line.chars().peekable();
    let end_word = |word: &mut Word, words: &mut Vec<Word>, in_word: &mut bool| {
        if *in_word {
            words.push(std::mem::take(word));
            *in_word = false;
        }
    };
    while let Some(c) = chars.next() {
        if c.is_control() && c != '\t' && c != '\n' {
            return None;
        }
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else if q == '"' && matches!(c, '$' | '`' | '\\') {
                return None;
            } else {
                word.text.push(c);
            }
            continue;
        }
        let tail = std::mem::replace(&mut redirect_tail, false);
        match c {
            '\\' | '$' | '`' | '{' | '}' => return None,
            '\'' | '"' => {
                quote = Some(c);
                in_word = true;
            }
            // `2>&1`, `>&2`, `<&3`, `&>file`, `&>>file`: a redirection.
            '&' if tail || chars.peek() == Some(&'>') => {
                word.text.push(c);
                in_word = true;
            }
            ';' | '&' | '|' | '\n' | '(' | ')' => {
                end_word(&mut word, &mut words, &mut in_word);
                if !words.is_empty() {
                    commands.push(std::mem::take(&mut words));
                }
            }
            ' ' | '\t' => end_word(&mut word, &mut words, &mut in_word),
            c => {
                if matches!(c, '*' | '?' | '[') {
                    word.glob = true;
                }
                redirect_tail = matches!(c, '<' | '>');
                word.text.push(c);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        return None;
    }
    end_word(&mut word, &mut words, &mut in_word);
    if !words.is_empty() {
        commands.push(words);
    }
    Some(commands)
}

/// Whether a simple command (its words) only runs what the line shows: its first word is
/// a program of [`SAFE_PROGRAMS`], plainly named (no glob, no path, no assignment or
/// redirection in front), with none of its options that run something.
fn runs_only_what_it_shows(words: &[Word]) -> bool {
    let Some((program, args)) = words.split_first() else {
        return true;
    };
    if program.glob {
        return false;
    }
    let Some(safe) = SAFE_PROGRAMS.iter().find(|p| p.name == program.text) else {
        return false;
    };
    if safe.has_dangerous_options() && args.iter().any(|w| w.glob) {
        return false;
    }
    !args.iter().any(|w| safe.forbids(&w.text))
}

/// Whether this command call only runs what its line shows ([`runs_only_what_it_shows`]
/// on each of its simple commands). `false` for a line that cannot be read for sure, an
/// argv, no command at all.
fn command_is_grantable(call: &AskedCall) -> bool {
    match call.input.get("command") {
        Some(Value::String(line)) => simple_commands(line)
            .is_some_and(|commands| commands.iter().all(|words| runs_only_what_it_shows(words))),
        _ => false,
    }
}

/// The grant an approval "for the session" of this call gives, or `None` when the call
/// cannot be granted for the session (see the module: an input that is not the whole
/// call, a tool not on the allowlist, a command running a program not on the allowlist):
/// the answer `session` is then refused, typed.
pub fn grant_for(call: &AskedCall) -> Option<SessionGrant> {
    if call.tool.is_empty() || call.tool == "unknown" {
        return None;
    }
    match call.kind()? {
        Kind::WholeTool => Some(SessionGrant::WholeTool {
            tool: call.tool.clone(),
        }),
        Kind::Command if !command_is_grantable(call) => None,
        Kind::Exact | Kind::Command => Some(SessionGrant::Exact {
            tool: call.tool.clone(),
            input: canonical_input(call)?,
        }),
    }
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

    /// The grant that covers this call, if any: a grant covers a call when that call,
    /// approved for the session, would give this very grant (every rule of [`grant_for`]
    /// applies again to the later call).
    pub fn covering(&self, call: &AskedCall) -> Option<&SessionGrant> {
        let grant = grant_for(call)?;
        self.grants.iter().find(|g| **g == grant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(asker: Asker, tool: &str, canonical: Option<&str>, input: Value) -> AskedCall {
        AskedCall {
            asker,
            tool: tool.into(),
            canonical: canonical.map(str::to_string),
            input,
        }
    }

    /// A `Bash` call of the Claude Code CLI.
    fn bash(command: &str) -> AskedCall {
        call(
            Asker::ClaudeCode,
            "Bash",
            None,
            json!({ "command": command, "description": "d" }),
        )
    }

    /// A `Bash` call of `nexus-tools` (native engine).
    fn native_bash(command: &str) -> AskedCall {
        call(
            Asker::Native,
            "mcp__nexus__Bash",
            Some("Bash"),
            json!({ "command": command }),
        )
    }

    fn granted(call: &AskedCall) -> SessionGrants {
        let mut grants = SessionGrants::default();
        grants.add(grant_for(call).expect("grantable"));
        grants
    }

    /// Refused for the session on both engines that run a command line.
    fn refused_everywhere(command: &str) {
        assert!(grant_for(&bash(command)).is_none(), "CLI: {command:?}");
        assert!(
            grant_for(&native_bash(command)).is_none(),
            "native: {command:?}"
        );
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
            "/bin/ls",
            "bash -c 'ls'",
            "sh -c ls",
            "timeout 5 ls",
            "xargs rm",
            "npx some-package",
            "ssh host ls",
            "docker exec box ls",
            "command ls",
            "builtin echo",
            "exec ls",
            "nohup ls",
        ] {
            refused_everywhere(command);
        }
        // ...nor covered by a grant of the command they wrap.
        let grants = granted(&bash("ls"));
        for wrapped in ["/usr/bin/env ls", "sudo ls", "bash -c ls"] {
            assert!(grants.covering(&bash(wrapped)).is_none(), "{wrapped:?}");
        }
    }

    /// The model can edit a script, a Makefile, a test, a git hook or a tool's
    /// configuration (edits may be auto-accepted) and rerun the "identical" call: the code
    /// that would run under the grant is not the code the user approved.
    #[test]
    fn a_script_an_interpreter_or_a_task_runner_is_never_granted_for_the_session() {
        for command in [
            // Local scripts and programs given by path.
            "./x.sh",
            "./x.sh --flag",
            "scripts/deploy",
            "bin/rails test",
            "node_modules/.bin/jest",
            "~/bin/tool",
            // Interpreters, a script or inline code.
            "python3 x.py",
            "python x.py",
            "python3.12 -m pytest",
            "node -e 'require(\"./x\")'",
            "node x.js",
            "ruby x.rb",
            "perl x.pl",
            "deno run x.ts",
            "bun x.ts",
            "awk -f prog.awk data",
            "go run .",
            // Task runners, tests, package scripts, build scripts.
            "make",
            "make test",
            "npm test",
            "npm run build",
            "npm exec foo",
            "pnpm run lint",
            "yarn test",
            "cargo run",
            "cargo test",
            "cargo build",
            "just check",
            "task build",
            "pytest",
            "gradle build",
            // git runs hooks and the repository's own configuration.
            "git commit -m x",
            "git push",
            "git status",
            "git diff",
            "git log --oneline",
            "git show HEAD",
            "git -c core.fsmonitor=./x status",
            // `find` running a command on what it finds.
            "find . -name '*.sh' -exec ./x.sh {} ;",
            "find . -name x -execdir ./x.sh ';'",
            "find . -ok ./x.sh ';'",
            "fd -x ./x.sh",
            "fd -HX ./x.sh",
            "fd --exec ./x.sh",
            "fd --exec-batch ./x.sh",
        ] {
            refused_everywhere(command);
        }
    }

    /// The independent review of #681 (finding 3): a denylist cannot be complete. Each of
    /// these ran code the user never saw under a grant of the "identical" line; the
    /// allowlist refuses them all.
    #[test]
    fn the_programs_of_the_review_that_run_unseen_code_are_refused() {
        for command in [
            // A file run through an option of an otherwise harmless program.
            "rg --pre ./x.sh foo",
            "rg --pre=./x.sh foo",
            "rg --pre-glob '*.md' --pre ./x.sh foo",
            "rg --hostname-bin=./x.sh --hyperlink-format=default foo",
            "rg --pr ./x.sh foo",
            "sort --compress-program=./x.sh big.txt",
            "sort --compress-prog ./x.sh big.txt",
            "sort -S 1 --compress=./x.sh big.txt",
            "tar -I ./x.sh -cf out.tar src",
            "tar --use-compress-program=./x.sh -xf a.tar",
            "arch -arm64 ./x.sh",
            "sandbox-exec -p '(version 1)' ./x.sh",
            "enable -f ./x.so ls",
            // A change of how a name resolves.
            "hash -p ./x.sh ls; ls",
            "export PATH=./bin:/usr/bin; ls",
            "PATH=./bin ls",
            "alias ls=./x.sh",
            "printf -v PATH ./bin; ls",
            "printf -vPATH ./bin",
            "cd bin",
            "set -a",
            "declare -x PATH=./bin",
            // Tools that load the project's own code or configuration.
            "eslint .",
            "vite build",
            "uvicorn app:app",
            "flask run",
            "mypy src",
            "gulp",
            "sed -f script.sed file",
            "sed -n 1p file",
            "sqlite3 -init ./x.sql db.sqlite",
            "jq -f prog.jq data.json",
            "less README.md",
            "prettier --check .",
            "tsc",
            "webpack",
            "ruff check",
            "black .",
            "rustc x.rs",
            "gcc x.c",
            "clang x.c",
            "swift x.swift",
        ] {
            refused_everywhere(command);
        }
    }

    /// A glob next to a program with a dangerous option: the shell could expand it to a
    /// file the model named `--pre=./x.sh`.
    #[test]
    fn a_glob_is_refused_where_it_could_become_an_option_that_runs_something() {
        for command in [
            "rg foo *",
            "rg foo src/*",
            "sort *.txt",
            "find * -name x",
            "fd x ?",
            "printf %s *",
            "ls; rg foo [a-z]*",
        ] {
            refused_everywhere(command);
        }
        // A quoted glob is a word, and a glob for a program without such an option is
        // only file names.
        for command in [
            "rg 'foo*' src",
            "find . -name '*.rs'",
            "ls *.rs",
            "wc -l src/*.rs",
        ] {
            assert!(grant_for(&bash(command)).is_some(), "{command:?}");
        }
    }

    /// The independent review of #681 (finding 2): the words read must be the words the
    /// shell runs. Brace expansion (`{bash,x.sh}` runs `bash x.sh`) and a line continuation
    /// (`ba\⏎sh x.sh` runs `bash x.sh`) are not read: no session grant.
    #[test]
    fn brace_expansion_and_a_line_continuation_make_the_line_unreadable() {
        for command in [
            "{bash,x.sh}",
            "{ls,-la}",
            "ls; {bash,x.sh}",
            "echo {a,b}",
            "ls x{1..3}",
            "{ ls; ./x.sh; }",
            "{ ls; }",
            "ba\\\nsh x.sh",
            "ls \\\n -la",
            "\"ba\\\nsh\" x.sh",
            "l\\s",
            "ls a\\ b",
            "ls \"a\\\"b\"",
            "ls \\",
        ] {
            refused_everywhere(command);
        }
        // Quoted, a brace is a character.
        assert!(grant_for(&bash("grep -n '{bash,x.sh}' src")).is_some());
        assert!(grant_for(&bash("echo \"{a,b}\"")).is_some());
    }

    #[test]
    fn a_program_hidden_after_an_operator_or_a_substitution_is_checked_too() {
        for command in [
            "ls && ./x.sh",
            "ls || make",
            "true; npm test",
            "cat a | python3",
            "ls & ./x.sh",
            "ls |& ./x.sh",
            "(cd sub && npm test)",
            "(ls; ./x.sh)",
            "ls\n./x.sh",
            "if true; then ./x.sh; fi",
            "while true; do make; done",
            "for f in *.sh; do $f; done",
            "! ./x.sh",
            // A quoted `>` is no redirection: the `&` after it puts `echo` in the
            // background and `./x.sh` runs.
            "echo '>'& ./x.sh",
            "echo \">\"& ./x.sh",
            // Substitutions and expansions: refused wherever they are, even quoted.
            "echo $(./x.sh)",
            "echo \"$(./x.sh)\"",
            "ls `./x.sh`",
            "diff <(./x.sh) b",
            "cat >(./x.sh)",
            "echo $HOME",
            "echo ${PATH:=./bin}",
            "echo $((1+1))",
            // zsh glob qualifiers run code: the parenthesis is a separator, its content
            // a command of its own.
            "ls *(e:./x.sh:)",
            // A line that cannot be read safely.
            "ls 'unterminated",
            "ls \"unterminated",
            "ls\u{a0}-la",
            "ls\u{0}-la",
            "ls\r",
            // The program not plainly named.
            "$CMD",
            "\"$CMD\" x",
            "*.sh",
            "l? x",
            "2>/dev/null ./x.sh",
            ">out ls",
            "&>out ls",
        ] {
            refused_everywhere(command);
        }
    }

    #[test]
    fn a_line_of_safe_programs_stays_grantable_for_the_identical_call() {
        for command in [
            "ls -la",
            "ls",
            "pwd",
            "cat README.md",
            "grep -rn 'foo;bar' src",
            "echo 'a | ./x.sh'",
            "echo 'a $(./x.sh) `x` {a,b} \\'",
            "find . -name '*.rs'",
            "find . -type f -newer Cargo.toml",
            "fd -e rs src",
            "wc -l a.rs | sort -n",
            "rg -n foo src",
            "rg --no-pre foo",
            "head -n 20 src/main.rs",
            "tail -n 5 log.txt 2>&1",
            "ls >&2",
            "ls 2>/dev/null",
            "ls &>/dev/null",
            "diff -u a.rs b.rs",
            "stat -f %z Cargo.toml",
            "du -sh target && df -h .",
            "which cargo",
            "printf '%s\\n' a b",
            "cat a.txt | sort | uniq -c | sort -rn | head",
            "echo done > out.txt",
        ] {
            assert!(
                matches!(grant_for(&bash(command)), Some(SessionGrant::Exact { .. })),
                "CLI: {command:?}"
            );
            assert!(
                matches!(
                    grant_for(&native_bash(command)),
                    Some(SessionGrant::Exact { .. })
                ),
                "native: {command:?}"
            );
        }
    }

    #[test]
    fn the_allowlist_is_the_documented_one() {
        assert_eq!(
            safe_program_names(),
            vec![
                "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "fd",
                "fdfind", "sort", "uniq", "cut", "tr", "nl", "diff", "cmp", "stat", "file", "du",
                "df", "tree", "pwd", "echo", "printf", "which", "basename", "dirname", "realpath",
                "readlink", "whoami", "uname", "id", "true", "false",
            ]
        );
    }

    #[test]
    fn a_command_grant_ignores_the_description_but_not_what_changes_the_execution() {
        let with = |description: &str, extra: Value| {
            let mut input = json!({ "command": "ls -la", "description": description });
            if let (Some(i), Some(e)) = (input.as_object_mut(), extra.as_object()) {
                i.extend(e.clone());
            }
            call(Asker::ClaudeCode, "Bash", None, input)
        };
        let grants = granted(&with("list the files", json!({})));
        assert!(grants
            .covering(&with("show me the directory", json!({})))
            .is_some());
        for extra in [
            json!({ "run_in_background": true }),
            json!({ "dangerouslyDisableSandbox": true }),
            json!({ "timeout": 600000 }),
        ] {
            assert!(
                grants
                    .covering(&with("list the files", extra.clone()))
                    .is_none(),
                "{extra}"
            );
        }
    }

    #[test]
    fn the_monitor_tool_of_nexus_goes_through_the_same_command_checks() {
        let monitor = |command: &str| {
            call(
                Asker::Native,
                "mcp__nexus__Monitor",
                Some("Monitor"),
                json!({ "command": command }),
            )
        };
        assert!(grant_for(&monitor("tail -f log.txt")).is_some());
        assert!(grant_for(&monitor("./watch.sh")).is_none());
        assert!(grant_for(&monitor("npm run dev")).is_none());
    }

    #[test]
    fn a_read_only_tool_of_nexus_is_granted_whole_any_other_only_for_the_identical_call() {
        let read = |path: &str| {
            call(
                Asker::Native,
                "mcp__nexus__Read",
                Some("Read"),
                json!({ "file_path": path }),
            )
        };
        let grants = granted(&read("a.rs"));
        assert!(grants.covering(&read("b.rs")).is_some());
        assert_eq!(
            grant_for(&read("a.rs")).unwrap().describe(),
            "mcp__nexus__Read (read-only)"
        );

        let write = |path: &str, content: &str| {
            call(
                Asker::Native,
                "mcp__nexus__Write",
                Some("Write"),
                json!({ "file_path": path, "content": content }),
            )
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

    /// A third party's MCP tool named like a built-in is not that built-in: it can do
    /// anything. Only the identical call is granted.
    #[test]
    fn a_third_party_tool_named_like_a_read_only_built_in_gets_only_an_exact_grant() {
        for name in ["Read", "Glob", "Grep", "LS", "NotebookRead"] {
            for canonical in [None, Some(name)] {
                for asker in [Asker::Native, Asker::ClaudeCode] {
                    let at = |path: &str| {
                        call(
                            asker,
                            &format!("mcp__other__{name}"),
                            canonical,
                            json!({ "file_path": path }),
                        )
                    };
                    let grant = grant_for(&at("a.rs")).unwrap();
                    assert!(
                        matches!(grant, SessionGrant::Exact { .. }),
                        "mcp__other__{name} ({canonical:?}, {asker:?}): {grant:?}"
                    );
                    let mut grants = SessionGrants::default();
                    grants.add(grant);
                    assert!(grants.covering(&at("a.rs")).is_some());
                    assert!(grants.covering(&at("/etc/passwd")).is_none());
                }
            }
        }
        // A whole-tool grant never covers a third party's tool of that name.
        let mut grants = SessionGrants::default();
        grants.add(SessionGrant::WholeTool {
            tool: "mcp__other__Read".into(),
        });
        assert!(grants
            .covering(&call(
                Asker::Native,
                "mcp__other__Read",
                None,
                json!({ "file_path": "x" }),
            ))
            .is_none());
        // Nor a `nexus` tool the adapter did not name.
        assert!(grant_for(&call(
            Asker::Native,
            "mcp__nexus__Read",
            None,
            json!({ "file_path": "x" }),
        ))
        .is_none());
    }

    /// A third party's MCP tool named like a command tool goes through the command checks.
    #[test]
    fn a_third_party_tool_named_like_a_command_tool_is_checked_like_a_command() {
        for asker in [Asker::Native, Asker::ClaudeCode] {
            let shell = |command: &str| {
                call(
                    asker,
                    "mcp__acme__shell",
                    None,
                    json!({ "command": command }),
                )
            };
            assert!(grant_for(&shell("./x.sh")).is_none(), "{asker:?}");
            assert!(grant_for(&shell("ls -la")).is_some(), "{asker:?}");
        }
    }

    /// The Claude Code CLI only asks a read OUTSIDE the working directory: a whole-tool
    /// grant on its first ask would open `~/.ssh`, `~/.aws/credentials`...
    #[test]
    fn a_read_of_the_claude_code_cli_is_granted_only_for_the_identical_call() {
        for (name, canonical) in [
            ("Read", None),
            ("Read", Some("Read")),
            ("Grep", None),
            ("Glob", None),
            ("LS", None),
            ("NotebookRead", None),
        ] {
            let at = |path: &str| {
                call(
                    Asker::ClaudeCode,
                    name,
                    canonical,
                    json!({ "file_path": path }),
                )
            };
            let grant = grant_for(&at("/tmp/notes.txt")).unwrap();
            assert!(
                matches!(grant, SessionGrant::Exact { .. }),
                "{name}: {grant:?}"
            );
            let mut grants = SessionGrants::default();
            grants.add(grant);
            assert!(grants.covering(&at("/tmp/notes.txt")).is_some());
            assert!(
                grants.covering(&at("/Users/me/.ssh/id_ed25519")).is_none(),
                "{name}"
            );
        }
    }

    /// The review of #681 (finding 5): a slash command or a skill runs a file
    /// (`.claude/commands/*.md`, `SKILL.md`) the model can edit; a subagent does whatever
    /// its prompt leads to. A tool the backend does not know is refused too.
    #[test]
    fn a_slash_command_a_skill_a_subagent_or_an_unknown_tool_is_never_granted() {
        for tool in [
            "SlashCommand",
            "Skill",
            "Task",
            "Agent",
            "ExitPlanMode",
            "BashOutput",
            "KillShell",
            "TodoWrite",
            "SomethingNew",
        ] {
            let input = json!({ "command": "/deploy", "skill": "x", "prompt": "p" });
            assert!(
                grant_for(&call(Asker::ClaudeCode, tool, None, input.clone())).is_none(),
                "CLI {tool}"
            );
            assert!(
                grant_for(&call(Asker::ClaudeCode, tool, Some(tool), input.clone())).is_none(),
                "CLI {tool} (canonical)"
            );
            assert!(
                grant_for(&call(
                    Asker::Native,
                    &format!("mcp__nexus__{tool}"),
                    Some(tool),
                    input,
                ))
                .is_none(),
                "native {tool}"
            );
        }
        // `TaskStop` of nexus-tools: not on the allowlist either.
        assert!(grant_for(&call(
            Asker::Native,
            "mcp__nexus__TaskStop",
            Some("TaskStop"),
            json!({ "task_id": "t" }),
        ))
        .is_none());
        // Even a grant made up for them does not cover their next call.
        let mut grants = SessionGrants::default();
        let skill = call(Asker::ClaudeCode, "Skill", None, json!({ "skill": "x" }));
        grants.add(SessionGrant::Exact {
            tool: "Skill".into(),
            input: canonical_json(&skill.input),
        });
        assert!(grants.covering(&skill).is_none());
    }

    // ------------------------------------------------------------------ Codex

    /// What nexus (`codex/map.rs`) gives for a `FileChangeApproval`: `{reason}` only.
    fn codex_patch(reason: Option<&str>) -> AskedCall {
        call(
            Asker::Codex,
            "apply_patch",
            Some("Edit"),
            json!({ "reason": reason }),
        )
    }

    /// The review of #681 (finding 1): the input of Codex's `apply_patch` holds neither
    /// the path nor the diff, so a grant of one patch would cover every later patch
    /// (`~/.zshrc`, `~/.ssh/authorized_keys`...). Never granted for the session.
    #[test]
    fn two_codex_patches_with_different_changes_are_never_covered_by_one_grant() {
        // The first patch (to README.md) and the second (to ~/.zshrc) read the same.
        let first = codex_patch(None);
        let second = codex_patch(None);
        assert_eq!(first, second, "nexus gives no path nor diff");
        assert!(grant_for(&first).is_none(), "no session grant of a patch");
        // The grant #681 kept would have covered the second patch: it no longer does.
        let mut grants = SessionGrants::default();
        grants.add(SessionGrant::Exact {
            tool: "apply_patch".into(),
            input: canonical_json(&first.input),
        });
        assert!(grants.covering(&second).is_none());
        assert!(grant_for(&codex_patch(Some("write the config"))).is_none());
    }

    /// The review of #681 (finding 4): nexus joins Codex's argv with spaces
    /// (`["rm","-rf","a b"]` and `["rm","-rf","a","b"]` give one line): `shell` is never
    /// granted for the session, whatever the command (the pinned nexus passes no raw argv).
    #[test]
    fn the_codex_shell_tool_is_never_granted_for_the_session() {
        for command in [json!("ls -la"), json!("rm -rf a b"), json!(["ls", "-la"])] {
            let shell = call(
                Asker::Codex,
                "shell",
                Some("Bash"),
                json!({ "command": command, "cwd": "/w", "reason": null }),
            );
            assert!(grant_for(&shell).is_none(), "{command}");
            let mut grants = SessionGrants::default();
            grants.add(SessionGrant::Exact {
                tool: "shell".into(),
                input: canonical_json(&shell.input),
            });
            assert!(grants.covering(&shell).is_none(), "{command}");
        }
        // An argv is never a command line, on any engine.
        let argv = call(
            Asker::Native,
            "mcp__nexus__Bash",
            Some("Bash"),
            json!({ "command": ["ls", "-la"] }),
        );
        assert!(grant_for(&argv).is_none());
    }

    #[test]
    fn a_codex_mcp_call_is_granted_only_when_its_elicitation_carries_the_arguments() {
        let mcp = |canonical: Option<&str>, input: Value| {
            call(Asker::Codex, "mcp__acme__deploy", canonical, input)
        };
        // With `tool_params`: the identical call only.
        let with = |target: &str| mcp(Some("mcp__acme__deploy"), json!({ "target": target }));
        let grants = granted(&with("staging"));
        assert!(grants.covering(&with("staging")).is_some());
        assert!(grants.covering(&with("production")).is_none());
        // Without them nexus gives the elicitation's message: never granted, and two
        // calls with different arguments are not covered by one grant.
        let without = mcp(
            Some("mcp__acme__deploy"),
            json!({ "message": "Allow acme to run deploy?" }),
        );
        assert!(grant_for(&without).is_none());
        let mut grants = SessionGrants::default();
        grants.add(SessionGrant::Exact {
            tool: "mcp__acme__deploy".into(),
            input: canonical_json(&without.input),
        });
        assert!(grants.covering(&without).is_none());
        // No tool name (`mcp__<server>` only, no canonical): never granted.
        assert!(grant_for(&call(
            Asker::Codex,
            "mcp__acme",
            None,
            json!({ "target": "staging" }),
        ))
        .is_none());
        assert!(grant_for(&mcp(None, json!({ "target": "staging" }))).is_none());
        // A non-object input is not a call's arguments.
        assert!(grant_for(&mcp(Some("mcp__acme__deploy"), json!("staging"))).is_none());
        // A Codex MCP tool named like a command tool goes through the command checks.
        let shell = |command: &str| {
            call(
                Asker::Codex,
                "mcp__acme__shell",
                Some("mcp__acme__shell"),
                json!({ "command": command }),
            )
        };
        assert!(grant_for(&shell("./x.sh")).is_none());
        assert!(grant_for(&shell("ls")).is_some());
    }

    #[test]
    fn codex_permission_profiles_and_other_providers_are_never_granted() {
        assert!(grant_for(&call(
            Asker::Codex,
            "request_permissions",
            None,
            json!({ "network": true }),
        ))
        .is_none());
        // A Codex tool named like a built-in is not one.
        for tool in ["Read", "Write", "Bash", "mcp__nexus__Read"] {
            assert!(
                grant_for(&call(
                    Asker::Codex,
                    tool,
                    Some("Read"),
                    json!({ "file_path": "a" })
                ))
                .is_none(),
                "{tool}"
            );
        }
        // ACP, a test kit: nothing.
        for (tool, input) in [
            ("Read", json!({ "file_path": "a" })),
            ("Bash", json!({ "command": "ls" })),
            ("mcp__acme__deploy", json!({ "target": "x" })),
        ] {
            assert!(
                grant_for(&call(Asker::Other, tool, Some(tool), input)).is_none(),
                "{tool}"
            );
        }
        assert_eq!(Asker::from_provider_kind("claude_code"), Asker::ClaudeCode);
        assert_eq!(Asker::from_provider_kind("native"), Asker::Native);
        assert_eq!(Asker::from_provider_kind("codex"), Asker::Codex);
        assert_eq!(Asker::from_provider_kind("acp"), Asker::Other);
        assert_eq!(Asker::from_provider_kind("scripted"), Asker::Other);
        assert_eq!(Asker::from_provider_kind(""), Asker::Other);
    }

    #[test]
    fn a_grant_says_exactly_what_it_covers() {
        assert_eq!(
            grant_for(&bash("ls -la src")).unwrap().describe(),
            "Bash: ls -la src"
        );
    }

    #[test]
    fn a_request_without_a_tool_or_a_command_is_not_grantable() {
        assert!(grant_for(&call(Asker::ClaudeCode, "unknown", None, json!({}))).is_none());
        assert!(grant_for(&call(Asker::ClaudeCode, "", None, json!({}))).is_none());
        assert!(grant_for(&call(Asker::ClaudeCode, "Bash", None, json!({}))).is_none());
        assert!(grant_for(&call(
            Asker::ClaudeCode,
            "Bash",
            None,
            json!({ "command": 42 })
        ))
        .is_none());
    }
}
