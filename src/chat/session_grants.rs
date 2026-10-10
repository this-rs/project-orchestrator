//! What a permission granted "for the session" covers (P11a), decided by the backend.
//!
//! A `session` approval is NEVER handed to a provider's allow list (a nexus `allow`
//! pattern, the CLI's `updatedPermissions`): those match on loose forms (a command split
//! on `; & |` only, redirections ignored, a whole program as a prefix) and the security
//! review of P11 showed an approval of `cat README.md` turning into `cat x >> ~/.zshrc`.
//! The provider is answered `once`; the backend keeps the grant in the session's memory
//! and answers ITSELF the later requests of the SAME session it covers. A grant never
//! covers more than what the user approved:
//!
//! - a read-only built-in tool of `nexus-tools` (`Read`, `Glob`, `Grep`, `LS`,
//!   `NotebookRead`), identified FOR SURE (the adapter's `canonical` name AND the
//!   `mcp__nexus__` name it belongs to), is covered whole: any later call of that tool.
//!   Never by the name's suffix: `mcp__acme__Read` is a third party's tool, whatever it
//!   does. Never on the Claude Code CLI either: it only asks a read OUTSIDE the working
//!   directory, so a whole-tool grant there would open `~/.ssh`, `~/.aws`...;
//! - any other call only when IDENTICAL: the same tool, the same input (canonical JSON;
//!   for a command, the command line with its surrounding blanks trimmed and every field
//!   but the cosmetic `description`). `cat README.md` covers `cat README.md`, never
//!   `cat x >> ~/.zshrc`; `ls` never `ls > ~/.ssh/authorized_keys`;
//! - a command whose identical form can still run something the user never saw gets no
//!   session grant at all (the answer is refused, typed): a command that runs ANOTHER
//!   command (`env`, `sudo`, `bash -c`, `xargs`, `ssh`, `docker exec`..., by basename:
//!   `/usr/bin/env` is `env`), a program given by path (`./x.sh`, `bin/run`), an
//!   interpreter (`python3 x.py`, `node -e`...), a task runner or build tool (`make`,
//!   `npm test`, `cargo run`...), a git command that may run hooks (`git commit`), a
//!   command substitution. The model can edit a script (edits may be auto-accepted) and
//!   rerun the "identical" call: the code that runs would not be the one approved.
//!
//! Order: the backend only sees the requests the provider ASKED, i.e. after the
//! provider's own policy (read-only access, explicit denies, trust, plan mode) and the
//! project's consent decided to ask; a grant can therefore never allow what the policy
//! refuses, and never answers a request that was not asked. Grants live in the session's
//! memory: another session (or the same one after a restart) asks again.
//!
//! Lasting approvals (`always`) are deliberately absent from this lot (P11b).

use serde_json::Value;

/// Read-only built-in tools of `nexus-tools` (no side effect): a session grant covers any
/// later call of them, once identified for sure ([`AskedCall::is_nexus_read_only`]).
const READ_ONLY: &[&str] = &["Read", "Glob", "Grep", "LS", "NotebookRead"];

/// The MCP server under which the native engine serves its built-in tools
/// (`mcp__nexus__Read`).
const NEXUS_TOOLS_SERVER: &str = nexus_claude::providers::native::NEXUS_TOOLS_SERVER;

/// Tool names (alias, exact name, or the last segment of an MCP name) that run a command
/// line: their calls go through the command checks. Matching more names here only ever
/// refuses more (a command check never widens a grant).
const COMMAND_TOOLS: &[&str] = &["Bash", "Monitor", "shell", "exec_command"];

/// Programs that run another command, a script or a remote command: never granted for
/// the session (compared by basename).
const RUNS_ANOTHER: &[&str] = &[
    "env",
    "sudo",
    "doas",
    "su",
    "exec",
    "command",
    "builtin",
    "nohup",
    "nice",
    "ionice",
    "setsid",
    "time",
    "timeout",
    "watch",
    "stdbuf",
    "flock",
    "chroot",
    "strace",
    "ltrace",
    "xargs",
    "parallel",
    "eval",
    "source",
    ".",
    "sh",
    "bash",
    "zsh",
    "dash",
    "ksh",
    "ash",
    "fish",
    "csh",
    "tcsh",
    "npx",
    "uvx",
    "bunx",
    "pnpx",
    "ssh",
    "docker",
    "podman",
    "kubectl",
    "script",
    "expect",
    "busybox",
    "alias",
    "function",
    "trap",
    "coproc",
    "unshare",
    "nsenter",
    "runuser",
    "sg",
    "caffeinate",
    "open",
    "osascript",
    "launchctl",
    "systemd-run",
    "at",
    "batch",
    "crontab",
];

/// Interpreters: they run a script file the model can edit, or code given on the line
/// that may load local files. `python*` / `pypy*` are matched by prefix.
const INTERPRETERS: &[&str] = &[
    "node",
    "nodejs",
    "deno",
    "bun",
    "ts-node",
    "tsx",
    "ruby",
    "irb",
    "perl",
    "php",
    "lua",
    "luajit",
    "Rscript",
    "R",
    "julia",
    "java",
    "jshell",
    "groovy",
    "kotlin",
    "scala",
    "swift",
    "pwsh",
    "powershell",
    "tclsh",
    "wish",
    "awk",
    "gawk",
    "mawk",
    "nawk",
    "ipython",
    "dotnet",
    "go",
    "elixir",
    "erl",
    "escript",
    "ghc",
    "runghc",
    "stack",
    "cabal",
    "racket",
    "guile",
];

/// Task runners, build tools, test runners and package managers: they run the project's
/// own code (a Makefile, a `package.json` script, a `build.rs`, a test), which the model
/// can edit between two "identical" calls.
const RUNNERS: &[&str] = &[
    "make",
    "gmake",
    "bmake",
    "cmake",
    "ctest",
    "ninja",
    "meson",
    "just",
    "task",
    "mage",
    "rake",
    "gradle",
    "gradlew",
    "mvn",
    "mvnw",
    "ant",
    "sbt",
    "mill",
    "bazel",
    "bazelisk",
    "buck",
    "buck2",
    "pants",
    "nx",
    "turbo",
    "lerna",
    "pytest",
    "py.test",
    "tox",
    "nox",
    "jest",
    "vitest",
    "mocha",
    "ava",
    "karma",
    "playwright",
    "cypress",
    "rspec",
    "phpunit",
    "composer",
    "pip",
    "pip3",
    "pipx",
    "pipenv",
    "poetry",
    "uv",
    "pdm",
    "hatch",
    "conda",
    "mamba",
    "npm",
    "pnpm",
    "yarn",
    "cargo",
    "rustup",
    "bundle",
    "bundler",
    "gem",
    "mix",
    "rebar3",
    "swiftc",
    "xcodebuild",
    "pod",
    "flutter",
    "dart",
    "terraform",
    "tofu",
    "pulumi",
    "ansible",
    "ansible-playbook",
    "vagrant",
    "pre-commit",
    "husky",
    "lefthook",
    // Every git command can run project code: hooks (`commit`, `push`, `checkout`...),
    // and the repository's own configuration (`core.fsmonitor` on `status`, diff
    // drivers, aliases), which the model can edit like any file.
    "git",
];

/// Options of `find` / `fd` that run a command on what they find.
const EXEC_OPTIONS: &[&str] = &[
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-x",
    "-X",
    "--exec",
    "--exec-batch",
];

/// Words of a shell line that only open a construct; the program is the next word.
const KEYWORDS: &[&str] = &["!", "if", "then", "else", "elif", "do", "while", "until"];

/// Words of a shell line that open a construct whose own words are not programs (the
/// body follows after `;` / `do` / `)`).
const HEADERS: &[&str] = &["for", "case", "select", "in", "fi", "done", "esac"];

/// A permission request, as the backend saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct AskedCall {
    /// The tool's name as the provider gave it (`Bash`, `mcp__nexus__Bash`).
    pub tool: String,
    /// Its stable alias (`Bash`), when the adapter gave one. The Claude Code CLI path
    /// gives none.
    pub canonical: Option<String>,
    pub input: Value,
}

impl AskedCall {
    /// A read-only built-in tool of `nexus-tools`, identified for sure: the adapter named
    /// it (`canonical`) AND the call is that tool of the `nexus` server. Never the suffix
    /// of a name: `mcp__acme__Read` is a third party's tool.
    fn is_nexus_read_only(&self) -> bool {
        self.canonical.as_deref().is_some_and(|canonical| {
            READ_ONLY.contains(&canonical)
                && self.tool == format!("mcp__{NEXUS_TOOLS_SERVER}__{canonical}")
        })
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
            Self::WholeTool { tool } => *tool == call.tool && call.is_nexus_read_only(),
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

/// The input an exact grant compares. A command: its line with the surrounding blanks
/// trimmed, and every other field (`timeout`, `run_in_background`, a sandbox switch...)
/// but the cosmetic `description`, which the model rewrites freely. `None`: a command
/// with no command text.
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
        if let Some(fields) = input.as_object_mut() {
            fields.remove("description");
        }
        return Some(canonical_json(&input));
    }
    Some(canonical_json(&call.input))
}

/// The simple commands of a shell line, each as its words (quotes removed), split on
/// `; & | ( ) { }` and new lines outside quotes. `None` when the line cannot be read
/// safely: an unterminated quote, a command substitution (`$(...)`, backquotes) or a
/// process substitution (`<(...)`, `>(...)`), anywhere, even quoted.
fn simple_commands(line: &str) -> Option<Vec<Vec<String>>> {
    if ["`", "$(", "<(", ">("].iter().any(|s| line.contains(s)) {
        return None;
    }
    let mut commands = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in line.chars() {
        if escaped {
            word.push(c);
            in_word = true;
            escaped = false;
            continue;
        }
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else if c == '\\' && q == '"' {
                escaped = true;
            } else {
                word.push(c);
            }
            continue;
        }
        match c {
            '\\' => escaped = true,
            '\'' | '"' => {
                quote = Some(c);
                in_word = true;
            }
            ';' | '&' | '|' | '\n' | '(' | ')' | '{' | '}' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
                if !words.is_empty() {
                    commands.push(std::mem::take(&mut words));
                }
            }
            c if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            c => {
                word.push(c);
                in_word = true;
            }
        }
    }
    if quote.is_some() || escaped {
        return None;
    }
    if in_word {
        words.push(word);
    }
    if !words.is_empty() {
        commands.push(words);
    }
    Some(commands)
}

/// Whether a simple command (its words) may run code the user did not see in the line:
/// another command, a script, an interpreter, a task runner, git, `find -exec`.
fn runs_unseen_code(words: &[String]) -> bool {
    let mut words = words
        .iter()
        .map(String::as_str)
        .skip_while(|w| KEYWORDS.contains(w));
    let Some(program) = words.next() else {
        return false;
    };
    if HEADERS.contains(&program) {
        return false;
    }
    // `FOO=1 cmd` (an assignment hides the program), `$cmd`, a glob, a redirection in
    // front, `~/x`: the program is not plainly named.
    if program.is_empty()
        || program.contains(['=', '$', '*', '?', '[', '~', '<', '>'])
        || (program.starts_with(|c: char| c.is_ascii_digit()) && program.contains(['<', '>']))
    {
        return true;
    }
    // A program given by path (`./x.sh`, `bin/run`, `/usr/bin/env`): a local script, or a
    // file the model may have written.
    if program.contains('/') {
        return true;
    }
    if RUNS_ANOTHER.contains(&program)
        || INTERPRETERS.contains(&program)
        || program.starts_with("python")
        || program.starts_with("pypy")
        || RUNNERS.contains(&program)
    {
        return true;
    }
    matches!(program, "find" | "fd" | "fdfind") && words.any(|w| EXEC_OPTIONS.contains(&w))
}

/// Whether this command call may run code the user did not see, whatever the line says
/// ([`runs_unseen_code`] on each of its simple commands). `None`: no command text.
fn command_runs_unseen_code(call: &AskedCall) -> Option<bool> {
    match call.input.get("command")? {
        Value::String(line) => Some(match simple_commands(line) {
            Some(commands) => commands.iter().any(|words| runs_unseen_code(words)),
            None => true,
        }),
        // Codex's argv form: one command, its words as given.
        Value::Array(argv) => {
            let words: Option<Vec<String>> = argv
                .iter()
                .map(|w| w.as_str().map(str::to_string))
                .collect();
            Some(words.is_none_or(|words| runs_unseen_code(&words)))
        }
        _ => None,
    }
}

/// The grant an approval "for the session" of this call gives, or `None` when the call
/// cannot be granted for the session (a command that may run code the user did not see,
/// a command with no text): the answer `session` is then refused, typed.
pub fn grant_for(call: &AskedCall) -> Option<SessionGrant> {
    if call.tool.is_empty() || call.tool == "unknown" {
        return None;
    }
    if call.is_nexus_read_only() {
        return Some(SessionGrant::WholeTool {
            tool: call.tool.clone(),
        });
    }
    if call.is_command() && command_runs_unseen_code(call)? {
        return None;
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

    /// The model can edit a script, a Makefile, a test or a git hook (edits may be
    /// auto-accepted) and rerun the "identical" call: the code that would run under the
    /// grant is not the code the user approved.
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
            "git -c core.fsmonitor=./x status",
            // `find` running a command on what it finds.
            "find . -name '*.sh' -exec ./x.sh {} ;",
            "fd -x ./x.sh",
        ] {
            assert!(grant_for(&bash(command)).is_none(), "{command:?}");
        }
    }

    #[test]
    fn a_program_hidden_after_an_operator_a_keyword_or_a_substitution_is_checked_too() {
        for command in [
            "ls && ./x.sh",
            "ls || make",
            "true; npm test",
            "cat a | python3",
            "ls & ./x.sh",
            "(cd sub && npm test)",
            "{ ls; ./x.sh; }",
            "ls\n./x.sh",
            "if true; then ./x.sh; fi",
            "while true; do make; done",
            "for f in *.sh; do $f; done",
            "! ./x.sh",
            // Substitutions: refused wherever they are, even quoted.
            "echo $(./x.sh)",
            "echo \"$(./x.sh)\"",
            "ls `./x.sh`",
            "diff <(./x.sh) b",
            // A line that cannot be read safely.
            "ls 'unterminated",
            "ls \\",
            // The program not plainly named.
            "$CMD",
            "\"$CMD\" x",
            "*.sh",
            "2>/dev/null ./x.sh",
        ] {
            assert!(grant_for(&bash(command)).is_none(), "{command:?}");
        }
    }

    #[test]
    fn a_plain_command_stays_grantable_for_the_identical_call() {
        for command in [
            "ls -la",
            "cat README.md",
            "grep -rn 'foo;bar' src",
            "echo 'a | ./x.sh'",
            "find . -name '*.rs'",
            "wc -l a.rs | sort -n",
            "rg -n foo src",
            "head -n 20 src/main.rs",
        ] {
            assert!(
                matches!(grant_for(&bash(command)), Some(SessionGrant::Exact { .. })),
                "{command:?}"
            );
        }
    }

    #[test]
    fn a_command_in_argv_form_is_checked_like_a_line() {
        let argv = |words: &[&str]| AskedCall {
            tool: "shell".into(),
            canonical: Some("Bash".into()),
            input: json!({ "command": words }),
        };
        for words in [
            &["python3", "x.py"][..],
            &["./x.sh"],
            &["make", "test"],
            &["bash", "-lc", "ls"],
        ] {
            assert!(grant_for(&argv(words)).is_none(), "{words:?}");
        }
        assert!(grant_for(&argv(&["ls", "-la"])).is_some());
    }

    #[test]
    fn a_command_grant_ignores_the_description_but_not_what_changes_the_execution() {
        let call = |description: &str, extra: Value| {
            let mut input = json!({ "command": "ls -la", "description": description });
            if let (Some(i), Some(e)) = (input.as_object_mut(), extra.as_object()) {
                i.extend(e.clone());
            }
            AskedCall {
                tool: "Bash".into(),
                canonical: None,
                input,
            }
        };
        let grants = granted(&call("list the files", json!({})));
        assert!(grants
            .covering(&call("show me the directory", json!({})))
            .is_some());
        for extra in [
            json!({ "run_in_background": true }),
            json!({ "dangerouslyDisableSandbox": true }),
            json!({ "timeout": 600000 }),
        ] {
            assert!(
                grants
                    .covering(&call("list the files", extra.clone()))
                    .is_none(),
                "{extra}"
            );
        }
    }

    #[test]
    fn a_read_only_tool_of_nexus_is_granted_whole_any_other_only_for_the_identical_call() {
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

    /// A third party's MCP tool named like a built-in is not that built-in: it can do
    /// anything. Only the identical call is granted.
    #[test]
    fn a_third_party_tool_named_like_a_read_only_built_in_gets_only_an_exact_grant() {
        for name in ["Read", "Glob", "Grep", "LS", "NotebookRead"] {
            for canonical in [None, Some(name.to_string())] {
                let call = |path: &str| AskedCall {
                    tool: format!("mcp__other__{name}"),
                    canonical: canonical.clone(),
                    input: json!({ "file_path": path }),
                };
                let grant = grant_for(&call("a.rs")).unwrap();
                assert!(
                    matches!(grant, SessionGrant::Exact { .. }),
                    "mcp__other__{name} ({canonical:?}): {grant:?}"
                );
                let mut grants = SessionGrants::default();
                grants.add(grant);
                assert!(grants.covering(&call("a.rs")).is_some());
                assert!(grants.covering(&call("/etc/passwd")).is_none());
            }
        }
        // A whole-tool grant of nexus never covers a third party's tool of that name.
        let mut grants = SessionGrants::default();
        grants.add(SessionGrant::WholeTool {
            tool: "mcp__other__Read".into(),
        });
        assert!(grants
            .covering(&AskedCall {
                tool: "mcp__other__Read".into(),
                canonical: None,
                input: json!({ "file_path": "x" }),
            })
            .is_none());
    }

    /// The Claude Code CLI only asks a read OUTSIDE the working directory: a whole-tool
    /// grant on its first ask would open `~/.ssh`, `~/.aws/credentials`...
    #[test]
    fn a_read_of_the_claude_code_cli_is_granted_only_for_the_identical_call() {
        for (name, canonical) in [
            ("Read", None),
            ("Read", Some("Read".to_string())),
            ("Grep", None),
            ("Glob", None),
            ("LS", None),
            ("NotebookRead", None),
        ] {
            let call = |path: &str| AskedCall {
                tool: name.into(),
                canonical: canonical.clone(),
                input: json!({ "file_path": path }),
            };
            let grant = grant_for(&call("/tmp/notes.txt")).unwrap();
            assert!(
                matches!(grant, SessionGrant::Exact { .. }),
                "{name}: {grant:?}"
            );
            let mut grants = SessionGrants::default();
            grants.add(grant);
            assert!(grants.covering(&call("/tmp/notes.txt")).is_some());
            assert!(
                grants
                    .covering(&call("/Users/me/.ssh/id_ed25519"))
                    .is_none(),
                "{name}"
            );
        }
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
