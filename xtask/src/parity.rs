//! `cargo xtask parity` — the OPPOSABLE inventory of the backend's contract:
//! every REST operation the router mounts, and every action of every mega-tool
//! the MCP dispatcher knows.
//!
//! Why a gate and not a document: a hand-written inventory can forget exactly
//! the awkward routes, and an inventory regenerated *after* a migration
//! compares the new router with itself. So this subcommand never reads a
//! written list — it EXTRACTS the contract from the source (`src/api/routes.rs`
//! for the routes, the `MEGA_TOOL_ACTIONS` table in `src/mcp/handlers.rs` for
//! the MCP actions), renders a sorted TSV, and compares it byte for byte with
//! the committed file (`docs/parity/inventory.tsv`).
//!
//! Zero dependency, no grammar: a ~150-line tokenizer that knows comments,
//! string literals (regular and raw), char literals and lifetimes is enough to
//! find `.route(` calls and the string tuples of the dispatch table without a
//! single false positive from a comment or a documentation block. Everything
//! else here is a pure function over text; [`run`] is the ONLY place touching
//! disk.
//!
//! Modes: `--check` (the DEFAULT, and the only mode the CI runs) regenerates in
//! memory and compares; `--write` regenerates the file (refused when `CI` is
//! set, so a pipeline can never bless its own drift).
//!
//! Exit codes: 0 = the inventory and the source agree, 1 = a divergence was
//! CONSTATED (a route or an action added, removed or moved, or the inventory
//! is missing), 2 = could not verify (source unreadable, the parser no longer
//! understands the source, no route and no action found, bad arguments).
//!
//! What it does NOT prove: that the router serves what the file says, nor that
//! any response body agrees with anything. That is the replay
//! (`tests/parity.rs`, `record` / `replay`), which needs JSON and the real
//! router and therefore does not live in this crate.

use crate::git::Repo;
use std::collections::BTreeMap;
use std::path::PathBuf;

const HELP: &str = "\
usage: cargo xtask parity [inventory] [options]

  --inventory <file>  inventory to compare with or to write,
                      default: docs/parity/inventory.tsv
  --check             (default) regenerate in memory and compare; never writes
  --write             regenerate the inventory from the source (refused when CI is set)
  --repo <path>       checkout to read, default: the one containing the cwd

exit 0 the inventory and the source agree, 1 they diverge (or the inventory is
missing), 2 could not verify.";

/// The production router file the routes are extracted from.
pub const ROUTES_SOURCE: &str = "src/api/routes.rs";
/// The file holding the mega-tool dispatch table.
pub const HANDLERS_SOURCE: &str = "src/mcp/handlers.rs";
/// Default inventory path, relative to the repo root.
pub const DEFAULT_INVENTORY: &str = "docs/parity/inventory.tsv";
/// The mega-tool dispatch table: `(mega_tool, action, internal_name)`.
const ACTION_TABLE: &str = "MEGA_TOOL_ACTIONS";
/// The `MethodRouter` builders `Router::route` accepts as its second argument.
const VERBS: &[&str] = &[
    "get", "post", "put", "patch", "delete", "head", "options", "any",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub inventory: PathBuf,
    /// `--write`; its absence means check (the default, and the CI's mode).
    pub write: bool,
    pub repo: Option<PathBuf>,
    pub help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            inventory: PathBuf::from(DEFAULT_INVENTORY),
            write: false,
            repo: None,
            help: false,
        }
    }
}

pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut check = false;
    let mut action_seen = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--inventory" => {
                let v = it.next().ok_or("--inventory attend un chemin")?;
                o.inventory = PathBuf::from(v);
            }
            "--check" => check = true,
            "--write" => o.write = true,
            "--repo" => {
                let v = it.next().ok_or("--repo attend un chemin")?;
                o.repo = Some(PathBuf::from(v));
            }
            "-h" | "--help" => o.help = true,
            // `parity inventory --check` (the spelling of the design note) and
            // `parity --check` are the same command: the action is optional.
            "inventory" if !action_seen => action_seen = true,
            other if other.starts_with('-') => return Err(format!("option inconnue « {other} »")),
            other => {
                return Err(format!(
                    "action inconnue « {other} » — seule `inventory` existe ici (le rejeu vit dans tests/parity.rs)"
                ))
            }
        }
    }
    if check && o.write {
        return Err("--check et --write s'excluent : l'un compare, l'autre écrit".into());
    }
    Ok(o)
}

// ============================================================================
// Tokenizer — enough Rust to never confuse code with a comment or a string
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokKind {
    Ident,
    /// A string literal, quotes included (see [`str_content`]).
    Str,
    /// A single punctuation character.
    Punct(char),
    /// A char literal or any other atom nobody here needs to look inside.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tok {
    kind: TokKind,
    /// Byte offset of the first byte, in the source the tokenizer was given.
    start: usize,
    /// Byte offset just past the last byte.
    end: usize,
    /// 1-based line of the first byte.
    line: usize,
}

impl Tok {
    fn text<'a>(&self, src: &'a str) -> &'a str {
        &src[self.start..self.end]
    }

    fn is_ident(&self, src: &str, name: &str) -> bool {
        self.kind == TokKind::Ident && self.text(src) == name
    }

    fn is_punct(&self, c: char) -> bool {
        self.kind == TokKind::Punct(c)
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

/// End (exclusive) of the regular string starting at `i` (the opening quote).
fn string_end(b: &[u8], i: usize) -> usize {
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\\' => j += 2,
            b'"' => return j + 1,
            _ => j += 1,
        }
    }
    b.len()
}

/// End (exclusive) of the raw string starting at `i`, or `None` when `i` is
/// merely the letter `r`.
fn raw_string_end(b: &[u8], i: usize) -> Option<usize> {
    if b.get(i) != Some(&b'r') {
        return None;
    }
    let mut j = i + 1;
    let mut hashes = 0;
    while b.get(j) == Some(&b'#') {
        hashes += 1;
        j += 1;
    }
    if b.get(j) != Some(&b'"') {
        return None;
    }
    j += 1;
    while j < b.len() {
        if b[j] == b'"' {
            let mut k = j + 1;
            let mut n = 0;
            while n < hashes && b.get(k) == Some(&b'#') {
                n += 1;
                k += 1;
            }
            if n == hashes {
                return Some(k);
            }
        }
        j += 1;
    }
    Some(b.len())
}

/// End (exclusive) of the char literal starting at `i`, or `None` when the
/// `'` opens a lifetime (`&'static str`) instead.
fn char_literal_end(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    if b.get(j) == Some(&b'\\') {
        j += 2;
        while j < b.len() && b[j] != b'\'' {
            j += 1;
        }
        return if j < b.len() { Some(j + 1) } else { None };
    }
    match b.get(j) {
        Some(&c) if is_ident_start(c) => {
            while j < b.len() && is_ident_byte(b[j]) {
                j += 1;
            }
            if b.get(j) == Some(&b'\'') {
                Some(j + 1)
            } else {
                None // a lifetime: the `'` stays a punctuation token
            }
        }
        // `' '`, `'('`, `'"'` …: exactly one byte between the quotes.
        Some(_) if b.get(j + 1) == Some(&b'\'') => Some(j + 2),
        _ => None,
    }
}

/// Split `src` into tokens, dropping comments and whitespace. Every token
/// keeps its byte span and its line, so any later rule can quote the source
/// exactly as written.
fn tokenize(src: &str) -> Vec<Tok> {
    let b = src.as_bytes();
    let mut toks: Vec<Tok> = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i < b.len() {
                if b[i] == b'\n' {
                    line += 1;
                }
                if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        let start = i;
        let start_line = line;
        let kind = if c == b'"' {
            i = string_end(b, i);
            TokKind::Str
        } else if let Some(end) = raw_string_end(b, i) {
            i = end;
            TokKind::Str
        } else if c == b'\'' {
            match char_literal_end(b, i) {
                Some(end) => {
                    i = end;
                    TokKind::Other
                }
                None => {
                    i += 1;
                    TokKind::Punct('\'')
                }
            }
        } else if is_ident_start(c) {
            while i < b.len() && is_ident_byte(b[i]) {
                i += 1;
            }
            TokKind::Ident
        } else {
            i += 1;
            TokKind::Punct(c as char)
        };
        // A token may span lines (a multi-line regular string). Every byte in
        // `start..i` is part of this token, so the count cannot double up with
        // the top-of-loop `\n` branch.
        for byte in &b[start..i] {
            if *byte == b'\n' {
                line += 1;
            }
        }
        toks.push(Tok {
            kind,
            start,
            end: i,
            line: start_line,
        });
    }
    toks
}

/// The content of a string literal token: quotes, `r` and `#` delimiters
/// removed. Escapes are left as written (no path or table entry needs them).
pub fn str_content(raw: &str) -> &str {
    let (hashes, rest) = match raw.strip_prefix('r') {
        Some(rest) => (rest.bytes().take_while(|b| *b == b'#').count(), rest),
        None => (0, raw),
    };
    let rest = &rest[hashes..];
    let body = rest.strip_prefix('"').unwrap_or(rest);
    // A raw string closes on `"` followed by its own hashes, a regular one on
    // a bare `"`.
    let body = if hashes == 0 {
        body
    } else {
        body.strip_suffix(&"#".repeat(hashes)).unwrap_or(body)
    };
    body.strip_suffix('"').unwrap_or(body)
}

/// Index of the token closing the bracket opened at `open`, counting only the
/// bracket pair `(open_char, close_char)`.
fn matching(toks: &[Tok], open: usize, open_char: char, close_char: char) -> Option<usize> {
    let mut depth = 0usize;
    for (j, t) in toks.iter().enumerate().skip(open + 1) {
        if t.is_punct(open_char) {
            depth += 1;
        } else if t.is_punct(close_char) {
            if depth == 0 {
                return Some(j);
            }
            depth -= 1;
        }
    }
    None
}

// ============================================================================
// Routes
// ============================================================================

/// Cut a source at its first `#[cfg(test)]` attribute: test modules mount
/// routers too, and none of those routes is served by the production router.
pub fn cut_tests(src: &str) -> (&str, Option<usize>) {
    let mut offset = 0;
    for (n, line) in src.split_inclusive('\n').enumerate() {
        if line.trim_start().starts_with("#[cfg(test)]") {
            return (&src[..offset], Some(n + 1));
        }
        offset += line.len();
    }
    (src, None)
}

/// One served operation: a path plus the verb of one `MethodRouter` builder.
/// A single `.route(path, get(a).post(b))` yields two of these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteOp {
    pub method: String,
    pub path: String,
    pub tier: String,
    pub handler: String,
    pub line: usize,
}

/// What one pass over `src/api/routes.rs` found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routes {
    /// Number of `.route(` calls understood.
    pub calls: usize,
    /// Number of `.route(` occurrences in the same text, counted naively (what
    /// `grep -o '.route(' | wc -l` prints). The two must agree: a parser that
    /// silently loses a route cannot pass, because the naive count would not
    /// move with it.
    pub naive_occurrences: usize,
    pub ops: Vec<RouteOp>,
}

/// The auth tier of each route group, decided by the name of the function that
/// mounts it. An unknown name is an ERROR and never a guess: a new group would
/// otherwise be silently filed under the wrong tier.
pub fn tier_for_fn(name: &str) -> Option<&'static str> {
    let ends = |suffix: &str| name == &suffix[1..] || name.ends_with(suffix);
    if ends("_protected_routes") {
        Some("protected")
    } else if ends("_websocket_routes") {
        Some("websocket")
    } else if ends("_bounded_public_routes") || name == "bounded_public_routes" {
        Some("bounded_public")
    } else if ends("_public_routes") || ends("_ingress_routes") {
        Some("public")
    } else {
        None
    }
}

/// Collapse a multi-line expression into one line, so a TSV cell never carries
/// a newline or a tab.
fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Naive occurrence count of `.route(` (comments and strings included, on
/// purpose: it is the independent second count, not a second parser).
pub fn count_route_occurrences(code: &str) -> usize {
    code.matches(".route(").count()
}

/// Extract every operation mounted by `src/api/routes.rs`.
pub fn parse_routes(src: &str) -> Result<Routes, String> {
    let (code, _) = cut_tests(src);
    let toks = tokenize(code);
    let naive_occurrences = count_route_occurrences(code);
    let mut ops = Vec::new();
    let mut calls = 0usize;
    let mut current_fn: Option<String> = None;

    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        if t.kind == TokKind::Ident && t.text(code) == "fn" {
            if let Some(next) = toks.get(i + 1) {
                if next.kind == TokKind::Ident {
                    current_fn = Some(next.text(code).to_string());
                }
            }
            i += 1;
            continue;
        }
        let next_is_punct = |c: char| toks.get(i + 1).is_some_and(|n| n.is_punct(c));
        if (t.is_ident(code, "nest") || t.is_ident(code, "route_service")) && next_is_punct('(') {
            return Err(format!(
                "ligne {} : `{}` monte des routes sous un préfixe que la génération ne sait pas deviner — refus plutôt qu'un inventaire incomplet",
                t.line,
                t.text(code)
            ));
        }
        let is_route_call =
            t.is_ident(code, "route") && i > 0 && toks[i - 1].is_punct('.') && next_is_punct('(');
        if !is_route_call {
            i += 1;
            continue;
        }

        let open = i + 1;
        let close = matching(&toks, open, '(', ')').ok_or_else(|| {
            format!(
                "ligne {} : `.route(` dont la parenthèse fermante est introuvable",
                t.line
            )
        })?;
        let group = current_fn.clone().ok_or_else(|| {
            format!(
                "ligne {} : `.route(` hors de toute `fn` — le niveau d'authentification est indécidable",
                t.line
            )
        })?;
        let tier = tier_for_fn(&group).ok_or_else(|| {
            format!(
                "ligne {} : groupe de routes « {group} » inconnu — ajouter son niveau à tier_for_fn plutôt que de le deviner",
                t.line
            )
        })?;
        let first = toks.get(open + 1).filter(|p| p.kind == TokKind::Str);
        let path = first
            .map(|p| str_content(p.text(code)).to_string())
            .ok_or_else(|| {
                format!(
                    "ligne {} : `.route(` dont le premier argument n'est pas un chemin littéral",
                    t.line
                )
            })?;
        calls += 1;

        let mut j = open + 1;
        while j < close {
            let v = toks[j];
            // A verb counts at ANY depth of the call: `get(a).merge(post(b))` is
            // how POST /api/documents carries its own body limit, and a handler
            // nested in a builder is still a served operation.
            let is_verb = v.kind == TokKind::Ident
                && VERBS.contains(&v.text(code))
                && toks.get(j + 1).is_some_and(|n| n.is_punct('('));
            if !is_verb {
                j += 1;
                continue;
            }
            let vclose = matching(&toks, j + 1, '(', ')').ok_or_else(|| {
                format!(
                    "ligne {} : `{}` sans parenthèse fermante",
                    v.line,
                    v.text(code)
                )
            })?;
            let handler = if vclose > j + 2 {
                collapse_ws(&code[toks[j + 2].start..toks[vclose - 1].end])
            } else {
                String::new()
            };
            ops.push(RouteOp {
                method: v.text(code).to_uppercase(),
                path: path.clone(),
                tier: tier.to_string(),
                handler,
                line: t.line,
            });
            j = vclose + 1;
        }
        // Do not jump past the whole call: a nested `.route(` would be lost and
        // only the naive count would notice.
        i = open + 1;
    }

    Ok(Routes {
        calls,
        naive_occurrences,
        ops,
    })
}

// ============================================================================
// MCP actions
// ============================================================================

/// One action of one mega-tool, as the dispatch table declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpAction {
    pub tool: String,
    pub action: String,
    /// The legacy tool name the action dispatches to.
    pub internal: String,
    pub line: usize,
}

/// Extract every `(mega_tool, action, legacy_name)` of `MEGA_TOOL_ACTIONS`.
pub fn parse_mega_actions(src: &str) -> Result<Vec<McpAction>, String> {
    let (code, _) = cut_tests(src);
    let toks = tokenize(code);
    // The table is also *used* elsewhere in the file (`MEGA_TOOL_ACTIONS.iter()`,
    // tests…): take the DEFINITION, the occurrence followed by `:`.
    let at = toks
        .iter()
        .enumerate()
        .find(|(k, t)| {
            t.is_ident(code, ACTION_TABLE) && toks.get(k + 1).is_some_and(|n| n.is_punct(':'))
        })
        .map(|(k, _)| k)
        .ok_or_else(|| {
            format!("table `{ACTION_TABLE}` introuvable dans {HANDLERS_SOURCE} — le dispatch a-t-il bougé ?")
        })?;
    // The declared type has brackets too (`&[(&str, &str, &str)]`): start after
    // the `=`, so the `[` found is the one of the VALUE.
    let eq = (at..toks.len())
        .find(|&k| toks[k].is_punct('='))
        .ok_or_else(|| format!("`{ACTION_TABLE}` : `=` introuvable"))?;
    let open = (eq..toks.len())
        .find(|&k| toks[k].is_punct('['))
        .ok_or_else(|| format!("`{ACTION_TABLE}` : `[` introuvable"))?;
    let close = matching(&toks, open, '[', ']')
        .ok_or_else(|| format!("`{ACTION_TABLE}` : `]` introuvable"))?;

    let mut out: Vec<McpAction> = Vec::new();
    let mut j = open + 1;
    while j < close {
        if !toks[j].is_punct('(') {
            j += 1;
            continue;
        }
        let end = matching(&toks, j, '(', ')').ok_or_else(|| {
            format!(
                "ligne {} : entrée de {ACTION_TABLE} non fermée",
                toks[j].line
            )
        })?;
        let strs: Vec<&str> = (j + 1..end)
            .filter(|&k| toks[k].kind == TokKind::Str)
            .map(|k| str_content(toks[k].text(code)))
            .collect();
        if strs.len() != 3 {
            return Err(format!(
                "ligne {} : {} littéral(aux) au lieu de 3 — la table n'a plus la forme (mega_tool, action, legacy_name)",
                toks[j].line,
                strs.len()
            ));
        }
        out.push(McpAction {
            tool: strs[0].to_string(),
            action: strs[1].to_string(),
            internal: strs[2].to_string(),
            line: toks[j].line,
        });
        j = end + 1;
    }

    let mut seen: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for a in &out {
        if let Some(first) = seen.insert((&a.tool, &a.action), a.line) {
            return Err(format!(
                "ligne {} : (« {} », « {} ») déjà déclaré ligne {first} — le dispatch est ambigu",
                a.line, a.tool, a.action
            ));
        }
    }
    Ok(out)
}

// ============================================================================
// The inventory
// ============================================================================

/// One TSV line, whatever its kind: the four columns the gate owns, plus
/// `key=value` extras that a reviewer reads but the sort ignores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// `R` for a REST operation, `M` for an MCP action.
    pub kind: char,
    /// The path (routes) or the mega-tool name (actions).
    pub target: String,
    /// The HTTP verb (routes) or the action name (actions).
    pub action: String,
    pub file: String,
    pub line: usize,
    pub extras: Vec<String>,
}

impl Row {
    fn render(&self) -> String {
        let mut s = format!(
            "{}\t{}\t{}\t{}:{}",
            self.kind, self.target, self.action, self.file, self.line
        );
        for e in &self.extras {
            s.push('\t');
            s.push_str(e);
        }
        s
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inventory {
    pub rows: Vec<Row>,
    pub route_calls: usize,
    pub route_ops: usize,
    pub naive_route_occurrences: usize,
    pub mcp_tools: usize,
    pub mcp_actions: usize,
}

/// Stable order of the inventory: kind, then the path or the tool, then the
/// verb or the action, then the location. Sorting by content and not by line
/// keeps a diff readable when a block moves without changing.
pub fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| {
        (a.kind, &a.target, &a.action, &a.file, a.line)
            .cmp(&(b.kind, &b.target, &b.action, &b.file, b.line))
    });
}

/// Parse both sources and build the sorted inventory. Errors are all
/// "could not verify": they mean the gate does not understand the source any
/// more, not that the source and the inventory disagree.
pub fn build_inventory(routes_src: &str, handlers_src: &str) -> Result<Inventory, String> {
    let routes = parse_routes(routes_src)?;
    if routes.naive_occurrences != routes.calls {
        return Err(format!(
            "recomptage indépendant : {} occurrence(s) de `.route(` mais {} appel(s) analysé(s) — un commentaire ou une chaîne cite le motif, ou le parseur ne comprend plus la source",
            routes.naive_occurrences, routes.calls
        ));
    }
    if routes.ops.is_empty() {
        return Err(format!(
            "{ROUTES_SOURCE} : aucune opération `.route(` — mauvais fichier, ou parseur aveugle"
        ));
    }
    let actions = parse_mega_actions(handlers_src)?;
    if actions.is_empty() {
        return Err(format!(
            "{HANDLERS_SOURCE} : `{ACTION_TABLE}` ne contient aucune action — table vidée, ou parseur aveugle"
        ));
    }

    let mut rows = Vec::with_capacity(routes.ops.len() + actions.len());
    for op in &routes.ops {
        rows.push(Row {
            kind: 'R',
            target: op.path.clone(),
            action: op.method.clone(),
            file: ROUTES_SOURCE.to_string(),
            line: op.line,
            extras: vec![format!("tier={}", op.tier), format!("h={}", op.handler)],
        });
    }
    for a in &actions {
        rows.push(Row {
            kind: 'M',
            target: a.tool.clone(),
            action: a.action.clone(),
            file: HANDLERS_SOURCE.to_string(),
            line: a.line,
            extras: vec![format!("internal={}", a.internal)],
        });
    }
    sort_rows(&mut rows);

    let mut tools: Vec<&str> = actions.iter().map(|a| a.tool.as_str()).collect();
    tools.sort_unstable();
    tools.dedup();

    Ok(Inventory {
        rows,
        route_calls: routes.calls,
        route_ops: routes.ops.len(),
        naive_route_occurrences: routes.naive_occurrences,
        mcp_tools: tools.len(),
        mcp_actions: actions.len(),
    })
}

/// The whole file, header included. Two runs on the same source must print the
/// same bytes: the CI compares files, not sets.
pub fn render(inv: &Inventory) -> String {
    let mut out = String::new();
    out.push_str("# GENERATED by `cargo xtask parity --write` — DO NOT EDIT.\n");
    out.push_str(&format!(
        "# source {ROUTES_SOURCE} (production only: everything before the first `#[cfg(test)]`)\n"
    ));
    out.push_str(&format!("# source {HANDLERS_SOURCE} ({ACTION_TABLE})\n"));
    out.push_str("# columns kind<TAB>path_or_tool<TAB>action<TAB>file:line<TAB>extras\n");
    out.push_str(&format!(
        "# counts routes={} operations={} mcp_tools={} mcp_actions={} naive_route_occurrences={}\n",
        inv.route_calls, inv.route_ops, inv.mcp_tools, inv.mcp_actions, inv.naive_route_occurrences
    ));
    for row in &inv.rows {
        out.push_str(&row.render());
        out.push('\n');
    }
    out
}

/// The data lines of an inventory (the `#` header is not a row).
pub fn data_rows(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(str::to_string)
        .collect()
}

/// What differs between two inventories, as multisets of data lines.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Drift {
    /// Lines the generated inventory has and the committed one has not.
    pub added: Vec<String>,
    /// Lines the committed inventory has and the generated one has not.
    pub removed: Vec<String>,
}

impl Drift {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// Multiset difference over data lines, so a duplicate line is told apart from
/// a moved one.
pub fn diff_rows(committed: &str, generated: &str) -> Drift {
    fn counts(text: &str) -> BTreeMap<String, usize> {
        let mut m = BTreeMap::new();
        for row in data_rows(text) {
            *m.entry(row).or_insert(0) += 1;
        }
        m
    }
    let old = counts(committed);
    let new = counts(generated);
    let mut drift = Drift::default();
    for (row, n) in &new {
        let was = old.get(row).copied().unwrap_or(0);
        for _ in 0..n.saturating_sub(was) {
            drift.added.push(row.clone());
        }
    }
    for (row, n) in &old {
        let is = new.get(row).copied().unwrap_or(0);
        for _ in 0..n.saturating_sub(is) {
            drift.removed.push(row.clone());
        }
    }
    drift
}

/// A row without its location: what the inventory says about the CONTRACT, as
/// opposed to where the line currently sits in the source. Inserting one route
/// shifts every line below it — without this, a one-line change reads as a
/// 600-line diff and nobody reviews it any more.
pub fn row_without_location(row: &str) -> String {
    let mut parts: Vec<&str> = row.split('\t').collect();
    if parts.len() > 3 {
        parts.remove(3);
    }
    parts.join("\t")
}

/// Split a byte-level drift into what really changed and what merely moved.
/// Returns the rows whose contract differs, and the number of rows that are
/// the same contract at another line.
pub fn split_moves(drift: &Drift) -> (Drift, usize) {
    let mut pool: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for row in &drift.removed {
        pool.entry(row_without_location(row))
            .or_default()
            .push(row.clone());
    }
    let mut added = Vec::new();
    let mut moved = 0usize;
    for row in &drift.added {
        let key = row_without_location(row);
        match pool.get_mut(&key).and_then(Vec::pop) {
            Some(_) => moved += 1,
            None => added.push(row.clone()),
        }
    }
    let mut removed: Vec<String> = pool.into_values().flatten().collect();
    removed.sort();
    (Drift { added, removed }, moved)
}

/// How many lines of each direction a red run prints before saying « … et N
/// autre(s) ».
const PRINTED: usize = 40;

fn render_drift(inventory: &str, drift: &Drift, moved: usize) -> String {
    let mut out = format!("parity : écart — {inventory} ne correspond plus à la source\n");
    for row in drift.added.iter().take(PRINTED) {
        out.push_str(&format!("  + {row}\n"));
    }
    if drift.added.len() > PRINTED {
        out.push_str(&format!(
            "  + … et {} autre(s)\n",
            drift.added.len() - PRINTED
        ));
    }
    for row in drift.removed.iter().take(PRINTED) {
        out.push_str(&format!("  - {row}\n"));
    }
    if drift.removed.len() > PRINTED {
        out.push_str(&format!(
            "  - … et {} autre(s)\n",
            drift.removed.len() - PRINTED
        ));
    }
    out.push_str(&format!(
        "{} ajoutée(s), {} retirée(s), {} déplacée(s) (même contrat, autre file:line)\n",
        drift.added.len(),
        drift.removed.len(),
        moved
    ));
    out.push_str("régénérer : cargo xtask parity --write   (le CI ne fait que --check)\n");
    out
}

fn summary(inv: &Inventory) -> String {
    format!(
        "routes={} operations={} mcp_tools={} mcp_actions={}",
        inv.route_calls, inv.route_ops, inv.mcp_tools, inv.mcp_actions
    )
}

/// The only function of this module that touches disk.
pub fn run(args: &[String]) -> i32 {
    let o = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("parity : {e}\n\n{HELP}");
            return 2;
        }
    };
    if o.help {
        println!("{HELP}");
        return 0;
    }
    if o.write && std::env::var("CI").is_ok() {
        eprintln!(
            "parity : non vérifiable — --write interdit en CI (une pipeline ne blanchit pas sa propre dérive)"
        );
        return 2;
    }
    let start = o.repo.clone().unwrap_or_else(|| PathBuf::from("."));
    let root = match Repo::discover(&start) {
        Some(r) => r.path,
        None => {
            eprintln!(
                "parity : non vérifiable — {} n'est pas dans un dépôt git",
                start.display()
            );
            return 2;
        }
    };

    let mut sources = Vec::with_capacity(2);
    for file in [ROUTES_SOURCE, HANDLERS_SOURCE] {
        match std::fs::read_to_string(root.join(file)) {
            Ok(c) => sources.push(c),
            Err(e) => {
                eprintln!(
                    "parity : non vérifiable — {} : {e}",
                    root.join(file).display()
                );
                return 2;
            }
        }
    }

    let inv = match build_inventory(&sources[0], &sources[1]) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("parity : non vérifiable — {e}");
            return 2;
        }
    };
    let generated = render(&inv);
    let shown = o.inventory.display().to_string();
    let path = root.join(&o.inventory);

    if o.write {
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!("parity : non vérifiable — {} : {e}", dir.display());
                return 2;
            }
        }
        return match std::fs::write(&path, &generated) {
            Ok(()) => {
                println!("parity : inventaire écrit — {} — {shown}", summary(&inv));
                0
            }
            Err(e) => {
                eprintln!("parity : non vérifiable — {} : {e}", path.display());
                2
            }
        };
    }

    let committed = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "parity : écart — {shown} absent : lancer `cargo xtask parity --write` et le commiter"
            );
            return 1;
        }
        Err(e) => {
            eprintln!("parity : non vérifiable — {} : {e}", path.display());
            return 2;
        }
    };

    let drift = diff_rows(&committed, &generated);
    if committed == generated {
        println!("parity : inventaire à jour — {shown} — {}", summary(&inv));
        return 0;
    }
    if drift.is_empty() {
        println!(
            "parity : écart — l'en-tête de {shown} a bougé (comptes) : relancer `cargo xtask parity --write`"
        );
        return 1;
    }
    let (real, moved) = split_moves(&drift);
    print!("{}", render_drift(&shown, &real, moved));
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUTES_FIXTURE: &str = r#"
fn public_routes() -> Router<State> {
    Router::new()
        .route("/health", get(handlers::health))
        .route(
            "/api/things",
            get(things::list).post(things::create),
        )
        .route("/api/things/{id}", axum::routing::delete(things::remove))
}

fn websocket_routes() -> Router<State> {
    Router::new().route("/ws/feed", get(ws::feed))
}

fn protected_routes() -> Router<State> {
    Router::new().route("/api/private", get(handlers::private))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_test_router_is_not_the_production_router() {
        let _ = Router::new().route("/only-in-tests", get(handlers::ghost));
    }
}
"#;

    const HANDLERS_FIXTURE: &str = r#"
/// Single source of truth for mega-tool dispatch.
#[rustfmt::skip]
pub(crate) const MEGA_TOOL_ACTIONS: &[(&str, &str, &str)] = &[
    // Project
    ("project", "list", "list_projects"),
    ("project", "create", "create_project"),

    // Plan
    (
        "plan",
        "list",
        "list_plans",
    ),
];

fn other() {}
"#;

    #[test]
    fn parse_routes_reads_every_verb_of_a_multi_line_call() {
        let routes = parse_routes(ROUTES_FIXTURE).expect("fixture");
        assert_eq!(routes.calls, 5, "cinq appels `.route(` hors tests");
        assert_eq!(routes.ops.len(), 6, "six opérations");
        let list = routes
            .ops
            .iter()
            .find(|o| o.path == "/api/things" && o.method == "GET")
            .expect("GET /api/things");
        assert_eq!(list.handler, "things::list");
        let create = routes
            .ops
            .iter()
            .find(|o| o.path == "/api/things" && o.method == "POST")
            .expect("POST /api/things");
        assert_eq!(create.handler, "things::create");
        assert_eq!(create.tier, "public");
        assert_eq!(create.line, 5, "la ligne du `.route(`, pas celle du verbe");
        // Un verbe qualifié (`axum::routing::delete`) compte comme les autres.
        let delete = routes
            .ops
            .iter()
            .find(|o| o.method == "DELETE")
            .expect("DELETE");
        assert_eq!(delete.path, "/api/things/{id}");
        assert_eq!(delete.handler, "things::remove");
    }

    #[test]
    fn parse_routes_counts_a_verb_nested_in_a_merge_builder() {
        // Measured on the real file: `POST /api/documents` lives inside
        // `get(…).merge(post(…).layer(DefaultBodyLimit::…))`. A parser that only
        // looked at depth 0 lost that operation (628 instead of 629).
        let src = r#"
fn protected_routes() -> Router<State> {
    Router::new().route(
        "/api/documents",
        get(docs::list).merge(
            post(docs::upload).layer(DefaultBodyLimit::max(docs::cap())),
        ),
    )
}
"#;
        let routes = parse_routes(src).expect("fixture");
        assert_eq!(routes.calls, 1);
        let methods: Vec<&str> = routes.ops.iter().map(|o| o.method.as_str()).collect();
        assert_eq!(methods, vec!["GET", "POST"], "les deux verbes sont servis");
        assert_eq!(routes.ops[1].handler, "docs::upload");
        assert_eq!(routes.ops[1].tier, "protected");
    }

    #[test]
    fn parse_routes_cuts_at_the_first_cfg_test() {
        let (code, at) = cut_tests(ROUTES_FIXTURE);
        let line = at.expect("la fixture a un bloc de tests");
        assert_eq!(code.lines().count(), line - 1);
        let routes = parse_routes(ROUTES_FIXTURE).expect("fixture");
        assert!(
            routes.ops.iter().all(|o| o.path != "/only-in-tests"),
            "une route de test ne doit jamais entrer dans l'inventaire"
        );
    }

    #[test]
    fn parse_routes_refuses_a_nest_instead_of_guessing_the_prefix() {
        let src = r#"
fn protected_routes() -> Router<State> {
    Router::new().nest("/api", sub())
}
"#;
        let err = parse_routes(src).expect_err("`.nest(` doit être refusé");
        assert!(err.contains("nest"), "{err}");
    }

    #[test]
    fn parse_routes_refuses_an_unknown_route_group() {
        let src = r#"
fn misc_routes() -> Router<State> {
    Router::new().route("/api/misc", get(handlers::misc))
}
"#;
        let err = parse_routes(src).expect_err("un groupe inconnu doit être refusé");
        assert!(err.contains("misc_routes"), "{err}");
    }

    #[test]
    fn parse_routes_does_not_invent_a_route_from_a_comment_or_a_string() {
        let src = r#"
fn public_routes() -> Router<State> {
    // .route("/ghost", get(handlers::ghost))
    let doc = ".route(\"/quoted\", get(handlers::quoted))";
    Router::new().route("/health", get(handlers::health))
}
"#;
        let routes = parse_routes(src).expect("fixture");
        assert_eq!(routes.calls, 1);
        assert_eq!(routes.ops.len(), 1);
        assert_eq!(routes.ops[0].path, "/health");
        // …et le recomptage naïf, lui, voit les trois : c'est le garde-fou.
        assert_eq!(routes.naive_occurrences, 3);
        let err = build_inventory(src, HANDLERS_FIXTURE).expect_err("recomptage");
        assert!(err.contains("recomptage"), "{err}");
    }

    #[test]
    fn tier_for_fn_files_every_known_group_and_refuses_the_rest() {
        assert_eq!(tier_for_fn("public_routes"), Some("public"));
        assert_eq!(tier_for_fn("remote_mcp_public_routes"), Some("public"));
        assert_eq!(tier_for_fn("ingress_routes"), Some("public"));
        assert_eq!(tier_for_fn("bounded_public_routes"), Some("bounded_public"));
        assert_eq!(tier_for_fn("protected_routes"), Some("protected"));
        assert_eq!(
            tier_for_fn("remote_mcp_protected_routes"),
            Some("protected")
        );
        assert_eq!(tier_for_fn("websocket_routes"), Some("websocket"));
        assert_eq!(tier_for_fn("main"), None);
    }

    #[test]
    fn parse_mega_actions_reads_the_table_with_its_lines() {
        let actions = parse_mega_actions(HANDLERS_FIXTURE).expect("fixture");
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0].tool, "project");
        assert_eq!(actions[0].action, "list");
        assert_eq!(actions[0].internal, "list_projects");
        assert_eq!(actions[0].line, 6);
        assert_eq!(actions[2].tool, "plan");
        assert_eq!(actions[2].line, 10, "la ligne de la parenthèse ouvrante");
    }

    #[test]
    fn parse_mega_actions_refuses_a_duplicate_and_a_malformed_entry() {
        let dup = r#"
pub(crate) const MEGA_TOOL_ACTIONS: &[(&str, &str, &str)] = &[
    ("project", "list", "list_projects"),
    ("project", "list", "list_projects_again"),
];
"#;
        let err = parse_mega_actions(dup).expect_err("doublon");
        assert!(err.contains("déjà déclaré"), "{err}");

        let short = r#"
pub(crate) const MEGA_TOOL_ACTIONS: &[(&str, &str, &str)] = &[
    ("project", "list"),
];
"#;
        let err = parse_mega_actions(short).expect_err("entrée à deux littéraux");
        assert!(err.contains("au lieu de 3"), "{err}");

        let absent = "fn main() {}";
        let err = parse_mega_actions(absent).expect_err("table absente");
        assert!(err.contains("introuvable"), "{err}");
    }

    #[test]
    fn build_inventory_refuses_an_empty_inventory() {
        let no_routes = r#"
fn public_routes() -> Router<State> {
    Router::new()
}
"#;
        let err = build_inventory(no_routes, HANDLERS_FIXTURE).expect_err("aucune route");
        assert!(err.contains("aucune opération"), "{err}");

        let no_actions = r#"
pub(crate) const MEGA_TOOL_ACTIONS: &[(&str, &str, &str)] = &[];
"#;
        let err = build_inventory(ROUTES_FIXTURE, no_actions).expect_err("aucune action");
        assert!(err.contains("aucune action"), "{err}");
    }

    #[test]
    fn rows_are_sorted_and_rendered_as_tsv() {
        let inv = build_inventory(ROUTES_FIXTURE, HANDLERS_FIXTURE).expect("fixture");
        assert_eq!(inv.route_calls, 5);
        assert_eq!(inv.route_ops, 6);
        assert_eq!(inv.mcp_tools, 2);
        assert_eq!(inv.mcp_actions, 3);
        assert_eq!(inv.rows.len(), 9);

        // Tri : `M` avant `R`, puis par cible, action, fichier, ligne.
        let keys: Vec<(char, String, String)> = inv
            .rows
            .iter()
            .map(|r| (r.kind, r.target.clone(), r.action.clone()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "l'inventaire doit être trié");

        let health = inv
            .rows
            .iter()
            .find(|r| r.target == "/health")
            .map(Row::render)
            .expect("ligne /health");
        assert_eq!(
            health, "R\t/health\tGET\tsrc/api/routes.rs:4\ttier=public\th=handlers::health",
            "une opération par ligne, quatre colonnes fixes puis les extras"
        );
        let action = inv
            .rows
            .iter()
            .find(|r| r.kind == 'M')
            .map(Row::render)
            .expect("ligne M");
        assert_eq!(
            action,
            "M\tplan\tlist\tsrc/mcp/handlers.rs:10\tinternal=list_plans"
        );
    }

    #[test]
    fn render_is_deterministic_and_carries_the_counts() {
        let a = build_inventory(ROUTES_FIXTURE, HANDLERS_FIXTURE).expect("fixture");
        let b = build_inventory(ROUTES_FIXTURE, HANDLERS_FIXTURE).expect("fixture");
        assert_eq!(render(&a), render(&b), "même source, mêmes octets");
        let text = render(&a);
        assert!(text.starts_with("# GENERATED by `cargo xtask parity --write`"));
        assert!(
            text.contains("# counts routes=5 operations=6 mcp_tools=2 mcp_actions=3"),
            "{text}"
        );
        assert_eq!(data_rows(&text).len(), 9, "l'en-tête n'est pas une ligne");
    }

    #[test]
    fn diff_rows_sees_an_added_then_a_removed_route() {
        let generated =
            render(&build_inventory(ROUTES_FIXTURE, HANDLERS_FIXTURE).expect("fixture"));
        let without_ws: String = generated
            .lines()
            .filter(|l| !l.contains("\t/ws/feed\t"))
            .map(|l| format!("{l}\n"))
            .collect();

        let added = diff_rows(&without_ws, &generated);
        assert_eq!(added.added.len(), 1, "la route ajoutée est vue");
        assert!(added.added[0].contains("/ws/feed"), "{:?}", added.added);
        assert!(added.removed.is_empty());

        let removed = diff_rows(&generated, &without_ws);
        assert_eq!(removed.removed.len(), 1, "la route retirée est vue");
        assert!(removed.removed[0].contains("/ws/feed"));

        assert!(
            diff_rows(&generated, &generated).is_empty(),
            "inventaire identique : aucun écart"
        );
    }

    #[test]
    fn split_moves_tells_a_line_shift_from_a_contract_change() {
        let row = |line: usize| {
            format!("R\t/api/things\tGET\tsrc/api/routes.rs:{line}\ttier=public\th=things::list\n")
        };
        // Même contrat, une ligne plus bas : un déplacement, pas un écart.
        let drift = diff_rows(&row(10), &row(11));
        assert_eq!(drift.added.len(), 1, "octet à octet, la ligne a bien bougé");
        let (real, moved) = split_moves(&drift);
        assert_eq!(moved, 1);
        assert!(real.is_empty(), "aucun changement de contrat : {real:?}");

        // Contrat différent au même endroit : un vrai écart.
        let added = "R\t/api/things\tPOST\tsrc/api/routes.rs:10\ttier=public\th=things::create\n";
        let drift = diff_rows(&row(10), added);
        let (real, moved) = split_moves(&drift);
        assert_eq!(moved, 0);
        assert_eq!(real.added.len(), 1);
        assert_eq!(real.removed.len(), 1);
        assert!(real.added[0].contains("POST"));

        assert_eq!(
            row_without_location(&row(10)),
            "R\t/api/things\tGET\ttier=public\th=things::list\n"
        );
    }

    #[test]
    fn str_content_strips_the_delimiters_of_every_literal_form() {
        assert_eq!(str_content("\"/health\""), "/health");
        assert_eq!(str_content("r#\"/a\"b\"#"), "/a\"b");
        assert_eq!(str_content(r#"r"plain""#), "plain");
    }

    #[test]
    fn tokenize_ignores_comments_and_keeps_char_literals_apart() {
        let src =
            "let a: &'static str = \"x\"; // zzcomment\nlet c = '/'; let e = 1 / 2; /* bbblock */";
        let toks = tokenize(src);
        let idents: Vec<&str> = toks
            .iter()
            .filter(|t| t.kind == TokKind::Ident)
            .map(|t| t.text(src))
            .collect();
        assert!(idents.contains(&"static"), "{idents:?}");
        assert!(idents.contains(&"e"), "{idents:?}");
        assert!(
            !idents.contains(&"zzcomment") && !idents.contains(&"bbblock"),
            "les commentaires sont retirés : {idents:?}"
        );
        assert!(
            toks.iter().any(|t| t.is_punct('/')),
            "le caractère '/' reste un jeton"
        );
        let strs: Vec<&str> = toks
            .iter()
            .filter(|t| t.kind == TokKind::Str)
            .map(|t| t.text(src))
            .collect();
        assert_eq!(strs, vec!["\"x\""]);
    }

    #[test]
    fn parse_args_defaults_to_check_mode() {
        let o = parse_args(&[]).expect("défaut");
        assert_eq!(o.inventory, PathBuf::from(DEFAULT_INVENTORY));
        assert!(!o.write, "sans --write on compare, on n'écrit pas");

        let o = parse_args(&[
            "inventory".to_string(),
            "--inventory".to_string(),
            "x.tsv".to_string(),
            "--write".to_string(),
        ])
        .expect("écriture");
        assert!(o.write);
        assert_eq!(o.inventory, PathBuf::from("x.tsv"));

        assert!(parse_args(&["--check".into(), "--write".into()]).is_err());
        assert!(parse_args(&["--nope".into()]).is_err());
        assert!(parse_args(&["verify".into()]).is_err());
    }
}
