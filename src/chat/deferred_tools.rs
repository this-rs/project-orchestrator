//! The tools whose definitions the native harness DEFERS (CTX2).
//!
//! Every request to a model of the native harness used to carry the full definition of
//! every tool the session has: the PO mega-tools (`mcp__project-orchestrator__*`, 32 of
//! them with their JSON schemas) and the `nexus-tools` (Read, Edit, Bash...). Thousands of
//! tokens, re-sent at every turn, and a long list for a small model to pick from.
//!
//! nexus (#116) lets a session defer tools: their definitions are withheld and the model is
//! offered `ToolSearch` instead (same name and shape as Claude Code's). A tool it loads is
//! sent with every later request, APPENDED after the tools already sent, so the cached
//! prefix never moves (CTX0); a deferred tool called before it is loaded gets an explicit
//! error, never a silent run. This module decides WHICH tools are deferred:
//!
//! - a default list ([`default_deferred_tools`]): the PO mega-tools an ordinary turn does
//!   not use. The everyday ones ([`ALWAYS_LOADED`]) and every `nexus-tools` tool stay
//!   loaded;
//! - one document, `LlmSetting {scope: "global", key: "deferred_tools"}`, that an operator
//!   replaces WHILE the server runs (`PUT /api/chat/deferred-tools`), read at startup into
//!   the [`ChatManager`]. An EMPTY stored list defers nothing (the switch off); no document
//!   means the default list.
//!
//! Rules that keep the setting safe:
//! - Only the root account writes it, never an agent: what a model is offered is a
//!   server-wide decision, as the delegation cap is (PLAFOND-RT).
//! - A pattern is an exact tool name or a prefix ending in `*`, never `*` alone; the list is
//!   bounded ([`MAX_PATTERNS`]); an unreadable or invalid stored document is ignored and
//!   the default holds, with the reason logged.
//! - The list of a SESSION is fixed when its provider is built: a change applies to the
//!   sessions opened after it (the native provider is rebuilt, see
//!   `ChatManager::provider_for`), never to a running conversation — its cached prefix
//!   would break.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::manager::ChatManager;
use super::provider::settings::GLOBAL;
use crate::neo4j::AnalyticsStore;
use crate::neo4j::GraphStore;

/// Key of the stored document, under the [`GLOBAL`] scope.
pub const DEFERRED_TOOLS_KEY: &str = "deferred_tools";

/// Most patterns a list may hold.
pub const MAX_PATTERNS: usize = 256;

/// Longest pattern.
pub const MAX_PATTERN_LEN: usize = 128;

/// How the PO MCP server names its tools on the native engine.
pub const PO_TOOL_PREFIX: &str = "mcp__project-orchestrator__";

/// The PO mega-tools an ordinary turn uses: never deferred by default. The rest of the
/// mega-tools are, and the model reaches them through `ToolSearch` (the system prompt's
/// reference names them all, so it knows what to ask for).
pub const ALWAYS_LOADED: &[&str] = &[
    "project", "plan", "task", "step", "decision", "code", "note", "chat", "commit",
];

/// The PO mega-tools deferred by default (everything `all_tools` has that is not in
/// [`ALWAYS_LOADED`]), as full `mcp__project-orchestrator__<name>` names.
pub fn default_deferred_tools() -> Vec<String> {
    crate::mcp::tools::all_tools()
        .into_iter()
        .map(|t| t.name)
        .filter(|name| !ALWAYS_LOADED.contains(&name.as_str()))
        .map(|name| format!("{PO_TOOL_PREFIX}{name}"))
        .collect()
}

/// The stored document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeferredTools {
    /// Exact tool names or prefixes ending in `*`. Empty = defer nothing.
    pub tools: Vec<String>,
}

/// Why a list is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidPattern {
    /// More than [`MAX_PATTERNS`] entries.
    TooMany(usize),
    /// An empty, too long or ill-formed entry (not `[A-Za-z0-9_][A-Za-z0-9_.-]*` with an
    /// optional trailing `*`), or `*` alone.
    Malformed(String),
}

impl std::fmt::Display for InvalidPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooMany(n) => write!(f, "{n} patterns, at most {MAX_PATTERNS} are allowed"),
            Self::Malformed(p) => write!(
                f,
                "{p:?} is not a tool name or a prefix ending in `*` (letters, digits, `_`, `.`, `-`; \
                 at most {MAX_PATTERN_LEN} characters; `*` alone is refused)"
            ),
        }
    }
}

/// The list, trimmed and deduplicated in order, when every entry is a valid pattern.
pub fn settle(tools: &[String]) -> Result<Vec<String>, InvalidPattern> {
    let mut out: Vec<String> = Vec::with_capacity(tools.len());
    for raw in tools {
        let p = raw.trim();
        let body = p.strip_suffix('*').unwrap_or(p);
        let well_formed = !body.is_empty()
            && p.len() <= MAX_PATTERN_LEN
            && body.chars().enumerate().all(|(i, c)| {
                c.is_ascii_alphanumeric() || c == '_' || (i > 0 && (c == '.' || c == '-'))
            });
        if !well_formed {
            return Err(InvalidPattern::Malformed(raw.clone()));
        }
        if !out.iter().any(|o| o == p) {
            out.push(p.to_string());
        }
    }
    if out.len() > MAX_PATTERNS {
        return Err(InvalidPattern::TooMany(out.len()));
    }
    Ok(out)
}

/// Whether `name` is deferred by `patterns` — the rule nexus applies
/// (`providers::native::tools::is_deferred`): exact name, or prefix when the pattern ends
/// in `*`. Kept here so the default list can be checked against the real tool names.
pub fn is_deferred(name: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|pattern| match pattern.strip_suffix('*') {
            Some(prefix) => name.starts_with(prefix),
            None => name == pattern,
        })
}

/// The stored list; `None` when nothing is stored or the document is unusable (the default
/// then applies, and the reason is logged). `Some(empty)` is a real value: defer nothing.
pub async fn load(graph: &dyn GraphStore) -> anyhow::Result<Option<Vec<String>>> {
    let Some(raw) = graph.get_llm_setting(GLOBAL, DEFERRED_TOOLS_KEY).await? else {
        return Ok(None);
    };
    match serde_json::from_str::<DeferredTools>(&raw) {
        Ok(doc) => match settle(&doc.tools) {
            Ok(tools) => Ok(Some(tools)),
            Err(error) => {
                tracing::warn!(%error, "stored deferred_tools invalid, ignored: the default list holds");
                Ok(None)
            }
        },
        Err(error) => {
            tracing::warn!(%error, "unreadable deferred_tools document ignored: the default list holds");
            Ok(None)
        }
    }
}

/// Replaces the stored list. The caller has already [`settle`]d it.
pub async fn store(graph: &dyn GraphStore, tools: &[String]) -> anyhow::Result<()> {
    let document = serde_json::to_string(&DeferredTools {
        tools: tools.to_vec(),
    })?;
    graph
        .put_llm_setting(GLOBAL, DEFERRED_TOOLS_KEY, &document)
        .await
}

/// Removes the stored list: the default applies again.
pub async fn clear(graph: &dyn GraphStore) -> anyhow::Result<()> {
    graph.delete_llm_setting(GLOBAL, DEFERRED_TOOLS_KEY).await?;
    Ok(())
}

impl ChatManager {
    /// The list in force NOW for the native providers built from here on. A session keeps
    /// the list its provider was built with.
    pub fn deferred_tools(&self) -> Arc<Vec<String>> {
        Arc::clone(
            &self
                .deferred_tools
                .read()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    fn replace_deferred_tools(&self, tools: Vec<String>) {
        *self
            .deferred_tools
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Arc::new(tools);
    }

    /// Reads the stored list into the running manager (startup). A failing read keeps the
    /// default and says so; it never blocks the server.
    pub async fn load_deferred_tools(&self) {
        match load(self.graph.as_ref()).await {
            Ok(Some(tools)) => self.replace_deferred_tools(tools),
            Ok(None) => {}
            Err(error) => tracing::warn!(
                %error,
                "could not read deferred_tools: the default list holds"
            ),
        }
    }

    /// Sets the list for the providers built next and stores it for the next start.
    /// Stored FIRST: a failed write leaves the running value as it was, so memory and graph
    /// never disagree. `Err(reason)` when the list is invalid.
    pub async fn set_deferred_tools(
        &self,
        tools: &[String],
    ) -> Result<anyhow::Result<Vec<String>>, InvalidPattern> {
        let tools = settle(tools)?;
        Ok(match store(self.graph.as_ref(), &tools).await {
            Ok(()) => {
                self.replace_deferred_tools(tools.clone());
                Ok(tools)
            }
            Err(error) => Err(error),
        })
    }

    /// Removes the stored list and goes back to the default; returns it.
    pub async fn reset_deferred_tools(&self) -> anyhow::Result<Vec<String>> {
        clear(self.graph.as_ref()).await?;
        let tools = default_deferred_tools();
        self.replace_deferred_tools(tools.clone());
        Ok(tools)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tools::all_tools;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::{
        AnalyticsStore, CodeStore, CommitStore, DecisionStore, NoteStore, PlanStore, ProjectStore,
        ReleaseStore, WorkspaceStore,
    };

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// The default list names only real mega-tools, and none of the everyday ones: a
    /// renamed or removed mega-tool breaks this test instead of silently deferring nothing.
    #[test]
    fn the_default_defers_real_tools_and_keeps_the_everyday_ones_loaded() {
        let all: Vec<String> = all_tools()
            .into_iter()
            .map(|t| format!("{PO_TOOL_PREFIX}{}", t.name))
            .collect();
        let deferred = default_deferred_tools();
        assert!(!deferred.is_empty());
        for d in &deferred {
            assert!(all.contains(d), "{d} is not a mega-tool");
        }
        for name in ALWAYS_LOADED {
            let full = format!("{PO_TOOL_PREFIX}{name}");
            assert!(all.contains(&full), "{name} is not a mega-tool any more");
            assert!(!is_deferred(&full, &deferred), "{name} must stay loaded");
        }
        // No nexus tool is deferred by the default.
        for tool in [
            "mcp__nexus__Read",
            "mcp__nexus__Bash",
            "Read",
            "Agent",
            "ToolSearch",
        ] {
            assert!(!is_deferred(tool, &deferred), "{tool}");
        }
        assert_eq!(deferred.len(), all.len() - ALWAYS_LOADED.len());
        settle(&deferred).expect("the default list is valid");
    }

    /// STATIC MEASURE of what the default withholds from every request: the bytes of the
    /// JSON definitions (name, description, schema) of the deferred mega-tools against all
    /// of them. Printed for the record (`--nocapture`); the live figure, with the cache,
    /// is CTX0's to read on a real session.
    #[test]
    fn the_default_withholds_most_of_the_mega_tool_definitions() {
        let deferred = default_deferred_tools();
        let (mut all_bytes, mut deferred_bytes, mut kept) = (0usize, 0usize, Vec::new());
        for tool in all_tools() {
            let bytes = serde_json::to_string(&tool).unwrap().len();
            all_bytes += bytes;
            if is_deferred(&format!("{PO_TOOL_PREFIX}{}", tool.name), &deferred) {
                deferred_bytes += bytes;
            } else {
                kept.push(tool.name);
            }
        }
        println!(
            "mega-tool definitions: {all_bytes} bytes in all, {deferred_bytes} withheld by the \
             default ({:.0}%), {} bytes kept for {kept:?}",
            100.0 * deferred_bytes as f64 / all_bytes as f64,
            all_bytes - deferred_bytes
        );
        assert!(
            deferred_bytes * 2 >= all_bytes,
            "the default must withhold at least half: {deferred_bytes}/{all_bytes}"
        );
    }

    #[test]
    fn only_names_and_prefixes_are_accepted_deduplicated_in_order() {
        assert_eq!(
            settle(&s(&[
                " mcp__po__admin ",
                "mcp__po__*",
                "mcp__po__admin",
                "Read"
            ]))
            .unwrap(),
            s(&["mcp__po__admin", "mcp__po__*", "Read"])
        );
        for bad in [
            "", " ", "*", "**", "a b", "a*b", "-lead", ".lead", "a/b", "é",
        ] {
            assert!(settle(&s(&[bad])).is_err(), "{bad:?}");
        }
        let long = "a".repeat(MAX_PATTERN_LEN + 1);
        assert!(settle(&[long]).is_err());
        let many: Vec<String> = (0..=MAX_PATTERNS).map(|i| format!("t{i}")).collect();
        assert_eq!(
            settle(&many),
            Err(InvalidPattern::TooMany(MAX_PATTERNS + 1))
        );
        assert_eq!(settle(&[]).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn the_matching_rule_is_the_one_nexus_applies() {
        let p = s(&["mcp__po__admin", "mcp__nexus__Web*"]);
        assert!(is_deferred("mcp__po__admin", &p));
        assert!(!is_deferred("mcp__po__admin2", &p));
        assert!(is_deferred("mcp__nexus__WebFetch", &p));
        assert!(!is_deferred("mcp__nexus__Read", &p));
    }

    #[tokio::test]
    async fn the_document_round_trips_an_empty_list_is_off_and_clear_restores_the_default() {
        let graph = MockGraphStore::new();
        assert_eq!(
            load(&graph).await.unwrap(),
            None,
            "nothing stored = default"
        );
        store(&graph, &s(&["mcp__po__admin"])).await.unwrap();
        assert_eq!(load(&graph).await.unwrap(), Some(s(&["mcp__po__admin"])));
        store(&graph, &[]).await.unwrap();
        assert_eq!(
            load(&graph).await.unwrap(),
            Some(vec![]),
            "empty = defer nothing"
        );
        clear(&graph).await.unwrap();
        assert_eq!(load(&graph).await.unwrap(), None);
    }

    #[tokio::test]
    async fn an_unusable_stored_document_leaves_the_default() {
        let graph = MockGraphStore::new();
        graph
            .put_llm_setting(GLOBAL, DEFERRED_TOOLS_KEY, "not json")
            .await
            .unwrap();
        assert_eq!(load(&graph).await.unwrap(), None);
        graph
            .put_llm_setting(GLOBAL, DEFERRED_TOOLS_KEY, r#"{"tools":["*"]}"#)
            .await
            .unwrap();
        assert_eq!(load(&graph).await.unwrap(), None, "`*` alone is not a list");
    }
}
