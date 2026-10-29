//! PostToolUse Hook — Feedback intelligent après exécution d'un tool
//!
//! Analyse le résultat d'un tool après exécution et injecte des suggestions
//! contextuelles :
//! - Grep bruité (> 20 matches) + symbole → suggestion MCP find_references
//! - Edit/Write sur fichier bridge → alerte co-changers
//!
//! # Error Policy
//!
//! **NEVER block tool execution.** Any error → passthrough (continue_: true).

use crate::neo4j::traits::GraphStore;
use crate::neo4j::{
    AnalyticsStore, CodeStore, CommitStore, DecisionStore, NoteStore, PlanStore, ProjectStore,
    ReleaseStore, WorkspaceStore,
};
use crate::neurons::AutoReinforcementConfig;
use crate::skills::hook_extractor::{
    enrich_redirect_with_context_card, extract_file_context, generate_redirect_suggestion,
    EnrichedRedirectSuggestion,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use tracing::debug;

/// Throttle TTL: don't suggest redirects for the same file within this window.
const THROTTLE_TTL_SECS: u64 = 60;

/// Minimum result lines to consider a Grep result "noisy" enough for redirect.
const MIN_NOISY_LINES: usize = 20;

/// PostToolUse hook that suggests MCP alternatives after noisy Grep results
/// and alerts about co-changers after Edit/Write on bridge files.
pub(crate) struct PostToolUseRedirectHook {
    graph_store: Arc<dyn GraphStore>,
    /// Per-file throttle to avoid spamming suggestions.
    throttle: Mutex<HashMap<String, Instant>>,
    /// The resolved project of the session (mode `on`); `None`: historical resolution.
    session_project: Option<Arc<super::anchor_resolver::SessionProject>>,
    /// The session's injection ledger, in place of the time throttle: an advice is
    /// given once, again after `REINJECT_AFTER_CALLS` tool calls or a compaction
    /// (`HookLedger::reset`). Set by the host's fallback for a provider without hooks
    /// (`provider::hooks`, B14); `None`: the 60 s throttle.
    ledger: Option<Arc<super::hook_ledger::HookLedger>>,
}

impl PostToolUseRedirectHook {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self {
            graph_store,
            throttle: Mutex::new(HashMap::new()),
            session_project: None,
            ledger: None,
        }
    }

    /// Deduplicate by the session's ledger instead of the time throttle.
    pub fn with_ledger(mut self, ledger: Arc<super::hook_ledger::HookLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// Work for the project the session resolved instead of the tool cwd's.
    pub fn with_session_project(
        mut self,
        session_project: Arc<super::anchor_resolver::SessionProject>,
    ) -> Self {
        self.session_project = Some(session_project);
        self
    }

    fn passthrough() -> nexus_claude::HookJSONOutput {
        nexus_claude::HookJSONOutput::Sync(nexus_claude::SyncHookJSONOutput {
            continue_: Some(true),
            ..Default::default()
        })
    }

    /// Check if a file is throttled (recently suggested).
    fn is_throttled(&self, key: &str) -> bool {
        let mut throttle = self.throttle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(last) = throttle.get(key) {
            if last.elapsed().as_secs() < THROTTLE_TTL_SECS {
                return true;
            }
        }
        // Clean old entries while we're at it (cheap, bounded by unique files)
        throttle.retain(|_, v| v.elapsed().as_secs() < THROTTLE_TTL_SECS * 2);
        throttle.insert(key.to_string(), Instant::now());
        false
    }

    /// Count approximate lines in a tool result (simple newline count).
    fn count_result_lines(result: &str) -> usize {
        result.lines().count()
    }

    /// Build a co-changer alert for a file that was just modified.
    async fn build_edit_alert(&self, file_path: &str, project_id: uuid::Uuid) -> Option<String> {
        let project_id_str = project_id.to_string();
        let card = match self
            .graph_store
            .get_context_card(file_path, &project_id_str)
            .await
        {
            Ok(Some(c)) => c,
            _ => return None,
        };

        let mut alerts = Vec::new();

        // Bridge alert
        if card.cc_betweenness > 0.5 {
            alerts.push(format!(
                "🌉 **Bridge file modified** (betweenness: {:.2}) — this file connects multiple clusters. Use `code(action: \"analyze_impact\", target: \"{}\")` to check cascading effects.",
                card.cc_betweenness, file_path
            ));
        }

        // Co-changers alert
        if !card.cc_co_changers_top5.is_empty() {
            let changers: Vec<&str> = card
                .cc_co_changers_top5
                .iter()
                .take(3)
                .map(|s| s.as_str())
                .collect();
            alerts.push(format!(
                "🔗 **Co-changers**: you modified `{}` — these files often change together: {}. Consider reviewing them.",
                file_path.rsplit('/').next().unwrap_or(file_path),
                changers.join(", ")
            ));
        }

        if alerts.is_empty() {
            None
        } else {
            // Fire-and-forget: reinforce notes linked to this file (Hebbian feedback)
            self.spawn_file_reinforcement(file_path, project_id);

            Some(format!(
                "## 📝 Post-Edit Intelligence\n{}",
                alerts.join("\n")
            ))
        }
    }

    /// Fire-and-forget Hebbian reinforcement for notes linked to a file.
    /// Fetches note IDs then reinforces synapses between co-surfaced notes.
    fn spawn_file_reinforcement(&self, file_path: &str, project_id: uuid::Uuid) {
        let graph = self.graph_store.clone();
        let path = file_path.to_string();
        tokio::spawn(async move {
            reinforce_file_notes(graph.as_ref(), &path, project_id).await;
        });
    }
}

/// Hebbian reinforcement of the notes anchored on a file the agent touched.
///
/// Only the notes of the session's project: the same path can anchor notes of
/// other projects, which this session must not warm.
async fn reinforce_file_notes(graph: &dyn GraphStore, path: &str, project_id: uuid::Uuid) {
    let config = AutoReinforcementConfig::default();
    match crate::notes::coactivation::project_notes_for_entity(
        graph,
        &crate::notes::EntityType::File,
        path,
        project_id,
    )
    .await
    {
        Ok(notes) if notes.len() >= 2 => {
            if !config.enabled {
                return;
            }
            let note_ids: Vec<uuid::Uuid> = notes.iter().map(|n| n.id).collect();
            if let Err(e) =
                crate::skills::activation::reinforce_hook_activation(graph, &note_ids, &config)
                    .await
            {
                debug!(path = %path, error = %e, "PostToolUse file reinforcement failed");
            }
        }
        Ok(notes) if notes.len() == 1 => {
            // Single note: just boost energy, no synapse reinforcement needed
            let _ = graph
                .boost_energy(notes[0].id, config.hook_energy_boost)
                .await;
        }
        _ => {} // No notes or error → skip silently
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for PostToolUseRedirectHook {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        // Only process PostToolUse events
        let post_tool = match input {
            nexus_claude::HookInput::PostToolUse(post) => post,
            _ => return Ok(Self::passthrough()),
        };

        // Throttle check using tool_name + file context as key
        let throttle_key = format!(
            "{}:{}",
            post_tool.tool_name,
            extract_file_context(&post_tool.tool_name, &post_tool.tool_input).unwrap_or_default()
        );
        let ledger_key = super::hook_ledger::InjectionKey::Redirect(throttle_key.clone());
        let held_back = match &self.ledger {
            Some(ledger) => !ledger.is_fresh(&ledger_key),
            None => self.is_throttled(&throttle_key),
        };
        if held_back {
            return Ok(Self::passthrough());
        }

        // Resolve project
        let project_id = match super::anchor_resolver::hook_project(
            self.session_project.as_deref(),
            &*self.graph_store,
            &post_tool.tool_name,
            &post_tool.tool_input,
            &post_tool.cwd,
        )
        .await
        {
            Ok(Some(id)) => id,
            _ => return Ok(Self::passthrough()),
        };

        let mut context_parts: Vec<String> = Vec::new();

        match post_tool.tool_name.as_str() {
            // Grep/Bash: if noisy result + symbol pattern → suggest MCP
            "Grep" | "Bash" => {
                let result_text = post_tool.tool_response.as_str().unwrap_or("");
                let line_count = Self::count_result_lines(result_text);

                if line_count >= MIN_NOISY_LINES {
                    if let Some(suggestion) =
                        generate_redirect_suggestion(&post_tool.tool_name, &post_tool.tool_input)
                    {
                        // Try to enrich with ContextCard
                        let file_path =
                            extract_file_context(&post_tool.tool_name, &post_tool.tool_input);
                        let enriched = if let Some(ref fp) = file_path {
                            let pid = project_id.to_string();
                            match self.graph_store.get_context_card(fp, &pid).await {
                                Ok(Some(card)) => {
                                    enrich_redirect_with_context_card(suggestion, &card)
                                }
                                _ => EnrichedRedirectSuggestion {
                                    suggestion,
                                    context_warnings: vec![],
                                },
                            }
                        } else {
                            EnrichedRedirectSuggestion {
                                suggestion,
                                context_warnings: vec![],
                            }
                        };

                        context_parts.push(format!(
                            "## ⚡ Post-Tool Redirect (result had {} lines)\n{}",
                            line_count, enriched
                        ));

                        // Fire-and-forget: reinforce notes linked to the file being searched
                        if let Some(ref fp) = file_path {
                            self.spawn_file_reinforcement(fp, project_id);
                        }

                        debug!(
                            tool = %post_tool.tool_name,
                            lines = line_count,
                            "PostToolUse redirect suggestion for noisy result"
                        );
                    }
                }
            }

            // Edit/Write: alert about co-changers and bridge status
            "Edit" | "Write" => {
                if let Some(fp) = extract_file_context(&post_tool.tool_name, &post_tool.tool_input)
                {
                    if let Some(alert) = self.build_edit_alert(&fp, project_id).await {
                        context_parts.push(alert);
                    }
                }
            }

            _ => {}
        }

        if context_parts.is_empty() {
            return Ok(Self::passthrough());
        }

        if let Some(ledger) = &self.ledger {
            ledger.record(ledger_key);
        }
        let combined = context_parts.join("\n\n");
        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                hook_specific_output: Some(nexus_claude::HookSpecificOutput::PostToolUse(
                    nexus_claude::PostToolUseHookSpecificOutput {
                        additional_context: Some(combined),
                    },
                )),
                ..Default::default()
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::{
        AnalyticsStore, CodeStore, CommitStore, DecisionStore, NoteStore, PlanStore, ProjectStore,
        ReleaseStore, WorkspaceStore,
    };
    use nexus_claude::{HookCallback, HookContext};

    fn test_context() -> HookContext {
        HookContext { signal: None }
    }

    /// T0f-bis: a touched file warms the notes of the session's project only.
    #[tokio::test]
    async fn file_reinforcement_touches_only_the_session_project() {
        let mock = MockGraphStore::new();
        let (mine, other) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let mut ids = Vec::new();
        for owner in [Some(mine), Some(mine), Some(other), Some(other), None] {
            let mut n = crate::notes::Note::new(
                owner,
                crate::notes::NoteType::Guideline,
                "n".into(),
                "t".into(),
            );
            n.energy = 0.3;
            n.last_activated = None;
            mock.create_note(&n).await.unwrap();
            mock.link_note_to_entity(
                n.id,
                &crate::notes::EntityType::File,
                "src/x.rs",
                None,
                None,
            )
            .await
            .unwrap();
            ids.push(n.id);
        }
        reinforce_file_notes(&mock, "src/x.rs", mine).await;
        for id in &ids[..2] {
            assert!(mock.get_note(*id).await.unwrap().unwrap().energy > 0.3);
        }
        for id in &ids[2..] {
            let n = mock.get_note(*id).await.unwrap().unwrap();
            assert!(
                (n.energy - 0.3).abs() < 1e-9,
                "foreign/global note untouched"
            );
            assert!(n.last_activated.is_none());
            assert!(mock.get_synapses(*id).await.unwrap().is_empty());
        }

        // One own note + foreign ones: the lone own note alone is boosted.
        let lone = uuid::Uuid::new_v4();
        let mut n = crate::notes::Note::new(
            Some(lone),
            crate::notes::NoteType::Guideline,
            "n".into(),
            "t".into(),
        );
        n.energy = 0.3;
        mock.create_note(&n).await.unwrap();
        mock.link_note_to_entity(
            n.id,
            &crate::notes::EntityType::File,
            "src/x.rs",
            None,
            None,
        )
        .await
        .unwrap();
        reinforce_file_notes(&mock, "src/x.rs", lone).await;
        assert!(mock.get_note(n.id).await.unwrap().unwrap().energy > 0.3);
        let f = mock.get_note(ids[2]).await.unwrap().unwrap();
        assert!((f.energy - 0.3).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_non_post_tool_returns_passthrough() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = PostToolUseRedirectHook::new(mock_store);

        let input = nexus_claude::HookInput::PreToolUse(nexus_claude::PreToolUseHookInput {
            // Main thread, not a sub-agent — the CLI omits both fields there.
            agent_id: None,
            agent_type: None,
            session_id: "test".to_string(),
            transcript_path: "/tmp/t".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Grep".to_string(),
            tool_input: serde_json::json!({}),
        });

        let result = hook.execute(&input, None, &test_context()).await.unwrap();
        match result {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(sync.continue_, Some(true));
                assert!(sync.hook_specific_output.is_none());
            }
            _ => panic!("Expected Sync output"),
        }
    }

    #[test]
    fn test_throttle_works() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = PostToolUseRedirectHook::new(mock_store);

        assert!(!hook.is_throttled("test_key"));
        assert!(hook.is_throttled("test_key")); // second call within TTL → throttled
        assert!(!hook.is_throttled("other_key")); // different key → not throttled
    }

    #[test]
    fn test_count_result_lines() {
        assert_eq!(PostToolUseRedirectHook::count_result_lines(""), 0);
        assert_eq!(
            PostToolUseRedirectHook::count_result_lines("one\ntwo\nthree"),
            3
        );
        let many = (0..30)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(PostToolUseRedirectHook::count_result_lines(&many), 30);
    }
}
