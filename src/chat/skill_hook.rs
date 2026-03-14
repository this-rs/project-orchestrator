//! SkillActivationHook — in-process PreToolUse hook for Neural Skills
//!
//! Intercepts every PreToolUse event from the Claude Code CLI, resolves the
//! project from the tool context (file path or cwd), activates matching skills,
//! and returns the skill context as `additionalContext` in the hook response.
//!
//! This replaces the old external hook flow (.po-config + pre-tool-use.cjs)
//! with a zero-config, in-process solution that is:
//! - ~100x faster (~1-5ms vs ~300ms HTTP round-trip)
//! - Zero-config (no .po-config, no settings.json, no hook scripts)
//! - Lock-free (runs inside the SDK's hook dispatch, no Mutex contention)
//!
//! # Error Policy
//!
//! **NEVER block tool execution.** Any error → passthrough (continue_: true).
//! Skills are a nice-to-have context injection, not a gating mechanism.

use crate::api::hook_handlers::skill_cache;
use crate::neo4j::traits::GraphStore;
use crate::neurons::AutoReinforcementConfig;
use crate::skills::activation::{
    activate_for_hook_cached, spawn_activation_increment, spawn_hook_reinforcement,
    HookActivationConfig,
};
use crate::skills::hook_extractor::{
    enrich_redirect_with_context_card, extract_file_context, generate_redirect_suggestion,
};
use crate::skills::project_resolver::resolve_project_from_context;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Source of a persona-file match (for distinguishing fallback levels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersonaMatchSource {
    /// Direct KNOWS relation between persona and file.
    DirectKnows,
    /// Community-based match via SCOPED_TO FeatureGraph (lower confidence).
    CommunityMatch,
    /// Directory-prefix match (persona KNOWS ≥ 2 files in the same directory).
    DirectoryPrefix,
}

/// In-memory cache entry for persona file index.
/// Maps file_path → Vec<(persona_id, persona_name, weight, source)>.
struct PersonaFileIndex {
    entries: HashMap<String, Vec<(Uuid, String, f64, PersonaMatchSource)>>,
    /// Directory index: dir_path → Vec<(persona_id, persona_name, avg_weight, file_count)>.
    /// Built from KNOWS entries for directory-prefix fallback.
    dir_index: HashMap<String, Vec<(Uuid, String, f64, usize)>>,
    loaded_at: std::time::Instant,
}

/// TTL for persona file index cache (2 minutes).
const PERSONA_INDEX_TTL: std::time::Duration = std::time::Duration::from_secs(120);

/// In-process hook callback that activates Neural Skills on every PreToolUse event.
///
/// Registered in `create_session()` and `resume_session()` alongside `CompactionNotifier`.
/// Uses the same `SkillCache` and activation pipeline as the REST endpoint.
/// Also matches file access against persona KNOWS relations for persona context injection.
pub(crate) struct SkillActivationHook {
    /// Access to Neo4j graph store for project resolution + skill loading.
    graph_store: Arc<dyn GraphStore>,
    /// Per-project persona file index cache: project_id → PersonaFileIndex.
    persona_index: RwLock<HashMap<Uuid, PersonaFileIndex>>,
}

impl SkillActivationHook {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self {
            graph_store,
            persona_index: RwLock::new(HashMap::new()),
        }
    }

    /// Passthrough response — continue without injecting context.
    fn passthrough() -> nexus_claude::HookJSONOutput {
        nexus_claude::HookJSONOutput::Sync(nexus_claude::SyncHookJSONOutput {
            continue_: Some(true),
            ..Default::default()
        })
    }

    /// Extract the parent directory from a file path (with trailing slash).
    /// Returns None for root-level files without a directory.
    fn parent_dir(file_path: &str) -> Option<String> {
        file_path
            .rsplit_once('/')
            .map(|(dir, _)| format!("{dir}/"))
    }

    /// Directory-prefix match: check if the persona KNOWS ≥ threshold files in the same directory.
    /// Returns (persona_id, persona_name, weight) where weight = avg(KNOWS weights in dir) * 0.7.
    fn directory_prefix_match(
        index: &PersonaFileIndex,
        file_path: &str,
        threshold: usize,
    ) -> Option<(Uuid, String, f64)> {
        let dir = Self::parent_dir(file_path)?;
        let dir_matches = index.dir_index.get(&dir)?;

        // Find the first persona with file_count >= threshold
        dir_matches
            .iter()
            .find(|(_id, _name, _avg_weight, count)| *count >= threshold)
            .map(|(id, name, avg_weight, _count)| (*id, name.clone(), *avg_weight * 0.7))
    }

    /// Match a file path against the persona file index.
    /// Returns the best matching persona (highest weight) if found.
    ///
    /// Fallback chain:
    /// 1. Exact KNOWS match (direct persona-file relation)
    /// 2. Directory-prefix match (persona KNOWS ≥ 2 files in same dir, weight = avg * 0.7)
    /// 3. Community match (SCOPED_TO FeatureGraph, weight 0.3)
    ///
    /// All levels are cached in PersonaFileIndex with their source tag.
    async fn match_persona_for_file(
        &self,
        project_id: Uuid,
        file_path: &str,
    ) -> Option<(Uuid, String, f64)> {
        // Check cache first (read lock)
        {
            let cache = self.persona_index.read().await;
            if let Some(index) = cache.get(&project_id) {
                if index.loaded_at.elapsed() < PERSONA_INDEX_TTL {
                    // Try exact match from cache
                    if let Some(result) = index
                        .entries
                        .get(file_path)
                        .and_then(|v| v.first())
                        .map(|(id, name, w, _source)| (*id, name.clone(), *w))
                    {
                        return Some(result);
                    }
                    // Try directory-prefix fallback from cache
                    if let Some(result) = Self::directory_prefix_match(index, file_path, 2) {
                        return Some(result);
                    }
                    // No cache hit at any level
                    return None;
                }
            }
        }

        // Cache miss or expired — load from Neo4j
        // find_personas_for_file already includes UNION with SCOPED_TO FeatureGraph
        // (community match at weight 0.3). We tag each result by its source:
        // - weight > 0.3 → DirectKnows (came from the KNOWS branch)
        // - weight == 0.3 → CommunityMatch (came from the SCOPED_TO branch)
        match self
            .graph_store
            .find_personas_for_file(file_path, project_id)
            .await
        {
            Ok(matches) if !matches.is_empty() => {
                let first = matches.first().map(|(p, w)| (p.id, p.name.clone(), *w));

                // Populate cache (write lock) — tag each entry with its source
                {
                    let mut cache = self.persona_index.write().await;
                    let index = cache.entry(project_id).or_insert_with(|| PersonaFileIndex {
                        entries: HashMap::new(),
                        dir_index: HashMap::new(),
                        loaded_at: std::time::Instant::now(),
                    });
                    index.entries.insert(
                        file_path.to_string(),
                        matches
                            .iter()
                            .map(|(p, w)| {
                                let source = if (*w - 0.3).abs() < f64::EPSILON {
                                    PersonaMatchSource::CommunityMatch
                                } else {
                                    PersonaMatchSource::DirectKnows
                                };
                                (p.id, p.name.clone(), *w, source)
                            })
                            .collect(),
                    );

                    // Also populate dir_index: aggregate KNOWS files by directory
                    // for directory-prefix fallback
                    Self::update_dir_index(index, file_path, &matches);
                }

                first
            }
            Ok(_) => {
                // No exact/community match — try directory-prefix from existing cache
                let cache = self.persona_index.read().await;
                if let Some(index) = cache.get(&project_id) {
                    return Self::directory_prefix_match(index, file_path, 2);
                }
                None
            }
            Err(e) => {
                debug!(
                    file_path = %file_path,
                    error = %e,
                    "Persona file match failed"
                );
                None
            }
        }
    }

    /// Update the directory index in PersonaFileIndex from KNOWS entries.
    /// Aggregates persona presence per directory for directory-prefix fallback.
    fn update_dir_index(
        index: &mut PersonaFileIndex,
        _current_file: &str,
        _matches: &[(crate::neo4j::models::PersonaNode, f64)],
    ) {
        // Rebuild dir_index from all cached entries (not just current file)
        // This ensures the dir_index reflects the full set of cached KNOWS relations
        let mut dir_persona_data: HashMap<String, HashMap<Uuid, (String, Vec<f64>)>> =
            HashMap::new();

        for (path, personas) in &index.entries {
            if let Some(dir) = path
                .rsplit_once('/')
                .map(|(d, _)| format!("{d}/"))
            {
                for (persona_id, persona_name, weight, source) in personas {
                    // Only count DirectKnows for directory index (not community matches)
                    if *source == PersonaMatchSource::DirectKnows {
                        let dir_map = dir_persona_data.entry(dir.clone()).or_default();
                        let entry = dir_map
                            .entry(*persona_id)
                            .or_insert_with(|| (persona_name.clone(), Vec::new()));
                        entry.1.push(*weight);
                    }
                }
            }
        }

        // Convert to dir_index format: dir → Vec<(persona_id, name, avg_weight, file_count)>
        index.dir_index.clear();
        for (dir, persona_map) in dir_persona_data {
            let mut dir_entries: Vec<(Uuid, String, f64, usize)> = persona_map
                .into_iter()
                .map(|(pid, (name, weights))| {
                    let count = weights.len();
                    let avg = weights.iter().sum::<f64>() / count as f64;
                    (pid, name, avg, count)
                })
                .collect();
            // Sort by file count desc, then avg weight desc
            dir_entries.sort_by(|a, b| {
                b.3.cmp(&a.3)
                    .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            });
            index.dir_index.insert(dir, dir_entries);
        }
    }

    /// Build persona context string for injection into additionalContext.
    async fn build_persona_context(
        &self,
        persona_id: Uuid,
        persona_name: &str,
        weight: f64,
    ) -> Option<String> {
        let subgraph = match self.graph_store.get_persona_subgraph(persona_id).await {
            Ok(sg) => sg,
            Err(e) => {
                debug!(
                    persona_id = %persona_id,
                    error = %e,
                    "Failed to load persona subgraph for hook"
                );
                return None;
            }
        };

        let mut ctx = format!(
            "## 🎭 Persona: {} (energy: {:.2}, weight: {:.2})\n",
            persona_name,
            // We don't have energy in subgraph, use a placeholder from stats
            subgraph.stats.freshness,
            weight
        );

        // Top notes (by weight, max 5)
        if !subgraph.notes.is_empty() {
            ctx.push_str("**Knowledge notes:**\n");
            for rel in subgraph.notes.iter().take(5) {
                ctx.push_str(&format!(
                    "- [note:{}] (w:{:.2})\n",
                    rel.entity_id, rel.weight
                ));
            }
        }

        // Top files (by weight, max 5)
        if !subgraph.files.is_empty() {
            ctx.push_str("**Known files:**\n");
            for rel in subgraph.files.iter().take(5) {
                ctx.push_str(&format!("- `{}` (w:{:.2})\n", rel.entity_id, rel.weight));
            }
        }

        // Truncate to ~2000 chars for additionalContext budget
        if ctx.len() > 2000 {
            ctx.truncate(1996);
            ctx.push_str("...\n");
        }

        Some(ctx)
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for SkillActivationHook {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        // 1. Only process PreToolUse events — passthrough everything else
        let pre_tool = match input {
            nexus_claude::HookInput::PreToolUse(pre) => pre,
            _ => return Ok(Self::passthrough()),
        };

        // 2. Resolve project from tool context (file path → project, or cwd → project)
        let project_id = match resolve_project_from_context(
            &*self.graph_store,
            &pre_tool.tool_name,
            &pre_tool.tool_input,
            &pre_tool.cwd,
        )
        .await
        {
            Ok(Some(id)) => id,
            Ok(None) => {
                debug!(
                    tool = %pre_tool.tool_name,
                    cwd = %pre_tool.cwd,
                    "Skill hook: no project matched"
                );
                return Ok(Self::passthrough());
            }
            Err(e) => {
                warn!(
                    tool = %pre_tool.tool_name,
                    "Skill hook: project resolution failed: {}", e
                );
                return Ok(Self::passthrough());
            }
        };

        // 3. Activate skills via the cached pipeline (same as REST endpoint)
        let config = HookActivationConfig::default();
        let cache = skill_cache();
        let skill_outcome = activate_for_hook_cached(
            &*self.graph_store,
            project_id,
            &pre_tool.tool_name,
            &pre_tool.tool_input,
            &config,
            cache,
        )
        .await;

        // 3b. Persona matching: on Read/Edit/Write, match file against persona KNOWS
        let file_path = extract_file_context(&pre_tool.tool_name, &pre_tool.tool_input);
        let persona_context = if let Some(ref fp) = file_path {
            match self.match_persona_for_file(project_id, fp).await {
                Some((pid, pname, weight)) => {
                    info!(
                        persona_name = %pname,
                        weight = weight,
                        file_path = %fp,
                        "Persona matched via file access"
                    );
                    self.build_persona_context(pid, &pname, weight).await
                }
                None => {
                    // Auto-grow: file not in KNOWS, check if adjacent to a persona's scope
                    // Fire-and-forget — must not slow down the PreToolUse response
                    let graph = self.graph_store.clone();
                    let fp_owned = fp.clone();
                    tokio::spawn(async move {
                        match graph.find_adjacent_personas(&fp_owned, project_id).await {
                            Ok(adjacent) if !adjacent.is_empty() => {
                                for (persona_id, persona_name) in &adjacent {
                                    if let Err(e) = graph
                                        .auto_grow_file_knows(*persona_id, &fp_owned, 0.3)
                                        .await
                                    {
                                        tracing::debug!(
                                            persona_id = %persona_id,
                                            file_path = %fp_owned,
                                            error = %e,
                                            "Auto-grow KNOWS failed (non-fatal)"
                                        );
                                    } else {
                                        tracing::debug!(
                                            persona_name = %persona_name,
                                            file_path = %fp_owned,
                                            "Auto-grow: added KNOWS(0.3) for adjacent file"
                                        );
                                    }
                                }
                            }
                            Ok(_) => {} // not adjacent to any persona
                            Err(e) => {
                                tracing::debug!(
                                    file_path = %fp_owned,
                                    error = %e,
                                    "Auto-grow: find_adjacent_personas failed (non-fatal)"
                                );
                            }
                        }
                    });
                    None
                }
            }
        } else {
            None
        };

        // If neither skill nor persona matched → passthrough
        let outcome = match skill_outcome {
            Ok(Some(outcome)) => Some(outcome),
            Ok(None) => {
                if persona_context.is_none() {
                    debug!(
                        tool = %pre_tool.tool_name,
                        "Skill hook: no skill or persona matched"
                    );
                    return Ok(Self::passthrough());
                }
                None
            }
            Err(e) => {
                warn!(
                    tool = %pre_tool.tool_name,
                    "Skill hook: activation failed: {}", e
                );
                if persona_context.is_none() {
                    return Ok(Self::passthrough());
                }
                None
            }
        };

        // 4. Hebbian reinforcement (fire-and-forget, async background)
        if let Some(ref outcome) = outcome {
            let reinforcement_config = AutoReinforcementConfig::default();
            spawn_hook_reinforcement(
                self.graph_store.clone(),
                outcome.activated_note_ids.clone(),
                reinforcement_config,
            );

            // 4b. Increment activation_count (fire-and-forget)
            spawn_activation_increment(self.graph_store.clone(), outcome.response.skill_id);
        }

        // 5. Build combined context: skill context + persona context + redirect suggestion
        let mut combined_context = outcome
            .as_ref()
            .map(|o| o.response.context.clone())
            .unwrap_or_default();

        // 5a. Inject persona context (after skill context)
        if let Some(ref pc) = persona_context {
            if !combined_context.is_empty() {
                combined_context.push_str("\n\n");
            }
            combined_context.push_str(pc);
        }

        // 5b. Generate redirect suggestion for Grep/Bash tools
        if let Some(suggestion) =
            generate_redirect_suggestion(&pre_tool.tool_name, &pre_tool.tool_input)
        {
            // Try to enrich with ContextCard (best-effort, don't block on failure)
            let redirect_fp = extract_file_context(&pre_tool.tool_name, &pre_tool.tool_input);
            let enriched = if let Some(ref fp) = redirect_fp {
                let project_id_str = project_id.to_string();
                match self.graph_store.get_context_card(fp, &project_id_str).await {
                    Ok(Some(card)) => enrich_redirect_with_context_card(suggestion, &card),
                    _ => crate::skills::hook_extractor::EnrichedRedirectSuggestion {
                        suggestion,
                        context_warnings: vec![],
                    },
                }
            } else {
                crate::skills::hook_extractor::EnrichedRedirectSuggestion {
                    suggestion,
                    context_warnings: vec![],
                }
            };

            // Append redirect suggestion to context (never overwrite skill context)
            combined_context.push_str("\n\n");
            combined_context.push_str(&enriched.to_string());

            debug!(
                tool = %pre_tool.tool_name,
                mcp_tool = %enriched.suggestion.mcp_tool,
                "Redirect suggestion injected"
            );
        }

        let skill_name = outcome
            .as_ref()
            .map(|o| o.response.skill_name.as_str())
            .unwrap_or("none");
        let confidence = outcome
            .as_ref()
            .map(|o| o.response.confidence)
            .unwrap_or(0.0);
        let has_persona = persona_context.is_some();

        info!(
            skill_name = %skill_name,
            confidence = confidence,
            has_persona = has_persona,
            tool = %pre_tool.tool_name,
            "Hook activated via PreToolUse"
        );

        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                hook_specific_output: Some(nexus_claude::HookSpecificOutput::PreToolUse(
                    nexus_claude::PreToolUseHookSpecificOutput {
                        permission_decision: None,
                        permission_decision_reason: None,
                        updated_input: None,
                        additional_context: Some(combined_context),
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
    use nexus_claude::{HookCallback, HookContext, HookInput};

    /// Helper to create a minimal HookContext for tests
    fn test_context() -> HookContext {
        HookContext { signal: None }
    }

    #[tokio::test]
    async fn test_non_pre_tool_use_input_returns_passthrough() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = SkillActivationHook::new(mock_store);

        // PreCompact is not PreToolUse → should passthrough
        let input = HookInput::PreCompact(nexus_claude::PreCompactHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/transcript".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            trigger: "context_full".to_string(),
            custom_instructions: None,
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

    #[tokio::test]
    async fn test_pre_tool_use_no_project_match_returns_passthrough() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = SkillActivationHook::new(mock_store);

        let input = HookInput::PreToolUse(nexus_claude::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/transcript".to_string(),
            cwd: "/completely/unknown/path".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: serde_json::json!({"file_path": "/completely/unknown/file.rs"}),
        });

        let result = hook.execute(&input, None, &test_context()).await.unwrap();
        match result {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(sync.continue_, Some(true));
                // No project matched → no additional context
                assert!(sync.hook_specific_output.is_none());
            }
            _ => panic!("Expected Sync output"),
        }
    }

    #[test]
    fn test_passthrough_has_continue_true() {
        let output = SkillActivationHook::passthrough();
        match output {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(sync.continue_, Some(true));
                assert!(sync.hook_specific_output.is_none());
                assert!(sync.reason.is_none());
            }
            _ => panic!("Expected Sync output"),
        }
    }

    // ========================================================================
    // PersonaFileIndex tests
    // ========================================================================

    #[test]
    fn test_persona_file_index_creation_empty() {
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        assert!(index.entries.is_empty());
        assert!(index.dir_index.is_empty());
        assert!(index.loaded_at.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn test_persona_file_index_lookup_hit_with_source() {
        let persona_id = Uuid::new_v4();
        let mut entries = HashMap::new();
        entries.insert(
            "src/main.rs".to_string(),
            vec![(persona_id, "rust-expert".to_string(), 0.95, PersonaMatchSource::DirectKnows)],
        );
        let index = PersonaFileIndex {
            entries,
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        let result = index.entries.get("src/main.rs");
        assert!(result.is_some());
        let vec = result.unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec[0].0, persona_id);
        assert_eq!(vec[0].1, "rust-expert");
        assert!((vec[0].2 - 0.95).abs() < f64::EPSILON);
        assert_eq!(vec[0].3, PersonaMatchSource::DirectKnows);
    }

    #[test]
    fn test_persona_file_index_community_match_tagged() {
        let persona_id = Uuid::new_v4();
        let mut entries = HashMap::new();
        entries.insert(
            "src/community_file.rs".to_string(),
            vec![(persona_id, "community-persona".to_string(), 0.3, PersonaMatchSource::CommunityMatch)],
        );
        let index = PersonaFileIndex {
            entries,
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        let result = index.entries.get("src/community_file.rs").unwrap();
        assert_eq!(result[0].3, PersonaMatchSource::CommunityMatch);
        assert!((result[0].2 - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_persona_file_index_lookup_miss() {
        let persona_id = Uuid::new_v4();
        let mut entries = HashMap::new();
        entries.insert(
            "src/main.rs".to_string(),
            vec![(persona_id, "rust-expert".to_string(), 0.9, PersonaMatchSource::DirectKnows)],
        );
        let index = PersonaFileIndex {
            entries,
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        assert!(!index.entries.contains_key("src/lib.rs"));
    }

    #[test]
    fn test_persona_file_index_multiple_personas_per_file() {
        let p1 = Uuid::new_v4();
        let p2 = Uuid::new_v4();
        let mut entries = HashMap::new();
        entries.insert(
            "src/shared.rs".to_string(),
            vec![
                (p1, "persona-a".to_string(), 0.9, PersonaMatchSource::DirectKnows),
                (p2, "persona-b".to_string(), 0.3, PersonaMatchSource::CommunityMatch),
            ],
        );
        let index = PersonaFileIndex {
            entries,
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        let result = index.entries.get("src/shared.rs").unwrap();
        assert_eq!(result.len(), 2);
        // First entry (DirectKnows) should have higher weight than second (CommunityMatch)
        assert!(result[0].2 > result[1].2);
        assert_eq!(result[0].3, PersonaMatchSource::DirectKnows);
        assert_eq!(result[1].3, PersonaMatchSource::CommunityMatch);
    }

    #[test]
    fn test_persona_index_ttl_constant() {
        // Verify the TTL is 120 seconds (2 minutes)
        assert_eq!(PERSONA_INDEX_TTL, std::time::Duration::from_secs(120));
    }

    #[test]
    fn test_persona_file_index_fresh_within_ttl() {
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        // Freshly created index should be within TTL
        assert!(index.loaded_at.elapsed() < PERSONA_INDEX_TTL);
    }

    #[test]
    fn test_persona_file_index_dir_index_populated() {
        let p1 = Uuid::new_v4();
        let mut dir_index = HashMap::new();
        dir_index.insert(
            "src/neo4j/".to_string(),
            vec![(p1, "neo4j-expert".to_string(), 0.85, 3)],
        );
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index,
            loaded_at: std::time::Instant::now(),
        };
        let dir_matches = index.dir_index.get("src/neo4j/");
        assert!(dir_matches.is_some());
        let matches = dir_matches.unwrap();
        assert_eq!(matches[0].0, p1);
        assert_eq!(matches[0].3, 3); // 3 files in dir
    }

    // ========================================================================
    // Directory-prefix match tests (Task 4)
    // ========================================================================

    #[test]
    fn test_directory_prefix_match_found_above_threshold() {
        let p1 = Uuid::new_v4();
        let mut dir_index = HashMap::new();
        // Persona knows 3 files in src/neo4j/ → above threshold of 2
        dir_index.insert(
            "src/neo4j/".to_string(),
            vec![(p1, "neo4j-expert".to_string(), 0.8, 3)],
        );
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index,
            loaded_at: std::time::Instant::now(),
        };

        let result = SkillActivationHook::directory_prefix_match(
            &index,
            "src/neo4j/new_file.rs",
            2,
        );
        assert!(result.is_some());
        let (id, name, weight) = result.unwrap();
        assert_eq!(id, p1);
        assert_eq!(name, "neo4j-expert");
        // Weight should be avg(0.8) * 0.7 = 0.56
        assert!((weight - 0.56).abs() < f64::EPSILON);
    }

    #[test]
    fn test_directory_prefix_match_below_threshold() {
        let p1 = Uuid::new_v4();
        let mut dir_index = HashMap::new();
        // Persona knows only 1 file in src/api/ → below threshold of 2
        dir_index.insert(
            "src/api/".to_string(),
            vec![(p1, "api-expert".to_string(), 0.9, 1)],
        );
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index,
            loaded_at: std::time::Instant::now(),
        };

        let result = SkillActivationHook::directory_prefix_match(
            &index,
            "src/api/new_handler.rs",
            2,
        );
        assert!(result.is_none(), "Should not match with only 1 file in dir");
    }

    #[test]
    fn test_directory_prefix_match_weight_calculation() {
        let p1 = Uuid::new_v4();
        let mut dir_index = HashMap::new();
        // avg_weight = 0.6, so result weight = 0.6 * 0.7 = 0.42
        dir_index.insert(
            "src/chat/".to_string(),
            vec![(p1, "chat-expert".to_string(), 0.6, 4)],
        );
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index,
            loaded_at: std::time::Instant::now(),
        };

        let result = SkillActivationHook::directory_prefix_match(
            &index,
            "src/chat/new_module.rs",
            2,
        );
        assert!(result.is_some());
        let (_, _, weight) = result.unwrap();
        assert!((weight - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn test_directory_prefix_match_no_dir_in_path() {
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        // Root-level file has no parent directory
        let result = SkillActivationHook::directory_prefix_match(&index, "main.rs", 2);
        // parent_dir("main.rs") returns None since no '/' → should not panic
        assert!(result.is_none());
    }

    #[test]
    fn test_directory_prefix_match_unknown_dir() {
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };
        // Dir not in index → None
        let result = SkillActivationHook::directory_prefix_match(
            &index,
            "src/unknown/file.rs",
            2,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_directory_prefix_match_multiple_personas_picks_first() {
        let p1 = Uuid::new_v4();
        let p2 = Uuid::new_v4();
        let mut dir_index = HashMap::new();
        // Two personas know files in the same dir. p1 has more files (sorted first).
        dir_index.insert(
            "src/neo4j/".to_string(),
            vec![
                (p1, "neo4j-expert".to_string(), 0.9, 5),
                (p2, "generalist".to_string(), 0.7, 2),
            ],
        );
        let index = PersonaFileIndex {
            entries: HashMap::new(),
            dir_index,
            loaded_at: std::time::Instant::now(),
        };

        let result = SkillActivationHook::directory_prefix_match(
            &index,
            "src/neo4j/something.rs",
            2,
        );
        assert!(result.is_some());
        let (id, _, _) = result.unwrap();
        // Should pick p1 (first in sorted order, most files)
        assert_eq!(id, p1);
    }

    #[test]
    fn test_parent_dir_extraction() {
        assert_eq!(
            SkillActivationHook::parent_dir("src/neo4j/client.rs"),
            Some("src/neo4j/".to_string())
        );
        assert_eq!(
            SkillActivationHook::parent_dir("/abs/path/file.rs"),
            Some("/abs/path/".to_string())
        );
        assert_eq!(
            SkillActivationHook::parent_dir("file.rs"),
            None
        );
    }

    #[test]
    fn test_update_dir_index_from_entries() {
        let p1 = Uuid::new_v4();
        let mut entries = HashMap::new();
        // p1 KNOWS 3 files in src/neo4j/
        entries.insert(
            "src/neo4j/client.rs".to_string(),
            vec![(p1, "neo4j-expert".to_string(), 0.9, PersonaMatchSource::DirectKnows)],
        );
        entries.insert(
            "src/neo4j/traits.rs".to_string(),
            vec![(p1, "neo4j-expert".to_string(), 0.8, PersonaMatchSource::DirectKnows)],
        );
        entries.insert(
            "src/neo4j/mock.rs".to_string(),
            vec![(p1, "neo4j-expert".to_string(), 0.7, PersonaMatchSource::DirectKnows)],
        );
        // Community match should NOT be counted in dir_index
        entries.insert(
            "src/api/handlers.rs".to_string(),
            vec![(p1, "neo4j-expert".to_string(), 0.3, PersonaMatchSource::CommunityMatch)],
        );

        let mut index = PersonaFileIndex {
            entries,
            dir_index: HashMap::new(),
            loaded_at: std::time::Instant::now(),
        };

        SkillActivationHook::update_dir_index(&mut index, "", &[]);

        // src/neo4j/ should have 3 DirectKnows files for p1
        let neo4j_dir = index.dir_index.get("src/neo4j/");
        assert!(neo4j_dir.is_some());
        let neo4j_entries = neo4j_dir.unwrap();
        assert_eq!(neo4j_entries.len(), 1);
        assert_eq!(neo4j_entries[0].0, p1);
        assert_eq!(neo4j_entries[0].3, 3); // 3 files
        // avg weight = (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert!((neo4j_entries[0].2 - 0.8).abs() < f64::EPSILON);

        // src/api/ should NOT be in dir_index (only CommunityMatch, not DirectKnows)
        assert!(index.dir_index.get("src/api/").is_none());
    }

    // ========================================================================
    // match_persona_for_file tests
    // ========================================================================

    /// Helper to create a PersonaNode for tests
    fn test_persona(project_id: Uuid, name: &str) -> crate::neo4j::models::PersonaNode {
        crate::neo4j::models::PersonaNode {
            id: Uuid::new_v4(),
            project_id: Some(project_id),
            name: name.to_string(),
            description: format!("Test persona: {}", name),
            status: crate::neo4j::models::PersonaStatus::Active,
            complexity_default: None,
            timeout_secs: None,
            max_cost_usd: None,
            model_preference: None,
            system_prompt_override: None,
            energy: 0.8,
            cohesion: 0.5,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            origin: crate::neo4j::models::PersonaOrigin::Manual,
            created_at: chrono::Utc::now(),
            updated_at: None,
        }
    }

    #[tokio::test]
    async fn test_match_persona_for_file_cache_miss_no_match() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = SkillActivationHook::new(mock_store);

        let project_id = Uuid::new_v4();
        let result = hook
            .match_persona_for_file(project_id, "src/unknown.rs")
            .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_match_persona_for_file_found() {
        let mock_store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Set up a persona with a file relation
        let persona = test_persona(project_id, "neo4j-expert");
        mock_store.create_persona(&persona).await.unwrap();
        mock_store
            .add_persona_file(persona.id, "src/neo4j/client.rs", 0.85)
            .await
            .unwrap();

        let hook = SkillActivationHook::new(mock_store);
        let result = hook
            .match_persona_for_file(project_id, "src/neo4j/client.rs")
            .await;

        assert!(result.is_some());
        let (pid, pname, weight) = result.unwrap();
        assert_eq!(pid, persona.id);
        assert_eq!(pname, "neo4j-expert");
        assert!((weight - 0.85).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_match_persona_for_file_populates_cache() {
        let mock_store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let persona = test_persona(project_id, "cache-test");
        mock_store.create_persona(&persona).await.unwrap();
        mock_store
            .add_persona_file(persona.id, "src/cached.rs", 0.7)
            .await
            .unwrap();

        let hook = SkillActivationHook::new(mock_store);

        // First call: cache miss → loads from store
        let result1 = hook
            .match_persona_for_file(project_id, "src/cached.rs")
            .await;
        assert!(result1.is_some());

        // Verify cache was populated
        let cache = hook.persona_index.read().await;
        assert!(cache.contains_key(&project_id));
        let index = cache.get(&project_id).unwrap();
        assert!(index.entries.contains_key("src/cached.rs"));
    }

    #[tokio::test]
    async fn test_match_persona_for_file_uses_cache_on_second_call() {
        let mock_store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let persona = test_persona(project_id, "double-hit");
        mock_store.create_persona(&persona).await.unwrap();
        mock_store
            .add_persona_file(persona.id, "src/double.rs", 0.75)
            .await
            .unwrap();

        let hook = SkillActivationHook::new(mock_store);

        // First call populates cache
        let r1 = hook
            .match_persona_for_file(project_id, "src/double.rs")
            .await;
        assert!(r1.is_some());

        // Second call should hit cache (same result, no store access needed)
        let r2 = hook
            .match_persona_for_file(project_id, "src/double.rs")
            .await;
        assert!(r2.is_some());
        assert_eq!(r1.unwrap().0, r2.unwrap().0);
    }

    #[tokio::test]
    async fn test_match_persona_wrong_project_returns_none() {
        let mock_store = Arc::new(MockGraphStore::new());
        let project_a = Uuid::new_v4();
        let project_b = Uuid::new_v4();

        let persona = test_persona(project_a, "project-a-expert");
        mock_store.create_persona(&persona).await.unwrap();
        mock_store
            .add_persona_file(persona.id, "src/shared.rs", 0.9)
            .await
            .unwrap();

        let hook = SkillActivationHook::new(mock_store);

        // Query with wrong project ID → should not find the persona
        let result = hook
            .match_persona_for_file(project_b, "src/shared.rs")
            .await;
        assert!(result.is_none());
    }

    // ========================================================================
    // build_persona_context tests
    // ========================================================================

    #[tokio::test]
    async fn test_build_persona_context_with_subgraph() {
        let mock_store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let persona = test_persona(project_id, "context-builder");
        mock_store.create_persona(&persona).await.unwrap();

        // Add some file and note relations to build a richer subgraph
        mock_store
            .add_persona_file(persona.id, "src/main.rs", 0.9)
            .await
            .unwrap();
        mock_store
            .add_persona_file(persona.id, "src/lib.rs", 0.7)
            .await
            .unwrap();

        let hook = SkillActivationHook::new(mock_store);
        let ctx = hook
            .build_persona_context(persona.id, "context-builder", 0.85)
            .await;

        assert!(ctx.is_some());
        let context_str = ctx.unwrap();
        assert!(context_str.contains("Persona: context-builder"));
        assert!(context_str.contains("weight: 0.85"));
        assert!(context_str.contains("Known files:"));
        assert!(context_str.contains("src/main.rs"));
    }

    #[tokio::test]
    async fn test_build_persona_context_nonexistent_persona() {
        let mock_store = Arc::new(MockGraphStore::new());
        let hook = SkillActivationHook::new(mock_store);

        // Persona does not exist → should return None (error handled gracefully)
        let ctx = hook
            .build_persona_context(Uuid::new_v4(), "ghost", 0.5)
            .await;
        assert!(ctx.is_none());
    }

    #[tokio::test]
    async fn test_build_persona_context_truncates_long_output() {
        let mock_store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Create persona with a very long name to test truncation logic
        let long_name = "x".repeat(500);
        let persona = test_persona(project_id, &long_name);
        mock_store.create_persona(&persona).await.unwrap();

        // Add many files to inflate context size
        for i in 0..100 {
            let path = format!(
                "src/very/deep/nested/module{}/file_with_long_name_{}.rs",
                i, i
            );
            mock_store
                .add_persona_file(persona.id, &path, 0.5)
                .await
                .unwrap();
        }

        let hook = SkillActivationHook::new(mock_store);
        let ctx = hook
            .build_persona_context(persona.id, &long_name, 0.9)
            .await;

        assert!(ctx.is_some());
        let context_str = ctx.unwrap();
        // Context should be truncated to ~2000 chars
        assert!(context_str.len() <= 2001);
    }
}
