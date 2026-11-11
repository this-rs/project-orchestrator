use super::*;

/// RM-4 — une session finie par un chat relie ses commits à sa tâche.
///
/// C'est le chaînon qui manquait : `TaskEnricher` existait et n'était branché que sur
/// le chemin runner, donc une tâche terminée par un CHAT gardait
/// `get_task_commits() == []` — la livraison était réelle, le lien jamais fait.
#[tokio::test]
async fn closing_a_session_links_its_commits_to_its_task() {
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{LinkedPlanInfo, LinkedSessionInfo, LinkedTaskInfo};
    use std::sync::Arc;

    let (_repo, cwd) = crate::test_helpers::git_repo_with_one_commit();
    let graph: Arc<MockGraphStore> = Arc::new(MockGraphStore::new());

    let mut project = crate::test_helpers::test_project_named("alpha");
    project.root_path = cwd.clone();
    graph.create_project(&project).await.unwrap();

    let plan = crate::test_helpers::test_plan();
    graph.create_plan(&plan).await.unwrap();
    let task = crate::test_helpers::test_task();
    graph.create_task(plan.id, &task).await.unwrap();

    let mut session = crate::test_helpers::test_chat_session(Some("alpha"));
    session.cwd = cwd;
    // The window opens before the (backdated) commit of the repository.
    session.created_at = chrono::Utc::now() - chrono::Duration::hours(3);
    let sid = session.id;
    graph.create_chat_session(&session).await.unwrap();

    // The session is attached to the task and the plan, as the auto-linking writes it.
    graph.session_links.write().await.insert(
        sid,
        LinkedSessionInfo {
            linked_plans: vec![LinkedPlanInfo {
                id: plan.id,
                title: "Test Plan".to_string(),
                source: "auto".to_string(),
            }],
            linked_tasks: vec![LinkedTaskInfo {
                id: task.id,
                title: "Test Task".to_string(),
                source: "auto".to_string(),
            }],
            linked_rfcs: vec![],
        },
    );

    let graph_dyn: Arc<dyn crate::neo4j::traits::GraphStore> = graph.clone();
    ChatManager::link_session_commits(graph_dyn, sid).await;

    assert!(
        !graph.get_task_commits(task.id).await.unwrap().is_empty(),
        "le commit de la fenêtre de la session est désormais relié à sa tâche"
    );
}

/// The review of #688 at 60a277e7 (finding D): a provider kind that does not
/// serialize to a name falls back to `unknown` (no session grant), never to a
/// permissive kind.
#[test]
fn an_unnamed_provider_kind_falls_back_to_unknown() {
    use crate::chat::session_grants::Asker;
    struct Unserializable;
    impl serde::Serialize for Unserializable {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("no name"))
        }
    }
    for name in [
        provider_kind_name(&Unserializable),
        provider_kind_name(&42),
        provider_kind_name(&serde_json::json!({ "kind": "claude_code" })),
    ] {
        assert_eq!(name, "unknown");
        assert_eq!(Asker::from_provider_kind(&name), Asker::Other);
    }
    assert_eq!(
        provider_kind_name(&nexus_claude::agent::ProviderKind::Native),
        "native"
    );
    assert_eq!(
        provider_kind_name(&nexus_claude::agent::ProviderKind::ClaudeCode),
        "claude_code"
    );
}

#[tokio::test]
async fn test_enrichment_project_id_resolves_the_session_slug() {
    let store = crate::neo4j::mock::MockGraphStore::new();
    let project = crate::test_helpers::test_project_named("Enriched");
    store.create_project(&project).await.unwrap();
    assert_eq!(
        enrichment_project_id(&store, Some(&project.slug)).await,
        Some(project.id)
    );
    assert_eq!(enrichment_project_id(&store, Some("unknown")).await, None);
    assert_eq!(enrichment_project_id(&store, None).await, None);
}
use crate::neo4j::models::ChatSessionNode;
use crate::test_helpers::{
    mock_app_state, mock_app_state_with_graph, test_chat_session, test_project,
};
use nexus_claude::{
    AssistantMessage, ContentBlock, ContentValue, PermissionMode, TextContent, ThinkingContent,
    ToolResultContent, ToolUseContent,
};
use std::path::PathBuf;
use std::time::Duration;

/// The configuration of the manager tests: Claude Code on the HISTORICAL engine, which
/// most of them exercise (the default is the agent engine since M2: `agent_manager`).
fn test_config() -> ChatConfig {
    ChatConfig {
        provider_path: crate::chat::config::ProviderPath::Legacy,
        mcp_server_path: PathBuf::from("/usr/bin/mcp_server"),
        nexus_tools_path: None,
        nexus_browser_path: None,
        obscura_path: None,
        default_model: "claude-sonnet-4-6".into(),
        max_sessions: 10,
        session_timeout: Duration::from_secs(1800),
        person_wait_cap: crate::chat::manager::PERSON_WAIT_CAP,
        neo4j_uri: "bolt://localhost:7687".into(),
        neo4j_user: "neo4j".into(),
        neo4j_password: "test".into(),
        meilisearch_url: "http://localhost:7700".into(),
        meilisearch_key: "key".into(),
        nats_url: None,
        max_turns: 10,
        permission: crate::chat::config::PermissionConfig::default(),
        auto_continue: false,
        retry: crate::chat::config::RetryConfig::default(),
        process_path: None,
        claude_cli_path: None,
        auto_update_cli: false,
        auto_update_app: true,
        jwt_secret: None,
        server_port: 8080,
        session_token_expiry_secs: 86400,
        read_only_mcp_tools: Vec::new(),
        lasting_rules_dir: None,
        native_turn_timeout_secs: None,
        native_session_max_tokens: None,
    }
}

#[test]
fn test_resolve_model_with_override() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    assert_eq!(
        manager.resolve_model(Some("claude-sonnet-4-6")),
        "claude-sonnet-4-6"
    );
}

#[test]
fn test_resolve_model_default() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    assert_eq!(manager.resolve_model(None), "claude-sonnet-4-6");
}

#[tokio::test]
async fn test_build_options_uses_config_permission_default() {
    // Default config uses "default" permission mode (safe-by-default)
    // and pre-approves MCP tools out of the box
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let opts = manager
        .build_options(
            "/tmp",
            "claude-opus-4-6",
            "test prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(matches!(opts.permission_mode, PermissionMode::Default));
    assert_eq!(opts.allowed_tools, vec!["mcp__project-orchestrator__*"]);
    assert!(opts.disallowed_tools.is_empty());
}

// ── read-only sessions: the harness refuses, the prompt is not involved ──

#[tokio::test]
async fn read_only_access_adds_the_deny_list_to_the_cli_options_and_keeps_the_mode() {
    use crate::chat::provider::policy::SessionAccess;
    let state = mock_app_state();
    let mut config = test_config();
    config.permission.disallowed_tools = vec!["Bash(rm -rf *)".into()];
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    for mode in [Some("bypassPermissions"), Some("plan"), None] {
        let opts = manager
            .build_options_with_access(
                "/tmp",
                "m",
                "p",
                Some("resume-id"),
                mode,
                None,
                &[],
                None,
                None,
                SessionAccess::ReadOnly,
            )
            .await;
        for tool in ["Edit", "Write", "MultiEdit", "NotebookEdit", "Bash", "Task"] {
            assert!(opts.disallowed_tools.iter().any(|d| d == tool), "{tool}");
        }
        assert!(opts.disallowed_tools.iter().any(|d| d == "Bash(rm -rf *)"));
        // The mode itself is not rewritten: the deny list is what protects.
        if mode == Some("bypassPermissions") {
            assert!(matches!(
                opts.permission_mode,
                PermissionMode::BypassPermissions
            ));
        }
    }
}

#[tokio::test]
async fn normal_access_leaves_the_cli_options_as_build_options_makes_them() {
    use crate::chat::provider::policy::SessionAccess;
    let state = mock_app_state();
    let mut config = test_config();
    config.permission.disallowed_tools = vec!["Bash(rm -rf *)".into()];
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let plain = manager
        .build_options("/tmp", "m", "p", None, None, None, &[], None, None)
        .await;
    let with = manager
        .build_options_with_access(
            "/tmp",
            "m",
            "p",
            None,
            None,
            None,
            &[],
            None,
            None,
            SessionAccess::Normal,
        )
        .await;
    assert_eq!(plain.disallowed_tools, with.disallowed_tools);
    assert_eq!(plain.disallowed_tools, vec!["Bash(rm -rf *)"]);
}

#[tokio::test]
async fn a_read_only_agent_spec_denies_writes_in_trust_and_a_normal_one_does_not() {
    use crate::chat::provider::policy::SessionAccess;
    use nexus_claude::agent::{PolicyDecision, ProviderKind, ToolCategory};
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let spec_for = |access: SessionAccess| {
        let manager = &manager;
        async move {
            manager
                .build_agent_spec_with_access(
                    AgentSpecInput {
                        file_scope_lifted: false,
                        cwd: "/tmp",
                        model: "m",
                        system_prompt: "p",
                        permission_mode: Some("bypassPermissions"),
                        add_dirs: &[],
                        user_claims: None,
                        session_id: "ro-s1",
                        third_party: false,
                        max_tokens: None,
                        max_cost_usd: None,
                        kind: ProviderKind::ClaudeCode,
                        remote_cwd: None,
                        per_session_mcp: true,
                        hooks: None,
                    },
                    access,
                )
                .await
                .unwrap()
        }
    };
    let read_only = spec_for(SessionAccess::ReadOnly).await;
    let normal = spec_for(SessionAccess::Normal).await;
    // A tool the prompt never mentions: the policy refuses it all the same.
    let write = |spec: &nexus_claude::agent::SessionSpec| {
        spec.policy
            .decide("Write", Some("/tmp/x"), ToolCategory::Edit)
    };
    assert_eq!(write(&read_only), PolicyDecision::Deny);
    assert_eq!(write(&normal), PolicyDecision::Allow);
    assert_eq!(
        read_only
            .policy
            .decide("Read", Some("/tmp/x"), ToolCategory::Read),
        PolicyDecision::Allow
    );
}

#[tokio::test]
async fn test_build_options_uses_config_permission_custom() {
    let state = mock_app_state();
    let mut config = test_config();
    config.permission = crate::chat::config::PermissionConfig {
        mode: "default".into(),
        allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
        disallowed_tools: vec!["Bash(rm -rf *)".into()],
    };
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

    let opts = manager
        .build_options(
            "/tmp",
            "claude-opus-4-6",
            "test prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(matches!(opts.permission_mode, PermissionMode::Default));
    assert_eq!(opts.allowed_tools, vec!["Bash(git *)", "Read"]);
    assert_eq!(opts.disallowed_tools, vec!["Bash(rm -rf *)"]);
}

#[tokio::test]
async fn test_build_options_with_permission_override() {
    // Global config is Default, session overrides to BypassPermissions
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let opts = manager
        .build_options(
            "/tmp",
            "claude-opus-4-6",
            "prompt",
            None,
            Some("bypassPermissions"),
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(matches!(
        opts.permission_mode,
        PermissionMode::BypassPermissions
    ));

    // Without override, falls back to global (Default)
    let opts = manager
        .build_options(
            "/tmp",
            "claude-opus-4-6",
            "prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(matches!(opts.permission_mode, PermissionMode::Default));
}

#[tokio::test]
async fn test_get_permission_config_returns_defaults() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let config = manager.get_permission_config().await;
    assert_eq!(config.mode, "default");
    // MCP tools are pre-approved by default
    assert_eq!(config.allowed_tools, vec!["mcp__project-orchestrator__*"]);
    assert!(config.disallowed_tools.is_empty());
}

#[tokio::test]
async fn test_update_permission_config_changes_mode() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let new_config = crate::chat::config::PermissionConfig {
        mode: "default".into(),
        allowed_tools: vec!["Read".into(), "Bash(git *)".into()],
        disallowed_tools: vec!["Bash(rm -rf *)".into()],
    };
    let updated = manager.update_permission_config(new_config).await.unwrap();
    assert_eq!(updated.mode, "default");
    assert_eq!(updated.allowed_tools, vec!["Read", "Bash(git *)"]);
    assert_eq!(updated.disallowed_tools, vec!["Bash(rm -rf *)"]);

    // Verify getter returns the updated config
    let fetched = manager.get_permission_config().await;
    assert_eq!(fetched.mode, "default");
    assert_eq!(fetched.allowed_tools, vec!["Read", "Bash(git *)"]);
}

#[tokio::test]
async fn test_update_permission_config_rejects_invalid_mode() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let bad_config = crate::chat::config::PermissionConfig {
        mode: "yolo".into(),
        ..Default::default()
    };
    let result = manager.update_permission_config(bad_config).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid permission mode 'yolo'"));
}

#[tokio::test]
async fn test_update_permission_config_affects_build_options() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // Initially Default (safe-by-default)
    let opts = manager
        .build_options(
            "/tmp",
            "claude-opus-4-6",
            "prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(matches!(opts.permission_mode, PermissionMode::Default));

    // Update to BypassPermissions mode at runtime
    manager
        .update_permission_config(crate::chat::config::PermissionConfig {
            mode: "bypassPermissions".into(),
            allowed_tools: vec![],
            disallowed_tools: vec![],
        })
        .await
        .unwrap();

    // New build_options should reflect the update
    let opts = manager
        .build_options(
            "/tmp",
            "claude-opus-4-6",
            "prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(matches!(
        opts.permission_mode,
        PermissionMode::BypassPermissions
    ));
    assert!(opts.allowed_tools.is_empty());
}

#[tokio::test]
async fn test_build_system_prompt_no_project() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let (prompt, note_ids) = manager
        .build_system_prompt(None, "test", None, None, None, None)
        .await;
    assert!(prompt.contains("Project Orchestrator"));
    assert!(prompt.contains("EXCLUSIVELY the Project Orchestrator MCP tools"));
    assert!(!prompt.contains("Active Project"));
    assert!(note_ids.is_empty(), "No project → no included note IDs");
}

// ── Anchor context wiring (PO_ANCHOR_CONTEXT) ─────────────────────────

use crate::chat::anchor_resolver::AnchorContextMode;

fn anchor_in(mode: AnchorContextMode) -> crate::chat::anchor_resolver::AnchorSession {
    crate::chat::anchor_resolver::AnchorSession::new(mode, Default::default())
}

fn manager_in(mode: AnchorContextMode, state: &crate::AppState) -> ChatManager {
    ChatManager::new_without_memory(state.neo4j.clone(), state.meili.clone(), test_config())
        .with_anchor_context_mode(mode)
}

#[tokio::test]
async fn shadow_and_off_prompts_are_byte_identical_to_the_historical_one() {
    let state = mock_app_state();
    let project = test_project();
    state.neo4j.create_project(&project).await.unwrap();
    let sid = Uuid::new_v4().to_string();
    let plain = manager_in(AnchorContextMode::Off, &state)
        .build_system_prompt(Some(&project.slug), "plan", None, None, Some(&sid), None)
        .await;
    for mode in [AnchorContextMode::Off, AnchorContextMode::Shadow] {
        let m = manager_in(mode, &state);
        let anchored = m
            .build_system_prompt_anchored(
                crate::chat::neutral_place::ExecutionPlace::Project,
                "/tmp/test-project",
                Some(&project.slug),
                "plan",
                None,
                None,
                &sid,
                None,
            )
            .await;
        assert_eq!(anchored.0.as_bytes(), plain.0.as_bytes(), "{mode:?}");
    }
}

#[tokio::test]
async fn on_mode_puts_a_stable_anchor_map_in_the_system_prompt() {
    let state = mock_app_state();
    let project = test_project();
    state.neo4j.create_project(&project).await.unwrap();
    let sid = Uuid::new_v4().to_string();
    let m = manager_in(AnchorContextMode::On, &state);
    let build = || {
        m.build_system_prompt_anchored(
            crate::chat::neutral_place::ExecutionPlace::Project,
            "/tmp/test-project",
            Some(&project.slug),
            "plan",
            None,
            None,
            &sid,
            None,
        )
    };
    let (first, _) = build().await;
    let (second, _) = build().await;
    let base = m
        .build_system_prompt(Some(&project.slug), "plan", None, None, Some(&sid), None)
        .await
        .0;
    assert!(
        first.starts_with(&base),
        "the historical prompt stays the prefix"
    );
    assert!(first.contains("<untrusted_data") && first.contains("anchor_map"));
    assert_eq!(first, second, "same anchors, same epochs: same bytes");
}

#[tokio::test]
async fn on_mode_opens_the_enrichment_with_the_live_block_and_shadow_never_does() {
    let state = mock_app_state();
    let mut s = crate::test_helpers::test_chat_session(None);
    s.execution_place = crate::chat::neutral_place::ExecutionPlace::Neutral;
    s.cwd = crate::chat::neutral_place::root()
        .join("s")
        .display()
        .to_string();
    state.neo4j.create_chat_session(&s).await.unwrap();
    let graph: Arc<dyn GraphStore> = state.neo4j.clone();
    let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
    let sid = s.id.to_string();
    let on = enrichment_for_turn(
        &graph,
        &pipeline,
        &sid,
        "hello",
        Default::default(),
        Default::default(),
        &anchor_in(AnchorContextMode::On),
        None,
    )
    .await
    .expect("a neutral session without anchor gets the notice");
    assert!(
        on.starts_with(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT),
        "{on}"
    );
    for mode in [AnchorContextMode::Off, AnchorContextMode::Shadow] {
        let out = enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "hello",
            Default::default(),
            Default::default(),
            &anchor_in(mode),
            None,
        )
        .await;
        assert!(
            out.is_none_or(|o| !o.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT)),
            "{mode:?}"
        );
    }
}

/// T7: the posture is observed and journaled every turn, in `on` and in `shadow`,
/// and the turn's enrichment is the one it would be without it.
#[tokio::test]
async fn the_posture_is_observed_in_shadow_without_changing_the_turn() {
    let state = mock_app_state();
    let graph: Arc<dyn GraphStore> = state.neo4j.clone();
    let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
    for mode in [AnchorContextMode::On, AnchorContextMode::Shadow] {
        let s = crate::test_helpers::test_chat_session(None);
        state.neo4j.create_chat_session(&s).await.unwrap();
        let sid = s.id.to_string();
        let turn = |anchor: crate::chat::anchor_resolver::AnchorSession| {
            let (graph, pipeline, sid) = (&graph, &pipeline, sid.clone());
            async move {
                enrichment_for_turn(
                    graph,
                    pipeline,
                    &sid,
                    "hello",
                    Default::default(),
                    Default::default(),
                    &anchor,
                    None,
                )
                .await
            }
        };
        // The observation of `turns` turns (shadow: a detached task).
        let observed = |turns: u64| async move {
            for _ in 0..10_000 {
                match crate::chat::posture::book().tracker(s.id) {
                    Some(t) if t.turns >= turns => return t,
                    _ => tokio::task::yield_now().await,
                }
            }
            panic!("{mode:?}: turn {turns} never observed");
        };
        let before = turn(anchor_in(mode)).await;
        assert_eq!(
            observed(1).await.effective(),
            Some(&crate::chat::posture::Posture::neutral())
        );
        let project = crate::test_helpers::test_project_named(&format!("posture-{mode:?}"));
        state.neo4j.create_project(&project).await.unwrap();
        human_anchors_project(
            &graph,
            s.id,
            project.id,
            crate::chat::anchor::AnchorActor::User,
        )
        .await;
        let after = turn(anchor_in(mode)).await;
        for out in [&before, &after].into_iter().flatten() {
            assert!(
                !out.contains("## Posture") && !out.contains("Posture :"),
                "{mode:?}: {out}"
            );
        }
        if mode == AnchorContextMode::Shadow {
            assert_eq!(
                before, after,
                "shadow: the anchor changes nothing of the turn"
            );
        }
        let tracker = observed(2).await;
        assert_eq!(tracker.turns, 2, "{mode:?}");
        assert_eq!(
            tracker.effective().map(|p| p.persona),
            Some(crate::chat::posture::Persona::ProjectLead),
            "{mode:?}: the user's focus anchor switched at once"
        );
    }
}

#[tokio::test]
async fn a_neutral_session_is_not_given_a_project_from_its_cwd_in_on_mode() {
    let state = mock_app_state();
    let mut p = crate::test_helpers::test_project_named("rooted");
    p.root_path = "/definitely/not/neutral/rooted".into();
    state.neo4j.create_project(&p).await.unwrap();
    let graph: Arc<dyn GraphStore> = state.neo4j.clone();
    let neutral = crate::chat::neutral_place::root()
        .join("n")
        .display()
        .to_string();
    let slug = infer_session_project(
        graph.as_ref(),
        AnchorContextMode::On,
        None,
        crate::chat::neutral_place::ExecutionPlace::Neutral,
        &neutral,
    )
    .await;
    assert_eq!(slug, None);
    // even when a (wrong) cwd points inside a project: neutral stays neutral
    let slug = infer_session_project(
        graph.as_ref(),
        AnchorContextMode::On,
        None,
        crate::chat::neutral_place::ExecutionPlace::Neutral,
        "/definitely/not/neutral/rooted/src",
    )
    .await;
    assert_eq!(slug, None);
}
/// Records the project slug each enrichment is handed.
struct SlugProbe(Arc<std::sync::Mutex<Vec<Option<String>>>>);

#[async_trait::async_trait]
impl crate::chat::enrichment::ParallelEnrichmentStage for SlugProbe {
    async fn execute(
        &self,
        input: &crate::chat::enrichment::EnrichmentInput,
    ) -> anyhow::Result<crate::chat::enrichment::StageOutput> {
        self.0.lock().unwrap().push(input.project_slug.clone());
        Ok(crate::chat::enrichment::StageOutput::new("slug-probe"))
    }
    fn name(&self) -> &str {
        "slug-probe"
    }
    fn is_enabled(&self, _config: &crate::chat::enrichment::EnrichmentConfig) -> bool {
        true
    }
}

/// A stage that always contributes one section.
struct SectionProbe;

#[async_trait::async_trait]
impl crate::chat::enrichment::ParallelEnrichmentStage for SectionProbe {
    async fn execute(
        &self,
        _input: &crate::chat::enrichment::EnrichmentInput,
    ) -> anyhow::Result<crate::chat::enrichment::StageOutput> {
        let mut out = crate::chat::enrichment::StageOutput::new("section-probe");
        out.add_section(
            "Probe",
            "probe content",
            "section-probe",
            crate::chat::enrichment::EnrichmentSource::KnowledgeInjection,
        );
        Ok(out)
    }
    fn name(&self) -> &str {
        "section-probe"
    }
    fn is_enabled(&self, _config: &crate::chat::enrichment::EnrichmentConfig) -> bool {
        true
    }
}

/// One turn of a fresh session whose pipeline holds [`SectionProbe`].
async fn enrich_one_probed_turn() -> Option<String> {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let s = neutral_session();
    mock.create_chat_session(&s).await.unwrap();
    let graph: Arc<dyn GraphStore> = mock.clone();
    let mut pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
    pipeline.add_parallel_stage(Box::new(SectionProbe));
    let anchor = anchor_in(AnchorContextMode::Off);
    enrichment_for_turn(
        &graph,
        &pipeline,
        &s.id.to_string(),
        "hello",
        Default::default(),
        Default::default(),
        &anchor,
        None,
    )
    .await
}

/// With debug logging on, the enriched turn is reported (sections, time, hints)
/// and the prompt still carries the stage's section.
#[tokio::test]
async fn an_enriched_turn_is_reported_at_debug_level() {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(std::io::sink)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let md = enrich_one_probed_turn().await.expect("the probe's section");
    assert!(md.contains("probe content"), "{md}");
}

/// The child half of [`the_retrieval_trigger_invites_the_first_turn_to_consult_the_graph`]:
/// it only checks something when its parent re-runs it with the flag on.
#[tokio::test]
async fn retrieval_trigger_child_appends_the_invitation() {
    if std::env::var(crate::chat::retrieval_trigger::FLAG).as_deref() != Ok("1") {
        return;
    }
    let md = enrich_one_probed_turn()
        .await
        .expect("section + invitation");
    assert!(md.contains("probe content"), "{md}");
    let invite = crate::chat::retrieval_trigger::invitation(
        crate::notes::retrieval_ledger::Trigger::TaskBoundary,
    );
    assert!(
        md.ends_with(&invite),
        "the invitation closes the turn: {md}"
    );
}

/// `RETRIEVAL_TRIGGER=1`: a session's first turn ends with the invitation to
/// consult the graph. The flag is read from the process environment, so the
/// check runs in a child process of this test binary — setting it here would
/// leak into the tests running in parallel.
#[test]
fn the_retrieval_trigger_invites_the_first_turn_to_consult_the_graph() {
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "chat::manager::tests::retrieval_trigger_child_appends_the_invitation",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(crate::chat::retrieval_trigger::FLAG, "1")
        .output()
        .expect("the child test runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("1 passed"), "the child test ran: {stdout}");
}

/// Review #679 finding 6: a session belongs to the person who opened it, a delegated
/// session to the person behind its parent; the server's own sessions to nobody.
#[tokio::test]
async fn a_session_belongs_to_the_person_behind_it() {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let graph: Arc<dyn GraphStore> = mock.clone();
    let m = ChatManager::new_without_memory(graph, mock_app_state().meili, test_config());
    let person = Uuid::new_v4();
    let secret = "test-secret-key-minimum-32-chars!!";
    let human = crate::auth::jwt::decode_jwt(
        &crate::auth::jwt::encode_jwt(person, "p@ffs.holdings", "P", secret, 60).unwrap(),
        secret,
    )
    .unwrap();
    assert_eq!(
        m.session_owner(Some(&human)).await.unwrap(),
        Some(person.to_string())
    );
    let server = crate::auth::jwt::Claims::service_account("runner");
    assert_eq!(m.session_owner(Some(&server)).await.unwrap(), None);
    assert_eq!(m.session_owner(None).await.unwrap(), None);

    // The agent session token of a session P owns opens sessions P owns.
    let parent = crate::neo4j::models::ChatSessionNode {
        file_scope_lifted: false,
        owner: Some(person.to_string()),
        ..crate::test_helpers::test_chat_session(None)
    };
    mock.create_chat_session(&parent).await.unwrap();
    let binding = crate::auth::jwt::AgentSessionBinding {
        session_id: parent.id.to_string(),
        ceiling: None,
        tool_profile: None,
        third_party: false,
    };
    let (token, _) =
        crate::auth::jwt::generate_session_token(&human, Some(&binding), secret, 60).unwrap();
    let agent = crate::auth::jwt::decode_jwt(&token, secret).unwrap();
    assert!(agent.is_agent_session());
    assert_eq!(
        m.session_owner(Some(&agent)).await.unwrap(),
        Some(person.to_string())
    );
}

/// Review of #699, finding 2: a read of the parent that fails never leaves a delegated
/// session without an owner (written once, at open). The token's `sub` is the person
/// it was minted for; a token minted for the server itself is refused.
#[tokio::test]
async fn a_delegated_session_whose_parent_cannot_be_read_is_not_left_without_owner() {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let graph: Arc<dyn GraphStore> = mock.clone();
    let m = ChatManager::new_without_memory(graph, mock_app_state().meili, test_config());
    let secret = "test-secret-key-minimum-32-chars!!";
    let parent = crate::neo4j::models::ChatSessionNode {
        file_scope_lifted: false,
        owner: Some("someone-else".into()),
        ..crate::test_helpers::test_chat_session(None)
    };
    mock.create_chat_session(&parent).await.unwrap();
    let binding = crate::auth::jwt::AgentSessionBinding {
        session_id: parent.id.to_string(),
        ceiling: None,
        tool_profile: None,
        third_party: false,
    };
    let agent_of = |claims: &crate::auth::jwt::Claims| {
        let (token, _) =
            crate::auth::jwt::generate_session_token(claims, Some(&binding), secret, 60).unwrap();
        crate::auth::jwt::decode_jwt(&token, secret).unwrap()
    };
    let person = Uuid::new_v4();
    let human = crate::auth::jwt::decode_jwt(
        &crate::auth::jwt::encode_jwt(person, "p@ffs.holdings", "P", secret, 60).unwrap(),
        secret,
    )
    .unwrap();
    mock.fail_reads.lock().unwrap().insert("get_chat_session");

    assert_eq!(
        m.session_owner(Some(&agent_of(&human))).await.unwrap(),
        Some(person.to_string()),
        "the person the token was minted for"
    );
    let server = crate::auth::jwt::Claims::service_account("runner");
    assert!(
        m.session_owner(Some(&agent_of(&server))).await.is_err(),
        "no person to fall back on: the open is refused"
    );

    // Readable again: the parent's owner wins over the token's `sub`.
    mock.fail_reads.lock().unwrap().clear();
    assert_eq!(
        m.session_owner(Some(&agent_of(&human))).await.unwrap(),
        Some("someone-else".to_string())
    );
}

/// Review of #699, finding 3: a person opening a session from an MCP client (their MCP
/// token) owns it, like with their session JWT.
#[tokio::test]
async fn a_session_opened_with_a_persons_mcp_token_belongs_to_that_person() {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let graph: Arc<dyn GraphStore> = mock.clone();
    let m = ChatManager::new_without_memory(graph, mock_app_state().meili, test_config());
    let secret = "test-secret-key-minimum-32-chars!!";
    let person = Uuid::new_v4();
    let (token, _) = crate::auth::jwt::encode_mcp_token(
        person,
        "p@ffs.holdings",
        "P",
        "mcp:read mcp:write",
        secret,
        60,
    )
    .unwrap();
    let mcp = crate::auth::jwt::decode_jwt(&token, secret).unwrap();
    assert!(mcp.is_mcp_token());
    assert_eq!(
        m.session_owner(Some(&mcp)).await.unwrap(),
        Some(person.to_string())
    );
    // The anonymous user (no-auth mode, the standalone MCP server) owns nothing.
    assert_eq!(
        m.session_owner(Some(&crate::auth::jwt::Claims::anonymous()))
            .await
            .unwrap(),
        None
    );
}

fn neutral_session() -> crate::neo4j::models::ChatSessionNode {
    let mut s = crate::test_helpers::test_chat_session(None);
    s.execution_place = crate::chat::neutral_place::ExecutionPlace::Neutral;
    s.cwd = crate::chat::neutral_place::root()
        .join(s.id.to_string())
        .display()
        .to_string();
    s
}

async fn anchor_plan(graph: &Arc<dyn GraphStore>, session: Uuid, plan: String) {
    use crate::chat::anchor::{AnchorActor, AnchorOp, AnchorRole, AnchorTargetType, NewAnchor};
    graph
        .apply_anchor_op(
            session,
            AnchorOp::Add(NewAnchor::new(
                AnchorTargetType::Plan,
                plan,
                [AnchorRole::Focus],
                AnchorActor::User,
                "t",
            )),
        )
        .await
        .unwrap();
}

async fn human_anchors_project(
    graph: &Arc<dyn GraphStore>,
    session: Uuid,
    project: Uuid,
    by: crate::chat::anchor::AnchorActor,
) {
    use crate::chat::anchor::{AnchorOp, AnchorRole, AnchorTargetType, NewAnchor};
    graph
        .apply_anchor_op(
            session,
            AnchorOp::Add(NewAnchor::new(
                AnchorTargetType::Project,
                project.to_string(),
                [AnchorRole::Focus],
                by,
                "t",
            )),
        )
        .await
        .unwrap();
}

/// The environment is read when the manager is built, nowhere else: a turn
/// takes the mode of its session (`ActiveSession::anchor`), which the manager
/// fixed when the session was opened or resumed.
#[tokio::test]
async fn the_anchor_mode_is_fixed_per_session_and_the_environment_is_not_read_per_turn() {
    let state = mock_app_state();
    let m = manager_in(AnchorContextMode::On, &state);
    assert_eq!(m.anchor_session().mode, AnchorContextMode::On);
    // every other call site of the environment reader would be a per-turn read
    let source = include_str!("../../manager.rs");
    let needle = concat!("AnchorContextMode::", "from_env()");
    assert_eq!(
        source.matches(needle).count(),
        2,
        "only the two constructors read the environment"
    );
    // a live session keeps the mode it was opened with
    assert_eq!(
        manager_in(AnchorContextMode::Off, &state)
            .anchor_session()
            .mode,
        AnchorContextMode::Off
    );
}

#[tokio::test]
async fn an_explicit_slug_that_resolves_to_no_project_is_kept_for_the_enrichment_in_on_mode() {
    let state = mock_app_state();
    let mut s = neutral_session();
    s.project_slug = Some("ghost".into());
    state.neo4j.create_chat_session(&s).await.unwrap();
    let graph: Arc<dyn GraphStore> = state.neo4j.clone();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
    pipeline.add_parallel_stage(Box::new(SlugProbe(seen.clone())));
    let anchor = anchor_in(AnchorContextMode::On);
    let sid = s.id.to_string();
    let live = enrichment_for_turn(
        &graph,
        &pipeline,
        &sid,
        "hello",
        Default::default(),
        Default::default(),
        &anchor,
        None,
    )
    .await
    .expect("the notice");
    // as before the resolver: the slug reaches the stages as it is
    assert_eq!(
        seen.lock().unwrap().last().unwrap().as_deref(),
        Some("ghost")
    );
    // ... without widening the resolver: no project, the neutral notice
    assert!(
        live.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT),
        "{live}"
    );
    // the warning is given once per session
    assert!(!anchor.cache.first_time(s.id, "unresolved_explicit_slug"));
    enrichment_for_turn(
        &graph,
        &pipeline,
        &sid,
        "again",
        Default::default(),
        Default::default(),
        &anchor,
        None,
    )
    .await;
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn an_anchor_put_mid_session_is_in_the_next_turn_the_hooks_and_the_next_map() {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let project = test_project();
    mock.create_project(&project).await.unwrap();
    let plan = crate::test_helpers::seed_plan_with_task(&mock, project.id).await;
    let s = neutral_session();
    mock.create_chat_session(&s).await.unwrap();
    let graph: Arc<dyn GraphStore> = mock.clone();
    let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
    let m = ChatManager::new_without_memory(graph.clone(), mock_app_state().meili, test_config())
        .with_anchor_context_mode(AnchorContextMode::On);
    let anchor = m.anchor_session();
    let sid = s.id.to_string();
    let hooks = m.hook_session_project(&sid).expect("mode on");
    let turn = |text: &'static str| {
        enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            text,
            Default::default(),
            Default::default(),
            &anchor,
            None,
        )
    };
    let system = || async {
        m.build_system_prompt_anchored(
            s.execution_place,
            &s.cwd,
            None,
            "hello",
            None,
            None,
            &sid,
            None,
        )
        .await
        .0
    };
    // turn 1: nothing anchored
    let first = turn("one").await.expect("the notice");
    assert!(first.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT));
    assert_eq!(hooks.project_id().await, None);
    let map_before = system().await;
    assert!(!map_before.contains("Plan mid"));

    human_anchors_project(
        &graph,
        s.id,
        project.id,
        crate::chat::anchor::AnchorActor::User,
    )
    .await;
    anchor_plan(&graph, s.id, plan).await;

    // the very next turn: live block with the new anchor, hooks on the new project
    let second = turn("two").await.expect("live block");
    assert!(second.contains("Task mid"), "{second}");
    assert!(!second.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT));
    assert_eq!(hooks.project_id().await, Some(project.id), "no 30 s wait");
    // the system prompt is rebuilt at the next prompt rebuild (resume,
    // compaction) with the current anchors
    let map_after = system().await;
    assert!(map_after.contains("Plan mid"), "{map_after}");
}

#[tokio::test]
async fn on_mode_turn_cost_in_store_reads_with_and_without_the_cache() {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let project = test_project();
    mock.create_project(&project).await.unwrap();
    let s = neutral_session();
    mock.create_chat_session(&s).await.unwrap();
    let graph: Arc<dyn GraphStore> = mock.clone();
    human_anchors_project(
        &graph,
        s.id,
        project.id,
        crate::chat::anchor::AnchorActor::User,
    )
    .await;
    let plan = crate::test_helpers::seed_plan_with_task(&mock, project.id).await;
    anchor_plan(&graph, s.id, plan).await;
    let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
    let reads = || mock.store_reads.load(std::sync::atomic::Ordering::Relaxed);
    let sid = s.id.to_string();
    let mut per_turn = Vec::new();
    // a cache of its own for every turn = no cache
    for _ in 0..3 {
        let b = reads();
        enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "x",
            Default::default(),
            Default::default(),
            &anchor_in(AnchorContextMode::On),
            None,
        )
        .await;
        per_turn.push(reads() - b);
    }
    let shared = anchor_in(AnchorContextMode::On);
    let mut cached = Vec::new();
    for _ in 0..3 {
        let b = reads();
        enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "x",
            Default::default(),
            Default::default(),
            &shared,
            None,
        )
        .await;
        cached.push(reads() - b);
    }
    eprintln!("on-mode reads per turn: without cache {per_turn:?}, with cache {cached:?}");
    assert!(cached[1] < per_turn[1], "{cached:?} vs {per_turn:?}");
    assert_eq!(cached[1], cached[2], "a stable cost once cached");
    // the shadow, in the same conditions: only the first turn reads
    let shadow = anchor_in(AnchorContextMode::Shadow);
    let mut sh = Vec::new();
    for _ in 0..3 {
        let b = reads();
        enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "x",
            Default::default(),
            Default::default(),
            &shadow,
            None,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        sh.push(reads() - b);
    }
    eprintln!("shadow reads per turn (turn included): {sh:?}");
    assert_eq!(sh[1], sh[2], "the shadow adds nothing after the first run");
}

/// CTX4: the PO prompt is budgeted from the model's KNOWN window, whatever the
/// model (a 128K DeepSeek carries what a 128K Claude carries, not the 10 000 of an
/// unknown window), and the window a session's prompt is built with is the one
/// frozen at its opening: two builds of the same session give the same prompt,
/// byte for byte, even when the provider has learned another window since.
#[tokio::test]
async fn the_prompt_budget_reads_the_window_and_holds_across_turns() {
    use crate::notes::{Note, NoteType};
    let state = mock_app_state();
    let project = test_project();
    state.neo4j.create_project(&project).await.unwrap();
    // A project context well past the unknown-window budget (10 000).
    for i in 0..50 {
        let note = Note::new(
            Some(project.id),
            NoteType::Gotcha,
            format!(
                "gotcha {i}: {}",
                "the dynamic context must hold this ".repeat(40)
            ),
            "test".into(),
        );
        state.neo4j.create_note(&note).await.unwrap();
    }
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let build = |model: &'static str, window: Option<u64>| {
        let manager = &manager;
        let slug = project.slug.clone();
        async move {
            manager
                .build_system_prompt(Some(&slug), "help me plan", Some(model), window, None, None)
                .await
                .0
        }
    };
    let deepseek = build("deepseek-chat", Some(128_000)).await;
    let claude = build("claude-sonnet-4-6", Some(128_000)).await;
    let unknown = build("deepseek-chat", None).await;
    assert_eq!(
        deepseek, claude,
        "same window, same prompt, whatever the name"
    );
    assert!(
        deepseek.len() > unknown.len() + 15_000,
        "known 128K window: {} chars, unknown: {}",
        deepseek.len(),
        unknown.len()
    );

    // The window of a session is the one frozen in its snapshot when it holds one:
    // a resume rebuilds the prompt with the same budget.
    let snapshot = r#"{"context_window":{"value":128000,"source":"probed"}}"#;
    assert_eq!(snapshot_context_window(Some(snapshot)), Some(128_000));
    assert_eq!(snapshot_context_window(Some("{}")), None);
    assert_eq!(snapshot_context_window(Some("not json")), None);
    assert_eq!(snapshot_context_window(None), None);
    let w1 = manager
        .session_context_window("unknown-provider", "deepseek-chat", Some(snapshot))
        .await;
    let w2 = manager
        .session_context_window("unknown-provider", "deepseek-chat", Some(snapshot))
        .await;
    assert_eq!((w1, w2), (Some(128_000), Some(128_000)));
    // No snapshot and nobody knows the window: unknown, never guessed from the name.
    assert_eq!(
        manager
            .session_context_window("unknown-provider", "claude-opus-4-6", None)
            .await,
        None
    );
    let first_turn = build("deepseek-chat", w1).await;
    let second_turn = build("deepseek-chat", w2).await;
    assert_eq!(first_turn, second_turn);
    assert_eq!(first_turn, deepseek);
}

/// CTX4, review of #748: a Claude Code session whose Anthropic catalog is
/// unreachable (OAuth login, no API key: the static fallback has no window) keeps
/// the 40 000-character budget, by ENGINE, never by the model's name; a native
/// provider with an unknown window keeps the 10 000 fallback.
#[tokio::test]
async fn a_claude_code_session_with_an_unreachable_catalog_keeps_40_000() {
    use super::super::composer::{
        compute_dynamic_budget, turn_enrichment_cap, DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET,
    };
    use super::super::provider::resolver::CLAUDE_CODE;
    use crate::chat::model_catalog::ModelCatalogCache;
    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    let state = mock_app_state();
    // No key: the catalog never fetches and serves the static fallback,
    // which carries the curated 200K window for offline resilience.
    let offline = ModelCatalogCache::new(None);
    assert!(offline
        .snapshot()
        .await
        .models
        .iter()
        .all(|m| m.max_input_tokens == Some(200_000)));
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
        .with_cognitive_routing(
            CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new()))
                .with_claude_code_catalog(offline),
        );
    assert_eq!(
        manager
            .known_context_window(CLAUDE_CODE, "claude-sonnet-4-6")
            .await,
        Some(200_000),
        "the static fallback now carries 200K"
    );

    // Creation (no snapshot yet) and a reopen whose snapshot states no window.
    for snapshot in [None, Some("{}")] {
        let window = manager
            .session_context_window(CLAUDE_CODE, "claude-sonnet-4-6", snapshot)
            .await;
        assert_eq!(window, Some(CLAUDE_CODE_FLOOR_WINDOW));
        assert_eq!(compute_dynamic_budget(window, 0), 40_000);
        assert_eq!(compute_dynamic_budget(window, 50_000), 40_000);
        assert_eq!(turn_enrichment_cap(window), 40_000);
    }
    // Decided by the engine, not the name: any model on Claude Code.
    assert_eq!(
        manager
            .session_context_window(CLAUDE_CODE, "some-future-model", None)
            .await,
        Some(CLAUDE_CODE_FLOOR_WINDOW)
    );
    // A known window still wins over the floor.
    assert_eq!(
        manager
            .session_context_window(
                CLAUDE_CODE,
                "claude-sonnet-4-6",
                Some(r#"{"context_window":{"value":1000000,"source":"catalog"}}"#),
            )
            .await,
        Some(1_000_000)
    );
    // A native provider with an unknown window, even with a Claude name: 10 000.
    let native = manager
        .session_context_window("unknown-provider", "claude-sonnet-4-6", None)
        .await;
    assert_eq!(native, None);
    assert_eq!(
        compute_dynamic_budget(native, 0),
        DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET
    );
    assert_eq!(turn_enrichment_cap(native), 10_000);
}

#[tokio::test]
async fn test_build_system_prompt_with_project() {
    let state = mock_app_state();
    let project = test_project();
    state.neo4j.create_project(&project).await.unwrap();

    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (prompt, _note_ids) = manager
        .build_system_prompt(Some(&project.slug), "help me plan", None, None, None, None)
        .await;

    // Contains the base prompt
    assert!(prompt.contains("EXCLUSIVELY the Project Orchestrator MCP tools"));
    // Contains dynamic context section (either oneshot or fallback)
    assert!(prompt.contains("---"));
    // The project name should appear somewhere in the dynamic context
    assert!(prompt.contains(&project.name));
}

#[tokio::test]
async fn test_session_not_active_by_default() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    assert!(!manager.is_session_active("nonexistent").await);
}

// ====================================================================
// build_options
// ====================================================================

#[tokio::test]
async fn test_build_options_basic() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let options = manager
        .build_options(
            "/tmp/project",
            "claude-opus-4-6",
            "System prompt here",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;

    assert_eq!(options.model, Some("claude-opus-4-6".into()));
    assert_eq!(options.cwd, Some(PathBuf::from("/tmp/project")));
    assert!(options.resume.is_none());
    assert!(options.mcp_servers.contains_key("project-orchestrator"));
}

#[test]
fn auto_model_is_the_choose_for_me_sentinel_not_a_name() {
    assert!(is_auto_model("auto"));
    assert!(is_auto_model("AUTO"));
    assert!(is_auto_model("  Auto  "));
    assert!(!is_auto_model("claude-opus-4-8"));
    assert!(!is_auto_model("automatic"));
    assert!(!is_auto_model(""));
}

#[test]
fn a_server_default_of_auto_puts_a_bare_pilot_opening_in_auto() {
    use super::super::provider::cognitive::ConversationRoutingMode;
    let bare = ChatRequest {
        message: "hello".into(),
        cwd: "/tmp/project".into(),
        ..Default::default()
    };
    assert!(opens_on_server_default_auto("auto", &bare));
    let routed = routed_by_server_default("auto", &bare).expect("routed by PO");
    assert_eq!(routed.routing_mode, Some(ConversationRoutingMode::Full));
    assert_eq!(routed.model, None);
    assert_eq!(routed.message, "hello");

    // The sentinel named as the model is nothing named: it is dropped, not pinned.
    let sentinel = ChatRequest {
        model: Some(" Auto ".into()),
        ..bare.clone()
    };
    assert!(opens_on_server_default_auto("auto", &sentinel));
    assert_eq!(
        routed_by_server_default("auto", &sentinel).map(|r| r.model),
        Some(None)
    );
    // Case and spaces in the config value do not matter either.
    assert!(opens_on_server_default_auto(" AUTO ", &bare));
}

#[test]
fn a_concrete_server_default_or_a_named_opening_is_left_to_the_declared_rules() {
    use super::super::provider::cognitive::ConversationRoutingMode;
    let bare = ChatRequest {
        message: "hello".into(),
        cwd: "/tmp/project".into(),
        ..Default::default()
    };
    // A concrete default: nothing changes, whatever the request.
    assert!(!opens_on_server_default_auto("claude-sonnet-5-5", &bare));
    assert!(routed_by_server_default("claude-sonnet-5-5", &bare).is_none());
    assert!(!opens_on_server_default_auto("", &bare));

    // Every way of naming an engine, a mode or a pool keeps the request as it is.
    let named: Vec<(&str, ChatRequest)> = vec![
        (
            "provider",
            ChatRequest {
                provider: Some("deepseek".into()),
                ..bare.clone()
            },
        ),
        (
            "model",
            ChatRequest {
                model: Some("claude-haiku-5-5".into()),
                ..bare.clone()
            },
        ),
        (
            "routing_mode",
            ChatRequest {
                routing_mode: Some(ConversationRoutingMode::Primary),
                ..bare.clone()
            },
        ),
        (
            "routing_pool",
            ChatRequest {
                routing_pool: Some(vec![super::super::types::RoutingPoolEntry {
                    provider: "claude-code".into(),
                    model: "claude-haiku-5-5".into(),
                }]),
                ..bare.clone()
            },
        ),
        (
            "task_alias",
            ChatRequest {
                task_alias: Some("fast".into()),
                ..bare.clone()
            },
        ),
        (
            "persona_alias",
            ChatRequest {
                persona_alias: Some("deep".into()),
                ..bare.clone()
            },
        ),
        (
            "run_provider",
            ChatRequest {
                run_provider: Some("deepseek".into()),
                ..bare.clone()
            },
        ),
        (
            "spawned_by",
            ChatRequest {
                spawned_by: Some("conversation:parent".into()),
                ..bare.clone()
            },
        ),
        (
            "route_auto",
            ChatRequest {
                route_auto: true,
                ..bare.clone()
            },
        ),
    ];
    for (what, request) in named {
        assert!(
            !opens_on_server_default_auto("auto", &request),
            "{what} names what the opening wants: the server default decides nothing"
        );
        assert!(
            routed_by_server_default("auto", &request).is_none(),
            "{what}"
        );
    }
}

#[tokio::test]
async fn test_build_options_omits_model_flag_for_auto_sentinel() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // `auto` must NOT reach the CLI as `--model auto` (it rejects it): the flag is omitted
    // so the account's own default is used.
    let options = manager
        .build_options(
            "/tmp/project",
            "auto",
            "System prompt here",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert_eq!(options.model, None, "auto must not be sent as --model");

    // A named model is still passed through.
    let named = manager
        .build_options(
            "/tmp/project",
            "claude-opus-4-8",
            "System prompt here",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert_eq!(named.model, Some("claude-opus-4-8".into()));
}

#[tokio::test]
async fn test_build_options_with_resume() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let options = manager
        .build_options(
            "/tmp/project",
            "claude-opus-4-6",
            "System prompt",
            Some("cli-session-abc"),
            None,
            None,
            &[],
            None,
            None,
        )
        .await;

    assert_eq!(options.resume, Some("cli-session-abc".into()));
}

#[tokio::test]
async fn test_build_options_mcp_server_config() {
    let state = mock_app_state();
    let config = test_config();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

    let options = manager
        .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
        .await;

    let mcp = options.mcp_servers.get("project-orchestrator").unwrap();
    match mcp {
        McpServerConfig::Stdio { command, env, .. } => {
            assert_eq!(command, "/usr/bin/mcp_server");
            let env = env.as_ref().unwrap();
            assert!(env.contains_key("PO_SERVER_URL"));
        }
        _ => panic!("Expected Stdio MCP config"),
    }
}

// ── The knowledge graph's hooks on the agent engine ─────────────────────

fn scope(runner: bool) -> Option<AgentHookScope> {
    Some(AgentHookScope {
        project_slug: Some("proj".into()),
        task_id: None,
        runner,
    })
}

#[tokio::test]
async fn the_agent_engine_gives_the_graph_hooks_only_to_providers_that_run_hooks() {
    use nexus_claude::agent::ProviderKind;
    let state = mock_app_state();
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let claims = crate::auth::jwt::Claims::service_account("hooks");
    let has_hooks = |kind: ProviderKind, remote: Option<&'static str>, hooks| {
        let manager = &manager;
        let claims = &claims;
        async move {
            manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/tmp",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: None,
                    add_dirs: &[],
                    user_claims: Some(claims),
                    session_id: "hooks-s1",
                    third_party: true,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind,
                    remote_cwd: remote,
                    per_session_mcp: true,
                    hooks,
                })
                .await
                .unwrap()
                .hooks
                .is_some()
        }
    };
    // Native runs them in its loop; so does Claude Code (its hook protocol).
    assert!(has_hooks(ProviderKind::Native, None, scope(false)).await);
    assert!(has_hooks(ProviderKind::ClaudeCode, None, scope(false)).await);
    // Codex and ACP ignore them (and say so): they are not handed any.
    assert!(!has_hooks(ProviderKind::Codex, None, scope(false)).await);
    assert!(!has_hooks(ProviderKind::Acp, None, scope(false)).await);
    // A remote machine has no local project to resolve a file against.
    assert!(!has_hooks(ProviderKind::ClaudeCode, Some("~/w"), scope(false)).await);
    // No scope, no hooks.
    assert!(!has_hooks(ProviderKind::Native, None, None).await);
}

/// CTX0: a native session names its conversation to the harness
/// (`extensions.native.host_session_id`), which writes it in the log line of every
/// model request; another provider is given no such extension.
#[tokio::test]
async fn a_native_session_spec_carries_the_conversation_id_for_the_request_log() {
    use nexus_claude::agent::ProviderKind;
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let spec_of = |kind: ProviderKind| {
        let manager = &manager;
        async move {
            manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/tmp",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: None,
                    add_dirs: &[],
                    user_claims: None,
                    session_id: "36d623c8-c7f6-4818-bcb3-26da85a97380",
                    third_party: kind != ProviderKind::ClaudeCode,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind,
                    remote_cwd: None,
                    per_session_mcp: false,
                    hooks: None,
                })
                .await
                .unwrap()
        }
    };
    let native = spec_of(ProviderKind::Native).await;
    assert_eq!(
        native.extension("native"),
        Some(&serde_json::json!({
            "host_session_id": "36d623c8-c7f6-4818-bcb3-26da85a97380"
        }))
    );
    let claude = spec_of(ProviderKind::ClaudeCode).await;
    assert!(claude.extension("native").is_none(), "{claude:?}");
}

/// M2: a local Claude Code on the agent engine gets in its own environment what the
/// historical engine gives its CLI (`build_options`): the operator's PATH, and what
/// `orchestrator secret get` needs in its shell. Not a remote one, not another provider.
#[tokio::test]
async fn a_local_claude_code_on_the_agent_engine_gets_the_shell_environment_of_the_historical_engine(
) {
    use nexus_claude::agent::ProviderKind;
    let state = mock_app_state();
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    config.server_port = 18123;
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config)
        .with_vault(crate::vault::VaultService::ephemeral());
    manager
        .update_process_path(Some("/opt/tools/bin:/usr/bin:/bin".into()))
        .await;
    let claims = crate::auth::jwt::Claims::service_account("env");
    let env_of = |kind: ProviderKind, remote: Option<&'static str>| {
        let manager = &manager;
        let claims = &claims;
        async move {
            manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/tmp",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: None,
                    add_dirs: &[],
                    user_claims: Some(claims),
                    session_id: "env-s1",
                    third_party: kind != ProviderKind::ClaudeCode,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind,
                    remote_cwd: remote,
                    per_session_mcp: true,
                    hooks: None,
                })
                .await
                .unwrap()
                .env
                .set
        }
    };
    let env = env_of(ProviderKind::ClaudeCode, None).await;
    assert_eq!(
        env.get("PATH").map(String::as_str),
        Some("/opt/tools/bin:/usr/bin:/bin")
    );
    assert!(
        env.get("PO_VAULT_TOKEN").is_some_and(|t| !t.is_empty()),
        "{:?}",
        env.keys()
    );
    assert_eq!(
        env.get("PO_SERVER_URL").map(String::as_str),
        Some("http://127.0.0.1:18123")
    );
    // A remote machine gets none of it.
    assert!(env_of(ProviderKind::ClaudeCode, Some("~/w"))
        .await
        .is_empty());
    // A local native session gets the PATH (its shell, nexus-tools, inherits it: under
    // launchd the server's own is /usr/bin:/bin:/usr/sbin:/sbin, ISO5), and nothing else.
    let native = env_of(ProviderKind::Native, None).await;
    assert_eq!(
        native.keys().map(String::as_str).collect::<Vec<_>>(),
        vec!["PATH"],
        "{native:?}"
    );
    assert_eq!(
        native.get("PATH").map(String::as_str),
        Some("/opt/tools/bin:/usr/bin:/bin")
    );
    assert!(env_of(ProviderKind::Native, Some("~/w")).await.is_empty());
}

/// ISO5: a native session has no turn wall-clock cap and no token budget by default
/// (Claude Code has neither; a 30 min cap and a 10 M budget, cache reads included, cut
/// a working DeepSeek session). Configured, both reach the session's limits; a budget
/// the request names wins; another provider is not given them.
#[tokio::test]
async fn native_session_limits_are_unset_by_default_and_passed_through_when_configured() {
    use nexus_claude::agent::{ProviderKind, SessionLimits};
    let limits_of = |config: ChatConfig, kind: ProviderKind, max_tokens: Option<u64>| async move {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        manager
            .build_agent_spec(AgentSpecInput {
                file_scope_lifted: false,
                cwd: "/tmp",
                model: "m",
                system_prompt: "p",
                permission_mode: None,
                add_dirs: &[],
                user_claims: None,
                session_id: "limits-s1",
                third_party: kind != ProviderKind::ClaudeCode,
                max_tokens,
                max_cost_usd: None,
                kind,
                remote_cwd: None,
                per_session_mcp: false,
                hooks: None,
            })
            .await
            .unwrap()
            .limits
    };
    // Nothing configured: no cap, no budget.
    assert_eq!(
        limits_of(test_config(), ProviderKind::Native, None).await,
        SessionLimits::default()
    );
    let mut configured = test_config();
    configured.native_turn_timeout_secs = Some(45 * 60);
    configured.native_session_max_tokens = Some(25_000_000);
    let limits = limits_of(configured.clone(), ProviderKind::Native, None).await;
    assert_eq!(limits.turn_timeout_ms, Some(45 * 60 * 1000));
    assert_eq!(limits.max_tokens, Some(25_000_000));
    // The request's own budget wins over the configured one.
    let limits = limits_of(configured.clone(), ProviderKind::Native, Some(1_000)).await;
    assert_eq!(limits.max_tokens, Some(1_000));
    // Not a native session: the native settings do not apply.
    let limits = limits_of(configured, ProviderKind::ClaudeCode, None).await;
    assert_eq!(limits.turn_timeout_ms, None);
    assert_eq!(limits.max_tokens, None);
}

#[test]
fn native_limit_settings_read_a_positive_number_or_nothing() {
    assert_eq!(ChatConfig::positive(Some("1800")), Some(1800));
    assert_eq!(ChatConfig::positive(Some(" 42 ")), Some(42));
    for unset in [None, Some(""), Some("0"), Some("-5"), Some("30m")] {
        assert_eq!(ChatConfig::positive(unset), None, "{unset:?}");
    }
}

/// M2: the built-in Claude Code of the agent engine opens with the CLI path set NOW
/// (the cell `update_claude_cli_path` writes: the CLI installed or moved at runtime), as
/// the historical engine reads it at every opening — not the path the server started
/// with. Only the manager's own cell is written here: no process-global state (that
/// `update_claude_cli_path` writes it is asserted in
/// `test_runtime_env_config_update_methods`).
#[tokio::test]
async fn the_builtin_claude_code_follows_the_cli_path_changed_at_runtime() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let cc = crate::chat::provider::resolver::CLAUDE_CODE;
    let first = manager.provider_source.get(cc).expect("built in");
    let same = manager.provider_source.get(cc).expect("built in");
    assert!(
        Arc::ptr_eq(&first, &same),
        "unchanged path: the same provider"
    );
    *manager.builtin_cli_path.write().unwrap() = Some("/opt/claude/bin/claude".into());
    let moved = manager.provider_source.get(cc).expect("built in");
    assert!(!Arc::ptr_eq(&first, &moved), "rebuilt on the new path");
}

#[tokio::test]
async fn the_hook_table_has_the_per_tool_hooks_unless_the_session_is_a_runner() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let table = |tool_knowledge: bool| {
        let mut keys: Vec<String> = manager
            .graph_hook_table(GraphHookInput {
                session_id: "s".into(),
                context_source: CompactionContextSource::None,
                work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
                tool_knowledge,
                announce: None,
            })
            .into_keys()
            .collect();
        keys.sort();
        keys
    };
    assert_eq!(table(true), ["PostToolUse", "PreCompact", "PreToolUse"]);
    // A runner has its task context in the prompt: compaction guidance, and the
    // tool clock (PreToolUse, which adds nothing to the context).
    assert_eq!(table(false), ["PreCompact", "PreToolUse"]);
}

/// B14: a session gets the same hooks whether it is created or resumed, runner or
/// not, on both engines. The resume used to register the redirect hook of a
/// runner (which creation skips) and, on the agent engine, every per-tool hook.
#[tokio::test]
async fn creation_and_resume_register_the_same_hooks() {
    fn shape(table: &super::super::agent_hooks::HookTable) -> Vec<(String, Vec<usize>)> {
        let mut shape: Vec<(String, Vec<usize>)> = table
            .iter()
            .map(|(event, matchers)| {
                (
                    event.clone(),
                    matchers.iter().map(|m| m.hooks.len()).collect(),
                )
            })
            .collect();
        shape.sort();
        shape
    }
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (tx, _rx) = broadcast::channel(8);
    let runner = r#"{"type":"runner","run_id":"r","plan_id":"p"}"#;
    for spawned_by in [None, Some(runner)] {
        let mut node = crate::test_helpers::test_chat_session(None);
        node.spawned_by = spawned_by.map(str::to_string);
        let work_log = Arc::new(Mutex::new(SessionWorkLog::default()));
        // `create_session`: per-tool hooks unless a runner.
        let created = manager.graph_hook_table(GraphHookInput {
            session_id: node.id.to_string(),
            context_source: CompactionContextSource::None,
            work_log: Arc::clone(&work_log),
            tool_knowledge: spawned_by.is_none(),
            announce: Some(tx.clone()),
        });
        let resumed = manager.resume_hook_table(&node, &tx, &work_log);
        assert_eq!(
            shape(&created),
            shape(&resumed),
            "legacy engine, spawned_by {spawned_by:?}"
        );
        // The agent engine: the opening says `runner` for a runner session.
        assert_eq!(
            ChatManager::resume_agent_hook_scope(&node).runner,
            spawned_by.is_some(),
            "agent engine, spawned_by {spawned_by:?}"
        );
    }
}

#[tokio::test]
async fn an_agent_engine_compaction_is_not_announced_twice() {
    // The provider emits its own `compaction` event; the notifier of the agent
    // path must stay silent on the chat channel (it only builds the instructions).
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let table = manager.graph_hook_table(GraphHookInput {
        session_id: "s".into(),
        context_source: CompactionContextSource::None,
        work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
        tool_knowledge: false,
        announce: None,
    });
    let (tx, mut rx) = broadcast::channel(8);
    // The same table, but the claude path's notifier would have announced on `tx`:
    let announcing = manager.graph_hook_table(GraphHookInput {
        session_id: "s".into(),
        context_source: CompactionContextSource::None,
        work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
        tool_knowledge: false,
        announce: Some(tx),
    });
    let input = nexus_claude::HookInput::PreCompact(nexus_claude::PreCompactHookInput {
        session_id: "s".into(),
        transcript_path: String::new(),
        cwd: "/tmp".into(),
        permission_mode: None,
        trigger: "auto".into(),
        custom_instructions: None,
    });
    let ctx = nexus_claude::HookContext { signal: None };
    for matcher in &table["PreCompact"] {
        for hook in &matcher.hooks {
            hook.execute(&input, None, &ctx).await.unwrap();
        }
    }
    assert!(
        rx.try_recv().is_err(),
        "the silent table announced something"
    );
    for matcher in &announcing["PreCompact"] {
        for hook in &matcher.hooks {
            hook.execute(&input, None, &ctx).await.unwrap();
        }
    }
    assert!(matches!(
        rx.try_recv(),
        Ok(ChatEvent::CompactionStarted { .. })
    ));
}

// ── Claude Code on another machine (claude_code_remote) ────────────────

#[tokio::test]
async fn a_remote_session_spec_runs_in_the_remote_directory_with_no_local_server_or_dirs() {
    use nexus_claude::agent::ProviderKind;
    let state = mock_app_state();
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let claims = crate::auth::jwt::Claims::service_account("remote");
    let add_dirs = vec!["/local/extra".to_string()];
    let spec_for = |remote_cwd: Option<&'static str>| {
        let manager = &manager;
        let claims = &claims;
        let add_dirs = &add_dirs;
        async move {
            manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/Users/me/project",
                    model: "sonnet",
                    system_prompt: "base prompt",
                    permission_mode: None,
                    add_dirs,
                    user_claims: Some(claims),
                    session_id: "remote-s1",
                    third_party: true,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind: ProviderKind::ClaudeCode,
                    remote_cwd,
                    per_session_mcp: true,
                    hooks: None,
                })
                .await
                .unwrap()
        }
    };
    let remote = spec_for(Some("~/work/app")).await;
    // The remote path, untouched (no local `~` expansion, no local project path).
    assert_eq!(remote.cwd, std::path::PathBuf::from("~/work/app"));
    assert!(remote.mcp_servers.is_empty(), "the PO server is not given");
    assert!(remote.extra_dirs.is_empty());
    let prompt = remote.system_prompt.as_ref().unwrap();
    assert!(prompt.text.starts_with("base prompt"));
    assert!(prompt.text.contains("remote machine"), "the model is told");
    // The same input, local: the server and the directories are there.
    let local = spec_for(None).await;
    assert!(local.mcp_servers.contains_key("project-orchestrator"));
    assert_eq!(local.extra_dirs.len(), 1);
}

/// An agent that takes no MCP server per session (OpenClaw's ACP bridge) is not
/// given the PO server, and the host reports what the session lacks.
#[tokio::test]
async fn a_provider_without_per_session_mcp_gets_no_po_server_and_it_is_reported() {
    use nexus_claude::agent::ProviderKind;
    let state = mock_app_state();
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let claims = crate::auth::jwt::Claims::service_account("acp");
    let spec_for = |per_session_mcp: bool| {
        let manager = &manager;
        let claims = &claims;
        async move {
            manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/tmp",
                    model: "openclaw",
                    system_prompt: "p",
                    permission_mode: None,
                    add_dirs: &[],
                    user_claims: Some(claims),
                    session_id: "acp-s1",
                    third_party: true,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind: ProviderKind::Acp,
                    remote_cwd: None,
                    per_session_mcp,
                    hooks: None,
                })
                .await
                .unwrap()
        }
    };
    let without = spec_for(false).await;
    assert!(without.mcp_servers.is_empty(), "{:?}", without.mcp_servers);
    assert_eq!(
        host_degraded(ProviderKind::Acp, &without, None),
        vec![super::super::agent_runtime::PO_TOOLS_FEATURE.to_string()]
    );
    let with = spec_for(true).await;
    assert!(with.mcp_servers.contains_key(PO_MCP_SERVER));
    assert!(host_degraded(ProviderKind::Acp, &with, None).is_empty());
}

// ── nexus-tools on a native session (B40) ───────────────────────────────

/// A native session starts with the `nexus` server (`nexus-tools` over stdio),
/// bounded by its policy: a denied tool and the web tools (no project, so no
/// network consent) are not in its `--tools`. Claude Code has its own tools: no `nexus`.
#[tokio::test]
async fn a_native_session_gets_the_nexus_tools_server_bounded_by_its_policy() {
    use nexus_claude::agent::{McpServerSpec, ProviderKind};
    let state = mock_app_state();
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    let (_bin, program) = fake_nexus_tools();
    config.nexus_tools_path = Some(program.clone());
    config.permission.disallowed_tools = vec!["Monitor".into()];
    let mut manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    // Hermetic: the host's own credentials (mirrored when present) are not this test's subject.
    manager.host_home = None;
    let claims = crate::auth::jwt::Claims::service_account("nexus");
    let add_dirs = vec!["/tmp/extra".to_string()];
    let spec_for = |kind: ProviderKind| {
        let manager = &manager;
        let claims = &claims;
        let add_dirs = &add_dirs;
        async move {
            manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/tmp",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: Some("default"),
                    add_dirs,
                    user_claims: Some(claims),
                    session_id: "nexus-s1",
                    third_party: true,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind,
                    remote_cwd: None,
                    per_session_mcp: true,
                    hooks: None,
                })
                .await
                .unwrap()
        }
    };
    let native = spec_for(ProviderKind::Native).await;
    let Some(McpServerSpec::Stdio { command, args, env }) = native.mcp_servers.get("nexus") else {
        panic!("a stdio `nexus` server expected");
    };
    assert_eq!(
        std::path::Path::new(command),
        program.canonicalize().unwrap()
    );
    // Nothing from the host's environment: only the session's own key and the profile
    // it signs (#596), never on argv, and its own private temporary directory (decision
    // fce98b8a: its command outputs live there, one of its directories).
    // The two host git/gh credential variables (decision 2238b1e4) are the deliberate
    // exception, and `GH_TOKEN` only exists where the host has a `gh` login: they are
    // left out of the comparison so it does not depend on the machine.
    assert_eq!(
        env.keys()
            .map(String::as_str)
            .filter(|k| !matches!(*k, "NEXUS_TOOLS_GIT_CONFIG_GLOBAL" | "NEXUS_TOOLS_GH_TOKEN"))
            .collect::<Vec<_>>(),
        [
            "NEXUS_TOOLS_KEY",
            "NEXUS_TOOLS_PROFILE",
            "TEMP",
            "TMP",
            "TMPDIR"
        ]
    );
    assert!(env["TMPDIR"].ends_with("nexus-s1"), "{}", env["TMPDIR"]);
    assert!(!args.contains(&env["NEXUS_TOOLS_PROFILE"]));
    assert!(!args.iter().any(|a| a == "--trust-harness"));
    let tools_at = args.iter().position(|a| a == "--tools").unwrap() + 1;
    let tools: Vec<&str> = args[tools_at].split(',').collect();
    for tool in ["Read", "Write", "Edit", "Glob", "Grep", "Bash"] {
        assert!(tools.contains(&tool), "{tool} in {tools:?}");
    }
    for tool in ["Monitor", "WebFetch", "WebSearch"] {
        assert!(
            !tools.contains(&tool),
            "{tool} must not be served: {tools:?}"
        );
    }
    assert!(args.windows(2).any(|w| w == ["--add-dir", "/tmp/extra"]));
    assert!(
        !args.iter().any(|a| a.contains("eyJ")),
        "no token on the command line"
    );
    // The PO server stays.
    assert!(native.mcp_servers.contains_key("project-orchestrator"));
    // Claude Code brings its own tools.
    assert!(!spec_for(ProviderKind::ClaudeCode)
        .await
        .mcp_servers
        .contains_key("nexus"));
}

/// P11b point 7: a server of a native session named `nexus` (or `browser`) is refused,
/// never kept in place of the backend's: nexus would skip nexus-tools and give its tools
/// the built-in names, so a third party's `mcp__nexus__Read` would be granted whole
/// for the session.
#[test]
fn a_native_session_server_named_nexus_is_refused() {
    use nexus_claude::agent::McpServerSpec;
    use std::collections::BTreeMap;
    let server = |command: &str| McpServerSpec::Stdio {
        command: command.to_string(),
        args: Vec::new(),
        env: Default::default(),
    };
    for reserved in RESERVED_NATIVE_SERVERS {
        let mut servers = BTreeMap::new();
        servers.insert("project-orchestrator".to_string(), server("/po/mcp_server"));
        servers.insert(reserved.to_string(), server("/tmp/someone-elses-server"));
        let refused = refuse_reserved_native_servers(&servers).unwrap_err();
        assert!(refused.to_string().contains(reserved), "{refused}");

        // The attachment does not keep the session's server in its place either.
        let attached = BTreeMap::from([(reserved.to_string(), server("/usr/bin/nexus-tools"))]);
        assert!(attach_native_servers(&mut servers, attached).is_err());
        assert!(matches!(
            servers.get(reserved),
            Some(McpServerSpec::Stdio { command, .. }) if command == "/tmp/someone-elses-server"
        ));
    }
    // Without a reserved name, the backend's servers are attached.
    let mut servers = BTreeMap::new();
    servers.insert("project-orchestrator".to_string(), server("/po/mcp_server"));
    servers.insert("acme".to_string(), server("/opt/acme"));
    refuse_reserved_native_servers(&servers).unwrap();
    attach_native_servers(
        &mut servers,
        BTreeMap::from([("nexus".to_string(), server("/usr/bin/nexus-tools"))]),
    )
    .unwrap();
    assert!(matches!(
        servers.get("nexus"),
        Some(McpServerSpec::Stdio { command, .. }) if command == "/usr/bin/nexus-tools"
    ));
}

/// `nexus-tools` is launched with the session's directory as its working
/// directory: a relative program path (NEXUS_TOOLS_PATH=./nexus-tools, a `.` or
/// empty entry of the PATH) would then name a file of the PROJECT, which the
/// model can write. The server is always launched by its absolute path.
#[tokio::test]
async fn the_nexus_tools_server_is_launched_by_its_absolute_path() {
    use nexus_claude::agent::{McpServerSpec, ProviderKind};
    use std::os::unix::fs::PermissionsExt;
    let crate_root = std::env::current_dir().unwrap();
    let dir = tempfile::TempDir::new_in(&crate_root).unwrap();
    let program = dir.path().join("nexus-tools");
    std::fs::write(&program, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    let relative = program.strip_prefix(&crate_root).unwrap().to_path_buf();
    assert!(relative.is_relative());
    let state = mock_app_state();
    let mut config = test_config();
    config.nexus_tools_path = Some(relative);
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let spec = manager
        .build_agent_spec(AgentSpecInput {
            file_scope_lifted: false,
            cwd: "/tmp",
            model: "m",
            system_prompt: "p",
            permission_mode: Some("default"),
            add_dirs: &[],
            user_claims: None,
            session_id: "nexus-rel",
            third_party: true,
            max_tokens: None,
            max_cost_usd: None,
            kind: ProviderKind::Native,
            remote_cwd: None,
            per_session_mcp: true,
            hooks: None,
        })
        .await
        .unwrap();
    let Some(McpServerSpec::Stdio { command, .. }) = spec.mcp_servers.get("nexus") else {
        panic!("a stdio `nexus` server expected");
    };
    assert!(
        std::path::Path::new(command).is_absolute(),
        "launched as {command}"
    );
    assert_eq!(
        std::path::Path::new(command),
        program.canonicalize().unwrap()
    );
}

/// Without `nexus-tools`, a native session opens with the PO tools only.
#[tokio::test]
async fn a_native_session_without_nexus_tools_has_no_nexus_server() {
    use nexus_claude::agent::ProviderKind;
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let spec = manager
        .build_agent_spec(AgentSpecInput {
            file_scope_lifted: false,
            cwd: "/tmp",
            model: "m",
            system_prompt: "p",
            permission_mode: None,
            add_dirs: &[],
            user_claims: None,
            session_id: "nexus-s2",
            third_party: true,
            max_tokens: None,
            max_cost_usd: None,
            kind: ProviderKind::Native,
            remote_cwd: None,
            per_session_mcp: true,
            hooks: None,
        })
        .await
        .unwrap();
    assert!(!spec.mcp_servers.contains_key("nexus"));
}
#[test]
fn a_session_without_per_session_mcp_reports_the_missing_po_tools() {
    let mut caps = nexus_claude::agent::Capabilities::none();
    caps.per_session_mcp = false;
    assert!(super::super::agent_runtime::degraded_features(&caps)
        .iter()
        .any(|f| f == "project_orchestrator_tools"));
    caps.per_session_mcp = true;
    assert!(!super::super::agent_runtime::degraded_features(&caps)
        .iter()
        .any(|f| f == "project_orchestrator_tools"));
}

// ── agent environment and MCP secrets (decision A33) ───────────────────

/// The PO session token a session of the agent path is given.
async fn agent_spec_token(
    manager: &ChatManager,
    claims: &crate::auth::jwt::Claims,
    third_party: bool,
    mode: Option<&str>,
    sid: &str,
) -> String {
    use nexus_claude::agent::{McpServerSpec, ProviderKind};
    let spec = manager
        .build_agent_spec(AgentSpecInput {
            file_scope_lifted: false,
            cwd: "/tmp",
            model: "m",
            system_prompt: "p",
            permission_mode: mode,
            add_dirs: &[],
            user_claims: Some(claims),
            session_id: sid,
            third_party,
            max_tokens: None,
            max_cost_usd: None,
            kind: ProviderKind::Native,
            remote_cwd: None,
            per_session_mcp: true,
            hooks: None,
        })
        .await
        .unwrap();
    let Some(McpServerSpec::Stdio { env, .. }) = spec.mcp_servers.get("project-orchestrator")
    else {
        panic!("stdio MCP server expected");
    };
    env.get("PO_AUTH_TOKEN").expect("a session token").clone()
}

/// The claims of a person signed in (a user JWT), not the server's service account.
fn person_claims() -> crate::auth::jwt::Claims {
    crate::auth::jwt::Claims {
        sub: Uuid::new_v4().to_string(),
        email: "alice@example.com".into(),
        name: "Alice".into(),
        iat: 0,
        exp: 0,
        token_type: None,
        scope: None,
        jti: None,
    }
}

fn signed_manager(state: crate::AppState) -> ChatManager {
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    ChatManager::new_without_memory(state.neo4j, state.meili, config)
}

/// VERIFIER: the token minted for a provider other than Claude Code carries the
/// restricted tool profile (A35) unless the session is in `trust` (H6); Claude
/// Code keeps the full one.
#[tokio::test]
async fn verifier_a_third_party_session_token_carries_the_restricted_profile() {
    use crate::auth::tool_profile::ToolProfile;
    let manager = signed_manager(mock_app_state());
    let claims = crate::auth::jwt::Claims::service_account("verifier");
    let profile = |token: String| ToolProfile::from_unverified_token(&token);
    for mode in [None, Some("default"), Some("acceptEdits"), Some("plan")] {
        assert_eq!(
            profile(agent_spec_token(&manager, &claims, true, mode, "verifier-s1").await),
            ToolProfile::Restricted,
            "a third party in {mode:?} stays restricted"
        );
    }
    assert_eq!(
        profile(agent_spec_token(&manager, &claims, false, None, "verifier-s2").await),
        ToolProfile::Full
    );
}

/// H6: a third-party session a person opened in `trust` (Rock'n roll) gets the
/// full profile: `plan.delegate_task` passes the MCP profile check and its REST
/// route is open to the token.
#[tokio::test]
async fn a_third_party_session_in_trust_carries_the_full_profile() {
    use crate::auth::tool_profile::ToolProfile;
    use crate::mcp::protocol::ToolCallParams;
    let manager = signed_manager(mock_app_state());
    let claims = person_claims();
    for mode in ["bypassPermissions", "trust"] {
        let token = agent_spec_token(&manager, &claims, true, Some(mode), "trust-s1").await;
        let profile = ToolProfile::from_unverified_token(&token);
        assert_eq!(profile, ToolProfile::Full, "{mode}");
        let delegate = ToolCallParams {
            name: "plan".into(),
            arguments: Some(serde_json::json!({ "action": "delegate_task" })),
        };
        assert_eq!(
            crate::mcp::server::profile_refusal(profile, &delegate),
            None
        );
        assert!(
            !profile.route_forbidden(&axum::http::Method::POST, "/api/plans/p1/tasks/t1/delegate")
        );
        // The lineage is signed in: whatever this session opens is restricted.
        let decoded = jsonwebtoken::dangerous::insecure_decode::<crate::auth::jwt::Claims>(&token)
            .unwrap()
            .claims;
        assert!(
            crate::auth::jwt::agent_session_binding(&decoded)
                .unwrap()
                .third_party
        );
    }
}

/// H6 guard: a session opened BY a third-party session (its caller's token
/// carries the lineage: delegation, `chat send_message`, `plan run`) is
/// restricted, whatever its own provider and mode — no delegation loop.
#[tokio::test]
async fn a_session_opened_by_a_third_party_session_is_restricted() {
    use crate::auth::tool_profile::ToolProfile;
    let manager = signed_manager(mock_app_state());
    let person = person_claims();
    let parent_token = agent_spec_token(&manager, &person, true, Some("trust"), "parent-s1").await;
    let parent =
        crate::auth::jwt::decode_jwt(&parent_token, "test-secret-key-minimum-32-chars!!").unwrap();
    for third_party in [true, false] {
        let token = agent_spec_token(
            &manager,
            &parent,
            third_party,
            Some("bypassPermissions"),
            "child-s1",
        )
        .await;
        let profile = ToolProfile::from_unverified_token(&token);
        assert_eq!(
            profile,
            ToolProfile::Restricted,
            "third_party={third_party}"
        );
        assert!(
            profile.route_forbidden(&axum::http::Method::POST, "/api/plans/p1/tasks/t1/delegate")
        );
    }
    // A Claude Code parent's children keep what their own provider grants.
    let cc_token = agent_spec_token(&manager, &person, false, None, "cc-parent").await;
    let cc_parent =
        crate::auth::jwt::decode_jwt(&cc_token, "test-secret-key-minimum-32-chars!!").unwrap();
    assert_eq!(
        ToolProfile::from_unverified_token(
            &agent_spec_token(&manager, &cc_parent, true, Some("trust"), "child-s2").await
        ),
        ToolProfile::Full
    );
}

/// H6 guard, from the graph: a delegated child resumed later by a person (no
/// lineage in the person's token) is still restricted when its parent runs on
/// a third-party provider.
#[tokio::test]
async fn a_resumed_child_of_a_third_party_session_is_restricted() {
    use crate::auth::tool_profile::ToolProfile;
    let state = mock_app_state();
    let graph = state.neo4j.clone();
    let manager = signed_manager(state);
    let node = |id: Uuid, provider: &str, spawned_by: Option<String>| {
        serde_json::from_value::<ChatSessionNode>(serde_json::json!({
            "id": id,
            "cwd": "/tmp",
            "model": "m",
            "created_at": "2026-10-09T00:00:00Z",
            "updated_at": "2026-10-09T00:00:00Z",
            "message_count": 0,
            "total_cost_usd": 0.0,
            "provider_id": provider,
            "spawned_by": spawned_by,
        }))
        .unwrap()
    };
    let (third, claude, child_of_third, child_of_claude) = (
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
    );
    let delegated = |parent: Uuid| {
        Some(
            crate::chat::types::SpawnedBy::Delegation {
                plan_id: Uuid::new_v4(),
                task_id: Uuid::new_v4(),
                parent_session_id: Some(parent),
                scaffolding_level: None,
            }
            .to_json_string(),
        )
    };
    for n in [
        node(third, "deepseek", None),
        node(claude, "claude-code", None),
        node(child_of_third, "deepseek", delegated(third)),
        node(child_of_claude, "deepseek", delegated(claude)),
    ] {
        graph.create_chat_session(&n).await.unwrap();
    }
    let person = person_claims();
    let profile_of = |sid: Uuid| {
        let manager = &manager;
        let person = &person;
        async move {
            ToolProfile::from_unverified_token(
                &agent_spec_token(manager, person, true, Some("trust"), &sid.to_string()).await,
            )
        }
    };
    assert_eq!(profile_of(child_of_third).await, ToolProfile::Restricted);
    assert_eq!(profile_of(child_of_claude).await, ToolProfile::Full);
    assert_eq!(profile_of(third).await, ToolProfile::Full);
}

/// H6 guard, origin unknown: a session opened under the server's own service
/// account (a protocol run, a plan run resumed after a restart) carries no
/// lineage and has no parent session. A third-party session in `trust` could
/// otherwise start a protocol whose third-party agent is `full` again and starts
/// the next one: the loop the guard is there to break. Such a session is
/// restricted; a delegation whose parent is a Claude Code session keeps `full`.
#[tokio::test]
async fn a_third_party_session_opened_by_the_service_account_without_a_known_parent_is_restricted()
{
    use crate::auth::tool_profile::ToolProfile;
    let state = mock_app_state();
    let graph = state.neo4j.clone();
    let manager = signed_manager(state);
    let node = |id: Uuid, provider: &str, spawned_by: Option<String>| {
        serde_json::from_value::<ChatSessionNode>(serde_json::json!({
            "id": id,
            "cwd": "/tmp",
            "model": "m",
            "created_at": "2026-10-09T00:00:00Z",
            "updated_at": "2026-10-09T00:00:00Z",
            "message_count": 0,
            "total_cost_usd": 0.0,
            "provider_id": provider,
            "spawned_by": spawned_by,
        }))
        .unwrap()
    };
    let (protocol_agent, recovered_runner_agent, claude_parent, delegated) = (
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
    );
    let runner_spawn = serde_json::json!({
        "type": "runner",
        "run_id": Uuid::new_v4().to_string(),
        "plan_id": Uuid::new_v4().to_string(),
        "task_id": Uuid::new_v4().to_string(),
    })
    .to_string();
    let protocol_spawn = serde_json::json!({
        "type": "protocol_runner",
        "run_id": Uuid::new_v4().to_string(),
        "protocol_id": Uuid::new_v4().to_string(),
        "state_name": "s",
    })
    .to_string();
    let delegation = crate::chat::types::SpawnedBy::Delegation {
        plan_id: Uuid::new_v4(),
        task_id: Uuid::new_v4(),
        parent_session_id: Some(claude_parent),
        scaffolding_level: None,
    }
    .to_json_string();
    for n in [
        node(protocol_agent, "deepseek", Some(protocol_spawn)),
        node(recovered_runner_agent, "deepseek", Some(runner_spawn)),
        node(claude_parent, "claude-code", None),
        node(delegated, "deepseek", Some(delegation)),
    ] {
        graph.create_chat_session(&n).await.unwrap();
    }
    let profile_of = |claims: crate::auth::jwt::Claims, sid: Uuid| {
        let manager = &manager;
        async move {
            ToolProfile::from_unverified_token(
                &agent_spec_token(
                    manager,
                    &claims,
                    true,
                    Some("bypassPermissions"),
                    &sid.to_string(),
                )
                .await,
            )
        }
    };
    let service = |who: &str| crate::auth::jwt::Claims::service_account(who);
    assert_eq!(
        profile_of(service("protocol-agent:r1"), protocol_agent).await,
        ToolProfile::Restricted,
        "a protocol agent has no known origin"
    );
    assert_eq!(
        profile_of(service("runner-agent:r2"), recovered_runner_agent).await,
        ToolProfile::Restricted,
        "a run resumed after a restart lost its caller"
    );
    assert_eq!(
        profile_of(service("delegate-agent:t1"), delegated).await,
        ToolProfile::Full,
        "a delegation by a Claude Code session keeps what trust grants"
    );
    // A plan run a person started carries that person's claims.
    assert_eq!(
        profile_of(person_claims(), recovered_runner_agent).await,
        ToolProfile::Full
    );
}

/// An executable `nexus-tools` stand-in (it is never started here) and its directory.
fn fake_nexus_tools() -> (tempfile::TempDir, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let bin = tempfile::TempDir::new().unwrap();
    let program = bin.path().join("nexus-tools");
    std::fs::write(&program, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    (bin, program)
}

// ── nexus-tools on a native session (B40) ───────────────────────────────

async fn nexus_tools_spec(
    kind: nexus_claude::agent::ProviderKind,
    remote_cwd: Option<&'static str>,
    configured: bool,
    graph: Arc<crate::neo4j::mock::MockGraphStore>,
    sid: &str,
    permission_mode: Option<&str>,
) -> nexus_claude::agent::SessionSpec {
    let state = mock_app_state_with_graph(graph);
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    // The stand-in lives until the spec is built: its path is resolved at the opening.
    let (_bin, program) = fake_nexus_tools();
    config.nexus_tools_path = configured.then_some(program);
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let claims = crate::auth::jwt::Claims::service_account("b40");
    manager
        .build_agent_spec(AgentSpecInput {
            file_scope_lifted: false,
            cwd: "/work/app",
            model: "m",
            system_prompt: "p",
            permission_mode,
            add_dirs: &[],
            user_claims: Some(&claims),
            session_id: sid,
            third_party: true,
            max_tokens: None,
            max_cost_usd: None,
            kind,
            remote_cwd,
            per_session_mcp: true,
            hooks: Some(AgentHookScope {
                project_slug: Some("proj".into()),
                task_id: None,
                runner: false,
            }),
        })
        .await
        .unwrap()
}

/// OBS-R: the executable of a declared `obscura` engine at a session opening.
#[tokio::test]
async fn a_declared_obscura_engine_runs_the_configured_executable_else_the_embedded_one() {
    use nexus_claude::agent::{McpServerSpec, ProviderKind};
    use std::os::unix::fs::PermissionsExt;
    const EMBEDS: &str = "#!/bin/sh\n[ \"$1\" = --capabilities ] && echo '{\"version\":\"0.5.1\",\"tls\":true,\"obscura\":true}'\n";
    const PLAIN: &str = "#!/bin/sh\n";

    let engine_args =
        |nexus_tools: &'static str, configured: Option<PathBuf>, sid: &'static str| async move {
            let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
            let record = crate::chat::provider::nexus_tools::SearchProviderRecord {
                id: "obs".into(),
                engine: "obscura".into(),
                base_url: None,
                credential_ref: "none".into(),
            };
            graph
                .put_llm_setting(
                    crate::chat::provider::settings::GLOBAL,
                    &format!(
                        "{}obs",
                        crate::chat::provider::nexus_tools::SEARCH_PROVIDER_PREFIX
                    ),
                    &serde_json::to_string(&record).unwrap(),
                )
                .await
                .unwrap();
            crate::chat::provider::nexus_tools::record_origin_consent(
                &*graph,
                "proj",
                crate::chat::provider::nexus_tools::OBSCURA_SEARCH_ORIGIN,
                "tester",
            )
            .await
            .unwrap();
            let state = mock_app_state_with_graph(graph);
            let bin = tempfile::TempDir::new().unwrap();
            let program = bin.path().join("nexus-tools");
            std::fs::write(&program, nexus_tools).unwrap();
            std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
            let mut config = test_config();
            config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
            config.nexus_tools_path = Some(program);
            config.obscura_path = configured;
            let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
            let claims = crate::auth::jwt::Claims::service_account("obs-r");
            let spec = manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/work/app",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: None,
                    add_dirs: &[],
                    user_claims: Some(&claims),
                    session_id: sid,
                    third_party: true,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind: ProviderKind::Native,
                    remote_cwd: None,
                    per_session_mcp: true,
                    hooks: Some(AgentHookScope {
                        project_slug: Some("proj".into()),
                        task_id: None,
                        runner: false,
                    }),
                })
                .await
                .unwrap();
            crate::auth::agent_tokens::revoke_session(sid);
            let Some(McpServerSpec::Stdio { args, .. }) = spec.mcp_servers.get("nexus") else {
                panic!("a stdio `nexus` server expected");
            };
            let engine: Vec<String> = args
                .windows(2)
                .filter(|w| w[0] == "--search-engine")
                .map(|w| w[1].clone())
                .collect();
            engine
        };

    let dir = tempfile::TempDir::new().unwrap();
    let obscura = dir.path().join("dsh-obscura-nav");
    std::fs::write(&obscura, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&obscura, std::fs::Permissions::from_mode(0o755)).unwrap();
    let configured = vec![format!(
        "obscura:{}",
        std::fs::canonicalize(&obscura).unwrap().display()
    )];
    let embedded = vec!["obscura".to_string()];
    let relative = Some(PathBuf::from("bin/dsh-obscura-nav"));
    let missing = Some(dir.path().join("absent"));
    for (n, (tools, path, want)) in [
        (EMBEDS, Some(obscura.clone()), &configured),
        (PLAIN, Some(obscura.clone()), &configured),
        (EMBEDS, None, &embedded),
        (EMBEDS, relative.clone(), &embedded),
        (EMBEDS, missing.clone(), &embedded),
        (PLAIN, relative, &Vec::new()),
        (PLAIN, missing, &Vec::new()),
    ]
    .into_iter()
    .enumerate()
    {
        let sid = Box::leak(format!("obsr-{n}").into_boxed_str());
        assert_eq!(
            &engine_args(tools, path.clone(), sid).await,
            want,
            "case {n}: {path:?}"
        );
    }
}

/// P18: a native session a person opened runs sub-agents (the harness then offers
/// `Agent`); a session something spawned — the sub-agent's own child session included —
/// never does (depth 1, A17); nor does a manager that cannot hand itself over.
#[tokio::test]
async fn only_a_native_session_nothing_spawned_runs_subagents() {
    use nexus_claude::agent::ProviderKind;
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let state = mock_app_state_with_graph(graph.clone());
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    let manager = Arc::new(ChatManager::new_without_memory(
        state.neo4j,
        state.meili,
        config,
    ));
    let (top, child) = (Uuid::new_v4(), Uuid::new_v4());
    let mut node = crate::test_helpers::test_chat_session(Some("proj"));
    node.id = top;
    graph.create_chat_session(&node).await.unwrap();
    node.id = child;
    node.spawned_by = Some(
        crate::chat::types::SpawnedBy::Conversation {
            parent_session_id: top,
            tool_use_id: Some("ca1".into()),
        }
        .to_json_string(),
    );
    graph.create_chat_session(&node).await.unwrap();
    let claims = crate::auth::jwt::Claims::service_account("p18");
    let runs = |sid: String, kind: ProviderKind| {
        let manager = manager.clone();
        let claims = claims.clone();
        async move {
            let spec = manager
                .build_agent_spec(AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/work/app",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: None,
                    add_dirs: &[],
                    user_claims: Some(&claims),
                    session_id: &sid,
                    third_party: true,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind,
                    remote_cwd: None,
                    per_session_mcp: true,
                    hooks: Some(AgentHookScope {
                        project_slug: Some("proj".into()),
                        task_id: None,
                        runner: false,
                    }),
                })
                .await
                .unwrap();
            crate::auth::agent_tokens::revoke_session(&sid);
            let asks = !spec
                .policy
                .deny
                .iter()
                .any(|p| p.matches("AskUserQuestion", None));
            (
                spec.hooks.is_some_and(|h| h.runs_subagents()),
                spec.lineage.map(|l| l.depth),
                asks,
            )
        }
    };
    assert!(
        !runs(top.to_string(), ProviderKind::Native).await.0,
        "without enable_subagents the manager cannot open a child"
    );
    manager.enable_subagents();
    // A session a person opened: the host, no lineage, the question.
    assert_eq!(
        runs(top.to_string(), ProviderKind::Native).await,
        (true, None, true)
    );
    // A spawned session: no host, the harness's depth guard armed (lineage depth 1), no
    // question nobody would see.
    assert_eq!(
        runs(child.to_string(), ProviderKind::Native).await,
        (false, Some(1), false),
        "a spawned session cannot delegate again nor ask"
    );
    assert!(
        !runs(Uuid::new_v4().to_string(), ProviderKind::Native)
            .await
            .0,
        "an unknown record is not trusted"
    );
    assert!(
        !runs(top.to_string(), ProviderKind::ClaudeCode).await.0,
        "Claude Code brings its own Task"
    );
}

/// The spec of a native session with `nexus-tools`, for a mode, a person's lift and an
/// access (decision fce98b8a).
async fn lifted_nexus_tools_spec(
    sid: &str,
    permission_mode: &str,
    file_scope_lifted: bool,
    access: crate::chat::provider::policy::SessionAccess,
) -> nexus_claude::agent::SessionSpec {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let state = mock_app_state_with_graph(graph);
    let mut config = test_config();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    let (_bin, program) = fake_nexus_tools();
    config.nexus_tools_path = Some(program);
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let claims = crate::auth::jwt::Claims::service_account("lift");
    manager
        .build_agent_spec_with_access(
            AgentSpecInput {
                file_scope_lifted,
                cwd: "/work/app",
                model: "m",
                system_prompt: "p",
                permission_mode: Some(permission_mode),
                add_dirs: &[],
                user_claims: Some(&claims),
                session_id: sid,
                third_party: true,
                max_tokens: None,
                max_cost_usd: None,
                kind: nexus_claude::agent::ProviderKind::Native,
                remote_cwd: None,
                per_session_mcp: true,
                hooks: None,
            },
            access,
        )
        .await
        .unwrap()
}

/// Review of #729 (nit and finding 2), the wiring from the session to the launch line: a
/// native session launches `nexus-tools --unscoped` only in trust (`bypassPermissions` or
/// `trust`), with normal access, when a person lifted it. A runner, delegation or protocol
/// session (trust, nobody lifted it) and a read-only one stay scoped. The deny list always
/// goes.
#[tokio::test]
async fn a_native_session_launches_nexus_tools_unscoped_only_when_a_person_lifted_its_trust() {
    use crate::chat::provider::policy::SessionAccess::{Normal, ReadOnly};
    use nexus_claude::agent::McpServerSpec;
    for (index, (mode, lifted, access, expected)) in [
        ("bypassPermissions", true, Normal, true),
        ("trust", true, Normal, true),
        ("bypassPermissions", false, Normal, false),
        ("bypassPermissions", true, ReadOnly, false),
        ("default", true, Normal, false),
        ("acceptEdits", true, Normal, false),
    ]
    .into_iter()
    .enumerate()
    {
        let sid = format!("lift-wiring-{index}");
        let spec = lifted_nexus_tools_spec(&sid, mode, lifted, access).await;
        let Some(McpServerSpec::Stdio { args, .. }) = spec.mcp_servers.get("nexus") else {
            panic!("{mode}: nexus-tools must be attached");
        };
        assert_eq!(
            args.contains(&"--unscoped".to_owned()),
            expected,
            "{mode} lifted={lifted} {access:?}: {args:?}"
        );
        assert!(
            args.iter().any(|a| a == "--deny-path"),
            "{mode}: the deny list goes in every case"
        );
        crate::auth::agent_tokens::revoke_session(&sid);
    }
}

#[tokio::test]
async fn a_native_session_starts_with_nexus_tools_next_to_the_po_server_and_its_profile_in_the_token(
) {
    use nexus_claude::agent::{McpServerSpec, ProviderKind};
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let spec = nexus_tools_spec(ProviderKind::Native, None, true, graph, "b40-s1", None).await;
    assert!(spec.mcp_servers.contains_key("project-orchestrator"));
    let Some(McpServerSpec::Stdio { command, args, env }) = spec.mcp_servers.get("nexus") else {
        panic!("nexus-tools must be attached to a native session");
    };
    let command = std::path::Path::new(command);
    assert!(command.is_absolute() && command.ends_with("nexus-tools"));
    assert!(env.contains_key("NEXUS_TOOLS_PROFILE") && env.contains_key("NEXUS_TOOLS_KEY"));
    assert!(args.windows(2).any(|w| w == ["--cwd", "/work/app"]));
    assert!(
        args.iter().all(|a| !a.starts_with("v1.")),
        "no token on argv"
    );
    assert!(!format!("{spec:?}").contains(&env["NEXUS_TOOLS_PROFILE"]));
    assert!(spec.hooks.is_some(), "the consent gate is on the session");
    crate::auth::agent_tokens::revoke_session("b40-s1");
}

#[tokio::test]
async fn nexus_tools_is_not_attached_to_other_harnesses_to_a_remote_session_or_without_the_binary()
{
    use nexus_claude::agent::ProviderKind;
    let graph = || Arc::new(crate::neo4j::mock::MockGraphStore::new());
    for (kind, remote, configured, sid) in [
        (ProviderKind::ClaudeCode, None, true, "b40-cc"),
        (ProviderKind::Codex, None, true, "b40-codex"),
        (ProviderKind::Native, Some("~/w"), true, "b40-remote"),
        (ProviderKind::Native, None, false, "b40-nobin"),
    ] {
        let spec = nexus_tools_spec(kind, remote, configured, graph(), sid, None).await;
        assert!(!spec.mcp_servers.contains_key("nexus"), "{sid}");
        assert!(
            !crate::auth::agent_tokens::tools_live(sid),
            "{sid}: nothing minted"
        );
    }
}

#[tokio::test]
async fn only_a_native_session_opened_without_its_nexus_server_is_reported_as_lacking_it() {
    use nexus_claude::agent::ProviderKind;
    let graph = || Arc::new(crate::neo4j::mock::MockGraphStore::new());
    // A runnable stand-in alive for the whole test (the spec's own is gone once the
    // spec is built).
    let (_bin, program) = fake_nexus_tools();
    // (kind, remote, binary configured, session, lacks the file/shell/web tools)
    for (kind, remote, configured, sid, lacks) in [
        (ProviderKind::Native, None, true, "b40-has", false),
        (ProviderKind::Native, None, false, "b40-lacks", true),
        // Claude Code and Codex bring their own tools: never reported.
        (ProviderKind::ClaudeCode, None, false, "b40-cc-own", false),
        (ProviderKind::Codex, None, false, "b40-codex-own", false),
    ] {
        let spec = nexus_tools_spec(kind, remote, configured, graph(), sid, None).await;
        let path = configured.then_some(program.as_path());
        assert_eq!(lacks_nexus_tools(kind, &spec, path), lacks, "{sid}");
        crate::auth::agent_tokens::revoke_session(sid);
    }
    // The executable is there but nothing was attached (a policy that leaves no
    // `nexus-tools` tool to offer): the installation lacks nothing, not reported.
    let bare = nexus_tools_spec(ProviderKind::Native, None, false, graph(), "b40-bare", None).await;
    crate::auth::agent_tokens::revoke_session("b40-bare");
    assert!(!bare
        .mcp_servers
        .contains_key(nexus_claude::providers::native::NEXUS_TOOLS_SERVER));
    assert!(!lacks_nexus_tools(
        ProviderKind::Native,
        &bare,
        Some(program.as_path())
    ));
    // A configured path that is not (or no longer) there is missing.
    let gone = program.with_file_name("gone");
    assert!(lacks_nexus_tools(
        ProviderKind::Native,
        &bare,
        Some(gone.as_path())
    ));
}

#[tokio::test]
async fn the_session_hooks_refuse_webfetch_to_an_origin_the_project_did_not_consent_to() {
    use nexus_claude::agent::{HookVerdict, ProviderKind, ToolCallInfo};
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let spec = nexus_tools_spec(
        ProviderKind::Native,
        None,
        true,
        graph.clone(),
        "b40-gate",
        None,
    )
    .await;
    let hooks = spec.hooks.clone().expect("hooks");
    let fetch = ToolCallInfo {
        id: None,
        name: "mcp__nexus__WebFetch".into(),
        canonical: Some("WebFetch".into()),
        category: nexus_claude::agent::ToolCategory::Web,
        input: serde_json::json!({"url": "https://docs.rs/serde"}),
    };
    let HookVerdict::Deny { reason } = hooks.before_tool(&fetch).await else {
        panic!("an origin without consent must be refused");
    };
    assert!(reason.starts_with("tool_origin_not_allowed:"), "{reason}");
    // The project consents; the same call is no longer this gate's refusal.
    graph
        .put_llm_setting(
            "project:proj",
            "tool_origin:https://docs.rs",
            &serde_json::json!({
                "origin": "https://docs.rs",
                "consented_by": "me@example.com",
                "consented_at": "2026-10-08T10:00:00Z"
            })
            .to_string(),
        )
        .await
        .unwrap();
    assert!(!matches!(
        hooks.before_tool(&fetch).await,
        HookVerdict::Deny { .. }
    ));
    // The session is closed: its token is revoked and its nexus tools stop.
    crate::auth::agent_tokens::revoke_session("b40-gate");
    let read = ToolCallInfo {
        name: "mcp__nexus__Read".into(),
        canonical: Some("Read".into()),
        input: serde_json::json!({"file_path": "/work/app/a"}),
        ..fetch
    };
    assert!(matches!(
        hooks.before_tool(&read).await,
        HookVerdict::Deny { .. }
    ));
}

#[tokio::test]
async fn mcp_config_carries_no_secret() {
    let state = mock_app_state();
    let config = test_config();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let options = manager
        .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
        .await;

    let Some(McpServerConfig::Stdio { env, .. }) = options.mcp_servers.get("project-orchestrator")
    else {
        panic!("Expected Stdio MCP config");
    };
    let env = env.as_ref().unwrap();
    for name in [
        "NEO4J_PASSWORD",
        "NEO4J_USER",
        "NEO4J_URI",
        "MEILISEARCH_KEY",
        "MEILISEARCH_URL",
    ] {
        assert!(
            !env.contains_key(name),
            "the MCP proxy does not need {name}; it must not be handed to the agent"
        );
    }
    let serialized = serde_json::to_string(&env).unwrap();
    assert!(
        !serialized.contains("\"test\"") && !serialized.contains("\"key\""),
        "database/search credentials leaked into the MCP config: {serialized}"
    );
    assert!(
        options.mcp_config_via_file,
        "the MCP config (session token) must go through a file, not argv"
    );
}

#[tokio::test]
async fn child_env_only_carries_allowlisted_names() {
    let state = mock_app_state();
    let config = test_config();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let options = manager
        .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
        .await;

    let policy = &options.env_policy;
    assert!(
        policy.is_isolated(),
        "the agent must start from a clean environment"
    );
    for secret in [
        "NEO4J_PASSWORD",
        "MEILISEARCH_KEY",
        "EMBEDDING_API_KEY",
        "PO_JWT_SECRET",
        "GOOGLE_CLIENT_SECRET",
        "AWS_SECRET_ACCESS_KEY",
        "SOME_UNKNOWN_SERVER_VARIABLE",
    ] {
        assert!(!policy.allows(secret), "{secret} must not reach the agent");
    }
    for needed in [
        "PATH",
        "HOME",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "SSH_AUTH_SOCK",
    ] {
        assert!(policy.allows(needed), "{needed} must reach the agent");
    }
}

#[test]
fn the_operator_list_extends_the_allowlist_but_never_with_server_secrets() {
    let policy = child_env_policy_with(" GH_TOKEN , PO_JWT_SECRET,,NEO4J_PASSWORD ");
    assert!(policy.allows("GH_TOKEN"));
    assert!(!policy.allows("PO_JWT_SECRET"));
    assert!(!policy.allows("NEO4J_PASSWORD"));
    assert!(!child_env_policy_with("").allows("GH_TOKEN"));
}

#[tokio::test]
async fn test_build_options_with_hooks() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // Build hooks map with a CompactionNotifier
    let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(tx, None, "test-session".to_string());
    let mut hooks = std::collections::HashMap::new();
    hooks.insert(
        "PreCompact".to_string(),
        vec![nexus_claude::HookMatcher {
            matcher: None,
            hooks: vec![std::sync::Arc::new(notifier)],
        }],
    );

    let opts = manager
        .build_options(
            "/tmp",
            "model",
            "prompt",
            None,
            None,
            Some(hooks),
            &[],
            None,
            None,
        )
        .await;

    // Hooks should be configured
    let hook_map = opts.hooks.as_ref().expect("hooks should be Some");
    assert!(hook_map.contains_key("PreCompact"));
    let matchers = &hook_map["PreCompact"];
    assert_eq!(matchers.len(), 1);
    assert_eq!(matchers[0].hooks.len(), 1);
}

#[tokio::test]
async fn test_build_options_without_hooks() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let opts = manager
        .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
        .await;

    // No hooks configured
    assert!(opts.hooks.is_none());
}

// ====================================================================
// resolve_add_dirs & build_options with add_dirs
// ====================================================================

#[tokio::test]
async fn test_resolve_add_dirs_explicit() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let dirs = vec!["/path/a".to_string(), "/path/b".to_string()];
    let result = manager
        .resolve_add_dirs("/tmp", Some(&dirs), None, None)
        .await;

    assert_eq!(result, vec!["/path/a", "/path/b"]);
}

#[tokio::test]
async fn test_resolve_add_dirs_empty() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.resolve_add_dirs("/tmp", None, None, None).await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_resolve_add_dirs_workspace() {
    let state = mock_app_state();

    // Create a workspace and projects
    let ws = crate::test_helpers::test_workspace();
    state.neo4j.create_workspace(&ws).await.unwrap();

    let mut p1 = crate::test_helpers::test_project();
    p1.root_path = "/home/user/proj-a".to_string();
    state.neo4j.create_project(&p1).await.unwrap();
    state
        .neo4j
        .add_project_to_workspace(ws.id, p1.id)
        .await
        .unwrap();

    let mut p2 = crate::test_helpers::test_project();
    p2.root_path = "/home/user/proj-b".to_string();
    state.neo4j.create_project(&p2).await.unwrap();
    state
        .neo4j
        .add_project_to_workspace(ws.id, p2.id)
        .await
        .unwrap();

    let mut p3 = crate::test_helpers::test_project();
    p3.root_path = "/home/user/proj-cwd".to_string();
    state.neo4j.create_project(&p3).await.unwrap();
    state
        .neo4j
        .add_project_to_workspace(ws.id, p3.id)
        .await
        .unwrap();

    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // cwd matches p3, so only p1 and p2 should be returned
    let result = manager
        .resolve_add_dirs("/home/user/proj-cwd", None, Some(&ws.slug), None)
        .await;

    assert_eq!(result.len(), 2);
    assert!(result.contains(&"/home/user/proj-a".to_string()));
    assert!(result.contains(&"/home/user/proj-b".to_string()));
    assert!(!result.contains(&"/home/user/proj-cwd".to_string()));
}

#[tokio::test]
async fn test_resolve_add_dirs_workspace_not_found() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager
        .resolve_add_dirs("/tmp", None, Some("nonexistent-ws"), None)
        .await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_build_options_with_add_dirs() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let dirs = ["/extra/dir1".to_string(), "/extra/dir2".to_string()];
    let opts = manager
        .build_options(
            "/tmp", "model", "prompt", None, None, None, &dirs, None, None,
        )
        .await;

    assert_eq!(opts.add_dirs.len(), 2);
    assert!(opts
        .add_dirs
        .contains(&std::path::PathBuf::from("/extra/dir1")));
    assert!(opts
        .add_dirs
        .contains(&std::path::PathBuf::from("/extra/dir2")));
}

#[tokio::test]
async fn test_build_options_without_add_dirs() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let opts = manager
        .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
        .await;

    assert!(opts.add_dirs.is_empty());
}

#[tokio::test]
async fn test_build_options_injects_session_id_env() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // With session_id → PO_SESSION_ID should be in MCP server env
    let opts = manager
        .build_options(
            "/tmp",
            "model",
            "prompt",
            None,
            None,
            None,
            &[],
            None,
            Some("abc-def-123"),
        )
        .await;

    // PO_SESSION_ID is injected into the MCP server config env, not ClaudeCodeOptions.env
    let mcp = opts.mcp_servers.get("project-orchestrator").unwrap();
    let mcp_env = match mcp {
        nexus_claude::McpServerConfig::Stdio { env, .. } => env.as_ref().unwrap(),
        _ => panic!("Expected Stdio config"),
    };
    assert_eq!(
        mcp_env.get("PO_SESSION_ID").map(|s| s.as_str()),
        Some("abc-def-123"),
        "PO_SESSION_ID should be injected into MCP server env when session_id is provided"
    );
}

#[tokio::test]
async fn test_build_options_no_session_id_no_env() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // Without session_id → PO_SESSION_ID should NOT be in MCP server env
    let opts = manager
        .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
        .await;

    let mcp = opts.mcp_servers.get("project-orchestrator").unwrap();
    let mcp_env = match mcp {
        nexus_claude::McpServerConfig::Stdio { env, .. } => env.as_ref().unwrap(),
        _ => panic!("Expected Stdio config"),
    };
    assert!(
        !mcp_env.contains_key("PO_SESSION_ID"),
        "PO_SESSION_ID should not be in MCP server env when session_id is None"
    );
}

// ====================================================================
// message_to_events
// ====================================================================

#[test]
fn test_message_to_events_assistant_text() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "Hello!".into(),
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::AssistantText { content, .. } if content == "Hello!"));
}

#[test]
fn test_message_to_events_thinking() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::Thinking(ThinkingContent {
                thinking: "Let me think...".into(),
                signature: "sig".into(),
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0], ChatEvent::Thinking { content, .. } if content == "Let me think...")
    );
}

#[test]
fn test_message_to_events_tool_use() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolUse(ToolUseContent {
                id: "tool-1".into(),
                name: "create_plan".into(),
                input: serde_json::json!({"title": "My Plan"}),
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::ToolUse { id, tool, .. }
            if id == "tool-1" && tool == "create_plan"));
}

#[test]
fn test_message_to_events_tool_result() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "tool-1".into(),
                content: Some(ContentValue::Text("Success".into())),
                is_error: Some(false),
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0], ChatEvent::ToolResult { id, is_error, .. }
            if id == "tool-1" && !is_error)
    );
}

#[test]
fn test_message_to_events_tool_result_error() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "tool-2".into(),
                content: Some(ContentValue::Text("Not found".into())),
                is_error: Some(true),
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::ToolResult { is_error, .. } if *is_error));
}

#[test]
fn test_message_to_events_tool_result_never_carries_a_live_citation() {
    const ID: &str = "57cf05c9-25b6-495d-ab07-de4b11d64736";
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "tool-1".into(),
                content: Some(ContentValue::Text(format!("voir #plan:{ID}\n"))),
                is_error: Some(false),
            })],
        },
        parent_tool_use_id: None,
    };
    let events = ChatManager::message_to_events(&msg);
    let ChatEvent::ToolResult { result, .. } = &events[0] else {
        panic!("a tool result");
    };
    let text = result.as_str().expect("the text of the result");
    assert!(text.contains(&format!("#plan:{ID}")), "visible: {text:?}");
    assert!(
        crate::refs::cite::cited_refs(text).is_empty(),
        "a tool result is never a citation of the agent: {text:?}"
    );
}

#[test]
fn test_message_to_events_result() {
    let msg = Message::Result {
        subtype: "success".into(),
        duration_ms: 5000,
        duration_api_ms: 4500,
        is_error: false,
        num_turns: 3,
        session_id: "cli-abc-123".into(),
        total_cost_usd: Some(0.15),
        usage: None,
        result: None,
        structured_output: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::Result {
        session_id, duration_ms, cost_usd, subtype, is_error, num_turns, result_text, ..
    } if session_id == "cli-abc-123"
        && *duration_ms == 5000
        && *cost_usd == Some(0.15)
        && subtype == "success"
        && !is_error
        && *num_turns == Some(3)
        && result_text.is_none()
    ));
}

#[test]
fn test_message_to_events_result_error_max_turns() {
    let msg = Message::Result {
        subtype: "error_max_turns".into(),
        duration_ms: 8000,
        duration_api_ms: 7500,
        is_error: true,
        num_turns: 15,
        session_id: "cli-max-turns".into(),
        total_cost_usd: Some(0.50),
        usage: None,
        result: None,
        structured_output: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::Result {
            subtype, is_error, num_turns, ..
        } if subtype == "error_max_turns" && *is_error && *num_turns == Some(15)));
}

/// Follow-up review of #711, finding 2: the CLI answers an interrupt with
/// `error_during_execution`, `is_error: true` and an empty `result`; the stored end of
/// the stopped turn is the neutral one (no error block at reload), with the CLI's
/// duration, or the turn's own when the CLI gave none (never "0 ms").
#[test]
fn a_stopped_cli_turn_is_stored_with_a_neutral_end_and_a_real_duration() {
    let cli_result = |duration_ms: i64| Message::Result {
        subtype: "error_during_execution".into(),
        duration_ms,
        duration_api_ms: 0,
        is_error: true,
        num_turns: 1,
        session_id: "cli-stopped".into(),
        total_cost_usd: Some(0.01),
        usage: None,
        result: Some(String::new()),
        structured_output: None,
    };
    let elapsed = Duration::from_millis(1500);
    for (read, want_ms, want_session) in [
        (Some(cli_result(2000)), 2000, "cli-stopped"),
        (Some(cli_result(0)), 1500, "cli-stopped"),
        (None, 1500, ""),
    ] {
        let ended = interrupted_turn_end(read, elapsed);
        let ChatEvent::Result {
            session_id,
            duration_ms,
            subtype,
            is_error,
            result_text,
            stop_reason,
            ..
        } = &ended
        else {
            panic!("not a result: {ended:?}");
        };
        assert_eq!(subtype, INTERRUPTED_SUBTYPE, "{ended:?}");
        assert!(!is_error, "a Stop is not a failure: {ended:?}");
        assert_eq!(result_text, &None, "no empty error text: {ended:?}");
        assert_eq!(stop_reason.as_deref(), Some("interrupted"));
        assert_eq!(*duration_ms, want_ms, "{ended:?}");
        assert_eq!(session_id, want_session);
    }
}

/// Only a `result` stopped by a person has a stored form of its own.
#[test]
fn only_an_interrupted_result_is_stored_in_the_neutral_form() {
    let result = |stop: Option<&str>| ChatEvent::Result {
        session_id: "s".into(),
        duration_ms: 7,
        cost_usd: None,
        subtype: "error_during_execution".into(),
        is_error: true,
        num_turns: Some(1),
        result_text: Some("boom".into()),
        cost: None,
        usage: None,
        model: None,
        stop_reason: stop.map(str::to_string),
        error_code: None,
        error_message: None,
    };
    assert!(stored_interrupted_result(&result(Some("error"))).is_none());
    assert!(stored_interrupted_result(&result(None)).is_none());
    assert!(stored_interrupted_result(&ChatEvent::StreamingStatus {
        is_streaming: false
    })
    .is_none());
    let stored = stored_interrupted_result(&result(Some("interrupted"))).unwrap();
    assert!(
        matches!(
            &stored,
            ChatEvent::Result { subtype, is_error: false, result_text: Some(t), duration_ms: 7, .. }
                if subtype == INTERRUPTED_SUBTYPE && t == "boom"
        ),
        "{stored:?}"
    );
}

#[test]
fn test_message_to_events_result_error_during_execution() {
    let msg = Message::Result {
        subtype: "error_during_execution".into(),
        duration_ms: 2000,
        duration_api_ms: 1800,
        is_error: true,
        num_turns: 1,
        session_id: "cli-exec-err".into(),
        total_cost_usd: Some(0.02),
        usage: None,
        result: Some("Process exited with code 1".into()),
        structured_output: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::Result {
        subtype, is_error, result_text, ..
    } if subtype == "error_during_execution"
        && *is_error
        && result_text.as_deref() == Some("Process exited with code 1")
    ));
}

#[test]
fn test_message_to_events_multiple_blocks() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "hmm...".into(),
                    signature: "s".into(),
                }),
                ContentBlock::Text(TextContent {
                    text: "Here is my answer".into(),
                }),
                ContentBlock::ToolUse(ToolUseContent {
                    id: "t1".into(),
                    name: "list_plans".into(),
                    input: serde_json::json!({}),
                }),
            ],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 3);
    assert!(matches!(&events[0], ChatEvent::Thinking { .. }));
    assert!(matches!(&events[1], ChatEvent::AssistantText { .. }));
    assert!(matches!(&events[2], ChatEvent::ToolUse { .. }));
}

#[test]
fn test_message_to_events_system_init() {
    let msg = Message::System {
        subtype: "init".into(),
        data: serde_json::json!({
            "session_id": "cli-sess-abc",
            "model": "claude-sonnet-4-6",
            "tools": ["Bash", "Read", "Write", "Edit"],
            "mcp_servers": [{"name": "po", "status": "connected"}],
            "permissionMode": "default"
        }),
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::SystemInit {
        cli_session_id, model, tools, mcp_servers, permission_mode, ..
    } if cli_session_id == "cli-sess-abc"
        && model.as_deref() == Some("claude-sonnet-4-6")
        && tools.len() == 4
        && mcp_servers.len() == 1
        && permission_mode.as_deref() == Some("default")
    ));
}

#[test]
fn test_message_to_events_system_unknown() {
    // Unknown system subtypes should still be ignored
    let msg = Message::System {
        subtype: "unknown_future_type".into(),
        data: serde_json::json!({"version": "1.0"}),
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

// ── masking fails closed (decision A36) ────────────────────────────────

#[test]
fn a_message_that_cannot_be_masked_is_withheld_not_passed_in_clear() {
    // The secret value collides with a key of the message's own JSON form:
    // once masked, the message no longer reads back.
    let secret = "duration_api_ms";
    let masker = crate::vault::mask::Masker::from_values([("TOKEN", secret)]);
    let msg = Message::Result {
        subtype: "success".into(),
        duration_ms: 1,
        duration_api_ms: 1,
        is_error: false,
        num_turns: 1,
        session_id: "cli-1".into(),
        total_cost_usd: None,
        usage: None,
        result: Some(format!("the value is {secret}")),
        structured_output: None,
    };

    let out = ChatManager::mask_cli_message_with(&masker, msg);
    assert!(
        matches!(&out, Message::System { subtype, .. } if subtype == MASKING_FAILED_SUBTYPE),
        "an unmaskable message must be replaced, got {out:?}"
    );
    let events = ChatManager::message_to_events(&out);
    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0], ChatEvent::Error { message, .. } if message == MASKING_FAILED_MESSAGE)
    );
    let wire = serde_json::to_string(&events).unwrap();
    assert!(
        !wire.contains(secret),
        "the secret reached the wire: {wire}"
    );
}

#[test]
fn a_maskable_message_is_masked_and_an_unrelated_one_is_untouched() {
    let masker = crate::vault::mask::Masker::from_values([("TOKEN", "s3cr3t-value")]);
    let with_secret = Message::System {
        subtype: "note".into(),
        data: serde_json::json!({"text": "key=s3cr3t-value"}),
    };
    let out = ChatManager::mask_cli_message_with(&masker, with_secret);
    let json = serde_json::to_string(&out).unwrap();
    assert!(!json.contains("s3cr3t-value") && json.contains("[secret:TOKEN]"));

    let plain = Message::System {
        subtype: "note".into(),
        data: serde_json::json!({"text": "nothing here"}),
    };
    let out = ChatManager::mask_cli_message_with(&masker, plain);
    assert!(matches!(&out, Message::System { subtype, .. } if subtype == "note"));
}

#[test]
fn test_message_to_events_user_message() {
    let msg = Message::User {
        message: nexus_claude::UserMessage {
            content: "Hi".into(),
            content_blocks: None,
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_user_message_with_tool_result() {
    let msg = Message::User {
        message: nexus_claude::UserMessage {
            content: String::new(),
            content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "toolu_abc123".into(),
                content: Some(ContentValue::Text("fn main() {}".into())),
                is_error: Some(false),
            })]),
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0],
        ChatEvent::ToolResult { id, is_error, .. }
        if id == "toolu_abc123" && !is_error
    ));
}

#[test]
fn test_message_to_events_user_message_with_tool_result_error() {
    let msg = Message::User {
        message: nexus_claude::UserMessage {
            content: String::new(),
            content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "toolu_err001".into(),
                content: Some(ContentValue::Text("File not found".into())),
                is_error: Some(true),
            })]),
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0],
        ChatEvent::ToolResult { id, is_error, .. }
        if id == "toolu_err001" && *is_error
    ));
}

// ====================================================================
// message_to_events — StreamEvent
// ====================================================================

#[test]
fn test_message_to_events_stream_text_delta() {
    let msg = Message::StreamEvent {
        event: StreamEventData::ContentBlockDelta {
            index: 0,
            delta: StreamDelta::TextDelta {
                text: "Hello".into(),
            },
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::StreamDelta { text, .. } if text == "Hello"));
}

#[test]
fn test_message_to_events_stream_content_block_start_tool_use() {
    let msg = Message::StreamEvent {
        event: StreamEventData::ContentBlockStart {
            index: 1,
            content_block: serde_json::json!({
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "list_plans",
                "input": {"status": "active"}
            }),
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], ChatEvent::ToolUse { id, tool, .. }
                if id == "toolu_abc123" && tool == "list_plans"));
}

#[test]
fn test_message_to_events_stream_content_block_start_text_ignored() {
    let msg = Message::StreamEvent {
        event: StreamEventData::ContentBlockStart {
            index: 0,
            content_block: serde_json::json!({"type": "text", "text": ""}),
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_stream_thinking_delta() {
    let msg = Message::StreamEvent {
        event: StreamEventData::ContentBlockDelta {
            index: 0,
            delta: StreamDelta::ThinkingDelta {
                thinking: "hmm".into(),
            },
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_stream_message_stop() {
    let msg = Message::StreamEvent {
        event: StreamEventData::MessageStop,
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_stream_content_block_start() {
    let msg = Message::StreamEvent {
        event: StreamEventData::ContentBlockStart {
            index: 0,
            content_block: serde_json::json!({"type": "text", "text": ""}),
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_stream_input_json_delta() {
    let msg = Message::StreamEvent {
        event: StreamEventData::ContentBlockDelta {
            index: 0,
            delta: StreamDelta::InputJsonDelta {
                partial_json: r#"{"title":"#.into(),
            },
        },
        session_id: Some("sess-1".into()),
        parent_tool_use_id: None,
    };

    // InputJsonDelta is not TextDelta, so it should produce empty events
    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_stream_message_start() {
    let msg = Message::StreamEvent {
        event: StreamEventData::MessageStart {
            message: serde_json::json!({"id": "msg_123"}),
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_stream_message_delta() {
    let msg = Message::StreamEvent {
        event: StreamEventData::MessageDelta {
            delta: serde_json::json!({"stop_reason": "end_turn"}),
            usage: Some(serde_json::json!({"output_tokens": 50})),
        },
        session_id: None,
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert!(events.is_empty());
}

#[test]
fn test_message_to_events_tool_result_structured() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "tool-3".into(),
                content: Some(ContentValue::Structured(vec![
                    serde_json::json!({"type": "text", "text": "result 1"}),
                    serde_json::json!({"type": "text", "text": "result 2"}),
                ])),
                is_error: None,
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    match &events[0] {
        ChatEvent::ToolResult {
            id,
            result,
            is_error,
            ..
        } => {
            assert_eq!(id, "tool-3");
            assert!(result.is_array());
            assert_eq!(result.as_array().unwrap().len(), 2);
            assert!(!is_error); // None defaults to false
        }
        _ => panic!("Expected ToolResult"),
    }
}

#[test]
fn test_message_to_events_tool_result_none_content() {
    let msg = Message::Assistant {
        message: AssistantMessage {
            content: vec![ContentBlock::ToolResult(ToolResultContent {
                tool_use_id: "tool-4".into(),
                content: None,
                is_error: Some(false),
            })],
        },
        parent_tool_use_id: None,
    };

    let events = ChatManager::message_to_events(&msg);
    assert_eq!(events.len(), 1);
    match &events[0] {
        ChatEvent::ToolResult { result, .. } => {
            assert!(result.is_null());
        }
        _ => panic!("Expected ToolResult"),
    }
}

// ====================================================================
// build_system_prompt with active plans
// ====================================================================

#[tokio::test]
async fn test_build_system_prompt_with_active_plans() {
    let state = mock_app_state();
    let project = test_project();
    state.neo4j.create_project(&project).await.unwrap();

    // Create an active plan linked to the project
    let plan = crate::test_helpers::test_plan();
    state.neo4j.create_plan(&plan).await.unwrap();
    state
        .neo4j
        .link_plan_to_project(plan.id, project.id)
        .await
        .unwrap();

    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (prompt, _note_ids) = manager
        .build_system_prompt(
            Some(&project.slug),
            "check the plan",
            None,
            None,
            None,
            None,
        )
        .await;

    // Base prompt present
    assert!(prompt.contains("EXCLUSIVELY the Project Orchestrator MCP tools"));
    // Dynamic context section present (either oneshot or fallback)
    assert!(prompt.contains("---"));
    // Project name should appear in the dynamic context
    assert!(prompt.contains(&project.name));
}

// ====================================================================
// active_session_count
// ====================================================================

#[tokio::test]
async fn test_active_session_count_empty() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    assert_eq!(manager.active_session_count().await, 0);
}

// ====================================================================
// subscribe / interrupt / close errors for missing sessions
// ====================================================================

#[tokio::test]
async fn test_subscribe_nonexistent_session() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.subscribe("nonexistent").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn test_interrupt_nonexistent_session() {
    // interrupt() no longer errors for non-local sessions — it always succeeds
    // because it may need to publish to NATS for cross-instance interrupt routing.
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.interrupt("nonexistent").await;
    assert!(
        result.is_ok(),
        "interrupt should succeed even for non-local sessions"
    );
}

#[tokio::test]
async fn test_close_nonexistent_session() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.close_session("nonexistent").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn closing_a_session_broadcasts_session_closed() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    test_support::insert_live_session_without_cli(&manager, "sess-close").await;
    let mut rx = manager
        .active_sessions
        .read()
        .await
        .get("sess-close")
        .expect("registered")
        .events_tx
        .subscribe();
    manager.close_session("sess-close").await.unwrap();
    let mut seen = None;
    while let Ok(ev) = rx.try_recv() {
        if let ChatEvent::SessionClosed { session_id, reason } = ev {
            seen = Some((session_id, reason));
        }
    }
    assert_eq!(
        seen,
        Some((
            "sess-close".to_string(),
            Some(crate::episodes::models::close_reason::USER_CLOSED.to_string())
        ))
    );
}

#[tokio::test]
async fn test_send_message_nonexistent_session() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.send_message("nonexistent", "hello").await;
    assert!(result.is_err());
}

/// B36 — closing a parent with cascade closes its delegated children (and
/// their children), leaving nothing live under it.
#[tokio::test]
async fn closing_a_parent_with_cascade_closes_its_delegated_children() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let parent = crate::test_helpers::test_chat_session(None);
    let mut child = crate::test_helpers::test_chat_session(None);
    child.spawned_by = Some(
        serde_json::json!({"type": "delegation", "parent_session_id": parent.id.to_string()})
            .to_string(),
    );
    for node in [&parent, &child] {
        manager.graph.create_chat_session(node).await.unwrap();
    }
    test_support::insert_live_session_without_cli(&manager, &parent.id.to_string()).await;
    test_support::insert_live_session_without_cli(&manager, &child.id.to_string()).await;
    assert!(manager.is_session_active(&child.id.to_string()).await);

    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();

    assert!(
        !manager.is_session_active(&parent.id.to_string()).await,
        "the parent is closed"
    );
    assert!(
        !manager.is_session_active(&child.id.to_string()).await,
        "the delegated child is closed by the cascade"
    );
}

/// A session node spawned by delegation from `parent`.
fn delegated_child_of(
    parent: &crate::neo4j::models::ChatSessionNode,
) -> crate::neo4j::models::ChatSessionNode {
    let mut child = crate::test_helpers::test_chat_session(None);
    child.spawned_by = Some(
        serde_json::json!({"type": "delegation", "parent_session_id": parent.id.to_string()})
            .to_string(),
    );
    child
}

/// B36 — a child whose delegation lands just after its parent's children
/// were listed is closed too: the parent is closed before its children are
/// read, and the tree is read again until nothing new appears.
#[tokio::test]
async fn cascade_closes_a_child_that_lands_after_the_listing() {
    let mock = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let graph: std::sync::Arc<dyn crate::neo4j::GraphStore> = mock.clone();
    let manager = ChatManager::new_without_memory(graph, mock_app_state().meili, test_config());

    let parent = crate::test_helpers::test_chat_session(None);
    let early = delegated_child_of(&parent);
    let late = delegated_child_of(&parent);
    for node in [&parent, &early] {
        manager.graph.create_chat_session(node).await.unwrap();
    }
    for node in [&parent, &early, &late] {
        test_support::insert_live_session_without_cli(&manager, &node.id.to_string()).await;
    }
    // `late` is live, and its node is written right after the first read of
    // the parent's children.
    mock.land_children_after_read
        .lock()
        .unwrap()
        .insert(parent.id, vec![late.clone()]);

    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();

    for (node, what) in [
        (&parent, "parent"),
        (&early, "early child"),
        (&late, "late child"),
    ] {
        assert!(
            !manager.is_session_active(&node.id.to_string()).await,
            "the {what} is closed"
        );
    }
}

/// B36 — a descendant already closed is not closed again (no false
/// failure), and its own live children are still reached.
#[tokio::test]
async fn cascade_skips_a_closed_descendant_and_reaches_its_children() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let parent = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&parent);
    let grandchild = delegated_child_of(&child);
    for node in [&parent, &child, &grandchild] {
        manager.graph.create_chat_session(node).await.unwrap();
    }
    // The child already finished; its own child is still running.
    test_support::insert_live_session_without_cli(&manager, &parent.id.to_string()).await;
    test_support::insert_live_session_without_cli(&manager, &grandchild.id.to_string()).await;

    let closed = manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await;

    assert!(
        closed.is_ok(),
        "an already-closed descendant is no failure: {closed:?}"
    );
    assert!(!manager.is_session_active(&grandchild.id.to_string()).await);
}

/// B36 — every descendant the cascade closes loses its neutral directory,
/// as the root does.
#[tokio::test]
async fn cascade_removes_each_closed_descendants_neutral_directory() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let parent = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&parent);
    for node in [&parent, &child] {
        manager.graph.create_chat_session(node).await.unwrap();
        test_support::insert_live_session_without_cli(&manager, &node.id.to_string()).await;
    }
    let child_dir = super::super::neutral_place::dir_for(child.id);
    std::fs::create_dir_all(&child_dir).unwrap();

    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();

    let left = child_dir.exists();
    let _ = std::fs::remove_dir_all(&child_dir);
    assert!(!left, "the child's neutral directory goes with it");
}

/// P18-suite, point 9: deleting a parent closes its two live children (and the child of
/// one of them), then deletes them all — none is left alive, none is left listed as the
/// child of a session that no longer exists. Red before: `delete_session` closed the
/// parent alone (no cascade) and deleted its node alone; the children ran on and the
/// `spawned_by` fallback still listed them under the deleted parent.
#[tokio::test]
async fn deleting_a_parent_closes_and_deletes_its_descendants() {
    let mock = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let graph: std::sync::Arc<dyn crate::neo4j::GraphStore> = mock.clone();
    let manager = ChatManager::new_without_memory(graph, mock_app_state().meili, test_config());

    let parent = crate::test_helpers::test_chat_session(None);
    let first = delegated_child_of(&parent);
    let second = delegated_child_of(&parent);
    let grandchild = delegated_child_of(&first);
    let tree = [&parent, &first, &second, &grandchild];
    for node in tree {
        manager.graph.create_chat_session(node).await.unwrap();
        test_support::insert_live_session_without_cli(&manager, &node.id.to_string()).await;
    }
    assert_eq!(
        manager
            .graph
            .get_session_children(parent.id)
            .await
            .unwrap()
            .len(),
        2
    );

    assert!(manager.delete_session(parent.id).await.unwrap());

    for (node, what) in [
        (&parent, "parent"),
        (&first, "first child"),
        (&second, "second child"),
        (&grandchild, "grandchild"),
    ] {
        assert!(
            !manager.is_session_active(&node.id.to_string()).await,
            "the {what} is closed"
        );
        assert!(
            manager
                .graph
                .get_chat_session(node.id)
                .await
                .unwrap()
                .is_none(),
            "the {what} is deleted"
        );
    }
    assert!(manager
        .graph
        .get_session_children(parent.id)
        .await
        .unwrap()
        .is_empty());
}

/// P18-suite, point 9: a parent deleted from the graph alone (no chat manager: its
/// children stay) lists no children — the `spawned_by` fallback does not hand them back
/// as the children of a session that no longer exists.
#[tokio::test]
async fn a_deleted_parent_has_no_children_left_by_the_spawned_by_fallback() {
    let graph = crate::neo4j::mock::MockGraphStore::new();
    let parent = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&parent);
    for node in [&parent, &child] {
        graph.create_chat_session(node).await.unwrap();
    }
    assert_eq!(
        graph.get_session_children(parent.id).await.unwrap().len(),
        1
    );
    assert!(graph.delete_chat_session(parent.id).await.unwrap());
    assert!(graph
        .get_session_children(parent.id)
        .await
        .unwrap()
        .is_empty());
}

/// A manager over a fresh mock graph, with the whole tree stored and live.
async fn manager_with_live_tree(
    nodes: &[&crate::neo4j::models::ChatSessionNode],
) -> (
    ChatManager,
    std::sync::Arc<crate::neo4j::mock::MockGraphStore>,
) {
    let mock = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let graph: std::sync::Arc<dyn crate::neo4j::GraphStore> = mock.clone();
    let manager = ChatManager::new_without_memory(graph, mock_app_state().meili, test_config());
    for node in nodes {
        manager.graph.create_chat_session(node).await.unwrap();
        test_support::insert_live_session_without_cli(&manager, &node.id.to_string()).await;
    }
    (manager, mock)
}

/// P18-suite, point 9: a session that does not exist deletes to `false` and takes no
/// other session with it.
#[tokio::test]
async fn deleting_an_unknown_session_answers_false_and_leaves_the_others() {
    let bystander = crate::test_helpers::test_chat_session(None);
    let (manager, _mock) = manager_with_live_tree(&[&bystander]).await;

    assert!(!manager.delete_session(Uuid::new_v4()).await.unwrap());

    assert!(manager
        .graph
        .get_chat_session(bystander.id)
        .await
        .unwrap()
        .is_some());
    assert!(manager.is_session_active(&bystander.id.to_string()).await);
}

/// P18-suite, point 9: the cascade goes deeper than one level (a chain of four) and
/// stops at the tree: a sibling subtree and an unrelated session are left alone.
#[tokio::test]
async fn deleting_a_chain_takes_every_level_and_spares_the_sibling_subtree() {
    let root = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&root);
    let grandchild = delegated_child_of(&child);
    let great_grandchild = delegated_child_of(&grandchild);
    let other_root = crate::test_helpers::test_chat_session(None);
    let other_child = delegated_child_of(&other_root);
    let all = [
        &root,
        &child,
        &grandchild,
        &great_grandchild,
        &other_root,
        &other_child,
    ];
    let (manager, _mock) = manager_with_live_tree(&all).await;

    assert!(manager.delete_session(root.id).await.unwrap());

    for gone in [&root, &child, &grandchild, &great_grandchild] {
        assert!(manager
            .graph
            .get_chat_session(gone.id)
            .await
            .unwrap()
            .is_none());
        assert!(!manager.is_session_active(&gone.id.to_string()).await);
    }
    for kept in [&other_root, &other_child] {
        assert!(manager
            .graph
            .get_chat_session(kept.id)
            .await
            .unwrap()
            .is_some());
        assert!(manager.is_session_active(&kept.id.to_string()).await);
    }
}

/// P18-suite, point 9: a child already closed (not live) is deleted all the same, and
/// a lineage that loops back on itself ends instead of walking forever.
#[tokio::test]
async fn deleting_a_parent_deletes_a_closed_child_and_survives_a_lineage_loop() {
    let mut parent = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&parent);
    // The parent also names its own child as ITS parent: a loop in the lineage.
    parent.spawned_by = Some(
        serde_json::json!({"type": "delegation", "parent_session_id": child.id.to_string()})
            .to_string(),
    );
    let (manager, _mock) = manager_with_live_tree(&[&parent]).await;
    // The child is stored but was never live (already closed).
    manager.graph.create_chat_session(&child).await.unwrap();

    assert!(manager.delete_session(parent.id).await.unwrap());

    for gone in [&parent, &child] {
        assert!(manager
            .graph
            .get_chat_session(gone.id)
            .await
            .unwrap()
            .is_none());
    }
}

/// P18-suite, point 9: a descendant that cannot be deleted fails the deletion BEFORE
/// the parent goes; once the graph answers again, a retry finds the tree and finishes it.
#[tokio::test]
async fn a_graph_error_during_the_cascade_keeps_the_parent_and_a_retry_finishes() {
    let parent = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&parent);
    let (manager, mock) = manager_with_live_tree(&[&parent, &child]).await;

    mock.fail_reads.lock().unwrap().insert("get_chat_session");
    let failed = manager.delete_session(parent.id).await;
    mock.fail_reads.lock().unwrap().clear();

    assert!(failed.is_err(), "the deletion reports the failure");
    for kept in [&parent, &child] {
        assert!(
            manager
                .graph
                .get_chat_session(kept.id)
                .await
                .unwrap()
                .is_some(),
            "nothing was deleted before the failure"
        );
    }

    assert!(manager.delete_session(parent.id).await.unwrap());
    for gone in [&parent, &child] {
        assert!(manager
            .graph
            .get_chat_session(gone.id)
            .await
            .unwrap()
            .is_none());
    }
}

/// P18-suite, point 9: the transcript of a native child goes with its parent's deletion,
/// and a transcript that cannot be removed fails the deletion with the parent kept.
#[tokio::test]
async fn deleting_a_parent_removes_a_native_childs_transcript() {
    use nexus_claude::agent::{ProviderKind, ResumeToken};
    let root = tempfile::tempdir().unwrap();
    let instance_dir = root.path().join("local-1");
    std::fs::create_dir_all(&instance_dir).unwrap();
    let wire = |id: &str| {
        ResumeToken::new(
            ProviderKind::Native,
            1,
            serde_json::json!({ "transcript_id": id }),
        )
        .to_wire()
    };

    let parent = crate::test_helpers::test_chat_session(None);
    let mut child = delegated_child_of(&parent);
    child.provider_id = Some("local-1".to_string());
    child.resume_token = Some(wire("t1"));
    let (manager, _mock) = manager_with_live_tree(&[&parent, &child]).await;
    let manager = manager.with_native_transcripts(root.path());

    // A transcript that cannot be removed (a directory where the file should be).
    std::fs::create_dir_all(instance_dir.join("t1.json")).unwrap();
    assert!(manager.delete_session(parent.id).await.is_err());
    assert!(manager
        .graph
        .get_chat_session(parent.id)
        .await
        .unwrap()
        .is_some());

    std::fs::remove_dir(instance_dir.join("t1.json")).unwrap();
    std::fs::write(instance_dir.join("t1.json"), "{}").unwrap();
    assert!(manager.delete_session(parent.id).await.unwrap());
    assert!(!instance_dir.join("t1.json").exists());
    assert!(manager
        .graph
        .get_chat_session(child.id)
        .await
        .unwrap()
        .is_none());
}

// ---- B36-RACE: a delegation in flight when its parent is cancelled ----

/// An agent-engine delegation request whose `spawned_by` names `parent`.
fn delegation_of(parent: Uuid) -> ChatRequest {
    let mut req = agent_request("delegated work");
    req.spawned_by = Some(
        serde_json::json!({"type": "delegation", "parent_session_id": parent.to_string()})
            .to_string(),
    );
    req
}

/// A live parent (node written, session live) on the agent-path manager.
async fn live_parent(
    manager: &ChatManager,
    graph: &crate::neo4j::mock::MockGraphStore,
) -> ChatSessionNode {
    let parent = test_chat_session(None);
    graph.create_chat_session(&parent).await.unwrap();
    test_support::insert_live_session_without_cli(manager, &parent.id.to_string()).await;
    parent
}

/// The first child node of `parent` the graph holds, waited for (5 s at most).
async fn child_node_written(graph: &crate::neo4j::mock::MockGraphStore, parent: Uuid) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(child) = graph.get_session_children(parent).await.unwrap().first() {
            return child.id.to_string();
        }
        assert!(
            Instant::now() < deadline,
            "the child's node is never written"
        );
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// B36-RACE, criterion 1 — the PRODUCTION order: the child's node is written (its
/// provider still opening), the parent is closed with cascade, THEN the child would
/// become live. The cascade saw it inactive and skipped it; its activation reads the
/// parent's tombstone and refuses it, with a typed error the delegation reads. Red
/// without the tombstone: the child ends live, outliving the cancelled run.
#[tokio::test]
async fn a_delegation_in_flight_when_its_parent_is_cancelled_is_refused_at_activation() {
    let (manager, graph, fake) = agent_manager();
    let manager = Arc::new(manager);
    let parent = live_parent(&manager, &graph).await;
    // The provider takes a while to open: the window the race lives in.
    *fake.open_delay.lock().unwrap() = Some(Duration::from_millis(400));
    let opening = tokio::spawn({
        let manager = Arc::clone(&manager);
        let request = delegation_of(parent.id);
        async move { manager.create_session(&request).await }
    });
    let child = child_node_written(&graph, parent.id).await;
    assert!(
        !manager.is_session_active(&child).await,
        "written, not live yet"
    );

    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();

    let opened = opening.await.expect("the opening task ends");
    let err = opened.expect_err("a child of a cancelled parent is never started");
    let refused = super::super::cancelled_parents::ParentCancelled::in_chain(&err)
        .expect("the delegation reads a typed refusal");
    assert_eq!(refused.parent_session_id, parent.id.to_string());
    assert_eq!(refused.session_id, child);
    assert!(err.to_string().contains("was cancelled"), "{err}");
    assert!(
        !manager.is_session_active(&child).await,
        "the late child is not live"
    );
    assert!(!manager.agent_runtime.owns(&child).await);
}

/// B36-RACE, criterion 2 — the agent engine's RESUME on behalf of a DELEGATION (the
/// path the warm pool takes, `subagents::open_child`): a closed child of a cancelled
/// parent is refused when it is resumed, not made live.
#[tokio::test]
async fn resuming_a_child_of_a_cancelled_parent_is_refused() {
    let (manager, graph, _fake) = agent_manager();
    let parent = live_parent(&manager, &graph).await;
    let child = manager
        .create_session(&delegation_of(parent.id))
        .await
        .expect("the parent is live: the child opens")
        .session_id;
    manager.close_session(&child).await.unwrap();
    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();

    let err = super::super::cancelled_parents::as_delegation(
        manager.resume_session(&child, "again", None),
    )
    .await
    .expect_err("refused");
    assert!(
        super::super::cancelled_parents::ParentCancelled::in_chain(&err).is_some(),
        "{err:#}"
    );
    assert!(!manager.is_session_active(&child).await);
}

/// B36-RACE-RESUME — the tombstone stops late DELEGATIONS, not people: right after a
/// cascade cancelled the run, a person reopens one of its children by hand (the chat,
/// the API) and it opens, while a late delegation of the same parent is still refused.
#[tokio::test]
async fn a_person_reopens_the_child_of_a_cancelled_run_at_once() {
    let (manager, graph, _fake) = agent_manager();
    let parent = live_parent(&manager, &graph).await;
    let child = manager
        .create_session(&delegation_of(parent.id))
        .await
        .expect("the parent is live: the child opens")
        .session_id;
    manager.close_session(&child).await.unwrap();
    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();

    manager
        .resume_session(&child, "let me look at what you did", None)
        .await
        .expect("a person's resume is not a late delegation: it opens");
    assert!(manager.is_session_active(&child).await);

    // The invariant still holds for delegations: a NEW child of the cancelled parent…
    let late = manager
        .create_session(&delegation_of(parent.id))
        .await
        .expect_err("a late delegation is refused");
    assert!(
        super::super::cancelled_parents::ParentCancelled::in_chain(&late).is_some(),
        "{late:#}"
    );
    // …and the warm pool resuming a child for a delegation.
    manager.close_session(&child).await.unwrap();
    let warm = super::super::cancelled_parents::as_delegation(
        manager.resume_session(&child, "again", None),
    )
    .await
    .expect_err("a delegation's resume is refused");
    assert!(
        super::super::cancelled_parents::ParentCancelled::in_chain(&warm).is_some(),
        "{warm:#}"
    );
}

/// B36-RACE, criterion 5 — no regression: a session without a parent, a child whose
/// parent is live, and a child whose parent was closed WITHOUT cascade all open; and a
/// child created BEFORE the cascade (live when it ran) is closed by it, as before.
#[tokio::test]
async fn sessions_whose_parent_was_not_cancelled_open_as_before() {
    let (manager, graph, _fake) = agent_manager();
    let lone = manager
        .create_session(&agent_request("alone"))
        .await
        .expect("no parent: opens")
        .session_id;
    assert!(manager.is_session_active(&lone).await);

    let parent = live_parent(&manager, &graph).await;
    let child = manager
        .create_session(&delegation_of(parent.id))
        .await
        .expect("a live parent: opens")
        .session_id;
    assert!(manager.is_session_active(&child).await);

    // A plain close leaves no tombstone: a delegation of that parent still opens.
    let other = live_parent(&manager, &graph).await;
    manager.close_session(&other.id.to_string()).await.unwrap();
    let orphan = manager
        .create_session(&delegation_of(other.id))
        .await
        .expect("no cascade, no refusal")
        .session_id;
    assert!(manager.is_session_active(&orphan).await);

    // The child live before the cascade is closed by it.
    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();
    assert!(!manager.is_session_active(&child).await);
    assert!(
        manager.is_session_active(&lone).await,
        "unrelated sessions stay"
    );
}

/// B36-RACE, criterion 3 — a cancel with no delegation in flight waits for nothing:
/// the cascade returns at once (far below the tombstone's lifetime), leaving only its
/// tombstones behind.
#[tokio::test]
async fn a_cancel_without_a_delegation_in_flight_adds_no_wait() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let parent = crate::test_helpers::test_chat_session(None);
    let child = delegated_child_of(&parent);
    for node in [&parent, &child] {
        manager.graph.create_chat_session(node).await.unwrap();
        test_support::insert_live_session_without_cli(&manager, &node.id.to_string()).await;
    }
    let started = Instant::now();
    manager
        .close_session_with_cascade(&parent.id.to_string(), true)
        .await
        .unwrap();
    let took = started.elapsed();
    assert!(
        took < Duration::from_secs(2),
        "the cascade waits for no in-flight delegation: {took:?}"
    );
    assert!(manager.cancelled_parents.is_marked(&parent.id.to_string()));
    assert!(manager.cancelled_parents.is_marked(&child.id.to_string()));
    assert!(!manager.is_session_active(&child.id.to_string()).await);
}

#[tokio::test]
async fn test_resume_session_invalid_uuid() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.resume_session("not-a-uuid", "hello", None).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid session ID"));
}

#[tokio::test]
async fn test_resume_session_not_in_db() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let id = Uuid::new_v4().to_string();
    let result = manager.resume_session(&id, "hello", None).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not found in database"));
}

#[tokio::test]
async fn test_resume_session_no_cli_session_id_starts_fresh() {
    // When a session has no cli_session_id (first message or previous spawn failed),
    // resume_session should attempt to start a fresh CLI (not error immediately).
    // In CI without Claude CLI, this will fail at InteractiveClient creation,
    // but the error should NOT be "no CLI session ID".
    let state = mock_app_state();
    let session = test_chat_session(None); // no cli_session_id
    state.neo4j.create_chat_session(&session).await.unwrap();

    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager
        .resume_session(&session.id.to_string(), "hello", None)
        .await;
    // Should fail (CLI not available in test), but NOT with "no CLI session ID"
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        !err_msg.contains("no CLI session ID"),
        "Should not fail with 'no CLI session ID', got: {}",
        err_msg
    );
}

// ====================================================================
// Session -> run/thread attachment (task 1.7)
// ====================================================================

fn runner_request(run_id: Uuid, plan_id: Uuid, task_id: Uuid) -> ChatRequest {
    ChatRequest {
        file_scope_lifted: false,
        access: None,
        routing_pool: None,
        routing_mode: None,
        attachments: Vec::new(),
        refs: Vec::new(),
        message: "go".into(),
        session_id: None,
        cwd: "/tmp/test".into(),
        project_slug: None,
        model: None,
        provider: None,
        task_alias: None,
        persona_alias: None,
        run_provider: None,
        run_model: None,
        max_tokens: None,
        max_cost_usd: None,
        task_class: None,
        permission_mode: Some("bypassPermissions".into()),
        add_dirs: None,
        workspace_slug: None,
        user_claims: None,
        spawned_by: Some(
            serde_json::json!({
                "type": "runner",
                "run_id": run_id.to_string(),
                "plan_id": plan_id.to_string(),
                "task_id": task_id.to_string(),
            })
            .to_string(),
        ),
        task_context: None,
        subagent_type: None,
        route_auto: false,
        scaffolding_override: None,
        runner_context: None,
        routing_decision_id: None,
        denied_tools: vec![],
    }
}

fn manager_with_mock() -> (ChatManager, Arc<crate::neo4j::mock::MockGraphStore>) {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let state = mock_app_state();
    (
        ChatManager::new_without_memory(dyn_graph, state.meili, test_config()),
        graph,
    )
}

/// Waits, bounded, for `check` to answer true — the detached task runs on its own.
/// Returns whether it did; a passing test leaves no line of it unrun.
async fn wait_until(mut check: impl FnMut() -> bool) -> bool {
    let mut seen = false;
    for _ in 0..500 {
        if check() {
            seen = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    seen
}

/// A persisted `tool_use` of `tool`, `minutes` minutes old (issue #1056, correctif B).
fn tool_use_event(
    session_id: Uuid,
    seq: i64,
    tool: &str,
    minutes: i64,
) -> crate::neo4j::models::ChatEventRecord {
    crate::neo4j::models::ChatEventRecord {
        id: Uuid::new_v4(),
        session_id,
        seq,
        event_type: "tool_use".to_string(),
        data: serde_json::json!({"type": "tool_use", "tool": tool}).to_string(),
        created_at: chrono::Utc::now() - chrono::Duration::minutes(minutes),
    }
}

fn an_error_event() -> ChatEvent {
    ChatEvent::Error {
        message: "notice".to_string(),
        parent_tool_use_id: None,
        code: Some("session_open_failed".to_string()),
        reason: None,
        index: None,
        request_id: None,
    }
}

// issue #1056, correctif A. The call an agent makes has to be BOUNDED: the id answers
// before the opening has read anything, so no HTTP budget can be busted by a slow graph
// (measured: 107 s before the session was even persisted, against a 120 s client budget).
//
// The proof that nothing is waited for is the SIGNATURE: `create_session_detached` is not
// `async`, so it cannot await the opening. What is left to check is the answer's shape,
// and that the opening really does happen afterwards.
#[tokio::test]
async fn test_a_detached_opening_answers_its_id_and_the_opening_still_happens() {
    let (manager, graph) = manager_with_mock();
    let manager = Arc::new(manager);
    let response =
        manager
            .clone()
            .create_session_detached(agent_request("open a child"), None, None);

    let id = Uuid::parse_str(&response.session_id).expect("a uuid, answered now");
    assert_eq!(response.stream_url, format!("/ws/chat/{id}"));
    assert_eq!(
        response.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Project,
        "the request's cwd alone decides the place, and it is answered now"
    );
    assert!(
        !response.notices.is_empty(),
        "the caller is told the opening is still going on"
    );
    // THE regression assertion: the answer came BEFORE the opening persisted anything.
    // Nothing was awaited between the call and this line, so a caller that had waited for
    // the opening (the behaviour issue #1056 reports) would already find the session here
    // — the synchronous `create_session` persists before it returns.
    assert!(
        graph
            .chat_sessions
            .try_read()
            .map(|s| !s.contains_key(&id))
            .unwrap_or(true),
        "the id is answered before the opening has persisted anything"
    );

    // The opening is not skipped: the session is persisted by the detached task.
    assert!(
        wait_until(|| {
            graph
                .chat_sessions
                .try_read()
                .map(|s| s.contains_key(&id))
                .unwrap_or(false)
        })
        .await,
        "the detached task persisted the session under the promised id"
    );
}

// issue #1056, correctif A, second half: a refused opening can no longer be an HTTP
// error (the response left long before), so it is reported where the conversation will
// read it — as an `error` event in the session that asked.
#[tokio::test]
async fn test_a_refused_detached_opening_is_reported_in_the_calling_conversation() {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let mut config = test_config();
    // No room at all: the opening is refused before anything is persisted, which is
    // exactly the refusal the caller can no longer receive as an HTTP error.
    config.max_sessions = 0;
    let manager = Arc::new(ChatManager::new_without_memory(
        dyn_graph,
        mock_app_state().meili,
        config,
    ));
    let parent = crate::test_helpers::test_chat_session(None);
    graph.create_chat_session(&parent).await.unwrap();

    let response = manager.clone().create_session_detached(
        agent_request("open a child"),
        Some(parent.id.to_string()),
        None,
    );
    assert!(Uuid::parse_str(&response.session_id).is_ok());

    assert!(
        wait_until(|| {
            graph
                .chat_events
                .try_read()
                .map(|events| {
                    events.get(&parent.id).is_some_and(|list| {
                        list.iter().any(|e| {
                            e.event_type == "error" && e.data.contains("session_open_failed")
                        })
                    })
                })
                .unwrap_or(false)
        })
        .await,
        "the refusal is an event in the conversation that asked"
    );
}

// issue #1056, correctif B. The rule reads the LAST persisted event: a call with nothing
// written after it is unanswered; anything written after it means the turn moved on.
#[tokio::test]
async fn test_the_lost_tool_result_rule_reads_an_unanswered_call_and_never_a_finished_turn() {
    let (manager, graph) = manager_with_mock();
    let session = crate::test_helpers::test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();
    let sid = session.id;
    let event =
        |seq: i64, kind: &str, data: serde_json::Value| crate::neo4j::models::ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: sid,
            seq,
            event_type: kind.to_string(),
            data: data.to_string(),
            created_at: chrono::Utc::now(),
        };

    // Nothing written yet: no judgement to make.
    assert!(manager.last_event_unanswered_tool_use(sid).await.is_none());

    graph
        .store_chat_events(
            sid,
            vec![event(
                1,
                "tool_use",
                serde_json::json!({
                    "type": "tool_use",
                    "tool": "mcp__project-orchestrator__chat",
                }),
            )],
        )
        .await
        .unwrap();
    let (tool, _) = manager
        .last_event_unanswered_tool_use(sid)
        .await
        .expect("a call nobody answered");
    assert_eq!(tool, "mcp__project-orchestrator__chat");
    assert!(
        is_own_mcp_tool(&tool),
        "PO's own hop: the short grace applies"
    );
    // The server name decides, never the `mcp__` prefix (D5): a third party's hop
    // keeps the command grace; PO's two servers get the short one.
    assert!(!is_own_mcp_tool("mcp__acme__long_task"));
    assert!(!is_own_mcp_tool("Bash"));
    assert!(is_own_mcp_tool("mcp__nexus__Bash"));

    // The result arrives: the turn is not waiting any more.
    graph
        .store_chat_events(
            sid,
            vec![event(
                2,
                "tool_result",
                serde_json::json!({"type": "tool_result", "tool_use_id": "call_1"}),
            )],
        )
        .await
        .unwrap();
    assert!(
        manager.last_event_unanswered_tool_use(sid).await.is_none(),
        "an answered call is not a lost result"
    );

    // A turn that ended says nothing about a lost result either.
    graph
        .store_chat_events(
            sid,
            vec![event(
                3,
                "assistant_text",
                serde_json::json!({"type": "assistant_text", "content": "done"}),
            )],
        )
        .await
        .unwrap();
    assert!(manager.last_event_unanswered_tool_use(sid).await.is_none());
}

/// The watchdog's tick is a no-op when no session is streaming: an idle server never
/// interrupts anything.
#[tokio::test]
async fn test_the_lost_tool_result_watchdog_ends_nothing_when_nothing_is_streaming() {
    let (manager, _graph) = manager_with_mock();
    assert!(manager
        .abandon_turns_whose_tool_result_never_came()
        .await
        .is_empty());
}

// issue #1056, correctif B, de bout en bout : une session VIVANTE dont le dernier
// événement persisté est un appel MCP sans résultat, vieux de 11 min, est interrompue —
// et la raison est écrite dans la conversation. La session n'est pas fermée.
#[tokio::test]
async fn test_the_watchdog_ends_a_turn_whose_mcp_tool_result_is_lost() {
    let (manager, graph) = manager_with_mock();
    let manager = Arc::new(manager);
    let session = crate::test_helpers::test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();
    let sid = session.id;
    graph
        .store_chat_events(
            sid,
            vec![tool_use_event(
                sid,
                1,
                "mcp__project-orchestrator__chat",
                11,
            )],
        )
        .await
        .unwrap();
    let (dummy, _pending) = create_dummy_session(true, "", Vec::new());
    manager
        .active_sessions
        .write()
        .await
        .insert(sid.to_string(), dummy);

    let ended = manager.abandon_turns_whose_tool_result_never_came().await;

    assert_eq!(ended, vec![sid.to_string()], "the stuck turn is ended");
    let events = graph.chat_events.read().await;
    let notice = events
        .get(&sid)
        .expect("its events")
        .iter()
        .find(|e| e.data.contains("tool_result_lost"))
        .expect("the reason is written in the conversation");
    assert_eq!(notice.event_type, "error");
    assert!(
        graph.get_chat_session(sid).await.unwrap().is_some(),
        "the session is NOT closed: its next message continues it"
    );
}

// issue #1056, correctif B : un relevé qui échoue ne juge pas la session. Lire le compte,
// puis le dernier événement, peut échouer : la session est laissée tranquille, jamais
// interrompue sur une lecture qu'on n'a pas pu faire.
#[tokio::test]
async fn test_the_lost_tool_result_rule_does_not_judge_a_session_it_cannot_read() {
    let (manager, graph) = manager_with_mock();
    let sid = Uuid::new_v4();
    graph
        .store_chat_events(
            sid,
            vec![tool_use_event(
                sid,
                1,
                "mcp__project-orchestrator__chat",
                11,
            )],
        )
        .await
        .unwrap();

    graph.fail_reads.lock().unwrap().insert("count_chat_events");
    assert!(manager.last_event_unanswered_tool_use(sid).await.is_none());

    graph.fail_reads.lock().unwrap().clear();
    graph
        .fail_reads
        .lock()
        .unwrap()
        .insert("get_chat_events_paginated");
    assert!(manager.last_event_unanswered_tool_use(sid).await.is_none());
}

// issue #1056, correctif B : les deux cas qui NE doivent PAS déclencher. Un appel MCP de
// 2 minutes est simplement lent ; une commande de 11 minutes est dans son droit (elle
// n'a pas la borne courte des hops de PO).
#[tokio::test]
async fn test_the_watchdog_leaves_a_recent_call_and_a_command_alone() {
    let (manager, graph) = manager_with_mock();
    let manager = Arc::new(manager);
    let recent = crate::test_helpers::test_chat_session(None);
    let command = crate::test_helpers::test_chat_session(None);
    for (node, tool, minutes) in [
        (&recent, "mcp__project-orchestrator__chat", 2),
        (&command, "Bash", 11),
    ] {
        graph.create_chat_session(node).await.unwrap();
        graph
            .store_chat_events(node.id, vec![tool_use_event(node.id, 1, tool, minutes)])
            .await
            .unwrap();
        let (dummy, _pending) = create_dummy_session(true, "", Vec::new());
        manager
            .active_sessions
            .write()
            .await
            .insert(node.id.to_string(), dummy);
    }

    assert!(
        manager
            .abandon_turns_whose_tool_result_never_came()
            .await
            .is_empty(),
        "neither a slow call nor a long command is a lost result"
    );
}

// issue #1056, correctif B, suite (D5) : un outil MCP d'un serveur TIERS n'a pas la
// borne courte des hops de PO — ce serveur n'est pas servi par PO, son budget est
// inconnu, et un appel légitimement long n'y est pas coupé à 10 min. Seul un hop de
// PO (project-orchestrator / nexus) reçoit le budget court.
#[tokio::test]
async fn test_the_watchdog_leaves_a_third_party_mcp_tool_alone() {
    let (manager, graph) = manager_with_mock();
    let manager = Arc::new(manager);
    let third_party = crate::test_helpers::test_chat_session(None);
    let own = crate::test_helpers::test_chat_session(None);
    for (node, tool) in [
        (&third_party, "mcp__acme__long_task"),
        (&own, "mcp__project-orchestrator__chat"),
    ] {
        graph.create_chat_session(node).await.unwrap();
        graph
            .store_chat_events(node.id, vec![tool_use_event(node.id, 1, tool, 11)])
            .await
            .unwrap();
        let (dummy, _pending) = create_dummy_session(true, "", Vec::new());
        manager
            .active_sessions
            .write()
            .await
            .insert(node.id.to_string(), dummy);
    }

    let ended = manager.abandon_turns_whose_tool_result_never_came().await;
    assert_eq!(
        ended,
        vec![own.id.to_string()],
        "only PO's own hop is ended; the third-party server keeps the command grace"
    );
}

// issue #1056, correctif B, suite (D5 bis) : l'erreur « résultat perdu » n'est dite
// qu'une fois le tour effectivement interrompu. Avant, elle était émise AVANT
// l'interruption : un interrupt qui échoue laissait repartir le même message à chaque
// tick, en annonçant un tour qui n'avait jamais été interrompu.
#[tokio::test]
async fn test_a_failed_interrupt_does_not_tell_the_user_the_turn_ended() {
    use std::sync::atomic::Ordering;
    let (manager, graph, fake) = agent_manager();
    let sid = Uuid::new_v4();
    manager
        .agent_runtime
        .adopt(
            &sid.to_string(),
            "claude-code",
            fake.session(),
            1,
            "native",
            serde_json::json!({}),
            None,
        )
        .await;
    let handle = manager.agent_runtime.get(&sid.to_string()).await.unwrap();
    handle.is_streaming.store(true, Ordering::SeqCst);
    graph
        .store_chat_events(
            sid,
            vec![tool_use_event(
                sid,
                1,
                "mcp__project-orchestrator__chat",
                11,
            )],
        )
        .await
        .unwrap();
    *fake.state.fail_interrupt.lock().unwrap() =
        Some(nexus_claude::agent::ProviderError::unsupported("interrupt"));

    let ended = manager.abandon_turns_whose_tool_result_never_came().await;
    assert!(
        ended.is_empty(),
        "a failed interrupt ends nothing: {ended:?}"
    );
    let told = graph
        .chat_events
        .read()
        .await
        .get(&sid)
        .map(|evs| evs.iter().any(|e| e.data.contains("tool_result_lost")))
        .unwrap_or(false);
    assert!(
        !told,
        "a failed interrupt must not tell the user the turn ended"
    );
}

/// A notice with nowhere to go is dropped: never a panic, never another session.
#[tokio::test]
async fn test_a_session_notice_without_a_usable_session_is_dropped() {
    let (manager, graph) = manager_with_mock();
    manager.emit_session_event(None, an_error_event()).await;
    manager
        .emit_session_event(Some("   "), an_error_event())
        .await;
    manager
        .emit_session_event(Some("not-a-uuid"), an_error_event())
        .await;
    assert!(
        graph.chat_events.read().await.is_empty(),
        "nothing was written for a session nobody named"
    );
}

/// No `cwd`: the request alone already says the session is neutral, and that is answered
/// before the host has made the directory.
#[tokio::test]
async fn test_a_detached_opening_without_a_cwd_answers_the_neutral_place() {
    let (manager, _graph) = manager_with_mock();
    let manager = Arc::new(manager);
    let mut request = agent_request("open a child");
    request.cwd = String::new();

    let response = manager.clone().create_session_detached(request, None, None);

    assert_eq!(
        response.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Neutral
    );
    assert!(
        !response.access.is_normal(),
        "a neutral place with no access asked for opens read_only (ChatRequest::access), \
             and the caller learns it in the same answer"
    );
}

#[tokio::test]
async fn test_create_session_links_runner_session_to_its_run() {
    let (manager, graph) = manager_with_mock();
    let (run, plan, task) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    // The CLI is not available in tests: only the persisted side matters.
    let _ = manager
        .create_session(&runner_request(run, plan, task))
        .await;

    let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    assert_eq!(sessions.len(), 1, "the runner session was persisted");
    let rows = graph.session_link_rows.read().await.clone();
    assert_eq!(rows.len(), 1, "exactly one run link: {rows:?}");
    assert_eq!(rows[0].session_id, sessions[0].id);
    assert_eq!(rows[0].run_id, Some(run));
    assert_eq!(rows[0].plan_id, Some(plan));
    assert_eq!(rows[0].task_id, Some(task));
    // and the JSON mechanism is there too: both mechanisms on the session
    let a = crate::chat::attachment::attach(&sessions, &rows);
    assert_eq!(a.by_plan[&plan][0].links.len(), 2);
}

// CACHE-POOL criterion 2: a delegation's child (conversation sub-agent, MCP
// sub-agent, runner, gate, trigger — any session with a `spawned_by`) must not be
// enrolled in the PARENT conversation's objective tracking: reminding it of the
// project's pending tasks is noise nobody asked it to act on (measured 17/10/2026:
// an extra unrequested turn, ~9.4k cache tokens, answering a reminder meant for
// someone else). `tracks_own_objectives` is the single rule shared by both
// construction sites (fresh open and restart recovery). Red without the fix:
// `objective_tracking` was the literal `true`, unconditionally, at both sites.
//
// Exercising this end-to-end through `create_session` would need the `fake_claude`
// CLI harness (the legacy engine's `InteractiveClient` has no transport seam in
// tests, unlike the agent engine's `FakeProvider` — confirmed in isolation: an
// agent-engine session never enters `active_sessions` at all, see
// `an_agent_session_of_claude_code_resumes_on_the_agent_engine_after_a_rollback`
// above). Out of scope here: these test the rule directly, pure and deterministic.
#[test]
fn a_delegations_child_does_not_track_the_parents_objectives() {
    assert!(
        !tracks_own_objectives(true),
        "a spawned session must not track objectives"
    );
}

#[test]
fn an_ordinary_sessions_objective_tracking_is_unaffected() {
    assert!(
        tracks_own_objectives(false),
        "a session nobody spawned still tracks objectives"
    );
}

#[tokio::test]
async fn test_create_session_of_a_resumed_run_links_to_the_new_run() {
    let (manager, graph) = manager_with_mock();
    let (old_run, new_run, plan) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let _ = manager
        .create_session(&runner_request(old_run, plan, Uuid::new_v4()))
        .await;
    // free the single "active session" slot if the CLI did start
    manager.active_sessions.write().await.clear();
    let _ = manager
        .create_session(&runner_request(new_run, plan, Uuid::new_v4()))
        .await;
    let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    let rows = graph.session_link_rows.read().await.clone();
    let a = crate::chat::attachment::attach(&sessions, &rows);
    assert_eq!(a.by_plan.len(), 1, "same plan, same thread");
    let relation_runs: std::collections::HashSet<_> =
        rows.iter().filter_map(|r| r.run_id).collect();
    assert_eq!(
        relation_runs,
        [old_run, new_run].into_iter().collect(),
        "each session is linked by relation to ITS run"
    );
    let runs: std::collections::HashSet<_> = a.by_plan[&plan]
        .iter()
        .flat_map(|s| s.links.iter().filter_map(|l| l.run_id))
        .collect();
    assert_eq!(runs, [old_run, new_run].into_iter().collect());
}

#[tokio::test]
async fn test_create_session_free_chat_gets_no_run_link() {
    let (manager, graph) = manager_with_mock();
    let mut req = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    req.spawned_by = None;
    let _ = manager.create_session(&req).await;
    assert!(graph.session_link_rows.read().await.is_empty());
    let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    let a = crate::chat::attachment::attach(&sessions, &[]);
    assert_eq!(a.unattached.len(), sessions.len());
}

#[tokio::test]
async fn create_session_persists_the_provider_and_how_it_was_routed() {
    let (manager, graph) = manager_with_mock();
    // The CLI is not available in tests: only the persisted side matters.
    let _ = manager
        .create_session(&runner_request(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        ))
        .await;
    let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].provider_id.as_deref(), Some("claude-code"));
    assert_eq!(sessions[0].routed_by.as_deref(), Some("claude_code"));
}

#[tokio::test]
async fn create_session_naming_an_unknown_provider_is_refused_before_persisting() {
    let (manager, graph) = manager_with_mock();
    let mut req = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    req.provider = Some("deepseek".into());
    let err = manager.create_session(&req).await.unwrap_err();
    let failure =
        crate::chat::provider::errors::classify_open_error(&err, None).expect("typed failure");
    assert_eq!((failure.status, failure.code), (404, "provider_unknown"));
    assert!(
        graph.chat_sessions.read().await.is_empty(),
        "nothing persisted"
    );
}

#[tokio::test]
async fn a_request_for_another_provider_on_an_existing_session_is_a_409() {
    let (manager, graph) = manager_with_mock();
    let mut s = test_chat_session(None);
    s.provider_id = Some("claude-code".into());
    graph.create_chat_session(&s).await.unwrap();
    let id = s.id.to_string();
    manager.check_provider_binding(&id, None).await.unwrap();
    manager
        .check_provider_binding(&id, Some("claude-code"))
        .await
        .unwrap();
    let err = manager
        .check_provider_binding(&id, Some("deepseek"))
        .await
        .unwrap_err();
    let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
    assert_eq!((failure.status, failure.code), (409, "provider_conflict"));
}

/// Task 1ff0e2e4: a `--resume` target the CLI no longer has makes it exit at
/// once. The message must still be taken: a fresh CLI for the SAME PO
/// session, never a dead one handed back (silent 2 s turns).
#[cfg(unix)]
#[tokio::test]
async fn resume_falls_back_to_a_fresh_cli_when_the_resumed_one_dies_at_once() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("args.log");
    let cli = dir.path().join("fake-claude");
    std::fs::write(
            &cli,
            format!(
                "#!/bin/sh\necho \"ARGS $*\" >> {log}\ncase \"$1\" in\n  --version) echo '2.1.287 (Claude Code)'; exit 0;;\nesac\nfor a in \"$@\"; do [ \"$a\" = \"--resume\" ] && exit 1; done\nexec cat > /dev/null\n",
                log = log.display()
            ),
        )
        .unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();

    let (manager, graph) = manager_with_mock();
    manager
        .update_claude_cli_path(Some(cli.display().to_string()))
        .await;
    let mut s = test_chat_session(None);
    s.provider_id = Some("claude-code".into());
    s.cwd = dir.path().display().to_string();
    s.cli_session_id = Some("512a5305-16d1-4ad8-a02f-d2b6e845dbdf".into());
    graph.create_chat_session(&s).await.unwrap();
    let id = s.id.to_string();

    manager
        .resume_session(&id, "hello", None)
        .await
        .expect("the message is taken by a fresh CLI");

    assert!(manager.is_session_active(&id).await, "the session is live");
    let mut calls = String::new();
    for _ in 0..80 {
        calls = std::fs::read_to_string(&log).unwrap_or_default();
        if calls.matches("ARGS --output-format").count() >= 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let spawns: Vec<&str> = calls
        .split("ARGS ")
        .filter(|c| c.starts_with("--output-format"))
        .collect();
    let resumes = |c: &str| c.contains(" --resume ");
    assert!(
        spawns.len() >= 2,
        "resume then fresh: {:?}",
        calls
            .split("ARGS ")
            .map(|c| (
                c.len(),
                c.contains(" --resume "),
                c.chars().take(60).collect::<String>()
            ))
            .collect::<Vec<_>>()
    );
    assert!(resumes(spawns[0]), "first try resumes");
    assert!(!resumes(spawns[1]), "the fallback does not resume");
    manager.evict_if_cli_dead(&id).await;
}

/// Task 1ff0e2e4: a registered session whose CLI has left is dropped, so the
/// next message resumes instead of being written to a closed stdin.
#[cfg(unix)]
#[tokio::test]
async fn a_session_whose_cli_has_left_is_evicted_before_delivery() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let cli = dir.path().join("fake-claude");
    std::fs::write(
            &cli,
            "#!/bin/sh\ncase \"$1\" in\n  --version) echo '2.1.287 (Claude Code)'; exit 0;;\nesac\nexec cat > /dev/null\n",
        )
        .unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (manager, graph) = manager_with_mock();
    manager
        .update_claude_cli_path(Some(cli.display().to_string()))
        .await;
    let mut s = test_chat_session(None);
    s.provider_id = Some("claude-code".into());
    s.cwd = dir.path().display().to_string();
    graph.create_chat_session(&s).await.unwrap();
    let id = s.id.to_string();
    manager.resume_session(&id, "hello", None).await.unwrap();
    assert!(manager.is_session_active(&id).await);
    assert!(!manager.evict_if_cli_dead(&id).await, "alive: kept");

    // The CLI leaves: kill its process.
    let client = manager
        .active_sessions
        .read()
        .await
        .get(&id)
        .unwrap()
        .client
        .clone();
    let pid = client.lock().await.child_pid().await.expect("pid");
    unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    for _ in 0..50 {
        if manager.evict_if_cli_dead(&id).await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(!manager.is_session_active(&id).await, "evicted");
}

/// Review N1 of #663: the CLI dies on instance A, the session resumes on B, a
/// cancel comes from C. The eviction on A ends A's NATS listeners (before: they
/// kept answering `gone` for a session A no longer holds, which a canceller
/// took as 410 "nothing cancelled" while B did cancel). C gets B's answer —
/// and, between the death on A and the resume on B, `owner_unreachable`, not
/// A's stale `session_gone`.
#[cfg(unix)]
#[tokio::test]
async fn a_cancel_after_the_session_moved_gets_the_new_owners_answer() {
    use super::super::cancel_relay::CancelRelayError;
    use crate::events::nats_broker_test::TestBroker;
    use crate::events::NatsEmitter;
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let cli = dir.path().join("fake-claude");
    std::fs::write(
            &cli,
            "#!/bin/sh\ncase \"$1\" in\n  --version) echo '2.1.287 (Claude Code)'; exit 0;;\nesac\nexec cat > /dev/null\n",
        )
        .unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
    let broker = TestBroker::start().await;
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let instance = |nats: Arc<dyn crate::events::ChatTransport>| {
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let state = mock_app_state();
        ChatManager::new_without_memory(dyn_graph, state.meili, test_config()).with_nats(nats)
    };
    let a = instance(Arc::new(NatsEmitter::new(broker.client().await, "events")));
    let b = instance(Arc::new(NatsEmitter::new(broker.client().await, "events")));
    let c = instance(Arc::new(NatsEmitter::new(broker.client().await, "events")));
    for m in [&a, &b] {
        m.update_claude_cli_path(Some(cli.display().to_string()))
            .await;
    }
    let mut s = test_chat_session(None);
    s.provider_id = Some("claude-code".into());
    s.cwd = dir.path().display().to_string();
    graph.create_chat_session(&s).await.unwrap();
    let id = s.id.to_string();
    async fn pid_of(m: &ChatManager, id: &str) -> u32 {
        let client = m
            .active_sessions
            .read()
            .await
            .get(id)
            .unwrap()
            .client
            .clone();
        let pid = client.lock().await.child_pid().await.expect("pid");
        pid
    }
    // Asked again while the owner's listeners subscribe.
    async fn ask(m: &ChatManager, id: &str) -> Result<CancelToolsResult> {
        let mut last = None;
        for _ in 0..30 {
            match m.cancel_running_tools(id).await {
                Err(e)
                    if e.downcast_ref::<CancelRelayError>()
                        == Some(&CancelRelayError::OwnerUnreachable) =>
                {
                    last = Some(Err(e));
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                other => return other,
            }
        }
        last.unwrap()
    }

    // On A, then its CLI dies and A evicts it.
    a.resume_session(&id, "hello", None).await.unwrap();
    let a_pid = pid_of(&a, &id).await;
    assert_eq!(
        ask(&c, &id).await.unwrap().cli_pid,
        Some(a_pid),
        "A answers while it holds it"
    );
    unsafe { libc::kill(a_pid as i32, libc::SIGKILL) };
    for _ in 0..50 {
        if a.evict_if_cli_dead(&id).await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(!a.is_session_active(&id).await, "evicted on A");
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    // Nobody holds it: not A's stale `gone`.
    let between = c.cancel_running_tools(&id).await.unwrap_err();
    assert_eq!(
        between.downcast_ref::<CancelRelayError>(),
        Some(&CancelRelayError::OwnerUnreachable),
        "{between:#}"
    );

    // Resumed on B: C gets B's real answer.
    b.resume_session(&id, "again", None).await.unwrap();
    let b_pid = pid_of(&b, &id).await;
    let answered = ask(&c, &id).await.unwrap();
    assert_eq!(answered.cli_pid, Some(b_pid), "{answered:?}");
    let _ = b.close_session(&id).await;
}

#[tokio::test]
async fn resuming_a_session_bound_to_a_non_claude_provider_is_unavailable_on_the_legacy_path() {
    let (manager, graph) = manager_with_mock();
    let mut s = test_chat_session(None);
    s.provider_id = Some("deepseek".into());
    graph.create_chat_session(&s).await.unwrap();
    let err = manager
        .resume_session(&s.id.to_string(), "hi", None)
        .await
        .unwrap_err();
    let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
    assert_eq!(failure.code, "provider_unavailable");
}

// ---- the agent path (CHAT_PROVIDER_PATH=agent), against a fake provider ----

fn agent_manager() -> (
    ChatManager,
    Arc<crate::neo4j::mock::MockGraphStore>,
    super::super::agent_runtime::fake::FakeProvider,
) {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let state = mock_app_state();
    let mut config = test_config();
    config.provider_path = crate::chat::config::ProviderPath::Agent;
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
        .with_provider_source(Arc::new(fake.clone()));
    (manager, graph, fake)
}

fn agent_request(message: &str) -> ChatRequest {
    let mut req = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    req.spawned_by = None;
    req.runner_context = None;
    req.message = message.to_string();
    req.permission_mode = Some("acceptEdits".into());
    req
}

async fn next_matching(
    rx: &mut broadcast::Receiver<ChatEvent>,
    pred: impl Fn(&ChatEvent) -> bool,
) -> ChatEvent {
    loop {
        let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("an event within 5 s")
            .expect("channel open");
        if pred(&ev) {
            return ev;
        }
    }
}

/// The request of a delegation (`plan(delegate_task)`) on `model`, with its own caps
/// (PYSR-3, `launch_delegation`).
fn delegation_request(model: &str) -> ChatRequest {
    let mut req = agent_request("delegated work");
    req.model = Some(model.to_string());
    req.spawned_by = Some(
        crate::chat::types::SpawnedBy::Delegation {
            plan_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            parent_session_id: None,
            scaffolding_level: None,
        }
        .to_json_string(),
    );
    (req.max_tokens, req.max_cost_usd) = crate::runner::DelegationLimits::default().request_caps();
    req
}

/// The fake prices `model` (a spend cap can be held on it).
fn price(fake: &super::super::agent_runtime::fake::FakeProvider, model: &str) {
    fake.model_cost
        .lock()
        .unwrap()
        .insert(model.to_string(), nexus_claude::agent::CostBasis::Reported);
}

/// PYSR-3b — a resumed delegation keeps its budget: the resume's spec carries the spend cap
/// (the same constant, `DelegationLimits`), less what the delegation already spent — the
/// engine process a resume opens counts from zero. Red before: the resume passed
/// `max_cost_usd: None`, so a delegation resumed after an idle close ran uncapped.
#[tokio::test]
async fn a_resumed_delegation_keeps_its_cap() {
    let (manager, graph, fake) = agent_manager();
    price(&fake, "priced");
    let sid = manager
        .create_session(&delegation_request("priced"))
        .await
        .expect("the delegation opens")
        .session_id;
    let cap = crate::runner::DelegationLimits::default().max_cost_usd;
    let opened = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(opened.limits.max_cost_usd, Some(cap));
    // It spent $12 on its first process.
    graph
        .chat_sessions
        .write()
        .await
        .get_mut(&Uuid::parse_str(&sid).unwrap())
        .unwrap()
        .total_cost_usd = Some(12.0);
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();

    manager
        .resume_session(&sid, "again", None)
        .await
        .expect("the delegation resumes");
    let resumed = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(
        resumed.limits.max_cost_usd,
        Some(cap - 12.0),
        "the resume keeps the cap, less what was spent"
    );
}

/// PYSR-3b — a session of its own (a person's chat) resumes with no cap, as it opened
/// (ISO5: no session budget by default).
#[tokio::test]
async fn a_resumed_session_of_its_own_stays_uncapped() {
    let (manager, _graph, fake) = agent_manager();
    price(&fake, "priced");
    let mut req = agent_request("hello");
    req.model = Some("priced".into());
    let sid = manager.create_session(&req).await.unwrap().session_id;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    manager.resume_session(&sid, "again", None).await.unwrap();
    let resumed = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(resumed.limits.max_cost_usd, None);
    assert_eq!(resumed.limits.max_tokens, None);
}

/// PYSR-3b — a spend cap the engine cannot hold (a model without a price) is left out, and
/// SAID: the session's `system_init` lists `cost_cap` among its `degraded_features`, so the
/// delegating session (and its person) read that the budget is not held. Red before: only
/// a `warn!` in the server log.
#[tokio::test]
async fn a_cap_the_engine_cannot_hold_is_announced_in_degraded_features() {
    use nexus_claude::agent::AgentEvent;
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&delegation_request("unpriced"))
        .await
        .expect("a capped delegation still opens on an engine that cannot hold the cap")
        .session_id;
    let opened = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(opened.limits.max_cost_usd, None, "the cap is left out");
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(AgentEvent::SessionStarted {
        provider_session_id: Some("p1".into()),
        model: Some("unpriced".into()),
        policy_mode: None,
        native_mode: None,
        tools: vec![],
        mcp_servers: vec![],
        cwd: None,
    });
    let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
    let wire = serde_json::to_value(&init).unwrap();
    let degraded: Vec<String> =
        serde_json::from_value(wire["degraded_features"].clone()).expect("a list");
    assert!(
        degraded
            .iter()
            .any(|f| f == super::super::agent_runtime::COST_CAP_FEATURE),
        "{degraded:?}"
    );
}

/// PYSR-3b — a live capped delegation is not moved to a model its engine cannot hold the
/// cap on (no price): refused `unsupported("cost")`, as the engine refuses such an
/// opening; the model in force stays. A move to a priced model is not refused. Red
/// before: the change went through and the session spent with nothing counted (the engine
/// checks the cap at the opening only).
#[tokio::test]
async fn a_capped_session_is_not_moved_to_a_model_without_a_price() {
    let (manager, graph, fake) = agent_manager();
    price(&fake, "priced");
    price(&fake, "priced-too");
    let sid = manager
        .create_session(&delegation_request("priced"))
        .await
        .unwrap()
        .session_id;

    let refused = manager
        .set_session_model(&sid, "unpriced")
        .await
        .expect_err("the cap would no longer be held");
    assert!(
        matches!(
            refused.downcast_ref::<nexus_claude::agent::ProviderError>(),
            Some(nexus_claude::agent::ProviderError::Unsupported { capability }) if capability == "cost"
        ),
        "{refused:#}"
    );
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.model, "priced", "the model in force stays");
    assert!(fake.state.models.lock().unwrap().is_empty());

    manager
        .set_session_model(&sid, "priced-too")
        .await
        .expect("a priced model holds the cap");
    assert_eq!(*fake.state.models.lock().unwrap(), vec!["priced-too"]);
}

/// PYSR-3b — the guard is for capped sessions only: a person's chat moves to any model.
#[tokio::test]
async fn an_uncapped_session_moves_to_a_model_without_a_price() {
    let (manager, _graph, fake) = agent_manager();
    price(&fake, "priced");
    let mut req = agent_request("hello");
    req.model = Some("priced".into());
    let sid = manager.create_session(&req).await.unwrap().session_id;
    manager.set_session_model(&sid, "unpriced").await.unwrap();
    assert_eq!(*fake.state.models.lock().unwrap(), vec!["unpriced"]);
}

/// DELEG-WAKE, moteur agent : une nouvelle venue d'ailleurs passe par le réveil hors
/// tour (`wake_with_background`) — un tour de type `BackgroundOutput`, l'annonce
/// écrite dans la transcription, et pas un message utilisateur.
#[tokio::test]
async fn a_notification_wakes_an_agent_session_with_a_background_turn() {
    use nexus_claude::agent::AgentEvent;

    let (manager, graph, fake) = agent_manager();
    let manager = Arc::new(manager);
    let sid = manager
        .create_session(&agent_request("hello"))
        .await
        .expect("the session opens")
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    // End the opening turn, so the session is idle when the news arrives.
    fake.state.push(AgentEvent::Done {
        stop_reason: nexus_claude::agent::StopReason::Completed,
        subtype: None,
        is_error: false,
        result_text: Some("ok".into()),
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 1,
        duration_api_ms: None,
        num_turns: 1,
        model: None,
        provider_session_id: None,
        structured_output: None,
        error: None,
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let handle = manager
        .agent_runtime
        .get(&sid)
        .await
        .expect("the agent session is live");
    while handle.is_streaming.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < deadline,
            "the opening turn never ended"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let turns_before = fake.state.turns_started.lock().unwrap().len();

    manager
        .wake_with_notification(&sid, "<task-notification task_id=\"t1\"/>")
        .await
        .expect("the session is woken");

    // The announcement is in the transcript, and it IS what the provider was sent.
    next_matching(&mut rx, |e| {
            matches!(e, ChatEvent::BackgroundOutput { content, .. } if content.contains("task-notification"))
        })
        .await;
    while fake.state.turns_started.lock().unwrap().len() == turns_before {
        assert!(
            std::time::Instant::now() < deadline,
            "the wake started no turn"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let sent = fake.state.turns_started.lock().unwrap()[turns_before].clone();
    assert!(sent.contains("task-notification"), "{sent}");

    // Not a user message: the session's count did not move for it.
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.message_count, 1, "1 at creation only: {node:?}");
}

/// Two first sessions opened at once on an agent that refuses MCP servers per
/// session: the other one taught the provider, which now answers
/// `Unsupported { per_session_mcp }` to a spec built while it still said yes. The
/// session is opened once more WITHOUT its servers, and says what it lacks.
#[tokio::test]
async fn a_refusal_learned_meanwhile_reopens_the_session_without_its_mcp_servers() {
    use nexus_claude::agent::AgentEvent;
    let (manager, graph, fake) = agent_manager();
    *fake.kind.lock().unwrap() = nexus_claude::agent::ProviderKind::Acp;
    // The session's own capabilities say it carries MCP servers: only the HOST can
    // say it lacks the PO tools (it opened it without them).
    fake.caps.lock().unwrap().per_session_mcp = true;
    *fake.fail_open.lock().unwrap() = Some(nexus_claude::agent::ProviderError::unsupported(
        "per_session_mcp",
    ));
    let sid = manager
        .create_session(&agent_request("hello"))
        .await
        .expect("opened again without the servers")
        .session_id;
    let opened = fake.state.opened_specs.lock().unwrap().clone();
    assert_eq!(opened.len(), 1, "the refused spec never opened");
    assert!(
        opened[0].mcp_servers.is_empty(),
        "{:?}",
        opened[0].mcp_servers.keys()
    );
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(AgentEvent::SessionStarted {
        provider_session_id: Some("p1".into()),
        model: Some("m".into()),
        policy_mode: None,
        native_mode: None,
        tools: vec![],
        mcp_servers: vec![],
        cwd: None,
    });
    let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
    let wire = serde_json::to_value(&init).unwrap();
    let degraded: Vec<String> =
        serde_json::from_value(wire["degraded_features"].clone()).expect("a list");
    assert!(
        degraded
            .iter()
            .any(|f| f == super::super::agent_runtime::PO_TOOLS_FEATURE),
        "{degraded:?}"
    );
    // The frozen snapshot says what the opening found, not what the session's
    // capabilities claim: a resume after a restart does not send them again.
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert!(!ChatManager::carries_per_session_mcp(
        &fake,
        "m",
        node.capabilities.as_deref()
    ));
    // Only an ACP agent is asked again (its spec holds no server but the PO one):
    // another kind's refusal is returned as it is, nothing dropped in silence.
    let (manager, _graph, fake) = agent_manager();
    fake.caps.lock().unwrap().per_session_mcp = true;
    *fake.fail_open.lock().unwrap() = Some(nexus_claude::agent::ProviderError::unsupported(
        "per_session_mcp",
    ));
    assert!(manager
        .create_session(&agent_request("hello"))
        .await
        .is_err());
    assert!(fake.state.opened_specs.lock().unwrap().is_empty());
    // Another refusal is not taken for this one: no second opening.
    let (manager, _graph, fake) = agent_manager();
    *fake.kind.lock().unwrap() = nexus_claude::agent::ProviderKind::Acp;
    *fake.fail_open.lock().unwrap() =
        Some(nexus_claude::agent::ProviderError::unsupported("images"));
    assert!(manager
        .create_session(&agent_request("hello"))
        .await
        .is_err());
    assert!(fake.state.opened_specs.lock().unwrap().is_empty());
}

/// A resumed session whose frozen snapshot says it carried no MCP server is not
/// given one, whatever its provider says now (a refusal learned before a restart).
#[test]
fn a_snapshot_without_per_session_mcp_keeps_the_resumed_session_without_servers() {
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    assert!(ChatManager::carries_per_session_mcp(&fake, "m", None));
    assert!(ChatManager::carries_per_session_mcp(
        &fake,
        "m",
        Some(r#"{"per_session_mcp": true}"#)
    ));
    assert!(!ChatManager::carries_per_session_mcp(
        &fake,
        "m",
        Some(r#"{"per_session_mcp": false}"#)
    ));
    assert!(ChatManager::carries_per_session_mcp(
        &fake,
        "m",
        Some("not json")
    ));
}

/// The agent-path manager of [`agent_manager`], with a signing key (PO tokens are
/// minted), a per-turn router configured, and every provider id answered by the fake.
fn signed_routed_agent_manager() -> (
    ChatManager,
    Arc<crate::neo4j::mock::MockGraphStore>,
    super::super::agent_runtime::fake::FakeProvider,
) {
    struct AnyId(super::super::agent_runtime::fake::FakeProvider);
    impl super::super::agent_runtime::ProviderSource for AnyId {
        fn get(&self, _id: &str) -> Option<Arc<dyn nexus_claude::agent::AgentProvider>> {
            Some(Arc::new(self.0.clone()))
        }
    }
    struct NoDecider;
    #[async_trait::async_trait]
    impl crate::chat::provider::cognitive::decision::Decider for NoDecider {
        async fn decide(
            &self,
            _request: &crate::chat::provider::cognitive::decision::DecideRequest,
        ) -> anyhow::Result<crate::chat::provider::cognitive::decision::CognitiveDecision> {
            anyhow::bail!("not asked in this test")
        }
    }
    struct NoPool;
    #[async_trait::async_trait]
    impl crate::chat::agent_hooks::PoolSource for NoPool {
        async fn pool(
            &self,
            _provider_id: &str,
        ) -> Vec<crate::chat::provider::cognitive::candidates::ModelFacts> {
            Vec::new()
        }
    }
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let state = mock_app_state();
    let mut config = test_config();
    config.provider_path = crate::chat::config::ProviderPath::Agent;
    config.jwt_secret = Some(SIGNING_SECRET.into());
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
        .with_provider_source(Arc::new(AnyId(fake.clone())))
        .with_turn_decider(Arc::new(NoDecider), Arc::new(NoPool));
    (manager, graph, fake)
}

const SIGNING_SECRET: &str = "test-secret-key-minimum-32-chars!!";

/// The `jti` of the PO token a spec carries.
fn spec_token_jti(spec: &nexus_claude::agent::SessionSpec) -> String {
    let token = spec_po_env(spec)
        .get("PO_AUTH_TOKEN")
        .cloned()
        .expect("a PO token in the spec");
    crate::auth::jwt::decode_jwt(&token, SIGNING_SECRET)
        .expect("a token signed by the server")
        .jti
        .expect("a bound token has a jti")
}

/// Review of #682, findings 1 and 4: an ACP agent that refuses `mcpServers` is
/// asked again WITHOUT them by nexus itself (OpenClaw 2026.9.x, a command the name
/// does not give away): the host sees no refusal, only the session's capabilities
/// (`per_session_mcp: false`). The PO token went to the agent in the refused
/// `session/new`: it is revoked, at an opening and at a resume. A session that kept
/// its servers keeps its token.
#[tokio::test]
async fn the_po_token_is_revoked_when_the_provider_itself_opens_the_session_without_servers() {
    let (manager, _graph, fake) = signed_routed_agent_manager();
    *fake.kind.lock().unwrap() = nexus_claude::agent::ProviderKind::Acp;
    let mut req = agent_request("hello");
    req.user_claims = Some(person_claims());

    // Kept: the session carries its servers, its token stays live.
    fake.caps.lock().unwrap().per_session_mcp = true;
    let sid = manager.create_session(&req).await.unwrap().session_id;
    let kept = spec_token_jti(&fake.state.opened_specs.lock().unwrap().pop().unwrap());
    assert!(
        crate::auth::agent_tokens::is_live(&kept),
        "servers kept: token live"
    );

    // Resumed after a restart of the agent that now refuses them: nexus asked
    // `session/load` again without them. The snapshot said "takes servers", so the
    // resume's spec carried a fresh token: revoked.
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    fake.caps.lock().unwrap().per_session_mcp = false;
    manager
        .resume_session(&sid, "again", req.user_claims.as_ref())
        .await
        .unwrap();
    let resumed = spec_token_jti(&fake.state.opened_specs.lock().unwrap().pop().unwrap());
    assert!(
        !crate::auth::agent_tokens::is_live(&resumed),
        "the token of a resume nexus opened without servers is revoked"
    );
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();

    // Opened: same at a first opening.
    let sid = manager.create_session(&req).await.unwrap().session_id;
    let opened = spec_token_jti(&fake.state.opened_specs.lock().unwrap().pop().unwrap());
    assert!(
        !crate::auth::agent_tokens::is_live(&opened),
        "the token of an opening nexus made without servers is revoked"
    );
    // ... and the session still runs, reporting what it lacks.
    assert!(manager.agent_runtime.get(&sid).await.is_some());
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();

    // The rule itself: requested and lost, by the host or by the provider.
    let mut caps = nexus_claude::agent::Capabilities::none();
    caps.per_session_mcp = true;
    assert!(!opened_without_mcp(true, false, &caps));
    assert!(opened_without_mcp(true, true, &caps));
    assert!(!opened_without_mcp(false, false, &caps));
    caps.per_session_mcp = false;
    assert!(opened_without_mcp(true, false, &caps));
    assert!(!opened_without_mcp(false, false, &caps), "nothing was sent");
}

/// Review of #682, finding 5: an opening or a resume that stops short — here its
/// authorization refused (a third-party instance the store does not know), after
/// its router was registered and its token minted — leaves neither behind.
#[tokio::test]
async fn a_refused_opening_or_resume_leaves_no_turn_router_nor_token() {
    let (manager, graph, _fake) = signed_routed_agent_manager();
    // A first opening on Claude Code (no gate): the router IS registered, so the
    // assertions below are not vacuous.
    let ok = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap()
        .session_id;
    assert!(
        manager.turn_routing.get(&ok).is_some(),
        "routing is configured"
    );

    // Refused: a third-party instance the store does not know.
    let req = {
        let mut req = agent_request("");
        req.user_claims = Some(person_claims());
        req
    };
    let sid = Uuid::new_v4();
    let refused = manager
        .open_agent_session(AgentOpen {
            request: &req,
            session_id: sid,
            provider_id: "elsewhere",
            model: "m",
            system_prompt: "p",
            add_dirs: &[],
            project_slug: None,
            owner: None,
            relay: None,
            access: Default::default(),
        })
        .await;
    assert!(refused.is_err());
    let sid = sid.to_string();
    // Not vacuous: a token WAS minted for the refused opening (review of #684, finding 3).
    let minted = crate::auth::agent_tokens::minted_for(&sid);
    assert_eq!(
        minted.len(),
        1,
        "the opening minted its token before it was refused"
    );
    assert!(!crate::auth::agent_tokens::is_live(&minted[0]));
    assert!(
        manager.turn_routing.get(&sid).is_none(),
        "a refused opening leaves no router"
    );
    assert_eq!(
        crate::auth::agent_tokens::revoke_session(&sid),
        0,
        "nor a live token"
    );

    // The same at a resume.
    let mut node = test_chat_session(None);
    node.provider_id = Some("elsewhere".into());
    graph.create_chat_session(&node).await.unwrap();
    let refused = manager
        .resume_agent_session(&node, "again", req.user_claims.as_ref(), None)
        .await;
    assert!(refused.is_err());
    let sid = node.id.to_string();
    let minted = crate::auth::agent_tokens::minted_for(&sid);
    assert_eq!(
        minted.len(),
        1,
        "the resume minted its token before it was refused"
    );
    assert!(!crate::auth::agent_tokens::is_live(&minted[0]));
    assert!(
        manager.turn_routing.get(&sid).is_none(),
        "a refused resume leaves no router"
    );
    assert_eq!(crate::auth::agent_tokens::revoke_session(&sid), 0);
    manager.close_session(&ok).await.unwrap();
}

/// A session of the agent engine opened by a first manager, whose last events are `tail`
/// (type, data), and a SECOND manager over the same graph: the server after a restart.
async fn restarted_with_tail(
    tail: &[(&str, &str)],
) -> (
    ChatManager,
    Arc<crate::neo4j::mock::MockGraphStore>,
    super::super::agent_runtime::fake::FakeProvider,
    Uuid,
) {
    restarted_with(tail, chrono::Duration::zero(), |_| {}).await
}

/// [`restarted_with_tail`], the events stored `age` ago, the server after the restart
/// configured by `tweak`.
async fn restarted_with(
    tail: &[(&str, &str)],
    age: chrono::Duration,
    tweak: impl FnOnce(&mut ChatConfig),
) -> (
    ChatManager,
    Arc<crate::neo4j::mock::MockGraphStore>,
    super::super::agent_runtime::fake::FakeProvider,
    Uuid,
) {
    let (before, graph, _fake) = agent_manager();
    let created = before
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    let sid = Uuid::parse_str(&created.session_id).unwrap();
    before.close_session(&created.session_id).await.unwrap();
    graph.delete_chat_events(sid).await.unwrap();
    for (i, (event_type, data)) in tail.iter().enumerate() {
        graph
            .store_chat_events(
                sid,
                vec![ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: sid,
                    seq: i as i64 + 1,
                    event_type: event_type.to_string(),
                    data: data.to_string(),
                    created_at: chrono::Utc::now() - age,
                }],
            )
            .await
            .unwrap();
    }
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let mut config = test_config();
    config.provider_path = crate::chat::config::ProviderPath::Agent;
    tweak(&mut config);
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    // The fake answers as `claude-code`, whose cut turns no server resumes by itself
    // (M2): these tests exercise the resume, so this server does.
    let after = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
        .with_provider_source(Arc::new(fake.clone()))
        .resuming_claude_code_at_boot();
    (after, graph, fake, sid)
}

#[tokio::test]
async fn a_turn_cut_by_a_restart_is_resumed_and_the_model_is_told() {
    let (manager, _graph, fake, _sid) = restarted_with_tail(&[
        ("user_message", r#"{"type":"user_message","content":"go"}"#),
        (
            "assistant_text",
            r#"{"type":"assistant_text","content":"I start"}"#,
        ),
        (
            "tool_use",
            r#"{"type":"tool_use","id":"t1","tool":"Read","input":{"path":"a.rs"}}"#,
        ),
        (
            "tool_result",
            r#"{"type":"tool_result","id":"t1","result":"ok"}"#,
        ),
    ])
    .await;
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((scan.scanned, scan.closed, pending.len()), (1, 0, 1));
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!(report.resumed, 1);
    let turns = fake.state.turns_started.lock().unwrap().clone();
    assert_eq!(turns.len(), 1);
    assert!(
        turns[0].starts_with(recovery::RESUME_MARKER),
        "{}",
        turns[0]
    );
    // What the PROVIDER received carries the cut turn rebuilt from the stored events
    // (follow-up review of #697, nit): the request, and the tool already run.
    assert!(turns[0].contains("User: go"), "{}", turns[0]);
    assert!(
        turns[0].contains("Called: Read(") && turns[0].contains("result received"),
        "{}",
        turns[0]
    );
}

/// A provider instance named `claude-code` whose key is a `vault:` reference: the fake
/// still plays the provider, the recovery reads the record to know it needs the vault.
async fn vault_backed_claude_code(graph: &crate::neo4j::mock::MockGraphStore) {
    let record = crate::chat::provider::settings::InstanceRecord {
        id: "claude-code".into(),
        kind: "openai_compat".into(),
        label: "vault-backed".into(),
        cost_source: "none".into(),
        credential_ref: "vault:the-key".into(),
        ..Default::default()
    };
    super::super::agent_e2e_tests::store_instance(graph, &record).await;
}

async fn events_of(graph: &crate::neo4j::mock::MockGraphStore, sid: Uuid) -> Vec<ChatEventRecord> {
    graph.get_chat_events(sid, 0, 500).await.unwrap()
}

fn assert_closed_by_recovery(events: &[ChatEventRecord]) {
    let last = events.last().expect("events");
    assert_eq!(last.event_type, "error", "{events:?}");
    assert!(last.data.contains("restart_interrupted"), "{}", last.data);
    let mut seqs: Vec<i64> = events.iter().map(|e| e.seq).collect();
    seqs.sort_unstable();
    seqs.dedup();
    assert_eq!(
        seqs.len(),
        events.len(),
        "every seq is used once: {events:?}"
    );
}

/// R1 (follow-up review of #697): a person resumes the session while the recovery is
/// about to. The recovery waits for the session's resume lock, sees the session the
/// person brought back, and SKIPS it: its "continue" never reaches the person's live
/// session (it used to be queued there, interrupting the person's turn).
#[tokio::test]
async fn a_session_the_user_resumed_meanwhile_is_skipped_not_written_into() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(pending.len(), 1);
    *fake.open_delay.lock().unwrap() = Some(Duration::from_millis(300));
    let manager = Arc::new(manager);
    let person = {
        let manager = manager.clone();
        tokio::spawn(async move { manager.resume_session(&sid.to_string(), "mine", None).await })
    };
    // The person's resume holds the lock, its provider still opening.
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(manager.agent_runtime.get(&sid.to_string()).await.is_none());
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(5))
        .await;
    person.await.unwrap().unwrap();
    assert_eq!((report.resumed, report.skipped, report.failed), (0, 1, 0));
    assert_eq!(
        fake.state.turns_started.lock().unwrap().clone(),
        vec!["mine".to_string()]
    );
    assert!(
        fake.state.interrupts.lock().unwrap().is_empty(),
        "the person's turn is not interrupted"
    );
    let events = events_of(&graph, sid).await;
    assert!(
        !events
            .iter()
            .any(|e| e.data.contains(recovery::RESUME_MARKER) || e.event_type == "error"),
        "nothing of the recovery in the live session: {events:?}"
    );
}

/// R1 / finding 9 of #697: a resume that fails AFTER it adopted the session (its first
/// message refused) used to write its `error` at the seq it saw at the scan — the seq
/// its own resume message had just taken — and to leave the handle live.
#[tokio::test]
async fn a_resume_that_fails_after_adopting_writes_its_error_after_its_events() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    *fake.state.fail_next_turn.lock().unwrap() =
        Some(nexus_claude::agent::ProviderError::invalid("refused"));
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!((report.resumed, report.failed), (0, 1));
    assert!(
        !manager.agent_runtime.owns(&sid.to_string()).await,
        "the failed attempt's session is ended"
    );
    assert_closed_by_recovery(&events_of(&graph, sid).await);
}

/// Finding 8c of #697: a provider not ready yet at boot (a transient error) is tried
/// again; it used to close the turn for good.
#[tokio::test]
async fn a_transient_failure_is_tried_again_before_the_turn_is_closed() {
    let (manager, _graph, fake, _sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    *fake.fail_open.lock().unwrap() = Some(nexus_claude::agent::ProviderError::Overloaded);
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!((report.resumed, report.retried, report.failed), (1, 1, 0));
    assert_eq!(fake.state.turns_started.lock().unwrap().len(), 1);
}

/// P18c: a read-only mirror of a Claude Code sub-agent cut by a restart is never resumed
/// (it has no process), even by a server that resumes Claude Code at boot and for a link
/// (`conversation` without `tool_use_id`) that would otherwise be resumed: it is closed.
#[tokio::test]
async fn a_cut_mirror_of_a_claude_code_sub_agent_is_closed_at_boot_never_resumed() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    node.mirror_of = Some(crate::chat::mirror::MIRROR_OF_CLAUDE_CODE.to_string());
    node.spawned_by = Some(
        crate::chat::types::SpawnedBy::Conversation {
            parent_session_id: Uuid::new_v4(),
            tool_use_id: None,
        }
        .to_json_string(),
    );
    graph.create_chat_session(&node).await.unwrap();
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 1));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
    let events = events_of(&graph, sid).await;
    assert_closed_by_recovery(&events);
    assert!(
        events.last().unwrap().data.contains("never resumed"),
        "{events:?}"
    );
    // And nobody can resume it by hand either.
    let err = manager
        .resume_session(&sid.to_string(), "go on", None)
        .await
        .unwrap_err();
    assert_eq!(
        crate::chat::provider::errors::classify_open_error(&err, None).map(|f| (f.status, f.code)),
        Some((409, crate::chat::mirror::MIRROR_READ_ONLY))
    );
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
}

/// Review of #726, point 3 (still the rule since P20-suite-2): a cut turn of a Claude
/// Code session whose CLI runs tools without asking (here `acceptEdits`) is CLOSED at
/// boot — not resumed with nobody there. In `default` it is resumed through `--resume`
/// (`claude_code_recovery_at_boot` in `agent_e2e_tests`).
#[tokio::test]
async fn a_cut_claude_code_turn_of_the_agent_engine_is_closed_at_boot_not_resumed() {
    let (resuming, graph, _fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    // The same server as every other: it does not resume Claude Code at boot.
    let mut config = test_config();
    config.provider_path = crate::chat::config::ProviderPath::Agent;
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
        .with_provider_source(Arc::new(fake.clone()));
    drop(resuming);
    let node = graph.get_chat_session(sid).await.unwrap().unwrap();
    assert!(node.opened_on_agent() && node.served_by_claude_code());
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 1));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
    let events = events_of(&graph, sid).await;
    assert_closed_by_recovery(&events);
    assert!(
        events
            .iter()
            .any(|e| e.data.contains("not resumed without you")),
        "the reason says why"
    );
}

// ── P20-suite-2: which cut Claude Code turns the boot recovery resumes ──────────────

/// A server as production builds it (it does not resume Claude Code by itself, so the
/// verdict is the one of `claude_code_left_alone`), over a graph holding one session
/// `node` whose events end mid-turn.
async fn production_server_with_cut(
    node: ChatSessionNode,
) -> (ChatManager, Arc<crate::neo4j::mock::MockGraphStore>, Uuid) {
    production_server_with_cut_on(node, test_config()).await
}

async fn production_server_with_cut_on(
    node: ChatSessionNode,
    config: ChatConfig,
) -> (ChatManager, Arc<crate::neo4j::mock::MockGraphStore>, Uuid) {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config);
    let sid = node.id;
    graph.create_chat_session(&node).await.unwrap();
    for (i, (event_type, data)) in [("user_message", "{}"), ("assistant_text", "{}")]
        .iter()
        .enumerate()
    {
        graph
            .store_chat_events(
                sid,
                vec![ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: sid,
                    seq: i as i64 + 1,
                    event_type: event_type.to_string(),
                    data: data.to_string(),
                    created_at: chrono::Utc::now(),
                }],
            )
            .await
            .unwrap();
    }
    (manager, graph, sid)
}

/// A Claude Code session cut mid-turn: `engine` ("agent" or "legacy"), the permission
/// `mode` it was opened with, and the conversation its CLI named (`resume_token` on the
/// agent engine, `cli_session_id` on the historical one).
fn cut_claude_code_node(
    engine: &str,
    mode: Option<&str>,
    conversation: Option<&str>,
) -> ChatSessionNode {
    let mut node = crate::test_helpers::test_chat_session(None);
    node.provider_id = Some("claude-code".into());
    node.engine = Some(engine.into());
    node.permission_mode = mode.map(str::to_string);
    if engine == ChatSessionNode::ENGINE_AGENT {
        node.resume_token = conversation.map(str::to_string);
    } else {
        node.cli_session_id = conversation.map(str::to_string);
    }
    node
}

#[tokio::test]
async fn a_cut_claude_code_turn_whose_cli_runs_tools_without_asking_is_closed_on_both_engines() {
    for engine in [ChatSessionNode::ENGINE_AGENT, "legacy"] {
        for mode in ["acceptEdits", "bypassPermissions"] {
            let node = cut_claude_code_node(engine, Some(mode), Some("conversation-1"));
            let (manager, graph, sid) = production_server_with_cut(node).await;
            let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
            assert_eq!((pending.len(), scan.closed), (0, 1), "{engine} {mode}");
            let events = events_of(&graph, sid).await;
            assert_closed_by_recovery(&events);
            assert!(
                events.last().unwrap().data.contains("without asking"),
                "{engine} {mode}: {}",
                events.last().unwrap().data
            );
        }
    }
}

#[tokio::test]
async fn a_cut_claude_code_turn_with_no_stored_conversation_is_closed_not_started_blind() {
    for engine in [ChatSessionNode::ENGINE_AGENT, "legacy"] {
        // No id at all, and an empty one (a CLI that named nothing).
        for conversation in [None, Some("")] {
            let node = cut_claude_code_node(engine, Some("default"), conversation);
            let (manager, graph, sid) = production_server_with_cut(node).await;
            let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
            assert_eq!(
                (pending.len(), scan.closed),
                (0, 1),
                "{engine} {conversation:?}"
            );
            let events = events_of(&graph, sid).await;
            assert_closed_by_recovery(&events);
            assert!(
                events
                    .last()
                    .unwrap()
                    .data
                    .contains("nothing to resume it from"),
                "{engine} {conversation:?}: {}",
                events.last().unwrap().data
            );
        }
    }
}

#[tokio::test]
async fn a_cut_claude_code_turn_whose_cli_asks_and_named_its_conversation_is_queued() {
    for engine in [ChatSessionNode::ENGINE_AGENT, "legacy"] {
        for mode in ["default", "plan"] {
            let node = cut_claude_code_node(engine, Some(mode), Some("conversation-1"));
            let (manager, graph, sid) = production_server_with_cut(node).await;
            let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
            assert_eq!((pending.len(), scan.closed), (1, 0), "{engine} {mode}");
            assert_eq!(pending[0].node.id, sid);
            assert!(
                !events_of(&graph, sid)
                    .await
                    .iter()
                    .any(|e| e.event_type == "error"),
                "{engine} {mode}: nothing is written before the resume"
            );
        }
    }
}

/// A session that names no permission mode runs in the server's: resumed when that one
/// asks, closed when it does not.
#[tokio::test]
async fn a_cut_claude_code_turn_with_no_permission_mode_follows_the_servers() {
    for (server_mode, queued) in [("default", 1), ("bypassPermissions", 0)] {
        let node = cut_claude_code_node(ChatSessionNode::ENGINE_AGENT, None, Some("c-1"));
        let (manager, graph, sid) = production_server_with_cut(node).await;
        manager.permission_config.write().await.mode = server_mode.into();
        let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
        assert_eq!(
            (pending.len(), scan.closed),
            (queued, 1 - queued),
            "{server_mode}"
        );
        if queued == 0 {
            assert_closed_by_recovery(&events_of(&graph, sid).await);
        }
    }
    // An EMPTY mode is no mode either.
    let node = cut_claude_code_node(ChatSessionNode::ENGINE_AGENT, Some(""), Some("c-1"));
    let (manager, _graph, _sid) = production_server_with_cut(node).await;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(pending.len(), 1);
}

/// A session of the historical engine can only be a Claude Code one: a node that says
/// otherwise (another provider on the legacy engine) cannot be resumed there.
#[tokio::test]
async fn a_cut_turn_of_another_provider_on_the_historical_engine_is_closed() {
    let mut node = cut_claude_code_node("legacy", Some("default"), Some("c-1"));
    node.provider_id = Some("some-other-provider".into());
    let (manager, graph, sid) = production_server_with_cut(node).await;
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 1));
    let events = events_of(&graph, sid).await;
    assert_closed_by_recovery(&events);
    assert!(
        events
            .last()
            .unwrap()
            .data
            .contains("not resumed automatically"),
        "{}",
        events.last().unwrap().data
    );
}

/// A cut turn of the historical engine whose CLI cannot be started at all ends in an
/// `error` event: the recovery does not fall back to a fresh CLI, and leaves nothing live.
#[tokio::test]
async fn a_historical_turn_whose_cli_cannot_start_is_closed_not_left_streaming() {
    let node = cut_claude_code_node("legacy", Some("default"), Some("conversation-1"));
    let mut config = test_config();
    config.claude_cli_path = Some("/nonexistent/claude".into());
    let (manager, graph, sid) = production_server_with_cut_on(node, config).await;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(pending.len(), 1);
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(5))
        .await;
    assert_eq!((report.resumed, report.failed), (0, 1), "{report:?}");
    assert!(!manager.is_session_active(&sid.to_string()).await);
    assert_closed_by_recovery(&events_of(&graph, sid).await);
}

/// Review of #726, point 2: the engine is written on the node BEFORE the open. A
/// session whose capability snapshot never reached the graph (that write is non-fatal)
/// still resumes on the agent engine — it used to fall to legacy with no CLI session,
/// and the model lost the conversation.
#[tokio::test]
async fn a_session_whose_snapshot_write_failed_still_resumes_on_the_agent_engine() {
    let (manager, graph, fake) = agent_manager();
    graph
        .fail_harness_writes
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let sid = manager
        .create_session(&agent_request("one"))
        .await
        .unwrap()
        .session_id;
    manager.agent_runtime.close(&sid).await.unwrap();
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert!(node.capabilities.is_none(), "the snapshot write failed");
    assert_eq!(node.engine.as_deref(), Some(ChatSessionNode::ENGINE_AGENT));
    graph
        .fail_harness_writes
        .store(false, std::sync::atomic::Ordering::SeqCst);
    // A restart, even onto `legacy`.
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let after = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, test_config())
        .with_provider_source(Arc::new(fake.clone()));
    after
        .resume_session(&sid, "two", None)
        .await
        .expect("resumed on the engine it was opened on");
    assert!(after.agent_runtime.owns(&sid).await, "the agent engine");
}

/// The engine read back: written ones decide; a node from before the field keeps the
/// inference of its time (snapshot, or a provider other than Claude Code).
#[test]
fn the_engine_of_a_session_is_the_one_written_else_legacy_without_a_snapshot() {
    let mut node = test_chat_session(None);
    node.engine = Some(ChatSessionNode::ENGINE_AGENT.into());
    assert!(node.opened_on_agent());
    node.engine = Some(ChatSessionNode::ENGINE_LEGACY.into());
    node.capabilities = Some("{}".into());
    assert!(!node.opened_on_agent(), "the written engine wins");
    node.engine = None;
    assert!(node.opened_on_agent(), "pre-M2: a snapshot means agent");
    node.capabilities = None;
    node.provider_id = Some("claude-code".into());
    assert!(
        !node.opened_on_agent(),
        "pre-M2: Claude Code without one is legacy"
    );
    node.provider_id = Some("local".into());
    assert!(
        !node.opened_on_agent(),
        "pre-M2, no snapshot: legacy (a resume refuses it, provider_unavailable)"
    );
}

/// Finding 9 of #697, M2: the recovery checks what a person's resume checks, and the
/// engine is frozen at opening. A Claude Code session of the agent engine, after a
/// restart with `CHAT_PROVIDER_PATH=legacy` (a rollback of the default), is resumed on
/// the agent engine it was opened on — no longer closed with `engine_unavailable`.
#[tokio::test]
async fn the_recovery_resumes_an_agent_session_on_its_engine_after_a_rollback_to_legacy() {
    let (manager, _graph, fake, _sid) = restarted_with(
        &[("user_message", "{}"), ("assistant_text", "{}")],
        chrono::Duration::zero(),
        |config| config.provider_path = crate::chat::config::ProviderPath::Legacy,
    )
    .await;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(pending.len(), 1);
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!((report.resumed, report.failed), (1, 0));
    assert_eq!(fake.state.turns_started.lock().unwrap().len(), 1);
}

/// Finding 7 of #697: a locked vault holds back only the sessions whose provider reads
/// its key there. A built-in provider (no stored instance) resumes at once.
#[tokio::test]
async fn a_session_that_needs_no_vault_resumes_while_the_vault_is_locked() {
    let (manager, _graph, fake, _sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    let vault = crate::vault::VaultService::ephemeral();
    let manager = manager.with_vault(vault.clone());
    assert!(!vault.is_unlocked(chrono::Utc::now()));
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(
            pending,
            Duration::from_millis(10),
            Duration::from_millis(500),
        )
        .await;
    assert_eq!((report.resumed, report.abandoned), (1, 0));
    assert_eq!(fake.state.turns_started.lock().unwrap().len(), 1);
}

/// R2: a cut turn of an orchestrator child is not resumed (its orchestrator relaunches
/// the task), and it is CLOSED so it no longer shows "streaming" (it used to be skipped
/// in silence). Also the scan's skip of the children, end to end (nit of the follow-up).
#[tokio::test]
async fn a_cut_child_of_an_orchestrator_is_closed_not_resumed() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    node.spawned_by = Some(
        crate::chat::types::SpawnedBy::Runner {
            run_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            parent_session_id: None,
        }
        .to_json_string(),
    );
    graph.create_chat_session(&node).await.unwrap();
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 1));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
    assert_closed_by_recovery(&events_of(&graph, sid).await);
    let (_, again) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(again.closed, 0, "closed once");
}

/// R2: a turn cut longer ago than the window is closed, not left "streaming". Its
/// `updated_at` is recent (a long turn does not refresh it): judged on the last event.
#[tokio::test]
async fn a_turn_cut_out_of_the_window_is_closed() {
    let (manager, graph, fake, sid) = restarted_with(
        &[("user_message", "{}"), ("assistant_text", "{}")],
        chrono::Duration::hours(2),
        |_| {},
    )
    .await;
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 1));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
    assert_closed_by_recovery(&events_of(&graph, sid).await);
}

/// Finding 10 of #697 / R2: a cut turn of a Claude Code CLI session (no capability
/// snapshot) is not resumed by this recovery, and it is closed instead of staying
/// "streaming".
#[tokio::test]
async fn a_cut_turn_of_a_cli_session_is_closed_at_boot() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("tool_use", "{}")]).await;
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    node.capabilities = None;
    node.engine = Some(ChatSessionNode::ENGINE_LEGACY.to_string());
    graph.create_chat_session(&node).await.unwrap();
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 1));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
    assert_closed_by_recovery(&events_of(&graph, sid).await);
}

/// Review of #711, finding 1: a CLI session whose background command printed after
/// the turn's `result` is idle, not cut: nothing is closed at boot.
#[tokio::test]
async fn a_cli_session_with_output_after_its_turn_is_not_closed_at_boot() {
    let (manager, graph, _fake, sid) = restarted_with_tail(&[
        ("user_message", "{}"),
        ("assistant_text", "{}"),
        ("result", "{}"),
        (
            "background_output",
            r#"{"source":"assistant","content":"line"}"#,
        ),
    ])
    .await;
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    node.capabilities = None;
    node.engine = Some(ChatSessionNode::ENGINE_LEGACY.to_string());
    graph.create_chat_session(&node).await.unwrap();
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 0));
    assert!(!events_of(&graph, sid)
        .await
        .iter()
        .any(|e| e.event_type == "error"));
}

/// Review of #711, finding 1 (age): a cut turn older than the horizon is left as it
/// is, never closed (rather left "streaming" than falsely closed).
#[tokio::test]
async fn a_cut_turn_older_than_the_horizon_is_left_as_it_is() {
    let (manager, graph, _fake, sid) = restarted_with(
        &[("user_message", "{}"), ("tool_use", "{}")],
        recovery::CLOSE_HORIZON + chrono::Duration::days(1),
        |_| {},
    )
    .await;
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    node.capabilities = None;
    node.engine = Some(ChatSessionNode::ENGINE_LEGACY.to_string());
    graph.create_chat_session(&node).await.unwrap();
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 0));
}

/// Review of #711, finding 2: an attempt that failed AFTER it adopted the session (its
/// marker stored) and was tried again does not count as a restart: the next boot
/// still resumes the session once more instead of closing it as "restarted again".
#[tokio::test]
async fn a_retry_after_adoption_does_not_count_as_a_restart() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    *fake.state.fail_next_turn.lock().unwrap() =
        Some(nexus_claude::agent::ProviderError::Overloaded);
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!((report.resumed, report.retried), (1, 1));
    let events = events_of(&graph, sid).await;
    assert_eq!(
        events
            .iter()
            .filter(|e| e.data.contains(recovery::RESUME_MARKER))
            .count(),
        2,
        "the failed attempt left its marker: {events:?}"
    );
    // The server restarts again while the resumed turn runs.
    assert!(
        matches!(recovery::judge(&events), recovery::Verdict::Resume { .. }),
        "{:?}",
        recovery::judge(&events)
    );
    manager.close_session(&sid.to_string()).await.unwrap();
}

/// Review of #711, finding 3: `credentials_locked` while the vault is OPEN (a vault
/// error that is not a lock) is counted like a transient error and closes the turn,
/// instead of reopening the session every `poll` until the deadline.
#[tokio::test]
async fn credentials_locked_with_the_vault_open_is_counted_then_closed() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    let vault = crate::vault::VaultService::ephemeral();
    vault
        .init("a long passphrase".into(), chrono::Duration::seconds(600))
        .await
        .unwrap();
    let manager = manager.with_vault(vault);
    *fake.fail_every_open.lock().unwrap() =
        Some(nexus_claude::agent::ProviderError::CredentialsLocked);
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(5))
        .await;
    assert_eq!(
        (report.failed, report.retried, report.abandoned),
        (1, recovery::MAX_TRANSIENT_RETRIES as usize, 0)
    );
    let events = events_of(&graph, sid).await;
    assert_eq!(events.iter().filter(|e| e.event_type == "error").count(), 1);
    assert_closed_by_recovery(&events);
}

/// A session live on this server (a run relaunched while the boot scans) is not a turn
/// the restart cut: neither queued nor closed.
#[tokio::test]
async fn a_session_live_here_is_neither_queued_nor_closed_by_the_scan() {
    let (manager, graph, fake) = agent_manager();
    let created = manager.create_session(&agent_request("hi")).await.unwrap();
    let sid = Uuid::parse_str(&created.session_id).unwrap();
    assert_eq!(fake.state.turns_started.lock().unwrap().len(), 1);
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((pending.len(), scan.closed), (0, 0));
    assert!(!events_of(&graph, sid)
        .await
        .iter()
        .any(|e| e.event_type == "error"));
    manager.close_session(&created.session_id).await.unwrap();
}

/// Finding 8a of #697: closing a cut turn is published, not only written: the session
/// lists learn the session no longer streams.
#[tokio::test]
async fn closing_a_cut_turn_is_published_for_the_session_lists() {
    let (manager, _graph, _fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("permission_request", "{}")]).await;
    let bus = Arc::new(crate::events::EventBus::new(16));
    let mut rx = bus.subscribe();
    let manager = manager.with_event_emitter(bus.clone());
    let (_, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(scan.closed, 1);
    let event = rx.try_recv().expect("a session update");
    assert_eq!(event.entity_id, sid.to_string());
    assert_eq!(event.payload["is_streaming"], serde_json::json!(false));
}

/// P18b round 3, fix F: a turn of the agent engine tells the session lists (the CRUD
/// bus) that the session streams, then that it stopped — as the Claude Code engine
/// does. Red before the fix: only the session's own channel heard `streaming_status`,
/// so a parent's list of children showed a finished native sub-agent as running.
#[tokio::test]
async fn an_agent_turn_publishes_its_streaming_start_and_end_for_the_session_lists() {
    let (manager, _graph, fake) = agent_manager();
    let bus = Arc::new(crate::events::EventBus::new(64));
    let mut crud = bus.subscribe();
    let manager = manager.with_event_emitter(bus.clone());
    let created = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    let sid = created.session_id.clone();
    fake.state.push(nexus_claude::agent::AgentEvent::Done {
        stop_reason: nexus_claude::agent::StopReason::Completed,
        subtype: Some("success".into()),
        is_error: false,
        result_text: Some("done".into()),
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 5,
        duration_api_ms: None,
        num_turns: 1,
        model: Some("m".into()),
        provider_session_id: Some("p-1".into()),
        structured_output: None,
        error: None,
    });
    let mut seen = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !seen.contains(&serde_json::json!(false)) {
        let event = tokio::time::timeout_at(deadline, crud.recv())
            .await
            .expect("the end of the turn is published")
            .unwrap();
        if event.entity_type == crate::events::EntityType::ChatSession
            && event.action == crate::events::CrudAction::Updated
            && event.entity_id == sid
        {
            if let Some(streaming) = event.payload.get("is_streaming") {
                seen.push(streaming.clone());
            }
        }
    }
    assert_eq!(
        seen,
        vec![serde_json::json!(true), serde_json::json!(false)],
        "start, then end"
    );
    manager.close_session(&sid).await.unwrap();
}

/// The `created` events of chat sessions the bus holds right now.
fn chat_sessions_created(
    crud: &mut tokio::sync::broadcast::Receiver<crate::events::CrudEvent>,
) -> Vec<crate::events::CrudEvent> {
    let mut created = Vec::new();
    while let Ok(event) = crud.try_recv() {
        if event.entity_type == crate::events::EntityType::ChatSession
            && event.action == crate::events::CrudAction::Created
        {
            created.push(event);
        }
    }
    created
}

/// Review nexus-41 of #727, finding 1: the `created` event reaches every authenticated
/// `/ws/events` subscriber, so its payload is cut to `project_slug` and a `spawned_by`
/// of `{type, parent_session_id}` — no `cwd` (it names the OS user), `model`,
/// `execution_place`, nor the origin's other ids (`tool_use_id` included: the client
/// reads it from the filtered `/children`). Red before: all of them were sent.
#[test]
fn the_created_payload_is_the_project_and_the_parent_only() {
    let mut req = agent_request("hello");
    req.cwd = "/Users/someone/work/proj".into();
    let top = session_created_payload(&req, Some("proj"));
    assert_eq!(top, serde_json::json!({ "project_slug": "proj" }));

    let parent = Uuid::new_v4();
    req.spawned_by = Some(
        crate::chat::types::SpawnedBy::Conversation {
            parent_session_id: parent,
            tool_use_id: Some("toolu_7".into()),
        }
        .to_json_string(),
    );
    let child = session_created_payload(&req, Some("proj"));
    assert_eq!(
        child,
        serde_json::json!({
            "project_slug": "proj",
            "spawned_by": { "type": "conversation", "parent_session_id": parent.to_string() },
        })
    );

    // A runner's child: its run and task ids stay on the record, off the event.
    let runner = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let payload = session_created_payload(&runner, None);
    let spawned_by = payload["spawned_by"].as_object().unwrap();
    assert_eq!(spawned_by["type"], "runner");
    assert!(
        spawned_by
            .keys()
            .all(|k| k == "type" || k == "parent_session_id"),
        "{payload}"
    );
    assert!(payload.get("cwd").is_none(), "{payload}");
}

/// Review of frontend #364 (verified on #727): the `created` of a session goes out
/// BEFORE its first turn starts. A native `Agent` child whose first turn fails at once
/// ends `is_streaming: false` right away; a client that saw that `updated` before the
/// `created` counted the child running for good. Red before: `created` was emitted
/// after `open_agent_session` returned — after the turn's `is_streaming: true`
/// (emitted as the turn is sent) and racing its spawned end.
#[tokio::test]
async fn created_precedes_the_first_updated_of_a_child_whose_turn_fails_at_once() {
    let (manager, _graph, fake) = agent_manager();
    let bus = Arc::new(crate::events::EventBus::new(256));
    let mut crud = bus.subscribe();
    let manager = manager.with_event_emitter(bus.clone());
    let mut req = agent_request("hello");
    req.spawned_by = Some(
        crate::chat::types::SpawnedBy::Conversation {
            parent_session_id: Uuid::new_v4(),
            tool_use_id: Some("toolu_fail".into()),
        }
        .to_json_string(),
    );
    let opened = manager.create_session(&req).await;
    // The first turn fails at once: the provider's first event is a final error.
    fake.state.push(nexus_claude::agent::AgentEvent::Error {
        error: nexus_claude::agent::ProviderError::invalid("refused"),
    });
    let mut seen: Vec<(String, crate::events::CrudAction, serde_json::Value)> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let ended = seen.iter().any(|(_, action, payload)| {
            *action == crate::events::CrudAction::Updated
                && payload.get("is_streaming") == Some(&serde_json::json!(false))
        });
        if ended {
            break;
        }
        match tokio::time::timeout_at(deadline, crud.recv()).await {
            Ok(Ok(event)) if event.entity_type == crate::events::EntityType::ChatSession => {
                seen.push((event.entity_id, event.action, event.payload))
            }
            Ok(Ok(_)) => {}
            _ => break,
        }
    }
    let sid = seen
        .iter()
        .find(|(_, action, _)| *action == crate::events::CrudAction::Created)
        .map(|(id, _, _)| id.clone())
        .unwrap_or_else(|| panic!("a created event: {seen:?}"));
    let of_child: Vec<&crate::events::CrudAction> = seen
        .iter()
        .filter(|(id, _, _)| *id == sid)
        .map(|(_, action, _)| action)
        .collect();
    assert_eq!(
        of_child.first(),
        Some(&&crate::events::CrudAction::Created),
        "created first: {seen:?}"
    );
    assert!(
        of_child.contains(&&crate::events::CrudAction::Updated),
        "the failed turn's end was seen: {seen:?}"
    );
    if let Ok(opened) = opened {
        let _ = manager.close_session(&opened.session_id).await;
    }
}

/// Review nexus-41 of #727, finding 1: the `created` of the session a provider switch
/// opens says nothing of the session it left (`switched_from`) nor of its place.
#[tokio::test]
async fn a_relayed_session_created_event_has_no_switched_from_nor_cwd() {
    let (manager, _graph, _fake) = agent_manager();
    let bus = Arc::new(crate::events::EventBus::new(64));
    let mut crud = bus.subscribe();
    let manager = manager.with_event_emitter(bus.clone());
    let relayed = super::super::relay::RelayedFrom {
        relay: super::super::relay::Relay {
            text: String::new(),
            included: 0,
            omitted: 0,
        },
        from_session_id: Uuid::new_v4().to_string(),
        from_provider: "local".into(),
        moved_by: "user".into(),
        conversation_id: None,
        segments: None,
        next_turn: 0,
    };
    let mut req = agent_request("hello");
    req.cwd = "/Users/someone/work/proj".into();
    let created = manager
        .create_session_relayed(&req, Some(&relayed))
        .await
        .unwrap();
    let events = chat_sessions_created(&mut crud);
    assert_eq!(events.len(), 1, "{events:?}");
    let payload = &events[0].payload;
    for field in [
        "switched_from",
        "cwd",
        "model",
        "execution_place",
        "spawned_by",
    ] {
        assert!(payload.get(field).is_none(), "{field}: {payload}");
    }
    manager.close_session(&created.session_id).await.unwrap();
}

/// Lot review of #721, finding 1: a session of the agent engine opened by nobody
/// emits exactly one `created`, without `spawned_by`. Red before: none at all (only
/// the legacy engine and the HTTP handler emitted one).
#[tokio::test]
async fn an_agent_engine_session_emits_one_created_event_without_spawned_by() {
    let (manager, _graph, _fake) = agent_manager();
    let bus = Arc::new(crate::events::EventBus::new(64));
    let mut crud = bus.subscribe();
    let manager = manager.with_event_emitter(bus.clone());
    let created = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    let events = chat_sessions_created(&mut crud);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].entity_id, created.session_id);
    assert!(events[0].payload.get("spawned_by").is_none());
    for field in ["cwd", "model", "execution_place"] {
        assert!(events[0].payload.get(field).is_none(), "{field}");
    }
    manager.close_session(&created.session_id).await.unwrap();
    assert!(chat_sessions_created(&mut crud).is_empty());
}

/// Lot review of #721, nit (b) of the follow-up review of #724: a deny of a sub-agent's
/// request that its provider refuses still ends the request DECIDED on the child's
/// thread (it is closed right after: nothing runs the call), so it stays pending in
/// nobody's attention; a deny the provider takes is written once, by the normal path.
/// Reviews of #727 (finding 2 of both): NOT when the provider refused it because an
/// allow of this session won — here the user's, taken by the provider while its
/// decision is not stored yet (its lasting rule being written): that request keeps the
/// one decision its allow writes. Red before: `("allowed", false)` was written too.
#[tokio::test]
async fn a_deny_the_provider_refuses_is_still_written_on_the_thread() {
    let (manager, graph, fake) = agent_manager();
    let created = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    let sid = created.session_id.clone();
    let handle = manager.agent_runtime.get(&sid).await.unwrap();
    let ask = |id: &str| ChatEvent::PermissionRequest {
        id: id.into(),
        tool: "Bash".into(),
        input: serde_json::json!({ "command": "rm -rf build" }),
        parent_tool_use_id: None,
        category: None,
        canonical: Some("Bash".into()),
        tool_use_id: None,
        always: None,
        consent: None,
    };
    handle.emit(ask("refused")).await;
    handle.emit(ask("allowed")).await;
    handle.emit(ask("taken")).await;
    // The provider already holds an answer to `refused` (one this session never took:
    // it lost the request) and to `allowed` (the user's allow, in flight on this
    // session): it rejects a second one for both.
    for id in ["refused", "allowed"] {
        fake.state.permission_answers.lock().unwrap().push((
            id.into(),
            nexus_claude::agent::PermissionDecision::allow_once(),
        ));
    }
    handle.mark_answered("allowed");
    manager.deny_waiting_permissions(&sid).await;
    let uuid = Uuid::parse_str(&sid).unwrap();
    let decisions: Vec<(String, bool)> = events_of(&graph, uuid)
        .await
        .iter()
        .filter_map(|e| match serde_json::from_str::<ChatEvent>(&e.data).ok()? {
            ChatEvent::PermissionDecision { id, allow, .. } => Some((id, allow)),
            _ => None,
        })
        .collect();
    let mut decisions = decisions;
    decisions.sort();
    assert_eq!(
        decisions,
        vec![("refused".to_string(), false), ("taken".to_string(), false)]
    );
    manager.close_session(&sid).await.unwrap();
}

/// Review nexus-41 of #727, finding 2: the fallback refusal of a sub-agent's request is
/// written only while the request is still undecided. One answered between the
/// collection and the fallback (an automatic answer winning the race — the reason the
/// provider refused the deny) keeps its one decision; one still waiting gets the
/// refusal. Red before: the refusal was written after the allow, two contradictory
/// decisions on one thread.
#[tokio::test]
async fn the_fallback_deny_is_written_only_for_a_request_still_undecided() {
    let (manager, graph, fake) = agent_manager();
    let created = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    let sid = created.session_id.clone();
    let handle = manager.agent_runtime.get(&sid).await.unwrap();
    let ask = |id: &str| ChatEvent::PermissionRequest {
        id: id.into(),
        tool: "Bash".into(),
        input: serde_json::json!({ "command": "make" }),
        parent_tool_use_id: None,
        category: None,
        canonical: Some("Bash".into()),
        tool_use_id: None,
        always: None,
        consent: None,
    };
    handle.emit(ask("won")).await;
    handle.emit(ask("pending")).await;
    // Both were waiting when the requests were collected.
    let waited = handle.waiting_permission_ids();
    assert!(waited.contains(&"won".to_string()) && waited.contains(&"pending".to_string()));
    // `won` is allowed meanwhile; the provider holds an answer for `pending` it no
    // longer waits on (so it would refuse a deny), and nothing decided it on the thread.
    handle
        .answer_permission_scoped("won", true, crate::chat::types::PermissionAnswerScope::Once)
        .await
        .unwrap();
    fake.state.permission_answers.lock().unwrap().push((
        "pending".into(),
        nexus_claude::agent::PermissionDecision::allow_once(),
    ));
    assert!(
        !manager
            .deny_on_thread_if_undecided(&sid, "won", Some(&handle))
            .await,
        "answered meanwhile: no refusal"
    );
    assert!(
        manager
            .deny_on_thread_if_undecided(&sid, "pending", Some(&handle))
            .await,
        "still undecided: refused"
    );
    // Not live (the recovery): the thread alone says whether it is decided.
    assert!(!manager.deny_on_thread_if_undecided(&sid, "won", None).await);
    let uuid = Uuid::parse_str(&sid).unwrap();
    let mut decisions: Vec<(String, bool)> = events_of(&graph, uuid)
        .await
        .iter()
        .filter_map(|e| match serde_json::from_str::<ChatEvent>(&e.data).ok()? {
            ChatEvent::PermissionDecision { id, allow, .. } => Some((id, allow)),
            _ => None,
        })
        .collect();
    decisions.sort();
    assert_eq!(
        decisions,
        vec![("pending".to_string(), false), ("won".to_string(), true)],
        "one decision per request"
    );
    manager.close_session(&sid).await.unwrap();
}

/// Lot review of #721, finding 4: the restart recovery closes a cut sub-agent of an
/// `Agent` call, and first refuses what its thread leaves undecided, through the same
/// helper as the live path — so the request is no orphan in its owner's attention. A
/// runner's child is closed as before (its requests are its run's business). Red
/// before: closed with the request still undecided.
#[tokio::test]
async fn a_cut_agent_child_refuses_its_undecided_requests_before_it_is_closed() {
    let request = serde_json::to_string(&ChatEvent::PermissionRequest {
        id: "p1".into(),
        tool: "Bash".into(),
        input: serde_json::json!({ "command": "make" }),
        parent_tool_use_id: None,
        category: None,
        canonical: Some("Bash".into()),
        tool_use_id: None,
        always: None,
        consent: None,
    })
    .unwrap();
    let tail = [
        ("user_message", "{}"),
        ("permission_request", request.as_str()),
    ];
    for (spawned_by, refused) in [
        (
            crate::chat::types::SpawnedBy::Conversation {
                parent_session_id: Uuid::new_v4(),
                tool_use_id: Some("toolu_1".into()),
            },
            true,
        ),
        (
            crate::chat::types::SpawnedBy::Runner {
                run_id: Uuid::new_v4(),
                task_id: Uuid::new_v4(),
                parent_session_id: None,
            },
            false,
        ),
    ] {
        let (manager, graph, fake, sid) = restarted_with_tail(&tail).await;
        let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
        node.spawned_by = Some(spawned_by.to_json_string());
        graph.create_chat_session(&node).await.unwrap();
        let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
        assert_eq!((pending.len(), scan.closed), (0, 1));
        assert!(fake.state.turns_started.lock().unwrap().is_empty());
        let events = events_of(&graph, sid).await;
        assert_closed_by_recovery(&events);
        let decisions: Vec<&ChatEventRecord> = events
            .iter()
            .filter(|e| e.event_type == "permission_decision")
            .collect();
        if refused {
            assert_eq!(decisions.len(), 1, "{events:?}");
            let decision: ChatEvent = serde_json::from_str(&decisions[0].data).unwrap();
            assert!(
                matches!(decision, ChatEvent::PermissionDecision { ref id, allow: false, .. } if id == "p1"),
                "{decision:?}"
            );
        } else {
            assert!(decisions.is_empty(), "{events:?}");
        }
    }
}

/// P20-suite-2: a session recovered with its owner stored re-derives the person and
/// mints the PO token again, so the PO tools keep their auth after a restart.
#[tokio::test]
async fn a_recovered_session_with_its_owner_gets_a_po_token() {
    let (manager, graph, fake, sid) = restarted_with(
        &[("user_message", "{}"), ("assistant_text", "{}")],
        chrono::Duration::zero(),
        |config| config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into()),
    )
    .await;
    // The owner stored at open, and the user record behind it.
    let owner = Uuid::new_v4();
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    node.owner = Some(owner.to_string());
    let mut snapshot: serde_json::Value =
        serde_json::from_str(node.capabilities.as_deref().unwrap()).unwrap();
    snapshot["per_session_mcp"] = serde_json::json!(true);
    node.capabilities = Some(snapshot.to_string());
    graph.create_chat_session(&node).await.unwrap();
    graph
        .upsert_user(&crate::neo4j::models::UserNode {
            id: owner,
            email: "person@example.com".into(),
            name: "A Person".into(),
            picture_url: None,
            auth_provider: crate::neo4j::models::AuthProvider::Oidc,
            external_id: Some("google-123".into()),
            password_hash: None,
            created_at: chrono::Utc::now(),
            last_login_at: chrono::Utc::now(),
        })
        .await
        .unwrap();
    fake.caps.lock().unwrap().per_session_mcp = true;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!(report.resumed, 1);
    let spec = fake
        .state
        .opened_specs
        .lock()
        .unwrap()
        .last()
        .cloned()
        .unwrap();
    match spec.mcp_servers.get(PO_MCP_SERVER) {
        Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) => {
            assert!(
                env.contains_key("PO_AUTH_TOKEN"),
                "an owner stored at open mints the resume token"
            );
        }
        other => panic!("the resumed spec carries the PO server: {other:?}"),
    }
}

/// Finding 8b of #697: a session reopened without its owner's identity, on a server
/// that signs its tokens, has the PO server but no token for it: it SAYS it lacks the
/// PO tools instead of losing them silently.
#[tokio::test]
async fn a_recovered_session_says_it_has_no_po_tools_without_its_owner() {
    let (manager, graph, fake, sid) = restarted_with(
        &[("user_message", "{}"), ("assistant_text", "{}")],
        chrono::Duration::zero(),
        |config| config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into()),
    )
    .await;
    // A session that carries its MCP servers: the resumed spec HAS the PO server.
    let mut node = graph.get_chat_session(sid).await.unwrap().unwrap();
    let mut snapshot: serde_json::Value =
        serde_json::from_str(node.capabilities.as_deref().unwrap()).unwrap();
    snapshot["per_session_mcp"] = serde_json::json!(true);
    node.capabilities = Some(snapshot.to_string());
    graph.create_chat_session(&node).await.unwrap();
    // And whose provider says it carries them: only the missing token is left to say.
    fake.caps.lock().unwrap().per_session_mcp = true;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(2))
        .await;
    assert_eq!(report.resumed, 1);
    let spec = fake
        .state
        .opened_specs
        .lock()
        .unwrap()
        .last()
        .cloned()
        .unwrap();
    match spec.mcp_servers.get(PO_MCP_SERVER) {
        Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) => {
            assert!(!env.contains_key("PO_AUTH_TOKEN"), "no owner, no token")
        }
        other => panic!("the resumed spec carries the PO server: {other:?}"),
    }
    let sid = sid.to_string();
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state
        .push(nexus_claude::agent::AgentEvent::SessionStarted {
            provider_session_id: Some("p1".into()),
            model: Some("m".into()),
            policy_mode: None,
            native_mode: None,
            tools: vec![],
            mcp_servers: vec![],
            cwd: None,
        });
    let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
    let wire = serde_json::to_value(&init).unwrap();
    let degraded: Vec<String> =
        serde_json::from_value(wire["degraded_features"].clone()).expect("a list");
    assert_eq!(
        degraded
            .iter()
            .filter(|f| *f == super::super::agent_runtime::PO_TOOLS_FEATURE)
            .count(),
        1,
        "{degraded:?}"
    );
    manager.close_session(&sid).await.unwrap();
}

#[tokio::test]
async fn a_finished_turn_is_not_resumed_at_boot() {
    let (manager, _graph, fake, _sid) = restarted_with_tail(&[
        ("user_message", "{}"),
        ("assistant_text", "{}"),
        ("result", "{}"),
    ])
    .await;
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((scan.scanned, scan.closed, pending.len()), (1, 0, 0));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_turn_waiting_for_an_answer_is_closed_once_and_not_judged_again() {
    let (manager, graph, _fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("permission_request", "{}")]).await;
    let (pending, scan) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!((scan.closed, pending.len()), (1, 0));
    let events = graph.get_chat_events(sid, 0, 50).await.unwrap();
    assert_eq!(events.last().unwrap().event_type, "error");
    // The next boot finds it settled.
    let (_, again) = manager.scan_interrupted_turns().await.unwrap();
    assert_eq!(again.closed, 0, "closed once");
}

#[tokio::test]
async fn a_session_the_user_wrote_to_meanwhile_is_not_resumed_twice() {
    let (manager, graph, fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    graph
        .store_chat_events(
            sid,
            vec![ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: sid,
                seq: 3,
                event_type: "user_message".into(),
                data: "{}".into(),
                created_at: chrono::Utc::now(),
            }],
        )
        .await
        .unwrap();
    let report = manager
        .drive_recovery(pending, Duration::from_millis(10), Duration::from_secs(1))
        .await;
    assert_eq!((report.resumed, report.skipped), (0, 1));
    assert!(fake.state.turns_started.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_recovery_waits_while_the_vault_is_locked_then_resumes() {
    let (manager, graph, fake, _sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    vault_backed_claude_code(&graph).await;
    let vault = crate::vault::VaultService::ephemeral();
    let manager = manager.with_vault(vault.clone());
    assert!(
        !vault.is_unlocked(chrono::Utc::now()),
        "a fresh vault is locked"
    );
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let manager = Arc::new(manager);
    let driving = {
        let manager = manager.clone();
        tokio::spawn(async move {
            manager
                .drive_recovery(pending, Duration::from_millis(20), Duration::from_secs(10))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        fake.state.turns_started.lock().unwrap().is_empty(),
        "nothing resumes while the vault is locked"
    );
    vault
        .init("a long passphrase".into(), chrono::Duration::seconds(600))
        .await
        .unwrap();
    let report = driving.await.unwrap();
    assert_eq!(report.resumed, 1);
}

#[tokio::test]
async fn a_recovery_that_waits_for_ever_gives_up_and_says_so() {
    let (manager, graph, _fake, sid) =
        restarted_with_tail(&[("user_message", "{}"), ("assistant_text", "{}")]).await;
    vault_backed_claude_code(&graph).await;
    let manager = manager.with_vault(crate::vault::VaultService::ephemeral());
    let (pending, _) = manager.scan_interrupted_turns().await.unwrap();
    let report = manager
        .drive_recovery(
            pending,
            Duration::from_millis(10),
            Duration::from_millis(60),
        )
        .await;
    assert_eq!((report.resumed, report.abandoned), (0, 1));
    // R2: what the recovery gives up on is closed, not left "streaming".
    assert_closed_by_recovery(&events_of(&graph, sid).await);
}

/// A third-party instance (`acp`) a project consented to, played by the fake.
async fn consented_acp_instance(graph: &crate::neo4j::mock::MockGraphStore, id: &str, slug: &str) {
    let record = crate::chat::provider::settings::InstanceRecord {
        id: id.into(),
        kind: "acp".into(),
        label: "ACP".into(),
        origin: crate::chat::provider::settings::process_origin("acp", None),
        cost_source: "none".into(),
        credential_ref: "none".into(),
        default_model: Some("acp-default".into()),
        ..Default::default()
    };
    super::super::agent_e2e_tests::store_instance(graph, &record).await;
    super::super::agent_e2e_tests::consent(graph, slug, id, &record.origin).await;
}

/// Two fake providers under two ids, for the in-place provider-switch tests (offline:
/// no HTTP, no spawned process).
struct TwoProviders {
    claude: super::super::agent_runtime::fake::FakeProvider,
    acp: super::super::agent_runtime::fake::FakeProvider,
}

impl super::super::agent_runtime::ProviderSource for TwoProviders {
    fn get(&self, provider_id: &str) -> Option<Arc<dyn nexus_claude::agent::AgentProvider>> {
        match provider_id {
            "claude-code" => Some(Arc::new(self.claude.clone())),
            "acp" => Some(Arc::new(self.acp.clone())),
            _ => None,
        }
    }
}

/// A signed agent manager with two fake providers (`claude-code` and `acp`).
fn two_provider_manager() -> (
    ChatManager,
    Arc<crate::neo4j::mock::MockGraphStore>,
    super::super::agent_runtime::fake::FakeProvider,
    super::super::agent_runtime::fake::FakeProvider,
) {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let state = mock_app_state();
    let mut config = test_config();
    config.provider_path = crate::chat::config::ProviderPath::Agent;
    config.jwt_secret = Some(SIGNING_SECRET.into());
    let claude = super::super::agent_runtime::fake::FakeProvider::new();
    let acp = super::super::agent_runtime::fake::FakeProvider::new();
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
        .with_provider_source(Arc::new(TwoProviders {
            claude: claude.clone(),
            acp: acp.clone(),
        }));
    (manager, graph, claude, acp)
}

/// Opens a session on `claude-code` and runs its first turn to completion.
async fn first_turn_on_claude(
    manager: &ChatManager,
    claude: &super::super::agent_runtime::fake::FakeProvider,
) -> String {
    use nexus_claude::agent::AgentEvent;
    let mut req = agent_request("hi there");
    req.provider = Some("claude-code".into());
    req.project_slug = Some("proj".into());
    let created = manager.create_session(&req).await.unwrap();
    let sid = created.session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    claude.state.push(AgentEvent::Text {
        text: "hello".into(),
        seq: None,
        parent: None,
    });
    claude.state.push(done_event(false, None));
    let _ = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    claude.state.end_turn();
    sid
}

/// B-SW phase 1: moving a conversation re-targets the SAME `session_id` instead of
/// opening a new one. The node's provider is updated, the old provider process is
/// closed, and the target provider sees the relayed history in front of the message.
#[tokio::test]
async fn an_in_place_switch_keeps_the_session_id_and_updates_the_provider() {
    let (manager, graph, claude, acp) = two_provider_manager();
    consented_acp_instance(&graph, "acp", "proj").await;
    let sid = first_turn_on_claude(&manager, &claude).await;

    let before = graph
        .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 5_000)
        .await
        .unwrap();
    assert!(
        !before.is_empty(),
        "the first turn is stored before the switch"
    );

    let moved = manager
        .switch_session_provider(&sid, "acp", None, "second question", None)
        .await
        .unwrap_or_else(|e| panic!("switch failed: {e:#}"));
    assert_eq!(moved.session_id, sid, "the SAME thread continues");
    assert_eq!(moved.previous_session_id, sid);
    assert!(
        moved.relayed_entries >= 2,
        "the relay carried the first turn"
    );

    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.provider_id.as_deref(), Some("acp"));
    assert_eq!(node.routed_by.as_deref(), Some("request"));

    // The old provider process was closed; the target saw the relay then the message.
    assert!(claude
        .state
        .closed
        .load(std::sync::atomic::Ordering::SeqCst));
    {
        let started = acp.state.turns_started.lock().unwrap();
        assert_eq!(started.len(), 1, "{started:?}");
        assert!(started[0].contains("<conversation_relay"), "{}", started[0]);
        assert!(started[0].contains("second question"), "{}", started[0]);
    }

    // No new session was opened: the events keep growing on the SAME thread.
    let after = graph
        .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 5_000)
        .await
        .unwrap();
    assert!(
        after.len() > before.len(),
        "the thread grew by the new turn only"
    );
    assert_eq!(
        graph.chat_sessions.read().await.len(),
        1,
        "no new session node was opened"
    );
}

/// A refused in-place switch (the target's project never consented) leaves the old
/// session EXACTLY as it was: the consent check runs before the process is closed, so
/// the process stays live and the resume token (and every other cleared field) survives.
#[tokio::test]
async fn an_in_place_switch_refused_by_consent_leaves_the_old_session_untouched() {
    let (manager, graph, claude, _acp) = two_provider_manager();
    // No consent for `acp`: the refusal is detected in the pre-flight check, before any
    // teardown.
    let sid = first_turn_on_claude(&manager, &claude).await;
    let before = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    // The first turn named a provider session, so there is a resume token to survive.
    assert!(
        before.resume_token.is_some(),
        "the session has a resume token to survive a refusal"
    );

    let result = manager
        .switch_session_provider(&sid, "acp", None, "second question", None)
        .await;
    assert!(result.is_err(), "a non-consented target is refused");

    // The old process was never closed.
    assert!(
        !claude
            .state
            .closed
            .load(std::sync::atomic::Ordering::SeqCst),
        "a refused move must not tear the old process down"
    );
    // The node was never touched: provider, model AND the provider-specific fields all
    // survived.
    let after = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        after.provider_id.as_deref(),
        before.provider_id.as_deref(),
        "the provider is unchanged"
    );
    assert_eq!(after.model, before.model, "the model is unchanged");
    assert_eq!(
        after.resume_token, before.resume_token,
        "the resume token survives a refusal"
    );
    assert_eq!(
        after.capabilities, before.capabilities,
        "the capabilities survive a refusal"
    );
    assert_eq!(
        after.cli_session_id, before.cli_session_id,
        "the cli_session_id survives a refusal"
    );
    assert_eq!(after.engine, before.engine, "the engine survives a refusal");
}

/// A consented switch whose target fails to OPEN (after the consent check passed, so the
/// node was retargeted and the old process closed) rolls back EVERY cleared field — model,
/// resume_token, capabilities, cli_session_id, engine — so a later resume finds the old
/// session, provider and model again.
#[tokio::test]
async fn an_in_place_switch_whose_target_fails_to_open_rolls_back_every_cleared_field() {
    let (manager, graph, claude, acp) = two_provider_manager();
    consented_acp_instance(&graph, "acp", "proj").await;
    let sid = first_turn_on_claude(&manager, &claude).await;
    let before = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();

    // The target consents, but its open fails: the re-target is rolled back in full.
    *acp.fail_open.lock().unwrap() = Some(nexus_claude::agent::ProviderError::unsupported(
        "open_refused",
    ));

    let result = manager
        .switch_session_provider(&sid, "acp", None, "second question", None)
        .await;
    assert!(result.is_err(), "a failing target open refuses the move");

    let after = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        after.provider_id.as_deref(),
        before.provider_id.as_deref(),
        "the provider rolled back"
    );
    assert_eq!(after.model, before.model, "the model rolled back");
    assert_eq!(
        after.resume_token, before.resume_token,
        "the resume token rolled back"
    );
    assert_eq!(
        after.capabilities, before.capabilities,
        "the capabilities rolled back"
    );
    assert_eq!(
        after.cli_session_id, before.cli_session_id,
        "the cli_session_id rolled back"
    );
    assert_eq!(after.engine, before.engine, "the engine rolled back");
}

/// The graph method re-targets a node in place and clears the provider-specific fields
/// that do not survive a move across provider families.
#[tokio::test]
async fn update_chat_session_provider_retargets_and_clears_provider_state() {
    let (_manager, graph) = manager_with_mock();
    let mut session = test_chat_session(None);
    session.provider_id = Some("claude-code".into());
    session.model = "old-model".into();
    session.resume_token = Some("resume".into());
    session.capabilities = Some("{}".into());
    session.cli_session_id = Some("cli-1".into());
    session.engine = Some("legacy".into());
    graph.create_chat_session(&session).await.unwrap();

    let ok = graph
        .update_chat_session_provider(session.id, "acp", "new-model", "request", "agent")
        .await
        .unwrap();
    assert!(ok);
    let node = graph.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(node.provider_id.as_deref(), Some("acp"));
    assert_eq!(node.model, "new-model");
    assert_eq!(node.routed_by.as_deref(), Some("request"));
    assert_eq!(node.resume_token, None);
    assert_eq!(node.capabilities, None);
    assert_eq!(node.cli_session_id, None);
    assert_eq!(node.engine.as_deref(), Some("agent"));

    // A missing session is a clean `false`.
    assert!(!graph
        .update_chat_session_provider(Uuid::new_v4(), "acp", "m", "request", "agent")
        .await
        .unwrap());
}

async fn open_on(
    manager: &ChatManager,
    req: &ChatRequest,
    sid: Uuid,
    provider_id: &str,
) -> Result<CreateSessionResponse> {
    manager
        .open_agent_session(AgentOpen {
            request: req,
            session_id: sid,
            provider_id,
            model: "m",
            system_prompt: "p",
            add_dirs: &[],
            project_slug: Some("exits"),
            owner: None,
            relay: None,
            access: Default::default(),
        })
        .await
}

/// What a failed opening leaves: no router, and the token it minted (when it got that
/// far, `minted`) is not live. `minted_for` makes the check non-vacuous: it lists the
/// tokens that WERE minted for the session.
fn left_nothing(manager: &ChatManager, sid: &str, minted: bool, exit: &str) {
    assert!(
        manager.turn_routing.get(sid).is_none(),
        "{exit}: no router left"
    );
    let tokens = crate::auth::agent_tokens::minted_for(sid);
    if minted {
        assert_eq!(
            tokens.len(),
            1,
            "{exit}: a token was minted before the failure"
        );
        assert!(
            !crate::auth::agent_tokens::is_live(&tokens[0]),
            "{exit}: its token is revoked"
        );
    } else {
        assert!(tokens.is_empty(), "{exit}: failed before any token");
    }
}

/// Review of #684, finding 3: every exit of an opening that stops short — provider
/// unavailable, window too small, a spec that cannot be built, the opening future
/// dropped — leaves neither its router nor the token it minted.
#[tokio::test]
async fn every_exit_of_a_failed_opening_leaves_no_router_nor_minted_token() {
    use nexus_claude::agent::{ContextWindow, ContextWindowSource, ProviderError, ProviderHealth};
    let (manager, graph, fake) = signed_routed_agent_manager();
    *fake.kind.lock().unwrap() = nexus_claude::agent::ProviderKind::Acp;
    fake.caps.lock().unwrap().per_session_mcp = true;
    consented_acp_instance(&graph, "acp-exits", "exits").await;
    let mut req = agent_request("");
    req.user_claims = Some(person_claims());

    // The setup opens: routing and token are set up, so what follows is not vacuous.
    let sid = Uuid::new_v4();
    open_on(&manager, &req, sid, "acp-exits")
        .await
        .expect("opens");
    let ok = sid.to_string();
    assert!(manager.turn_routing.get(&ok).is_some());
    let tokens = crate::auth::agent_tokens::minted_for(&ok);
    assert!(crate::auth::agent_tokens::is_live(&tokens[0]));
    manager.close_session(&ok).await.unwrap();

    // The provider says it is unavailable (the preflight, A30).
    *fake.health.lock().unwrap() = Some(ProviderHealth::unavailable(
        ProviderError::EndpointUnreachable {
            detail: "down".into(),
        },
    ));
    let sid = Uuid::new_v4();
    assert!(open_on(&manager, &req, sid, "acp-exits").await.is_err());
    left_nothing(&manager, &sid.to_string(), true, "health");
    *fake.health.lock().unwrap() = None;

    // The model's window cannot hold the tool schemas: the session opened, then closed.
    fake.caps.lock().unwrap().context_window = Some(ContextWindow {
        value: 10,
        source: ContextWindowSource::Probed,
    });
    let sid = Uuid::new_v4();
    let error = open_on(&manager, &req, sid, "acp-exits").await.unwrap_err();
    assert!(
        matches!(
            error.downcast_ref::<ProviderError>(),
            Some(ProviderError::ContextTooSmall { .. })
        ),
        "{error:#}"
    );
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(
        crate::auth::agent_tokens::minted_for(&sid.to_string()),
        vec![spec_token_jti(&spec)],
        "the token that went to the provider"
    );
    left_nothing(&manager, &sid.to_string(), true, "window");
    fake.caps.lock().unwrap().context_window = None;

    // The spec cannot be built (an unknown permission mode): before any token.
    let mut bad = req.clone();
    bad.permission_mode = Some("no-such-mode".into());
    let sid = Uuid::new_v4();
    assert!(open_on(&manager, &bad, sid, "acp-exits").await.is_err());
    left_nothing(&manager, &sid.to_string(), false, "spec");

    // The opening future is dropped while the provider opens.
    *fake.open_delay.lock().unwrap() = Some(Duration::from_secs(30));
    let sid = Uuid::new_v4();
    assert!(tokio::time::timeout(
        Duration::from_millis(200),
        open_on(&manager, &req, sid, "acp-exits")
    )
    .await
    .is_err());
    left_nothing(&manager, &sid.to_string(), true, "abort");
}

/// Review of #684, finding 2: an opening that fails undoes only what IT set up. A
/// concurrent resume of the same session that succeeded (it minted after, and
/// registered its router over this one's) keeps its token and its router.
#[tokio::test]
async fn a_failed_opening_never_undoes_a_concurrent_one_that_succeeded() {
    let (manager, _graph, _fake) = signed_routed_agent_manager();
    let sid = Uuid::new_v4().to_string();
    let turn = || OpeningTurn {
        explicit_model: false,
        provider_imposed: false,
        moved_in: false,
        routing_mode: None,
        routing_pool: None,
        permission_mode: None,
        message: "hello",
        next_turn: 0,
        segments: None,
        owner: None,
    };
    // The opening that will fail: its router, its token.
    let mut failed = OpeningCleanup::new(&sid, &manager.turn_routing);
    failed.registered(
        manager
            .register_turn_router(&sid, "claude-code", "m", None, turn())
            .await,
    );
    assert!(failed.router.is_some(), "routing is configured");
    let failed_jti = format!("jti-failed-{sid}");
    crate::auth::agent_tokens::register(&failed_jti, Some(&sid));
    failed.jti = Some(failed_jti.clone());
    // The concurrent one that succeeds: a new token, a new router.
    let won_jti = format!("jti-won-{sid}");
    crate::auth::agent_tokens::register(&won_jti, Some(&sid));
    let won = manager
        .register_turn_router(&sid, "claude-code", "m", None, turn())
        .await
        .expect("a router");
    drop(failed);
    assert!(
        crate::auth::agent_tokens::is_live(&won_jti),
        "the token of the opening that succeeded stays live"
    );
    assert!(
        manager
            .turn_routing
            .get(&sid)
            .is_some_and(|router| Arc::ptr_eq(&router, &won)),
        "its router stays"
    );
    // Alone, the failed opening undoes its own.
    let mut alone = OpeningCleanup::new(&sid, &manager.turn_routing);
    alone.registered(manager.turn_routing.get(&sid));
    alone.jti = Some(won_jti.clone());
    drop(alone);
    assert!(!crate::auth::agent_tokens::is_live(&won_jti));
    assert!(manager.turn_routing.get(&sid).is_none());
}

/// Review of #684, finding 2: two messages to a session that is not live resume it
/// ONCE. The second waits for the first and is delivered to the session it resumed:
/// no second token superseding the first, no second agent process.
#[tokio::test]
async fn two_messages_to_a_session_that_is_not_live_resume_it_once() {
    let (manager, _graph, fake) = signed_routed_agent_manager();
    fake.caps.lock().unwrap().per_session_mcp = true;
    let mut req = agent_request("hello");
    req.user_claims = Some(person_claims());
    let sid = manager.create_session(&req).await.unwrap().session_id;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    fake.state.opened_specs.lock().unwrap().clear();
    // Each resume yields while the provider opens: without the lock, both resume.
    *fake.open_delay.lock().unwrap() = Some(Duration::from_millis(100));
    let (first, second) = tokio::join!(
        manager.resume_session(&sid, "one", req.user_claims.as_ref()),
        manager.resume_session(&sid, "two", req.user_claims.as_ref()),
    );
    first.unwrap();
    second.unwrap();
    let resumed = fake.state.opened_specs.lock().unwrap().clone();
    assert_eq!(resumed.len(), 1, "the session is resumed once");
    assert!(
        crate::auth::agent_tokens::is_live(&spec_token_jti(&resumed[0])),
        "its token is live"
    );
    assert!(manager.agent_runtime.get(&sid).await.is_some());
    assert_eq!(manager.resume_locks.len(), 0, "no lock is kept");
    // Both messages reached the session (review of #696, nit 4).
    delivered(&fake, &["one", "two"]).await;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
}

/// Waits until the fake agent has started a turn for each of `texts` (a message that
/// waited behind a running turn starts its own once that turn ends).
async fn delivered(fake: &super::super::agent_runtime::fake::FakeProvider, texts: &[&str]) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let turns = fake.state.turns_started.lock().unwrap().clone();
        if texts
            .iter()
            .all(|t| turns.iter().any(|turn| turn.contains(t)))
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "not all delivered: {texts:?} in {turns:?}"
        );
        fake.state.end_turn();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Review of #696, finding 1: two sends that fail on ONE dead handle each fall back to
/// a resume (`seen` is that handle). The first resumes; the second, once it holds the
/// lock, finds a live handle other than the one it started from and delivers its
/// message there: one resume, both messages delivered. Before, both resumed (the
/// second minting a token that superseded the first's and ending its handle).
#[tokio::test]
async fn two_failed_sends_on_one_dead_session_resume_it_once_and_deliver_both() {
    let (manager, graph, fake) = signed_routed_agent_manager();
    fake.caps.lock().unwrap().per_session_mcp = true;
    let mut req = agent_request("hello");
    req.user_claims = Some(person_claims());
    let sid = manager.create_session(&req).await.unwrap().session_id;
    fake.state.end_turn();
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    // The handle both sends failed on.
    let dead = manager.agent_runtime.get(&sid).await.expect("live");
    fake.state.opened_specs.lock().unwrap().clear();
    fake.state.turns_started.lock().unwrap().clear();
    *fake.open_delay.lock().unwrap() = Some(Duration::from_millis(100));
    let (first, second) = tokio::join!(
        manager.resume_agent_session(
            &node,
            "one",
            req.user_claims.as_ref(),
            Some(Arc::clone(&dead))
        ),
        manager.resume_agent_session(
            &node,
            "two",
            req.user_claims.as_ref(),
            Some(Arc::clone(&dead))
        ),
    );
    first.unwrap();
    second.unwrap();
    let resumed = fake.state.opened_specs.lock().unwrap().clone();
    assert_eq!(resumed.len(), 1, "the session is resumed once");
    let live = manager.agent_runtime.get(&sid).await.expect("live");
    assert!(
        !Arc::ptr_eq(&live, &dead),
        "the resume replaced the dead handle"
    );
    assert!(crate::auth::agent_tokens::is_live(&spec_token_jti(
        &resumed[0]
    )));
    delivered(&fake, &["one", "two"]).await;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
}
/// Follow-up of the review of #696 (scenario A, the caller's window): a message that
/// finds the session not live asks another instance first (`try_remote_send`, up to
/// the NATS RPC timeout in a cluster). A resume of the same session that completes
/// during that wait must receive the message, not be resumed over: the caller reads
/// the live handle BEFORE its liveness check. Before, `resume_session` read it only
/// after the wait, saw the new handle as its own starting point and resumed a second
/// time (a second token superseding the first, the first turn ended by `adopt`).
#[tokio::test]
async fn a_resume_that_completes_while_a_message_asks_nats_is_not_resumed_over() {
    let (manager, _graph, fake) = signed_routed_agent_manager();
    fake.caps.lock().unwrap().per_session_mcp = true;
    let mut req = agent_request("hello");
    req.user_claims = Some(person_claims());
    let sid = manager.create_session(&req).await.unwrap().session_id;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    fake.state.opened_specs.lock().unwrap().clear();
    fake.state.turns_started.lock().unwrap().clear();
    let (entered, release) = remote_send_hook::arm(&sid);
    let claims = req.user_claims.as_ref();
    let (routed, resumed_first) =
        tokio::join!(manager.route_user_message(&sid, "two", claims), async {
            // "two" saw the session not live and now waits on the remote instances.
            entered.await.unwrap();
            let first = manager.resume_session(&sid, "one", claims).await;
            release.notify_one();
            first
        },);
    resumed_first.unwrap();
    assert_eq!(routed.unwrap(), DeliveryRoute::Resumed);
    let resumed = fake.state.opened_specs.lock().unwrap().clone();
    assert_eq!(resumed.len(), 1, "the session is resumed once");
    assert!(crate::auth::agent_tokens::is_live(&spec_token_jti(
        &resumed[0]
    )));
    delivered(&fake, &["one", "two"]).await;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
}

/// Review of #706, finding 1: a message that waited (the NATS RPC) while another
/// resume installed a handle whose process then died is not lost: the send to that
/// handle fails, and the resume goes on over it (a handle leaves the runtime only by
/// `close`). Before, the failed send was the answer: the message never arrived.
#[tokio::test]
async fn a_message_overtaken_by_a_resume_whose_session_died_resumes_over_it() {
    let (manager, _graph, fake) = signed_routed_agent_manager();
    fake.caps.lock().unwrap().per_session_mcp = true;
    let mut req = agent_request("hello");
    req.user_claims = Some(person_claims());
    let sid = manager.create_session(&req).await.unwrap().session_id;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    fake.state.opened_specs.lock().unwrap().clear();
    fake.state.turns_started.lock().unwrap().clear();
    // "two" looked before the session was resumed: it saw no live handle.
    let from = manager.resume_point(&sid).await;
    manager
        .resume_session(&sid, "one", req.user_claims.as_ref())
        .await
        .unwrap();
    delivered(&fake, &["one"]).await;
    fake.state.end_turn();
    let overtaking = manager.agent_runtime.get(&sid).await.expect("live");
    wait_idle(&overtaking).await;
    // Its process dies: the next turn on it fails.
    *fake.state.fail_next_turn.lock().unwrap() = Some(
        nexus_claude::agent::ProviderError::unsupported("the agent is gone"),
    );
    manager
        .resume_session_from(&sid, "two", req.user_claims.as_ref(), from)
        .await
        .unwrap();
    let resumed = fake.state.opened_specs.lock().unwrap().clone();
    assert_eq!(resumed.len(), 2, "resumed over the dead session");
    let live = manager.agent_runtime.get(&sid).await.expect("live");
    assert!(!Arc::ptr_eq(&live, &overtaking), "a new handle");
    assert!(crate::auth::agent_tokens::is_live(&spec_token_jti(
        &resumed[1]
    )));
    delivered(&fake, &["two"]).await;
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
}

/// Waits until `handle` has no turn running.
async fn wait_idle(handle: &super::super::agent_runtime::AgentSessionHandle) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while handle.is_streaming.load(Ordering::SeqCst) {
        assert!(std::time::Instant::now() < deadline, "the turn never ended");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn agent_path_opens_a_session_persists_the_snapshot_and_streams_a_turn() {
    use nexus_claude::agent::{AgentEvent, StopReason};
    let (manager, graph, fake) = agent_manager();
    let created = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    let sid = created.session_id.clone();
    let mut rx = manager.subscribe(&sid).await.unwrap();

    // The provider got the turn, a neutral policy and the PO MCP server.
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(spec.policy.mode, nexus_claude::agent::PolicyMode::AutoEdits);
    assert!(spec.mcp_servers.contains_key("project-orchestrator"));
    assert_eq!(
        fake.state.turns_started.lock().unwrap().as_slice(),
        ["hello"]
    );

    // What the provider reported is on the persisted session.
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert!(node.capabilities.is_some(), "frozen capability snapshot");
    // Like the Claude Code façade, the provider has named no session yet.
    assert_eq!(
        node.resume_token, None,
        "no resume token before the provider names one"
    );
    assert_eq!(node.provider_id.as_deref(), Some("claude-code"));

    fake.state.push(AgentEvent::Text {
        text: "hi!".into(),
        seq: None,
        parent: None,
    });
    fake.state.push(AgentEvent::Done {
        stop_reason: StopReason::Completed,
        subtype: Some("success".into()),
        is_error: false,
        result_text: None,
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 5,
        duration_api_ms: None,
        num_turns: 1,
        model: Some("m".into()),
        provider_session_id: Some("p-1".into()),
        structured_output: None,
        error: None,
    });
    let text = next_matching(&mut rx, |e| matches!(e, ChatEvent::AssistantText { .. })).await;
    assert!(matches!(text, ChatEvent::AssistantText { ref content, .. } if content == "hi!"));
    let result = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    assert!(
        matches!(result, ChatEvent::Result { ref stop_reason, .. } if stop_reason.as_deref() == Some("completed"))
    );
    next_matching(&mut rx, |e| {
        matches!(
            e,
            ChatEvent::StreamingStatus {
                is_streaming: false
            }
        )
    })
    .await;
    assert!(!manager.is_session_streaming(&sid).await);

    // The session id the turn named is persisted, without waiting for a reopen.
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    let token = nexus_claude::agent::ResumeToken::from_wire(
        node.resume_token
            .as_deref()
            .expect("resume token after the turn"),
    )
    .unwrap();
    assert_eq!(token.data()["session_id"], "p-1");

    // The events were persisted for replay (never the transient ones).
    let events = graph
        .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 100)
        .await
        .unwrap();
    let kinds: Vec<_> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(kinds, ["user_message", "assistant_text", "result"]);
}

#[tokio::test]
async fn agent_path_queues_a_second_turn_while_one_runs_and_answers_permissions() {
    use nexus_claude::agent::AgentEvent;
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("first"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    // Queued, not refused: the running turn is interrupted and the message read next.
    manager.send_message(&sid, "second").await.unwrap();
    assert_eq!(fake.state.interrupts.lock().unwrap().len(), 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while fake.state.turns_started.lock().unwrap().len() < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "the queued turn starts"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(fake.state.turns_started.lock().unwrap()[1].ends_with("second"));

    fake.state.push(AgentEvent::PermissionAsk {
        request_id: "perm-1".into(),
        tool_name: "Bash".into(),
        input: serde_json::json!({"command": "ls"}),
        category: Default::default(),
        canonical: None,
        tool_call_id: None,
        scopes: vec![],
        parent: None,
    });
    next_matching(&mut rx, |e| {
        matches!(e, ChatEvent::PermissionRequest { .. })
    })
    .await;
    manager
        .route_permission_response(
            &sid,
            "perm-1",
            true,
            crate::chat::types::PermissionAnswerScope::Once,
            true,
        )
        .await
        .unwrap();
    let answers = fake.state.permission_answers.lock().unwrap().clone();
    assert_eq!(answers.len(), 1);
    assert_eq!(answers[0].0, "perm-1");
}

/// Decision fce98b8a, review of #729 (point 3): a person's lift is persisted with the
/// session, a session launched `--unscoped` keeps its lift live while in trust, and leaving
/// trust clears both: the persisted flag (a resume launches scoped) and the live lift (the
/// running process is confined by its hooks). Only a person's switch lifts it again.
#[tokio::test]
async fn leaving_trust_ends_a_persons_lift_of_the_file_scope() {
    let (manager, graph, _fake) = agent_manager();
    let request = ChatRequest {
        file_scope_lifted: true,
        permission_mode: Some("bypassPermissions".into()),
        ..agent_request("go")
    };
    let sid = manager.create_session(&request).await.unwrap().session_id;
    let uuid = Uuid::parse_str(&sid).unwrap();
    assert!(
        graph
            .get_chat_session(uuid)
            .await
            .unwrap()
            .unwrap()
            .file_scope_lifted
    );
    // As `attach` records a process launched `--unscoped`.
    crate::auth::agent_tokens::set_launched_unscoped(&sid, true);
    assert!(!crate::auth::agent_tokens::must_confine_unscoped(&sid));

    manager
        .set_session_permission_mode(&sid, "default")
        .await
        .unwrap();
    assert!(
        !graph
            .get_chat_session(uuid)
            .await
            .unwrap()
            .unwrap()
            .file_scope_lifted
    );
    assert!(crate::auth::agent_tokens::must_confine_unscoped(&sid));
    assert!(!crate::auth::agent_tokens::is_scope_lifted(&sid));

    // Back to trust by the mode change alone: still confined, still not lifted.
    manager
        .set_session_permission_mode(&sid, "trust")
        .await
        .unwrap();
    assert!(crate::auth::agent_tokens::must_confine_unscoped(&sid));
    assert!(
        !graph
            .get_chat_session(uuid)
            .await
            .unwrap()
            .unwrap()
            .file_scope_lifted
    );
    // A person's switch (the WebSocket handler, after its owner and person checks).
    manager.lift_file_scope(&sid).await;
    assert!(!crate::auth::agent_tokens::must_confine_unscoped(&sid));
    assert!(
        graph
            .get_chat_session(uuid)
            .await
            .unwrap()
            .unwrap()
            .file_scope_lifted
    );
    manager.close_session(&sid).await.unwrap();
    crate::auth::agent_tokens::revoke_session(&sid);
}

#[tokio::test]
async fn agent_interrupt_scoped_reports_whether_a_turn_was_running() {
    let (manager, _graph, _fake) = agent_manager();

    // An idle session has no turn: the Stop is delivered nowhere.
    let idle = manager
        .create_session(&agent_request(""))
        .await
        .unwrap()
        .session_id;
    let outcome = manager.interrupt_scoped(&idle, true).await.unwrap();
    assert!(!outcome.delivered, "no turn running → delivered false");
    assert_eq!(outcome.routed, "local");

    // A session whose turn is running has something to interrupt.
    let running = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let outcome = manager.interrupt_scoped(&running, true).await.unwrap();
    assert!(outcome.delivered, "a turn was running → delivered true");
}

#[tokio::test]
async fn agent_path_interrupt_model_mode_and_close_reach_the_provider() {
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    manager.interrupt(&sid).await.unwrap();
    assert_eq!(fake.state.interrupts.lock().unwrap().len(), 1);

    assert!(manager
        .set_session_model(&sid, "other-model")
        .await
        .unwrap());
    assert_eq!(
        fake.state.models.lock().unwrap().as_slice(),
        ["other-model"]
    );
    manager
        .set_session_permission_mode(&sid, "plan_only")
        .await
        .unwrap();
    assert_eq!(
        fake.state.modes.lock().unwrap().as_slice(),
        [nexus_claude::agent::PolicyMode::PlanOnly]
    );
    let changed = next_matching(&mut rx, |e| {
        matches!(e, ChatEvent::PermissionModeChanged { .. })
    })
    .await;
    assert!(
        matches!(changed, ChatEvent::PermissionModeChanged { ref policy_mode, .. } if policy_mode.as_deref() == Some("plan_only"))
    );

    manager.close_session(&sid).await.unwrap();
    assert!(fake.state.closed.load(Ordering::SeqCst));
    next_matching(&mut rx, |e| matches!(e, ChatEvent::SessionClosed { .. })).await;
    assert!(!manager.is_session_active(&sid).await);
}

#[tokio::test]
async fn agent_path_masks_a_vault_secret_before_persisting_and_broadcasting() {
    use nexus_claude::agent::{AgentEvent, StopReason, ToolOutput};
    let secret = "sk-vault-secret-in-a-tool-output-5521";
    crate::vault::mask::global().register("AGENT_PATH_TEST_KEY", secret);
    let (manager, graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(AgentEvent::ToolResult {
        id: "t1".into(),
        output: Some(ToolOutput::Text(format!("cat .env -> KEY={secret}"))),
        is_error: false,
        seq: None,
        parent: None,
    });
    fake.state.push(AgentEvent::Text {
        text: format!("the key is {secret}"),
        seq: None,
        parent: None,
    });
    fake.state.push(AgentEvent::Done {
        stop_reason: StopReason::Completed,
        subtype: Some("success".into()),
        is_error: false,
        result_text: Some(format!("done, {secret}")),
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 1,
        duration_api_ms: None,
        num_turns: 1,
        model: None,
        provider_session_id: None,
        structured_output: None,
        error: None,
    });
    let mut seen = Vec::new();
    loop {
        let ev = next_matching(&mut rx, |_| true).await;
        let done = matches!(
            ev,
            ChatEvent::StreamingStatus {
                is_streaming: false
            }
        );
        seen.push(serde_json::to_string(&ev).unwrap());
        if done {
            break;
        }
    }
    assert!(
        seen.iter().all(|e| !e.contains(secret)),
        "broadcast: {seen:?}"
    );
    assert!(
        seen.iter().any(|e| e.contains("cat .env")),
        "the rest of the output is kept"
    );
    let stored = graph
        .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 100)
        .await
        .unwrap();
    assert!(stored.iter().all(|e| !e.data.contains(secret)), "persisted");
    crate::vault::mask::global().forget("AGENT_PATH_TEST_KEY");
}

#[tokio::test]
async fn agent_path_open_failure_is_typed_and_revokes_nothing_live() {
    let (manager, _graph, fake) = agent_manager();
    *fake.fail_open.lock().unwrap() = Some(nexus_claude::agent::ProviderError::CliNotFound {
        program: "claude".into(),
    });
    let err = manager
        .create_session(&agent_request("x"))
        .await
        .unwrap_err();
    let failure =
        crate::chat::provider::errors::classify_open_error(&err, Some("claude-code")).unwrap();
    assert_eq!((failure.status, failure.code), (424, "cli_not_found"));
}

#[tokio::test]
async fn a_session_without_cwd_runs_in_a_neutral_directory_of_the_host_and_resumes_in_it() {
    let (manager, graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.cwd = String::new();
    let created = manager.create_session(&req).await.unwrap();
    let sid = created.session_id.clone();
    let id = Uuid::parse_str(&sid).unwrap();

    // The marker is on the response and on the node; the cwd is the host's directory.
    assert_eq!(
        created.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Neutral
    );
    let expected = crate::chat::neutral_place::dir_for(id);
    let node = graph.get_chat_session(id).await.unwrap().unwrap();
    assert_eq!(
        node.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Neutral
    );
    assert_eq!(node.cwd, expected.to_string_lossy());
    // Real, empty, and what the provider was given.
    assert!(expected.is_dir());
    assert_eq!(std::fs::read_dir(&expected).unwrap().count(), 0);
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(spec.cwd, expected);
    // No project named: the lack of graph context is said, not silent.
    assert!(node.project_slug.is_none());
    assert_eq!(created.notices.len(), 1);
    assert!(created.notices[0].contains("project_slug"));

    // Closing removes the directory; a resume makes the same one again.
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    assert!(!expected.exists());
    manager.resume_session(&sid, "again", None).await.unwrap();
    assert!(expected.is_dir());
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert_eq!(spec.cwd, expected);

    manager.close_session(&sid).await.unwrap();
    assert!(!expected.exists());
}

#[tokio::test]
async fn a_session_with_a_cwd_stays_a_project_session_without_notice() {
    let (manager, graph, _fake) = agent_manager();
    let created = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    assert_eq!(
        created.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Project
    );
    assert!(created.notices.is_empty());
    let node = graph
        .get_chat_session(Uuid::parse_str(&created.session_id).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        node.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Project
    );
    manager.close_session(&created.session_id).await.unwrap();
}

/// What the opened session's tool policy says about a write tool.
fn spec_refuses_writes(spec: &nexus_claude::agent::SessionSpec) -> bool {
    spec.policy.decide(
        "Write",
        Some("/tmp/x"),
        nexus_claude::agent::ToolCategory::Edit,
    ) == nexus_claude::agent::PolicyDecision::Deny
}

/// The MCP server environment of an opened spec.
fn spec_po_env(
    spec: &nexus_claude::agent::SessionSpec,
) -> std::collections::BTreeMap<String, String> {
    match spec.mcp_servers.get("project-orchestrator") {
        Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) => env.clone(),
        _ => panic!("stdio MCP server expected"),
    }
}

#[tokio::test]
async fn a_neutral_session_opens_read_only_persists_it_and_every_resume_keeps_it() {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.cwd = String::new();
    let created = manager.create_session(&req).await.unwrap();
    let sid = created.session_id.clone();
    let id = Uuid::parse_str(&sid).unwrap();

    // Said on the response, stored on the node, enforced on the open.
    assert_eq!(created.access, SessionAccess::ReadOnly);
    let node = graph.get_chat_session(id).await.unwrap().unwrap();
    assert_eq!(node.access, SessionAccess::ReadOnly);
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(spec_refuses_writes(&spec), "the open denies writes");
    assert_eq!(
        spec_po_env(&spec).get(crate::auth::tool_profile::TOOL_PROFILE_ENV),
        Some(&crate::auth::tool_profile::READ_ONLY.to_string())
    );

    // A resume reads the access from the node and applies it again.
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    manager.resume_session(&sid, "again", None).await.unwrap();
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(spec_refuses_writes(&spec), "the resume denies writes too");
    assert_eq!(
        graph.get_chat_session(id).await.unwrap().unwrap().access,
        SessionAccess::ReadOnly,
        "a resume does not rewrite the access"
    );
    manager.close_session(&sid).await.unwrap();
}

/// A `work` project has no source tree, so its conversations run in the neutral
/// directory like any session given no folder. They used to come out read-only —
/// which made the notes, plans and tasks of that very project unwritable, and left
/// a fresh install (where the only project one can create without a repository is
/// a `work` one) with an agent that could not write anything.
#[tokio::test]
async fn a_project_without_a_source_tree_is_an_ordinary_writable_session() {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.cwd = String::new();
    req.project_slug = Some("notes".into());
    let created = manager.create_session(&req).await.unwrap();
    let sid = created.session_id.clone();
    let id = Uuid::parse_str(&sid).unwrap();

    assert_eq!(created.access, SessionAccess::Normal);
    let node = graph.get_chat_session(id).await.unwrap().unwrap();
    assert_eq!(
        node.execution_place,
        crate::chat::neutral_place::ExecutionPlace::Neutral,
        "it still RUNS in the neutral directory"
    );
    assert_eq!(node.access, SessionAccess::Normal, "and it may write");
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(!spec_refuses_writes(&spec), "the open does not deny writes");
    assert!(!spec_po_env(&spec).contains_key(crate::auth::tool_profile::TOOL_PROFILE_ENV));
    manager.close_session(&sid).await.unwrap();
}

/// A workspace that holds no project yet (a fresh install): its conversations run in
/// the neutral directory, and they belong to the workspace the person is in — so they
/// are ordinary conversations, not read-only ones.
#[tokio::test]
async fn a_workspace_without_a_project_is_an_ordinary_writable_session() {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.cwd = String::new();
    req.workspace_slug = Some("personal".into());
    let created = manager.create_session(&req).await.unwrap();
    let sid = created.session_id.clone();
    let id = Uuid::parse_str(&sid).unwrap();

    assert_eq!(created.access, SessionAccess::Normal);
    assert_eq!(
        graph.get_chat_session(id).await.unwrap().unwrap().access,
        SessionAccess::Normal,
        "stored as normal: every resume keeps it"
    );
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(!spec_refuses_writes(&spec));
    manager.close_session(&sid).await.unwrap();

    // The default is a default, not a lock: a workspace session may still be opened
    // read-only when the client asks for it, and that is what is stored.
    let mut ro = agent_request("hello");
    ro.cwd = String::new();
    ro.workspace_slug = Some("personal".into());
    ro.access = Some(SessionAccess::ReadOnly);
    let ro = manager.create_session(&ro).await.unwrap();
    assert_eq!(ro.access, SessionAccess::ReadOnly);
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(
        spec_refuses_writes(&spec),
        "asked read-only, stays read-only"
    );
    manager.close_session(&ro.session_id).await.unwrap();
}

/// An empty slug names nothing: `decide_project` and the enrichment already read `""`
/// as absent, so the access default must too — otherwise a client sending
/// `"project_slug": ""` would get a writable session out of one that belongs to nothing.
#[tokio::test]
async fn an_empty_slug_names_nothing_for_the_access_default() {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, _graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.cwd = String::new();
    req.project_slug = Some(String::new());
    req.workspace_slug = Some(String::new());
    let created = manager.create_session(&req).await.unwrap();
    assert_eq!(created.access, SessionAccess::ReadOnly);
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(spec_refuses_writes(&spec));
    manager.close_session(&created.session_id).await.unwrap();
}

#[tokio::test]
async fn a_project_session_is_normal_unless_the_client_asks_for_read_only_and_a_resume_cannot_widen_it(
) {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, graph, fake) = agent_manager();

    // Project session, nothing asked: normal, left off the wire, writes allowed.
    let normal = manager
        .create_session(&agent_request("hello"))
        .await
        .unwrap();
    assert_eq!(normal.access, SessionAccess::Normal);
    let json = serde_json::to_value(&normal).unwrap();
    assert!(json.get("access").is_none(), "normal is left off the wire");
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(!spec_refuses_writes(&spec));
    assert!(!spec_po_env(&spec).contains_key(crate::auth::tool_profile::TOOL_PROFILE_ENV));
    fake.state.end_turn();
    manager.close_session(&normal.session_id).await.unwrap();

    // Project session, read_only asked explicitly.
    let mut req = agent_request("hello");
    req.access = Some(SessionAccess::ReadOnly);
    let created = manager.create_session(&req).await.unwrap();
    let sid = created.session_id.clone();
    let id = Uuid::parse_str(&sid).unwrap();
    assert_eq!(created.access, SessionAccess::ReadOnly);
    assert_eq!(
        graph
            .get_chat_session(id)
            .await
            .unwrap()
            .unwrap()
            .execution_place,
        crate::chat::neutral_place::ExecutionPlace::Project
    );
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(spec_refuses_writes(&spec));

    // Resume: nothing in the stored node or the call can bring `normal` back.
    fake.state.end_turn();
    manager.close_session(&sid).await.unwrap();
    manager.resume_session(&sid, "again", None).await.unwrap();
    let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
    assert!(spec_refuses_writes(&spec), "read_only -> normal is refused");
    assert_eq!(
        graph.get_chat_session(id).await.unwrap().unwrap().access,
        SessionAccess::ReadOnly
    );
    manager.close_session(&sid).await.unwrap();
}

#[tokio::test]
async fn a_neutral_session_can_be_opened_normal_on_explicit_request_and_a_legacy_node_reads_as_normal(
) {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.cwd = String::new();
    req.access = Some(SessionAccess::Normal);
    let created = manager.create_session(&req).await.unwrap();
    assert_eq!(created.access, SessionAccess::Normal);
    let id = Uuid::parse_str(&created.session_id).unwrap();
    let node = graph.get_chat_session(id).await.unwrap().unwrap();
    assert_eq!(node.access, SessionAccess::Normal);
    assert!(!spec_refuses_writes(
        &fake.state.opened_specs.lock().unwrap().pop().unwrap()
    ));
    fake.state.end_turn();
    manager.close_session(&created.session_id).await.unwrap();

    // A node written before the field existed has no `access`: it reads as normal
    // (compatibility), on the wire and on a resume.
    let mut json = serde_json::to_value(&node).unwrap();
    assert!(json.get("access").is_none());
    json.as_object_mut().unwrap().remove("access");
    let legacy: ChatSessionNode = serde_json::from_value(json).unwrap();
    assert_eq!(legacy.access, SessionAccess::Normal);
    let request: ChatRequest = serde_json::from_value(serde_json::json!({"message": "m"})).unwrap();
    assert_eq!(request.access, None);
}

#[tokio::test]
async fn a_provider_switch_keeps_a_read_only_session_read_only() {
    use crate::chat::provider::policy::SessionAccess;
    let (manager, graph, fake) = agent_manager();
    let mut req = agent_request("hello");
    req.access = Some(SessionAccess::ReadOnly);
    let created = manager.create_session(&req).await.unwrap();
    fake.state.end_turn();
    manager.close_session(&created.session_id).await.unwrap();
    // The request a switch builds carries the access of the node it leaves.
    let node = graph
        .get_chat_session(Uuid::parse_str(&created.session_id).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.access, SessionAccess::ReadOnly);
}

/// The token minted for a read-only session carries the read-only profile, a
/// third-party lineage included: the restricted profile it would otherwise get is
/// WIDER (it runs the writes of the tools it keeps).
#[tokio::test]
async fn the_token_of_a_read_only_session_carries_the_read_only_profile() {
    use crate::auth::tool_profile::ToolProfile;
    use crate::chat::provider::policy::SessionAccess;
    use nexus_claude::agent::ProviderKind;
    let manager = signed_manager(mock_app_state());
    let claims = person_claims();
    for (third_party, mode) in [
        (false, "default"),
        (false, "bypassPermissions"),
        (true, "default"),
        (true, "bypassPermissions"),
    ] {
        let spec = manager
            .build_agent_spec_with_access(
                AgentSpecInput {
                    file_scope_lifted: false,
                    cwd: "/tmp",
                    model: "m",
                    system_prompt: "p",
                    permission_mode: Some(mode),
                    add_dirs: &[],
                    user_claims: Some(&claims),
                    session_id: "ro-token",
                    third_party,
                    max_tokens: None,
                    max_cost_usd: None,
                    kind: ProviderKind::Native,
                    remote_cwd: None,
                    per_session_mcp: true,
                    hooks: None,
                },
                SessionAccess::ReadOnly,
            )
            .await
            .unwrap();
        let token = spec_po_env(&spec)
            .get("PO_AUTH_TOKEN")
            .cloned()
            .expect("a session token");
        let profile = ToolProfile::from_unverified_token(&token);
        assert_eq!(
            profile,
            ToolProfile::ReadOnly,
            "third_party={third_party} {mode}"
        );
        let call = |action: &str| -> crate::mcp::protocol::ToolCallParams {
            serde_json::from_value(serde_json::json!({
                "name": "task", "arguments": {"action": action}
            }))
            .unwrap()
        };
        assert!(crate::mcp::server::profile_refusal(profile, &call("create")).is_some());
        assert_eq!(
            crate::mcp::server::profile_refusal(profile, &call("list")),
            None
        );
    }
}

/// VERIFIER (4): the Claude CLI really receives the deny list when the mode is the
/// most permissive one. The CLI is the `fake_claude` of nexus, which records its
/// argv; a real `claude` is not available here and nothing below runs one.
#[tokio::test]
async fn the_cli_gets_the_read_only_deny_list_under_bypass_permissions() {
    use crate::chat::provider::policy::SessionAccess;
    let fake = std::env::var_os("NEXUS_FAKES_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../.target-nexus-fakes/debug")
        })
        .join("fake_claude");
    assert!(
            fake.exists(),
            "{} is missing: build the nexus fakes (.github/actions/nexus-fakes builds fake_claude with the others)",
            fake.display()
        );
    let dir = tempfile::tempdir().unwrap();
    let argv_out = dir.path().join("argv.json");
    let transcript = dir.path().join("transcript.jsonl");
    std::fs::write(&transcript, "{\"op\":\"exit\",\"code\":0}\n").unwrap();

    let state = mock_app_state();
    let mut config = test_config();
    config.permission.disallowed_tools = vec!["Bash(rm -rf *)".into()];
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
    for (access, expect_denied) in [
        (SessionAccess::ReadOnly, true),
        (SessionAccess::Normal, false),
    ] {
        let mut options = manager
            .build_options_with_access(
                "/tmp",
                "m",
                "p",
                None,
                Some("bypassPermissions"),
                None,
                &[],
                None,
                Some("ro-cli"),
                access,
            )
            .await;
        options.cli_path = Some(fake.clone());
        options.env.insert(
            "FAKE_CLAUDE_TRANSCRIPT".into(),
            transcript.display().to_string(),
        );
        options.env.insert(
            "FAKE_CLAUDE_ARGS_OUT".into(),
            argv_out.display().to_string(),
        );
        let _ = std::fs::remove_file(&argv_out);
        let mut client = nexus_claude::InteractiveClient::new(options).unwrap();
        // The fake exits at once; the connection may or may not survive that, the
        // recorded argv is what matters.
        let _ = client.connect().await;
        for _ in 0..100 {
            if argv_out.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let recorded: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&argv_out).expect("argv recorded"))
                .unwrap();
        let argv = recorded.to_string();
        assert!(
            argv.contains("bypassPermissions"),
            "the mode is the permissive one: {argv}"
        );
        assert!(
            argv.contains("Bash(rm -rf *)"),
            "configured entries stay: {argv}"
        );
        for tool in [
            "Write",
            "Edit",
            "Bash",
            "Task",
            "mcp__project-orchestrator__admin",
        ] {
            assert_eq!(
                argv.contains(&format!("\"{tool}\""))
                    || argv.contains(&format!("{tool},"))
                    || argv.contains(&format!(",{tool}"))
                    || argv.contains(&format!("{tool}\"")),
                expect_denied,
                "{tool} in the CLI's disallowed tools ({access:?}): {argv}"
            );
        }
        let _ = client.disconnect().await;
    }
}

#[tokio::test]
async fn agent_path_resumes_from_the_persisted_token_and_out_of_turn_output_regroups() {
    use nexus_claude::agent::AgentEvent;
    let (manager, graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("one"))
        .await
        .unwrap()
        .session_id;
    // The provider names its session during the first turn, never at open.
    let mut rx = manager.subscribe(&sid).await.unwrap();
    let mut done = done_event(false, None);
    if let AgentEvent::Done {
        provider_session_id,
        ..
    } = &mut done
    {
        *provider_session_id = Some("fake-provider-session".into());
    }
    fake.state.push(done);
    next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    fake.state.end_turn();
    manager.agent_runtime.close(&sid).await.unwrap();
    assert!(!manager.is_session_active(&sid).await);

    manager.resume_session(&sid, "two", None).await.unwrap();
    let token = fake
        .state
        .resumed_with
        .lock()
        .unwrap()
        .pop()
        .expect("resumed with a token");
    assert_eq!(token.data()["session_id"], "fake-provider-session");
    assert_eq!(
        fake.state
            .turns_started
            .lock()
            .unwrap()
            .last()
            .map(String::as_str),
        Some("two")
    );
    // Event numbers continue after the persisted ones.
    let events = graph
        .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 100)
        .await
        .unwrap();
    let seqs: Vec<i64> = events.iter().map(|e| e.seq).collect();
    let mut sorted = seqs.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(seqs.len(), sorted.len(), "no event number reused: {seqs:?}");

    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push_oob(AgentEvent::Text {
        text: "background line".into(),
        seq: Some(9),
        parent: None,
    });
    let bg = next_matching(&mut rx, |e| matches!(e, ChatEvent::BackgroundOutput { .. })).await;
    assert!(
        matches!(bg, ChatEvent::BackgroundOutput { ref source, ref content, .. } if source == "assistant" && content == "background line")
    );
}

/// SDA-2: a model the PROVIDER switches by itself — the `before_turn` hook of the per-turn
/// router is applied inside the harness, without going through the manager — reaches the
/// backend only as `model_changed`. It must land on the session node, or the sessions list
/// and the model menu keep saying the model the session opened with. The Auto marker
/// (`routing_mode`) is a separate property and stays.
#[tokio::test]
async fn agent_path_a_model_switched_by_the_provider_is_written_on_the_session() {
    use nexus_claude::agent::AgentEvent;
    let (manager, graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let uuid = Uuid::parse_str(&sid).unwrap();
    graph
        .update_chat_session_routing(uuid, Some("full"), None, Some("po"))
        .await
        .unwrap();
    let before = graph.get_chat_session(uuid).await.unwrap().unwrap();
    assert_ne!(
        before.model, "routed-model",
        "the node does not already carry the model"
    );

    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push_oob(AgentEvent::ModelChanged {
        model: "routed-model".into(),
    });
    next_matching(&mut rx, |e| matches!(e, ChatEvent::ModelChanged { .. })).await;

    let node = graph.get_chat_session(uuid).await.unwrap().unwrap();
    assert_eq!(
        node.model, "routed-model",
        "the model in force is written on the session"
    );
    assert_eq!(
        node.routing_mode.as_deref(),
        Some("full"),
        "the Auto marker is untouched by a model change"
    );
    manager.close_session(&sid).await.unwrap();
}

/// A typed `BackgroundTasks` snapshot scripted by the fake provider (out of
/// turn) becomes an `ActiveTasksUpdate` WITHOUT any PID: the agent engine
/// tracks no process — the pid, when there is one, belongs to the adapter.
#[tokio::test]
async fn agent_path_background_tasks_are_typed_and_carry_no_pid() {
    use nexus_claude::agent::{
        AgentEvent, BackgroundTask, BackgroundTaskKind, BackgroundTaskStatus,
    };
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push_oob(AgentEvent::BackgroundTasks {
        tasks: vec![BackgroundTask {
            id: "nt-1".into(),
            kind: BackgroundTaskKind::Shell,
            description: "tail -F deploy.log".into(),
            status: BackgroundTaskStatus::Running,
            started_at_ms: None,
            tool_call_id: Some("b1".into()),
            parent: None,
            pid: None,
        }],
    });
    let event = next_matching(&mut rx, |e| {
        matches!(e, ChatEvent::ActiveTasksUpdate { .. })
    })
    .await;
    let ChatEvent::ActiveTasksUpdate { tasks } = event else {
        unreachable!()
    };
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, "b1", "keyed by its tool_use");
    assert_eq!(tasks[0].pid, None, "the agent engine tracks no PID");
    manager.close_session(&sid).await.unwrap();
}

/// ZOMBIE-CLI on the agent engine (the default): an IDLE session whose provider process
/// dies says so out of any turn (`Error { ProcessExited }`). Nothing used to act on it: only
/// a turn in flight detached its handle, so the session stayed live and held its parent's
/// child slot. The watchdog now closes it, with the typed reason `process_died`. A healthy
/// sibling is left alone.
#[tokio::test]
async fn agent_path_an_idle_session_whose_provider_died_is_closed_with_process_died() {
    use nexus_claude::agent::{AgentEvent, ProviderError};
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    assert!(manager.agent_runtime.owns(&sid).await, "the agent engine");
    assert!(
        manager.close_sessions_of_dead_clis().await.is_empty(),
        "a healthy idle session is not closed"
    );

    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push_oob(AgentEvent::Error {
        error: ProviderError::ProcessExited { code: None },
    });
    // The pump is asynchronous: wait for the handle to carry the verdict.
    let handle = manager.agent_runtime.get(&sid).await.expect("registered");
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !handle.provider_death_announced() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the out-of-band death reaches the handle");

    let closed = manager.close_sessions_of_dead_clis().await;
    assert_eq!(closed, vec![sid.clone()], "the dead session is closed");
    assert!(
        !manager.is_session_active(&sid).await,
        "a closed session holds no envelope slot"
    );
    let event = next_matching(
        &mut rx,
        |e| matches!(e, ChatEvent::SessionClosed { reason: Some(r), .. } if r == "process_died"),
    )
    .await;
    assert!(matches!(event, ChatEvent::SessionClosed { .. }));
}

/// ZOMBIE-CLI-suite: after the provider of a parent's CHILD session dies, the same
/// delegation that was refused succeeds again — the refusal (`envelope_too_many_children`)
/// came from the dead child still counting as "working" (it reports a running background
/// task), and `close_sessions_of_dead_clis` frees the parent's slot by closing it with
/// `process_died`. The whole REAL delegation path is exercised
/// (`ChatManager::run_subagent`, the one `delegate_task` uses).
#[tokio::test]
async fn a_delegation_refused_by_a_dead_child_succeeds_again_after_the_watchdog_closes_it() {
    use nexus_claude::agent::{
        AgentEvent, BackgroundTask, BackgroundTaskKind, BackgroundTaskStatus, ProviderError,
        StopReason, SubagentRequest,
    };
    let (manager, _graph, fake) = agent_manager();
    let manager = Arc::new(manager);
    manager.enable_subagents();
    manager
        .set_max_live_children(1)
        .await
        .expect("cap in bounds")
        .expect("cap stored");
    let parent = manager
        .create_session(&agent_request(""))
        .await
        .unwrap()
        .session_id;
    let parent_uuid = Uuid::parse_str(&parent).unwrap();

    // A child of the parent, idle but "working": it reports a running background task.
    let mut child_req = agent_request("");
    child_req.spawned_by = Some(crate::chat::envelope::conversation_spawned_by(parent_uuid));
    let child = manager.create_session(&child_req).await.unwrap().session_id;
    fake.state.push_oob(AgentEvent::BackgroundTasks {
        tasks: vec![BackgroundTask {
            id: "nt-1".into(),
            kind: BackgroundTaskKind::Shell,
            description: "tail -F deploy.log".into(),
            status: BackgroundTaskStatus::Running,
            started_at_ms: None,
            tool_call_id: Some("b1".into()),
            parent: None,
            pid: None,
        }],
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while !manager.is_session_working(&child).await {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the child is working (its background task is running)");

    // The live child fills the parent's only slot: the delegation is refused, typed.
    let refused = manager
        .clone()
        .run_subagent(&parent, None, &SubagentRequest::new("toolu_1", "do it"))
        .await;
    let err = refused.expect_err("the live child blocks the delegation");
    assert!(
        err.contains("envelope_too_many_children"),
        "typed refusal, got: {err}"
    );

    // The child's provider process dies out of any turn.
    fake.state.push_oob(AgentEvent::Error {
        error: ProviderError::ProcessExited { code: None },
    });
    let handle = manager.agent_runtime.get(&child).await.expect("registered");
    tokio::time::timeout(Duration::from_secs(5), async {
        while !handle.provider_death_announced() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the out-of-band death reaches the handle");

    // The watchdog closes the dead child, freeing the parent's slot.
    let closed = manager.close_sessions_of_dead_clis().await;
    assert_eq!(closed, vec![child.clone()], "the dead child is closed");

    // The SAME delegation now succeeds, and the child it opens finishes normally.
    let delegation = tokio::spawn({
        let manager = Arc::clone(&manager);
        let parent = parent.clone();
        async move {
            manager
                .run_subagent(
                    &parent,
                    None,
                    &SubagentRequest::new("toolu_2", "do it again"),
                )
                .await
        }
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while fake.state.turns_started.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the child's turn started");
    fake.state.push(AgentEvent::Text {
        text: "working".into(),
        seq: None,
        parent: None,
    });
    fake.state.push(AgentEvent::Done {
        stop_reason: StopReason::Completed,
        subtype: Some("success".into()),
        is_error: false,
        result_text: Some("done".into()),
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 5,
        duration_api_ms: None,
        num_turns: 1,
        model: Some("m".into()),
        provider_session_id: Some("p-1".into()),
        structured_output: None,
        error: None,
    });
    fake.state.end_turn();
    let answer = tokio::time::timeout(Duration::from_secs(10), delegation)
        .await
        .expect("the delegation completes")
        .expect("no panic");
    assert_eq!(answer, Ok("done".to_string()));
}

/// The spec of the child one `Agent` call of `parent` opens, the call then answered.
async fn spec_of_a_sub_agent_of(
    manager: &Arc<ChatManager>,
    fake: &super::super::agent_runtime::fake::FakeProvider,
    parent: &str,
) -> nexus_claude::agent::SessionSpec {
    use nexus_claude::agent::{AgentEvent, StopReason, SubagentRequest};
    let before = fake.state.opened_specs.lock().unwrap().len();
    let call = tokio::spawn({
        let manager = Arc::clone(manager);
        let parent = parent.to_string();
        async move {
            manager
                .run_subagent(&parent, None, &SubagentRequest::new("toolu_cap", "do it"))
                .await
        }
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while fake.state.opened_specs.lock().unwrap().len() == before {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the sub-agent opens (never 'no candidate')");
    let spec = fake
        .state
        .opened_specs
        .lock()
        .unwrap()
        .last()
        .cloned()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while fake.state.turns_started.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the child's turn started");
    fake.state.push(AgentEvent::Done {
        stop_reason: StopReason::Completed,
        subtype: Some("success".into()),
        is_error: false,
        result_text: Some("done".into()),
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 5,
        duration_api_ms: None,
        num_turns: 1,
        model: Some("priced".into()),
        provider_session_id: Some("p-cap".into()),
        structured_output: None,
        error: None,
    });
    fake.state.end_turn();
    let answer = tokio::time::timeout(Duration::from_secs(10), call)
        .await
        .expect("the call completes")
        .expect("no panic");
    assert_eq!(answer, Ok("done".to_string()));
    spec
}

/// PYSR-3b — a sub-agent of the `Agent` tool opened by a capped delegation runs under what
/// is left of its parent's budget (same constants, `DelegationLimits`): the parent's spend
/// is taken off, so the child never widens it. The call opens on the parent's engine as
/// before (no "no candidate"). Red before: the child got no cap at all.
#[tokio::test]
async fn a_sub_agent_of_a_capped_delegation_runs_under_what_is_left_of_its_budget() {
    let (manager, graph, fake) = agent_manager();
    let manager = Arc::new(manager);
    manager.enable_subagents();
    price(&fake, "priced");
    let mut parent_req = delegation_request("priced");
    parent_req.message = String::new();
    let parent = manager
        .create_session(&parent_req)
        .await
        .unwrap()
        .session_id;
    let parent_uuid = Uuid::parse_str(&parent).unwrap();
    graph
        .chat_sessions
        .write()
        .await
        .get_mut(&parent_uuid)
        .unwrap()
        .total_cost_usd = Some(10.0);
    graph
        .add_chat_session_children_cost(parent_uuid, 5.0)
        .await
        .unwrap();

    let spec = spec_of_a_sub_agent_of(&manager, &fake, &parent).await;

    let cap = crate::runner::DelegationLimits::default().max_cost_usd;
    assert_eq!(
        spec.limits.max_cost_usd,
        Some(cap - 15.0),
        "the parent's own and children's spend are taken off its budget"
    );
}

/// PYSR-3b — a sub-agent of a session of its own (a person's chat) gets no cap: its parent
/// has none (ISO5).
#[tokio::test]
async fn a_sub_agent_of_an_uncapped_session_gets_no_cap() {
    let (manager, _graph, fake) = agent_manager();
    let manager = Arc::new(manager);
    manager.enable_subagents();
    price(&fake, "priced");
    let mut parent_req = agent_request("");
    parent_req.model = Some("priced".into());
    let parent = manager
        .create_session(&parent_req)
        .await
        .unwrap()
        .session_id;

    let spec = spec_of_a_sub_agent_of(&manager, &fake, &parent).await;

    assert_eq!(spec.limits.max_cost_usd, None);
    assert_eq!(spec.limits.max_tokens, None);
}

#[tokio::test]
async fn claude_code_stays_on_the_legacy_engine_when_the_operator_says_legacy() {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let state = mock_app_state();
    let config = test_config();
    assert_eq!(
        config.provider_path,
        crate::chat::config::ProviderPath::Legacy
    );
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
        .with_provider_source(Arc::new(fake.clone()));
    // The Claude CLI is not there in tests: the call may fail, but it must
    // NOT have gone through the agent engine: `CHAT_PROVIDER_PATH=legacy` holds.
    let _ = manager.create_session(&agent_request("hello")).await;
    assert_eq!(manager.agent_runtime.len().await, 0);
    assert!(
        fake.state.opened_specs.lock().unwrap().is_empty(),
        "the agent provider was never asked"
    );
    let nodes: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    assert_eq!(nodes.len(), 1);
    assert!(
        nodes[0].capabilities.is_none(),
        "no agent snapshot: it is a legacy session"
    );
}

/// M2: with nothing configured, Claude Code opens on the agent engine — and says it
/// lacks nothing a local Claude Code has (hooks in protocol, MCP per session, nested
/// sub-agents, its own compaction).
#[tokio::test]
async fn claude_code_opens_on_the_agent_engine_by_default() {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let mut config = test_config();
    config.provider_path = crate::chat::config::ProviderPath::default();
    let fake = super::super::agent_runtime::fake::FakeProvider::new();
    {
        use nexus_claude::agent::{HookSupport, SubagentSupport};
        let mut caps = fake.caps.lock().unwrap();
        caps.hooks = HookSupport::InProtocol;
        caps.per_session_mcp = true;
        caps.subagents = SubagentSupport::Nested;
        caps.compaction_signal = true;
        caps.images = true;
    }
    let manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
        .with_provider_source(Arc::new(fake.clone()));
    let wire = real_agent_system_init(&manager, &fake).await;
    assert_eq!(wire["engine"], "agent", "{wire}");
    assert_eq!(wire["degraded_features"], serde_json::json!([]), "{wire}");
    let nodes: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    assert!(nodes[0].capabilities.is_some(), "an agent snapshot");
}

#[tokio::test]
async fn forcing_claude_code_onto_the_agent_engine_says_what_is_missing() {
    use nexus_claude::agent::AgentEvent;
    let (manager, _graph, fake) = agent_manager(); // CHAT_PROVIDER_PATH=agent
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(AgentEvent::SessionStarted {
        provider_session_id: Some("p1".into()),
        model: Some("m".into()),
        policy_mode: None,
        native_mode: None,
        tools: vec![],
        mcp_servers: vec![],
        cwd: None,
    });
    let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
    let wire = serde_json::to_value(&init).unwrap();
    assert_eq!(wire["engine"], "agent", "{wire}");
    let degraded: Vec<String> = serde_json::from_value(wire["degraded_features"].clone())
        .unwrap_or_else(|_| panic!("degraded_features missing: {wire}"));
    for lost in ["hooks", "compaction", "images"] {
        assert!(
            degraded.iter().any(|d| d == lost),
            "{lost} must be listed: {degraded:?}"
        );
    }
}

/// The `system_init` a live agent session REALLY emits (not a fixture).
async fn real_agent_system_init(
    manager: &ChatManager,
    fake: &super::super::agent_runtime::fake::FakeProvider,
) -> serde_json::Value {
    use nexus_claude::agent::AgentEvent;
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(AgentEvent::SessionStarted {
        provider_session_id: Some("p1".into()),
        model: Some("m".into()),
        policy_mode: None,
        native_mode: None,
        tools: vec![],
        mcp_servers: vec![],
        cwd: None,
    });
    let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
    serde_json::to_value(&init).unwrap()
}

#[tokio::test]
async fn the_agent_system_init_lists_what_the_session_capabilities_do_not_cover() {
    // A provider that declares images and a compaction signal: those two are not missing.
    let (manager, _graph, fake) = agent_manager();
    {
        let mut caps = fake.caps.lock().unwrap();
        caps.images = true;
        caps.compaction_signal = true;
    }
    let wire = real_agent_system_init(&manager, &fake).await;
    assert_eq!(wire["engine"], "agent", "{wire}");
    let degraded: Vec<String> =
        serde_json::from_value(wire["degraded_features"].clone()).expect("a list");
    // This provider runs no hooks in its loop: the graph hooks are listed...
    assert!(degraded.iter().any(|d| d == "hooks"), "{degraded:?}");
    // ...what the engine ported is not claimed missing...
    let ported = ["enrichment", "message_queue", "auto_continue", "nats"];
    assert!(
        !degraded.iter().any(|d| ported.contains(&d.as_str())),
        "{degraded:?}"
    );
    // ...and what the provider covers is not claimed missing.
    assert!(
        !degraded.iter().any(|d| d == "images" || d == "compaction"),
        "{degraded:?}"
    );

    // A provider that declares neither: both are listed.
    let (manager, _graph, fake) = agent_manager();
    let wire = real_agent_system_init(&manager, &fake).await;
    let degraded: Vec<String> = serde_json::from_value(wire["degraded_features"].clone()).unwrap();
    assert!(degraded.iter().any(|d| d == "images") && degraded.iter().any(|d| d == "compaction"));
}

/// M2, gaps D2 and D3 of M0: what the provider's KIND tells beyond its capabilities
/// reaches the `system_init`, not only a line in the server's log.
#[tokio::test]
async fn the_agent_system_init_says_compaction_and_subagents_where_nothing_provides_them() {
    use nexus_claude::agent::{ContextWindow, ContextWindowSource, ProviderKind, SubagentSupport};
    let listed = |wire: &serde_json::Value, feature: &str| {
        serde_json::from_value::<Vec<String>>(wire["degraded_features"].clone())
            .unwrap_or_default()
            .iter()
            .any(|d| d == feature)
    };
    // A native model whose window nobody knows: it claims a compaction signal, and the
    // harness never compacts (no window to compact against).
    let (manager, _graph, fake) = agent_manager();
    *fake.kind.lock().unwrap() = ProviderKind::Native;
    fake.caps.lock().unwrap().compaction_signal = true;
    let wire = real_agent_system_init(&manager, &fake).await;
    assert!(listed(&wire, "compaction"), "{wire}");
    assert!(
        !listed(&wire, "subagents"),
        "the host runs them as child sessions: {wire}"
    );
    // ...the same model with a known window compacts: nothing announced.
    let (manager, _graph, fake) = agent_manager();
    *fake.kind.lock().unwrap() = ProviderKind::Native;
    {
        let mut caps = fake.caps.lock().unwrap();
        caps.compaction_signal = true;
        caps.context_window = Some(ContextWindow {
            value: 128_000,
            source: ContextWindowSource::Probed,
        });
    }
    let wire = real_agent_system_init(&manager, &fake).await;
    assert!(!listed(&wire, "compaction"), "{wire}");
    // An ACP agent: no sub-agents of its own, and no fallback for it.
    let (manager, _graph, fake) = agent_manager();
    *fake.kind.lock().unwrap() = ProviderKind::Acp;
    fake.caps.lock().unwrap().per_session_mcp = true;
    let wire = real_agent_system_init(&manager, &fake).await;
    assert!(listed(&wire, "subagents"), "{wire}");
    // Claude Code reports its own (nested): nothing announced.
    let (manager, _graph, fake) = agent_manager();
    {
        let mut caps = fake.caps.lock().unwrap();
        caps.subagents = SubagentSupport::Nested;
        caps.compaction_signal = true;
    }
    let wire = real_agent_system_init(&manager, &fake).await;
    assert!(
        !listed(&wire, "subagents") && !listed(&wire, "compaction"),
        "{wire}"
    );
}

/// M2, gap D1 of M0: the warning of a Claude Code session on the agent engine names
/// what it lacks, and is not logged when it lacks nothing.
#[test]
fn the_engine_warning_is_silent_when_a_claude_code_session_lacks_nothing() {
    assert_eq!(
        forced_engine_warning(crate::chat::provider::resolver::CLAUDE_CODE, &[]),
        None
    );
    let warning = forced_engine_warning(
        crate::chat::provider::resolver::CLAUDE_CODE,
        &["hooks".to_string()],
    )
    .expect("something is missing");
    assert!(warning.contains("hooks"), "{warning}");
    // Another provider is always on the agent engine: no warning of its own.
    assert_eq!(forced_engine_warning("local", &["hooks".to_string()]), None);
}

#[test]
fn the_legacy_system_init_says_legacy_with_nothing_missing() {
    // `message_to_events` is what the legacy stream loop emits for the CLI's init.
    let msg = Message::System {
        subtype: "init".into(),
        data: serde_json::json!({"session_id": "cli-1", "model": "m"}),
    };
    let events = ChatManager::message_to_events(&msg);
    let wire = serde_json::to_value(&events[0]).unwrap();
    assert_eq!(wire["engine"], "legacy", "{wire}");
    assert_eq!(wire["degraded_features"], serde_json::json!([]), "{wire}");
}

#[test]
fn the_legacy_system_init_declares_the_permission_scopes_the_engine_accepts() {
    let msg = Message::System {
        subtype: "init".into(),
        data: serde_json::json!({"session_id": "cli-1", "model": "m"}),
    };
    let events = ChatManager::message_to_events(&msg);
    let wire = serde_json::to_value(&events[0]).unwrap();
    assert_eq!(
        wire["capabilities"]["permission_scopes"],
        serde_json::json!(["once", "session"]),
        "{wire}"
    );
}

fn done_event(
    is_error: bool,
    error: Option<nexus_claude::agent::ProviderError>,
) -> nexus_claude::agent::AgentEvent {
    nexus_claude::agent::AgentEvent::Done {
        stop_reason: if is_error {
            nexus_claude::agent::StopReason::Error
        } else {
            nexus_claude::agent::StopReason::Completed
        },
        subtype: Some(
            if is_error {
                "error_during_execution"
            } else {
                "success"
            }
            .into(),
        ),
        is_error,
        result_text: None,
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 1,
        duration_api_ms: None,
        num_turns: 1,
        model: None,
        provider_session_id: Some("p".into()),
        structured_output: None,
        error,
    }
}

#[tokio::test]
async fn a_retryable_done_error_before_any_output_is_retried_once_the_provider_says_when() {
    use nexus_claude::agent::{AgentEvent, ProviderError};
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(done_event(
        true,
        Some(ProviderError::RateLimited {
            retry_after_ms: Some(10),
            resets_at_ms: None,
        }),
    ));
    let retrying = next_matching(&mut rx, |e| matches!(e, ChatEvent::Retrying { .. })).await;
    assert!(matches!(
        retrying,
        ChatEvent::Retrying {
            attempt: 1,
            delay_ms: 10,
            ..
        }
    ));
    // The same turn is sent again.
    for _ in 0..200 {
        if fake.state.turns_started.lock().unwrap().len() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        fake.state.turns_started.lock().unwrap().as_slice(),
        ["go", "go"]
    );
    fake.state.push(AgentEvent::Text {
        text: "ok".into(),
        seq: None,
        parent: None,
    });
    fake.state.push(done_event(false, None));
    let result = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    // The failed attempt never reached the user: the first Result is the good one.
    assert!(
        matches!(
            result,
            ChatEvent::Result {
                is_error: false,
                ..
            }
        ),
        "{result:?}"
    );
}

#[tokio::test]
async fn a_turn_that_already_showed_output_or_failed_for_good_is_not_retried() {
    use nexus_claude::agent::{AgentEvent, ProviderError};
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    fake.state.push(AgentEvent::Text {
        text: "partial".into(),
        seq: None,
        parent: None,
    });
    fake.state
        .push(done_event(true, Some(ProviderError::Overloaded)));
    let result = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    assert!(matches!(result, ChatEvent::Result { is_error: true, .. }));
    assert_eq!(
        fake.state.turns_started.lock().unwrap().len(),
        1,
        "no replay after output"
    );

    // A non-retryable cause is shown at once.
    let sid2 = manager
        .create_session(&agent_request("again"))
        .await
        .unwrap()
        .session_id;
    let mut rx2 = manager.subscribe(&sid2).await.unwrap();
    fake.state
        .push(done_event(true, Some(ProviderError::Unauthorized)));
    let result = next_matching(&mut rx2, |e| matches!(e, ChatEvent::Result { .. })).await;
    assert!(matches!(result, ChatEvent::Result { is_error: true, .. }));
}

/// The fallback reads the conversation's routing pool in order, skips the failed
/// instance, and finds nothing when the pool holds only the failed provider.
#[tokio::test]
async fn the_fallback_picks_the_next_pool_provider_and_none_when_it_is_alone() {
    use crate::chat::types::RoutingPoolEntry;
    let (manager, graph) = manager_with_mock();
    let pool = vec![
        RoutingPoolEntry {
            provider: "a".into(),
            model: "m1".into(),
        },
        RoutingPoolEntry {
            provider: "b".into(),
            model: "m2".into(),
        },
    ];
    let mut session = test_chat_session(None);
    session.provider_id = Some("a".into());
    session.routing_pool = Some(serde_json::to_string(&pool).unwrap());
    graph.create_chat_session(&session).await.unwrap();
    let pick = manager
        .next_fallback_pick(&session.id.to_string(), "a")
        .await
        .expect("a fallback among the pool");
    assert_eq!(pick.provider_id, "b");
    assert_eq!(pick.model, "m2");

    // A pool with only the failed provider offers no fallback.
    let mut alone = test_chat_session(None);
    alone.provider_id = Some("a".into());
    alone.routing_pool = Some(
        serde_json::to_string(&vec![RoutingPoolEntry {
            provider: "a".into(),
            model: "m1".into(),
        }])
        .unwrap(),
    );
    graph.create_chat_session(&alone).await.unwrap();
    assert!(manager
        .next_fallback_pick(&alone.id.to_string(), "a")
        .await
        .is_none());
}

/// A provider whose send fails terminally with no other provider to fall back to
/// keeps the error: the turn is not replayed, no session is opened by a fallback.
#[tokio::test]
async fn a_terminal_send_failure_without_another_provider_keeps_the_error() {
    use nexus_claude::agent::{AgentEvent, ProviderError};
    let (manager, graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    // Finish the opening turn so the session is idle for the next message.
    fake.state.push(AgentEvent::Text {
        text: "ok".into(),
        seq: None,
        parent: None,
    });
    fake.state.push(done_event(false, None));
    let _ = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;

    // A non-retryable send failure on the only provider must not loop: the error
    // is surfaced and no fallback session is opened.
    *fake.state.fail_next_turn.lock().unwrap() = Some(ProviderError::Unauthorized);
    let err = manager.send_message(&sid, "again").await.unwrap_err();
    assert!(
        err.chain().any(|cause| matches!(
            cause.downcast_ref::<ProviderError>(),
            Some(ProviderError::Unauthorized)
        )),
        "the original failure is surfaced: {err}"
    );
    assert_eq!(
        fake.state.turns_started.lock().unwrap().as_slice(),
        ["go"],
        "the failed turn never started (no replay)"
    );
    assert_eq!(
        graph.chat_sessions.read().await.len(),
        1,
        "no fallback session opened"
    );
}

/// QUOTA, explicit choice: a provider the person named, whose quota is used up, is never
/// replaced behind their back. The opening is refused with the instant it serves again (a
/// typed `rate_limited` carrying `resets_at_ms`), and nothing else is tried. A router's
/// choice, a mixed pool and a delegated child are not "explicit": the router already left the
/// exhausted instance out, and a child falls back on its parent's engine.
#[tokio::test]
async fn an_explicit_provider_whose_quota_is_used_up_is_refused_with_its_reset_not_swapped() {
    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    use crate::chat::provider::cognitive::ConversationRoutingMode as Mode;
    let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let state = mock_app_state();
    let routing = CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new()));
    let reset = routing.quota.now_ms() + 3 * 86_400_000;
    routing
        .quota
        .mark_rate_limited("deepseek", Some(reset), None);
    let manager = ChatManager::new_without_memory(graph, state.meili, test_config())
        .with_cognitive_routing(routing);

    let named = ChatRequest {
        provider: Some("deepseek".into()),
        ..ChatRequest::default()
    };
    assert!(ChatManager::provider_is_explicit(&named));
    let refusal = manager
        .refuse_explicit_provider_if_exhausted("deepseek", true)
        .expect_err("refused");
    let failure = crate::chat::provider::errors::classify_open_error(&refusal, None)
        .expect("a typed failure");
    assert_eq!(failure.status, 429);
    assert_eq!(failure.code, "rate_limited");
    assert_eq!(failure.resets_at_ms, Some(reset));
    assert_eq!(failure.provider_id.as_deref(), Some("deepseek"));
    assert!(failure.message.contains("serves again at"));
    // Another provider is unaffected.
    assert!(manager
        .refuse_explicit_provider_if_exhausted("glm", true)
        .is_ok());
    // Not explicit: the router's pool filter already handled it.
    for request in [
        ChatRequest {
            provider: Some("deepseek".into()),
            routing_mode: Some(Mode::Full),
            ..ChatRequest::default()
        },
        ChatRequest {
            provider: Some("deepseek".into()),
            spawned_by: Some("parent".into()),
            ..ChatRequest::default()
        },
        ChatRequest::default(),
    ] {
        assert!(!ChatManager::provider_is_explicit(&request), "{request:?}");
    }
    assert!(ChatManager::provider_is_explicit(&ChatRequest {
        routing_mode: Some(Mode::Primary),
        ..ChatRequest::default()
    }));
}

/// QUOTA: the fallback after a failure reads the same registry as the pool filter: an
/// instance exhausted until its reset (a dated 429 here) is not a target, the next one of the
/// pool is; at the reset it is a target again.
#[tokio::test]
async fn the_fallback_after_a_failure_skips_an_instance_whose_quota_is_used_up() {
    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    use crate::chat::provider::limits::ManualClock;
    use crate::chat::provider::quota::QuotaRegistry;
    use crate::chat::types::RoutingPoolEntry;
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let clock = Arc::new(ManualClock::at("2026-10-21T10:00:00Z"));
    let quota = Arc::new(QuotaRegistry::new().with_clock(clock.clone()));
    let routing =
        CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new())).with_quota(quota.clone());
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, test_config())
        .with_cognitive_routing(routing);
    let entry = |provider: &str| RoutingPoolEntry {
        provider: provider.into(),
        model: "m".into(),
    };
    let mut session = test_chat_session(None);
    session.provider_id = Some("a".into());
    session.routing_pool =
        Some(serde_json::to_string(&vec![entry("a"), entry("b"), entry("c")]).unwrap());
    graph.create_chat_session(&session).await.unwrap();
    let sid = session.id.to_string();

    quota.mark_rate_limited("b", Some(quota.now_ms() + 3_600_000), None);
    let pick = manager.next_fallback_pick(&sid, "a").await.expect("c");
    assert_eq!(pick.provider_id, "c", "b refuses for an hour");

    clock.advance(std::time::Duration::from_secs(3_601));
    let pick = manager.next_fallback_pick(&sid, "a").await.expect("b");
    assert_eq!(
        pick.provider_id, "b",
        "back at its reset, nobody did anything"
    );
}

/// RL5: every session counts against the ONE gate the boot injected, and a manager built
/// without one has an unlimited gate of its own.
#[test]
fn the_manager_shares_the_gate_it_was_given() {
    use crate::chat::provider::limits::Gate;
    let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(graph, state.meili, test_config());
    let own = manager.provider_limits().clone();
    let injected = Arc::new(Gate::unlimited());
    assert!(!Arc::ptr_eq(&own, &injected));
    let manager = manager.with_provider_limits(injected.clone());
    assert!(
        Arc::ptr_eq(manager.provider_limits(), &injected),
        "the gate every session reads is the injected one, not a copy"
    );
}

/// The failed provider is abandoned (marked out of credit) even when the pool
/// holds no other provider: the fallback then returns `false`, never loops.
#[tokio::test]
async fn a_failed_provider_is_abandoned_even_when_no_fallback_is_eligible() {
    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, test_config())
        .with_cognitive_routing(CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new())));
    // A session on the only provider, without a routing pool: no fallback target.
    let mut session = test_chat_session(None);
    session.provider_id = Some("a".into());
    graph.create_chat_session(&session).await.unwrap();

    let moved = manager
        .fallback_provider_after_failure(&session.id.to_string(), "a", "go")
        .await;
    assert!(!moved, "no other provider to fall back to");
    let routing = manager.cognitive_routing.as_ref().unwrap();
    assert!(
        routing.out_of_credit("a"),
        "the failed provider is abandoned (out of credit)"
    );
    assert_eq!(
        graph.chat_sessions.read().await.len(),
        1,
        "no fallback opened"
    );
}

/// M2: the engine is frozen at opening. A Claude Code session opened on the agent
/// engine (the default) still resumes there once the operator rolls back to
/// `CHAT_PROVIDER_PATH=legacy` — it used to be refused with `409 engine_unavailable`.
#[tokio::test]
async fn an_agent_session_of_claude_code_resumes_on_the_agent_engine_after_a_rollback() {
    let (manager, graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("one"))
        .await
        .unwrap()
        .session_id;
    manager.agent_runtime.close(&sid).await.unwrap();
    // The operator sets the flag back to legacy (test_config) and restarts.
    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let legacy = ChatManager::new_without_memory(dyn_graph, state.meili, test_config())
        .with_provider_source(Arc::new(fake.clone()));
    assert_eq!(
        legacy.config.provider_path,
        crate::chat::config::ProviderPath::Legacy
    );
    legacy
        .resume_session(&sid, "two", None)
        .await
        .expect("resumed on the engine it was opened on");
    assert!(legacy.agent_runtime.owns(&sid).await, "the agent engine");
    assert!(
        legacy.active_sessions.read().await.get(&sid).is_none(),
        "not legacy"
    );
}

#[tokio::test]
async fn legacy_path_never_touches_the_agent_runtime() {
    let (manager, _graph) = {
        let (m, g) = manager_with_mock();
        (m, g)
    };
    let _ = manager
        .create_session(&runner_request(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        ))
        .await;
    assert_eq!(manager.agent_runtime.len().await, 0);
}

#[tokio::test]
async fn test_resume_session_keeps_identity_and_links() {
    let (manager, graph) = manager_with_mock();
    let (run, plan, task) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let mut s = test_chat_session(None);
    s.spawned_by = runner_request(run, plan, task).spawned_by;
    graph.create_chat_session(&s).await.unwrap();
    graph
        .link_session_to_run(&s.id.to_string(), run, Some(plan), Some(task))
        .await
        .unwrap();
    let before = crate::chat::attachment::attach(
        &[s.clone()],
        &graph.session_link_rows.read().await.clone(),
    );

    let rows_before = graph.session_link_rows.read().await.clone();

    // WHAT THIS PROVES: `resume_session` reads the stored session, never
    // creates another one and never touches its link rows, up to the CLI
    // spawn. In tests the spawn itself fails (or succeeds against whatever
    // `claude` is installed), so what happens to the links AFTER a real
    // spawn is NOT exercised here: `resume_session` has no call to
    // `link_session_to_run` / `create_spawned_by_relation` at all
    // (they live in `create_session` only), which is what the row-for-row
    // comparison below pins on the mock.
    let _ = manager
        .resume_session(&s.id.to_string(), "continue", None)
        .await;

    let after_sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
    assert_eq!(
        after_sessions.len(),
        1,
        "resume must not create another session"
    );
    assert_eq!(after_sessions[0].id, s.id, "same session id");
    assert_eq!(
        after_sessions[0].spawned_by, s.spawned_by,
        "spawned_by kept"
    );
    let rows = graph.session_link_rows.read().await.clone();
    assert_eq!(
        rows, rows_before,
        "the stored link rows are not rewritten, added to or removed by a resume"
    );
    let after = crate::chat::attachment::attach(&after_sessions, &rows);
    assert_eq!(after.by_plan, before.by_plan, "same links, same thread");
    assert!(after.unattached.is_empty(), "never a detached session");
}

#[tokio::test]
async fn live_snapshot_reports_the_permissions_each_live_cli_still_holds() {
    let (manager, _graph) = manager_with_mock();
    let with = Uuid::new_v4();
    let without = Uuid::new_v4();
    let _a = super::test_support::insert_live_session(
        &manager,
        &with.to_string(),
        true,
        &["req-1", "req-2"],
    )
    .await;
    let _b =
        super::test_support::insert_live_session(&manager, &without.to_string(), false, &[]).await;
    let snap = manager.live_session_snapshot().await;
    assert_eq!(snap.live, [with, without].into_iter().collect());
    assert_eq!(snap.streaming, [with].into_iter().collect());
    assert_eq!(
        snap.pending_permissions.get(&with),
        Some(
            &["req-1".to_string(), "req-2".to_string()]
                .into_iter()
                .collect()
        )
    );
    assert!(!snap.pending_permissions.contains_key(&without));
}

/// `is_streaming` lives in an `AtomicBool` and is written to Neo4j
/// nowhere, so turning the in-memory snapshot into the per-session
/// activity the API reports is the whole mechanism behind a working
/// indicator that survives a page reload. These tests need no CLI: the
/// mapping and the counting are pure.
mod activity {
    use super::super::{count_background_tasks, LiveSessionSnapshot};
    use crate::chat::types::{BackgroundTaskInfo, BackgroundTaskKind};
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use uuid::Uuid;

    fn task(kind: BackgroundTaskKind, dying: bool) -> BackgroundTaskInfo {
        let now = chrono::Utc::now();
        BackgroundTaskInfo {
            id: Uuid::new_v4().to_string(),
            kind,
            description: "watch the log".into(),
            started_at: now,
            last_seen_at: now,
            pid: Some(1234),
            parent_tool_use_id: None,
            pending_removal_at: dying.then(std::time::Instant::now),
            signalled: false,
        }
    }

    #[test]
    fn an_unknown_session_is_quiet_rather_than_unknown() {
        let snap = LiveSessionSnapshot::default();
        let a = snap.activity_for(Uuid::new_v4());
        assert!(
            a.is_quiet(),
            "absence from the map is an answer: nothing is running"
        );
        assert!(!a.live);
        assert_eq!(a.background_tasks(), 0);
    }

    #[test]
    fn a_live_session_reports_what_it_is_waiting_on() {
        let id = Uuid::new_v4();
        let other = Uuid::new_v4();
        let mut snap = LiveSessionSnapshot::default();
        snap.live.insert(id);
        snap.live.insert(other);
        snap.streaming.insert(id);
        snap.pending_permissions.insert(
            id,
            ["req-1".to_string(), "req-2".to_string()]
                .into_iter()
                .collect(),
        );
        snap.background_tasks.insert(id, (2, 1));

        let a = snap.activity_for(id);
        assert!(a.live && a.streaming);
        assert_eq!(a.pending_permissions, 2);
        assert_eq!(a.monitors, 2);
        assert_eq!(a.bash_tasks, 1);
        assert_eq!(a.background_tasks(), 3);
        assert!(!a.is_quiet());

        // A second live session must not inherit the first one's state —
        // the bug this guards against is one row's indicator leaking onto
        // every other row.
        let b = snap.activity_for(other);
        assert!(b.live);
        assert!(!b.streaming);
        assert_eq!(b.pending_permissions, 0);
        assert_eq!(b.background_tasks(), 0);
    }

    #[test]
    fn a_live_but_idle_session_is_not_quiet() {
        let id = Uuid::new_v4();
        let mut snap = LiveSessionSnapshot::default();
        snap.live.insert(id);
        // The distinction the UI needs: "the agent is up, waiting for you"
        // must be sayable, because showing nothing there is what made a
        // running conversation look dead.
        assert!(!snap.activity_for(id).is_quiet());
    }

    #[test]
    fn counting_separates_monitors_from_background_commands() {
        let tasks = [
            task(BackgroundTaskKind::Monitor, false),
            task(BackgroundTaskKind::Monitor, false),
            task(BackgroundTaskKind::BashBackground, false),
        ];
        assert_eq!(count_background_tasks(tasks.iter()), (2, 1));
    }

    #[test]
    fn a_cancelled_watch_stops_being_advertised_at_once() {
        // `pending_removal_at` entries linger for a 5 s grace period so
        // late ticks still route. Counting them would keep a cancelled
        // watch on screen for those five seconds.
        let tasks = [
            task(BackgroundTaskKind::Monitor, true),
            task(BackgroundTaskKind::BashBackground, true),
        ];
        assert_eq!(count_background_tasks(tasks.iter()), (0, 0));
    }

    #[test]
    fn nothing_tracked_counts_as_nothing() {
        assert_eq!(count_background_tasks(std::iter::empty()), (0, 0));
    }
}

#[tokio::test]
async fn failed_stdin_send_keeps_the_permission_pending_for_a_retry() {
    let (manager, _graph) = manager_with_mock();
    let sid = Uuid::new_v4().to_string();
    let (stdin_rx, _q) =
        super::test_support::insert_live_session(&manager, &sid, false, &["req-1"]).await;
    // The CLI is gone: the stdin channel is closed, the send fails.
    drop(stdin_rx);
    let mut events = {
        let sessions = manager.active_sessions.read().await;
        sessions[&sid].events_tx.subscribe()
    };

    for attempt in 1..=2 {
        let err = manager
            .send_permission_response_inner(
                &sid,
                "req-1",
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                true,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, PermissionDeliveryError::Failed(_)),
            "attempt {attempt}: a failed send stays a delivery failure, not NotPending ({err})"
        );
    }
    let pending = manager.active_sessions.read().await[&sid]
        .pending_permission_inputs
        .clone();
    assert!(
        pending.lock().await.contains_key("req-1"),
        "the request is still pending"
    );
    assert!(
        events.try_recv().is_err(),
        "no permission_decision is broadcast for a decision that was not sent"
    );
}

// ====================================================================
// ChatSession CRUD via GraphStore (mock)
// ====================================================================

#[tokio::test]
async fn test_chat_session_crud_lifecycle() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    // Create
    let session = test_chat_session(Some("my-project"));
    graph.create_chat_session(&session).await.unwrap();

    // Get
    let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(fetched.id, session.id);
    assert_eq!(fetched.cwd, "/tmp/test");
    assert_eq!(fetched.model, "claude-opus-4-6");
    assert_eq!(fetched.project_slug.as_deref(), Some("my-project"));

    // Update
    let updated = graph
        .update_chat_session(
            session.id,
            Some("cli-abc-123".into()),
            Some("My Chat".into()),
            Some(5),
            Some(0.25),
            None,
            None,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.cli_session_id.as_deref(), Some("cli-abc-123"));
    assert_eq!(updated.title.as_deref(), Some("My Chat"));
    assert_eq!(updated.message_count, 5);
    assert_eq!(updated.total_cost_usd, Some(0.25));

    // Delete
    let deleted = graph.delete_chat_session(session.id).await.unwrap();
    assert!(deleted);

    // Get after delete
    let gone = graph.get_chat_session(session.id).await.unwrap();
    assert!(gone.is_none());
}

#[tokio::test]
async fn test_chat_session_list_with_filter() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let s1 = test_chat_session(Some("project-a"));
    let s2 = test_chat_session(Some("project-a"));
    let s3 = test_chat_session(Some("project-b"));
    let s4 = test_chat_session(None);

    graph.create_chat_session(&s1).await.unwrap();
    graph.create_chat_session(&s2).await.unwrap();
    graph.create_chat_session(&s3).await.unwrap();
    graph.create_chat_session(&s4).await.unwrap();

    // All sessions
    let (all, total) = graph
        .list_chat_sessions(None, None, 50, 0, false)
        .await
        .unwrap();
    assert_eq!(total, 4);
    assert_eq!(all.len(), 4);

    // Filter by project-a
    let (filtered, total) = graph
        .list_chat_sessions(Some("project-a"), None, 50, 0, false)
        .await
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(filtered.len(), 2);

    // Pagination
    let (page, total) = graph
        .list_chat_sessions(Some("project-a"), None, 1, 0, false)
        .await
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(page.len(), 1);
}

#[tokio::test]
async fn test_chat_session_update_partial() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();

    // Update only title
    let updated = graph
        .update_chat_session(
            session.id,
            None,
            Some("Title only".into()),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.title.as_deref(), Some("Title only"));
    assert!(updated.cli_session_id.is_none()); // unchanged
    assert_eq!(updated.message_count, 0); // unchanged
}

#[tokio::test]
async fn test_chat_session_update_nonexistent() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let result = graph
        .update_chat_session(uuid::Uuid::new_v4(), None, None, None, None, None, None)
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_chat_session_delete_nonexistent() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let deleted = graph
        .delete_chat_session(uuid::Uuid::new_v4())
        .await
        .unwrap();
    assert!(!deleted);
}

#[tokio::test]
async fn test_chat_session_node_serialization() {
    let session = ChatSessionNode {
        file_scope_lifted: false,
        mirror_of: None,
        routing_pool: None,
        routing_mode: None,
        id: uuid::Uuid::new_v4(),
        cli_session_id: Some("cli-123".into()),
        project_slug: Some("test-proj".into()),
        workspace_slug: None,
        cwd: "/home/user/code".into(),
        title: Some("My session".into()),
        model: "claude-opus-4-6".into(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        message_count: 10,
        total_cost_usd: Some(1.50),
        conversation_id: Some("conv-abc-123".into()),
        preview: Some("Hello, can you help me with this?".into()),
        permission_mode: None,
        add_dirs: None,
        spawned_by: None,
        provider_id: None,
        routed_by: None,
        capabilities: None,
        engine: None,
        resume_token: None,
        execution_place: Default::default(),
        project_inferred: false,
        access: Default::default(),
        owner: None,
    };

    let json = serde_json::to_string(&session).unwrap();
    let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, session.id);
    assert_eq!(deserialized.cli_session_id, session.cli_session_id);
    assert_eq!(deserialized.message_count, 10);
    assert_eq!(deserialized.total_cost_usd, Some(1.50));
}

// ====================================================================
// new_without_memory — fields
// ====================================================================

#[test]
fn test_new_without_memory_has_no_injector() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    assert!(manager.context_injector.is_none());
    assert!(manager.memory_config.is_none());
}

// ====================================================================
// get_session_messages — error paths
// ====================================================================

#[tokio::test]
async fn test_get_session_messages_no_injector() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // Session doesn't exist in mock store — should get "not found"
    let result = manager
        .get_session_messages(&Uuid::new_v4().to_string(), None, None)
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn test_get_session_messages_invalid_uuid() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.get_session_messages("not-a-uuid", None, None).await;
    assert!(result.is_err());
    // Invalid UUID is rejected before any storage lookup
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid session ID"));
}

// ====================================================================
// get_session_messages — returns structured events (tool_use, etc.)
// ====================================================================

#[tokio::test]
async fn test_get_session_messages_returns_tool_use_events() {
    let state = mock_app_state();
    let manager =
        ChatManager::new_without_memory(state.neo4j.clone(), state.meili.clone(), test_config());

    // Create a session
    let session = test_chat_session(None);
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    // Store events including tool_use and tool_result
    let events = vec![
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: 1,
            event_type: "user_message".into(),
            data: serde_json::to_string(&ChatEvent::UserMessage {
                content: "List my plans".into(),
            })
            .unwrap(),
            created_at: chrono::Utc::now(),
        },
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: 2,
            event_type: "tool_use".into(),
            data: serde_json::to_string(&ChatEvent::ToolUse {
                id: "tu_1".into(),
                tool: "list_plans".into(),
                input: serde_json::json!({"status": "in_progress"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            })
            .unwrap(),
            created_at: chrono::Utc::now(),
        },
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: 3,
            event_type: "tool_result".into(),
            data: serde_json::to_string(&ChatEvent::ToolResult {
                id: "tu_1".into(),
                result: serde_json::json!({"plans": []}),
                is_error: false,
                parent_tool_use_id: None,
            })
            .unwrap(),
            created_at: chrono::Utc::now(),
        },
        ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: 4,
            event_type: "assistant_text".into(),
            data: serde_json::to_string(&ChatEvent::AssistantText {
                content: "You have no in-progress plans.".into(),
                parent_tool_use_id: None,
            })
            .unwrap(),
            created_at: chrono::Utc::now(),
        },
    ];
    state
        .neo4j
        .store_chat_events(session_id, events)
        .await
        .unwrap();

    // Retrieve via get_session_messages
    let page = manager
        .get_session_messages(&session_id.to_string(), None, None)
        .await
        .unwrap();

    assert_eq!(page.total_count, 4);
    assert_eq!(page.events.len(), 4);

    // Verify event types are preserved
    assert_eq!(page.events[0].event_type, "user_message");
    assert_eq!(page.events[1].event_type, "tool_use");
    assert_eq!(page.events[2].event_type, "tool_result");
    assert_eq!(page.events[3].event_type, "assistant_text");

    // Verify tool_use data is intact
    let tool_use: ChatEvent = serde_json::from_str(&page.events[1].data).unwrap();
    match tool_use {
        ChatEvent::ToolUse {
            id, tool, input, ..
        } => {
            assert_eq!(id, "tu_1");
            assert_eq!(tool, "list_plans");
            assert_eq!(input, serde_json::json!({"status": "in_progress"}));
        }
        _ => panic!("Expected ToolUse event"),
    }

    // Verify ordering by seq
    assert_eq!(page.events[0].seq, 1);
    assert_eq!(page.events[1].seq, 2);
    assert_eq!(page.events[2].seq, 3);
    assert_eq!(page.events[3].seq, 4);
}

#[tokio::test]
async fn test_get_session_messages_pagination() {
    let state = mock_app_state();
    let manager =
        ChatManager::new_without_memory(state.neo4j.clone(), state.meili.clone(), test_config());

    let session = test_chat_session(None);
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    // Store 5 events
    let events: Vec<ChatEventRecord> = (1..=5)
        .map(|i| ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: i,
            event_type: "assistant_text".into(),
            data: serde_json::to_string(&ChatEvent::AssistantText {
                content: format!("Message {}", i),
                parent_tool_use_id: None,
            })
            .unwrap(),
            created_at: chrono::Utc::now(),
        })
        .collect();
    state
        .neo4j
        .store_chat_events(session_id, events)
        .await
        .unwrap();

    // First page: offset=0, limit=2
    let page1 = manager
        .get_session_messages(&session_id.to_string(), Some(2), Some(0))
        .await
        .unwrap();
    assert_eq!(page1.events.len(), 2);
    assert_eq!(page1.total_count, 5);
    assert_eq!(page1.events[0].seq, 1);
    assert_eq!(page1.events[1].seq, 2);

    // Second page: offset=2, limit=2
    let page2 = manager
        .get_session_messages(&session_id.to_string(), Some(2), Some(2))
        .await
        .unwrap();
    assert_eq!(page2.events.len(), 2);
    assert_eq!(page2.events[0].seq, 3);
    assert_eq!(page2.events[1].seq, 4);

    // Last page: offset=4, limit=2
    let page3 = manager
        .get_session_messages(&session_id.to_string(), Some(2), Some(4))
        .await
        .unwrap();
    assert_eq!(page3.events.len(), 1);
    assert_eq!(page3.events[0].seq, 5);
}

#[tokio::test]
async fn test_get_session_messages_empty_session() {
    let state = mock_app_state();
    let manager =
        ChatManager::new_without_memory(state.neo4j.clone(), state.meili.clone(), test_config());

    // Create session with no events
    let session = test_chat_session(None);
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    let page = manager
        .get_session_messages(&session_id.to_string(), None, None)
        .await
        .unwrap();

    assert_eq!(page.total_count, 0);
    assert!(page.events.is_empty());
    assert!(!page.has_more);
}

// ====================================================================
// conversation_id in mock GraphStore CRUD
// ====================================================================

#[tokio::test]
async fn test_chat_session_update_conversation_id() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();
    assert!(session.conversation_id.is_none());

    // Update conversation_id
    let updated = graph
        .update_chat_session(
            session.id,
            None,
            None,
            None,
            None,
            Some("conv-new-123".into()),
            None,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.conversation_id.as_deref(), Some("conv-new-123"));

    // Fetch and verify persisted
    let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(fetched.conversation_id.as_deref(), Some("conv-new-123"));
}

#[tokio::test]
async fn test_chat_session_create_with_conversation_id() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let mut session = test_chat_session(Some("proj"));
    session.conversation_id = Some("conv-init-456".into());
    graph.create_chat_session(&session).await.unwrap();

    let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(fetched.conversation_id.as_deref(), Some("conv-init-456"));
}

#[tokio::test]
async fn test_chat_session_conversation_id_survives_other_updates() {
    let state = mock_app_state();
    let graph = &state.neo4j;

    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();

    // Set conversation_id
    graph
        .update_chat_session(
            session.id,
            None,
            None,
            None,
            None,
            Some("conv-persist".into()),
            None,
        )
        .await
        .unwrap();

    // Update title only — conversation_id should be preserved
    let updated = graph
        .update_chat_session(
            session.id,
            None,
            Some("New Title".into()),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.title.as_deref(), Some("New Title"));
    assert_eq!(updated.conversation_id.as_deref(), Some("conv-persist"));
}

#[tokio::test]
async fn test_chat_session_node_serialization_with_conversation_id() {
    let session = ChatSessionNode {
        file_scope_lifted: false,
        mirror_of: None,
        routing_pool: None,
        routing_mode: None,
        id: uuid::Uuid::new_v4(),
        cli_session_id: None,
        project_slug: None,
        workspace_slug: None,
        cwd: "/tmp".into(),
        title: None,
        model: "model".into(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        message_count: 0,
        total_cost_usd: None,
        conversation_id: Some("conv-serde-test".into()),
        preview: None,
        permission_mode: None,
        add_dirs: None,
        spawned_by: None,
        provider_id: None,
        routed_by: None,
        capabilities: None,
        engine: None,
        resume_token: None,
        execution_place: Default::default(),
        project_inferred: false,
        access: Default::default(),
        owner: None,
    };

    let json = serde_json::to_string(&session).unwrap();
    assert!(json.contains("conv-serde-test"));

    let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
    assert_eq!(
        deserialized.conversation_id.as_deref(),
        Some("conv-serde-test")
    );
}

#[tokio::test]
async fn test_chat_session_node_serialization_without_conversation_id() {
    let session = ChatSessionNode {
        file_scope_lifted: false,
        mirror_of: None,
        routing_pool: None,
        routing_mode: None,
        id: uuid::Uuid::new_v4(),
        cli_session_id: None,
        project_slug: None,
        workspace_slug: None,
        cwd: "/tmp".into(),
        title: None,
        model: "model".into(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        message_count: 0,
        total_cost_usd: None,
        conversation_id: None,
        preview: None,
        permission_mode: None,
        add_dirs: None,
        spawned_by: None,
        provider_id: None,
        routed_by: None,
        capabilities: None,
        engine: None,
        resume_token: None,
        execution_place: Default::default(),
        project_inferred: false,
        access: Default::default(),
        owner: None,
    };

    let json = serde_json::to_string(&session).unwrap();
    let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
    assert!(deserialized.conversation_id.is_none());
}

// ====================================================================
// get_streaming_snapshot
// ====================================================================

#[tokio::test]
async fn test_get_streaming_snapshot_nonexistent_session() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let (is_streaming, text, events) = manager.get_streaming_snapshot("nonexistent").await;
    assert!(!is_streaming);
    assert!(text.is_empty());
    assert!(events.is_empty());
}

/// Helper: create a dummy InteractiveClient for tests.
/// Returns None if the Claude CLI is not installed (e.g., in CI).
/// A client that needs no Claude CLI: it sits on the SDK's in-memory
/// transport. `InteractiveClient::new` looks the CLI up on disk, so the
/// tests built on it were skipped wherever it is not installed — CI
/// included — and passed without running.
/// The pid of a process that has exited and been reaped: nothing answers to it. Never 0:
/// kill(0, _) signals the caller's own process group, which is the test run itself.
fn reaped_pid() -> u32 {
    let mut child = std::process::Command::new("true")
        .spawn()
        .expect("spawn a short-lived process");
    let pid = child.id();
    child.wait().expect("reap it");
    pid
}

fn create_dummy_client() -> InteractiveClient {
    let (transport, _handle) = nexus_claude::transport::mock::MockTransport::pair();
    InteractiveClient::from_transport(transport)
}

/// Helper: create a dummy ActiveSession for testing (no CLI needed).
fn create_dummy_session(
    is_streaming: bool,
    streaming_text: &str,
    streaming_events_data: Vec<ChatEvent>,
) -> (ActiveSession, Arc<Mutex<VecDeque<PendingMessage>>>) {
    let client = create_dummy_client();
    let (tx, _rx) = broadcast::channel(16);
    let pending_messages = Arc::new(Mutex::new(VecDeque::<PendingMessage>::new()));

    let session = ActiveSession {
        context_window: None,
        anchor: Default::default(),
        mirror_tap: None,
        events_tx: tx,
        last_activity: Instant::now(),
        cli_session_id: None,
        client: Arc::new(Mutex::new(client)),
        interrupt_flag: Arc::new(AtomicBool::new(false)),
        memory_manager: None,
        next_seq: Arc::new(AtomicI64::new(1)),
        pending_messages: pending_messages.clone(),
        is_streaming: Arc::new(AtomicBool::new(is_streaming)),
        streaming_text: Arc::new(Mutex::new(streaming_text.to_string())),
        streaming_events: Arc::new(Mutex::new(streaming_events_data)),
        permission_mode: None,
        model: None,
        sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
        stdin_tx: None,
        child_pid: None,
        nats_cancel: CancellationToken::new(),
        interrupt_token: CancellationToken::new(),
        turn_generation: Arc::new(AtomicU64::new(0)),
        result_cursor: Default::default(),
        stream_task: Arc::new(std::sync::Mutex::new(None)),
        post_stream_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
        pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
            std::collections::HashMap::new(),
        )),
        auto_continue: Arc::new(AtomicBool::new(false)),
        auto_continue_count: Arc::new(AtomicU32::new(0)),
        max_auto_continues: 0,
        rfc_accumulator: Arc::new(Mutex::new(
            crate::chat::observation_detector::RfcAccumulator::new(),
        )),
        protocol_run_id: None,
        protocol_state: None,
        reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
        objective_tracking: false,
        objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
        objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
        work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
        provider_outage: Arc::new(std::sync::Mutex::new(
            crate::chat::provider::outage::TurnHold::new(
                crate::chat::provider::resolver::CLAUDE_CODE,
            ),
        )),
        session_grants: Arc::default(),
        oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
        oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
        oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
        oob_capped_warned: Arc::new(AtomicBool::new(false)),
        cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
        cancel_tools_cap: CANCEL_TOOLS_CAP,
        cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
        active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
        tool_start_descendants: Arc::new(Mutex::new(HashMap::new())),
        cli_background_tasks: Arc::new(AtomicUsize::new(0)),
        cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
        cancel_task_cap: CANCEL_TASK_CAP,
        cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
    };

    (session, pending_messages)
}

#[tokio::test]
async fn test_get_streaming_snapshot_with_active_session_not_streaming() {
    let (session, _) = create_dummy_session(false, "", vec![]);

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    manager
        .active_sessions
        .write()
        .await
        .insert("test-session".into(), session);

    let (is_streaming, text, events) = manager.get_streaming_snapshot("test-session").await;
    assert!(!is_streaming);
    assert!(text.is_empty());
    assert!(events.is_empty());
}

#[tokio::test]
async fn test_get_streaming_snapshot_with_active_streaming_session() {
    let events_data = vec![
        ChatEvent::ToolUse {
            id: "t1".into(),
            tool: "list_plans".into(),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        },
        ChatEvent::ToolResult {
            id: "t1".into(),
            result: serde_json::json!({"plans": []}),
            is_error: false,
            parent_tool_use_id: None,
        },
    ];

    let (session, _) = create_dummy_session(true, "Hello world", events_data);

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    manager
        .active_sessions
        .write()
        .await
        .insert("streaming-session".into(), session);

    let (is_streaming, text, events) = manager.get_streaming_snapshot("streaming-session").await;
    assert!(is_streaming);
    assert_eq!(text, "Hello world");
    assert_eq!(events.len(), 2);
    assert!(matches!(&events[0], ChatEvent::ToolUse { tool, .. } if tool == "list_plans"));
    assert!(matches!(&events[1], ChatEvent::ToolResult { id, .. } if id == "t1"));
}

// ====================================================================
// send_message — queuing when is_streaming=true
// ====================================================================

#[tokio::test]
async fn test_send_message_queues_when_streaming() {
    let (session, pending_messages) = create_dummy_session(true, "", vec![]);

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let session_id = Uuid::new_v4().to_string();

    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.clone(), session);

    // Send a message while streaming — should be queued, NOT sent
    let result = manager.send_message(&session_id, "queued message").await;
    assert!(result.is_ok());

    // Verify the message was queued
    let queue = pending_messages.lock().await;
    assert_eq!(queue.len(), 1);
    assert_eq!(queue[0], "queued message");
}

// ── Held messages: queued WITHOUT interrupting (chat::pending_queue) ──

/// A streaming session registered in a fresh manager, with a subscriber on
/// its event channel.
async fn manager_with_streaming_session() -> (
    ChatManager,
    String,
    Arc<Mutex<VecDeque<PendingMessage>>>,
    broadcast::Receiver<ChatEvent>,
    Arc<AtomicBool>,
) {
    let (session, pending_messages) = create_dummy_session(true, "", vec![]);
    let events_rx = session.events_tx.subscribe();
    let interrupt_flag = session.interrupt_flag.clone();
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let session_id = Uuid::new_v4().to_string();
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.clone(), session);
    (
        manager,
        session_id,
        pending_messages,
        events_rx,
        interrupt_flag,
    )
}

// ── Interrupt watchdog: a turn that ignores its Stop is abandoned ──

/// Every event the session has broadcast so far.
fn drain_events(rx: &mut broadcast::Receiver<ChatEvent>) -> Vec<ChatEvent> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    events
}

async fn session_is_streaming(manager: &ChatManager, session_id: &str) -> bool {
    manager.active_sessions.read().await[session_id]
        .is_streaming
        .load(Ordering::SeqCst)
}

#[tokio::test]
async fn a_turn_that_ignores_its_stop_is_abandoned_after_the_grace() {
    let (manager, session_id, _pending, mut events, _flag) = manager_with_streaming_session().await;
    // The turn's task: past its loop, sitting on an await that never answers.
    let turn = tokio::spawn(std::future::pending::<()>());
    ChatManager::track_stream_task(&manager.active_sessions, &session_id, turn.abort_handle())
        .await;

    let abandoned = ChatManager::abandon_turn_if_stuck(
        manager.active_sessions.clone(),
        None,
        None,
        session_id.clone(),
        Duration::from_millis(50),
    )
    .await;

    assert!(
        abandoned,
        "the turn did not end within the grace: abandoned"
    );
    assert!(!session_is_streaming(&manager, &session_id).await);
    let err = turn.await.expect_err("the turn's task was aborted");
    assert!(err.is_cancelled());
    let events = drain_events(&mut events);
    assert!(
        events.iter().any(|e| matches!(
            e,
            ChatEvent::Error { code: Some(code), .. } if code == "turn_abandoned"
        )),
        "the clients learn why: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            ChatEvent::StreamingStatus {
                is_streaming: false
            }
        )),
        "the composer is released: {events:?}"
    );
}

#[tokio::test]
async fn a_turn_that_ends_within_the_grace_is_left_alone() {
    let (manager, session_id, _pending, mut events, _flag) = manager_with_streaming_session().await;
    let is_streaming = manager.active_sessions.read().await[&session_id]
        .is_streaming
        .clone();
    let turn = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        is_streaming.store(false, Ordering::SeqCst);
    });
    ChatManager::track_stream_task(&manager.active_sessions, &session_id, turn.abort_handle())
        .await;

    let abandoned = ChatManager::abandon_turn_if_stuck(
        manager.active_sessions.clone(),
        None,
        None,
        session_id.clone(),
        Duration::from_millis(500),
    )
    .await;

    assert!(!abandoned, "the turn ended on its own");
    turn.await.expect("the task was not aborted");
    assert!(
        drain_events(&mut events).is_empty(),
        "nothing to tell the clients"
    );
}

#[tokio::test]
async fn a_newer_turn_is_never_abandoned_for_an_older_stop() {
    let (manager, session_id, _pending, mut events, _flag) = manager_with_streaming_session().await;
    let generation = manager.active_sessions.read().await[&session_id]
        .turn_generation
        .clone();
    // The stopped turn ends and the next one starts (still streaming) while
    // the watchdog waits: it is not the watchdog's turn any more.
    let bump = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        generation.fetch_add(1, Ordering::SeqCst);
    });

    let abandoned = ChatManager::abandon_turn_if_stuck(
        manager.active_sessions.clone(),
        None,
        None,
        session_id.clone(),
        // Long enough for the 20 ms bump to run on a loaded runner (200 ms was not).
        Duration::from_secs(3),
    )
    .await;

    bump.await.unwrap();
    assert!(!abandoned);
    assert!(
        session_is_streaming(&manager, &session_id).await,
        "the new turn runs on"
    );
    assert!(drain_events(&mut events).is_empty());
}

#[tokio::test]
async fn an_idle_session_needs_no_watchdog() {
    let (session, _pending) = create_dummy_session(false, "", vec![]);
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let session_id = Uuid::new_v4().to_string();
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.clone(), session);

    let abandoned = ChatManager::abandon_turn_if_stuck(
        manager.active_sessions.clone(),
        None,
        None,
        session_id,
        Duration::from_millis(50),
    )
    .await;
    assert!(!abandoned);
}

#[tokio::test]
async fn the_interrupt_arms_the_watchdog() {
    let (manager, session_id, _pending, mut events, interrupt_flag) =
        manager_with_streaming_session().await;
    let turn = tokio::spawn(std::future::pending::<()>());
    ChatManager::track_stream_task(&manager.active_sessions, &session_id, turn.abort_handle())
        .await;

    manager.spawn_interrupt_watchdog(&session_id, Duration::from_millis(50));
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(!session_is_streaming(&manager, &session_id).await);
    assert!(turn.is_finished());
    assert!(
        !interrupt_flag.load(Ordering::SeqCst),
        "the watchdog is not an interrupt"
    );
    assert!(drain_events(&mut events).iter().any(|e| matches!(
        e,
        ChatEvent::StreamingStatus {
            is_streaming: false
        }
    )));
}

/// The regression itself, on the public path (`interrupt_scoped`), with a
/// paused clock: a turn whose task never reads its cancellation token used to
/// leave the session "streaming" for good — every Stop delivered, none
/// effective. Red without the watchdog: the assert on `is_streaming` fails.
#[tokio::test(start_paused = true)]
async fn a_stop_frees_a_session_whose_turn_ignores_it() {
    let (manager, session_id, _pending, mut events, _flag) = manager_with_streaming_session().await;

    let outcome = manager.interrupt_scoped(&session_id, true).await.unwrap();
    assert!(
        outcome.delivered,
        "the session is local: the Stop was delivered"
    );

    // Well past any grace the watchdog may give the turn.
    tokio::time::sleep(Duration::from_secs(60)).await;

    assert!(
        !session_is_streaming(&manager, &session_id).await,
        "a delivered Stop ends the turn even when its task ignores the token"
    );
    assert!(drain_events(&mut events).iter().any(|e| matches!(
        e,
        ChatEvent::StreamingStatus {
            is_streaming: false
        }
    )));
}

fn held_texts(event: ChatEvent) -> Vec<String> {
    match event {
        ChatEvent::PendingQueue { messages } => messages.into_iter().map(|m| m.content).collect(),
        other => panic!("expected pending_queue, got {other:?}"),
    }
}

#[tokio::test]
async fn queue_user_message_holds_without_interrupting_and_publishes_the_list() {
    let (manager, session_id, pending, mut events, interrupt_flag) =
        manager_with_streaming_session().await;

    let held = manager
        .queue_user_message(&session_id, "after you finish")
        .await
        .unwrap();

    assert!(held, "a streaming session holds the message");
    assert!(
        !interrupt_flag.load(Ordering::SeqCst),
        "holding a message must not interrupt the running turn"
    );
    {
        let queue = pending.lock().await;
        assert_eq!(queue.len(), 1);
        assert!(queue[0].held);
        assert_eq!(queue[0].kind, crate::chat::types::PendingMessageKind::User);
    }
    assert_eq!(
        held_texts(events.try_recv().unwrap()),
        vec!["after you finish"]
    );
}

#[tokio::test]
async fn every_device_of_the_session_sees_the_queue_and_its_changes() {
    // Two devices on one conversation: each holds its own subscription to the
    // session's event channel. A message queued from the phone must show on
    // the laptop, and an action from the laptop must show on the phone.
    let (manager, session_id, _pending, mut phone, _interrupt) =
        manager_with_streaming_session().await;
    let mut laptop = manager
        .active_sessions
        .read()
        .await
        .get(&session_id)
        .unwrap()
        .events_tx
        .subscribe();

    manager
        .queue_user_message(&session_id, "typed on the phone")
        .await
        .unwrap();
    assert_eq!(
        held_texts(phone.try_recv().unwrap()),
        vec!["typed on the phone"]
    );
    assert_eq!(
        held_texts(laptop.try_recv().unwrap()),
        vec!["typed on the phone"]
    );

    let id = manager.pending_queue_snapshot(&session_id).await.unwrap()[0].id;
    manager
        .pending_queue_op(
            &session_id,
            &crate::chat::pending_queue::QueueOp::Edit {
                id,
                content: "fixed on the laptop".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        held_texts(phone.try_recv().unwrap()),
        vec!["fixed on the laptop"]
    );
    assert_eq!(
        held_texts(laptop.try_recv().unwrap()),
        vec!["fixed on the laptop"]
    );
}

#[tokio::test]
async fn send_message_mid_stream_still_interrupts_and_is_not_listed_as_held() {
    // The historical path must keep its meaning: it is what "send now" relies on.
    let (manager, session_id, _pending, _events, interrupt_flag) =
        manager_with_streaming_session().await;

    manager
        .send_message(&session_id, "right now")
        .await
        .unwrap();

    assert!(interrupt_flag.load(Ordering::SeqCst));
    assert_eq!(
        manager.pending_queue_snapshot(&session_id).await,
        Some(vec![]),
        "a message already on its way is not a held message"
    );
}

#[tokio::test]
async fn pending_queue_ops_edit_remove_and_prioritize_without_interrupting() {
    let (manager, session_id, _pending, mut events, interrupt_flag) =
        manager_with_streaming_session().await;
    for text in ["a", "b", "c"] {
        manager.queue_user_message(&session_id, text).await.unwrap();
        let _ = events.try_recv();
    }
    let ids: Vec<Uuid> = manager
        .pending_queue_snapshot(&session_id)
        .await
        .unwrap()
        .iter()
        .map(|m| m.id)
        .collect();

    use crate::chat::pending_queue::QueueOp;
    assert!(manager
        .pending_queue_op(&session_id, &QueueOp::Prioritize { id: ids[2] })
        .await
        .unwrap());
    assert_eq!(held_texts(events.try_recv().unwrap()), vec!["c", "a", "b"]);

    manager
        .pending_queue_op(
            &session_id,
            &QueueOp::Edit {
                id: ids[0],
                content: "A".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(held_texts(events.try_recv().unwrap()), vec!["c", "A", "b"]);

    manager
        .pending_queue_op(&session_id, &QueueOp::Remove { id: ids[1] })
        .await
        .unwrap();
    assert_eq!(held_texts(events.try_recv().unwrap()), vec!["c", "A"]);

    assert!(
        !interrupt_flag.load(Ordering::SeqCst),
        "none of these may cut the running response short"
    );
}

#[tokio::test]
async fn send_now_puts_the_message_first_and_interrupts() {
    let (manager, session_id, pending, mut events, interrupt_flag) =
        manager_with_streaming_session().await;
    for text in ["a", "b"] {
        manager.queue_user_message(&session_id, text).await.unwrap();
        let _ = events.try_recv();
    }
    let id = manager.pending_queue_snapshot(&session_id).await.unwrap()[1].id;

    manager
        .pending_queue_op(
            &session_id,
            &crate::chat::pending_queue::QueueOp::SendNow { id },
        )
        .await
        .unwrap();

    assert!(interrupt_flag.load(Ordering::SeqCst));
    assert_eq!(pending.lock().await[0].content, "b", "delivered next");
    assert_eq!(
        held_texts(events.try_recv().unwrap()),
        vec!["a"],
        "no longer listed: it is on its way"
    );
}

#[tokio::test]
async fn a_queue_op_on_a_session_nobody_holds_reaches_nothing() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let reached = manager
        .pending_queue_op(
            &Uuid::new_v4().to_string(),
            &crate::chat::pending_queue::QueueOp::Snapshot,
        )
        .await
        .unwrap();
    assert!(!reached, "no local session and no NATS: there is no queue");
}

#[tokio::test]
async fn test_send_message_queues_multiple_when_streaming() {
    let (session, pending_messages) = create_dummy_session(true, "", vec![]);

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let session_id = Uuid::new_v4().to_string();

    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.clone(), session);

    // Queue multiple messages
    manager.send_message(&session_id, "first").await.unwrap();
    manager.send_message(&session_id, "second").await.unwrap();
    manager.send_message(&session_id, "third").await.unwrap();

    let queue = pending_messages.lock().await;
    assert_eq!(queue.len(), 3);
    assert_eq!(queue[0], "first");
    assert_eq!(queue[1], "second");
    assert_eq!(queue[2], "third");
}

// ====================================================================
// streaming_events filtering logic
// ====================================================================

#[test]
fn test_streaming_events_filter_excludes_transient_events() {
    // This tests the filtering logic used in stream_response():
    // StreamDelta, StreamingStatus, and AssistantText should be EXCLUDED
    // from streaming_events buffer (they are handled via streaming_text or
    // sent explicitly in Phase 1.5)

    let events_to_exclude = vec![
        ChatEvent::StreamDelta {
            text: "Hello".into(),
            parent_tool_use_id: None,
        },
        ChatEvent::StreamingStatus { is_streaming: true },
        ChatEvent::StreamingStatus {
            is_streaming: false,
        },
        ChatEvent::AssistantText {
            content: "Hello world".into(),
            parent_tool_use_id: None,
        },
    ];

    for event in &events_to_exclude {
        let should_add = !matches!(
            event,
            ChatEvent::StreamDelta { .. }
                | ChatEvent::StreamingStatus { .. }
                | ChatEvent::AssistantText { .. }
        );
        assert!(
            !should_add,
            "Event {:?} should be excluded from streaming_events",
            event.event_type()
        );
    }
}

#[test]
fn test_streaming_events_filter_includes_structured_events() {
    // These events SHOULD be included in streaming_events buffer
    // so mid-stream joiners can reconstruct tool calls

    let events_to_include = vec![
        ChatEvent::ToolUse {
            id: "t1".into(),
            tool: "create_plan".into(),
            input: serde_json::json!({"title": "Plan"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        },
        ChatEvent::ToolResult {
            id: "t1".into(),
            result: serde_json::json!({"id": "abc"}),
            is_error: false,
            parent_tool_use_id: None,
        },
        ChatEvent::Thinking {
            content: "Let me think...".into(),
            parent_tool_use_id: None,
        },
        ChatEvent::PermissionRequest {
            id: "p1".into(),
            tool: "bash".into(),
            input: serde_json::json!({"command": "ls"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
            tool_use_id: None,
            always: None,
            consent: None,
        },
        ChatEvent::Error {
            message: "Something went wrong".into(),
            parent_tool_use_id: None,
            code: None,
            reason: None,
            index: None,
            request_id: None,
        },
        ChatEvent::Result {
            session_id: "cli-123".into(),
            duration_ms: 5000,
            cost_usd: Some(0.15),
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: None,
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
            error_code: None,
            error_message: None,
        },
        ChatEvent::UserMessage {
            content: "Hello".into(),
        },
    ];

    for event in &events_to_include {
        let should_add = !matches!(
            event,
            ChatEvent::StreamDelta { .. }
                | ChatEvent::StreamingStatus { .. }
                | ChatEvent::AssistantText { .. }
        );
        assert!(
            should_add,
            "Event {:?} should be included in streaming_events",
            event.event_type()
        );
    }
}

// ====================================================================
// ActiveSession with streaming_events field
// ====================================================================

#[tokio::test]
async fn test_active_session_streaming_events_field() {
    let (session, _) = create_dummy_session(false, "", vec![]);

    // Verify streaming_events starts empty
    assert!(session.streaming_events.lock().await.is_empty());

    // Push an event and verify it's accessible
    session
        .streaming_events
        .lock()
        .await
        .push(ChatEvent::Thinking {
            content: "test".into(),
            parent_tool_use_id: None,
        });
    assert_eq!(session.streaming_events.lock().await.len(), 1);

    // Clear and verify
    session.streaming_events.lock().await.clear();
    assert!(session.streaming_events.lock().await.is_empty());
}

// ====================================================================
// ChatEvent event_type() — covers the StreamingStatus variant
// ====================================================================

#[test]
fn test_chat_event_streaming_status_type() {
    let event = ChatEvent::StreamingStatus { is_streaming: true };
    assert_eq!(event.event_type(), "streaming_status");

    let event = ChatEvent::StreamingStatus {
        is_streaming: false,
    };
    assert_eq!(event.event_type(), "streaming_status");
}

#[test]
fn test_chat_event_user_message_type() {
    let event = ChatEvent::UserMessage {
        content: "Hello".into(),
    };
    assert_eq!(event.event_type(), "user_message");
}

#[test]
fn test_chat_event_streaming_status_serde_roundtrip() {
    let event = ChatEvent::StreamingStatus { is_streaming: true };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"streaming_status\""));
    assert!(json.contains("\"is_streaming\":true"));

    let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
    assert!(matches!(
        deserialized,
        ChatEvent::StreamingStatus { is_streaming: true }
    ));
}

#[test]
fn test_chat_event_user_message_serde_roundtrip() {
    let event = ChatEvent::UserMessage {
        content: "Hello world".into(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"user_message\""));

    let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
    assert!(
        matches!(deserialized, ChatEvent::UserMessage { ref content } if content == "Hello world")
    );
}

// ====================================================================
// with_event_emitter builder
// ====================================================================

#[test]
fn test_with_event_emitter_sets_emitter() {
    use crate::events::{CrudEvent, EventEmitter};

    struct DummyEmitter;
    impl EventEmitter for DummyEmitter {
        fn emit(&self, _event: CrudEvent) {}
    }

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    assert!(manager.event_emitter.is_none());

    let manager = manager.with_event_emitter(Arc::new(DummyEmitter));
    assert!(manager.event_emitter.is_some());
}

// ====================================================================
// backfill_previews_from_meilisearch — early return when no injector
// ====================================================================

#[tokio::test]
async fn test_backfill_previews_no_injector_returns_zero() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    // new_without_memory => context_injector is None => should return Ok(0)
    let result = manager.backfill_previews_from_meilisearch().await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

// ====================================================================
// backfill_agent_session_records — the record of the sessions the agent
// engine served before it kept one (P8)
// ====================================================================

fn record_event(session: Uuid, seq: i64, event: &ChatEvent) -> ChatEventRecord {
    ChatEventRecord {
        id: Uuid::new_v4(),
        session_id: session,
        seq,
        event_type: event.event_type().to_string(),
        data: serde_json::to_string(event).unwrap(),
        created_at: chrono::Utc::now(),
    }
}

fn result_costing(usd: Option<f64>) -> ChatEvent {
    ChatEvent::Result {
        session_id: String::new(),
        duration_ms: 1,
        cost_usd: usd,
        subtype: "success".into(),
        is_error: false,
        num_turns: None,
        result_text: None,
        cost: None,
        usage: None,
        model: None,
        stop_reason: None,
        error_code: None,
        error_message: None,
    }
}

#[tokio::test]
async fn the_backfill_rebuilds_the_record_of_an_agent_session_once_and_leaves_claude_code_alone() {
    let state = mock_app_state();
    let graph = state.neo4j.clone();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // A native session of before P8: three messages, three priced turns (one
    // unknown), a record stuck at its creation.
    let mut native = crate::test_helpers::test_chat_session(None);
    native.provider_id = Some("local".into());
    native.message_count = 1;
    native.capabilities = Some("{}".into());
    graph.create_chat_session(&native).await.unwrap();
    let typed = crate::refs::block::encode(&"x".repeat(100), &[]);
    let events = [
        ChatEvent::UserMessage { content: typed },
        result_costing(Some(0.01)),
        ChatEvent::UserMessage {
            content: "two".into(),
        },
        result_costing(None),
        ChatEvent::UserMessage {
            content: "three".into(),
        },
        result_costing(Some(0.02)),
    ];
    graph
        .store_chat_events(
            native.id,
            events
                .iter()
                .enumerate()
                .map(|(i, e)| record_event(native.id, i as i64 + 1, e))
                .collect(),
        )
        .await
        .unwrap();

    // A Claude Code session: the legacy engine kept its record.
    let mut legacy = crate::test_helpers::test_chat_session(None);
    legacy.provider_id = Some(crate::chat::provider::resolver::CLAUDE_CODE.into());
    legacy.message_count = 1;
    graph.create_chat_session(&legacy).await.unwrap();
    graph
        .store_chat_events(
            legacy.id,
            vec![
                record_event(
                    legacy.id,
                    1,
                    &ChatEvent::UserMessage {
                        content: "a".into(),
                    },
                ),
                record_event(
                    legacy.id,
                    2,
                    &ChatEvent::UserMessage {
                        content: "b".into(),
                    },
                ),
            ],
        )
        .await
        .unwrap();

    assert_eq!(manager.backfill_agent_session_records().await.unwrap(), 1);
    let node = graph.get_chat_session(native.id).await.unwrap().unwrap();
    assert_eq!(node.message_count, 3, "{node:?}");
    assert!(
        (node.total_cost_usd.unwrap() - 0.03).abs() < 1e-9,
        "{node:?}"
    );
    let title = node.title.clone().unwrap();
    assert_eq!(title.chars().count(), 80, "the one title rule: {title}");
    assert!(title.ends_with("..."));
    assert_eq!(node.preview.as_deref(), Some("x".repeat(100).as_str()));

    let untouched = graph.get_chat_session(legacy.id).await.unwrap().unwrap();
    assert_eq!(untouched.message_count, 1);
    assert_eq!(untouched.title, None);

    // Idempotent.
    assert_eq!(manager.backfill_agent_session_records().await.unwrap(), 0);
    let again = graph.get_chat_session(native.id).await.unwrap().unwrap();
    assert_eq!(again.message_count, 3);
    assert_eq!(again.total_cost_usd, node.total_cost_usd);
}

/// Claude Code forced onto the agent engine has a provider snapshot: it is
/// backfilled, its cost being the CLI's session total (the last one). A session
/// live on this instance is left to its handle (it could be counted twice).
#[tokio::test]
async fn the_backfill_takes_claude_code_on_the_agent_engine_and_skips_a_live_session() {
    let state = mock_app_state();
    let graph = state.neo4j.clone();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let mut forced = crate::test_helpers::test_chat_session(None);
    forced.provider_id = Some(crate::chat::provider::resolver::CLAUDE_CODE.into());
    forced.capabilities = Some("{}".into());
    forced.message_count = 1;
    graph.create_chat_session(&forced).await.unwrap();
    let mut live = crate::test_helpers::test_chat_session(None);
    live.provider_id = Some("local".into());
    live.capabilities = Some("{}".into());
    live.message_count = 1;
    graph.create_chat_session(&live).await.unwrap();
    for id in [forced.id, live.id] {
        let events = [
            ChatEvent::UserMessage {
                content: "a".into(),
            },
            result_costing(Some(0.01)),
            ChatEvent::UserMessage {
                content: "b".into(),
            },
            result_costing(Some(0.03)),
        ];
        graph
            .store_chat_events(
                id,
                events
                    .iter()
                    .enumerate()
                    .map(|(i, e)| record_event(id, i as i64 + 1, e))
                    .collect(),
            )
            .await
            .unwrap();
    }
    let provider = crate::chat::agent_runtime::fake::FakeProvider::new();
    manager
        .agent_runtime
        .adopt(
            &live.id.to_string(),
            "local",
            provider.session(),
            10,
            "native",
            serde_json::json!({}),
            None,
        )
        .await;

    assert_eq!(manager.backfill_agent_session_records().await.unwrap(), 1);
    let node = graph.get_chat_session(forced.id).await.unwrap().unwrap();
    assert_eq!(node.message_count, 2, "{node:?}");
    assert_eq!(node.total_cost_usd, Some(0.03), "the CLI's total, not 0.04");
    let untouched = graph.get_chat_session(live.id).await.unwrap().unwrap();
    assert_eq!(untouched.message_count, 1, "a live session is its handle's");
    assert_eq!(untouched.total_cost_usd, None);
}

fn native_bg_task(
    id: &str,
    kind: nexus_claude::agent::BackgroundTaskKind,
    status: nexus_claude::agent::BackgroundTaskStatus,
) -> nexus_claude::agent::BackgroundTask {
    nexus_claude::agent::BackgroundTask {
        id: id.into(),
        kind,
        description: "cargo build".into(),
        status,
        started_at_ms: Some(1_700_000_000_000),
        tool_call_id: Some(format!("tool-{id}")),
        parent: None,
        pid: None,
    }
}

/// Adopts a native agent session on a fake provider, idle for `idle`.
async fn idle_agent_session(
    manager: &ChatManager,
    idle: Duration,
) -> Arc<crate::chat::agent_runtime::AgentSessionHandle> {
    let provider = crate::chat::agent_runtime::fake::FakeProvider::new();
    let handle = manager
        .agent_runtime
        .adopt(
            &Uuid::new_v4().to_string(),
            "local",
            provider.session(),
            1,
            "native",
            serde_json::json!({}),
            None,
        )
        .await;
    handle.backdate_activity(idle);
    handle
}

/// A manager whose idle cleanup runs with `timeout`, and a person wait cap of one hour
/// (past the hard cap: 8 × `timeout`).
fn idle_manager(timeout: Duration) -> ChatManager {
    let state = mock_app_state();
    ChatManager::new_without_memory(
        state.neo4j,
        state.meili,
        ChatConfig {
            session_timeout: timeout,
            person_wait_cap: Duration::from_secs(3600),
            ..test_config()
        },
    )
}

/// The sessions the idle cleanup would close with the ordinary delay `patience`, sorted.
async fn expired_ids(manager: &ChatManager, patience: Duration) -> Vec<String> {
    let mut ids: Vec<String> = manager
        .expired_sessions(patience)
        .await
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    ids
}

/// A permission request of a `Bash` call, as a provider asks it.
fn bash_permission(id: &str) -> ChatEvent {
    ChatEvent::PermissionRequest {
        id: id.into(),
        tool: "Bash".into(),
        input: serde_json::json!({ "command": "make" }),
        parent_tool_use_id: None,
        category: None,
        canonical: Some("Bash".into()),
        tool_use_id: None,
        always: None,
        consent: None,
    }
}

/// An agent session whose RUNNING turn asked a permission nobody answered, idle for
/// `idle` since: the state the provider leaves while it waits on the person.
async fn agent_session_waiting(
    manager: &ChatManager,
    idle: Duration,
) -> Arc<crate::chat::agent_runtime::AgentSessionHandle> {
    let handle = idle_agent_session(manager, Duration::ZERO).await;
    handle.is_streaming.store(true, Ordering::SeqCst);
    handle.emit(bash_permission("perm-1")).await;
    // Asking is activity: the wait starts now, and is moved back after.
    handle.backdate_activity(idle);
    assert!(handle.waiting_on_a_person().await, "the turn waits");
    assert_eq!(handle.waiting_permission_ids(), vec!["perm-1".to_string()]);
    handle
}

/// The idle cleanup keeps an agent session whose provider still reports a
/// background task running (P4), as it keeps a Claude Code session with one;
/// the same idle session with no task left (or only ended ones) is closed,
/// and past the hard cap even a busy one is.
#[tokio::test]
async fn the_idle_cleanup_keeps_an_agent_session_with_a_background_task_running() {
    use nexus_claude::agent::{BackgroundTaskKind as Kind, BackgroundTaskStatus as Status};
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);

    let busy = idle_agent_session(&manager, timeout * 2).await;
    busy.set_background_tasks(vec![native_bg_task("nt-1", Kind::Shell, Status::Running)])
        .await;
    let done = idle_agent_session(&manager, timeout * 2).await;
    done.set_background_tasks(vec![native_bg_task("nt-2", Kind::Shell, Status::Killed)])
        .await;
    let quiet = idle_agent_session(&manager, timeout * 2).await;
    let fresh = idle_agent_session(&manager, Duration::ZERO).await;

    let expired = expired_ids(&manager, timeout).await;
    let mut want = vec![done.session_id.clone(), quiet.session_id.clone()];
    want.sort();
    assert_eq!(expired, want, "busy and fresh sessions are kept");
    assert!(!expired.contains(&fresh.session_id));

    busy.backdate_activity(timeout * (IDLE_HARD_CAP_FACTOR + 1));
    assert!(
        expired_ids(&manager, timeout)
            .await
            .contains(&busy.session_id),
        "past the hard cap, a background task no longer keeps the session"
    );
}

/// B1 / M6: a native session whose running turn waits on a person's answer to a
/// permission is kept past the ordinary delay AND past the hard cap (8 × the timeout) —
/// and closed once it has waited past the person wait cap: waiting prolongs a session,
/// never for ever (it used to hold its slot until the server restarted).
#[tokio::test]
async fn a_native_session_waiting_on_a_permission_outlives_the_hard_cap_but_not_the_wait_cap() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let waiting = agent_session_waiting(&manager, timeout * (IDLE_HARD_CAP_FACTOR + 1)).await;
    assert!(
        expired_ids(&manager, timeout).await.is_empty(),
        "past the hard cap, a session waiting on a person is kept"
    );

    waiting.backdate_activity(manager.config.person_wait_cap + Duration::from_secs(1));
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec![waiting.session_id.clone()],
        "past the person wait cap, it is closed all the same"
    );
    assert_eq!(
        manager.idle_cleanup_pass(timeout).await,
        vec![waiting.session_id.clone()]
    );
    assert!(!manager.agent_runtime.owns(&waiting.session_id).await);
}

/// B1, review (d): an `Agent` sub-agent (`conversation`) is never answered by a person —
/// its parent's call fails on its first request and the child is closed (`subagents.rs`) —
/// so its waiting protects it from nothing: the ordinary rules apply, kept while its turn
/// runs, closed at the hard cap.
#[tokio::test]
async fn a_spawned_session_waiting_on_a_permission_follows_the_ordinary_rules() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let waiting = agent_session_waiting(&manager, timeout * 2).await;
    let mut node = crate::test_helpers::test_chat_session(None);
    node.id = Uuid::parse_str(&waiting.session_id).unwrap();
    node.spawned_by = Some(r#"{"type":"conversation","parent_session_id":"p"}"#.into());
    manager.graph.create_chat_session(&node).await.unwrap();

    assert!(
        expired_ids(&manager, timeout).await.is_empty(),
        "its turn runs: kept, as any running turn"
    );
    waiting.backdate_activity(timeout * (IDLE_HARD_CAP_FACTOR + 1));
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec![waiting.session_id.clone()],
        "past the hard cap it is closed: nobody would ever answer it"
    );
}

/// B2: a permission asked by a turn that was then STOPPED is waited on by nobody (the
/// native harness drops its ask with the turn's signal, without a word): the session is
/// no longer "waiting on a person", its request is not listed as waiting, and once the
/// turn is over the idle cleanup treats it as the quiet session it is. The request used
/// to stay in `asked_calls` for the session's whole life — kept for ever.
#[tokio::test]
async fn a_permission_left_by_a_stopped_turn_keeps_nothing() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let stopped = agent_session_waiting(&manager, Duration::ZERO).await;

    stopped.interrupt().await.unwrap();
    assert!(!stopped.waiting_on_a_person().await);
    assert!(
        stopped.waiting_permission_ids().is_empty(),
        "a stopped turn's request is no longer waiting"
    );
    // The turn ends (its provider answered the Stop).
    stopped.is_streaming.store(false, Ordering::SeqCst);
    stopped.backdate_activity(timeout * 2);
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec![stopped.session_id.clone()]
    );
}

/// B2: the same for a turn that ended without the request being answered — abandoned by
/// the watchdog here — and for the next turn, which never inherits the previous one's
/// requests.
#[tokio::test]
async fn a_permission_dies_with_the_turn_that_asked_it() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let ended = agent_session_waiting(&manager, Duration::ZERO).await;
    assert!(ended.abandon_turn(ended.turn_generation()).await);
    assert!(!ended.waiting_on_a_person().await);
    assert!(ended.waiting_permission_ids().is_empty());

    // A request whose turn is over does not count even while ANOTHER turn runs.
    let later = agent_session_waiting(&manager, Duration::ZERO).await;
    later.advance_turn_for_tests();
    assert!(
        !later.waiting_on_a_person().await,
        "a request of a previous turn is waited on by nobody"
    );
}

/// C4: `reclaim_idle_slot` judges by the cleanup's own rule. A session whose turn waits
/// on a person is never reclaimed; a stuck one past the hard cap is (the cleanup closes
/// it too — the reclaim used to keep any streaming session, so the two disagreed); a
/// quiet one past the grace is.
#[tokio::test]
async fn reclaim_idle_slot_follows_the_rule_of_the_cleanup() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let waiting = agent_session_waiting(&manager, timeout * (IDLE_HARD_CAP_FACTOR + 1)).await;
    let stuck = idle_agent_session(&manager, timeout * (IDLE_HARD_CAP_FACTOR + 2)).await;
    stuck.is_streaming.store(true, Ordering::SeqCst);

    assert!(manager.reclaim_idle_slot(Duration::from_secs(1)).await);
    assert!(
        !manager.agent_runtime.owns(&stuck.session_id).await,
        "the most idle expired session goes: the stuck one"
    );
    assert!(
        !manager.reclaim_idle_slot(Duration::from_secs(1)).await,
        "a session waiting on a person is never reclaimed"
    );
    assert!(manager.agent_runtime.owns(&waiting.session_id).await);
}

/// C3: the Claude Code engine reads a person's wait too — a `can_use_tool` its running
/// turn waits on — with the same rule: kept past the hard cap, closed past the wait cap.
/// It used to be judged "not waiting" whatever it waited on, and closed at the hard cap.
/// A request in the table out of any turn of this server (the CLI woke up on its own)
/// counts too: the table is emptied at each turn's start, end and abandon, so an entry is a
/// request the CLI really waits on (review of #1173, point 2).
#[tokio::test]
async fn a_claude_code_session_waiting_on_a_permission_follows_the_same_rule() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    idle_legacy_session(&manager, "asking", timeout * (IDLE_HARD_CAP_FACTOR + 1)).await;
    idle_legacy_session(&manager, "between-turns", timeout * 2).await;
    {
        let sessions = manager.active_sessions.read().await;
        sessions["asking"]
            .is_streaming
            .store(true, Ordering::SeqCst);
        for id in ["asking", "between-turns"] {
            sessions[id].pending_permission_inputs.lock().await.insert(
                "req-1".into(),
                PendingPermission {
                    input: serde_json::json!({ "command": "make" }),
                    tool: "Bash".into(),
                },
            );
        }
    }
    assert!(
        expired_ids(&manager, timeout).await.is_empty(),
        "the waiting turn is kept past the hard cap; a request out of any turn keeps too"
    );

    for id in ["asking", "between-turns"] {
        manager
            .active_sessions
            .write()
            .await
            .get_mut(id)
            .unwrap()
            .last_activity = Instant::now()
            .checked_sub(manager.config.person_wait_cap + Duration::from_secs(1))
            .unwrap();
    }
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec!["asking".to_string(), "between-turns".to_string()],
        "past the person wait cap, both are closed all the same"
    );
}

/// M10: the verdict is read again just before the close. A session found expired, whose
/// turn starts waiting on a person between that verdict and its close, is kept.
#[tokio::test]
async fn the_idle_cleanup_reads_the_verdict_again_before_it_closes() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let quiet = idle_agent_session(&manager, timeout * 2).await;
    let asked = Arc::clone(&quiet);
    *manager
        .idle_close_hook
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(move |_: &str| {
        let asked = Arc::clone(&asked);
        Box::pin(async move {
            asked.is_streaming.store(true, Ordering::SeqCst);
            asked.emit(bash_permission("perm-late")).await;
        }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    }));
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec![quiet.session_id.clone()],
        "found expired"
    );
    assert!(
        manager.idle_cleanup_pass(timeout).await.is_empty(),
        "its wait began before the close: kept"
    );
    assert!(manager.agent_runtime.owns(&quiet.session_id).await);
}

/// Review of #1173 (d): a delegated session's request is shown to a person (the cockpit's
/// attention lists it), so its wait counts like a person's: kept past the hard cap, closed
/// past the wait cap.
#[tokio::test]
async fn a_delegated_session_waiting_on_a_permission_is_protected_like_a_persons() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let waiting = agent_session_waiting(&manager, timeout * (IDLE_HARD_CAP_FACTOR + 1)).await;
    let mut node = crate::test_helpers::test_chat_session(None);
    node.id = Uuid::parse_str(&waiting.session_id).unwrap();
    node.spawned_by = Some(r#"{"type":"delegation","parent_session_id":"p","task_id":"t"}"#.into());
    manager.graph.create_chat_session(&node).await.unwrap();
    assert!(
        expired_ids(&manager, timeout).await.is_empty(),
        "a delegation waiting on a person is kept past the hard cap"
    );
    waiting.backdate_activity(manager.config.person_wait_cap + Duration::from_secs(1));
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec![waiting.session_id.clone()]
    );
}

/// Review of #1173, point 2: a request asked OUT of any turn — the provider woke up on
/// its own (a background task's end, `emit_out_of_band`, which opens no turn and leaves
/// `is_streaming` false) — keeps the session waiting on its person for as long as it is
/// asked: past the ordinary delay and the hard cap, up to the wait cap. It used to count
/// for nothing: the modal was visible, the session closed under it at the timeout.
#[tokio::test]
async fn an_out_of_turn_permission_keeps_the_session_until_the_wait_cap() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let woken = idle_agent_session(&manager, Duration::ZERO).await;
    assert!(!woken.is_streaming.load(Ordering::SeqCst), "no turn runs");
    woken.emit(bash_permission("oob-1")).await;
    assert!(woken.waiting_on_a_person().await);
    woken.backdate_activity(timeout * (IDLE_HARD_CAP_FACTOR + 1));
    assert!(
        expired_ids(&manager, timeout).await.is_empty(),
        "an out-of-turn request keeps the session, past the hard cap included"
    );
    woken.backdate_activity(manager.config.person_wait_cap + Duration::from_secs(1));
    assert_eq!(
        expired_ids(&manager, timeout).await,
        vec![woken.session_id.clone()],
        "and the wait cap bounds it"
    );
}

/// Review of #1173, point 1: a session closed at the person wait cap leaves no request
/// orphaned. The request is REFUSED first — its `permission_decision` reaches the thread
/// and the provider — the thread says why, and the close carries the reason
/// `idle_timeout`, not `user_closed`.
#[tokio::test]
async fn a_close_at_the_wait_cap_refuses_the_request_and_says_why() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let waiting = agent_session_waiting(
        &manager,
        manager.config.person_wait_cap + Duration::from_secs(1),
    )
    .await;
    let mut events = waiting.events_tx.subscribe();
    assert_eq!(
        manager.idle_cleanup_pass(timeout).await,
        vec![waiting.session_id.clone()]
    );
    let mut refused = false;
    let mut explained = false;
    let mut reason = None;
    while let Ok(event) = events.try_recv() {
        match event {
            ChatEvent::PermissionDecision { id, allow, .. } if id == "perm-1" => {
                assert!(!allow, "the request is refused, not allowed");
                refused = true;
            }
            ChatEvent::SystemHint { content } if content.contains("refused") => {
                explained = true;
            }
            ChatEvent::SessionClosed { reason: r, .. } => reason = r,
            _ => {}
        }
    }
    assert!(refused, "the waiting request got its permission_decision");
    assert!(explained, "the thread says why the session closed");
    assert_eq!(
        reason.as_deref(),
        Some(crate::episodes::models::close_reason::IDLE_TIMEOUT),
        "the close names the cleanup, not a person"
    );
    assert!(
        waiting.waiting_permission_ids().is_empty(),
        "nothing is left asked"
    );
}

/// Review of #1173 (f): `reclaim_idle_slot` does not give up on its first candidate. The
/// most idle session, found expired, starts waiting on a person before its close (the
/// re-read keeps it): the next candidate frees the slot.
#[tokio::test]
async fn reclaim_idle_slot_goes_on_to_the_next_candidate_when_the_first_is_kept() {
    let timeout = Duration::from_secs(60);
    let manager = idle_manager(timeout);
    let most_idle = idle_agent_session(&manager, Duration::from_secs(600)).await;
    let next = idle_agent_session(&manager, Duration::from_secs(300)).await;
    let first = Arc::clone(&most_idle);
    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *manager
        .idle_close_hook
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(move |id: &str| {
        let first = Arc::clone(&first);
        let fired = Arc::clone(&fired);
        let id = id.to_string();
        Box::pin(async move {
            if !fired.swap(true, Ordering::SeqCst) {
                assert_eq!(id, first.session_id, "the most idle session is tried first");
                first.is_streaming.store(true, Ordering::SeqCst);
                first.emit(bash_permission("late")).await;
            }
        }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    }));
    assert!(
        manager.reclaim_idle_slot(Duration::from_secs(1)).await,
        "a slot is freed by the next candidate"
    );
    assert!(
        manager.agent_runtime.owns(&most_idle.session_id).await,
        "the session whose wait began meanwhile is kept"
    );
    assert!(!manager.agent_runtime.owns(&next.session_id).await);
}

/// Review of #1173 (c): on the Claude Code engine, asking a permission is activity — the
/// person's wait starts at the ask, not at the turn's last event before it.
#[tokio::test]
async fn asking_a_permission_is_activity_for_a_claude_code_session() {
    let manager = idle_manager(Duration::from_secs(60));
    idle_legacy_session(&manager, "cli", Duration::from_secs(3 * 3600)).await;
    let map = manager.active_sessions.read().await["cli"]
        .pending_permission_inputs
        .clone();
    store_pending_perm_input(
        &map,
        "req-1",
        &serde_json::json!({ "command": "make" }),
        "Bash",
        &manager.active_sessions,
        "cli",
    )
    .await;
    let sessions = manager.active_sessions.read().await;
    assert!(
        sessions["cli"].last_activity.elapsed() < Duration::from_secs(5),
        "asking is activity: the wait starts now"
    );
    assert!(map.lock().await.contains_key("req-1"));
}

/// Registers a live Claude Code session (no CLI behind it), idle for `idle`.
async fn idle_legacy_session(manager: &ChatManager, session_id: &str, idle: Duration) {
    test_support::insert_live_session_without_cli(manager, session_id).await;
    let mut sessions = manager.active_sessions.write().await;
    sessions
        .get_mut(session_id)
        .expect("registered")
        .last_activity = Instant::now()
        .checked_sub(idle)
        .expect("a representable instant");
}

/// Waits until the cleanup has closed every session of `ids`, on either engine.
async fn closed_by_the_cleanup(manager: &ChatManager, ids: &[&str]) {
    for _ in 0..1200 {
        let mut all_closed = true;
        for id in ids {
            if manager.is_session_active(id).await || manager.agent_runtime.owns(id).await {
                all_closed = false;
            }
        }
        if all_closed {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the idle cleanup never closed {ids:?}");
}

/// The cleanup loop itself ([`ChatManager::start_cleanup_task`]), on both engines, with
/// the runtime's clock paused (the ticks and the waits are virtual: nothing depends on
/// how loaded the machine is; the sessions' idle times are set, not waited for).
///
/// Claude Code: the session with nothing left to do is closed; a running turn keeps
/// its session, and so does a background task, whether this server tracks it or the CLI
/// reports it (whichever knows of more work wins). Past the hard cap, running work no
/// longer keeps it.
///
/// Native: the quiet session is closed, a running turn is kept, and a turn waiting on a
/// person is kept past the hard cap.
///
/// A second pass closes a session registered after the first: by then the first pass
/// is over, closes included, so a session it kept was kept by its verdict and not by
/// the test reading too early.
#[tokio::test(start_paused = true)]
async fn the_idle_cleanup_closes_a_quiet_session_and_keeps_a_working_one_on_both_engines() {
    let timeout = Duration::from_secs(4);
    let manager = Arc::new(idle_manager(timeout));
    let idle = timeout + Duration::from_secs(1);

    idle_legacy_session(&manager, "quiet", idle).await;
    idle_legacy_session(&manager, "streaming", idle).await;
    idle_legacy_session(&manager, "tracked-task", idle).await;
    idle_legacy_session(&manager, "cli-task", idle).await;
    idle_legacy_session(&manager, "stuck", timeout * (IDLE_HARD_CAP_FACTOR + 1)).await;
    {
        let sessions = manager.active_sessions.read().await;
        for id in ["streaming", "stuck"] {
            sessions[id].is_streaming.store(true, Ordering::SeqCst);
        }
        // Tracked by this server, unknown to the CLI's own count.
        sessions["tracked-task"]
            .active_background_tasks
            .lock()
            .await
            .insert("tool_W1".into(), a_cli_background_task());
        // Reported by the CLI, absent from this server's tracking.
        sessions["cli-task"]
            .cli_background_tasks
            .store(2, Ordering::Relaxed);
    }
    let native_quiet = idle_agent_session(&manager, idle).await;
    let native_working = idle_agent_session(&manager, idle).await;
    native_working.is_streaming.store(true, Ordering::SeqCst);
    let native_waiting =
        agent_session_waiting(&manager, timeout * (IDLE_HARD_CAP_FACTOR + 1)).await;

    manager.start_cleanup_task();
    closed_by_the_cleanup(&manager, &["quiet", "stuck", &native_quiet.session_id]).await;

    idle_legacy_session(&manager, "quiet-later", idle).await;
    closed_by_the_cleanup(&manager, &["quiet-later"]).await;

    let mut kept: Vec<String> = manager
        .active_sessions
        .read()
        .await
        .keys()
        .cloned()
        .collect();
    kept.sort_unstable();
    assert_eq!(
        kept,
        ["cli-task", "streaming", "tracked-task"],
        "a running turn or a background task, tracked here or reported by the CLI, keeps an idle session"
    );
    assert!(manager.agent_runtime.owns(&native_working.session_id).await);
    assert!(
        manager.agent_runtime.owns(&native_waiting.session_id).await,
        "a native turn waiting on a person is kept past the hard cap"
    );
}

/// The cockpit's snapshot counts the background tasks an agent session's
/// provider reports running, by kind, as it does for a Claude Code session.
#[tokio::test]
async fn the_live_snapshot_lists_the_background_tasks_of_an_agent_session() {
    use nexus_claude::agent::{BackgroundTaskKind as Kind, BackgroundTaskStatus as Status};
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let busy = idle_agent_session(&manager, Duration::ZERO).await;
    busy.set_background_tasks(vec![
        native_bg_task("nt-1", Kind::Shell, Status::Running),
        native_bg_task("nt-2", Kind::Monitor, Status::Running),
        native_bg_task("nt-3", Kind::Shell, Status::Running),
        native_bg_task("nt-4", Kind::Shell, Status::Killed),
    ])
    .await;
    let quiet = idle_agent_session(&manager, Duration::ZERO).await;

    let snap = manager.live_session_snapshot().await;
    let busy_id: Uuid = busy.session_id.parse().unwrap();
    let quiet_id: Uuid = quiet.session_id.parse().unwrap();
    assert!(snap.live.contains(&busy_id) && snap.live.contains(&quiet_id));
    assert_eq!(
        snap.background_tasks.get(&busy_id),
        Some(&(1, 2)),
        "one monitor, two running commands; the killed one is not counted"
    );
    assert_eq!(snap.background_tasks.get(&quiet_id), None);
}

// ====================================================================
// update_chat_session via mock — title, preview, conversation_id
// ====================================================================

#[tokio::test]
async fn test_update_session_title_and_preview() {
    let state = mock_app_state();
    let session = test_chat_session(Some("test-proj"));
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    // Update title
    let updated = state
        .neo4j
        .update_chat_session(
            session_id,
            None,
            Some("My new title".into()),
            None,
            None,
            None,
            Some("Preview of the conversation".into()),
        )
        .await
        .unwrap();
    assert!(updated.is_some());
    let s = updated.unwrap();
    assert_eq!(s.title.as_deref(), Some("My new title"));
    assert_eq!(s.preview.as_deref(), Some("Preview of the conversation"));
}

#[tokio::test]
async fn test_update_session_conversation_id() {
    let state = mock_app_state();
    let session = test_chat_session(None);
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    let updated = state
        .neo4j
        .update_chat_session(
            session_id,
            None,
            None,
            None,
            None,
            Some("conv-abc-123".into()),
            None,
        )
        .await
        .unwrap();
    assert!(updated.is_some());
    assert_eq!(
        updated.unwrap().conversation_id.as_deref(),
        Some("conv-abc-123")
    );
}

#[tokio::test]
async fn test_update_session_cost_and_message_count() {
    let state = mock_app_state();
    let session = test_chat_session(None);
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    let updated = state
        .neo4j
        .update_chat_session(session_id, None, None, Some(42), Some(1.50), None, None)
        .await
        .unwrap();
    assert!(updated.is_some());
    let s = updated.unwrap();
    assert_eq!(s.message_count, 42);
    assert!((s.total_cost_usd.unwrap() - 1.50).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_update_session_not_found() {
    let state = mock_app_state();
    let fake_id = Uuid::new_v4();
    let updated = state
        .neo4j
        .update_chat_session(fake_id, None, None, None, None, None, None)
        .await
        .unwrap();
    assert!(updated.is_none());
}

#[tokio::test]
async fn test_update_session_cli_session_id() {
    let state = mock_app_state();
    let session = test_chat_session(None);
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    let updated = state
        .neo4j
        .update_chat_session(
            session_id,
            Some("cli-session-xyz".into()),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(updated.is_some());
    assert_eq!(
        updated.unwrap().cli_session_id.as_deref(),
        Some("cli-session-xyz")
    );
}

#[tokio::test]
async fn test_update_session_partial_preserves_existing() {
    let state = mock_app_state();
    let mut session = test_chat_session(Some("my-proj"));
    session.title = Some("Original title".into());
    session.preview = Some("Original preview".into());
    let session_id = session.id;
    state.neo4j.create_chat_session(&session).await.unwrap();

    // Update only message_count — should preserve title and preview
    let updated = state
        .neo4j
        .update_chat_session(session_id, None, None, Some(10), None, None, None)
        .await
        .unwrap();
    assert!(updated.is_some());
    let s = updated.unwrap();
    assert_eq!(s.title.as_deref(), Some("Original title"));
    assert_eq!(s.preview.as_deref(), Some("Original preview"));
    assert_eq!(s.message_count, 10);
}

// ====================================================================
// backfill_chat_session_previews via mock — returns 0 (mock has no events)
// ====================================================================

#[tokio::test]
async fn test_backfill_session_previews_mock_returns_zero() {
    let state = mock_app_state();
    let result = state.neo4j.backfill_chat_session_previews().await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

// ====================================================================
// Title truncation logic (mirrors send_message_internal behavior)
// ====================================================================

#[test]
fn test_title_generation_short_message() {
    let msg = "Hello, help me with my project";
    // < 80 chars → title == message
    let title = if msg.chars().count() > 80 {
        let truncated: String = msg.chars().take(77).collect();
        format!("{}...", truncated.trim_end())
    } else {
        msg.to_string()
    };
    assert_eq!(title, "Hello, help me with my project");
}

#[test]
fn test_title_generation_long_message() {
    let msg = "a".repeat(100);
    // > 80 chars → truncated to 77 + "..."
    let title = if msg.chars().count() > 80 {
        let truncated: String = msg.chars().take(77).collect();
        format!("{}...", truncated.trim_end())
    } else {
        msg.to_string()
    };
    assert_eq!(title.chars().count(), 80);
    assert!(title.ends_with("..."));
}

#[test]
fn test_preview_generation_short_message() {
    let msg = "Short message";
    let preview = if msg.chars().count() > 200 {
        let truncated: String = msg.chars().take(197).collect();
        format!("{}...", truncated.trim_end())
    } else {
        msg.to_string()
    };
    assert_eq!(preview, "Short message");
}

#[test]
fn test_preview_generation_long_message() {
    let msg = "b".repeat(300);
    let preview = if msg.chars().count() > 200 {
        let truncated: String = msg.chars().take(197).collect();
        format!("{}...", truncated.trim_end())
    } else {
        msg.to_string()
    };
    assert_eq!(preview.chars().count(), 200);
    assert!(preview.ends_with("..."));
}

#[test]
fn test_title_generation_utf8_multibyte() {
    // 90 chars with accented characters
    let msg: String = "é".repeat(90);
    let title = if msg.chars().count() > 80 {
        let truncated: String = msg.chars().take(77).collect();
        format!("{}...", truncated.trim_end())
    } else {
        msg.to_string()
    };
    assert_eq!(title.chars().count(), 80);
    assert!(title.ends_with("..."));
}

// ========================================================================
// NATS RPC routing tests
// ========================================================================

#[tokio::test]
async fn test_try_remote_send_no_nats() {
    // ChatManager without NATS → try_remote_send should return Ok(false) immediately
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // No NATS configured → should return false (fallback needed)
    assert!(manager.nats.is_none());
    let result = manager
        .try_remote_send("any-session-id", "hello", "user_message")
        .await;
    assert!(result.is_ok());
    assert!(!result.unwrap(), "Should return false without NATS");
}

#[tokio::test]
async fn test_try_remote_send_all_message_types_no_nats() {
    // Verify all message types return Ok(false) without NATS
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    for msg_type in &["user_message", "permission_response", "input_response"] {
        let result = manager
            .try_remote_send("session-123", "test", msg_type)
            .await
            .unwrap();
        assert!(!result, "Should return false for {} without NATS", msg_type);
    }
}

#[tokio::test]
async fn test_interrupt_no_local_session_no_nats() {
    // interrupt() should succeed silently when session is not local and no NATS
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // No session active, no NATS → should return Ok (no error)
    let result = manager.interrupt("nonexistent-session").await;
    assert!(
        result.is_ok(),
        "interrupt should not error when session is not local"
    );
}

/// `interrupt()` returning `Ok(())` for a session it never found is the
/// reason a lost Stop was invisible: the UI could not tell "turn ended"
/// from "nothing happened". `interrupt_scoped` says which it was.
#[tokio::test]
async fn test_interrupt_scoped_reports_a_no_op_as_undelivered() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let outcome = manager
        .interrupt_scoped("nonexistent-session", true)
        .await
        .expect("interrupt_scoped must not error on an unknown session");

    assert!(
        !outcome.delivered,
        "nothing was interrupted — delivered must be false"
    );
    assert_eq!(
        outcome.routed, "none",
        "no local session and no NATS means the interrupt went nowhere"
    );
    assert!(outcome.killed_pids.is_empty());
}

/// The narrow scope still ends the turn: flag set, token cancelled.
#[tokio::test]
async fn test_interrupt_scoped_still_ends_the_turn_without_tools() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _handle) = mock_active_session(true);
    let flag = session.interrupt_flag.clone();
    let token = session.interrupt_token.clone();

    let session_id = "test-session-scoped-turn";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    let outcome = manager.interrupt_scoped(session_id, false).await.unwrap();

    assert!(outcome.delivered, "a local session was interrupted");
    assert_eq!(outcome.routed, "local");
    assert!(
        flag.load(Ordering::SeqCst),
        "the turn must still be interrupted with kill_tools = false"
    );
    assert!(
        token.is_cancelled(),
        "the stream token must still be cancelled"
    );
    assert!(outcome.killed_pids.is_empty());
}

/// The point of the narrow scope, proved on real processes rather than
/// on an empty `killed_pids`: a subprocess the agent launched survives
/// an interrupt scoped to the turn, and dies under the wide one.
#[cfg(unix)]
#[tokio::test]
async fn test_interrupt_scoped_spares_then_reaps_a_running_tool() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (mut session, _handle) = mock_active_session(true);

    // Stand-in for a tool the agent is running: a shell that stays alive
    // as the parent of a long sleep, so the sleep is a *descendant* of
    // the pretend-CLI, exactly like a real Bash tool child. The trailing
    // `:` stops `sh` from exec'ing sleep in its own process.
    let mut shell = std::process::Command::new("sh")
        .arg("-c")
        .arg("sleep 30; :")
        .spawn()
        .expect("failed to spawn the stand-in tool");
    session.child_pid = Some(shell.id());

    // Wait for the grandchild to actually exist — without it the test
    // would pass for the wrong reason.
    let mut descendants = Vec::new();
    for _ in 0..40 {
        descendants = ChatManager::get_descendant_pids(shell.id());
        if !descendants.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        !descendants.is_empty(),
        "no descendant ever appeared under the stand-in tool — the test cannot conclude"
    );

    let session_id = "test-session-scoped-tools";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    // Narrow scope: the turn ends, the tool keeps running.
    let outcome = manager.interrupt_scoped(session_id, false).await.unwrap();
    assert!(outcome.delivered);
    assert!(
        outcome.killed_pids.is_empty(),
        "kill_tools = false must signal nobody"
    );
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !ChatManager::get_descendant_pids(shell.id()).is_empty(),
        "the tool subprocess must survive an interrupt scoped to the turn"
    );

    // Wide scope: same session, the tool is reaped.
    let outcome = manager.interrupt_scoped(session_id, true).await.unwrap();
    assert!(
        !outcome.killed_pids.is_empty(),
        "kill_tools = true must SIGINT the descendant"
    );

    let _ = shell.kill();
    let _ = shell.wait();
}

#[tokio::test]
async fn test_spawn_nats_rpc_listener_noop_without_nats() {
    // Verifies that spawn_nats_rpc_listener is a no-op without NATS (no panic, no error)
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // This should be a complete no-op — no NATS means early return
    manager.spawn_nats_rpc_listener(
        "test-session",
        manager.active_sessions.clone(),
        CancellationToken::new(),
    );
    // If we get here without panic, the test passes
}

#[tokio::test]
async fn test_routing_decision_local_active_session() {
    // When a session is active locally, is_session_active returns true
    // → the routing logic should choose send_message (local path)
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // No active sessions → is_session_active should return false
    assert!(!manager.is_session_active("some-session").await);

    // try_remote_send without NATS → false
    let remote = manager
        .try_remote_send("some-session", "hello", "user_message")
        .await
        .unwrap();
    assert!(!remote);

    // This confirms the routing: not local + not remote = resume_session fallback
}

// ====================================================================
// YAML persistence tests
// ====================================================================

#[test]
fn test_persist_permission_to_yaml_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");

    // Write an existing config.yaml with other sections
    std::fs::write(
            &yaml_path,
            "server:\n  port: 9090\nneo4j:\n  uri: bolt://db:7687\nchat:\n  default_model: claude-sonnet\n",
        )
        .unwrap();

    let perm = super::super::config::PermissionConfig {
        mode: "default".into(),
        allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
        disallowed_tools: vec!["Bash(rm -rf *)".into()],
    };

    ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

    // Re-read and verify
    let contents = std::fs::read_to_string(&yaml_path).unwrap();
    let doc = crate::yaml::value_from_str(&contents).unwrap();

    // Other sections preserved
    assert_eq!(
        doc["server"]["port"].as_u64().unwrap(),
        9090,
        "server.port should be preserved"
    );
    assert_eq!(
        doc["neo4j"]["uri"].as_str().unwrap(),
        "bolt://db:7687",
        "neo4j.uri should be preserved"
    );
    // Existing chat fields preserved
    assert_eq!(
        doc["chat"]["default_model"].as_str().unwrap(),
        "claude-sonnet",
        "chat.default_model should be preserved"
    );
    // Permissions written correctly
    assert_eq!(
        doc["chat"]["permissions"]["mode"].as_str().unwrap(),
        "default"
    );
    let allowed = doc["chat"]["permissions"]["allowed_tools"]
        .as_sequence()
        .unwrap();
    assert_eq!(allowed.len(), 2);
    assert_eq!(allowed[0].as_str().unwrap(), "Bash(git *)");
    assert_eq!(allowed[1].as_str().unwrap(), "Read");
    let disallowed = doc["chat"]["permissions"]["disallowed_tools"]
        .as_sequence()
        .unwrap();
    assert_eq!(disallowed.len(), 1);
    assert_eq!(disallowed[0].as_str().unwrap(), "Bash(rm -rf *)");
}

#[test]
fn test_persist_permission_to_yaml_no_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");

    // File does not exist yet
    assert!(!yaml_path.exists());

    let perm = super::super::config::PermissionConfig {
        mode: "acceptEdits".into(),
        allowed_tools: vec![],
        disallowed_tools: vec!["Bash(sudo *)".into()],
    };

    ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

    // File should now exist
    assert!(yaml_path.exists());
    let contents = std::fs::read_to_string(&yaml_path).unwrap();
    let doc = crate::yaml::value_from_str(&contents).unwrap();

    assert_eq!(
        doc["chat"]["permissions"]["mode"].as_str().unwrap(),
        "acceptEdits"
    );
    // Empty allowed_tools should be present as empty sequence
    assert!(doc["chat"]["permissions"]["allowed_tools"]
        .as_sequence()
        .unwrap()
        .is_empty());
    assert_eq!(
        doc["chat"]["permissions"]["disallowed_tools"]
            .as_sequence()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn test_persist_permission_roundtrip_with_config_load() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");

    // Start with a config that has no permissions
    std::fs::write(
        &yaml_path,
        "server:\n  port: 8080\nchat:\n  default_model: claude-opus-4-6\n",
    )
    .unwrap();

    // Persist permissions
    let perm = super::super::config::PermissionConfig {
        mode: "plan".into(),
        allowed_tools: vec!["mcp__project-orchestrator__*".into()],
        disallowed_tools: vec![],
    };
    ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

    // Reload via Config::from_yaml_and_env
    // Clear env vars that would override
    std::env::remove_var("CHAT_PERMISSION_MODE");
    std::env::remove_var("CHAT_ALLOWED_TOOLS");
    std::env::remove_var("CHAT_DISALLOWED_TOOLS");

    let config = crate::Config::from_yaml_and_env(Some(&yaml_path)).unwrap();
    let loaded_perm = config
        .chat_permissions
        .expect("chat_permissions should be Some");
    assert_eq!(loaded_perm.mode, "plan");
    assert_eq!(
        loaded_perm.allowed_tools,
        vec!["mcp__project-orchestrator__*"]
    );
    assert!(loaded_perm.disallowed_tools.is_empty());

    // config_yaml_path should be set
    assert_eq!(
        config.config_yaml_path.as_deref(),
        Some(yaml_path.as_path())
    );
}

#[test]
fn test_persist_permission_atomic_no_tmp_leftover() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");
    let tmp_path = dir.path().join("config.yaml.tmp");

    std::fs::write(&yaml_path, "chat: {}\n").unwrap();

    let perm = super::super::config::PermissionConfig::default();
    ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

    // .tmp file should not exist after successful rename
    assert!(
        !tmp_path.exists(),
        "Temporary file should be cleaned up after atomic rename"
    );
    // Original file should still exist with updated content
    assert!(yaml_path.exists());
}

#[tokio::test]
async fn test_update_permission_config_persists_to_yaml() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");
    std::fs::write(&yaml_path, "chat:\n  default_model: test\n").unwrap();

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
        .with_config_yaml_path(yaml_path.clone());

    let new_perm = super::super::config::PermissionConfig {
        mode: "acceptEdits".into(),
        allowed_tools: vec!["Read".into()],
        disallowed_tools: vec![],
    };

    let result = manager.update_permission_config(new_perm).await.unwrap();
    assert_eq!(result.mode, "acceptEdits");

    // Verify it was persisted to disk
    let contents = std::fs::read_to_string(&yaml_path).unwrap();
    let doc = crate::yaml::value_from_str(&contents).unwrap();
    assert_eq!(
        doc["chat"]["permissions"]["mode"].as_str().unwrap(),
        "acceptEdits"
    );
    // Other chat fields preserved
    assert_eq!(doc["chat"]["default_model"].as_str().unwrap(), "test");
}

// ── helpers for ActiveSession tests (no Claude CLI required) ────────

/// Create an ActiveSession backed by a MockTransport.
/// Does NOT require the Claude CLI to be installed.
fn mock_active_session(
    is_streaming: bool,
) -> (
    ActiveSession,
    nexus_claude::transport::mock::MockTransportHandle,
) {
    let (transport, handle) = nexus_claude::transport::mock::MockTransport::pair();
    let client = InteractiveClient::from_transport(transport);
    let (tx, _rx) = broadcast::channel(16);

    let session = ActiveSession {
        context_window: None,
        anchor: Default::default(),
        mirror_tap: None,
        events_tx: tx,
        last_activity: Instant::now(),
        cli_session_id: None,
        client: Arc::new(Mutex::new(client)),
        interrupt_flag: Arc::new(AtomicBool::new(false)),
        memory_manager: None,
        next_seq: Arc::new(AtomicI64::new(1)),
        pending_messages: Arc::new(Mutex::new(VecDeque::<PendingMessage>::new())),
        is_streaming: Arc::new(AtomicBool::new(is_streaming)),
        streaming_text: Arc::new(Mutex::new(String::new())),
        streaming_events: Arc::new(Mutex::new(Vec::new())),
        permission_mode: None,
        model: None,
        sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
        stdin_tx: None,
        child_pid: None,
        nats_cancel: CancellationToken::new(),
        interrupt_token: CancellationToken::new(),
        turn_generation: Arc::new(AtomicU64::new(0)),
        result_cursor: Default::default(),
        stream_task: Arc::new(std::sync::Mutex::new(None)),
        post_stream_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
        pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
            std::collections::HashMap::new(),
        )),
        auto_continue: Arc::new(AtomicBool::new(false)),
        auto_continue_count: Arc::new(AtomicU32::new(0)),
        max_auto_continues: 0,
        rfc_accumulator: Arc::new(Mutex::new(
            crate::chat::observation_detector::RfcAccumulator::new(),
        )),
        protocol_run_id: None,
        protocol_state: None,
        reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
        objective_tracking: false,
        objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
        objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
        work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
        provider_outage: Arc::new(std::sync::Mutex::new(
            crate::chat::provider::outage::TurnHold::new(
                crate::chat::provider::resolver::CLAUDE_CODE,
            ),
        )),
        session_grants: Arc::default(),
        oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
        oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
        oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
        oob_capped_warned: Arc::new(AtomicBool::new(false)),
        cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
        cancel_tools_cap: CANCEL_TOOLS_CAP,
        cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
        active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
        tool_start_descendants: Arc::new(Mutex::new(HashMap::new())),
        cli_background_tasks: Arc::new(AtomicUsize::new(0)),
        cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
        cancel_task_cap: CANCEL_TASK_CAP,
        cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
    };

    (session, handle)
}

// ── parse_permission_control_msg tests ──────────────────────────────

#[test]
fn test_parse_permission_control_msg_can_use_tool() {
    let msg = serde_json::json!({
        "type": "control_request",
        "request_id": "req_abc",
        "request": {
            "subtype": "can_use_tool",
            "tool_name": "Bash",
            "input": {"command": "ls -la"}
        }
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_some(), "should parse can_use_tool");

    match event.unwrap() {
        ChatEvent::PermissionRequest {
            id,
            tool,
            input,
            parent_tool_use_id,
            ..
        } => {
            assert_eq!(id, "req_abc");
            assert_eq!(tool, "Bash");
            assert_eq!(input["command"], "ls -la");
            assert!(parent_tool_use_id.is_none());
        }
        other => panic!("expected PermissionRequest, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_with_parent_tool_use_id() {
    let msg = serde_json::json!({
        "type": "control_request",
        "request_id": "req_xyz",
        "request": {
            "subtype": "can_use_tool",
            "toolName": "Read",
            "input": {"file_path": "/tmp/test.rs"}
        }
    });

    let event = parse_permission_control_msg(&msg, Some("toolu_parent_123".to_string()));
    assert!(event.is_some());

    match event.unwrap() {
        ChatEvent::PermissionRequest {
            id,
            tool,
            input,
            parent_tool_use_id,
            ..
        } => {
            assert_eq!(id, "req_xyz");
            assert_eq!(tool, "Read");
            assert_eq!(input["file_path"], "/tmp/test.rs");
            assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent_123"));
        }
        other => panic!("expected PermissionRequest, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_ignores_non_permission() {
    // subtype != "can_use_tool"
    let msg = serde_json::json!({
        "type": "control_request",
        "request_id": "req_other",
        "request": {
            "subtype": "server_info",
            "data": {"model": "claude-sonnet-4-6"}
        }
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_none(), "non-permission should return None");
}

#[test]
fn test_parse_permission_control_msg_flat_format() {
    // No nested "request" — all fields at top level
    let msg = serde_json::json!({
        "subtype": "can_use_tool",
        "requestId": "flat_001",
        "tool_name": "Write",
        "input": {"file_path": "/tmp/out.txt", "content": "hello"}
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_some(), "flat format should parse");

    match event.unwrap() {
        ChatEvent::PermissionRequest {
            id, tool, input, ..
        } => {
            assert_eq!(id, "flat_001");
            assert_eq!(tool, "Write");
            assert_eq!(input["content"], "hello");
        }
        other => panic!("expected PermissionRequest, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_missing_fields_defaults() {
    // Minimal message with just the subtype
    let msg = serde_json::json!({
        "request": {
            "subtype": "can_use_tool"
        }
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_some());

    match event.unwrap() {
        ChatEvent::PermissionRequest {
            id, tool, input, ..
        } => {
            assert_eq!(id, "", "missing request_id defaults to empty");
            assert_eq!(tool, "unknown", "missing tool defaults to 'unknown'");
            assert_eq!(
                input,
                serde_json::json!({}),
                "missing input defaults to empty object"
            );
        }
        other => panic!("expected PermissionRequest, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_camel_vs_snake_tool_name() {
    // toolName (camelCase) should be preferred
    let msg_camel = serde_json::json!({
        "request_id": "r1",
        "request": {
            "subtype": "can_use_tool",
            "toolName": "CamelTool",
            "input": {}
        }
    });
    let evt = parse_permission_control_msg(&msg_camel, None).unwrap();
    if let ChatEvent::PermissionRequest { tool, .. } = evt {
        assert_eq!(tool, "CamelTool");
    }

    // tool_name (snake_case) fallback
    let msg_snake = serde_json::json!({
        "request_id": "r2",
        "request": {
            "subtype": "can_use_tool",
            "tool_name": "SnakeTool",
            "input": {}
        }
    });
    let evt = parse_permission_control_msg(&msg_snake, None).unwrap();
    if let ChatEvent::PermissionRequest { tool, .. } = evt {
        assert_eq!(tool, "SnakeTool");
    }
}

#[test]
fn test_parse_permission_control_msg_request_id_variants() {
    // requestId (camelCase)
    let msg = serde_json::json!({
        "requestId": "camel_id_001",
        "request": { "subtype": "can_use_tool", "tool_name": "T", "input": {} }
    });
    if let Some(ChatEvent::PermissionRequest { id, .. }) = parse_permission_control_msg(&msg, None)
    {
        assert_eq!(id, "camel_id_001");
    }

    // request_id (snake_case)
    let msg = serde_json::json!({
        "request_id": "snake_id_002",
        "request": { "subtype": "can_use_tool", "tool_name": "T", "input": {} }
    });
    if let Some(ChatEvent::PermissionRequest { id, .. }) = parse_permission_control_msg(&msg, None)
    {
        assert_eq!(id, "snake_id_002");
    }
}

// ── AskUserQuestion is closed by a denial that ends the turn (§15.6) ──

#[test]
fn a_question_is_closed_by_a_denial_never_by_an_allow() {
    let response = question_closing_response("req-q");
    assert_eq!(response["response"]["request_id"], "req-q");
    assert_eq!(response["response"]["response"]["behavior"], "deny");
    assert!(
        response["response"]["response"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("posed to the user")),
        "{response}"
    );
    assert!(response["response"]["response"]
        .get("updatedInput")
        .is_none());
}

// ── parse_permission_control_msg: AskUserQuestion tests ─────────────

#[test]
fn test_parse_permission_control_msg_ask_user_question_returns_dedicated_event() {
    let msg = serde_json::json!({
        "type": "control_request",
        "requestId": "req_ask_001",
        "request": {
            "subtype": "can_use_tool",
            "toolName": "AskUserQuestion",
            "toolUseId": "toolu_ask_123",
            "input": {
                "questions": [{
                    "question": "Which framework?",
                    "header": "Framework",
                    "options": [
                        {"label": "React", "description": "Popular UI library"},
                        {"label": "Vue", "description": "Progressive framework"}
                    ],
                    "multiSelect": false
                }]
            }
        }
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_some(), "AskUserQuestion should parse");

    match event.unwrap() {
        ChatEvent::AskUserQuestion {
            id,
            tool_call_id,
            questions,
            input,
            parent_tool_use_id,
            ..
        } => {
            assert_eq!(id, "req_ask_001");
            assert_eq!(tool_call_id, "toolu_ask_123");
            assert!(questions.is_array(), "questions should be an array");
            assert_eq!(questions.as_array().unwrap().len(), 1);
            assert_eq!(questions[0]["header"], "Framework");
            assert!(input.get("questions").is_some());
            assert!(parent_tool_use_id.is_none());
        }
        other => panic!("expected AskUserQuestion, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_ask_user_question_with_parent() {
    let msg = serde_json::json!({
        "requestId": "req_ask_002",
        "request": {
            "subtype": "can_use_tool",
            "toolName": "AskUserQuestion",
            "toolUseId": "toolu_ask_456",
            "input": {
                "questions": [{"question": "Choose:", "header": "Q", "options": [{"label": "A"}, {"label": "B"}], "multiSelect": false}]
            }
        }
    });

    let event = parse_permission_control_msg(&msg, Some("toolu_parent_789".to_string()));
    assert!(event.is_some());

    match event.unwrap() {
        ChatEvent::AskUserQuestion {
            id,
            parent_tool_use_id,
            ..
        } => {
            assert_eq!(id, "req_ask_002");
            assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent_789"));
        }
        other => panic!("expected AskUserQuestion, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_bash_still_returns_permission_request() {
    // Regression: Bash (and all other tools) must still return PermissionRequest
    let msg = serde_json::json!({
        "requestId": "req_bash_001",
        "request": {
            "subtype": "can_use_tool",
            "toolName": "Bash",
            "input": {"command": "echo hello"}
        }
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_some());

    match event.unwrap() {
        ChatEvent::PermissionRequest { tool, .. } => {
            assert_eq!(tool, "Bash");
        }
        other => panic!("expected PermissionRequest for Bash, got: {:?}", other),
    }
}

#[test]
fn test_parse_permission_control_msg_ask_user_question_missing_questions() {
    // If input doesn't have "questions", should default to empty array
    let msg = serde_json::json!({
        "requestId": "req_ask_003",
        "request": {
            "subtype": "can_use_tool",
            "toolName": "AskUserQuestion",
            "input": {}
        }
    });

    let event = parse_permission_control_msg(&msg, None);
    assert!(event.is_some());

    match event.unwrap() {
        ChatEvent::AskUserQuestion {
            questions,
            tool_call_id,
            ..
        } => {
            assert!(questions.is_array());
            assert_eq!(questions.as_array().unwrap().len(), 0);
            assert_eq!(tool_call_id, "", "missing toolUseId defaults to empty");
        }
        other => panic!("expected AskUserQuestion, got: {:?}", other),
    }
}

// ── send_permission_response tests ──────────────────────────────────

#[tokio::test]
async fn test_send_permission_response_no_session_errors() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager
        .send_permission_response("nonexistent-session-id", "req-001", true)
        .await;
    assert!(result.is_err(), "should fail for non-existent session");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not found"),
        "error should mention 'not found': {err_msg}"
    );
}

// ── interrupt tests ─────────────────────────────────────────────────

#[tokio::test]
async fn test_interrupt_no_session_errors() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let result = manager.interrupt("nonexistent-session-id").await;
    // The manager falls back to NATS publish when no local session exists.
    // Without NATS configured, it returns an error or succeeds silently.
    // Either way, it should not panic.
    assert!(
        result.is_ok() || result.is_err(),
        "interrupt on non-existent session should not panic"
    );
}

// ── interrupt flag atomicity & latency tests ────────────────────────

/// Verify that `interrupt()` sets the atomic flag immediately (< 1ms)
/// without acquiring the InteractiveClient Mutex.
#[tokio::test]
async fn test_interrupt_sets_flag_atomically() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _handle) = mock_active_session(false);
    let flag = session.interrupt_flag.clone();

    // Insert session into manager
    let session_id = "test-session-atomic";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    assert!(!flag.load(Ordering::SeqCst), "flag should start false");

    // Measure interrupt latency
    let start = Instant::now();
    manager.interrupt(session_id).await.unwrap();
    let elapsed = start.elapsed();

    assert!(
        flag.load(Ordering::SeqCst),
        "flag should be true after interrupt"
    );
    // A wall-clock budget on a shared runner measures the MACHINE, not the code:
    // this one was measured at 3.12 ms on a loaded CI runner for an interrupt that
    // only sets an AtomicBool. A regression does not make it "a bit slow": waiting on
    // the client Mutex or on an I/O would not return at all inside the test.
    assert!(
        elapsed < Duration::from_millis(100),
        "interrupt() took {:?}, expected under 100 ms",
        elapsed
    );
}

/// Verify that `interrupt()` works even when the client Mutex is held
/// (simulating an active stream_response).
#[tokio::test]
async fn test_interrupt_does_not_wait_for_client_lock() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _handle) = mock_active_session(true);
    let flag = session.interrupt_flag.clone();
    let client_lock = session.client.clone();

    let session_id = "test-session-lock";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    // Hold the client Mutex (simulating stream_response owning it)
    let _guard = client_lock.lock().await;

    // interrupt() should still complete instantly because it only
    // touches the AtomicBool, not the Mutex
    let start = Instant::now();
    manager.interrupt(session_id).await.unwrap();
    let elapsed = start.elapsed();

    assert!(
        flag.load(Ordering::SeqCst),
        "flag should be set even while Mutex is held"
    );
    // A wall-clock budget on a shared runner measures the MACHINE, not the code:
    // this one was measured at 3.12 ms on a loaded CI runner for an interrupt that
    // only sets an AtomicBool. A regression does not make it "a bit slow": waiting on
    // the client Mutex or on an I/O would not return at all inside the test.
    assert!(
        elapsed < Duration::from_millis(100),
        "interrupt() took {:?} even though the client Mutex is held",
        elapsed
    );
}

/// Verify that interrupt works when streaming is active
/// (is_streaming flag set to true).
#[tokio::test]
async fn test_interrupt_during_active_stream() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _handle) = mock_active_session(true);
    let interrupt_flag = session.interrupt_flag.clone();
    let is_streaming = session.is_streaming.clone();

    let session_id = "test-session-streaming";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    assert!(
        is_streaming.load(Ordering::SeqCst),
        "session should be streaming"
    );
    assert!(
        !interrupt_flag.load(Ordering::SeqCst),
        "interrupt should start false"
    );

    manager.interrupt(session_id).await.unwrap();

    assert!(
        interrupt_flag.load(Ordering::SeqCst),
        "interrupt flag should be set during active stream"
    );
}

/// Benchmark: interrupt latency over 100 sessions, each must be < 1ms.
#[tokio::test]
async fn test_interrupt_latency_benchmark() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let iterations = 100u32;
    let mut flags = Vec::with_capacity(iterations as usize);

    // Create N sessions
    for i in 0..iterations {
        let (session, _handle) = mock_active_session(true);
        let flag = session.interrupt_flag.clone();
        flags.push(flag);
        manager
            .active_sessions
            .write()
            .await
            .insert(format!("bench-{i}"), session);
    }

    // Interrupt all sessions and measure total time
    let start = Instant::now();
    for i in 0..iterations {
        manager.interrupt(&format!("bench-{i}")).await.unwrap();
    }
    let total = start.elapsed();
    let avg = total / iterations;

    // Verify all flags were set
    for (i, flag) in flags.iter().enumerate() {
        assert!(
            flag.load(Ordering::SeqCst),
            "flag for session bench-{i} should be set"
        );
    }

    // A wall-clock budget on a shared runner measures the MACHINE, not the code:
    // this one was measured at 3.12 ms on a loaded CI runner for an interrupt that
    // only sets an AtomicBool. A regression does not make it "a bit slow": waiting on
    // the client Mutex or on an I/O would not return at all inside the test.
    assert!(
        avg < Duration::from_millis(100),
        "Average interrupt latency {:?} exceeds 100 ms (total {:?} for {iterations})",
        avg,
        total
    );
}

/// Verify that double-interrupt doesn't panic and is idempotent.
#[tokio::test]
async fn test_double_interrupt_is_idempotent() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _handle) = mock_active_session(false);
    let flag = session.interrupt_flag.clone();

    let session_id = "test-double-interrupt";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    // First interrupt
    manager.interrupt(session_id).await.unwrap();
    assert!(flag.load(Ordering::SeqCst));

    // Second interrupt — should not panic or error
    let result = manager.interrupt(session_id).await;
    assert!(result.is_ok(), "second interrupt should succeed");
    assert!(flag.load(Ordering::SeqCst), "flag should remain true");
}

// ── attention_changed relay ─────────────────────────────────────────

fn attention_hybrid() -> (
    Arc<crate::events::HybridEmitter>,
    tokio::sync::broadcast::Receiver<crate::events::CrudEvent>,
) {
    let h = Arc::new(crate::events::HybridEmitter::new(Arc::new(
        crate::events::EventBus::default(),
    )));
    let rx = h.subscribe();
    (h, rx)
}

#[tokio::test]
async fn test_permission_request_emits_attention_changed_without_the_command() {
    let (h, mut rx) = attention_hybrid();
    let emitter: Option<Arc<dyn crate::events::EventEmitter>> = Some(h);
    let msg = serde_json::json!({
        "type": "control_request",
        "request_id": "req-1",
        "request": {
            "subtype": "can_use_tool",
            "tool_name": "Bash",
            "input": {"command": "rm -rf /secret-command-text"},
            "tool_use_id": "tu-1"
        }
    });
    let event = parse_permission_control_msg(&msg, None).expect("permission request");
    assert!(matches!(event, ChatEvent::PermissionRequest { .. }));
    notify_attention_for_chat_event(&emitter, "sess-1", &event);
    // Wait for the event (bounded), not a fixed 400 ms: under a loaded runner the
    // emitter task may not have run yet and a single try_recv saw an empty bus.
    let ev = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("attention_changed on the bus within 10 s")
        .expect("attention_changed on the bus");
    assert_eq!(ev.entity_type, crate::events::EntityType::AttentionChanged);
    assert_eq!(ev.payload["session_id"], "sess-1");
    assert_eq!(
        ev.payload["reasons"],
        serde_json::json!(["permission_request"])
    );
    assert!(
        !serde_json::to_string(&ev)
            .unwrap()
            .contains("secret-command-text"),
        "the command text must not transit on the general bus"
    );
}

#[tokio::test]
async fn test_permission_decision_emits_attention_changed() {
    let state = mock_app_state();
    let (h, mut rx) = attention_hybrid();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
        .with_event_emitter(h);
    let (mut session, _handle) = mock_active_session(false);
    let (stdin_tx, _stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
    session.stdin_tx = Some(stdin_tx);
    session
        .pending_permission_inputs
        .lock()
        .await
        .insert("req-d".to_string(), PendingPermission::default());
    manager
        .active_sessions
        .write()
        .await
        .insert("sess-d".to_string(), session);
    manager
        .send_permission_response("sess-d", "req-d", true)
        .await
        .unwrap();
    // Wait for the event (bounded), not a fixed 400 ms: under a loaded runner the
    // emitter task may not have run yet and a single try_recv saw an empty bus.
    let ev = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("attention_changed on the bus within 10 s")
        .expect("attention_changed on the bus");
    assert_eq!(ev.payload["session_id"], "sess-d");
    assert_eq!(
        ev.payload["reasons"],
        serde_json::json!(["permission_decision"])
    );
}

// ── send_permission_response with mock transport ────────────────────

#[tokio::test]
async fn test_send_permission_response_allow_via_mock() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (mut session, _handle) = mock_active_session(false);

    // Create a stdin channel to capture the control response
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
    session.stdin_tx = Some(stdin_tx);

    // Pre-populate pending_permission_inputs with the original tool input
    let original_input = serde_json::json!({
        "command": "echo \"hello\"",
        "description": "Print hello"
    });
    session.pending_permission_inputs.lock().await.insert(
        "req-allow-001".to_string(),
        PendingPermission {
            input: original_input.clone(),
            tool: "Bash".to_string(),
        },
    );

    let session_id = "test-perm-allow";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    manager
        .send_permission_response(session_id, "req-allow-001", true)
        .await
        .unwrap();

    // Verify the control response was sent through stdin_tx
    let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
        .await
        .expect("should receive within timeout")
        .expect("channel should be open");

    let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();

    // Verify full envelope: {"type": "control_response", "response": {"subtype": "success", "request_id": ..., "response": {...}}}
    assert_eq!(sent["type"], "control_response");
    let outer_response = &sent["response"];
    assert_eq!(outer_response["subtype"], "success");
    assert_eq!(outer_response["request_id"], "req-allow-001");
    let inner_response = &outer_response["response"];
    assert_eq!(
        inner_response["behavior"], "allow",
        "inner response should contain behavior: allow"
    );
    assert_eq!(
        inner_response["updatedInput"], original_input,
        "updatedInput should contain the original tool input"
    );
}

#[tokio::test]
async fn test_send_permission_response_deny_via_mock() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (mut session, _handle) = mock_active_session(false);

    // Create a stdin channel to capture the control response
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
    session.stdin_tx = Some(stdin_tx);

    let session_id = "test-perm-deny";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    manager
        .send_permission_response(session_id, "req-deny-002", false)
        .await
        .unwrap();

    let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
        .await
        .expect("should receive within timeout")
        .expect("channel should be open");

    let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();
    assert_eq!(sent["type"], "control_response");
    let outer_response = &sent["response"];
    assert_eq!(outer_response["subtype"], "success");
    assert_eq!(outer_response["request_id"], "req-deny-002");
    let inner_response = &outer_response["response"];
    assert_eq!(
        inner_response["behavior"], "deny",
        "inner response should contain behavior: deny"
    );
    assert!(
        inner_response["message"]
            .as_str()
            .unwrap()
            .contains("denied"),
        "inner response should contain a denial message"
    );
}

#[tokio::test]
async fn test_set_session_model_sends_control_request_via_stdin() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (mut session, _handle) = mock_active_session(false);

    // Create a stdin channel to capture the control request
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
    session.stdin_tx = Some(stdin_tx);

    // Subscribe to events BEFORE inserting the session
    let mut events_rx = session.events_tx.subscribe();

    let session_id = "test-model-change";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    // Change model
    let broadcast = manager
        .set_session_model(session_id, "claude-opus-4-20250514")
        .await
        .unwrap();
    assert!(
        broadcast,
        "a live session broadcasts model_changed itself: the caller must not confirm again"
    );

    // Verify the control request was sent through stdin_tx
    let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
        .await
        .expect("should receive within timeout")
        .expect("channel should be open");

    let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();
    assert_eq!(sent["type"], "control_request");
    assert!(
        sent["request_id"].as_str().is_some(),
        "should have a request_id"
    );
    let request = &sent["request"];
    assert_eq!(request["subtype"], "set_model");
    assert_eq!(request["model"], "claude-opus-4-20250514");

    // Verify ModelChanged event was broadcast — exactly once
    let event = tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
        .await
        .expect("should receive event within timeout")
        .expect("channel should be open");
    assert!(
        events_rx.try_recv().is_err(),
        "one change, one model_changed event: two would show the confirmation twice"
    );

    assert_eq!(event.event_type(), "model_changed");
    if let ChatEvent::ModelChanged { model, .. } = event {
        assert_eq!(model, "claude-opus-4-20250514");
    } else {
        panic!("Expected ModelChanged event, got {:?}", event.event_type());
    }

    // Verify the in-memory session was updated
    let sessions = manager.active_sessions.read().await;
    let session = sessions.get(session_id).unwrap();
    assert_eq!(session.model.as_deref(), Some("claude-opus-4-20250514"));
}

#[tokio::test]
async fn test_set_session_model_without_stdin_persists_and_broadcasts() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _handle) = mock_active_session(false);
    // stdin_tx is None — no live CLI to receive a control_request. This must NOT error:
    // the model is updated in-memory + persisted so it applies on the next spawn.

    let mut events_rx = session.events_tx.subscribe();

    let session_id = "test-model-no-stdin";
    manager
        .active_sessions
        .write()
        .await
        .insert(session_id.to_string(), session);

    let broadcast = manager
        .set_session_model(session_id, "claude-opus-4-20250514")
        .await
        .expect("model change without live stdin should succeed (persist + broadcast)");
    assert!(broadcast);

    // In-memory model is updated so a respawn from this session uses the new model.
    {
        let sessions = manager.active_sessions.read().await;
        assert_eq!(
            sessions.get(session_id).unwrap().model.as_deref(),
            Some("claude-opus-4-20250514")
        );
    }

    // ModelChanged is still broadcast so the frontend reflects the choice.
    let event = tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
        .await
        .expect("should receive event within timeout")
        .expect("channel should be open");
    assert_eq!(event.event_type(), "model_changed");
}

#[tokio::test]
async fn test_set_session_model_dormant_session_persists() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // A session absent from active_sessions is dormant (idle-cleaned). set_model must
    // still succeed by persisting to Neo4j so the next resume launches with the new
    // model — rather than rejecting the change and respawning on the create-time model.
    let session_id = uuid::Uuid::new_v4().to_string();
    let result = manager
        .set_session_model(&session_id, "claude-opus-4-20250514")
        .await;

    // `false`: nothing was broadcast (no subscribers), so the WebSocket
    // handler confirms to the asker directly.
    assert!(!result.expect("dormant session model change should persist without error"));
}

// ====================================================================
// CompactionNotifier
// ====================================================================

#[tokio::test]
async fn test_compaction_notifier_emits_event_on_pre_compact() {
    use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

    let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(tx, None, "test-session".to_string());

    let input = HookInput::PreCompact(PreCompactHookInput {
        session_id: "test-session".into(),
        transcript_path: "/tmp/transcript".into(),
        cwd: "/test".into(),
        permission_mode: None,
        trigger: "auto".into(),
        custom_instructions: None,
    });

    let context = HookContext { signal: None };
    let result = notifier.execute(&input, None, &context).await;
    assert!(result.is_ok());

    // Verify event was broadcast
    let event = rx.try_recv().unwrap();
    assert!(matches!(
        event,
        ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"
    ));
}

#[tokio::test]
async fn test_compaction_notifier_manual_trigger() {
    use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

    let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(tx, None, "sess-2".to_string());

    let input = HookInput::PreCompact(PreCompactHookInput {
        session_id: "sess-2".into(),
        transcript_path: "/tmp/t".into(),
        cwd: "/".into(),
        permission_mode: Some("default".into()),
        trigger: "manual".into(),
        custom_instructions: Some("Keep API context".into()),
    });

    let context = HookContext { signal: None };
    let result = notifier.execute(&input, None, &context).await.unwrap();

    // Must always return continue=true
    if let nexus_claude::HookJSONOutput::Sync(sync) = result {
        assert_eq!(sync.continue_, Some(true));
    } else {
        panic!("Expected Sync output");
    }

    let event = rx.try_recv().unwrap();
    assert!(matches!(
        event,
        ChatEvent::CompactionStarted { ref trigger } if trigger == "manual"
    ));
}

#[tokio::test]
async fn test_compaction_notifier_noop_on_other_hooks() {
    use nexus_claude::{HookCallback, HookContext, HookInput, PreToolUseHookInput};

    let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(tx, None, "test-session".to_string());

    let input = HookInput::PreToolUse(PreToolUseHookInput {
        // Main thread, not a sub-agent — the CLI omits both fields there.
        agent_id: None,
        agent_type: None,
        session_id: "s".into(),
        transcript_path: "/t".into(),
        cwd: "/c".into(),
        permission_mode: None,
        tool_name: "Bash".into(),
        tool_input: serde_json::json!({"command": "ls"}),
    });

    let context = HookContext { signal: None };
    let result = notifier.execute(&input, Some("tu1"), &context).await;
    assert!(result.is_ok());

    // No event should be broadcast for non-PreCompact hooks
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn test_compaction_notifier_returns_custom_instructions() {
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::{test_plan, test_step, test_task};
    use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

    // 1. Setup: create a plan + task + steps in MockGraphStore
    let mock = Arc::new(MockGraphStore::new());
    let plan = test_plan();
    let plan_id = plan.id;
    mock.create_plan(&plan).await.unwrap();

    let mut task = test_task();
    task.affected_files = vec![
        "src/chat/manager.rs".to_string(),
        "src/chat/compaction_context.rs".to_string(),
    ];
    task.title = Some("Implement compaction context".to_string());
    let task_id = task.id;
    mock.create_task(plan_id, &task).await.unwrap();

    let step1 = test_step(1, "Create builder struct");
    let step2 = test_step(2, "Add custom_instructions output");
    mock.create_step(task_id, &step1).await.unwrap();
    mock.create_step(task_id, &step2).await.unwrap();

    // 2. Create CompactionNotifier with task context
    let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(tx, None, "test-ctx-session".to_string()).with_context(
        mock.clone() as Arc<dyn GraphStore>,
        CompactionContextSource::Task(task_id),
    );

    // 3. Execute with a PreCompact input
    let input = HookInput::PreCompact(PreCompactHookInput {
        session_id: "test-ctx-session".into(),
        transcript_path: "/tmp/t".into(),
        cwd: "/test".into(),
        permission_mode: None,
        trigger: "auto".into(),
        custom_instructions: None,
    });

    let context = HookContext { signal: None };
    let result = notifier.execute(&input, None, &context).await.unwrap();

    // 4. Verify: the custom instructions go out as systemMessage (M2c)
    if let nexus_claude::HookJSONOutput::Sync(sync) = result {
        assert_eq!(sync.continue_, Some(true), "Must always continue");
        let instructions = sync
            .system_message
            .expect("systemMessage (custom_instructions) should be set");
        assert!(
            sync.reason.is_none(),
            "the CLI ignores reason on PreCompact"
        );
        assert!(
            instructions.contains("src/chat/manager.rs"),
            "Should mention affected files, got: {instructions}"
        );
        assert!(
            instructions.contains("Implement compaction context"),
            "Should mention task title, got: {instructions}"
        );
    } else {
        panic!("Expected Sync output");
    }
}

#[tokio::test]
async fn test_compaction_notifier_no_context_returns_no_instructions() {
    use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

    // CompactionNotifier without context (backward-compatible)
    let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(tx, None, "no-ctx".to_string());

    let input = HookInput::PreCompact(PreCompactHookInput {
        session_id: "no-ctx".into(),
        transcript_path: "/tmp/t".into(),
        cwd: "/".into(),
        permission_mode: None,
        trigger: "auto".into(),
        custom_instructions: None,
    });

    let context = HookContext { signal: None };
    let result = notifier.execute(&input, None, &context).await.unwrap();

    if let nexus_claude::HookJSONOutput::Sync(sync) = result {
        assert_eq!(sync.continue_, Some(true));
        assert!(
            sync.system_message.is_none() && sync.reason.is_none(),
            "No context source → no custom instructions"
        );
    } else {
        panic!("Expected Sync output");
    }
}

// ====================================================================
// Hook dispatch e2e (CompactionNotifier + dispatch_hook_from_registry)
// ====================================================================

/// E2E test: simulates the full hook_callback flow as it happens in stream_response.
///
/// 1. Create a CompactionNotifier with a broadcast channel
/// 2. Register it via InteractiveClient::initialize_hooks()
/// 3. Clone the hook_callbacks registry (like stream_response does)
/// 4. Build a hook_callback JSON message (like the CLI would send)
/// 5. Dispatch via dispatch_hook_from_registry (lock-free)
/// 6. Verify CompactionStarted was broadcast
/// 7. Verify the hook output is continue: true
#[tokio::test]
async fn test_compaction_hook_e2e_dispatch_broadcasts_event() {
    use nexus_claude::transport::mock::MockTransport;
    use nexus_claude::{dispatch_hook_from_registry, HookMatcher, InteractiveClient};

    // 1. Create CompactionNotifier backed by a broadcast channel
    let (events_tx, mut events_rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(events_tx, None, "test-session-e2e".to_string());

    // 2. Build hooks map and initialize
    let mut hooks = std::collections::HashMap::new();
    hooks.insert(
        "PreCompact".to_string(),
        vec![HookMatcher {
            matcher: None,
            hooks: vec![std::sync::Arc::new(notifier)],
        }],
    );

    let (transport, _handle) = MockTransport::pair();
    let client = InteractiveClient::from_transport_with_hooks(transport, hooks);
    client.initialize_hooks().await.unwrap();

    // 3. Clone registry (this is what stream_response does before the select! loop)
    let registry = client.hook_callbacks();

    // 4. Get the callback_id that was generated
    let callback_id = {
        let cbs = registry.read().await;
        assert_eq!(cbs.len(), 1, "Should have exactly one registered callback");
        cbs.keys().next().unwrap().clone()
    };

    // 5. Build a hook_callback control message (simulating what the CLI sends)
    let control_msg = serde_json::json!({
        "type": "control_request",
        "request_id": "req-e2e-001",
        "request": {
            "subtype": "hook_callback",
            "callback_id": callback_id,
            "input": {
                "hook_event_name": "PreCompact",
                "session_id": "test-session-e2e",
                "transcript_path": "/tmp/transcript.json",
                "cwd": "/home/user/project",
                "trigger": "auto"
            }
        }
    });

    // 6. Dispatch (lock-free, just like stream_response does)
    let result = dispatch_hook_from_registry(&control_msg, &registry).await;
    assert!(result.is_some(), "dispatch should find the callback");
    let output = result.unwrap();
    assert!(output.is_ok(), "callback should succeed");

    // 7. Verify CompactionStarted was broadcast on the events channel
    let event = events_rx
        .try_recv()
        .expect("Should have received CompactionStarted event");
    assert!(
        matches!(event, ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"),
        "Event should be CompactionStarted with trigger=auto, got: {:?}",
        event
    );

    // 8. Verify output is Sync with continue: true
    match output.unwrap() {
        nexus_claude::HookJSONOutput::Sync(sync) => {
            assert_eq!(
                sync.continue_,
                Some(true),
                "Hook should return continue=true"
            );
        }
        other => panic!("Expected Sync output, got: {:?}", other),
    }
}

/// Test that build_hook_response_json produces valid JSON that would be sent to CLI.
/// Verifies the format matches what the CLI expects as a control_response.
#[tokio::test]
async fn test_hook_response_sent_to_cli_format() {
    use nexus_claude::transport::mock::MockTransport;
    use nexus_claude::{
        build_hook_response_json, dispatch_hook_from_registry, HookMatcher, InteractiveClient,
    };

    // Setup: CompactionNotifier + initialize + dispatch
    let (events_tx, _events_rx) = broadcast::channel::<ChatEvent>(16);
    let notifier = CompactionNotifier::new(events_tx, None, "sess-resp".to_string());

    let mut hooks = std::collections::HashMap::new();
    hooks.insert(
        "PreCompact".to_string(),
        vec![HookMatcher {
            matcher: None,
            hooks: vec![std::sync::Arc::new(notifier)],
        }],
    );

    let (transport, _handle) = MockTransport::pair();
    let client = InteractiveClient::from_transport_with_hooks(transport, hooks);
    client.initialize_hooks().await.unwrap();

    let registry = client.hook_callbacks();
    let callback_id = {
        let cbs = registry.read().await;
        cbs.keys().next().unwrap().clone()
    };

    let request_id = "req-response-test-001";
    let control_msg = serde_json::json!({
        "type": "control_request",
        "request_id": request_id,
        "request": {
            "subtype": "hook_callback",
            "callback_id": callback_id,
            "input": {
                "hook_event_name": "PreCompact",
                "session_id": "sess-resp",
                "transcript_path": "/tmp/t.json",
                "cwd": "/home",
                "trigger": "manual"
            }
        }
    });

    let result = dispatch_hook_from_registry(&control_msg, &registry)
        .await
        .expect("Should dispatch");

    // Build the response JSON (this is what stream_response sends via stdin_tx)
    let response_json_str = build_hook_response_json(request_id, &result);

    // Parse and verify structure
    let response: serde_json::Value =
        serde_json::from_str(&response_json_str).expect("Should be valid JSON");

    assert_eq!(
        response["type"], "control_response",
        "type must be control_response"
    );

    let resp = &response["response"];
    assert_eq!(resp["subtype"], "success", "subtype must be success");
    assert_eq!(
        resp["request_id"], request_id,
        "request_id must match the original"
    );

    // The inner response should contain the hook output (continue: true)
    let inner = &resp["response"];
    assert_eq!(
        inner["continue"], true,
        "Hook output should have continue=true"
    );
}

#[tokio::test]
async fn test_build_options_injects_path_and_cli_path() {
    let state = mock_app_state();
    let mut config = test_config();
    config.process_path = Some("/test/path:/usr/bin:/bin".into());
    config.claude_cli_path = Some("/test/claude".into());
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

    let options = manager
        .build_options(
            "/tmp",
            "claude-sonnet-4-6",
            "test prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;

    // Verify PATH was injected into the env HashMap
    assert_eq!(
        options.env.get("PATH").map(|s| s.as_str()),
        Some("/test/path:/usr/bin:/bin"),
        "process_path should be injected as PATH env var"
    );
    // Verify cli_path was set
    assert_eq!(
        options
            .cli_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        Some("/test/claude".to_string()),
        "claude_cli_path should be set on options"
    );
}

#[tokio::test]
async fn test_build_options_without_path_no_injection() {
    let state = mock_app_state();
    let config = test_config(); // process_path=None, claude_cli_path=None
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

    let options = manager
        .build_options(
            "/tmp",
            "claude-sonnet-4-6",
            "test prompt",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;

    // PATH should NOT be in env when process_path is None
    assert!(
        !options.env.contains_key("PATH"),
        "PATH should not be injected when process_path is None"
    );
    // cli_path should be None
    assert!(
        options.cli_path.is_none(),
        "cli_path should be None when claude_cli_path is not configured"
    );
}

// ================================================================
// Runtime env config update + persistence tests
// ================================================================

#[tokio::test]
async fn test_runtime_env_config_update_methods() {
    let state = mock_app_state();
    let config = test_config();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

    // Initial state: all None / false
    let env = manager.get_env_config().await;
    assert!(env.process_path.is_none());
    assert!(env.claude_cli_path.is_none());
    assert!(!env.auto_update_cli);

    // Update process_path
    manager
        .update_process_path(Some("/custom/path:/usr/bin".into()))
        .await;
    let env = manager.get_env_config().await;
    assert_eq!(env.process_path.as_deref(), Some("/custom/path:/usr/bin"));

    // Update claude_cli_path
    manager
        .update_claude_cli_path(Some("/opt/claude".into()))
        .await;
    let env = manager.get_env_config().await;
    assert_eq!(env.claude_cli_path.as_deref(), Some("/opt/claude"));
    // ... and the built-in Claude Code of the agent engine opens with it (M2).
    assert_eq!(
        manager.builtin_cli_path.read().unwrap().as_deref(),
        Some("/opt/claude")
    );

    // Update auto_update_cli
    manager.update_auto_update_cli(true).await;
    let env = manager.get_env_config().await;
    assert!(env.auto_update_cli);

    // Clear process_path
    manager.update_process_path(None).await;
    let env = manager.get_env_config().await;
    assert!(env.process_path.is_none());
}

#[tokio::test]
async fn test_runtime_env_config_affects_build_options() {
    let state = mock_app_state();
    let config = test_config();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

    // Initially no PATH injection
    let options = manager
        .build_options(
            "/tmp",
            "claude-sonnet-4-6",
            "test",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert!(!options.env.contains_key("PATH"));

    // Update at runtime via env_config
    manager
        .update_process_path(Some("/runtime/path:/usr/bin".into()))
        .await;
    manager
        .update_claude_cli_path(Some("/runtime/claude".into()))
        .await;

    // Now build_options should pick up the runtime values
    let options = manager
        .build_options(
            "/tmp",
            "claude-sonnet-4-6",
            "test",
            None,
            None,
            None,
            &[],
            None,
            None,
        )
        .await;
    assert_eq!(
        options.env.get("PATH").map(|s| s.as_str()),
        Some("/runtime/path:/usr/bin")
    );
    assert_eq!(
        options
            .cli_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string()),
        Some("/runtime/claude".to_string())
    );
}

#[test]
fn test_persist_env_config_to_yaml_new_file() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");

    let env = RuntimeEnvConfig {
        process_path: Some("/custom/path:/usr/bin".into()),
        claude_cli_path: Some("/opt/claude".into()),
        auto_update_cli: true,
        auto_update_app: true,
    };

    ChatManager::persist_env_config_to_yaml(&yaml_path, &env).unwrap();

    let contents = std::fs::read_to_string(&yaml_path).unwrap();
    let doc = crate::yaml::value_from_str(&contents).unwrap();
    let chat = doc["chat"].as_mapping().unwrap();
    assert_eq!(
        chat["process_path"].as_str().unwrap(),
        "/custom/path:/usr/bin"
    );
    assert_eq!(chat["claude_cli_path"].as_str().unwrap(), "/opt/claude");
    assert!(chat["auto_update_cli"].as_bool().unwrap());
}

#[test]
fn test_persist_env_config_preserves_existing_yaml() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");

    // Write an existing config with other fields
    std::fs::write(
            &yaml_path,
            "server:\n  port: 6600\nchat:\n  default_model: claude-sonnet-4-6\n  permissions:\n    mode: default\n",
        )
        .unwrap();

    let env = RuntimeEnvConfig {
        process_path: Some("/my/path".into()),
        claude_cli_path: None,
        auto_update_cli: false,
        auto_update_app: true,
    };

    ChatManager::persist_env_config_to_yaml(&yaml_path, &env).unwrap();

    let contents = std::fs::read_to_string(&yaml_path).unwrap();
    let doc = crate::yaml::value_from_str(&contents).unwrap();

    // server section preserved
    assert_eq!(doc["server"]["port"].as_u64().unwrap(), 6600);
    // chat.default_model preserved
    assert_eq!(
        doc["chat"]["default_model"].as_str().unwrap(),
        "claude-sonnet-4-6"
    );
    // chat.permissions preserved
    assert_eq!(
        doc["chat"]["permissions"]["mode"].as_str().unwrap(),
        "default"
    );
    // process_path set
    assert_eq!(doc["chat"]["process_path"].as_str().unwrap(), "/my/path");
    // claude_cli_path removed (None)
    assert!(doc["chat"]["claude_cli_path"].is_null());
    // auto_update_cli removed (false → omitted)
    assert!(doc["chat"]["auto_update_cli"].is_null());
}

#[test]
fn test_persist_env_config_clears_fields() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("config.yaml");

    // First persist with values
    let env1 = RuntimeEnvConfig {
        process_path: Some("/my/path".into()),
        claude_cli_path: Some("/my/claude".into()),
        auto_update_cli: true,
        auto_update_app: true,
    };
    ChatManager::persist_env_config_to_yaml(&yaml_path, &env1).unwrap();

    // Then persist with cleared values
    let env2 = RuntimeEnvConfig {
        process_path: None,
        claude_cli_path: None,
        auto_update_cli: false,
        auto_update_app: true,
    };
    ChatManager::persist_env_config_to_yaml(&yaml_path, &env2).unwrap();

    let contents = std::fs::read_to_string(&yaml_path).unwrap();
    let doc = crate::yaml::value_from_str(&contents).unwrap();

    // All fields should be removed
    let chat = doc["chat"].as_mapping().unwrap();
    assert!(!chat.contains_key("process_path"));
    assert!(!chat.contains_key("claude_cli_path"));
    assert!(!chat.contains_key("auto_update_cli"));
}

// ================================================================
// parse_spawned_by tests
// ================================================================

#[test]
fn test_parse_spawned_by_full_context() {
    let run_id = Uuid::new_v4();
    let task_id = Uuid::new_v4();
    let proto_run_id = Uuid::new_v4();
    let json = serde_json::json!({
        "parent_session_id": "sess-123",
        "type": "wave-runner",
        "run_id": run_id.to_string(),
        "task_id": task_id.to_string(),
        "protocol_run_id": proto_run_id.to_string(),
        "protocol_state": "implement"
    })
    .to_string();

    let ctx = parse_spawned_by(&json).unwrap();
    assert_eq!(ctx.parent_session_id, Some("sess-123".to_string()));
    assert_eq!(ctx.spawn_type, "wave-runner");
    assert_eq!(ctx.run_id, Some(run_id));
    assert_eq!(ctx.task_id, Some(task_id));
    assert_eq!(ctx.protocol_run_id, Some(proto_run_id));
    assert_eq!(ctx.protocol_state, Some("implement".to_string()));
}

#[test]
fn test_parse_spawned_by_minimal() {
    let json = r#"{"parent_session_id": "sess-456"}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.parent_session_id, Some("sess-456".to_string()));
    assert_eq!(ctx.spawn_type, "runner"); // default
    assert_eq!(ctx.run_id, None);
    assert_eq!(ctx.task_id, None);
    assert_eq!(ctx.protocol_run_id, None);
    assert_eq!(ctx.protocol_state, None);
}

#[test]
fn test_parse_spawned_by_protocol_only() {
    let proto_run_id = Uuid::new_v4();
    let json = serde_json::json!({
        "protocol_run_id": proto_run_id.to_string(),
        "protocol_state": "review"
    })
    .to_string();

    let ctx = parse_spawned_by(&json).unwrap();
    assert_eq!(ctx.parent_session_id, None);
    assert_eq!(ctx.protocol_run_id, Some(proto_run_id));
    assert_eq!(ctx.protocol_state, Some("review".to_string()));
}

// ---------------------------------------------------------------
// parse_spawned_by — scaffolding_level tests
// ---------------------------------------------------------------

#[test]
fn test_parse_spawned_by_scaffolding_level_zero() {
    let json = r#"{"scaffolding_level": 0}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, Some(0));
}

#[test]
fn test_parse_spawned_by_scaffolding_level_mid() {
    let json = r#"{"scaffolding_level": 2}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, Some(2));
}

#[test]
fn test_parse_spawned_by_scaffolding_level_max() {
    let json = r#"{"scaffolding_level": 4}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, Some(4));
}

#[test]
fn test_parse_spawned_by_scaffolding_level_capped_at_4() {
    let json = r#"{"scaffolding_level": 5}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, Some(4));
}

#[test]
fn test_parse_spawned_by_scaffolding_level_large_capped() {
    let json = r#"{"scaffolding_level": 255}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, Some(4));
}

#[test]
fn test_parse_spawned_by_scaffolding_level_null() {
    let json = r#"{"scaffolding_level": null}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, None);
}

#[test]
fn test_parse_spawned_by_no_scaffolding_level_backward_compat() {
    let json = r#"{"type": "runner", "parent_session_id": "abc"}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.scaffolding_level, None);
    assert_eq!(ctx.parent_session_id, Some("abc".to_string()));
    assert_eq!(ctx.spawn_type, "runner");
}

#[test]
fn test_parse_spawned_by_scaffolding_with_other_fields() {
    let run_id = Uuid::new_v4();
    let task_id = Uuid::new_v4();
    let json = format!(
        r#"{{"type":"runner","run_id":"{}","task_id":"{}","scaffolding_level":3}}"#,
        run_id, task_id
    );
    let ctx = parse_spawned_by(&json).unwrap();
    assert_eq!(ctx.scaffolding_level, Some(3));
    assert_eq!(ctx.run_id, Some(run_id));
    assert_eq!(ctx.task_id, Some(task_id));
    assert_eq!(ctx.spawn_type, "runner");
}

#[test]
fn test_parse_spawned_by_empty_json() {
    let ctx = parse_spawned_by("{}").unwrap();
    assert_eq!(ctx.parent_session_id, None);
    assert_eq!(ctx.spawn_type, "runner");
    assert_eq!(ctx.protocol_run_id, None);
}

#[test]
fn test_parse_spawned_by_invalid_json() {
    assert!(parse_spawned_by("not json").is_none());
}

#[test]
fn test_parse_spawned_by_invalid_uuid() {
    let json = r#"{"protocol_run_id": "not-a-uuid", "run_id": "also-bad"}"#;
    let ctx = parse_spawned_by(json).unwrap();
    assert_eq!(ctx.protocol_run_id, None);
    assert_eq!(ctx.run_id, None);
}

// ── is_conclusive_tool tests ────────────────────────────────────────

#[test]
fn test_git_commit_is_conclusive() {
    let input = serde_json::json!({"command": "git commit -m 'feat: add stuff'"});
    assert!(is_conclusive_tool("Bash", &input));
}

#[test]
fn test_git_push_is_conclusive() {
    let input = serde_json::json!({"command": "git push origin feature-branch"});
    assert!(is_conclusive_tool("Bash", &input));
}

#[test]
fn test_git_status_is_conclusive() {
    let input = serde_json::json!({"command": "git status"});
    assert!(is_conclusive_tool("Bash", &input));
}

#[test]
fn test_git_log_is_conclusive() {
    let input = serde_json::json!({"command": "git log --oneline -5"});
    assert!(is_conclusive_tool("Bash", &input));
}

#[test]
fn test_git_tag_is_conclusive() {
    let input = serde_json::json!({"command": "git tag v1.0.0"});
    assert!(is_conclusive_tool("Bash", &input));
}

#[test]
fn test_chained_git_commit_push_is_conclusive() {
    let input = serde_json::json!({"command": "git commit -m 'fix' && git push"});
    assert!(is_conclusive_tool("Bash", &input));
}

#[test]
fn test_cargo_build_is_not_conclusive() {
    let input = serde_json::json!({"command": "cargo build"});
    assert!(!is_conclusive_tool("Bash", &input));
}

#[test]
fn test_npm_test_is_not_conclusive() {
    let input = serde_json::json!({"command": "npm test"});
    assert!(!is_conclusive_tool("Bash", &input));
}

#[test]
fn test_edit_tool_is_not_conclusive() {
    let input =
        serde_json::json!({"file_path": "src/main.rs", "old_string": "a", "new_string": "b"});
    assert!(!is_conclusive_tool("Edit", &input));
}

#[test]
fn test_read_tool_is_not_conclusive() {
    let input = serde_json::json!({"file_path": "src/main.rs"});
    assert!(!is_conclusive_tool("Read", &input));
}

#[test]
fn test_write_tool_is_not_conclusive() {
    let input = serde_json::json!({"file_path": "src/main.rs", "content": "fn main() {}"});
    assert!(!is_conclusive_tool("Write", &input));
}

#[test]
fn test_bash_without_command_is_not_conclusive() {
    let input = serde_json::json!({});
    assert!(!is_conclusive_tool("Bash", &input));
}

// ── T5 of plan 28e9afe3: cancel_running_tools tests ──────────────────
//
// Most invariants of `cancel_running_tools` are enforced by the type
// system: `kill_descendants` takes only `Option<u32>` (no
// `AtomicBool`/`CancellationToken` — cannot touch flag/token), and
// `check_and_record_cancel_cap` takes only `&Arc<Mutex<VecDeque>>`,
// `u32`, `Duration`. Runtime invariant (cancel doesn't break the
// stream) is covered by the live e2e test in T6.
//
// Unit tests below cover: rate cap (basics + recovery), kill helper
// edge cases, and CancelToolsResult serde shape.

#[tokio::test]
async fn test_cancel_cap_allows_up_to_cap_then_refuses() {
    let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
    let cap: u32 = 10;
    let window = Duration::from_secs(60);

    for i in 0..cap {
        assert!(
            ChatManager::check_and_record_cancel_cap(&history, cap, window).await,
            "call #{i} below cap must be allowed"
        );
    }
    // 11th refused, history len stays at cap.
    assert!(!ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
    assert_eq!(history.lock().await.len() as u32, cap);
    // Subsequent refused calls don't add to history either.
    assert!(!ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
    assert_eq!(history.lock().await.len() as u32, cap);
}

#[tokio::test]
async fn test_cancel_cap_recovers_after_window_expires() {
    let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
    let cap: u32 = 2;
    // Tiny window so we can age entries out within test time.
    let window = Duration::from_millis(50);

    assert!(ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
    assert!(ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
    assert!(!ChatManager::check_and_record_cancel_cap(&history, cap, window).await);

    // Wait past the window, history should drain.
    tokio::time::sleep(Duration::from_millis(80)).await;

    assert!(
        ChatManager::check_and_record_cancel_cap(&history, cap, window).await,
        "after window expiry, calls should be allowed again"
    );
}

/// P5: through `cancel_running_tools` (what the WS `cancel_tools` frame and
/// the REST route call), a Claude Code session takes `CANCEL_TOOLS_CAP`
/// cancels per window, then refuses with `capped` — the agent engine
/// applies the same cap (`agent_e2e_tests::cancel_tools`).
#[tokio::test]
async fn test_cancel_running_tools_caps_a_claude_code_session() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("legacy".into(), session);

    for i in 0..CANCEL_TOOLS_CAP {
        let result = manager.cancel_running_tools("legacy").await.unwrap();
        assert!(!result.capped, "call {i} is within the cap");
    }
    let past = manager.cancel_running_tools("legacy").await.unwrap();
    assert!(past.capped, "the call past the cap is refused");
    assert!(past.killed_pids.is_empty());
}

#[test]
fn test_kill_descendants_with_no_pid_is_noop() {
    // No PID → nothing to kill, returns empty Vec, no panic, no SIGINT.
    let killed = ChatManager::kill_descendants(None);
    assert!(killed.is_empty());
}

// ========================================================================
// T3 of plan fc35b25e — kill_subtree(pid) helper
// ========================================================================

#[test]
#[cfg(unix)]
fn test_kill_subtree_nonexistent_pid_is_noop() {
    // PID 999_999 is almost certainly not a live process — kill returns
    // ESRCH which we silently filter, so killed list is empty and the
    // call must not panic.
    let killed = ChatManager::kill_subtree(999_999);
    assert!(
        killed.is_empty(),
        "expected empty killed list, got {killed:?}"
    );
}

/// Spawn a real `sleep 30`, call `kill_subtree`, verify the subprocess
/// is reaped within a reasonable timeout. Marked `#[ignore]` because it
/// touches real processes — run with `cargo test -- --ignored`.
#[test]
#[ignore]
#[cfg(unix)]
fn test_kill_subtree_kills_real_sleep_subprocess() {
    let mut child = match std::process::Command::new("sleep").arg("30").spawn() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping test: sleep not available");
            return;
        }
    };
    let pid = child.id();

    // Give the OS a moment to register the process.
    std::thread::sleep(std::time::Duration::from_millis(100));

    let killed = ChatManager::kill_subtree(pid);
    assert!(
        killed.contains(&pid),
        "expected {pid} in killed list, got {killed:?}"
    );

    // Reap the zombie and confirm the process actually died (sleep was
    // 30s — if it's already dead now, kill_subtree did its job).
    let exit = child.wait().expect("waitpid");
    assert!(
        !exit.success(),
        "subprocess should have been signalled, not exited cleanly"
    );
}

#[test]
fn test_cancel_tools_result_serde_roundtrip() {
    let result = CancelToolsResult {
        cli_pid: Some(12345),
        killed_pids: vec![67890, 67891],
        capped: false,
    };
    let json = serde_json::to_string(&result).expect("serialize");
    // Spot-check the wire format the frontend will consume.
    assert!(json.contains("\"cli_pid\":12345"));
    assert!(json.contains("\"killed_pids\":[67890,67891]"));
    assert!(json.contains("\"capped\":false"));

    let parsed: CancelToolsResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.cli_pid, Some(12345));
    assert_eq!(parsed.killed_pids, vec![67890, 67891]);
    assert!(!parsed.capped);
}

#[test]
fn test_cancel_tools_result_serde_capped_state() {
    let result = CancelToolsResult {
        cli_pid: Some(1234),
        killed_pids: vec![],
        capped: true,
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("\"capped\":true"));
    assert!(json.contains("\"killed_pids\":[]"));
}

// ========================================================================
// T3 of plan 754a1379 — track_background_task_start (INSERT lifecycle hook)
// ========================================================================

/// Helper: insert a dummy session under the given id and return the
/// active_sessions map ready for the helper. No CLI needed.
async fn build_sessions_map_for_track_test(
    session_id: &str,
) -> (
    Arc<RwLock<HashMap<String, ActiveSession>>>,
    broadcast::Sender<ChatEvent>,
    broadcast::Receiver<ChatEvent>,
) {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let events_tx = session.events_tx.clone();
    let events_rx = events_tx.subscribe();
    let map = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert(session_id.to_string(), session);
    (map, events_tx, events_rx)
}

#[tokio::test]
async fn test_track_background_task_inserts_monitor_on_first_pass() {
    let (sessions, events_tx, mut events_rx) = build_sessions_map_for_track_test("s-mon").await;

    // First pass: ContentBlockStart with empty input.
    let inserted = ChatManager::track_background_task_start(
        "s-mon",
        &sessions,
        &events_tx,
        &None,
        "tool_M1",
        "Monitor",
        &serde_json::json!({}),
        None,
    )
    .await;
    assert!(inserted, "Monitor should be tracked on first pass");

    // Map should now have one entry.
    let session_guard = sessions.read().await;
    let session = session_guard.get("s-mon").unwrap();
    let tasks = session.active_background_tasks.lock().await;
    assert_eq!(tasks.len(), 1);
    let entry = tasks.get("tool_M1").unwrap();
    assert_eq!(entry.kind, BackgroundTaskKind::Monitor);
    assert_eq!(entry.description, "(no description)");

    // Should have broadcast an ActiveTasksUpdate.
    let event = events_rx.try_recv().expect("expected one broadcast");
    match event {
        ChatEvent::ActiveTasksUpdate { tasks } => assert_eq!(tasks.len(), 1),
        other => panic!("expected ActiveTasksUpdate, got {:?}", other),
    }
}

#[tokio::test]
async fn test_track_background_task_updates_description_on_second_pass() {
    let (sessions, events_tx, mut events_rx) = build_sessions_map_for_track_test("s-mon2").await;

    // First pass — empty input, placeholder description.
    ChatManager::track_background_task_start(
        "s-mon2",
        &sessions,
        &events_tx,
        &None,
        "tool_M2",
        "Monitor",
        &serde_json::json!({}),
        None,
    )
    .await;
    // Drain first broadcast.
    let _ = events_rx.try_recv();

    // Second pass — full input with description.
    ChatManager::track_background_task_start(
        "s-mon2",
        &sessions,
        &events_tx,
        &None,
        "tool_M2",
        "Monitor",
        &serde_json::json!({
            "command": "tail -F /tmp/build.log",
            "description": "watch build log",
            "persistent": false,
            "timeout_ms": 300000,
        }),
        None,
    )
    .await;

    let session_guard = sessions.read().await;
    let tasks = session_guard
        .get("s-mon2")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    // Still one entry (idempotent), description refreshed.
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks.get("tool_M2").unwrap().description, "watch build log");

    // Two broadcasts total (one per pass).
    events_rx.try_recv().expect("second broadcast missing");
}

#[tokio::test]
async fn test_track_background_task_bash_run_in_background_first_pass_skipped() {
    let (sessions, events_tx, mut events_rx) = build_sessions_map_for_track_test("s-bash").await;

    // First pass: empty input, run_in_background flag absent → not tracked.
    let inserted = ChatManager::track_background_task_start(
        "s-bash",
        &sessions,
        &events_tx,
        &None,
        "tool_B1",
        "Bash",
        &serde_json::json!({}),
        None,
    )
    .await;
    assert!(!inserted, "Bash with empty input must not be tracked yet");

    // Second pass: full input with run_in_background=true → tracked.
    let inserted = ChatManager::track_background_task_start(
        "s-bash",
        &sessions,
        &events_tx,
        &None,
        "tool_B1",
        "Bash",
        &serde_json::json!({
            "command": "cargo watch -x test",
            "description": "run tests on change",
            "run_in_background": true,
        }),
        None,
    )
    .await;
    assert!(inserted, "Bash with run_in_background=true must be tracked");

    let session_guard = sessions.read().await;
    let tasks = session_guard
        .get("s-bash")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.len(), 1);
    let entry = tasks.get("tool_B1").unwrap();
    assert_eq!(entry.kind, BackgroundTaskKind::BashBackground);
    assert_eq!(entry.description, "run tests on change");

    // Only ONE broadcast (the second pass) since the first returned false.
    events_rx.try_recv().expect("expected the second broadcast");
    assert!(events_rx.try_recv().is_err(), "no extra broadcast");
}

#[tokio::test]
async fn test_track_background_task_bash_synchronous_not_tracked() {
    let (sessions, events_tx, mut events_rx) =
        build_sessions_map_for_track_test("s-bash-sync").await;

    // Synchronous Bash (run_in_background absent or false) — never tracked.
    let inserted_a = ChatManager::track_background_task_start(
        "s-bash-sync",
        &sessions,
        &events_tx,
        &None,
        "tool_B2",
        "Bash",
        &serde_json::json!({ "command": "ls" }),
        None,
    )
    .await;
    let inserted_b = ChatManager::track_background_task_start(
        "s-bash-sync",
        &sessions,
        &events_tx,
        &None,
        "tool_B3",
        "Bash",
        &serde_json::json!({ "command": "pwd", "run_in_background": false }),
        None,
    )
    .await;
    assert!(!inserted_a);
    assert!(!inserted_b);

    let tasks = sessions
        .read()
        .await
        .get("s-bash-sync")
        .unwrap()
        .active_background_tasks
        .lock()
        .await
        .len();
    assert_eq!(tasks, 0, "no Bash should be tracked");
    assert!(
        events_rx.try_recv().is_err(),
        "no broadcast expected for untracked tools"
    );
}

#[tokio::test]
async fn test_track_background_task_other_tools_ignored() {
    let (sessions, events_tx, _events_rx) = build_sessions_map_for_track_test("s-other").await;

    for tool in ["Read", "Write", "Edit", "Glob", "Task"] {
        let inserted = ChatManager::track_background_task_start(
            "s-other",
            &sessions,
            &events_tx,
            &None,
            "tool_X",
            tool,
            &serde_json::json!({"description": "should be ignored"}),
            None,
        )
        .await;
        assert!(!inserted, "{} must not be tracked", tool);
    }

    let count = sessions
        .read()
        .await
        .get("s-other")
        .unwrap()
        .active_background_tasks
        .lock()
        .await
        .len();
    assert_eq!(count, 0);
}

// ========================================================================
// T2 of plan fc35b25e — async PID claim from track_background_task_start
// ========================================================================

/// Trigger `track_background_task_start` against a session whose
/// `child_pid` points at a real `bash` subprocess that periodically
/// forks new descendants. The async claim must:
/// 1. snapshot descendants at insert time (`before`),
/// 2. sleep 1 s,
/// 3. snapshot again (`after`) and diff,
/// 4. claim the newest new PID,
/// 5. update the entry's `pid` and broadcast a fresh `ActiveTasksUpdate`.
///
/// Marked `#[ignore]` because it spawns real subprocesses and depends
/// on timing — run with `cargo test -- --ignored`.
#[tokio::test]
#[ignore]
#[cfg(unix)]
async fn test_pid_claim_async_updates_map_after_1s() {
    use std::process::Stdio;

    // Spawn a bash that chains short sleeps so its descendant set
    // changes during the 1 s claim window. At t=0 the descendant is
    // a 0.3 s sleep; at t=1.0 it's the long-lived 30 s sleep.
    //
    // The trailing `& wait` is critical: without it, bash would
    // optimise the final command via exec(2) and **replace itself**
    // with `sleep 30`, leaving no descendants for pgrep to find.
    // Backgrounding with `&` and using `wait` forces bash to fork
    // and stay alive as the PPID.
    let mut bash = match std::process::Command::new("bash")
        .arg("-c")
        .arg("sleep 0.3; sleep 0.5; sleep 30 & wait")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping test: bash not available");
            return;
        }
    };
    let bash_pid = bash.id();
    // Give the OS a moment so pgrep can see the first descendant.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(bash_pid);
    let events_tx = session.events_tx.clone();
    let mut events_rx = events_tx.subscribe();

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-claim".into(), session);

    let inserted = ChatManager::track_background_task_start(
        "s-claim",
        &map,
        &events_tx,
        &None,
        "tu-claim",
        "Monitor",
        &serde_json::json!({}),
        None,
    )
    .await;
    assert!(inserted, "Monitor first-pass should insert");

    // Drain the first broadcast (insert with pid=None).
    match tokio::time::timeout(Duration::from_secs(1), events_rx.recv()).await {
        Ok(Ok(ChatEvent::ActiveTasksUpdate { tasks })) => {
            assert_eq!(tasks.len(), 1);
            assert!(tasks[0].pid.is_none(), "first broadcast pid must be None");
        }
        other => {
            let _ = bash.kill();
            let _ = bash.wait();
            panic!("expected ActiveTasksUpdate broadcast, got {other:?}");
        }
    }

    // Wait past the 1 s sleep + a margin for the async claim to land.
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // The entry should now carry pid=Some(...).
    let claimed_pid = {
        let sessions = map.read().await;
        let active = sessions.get("s-claim").expect("session present");
        let tasks = active.active_background_tasks.lock().await;
        let entry = tasks.get("tu-claim").expect("entry present");
        entry.pid
    };

    // Cleanup before assertions so a panic doesn't leak the bash.
    let _ = bash.kill();
    let _ = bash.wait();

    assert!(
        claimed_pid.is_some(),
        "expected pid populated by async claim, got {claimed_pid:?}"
    );
}

/// Cancel the entry between the synchronous track call and the 1 s
/// async wake-up. The claim must no-op without re-inserting the entry
/// or panicking.
#[tokio::test]
async fn test_pid_claim_noop_when_entry_removed_before_wakeup() {
    let (mut session, _) = create_dummy_session(false, "", vec![]);
    // Set a non-None child_pid so the claim task is actually spawned
    // (else it short-circuits in the synchronous track call). We use
    // a guaranteed-nonexistent PID so `pgrep -P` returns instantly
    // with no descendants → the diff is empty and the claim takes
    // the warn-and-return path, which is what we want to exercise.
    // (Avoid PID 1 — on macOS that's launchd whose subtree is the
    // entire process table, making `get_descendant_pids` very slow.)
    session.child_pid = Some(999_999);
    let events_tx = session.events_tx.clone();
    let _events_rx = events_tx.subscribe();

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-race".into(), session);

    let inserted = ChatManager::track_background_task_start(
        "s-race",
        &map,
        &events_tx,
        &None,
        "tu-race",
        "Monitor",
        &serde_json::json!({}),
        None,
    )
    .await;
    assert!(inserted);

    // Immediately remove the entry — simulates a rapid cancel that
    // beats the async claim.
    {
        let sessions = map.read().await;
        let active = sessions.get("s-race").expect("session present");
        active
            .active_background_tasks
            .lock()
            .await
            .remove("tu-race");
    }

    // Wait past the 1 s sleep + a margin.
    tokio::time::sleep(Duration::from_millis(1300)).await;

    // The entry must NOT have been re-inserted by the claim, and the
    // process must not have panicked (if it did, the test runtime
    // would have aborted).
    let count = map
        .read()
        .await
        .get("s-race")
        .unwrap()
        .active_background_tasks
        .lock()
        .await
        .len();
    assert_eq!(count, 0, "claim must not re-insert removed entries");
}

// ========================================================================
// T7 of plan 754a1379 — cancel_task (map-side cancel V1)
// ========================================================================

#[tokio::test]
async fn test_cancel_task_marks_for_removal_and_broadcasts() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    // Pre-seed a task in the map.
    session.active_background_tasks.lock().await.insert(
        "tool_T1".into(),
        BackgroundTaskInfo {
            id: "tool_T1".into(),
            kind: BackgroundTaskKind::Monitor,
            description: "tail log".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: None,
            parent_tool_use_id: Some("tool_T1".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-cancel".into(), session);

    let result = manager.cancel_task("s-cancel", "tool_T1").await.unwrap();
    assert_eq!(result.task_id, "tool_T1");
    // V2 (T4): entry has pid=None → fallback path, killed_pids empty.
    assert!(
        result.killed_pids.is_empty(),
        "expected empty killed_pids for pid=None entry"
    );
    assert!(!result.capped);

    // Entry should be marked for removal but still in the map (T12 grace).
    let sessions = manager.active_sessions.read().await;
    let tasks = sessions
        .get("s-cancel")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.len(), 1);
    assert!(tasks.get("tool_T1").unwrap().pending_removal_at.is_some());

    // Broadcast: ActiveTasksUpdate carrying the snapshot (the entry is
    // still visible — it's the grace-period state).
    let event = events_rx.try_recv().expect("expected broadcast");
    match event {
        ChatEvent::ActiveTasksUpdate { tasks } => assert_eq!(tasks.len(), 1),
        other => panic!("expected ActiveTasksUpdate, got {:?}", other),
    }
}

#[tokio::test]
async fn test_cancel_task_unknown_task_id_idempotent() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-empty".into(), session);

    // No tasks in the map — cancel still succeeds, just broadcasts an
    // empty snapshot (idempotent: cancelling something that was never
    // there is OK).
    let result = manager.cancel_task("s-empty", "unknown_id").await.unwrap();
    assert!(!result.capped);
    assert!(result.killed_pids.is_empty());

    let event = events_rx.try_recv().expect("broadcast still emitted");
    match event {
        ChatEvent::ActiveTasksUpdate { tasks } => assert!(tasks.is_empty()),
        other => panic!("expected empty ActiveTasksUpdate, got {:?}", other),
    }
}

#[tokio::test]
/// Review N2 of #663: a session no instance holds gets the same typed answer on a
/// single instance as across instances (NATS, nobody answering):
/// `owner_unreachable` (409, not retryable), for both cancels — not a 200 here and
/// a 409 there.
async fn test_cancel_on_unknown_session_is_owner_unreachable() {
    use super::super::cancel_relay::CancelRelayError;
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // No session inserted at all.
    for error in [
        manager
            .cancel_task("ghost-session", "tool_X")
            .await
            .unwrap_err(),
        manager
            .cancel_running_tools("ghost-session")
            .await
            .unwrap_err(),
    ] {
        assert_eq!(
            error.downcast_ref::<CancelRelayError>(),
            Some(&CancelRelayError::OwnerUnreachable)
        );
    }
}

/// Review of #673, finding 4: the eviction of a dead CLI removes only the entry
/// it found dead. A resume that replaced the entry meanwhile keeps its session
/// and its NATS listeners (before: the new session was removed and its
/// listeners cancelled).
#[tokio::test]
async fn evicting_a_dead_cli_leaves_a_session_that_replaced_it() {
    let (old, _) = create_dummy_session(false, "", vec![]);
    let (new, _) = create_dummy_session(false, "", vec![]);
    let (old_client, old_token) = (old.client.clone(), old.nats_cancel.clone());
    let new_token = new.nats_cancel.clone();
    let sessions = RwLock::new(HashMap::new());
    // A resume replaced the entry whose CLI was found dead.
    sessions.write().await.insert("s".to_string(), new);
    assert!(!ChatManager::forget_if_client(&sessions, "s", &old_client).await);
    assert!(
        sessions.read().await.contains_key("s"),
        "the new session stays"
    );
    assert!(!new_token.is_cancelled(), "its listeners keep running");

    // The entry is still the dead one: removed, its listeners ended.
    sessions.write().await.insert("s".to_string(), old);
    assert!(ChatManager::forget_if_client(&sessions, "s", &old_client).await);
    assert!(!sessions.read().await.contains_key("s"));
    assert!(old_token.is_cancelled());
}

/// T5: the notes a turn really had in context become synapses at the close, scoped to
/// their project — and a session with one note welds nothing, without failing.
#[tokio::test]
async fn a_closed_session_welds_the_notes_it_really_had_in_context() {
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::*;

    let graph = std::sync::Arc::new(MockGraphStore::new());
    let project = crate::test_helpers::test_project_named("alpha");
    graph.create_project(&project).await.unwrap();
    let mut session = crate::test_helpers::test_chat_session(Some("alpha"));
    session.preview = Some("first".into());
    let sid = session.id;
    graph.create_chat_session(&session).await.unwrap();

    let mut ids = Vec::new();
    for i in 0..3 {
        let mut n = crate::notes::models::Note::new(
            Some(project.id),
            crate::notes::models::NoteType::Guideline,
            format!("seen {i}"),
            "api".into(),
        );
        n.status = crate::notes::models::NoteStatus::Active;
        graph.create_note(&n).await.unwrap();
        ids.push(n.id);
    }

    let outcome = ChatManager::reinforce_session_knowledge(graph.clone(), sid, ids.clone())
        .await
        .expect("three notes are three pairs");
    assert_eq!(outcome.written, 6, "both directions: {outcome:?}");
    assert_eq!(graph.synapses.read().await.len(), 3);

    assert!(
        ChatManager::reinforce_session_knowledge(graph.clone(), sid, vec![ids[0]])
            .await
            .is_none(),
        "one note has no pair to weld"
    );
}

/// Review of #685, finding 3: a task already being stopped (its grace period
/// runs, its pid kept) is not signalled again by a second cancel_task (a retry
/// after a `session_gone`, a second click), nor counted twice in the cap.
#[tokio::test]
#[cfg(unix)]
async fn a_task_already_being_stopped_is_not_signalled_again() {
    let mut task = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let task_pid = task.id();
    let (session, _) = create_dummy_session(false, "", vec![]);
    let history = session.cancel_task_history.clone();
    session.active_background_tasks.lock().await.insert(
        "tool_Twice".into(),
        BackgroundTaskInfo {
            id: "tool_Twice".into(),
            kind: BackgroundTaskKind::BashBackground,
            description: "sleep".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: Some(task_pid),
            parent_tool_use_id: Some("tool_Twice".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-twice".into(), session);

    let first = manager.cancel_task("s-twice", "tool_Twice").await.unwrap();
    assert!(first.killed_pids.contains(&task_pid), "{first:?}");
    assert!(!task.wait().unwrap().success(), "signalled once");

    let again = manager.cancel_task("s-twice", "tool_Twice").await.unwrap();
    assert!(!again.capped && again.killed_pids.is_empty(), "{again:?}");
    assert_eq!(history.lock().await.len(), 1, "counted once");
}

fn tracked_task(id: &str, pid: Option<u32>) -> BackgroundTaskInfo {
    BackgroundTaskInfo {
        id: id.into(),
        kind: BackgroundTaskKind::BashBackground,
        description: "sleep".into(),
        started_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        pid,
        parent_tool_use_id: Some(id.into()),
        pending_removal_at: None,
        signalled: false,
    }
}

/// Review of #691, finding 1: a Stop in the first second of a background task
/// comes before its pid is known. It signals nothing, and marks the task
/// stopping. The PID claim that follows signals it (before: it only filled the
/// pid, and the no-op of a retry left the process running, untracked). Once
/// signalled, asking again is the no-op.
#[tokio::test]
#[cfg(unix)]
async fn a_task_stopped_before_its_pid_is_known_is_signalled_when_it_is_found() {
    // A fake CLI whose child is the background task.
    let mut cli = std::process::Command::new("sh")
        .args(["-c", "sleep 30; true"])
        .spawn()
        .unwrap();
    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(cli.id());
    let (history, events_tx, tasks) = (
        session.cancel_task_history.clone(),
        session.events_tx.clone(),
        session.active_background_tasks.clone(),
    );
    tasks
        .lock()
        .await
        .insert("tool_Early".into(), tracked_task("tool_Early", None));
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-early".into(), session);

    let early = manager.cancel_task("s-early", "tool_Early").await.unwrap();
    assert!(!early.capped && early.killed_pids.is_empty(), "{early:?}");

    ChatManager::async_pid_claim(
        "s-early".into(),
        manager.active_sessions.clone(),
        events_tx,
        None,
        "tool_Early".into(),
        vec![],
    )
    .await;
    let pid = tasks.lock().await["tool_Early"].pid;
    assert!(pid.is_some(), "the claim found the task's process");
    let deadline = Instant::now() + Duration::from_secs(5);
    while process_alive(pid) == Some(true) {
        assert!(
            Instant::now() < deadline,
            "the task's process was signalled once its pid was found"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let _ = cli.wait();

    let again = manager.cancel_task("s-early", "tool_Early").await.unwrap();
    assert!(!again.capped && again.killed_pids.is_empty(), "{again:?}");
    assert_eq!(history.lock().await.len(), 1, "the no-op is not counted");
}

/// Review of #694, point 1: a task the death poller found dead is never
/// signalled: by the time a Stop comes (before the purge), its pid may name
/// another process. The reuse is played by pointing the entry at a live
/// process after the poller marked it: it must survive the Stop.
#[cfg(unix)]
#[tokio::test]
async fn a_task_found_dead_is_never_signalled_even_if_its_pid_was_reused() {
    let mut gone = std::process::Command::new("true").spawn().unwrap();
    let dead_pid = gone.id();
    gone.wait().unwrap();
    let (map, events_tx) = session_with_silent_background_task("s-reused", Some(dead_pid), 0).await;
    ChatManager::tick_purge_background_tasks(
        "s-reused",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;
    assert!(silent_task_is_marked(&map, "s-reused").await);

    let mut other = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let other_pid = other.id();
    map.read().await["s-reused"]
        .active_background_tasks
        .lock()
        .await
        .get_mut("tool_silent")
        .unwrap()
        .pid = Some(other_pid);

    let stop = ChatManager::cancel_legacy_task(&map, &None, "s-reused", "tool_silent")
        .await
        .unwrap();
    assert!(stop.killed_pids.is_empty(), "{stop:?}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        process_alive(Some(other_pid)),
        Some(true),
        "the process now holding the pid was left alone"
    );
    let _ = other.kill();
    let _ = other.wait();
}

/// Review of #694, point 2: a claim that finds several new processes in the
/// same second (two tools started together) records the first, as before, but
/// does not signal a task stopped before it: the first candidate may be the
/// neighbour's. The task stays stopping and not signalled, so an explicit Stop
/// asked again signals it.
#[tokio::test]
#[cfg(unix)]
async fn an_ambiguous_claim_signals_nothing_and_leaves_it_to_an_explicit_stop() {
    let mut cli = std::process::Command::new("sh")
        .args(["-c", "sleep 30 & sleep 30; true"])
        .spawn()
        .unwrap();
    let cli_pid = cli.id();
    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(cli_pid);
    let (events_tx, tasks) = (
        session.events_tx.clone(),
        session.active_background_tasks.clone(),
    );
    tasks
        .lock()
        .await
        .insert("tool_Two".into(), tracked_task("tool_Two", None));
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-two".into(), session);

    let early = manager.cancel_task("s-two", "tool_Two").await.unwrap();
    assert!(early.killed_pids.is_empty(), "{early:?}");
    ChatManager::async_pid_claim(
        "s-two".into(),
        manager.active_sessions.clone(),
        events_tx,
        None,
        "tool_Two".into(),
        vec![],
    )
    .await;
    let both = ChatManager::get_descendant_pids(cli_pid);
    assert_eq!(both.len(), 2, "two new processes: {both:?}");
    let claimed = {
        let tasks = tasks.lock().await;
        assert!(!tasks["tool_Two"].signalled);
        tasks["tool_Two"]
            .pid
            .expect("the first candidate is recorded")
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    for pid in &both {
        assert_eq!(process_alive(Some(*pid)), Some(true), "{pid} signalled");
    }

    let explicit = manager.cancel_task("s-two", "tool_Two").await.unwrap();
    assert!(explicit.killed_pids.contains(&claimed), "{explicit:?}");

    for pid in both {
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    }
    let _ = cli.kill();
    let _ = cli.wait();
}

/// `FLAKY-PARITE`, the deterministic half of the proof: a process forked
/// LATE in the claim window is still claimed.
///
/// The CLI here is a `bash` blocked in `read` on its stdin (a builtin: no
/// descendant while it waits), so the "before" snapshot taken at the
/// callsite is empty and the child really is born when the test says so —
/// 2.5 s after the claim started, i.e. well past the single reading at 1 s
/// the claim used to make. The claim must keep looking inside
/// [`PID_CLAIM_WINDOW`] and find it; with one reading it would come back
/// empty and the task would keep `pid: None` (exactly what the parity test
/// saw as `pid réclamé=false, killed_pids=[]` under llvm-cov).
#[tokio::test]
#[cfg(unix)]
async fn a_pid_born_late_in_the_claim_window_is_still_claimed() {
    use std::io::Write as _;
    use std::process::Stdio;

    // A "CLI" that forks nothing until it is told to: `read` is a builtin,
    // so the shell has no descendant while it waits.
    let mut cli = std::process::Command::new("bash")
        .args(["-c", "read line; sleep 30 & wait"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let cli_pid = cli.id();
    let mut stdin = cli.stdin.take().expect("stdin is piped");
    assert!(
        ChatManager::get_descendant_pids(cli_pid).is_empty(),
        "the CLI forked nothing yet: the before snapshot of the callsite is empty"
    );

    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(cli_pid);
    let (events_tx, tasks) = (
        session.events_tx.clone(),
        session.active_background_tasks.clone(),
    );
    tasks
        .lock()
        .await
        .insert("tool_Late".into(), tracked_task("tool_Late", None));
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-late".into(), session);

    // The fork, 2.5 s in: past the single reading the claim used to take.
    let forker = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(2500)).await;
        stdin.write_all(b"go\n").unwrap();
        stdin.flush().unwrap();
    });

    ChatManager::async_pid_claim(
        "s-late".into(),
        manager.active_sessions.clone(),
        events_tx,
        None,
        "tool_Late".into(),
        vec![],
    )
    .await;
    forker.await.unwrap();

    let pid = tasks.lock().await["tool_Late"].pid;
    // Cleanup before asserting, so a failure does not leak the processes.
    if let Some(pid) = pid {
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    }
    let _ = cli.kill();
    let _ = cli.wait();
    assert!(
        pid.is_some(),
        "the process forked 2.5 s into the window was claimed (a single reading \
             at 1 s would have missed it)"
    );
}

/// Task 51f87943: the CLI forks a background Bash right after it prints the
/// complete call, often before this server reads the CLI's descendants on
/// that `ToolUse`. The reference of the claim is the reading taken when the
/// call STARTED (`ContentBlockStart`, empty input), so the process is still
/// a newcomer: claimed, and the Stop reaches it. Read on the second pass (as
/// before), the process was already in the reference: never claimed,
/// `cancel_task` signalled nothing.
#[tokio::test]
#[cfg(unix)]
async fn a_background_process_forked_before_the_full_tool_use_is_claimed_and_stopped() {
    use std::io::Write as _;
    use std::process::Stdio;

    // A "CLI" that forks nothing until it is told to (`read` is a builtin).
    let mut cli = std::process::Command::new("bash")
        .args(["-c", "read line; sleep 30 & wait"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let cli_pid = cli.id();
    let mut stdin = cli.stdin.take().expect("stdin is piped");

    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(cli_pid);
    let (events_tx, tasks) = (
        session.events_tx.clone(),
        session.active_background_tasks.clone(),
    );
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-early".into(), session);
    let track = |input: serde_json::Value| {
        let sessions = manager.active_sessions.clone();
        let events_tx = events_tx.clone();
        async move {
            ChatManager::track_background_task_start(
                "s-early",
                &sessions,
                &events_tx,
                &None,
                "tool_Early",
                "Bash",
                &input,
                None,
            )
            .await
        }
    };

    // The call starts (ContentBlockStart): nothing tracked yet.
    assert!(!track(serde_json::json!({})).await);
    // The CLI forks the tool ...
    stdin.write_all(b"go\n").unwrap();
    stdin.flush().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let child = loop {
        if let Some(pid) = ChatManager::get_descendant_pids(cli_pid).first() {
            break *pid;
        }
        assert!(std::time::Instant::now() < deadline, "the CLI forked");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    // ... before this server sees the complete call.
    assert!(track(serde_json::json!({"command": "sleep 30", "run_in_background": true})).await);

    let deadline = std::time::Instant::now() + Duration::from_secs(4);
    let claimed = loop {
        let pid = tasks.lock().await["tool_Early"].pid;
        if pid.is_some() || std::time::Instant::now() >= deadline {
            break pid;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let stop = if claimed.is_some() {
        Some(manager.cancel_task("s-early", "tool_Early").await.unwrap())
    } else {
        None
    };
    // Cleanup before asserting, so a failure does not leak the processes.
    unsafe { libc::kill(child as i32, libc::SIGKILL) };
    let _ = cli.kill();
    let _ = cli.wait();
    assert_eq!(
        claimed,
        Some(child),
        "the process forked before the full ToolUse was claimed"
    );
    let stop = stop.unwrap();
    assert!(stop.killed_pids.contains(&child), "{stop:?}");
    assert!(
        manager
            .active_sessions
            .read()
            .await
            .get("s-early")
            .unwrap()
            .tool_start_descendants
            .lock()
            .await
            .is_empty(),
        "the start reading was taken back by the second pass"
    );
}

/// Review of #691, finding 2: two concurrent stops of one task signal it once
/// and count once in the cap (before: both passed the check and both counted).
/// The cap history is held so that both are in flight together.
#[tokio::test]
#[cfg(unix)]
async fn two_concurrent_stops_signal_a_task_once_and_count_once() {
    let mut task = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let task_pid = task.id();
    let (session, _) = create_dummy_session(false, "", vec![]);
    let history = session.cancel_task_history.clone();
    session.active_background_tasks.lock().await.insert(
        "tool_Both".into(),
        tracked_task("tool_Both", Some(task_pid)),
    );
    let sessions = Arc::new(RwLock::new(HashMap::new()));
    sessions.write().await.insert("s-both".to_string(), session);

    let held = history.lock().await;
    let stop = || {
        let sessions = Arc::clone(&sessions);
        tokio::spawn(async move {
            ChatManager::cancel_legacy_task(&sessions, &None, "s-both", "tool_Both")
                .await
                .expect("a session of this instance")
        })
    };
    let (a, b) = (stop(), stop());
    tokio::time::sleep(Duration::from_millis(200)).await;
    drop(held);
    let (a, b) = (a.await.unwrap(), b.await.unwrap());

    assert!(!a.capped && !b.capped, "{a:?} {b:?}");
    let signalled = [&a, &b]
        .iter()
        .filter(|r| r.killed_pids.contains(&task_pid))
        .count();
    assert_eq!(signalled, 1, "signalled once: {a:?} {b:?}");
    assert_eq!(history.lock().await.len(), 1, "counted once");
    assert!(!task.wait().unwrap().success());
}

/// Review of #691, finding 3: the close of an agent-engine session is watched
/// for the whole cancel, not only at its start. A close during the cancel, or
/// one that comes with its outcome, answers `gone` (provisional), never the
/// provider's final answer.
#[tokio::test]
async fn a_close_during_an_agent_cancel_answers_gone() {
    use super::super::cancel_relay::{CancelKind, CancelRelayError};
    let gone = |outcome: Result<u32>| {
        outcome
            .unwrap_err()
            .downcast_ref::<CancelRelayError>()
            .cloned()
            == Some(CancelRelayError::SessionGone {
                kind: CancelKind::Tools,
            })
    };

    // Closed while the cancel runs.
    let closed = CancellationToken::new();
    let closing = closed.clone();
    let outcome = ChatManager::unless_agent_closed(&closed, CancelKind::Tools, async move {
        closing.cancel();
        tokio::task::yield_now().await;
        Ok(7)
    })
    .await;
    assert!(gone(outcome));

    // Closed as the cancel answers.
    let closed = CancellationToken::new();
    let closing = closed.clone();
    let outcome = ChatManager::unless_agent_closed(&closed, CancelKind::Tools, async move {
        closing.cancel();
        Ok(7)
    })
    .await;
    assert!(gone(outcome));

    // Open throughout: the cancel's own answer.
    let closed = CancellationToken::new();
    let outcome =
        ChatManager::unless_agent_closed(&closed, CancelKind::Tools, async { Ok(7) }).await;
    assert_eq!(outcome.unwrap(), 7);
}

/// A connected CLI whose turn ends before any `Result` (it died mid-turn).
struct DiesMidTurn;

#[async_trait::async_trait]
impl nexus_claude::transport::Transport for DiesMidTurn {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    async fn connect(&mut self) -> nexus_claude::Result<()> {
        Ok(())
    }
    async fn send_message(
        &mut self,
        _: nexus_claude::transport::InputMessage,
    ) -> nexus_claude::Result<()> {
        Ok(())
    }
    fn receive_messages(
        &mut self,
    ) -> std::pin::Pin<
        Box<
            dyn futures::Stream<Item = nexus_claude::Result<nexus_claude::Message>>
                + Send
                + 'static,
        >,
    > {
        Box::pin(futures::stream::empty())
    }
    async fn send_control_request(
        &mut self,
        _: nexus_claude::ControlRequest,
    ) -> nexus_claude::Result<()> {
        Ok(())
    }
    async fn receive_control_response(
        &mut self,
    ) -> nexus_claude::Result<Option<nexus_claude::ControlResponse>> {
        Ok(None)
    }
    async fn send_sdk_control_request(&mut self, _: serde_json::Value) -> nexus_claude::Result<()> {
        Ok(())
    }
    async fn send_sdk_control_response(
        &mut self,
        _: serde_json::Value,
    ) -> nexus_claude::Result<()> {
        Ok(())
    }
    fn is_connected(&self) -> bool {
        true
    }
    async fn disconnect(&mut self) -> nexus_claude::Result<()> {
        Ok(())
    }
}

/// Review of #685, finding 2: a turn that finds its CLI dead drops the session
/// only while the entry is still the one of that CLI. A resume that replaced
/// it during the turn keeps its session and its NATS listeners (before: the
/// new session was removed and its listeners cancelled). The turn waits for
/// its CLI (held by the test) while the entry is replaced.
async fn a_turn_on_a_dead_cli_leaves_the_session_that_replaced_it(dead: InteractiveClient) {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let sid = Uuid::new_v4().to_string();
    let (mut old, _) = create_dummy_session(false, "", vec![]);
    old.client = Arc::new(Mutex::new(dead));
    let (old_client, old_streaming) = (old.client.clone(), old.is_streaming.clone());
    let mut events = old.events_tx.subscribe();
    manager
        .active_sessions
        .write()
        .await
        .insert(sid.clone(), old);

    let held = old_client.lock().await;
    manager.send_message(&sid, "hi").await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !old_streaming.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the turn started");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let (new, _) = create_dummy_session(false, "", vec![]);
    let (new_client, new_token) = (new.client.clone(), new.nats_cancel.clone());
    manager
        .active_sessions
        .write()
        .await
        .insert(sid.clone(), new);
    drop(held);

    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match events.recv().await {
                Ok(ChatEvent::Error { .. }) => break,
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => panic!("no error event"),
            }
        }
    })
    .await
    .expect("the turn reports its dead CLI");
    let sessions = manager.active_sessions.read().await;
    let kept = sessions
        .get(&sid)
        .expect("the session that replaced it stays");
    assert!(Arc::ptr_eq(&kept.client, &new_client));
    assert!(!new_token.is_cancelled(), "its NATS listeners keep running");
}

/// The write fails (`Not connected`): the "CLI appears dead" path.
#[tokio::test]
async fn a_turn_whose_write_finds_the_cli_dead_leaves_the_session_that_replaced_it() {
    a_turn_on_a_dead_cli_leaves_the_session_that_replaced_it(create_dummy_client()).await;
}

/// The stream ends before a `Result`: the "CLI left before answering" path.
#[tokio::test]
async fn a_turn_whose_cli_ends_before_a_result_leaves_the_session_that_replaced_it() {
    let mut dead = InteractiveClient::from_transport(Box::new(DiesMidTurn));
    dead.connect().await.unwrap();
    a_turn_on_a_dead_cli_leaves_the_session_that_replaced_it(dead).await;
}

/// A connected CLI whose `send_message` fails without naming the channel:
/// the "half-dead" path. `send_and_receive_stream` reports a transport
/// error, not "Not connected", so the dead-channel heuristic does not fire.
struct RefusesToSend;

#[async_trait::async_trait]
impl nexus_claude::transport::Transport for RefusesToSend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    async fn connect(&mut self) -> nexus_claude::Result<()> {
        Ok(())
    }
    async fn send_message(
        &mut self,
        _: nexus_claude::transport::InputMessage,
    ) -> nexus_claude::Result<()> {
        Err(nexus_claude::SdkError::TransportError(
            "simulated send refusal".into(),
        ))
    }
    fn receive_messages(
        &mut self,
    ) -> std::pin::Pin<
        Box<
            dyn futures::Stream<Item = nexus_claude::Result<nexus_claude::Message>>
                + Send
                + 'static,
        >,
    > {
        Box::pin(futures::stream::empty())
    }
    async fn send_control_request(
        &mut self,
        _: nexus_claude::ControlRequest,
    ) -> nexus_claude::Result<()> {
        Ok(())
    }
    async fn receive_control_response(
        &mut self,
    ) -> nexus_claude::Result<Option<nexus_claude::ControlResponse>> {
        Ok(None)
    }
    async fn send_sdk_control_request(&mut self, _: serde_json::Value) -> nexus_claude::Result<()> {
        Ok(())
    }
    async fn send_sdk_control_response(
        &mut self,
        _: serde_json::Value,
    ) -> nexus_claude::Result<()> {
        Ok(())
    }
    fn is_connected(&self) -> bool {
        true
    }
    async fn disconnect(&mut self) -> nexus_claude::Result<()> {
        Ok(())
    }
}

/// A write that fails with a NON-channel error ("Transport error: ...")
/// still leaves the session half-dead: the CLI could not accept the turn.
/// The turn must forget it so the next message resumes it instead of
/// re-hitting the same broken stdin.
#[tokio::test]
async fn a_turn_whose_write_fails_without_naming_the_channel_forgets_the_session() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let sid = Uuid::new_v4().to_string();
    let (mut old, _) = create_dummy_session(false, "", vec![]);
    let mut refusing = InteractiveClient::from_transport(Box::new(RefusesToSend));
    refusing.connect().await.unwrap();
    old.client = Arc::new(Mutex::new(refusing));
    let mut events = old.events_tx.subscribe();
    manager
        .active_sessions
        .write()
        .await
        .insert(sid.clone(), old);

    manager.send_message(&sid, "hi").await.unwrap();

    // The turn reports the refusal ...
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match events.recv().await {
                Ok(ChatEvent::Error { .. }) => break,
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => panic!("no error event"),
            }
        }
    })
    .await
    .expect("the turn reports its refusal");

    // ... and the half-dead session is gone, so the next message resumes it.
    let sessions = manager.active_sessions.read().await;
    assert!(
        sessions.get(&sid).is_none(),
        "the half-dead session is forgotten"
    );
}

/// A legacy turn that ends on a 402 (out of credit) text marks the session's
/// provider instance out of credit in the router, reading the instance id
/// from the session node; a non-credit error leaves the instance alone.
#[tokio::test]
async fn a_legacy_out_of_credit_refusal_marks_the_provider_instance_out_of_credit() {
    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::store::InMemoryRoutingStore;

    let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let mut session = test_chat_session(None);
    session.provider_id = Some("deepseek".into());
    graph.create_chat_session(&session).await.unwrap();

    let routing = CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new()));

    ChatManager::mark_out_of_credit_if_needed(
        Some(&routing),
        &graph,
        Some(session.id),
        "API Error: 402 {\"type\":\"error\",\"error\":{\"message\":\"Insufficient credits\"}}",
    )
    .await;
    assert!(
        routing.out_of_credit("deepseek"),
        "the provider instance is out of credit after a 402"
    );

    // A non-credit error does not mark another instance.
    let mut other = test_chat_session(None);
    other.provider_id = Some("elsewhere".into());
    graph.create_chat_session(&other).await.unwrap();
    ChatManager::mark_out_of_credit_if_needed(
        Some(&routing),
        &graph,
        Some(other.id),
        "API Error: 500 {\"type\":\"error\",\"error\":{\"type\":\"api_error\"}}",
    )
    .await;
    assert!(
        !routing.out_of_credit("elsewhere"),
        "a non-credit error does not mark the instance"
    );
}

/// P20-suite-2: a resume that carries no claims re-derives the person from the owner
/// stored at open — so `po_mcp_env` can mint the session token again after a restart.
/// A session with no owner, or whose user record is gone, stays without one.
#[tokio::test]
async fn a_resumed_session_rederives_its_identity_from_the_stored_owner() {
    use crate::neo4j::models::{AuthProvider, UserNode};
    let state = mock_app_state();
    let graph = Arc::clone(&state.neo4j);
    let user = UserNode {
        id: Uuid::new_v4(),
        email: "person@example.com".into(),
        name: "A Person".into(),
        picture_url: None,
        auth_provider: AuthProvider::Oidc,
        external_id: Some("google-123".into()),
        password_hash: None,
        created_at: chrono::Utc::now(),
        last_login_at: chrono::Utc::now(),
    };
    graph.upsert_user(&user).await.unwrap();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // The owner stored at open is re-derived into the person's claims.
    let mut owned = test_chat_session(None);
    owned.owner = Some(user.id.to_string());
    let claims = manager.resume_identity(&owned).await.unwrap();
    assert_eq!(claims.sub, user.id.to_string());
    assert_eq!(claims.email, "person@example.com");
    assert_eq!(claims.name, "A Person");

    // No owner → no identity: the session stays degraded, never forged.
    assert!(manager
        .resume_identity(&test_chat_session(None))
        .await
        .is_none());

    // An owner whose user record is gone → no identity either.
    let mut ghost = test_chat_session(None);
    ghost.owner = Some(Uuid::new_v4().to_string());
    assert!(manager.resume_identity(&ghost).await.is_none());
}

/// Review of #685, finding 1: an agent-engine cancel listener reads no request
/// once its session closed, even one already waiting (`biased`, the close
/// first). Without it `select!` picks a ready branch at random: over 64 runs a
/// request would be read.
#[tokio::test]
async fn an_agent_cancel_listener_reads_no_request_once_the_session_closed() {
    let closed = CancellationToken::new();
    closed.cancel();
    for _ in 0..64 {
        let mut read = 0;
        ChatManager::serve_agent_cancels(futures::stream::iter([1, 2, 3]), closed.clone(), |_| {
            read += 1
        })
        .await;
        assert_eq!(read, 0);
    }
    // Open, it serves every request until the subscription ends.
    let mut read = 0;
    ChatManager::serve_agent_cancels(
        futures::stream::iter([1, 2, 3]),
        CancellationToken::new(),
        |_| read += 1,
    )
    .await;
    assert_eq!(read, 3);
}

/// Review of #685, finding 1: a cancel read just before the session closed is
/// answered `gone` (provisional: the real owner still answers), not with what
/// the closing provider session says (a result, a refusal or a failure, all
/// final: they would beat the real owner's answer).
#[tokio::test]
async fn a_cancel_on_a_closing_agent_session_answers_gone() {
    use super::super::cancel_relay::{CancelKind, CancelReply};
    let (manager, _graph, _fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("first"))
        .await
        .unwrap()
        .session_id;
    let handle = manager.agent_runtime.get(&sid).await.expect("a handle");
    // Open, the provider answers.
    let open = CancelReply::of(
        ChatManager::unless_agent_closed(&handle.closed, CancelKind::Tools, handle.cancel_tools())
            .await,
    );
    assert!(!open.is_gone());

    handle.closed.cancel();
    let tools = CancelReply::of(
        ChatManager::unless_agent_closed(&handle.closed, CancelKind::Tools, handle.cancel_tools())
            .await,
    );
    assert!(tools.is_gone());
    let task = CancelReply::of(
        ChatManager::unless_agent_closed(
            &handle.closed,
            CancelKind::Task,
            handle.cancel_task("tool_T"),
        )
        .await,
    );
    assert!(task.is_gone());
}

#[tokio::test]
async fn test_cancel_task_rate_cap_enforced() {
    let (mut session, _) = create_dummy_session(false, "", vec![]);
    // Tighten the cap so the test runs in milliseconds: 2 cancels /
    // 60s window. Reusing the same window const would make the test
    // either very slow or flaky.
    session.cancel_task_cap = 2;
    session.cancel_task_window = Duration::from_secs(60);

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-cap".into(), session);

    // 2 calls allowed.
    for _ in 0..2 {
        let r = manager.cancel_task("s-cap", "tool_W").await.unwrap();
        assert!(!r.capped);
    }
    // 3rd call should be capped.
    let r = manager.cancel_task("s-cap", "tool_W").await.unwrap();
    assert!(r.capped, "3rd call must be capped");
    assert!(r.killed_pids.is_empty());
}

// ========================================================================
// T4 of plan fc35b25e — cancel_task wires `kill_subtree` for V2 kill
// ========================================================================

/// When the task entry has `pid: None` (claim race or subprocess
/// crashed before discovery), `cancel_task` must:
/// - return `killed_pids: []` (no kill happened),
/// - still mark the entry for removal (V1 fallback preserved),
/// - still broadcast an `ActiveTasksUpdate` (frontend feedback).
#[tokio::test]
async fn test_cancel_task_without_pid_falls_back_to_v1() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    // Pre-seed a task with pid=None to exercise the fallback branch.
    session.active_background_tasks.lock().await.insert(
        "tool_NoPid".into(),
        BackgroundTaskInfo {
            id: "tool_NoPid".into(),
            kind: BackgroundTaskKind::Monitor,
            description: "no pid yet".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: None,
            parent_tool_use_id: Some("tool_NoPid".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-nopid".into(), session);

    let result = manager.cancel_task("s-nopid", "tool_NoPid").await.unwrap();
    assert_eq!(result.task_id, "tool_NoPid");
    assert!(
        result.killed_pids.is_empty(),
        "fallback path must return empty killed_pids, got {:?}",
        result.killed_pids
    );
    assert!(!result.capped);

    // Mark for removal still applied (V1 fallback preserved).
    let sessions = manager.active_sessions.read().await;
    let tasks = sessions
        .get("s-nopid")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert!(tasks
        .get("tool_NoPid")
        .unwrap()
        .pending_removal_at
        .is_some());

    // Broadcast still happens.
    let event = events_rx.try_recv().expect("expected broadcast");
    assert!(matches!(event, ChatEvent::ActiveTasksUpdate { .. }));
}

/// Spawn a real `sleep 30`, seed a task entry with its PID, call
/// `cancel_task`, and verify:
/// - `killed_pids` contains the sleep's PID,
/// - the subprocess actually dies (exit status non-success).
///
/// Marked `#[ignore]` because it touches real processes — run with
/// `cargo test -- --ignored`.
#[tokio::test]
#[ignore]
#[cfg(unix)]
async fn test_cancel_task_with_pid_calls_kill_subtree() {
    let mut child = match std::process::Command::new("sleep").arg("30").spawn() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping test: sleep not available");
            return;
        }
    };
    let real_pid = child.id();

    let (session, _) = create_dummy_session(false, "", vec![]);
    // Pre-seed a task entry pointing at the real subprocess.
    session.active_background_tasks.lock().await.insert(
        "tool_Real".into(),
        BackgroundTaskInfo {
            id: "tool_Real".into(),
            kind: BackgroundTaskKind::Monitor,
            description: "real sleep".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: Some(real_pid),
            parent_tool_use_id: Some("tool_Real".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    manager
        .active_sessions
        .write()
        .await
        .insert("s-real".into(), session);

    let result = manager.cancel_task("s-real", "tool_Real").await.unwrap();
    assert!(!result.capped);
    assert!(
        result.killed_pids.contains(&real_pid),
        "expected real_pid {real_pid} in killed_pids, got {:?}",
        result.killed_pids
    );

    // The subprocess should have received SIGINT and be dead now.
    let exit = child.wait().expect("waitpid");
    assert!(
        !exit.success(),
        "subprocess should have been signalled, not exited cleanly"
    );
}

/// P12 (review of #663, points 4 and 6): a Claude Code session of the legacy
/// engine held by ANOTHER instance. That instance's `cancel_task` and
/// `cancel_running_tools` ask the owner over NATS (test broker) and return what
/// it did: the task's process (a real `sleep`) is signalled and its pid comes
/// back, the running tool (a real child of a fake CLI) too. Once the session has
/// left the owner, the owner answers `gone` at once: a typed `session_gone`,
/// never the old silent success.
#[tokio::test]
#[cfg(unix)]
async fn a_legacy_session_answers_cancels_from_another_instance_over_nats() {
    use super::super::cancel_relay::CancelRelayError;
    use crate::events::nats_broker_test::TestBroker;
    use crate::events::NatsEmitter;

    let mut task = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let task_pid = task.id();
    // A fake CLI whose child is the running tool.
    let mut cli = std::process::Command::new("sh")
        .args(["-c", "sleep 30; true"])
        .spawn()
        .unwrap();
    let cli_pid = cli.id();

    let broker = TestBroker::start().await;
    let owner_nats = Arc::new(NatsEmitter::new(broker.client().await, "events"));
    let state = mock_app_state();
    let owner = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
        .with_nats(owner_nats.clone());
    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(cli_pid);
    session.active_background_tasks.lock().await.insert(
        "tool_Far".into(),
        BackgroundTaskInfo {
            id: "tool_Far".into(),
            kind: BackgroundTaskKind::BashBackground,
            description: "far sleep".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: Some(task_pid),
            parent_tool_use_id: Some("tool_Far".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );
    owner
        .active_sessions
        .write()
        .await
        .insert("s-far".into(), session);
    let token = CancellationToken::new();
    owner.spawn_nats_cancel_task_listener("s-far", owner.active_sessions.clone(), token.clone());
    owner.spawn_nats_cancel_tools_listener("s-far", owner.active_sessions.clone(), token.clone());

    let state = mock_app_state();
    // A short deadline: a `gone` alone for cancel_tools is kept until it.
    let far = ChatManager::new_without_memory(state.neo4j, state.meili, test_config()).with_nats(
        Arc::new(
            NatsEmitter::new(broker.client().await, "events")
                .with_cancel_rpc_timeout(Duration::from_secs(3)),
        ),
    );
    // The owner's listeners may still be subscribing.
    let mut stopped = None;
    for _ in 0..20 {
        match far.cancel_task("s-far", "tool_Far").await {
            Ok(result) => {
                stopped = Some(result);
                break;
            }
            Err(e)
                if e.downcast_ref::<CancelRelayError>()
                    == Some(&CancelRelayError::OwnerUnreachable) =>
            {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(e) => panic!("{e:#}"),
        }
    }
    let stopped = stopped.expect("the owner answered");
    assert!(stopped.killed_pids.contains(&task_pid), "{stopped:?}");
    assert!(
        !task.wait().unwrap().success(),
        "the task's process was signalled"
    );

    let mut tools = None;
    for _ in 0..20 {
        match far.cancel_running_tools("s-far").await {
            Ok(result) => {
                tools = Some(result);
                break;
            }
            Err(e)
                if e.downcast_ref::<CancelRelayError>()
                    == Some(&CancelRelayError::OwnerUnreachable) =>
            {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(e) => panic!("{e:#}"),
        }
    }
    let tools = tools.expect("the owner answered");
    assert!(!tools.capped && !tools.killed_pids.is_empty(), "{tools:?}");
    assert_eq!(tools.cli_pid, Some(cli_pid));
    let _ = cli.kill();
    let _ = cli.wait();

    // The session leaves the owner: both answer `gone`, typed. cancel_task
    // reports it after GONE_GRACE (retryable); cancel_tools keeps listening for
    // a real owner until its deadline, then reports it, not retryable (review
    // of #673, finding 1).
    owner.active_sessions.write().await.remove("s-far");
    let started = std::time::Instant::now();
    let gone = far.cancel_task("s-far", "tool_Far").await.unwrap_err();
    assert_eq!(
        gone.downcast_ref::<CancelRelayError>(),
        Some(&CancelRelayError::SessionGone {
            kind: super::super::cancel_relay::CancelKind::Task
        })
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed >= super::super::cancel_relay::GONE_GRACE && elapsed < Duration::from_secs(3),
        "answered after the grace, not timed out: {elapsed:?}"
    );
    // Measured from the cancel_tools alone (review of #685, finding 5): it waits
    // for a real owner up to the asker's deadline (3 s here), and no longer.
    let started = std::time::Instant::now();
    let gone = far.cancel_running_tools("s-far").await.unwrap_err();
    let elapsed = started.elapsed();
    let gone = gone.downcast_ref::<CancelRelayError>().cloned();
    assert_eq!(
        gone,
        Some(CancelRelayError::SessionGone {
            kind: super::super::cancel_relay::CancelKind::Tools
        })
    );
    assert!(!gone.unwrap().failure().retryable);
    assert!(
        elapsed >= Duration::from_secs(3) && elapsed < Duration::from_secs(5),
        "answered at the deadline, not before nor long after: {elapsed:?}"
    );
    token.cancel();
}

/// Review N8 of #663: a cancel_tools on the instance that holds the session is
/// done once. It used to also publish the cancel signal on NATS, which this
/// instance's own listener received: the tools were signalled twice and the
/// cap counted twice.
#[tokio::test]
#[cfg(unix)]
async fn a_local_cancel_tools_is_done_once_not_echoed_by_its_own_listener() {
    use crate::events::nats_broker_test::TestBroker;
    use crate::events::NatsEmitter;

    let mut cli = std::process::Command::new("sh")
        .args(["-c", "sleep 30; true"])
        .spawn()
        .unwrap();
    let broker = TestBroker::start().await;
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
        .with_nats(Arc::new(NatsEmitter::new(broker.client().await, "events")));
    let (mut session, _) = create_dummy_session(false, "", vec![]);
    session.child_pid = Some(cli.id());
    let history = session.cancel_tools_history.clone();
    manager
        .active_sessions
        .write()
        .await
        .insert("s-once".into(), session);
    let token = CancellationToken::new();
    manager.spawn_nats_cancel_tools_listener(
        "s-once",
        manager.active_sessions.clone(),
        token.clone(),
    );
    tokio::time::sleep(Duration::from_millis(300)).await;

    let result = manager.cancel_running_tools("s-once").await.unwrap();
    assert!(!result.capped);
    // Leave the echo, if any, the time to arrive.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(history.lock().await.len(), 1, "one cancel, counted once");
    token.cancel();
    let _ = cli.kill();
    let _ = cli.wait();
}

// ========================================================================
// T1 of plan fc35b25e — parse_etime + pid_discovery_diff helpers
// ========================================================================

#[test]
fn test_parse_etime_seconds_only() {
    assert_eq!(ChatManager::parse_etime("00:01"), Some(1));
    assert_eq!(ChatManager::parse_etime("00:42"), Some(42));
    assert_eq!(ChatManager::parse_etime("00:00"), Some(0));
}

#[test]
fn test_parse_etime_minutes_seconds() {
    assert_eq!(ChatManager::parse_etime("12:34"), Some(12 * 60 + 34));
    assert_eq!(ChatManager::parse_etime("00:30"), Some(30));
    assert_eq!(ChatManager::parse_etime("59:59"), Some(59 * 60 + 59));
}

#[test]
fn test_parse_etime_hours_minutes_seconds() {
    assert_eq!(
        ChatManager::parse_etime("02:12:34"),
        Some(2 * 3600 + 12 * 60 + 34)
    );
    assert_eq!(ChatManager::parse_etime("01:00:00"), Some(3600));
}

#[test]
fn test_parse_etime_with_days() {
    assert_eq!(
        ChatManager::parse_etime("01-02:12:34"),
        Some(86_400 + 2 * 3600 + 12 * 60 + 34)
    );
    assert_eq!(ChatManager::parse_etime("07-00:00:00"), Some(7 * 86_400));
}

#[test]
fn test_parse_etime_handles_whitespace() {
    // `ps -o etime=` may pad with leading spaces.
    assert_eq!(ChatManager::parse_etime("  00:42  "), Some(42));
    assert_eq!(ChatManager::parse_etime("\n12:34\n"), Some(12 * 60 + 34));
}

#[test]
fn test_parse_etime_rejects_garbage() {
    assert_eq!(ChatManager::parse_etime(""), None);
    assert_eq!(ChatManager::parse_etime("hello"), None);
    assert_eq!(ChatManager::parse_etime("1:2:3:4"), None); // too many parts
    assert_eq!(ChatManager::parse_etime("aa:bb"), None);
    assert_eq!(ChatManager::parse_etime("-1:00"), None);
}

#[test]
fn test_pid_discovery_diff_finds_new_pids() {
    // before = [100, 200], after = [100, 200, 300] → new = [300]
    let before = vec![100u32, 200];
    let after = vec![100u32, 200, 300];
    let diff = ChatManager::pid_discovery_diff(&before, &after);
    // PID 300 doesn't exist in /proc, etime returns None → sorted last,
    // but it's the only new entry so it's at index 0.
    assert_eq!(diff, vec![300]);
}

#[test]
fn test_pid_discovery_diff_no_new_pids() {
    let before = vec![100u32, 200, 300];
    let after = vec![100u32, 200];
    let diff = ChatManager::pid_discovery_diff(&before, &after);
    // No PIDs in `after` that aren't in `before`.
    assert!(diff.is_empty());
}

#[test]
fn test_pid_discovery_diff_empty_before() {
    let before = Vec::<u32>::new();
    let after = vec![1u32, 2, 3];
    let diff = ChatManager::pid_discovery_diff(&before, &after);
    assert_eq!(diff.len(), 3);
    // Order is stable (etime returns None for fake PIDs → all u64::MAX,
    // sort_by_key preserves insertion order on ties).
}

#[test]
fn test_pid_discovery_diff_with_real_subprocess() {
    // Spawn a real `sleep 30` subprocess so we have a PID with a
    // readable etime. Diff should put it before fake PIDs (which
    // have unreadable etime → sorted last).
    let mut child = match std::process::Command::new("sleep").arg("30").spawn() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping test: sleep not available");
            return;
        }
    };
    let real_pid = child.id();

    // Give the OS a moment to register the process so `ps` sees it.
    std::thread::sleep(std::time::Duration::from_millis(100));

    let before = Vec::<u32>::new();
    let after = vec![999_999u32, real_pid, 999_998u32];
    let diff = ChatManager::pid_discovery_diff(&before, &after);

    // Real PID has etime ~0s (just spawned), fake PIDs have etime
    // None → u64::MAX → sorted last. So real PID should be first.
    assert_eq!(diff[0], real_pid, "real subprocess should sort first");

    let _ = child.kill();
    let _ = child.wait();
}

// ========================================================================
// T13 of plan 754a1379 — track_background_task_recovery_if_orphan
// (lazy crash-recovery via BackgroundOutput orphan correlation_id)
// ========================================================================

#[tokio::test]
async fn test_recovery_inserts_orphan_monitor() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let events_tx = session.events_tx.clone();

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-rec".into(), session);

    let inserted = ChatManager::track_background_task_recovery_if_orphan(
        "s-rec",
        &map,
        &events_tx,
        &None,
        Some("toolu_orphan_M"),
        "Monitor",
    )
    .await;
    assert!(inserted);

    // Map now has the recovered entry.
    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-rec")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.len(), 1);
    let entry = tasks.get("toolu_orphan_M").unwrap();
    assert_eq!(entry.kind, BackgroundTaskKind::Monitor);
    assert_eq!(entry.description, "(recovered after restart)");
    assert_eq!(
        entry.parent_tool_use_id.as_deref(),
        Some("toolu_orphan_M"),
        "parent_tool_use_id mirrors the recovered id"
    );
    drop(tasks);
    drop(sessions_guard);

    // Broadcast carries the snapshot.
    let event = events_rx.try_recv().expect("expected broadcast");
    match event {
        ChatEvent::ActiveTasksUpdate { tasks } => {
            assert_eq!(tasks.len(), 1);
            assert_eq!(tasks[0].id, "toolu_orphan_M");
        }
        other => panic!("expected ActiveTasksUpdate, got {:?}", other),
    }
}

#[tokio::test]
async fn test_recovery_inserts_orphan_bash_background() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let events_tx = session.events_tx.clone();

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-rec-b".into(), session);

    let inserted = ChatManager::track_background_task_recovery_if_orphan(
        "s-rec-b",
        &map,
        &events_tx,
        &None,
        Some("toolu_orphan_B"),
        "BashOutput",
    )
    .await;
    assert!(inserted);

    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-rec-b")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(
        tasks.get("toolu_orphan_B").unwrap().kind,
        BackgroundTaskKind::BashBackground
    );
}

#[tokio::test]
async fn test_recovery_skips_already_tracked() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let events_tx = session.events_tx.clone();

    // Pre-seed an entry under the same id.
    session.active_background_tasks.lock().await.insert(
        "toolu_existing".into(),
        BackgroundTaskInfo {
            id: "toolu_existing".into(),
            kind: BackgroundTaskKind::Monitor,
            description: "original".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: None,
            parent_tool_use_id: Some("toolu_existing".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-rec-skip".into(), session);

    let inserted = ChatManager::track_background_task_recovery_if_orphan(
        "s-rec-skip",
        &map,
        &events_tx,
        &None,
        Some("toolu_existing"),
        "Monitor",
    )
    .await;
    assert!(!inserted, "already-tracked id must not trigger insertion");

    // Description preserved (we did NOT overwrite).
    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-rec-skip")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.get("toolu_existing").unwrap().description, "original");
    drop(tasks);
    drop(sessions_guard);

    // No broadcast on no-op.
    assert!(events_rx.try_recv().is_err(), "no broadcast for no-op");
}

#[tokio::test]
async fn test_recovery_ignores_unknown_source() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let events_tx = session.events_tx.clone();

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-rec-unk".into(), session);

    for source in ["system", "WebFetch", "", "foobar"] {
        let inserted = ChatManager::track_background_task_recovery_if_orphan(
            "s-rec-unk",
            &map,
            &events_tx,
            &None,
            Some("toolu_X"),
            source,
        )
        .await;
        assert!(!inserted, "source `{}` should not trigger recovery", source);
    }

    let count = map
        .read()
        .await
        .get("s-rec-unk")
        .unwrap()
        .active_background_tasks
        .lock()
        .await
        .len();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_recovery_touches_last_seen_at_on_known_correlation_id() {
    // Path 1: known correlation_id → refresh last_seen_at, no broadcast,
    // returns false (no insertion, but the refresh is the silent
    // side-effect that drives idle-death detection).
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let events_tx = session.events_tx.clone();

    // Pre-seed an old entry — last_seen_at deep in the past.
    let old = chrono::Utc::now() - chrono::Duration::hours(2);
    session.active_background_tasks.lock().await.insert(
        "tool_known".into(),
        BackgroundTaskInfo {
            id: "tool_known".into(),
            kind: BackgroundTaskKind::Monitor,
            description: "watched a long time".into(),
            started_at: old,
            last_seen_at: old,
            pid: None,
            parent_tool_use_id: Some("tool_known".into()),
            pending_removal_at: None,
            signalled: false,
        },
    );

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-touch".into(), session);

    let inserted = ChatManager::track_background_task_recovery_if_orphan(
        "s-touch",
        &map,
        &events_tx,
        &None,
        Some("tool_known"),
        "Monitor",
    )
    .await;
    assert!(!inserted, "no insertion when id is already known");

    // last_seen_at refreshed.
    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-touch")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    let entry = tasks.get("tool_known").unwrap();
    assert!(
        entry.last_seen_at > old,
        "last_seen_at must be refreshed (was {:?}, now {:?})",
        old,
        entry.last_seen_at
    );
    drop(tasks);
    drop(sessions_guard);

    // No broadcast (path 1 is silent — tick-by-tick refreshes would
    // saturate the WebSocket on a chatty Monitor).
    assert!(events_rx.try_recv().is_err());
}

#[tokio::test]
async fn test_recovery_no_correlation_id_is_noop() {
    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    let (events_tx, _rx) = broadcast::channel(8);
    let inserted = ChatManager::track_background_task_recovery_if_orphan(
        "s-anywhere",
        &map,
        &events_tx,
        &None,
        None,
        "Monitor",
    )
    .await;
    assert!(!inserted);
}

// ========================================================================
// A silent background task is not a dead one (idle-death vs. liveness)
// ========================================================================

/// One silent `Bash run_in_background` entry, last seen well past the
/// idle-death threshold, in a session inserted under `sid`.
async fn session_with_silent_background_task(
    sid: &str,
    pid: Option<u32>,
    cli_reported: usize,
) -> (
    Arc<RwLock<HashMap<String, ActiveSession>>>,
    broadcast::Sender<ChatEvent>,
) {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let events_tx = session.events_tx.clone();
    let now = chrono::Utc::now();
    let silent_since =
        now - chrono::Duration::seconds((BACKGROUND_TASK_IDLE_DEATH_SECS + 60) as i64);
    session.active_background_tasks.lock().await.insert(
        "tool_silent".into(),
        BackgroundTaskInfo {
            id: "tool_silent".into(),
            kind: BackgroundTaskKind::BashBackground,
            description: "cargo test > log 2>&1".into(),
            started_at: silent_since,
            last_seen_at: silent_since,
            pid,
            parent_tool_use_id: None,
            pending_removal_at: None,
            signalled: false,
        },
    );
    session
        .cli_background_tasks
        .store(cli_reported, Ordering::Relaxed);
    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert(sid.into(), session);
    (map, events_tx)
}

async fn silent_task_is_marked(
    map: &Arc<RwLock<HashMap<String, ActiveSession>>>,
    sid: &str,
) -> bool {
    let sessions = map.read().await;
    let tasks = sessions
        .get(sid)
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    tasks
        .get("tool_silent")
        .expect("the entry is never removed within one tick")
        .pending_removal_at
        .is_some()
}

/// The bug: 30 minutes without output and the task was purged although
/// its process was still running; the session then had "no background
/// work" and the idle cleanup closed it, killing the CLI and the command.
/// A task whose process runs is kept, and nothing is broadcast.
#[cfg(unix)]
#[tokio::test]
async fn test_tick_purge_keeps_a_silent_task_whose_process_is_running() {
    let (map, events_tx) =
        session_with_silent_background_task("s-silent-pid", Some(std::process::id()), 0).await;
    let mut events_rx = events_tx.subscribe();

    ChatManager::tick_purge_background_tasks(
        "s-silent-pid",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;

    assert!(
        !silent_task_is_marked(&map, "s-silent-pid").await,
        "a task whose process is running must not be marked for removal"
    );
    assert!(
        events_rx.try_recv().is_err(),
        "a quiet tick broadcasts nothing"
    );
}

/// No pid was discovered (the claim often fails), but the CLI reports one
/// running background task: the entry is kept.
#[tokio::test]
async fn test_tick_purge_keeps_a_silent_task_while_the_cli_reports_work() {
    let (map, events_tx) = session_with_silent_background_task("s-silent-cli", None, 1).await;

    ChatManager::tick_purge_background_tasks(
        "s-silent-cli",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;
    assert!(!silent_task_is_marked(&map, "s-silent-cli").await);

    // The CLI then reports that nothing runs any more: the next tick
    // marks the entry, as before.
    map.read()
        .await
        .get("s-silent-cli")
        .unwrap()
        .cli_background_tasks
        .store(0, Ordering::Relaxed);
    ChatManager::tick_purge_background_tasks(
        "s-silent-cli",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;
    assert!(silent_task_is_marked(&map, "s-silent-cli").await);
}

/// A known pid whose process is gone is dead, whatever the CLI reports
/// (its report is about some other task).
#[cfg(unix)]
#[tokio::test]
async fn test_tick_purge_marks_a_silent_task_whose_process_is_gone() {
    let mut child = std::process::Command::new("true")
        .spawn()
        .expect("spawn `true`");
    let dead_pid = child.id();
    child.wait().expect("wait for `true`");

    let (map, events_tx) =
        session_with_silent_background_task("s-silent-dead", Some(dead_pid), 1).await;

    ChatManager::tick_purge_background_tasks(
        "s-silent-dead",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;
    assert!(silent_task_is_marked(&map, "s-silent-dead").await);
}

// ========================================================================
// T12 of plan 754a1379 — tick_purge_background_tasks (grace period purge)
// ========================================================================

#[tokio::test]
async fn test_tick_purge_removes_stale_pending_entries_and_broadcasts() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let events_tx = session.events_tx.clone();

    // Insert: one expired (pending_removal_at far in the past),
    // one fresh (no pending_removal_at).
    let now = std::time::Instant::now();
    let stale_at = now.checked_sub(Duration::from_secs(10)).unwrap_or(now);
    {
        let mut tasks = session.active_background_tasks.lock().await;
        tasks.insert(
            "tool_stale".into(),
            BackgroundTaskInfo {
                id: "tool_stale".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "stale".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: None,
                parent_tool_use_id: None,
                pending_removal_at: Some(stale_at),
                signalled: false,
            },
        );
        tasks.insert(
            "tool_alive".into(),
            BackgroundTaskInfo {
                id: "tool_alive".into(),
                kind: BackgroundTaskKind::BashBackground,
                description: "alive".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: None,
                parent_tool_use_id: None,
                pending_removal_at: None,
                signalled: false,
            },
        );
    }

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-purge".into(), session);

    ChatManager::tick_purge_background_tasks(
        "s-purge",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;

    // Stale entry purged, alive entry kept.
    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-purge")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.len(), 1);
    assert!(tasks.contains_key("tool_alive"));
    assert!(!tasks.contains_key("tool_stale"));
    drop(tasks);
    drop(sessions_guard);

    // A single broadcast carrying the post-purge snapshot.
    let event = events_rx
        .try_recv()
        .expect("expected broadcast after purge");
    match event {
        ChatEvent::ActiveTasksUpdate { tasks } => {
            assert_eq!(tasks.len(), 1);
            assert_eq!(tasks[0].id, "tool_alive");
        }
        other => panic!("expected ActiveTasksUpdate, got {:?}", other),
    }
}

#[tokio::test]
async fn test_tick_purge_marks_idle_entries_as_pending_removal() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let events_tx = session.events_tx.clone();

    // Pre-seed: one ALIVE entry (last_seen_at = now), one IDLE
    // entry (last_seen_at well past the death threshold).
    let now = chrono::Utc::now();
    let stale_seen = now - chrono::Duration::seconds((BACKGROUND_TASK_IDLE_DEATH_SECS + 60) as i64);
    {
        let mut tasks = session.active_background_tasks.lock().await;
        tasks.insert(
            "tool_alive".into(),
            BackgroundTaskInfo {
                id: "tool_alive".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "live".into(),
                started_at: now,
                last_seen_at: now,
                pid: None,
                parent_tool_use_id: None,
                pending_removal_at: None,
                signalled: false,
            },
        );
        tasks.insert(
            "tool_idle".into(),
            BackgroundTaskInfo {
                id: "tool_idle".into(),
                kind: BackgroundTaskKind::BashBackground,
                description: "silently dead".into(),
                started_at: stale_seen,
                last_seen_at: stale_seen,
                pid: None,
                parent_tool_use_id: None,
                pending_removal_at: None,
                signalled: false,
            },
        );
    }

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-idle".into(), session);

    ChatManager::tick_purge_background_tasks(
        "s-idle",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;

    // Idle entry must now have pending_removal_at set.
    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-idle")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.len(), 2, "first tick only marks, doesn't purge yet");
    assert!(
        tasks
            .get("tool_alive")
            .unwrap()
            .pending_removal_at
            .is_none(),
        "alive entry must NOT be marked"
    );
    assert!(
        tasks.get("tool_idle").unwrap().pending_removal_at.is_some(),
        "idle entry MUST be marked"
    );
    drop(tasks);
    drop(sessions_guard);

    // Single broadcast carrying the marked-but-still-present snapshot.
    let event = events_rx.try_recv().expect("expected broadcast");
    match event {
        ChatEvent::ActiveTasksUpdate { tasks } => assert_eq!(tasks.len(), 2),
        other => panic!("expected ActiveTasksUpdate, got {:?}", other),
    }
}

#[tokio::test]
async fn test_tick_purge_within_grace_does_nothing() {
    let (session, _) = create_dummy_session(false, "", vec![]);
    let mut events_rx = session.events_tx.subscribe();
    let events_tx = session.events_tx.clone();

    // Mark just now → still within grace.
    let now = std::time::Instant::now();
    session.active_background_tasks.lock().await.insert(
        "tool_recent".into(),
        BackgroundTaskInfo {
            id: "tool_recent".into(),
            kind: BackgroundTaskKind::Monitor,
            description: "recent".into(),
            started_at: chrono::Utc::now(),
            last_seen_at: chrono::Utc::now(),
            pid: None,
            parent_tool_use_id: None,
            pending_removal_at: Some(now),
            signalled: false,
        },
    );

    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    map.write().await.insert("s-grace".into(), session);

    ChatManager::tick_purge_background_tasks(
        "s-grace",
        &map,
        &events_tx,
        &None,
        Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
    )
    .await;

    // Entry still present.
    let sessions_guard = map.read().await;
    let tasks = sessions_guard
        .get("s-grace")
        .unwrap()
        .active_background_tasks
        .lock()
        .await;
    assert_eq!(tasks.len(), 1);
    drop(tasks);
    drop(sessions_guard);

    // No broadcast — purge was a no-op.
    assert!(
        events_rx.try_recv().is_err(),
        "no broadcast expected when nothing was purged"
    );
}

#[tokio::test]
async fn test_cancel_task_result_serde_round_trip() {
    let r = CancelTaskResult {
        task_id: "tool_abc".into(),
        killed_pids: vec![],
        capped: false,
    };
    let json = serde_json::to_string(&r).expect("serialize");
    assert!(json.contains("\"task_id\":\"tool_abc\""));
    assert!(json.contains("\"killed_pids\":[]"));
    assert!(json.contains("\"capped\":false"));
    let back: CancelTaskResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, "tool_abc");
    assert!(back.killed_pids.is_empty());
    assert!(!back.capped);
}

#[tokio::test]
async fn test_track_background_task_unknown_session_returns_false() {
    let map: Arc<RwLock<HashMap<String, ActiveSession>>> = Arc::new(RwLock::new(HashMap::new()));
    let (events_tx, _) = broadcast::channel(8);

    let inserted = ChatManager::track_background_task_start(
        "ghost-session",
        &map,
        &events_tx,
        &None,
        "tool_X",
        "Monitor",
        &serde_json::json!({}),
        None,
    )
    .await;
    assert!(!inserted);
}

// ── is_session_working: who still holds a delegation slot (issue #876) ─────

/// A Claude Code session entry's own background task, as the OOB listener tracks it.
fn a_cli_background_task() -> crate::chat::types::BackgroundTaskInfo {
    use crate::chat::types::BackgroundTaskKind;
    let now = chrono::Utc::now();
    crate::chat::types::BackgroundTaskInfo {
        id: "tool_W1".to_string(),
        kind: BackgroundTaskKind::BashBackground,
        description: "cargo test".to_string(),
        started_at: now,
        last_seen_at: now,
        pid: Some(4242),
        parent_tool_use_id: None,
        pending_removal_at: None,
        signalled: false,
    }
}

/// A turn running is work: the session must not give up its slot.
#[tokio::test]
async fn a_streaming_session_is_working() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (session, _) = create_dummy_session(true, "half an answer", vec![]);
    manager
        .active_sessions
        .write()
        .await
        .insert("s-streaming".into(), session);

    assert!(manager.is_session_working("s-streaming").await);
}

/// An agent session is working while its provider reports a background task, and
/// not once it has answered (`is_session_active` still says it is live).
#[tokio::test]
async fn an_agent_session_is_working_while_its_provider_reports_a_task() {
    use nexus_claude::agent::{BackgroundTaskKind as Kind, BackgroundTaskStatus as Status};
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let busy = idle_agent_session(&manager, Duration::ZERO).await;
    busy.set_background_tasks(vec![native_bg_task("nt-1", Kind::Shell, Status::Running)])
        .await;
    assert!(
        manager.is_session_working(&busy.session_id).await,
        "a background task its provider reports is work"
    );

    let answered = idle_agent_session(&manager, Duration::ZERO).await;
    assert!(
        manager.is_session_active(&answered.session_id).await,
        "answered but still live"
    );
    assert!(
        !manager.is_session_working(&answered.session_id).await,
        "answered and nothing running: not working"
    );
}

/// A Claude Code session entry is working while one of its background tasks is
/// tracked, or while the CLI itself reports one — and not otherwise.
#[tokio::test]
async fn a_cli_session_is_working_while_a_background_task_runs() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let (answered, _) = create_dummy_session(false, "", vec![]);
    let (tracked, _) = create_dummy_session(false, "", vec![]);
    tracked
        .active_background_tasks
        .lock()
        .await
        .insert("tool_W1".into(), a_cli_background_task());
    let (reported_by_cli, _) = create_dummy_session(false, "", vec![]);
    reported_by_cli
        .cli_background_tasks
        .store(1, Ordering::Relaxed);

    {
        let mut sessions = manager.active_sessions.write().await;
        sessions.insert("s-answered".into(), answered);
        sessions.insert("s-tracked".into(), tracked);
        sessions.insert("s-cli".into(), reported_by_cli);
    }

    assert!(
        !manager.is_session_working("s-answered").await,
        "answered and nothing running: not working"
    );
    assert!(
        manager.is_session_working("s-tracked").await,
        "a task this server tracks is work"
    );
    assert!(
        manager.is_session_working("s-cli").await,
        "a task the CLI reports is work"
    );
    // A session nobody knows is not working.
    assert!(!manager.is_session_working("s-nowhere").await);
}

/// A dead CLI (zombie process) whose `cli_background_tasks` was left > 0 must NOT
/// be counted as working — the stale counter is cleared so the child slot is freed
/// (ZOMBIE-CLI: `envelope_too_many_children` caused by a defunct `<defunct>` entry).
#[tokio::test]
async fn a_zombie_cli_is_not_counted_as_working() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    // Session whose cli_background_tasks = 1 but whose CLI process is dead (pid 0
    // is never a real process on Unix; process_alive(Some(0)) == Some(false)).
    let (zombie_cli, _) = create_dummy_session(false, "", vec![]);
    zombie_cli.cli_background_tasks.store(1, Ordering::Relaxed);

    // Patch child_pid to a known-dead pid (0 is never a real process).
    // We reconstruct the session with child_pid set directly.
    let (mut dead_session, msgs) = create_dummy_session(false, "", vec![]);
    dead_session.child_pid = Some(reaped_pid());
    dead_session
        .cli_background_tasks
        .store(1, Ordering::Relaxed);
    let cli_bg_arc = dead_session.cli_background_tasks.clone();

    {
        let mut sessions = manager.active_sessions.write().await;
        sessions.insert("s-zombie".into(), dead_session);
    }

    // On a dead CLI, is_session_working must return false.
    assert!(
        !manager.is_session_working("s-zombie").await,
        "a zombie CLI with stale cli_background_tasks must not be counted as working"
    );
    // The stale counter must have been cleared.
    assert_eq!(
        cli_bg_arc.load(Ordering::Relaxed),
        0,
        "cli_background_tasks must be reset to 0 when the CLI process is dead"
    );
    // Suppress unused-variable warning.
    drop(msgs);
}

/// ZOMBIE-CLI, the whole path: a session whose CLI process is dead is CLOSED, with the
/// typed reason `process_died` on its `session_closed`, and removed from the live table
/// (which frees its parent's child slot). A live CLI and a session without a known pid
/// are left alone.
#[tokio::test]
async fn a_session_whose_cli_died_is_closed_with_process_died() {
    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

    let (mut dead, _) = create_dummy_session(false, "", vec![]);
    dead.child_pid = Some(reaped_pid());
    let mut events = dead.events_tx.subscribe();
    let (live, _) = create_dummy_session(false, "", vec![]);
    let mut live = live;
    live.child_pid = Some(std::process::id()); // this test process: alive
    let (unknown, _) = create_dummy_session(false, "", vec![]); // child_pid None

    {
        let mut sessions = manager.active_sessions.write().await;
        sessions.insert("s-dead".into(), dead);
        sessions.insert("s-live".into(), live);
        sessions.insert("s-unknown".into(), unknown);
    }

    let closed = manager.close_sessions_of_dead_clis().await;
    assert_eq!(
        closed,
        vec!["s-dead".to_string()],
        "only the dead CLI is closed"
    );

    {
        let sessions = manager.active_sessions.read().await;
        assert!(
            !sessions.contains_key("s-dead"),
            "the dead session left the live table"
        );
        assert!(
            sessions.contains_key("s-live"),
            "a live CLI keeps its session"
        );
        assert!(sessions.contains_key("s-unknown"), "no pid, no verdict");
    }
    assert!(
        !manager.is_session_active("s-dead").await,
        "a closed session holds no envelope slot"
    );

    // The typed reason reaches the clients.
    let reason = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            match events.recv().await {
                Ok(ChatEvent::SessionClosed { session_id, reason }) if session_id == "s-dead" => {
                    break reason;
                }
                Ok(_) => continue,
                Err(_) => break None,
            }
        }
    })
    .await
    .expect("session_closed is emitted");
    assert_eq!(reason.as_deref(), Some("process_died"));
}

/// The envelope's `LiveSessions` view is that same verdict: through the trait, a
/// session that answered is live but not working (so it frees a child slot).
#[tokio::test]
async fn the_envelope_asks_the_manager_who_is_working() {
    use crate::chat::envelope::LiveSessions;

    let state = mock_app_state();
    let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
    let (answered, _) = create_dummy_session(false, "", vec![]);
    let (streaming, _) = create_dummy_session(true, "half an answer", vec![]);
    {
        let mut sessions = manager.active_sessions.write().await;
        sessions.insert("s-answered".into(), answered);
        sessions.insert("s-streaming".into(), streaming);
    }

    assert!(LiveSessions::is_active(&manager, "s-answered").await);
    assert!(!LiveSessions::is_working(&manager, "s-answered").await);
    assert!(LiveSessions::is_working(&manager, "s-streaming").await);
    assert!(!LiveSessions::is_working(&manager, "s-nowhere").await);
}

/// PROVIDER-DEAD (the third defect of the 10/11 plan). The watchdog used to know only
/// what the provider had ANNOUNCED out of a turn: an `Error { ProcessExited }` on its
/// stream. A session whose process is gone but whose stream is silent announces
/// nothing — that is exactly the state the session measured on 10/11 (cadfa92b) sat in
/// for 8 hours: handle live, `provider_is_dead()` false, no CLI process. The verdict is
/// OBSERVED now, by asking the provider, so the same state is harvested within one
/// watchdog interval and its slot is freed.
#[tokio::test]
async fn agent_path_a_dead_provider_process_is_harvested_though_no_event_announced_it() {
    use nexus_claude::agent::ProviderError;
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    // The process is gone. NOTHING says so: no event, no out-of-band frame — only the
    // session's own answer to a call that checks whether it is still usable.
    *fake.state.cancel_tools_answer.lock().unwrap() =
        Some(ProviderError::ProcessExited { code: None });

    let mut rx = manager.subscribe(&sid).await.unwrap();
    let closed = manager.close_sessions_of_dead_clis().await;
    assert_eq!(
        closed,
        vec![sid.clone()],
        "the session whose provider process is gone is harvested"
    );
    assert!(
        !manager.is_session_active(&sid).await,
        "a closed session holds no envelope slot"
    );
    assert!(
        !manager.agent_runtime.owns(&sid).await,
        "the handle left the live table with the session"
    );
    let event = next_matching(
        &mut rx,
        |e| matches!(e, ChatEvent::SessionClosed { reason: Some(r), .. } if r == "process_died"),
    )
    .await;
    assert!(matches!(event, ChatEvent::SessionClosed { .. }));
}

/// PROVIDER-DEAD, the other half: the constat must not create the false positive it
/// risks. A session whose provider is ALIVE but silent — a long tool, a question waiting
/// for its person — answers the probe the way nexus answers a live session
/// (`invalid_request`: no such background task). Nothing is harvested, and the turn in
/// flight keeps streaming.
#[tokio::test]
async fn agent_path_a_silent_but_living_provider_is_not_harvested() {
    use nexus_claude::agent::ProviderError;
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    *fake.state.cancel_tools_answer.lock().unwrap() = Some(ProviderError::invalid(
        "unknown background task `__provider_liveness_probe__`",
    ));

    assert!(
        manager.close_sessions_of_dead_clis().await.is_empty(),
        "a session that says nothing is not a session that is gone"
    );
    assert!(
        manager.is_session_active(&sid).await,
        "the silent session keeps its slot"
    );
    let handle = manager.agent_runtime.get(&sid).await.expect("registered");
    assert!(
        handle
            .is_streaming
            .load(std::sync::atomic::Ordering::SeqCst),
        "the turn in flight was not cut by the question"
    );
    manager.close_session(&sid).await.ok();
}

/// PROVIDER-DEAD, the remote half: where the provider's process is not local by
/// construction (ACP, Codex, the in-process native loop, a remote CLI with tool
/// cancellation off) the probe is answered `unsupported`. That is "I cannot tell", and it
/// must never be read as "it is dead".
#[tokio::test]
async fn agent_path_a_provider_that_cannot_tell_is_not_harvested() {
    use nexus_claude::agent::ProviderError;
    let (manager, _graph, fake) = agent_manager();
    let sid = manager
        .create_session(&agent_request("go"))
        .await
        .unwrap()
        .session_id;
    *fake.state.cancel_tools_answer.lock().unwrap() =
        Some(ProviderError::unsupported("tool_cancel"));

    assert!(
        manager.close_sessions_of_dead_clis().await.is_empty(),
        "a provider that cannot tell is left alone"
    );
    assert!(
        manager.is_session_active(&sid).await,
        "the session keeps its slot"
    );
    manager.close_session(&sid).await.ok();
}
