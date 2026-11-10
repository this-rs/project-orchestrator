    use super::*;
    use crate::chat::provider::cognitive::ProviderRoutingMode;
    use crate::chat::provider::resolver::REMOTE_PREFIX;
    use crate::chat::provider::settings::InstanceRecord;
    use crate::chat::types::RoutingPoolEntry;
    fn instance(id: &str, credential_ref: &str) -> InstanceRecord {
        InstanceRecord {
            id: id.to_string(),
            credential_ref: credential_ref.to_string(),
            ..Default::default()
        }
    }

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn anyone(_: &str) -> bool {
        true
    }

    #[test]
    fn only_local_instances_keyed_in_the_vault_are_vault_backed() {
        let remote = format!("{REMOTE_PREFIX}box");
        let instances = vec![
            instance("openai", "vault:openai-key"),
            instance("ollama", "none"),
            instance("groq", "env:GROQ_API_KEY"),
            instance(&remote, "vault:ssh-key"),
        ];
        assert_eq!(
            vault_backed_candidates(&instances, None, anyone, false),
            ids(&["openai"])
        );
    }

    /// Review of #689, finding 2: Claude Code depends on the vault through the
    /// Anthropic catalog; when the locked vault left it offline, it counts.
    #[test]
    fn claude_code_is_vault_backed_while_its_catalog_waits_on_the_vault() {
        let instances = vec![instance("ollama", "none")];
        assert!(vault_backed_candidates(&instances, None, anyone, false).is_empty());
        assert_eq!(
            vault_backed_candidates(&instances, None, anyone, true),
            ids(&["claude-code"])
        );
    }

    /// Review of #689, finding 3: an instance the project has not consented to
    /// never enters the pool, so it never makes Auto refuse.
    #[test]
    fn an_instance_not_allowed_for_the_project_is_not_vault_backed() {
        let instances = vec![
            instance("openai", "vault:openai-key"),
            instance("mistral", "vault:mistral-key"),
        ];
        assert_eq!(
            vault_backed_candidates(&instances, None, |id| id == "mistral", false),
            ids(&["mistral"])
        );
    }

    #[test]
    fn ticked_models_restrict_the_vault_backed_candidates() {
        let instances = vec![
            instance("openai", "vault:openai-key"),
            instance("mistral", "vault:mistral-key"),
        ];
        let ticked = vec![
            RoutingPoolEntry {
                provider: "mistral".into(),
                model: "large".into(),
            },
            RoutingPoolEntry {
                provider: "claude-code".into(),
                model: "opus".into(),
            },
        ];
        assert_eq!(
            vault_backed_candidates(&instances, Some(&ticked), anyone, false),
            ids(&["mistral"])
        );
        assert_eq!(
            vault_backed_candidates(&instances, Some(&ticked), anyone, true),
            ids(&["claude-code", "mistral"])
        );
        let none_keyed = vec![RoutingPoolEntry {
            provider: "ollama".into(),
            model: "qwen3".into(),
        }];
        assert!(vault_backed_candidates(&instances, Some(&none_keyed), anyone, true).is_empty());
    }

    #[test]
    fn auto_is_refused_only_when_it_would_route_around_a_locked_vault() {
        let keyed = ids(&["openai"]);
        let full = ProviderRoutingMode::Full;
        assert!(auto_refused_by_locked_vault(
            full, true, false, true, &keyed
        ));
        // Unlocked vault, named choice, no keyed candidate, or not Auto: open as before.
        assert!(!auto_refused_by_locked_vault(
            full, true, false, false, &keyed
        ));
        assert!(!auto_refused_by_locked_vault(
            full, true, true, true, &keyed
        ));
        assert!(!auto_refused_by_locked_vault(full, true, false, true, &[]));
        // Review of #689, finding 3: the choice is not applied (stage `shadow`, a
        // pilot at `advisory`): the declared rules open, nothing is routed around.
        assert!(!auto_refused_by_locked_vault(
            full, false, false, true, &keyed
        ));
        for mode in [
            ProviderRoutingMode::Primary,
            ProviderRoutingMode::ExecutorsOnly,
        ] {
            assert!(!auto_refused_by_locked_vault(
                mode, true, false, true, &keyed
            ));
        }
    }

    fn decision_with(
        chosen: Option<(&str, &str)>,
        alternatives: &[(&str, &str, Option<f64>, bool)],
    ) -> crate::chat::provider::cognitive::decision::CognitiveDecision {
        use crate::chat::provider::cognitive::candidates::RejectReason;
        use crate::chat::provider::cognitive::decision::{
            CognitiveDecision, DecisionAlternative, Pick,
        };
        use crate::chat::provider::cognitive::mode::LearningStage;
        use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
        CognitiveDecision {
            snapshot: None,
            arms_snapshot: Vec::new(),
            id: Uuid::new_v4(),
            at: chrono::Utc::now(),
            signature: TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, None),
            chosen: chosen.map(|(p, m)| Pick::new(p, m)),
            score: None,
            explored: false,
            exploration_epsilon: 0.0,
            confidence: None,
            reason: "test".into(),
            alternatives: alternatives
                .iter()
                .map(|(p, m, score, rejected)| DecisionAlternative {
                    pick: Pick::new(*p, *m),
                    score: *score,
                    rejected: rejected.then_some(RejectReason::Unhealthy),
                    p_top1: None,
                    propensity: None,
                })
                .collect(),
            applied: true,
            mode: ProviderRoutingMode::Full,
            stage: LearningStage::Auto,
            session_id: None,
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: None,
            used: None,
        }
    }

    fn picks(f: &[crate::chat::provider::errors::FallbackModel]) -> Vec<(String, String)> {
        f.iter()
            .map(|f| (f.provider_id.clone(), f.model.clone()))
            .collect()
    }

    #[test]
    fn fallbacks_are_the_pick_then_eligible_alternatives_without_the_vault() {
        let d = decision_with(
            Some(("codex", "gpt-5")),
            &[
                ("codex", "gpt-5", Some(0.9), false),
                ("ollama", "qwen3", Some(0.4), false),
                ("openclaw", "default", Some(0.7), false),
                ("openai", "gpt-5", Some(0.95), false),
                ("claude-code", "opus", None, true),
                ("lmstudio", "llama", Some(0.1), false),
            ],
        );
        let out = locked_vault_fallbacks(&d, &ids(&["openai"]));
        assert_eq!(
            picks(&out),
            vec![
                ("codex".into(), "gpt-5".into()),
                ("openclaw".into(), "default".into()),
                ("ollama".into(), "qwen3".into()),
            ]
        );
    }

    #[test]
    fn every_model_in_the_vault_leaves_only_unlocking() {
        let d = decision_with(
            None,
            &[
                ("openai", "gpt-5", None, true),
                ("claude-code", "opus", None, true),
            ],
        );
        assert!(locked_vault_fallbacks(&d, &ids(&["openai"])).is_empty());
    }

    #[test]
    fn the_open_budget_stays_under_the_http_request_timeout() {
        let budget = ROUTING_POOL_BOUND + OPEN_PREFLIGHT_TIMEOUT + PROVIDER_OPEN_TIMEOUT;
        assert!(budget < Duration::from_secs(60), "{budget:?}");
    }

    /// A provider whose health answers at once (`healthy`) or never, and whose
    /// catalog never answers: the slow endpoint or agent process of the 408.
    struct Hanging {
        id: String,
        healthy: bool,
    }

    #[async_trait::async_trait]
    impl nexus_claude::agent::AgentProvider for Hanging {
        fn id(&self) -> &str {
            &self.id
        }
        fn kind(&self) -> nexus_claude::agent::ProviderKind {
            nexus_claude::agent::ProviderKind::Native
        }
        async fn health(&self) -> nexus_claude::agent::ProviderHealth {
            if !self.healthy {
                std::future::pending::<()>().await;
            }
            nexus_claude::agent::ProviderHealth {
                status: nexus_claude::agent::HealthStatus::Ok,
                version: None,
                detail: None,
                error: None,
                login_hint: None,
                checked_at_ms: 0,
            }
        }
        async fn catalog(
            &self,
        ) -> Result<Vec<nexus_claude::agent::ModelInfo>, nexus_claude::agent::ProviderError>
        {
            std::future::pending().await
        }
        fn capabilities(&self, _model: Option<&str>) -> nexus_claude::agent::Capabilities {
            nexus_claude::agent::Capabilities::default()
        }
        async fn open(
            &self,
            _spec: nexus_claude::agent::SessionSpec,
        ) -> Result<Arc<dyn nexus_claude::agent::AgentSession>, nexus_claude::agent::ProviderError>
        {
            Err(nexus_claude::agent::ProviderError::EndpointUnreachable {
                detail: "never opens".into(),
            })
        }
        async fn resume(
            &self,
            _spec: nexus_claude::agent::SessionSpec,
            _token: nexus_claude::agent::ResumeToken,
        ) -> Result<Arc<dyn nexus_claude::agent::AgentSession>, nexus_claude::agent::ProviderError>
        {
            Err(nexus_claude::agent::ProviderError::EndpointUnreachable {
                detail: "never opens".into(),
            })
        }
    }

    /// Every id is a [`Hanging`] provider; those named `mute-*` never answer their health.
    struct HangingSource;

    impl crate::chat::agent_runtime::ProviderSource for HangingSource {
        fn get(&self, provider_id: &str) -> Option<Arc<dyn nexus_claude::agent::AgentProvider>> {
            Some(Arc::new(Hanging {
                id: provider_id.to_string(),
                healthy: !provider_id.starts_with("mute-"),
            }))
        }
    }

    /// Review of #689, finding 4: the catalogs were fetched one instance after the
    /// other (2 s each), so the pool's build grew with the number of instances and
    /// the 408 came back past a few of them. Health and catalogs now run at once:
    /// the build stays under `ROUTING_POOL_BOUND` whatever the number of instances.
    #[tokio::test(start_paused = true)]
    async fn the_pool_is_bounded_whatever_the_number_of_slow_instances() {
        use crate::chat::provider::cognitive::decider::CognitiveRouting;
        use crate::chat::provider::cognitive::store::InMemoryRoutingStore;
        use crate::chat::provider::settings::{GLOBAL, INSTANCE_PREFIX};
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let ids = ["slow-1", "slow-2", "slow-3", "slow-4", "mute-1", "mute-2"];
        for id in ids {
            let record = InstanceRecord {
                id: id.into(),
                kind: "openai_compatible".into(),
                credential_ref: "none".into(),
                default_model: Some("m".into()),
                ..Default::default()
            };
            graph
                .put_llm_setting(
                    GLOBAL,
                    &format!("{INSTANCE_PREFIX}{id}"),
                    &serde_json::to_string(&record).unwrap(),
                )
                .await
                .unwrap();
        }
        let state = crate::test_helpers::mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph;
        let config = ChatConfig {
            provider_path: crate::chat::config::ProviderPath::Agent,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(HangingSource))
            .with_cognitive_routing(CognitiveRouting::new(Arc::new(InMemoryRoutingStore::new())));
        let start = tokio::time::Instant::now();
        let pool = manager.routing_pool_known(None).await;
        let took = start.elapsed();
        assert!(took <= ROUTING_POOL_BOUND, "built in {took:?}");
        let health = |id: &str| {
            pool.iter()
                .find(|f| f.provider_id == id)
                .unwrap_or_else(|| panic!("{id} in the pool: {pool:?}"))
                .healthy
        };
        // Still the right facts: a health that never answers is unavailable, a
        // catalog that never answers leaves the instance its default model.
        assert_eq!(health("slow-4"), Some(true));
        assert_eq!(health("mute-2"), Some(false));
    }
