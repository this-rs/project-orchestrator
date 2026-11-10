    use super::*;
    use crate::chat::agent_hooks::{PoolSource, TurnRouter, TurnRouterSpec, TurnRouting};
    use crate::chat::agent_runtime::TurnServices;
    use crate::chat::message_attachments::{self, MessageAttachment};
    use crate::chat::provider::cognitive::candidates::ModelFacts;
    use crate::chat::provider::cognitive::decider::decide_with;
    use crate::chat::provider::cognitive::decision::{CognitiveDecision, DecideRequest, Decider};
    use crate::chat::provider::cognitive::scorer::PriorHints;
    use crate::chat::provider::cognitive::{LearningStage, ProviderRoutingMode, RoutingSettings};
    struct Real;

    #[async_trait::async_trait]
    impl Decider for Real {
        async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            Ok(decide_with(
                Uuid::new_v4(),
                chrono::Utc::now(),
                request,
                &[],
                &PriorHints::new(),
            ))
        }
    }

    struct Pool;

    #[async_trait::async_trait]
    impl PoolSource for Pool {
        async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
            self.project_pool(None)
                .await
                .into_iter()
                .filter(|f| f.provider_id == provider_id)
                .collect()
        }
        async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
            ["flash", "vision"]
                .into_iter()
                .map(|model| ModelFacts {
                    provider_id: "deepseek".into(),
                    model: model.into(),
                    supports_tools: true,
                    supports_images: model == "vision",
                    context_window: Some(1_000_000),
                    window_unknown: None,
                    price: None,
                    cost_basis: nexus_claude::agent::CostBasis::Unknown,
                    healthy: Some(true),
                    allowed_for_project: true,
                    sandboxed: false,
                    quota_exhausted: None,
                    quota_utilization: None,
                })
                .collect()
        }
    }

    #[tokio::test]
    async fn prepare_hands_the_router_the_images_so_the_turn_is_routed_before_it_is_sent() {
        use crate::chat::provider::cognitive::store::{
            DecisionFilter, InMemoryRoutingStore, RoutingArmStore,
        };
        let store = Arc::new(InMemoryRoutingStore::new());
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let manager = ChatManager::new_without_memory(
            graph.clone(),
            crate::test_helpers::mock_app_state().meili,
            ChatConfig {
                max_sessions: 10,
                ..Default::default()
            },
        );
        let session = crate::test_helpers::test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();
        let sid = session.id.to_string();
        let router = Arc::new(TurnRouter::new(TurnRouterSpec {
            decider: Arc::new(Real),
            pool: Arc::new(Pool),
            routing: RoutingSettings {
                mode: ProviderRoutingMode::Full,
                stage: LearningStage::Auto,
                exploration_epsilon: 0.0,
                ..RoutingSettings::default()
            },
            provider_id: "deepseek".into(),
            session_id: Some(session.id),
            project_slug: None,
            trust: false,
            explicit_model: false,
            provider_imposed: false,
            allowed_models: None,
            routing_pool: None,
            auto_models: None,
            conversation_routes: false,
            current_model: "flash".into(),
            next_turn: 0,
            moved_in: false,
            segment_store: None,
            segments: None,
        }));
        router.set_model_live(true);
        let routing = Arc::new(TurnRouting::default());
        routing.insert(&sid, Arc::clone(&router));
        let services = ManagerTurnServices {
            graph: graph.clone(),
            enrichment_pipeline: manager.enrichment_pipeline.clone(),
            turn_routing: routing,
            documents: crate::documents::store::DocumentStore::new(std::env::temp_dir()),
            nats: None,
            anchor: Default::default(),
            session: AgentTurnState::default(),
            search: crate::test_helpers::mock_app_state().meili,
            event_emitter: None,
            context_injector: None,
            step_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
            stall_step: None,
            routing_store: Some(store.clone()),
            open_decisions: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            objective_ledgers: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            mirror: None,
            hook_fallback: None,
        };
        // Before any turn input, nothing says an image comes.
        assert_eq!(services.model_for_images(&sid).await, None);
        let stored = message_attachments::encode(
            "what is on this screenshot?",
            &[MessageAttachment {
                id: Uuid::new_v4(),
                filename: "shot.png".into(),
                mime_type: "image/png".into(),
                size_bytes: 4,
            }],
        );
        let turn = crate::refs::turn::expand_user_turn_if(&graph, &stored, false).await;
        services.prepare(&sid, &stored, &stored, &turn).await;
        assert_eq!(
            services.model_for_images(&sid).await.as_deref(),
            Some("vision")
        );
        // The host could not apply it: the router is put back as it was.
        services.image_model_not_applied(&sid, "vision").await;
        assert_eq!(router.next_turn_context(0).current_model, "flash");
        // ...and the decision, which said `applied: true`, is stored as what happened: not
        // applied, and why (as a legacy `set_model` that fails, or a refused move).
        let stored = store.decisions(&DecisionFilter::default()).await.unwrap();
        assert_eq!(
            stored.len(),
            1,
            "the image decision is written back: {stored:#?}"
        );
        assert!(!stored[0].applied, "{:#?}", stored[0]);
        assert_eq!(
            stored[0].chosen.as_ref().map(|p| p.model.as_str()),
            Some("vision")
        );
        assert!(
            stored[0]
                .reason
                .ends_with("; not_changed: set_model_failed"),
            "{}",
            stored[0].reason
        );
    }
