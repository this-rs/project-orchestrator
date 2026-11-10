    use super::*;
    use crate::chat::agent_runtime::{fake::FakeProvider, AgentRuntime};
    use crate::chat::provider::cognitive::decision::{CognitiveDecision, Pick};
    use crate::chat::provider::cognitive::signature::{TaskClass, TaskSignature};
    use crate::chat::provider::cognitive::store::{ArmKey, InMemoryRoutingStore, RoutingArmStore};
    use nexus_claude::agent::{AgentEvent, ProviderError, StopReason};

    fn done(stop_reason: StopReason, is_error: bool, error: Option<ProviderError>) -> AgentEvent {
        AgentEvent::Done {
            stop_reason,
            subtype: None,
            is_error,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error,
        }
    }

    fn open_decision(session: Uuid) -> CognitiveDecision {
        CognitiveDecision {
            snapshot: None,
            arms_snapshot: Vec::new(),
            id: Uuid::new_v4(),
            at: chrono::Utc::now(),
            signature: TaskSignature::utility(TaskClass::UtilityCompaction, 9_000, None),
            chosen: Some(Pick::new("deepseek", "v4")),
            score: None,
            explored: false,
            exploration_epsilon: 0.0,
            confidence: None,
            reason: "r0".into(),
            alternatives: vec![],
            applied: true,
            mode: crate::chat::provider::cognitive::mode::ProviderRoutingMode::Full,
            stage: crate::chat::provider::cognitive::mode::LearningStage::Auto,
            session_id: Some(session),
            task_id: None,
            run_id: None,
            turn_index: None,
            outcome: None,
            used: None,
        }
    }

    struct Rig {
        store: Arc<InMemoryRoutingStore>,
        provider: FakeProvider,
        handle: Arc<crate::chat::agent_runtime::AgentSessionHandle>,
        arm: ArmKey,
    }

    /// A native session whose turns go through `play()` and `ManagerTurnServices`, with a
    /// routing decision OPEN for it (as `remember_routing_decision` leaves it at open).
    async fn rig() -> Rig {
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
        let session = Uuid::new_v4();
        let sid = session.to_string();
        let decision = open_decision(session);
        store.put_decision(&decision).await.unwrap();
        let arm = decision.served_arm().unwrap();
        let services = ManagerTurnServices {
            graph: graph.clone(),
            enrichment_pipeline: manager.enrichment_pipeline.clone(),
            turn_routing: Arc::default(),
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
            open_decisions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::from([(
                sid.clone(),
                decision.id,
            )]))),
            objective_ledgers: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            mirror: None,
            hook_fallback: None,
        };
        let provider = FakeProvider::new();
        let handle = AgentRuntime::new(graph)
            .adopt(
                &sid,
                "deepseek",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                Some(Arc::new(services)),
            )
            .await;
        Rig {
            store,
            provider,
            handle,
            arm,
        }
    }

    impl Rig {
        /// One turn: the user's message, the provider's terminal `done`, then waits for the
        /// session to be idle (`after_turn` ran before the stream flag drops).
        async fn turn(&self, terminal: Option<AgentEvent>) {
            let before = self.provider.state.turns_started.lock().unwrap().len();
            self.handle.send_message("go").await.unwrap();
            wait("the turn started", || {
                self.provider.state.turns_started.lock().unwrap().len() > before
            })
            .await;
            match terminal {
                Some(event) => self.provider.state.push(event),
                None => self.provider.state.end_turn(),
            }
            wait("the turn ended", || {
                !self
                    .handle
                    .is_streaming
                    .load(std::sync::atomic::Ordering::SeqCst)
            })
            .await;
        }
        async fn alpha_beta_n(&self) -> Option<(f64, f64, u64)> {
            self.store
                .arm(&self.arm)
                .await
                .unwrap()
                .map(|a| (a.alpha, a.beta, a.n))
        }
    }

    async fn wait(what: &str, check: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !check() {
            assert!(std::time::Instant::now() < deadline, "{what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// AC1 + AC5. Red without the capture of the terminal `done` in `play()`
    /// (`outcome.stop_reason` stays `None`: `after_turn` feeds nothing).
    #[tokio::test]
    async fn a_native_turn_completed_on_an_open_decision_feeds_its_arm_once() {
        let rig = rig().await;
        assert_eq!(rig.alpha_beta_n().await, None, "no arm before any turn");
        rig.turn(Some(done(StopReason::Completed, false, None)))
            .await;
        let (alpha, beta, n) = rig.alpha_beta_n().await.expect("the arm was observed");
        assert_eq!((alpha, beta, n), (2.0, 1.0, 1), "Beta(1,1) + one success");
    }

    /// AC2: a turn with no terminal `done` (the stream closes) leaves the posterior alone.
    #[tokio::test]
    async fn a_turn_with_no_terminal_done_feeds_nothing() {
        let rig = rig().await;
        rig.turn(None).await;
        assert_eq!(rig.alpha_beta_n().await, None);
    }

    /// AC3: an ambiguous stop leaves the posterior untouched.
    #[tokio::test]
    async fn an_ambiguous_stop_leaves_the_posterior_untouched() {
        let rig = rig().await;
        for stop in [
            StopReason::Interrupted,
            StopReason::MaxTurns,
            StopReason::BudgetExceeded,
        ] {
            rig.turn(Some(done(stop, false, None))).await;
            assert_eq!(rig.alpha_beta_n().await, None, "{stop:?} fed the arm");
        }
    }

    /// AC4: a far-reset 429 ends the turn on a refusal of the ACCOUNT: the model is not
    /// penalised. (A real failure of the model, for contrast, does feed a zero.)
    #[tokio::test]
    async fn a_rate_limited_turn_does_not_penalise_the_model() {
        let rig = rig().await;
        let far = u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap() + 3 * 86_400_000;
        rig.turn(Some(done(
            StopReason::Error,
            true,
            Some(ProviderError::RateLimited {
                retry_after_ms: None,
                resets_at_ms: Some(far),
            }),
        )))
        .await;
        assert_eq!(
            rig.alpha_beta_n().await,
            None,
            "a 429 is not the model's fault"
        );
    }

    /// AC6: ten turns of one Auto conversation, counted. One decision is open per
    /// session, so ONE arm moves, ten times.
    #[tokio::test]
    async fn ten_turns_of_a_conversation_move_how_many_arms() {
        let rig = rig().await;
        for _ in 0..10 {
            rig.turn(Some(done(StopReason::Completed, false, None)))
                .await;
        }
        let moved = rig
            .store
            .arms_of_class(&rig.arm.class)
            .await
            .unwrap()
            .into_iter()
            .filter(|a| a.n > 0)
            .count();
        let (alpha, _, n) = rig.alpha_beta_n().await.unwrap();
        println!("R0-AC6: arms moved over 10 turns = {moved}; alpha = {alpha}, n = {n}");
        assert_eq!((moved, n), (1, 10));
        assert_eq!(alpha, 11.0);
    }
