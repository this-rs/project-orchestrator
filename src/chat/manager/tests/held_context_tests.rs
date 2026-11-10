    use super::*;
    use crate::chat::agent_runtime::{TurnOutcome, TurnServices};
    fn test_config() -> ChatConfig {
        ChatConfig {
            max_sessions: 10,
            ..Default::default()
        }
    }

    fn services(manager: &ChatManager, state: AgentTurnState) -> ManagerTurnServices {
        ManagerTurnServices {
            graph: manager.graph.clone(),
            enrichment_pipeline: manager.enrichment_pipeline.clone(),
            turn_routing: Arc::default(),
            documents: crate::documents::store::DocumentStore::new(std::env::temp_dir()),
            nats: None,
            anchor: Default::default(),
            session: state,
            search: crate::test_helpers::mock_app_state().meili,
            event_emitter: None,
            context_injector: None,
            step_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
            stall_step: None,
            routing_store: None,
            open_decisions: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            objective_ledgers: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            mirror: None,
            hook_fallback: None,
        }
    }

    async fn world() -> (ChatManager, String) {
        world_on(Arc::new(crate::neo4j::mock::MockGraphStore::new())).await
    }

    /// [`world`] on a store the test drives (slow, failing writes).
    async fn world_on(mock: Arc<crate::neo4j::mock::MockGraphStore>) -> (ChatManager, String) {
        let graph: Arc<dyn GraphStore> = mock;
        let manager = ChatManager::new_without_memory(
            graph.clone(),
            crate::test_helpers::mock_app_state().meili,
            test_config(),
        );
        let mut project = crate::test_helpers::test_project();
        project.slug = "held".into();
        graph.create_project(&project).await.unwrap();
        let session = crate::test_helpers::test_chat_session(Some("held"));
        graph.create_chat_session(&session).await.unwrap();
        (manager, session.id.to_string())
    }

    async fn prepared(s: &ManagerTurnServices, sid: &str, text: &str) -> String {
        let turn = crate::refs::turn::expand_user_turn_if(&s.graph, text, false).await;
        s.prepare(sid, text, text, &turn).await
    }

    fn compacted() -> TurnOutcome {
        TurnOutcome {
            compacted: true,
            ..Default::default()
        }
    }

    fn answered() -> TurnOutcome {
        TurnOutcome {
            answered: true,
            ..Default::default()
        }
    }

    const HELD: &str = "Post-Compaction Context";

    /// The clear of a delivered context fails (the store errs): the failure is not
    /// lost — the slot knows the store is behind, a reopen in this process does not
    /// reload the delivered context, and the next turn, whatever it is, writes again
    /// until the store follows (else a restart, an idle close or a relay would
    /// re-inject it, maybe days later).
    #[tokio::test]
    async fn a_failed_clear_is_retried_on_the_next_turn() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (manager, sid) = world_on(mock.clone()).await;
        let s = services(&manager, manager.agent_turn_state(&sid).await);
        s.after_turn(&sid, &compacted()).await;
        assert!(!HeldContext::load(&s.graph, &sid).await.unwrap().is_empty());
        mock.fail_llm_setting_writes(HeldContext::KEY, true);
        s.after_turn(&sid, &answered()).await;
        assert!(!prepared(&s, &sid, "one").await.contains(HELD), "delivered");
        assert!(
            !HeldContext::load(&s.graph, &sid).await.unwrap().is_empty(),
            "the store refused the clear"
        );
        assert!(s.session.held.unsynced(), "and the slot knows it");
        let reopened = services(&manager, manager.agent_turn_state(&sid).await);
        assert!(
            !prepared(&reopened, &sid, "two").await.contains(HELD),
            "a reopen keeps memory over a store known to be behind"
        );
        // Still failing: a plain turn tries again, and keeps knowing.
        s.after_turn(&sid, &TurnOutcome::default()).await;
        assert!(s.session.held.unsynced());
        // The store answers again: the next turn's write lands.
        mock.fail_llm_setting_writes(HeldContext::KEY, false);
        s.after_turn(&sid, &TurnOutcome::default()).await;
        assert!(
            HeldContext::load(&s.graph, &sid).await.unwrap().is_empty(),
            "retried"
        );
        assert!(!s.session.held.unsynced());
    }

    /// The clear of a delivered context is slow (the store does not answer): the
    /// turn does not wait past the step budget and says the write is late
    /// (`persistence_delayed`); the next turn compacts again meanwhile, and its
    /// store is ORDERED after that clear — the newer context is what the store ends
    /// with, not an empty one landed last, also once the stalled store answers.
    /// The budget (1 s: the clear dropped after 2 s) leaves the stalled clear in
    /// flight well after the newer store would land without the lock, so the test
    /// is red without it however slow the machine.
    #[tokio::test]
    async fn a_slow_clear_is_said_late_and_a_newer_store_lands_after_it() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (manager, sid) = world_on(mock.clone()).await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.step_budget = Duration::from_secs(1);
        s.after_turn(&sid, &compacted()).await;
        mock.stall_llm_setting_deletes(HeldContext::KEY);
        let started = std::time::Instant::now();
        let after = s.after_turn(&sid, &answered()).await;
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(late(&after, "clear_held_context"), "{:?}", after.events);
        // The next turn (not carrying a context) compacts again while the clear waits.
        s.after_turn(&sid, &compacted()).await;
        let stored_context = || async {
            HeldContext::load(&s.graph, &sid)
                .await
                .unwrap()
                .context
                .is_some()
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !stored_context().await {
            assert!(
                std::time::Instant::now() < deadline,
                "the newer context was never stored"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        mock.release_llm_setting_deletes(HeldContext::KEY);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            stored_context().await,
            "the newer context is stored, the late clear landed before it"
        );
    }

    /// Whether `after` says the write `step` is late (`persistence_delayed`).
    fn late(after: &crate::chat::agent_runtime::AfterTurn, step: &str) -> bool {
        after.events.iter().any(|e| {
            matches!(
                e,
                ChatEvent::Error { code: Some(code), reason: Some(reason), .. }
                    if code == crate::chat::post_stream::PERSISTENCE_DELAYED_CODE
                        && reason == step
            )
        })
    }

    /// Waits (5 s at most) until `cond` holds.
    async fn until(what: &str, cond: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !cond() {
            assert!(std::time::Instant::now() < deadline, "never: {what}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// The session is reopened (`close` does not stop the earlier handle's
    /// `after_turn`, `adopt` replaces a live handle) while the earlier handle's turn
    /// compacted and holds its new context in memory, not yet written (its steps
    /// still run). The reopen keeps memory instead of reloading the store, which
    /// does not have it yet: the new context is neither lost in memory nor
    /// overwritten in the store by the reloaded one. Red without the generation:
    /// the slot looked in step, the reopen reloaded nothing over it, and the turn's
    /// write then stored that nothing.
    #[tokio::test]
    async fn a_reopen_during_an_earlier_turn_keeps_the_context_it_holds() {
        let (manager, sid) = world().await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.step_budget = Duration::from_millis(500);
        s.stall_step = Some("objective_tracking");
        let s = Arc::new(s);
        let turn = {
            let (s, sid) = (Arc::clone(&s), sid.clone());
            tokio::spawn(async move { s.after_turn(&sid, &compacted()).await })
        };
        until("the compaction's context is held", || {
            s.session.held.held().context.is_some()
        })
        .await;
        let reopened = services(&manager, manager.agent_turn_state(&sid).await);
        turn.await.unwrap();
        assert!(
            prepared(&reopened, &sid, "next").await.contains(HELD),
            "the reopened session carries the new context"
        );
        assert!(
            HeldContext::load(&s.graph, &sid)
                .await
                .unwrap()
                .context
                .is_some(),
            "and the store has it"
        );
    }

    /// The other way round: the earlier handle's turn delivered the context and
    /// emptied memory, its clear not written yet; a reopen meanwhile does not
    /// reload the delivered context from the store (it would ride a second turn).
    #[tokio::test]
    async fn a_reopen_during_an_earlier_turn_does_not_reload_a_delivered_context() {
        let (manager, sid) = world().await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.after_turn(&sid, &compacted()).await;
        s.step_budget = Duration::from_millis(500);
        s.stall_step = Some("context");
        let s = Arc::new(s);
        let turn = {
            let (s, sid) = (Arc::clone(&s), sid.clone());
            tokio::spawn(async move { s.after_turn(&sid, &answered()).await })
        };
        until("the delivered context is dropped in memory", || {
            s.session.held.held().is_empty()
        })
        .await;
        let reopened = services(&manager, manager.agent_turn_state(&sid).await);
        assert!(
            !prepared(&reopened, &sid, "next").await.contains(HELD),
            "not reloaded from the store the clear has not reached yet"
        );
        turn.await.unwrap();
        assert!(HeldContext::load(&s.graph, &sid).await.unwrap().is_empty());
    }

    /// The store cannot be read when the session is reopened: memory is kept, not
    /// replaced by an empty context.
    #[tokio::test]
    async fn a_store_read_error_at_reopen_keeps_memory() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (manager, sid) = world_on(mock.clone()).await;
        let s = services(&manager, manager.agent_turn_state(&sid).await);
        s.after_turn(&sid, &compacted()).await;
        assert!(!s.session.held.unsynced());
        mock.fail_llm_setting_reads(HeldContext::KEY, true);
        let reopened = services(&manager, manager.agent_turn_state(&sid).await);
        assert!(
            prepared(&reopened, &sid, "next").await.contains(HELD),
            "a read error is not an empty context"
        );
        mock.fail_llm_setting_reads(HeldContext::KEY, false);
    }

    /// The store read of a reopen never answers: it is dropped after twice the step
    /// budget, memory is kept, and the session's write lock is released — the next
    /// writes are not stuck behind it. Red without the bound: the reopen never
    /// returns, holding the lock.
    #[tokio::test]
    async fn a_store_read_that_never_answers_at_reopen_releases_the_lock() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (manager, sid) = world_on(mock.clone()).await;
        let s = services(&manager, manager.agent_turn_state(&sid).await);
        s.after_turn(&sid, &compacted()).await;
        assert!(!s.session.held.unsynced());
        mock.stall_llm_setting_reads(HeldContext::KEY);
        let budget = Duration::from_millis(100);
        let reopen = tokio::time::timeout(
            Duration::from_secs(5),
            s.session.held.reopen(&s.graph, &sid, budget),
        )
        .await;
        assert!(reopen.is_ok(), "the reopen waited on the store for good");
        assert!(
            s.session.held.write.try_lock().is_ok(),
            "the write lock is released"
        );
        assert!(
            s.session.held.held().context.is_some(),
            "a read that never answers is not an empty context"
        );
        // A write goes through meanwhile.
        s.session.held.edit(|h| *h = HeldContext::default());
        s.session
            .held
            .write_once(&s.graph, &sid, budget)
            .await
            .unwrap();
        assert!(!s.session.held.unsynced());
        mock.release_llm_setting_reads(HeldContext::KEY);
        assert!(HeldContext::load(&s.graph, &sid).await.unwrap().is_empty());
    }

    /// A failed write is retried in the background, without waiting for a turn: a
    /// session closed or left right after keeps no delivered context in the store
    /// for a restart to reload.
    #[tokio::test]
    async fn a_failed_write_is_retried_without_a_turn() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (manager, sid) = world_on(mock.clone()).await;
        let s = services(&manager, manager.agent_turn_state(&sid).await);
        s.after_turn(&sid, &compacted()).await;
        mock.fail_llm_setting_writes(HeldContext::KEY, true);
        s.after_turn(&sid, &answered()).await;
        assert!(s.session.held.unsynced());
        mock.fail_llm_setting_writes(HeldContext::KEY, false);
        // No turn from here on.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while s.session.held.unsynced() {
            assert!(
                std::time::Instant::now() < deadline,
                "the failed clear was never retried"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(HeldContext::load(&s.graph, &sid).await.unwrap().is_empty());
    }

    /// A store write that never answers is dropped after twice the step budget: the
    /// session's write lock is released (later turns and reopens do not wait on
    /// it) and the slot stays unsynced (written again later). Red without the
    /// bound: the lock is held for as long as the store does not answer.
    #[tokio::test]
    async fn a_store_write_that_never_answers_releases_the_lock() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let (manager, sid) = world_on(mock.clone()).await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.step_budget = Duration::from_millis(100);
        s.after_turn(&sid, &compacted()).await;
        mock.stall_llm_setting_deletes(HeldContext::KEY);
        let after = s.after_turn(&sid, &answered()).await;
        assert!(late(&after, "clear_held_context"), "{:?}", after.events);
        until("the write lock is released", || {
            s.session.held.write.try_lock().is_ok()
        })
        .await;
        assert!(s.session.held.unsynced(), "the clear did not land");
        mock.release_llm_setting_deletes(HeldContext::KEY);
    }

    /// A turn carrying a context compacts and is answered, but its re-injection
    /// gives nothing in time (the step stalls): what it carried — the context AND
    /// its objective reminder — waits again, in memory and in the store.
    #[tokio::test]
    async fn a_compaction_without_a_fresh_context_holds_what_the_turn_carried() {
        let (manager, sid) = world().await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.step_budget = Duration::from_millis(200);
        s.stall_step = Some("post_compaction");
        let carried = HeldContext {
            context: Some("H1 carried context".into()),
            reminder: Some("R1 carried reminder".into()),
        };
        s.session.held.edit(|h| *h = carried.clone());
        s.after_turn(
            &sid,
            &TurnOutcome {
                answered: true,
                compacted: true,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(s.session.held.held(), carried);
        assert_eq!(HeldContext::load(&s.graph, &sid).await.unwrap(), carried);
    }

    /// The server goes down between the compaction and the next turn: the session
    /// resumed on a new state still gets the context in front of that turn.
    #[tokio::test]
    async fn the_held_context_survives_a_restart() {
        let (manager, sid) = world().await;
        let before = services(&manager, manager.agent_turn_state(&sid).await);
        before.after_turn(&sid, &compacted()).await;
        drop(before);
        let after = services(&manager, manager.agent_turn_state(&sid).await);
        let sent = prepared(&after, &sid, "next").await;
        assert!(sent.contains("Post-Compaction Context"), "{sent}");
    }

    /// A turn that was prepared but not answered (a Stop during the enrichment, a
    /// refused image, a provider error, a failure after `send_turn` accepted it)
    /// leaves the context for the next one; a turn answered drops it, in memory and
    /// in the store.
    #[tokio::test]
    async fn the_held_context_is_dropped_only_once_a_turn_is_answered() {
        let (manager, sid) = world().await;
        let s = services(&manager, manager.agent_turn_state(&sid).await);
        s.after_turn(&sid, &compacted()).await;
        let first = prepared(&s, &sid, "one").await;
        assert!(first.contains("Post-Compaction Context"));
        // The turn failed: not answered.
        s.after_turn(&sid, &TurnOutcome::default()).await;
        let again = prepared(&s, &sid, "two").await;
        assert!(again.contains("Post-Compaction Context"), "kept: {again}");
        assert!(!HeldContext::load(&s.graph, &sid).await.unwrap().is_empty());
        s.after_turn(
            &sid,
            &TurnOutcome {
                answered: true,
                ..Default::default()
            },
        )
        .await;
        let then = prepared(&s, &sid, "three").await;
        assert!(!then.contains("Post-Compaction Context"), "dropped: {then}");
        // And from the store too: written before `after_turn` returned.
        assert!(HeldContext::load(&s.graph, &sid).await.unwrap().is_empty());
    }

    /// B14 (review of #814): the post-tool advice of a provider without hooks is
    /// carried like the held context — a turn prepared but not answered (refused,
    /// failed, stopped) leaves it for the next one; a turn answered consumes it, once.
    #[tokio::test]
    async fn the_post_tool_advice_is_dropped_only_once_a_turn_is_answered() {
        use super::super::agent_runtime::TurnServices as _;
        let (manager, sid) = world().await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.hook_fallback = Some(Arc::new(super::super::provider::hooks::HookFallback::new(
            Arc::new(super::super::provider::hooks::TestAdvisor),
            Arc::new(super::super::hook_ledger::HookLedger::new()),
            sid.clone(),
            "/tmp",
        )));
        s.observe(
            &sid,
            &ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "Grep".into(),
                input: serde_json::json!({}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
        );
        s.observe(
            &sid,
            &ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!("30 lines"),
                is_error: false,
                parent_tool_use_id: None,
            },
        );
        // The turn of the Grep, answered: its advice waits for the next turn.
        s.after_turn(&sid, &answered()).await;
        let advice = "use find_references (30 lines)";
        assert!(prepared(&s, &sid, "one").await.contains(advice));
        // That turn failed: not answered, the advice is kept.
        s.after_turn(&sid, &TurnOutcome::default()).await;
        let again = prepared(&s, &sid, "two").await;
        assert!(again.contains(advice), "kept: {again}");
        // Answered: consumed, never carried twice.
        s.after_turn(&sid, &answered()).await;
        let then = prepared(&s, &sid, "three").await;
        assert!(!then.contains(advice), "dropped: {then}");
    }

    #[test]
    fn the_held_context_is_cut_to_a_share_of_the_window() {
        let long = format!(
            "<system-reminder>\n{}\n</system-reminder>",
            "x".repeat(20_000)
        );
        let small = HeldContext::cap(&long, Some(8_000));
        assert!(small.chars().count() <= 1_600, "{}", small.chars().count());
        assert!(small.ends_with("</system-reminder>"));
        let unknown = HeldContext::cap(&long, None);
        assert!(unknown.chars().count() <= 6_000);
        let big = HeldContext::cap(&long, Some(1_000_000));
        assert!(
            big.chars().count() <= 6_000,
            "never above the builder's cap"
        );
        assert_eq!(HeldContext::cap("short", Some(8_000)), "short");
    }

    /// One step that never answers costs that step only: the next steps run, each in
    /// its own budget (a shared budget would drop them with it).
    #[tokio::test]
    async fn a_stuck_step_does_not_take_the_next_ones_with_it() {
        let (manager, sid) = world().await;
        let mut s = services(&manager, manager.agent_turn_state(&sid).await);
        s.step_budget = Duration::from_millis(100);
        s.stall_step = Some("context");
        let started = std::time::Instant::now();
        let after = s.after_turn(&sid, &TurnOutcome::default()).await;
        assert!(
            after.events.iter().any(|e| matches!(
                e,
                ChatEvent::Error { reason: Some(step), .. } if step == "context"
            )),
            "{:?}",
            after.events
        );
        assert_eq!(
            s.session
                .objectives
                .turns_since
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the objective step still ran after the stuck one"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }
