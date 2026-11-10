    use super::*;
    #[tokio::test]
    async fn a_step_past_its_budget_is_said_and_the_next_one_still_runs() {
        let mut after = super::super::agent_runtime::AfterTurn::default();
        let budget = Duration::from_millis(20);
        let stuck = run_step(
            &mut after,
            "s",
            "post_compaction",
            budget,
            std::future::pending::<()>(),
        )
        .await;
        assert!(stuck.is_none());
        let next = run_step(&mut after, "s", "feedback", budget, async { 7 }).await;
        assert_eq!(next, Some(7), "the next step ran");
        assert!(
            matches!(
                after.events.as_slice(),
                [ChatEvent::Error { code: Some(code), reason: Some(step), .. }]
                    if code == super::super::post_stream::STEP_ABANDONED_CODE
                        && step == "post_compaction"
            ),
            "{:?}",
            after.events
        );
    }
