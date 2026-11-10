    use super::super::lasting_rules::{LastingRuleStore, LastingScope};
    use super::super::session_grants::{cover, AskedCall, Asker, Cover, SessionGrants};
    use super::super::types::PermissionAnswerScope;
    use super::*;
    /// The call the session's pending `req-1` asks (`test_support`).
    fn ls() -> AskedCall {
        AskedCall {
            asker: Asker::ClaudeCode,
            tool: "Bash".into(),
            canonical: None,
            input: serde_json::json!({ "command": "ls" }),
        }
    }

    struct Answered {
        /// What covered the identical call the moment the CLI could read the answer.
        at_the_cli: Option<Cover>,
        /// What covers it once the answer is done.
        after: Option<Cover>,
        /// What covered ANOTHER call (`pwd`, never approved) at the same moment.
        other_at_the_cli: Option<Cover>,
        result: std::result::Result<(), PermissionDeliveryError>,
        _dir: tempfile::TempDir,
    }

    /// A Claude Code session keeping lasting rules (in the project's root) whose `req-1`
    /// is answered with `scope`; `cli_gone` drops the CLI's stdin first.
    async fn answered(scope: PermissionAnswerScope, cli_gone: bool) -> Answered {
        answered_in(scope, cli_gone, false).await
    }

    /// `broken_store`: the lasting rules' directory cannot be written (a file stands there).
    async fn answered_in(
        scope: PermissionAnswerScope,
        cli_gone: bool,
        broken_store: bool,
    ) -> Answered {
        let state = crate::test_helpers::mock_app_state();
        let manager =
            ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config());
        let sid = uuid::Uuid::new_v4().to_string();
        let client = nexus_claude::InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
            model: Some("test".into()),
            cli_path: Some("/nonexistent/claude".into()),
            ..Default::default()
        })
        .unwrap();
        let (stdin_rx, _queue) =
            test_support::insert_session_with_client(&manager, &sid, client, false, &["req-1"])
                .await
                .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir_all(&root).unwrap();
        if broken_store {
            std::fs::write(dir.path().join("rules"), b"not a directory").unwrap();
        }
        let place = LastingScope::of_session(
            Some(&LastingRuleStore::new(dir.path().join("rules"))),
            Some("person"),
            Some("proj"),
            Some(&root),
            &root,
        )
        .expect("a session in the project's root keeps lasting rules");
        let grants = manager.active_sessions.read().await[&sid]
            .session_grants
            .clone();
        *grants.lock().unwrap() = SessionGrants::declaring(Vec::new()).with_lasting(Some(place));
        assert!(
            grants
                .lock()
                .unwrap()
                .always_grant(&ls())
                .is_some_and(|(_, lasting)| lasting),
            "`ls` can be approved for good here"
        );
        let (seen_tx, seen_rx) = tokio::sync::oneshot::channel();
        let (other_tx, other_rx) = tokio::sync::oneshot::channel();
        let probed = Arc::clone(&grants);
        answer_probe::on_cli_has_answer(
            &sid,
            Box::new(move || {
                Box::pin(async move {
                    let _ = seen_tx.send(cover(&probed, &ls()).await);
                    let pwd = AskedCall {
                        input: serde_json::json!({ "command": "pwd" }),
                        ..ls()
                    };
                    let _ = other_tx.send(cover(&probed, &pwd).await);
                })
            }),
        );
        let stdin_rx = if cli_gone {
            drop(stdin_rx);
            None
        } else {
            Some(stdin_rx)
        };
        let result = manager
            .send_permission_response_inner(&sid, "req-1", true, scope, true)
            .await;
        answer_probe::forget(&sid);
        if let Some(mut rx) = stdin_rx {
            assert!(rx.try_recv().unwrap().contains("req-1"), "the CLI has it");
        }
        Answered {
            at_the_cli: seen_rx.await.ok().flatten(),
            other_at_the_cli: other_rx.await.ok().flatten(),
            after: cover(&grants, &ls()).await,
            result,
            _dir: dir,
        }
    }

    #[tokio::test]
    async fn an_always_covers_the_identical_call_as_soon_as_the_cli_has_the_answer() {
        let a = answered(PermissionAnswerScope::Always, false).await;
        assert!(a.result.is_ok(), "{:?}", a.result);
        assert!(
            a.at_the_cli.as_ref().is_some_and(|c| c.lasting),
            "the CLI's next identical call would be left to the user: {:?}",
            a.at_the_cli
        );
        assert!(
            a.after.as_ref().is_some_and(|c| c.lasting),
            "the rule is kept: {:?}",
            a.after
        );
    }

    #[tokio::test]
    async fn a_session_grant_covers_the_identical_call_as_soon_as_the_cli_has_the_answer() {
        let a = answered(PermissionAnswerScope::Session, false).await;
        assert!(a.result.is_ok(), "{:?}", a.result);
        assert!(
            a.at_the_cli.as_ref().is_some_and(|c| !c.lasting),
            "the CLI's next identical call would be left to the user: {:?}",
            a.at_the_cli
        );
        assert!(
            a.after.as_ref().is_some_and(|c| !c.lasting),
            "{:?}",
            a.after
        );
    }

    #[tokio::test]
    async fn an_answer_the_cli_never_got_covers_nothing() {
        for scope in [
            PermissionAnswerScope::Always,
            PermissionAnswerScope::Session,
        ] {
            let a = answered(scope, true).await;
            assert!(
                matches!(a.result, Err(PermissionDeliveryError::Failed(_))),
                "{scope:?}: {:?}",
                a.result
            );
            assert!(a.after.is_none(), "{scope:?} still covers: {:?}", a.after);
        }
    }

    #[tokio::test]
    async fn a_grant_leaves_another_call_to_the_user_even_before_the_cli_reads_it() {
        for scope in [
            PermissionAnswerScope::Always,
            PermissionAnswerScope::Session,
        ] {
            let a = answered(scope, false).await;
            assert!(a.result.is_ok(), "{scope:?}: {:?}", a.result);
            assert!(
                a.at_the_cli.is_some(),
                "{scope:?}: the call itself is covered"
            );
            assert!(
                a.other_at_the_cli.is_none(),
                "{scope:?} covers another call: {:?}",
                a.other_at_the_cli
            );
        }
    }

    #[tokio::test]
    async fn an_always_whose_rule_cannot_be_written_stays_a_session_grant() {
        let a = answered_in(PermissionAnswerScope::Always, false, true).await;
        assert!(a.result.is_ok(), "{:?}", a.result);
        assert!(
            a.at_the_cli.as_ref().is_some_and(|c| c.lasting),
            "pending for good until the write is known: {:?}",
            a.at_the_cli
        );
        assert!(
            a.after.as_ref().is_some_and(|c| !c.lasting),
            "no rule on disk: granted for the session only: {:?}",
            a.after
        );
    }
