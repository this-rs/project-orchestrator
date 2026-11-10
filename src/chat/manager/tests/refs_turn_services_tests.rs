
use super::*;
use crate::chat::agent_runtime::TurnServices;
use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, EnrichmentPipeline, EnrichmentSource,
    ParallelEnrichmentStage, StageOutput,
};
#[allow(unused_imports)]
use crate::refs::types::{EntityRef, RefKind};
use std::sync::Mutex as StdMutex;

/// What one enrichment read: the message and the notes it must not inject.
type Seen = (String, std::collections::HashSet<String>);

/// Records what each enrichment reads, and answers one section.
#[derive(Default)]
struct Recording {
    seen: Arc<StdMutex<Vec<Seen>>>,
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for Recording {
    async fn execute(&self, input: &EnrichmentInput) -> anyhow::Result<StageOutput> {
        self.seen
            .lock()
            .unwrap()
            .push((input.message.clone(), input.excluded_note_ids.clone()));
        let mut out = StageOutput::new("recording");
        out.add_section("CTX", "graph context", "recording", EnrichmentSource::Other);
        Ok(out)
    }
    fn name(&self) -> &str {
        "recording"
    }
    fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
        true
    }
}

#[tokio::test]
async fn prepare_enriches_the_typed_text_once_and_sends_the_pointers_behind_the_relay() {
    let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let session = crate::test_helpers::test_chat_session(None);
    mock.create_chat_session(&session).await.unwrap();
    let graph: Arc<dyn GraphStore> = mock;
    let stage = Recording::default();
    let seen = Arc::clone(&stage.seen);
    let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
    pipeline.add_parallel_stage(Box::new(stage));
    let services = ManagerTurnServices {
        graph: graph.clone(),
        enrichment_pipeline: Arc::new(pipeline),
        turn_routing: Arc::default(),
        nats: None,
        documents: crate::documents::store::DocumentStore::new(std::env::temp_dir()),
        anchor: Default::default(),
        session: Default::default(),
        search: crate::test_helpers::mock_app_state().meili,
        event_emitter: None,
        context_injector: None,
        step_budget: crate::chat::post_stream::POST_STREAM_STEP_BUDGET,
        stall_step: None,
        routing_store: None,
        open_decisions: std::sync::Arc::new(
            std::sync::Mutex::new(std::collections::HashMap::new()),
        ),
        objective_ledgers: std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::HashMap::new(),
        )),
        mirror: None,
        hook_fallback: None,
    };

    let note = Uuid::new_v4();
    let typed = "look at #note:x";
    let stored = crate::refs::block::encode(typed, &[EntityRef::new(RefKind::Note, note)]);
    let sent = format!("RELAY HISTORY\n\n{stored}");
    let turn = crate::refs::turn::expand_user_turn_if(&graph, &stored, true).await;
    let out = services
        .prepare(&session.id.to_string(), &stored, &sent, &turn)
        .await;

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "one enrichment per turn: {seen:?}");
    assert_eq!(seen[0].0, typed, "the enrichment reads the typed text");
    assert!(
        seen[0].1.contains(&note.to_string()),
        "the note pointed at is not injected again: {seen:?}"
    );
    assert!(
            out.starts_with("## CTX\ngraph context\n\n---\n\nRELAY HISTORY\n\nlook at #note:x\n\n<po-context nonce=\""),
            "{out}"
        );
    assert!(!out.contains("<po-refs>"), "{out}");
}
