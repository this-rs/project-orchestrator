//! Generator state execution
//!
//! When a protocol run enters a Generator state, this module creates
//! RuntimeStates dynamically based on the GeneratorConfig. The generated
//! states are linked according to the LinkingStrategy (Sequential or Parallel).
//!
//! # Safety Limits
//! - MAX_RUNTIME_STATES_PER_GENERATOR = 50
//! - Generation is idempotent: if RuntimeStates already exist for the
//!   (run_id, generator_state_id) pair, generation is skipped.

use crate::neo4j::traits::GraphStore;
use crate::protocol::{GeneratorConfig, LinkingStrategy, ProtocolRun, ProtocolState, RuntimeState};
use anyhow::{bail, Result};

/// Maximum number of RuntimeStates a single Generator can produce.
pub const MAX_RUNTIME_STATES_PER_GENERATOR: usize = 50;

/// Generate RuntimeStates for a Generator state.
///
/// Called when a protocol run enters a state with `state_type == Generator`.
/// The `GeneratorConfig` on the state determines:
/// - `data_source`: what data to generate from (dispatched, for now "test" returns 3 items)
/// - `state_template`: name template for generated states
/// - `linking`: how states are connected (Sequential or Parallel)
///
/// # Idempotence
/// If RuntimeStates already exist for this run + generator state, returns
/// the existing ones without creating duplicates.
///
/// # Returns
/// The list of created (or existing) RuntimeStates, ordered by index.
pub async fn generate(
    state: &ProtocolState,
    run: &ProtocolRun,
    store: &dyn GraphStore,
) -> Result<Vec<RuntimeState>> {
    let config = match &state.generator_config {
        Some(c) => c,
        None => bail!(
            "Generator state '{}' ({}) has no generator_config",
            state.name,
            state.id
        ),
    };

    // Idempotence check: skip if RuntimeStates already exist for this run+state
    let existing = store.get_runtime_states(run.id).await?;
    let already_generated: Vec<_> = existing
        .iter()
        .filter(|rs| rs.generated_by == state.id)
        .cloned()
        .collect();
    if !already_generated.is_empty() {
        return Ok(already_generated);
    }

    // Dispatch data source to get item count
    let items = resolve_data_source(config).await?;

    if items > MAX_RUNTIME_STATES_PER_GENERATOR {
        bail!(
            "Generator '{}' would produce {} states, exceeding limit of {}",
            state.name,
            items,
            MAX_RUNTIME_STATES_PER_GENERATOR
        );
    }

    // Create RuntimeStates (without persisting yet — we need all IDs for linking)
    let mut runtime_states = Vec::with_capacity(items);
    for i in 0..items {
        let name = config.state_template.replace("{index}", &i.to_string());
        let mut rs = RuntimeState::new(run.id, state.id, name, i as u32);
        rs.sub_protocol_id = config.sub_protocol_id;
        rs.linking_strategy = config.linking;
        runtime_states.push(rs);
    }

    // Apply linking strategy
    match config.linking {
        LinkingStrategy::Sequential => {
            // Chain: state_0 → state_1 → state_2 → ... (last has next = None)
            for i in 0..runtime_states.len().saturating_sub(1) {
                let next_id = runtime_states[i + 1].id;
                runtime_states[i].next_runtime_state_id = Some(next_id);
            }
        }
        LinkingStrategy::Parallel => {
            // All states are independent — no next_runtime_state_id linking.
            // Fan-out from generator state to all, fan-in implicit when all complete.
        }
    }

    // Persist all RuntimeStates
    for rs in &runtime_states {
        store.create_runtime_state(rs).await?;
    }

    Ok(runtime_states)
}

/// Resolve a data source string to produce a count of items to generate.
///
/// For now, this is a simple dispatcher:
/// - "test" / "test:N" → returns N items (default 3)
/// - Other sources → returns 0 (future: query plans, tasks, etc.)
async fn resolve_data_source(config: &GeneratorConfig) -> Result<usize> {
    let source = config.data_source.trim();

    if source == "test" {
        return Ok(3);
    }

    if let Some(count_str) = source.strip_prefix("test:") {
        let count: usize = count_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid test count: '{}'", count_str))?;
        return Ok(count);
    }

    // "plan.get_waves" and similar plan-based sources require runtime context
    // (plan_id, store access). For now, they are not resolvable at generation time
    // and return 0 items. The caller should use "test:N" for testing or provide
    // a concrete plan-aware generator in the future.
    if source.starts_with("plan.") {
        return Ok(0);
    }

    // Unknown data sources return 0 items (no-op)
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use crate::protocol::{
        GeneratorConfig, LinkingStrategy, Protocol, ProtocolCategory, ProtocolState,
        ProtocolTransition, StateType,
    };
    use uuid::Uuid;

    /// Helper: create a project + protocol with a Generator state.
    async fn setup_generator_protocol(
        store: &MockGraphStore,
        data_source: &str,
        linking: LinkingStrategy,
    ) -> (Protocol, ProtocolState, ProtocolState, ProtocolState) {
        let project_id = Uuid::new_v4();
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "gen-test".to_string(),
            slug: "gen-test".to_string(),
            description: None,
            root_path: "/tmp/gen".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();
        let start = ProtocolState::start(protocol_id, "Start");

        let mut gen_state = ProtocolState::new(protocol_id, "Generator");
        gen_state.state_type = StateType::Generator;
        gen_state.generator_config = Some(GeneratorConfig {
            data_source: data_source.to_string(),
            state_template: "Item {index}".to_string(),
            sub_protocol_id: None,
            linking,
        });

        let done = ProtocolState::terminal(protocol_id, "Done");

        let mut protocol = Protocol::new_full(
            project_id,
            "Generator Protocol",
            "Test generator",
            start.id,
            vec![done.id],
            ProtocolCategory::Business,
        );
        protocol.id = protocol_id;

        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&start).await.unwrap();
        store.upsert_protocol_state(&gen_state).await.unwrap();
        store.upsert_protocol_state(&done).await.unwrap();

        let t1 = ProtocolTransition::new(protocol_id, start.id, gen_state.id, "generate");
        let t2 = ProtocolTransition::new(protocol_id, gen_state.id, done.id, "generated_complete");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        (protocol, start, gen_state, done)
    }

    #[tokio::test]
    async fn test_generate_creates_runtime_states() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test", LinkingStrategy::Sequential).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        let states = generate(&gen_state, &run, &store).await.unwrap();
        assert_eq!(states.len(), 3);
        assert_eq!(states[0].name, "Item 0");
        assert_eq!(states[1].name, "Item 1");
        assert_eq!(states[2].name, "Item 2");
        assert_eq!(states[0].index, 0);
        assert_eq!(states[2].index, 2);
        assert_eq!(states[0].status, "pending");

        // Verify persisted
        let loaded = store.get_runtime_states(run.id).await.unwrap();
        assert_eq!(loaded.len(), 3);
    }

    #[tokio::test]
    async fn test_generate_idempotent() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test", LinkingStrategy::Sequential).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        let states1 = generate(&gen_state, &run, &store).await.unwrap();
        let states2 = generate(&gen_state, &run, &store).await.unwrap();

        // Same IDs — no duplicates
        assert_eq!(states1.len(), states2.len());
        assert_eq!(states1[0].id, states2[0].id);

        // Only 3 in store total
        let loaded = store.get_runtime_states(run.id).await.unwrap();
        assert_eq!(loaded.len(), 3);
    }

    #[tokio::test]
    async fn test_generate_custom_count() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test:5", LinkingStrategy::Parallel).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        let states = generate(&gen_state, &run, &store).await.unwrap();
        assert_eq!(states.len(), 5);
    }

    #[tokio::test]
    async fn test_generate_max_limit() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test:51", LinkingStrategy::Sequential).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        let result = generate(&gen_state, &run, &store).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeding limit"));
    }

    #[tokio::test]
    async fn test_generate_no_config_error() {
        let store = MockGraphStore::new();
        let (protocol, _start, _gen_state, _done) =
            setup_generator_protocol(&store, "test", LinkingStrategy::Sequential).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Create a state without generator_config
        let bare_state = ProtocolState::new(protocol.id, "Bare");
        let result = generate(&bare_state, &run, &store).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no generator_config"));
    }

    #[tokio::test]
    async fn test_generate_unknown_source_returns_empty() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "unknown_source", LinkingStrategy::Sequential).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        let states = generate(&gen_state, &run, &store).await.unwrap();
        assert_eq!(states.len(), 0);
    }

    #[tokio::test]
    async fn test_delete_runtime_states_cascade() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test", LinkingStrategy::Sequential).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        generate(&gen_state, &run, &store).await.unwrap();
        assert_eq!(store.get_runtime_states(run.id).await.unwrap().len(), 3);

        store.delete_runtime_states(run.id).await.unwrap();
        assert_eq!(store.get_runtime_states(run.id).await.unwrap().len(), 0);
    }

    // ================================================================
    // T5.4 — Wave-dispatch Protocol PoC
    // ================================================================

    /// End-to-end test: a protocol with a Generator state configured with
    /// data_source "test", linking Sequential. Verifies that entering the
    /// Generator state via fire_transition creates RuntimeStates and the
    /// run can progress through to completion.
    #[tokio::test]
    async fn test_wave_dispatch_sequential_poc() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test", LinkingStrategy::Sequential).await;

        // 1. Start the run
        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(run.status, crate::protocol::RunStatus::Running);

        // 2. Fire "generate" trigger → transitions to Generator state
        //    fire_transition_inner should call generator::generate() automatically
        let result = crate::protocol::engine::fire_transition(&store, run.id, "generate")
            .await
            .unwrap();
        assert!(result.success, "Transition to Generator state failed");
        assert_eq!(result.current_state_name, "Generator");
        assert!(!result.run_completed);

        // 3. Verify RuntimeStates were created by fire_transition
        let runtime_states = store.get_runtime_states(run.id).await.unwrap();
        assert_eq!(
            runtime_states.len(),
            3,
            "Generator should produce 3 RuntimeStates from test data source"
        );

        // Verify sequential ordering
        assert_eq!(runtime_states[0].name, "Item 0");
        assert_eq!(runtime_states[0].index, 0);
        assert_eq!(runtime_states[1].name, "Item 1");
        assert_eq!(runtime_states[1].index, 1);
        assert_eq!(runtime_states[2].name, "Item 2");
        assert_eq!(runtime_states[2].index, 2);

        // Verify all belong to this run and were generated by the generator state
        for rs in &runtime_states {
            assert_eq!(rs.run_id, run.id);
            assert_eq!(rs.generated_by, gen_state.id);
            assert_eq!(rs.status, "pending");
        }

        // 4. Fire "generated_complete" → transitions to Done (terminal)
        let result2 =
            crate::protocol::engine::fire_transition(&store, run.id, "generated_complete")
                .await
                .unwrap();
        assert!(result2.success, "Transition to Done failed");
        assert_eq!(result2.current_state_name, "Done");
        assert!(result2.run_completed);
        assert_eq!(result2.status, crate::protocol::RunStatus::Completed);

        // 5. Verify run is completed
        let final_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(final_run.status, crate::protocol::RunStatus::Completed);
        assert!(final_run.completed_at.is_some());

        // Verify state visit history
        let names: Vec<_> = final_run
            .states_visited
            .iter()
            .map(|sv| sv.state_name.as_str())
            .collect();
        assert_eq!(names, vec!["Start", "Generator", "Done"]);
    }

    /// Verify that re-entering a Generator state (idempotence through engine)
    /// does not create duplicate RuntimeStates.
    #[tokio::test]
    async fn test_wave_dispatch_idempotent_via_engine() {
        let store = MockGraphStore::new();
        let (protocol, _start, gen_state, _done) =
            setup_generator_protocol(&store, "test:2", LinkingStrategy::Parallel).await;

        let run = crate::protocol::engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Transition to Generator state
        crate::protocol::engine::fire_transition(&store, run.id, "generate")
            .await
            .unwrap();

        let states1 = store.get_runtime_states(run.id).await.unwrap();
        assert_eq!(states1.len(), 2);

        // Manually call generate again (simulating re-entry)
        let run_now = store.get_protocol_run(run.id).await.unwrap().unwrap();
        let states2 = generate(&gen_state, &run_now, &store).await.unwrap();
        assert_eq!(states2.len(), 2);

        // Verify no duplicates
        let all = store.get_runtime_states(run.id).await.unwrap();
        assert_eq!(
            all.len(),
            2,
            "Idempotent: should still have exactly 2 RuntimeStates"
        );
    }
}
