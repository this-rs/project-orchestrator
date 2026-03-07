//! Neural Feedback Engine — post-response processing for knowledge reinforcement.
//!
//! After each LLM response, this module:
//! 1. **Entity detection**: Extracts code entities (files, functions, structs) from the response
//! 2. **Auto add_discussed**: Marks detected entities as DISCUSSED in Neo4j (async, non-blocking)
//! 3. **Session dedup cache**: Prevents duplicate DISCUSSED relations within a session
//! 4. **ReasoningTree path tracking**: Tracks followed reasoning paths for reinforcement
//! 5. **Observation detection**: Detects implicit insights and suggests note creation
//!
//! **Performance target:** < 50ms post-response, entirely asynchronous (zero latency added).

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::chat::entity_extractor::{extract_entities, validate_entities, ValidatedEntity};
use crate::chat::observation_detector::{detect_observations, DetectedObservation};
use crate::neo4j::traits::GraphStore;

// ============================================================================
// Types
// ============================================================================

/// Cache of entities already marked as DISCUSSED in the current session.
/// Prevents redundant Neo4j writes for entities mentioned multiple times.
#[derive(Debug, Clone)]
pub struct SessionDiscussedCache {
    /// Set of (entity_type, entity_id) already processed
    seen: Arc<Mutex<HashSet<(String, String)>>>,
}

impl SessionDiscussedCache {
    pub fn new() -> Self {
        Self {
            seen: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Check if an entity was already processed, and mark it if not.
    /// Returns true if the entity is NEW (not yet seen).
    async fn check_and_mark(&self, entity_type: &str, entity_id: &str) -> bool {
        let mut seen = self.seen.lock().await;
        seen.insert((entity_type.to_string(), entity_id.to_string()))
    }

    /// Get the number of unique entities tracked in this session.
    pub async fn len(&self) -> usize {
        self.seen.lock().await.len()
    }

    /// Check if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.seen.lock().await.is_empty()
    }
}

impl Default for SessionDiscussedCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ReasoningTree Path Tracker
// ============================================================================

/// A single tracked reasoning path: (tree_id, node_entity_ids).
type TrackedPath = (Uuid, Vec<Uuid>);

/// Tracks reasoning tree paths followed during a session for auto-reinforcement.
///
/// When the agent follows a ReasoningTree path and the action succeeds,
/// the tracker automatically reinforces the underlying neural connections
/// (boost energy + synapse reinforcement).
#[derive(Debug, Clone)]
pub struct ReasoningPathTracker {
    /// Accumulated paths: Vec<(tree_id, Vec<node_entity_ids>)>
    paths: Arc<Mutex<Vec<TrackedPath>>>,
}

impl ReasoningPathTracker {
    pub fn new() -> Self {
        Self {
            paths: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record a reasoning tree path that was followed.
    ///
    /// `node_ids` are the entity IDs (note/decision UUIDs) of the
    /// ReasoningTree nodes traversed during the action.
    pub async fn record_path(&self, tree_id: Uuid, node_ids: Vec<Uuid>) {
        if node_ids.is_empty() {
            return;
        }
        let mut paths = self.paths.lock().await;
        paths.push((tree_id, node_ids));
    }

    /// Get the number of tracked paths in this session.
    pub async fn len(&self) -> usize {
        self.paths.lock().await.len()
    }

    /// Check if there are no tracked paths.
    pub async fn is_empty(&self) -> bool {
        self.paths.lock().await.is_empty()
    }

    /// Consume all tracked paths and return them.
    async fn take_all(&self) -> Vec<TrackedPath> {
        let mut paths = self.paths.lock().await;
        std::mem::take(&mut *paths)
    }
}

impl Default for ReasoningPathTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Reinforce all tracked reasoning paths after successful actions.
///
/// For each tracked path:
/// - Boost energy of each traversed node (energy_boost = 0.15)
/// - Reinforce synapses between adjacent nodes (synapse_boost = 0.05)
///
/// This is the Hebbian learning rule: "neurons that fire together wire together."
pub async fn reinforce_tracked_paths(
    graph: Arc<dyn GraphStore>,
    tracker: &ReasoningPathTracker,
) -> usize {
    let paths = tracker.take_all().await;
    if paths.is_empty() {
        return 0;
    }

    let energy_boost = 0.15;
    let synapse_boost = 0.05;
    let mut total_reinforced = 0usize;

    for (_tree_id, node_ids) in &paths {
        // Boost energy for each node
        for node_id in node_ids {
            if let Err(e) = graph.boost_energy(*node_id, energy_boost).await {
                debug!(
                    "[feedback] Failed to boost energy for node {}: {}",
                    node_id, e
                );
            }
        }

        // Reinforce synapses between all nodes in the path
        if node_ids.len() >= 2 {
            match graph.reinforce_synapses(node_ids, synapse_boost).await {
                Ok(count) => {
                    total_reinforced += count;
                    debug!(
                        "[feedback] Reinforced {} synapses for path ({} nodes)",
                        count,
                        node_ids.len()
                    );
                }
                Err(e) => {
                    debug!("[feedback] Failed to reinforce synapses: {}", e);
                }
            }
        }
    }

    debug!(
        "[feedback] Reinforced {} paths, {} total synapse boosts",
        paths.len(),
        total_reinforced
    );
    total_reinforced
}

// ============================================================================
// Observation Detection (post-response)
// ============================================================================

/// Detect observations in an LLM response and return any suggestion.
///
/// Returns at most ONE `DetectedObservation` per response.
/// Only returns suggestions with confidence > 0.8.
pub fn detect_response_observations(response_text: &str) -> Option<DetectedObservation> {
    detect_observations(response_text)
}

// ============================================================================
// Auto add_discussed
// ============================================================================

/// Process an LLM response: extract entities and mark them as DISCUSSED.
///
/// This function is designed to be called via `tokio::spawn` so it adds
/// zero latency to the user's response. It:
/// 1. Extracts entities from the response text (regex, ~1ms)
/// 2. Validates them against Neo4j (graph lookups, ~30ms)
/// 3. Filters out already-discussed entities (session cache)
/// 4. Calls add_discussed for new entities (batch Neo4j write, ~10ms)
pub async fn process_response_entities(
    graph: Arc<dyn GraphStore>,
    session_id: Uuid,
    project_id: Option<Uuid>,
    response_text: &str,
    cache: &SessionDiscussedCache,
) -> usize {
    // Step 1: Extract entities from response text
    let extracted = extract_entities(response_text);
    if extracted.is_empty() {
        return 0;
    }

    debug!(
        "[feedback] Extracted {} raw entities from response",
        extracted.len()
    );

    // Step 2: Validate against Neo4j graph
    let validated = validate_entities(extracted, graph.as_ref(), project_id).await;
    if validated.is_empty() {
        debug!("[feedback] No validated entities after graph check");
        return 0;
    }

    // Step 3: Filter out already-discussed entities (session dedup)
    let mut new_entities: Vec<(String, String)> = Vec::new();
    for entity in &validated {
        let entity_type = entity.node_label.clone();
        let entity_id = resolve_entity_id(entity);

        if cache.check_and_mark(&entity_type, &entity_id).await {
            new_entities.push((entity_type, entity_id));
        }
    }

    if new_entities.is_empty() {
        debug!("[feedback] All entities already discussed in this session");
        return 0;
    }

    // Step 4: Batch add_discussed
    let entity_refs: Vec<(String, String)> = new_entities.clone();
    let count = entity_refs.len();

    match graph.add_discussed(session_id, &entity_refs).await {
        Ok(added) => {
            debug!(
                "[feedback] Added {} DISCUSSED relations ({} entities submitted)",
                added, count
            );
            added
        }
        Err(e) => {
            warn!(
                "[feedback] Failed to add DISCUSSED relations: {} — non-blocking",
                e
            );
            0
        }
    }
}

/// Resolve the entity identifier for add_discussed.
///
/// For files, use the file_path. For symbols, use the identifier.
fn resolve_entity_id(entity: &ValidatedEntity) -> String {
    match entity.file_path.as_deref() {
        Some(path) if entity.node_label == "File" => path.to_string(),
        _ => entity.identifier.clone(),
    }
}

/// Convenience function to spawn async feedback processing.
///
/// Call this after each LLM response. It spawns a background task
/// that processes entities without blocking the response stream.
pub fn spawn_feedback(
    graph: Arc<dyn GraphStore>,
    session_id: Uuid,
    project_id: Option<Uuid>,
    response_text: String,
    cache: SessionDiscussedCache,
) {
    tokio::spawn(async move {
        let start = tokio::time::Instant::now();
        let count =
            process_response_entities(graph, session_id, project_id, &response_text, &cache).await;
        let elapsed = start.elapsed().as_millis();
        if count > 0 {
            debug!(
                "[feedback] Async feedback completed: {} entities in {}ms",
                count, elapsed
            );
        }
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_cache_dedup() {
        let cache = SessionDiscussedCache::new();

        // First time → new
        assert!(cache.check_and_mark("File", "src/main.rs").await);
        // Second time → already seen
        assert!(!cache.check_and_mark("File", "src/main.rs").await);
        // Different entity → new
        assert!(cache.check_and_mark("Function", "build_prompt").await);

        assert_eq!(cache.len().await, 2);
    }

    #[tokio::test]
    async fn test_process_empty_response() {
        let cache = SessionDiscussedCache::new();
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let session_id = Uuid::new_v4();

        let count = process_response_entities(mock, session_id, None, "", &cache).await;
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_process_no_code_entities() {
        let cache = SessionDiscussedCache::new();
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let session_id = Uuid::new_v4();

        let count = process_response_entities(
            mock,
            session_id,
            None,
            "Bonjour, voici ma réponse sans code.",
            &cache,
        )
        .await;
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_process_with_entities_mock() {
        let cache = SessionDiscussedCache::new();
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let session_id = Uuid::new_v4();

        // Response mentioning entities — they won't validate against mock
        // (mock has no files), so count should be 0
        let count = process_response_entities(
            mock,
            session_id,
            None,
            "J'ai modifié `src/chat/manager.rs` et la fonction `build_prompt`",
            &cache,
        )
        .await;
        // Mock has no graph data, so validation filters everything out
        assert_eq!(count, 0);
    }

    // ========================================================================
    // E2E Scenario Tests (TP4.5)
    // ========================================================================

    /// E2E #3: ReasoningTree path tracked → reinforcement triggers
    #[tokio::test]
    async fn test_e2e_reasoning_path_reinforcement() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let tracker = ReasoningPathTracker::new();

        // Create notes in mock for reinforcement targets
        use crate::notes::models::{Note, NoteImportance, NoteType};
        let note1_id = Uuid::new_v4();
        let note2_id = Uuid::new_v4();
        let note3_id = Uuid::new_v4();
        {
            let mut n1 = Note::new(
                None,
                NoteType::Guideline,
                "Test note 1".into(),
                "test".into(),
            );
            n1.id = note1_id;
            n1.importance = NoteImportance::High;
            n1.energy = 0.5;
            mock.create_note(&n1).await.unwrap();

            let mut n2 = Note::new(None, NoteType::Gotcha, "Test note 2".into(), "test".into());
            n2.id = note2_id;
            n2.importance = NoteImportance::High;
            n2.energy = 0.3;
            mock.create_note(&n2).await.unwrap();

            let mut n3 = Note::new(None, NoteType::Pattern, "Test note 3".into(), "test".into());
            n3.id = note3_id;
            n3.importance = NoteImportance::Medium;
            n3.energy = 0.2;
            mock.create_note(&n3).await.unwrap();
        }

        let tree_id = Uuid::new_v4();
        // Record a path with 3 nodes
        tracker
            .record_path(tree_id, vec![note1_id, note2_id, note3_id])
            .await;
        assert_eq!(tracker.len().await, 1);

        // Reinforce
        let reinforced = reinforce_tracked_paths(mock.clone(), &tracker).await;
        // Mock reinforces synapses between 3 notes → creates up to 3 pairs
        assert!(reinforced > 0);

        // Verify energy was boosted
        let notes = mock.notes.read().await;
        let n1 = notes.get(&note1_id).unwrap();
        assert!(n1.energy > 0.5, "Energy should be boosted from 0.5");
        let n2 = notes.get(&note2_id).unwrap();
        assert!(n2.energy > 0.3, "Energy should be boosted from 0.3");

        // Tracker should be empty after reinforcement (consumed)
        assert_eq!(tracker.len().await, 0);
    }

    /// E2E #4: Response containing observation → suggestion detected
    #[test]
    fn test_e2e_observation_detection_suggestion() {
        let text = "After debugging, the root cause was that the pre-push hook runs both cargo fmt AND clippy. Be careful with this — you need to run both checks locally before pushing.";
        let obs = detect_response_observations(text);
        assert!(obs.is_some(), "Should detect an observation");
        let obs = obs.unwrap();
        assert!(obs.confidence >= 0.80, "Confidence should be >= 0.80");
        assert!(
            obs.note_type == "gotcha",
            "Should suggest a gotcha note, got: {}",
            obs.note_type
        );
    }

    /// E2E #7: Feedback is async — spawn_feedback doesn't block
    #[tokio::test]
    async fn test_e2e_feedback_async_nonblocking() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let session_id = Uuid::new_v4();
        let cache = SessionDiscussedCache::new();

        let start = tokio::time::Instant::now();

        // spawn_feedback should return immediately (non-blocking)
        spawn_feedback(
            mock,
            session_id,
            None,
            "I modified `src/chat/manager.rs` and the function `build_prompt`. The root cause was a Sized bound issue.".to_string(),
            cache,
        );

        let elapsed = start.elapsed();
        // spawn_feedback should complete in < 1ms (it just spawns a task)
        assert!(
            elapsed.as_millis() < 10,
            "spawn_feedback should be non-blocking, took {}ms",
            elapsed.as_millis()
        );
    }

    // ========================================================================
    // Original tests
    // ========================================================================

    #[test]
    fn test_resolve_entity_id_file() {
        let entity = ValidatedEntity {
            entity_type: crate::chat::entity_extractor::EntityType::File,
            identifier: "src/main.rs".to_string(),
            source: crate::chat::entity_extractor::ExtractionSource::FilePath,
            node_label: "File".to_string(),
            file_path: Some("src/main.rs".to_string()),
        };
        assert_eq!(resolve_entity_id(&entity), "src/main.rs");
    }

    #[test]
    fn test_resolve_entity_id_function() {
        let entity = ValidatedEntity {
            entity_type: crate::chat::entity_extractor::EntityType::Function,
            identifier: "build_prompt".to_string(),
            source: crate::chat::entity_extractor::ExtractionSource::Backtick,
            node_label: "Function".to_string(),
            file_path: Some("src/chat/prompt.rs".to_string()),
        };
        assert_eq!(resolve_entity_id(&entity), "build_prompt");
    }
}
