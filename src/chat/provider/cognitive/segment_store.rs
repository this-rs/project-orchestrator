//! Persistence of the conversation segments and their boundaries.
//!
//! The segmenter (`segment.rs`) is a pure model: it reads one [`Observation`]
//! per turn and, from the fingerprints, says where a direction changed. This
//! slice is where the resulting [`Segment`]s and [`Boundary`]s are kept for a
//! conversation. Both types derive `Serialize`/`Deserialize` with snake_case,
//! so a caller can snapshot them as JSON and read them back without loss; the
//! trait is what the wiring depends on, and the in-memory implementation serves
//! the tests of every caller until the graph-backed one is written.
use crate::neo4j::traits::*;

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use super::segment::{Boundary, Segment};

/// Where a conversation's segments and boundaries live.
#[async_trait]
pub trait SegmentStore: Send + Sync {
    /// Stores (replacing) the boundaries of a conversation.
    async fn put_boundaries(
        &self,
        conversation: &str,
        boundaries: &[Boundary],
    ) -> anyhow::Result<()>;
    /// The stored boundaries of a conversation, empty when there are none.
    async fn boundaries(&self, conversation: &str) -> anyhow::Result<Vec<Boundary>>;
    /// Stores (replacing) the segments of a conversation.
    async fn put_segments(&self, conversation: &str, segments: &[Segment]) -> anyhow::Result<()>;
    /// The stored segments of a conversation, empty when there are none.
    async fn segments(&self, conversation: &str) -> anyhow::Result<Vec<Segment>>;
}

/// In-memory store, for tests and for a run without a graph.
#[derive(Default)]
pub struct InMemorySegmentStore {
    boundaries: Mutex<HashMap<String, Vec<Boundary>>>,
    segments: Mutex<HashMap<String, Vec<Segment>>>,
}

impl InMemorySegmentStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[async_trait]
impl SegmentStore for InMemorySegmentStore {
    async fn put_boundaries(
        &self,
        conversation: &str,
        boundaries: &[Boundary],
    ) -> anyhow::Result<()> {
        lock(&self.boundaries).insert(conversation.to_owned(), boundaries.to_vec());
        Ok(())
    }

    async fn boundaries(&self, conversation: &str) -> anyhow::Result<Vec<Boundary>> {
        Ok(lock(&self.boundaries)
            .get(conversation)
            .cloned()
            .unwrap_or_default())
    }

    async fn put_segments(&self, conversation: &str, segments: &[Segment]) -> anyhow::Result<()> {
        lock(&self.segments).insert(conversation.to_owned(), segments.to_vec());
        Ok(())
    }

    async fn segments(&self, conversation: &str) -> anyhow::Result<Vec<Segment>> {
        Ok(lock(&self.segments)
            .get(conversation)
            .cloned()
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::super::segment::{Dimension, Fingerprint, Observation, SegmentPolicy, Segmenter};
    use super::*;
    use crate::neo4j::traits::*;

    fn obs(turn: u32, prints: &[(Dimension, &[&str])]) -> Observation {
        Observation {
            turn,
            prints: prints
                .iter()
                .map(|(dimension, labels)| (*dimension, Fingerprint::new(labels.iter().copied())))
                .collect(),
        }
    }

    /// Two turns on one axis that switch harness: a single boundary, two segments.
    fn history() -> Vec<Observation> {
        vec![
            obs(0, &[(Dimension::Harness, &["deepseek"])]),
            obs(1, &[(Dimension::Harness, &["claude-code"])]),
        ]
    }

    fn replay(seed: u64) -> Segmenter {
        Segmenter::replay(SegmentPolicy::default(), seed, &history()).unwrap()
    }

    #[test]
    fn a_boundary_and_a_segment_round_trip_through_json_losslessly() {
        let segmenter = replay(7);
        let boundary = segmenter.boundaries()[0].clone();
        let segment = segmenter.segments()[0].clone();

        let boundary_back: Boundary =
            serde_json::from_str(&serde_json::to_string(&boundary).unwrap()).unwrap();
        assert_eq!(boundary_back, boundary);
        // The wire form uses snake_case axes.
        assert_eq!(
            serde_json::to_value(&boundary).unwrap()["dimension"],
            "harness"
        );

        let segment_back: Segment =
            serde_json::from_str(&serde_json::to_string(&segment).unwrap()).unwrap();
        assert_eq!(segment_back, segment);
        assert_eq!(
            serde_json::to_value(&segment).unwrap()["dimension"],
            "harness"
        );
    }

    #[tokio::test]
    async fn a_store_reads_back_what_was_put() {
        let store = InMemorySegmentStore::new();
        let segmenter = replay(7);
        let segments = segmenter.segments();
        let boundaries = segmenter.boundaries().to_vec();

        assert!(store.segments("conv").await.unwrap().is_empty());
        assert!(store.boundaries("conv").await.unwrap().is_empty());

        store.put_segments("conv", &segments).await.unwrap();
        store.put_boundaries("conv", &boundaries).await.unwrap();

        assert_eq!(store.segments("conv").await.unwrap(), segments);
        assert_eq!(store.boundaries("conv").await.unwrap(), boundaries);
        // Another conversation is still empty.
        assert!(store.segments("other").await.unwrap().is_empty());
        assert!(store.boundaries("other").await.unwrap().is_empty());

        // Putting again replaces, never accumulates.
        store.put_segments("conv", &segments).await.unwrap();
        assert_eq!(store.segments("conv").await.unwrap(), segments);
    }

    #[test]
    fn two_replays_with_the_same_seed_serialize_identically() {
        let a = replay(42);
        let b = replay(42);
        assert_eq!(
            serde_json::to_string(&a.segments()).unwrap(),
            serde_json::to_string(&b.segments()).unwrap()
        );
        assert_eq!(
            serde_json::to_string(a.boundaries()).unwrap(),
            serde_json::to_string(b.boundaries()).unwrap()
        );
    }
}
