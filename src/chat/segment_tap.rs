//! The live half of `chat::segment_state`: one passive tap per live session records the
//! tail of its typed events, so the snapshot can classify what each session is doing.
//!
//! A tap is a receiver on the session's own broadcast (`events_tx`, the channel every
//! client of the session already reads, on both engines): it sees exactly what a client
//! sees, adds no second channel and changes nothing the session does. It is installed
//! lazily by `ChatManager::live_session_snapshot` the first time a live session is seen,
//! and replaced when the session's channel changes (a resume opens a new one); a session
//! that leaves the live set loses its tap. Before a tap exists nothing is known, which
//! `segment_state` reads as "no evidence" — never as `Done`.
//!
//! The registry lives in the `ChatManager` (not in a global), so two managers in one
//! process — tests — never prune each other's taps.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;
use uuid::Uuid;

use super::segment_state::{Activity, Addressee, Tail, Verdict};
use super::types::ChatEvent;

/// Milliseconds since the epoch: the one clock read of the feature, made here and passed
/// into the pure model.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct Tap {
    tail: Arc<Mutex<Tail>>,
    addressee: Arc<Mutex<Addressee>>,
    channel: broadcast::WeakSender<ChatEvent>,
    task: tokio::task::AbortHandle,
}

impl Drop for Tap {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A tap just installed: what the caller fills in from the session's record.
pub struct NewTap {
    tail: Arc<Mutex<Tail>>,
    addressee: Arc<Mutex<Addressee>>,
}

impl NewTap {
    /// The stored events that came before the tap (oldest first, with their times).
    pub fn seed(&self, older: &[(ChatEvent, u64)]) {
        self.tail
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .seed(older.iter().map(|(e, at)| (e, *at)));
    }

    /// Who the session's output is for.
    pub fn set_addressee(&self, addressee: Addressee) {
        *self.addressee.lock().unwrap_or_else(|e| e.into_inner()) = addressee;
    }
}

/// The taps of one manager, by session.
#[derive(Default)]
pub struct SegmentTaps {
    taps: Mutex<HashMap<Uuid, Tap>>,
}

impl SegmentTaps {
    /// Makes sure `id` is tapped on `channel`. Returns the new tap when one was
    /// (re)installed, so the caller can fill in what came before it (the stored thread)
    /// and who the session answers to; `None` when the tap was already there.
    pub fn ensure(&self, id: Uuid, channel: &broadcast::Sender<ChatEvent>) -> Option<NewTap> {
        let mut taps = self.taps.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tap) = taps.get(&id) {
            if tap
                .channel
                .upgrade()
                .is_some_and(|live| live.same_channel(channel))
            {
                return None;
            }
        }
        let tail = Arc::new(Mutex::new(Tail::starting_at(now_ms())));
        let mut rx = channel.subscribe();
        let writer = Arc::clone(&tail);
        let task = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => writer
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .record(&event, now_ms()),
                    // Events were dropped: the tail may miss an end, which the model reads
                    // as "no proven end" — conservative, never `Done` by mistake.
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        })
        .abort_handle();
        let addressee = Arc::new(Mutex::new(Addressee::Human));
        taps.insert(
            id,
            Tap {
                tail: Arc::clone(&tail),
                addressee: Arc::clone(&addressee),
                channel: channel.downgrade(),
                task,
            },
        );
        Some(NewTap { tail, addressee })
    }

    /// Forgets the taps of the sessions that are no longer live.
    pub fn retain(&self, live: &HashSet<Uuid>) {
        self.taps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|id, _| live.contains(id));
    }

    /// The verdict for `id` at `now_ms`, `None` when it is not tapped. `activity` is what
    /// the manager knows; the addressee comes from the tap.
    pub fn classify(&self, id: Uuid, activity: Activity, now_ms: u64) -> Option<Verdict> {
        let taps = self.taps.lock().unwrap_or_else(|e| e.into_inner());
        let tap = taps.get(&id)?;
        let activity = Activity {
            addressee: *tap.addressee.lock().unwrap_or_else(|e| e.into_inner()),
            ..activity
        };
        let tail = tap.tail.lock().unwrap_or_else(|e| e.into_inner());
        Some(tail.classify(&activity, now_ms))
    }
}

/// Who a stored `spawned_by` says the session answers to.
pub fn addressee_of(spawned_by: Option<&str>) -> Addressee {
    match spawned_by.map(str::trim) {
        Some(s) if !s.is_empty() && s != "null" => Addressee::Spawner,
        _ => Addressee::Human,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::segment_state::SessionState;

    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn a_tap_records_what_the_session_broadcasts_and_follows_a_new_channel() {
        let taps = SegmentTaps::default();
        let id = Uuid::new_v4();
        let (tx, _keep) = broadcast::channel(64);
        assert!(taps.ensure(id, &tx).is_some());
        assert!(
            taps.ensure(id, &tx).is_none(),
            "the same channel is tapped once"
        );
        tx.send(ChatEvent::UserMessage {
            content: "go".into(),
        })
        .unwrap();
        tx.send(ChatEvent::StreamDelta {
            text: "x".into(),
            parent_tool_use_id: None,
        })
        .unwrap();
        settle().await;
        let streaming = Activity {
            streaming: true,
            ..Default::default()
        };
        let got = taps.classify(id, streaming, now_ms()).unwrap();
        assert_eq!(got.state, SessionState::Writing, "{got:?}");

        // A resume opens a new channel: the tap follows it, from an empty tail.
        let (tx2, _keep2) = broadcast::channel(64);
        assert!(taps.ensure(id, &tx2).is_some());
        let got = taps.classify(id, streaming, now_ms()).unwrap();
        assert_eq!(got.state, SessionState::Working, "{got:?}");

        taps.retain(&HashSet::new());
        assert!(taps.classify(id, streaming, now_ms()).is_none());
    }

    #[test]
    fn a_spawned_session_answers_its_spawner() {
        assert_eq!(addressee_of(None), Addressee::Human);
        assert_eq!(addressee_of(Some("")), Addressee::Human);
        assert_eq!(
            addressee_of(Some(r#"{"type":"conversation","parent_session_id":"x"}"#)),
            Addressee::Spawner
        );
    }
}
