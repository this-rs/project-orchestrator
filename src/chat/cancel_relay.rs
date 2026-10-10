//! A cancel asked on one instance for a session another instance holds
//! (`cancel_tools`, `cancel_task`): NATS request/reply, and what each outcome
//! means. The asker never answers a success it did not get from the owner:
//!
//! | what happened | asker returns | HTTP |
//! |---|---|---|
//! | the owner did it | its `Ok(result)` | 200 |
//! | the owner's provider refused (`Unsupported`, …) | `Err(ProviderError)` | 422 `unsupported` … |
//! | nobody subscribes to the subject (NATS "no responders") | [`CancelRelayError::OwnerUnreachable`] | 409 `owner_unreachable` |
//! | no answer within the asker's timeout | [`CancelRelayError::OwnerTimeout`] | 504 `owner_timeout` |
//! | the owner gave up after [`OWNER_CANCEL_BOUND`] | [`CancelRelayError::OwnerTimeout`] | 504 `owner_timeout` |
//! | the session left the owner while asking | [`CancelRelayError::SessionGone`] | 410 `session_gone` |
//! | an answer that is not a [`CancelReply`] | [`CancelRelayError::OwnerProtocol`] | 502 `owner_protocol` |
//! | the owner failed otherwise | [`CancelRelayError::OwnerFailed`] | 502 `owner_failed` |
//!
//! The owner answers each request in its own task, bounded by
//! [`OWNER_CANCEL_BOUND`] (shorter than the asker's timeout, so a slow provider
//! is reported as such instead of a silence).

use std::future::Future;
use std::time::Duration;

use anyhow::Result;
use nexus_claude::agent::ProviderError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::provider::errors::OpenFailure;
use crate::events::{NatsEmitter, RelayFailure};

/// How long the owner works on one cancel before it answers `timeout`.
pub const OWNER_CANCEL_BOUND: Duration = Duration::from_secs(8);

/// The owner's answer on the reply subject.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReply<T> {
    /// What the owner did.
    Result(T),
    /// The owner's provider refused it.
    Refused(ProviderError),
    /// The owner failed otherwise (its error text).
    Failed(String),
    /// The owner gave up after [`OWNER_CANCEL_BOUND`].
    Timeout,
    /// The session is no longer held by the instance that was asked.
    Gone,
}

impl<T> CancelReply<T> {
    /// The answer for what the owner's call returned.
    pub fn of(outcome: Result<T>) -> Self {
        match outcome {
            Ok(result) => Self::Result(result),
            Err(error) => {
                if let Some(relay) = error.downcast_ref::<CancelRelayError>() {
                    return match relay {
                        CancelRelayError::SessionGone => Self::Gone,
                        CancelRelayError::OwnerTimeout => Self::Timeout,
                        other => Self::Failed(other.to_string()),
                    };
                }
                match error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<ProviderError>())
                {
                    Some(refusal) => Self::Refused(refusal.clone()),
                    None => Self::Failed(error.to_string()),
                }
            }
        }
    }

    /// What the asker returns for it: the same `Ok` / typed `Err` the owner's
    /// caller had.
    pub fn into_outcome(self) -> Result<T> {
        match self {
            Self::Result(result) => Ok(result),
            Self::Refused(refusal) => Err(anyhow::Error::new(refusal)),
            Self::Failed(text) => Err(anyhow::Error::new(CancelRelayError::OwnerFailed(text))),
            Self::Timeout => Err(anyhow::Error::new(CancelRelayError::OwnerTimeout)),
            Self::Gone => Err(anyhow::Error::new(CancelRelayError::SessionGone)),
        }
    }
}

/// Why a cancel relayed to another instance has no answer to give.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CancelRelayError {
    #[error("no instance holds this session live: nothing was cancelled")]
    OwnerUnreachable,
    #[error(
        "the instance holding this session did not answer in time: the cancel may still happen"
    )]
    OwnerTimeout,
    #[error("the session is no longer held by the instance that was asked: nothing was cancelled")]
    SessionGone,
    #[error("the instance holding this session sent an unreadable answer: {0}")]
    OwnerProtocol(String),
    #[error("the instance holding this session failed to cancel: {0}")]
    OwnerFailed(String),
}

impl CancelRelayError {
    /// The typed HTTP answer (body `{error, code, retryable}`).
    pub fn failure(&self) -> OpenFailure {
        let (status, code, retryable) = match self {
            Self::OwnerUnreachable => (409, "owner_unreachable", false),
            Self::OwnerTimeout => (504, "owner_timeout", true),
            Self::SessionGone => (410, "session_gone", false),
            Self::OwnerProtocol(_) => (502, "owner_protocol", false),
            Self::OwnerFailed(_) => (502, "owner_failed", false),
        };
        // The text names no internal detail (the owner's error stays in its log).
        let message = match self {
            Self::OwnerUnreachable => "No instance holds this session live: nothing was cancelled.",
            Self::OwnerTimeout => {
                "The instance holding this session did not answer in time: the cancel may still happen."
            }
            Self::SessionGone => {
                "The session is no longer held by the instance that was asked: nothing was cancelled."
            }
            Self::OwnerProtocol(_) => "The instance holding this session sent an unreadable answer.",
            Self::OwnerFailed(_) => "The instance holding this session failed to cancel.",
        }
        .to_string();
        OpenFailure {
            status,
            code,
            message,
            provider_id: None,
            action: None,
            retryable,
            retry_after_ms: None,
        }
    }
}

/// Asks the owner over NATS and returns its answer as the owner's caller would
/// have had it, or the typed reason there is none.
pub async fn relay<T: DeserializeOwned>(
    nats: &NatsEmitter,
    subject: String,
    payload: serde_json::Value,
) -> Result<T> {
    match nats.request_cancel(subject, payload).await {
        Ok(bytes) => serde_json::from_slice::<CancelReply<T>>(&bytes)
            .map_err(|e| anyhow::Error::new(CancelRelayError::OwnerProtocol(e.to_string())))?
            .into_outcome(),
        Err(RelayFailure::NoResponders) => {
            Err(anyhow::Error::new(CancelRelayError::OwnerUnreachable))
        }
        Err(RelayFailure::TimedOut) => Err(anyhow::Error::new(CancelRelayError::OwnerTimeout)),
        Err(RelayFailure::Transport(text)) => {
            Err(anyhow::Error::new(CancelRelayError::OwnerProtocol(text)))
        }
    }
}

/// The owner's side of one request: runs `work` bounded by `bound`, then
/// publishes the answer on `reply_to`. Spawned per request by the listeners, so
/// one slow cancel never holds the next one.
pub async fn answer<T, F>(
    nats: &NatsEmitter,
    reply_to: async_nats::Subject,
    bound: Duration,
    work: F,
) where
    T: Serialize,
    F: Future<Output = Result<T>>,
{
    let reply = match tokio::time::timeout(bound, work).await {
        Ok(outcome) => CancelReply::of(outcome),
        Err(_) => {
            tracing::warn!(subject = %reply_to, "cancel relayed from another instance: no outcome within the bound");
            CancelReply::Timeout
        }
    };
    publish(nats, reply_to, &reply).await;
}

/// Publishes one answer on its reply subject.
pub async fn publish<T: Serialize>(
    nats: &NatsEmitter,
    reply_to: async_nats::Subject,
    reply: &CancelReply<T>,
) {
    if let Ok(payload) = serde_json::to_vec(reply) {
        let _ = nats.client().publish(reply_to, payload.into()).await;
        let _ = nats.client().flush().await;
    }
}

/// The `task_id` of a cancel_task request (`{"task_id": …}`).
pub fn task_id_of(payload: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| {
            v.get("task_id")
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::manager::CancelTaskResult;

    #[test]
    fn every_answer_comes_back_as_the_owner_had_it() {
        let ok = CancelTaskResult {
            task_id: "b1".into(),
            killed_pids: vec![7],
            capped: false,
        };
        let back: CancelTaskResult = CancelReply::of(Ok(ok)).into_outcome().unwrap();
        assert_eq!(back.killed_pids, vec![7]);

        let refused: Result<CancelTaskResult> =
            Err(anyhow::Error::new(ProviderError::unsupported("tool_cancel")).context("cancel"));
        let error = CancelReply::of(refused).into_outcome().unwrap_err();
        assert_eq!(
            error.downcast_ref::<ProviderError>(),
            Some(&ProviderError::unsupported("tool_cancel"))
        );

        for (reply, expected) in [
            (
                CancelReply::<CancelTaskResult>::Timeout,
                CancelRelayError::OwnerTimeout,
            ),
            (CancelReply::Gone, CancelRelayError::SessionGone),
            (
                CancelReply::Failed("disk".into()),
                CancelRelayError::OwnerFailed("disk".into()),
            ),
        ] {
            let wire: CancelReply<CancelTaskResult> =
                serde_json::from_slice(&serde_json::to_vec(&reply).unwrap()).unwrap();
            let error = wire.into_outcome().unwrap_err();
            assert_eq!(error.downcast_ref::<CancelRelayError>(), Some(&expected));
        }
    }

    #[test]
    fn each_relay_failure_has_its_own_code() {
        let codes: Vec<(u16, &str)> = [
            CancelRelayError::OwnerUnreachable,
            CancelRelayError::OwnerTimeout,
            CancelRelayError::SessionGone,
            CancelRelayError::OwnerProtocol("x".into()),
            CancelRelayError::OwnerFailed("x".into()),
        ]
        .iter()
        .map(|e| {
            let f = e.failure();
            (f.status, f.code)
        })
        .collect();
        assert_eq!(
            codes,
            [
                (409, "owner_unreachable"),
                (504, "owner_timeout"),
                (410, "session_gone"),
                (502, "owner_protocol"),
                (502, "owner_failed"),
            ]
        );
        // No internal detail in the text.
        assert!(!CancelRelayError::OwnerFailed("secret path".into())
            .failure()
            .message
            .contains("secret"));
    }
}
