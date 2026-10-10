//! A cancel asked on one instance for a session another instance holds
//! (`cancel_tools`, `cancel_task`): NATS request/reply, and what each outcome
//! means. The asker never answers a success it did not get from the owner:
//!
//! | what happened | asker returns | HTTP | retryable |
//! |---|---|---|---|
//! | the owner did it | its `Ok(result)` | 200 | |
//! | the owner's provider refused (`Unsupported`, …) | `Err(ProviderError)` | 422 `unsupported` … | no |
//! | no instance holds the session live (NATS "no responders"; one instance without NATS: the same) | [`CancelRelayError::OwnerUnreachable`] | 409 `owner_unreachable` | no |
//! | no answer in time (asker's timeout, or the owner's [`OWNER_CANCEL_BOUND`]) | [`CancelRelayError::OwnerTimeout`] | 504 `owner_timeout` | cancel_task only: a late `cancel_tools` may have happened, a retry would stop tools started since |
//! | only `gone` came back: the session left the instance that was asked | [`CancelRelayError::SessionGone`] | 410 `session_gone` | yes (asking again reaches the new owner) |
//! | an answer that is not a [`CancelReply`] | [`CancelRelayError::OwnerProtocol`] | 502 `owner_protocol` | no |
//! | the owner failed otherwise | [`CancelRelayError::OwnerFailed`] | 502 `owner_failed` | no |
//! | the request could not be sent | [`CancelRelayError::RelayFailed`] | 502 `relay_failed` | no |
//!
//! A `gone` is never taken over a real answer: the asker keeps listening
//! [`GONE_GRACE`] after one (an instance that just lost the session may answer
//! before the one that now holds it), and only then reports `session_gone`.
//!
//! Wire (versioned, for rolling upgrades): the request carries `"v": 2`; a v2
//! owner answers `{"v":2, "result"|"refused"|"failed"|"timeout"|"gone": …}`. A
//! request without `v` comes from an older instance: it gets the formats it reads
//! (`result`, `refused`, `failed`), and nothing for `timeout` / `gone` (its own
//! timeout then gives its old no-op). The asker reads every format: v2 objects, the
//! bare `"timeout"` / `"gone"` strings of the first v2 draft, and old objects.

use std::future::Future;
use std::time::Duration;

use anyhow::Result;
use nexus_claude::agent::ProviderError;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

use super::provider::errors::OpenFailure;
use crate::events::{NatsEmitter, RelayFailure};

/// How long one cancel may take, on the owner and locally, before it is reported
/// as `owner_timeout` (shorter than the asker's 10 s, so a slow provider is
/// reported as such instead of a silence).
pub const OWNER_CANCEL_BOUND: Duration = Duration::from_secs(8);

/// How long the asker keeps listening for a real answer after a `gone`.
pub const GONE_GRACE: Duration = Duration::from_millis(1500);

/// Version of the request/reply payload.
pub const WIRE_VERSION: u64 = 2;

/// Which cancel: their retries are not equally safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelKind {
    /// Stop the running tools: not idempotent (a retry stops tools started since).
    Tools,
    /// Stop one background task: idempotent (a stopped task stays stopped).
    Task,
}

/// The owner's answer.
#[derive(Debug, Clone)]
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
                        CancelRelayError::OwnerTimeout { .. } => Self::Timeout,
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
    pub fn into_outcome(self, kind: CancelKind) -> Result<T> {
        match self {
            Self::Result(result) => Ok(result),
            Self::Refused(refusal) => Err(anyhow::Error::new(refusal)),
            Self::Failed(text) => Err(anyhow::Error::new(CancelRelayError::OwnerFailed(text))),
            Self::Timeout => Err(anyhow::Error::new(CancelRelayError::OwnerTimeout { kind })),
            Self::Gone => Err(anyhow::Error::new(CancelRelayError::SessionGone)),
        }
    }

    pub fn is_gone(&self) -> bool {
        matches!(self, Self::Gone)
    }
}

impl<T: Serialize> CancelReply<T> {
    /// The payload for an asker of version 2 (`asker_v2`) or older; `None`: nothing
    /// an older asker can read (it times out into its own old no-op).
    pub fn to_wire(&self, asker_v2: bool) -> Option<Value> {
        let mut body = match self {
            Self::Result(result) => json!({ "result": result }),
            Self::Refused(refusal) => json!({ "refused": refusal }),
            Self::Failed(text) => json!({ "failed": text }),
            Self::Timeout if asker_v2 => json!({ "timeout": true }),
            Self::Gone if asker_v2 => json!({ "gone": true }),
            Self::Timeout | Self::Gone => return None,
        };
        if asker_v2 {
            body["v"] = json!(WIRE_VERSION);
        }
        Some(body)
    }
}

impl<T: DeserializeOwned> CancelReply<T> {
    /// Reads any format an owner may send (see the module doc).
    pub fn from_wire(bytes: &[u8]) -> std::result::Result<Self, String> {
        let value: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if let Some(word) = value.as_str() {
            return match word {
                "timeout" => Ok(Self::Timeout),
                "gone" => Ok(Self::Gone),
                other => Err(format!("unknown answer `{other}`")),
            };
        }
        let field = |name: &str| value.get(name).cloned();
        if let Some(result) = field("result") {
            return serde_json::from_value(result)
                .map(Self::Result)
                .map_err(|e| e.to_string());
        }
        if let Some(refused) = field("refused") {
            return serde_json::from_value(refused)
                .map(Self::Refused)
                .map_err(|e| e.to_string());
        }
        if let Some(failed) = field("failed") {
            return Ok(Self::Failed(
                failed.as_str().unwrap_or("failed").to_string(),
            ));
        }
        if field("timeout").is_some() {
            return Ok(Self::Timeout);
        }
        if field("gone").is_some() {
            return Ok(Self::Gone);
        }
        Err("not a cancel answer".to_string())
    }
}

/// Why a cancel has no answer to give.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CancelRelayError {
    #[error("no instance holds this session live: nothing was cancelled")]
    OwnerUnreachable,
    #[error(
        "the instance holding this session did not answer in time: the cancel may still happen"
    )]
    OwnerTimeout { kind: CancelKind },
    #[error("the session is no longer held by the instance that was asked: nothing was cancelled")]
    SessionGone,
    #[error("the instance holding this session sent an unreadable answer: {0}")]
    OwnerProtocol(String),
    #[error("the instance holding this session failed to cancel: {0}")]
    OwnerFailed(String),
    #[error("the cancel could not be sent to the instance holding this session: {0}")]
    RelayFailed(String),
}

impl CancelRelayError {
    /// The typed HTTP answer (body `{error, code, retryable}`).
    pub fn failure(&self) -> OpenFailure {
        let (status, code, retryable) = match self {
            Self::OwnerUnreachable => (409, "owner_unreachable", false),
            // A late cancel_tools may have happened: a retry would stop the tools
            // started since. A task stopped twice is still one stopped task.
            Self::OwnerTimeout { kind } => (504, "owner_timeout", *kind == CancelKind::Task),
            // The session moved: asking again reaches the instance that holds it now.
            Self::SessionGone => (410, "session_gone", true),
            Self::OwnerProtocol(_) => (502, "owner_protocol", false),
            Self::OwnerFailed(_) => (502, "owner_failed", false),
            Self::RelayFailed(_) => (502, "relay_failed", false),
        };
        // The text names no internal detail (the owner's error stays in its log).
        let message = match self {
            Self::OwnerUnreachable => {
                "No instance holds this session live: nothing was cancelled (nothing runs there)."
            }
            Self::OwnerTimeout { .. } => {
                "The instance holding this session did not answer in time: the cancel may still happen."
            }
            Self::SessionGone => {
                "The session moved to another instance while it was asked: nothing was cancelled; ask again."
            }
            Self::OwnerProtocol(_) => "The instance holding this session sent an unreadable answer.",
            Self::OwnerFailed(_) => "The instance holding this session failed to cancel.",
            Self::RelayFailed(_) => "The cancel could not be sent to the instance holding this session.",
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
    mut payload: Value,
    kind: CancelKind,
) -> Result<T> {
    payload["v"] = json!(WIRE_VERSION);
    let is_gone = |bytes: &[u8]| CancelReply::<Value>::from_wire(bytes).is_ok_and(|r| r.is_gone());
    match nats
        .request_cancel(subject, payload, &is_gone, GONE_GRACE)
        .await
    {
        Ok(bytes) => CancelReply::<T>::from_wire(&bytes)
            .map_err(|e| anyhow::Error::new(CancelRelayError::OwnerProtocol(e)))?
            .into_outcome(kind),
        Err(RelayFailure::NoResponders) => {
            Err(anyhow::Error::new(CancelRelayError::OwnerUnreachable))
        }
        Err(RelayFailure::TimedOut) => {
            Err(anyhow::Error::new(CancelRelayError::OwnerTimeout { kind }))
        }
        Err(RelayFailure::Transport(text)) => {
            Err(anyhow::Error::new(CancelRelayError::RelayFailed(text)))
        }
    }
}

/// Whether a request comes from a v2 asker (`"v": 2`); older ones send no `v`.
pub fn asker_v2(payload: &[u8]) -> bool {
    serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|v| v.get("v").and_then(Value::as_u64))
        .is_some_and(|v| v >= WIRE_VERSION)
}

/// `work` bounded by [`OWNER_CANCEL_BOUND`]: past it, `owner_timeout` of `kind`.
///
/// The future is dropped at the bound, which does not undo what it may already
/// have asked of the provider (a cancel signal sent, a cap recorded): that is why
/// a timed-out `cancel_tools` is reported "may still happen" and not retryable.
pub async fn bounded<T>(kind: CancelKind, work: impl Future<Output = Result<T>>) -> Result<T> {
    match tokio::time::timeout(OWNER_CANCEL_BOUND, work).await {
        Ok(outcome) => outcome,
        Err(_) => Err(anyhow::Error::new(CancelRelayError::OwnerTimeout { kind })),
    }
}

/// The owner's side of one request: runs `work` (bounded, see [`bounded`]), then
/// publishes the answer on `reply_to` in the asker's format. Spawned per request by
/// the listeners, so one slow cancel never holds the next one.
pub async fn answer<T, F>(
    nats: &NatsEmitter,
    reply_to: async_nats::Subject,
    asker_v2: bool,
    kind: CancelKind,
    work: F,
) where
    T: Serialize,
    F: Future<Output = Result<T>>,
{
    let outcome = bounded(kind, work).await;
    if let Err(e) = &outcome {
        if e.downcast_ref::<CancelRelayError>().is_some() {
            tracing::warn!(subject = %reply_to, error = %e, "cancel relayed from another instance: no outcome within the bound");
        }
    }
    publish(nats, reply_to, asker_v2, &CancelReply::of(outcome)).await;
}

/// Publishes one answer on its reply subject (nothing when the asker cannot read it).
pub async fn publish<T: Serialize>(
    nats: &NatsEmitter,
    reply_to: async_nats::Subject,
    asker_v2: bool,
    reply: &CancelReply<T>,
) {
    let Some(body) = reply.to_wire(asker_v2) else {
        return;
    };
    if let Ok(payload) = serde_json::to_vec(&body) {
        let _ = nats.client().publish(reply_to, payload.into()).await;
        let _ = nats.client().flush().await;
    }
}

/// The `error` frame a WebSocket client gets for a `cancel_tools` that failed,
/// unless it was already told: a provider refusal (`Unsupported`) is announced to
/// every client of the session as `error { code: cancel_refused }` by the instance
/// holding it. Otherwise `error { code: "cancel_failed", reason: <code> }`, the
/// code being the one the HTTP route answers (`owner_unreachable`,
/// `owner_timeout`, `session_gone`, …, or the provider error kind).
pub fn ws_error_event(error: &anyhow::Error) -> Option<crate::chat::types::ChatEvent> {
    let (code, message) = if let Some(relay) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<CancelRelayError>())
    {
        let failure = relay.failure();
        (failure.code.to_string(), failure.message)
    } else if let Some(provider) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<ProviderError>())
    {
        if matches!(provider, ProviderError::Unsupported { .. }) {
            return None;
        }
        let failure = super::provider::errors::open_failure(provider, None);
        (failure.code.to_string(), failure.message)
    } else {
        (
            "internal".to_string(),
            "The tools could not be stopped.".to_string(),
        )
    };
    Some(crate::chat::types::ChatEvent::Error {
        message: format!("Error: {message}"),
        parent_tool_use_id: None,
        code: Some("cancel_failed".to_string()),
        reason: Some(code),
        index: None,
    })
}

/// The `task_id` of a cancel_task request (`{"task_id": …}`).
pub fn task_id_of(payload: &[u8]) -> String {
    serde_json::from_slice::<Value>(payload)
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

    fn done() -> CancelTaskResult {
        CancelTaskResult {
            task_id: "b1".into(),
            killed_pids: vec![7],
            capped: false,
        }
    }

    fn round_trip(reply: CancelReply<CancelTaskResult>) -> CancelReply<CancelTaskResult> {
        let wire = reply.to_wire(true).expect("a v2 asker reads every answer");
        assert_eq!(wire["v"], WIRE_VERSION);
        CancelReply::from_wire(&serde_json::to_vec(&wire).unwrap()).unwrap()
    }

    #[test]
    fn every_answer_comes_back_as_the_owner_had_it() {
        let back = round_trip(CancelReply::of(Ok(done())))
            .into_outcome(CancelKind::Task)
            .unwrap();
        assert_eq!(back.killed_pids, vec![7]);

        let refused: Result<CancelTaskResult> =
            Err(anyhow::Error::new(ProviderError::unsupported("tool_cancel")).context("cancel"));
        let error = round_trip(CancelReply::of(refused))
            .into_outcome(CancelKind::Task)
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<ProviderError>(),
            Some(&ProviderError::unsupported("tool_cancel"))
        );

        for (reply, expected) in [
            (
                CancelReply::<CancelTaskResult>::Timeout,
                CancelRelayError::OwnerTimeout {
                    kind: CancelKind::Tools,
                },
            ),
            (CancelReply::Gone, CancelRelayError::SessionGone),
            (
                CancelReply::Failed("disk".into()),
                CancelRelayError::OwnerFailed("disk".into()),
            ),
        ] {
            let error = round_trip(reply)
                .into_outcome(CancelKind::Tools)
                .unwrap_err();
            assert_eq!(error.downcast_ref::<CancelRelayError>(), Some(&expected));
        }
    }

    /// N4 (rolling upgrade): the asker reads the bare strings of the first v2 draft
    /// and the objects of the first release; an older asker (no `v`) gets only the
    /// formats it reads, and nothing it would take for a failure.
    #[test]
    fn every_format_of_every_version_is_read_and_old_askers_get_theirs() {
        for (bytes, gone) in [
            (&b"\"gone\""[..], true),
            (&b"\"timeout\""[..], false),
            (&b"{\"gone\":true}"[..], true),
        ] {
            let reply = CancelReply::<CancelTaskResult>::from_wire(bytes).unwrap();
            assert_eq!(reply.is_gone(), gone, "{}", String::from_utf8_lossy(bytes));
        }
        let old = br#"{"result":{"task_id":"b1","killed_pids":[3],"capped":false}}"#;
        let reply = CancelReply::<CancelTaskResult>::from_wire(old).unwrap();
        assert_eq!(
            reply.into_outcome(CancelKind::Task).unwrap().killed_pids,
            vec![3]
        );
        assert!(CancelReply::<CancelTaskResult>::from_wire(b"{\"ok\":true}").is_err());

        // An older asker: result / refused / failed in its format, no `v`;
        // nothing for timeout and gone (its own timeout gives its old no-op).
        let wire = CancelReply::Result(done()).to_wire(false).unwrap();
        assert!(wire.get("v").is_none() && wire.get("result").is_some());
        assert!(CancelReply::<CancelTaskResult>::Timeout
            .to_wire(false)
            .is_none());
        assert!(CancelReply::<CancelTaskResult>::Gone
            .to_wire(false)
            .is_none());
        assert!(asker_v2(br#"{"v":2,"task_id":"x"}"#));
        assert!(!asker_v2(br#"{"task_id":"x"}"#));
    }

    /// N2: retry only where it is safe.
    #[test]
    fn each_relay_failure_has_its_own_code_and_retry_rule() {
        let rows: Vec<(u16, &str, bool)> = [
            CancelRelayError::OwnerUnreachable,
            CancelRelayError::OwnerTimeout {
                kind: CancelKind::Tools,
            },
            CancelRelayError::OwnerTimeout {
                kind: CancelKind::Task,
            },
            CancelRelayError::SessionGone,
            CancelRelayError::OwnerProtocol("x".into()),
            CancelRelayError::OwnerFailed("x".into()),
            CancelRelayError::RelayFailed("x".into()),
        ]
        .iter()
        .map(|e| {
            let f = e.failure();
            (f.status, f.code, f.retryable)
        })
        .collect();
        assert_eq!(
            rows,
            [
                (409, "owner_unreachable", false),
                (504, "owner_timeout", false),
                (504, "owner_timeout", true),
                (410, "session_gone", true),
                (502, "owner_protocol", false),
                (502, "owner_failed", false),
                (502, "relay_failed", false),
            ]
        );
        // No internal detail in the text.
        assert!(!CancelRelayError::OwnerFailed("secret path".into())
            .failure()
            .message
            .contains("secret"));
    }
}
