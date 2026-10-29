//! Transport-agnostic handles for the event bus request/reply.
//!
//! The chat domain names these types without naming the transport library
//! behind them: [`Subject`] is a reply inbox, [`Message`] an inbound request,
//! [`Subscriber`] a stream of such requests. The concrete transport owns the
//! conversion in and out (`nats.rs`).
use crate::neo4j::traits::*;

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// An opaque subject on the transport (the reply inbox of a request).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subject(String);

impl Subject {
    /// The subject as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Subject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Subject {
    fn from(value: String) -> Self {
        Subject(value)
    }
}

impl From<&str> for Subject {
    fn from(value: &str) -> Self {
        Subject(value.to_string())
    }
}

/// An inbound message received on the transport.
#[derive(Debug, Clone)]
pub struct Message {
    /// The request payload.
    pub payload: Vec<u8>,
    /// The reply inbox, when the asker expects an answer.
    pub reply: Option<Subject>,
}

/// An opaque subscription: a stream of transport [`Message`]s.
pub struct Subscriber {
    inner: Pin<Box<dyn Stream<Item = Message> + Send>>,
}

impl Subscriber {
    pub(crate) fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Message> + Send + 'static,
    {
        Subscriber {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for Subscriber {
    type Item = Message;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
