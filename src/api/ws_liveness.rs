//! Détection d'un pair WebSocket mort : le `Ping` part, le `Pong` doit revenir.
//!
//! Avant ce module, les quatre sockets (`/ws/chat`, `/ws/events`, `/ws/activity`,
//! `/ws/run`) envoyaient un `Ping` toutes les 30 s et IGNORAIENT la réponse : un
//! pair dont le TCP reste ouvert mais qui ne répond plus (équipement éteint sans
//! FIN, onglet gelé, NAT qui a perdu l'état) n'était détecté que quand
//! l'écriture échouait, c'est-à-dire quand le tampon d'envoi du noyau était
//! plein — des minutes, voire jamais. Pendant ce temps le descripteur reste
//! ouvert. HYPOTHÈSE (non démontrée) : c'est l'une des façons d'accumuler des
//! descripteurs ; ce module rend le comportement borné et testable, il ne
//! prétend pas être la cause des saturations observées.
//!
//! Règles (les tests de `ws_liveness_tests.rs` les fixent) :
//! - chaque `Ping` porte un numéro de séquence (8 octets) que le pair doit
//!   renvoyer (RFC 6455 §5.5.3) ;
//! - un `Pong` ne compte que s'il renvoie un numéro émis et encore en attente :
//!   il acquitte ce `Ping` et tous les plus anciens. Un `Pong` sans charge, mal
//!   formé, dupliqué ou d'un numéro jamais émis est IGNORÉ : il ne repousse pas
//!   l'échéance ;
//! - le trafic applicatif (texte, binaire) ne compte pas comme preuve de vie
//!   ici : seul un `Pong` valide prolonge ;
//! - l'échéance est celle du plus ancien `Ping` sans réponse. Elle est
//!   contrôlée au tick suivant (donc la fermeture a lieu au plus une période
//!   après l'échéance).

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use futures::stream::SplitSink;
use futures::{Sink, SinkExt};
use tokio::time::{Instant, Sleep};

/// Période des `Ping`. Un test ne peut pas attendre 30 s : il a un rythme court.
#[cfg(not(test))]
pub const WS_PING_EVERY: Duration = Duration::from_secs(30);
#[cfg(test)]
pub const WS_PING_EVERY: Duration = Duration::from_millis(250);

/// Rythme d'une connexion : la période des `Ping` et l'échéance du `Pong`.
///
/// En production : 30 s et 60 s. `/ws/chat` garde en test le rythme court de
/// [`WS_PING_EVERY`] ; les trois autres sockets restent à 30 s / 60 s en test
/// (leurs tests existants laissent le client inactif plus d'une seconde), sauf
/// pour un test de ce module qui le demande par les en-têtes
/// `x-test-ws-ping-ms` / `x-test-ws-pong-ms` (lus seulement sous `cfg(test)`).
#[derive(Clone, Copy, Debug)]
pub struct Beat {
    pub ping_every: Duration,
    pub pong_timeout: Duration,
}

impl Beat {
    pub const PRODUCTION: Beat = Beat {
        ping_every: Duration::from_secs(30),
        pong_timeout: Duration::from_secs(60),
    };

    #[cfg(not(test))]
    pub fn from_headers(_: &axum::http::HeaderMap) -> Beat {
        Beat::PRODUCTION
    }

    #[cfg(test)]
    pub fn from_headers(h: &axum::http::HeaderMap) -> Beat {
        Beat::overridden(Beat::PRODUCTION, h)
    }

    /// `/ws/chat`: the short test rhythm of [`WS_PING_EVERY`] / [`WS_PONG_TIMEOUT`]
    /// under `cfg(test)` (its existing keepalive tests rely on it), overridable
    /// by the same headers.
    #[cfg(not(test))]
    pub fn for_chat(_: &axum::http::HeaderMap) -> Beat {
        Beat::PRODUCTION
    }

    #[cfg(test)]
    pub fn for_chat(h: &axum::http::HeaderMap) -> Beat {
        Beat::overridden(
            Beat {
                ping_every: WS_PING_EVERY,
                pong_timeout: WS_PONG_TIMEOUT,
            },
            h,
        )
    }

    #[cfg(test)]
    fn overridden(base: Beat, h: &axum::http::HeaderMap) -> Beat {
        let ms = |k: &str| {
            h.get(k)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .map(Duration::from_millis)
        };
        Beat {
            ping_every: ms("x-test-ws-ping-ms").unwrap_or(base.ping_every),
            pong_timeout: ms("x-test-ws-pong-ms").unwrap_or(base.pong_timeout),
        }
    }
}

/// Délai sans `Pong` valide après lequel le pair est déclaré mort : deux
/// périodes, de sorte qu'un seul `Pong` perdu ou retardé ne coupe pas.
#[cfg(not(test))]
pub const WS_PONG_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(test)]
pub const WS_PONG_TIMEOUT: Duration = Duration::from_millis(600);

/// Borne d'une écriture de contrôle : un pair qui ne lit plus remplit le
/// tampon d'envoi et une écriture sans borne ne rendrait jamais la main.
pub const WS_SEND_TIMEOUT: Duration = WS_PONG_TIMEOUT;

/// Débit minimal toléré d'une écriture qui avance (octets par seconde).
pub const WS_MIN_WRITE_RATE: usize = 8 * 1024;

fn message_len(m: &Message) -> usize {
    match m {
        Message::Text(t) => t.len(),
        Message::Binary(b) | Message::Ping(b) | Message::Pong(b) => b.len(),
        Message::Close(_) => 0,
    }
}

/// Le pair n'a pas répondu à un `Ping` dans le délai.
#[derive(Debug, PartialEq, Eq)]
pub struct PongOverdue {
    /// Depuis combien de temps le plus ancien `Ping` attend.
    pub waited: Duration,
    /// Nombre de `Ping` sans réponse.
    pub unanswered: usize,
}

/// L'état de vivacité d'UNE connexion.
#[derive(Debug)]
pub struct Liveness {
    timeout: Duration,
    next_seq: u64,
    unanswered: VecDeque<(u64, Instant)>,
}

impl Liveness {
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            next_seq: 0,
            unanswered: VecDeque::new(),
        }
    }

    /// Préparer le prochain `Ping`. `Err` si le plus ancien en attend un
    /// depuis `timeout` ou plus : l'appelant ferme la connexion.
    pub fn next_ping(&mut self, now: Instant) -> Result<Vec<u8>, PongOverdue> {
        if let Some(&(_, sent)) = self.unanswered.front() {
            let waited = now.saturating_duration_since(sent);
            if waited >= self.timeout {
                return Err(PongOverdue {
                    waited,
                    unanswered: self.unanswered.len(),
                });
            }
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.unanswered.push_back((seq, now));
        Ok(seq.to_be_bytes().to_vec())
    }

    /// Un `Pong` reçu. Renvoie `true` s'il a acquitté quelque chose.
    pub fn on_pong(&mut self, payload: &[u8]) -> bool {
        let Ok(bytes) = <[u8; 8]>::try_from(payload) else {
            return false;
        };
        let seq = u64::from_be_bytes(bytes);
        if !self.unanswered.iter().any(|&(s, _)| s == seq) {
            return false; // jamais émis, déjà acquitté, ou périmé
        }
        while let Some(&(s, _)) = self.unanswered.front() {
            self.unanswered.pop_front();
            if s == seq {
                break;
            }
        }
        true
    }

    /// Nombre de `Ping` en attente (diagnostic, tests).
    pub fn pending(&self) -> usize {
        self.unanswered.len()
    }
}

/// Écrire une trame avec une borne de temps. `false` si l'écriture a échoué OU
/// n'a pas abouti dans [`WS_SEND_TIMEOUT`] : dans les deux cas le pair est
/// inutilisable et l'appelant sort de sa boucle.
pub async fn send_bounded<S>(sink: &mut S, msg: Message) -> bool
where
    S: Sink<Message> + Unpin,
{
    send_within(sink, msg, WS_SEND_TIMEOUT).await
}

async fn send_within<S>(sink: &mut S, msg: Message, within: Duration) -> bool
where
    S: Sink<Message> + Unpin,
{
    matches!(
        tokio::time::timeout(within, sink.send(msg)).await,
        Ok(Ok(()))
    )
}

/// Le tick de `Ping` commun aux quatre sockets : fait partir le `Ping` suivant.
/// `false` : sortir de la boucle (échéance dépassée, ou écriture impossible).
pub async fn ping_tick<S>(live: &mut Liveness, sink: &mut S, what: &str) -> bool
where
    S: Sink<Message> + Unpin,
{
    match live.next_ping(Instant::now()) {
        Err(overdue) => {
            tracing::info!(
                socket = what,
                waited_ms = overdue.waited.as_millis() as u64,
                unanswered = overdue.unanswered,
                "WebSocket peer sent no valid Pong in time: closing the connection"
            );
            // Fermeture polie, au mieux : le pair est présumé ne plus lire.
            let _ = send_within(sink, Message::Close(None), Duration::from_secs(1)).await;
            false
        }
        Ok(payload) => send_bounded(sink, Message::Ping(payload.into())).await,
    }
}

/// Le côté écriture d'une socket de ce service.
pub type WsSender = BoundedSink<SplitSink<WebSocket, Message>>;

/// Une écriture qui n'aboutit pas dans [`WS_SEND_TIMEOUT`] PLUS le temps qu'un
/// débit minimal de [`WS_MIN_WRITE_RATE`] octets/s mettrait à écrire le message
/// échoue. Le délai est donc proportionnel à la taille : un message de plusieurs
/// Mo vers un client lent MAIS qui avance (au moins ce débit) n'est pas coupé ;
/// un client qui ne lit plus l'est. La minuterie repart à zéro à chaque fin
/// d'étape réussie (message écrit ou vidé). Pourquoi pas « réarmer à chaque
/// octet écrit » : le `SplitSink` d'une `WebSocket` n'expose pas la progression
/// d'une écriture en cours (elle est atomique de son point de vue), seul le
/// transport la connaît ; la proportionnalité à la taille en est l'approximation
/// observable ici.
///
/// Un client qui CESSE DE LIRE remplit le tampon d'émission TCP : un
/// `send(...).await` non borné ne rend alors jamais la main, la boucle de la
/// socket ne revient plus à son `select!`, l'échéance du `Pong` ne s'exécute
/// jamais et la connexion (donc son descripteur) reste retenue. Plutôt que de
/// borner chacune des dizaines d'écritures applicatives des quatre handlers,
/// on borne le puits : toute écriture, `Ping` comme message applicatif, échoue
/// en erreur au bout du délai, et les handlers traitent déjà une erreur
/// d'écriture comme « client parti » (sortie de boucle, socket lâchée).
pub struct BoundedSink<S> {
    inner: S,
    limit: Duration,
    stall: Option<Pin<Box<Sleep>>>,
    /// Taille du dernier message accepté : elle allonge le délai.
    last_len: usize,
    /// Set by the first failed or stalled write: every later operation fails at once.
    broken: bool,
}

impl<S> BoundedSink<S> {
    pub fn new(inner: S) -> Self {
        Self::with_limit(inner, WS_SEND_TIMEOUT)
    }
    pub fn with_limit(inner: S, limit: Duration) -> Self {
        Self {
            inner,
            limit,
            stall: None,
            last_len: 0,
            broken: false,
        }
    }

    /// A write failed or stalled: the connection is to be dropped.
    pub fn is_broken(&self) -> bool {
        self.broken
    }
}

impl<S> BoundedSink<S>
where
    S: Sink<Message, Error = axum::Error> + Unpin,
{
    fn poll_bounded(
        &mut self,
        cx: &mut Context<'_>,
        step: impl FnOnce(Pin<&mut S>, &mut Context<'_>) -> Poll<Result<(), axum::Error>>,
    ) -> Poll<Result<(), axum::Error>> {
        if self.broken {
            return Poll::Ready(Err(axum::Error::new(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "WebSocket sink already failed",
            ))));
        }
        match step(Pin::new(&mut self.inner), cx) {
            Poll::Ready(r) => {
                self.stall = None;
                if r.is_err() {
                    self.broken = true;
                }
                Poll::Ready(r)
            }
            Poll::Pending => {
                let limit =
                    self.limit + Duration::from_secs((self.last_len / WS_MIN_WRITE_RATE) as u64);
                let timer = self
                    .stall
                    .get_or_insert_with(|| Box::pin(tokio::time::sleep(limit)));
                if timer.as_mut().poll(cx).is_ready() {
                    self.stall = None;
                    self.broken = true;
                    Poll::Ready(Err(axum::Error::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "WebSocket write stalled: the peer stopped reading",
                    ))))
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

impl<S> Sink<Message> for BoundedSink<S>
where
    S: Sink<Message, Error = axum::Error> + Unpin,
{
    type Error = axum::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.get_mut().poll_bounded(cx, |s, cx| s.poll_ready(cx))
    }
    fn start_send(self: Pin<&mut Self>, item: Message) -> Result<(), Self::Error> {
        let me = self.get_mut();
        if me.broken {
            return Err(axum::Error::new(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "WebSocket sink already failed",
            )));
        }
        me.last_len = message_len(&item);
        let r = Pin::new(&mut me.inner).start_send(item);
        if r.is_err() {
            me.broken = true;
        }
        r
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.get_mut().poll_bounded(cx, |s, cx| s.poll_flush(cx))
    }
    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.get_mut().poll_bounded(cx, |s, cx| s.poll_close(cx))
    }
}
