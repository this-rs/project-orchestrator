//! Descripteurs de fichiers : relever la limite au démarrage, la surveiller, et
//! refuser proprement les connexions entrantes avant la saturation.
//!
//! CONTEXTE (mesuré le 21/11/2026 sur le service vivant sous launchd) : 957
//! `Too many open files` (os error 24) dans le journal, première occurrence
//! après ~18 h de service, `accept` en erreur et `ServeDir` incapable d'ouvrir
//! ses fichiers ; table de descripteurs plafonnée à 256 (255 le plus grand
//! numéro vu), `launchctl limit maxfiles` = 256 souple / illimité dur, aucune
//! clé de limite dans le plist, aucun `setrlimit` dans le code. La limite
//! effective du processus n'est pas lisible sans privilèges : le plafond de 256
//! est une INFÉRENCE (cohérente avec le plus grand numéro observé), pas une
//! preuve. La nature des connexions entrantes (une seule adresse LAN en
//! portait 133) n'est PAS établie : [`ConnTracker`] sert à l'établir.
//!
//! Ce module ne dépend que de `libc`, `tokio`, `axum` et `tracing`, pour être
//! testable isolément (`tests/fd_limit.rs` l'inclut par `#[path]`).

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::{TcpListener, TcpStream};

/// Cible par défaut de la limite souple. Largement au-dessus de ce que le
/// service consomme (~240 mesurés, dont ~50 vers Neo4j) avec de la marge pour
/// les sessions d'agents (pipes) et les clients LAN.
pub const DEFAULT_NOFILE_TARGET: u64 = 65_536;
/// Variable d'environnement : cible de la limite souple.
pub const ENV_NOFILE_TARGET: &str = "ORCHESTRATOR_NOFILE_TARGET";
/// Variable d'environnement : plafond de connexions entrantes simultanées
/// (`0` = pas de plafond). Défaut : voir [`default_max_connections`].
pub const ENV_MAX_CONNECTIONS: &str = "ORCHESTRATOR_MAX_CONNECTIONS";
/// Variable d'environnement : `1`/`true` active le suivi par adresse distante.
pub const ENV_CONN_STATS: &str = "ORCHESTRATOR_CONN_STATS";
/// Seuil d'alerte (fraction de la limite souple).
pub const ALERT_RATIO: f64 = 0.8;

/// Valeur « illimité » d'une limite.
pub const UNLIMITED: u64 = u64::MAX;

// ── Limite RLIMIT_NOFILE ───────────────────────────────────────────────

/// Ce qu'a fait [`raise_nofile_limit`], à journaliser tel quel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NofileReport {
    pub soft_before: u64,
    pub hard: u64,
    pub soft_after: u64,
    /// La première valeur tentée (0 si rien à relever).
    pub requested: u64,
    /// Pourquoi la limite n'a pas pu être relevée autant que voulu (jamais muet).
    pub problem: Option<String>,
}

impl NofileReport {
    fn fmt_limit(v: u64) -> String {
        if v == UNLIMITED {
            "unlimited".to_string()
        } else {
            v.to_string()
        }
    }

    /// Une ligne lisible : `nofile soft 256 -> 10240 (hard unlimited)`.
    pub fn summary(&self) -> String {
        format!(
            "nofile soft {} -> {} (hard {})",
            Self::fmt_limit(self.soft_before),
            Self::fmt_limit(self.soft_after),
            Self::fmt_limit(self.hard)
        )
    }
}

/// Valeurs successives à essayer pour passer de `soft` à (au plus) `target`
/// sans dépasser `hard`. macOS refuse (EINVAL) une valeur au-dessus de
/// `kern.maxfilesperproc` même si la limite dure est illimitée : on redescend
/// donc par paliers connus plutôt que d'échouer d'un bloc.
pub fn candidate_limits(soft: u64, hard: u64, target: u64) -> Vec<u64> {
    let want = target.min(hard);
    if want <= soft {
        return Vec::new();
    }
    let mut out = vec![want];
    for step in [245_760u64, 61_440, 10_240, 4_096, 1_024] {
        if step < want && step > soft {
            out.push(step);
        }
    }
    out
}

/// (souple, dure) courantes.
#[cfg(unix)]
pub fn nofile_limits() -> io::Result<(u64, u64)> {
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `rl` is a valid, writable rlimit for the duration of the call.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let norm = |v: libc::rlim_t| {
        if v == libc::RLIM_INFINITY {
            UNLIMITED
        } else {
            u64::from(v)
        }
    };
    Ok((norm(rl.rlim_cur), norm(rl.rlim_max)))
}

#[cfg(not(unix))]
pub fn nofile_limits() -> io::Result<(u64, u64)> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no RLIMIT_NOFILE",
    ))
}

#[cfg(unix)]
fn set_soft_nofile(soft: u64, hard: u64) -> io::Result<()> {
    let to_rlim = |v: u64| {
        if v == UNLIMITED {
            libc::RLIM_INFINITY
        } else {
            v as libc::rlim_t
        }
    };
    let rl = libc::rlimit {
        rlim_cur: to_rlim(soft),
        rlim_max: to_rlim(hard),
    };
    // SAFETY: `rl` is a valid rlimit; setrlimit only reads it.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rl) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_soft_nofile(_: u64, _: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no RLIMIT_NOFILE",
    ))
}

/// Relever la limite souple jusqu'à `min(dure, target)`. Ne panique pas, ne
/// retourne pas d'erreur : tout échec est dans [`NofileReport::problem`], que
/// l'appelant journalise.
pub fn raise_nofile_limit(target: u64) -> NofileReport {
    let (soft, hard) = match nofile_limits() {
        Ok(v) => v,
        Err(e) => {
            return NofileReport {
                soft_before: 0,
                hard: 0,
                soft_after: 0,
                requested: 0,
                problem: Some(format!("cannot read RLIMIT_NOFILE: {e}")),
            }
        }
    };
    let mut report = NofileReport {
        soft_before: soft,
        hard,
        soft_after: soft,
        requested: 0,
        problem: None,
    };
    let candidates = candidate_limits(soft, hard, target);
    let Some(&first) = candidates.first() else {
        if target > soft {
            report.problem = Some(format!(
                "soft limit {} is already the hard limit: cannot reach the target {}",
                NofileReport::fmt_limit(soft),
                target
            ));
        }
        return report;
    };
    report.requested = first;
    let mut last_err = None;
    for c in candidates {
        match set_soft_nofile(c, hard) {
            Ok(()) => {
                report.soft_after = nofile_limits().map(|(s, _)| s).unwrap_or(c);
                if c < target.min(hard) {
                    report.problem = Some(format!(
                        "asked {}, the system accepted only {} ({})",
                        first,
                        c,
                        last_err
                            .as_ref()
                            .map(|e: &io::Error| e.to_string())
                            .unwrap_or_default()
                    ));
                }
                return report;
            }
            Err(e) => last_err = Some(e),
        }
    }
    report.problem = Some(format!(
        "setrlimit refused every value tried: {}",
        last_err.map(|e| e.to_string()).unwrap_or_default()
    ));
    report
}

/// Cible lue dans l'environnement (valeur invalide ou nulle : le défaut).
pub fn nofile_target_from_env() -> u64 {
    std::env::var(ENV_NOFILE_TARGET)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_NOFILE_TARGET)
}

/// Relever la limite et la journaliser (info, ou warn si elle n'a pas pu l'être
/// comme voulu). À appeler une fois, tout au début du service.
pub fn raise_nofile_limit_and_log() -> NofileReport {
    let report = raise_nofile_limit(nofile_target_from_env());
    match &report.problem {
        None => tracing::info!("File descriptors: {}", report.summary()),
        Some(problem) => tracing::warn!("File descriptors: {} — {}", report.summary(), problem),
    }
    report
}

// ── Usage ──────────────────────────────────────────────────────────────

/// Résultat d'une mesure du nombre de descripteurs ouverts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdReading {
    Open(usize),
    /// La table est pleine (EMFILE) ou `/dev/fd` illisible : on NE DEVINE PAS.
    Unavailable {
        errno: Option<i32>,
    },
}

struct DirHandle(*mut libc::DIR);
// SAFETY: the handle is only ever used behind the mutex below.
unsafe impl Send for DirHandle {}

static METER: Mutex<Option<DirHandle>> = Mutex::new(None);

/// Ouvre `/dev/fd` UNE FOIS, au démarrage, et le garde ouvert : mesurer ne
/// consomme alors aucun descripteur, y compris quand la table est pleine —
/// précisément le moment où il faut pouvoir mesurer. (Choix retenu parmi : un
/// descripteur gardé ouvert d'avance, ou `proc_pidinfo` sous macOS — non
/// portable. `/dev/fd` existe sous Linux et macOS.) Idempotent.
#[cfg(unix)]
pub fn init_fd_meter() {
    let mut guard = METER.lock().unwrap();
    if guard.is_some() {
        return;
    }
    let path = std::ffi::CString::new("/dev/fd").unwrap();
    // SAFETY: valid NUL-terminated path.
    let dir = unsafe { libc::opendir(path.as_ptr()) };
    if !dir.is_null() {
        *guard = Some(DirHandle(dir));
    }
}

#[cfg(not(unix))]
pub fn init_fd_meter() {}

/// Mesure par le descripteur gardé ouvert (`None` si non initialisé ou en échec).
#[cfg(unix)]
fn read_via_held_dir() -> Option<usize> {
    let guard = METER.lock().unwrap();
    let DirHandle(dir) = guard.as_ref()?;
    // SAFETY: `dir` is a live DIR* protected by the mutex; entries are read
    // until NULL, d_name is a NUL-terminated array.
    unsafe {
        libc::rewinddir(*dir);
        let mut n = 0usize;
        loop {
            let ent = libc::readdir(*dir);
            if ent.is_null() {
                break;
            }
            let name = std::ffi::CStr::from_ptr((*ent).d_name.as_ptr());
            if !name.to_bytes().starts_with(b".") {
                n += 1;
            }
        }
        // The held directory's own descriptor is listed too.
        Some(n.saturating_sub(1))
    }
}

#[cfg(not(unix))]
fn read_via_held_dir() -> Option<usize> {
    None
}

/// Mesure en OUVRANT `/dev/fd` : consomme un descripteur, donc échoue quand la
/// table est pleine. C'est le chemin de repli, et ce que faisait la première
/// version du module.
pub fn measure_via_read_dir() -> FdReading {
    match std::fs::read_dir("/dev/fd") {
        Ok(rd) => FdReading::Open(rd.count().saturating_sub(1)),
        Err(e) => FdReading::Unavailable {
            errno: e.raw_os_error(),
        },
    }
}

/// Nombre de descripteurs ouverts : par le descripteur gardé, sinon repli.
pub fn read_open_fds() -> FdReading {
    match read_via_held_dir() {
        Some(n) => FdReading::Open(n),
        None => measure_via_read_dir(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdUsage {
    pub open: usize,
    pub soft_limit: u64,
}

impl FdUsage {
    pub fn ratio(&self) -> f64 {
        if self.soft_limit == UNLIMITED || self.soft_limit == 0 {
            0.0
        } else {
            self.open as f64 / self.soft_limit as f64
        }
    }
    pub fn alert(&self) -> bool {
        self.ratio() >= ALERT_RATIO
    }
}

/// Usage mesuré, ou « mesure impossible » (qui est lui-même une alerte).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FdStatus {
    Usage(FdUsage),
    Unavailable { errno: Option<i32> },
}

impl FdStatus {
    pub fn alert(&self) -> bool {
        match self {
            FdStatus::Usage(u) => u.alert(),
            FdStatus::Unavailable { .. } => true,
        }
    }
    /// Texte du journal. Jamais de pourcentage quand la mesure est impossible.
    pub fn describe(&self) -> String {
        match self {
            FdStatus::Usage(u) => format!(
                "{} open / {} ({}%)",
                u.open,
                NofileReport::fmt_limit(u.soft_limit),
                (u.ratio() * 100.0) as u32
            ),
            FdStatus::Unavailable { errno } => format!(
                "measurement impossible (errno {}): the descriptor table is probably full",
                errno.map_or("?".to_string(), |e| e.to_string())
            ),
        }
    }
}

pub fn fd_status() -> FdStatus {
    match read_open_fds() {
        FdReading::Unavailable { errno } => FdStatus::Unavailable { errno },
        FdReading::Open(open) => match nofile_limits() {
            Ok((soft_limit, _)) => FdStatus::Usage(FdUsage { open, soft_limit }),
            Err(e) => FdStatus::Unavailable {
                errno: e.raw_os_error(),
            },
        },
    }
}

/// Une période de surveillance : ce qu'il faut journaliser, calculé à part de
/// la boucle pour être testable.
#[derive(Debug, Clone)]
pub struct MonitorReport {
    pub status: FdStatus,
    pub alert: bool,
    pub message: String,
}

pub fn monitor_report() -> MonitorReport {
    let status = fd_status();
    MonitorReport {
        alert: status.alert(),
        message: status.describe(),
        status,
    }
}

/// Surveillance périodique : warn à chaque période si ≥ 80 % OU si la mesure
/// est impossible ; un rappel info toutes les 10 périodes sinon ; et, si le
/// suivi par pair est actif, les adresses distantes les plus chargées.
pub fn spawn_fd_monitor(period: Duration, tracker: ConnTracker) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(period);
        let mut n: u64 = 0;
        loop {
            ticker.tick().await;
            n += 1;
            let snap = tracker.snapshot();
            let r = monitor_report();
            if r.alert {
                tracing::warn!(
                    connections = snap.active,
                    cap = snap.max,
                    rejected_total = snap.rejected,
                    "File descriptors ALERT: {}",
                    r.message
                );
            } else if n % 10 == 1 {
                tracing::info!(
                    connections = snap.active,
                    cap = snap.max,
                    rejected_total = snap.rejected,
                    accept_errors_total = snap.accept_errors,
                    "File descriptors usage: {}",
                    r.message
                );
            }
            if tracker.per_peer_enabled() {
                for p in snap.peers.iter().take(5) {
                    tracing::info!(
                        peer = %p.ip,
                        connections = p.count,
                        held_secs = p.held_since.as_secs(),
                        newest_open_secs = p.newest.as_secs(),
                        opened_total = p.opened_total,
                        "Inbound connections by peer"
                    );
                }
                if snap.others > 0 {
                    tracing::info!(
                        connections = snap.others,
                        "Inbound connections from peers beyond the tracking bound"
                    );
                }
            }
        }
    })
}

// ── Plafond de connexions entrantes + suivi par pair ───────────────────

/// Marge gardée libre sous la limite, pour ce qui n'est pas une connexion
/// entrante admise : refus en vol (≤ 16 + la connexion en cours d'acceptation),
/// fichiers statiques servis, sockets sortants neufs.
pub const RESERVE_FDS: u64 = 64;
/// Plancher du plafond calculé : jamais en dessous.
pub const MIN_CAP: usize = 4;
/// Refus simultanés au plus (chacun tient un descripteur jusqu'à sa fermeture).
pub const MAX_REJECTING: usize = 16;
/// Nombre maximal d'adresses suivies individuellement ; les autres vont dans
/// un compteur agrégé « autres ».
pub const MAX_PEERS: usize = 256;

/// Plafond dynamique : ce que la limite laisse aux connexions entrantes une
/// fois retirés les descripteurs qui ne le sont PAS (`non_inbound`, mesurés à
/// l'exécution) et la marge [`RESERVE_FDS`] (refus en vol, fichiers statiques,
/// sockets sortants neufs).
pub fn dynamic_cap(soft_limit: u64, non_inbound: usize) -> usize {
    if soft_limit == UNLIMITED {
        return usize::MAX;
    }
    let room = soft_limit.saturating_sub(non_inbound as u64 + RESERVE_FDS);
    (room as usize).max(MIN_CAP)
}

#[derive(Debug, Clone)]
pub struct PeerStat {
    pub ip: IpAddr,
    pub count: usize,
    /// Depuis combien de temps cette adresse a au moins une connexion ouverte
    /// (la série en cours : une adresse revenue à zéro est oubliée).
    pub held_since: Duration,
    /// Âge de la connexion la plus récente.
    pub newest: Duration,
    /// Connexions admises pendant la série en cours.
    pub opened_total: u64,
}

#[derive(Debug, Clone)]
pub struct ConnSnapshot {
    pub active: usize,
    /// Plafond effectif à cet instant ; 0 = pas de plafond.
    pub max: usize,
    pub rejected: u64,
    pub accept_errors: u64,
    /// Refusals whose 503 could not be written (complete) to the socket.
    pub reject_write_failed: u64,
    /// Vide si le suivi par pair est inactif. Trié par `count` décroissant.
    pub peers: Vec<PeerStat>,
    /// Connexions ouvertes d'adresses au-delà de la borne de suivi.
    pub others: usize,
    /// Connexions admises depuis le démarrage, tous pairs confondus (cohérent
    /// avec la somme des séries suivies, des « autres » et des séries closes).
    pub admitted_total: u64,
}

struct PeerEntry {
    count: usize,
    opened_total: u64,
    held_since: Instant,
    last_open: Instant,
}

#[derive(Default)]
struct PeerBook {
    peers: HashMap<IpAddr, PeerEntry>,
    others: usize,
}

struct TrackerInner {
    /// Plafond configuré (`0` = aucun) : s'ajoute au plafond dynamique.
    max: usize,
    /// Limite souple connue : active le plafond dynamique.
    soft_limit: Option<u64>,
    max_peers: usize,
    active: AtomicUsize,
    admitted: AtomicU64,
    rejected: AtomicU64,
    accept_errors: AtomicU64,
    reject_write_failed: AtomicU64,
    rejecting: AtomicUsize,
    last_log_ms: AtomicU64,
    started: Instant,
    per_peer: bool,
    book: Mutex<PeerBook>,
}

/// Compteur de connexions entrantes partagé par le listener et le diagnostic.
#[derive(Clone)]
pub struct ConnTracker {
    inner: Arc<TrackerInner>,
}

/// Une admission : ce qu'il faut pour rendre la place au lâcher du flux.
struct Admission {
    ip: IpAddr,
    tracked: bool,
}

impl ConnTracker {
    /// `max == 0` : pas de plafond configuré. `per_peer` : suivi par adresse.
    pub fn new(max: usize, per_peer: bool) -> Self {
        Self::build(max, None, per_peer, MAX_PEERS)
    }

    /// Avec le plafond dynamique dérivé de la limite souple `soft_limit`.
    pub fn with_budget(max: usize, soft_limit: u64, per_peer: bool) -> Self {
        Self::build(max, Some(soft_limit), per_peer, MAX_PEERS)
    }

    pub fn with_peer_limit(max: usize, per_peer: bool, max_peers: usize) -> Self {
        Self::build(max, None, per_peer, max_peers)
    }

    fn build(max: usize, soft_limit: Option<u64>, per_peer: bool, max_peers: usize) -> Self {
        Self {
            inner: Arc::new(TrackerInner {
                max,
                soft_limit,
                max_peers,
                active: AtomicUsize::new(0),
                admitted: AtomicU64::new(0),
                rejected: AtomicU64::new(0),
                accept_errors: AtomicU64::new(0),
                reject_write_failed: AtomicU64::new(0),
                rejecting: AtomicUsize::new(0),
                last_log_ms: AtomicU64::new(0),
                started: Instant::now(),
                per_peer,
                book: Mutex::new(PeerBook::default()),
            }),
        }
    }

    /// Réglage depuis l'environnement ; le plafond dynamique est toujours actif.
    pub fn from_env(soft_limit: u64) -> Self {
        let max = std::env::var(ENV_MAX_CONNECTIONS)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let per_peer = std::env::var(ENV_CONN_STATS)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        Self::with_budget(max, soft_limit, per_peer)
    }

    pub fn per_peer_enabled(&self) -> bool {
        self.inner.per_peer
    }

    /// Descripteurs ouverts qui ne sont NI des connexions admises NI des refus
    /// en vol, mesurés à CHAQUE admission (pas de cache : un cache de 50 ms,
    /// essayé d'abord, laissait passer ~114 connexions au lieu de ~61 quand une
    /// salve d'ouvertures hors-entrant précédait de peu une salve entrante —
    /// mesuré par `tests/fd_limit.rs`). Le nombre d'entrantes, lui, est exact.
    /// `Err(errno)` : mesure impossible.
    fn non_inbound(&self) -> Result<usize, Option<i32>> {
        let i = &self.inner;
        match read_open_fds() {
            FdReading::Open(open) => {
                let inbound =
                    i.active.load(Ordering::Acquire) + i.rejecting.load(Ordering::Acquire);
                Ok(open.saturating_sub(inbound + 1))
            }
            FdReading::Unavailable { errno } => Err(errno),
        }
    }

    /// Plafond en vigueur, `None` = aucun. La mesure impossible vaut saturation :
    /// plafond 0, on refuse.
    pub fn effective_cap(&self) -> Option<usize> {
        let i = &self.inner;
        let configured = (i.max != 0).then_some(i.max);
        let dynamic =
            i.soft_limit
                .filter(|&s| s != UNLIMITED)
                .map(|soft| match self.non_inbound() {
                    Ok(n) => dynamic_cap(soft, n),
                    Err(_) => 0,
                });
        match (configured, dynamic) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn admit(&self, ip: IpAddr) -> Option<Admission> {
        let i = &self.inner;
        let cap = self.effective_cap();
        // Réserver une place avant de regarder le plafond : deux accept
        // concurrents ne peuvent pas dépasser ensemble.
        let before = i.active.fetch_add(1, Ordering::AcqRel);
        if cap.is_some_and(|c| before >= c) {
            i.active.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        i.admitted.fetch_add(1, Ordering::Relaxed);
        let mut tracked = false;
        if i.per_peer {
            let mut book = i.book.lock().unwrap();
            let now = Instant::now();
            if let Some(e) = book.peers.get_mut(&ip) {
                e.count += 1;
                e.opened_total += 1;
                e.last_open = now;
                tracked = true;
            } else if book.peers.len() < i.max_peers {
                book.peers.insert(
                    ip,
                    PeerEntry {
                        count: 1,
                        opened_total: 1,
                        held_since: now,
                        last_open: now,
                    },
                );
                tracked = true;
            } else {
                book.others += 1;
            }
        }
        Some(Admission { ip, tracked })
    }

    fn release(&self, a: &Admission) {
        let i = &self.inner;
        i.active.fetch_sub(1, Ordering::AcqRel);
        if i.per_peer {
            let mut book = i.book.lock().unwrap();
            if a.tracked {
                if let Some(e) = book.peers.get_mut(&a.ip) {
                    e.count -= 1;
                    if e.count == 0 {
                        book.peers.remove(&a.ip); // revenue à zéro : oubliée
                    }
                }
            } else {
                book.others = book.others.saturating_sub(1);
            }
        }
    }

    pub fn snapshot(&self) -> ConnSnapshot {
        let i = &self.inner;
        let mut peers = Vec::new();
        let mut others = 0;
        if i.per_peer {
            let now = Instant::now();
            let book = i.book.lock().unwrap();
            others = book.others;
            for (ip, e) in book.peers.iter() {
                peers.push(PeerStat {
                    ip: *ip,
                    count: e.count,
                    held_since: now.saturating_duration_since(e.held_since),
                    newest: now.saturating_duration_since(e.last_open),
                    opened_total: e.opened_total,
                });
            }
            peers.sort_by(|a, b| b.count.cmp(&a.count).then(a.ip.cmp(&b.ip)));
        }
        ConnSnapshot {
            active: i.active.load(Ordering::Acquire),
            max: self.effective_cap().unwrap_or(0),
            rejected: i.rejected.load(Ordering::Relaxed),
            accept_errors: i.accept_errors.load(Ordering::Relaxed),
            reject_write_failed: i.reject_write_failed.load(Ordering::Relaxed),
            peers,
            others,
            admitted_total: i.admitted.load(Ordering::Relaxed),
        }
    }

    /// Journal limité : au plus une ligne par 5 s (une salve de milliers de
    /// refus ne doit pas noyer le journal).
    fn log_limited(&self, what: &'static str, detail: String) {
        let i = &self.inner;
        let now_ms = i.started.elapsed().as_millis() as u64 + 1;
        let last = i.last_log_ms.load(Ordering::Relaxed);
        if last == 0 || now_ms.saturating_sub(last) >= 5_000 {
            i.last_log_ms.store(now_ms, Ordering::Relaxed);
            tracing::warn!(
                rejected_total = i.rejected.load(Ordering::Relaxed),
                accept_errors_total = i.accept_errors.load(Ordering::Relaxed),
                "{what}: {detail}"
            );
        }
    }
}

static GLOBAL: OnceLock<ConnTracker> = OnceLock::new();

/// Le suivi du service (posé par `start_server`), pour le diagnostic de santé.
pub fn install_global(tracker: ConnTracker) {
    let _ = GLOBAL.set(tracker);
}

pub fn global() -> Option<&'static ConnTracker> {
    GLOBAL.get()
}

/// Le corps du refus, et la réponse complète dont `Content-Length` est CALCULÉ
/// sur ce corps (jamais écrit en dur).
pub const REJECT_BODY: &str = "too many connections, retry soon";

pub fn reject_response() -> Vec<u8> {
    format!(
        "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nConnection: close\r\n\
         Content-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
        REJECT_BODY.len(),
        REJECT_BODY
    )
    .into_bytes()
}

/// Refuser proprement : la réponse 503 complète est écrite SUR-LE-CHAMP (un
/// socket neuf a toujours la place de ~170 octets : pas d'attente, pas de tâche),
/// puis fermeture. Le client lit donc un 503 entier même pendant une salve de
/// refus. Ensuite, au mieux et à raison de [`MAX_REJECTING`] à la fois, on garde
/// le socket 250 ms en lecture pour que la fermeture soit un FIN et pas un RST
/// (qui pourrait effacer la réponse chez un client qui a déjà envoyé sa requête) ;
/// au-delà, fermeture immédiate : les refus ne retiennent pas de descripteurs
/// sans limite.
fn reject(tracker: &ConnTracker, stream: TcpStream, peer: SocketAddr) {
    let i = &tracker.inner;
    i.rejected.fetch_add(1, Ordering::Relaxed);
    tracker.log_limited(
        "Inbound connection cap reached, refusing with 503",
        format!(
            "{} active (cap {:?}), latest peer {}",
            i.active.load(Ordering::Relaxed),
            tracker.effective_cap(),
            peer.ip()
        ),
    );
    // Straight on the socket: tokio's `try_write` would refuse (WouldBlock)
    // before the reactor has reported the socket writable, which a freshly
    // accepted one has not yet (measured: empty refusals in a burst).
    let Ok(std_stream) = stream.into_std() else {
        i.reject_write_failed.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let response = reject_response();
    match std::io::Write::write(&mut &std_stream, &response) {
        Ok(n) if n == response.len() => {}
        _ => {
            i.reject_write_failed.fetch_add(1, Ordering::Relaxed);
        }
    }
    if i.rejecting.fetch_add(1, Ordering::AcqRel) >= MAX_REJECTING {
        i.rejecting.fetch_sub(1, Ordering::AcqRel);
        return; // the socket is closed here
    }
    let t = tracker.clone();
    tokio::spawn(async move {
        let _ = tokio::time::timeout(Duration::from_millis(250), async {
            let Ok(mut stream) = TcpStream::from_std(std_stream) else {
                return;
            };
            let _ = stream.shutdown().await;
            let mut sink = [0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut sink).await;
        })
        .await;
        t.inner.rejecting.fetch_sub(1, Ordering::AcqRel);
    });
}

fn is_per_connection_error(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionRefused
            | io::ErrorKind::Interrupted
    )
}

/// Pause après un échec d'`accept` qui n'est pas propre à une connexion
/// (EMFILE/ENFILE/ENOBUFS...) : assez courte pour reprendre vite quand des
/// descripteurs se libèrent, assez longue pour ne pas tourner à vide.
pub const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

/// Un `TcpListener` qui plafonne les connexions entrantes simultanées. Au
/// plafond, la connexion est acceptée puis refusée en 503 ; les connexions déjà
/// établies (flux SSE/WebSocket longs compris) ne sont JAMAIS coupées.
///
/// Le plafond est dynamique (voir [`dynamic_cap`]) : la limite moins ce que
/// le processus tient déjà HORS connexions entrantes, moins [`RESERVE_FDS`].
/// Il ne promet pas l'absence d'EMFILE si, hors connexions entrantes, le
/// processus consomme soudainement plus que la réserve : l'accept en erreur
/// reprend alors (pause [`ACCEPT_BACKOFF`]) dès que des descripteurs se libèrent.
pub struct GuardedListener {
    inner: TcpListener,
    tracker: ConnTracker,
}

impl GuardedListener {
    pub fn new(inner: TcpListener, tracker: ConnTracker) -> Self {
        Self { inner, tracker }
    }
}

/// Le flux accepté : rend sa place (et son entrée par pair) en étant lâché.
pub struct GuardedStream {
    inner: TcpStream,
    admission: Admission,
    tracker: ConnTracker,
}

impl Drop for GuardedStream {
    fn drop(&mut self) {
        self.tracker.release(&self.admission);
    }
}

impl AsyncRead for GuardedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for GuardedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }
    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

impl axum::serve::Listener for GuardedListener {
    type Io = GuardedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.inner.accept().await {
                Ok((stream, peer)) => match self.tracker.admit(peer.ip()) {
                    Some(admission) => {
                        return (
                            GuardedStream {
                                inner: stream,
                                admission,
                                tracker: self.tracker.clone(),
                            },
                            peer,
                        )
                    }
                    None => reject(&self.tracker, stream, peer),
                },
                Err(e) if is_per_connection_error(&e) => continue,
                Err(e) => {
                    self.tracker
                        .inner
                        .accept_errors
                        .fetch_add(1, Ordering::Relaxed);
                    self.tracker.log_limited("accept error", e.to_string());
                    tokio::time::sleep(ACCEPT_BACKOFF).await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_stop_at_the_hard_limit_and_step_down() {
        assert_eq!(candidate_limits(256, 100_000, 65_536)[0], 65_536);
        let c = candidate_limits(256, UNLIMITED, 65_536);
        assert_eq!(c, vec![65_536, 61_440, 10_240, 4_096, 1_024]);
        assert_eq!(candidate_limits(256, 4_096, 65_536), vec![4_096, 1_024]);
        assert!(candidate_limits(4_096, 4_096, 65_536).is_empty());
        assert!(candidate_limits(10_000, UNLIMITED, 1_000).is_empty());
    }

    #[test]
    fn the_dynamic_cap_subtracts_what_is_not_inbound_and_the_reserve() {
        // Le service vivant : 238 ouverts dont 133 entrantes -> 105 hors entrantes.
        assert_eq!(dynamic_cap(256, 238 - 133), 256 - 105 - 64);
        assert_eq!(dynamic_cap(256, 0), 192);
        assert_eq!(dynamic_cap(64, 60), MIN_CAP);
        assert_eq!(dynamic_cap(UNLIMITED, 10), usize::MAX);
    }

    #[test]
    fn usage_alerts_at_eighty_percent_and_unavailable_is_an_alert() {
        let u = |open| FdUsage {
            open,
            soft_limit: 100,
        };
        assert!(!u(79).alert());
        assert!(u(80).alert());
        assert!(!FdUsage {
            open: 5_000,
            soft_limit: UNLIMITED
        }
        .alert());
        let none = FdStatus::Unavailable { errno: Some(24) };
        assert!(none.alert());
        assert!(!none.describe().contains('%'), "{}", none.describe());
        assert!(none.describe().contains("errno 24"));
        assert!(FdStatus::Usage(u(50)).describe().contains("50%"));
    }

    #[test]
    fn the_held_meter_sees_descriptors_come_and_go() {
        init_fd_meter();
        let FdReading::Open(base) = read_open_fds() else {
            panic!("unreadable")
        };
        let files: Vec<_> = (0..20)
            .map(|_| std::fs::File::open("/dev/null").unwrap())
            .collect();
        let FdReading::Open(up) = read_open_fds() else {
            panic!()
        };
        assert!(up >= base + 20, "{base} -> {up}");
        drop(files);
        let FdReading::Open(down) = read_open_fds() else {
            panic!()
        };
        assert!(down <= base + 2, "{base} -> {down}");
    }

    #[test]
    fn the_reject_response_is_framed_from_its_body() {
        let r = String::from_utf8(reject_response()).unwrap();
        let (head, body) = r.split_once("\r\n\r\n").unwrap();
        assert_eq!(body, REJECT_BODY);
        assert!(head.starts_with("HTTP/1.1 503 "));
        assert!(
            head.contains(&format!("Content-Length: {}", body.len())),
            "{head}"
        );
    }

    #[test]
    fn the_tracker_admits_up_to_the_cap_and_frees_on_release() {
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        let t = ConnTracker::new(2, true);
        let a = t.admit(ip).unwrap();
        let _b = t.admit(ip).unwrap();
        assert!(t.admit(ip).is_none());
        let s = t.snapshot();
        assert_eq!(
            (s.active, s.peers[0].count, s.peers[0].opened_total),
            (2, 2, 2)
        );
        t.release(&a);
        assert!(t.admit(ip).is_some());
        assert!(ConnTracker::new(0, false).admit(ip).is_some());
    }

    /// Bien plus d'adresses distinctes que la borne : la table reste bornée,
    /// les comptes restent cohérents, et une adresse revenue à zéro disparaît.
    #[test]
    fn peer_tracking_is_bounded_coherent_and_forgets_idle_peers() {
        let t = ConnTracker::with_peer_limit(0, true, 8);
        let mut held = Vec::new();
        for n in 0..10_000u32 {
            let ip = IpAddr::from([10, (n >> 16) as u8, (n >> 8) as u8, n as u8]);
            held.push(t.admit(ip).unwrap());
            let book = t.inner.book.lock().unwrap();
            assert!(book.peers.len() <= 8, "{} entries", book.peers.len());
        }
        let s = t.snapshot();
        let tracked: usize = s.peers.iter().map(|p| p.count).sum();
        assert_eq!(s.peers.len(), 8);
        assert_eq!(tracked + s.others, 10_000);
        assert_eq!(s.active, 10_000);
        assert_eq!(s.admitted_total, 10_000);
        // Une adresse suivie revient à zéro : son entrée disparaît.
        let first = held.remove(0);
        t.release(&first);
        assert!(!t.inner.book.lock().unwrap().peers.contains_key(&first.ip));
        // Tout rendre : plus rien.
        for a in held.drain(..) {
            t.release(&a);
        }
        let s = t.snapshot();
        assert_eq!((s.active, s.peers.len(), s.others), (0, 0, 0));
    }
}
