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
/// Variable d'environnement : nombre de connexions de LOOPBACK admises AU-DELÀ
/// du plafond (défaut [`DEFAULT_LOOPBACK_RESERVE`]), pour que la sonde de santé,
/// l'application desktop et les clients MCP locaux passent toujours.
pub const ENV_LOOPBACK_RESERVE: &str = "ORCHESTRATOR_LOOPBACK_RESERVE";
pub const DEFAULT_LOOPBACK_RESERVE: usize = 8;
/// Variable d'environnement : délai (secondes, minimum 1) au bout duquel une
/// connexion admise sur le quota loopback dont les en-têtes de requête ne sont pas
/// complets est fermée en lecture (défaut [`DEFAULT_LOOPBACK_MAX_AGE`]).
pub const ENV_LOOPBACK_MAX_AGE_SECS: &str = "ORCHESTRATOR_LOOPBACK_MAX_AGE_SECS";
/// Le quota loopback ne sert qu'à des REQUÊTES COURTES :
/// - toute réponse sur une connexion du quota porte `Connection: close` (pas de
///   keep-alive : la connexion se ferme APRÈS sa réponse) ;
/// - un upgrade WebSocket ou une SSE sur le quota est refusé en 503 propre, donc
///   le desktop, dont le backoff repart à 1 s à chaque ouverture réussie, n'y
///   obtient jamais de `onopen` et son backoff grandit ;
/// - rien ne coupe une requête dont les en-têtes sont complets ni une réponse en
///   cours (une réponse longue non-SSE n'est donc PAS bornée sur le quota) ;
/// - HTTP/2 en clair (h2c, sans `Connection: close`) est refusé sur le quota ;
/// - filet de sécurité (slowloris) : une connexion du quota dont les en-têtes ne
///   sont pas complets (rien reçu, ou en-têtes partiels) au bout de ce délai est
///   fermée EN LECTURE seulement, et seulement quand aucun octet n'est disponible.
pub const DEFAULT_LOOPBACK_MAX_AGE: Duration = Duration::from_secs(10);
/// Plancher du délai ci-dessus.
pub const MIN_LOOPBACK_MAX_AGE: Duration = Duration::from_secs(1);

/// Le quota est borné sous la marge [`RESERVE_FDS`] : il consomme de cette marge.
pub fn clamp_loopback_reserve(n: usize) -> usize {
    n.min(RESERVE_FDS as usize / 2)
}
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
    // `rlim_t` is u64 on Linux and macOS but narrower on some targets.
    #[allow(clippy::unnecessary_cast)]
    let norm = |v: libc::rlim_t| {
        if v == libc::RLIM_INFINITY {
            UNLIMITED
        } else {
            v as u64
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
/// Where descriptors are listed. Only tests change it (to simulate a sandbox
/// where `/dev/fd` is absent or unreadable).
static METER_PATH: Mutex<Option<String>> = Mutex::new(None);

fn meter_path() -> String {
    METER_PATH
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| "/dev/fd".to_string())
}

/// Point the meter at another directory (tests: an unavailable meter).
#[doc(hidden)]
pub fn override_meter_path_for_tests(path: &str) {
    *METER_PATH.lock().unwrap() = Some(path.to_string());
    *METER.lock().unwrap() = None;
}

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
    let Ok(path) = std::ffi::CString::new(meter_path()) else {
        return;
    };
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
    match std::fs::read_dir(meter_path()) {
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

/// EMFILE / ENFILE : la table est pleine, c'est une saturation. Tout autre
/// errno (ENOENT, EACCES : conteneur, bac à sable sans `/dev/fd`) veut dire
/// « je ne sais pas mesurer », pas « saturé ».
pub fn is_saturation_errno(errno: Option<i32>) -> bool {
    matches!(errno, Some(e) if e == libc::EMFILE || e == libc::ENFILE)
}

/// Plafond statique, sans mesure : la limite moins `max(64, 25 %)`, plancher 16.
/// C'est le repli quand le mètre est indisponible sans que la table soit pleine.
pub fn static_cap(soft_limit: u64) -> usize {
    if soft_limit == UNLIMITED {
        return usize::MAX;
    }
    let reserve = (soft_limit / 4).max(RESERVE_FDS);
    (soft_limit.saturating_sub(reserve)).max(16) as usize
}

/// Usage mesuré, ou « mesure impossible ». Seul EMFILE/ENFILE est une alerte.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FdStatus {
    Usage(FdUsage),
    Unavailable { errno: Option<i32> },
}

impl FdStatus {
    pub fn alert(&self) -> bool {
        match self {
            FdStatus::Usage(u) => u.alert(),
            FdStatus::Unavailable { errno } => is_saturation_errno(*errno),
        }
    }
    /// Un mètre aveugle (quel qu'en soit le motif) est DÉGRADÉ : le plafond
    /// statique reste en place, mais on ne voit plus la table. Journalisé en
    /// `warn` à chaque période et au démarrage, et `/health` passe `degraded`.
    pub fn degraded(&self) -> bool {
        self.alert() || matches!(self, FdStatus::Unavailable { .. })
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
            FdStatus::Unavailable { errno } if is_saturation_errno(*errno) => format!(
                "measurement impossible (errno {}): the descriptor table is full",
                errno.map_or("?".to_string(), |e| e.to_string())
            ),
            FdStatus::Unavailable { errno } => format!(
                "measurement unavailable (errno {}): static cap in use, no saturation signal",
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
    /// `alert` OU mètre indisponible.
    pub degraded: bool,
    pub message: String,
}

pub fn monitor_report() -> MonitorReport {
    let status = fd_status();
    MonitorReport {
        alert: status.alert(),
        degraded: status.degraded(),
        message: status.describe(),
        status,
    }
}

/// Au démarrage : un mètre indisponible est signalé tout de suite (`warn`).
pub fn log_meter_at_startup() {
    let status = fd_status();
    if status.degraded() {
        tracing::warn!(
            "File descriptors {}: {}",
            if status.alert() {
                "ALERT"
            } else {
                "meter DEGRADED"
            },
            status.describe()
        );
    } else {
        tracing::info!("File descriptors: {}", status.describe());
    }
}

/// Surveillance périodique : warn à chaque période si ≥ 80 % OU si le mètre est
/// indisponible (dégradé) ; un rappel info toutes les 10 périodes sinon ; et, si le
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
            if r.degraded {
                tracing::warn!(
                    connections = snap.active,
                    cap = snap.max,
                    rejected_total = snap.rejected,
                    "File descriptors {}: {}",
                    if r.alert { "ALERT" } else { "meter DEGRADED" },
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
/// Limite supposée quand `getrlimit` échoue (défaut d'un job launchd).
pub const UNKNOWN_LIMIT_FALLBACK: u64 = 256;
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
    /// Connexions loopback admises au-delà du plafond, et leur borne.
    loopback_reserve: AtomicUsize,
    loopback_in_use: AtomicUsize,
    loopback_max_age_ms: AtomicU64,
    /// Pairs des connexions admises sur le quota loopback, et leur état.
    quota_conns: Mutex<HashMap<SocketAddr, Arc<QuotaState>>>,
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

/// Durée en millisecondes, sans troncature silencieuse.
fn millis_u64(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// État d'une connexion du quota loopback.
#[derive(Debug, Default)]
pub struct QuotaState {
    /// Une requête a été reçue (en-têtes lus) : plus aucune coupure, jamais.
    pub request_seen: std::sync::atomic::AtomicBool,
}

/// Une admission : ce qu'il faut pour rendre la place au lâcher du flux.
struct Admission {
    ip: IpAddr,
    tracked: bool,
    /// Admise sur le quota loopback, au-delà du plafond.
    reserved: bool,
    /// Pair de la connexion (renseigné à l'acceptation, pour le quota).
    peer: Option<SocketAddr>,
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
                loopback_reserve: AtomicUsize::new(0),
                loopback_in_use: AtomicUsize::new(0),
                loopback_max_age_ms: AtomicU64::new(millis_u64(DEFAULT_LOOPBACK_MAX_AGE)),
                quota_conns: Mutex::new(HashMap::new()),
                rejecting: AtomicUsize::new(0),
                last_log_ms: AtomicU64::new(0),
                started: Instant::now(),
                per_peer,
                book: Mutex::new(PeerBook::default()),
            }),
        }
    }

    /// Réglage depuis l'environnement ; le plafond dynamique est toujours actif.
    /// Quota de connexions loopback admises au-delà du plafond (0 = aucun).
    pub fn set_loopback_reserve(&self, n: usize) {
        self.inner.loopback_reserve.store(n, Ordering::Relaxed);
    }

    /// Délai avant fermeture d'une connexion du quota sans requête (minimum 1 s :
    /// une valeur plus basse est relevée, avec un `warn`).
    pub fn set_loopback_max_age(&self, age: Duration) {
        let age = if age < MIN_LOOPBACK_MAX_AGE {
            tracing::warn!(
                requested_ms = millis_u64(age),
                used_ms = millis_u64(MIN_LOOPBACK_MAX_AGE),
                "{ENV_LOOPBACK_MAX_AGE_SECS} raised to the 1 s minimum"
            );
            MIN_LOOPBACK_MAX_AGE
        } else {
            age
        };
        self.inner
            .loopback_max_age_ms
            .store(millis_u64(age), Ordering::Relaxed);
    }

    /// L'état d'une connexion du quota (pour le middleware), si `peer` en est une.
    pub fn quota_state(&self, peer: &SocketAddr) -> Option<Arc<QuotaState>> {
        // Hot path of every request: no lock when no quota connection exists.
        if self.inner.loopback_in_use.load(Ordering::Acquire) == 0 {
            return None;
        }
        self.inner.quota_conns.lock().unwrap().get(peer).cloned()
    }

    pub fn loopback_max_age(&self) -> Duration {
        Duration::from_millis(self.inner.loopback_max_age_ms.load(Ordering::Relaxed))
    }

    pub fn from_env(soft_limit: u64) -> Self {
        // `getrlimit` failed (soft_limit 0): do not stay on the floor cap for the
        // life of the process. Assume the launchd default and say so.
        let soft_limit = if soft_limit == 0 {
            tracing::warn!(
                "File descriptor limit unreadable: assuming {UNKNOWN_LIMIT_FALLBACK} for the connection cap"
            );
            UNKNOWN_LIMIT_FALLBACK
        } else {
            soft_limit
        };
        let max = std::env::var(ENV_MAX_CONNECTIONS)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let per_peer = std::env::var(ENV_CONN_STATS)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let tracker = Self::with_budget(max, soft_limit, per_peer);
        let reserve = std::env::var(ENV_LOOPBACK_RESERVE)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_LOOPBACK_RESERVE);
        let clamped = clamp_loopback_reserve(reserve);
        if clamped != reserve {
            tracing::warn!(
                requested = reserve,
                used = clamped,
                "{ENV_LOOPBACK_RESERVE} lowered: it must stay under the {RESERVE_FDS}-descriptor margin"
            );
        }
        tracker.set_loopback_reserve(clamped);
        if let Some(secs) = std::env::var(ENV_LOOPBACK_MAX_AGE_SECS)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            tracker.set_loopback_max_age(Duration::from_secs(secs));
        }
        tracing::info!(
            loopback_reserve = clamped,
            loopback_max_age_secs = tracker.loopback_max_age().as_secs(),
            "Loopback quota over the connection cap"
        );
        tracker
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
                    // Table full: refuse. Any other reason the meter is blind
                    // (no /dev/fd): fall back to the static cap, as without a
                    // measurement, instead of refusing everything.
                    Err(errno) if is_saturation_errno(errno) => 0,
                    Err(_) => static_cap(soft),
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
        let mut reserved = false;
        if cap.is_some_and(|c| before >= c) {
            // Over the cap: only the loopback quota gets through (health probe,
            // desktop app, local MCP clients), so that the cap can never lock
            // the service out of its own health check.
            let loopback = ip.to_canonical().is_loopback();
            let quota = i.loopback_reserve.load(Ordering::Relaxed);
            if loopback && quota > 0 && i.loopback_in_use.fetch_add(1, Ordering::AcqRel) < quota {
                reserved = true;
            } else {
                if loopback && quota > 0 {
                    i.loopback_in_use.fetch_sub(1, Ordering::AcqRel);
                }
                i.active.fetch_sub(1, Ordering::AcqRel);
                return None;
            }
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
        Some(Admission {
            ip,
            tracked,
            reserved,
            peer: None,
        })
    }

    fn release(&self, a: &Admission) {
        let i = &self.inner;
        i.active.fetch_sub(1, Ordering::AcqRel);
        if a.reserved {
            i.loopback_in_use.fetch_sub(1, Ordering::AcqRel);
            if let Some(peer) = a.peer {
                i.quota_conns.lock().unwrap().remove(&peer);
            }
        }
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
        self.snapshot_inner(true)
    }

    /// Counters only (`max` left at 0): no second `/dev/fd` listing.
    pub fn snapshot_counters(&self) -> ConnSnapshot {
        self.snapshot_inner(false)
    }

    fn snapshot_inner(&self, with_cap: bool) -> ConnSnapshot {
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
            max: if with_cap {
                self.effective_cap().unwrap_or(0)
            } else {
                0
            },
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
    fn log_limited(&self, what: &'static str, detail: impl FnOnce() -> String) {
        let i = &self.inner;
        let now_ms = i.started.elapsed().as_millis() as u64 + 1;
        let last = i.last_log_ms.load(Ordering::Relaxed);
        if last == 0 || now_ms.saturating_sub(last) >= 5_000 {
            i.last_log_ms.store(now_ms, Ordering::Relaxed);
            tracing::warn!(
                rejected_total = i.rejected.load(Ordering::Relaxed),
                accept_errors_total = i.accept_errors.load(Ordering::Relaxed),
                "{what}: {}",
                detail()
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
    // The peer address is logged ONLY when per-peer statistics are enabled
    // (ORCHESTRATOR_CONN_STATS): it is personal data nobody opted into.
    tracker.log_limited("Inbound connection cap reached, refusing with 503", || {
        if i.per_peer {
            format!(
                "{} active (cap {:?}), latest peer {}",
                i.active.load(Ordering::Relaxed),
                tracker.effective_cap(),
                peer.ip()
            )
        } else {
            format!(
                "{} active (cap {:?})",
                i.active.load(Ordering::Relaxed),
                tracker.effective_cap()
            )
        }
    });
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
    /// Connexion du quota loopback : sa minuterie de filet de sécurité et son état.
    quota: Option<(Arc<QuotaState>, std::pin::Pin<Box<tokio::time::Sleep>>)>,
    /// Début du flux d'une connexion du quota, tant qu'on cherche la préface h2c.
    sniff: Option<Vec<u8>>,
}

/// Préface d'une connexion HTTP/2 « prior knowledge » (h2c).
const H2_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

impl GuardedStream {
    /// Filet de sécurité du quota (slowloris), EN LECTURE seulement : une
    /// connexion du quota qui n'a envoyé aucune requête à l'échéance est fermée.
    /// Dès qu'une requête a été vue, plus jamais rien ne la coupe (ni lecture, ni
    /// écriture). La minuterie est sondée avec le contexte de l'appelant : une
    /// connexion silencieuse est réveillée à l'échéance.
    fn check_idle(&mut self, cx: &mut Context<'_>) -> io::Result<()> {
        use std::future::Future;
        if let Some((state, timer)) = self.quota.as_mut() {
            if timer.as_mut().poll(cx).is_ready() && !state.request_seen.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "loopback quota connection sent no request in time",
                ));
            }
        }
        Ok(())
    }
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
        let before = buf.filled().len();
        match Pin::new(&mut self.inner).poll_read(cx, buf) {
            Poll::Ready(Ok(())) => {
                // h2c is refused on the quota: HTTP/2 has no `Connection: close`, so a
                // multiplexed connection could stay open and serve any number of
                // requests, long ones included. Only a quota connection sniffs.
                if let Some(seen) = self.sniff.as_mut() {
                    seen.extend_from_slice(&buf.filled()[before..]);
                    let n = seen.len().min(H2_PREFACE.len());
                    if seen[..n] != H2_PREFACE[..n] {
                        self.sniff = None; // not h2: stop looking
                    } else if seen.len() >= H2_PREFACE.len() {
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "HTTP/2 is not served on the loopback quota",
                        )));
                    }
                }
                Poll::Ready(Ok(()))
            }
            // No byte available: this is the only moment the idle net may cut.
            Poll::Pending => match self.check_idle(cx) {
                Ok(()) => Poll::Pending,
                Err(e) => Poll::Ready(Err(e)),
            },
            other => other,
        }
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
                    Some(mut admission) => {
                        let sniff = admission.reserved.then(Vec::new);
                        let quota = if admission.reserved {
                            admission.peer = Some(peer);
                            let state = Arc::new(QuotaState::default());
                            self.tracker
                                .inner
                                .quota_conns
                                .lock()
                                .unwrap()
                                .insert(peer, state.clone());
                            Some((
                                state,
                                Box::pin(tokio::time::sleep(self.tracker.loopback_max_age())),
                            ))
                        } else {
                            None
                        };
                        return (
                            GuardedStream {
                                inner: stream,
                                admission,
                                tracker: self.tracker.clone(),
                                quota,
                                sniff,
                            },
                            peer,
                        );
                    }
                    None => reject(&self.tracker, stream, peer),
                },
                Err(e) if is_per_connection_error(&e) => continue,
                Err(e) => {
                    self.tracker
                        .inner
                        .accept_errors
                        .fetch_add(1, Ordering::Relaxed);
                    self.tracker.log_limited("accept error", || e.to_string());
                    tokio::time::sleep(ACCEPT_BACKOFF).await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

/// `true` si `text/event-stream` est le SEUL type accepté. Les POST du transport
/// MCP « Streamable HTTP » envoient `Accept: application/json, text/event-stream`
/// (la spécification l'impose) : ils acceptent aussi du JSON et doivent être servis.
pub fn accepts_only_sse(accept: &str) -> bool {
    let mut any = false;
    for part in accept.split(',') {
        let media = part.split(';').next().unwrap_or("").trim();
        if media.is_empty() {
            continue;
        }
        if !media.eq_ignore_ascii_case("text/event-stream") {
            return false;
        }
        any = true;
    }
    any
}

fn header_matches(
    headers: &axum::http::HeaderMap,
    name: &'static str,
    f: impl Fn(&str) -> bool,
) -> bool {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .is_some_and(f)
}

/// Le 503 propre d'un upgrade WebSocket ou d'une SSE sur le quota loopback.
fn quota_refusal() -> axum::response::Response {
    use axum::http::{header, HeaderValue, StatusCode};
    let mut r = axum::response::Response::new(axum::body::Body::from(
        "the loopback quota only serves short requests, retry on a normal connection",
    ));
    *r.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
    r.headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    r.headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("close"));
    r
}

/// Middleware du quota loopback (à poser autour du routeur, avec
/// `from_fn_with_state(tracker, quota_guard)` et `ConnectInfo<SocketAddr>`).
/// Sur une connexion admise AU-DELÀ du plafond : une requête vue désarme le filet
/// de sécurité ; un upgrade WebSocket ou une SSE est refusé en 503 ; toute autre
/// réponse porte `Connection: close`, donc hyper ferme APRÈS la réponse. Sur une
/// connexion ordinaire, ou sans `ConnectInfo`, il ne fait rien.
pub async fn quota_guard(
    axum::extract::State(tracker): axum::extract::State<ConnTracker>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{header, HeaderValue};
    let state = req
        .extensions()
        .get::<axum::extract::ConnectInfo<SocketAddr>>()
        .and_then(|ci| tracker.quota_state(&ci.0));
    let Some(state) = state else {
        return next.run(req).await;
    };
    state.request_seen.store(true, Ordering::Release);
    let upgrade = header_matches(req.headers(), "upgrade", |v| {
        v.eq_ignore_ascii_case("websocket")
    });
    let sse = header_matches(req.headers(), "accept", accepts_only_sse);
    if upgrade || sse {
        return quota_refusal();
    }
    let mut resp = next.run(req).await;
    if resp
        .headers()
        .get(header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"))
    {
        return quota_refusal();
    }
    resp.headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("close"));
    resp
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
    fn only_emfile_is_a_saturation_other_errnos_fall_back_to_the_static_cap() {
        assert!(is_saturation_errno(Some(libc::EMFILE)));
        assert!(is_saturation_errno(Some(libc::ENFILE)));
        assert!(!is_saturation_errno(Some(libc::ENOENT)));
        assert!(!is_saturation_errno(Some(libc::EACCES)));
        assert!(!is_saturation_errno(None));
        assert!(!FdStatus::Unavailable {
            errno: Some(libc::ENOENT)
        }
        .alert());
        assert!(FdStatus::Unavailable {
            errno: Some(libc::EMFILE)
        }
        .alert());
        assert_eq!(static_cap(256), 192);
        assert_eq!(static_cap(64), 16);
        assert_eq!(static_cap(UNLIMITED), usize::MAX);
    }

    #[test]
    fn loopback_gets_a_reserved_quota_over_the_cap_others_do_not() {
        let t = ConnTracker::new(2, false);
        t.set_loopback_reserve(2);
        let lan: IpAddr = "192.168.1.42".parse().unwrap();
        let lo: IpAddr = "127.0.0.1".parse().unwrap();
        let lo6: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        let _a = t.admit(lan).unwrap();
        let _b = t.admit(lan).unwrap();
        assert!(t.admit(lan).is_none(), "a LAN peer is refused at the cap");
        let c = t.admit(lo).expect("loopback passes over the cap");
        let _d = t.admit(lo6).expect("a v4-mapped loopback too");
        assert!(t.admit(lo).is_none(), "the quota is bounded");
        t.release(&c);
        assert!(t.admit(lo).is_some(), "a released quota slot is reusable");
        assert!(t.admit(lan).is_none());
    }

    #[test]
    fn an_unreadable_limit_does_not_pin_the_cap_to_its_floor() {
        init_fd_meter();
        // getrlimit failed upstream: soft_after is 0. The cap must not be MIN_CAP
        // for the life of the process.
        let t = ConnTracker::from_env(0);
        assert_eq!(t.inner.soft_limit, Some(UNKNOWN_LIMIT_FALLBACK));
        assert!(t.effective_cap().is_some());
    }

    #[test]
    fn an_unavailable_meter_is_degraded_and_warned_but_not_a_saturation_alert() {
        let enoent = FdStatus::Unavailable {
            errno: Some(libc::ENOENT),
        };
        assert!(!enoent.alert());
        assert!(enoent.degraded());
        assert!(FdStatus::Unavailable {
            errno: Some(libc::EMFILE)
        }
        .degraded());
        let ok = FdStatus::Usage(FdUsage {
            open: 10,
            soft_limit: 256,
        });
        assert!(!ok.degraded());
        let msg = enoent.describe();
        assert!(
            msg.contains("unavailable (errno 2)") && !msg.contains("impossible"),
            "{msg}"
        );
        assert!(!msg.contains('%'));
    }

    #[test]
    fn the_loopback_reserve_stays_under_the_descriptor_margin() {
        assert_eq!(clamp_loopback_reserve(8), 8);
        assert_eq!(clamp_loopback_reserve(1000), RESERVE_FDS as usize / 2);
        assert_eq!(clamp_loopback_reserve(0), 0);
    }

    #[test]
    fn only_a_pure_event_stream_accept_is_an_sse_request() {
        assert!(accepts_only_sse("text/event-stream"));
        assert!(accepts_only_sse(" Text/Event-Stream ; q=1 "));
        // The MCP Streamable HTTP transport sends this on every POST: it must be served.
        assert!(!accepts_only_sse("application/json, text/event-stream"));
        assert!(!accepts_only_sse("text/event-stream, */*"));
        assert!(!accepts_only_sse("*/*"));
        assert!(!accepts_only_sse(""));
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
