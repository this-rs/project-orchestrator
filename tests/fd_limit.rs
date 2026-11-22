//! Descripteurs de fichiers : la limite est relevée, la saturation est refusée
//! proprement, rien n'est retenu après la fermeture des clients.
//!
//! Chaque test démarre un SOUS-PROCESSUS (ce même exécutable de test, relancé
//! sur `child_server`) sous `ulimit -n` bas, qui sert un routeur axum sur un
//! port éphémère de 127.0.0.1 (jamais 8080) avec le vrai `GuardedListener` du
//! service et un `ServeDir` sur un répertoire temporaire. Le test parle à ce
//! serveur en TCP brut. Aucun processus extérieur n'est touché.
//!
//! Le module est inclus par `#[path]` : ce fichier ne compile que `fd_guard.rs`
//! (libc, tokio, axum, tracing), pas le crate entier.
#![cfg(unix)]
#![allow(dead_code)]

#[path = "../src/fd_guard.rs"]
mod fd_guard;

use std::io::{BufRead, BufReader, Read};
use std::process::{Child as Proc, Command, Stdio};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Le serveur de test (sous-processus) ────────────────────────────────

/// Ne fait rien quand il est lancé comme test ordinaire ; sert quand
/// `FD_CHILD=1` (positionné par les tests ci-dessous, sous `ulimit`).
#[test]
fn child_server() {
    if std::env::var("FD_CHILD").as_deref() != Ok("1") {
        return;
    }
    use axum::extract::ConnectInfo;
    use axum::routing::get;
    use axum::serve::ListenerExt;
    use std::net::SocketAddr;

    let env_num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
    let target = env_num("FD_TARGET").unwrap_or(0);
    let max = env_num("FD_MAX").unwrap_or(0) as usize;
    let peers = std::env::var("FD_PEERS").as_deref() == Ok("1");
    let budget = std::env::var("FD_BUDGET").as_deref() != Ok("0");
    let dir = std::env::var("FD_DIR").unwrap();

    if let Ok(path) = std::env::var("FD_METER_PATH") {
        // Simulates a sandbox where /dev/fd is absent or unreadable.
        fd_guard::override_meter_path_for_tests(&path);
    }
    let loopback = env_num("FD_LOOPBACK").unwrap_or(0) as usize;
    let loopback_age = env_num("FD_LOOPBACK_AGE_MS");
    fd_guard::init_fd_meter();
    let report = fd_guard::raise_nofile_limit(target);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let tracker = if budget {
            fd_guard::ConnTracker::with_budget(max, report.soft_after, peers)
        } else {
            fd_guard::ConnTracker::new(max, peers)
        };
        tracker.set_loopback_reserve(loopback);
        if let Some(ms) = loopback_age {
            tracker.set_loopback_max_age(Duration::from_millis(ms));
        }
        let t2 = tracker.clone();
        let t3 = tracker.clone();
        let app = axum::Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/fds",
                get(|| async { format!("{:?}", fd_guard::read_open_fds()) }),
            )
            .route(
                "/stats",
                get(move || {
                    let s = t2.snapshot();
                    async move {
                        format!(
                            "{} {} {} {}",
                            s.active, s.rejected, s.accept_errors, s.reject_write_failed
                        )
                    }
                }),
            )
            .route(
                "/peers",
                get(move || {
                    let s = t3.snapshot();
                    async move {
                        s.peers
                            .iter()
                            .map(|p| {
                                format!(
                                    "{} {} {} {}\n",
                                    p.ip,
                                    p.count,
                                    p.held_since.as_secs(),
                                    p.opened_total
                                )
                            })
                            .collect::<String>()
                    }
                }),
            )
            .route("/mcp", axum::routing::post(|| async { "ok" }))
            .route(
                "/slow",
                get(|| async {
                    // A slow but ACTIVE request: nothing may cut it half way.
                    tokio::time::sleep(Duration::from_millis(2500)).await;
                    "done"
                }),
            )
            .route(
                "/stream",
                get(|| async { ([("content-type", "text/event-stream")], "data: hello\n\n") }),
            )
            .route(
                "/monitor",
                get(|| async {
                    let r = fd_guard::monitor_report();
                    format!(
                        "alert={} degraded={} msg={}",
                        r.alert, r.degraded, r.message
                    )
                }),
            )
            .route("/noninbound", get(|| async { hold_non_inbound() }))
            .route(
                "/burst",
                get(
                    |axum::extract::Query(q): axum::extract::Query<
                        std::collections::HashMap<String, String>,
                    >| async move {
                        // Outbound-style descriptors opened in a burst 200 ms from now
                        // (so while the inbound salve is in progress), held 1.5 s.
                        let n: usize = q.get("n").and_then(|v| v.parse().ok()).unwrap_or(0);
                        tokio::spawn(async move {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                            let mut files = Vec::new();
                            for _ in 0..n {
                                match std::fs::File::open("/dev/null") {
                                    Ok(f) => files.push(f),
                                    Err(_) => break,
                                }
                            }
                            tokio::time::sleep(Duration::from_millis(1500)).await;
                            drop(files);
                        });
                        "scheduled"
                    },
                ),
            )
            .route("/saturate", get(|| async { saturate_table() }))
            .route(
                "/who",
                get(|ConnectInfo(a): ConnectInfo<SocketAddr>| async move { a.ip().to_string() }),
            )
            .nest_service("/static", tower_http::services::ServeDir::new(dir));
        let app = app.layer(axum::middleware::from_fn_with_state(
            tracker.clone(),
            fd_guard::quota_guard,
        ));
        let std_l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = std_l.local_addr().unwrap().port();
        let listener = fd_guard::GuardedListener::new(std_l, tracker).tap_io(|_| {});
        println!(
            "READY {port} {} {} {}",
            report.soft_before, report.hard, report.soft_after
        );
        println!("PROBLEM {}", report.problem.is_some());
        // Quand le parent ferme notre stdin, on s'arrête.
        std::thread::spawn(|| {
            let mut sink = Vec::new();
            let _ = std::io::stdin().read_to_end(&mut sink);
            std::process::exit(0);
        });
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
}

static HELD: std::sync::Mutex<Vec<Box<dyn std::any::Any + Send>>> =
    std::sync::Mutex::new(Vec::new());

/// ~110 descripteurs qui ne sont PAS des connexions entrantes : fichiers,
/// pipes, et sockets sortants vers un écouteur local (les deux bouts sont dans
/// ce processus, donc 2 descripteurs par connexion).
fn hold_non_inbound() -> String {
    use std::os::fd::FromRawFd;
    let mut held = HELD.lock().unwrap();
    for _ in 0..40 {
        held.push(Box::new(std::fs::File::open("/dev/null").unwrap()));
    }
    for _ in 0..25 {
        let mut fds = [0i32; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        held.push(Box::new(unsafe { std::fs::File::from_raw_fd(fds[0]) }));
        held.push(Box::new(unsafe { std::fs::File::from_raw_fd(fds[1]) }));
    }
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    for _ in 0..15 {
        held.push(Box::new(std::net::TcpStream::connect(addr).unwrap()));
        held.push(Box::new(l.accept().unwrap().0));
    }
    held.push(Box::new(l));
    format!("{:?}", fd_guard::read_open_fds())
}

/// Remplit la table jusqu'à EMFILE, puis mesure des deux façons.
fn saturate_table() -> String {
    let mut files = Vec::new();
    while let Ok(f) = std::fs::File::open("/dev/null") {
        files.push(f);
    }
    let via_dir = fd_guard::measure_via_read_dir();
    let held = fd_guard::read_open_fds();
    let report = fd_guard::monitor_report();
    let out = format!(
        "via_dir={via_dir:?}\nheld={held:?}\nalert={}\nmsg={}\n",
        report.alert, report.message
    );
    drop(files);
    out
}

// ── Le parent ──────────────────────────────────────────────────────────

struct Server {
    proc: Proc,
    port: u16,
    soft_before: u64,
    hard: u64,
    soft_after: u64,
    problem: bool,
    _dir: tempdir::Dir,
}

mod tempdir {
    pub struct Dir(pub std::path::PathBuf);
    impl Dir {
        pub fn new() -> Dir {
            let p = std::env::temp_dir().join(format!(
                "fd_limit_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("a.txt"), "static-content").unwrap();
            Dir(p)
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.proc.kill(); // notre propre enfant, jamais un autre processus
        let _ = self.proc.wait();
    }
}

struct Cfg {
    /// Loopback connections admitted over the cap (0 in the cap tests, which
    /// connect from loopback themselves).
    loopback: u64,
    /// Maximum age (ms) of a connection admitted on the loopback quota.
    loopback_age_ms: Option<u64>,
    /// Another directory for the descriptor meter (an unavailable one).
    meter_path: Option<&'static str>,
    soft: u64,
    hard: u64,
    target: u64,
    max: u64,
    peers: bool,
    budget: bool,
}

impl Cfg {
    fn new(soft: u64, hard: u64) -> Cfg {
        Cfg {
            loopback: 0,
            loopback_age_ms: None,
            meter_path: None,
            soft,
            hard,
            target: 0,
            max: 0,
            peers: false,
            budget: true,
        }
    }
}

fn inherited_limits() -> (u64, u64) {
    fd_guard::nofile_limits().unwrap()
}

static RAN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SKIPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Le scénario peut-il tourner ? Lui faut-il une limite dure plus haute que
/// celle dont ce processus a hérité (qu'un processus non privilégié ne peut pas
/// relever) ? Dans ce cas il est SAUTÉ, bruyamment : la sortie dit quoi, et
/// pourquoi. Les scénarios qui ne font qu'ABAISSER les limites ne sont jamais
/// sautés tant que la limite héritée vaut au moins `need`.
fn runnable(name: &str, need_hard: u64) -> bool {
    let (_, inherited) = inherited_limits();
    if inherited >= need_hard {
        let n = RAN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        println!("FD_LIMIT scenario RAN: {name} (needs hard >= {need_hard}, inherited {inherited}; ran so far {n})");
        true
    } else {
        // In CI a skipped scenario is a failure, not a quiet pass: the runner must
        // provide the hard limit (or the scenario be removed on purpose).
        if std::env::var_os("CI").is_some() {
            panic!(
                "scenario `{name}` needs a hard RLIMIT_NOFILE of {need_hard}, the runner has {inherited}: \
                 raise the runner's hard limit (CI is set, so skipping is not allowed)"
            );
        }
        let n = SKIPPED.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        println!("FD_LIMIT scenario SKIPPED: {name}: required hard limit {need_hard}, inherited hard limit {inherited} (skipped so far {n})");
        eprintln!("FD_LIMIT scenario SKIPPED: {name}: required hard limit {need_hard}, inherited {inherited}");
        false
    }
}

fn spawn(cfg: Cfg) -> Server {
    use std::os::unix::process::CommandExt;
    let dir = tempdir::Dir::new();
    let exe = std::env::current_exe().unwrap();
    let (soft, hard) = (cfg.soft, cfg.hard);
    let mut cmd = Command::new(exe);
    if let Some(ms) = cfg.loopback_age_ms {
        cmd.env("FD_LOOPBACK_AGE_MS", ms.to_string());
    }
    if let Some(path) = cfg.meter_path {
        cmd.env("FD_METER_PATH", path);
    }
    cmd.args(["child_server", "--exact", "--nocapture", "--test-threads=1"])
        .env("FD_CHILD", "1")
        .env("FD_TARGET", cfg.target.to_string())
        .env("FD_MAX", cfg.max.to_string())
        .env("FD_PEERS", if cfg.peers { "1" } else { "0" })
        .env("FD_BUDGET", if cfg.budget { "1" } else { "0" })
        .env("FD_LOOPBACK", cfg.loopback.to_string())
        .env("FD_DIR", &dir.0)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // setrlimit in the child between fork and exec, soft AND hard in one call:
    // no shell, no `ulimit -S/-H` ordering differences between sh/bash/zsh/dash.
    // Lowering is always allowed; `runnable` has checked raising is not needed.
    unsafe {
        cmd.pre_exec(move || {
            let rl = libc::rlimit {
                rlim_cur: soft as libc::rlim_t,
                rlim_max: hard as libc::rlim_t,
            };
            if libc::setrlimit(libc::RLIMIT_NOFILE, &rl) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut proc = cmd.spawn().unwrap();
    let mut out = BufReader::new(proc.stdout.take().unwrap());
    let (mut port, mut sb, mut hd, mut sa) = (0u16, 0u64, 0u64, 0u64);
    let problem;
    let mut line = String::new();
    loop {
        line.clear();
        assert!(
            out.read_line(&mut line).unwrap() > 0,
            "child died before READY"
        );
        if let Some(rest) = line.find("READY ").map(|i| &line[i + 6..]) {
            let v: Vec<u64> = rest
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            (port, sb, hd, sa) = (v[0] as u16, v[1], v[2], v[3]);
        }
        if let Some(rest) = line.find("PROBLEM ").map(|i| &line[i + 8..]) {
            problem = rest.trim() == "true";
            break;
        }
    }
    // Le reste de la sortie du fils n'est plus lu : ne rien y écrire de volumineux.
    std::mem::forget(out);
    Server {
        proc,
        port,
        soft_before: sb,
        hard: hd,
        soft_after: sa,
        problem,
        _dir: dir,
    }
}

/// Le parent a besoin de ses propres descripteurs pour des centaines de clients.
fn raise_parent_limit() {
    let (soft, hard) = inherited_limits();
    let want = hard.min(2048);
    if soft < want {
        let rl = libc::rlimit {
            rlim_cur: want as libc::rlim_t,
            rlim_max: hard as libc::rlim_t,
        };
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rl) };
    }
}

/// Une seule instance à la fois : les tests partagent la limite du parent.
static ONE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const REQ: &str = "GET {} HTTP/1.1\r\nHost: t\r\nConnection: {}\r\n\r\n";

async fn send_get(s: &mut TcpStream, path: &str, close: bool) {
    let req =
        REQ.replacen("{}", path, 1)
            .replacen("{}", if close { "close" } else { "keep-alive" }, 1);
    s.write_all(req.as_bytes()).await.unwrap();
}

async fn send_get_with(s: &mut TcpStream, path: &str, headers: &str) {
    let req = format!("GET {path} HTTP/1.1\r\nHost: t\r\n{headers}\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
}

/// Lit une réponse HTTP complète (Content-Length) : (statut, en-têtes, corps).
async fn read_response(s: &mut TcpStream) -> Option<(u16, String, String)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 2048];
    loop {
        if let Some(end) = find(&buf, b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).to_string();
            let len = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            if buf.len() >= end + 4 + len {
                let status = head.split_whitespace().nth(1)?.parse().ok()?;
                let body = String::from_utf8_lossy(&buf[end + 4..end + 4 + len]).to_string();
                return Some((status, head, body));
            }
        }
        match s.read(&mut chunk).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

/// Une requête neuve, avec une échéance : `None` si elle n'aboutit pas.
async fn get(port: u16, path: &str, within: Duration) -> Option<(u16, String, String)> {
    tokio::time::timeout(within, async {
        let mut s = TcpStream::connect(("127.0.0.1", port)).await.ok()?;
        send_get(&mut s, path, true).await;
        read_response(&mut s).await
    })
    .await
    .ok()
    .flatten()
}

async fn eventually_ok(port: u16, path: &str, total: Duration) -> (u16, String, String) {
    let end = tokio::time::Instant::now() + total;
    loop {
        if let Some(r) = get(port, path, Duration::from_secs(2)).await {
            if r.0 == 200 {
                return r;
            }
        }
        assert!(
            tokio::time::Instant::now() < end,
            "GET {path} never answered 200 within {total:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn nums(s: &str) -> Vec<u64> {
    s.split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect()
}

// ── Les tests ──────────────────────────────────────────────────────────

/// ROUGE sans relèvement : sous une limite de 64 le serveur plafonne à ~55
/// connexions, les suivantes restent dans le backlog et n'obtiennent rien.
/// Seul scénario qui exige une limite dure héritée haute (4096) : sauté sinon.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_soft_limit_is_raised_when_the_hard_limit_allows_it() {
    let _g = ONE_AT_A_TIME.lock().await;
    raise_parent_limit();
    if !runnable("raise", 4096) {
        return;
    }
    let srv = spawn(Cfg {
        target: 1024,
        ..Cfg::new(64, 4096)
    });
    assert_eq!(srv.soft_before, 64);
    assert_eq!(srv.hard, 4096);
    assert_eq!(
        srv.soft_after, 1024,
        "the soft limit was not lifted to min(hard, target)"
    );
    assert!(!srv.problem);

    // Et c'est effectif : 100 connexions simultanées (> 64) répondent toutes.
    let mut conns = Vec::new();
    for _ in 0..100 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        conns.push(c);
    }
    for c in conns.iter_mut() {
        let r = tokio::time::timeout(Duration::from_secs(10), read_response(c))
            .await
            .expect("a connection beyond the old limit got no answer")
            .expect("closed");
        assert_eq!(r.0, 200);
    }
}

/// Limite dure basse : on ne la dépasse pas, on ne plante pas, et le problème
/// est SIGNALÉ (jamais d'échec silencieux). Ne fait qu'abaisser : tourne partout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_low_hard_limit_is_respected_and_reported() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("low-hard", 64) {
        return;
    }
    let srv = spawn(Cfg {
        target: 1024,
        ..Cfg::new(64, 64)
    });
    assert_eq!((srv.soft_before, srv.hard, srv.soft_after), (64, 64, 64));
    assert!(
        srv.problem,
        "a limit that could not be lifted must be reported"
    );
    assert_eq!(
        get(srv.port, "/health", Duration::from_secs(3))
            .await
            .unwrap()
            .0,
        200
    );
}

/// ROUGE sans plafond : les connexions au-delà du plafond reçoivent 200 au lieu
/// d'un 503 net. Les flux longs déjà établis ne sont pas coupés.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn at_the_cap_new_connections_get_503_and_service_resumes() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("cap-503", 128) {
        return;
    }
    let srv = spawn(Cfg {
        max: 20,
        ..Cfg::new(128, 128)
    });

    let mut holders = Vec::new();
    for _ in 0..20 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        let r = tokio::time::timeout(Duration::from_secs(5), read_response(&mut c))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(r.0, 200);
        holders.push(c);
    }
    for i in 0..30 {
        let r = get(srv.port, "/health", Duration::from_secs(3))
            .await
            .unwrap_or_else(|| panic!("connection {i} over the cap got no answer (hang)"));
        assert_eq!(r.0, 503, "connection {i} over the cap: {:?}", r.1);
        assert!(r.1.to_ascii_lowercase().contains("retry-after"));
    }
    for c in holders.iter_mut() {
        send_get(c, "/health", false).await;
        let r = tokio::time::timeout(Duration::from_secs(5), read_response(c))
            .await
            .expect("a long-lived connection was cut")
            .expect("closed");
        assert_eq!(r.0, 200);
    }
    drop(holders);
    eventually_ok(srv.port, "/health", Duration::from_secs(5)).await;
    let (_, _, body) = eventually_ok(srv.port, "/static/a.txt", Duration::from_secs(5)).await;
    assert_eq!(body, "static-content");
    let (_, _, stats) = eventually_ok(srv.port, "/stats", Duration::from_secs(5)).await;
    assert!(nums(&stats)[1] >= 30, "rejected counter: {stats}");
}

/// Cadrage HTTP du refus, au niveau des octets : statut, `Content-Length` égal à
/// la longueur RÉELLE du corps, corps exact, fermeture nette sans octet en trop.
/// ROUGE avec un `Content-Length` écrit en dur (31 pour un corps de 32 octets).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_503_is_framed_exactly_and_the_connection_closes_cleanly() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("503-framing", 128) {
        return;
    }
    let srv = spawn(Cfg {
        max: 1,
        ..Cfg::new(128, 128)
    });
    let mut holder = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    send_get(&mut holder, "/health", false).await;
    assert_eq!(read_response(&mut holder).await.unwrap().0, 200);

    let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    send_get(&mut c, "/health", false).await;
    let mut all = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), c.read_to_end(&mut all))
        .await
        .expect("the connection was not closed by the server")
        .unwrap();
    let end = find(&all, b"\r\n\r\n").expect("no end of headers");
    let head = String::from_utf8(all[..end].to_vec()).unwrap();
    let body = &all[end + 4..];
    assert!(
        head.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
        "{head}"
    );
    let declared: usize = head
        .lines()
        .find_map(|l| l.strip_prefix("Content-Length: "))
        .expect("no Content-Length")
        .parse()
        .unwrap();
    assert_eq!(body, b"too many connections, retry soon");
    assert_eq!(
        declared,
        body.len(),
        "Content-Length {declared} but the body is {} bytes",
        body.len()
    );
    drop(holder);
}

/// Deux régimes de refus. Dans la borne de [`fd_guard::MAX_REJECTING`] (16
/// refus simultanés) le socket reste ouvert 250 ms en lecture avant fermeture
/// (FIN). Au-delà il est fermé tout de suite. Dans les DEUX cas le 503 est écrit
/// sur-le-champ ; la différence est qu'au-delà, un client qui a déjà envoyé sa
/// requête (octets non lus côté serveur) peut recevoir un RST. Ici 100 clients
/// envoient leur requête en même temps : aucun ne se bloque (fermeture rapide),
/// et au moins 16 reçoivent le 503 complet ; le nombre exact est affiché.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refusals_within_and_beyond_the_linger_bound_all_close_fast() {
    let _g = ONE_AT_A_TIME.lock().await;
    raise_parent_limit();
    if !runnable("refusal-regimes", 256) {
        return;
    }
    let srv = spawn(Cfg {
        max: 1,
        ..Cfg::new(256, 256)
    });
    let mut holder = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    send_get(&mut holder, "/health", false).await;
    assert_eq!(read_response(&mut holder).await.unwrap().0, 200);

    let mut tasks = Vec::new();
    for _ in 0..100 {
        let port = srv.port;
        tasks.push(tokio::spawn(async move {
            let mut c = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            send_get(&mut c, "/health", true).await;
            let mut buf = Vec::new();
            let r = tokio::time::timeout(Duration::from_secs(3), c.read_to_end(&mut buf)).await;
            (r.is_ok(), buf)
        }));
    }
    let (mut complete, mut hung) = (0, 0);
    for t in tasks {
        let (closed, buf) = t.await.unwrap();
        if !closed {
            hung += 1;
        } else if String::from_utf8_lossy(&buf).starts_with("HTTP/1.1 503 ")
            && buf.ends_with(b"too many connections, retry soon")
        {
            complete += 1;
        }
    }
    println!(
        "FD_LIMIT refusal regimes: {complete} complete 503, {} reset/empty, {hung} hung (of 100)",
        100 - complete - hung
    );
    assert_eq!(hung, 0, "a refused connection hung");
    assert!(complete >= 16, "only {complete} complete 503 out of 100");
    drop(holder);
    eventually_ok(srv.port, "/health", Duration::from_secs(5)).await;
}

/// Dimensionnement : ~110 descripteurs HORS connexions entrantes (fichiers,
/// pipes, sockets sortants), limite 256, puis 150 vraies connexions entrantes
/// concurrentes bien au-delà du plafond dynamique, donc des dizaines de refus
/// simultanés. Aucun EMFILE sur accept ni sur les fichiers statiques, tous les
/// refus sont des 503 complets, et l'accept reprend. ROUGE sans le plafond.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn non_inbound_descriptors_and_a_burst_of_refusals_never_exhaust_the_table() {
    dimensioning("dimensioning", 30, true).await;
}

/// Le budget est une RÉDUCTION du risque, pas une garantie : une rafale de
/// descripteurs sortants BIEN AU-DELÀ de la marge (150 > 64 de réserve) pendant
/// la salve peut encore provoquer des EMFILE. On ne l'affirme pas absent ; on
/// vérifie que le service reprend ensuite (accept, fichiers statiques).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_outbound_burst_beyond_the_margin_may_hit_emfile_but_the_service_recovers() {
    dimensioning("burst-beyond-margin", 150, false).await;
}

async fn dimensioning(name: &str, burst: u64, expect_no_emfile: bool) {
    let _g = ONE_AT_A_TIME.lock().await;
    raise_parent_limit();
    if !runnable(name, 256) {
        return;
    }
    let srv = spawn(Cfg::new(256, 256));
    let (_, _, held) = eventually_ok(srv.port, "/noninbound", Duration::from_secs(5)).await;
    let open: u64 = held
        .trim_start_matches("Open(")
        .trim_end_matches(')')
        .parse()
        .unwrap();
    assert!(open >= 110, "non-inbound descriptors held: {held}");

    // Outbound descriptors opened in a burst DURING the inbound salve.
    eventually_ok(
        srv.port,
        &format!("/burst?n={burst}"),
        Duration::from_secs(5),
    )
    .await;
    let mut tasks = Vec::new();
    for _ in 0..150 {
        let port = srv.port;
        tasks.push(tokio::spawn(async move {
            let mut c = TcpStream::connect(("127.0.0.1", port)).await.ok()?;
            let mut buf = Vec::new();
            // Refused connections answer at once; an admitted one stays silent.
            match tokio::time::timeout(Duration::from_secs(3), c.read_to_end(&mut buf)).await {
                Err(_) => Some((true, Vec::new(), c)), // admitted, held open
                Ok(Ok(_)) => Some((false, buf, c)),
                // A reset keeps what was read before it; say so in a failure.
                Ok(Err(e)) => {
                    buf.extend_from_slice(format!("<<read error: {:?} {e}>>", e.kind()).as_bytes());
                    Some((false, buf, c))
                }
            }
        }));
    }
    let mut admitted = Vec::new();
    let mut refused = 0;
    let mut bad: Vec<String> = Vec::new();
    for t in tasks {
        let (is_admitted, bytes, c) = t
            .await
            .unwrap()
            .expect("a connection could not even be made");
        if is_admitted {
            admitted.push(c);
        } else {
            let text = String::from_utf8_lossy(&bytes).to_string();
            if text.starts_with("HTTP/1.1 503 ")
                && text.ends_with("too many connections, retry soon")
            {
                refused += 1;
            } else {
                bad.push(text);
            }
        }
    }
    if !bad.is_empty() {
        admitted.truncate(admitted.len().saturating_sub(10));
        tokio::time::sleep(Duration::from_millis(500)).await;
        let (_, _, st) = eventually_ok(srv.port, "/stats", Duration::from_secs(5)).await;
        panic!("{} refusals were not a complete 503 (first: {:?}); complete 503: {refused}; admitted {}; server stats [active rejected accept_errors reject_write_failed] = {st}", bad.len(), bad[0], admitted.len());
    }
    assert!(
        admitted.len() >= 4,
        "nothing admitted at all: {}",
        admitted.len()
    );
    assert!(
        !expect_no_emfile || refused >= 40,
        "only {refused} refusals out of 150 (admitted {})",
        admitted.len()
    );

    // Free a few slots: static files (which need a descriptor) are served now.
    admitted.truncate(admitted.len().saturating_sub(10));
    tokio::time::sleep(Duration::from_millis(2000)).await; // the burst is released
    let (_, _, body) = eventually_ok(srv.port, "/static/a.txt", Duration::from_secs(10)).await;
    assert_eq!(body, "static-content");
    let (_, _, stats) = eventually_ok(srv.port, "/stats", Duration::from_secs(5)).await;
    let n = nums(&stats);
    if expect_no_emfile {
        assert_eq!(n[2], 0, "accept hit EMFILE {} times (stats: {stats})", n[2]);
    }
    assert!(n[1] as usize >= refused);
    drop(admitted);
    eventually_ok(srv.port, "/health", Duration::from_secs(10)).await;
}

/// Épuisement RÉEL (pas de plafond, 64 descripteurs) : `accept` échoue en
/// EMFILE, le serveur ne se bloque pas, reprend quand les clients partent, et
/// sert alors ses fichiers statiques. Le compteur `accept_errors` prouve que
/// l'épuisement a bien eu lieu.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accept_survives_real_exhaustion_and_resumes() {
    let _g = ONE_AT_A_TIME.lock().await;
    raise_parent_limit();
    if !runnable("exhaustion", 64) {
        return;
    }
    let srv = spawn(Cfg {
        budget: false,
        ..Cfg::new(64, 64)
    });
    let mut flood = Vec::new();
    for _ in 0..90 {
        if let Ok(c) = TcpStream::connect(("127.0.0.1", srv.port)).await {
            flood.push(c);
        }
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(flood);
    eventually_ok(srv.port, "/health", Duration::from_secs(10)).await;
    let (_, _, body) = eventually_ok(srv.port, "/static/a.txt", Duration::from_secs(5)).await;
    assert_eq!(body, "static-content");
    let (_, _, stats) = eventually_ok(srv.port, "/stats", Duration::from_secs(5)).await;
    assert!(
        nums(&stats)[2] > 0,
        "no accept error: the exhaustion did not happen ({stats})"
    );
}

/// Mesure quand la table est pleine : le repli (ouvrir `/dev/fd`) échoue en
/// EMFILE et le dit — sans pourcentage inventé, et comme alerte ; le
/// descripteur gardé ouvert d'avance, lui, mesure encore.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_meter_reports_impossible_when_the_table_is_full_and_never_invents_a_percentage() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("meter-full-table", 64) {
        return;
    }
    let srv = spawn(Cfg {
        budget: false,
        ..Cfg::new(64, 64)
    });
    let (_, _, out) = eventually_ok(srv.port, "/saturate", Duration::from_secs(5)).await;
    assert!(
        out.contains("via_dir=Unavailable { errno: Some(24) }"),
        "{out}"
    );
    let held: u64 = out
        .lines()
        .find_map(|l| l.strip_prefix("held=Open("))
        .map(|v| v.trim_end_matches(')').parse().unwrap())
        .unwrap_or_else(|| panic!("the held descriptor could not measure: {out}"));
    assert!(held >= 55, "{out}");
    // The monitor's own report (held meter): a real usage, over 80 %, alerting.
    assert!(out.contains("alert=true"), "{out}");
    assert!(out.contains("% )") || out.contains("%)"), "{out}");
}

/// Aucun descripteur ni place retenus après la fermeture des clients.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nothing_is_retained_after_the_clients_close() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("no-retention", 256) {
        return;
    }
    let srv = spawn(Cfg {
        max: 100,
        peers: true,
        ..Cfg::new(256, 256)
    });
    let (_, _, base) = eventually_ok(srv.port, "/fds", Duration::from_secs(5)).await;
    let base: u64 = base
        .trim_start_matches("Open(")
        .trim_end_matches(')')
        .parse()
        .unwrap();

    let mut conns = Vec::new();
    for _ in 0..40 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        read_response(&mut c).await.unwrap();
        conns.push(c);
    }
    drop(conns);

    let end = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let (_, _, now) = eventually_ok(srv.port, "/fds", Duration::from_secs(5)).await;
        let (_, _, stats) = eventually_ok(srv.port, "/stats", Duration::from_secs(5)).await;
        let fds: u64 = now
            .trim_start_matches("Open(")
            .trim_end_matches(')')
            .parse()
            .unwrap();
        let active = nums(&stats)[0];
        if fds <= base && active <= 1 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < end,
            "retained after close: fds {fds} (baseline {base}), active {active}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Le suivi par pair dit QUI tient les connexions et DEPUIS QUAND, sans rien
/// présumer de leur nature.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn per_peer_stats_report_count_and_age_by_remote_address() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("per-peer", 256) {
        return;
    }
    let srv = spawn(Cfg {
        peers: true,
        ..Cfg::new(256, 256)
    });
    let mut held = Vec::new();
    for _ in 0..3 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        read_response(&mut c).await.unwrap();
        held.push(c);
    }
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let (_, _, peers) = eventually_ok(srv.port, "/peers", Duration::from_secs(5)).await;
    let line = peers
        .lines()
        .find(|l| l.starts_with("127.0.0.1 "))
        .expect(&peers);
    let f: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    assert!(f[0] >= 3, "count: {line}");
    assert!(f[1] >= 1, "held-since age in whole seconds: {line}");
    assert!(f[2] >= 3, "opened_total: {line}");
    let (_, _, who) = eventually_ok(srv.port, "/who", Duration::from_secs(5)).await;
    assert_eq!(who, "127.0.0.1");
    drop(held);
}

/// Au plafond, la sonde de santé passe encore : le quota LOOPBACK admet des
/// connexions au-delà du plafond (les LAN n'y ont pas droit : voir le test
/// unitaire du suivi). Choix : un quota réservé plutôt qu'un listener de santé
/// séparé (aucun port ni configuration ; la sonde, le desktop et le MCP local
/// sont en loopback). ROUGE sans le quota : `/health` reçoit un 503.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn at_the_cap_health_still_answers_through_the_loopback_quota() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("health-at-cap", 128) {
        return;
    }
    let srv = spawn(Cfg {
        max: 5,
        loopback: 4,
        ..Cfg::new(128, 128)
    });
    let mut holders = Vec::new();
    for _ in 0..5 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        assert_eq!(read_response(&mut c).await.unwrap().0, 200);
        holders.push(c);
    }
    for _ in 0..20 {
        let r = get(srv.port, "/health", Duration::from_secs(3))
            .await
            .expect("no answer at the cap");
        assert_eq!(r.0, 200, "health refused at the cap: {:?}", r.1);
    }
    drop(holders);
}

/// Le quota ne sert qu'aux REQUÊTES COURTES, et rien n'y coupe une requête en
/// cours. Cap 3 occupé par 3 connexions normales ; 8 de quota ; âge 1 s.
///  - une réponse sur le quota porte `Connection: close` et la connexion se ferme ;
///  - un upgrade WebSocket ou une SSE sur le quota reçoit un 503 propre (le desktop
///    n'y a donc jamais de `onopen`, son backoff grandit) ; sur une connexion
///    normale, les mêmes requêtes passent ;
///  - une requête LENTE mais active (2,5 s) sur le quota, plus longue que l'âge,
///    n'est pas coupée ;
///  - 8 connexions locales qui n'envoient rien épuisent le quota, la sonde est
///    refusée, puis le filet de sécurité (lecture seulement) les ferme et la sonde
///    repasse.
/// ROUGE sans middleware (pas de 503, pas de `Connection: close`), sans
/// `Connection: close`, ou si le filet coupe une requête commencée.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_loopback_quota_serves_short_requests_only_and_never_cuts_a_request_in_progress() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("quota-short-requests", 128) {
        return;
    }
    let srv = spawn(Cfg {
        max: 3,
        loopback: 8,
        loopback_age_ms: Some(1000),
        ..Cfg::new(128, 128)
    });
    let mut holders = Vec::new();
    for _ in 0..3 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        assert_eq!(read_response(&mut c).await.unwrap().0, 200);
        holders.push(c);
    }

    // A short request on the quota: Connection: close, then the connection closes.
    let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    send_get_with(&mut c, "/health", "").await; // keep-alive requested (HTTP/1.1 default)
    let (status, head, _) = read_response(&mut c).await.unwrap();
    assert_eq!(status, 200);
    assert!(
        head.to_ascii_lowercase().contains("connection: close"),
        "{head}"
    );
    let mut rest = Vec::new();
    tokio::time::timeout(Duration::from_secs(3), c.read_to_end(&mut rest))
        .await
        .expect("the connection stayed open after its response")
        .ok();
    assert!(rest.is_empty());

    // A WebSocket upgrade, and an SSE, on the quota: a clean 503.
    for (path, headers) in [
        ("/health", "Connection: Upgrade\r\nUpgrade: websocket\r\n"),
        ("/stream", "Accept: text/event-stream\r\n"),
        ("/stream", ""), // SSE chosen by the handler, no Accept header
    ] {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get_with(&mut c, path, headers).await;
        let (status, head, body) = read_response(&mut c).await.unwrap();
        assert_eq!(status, 503, "{path} {headers:?}: {head}");
        assert!(head.to_ascii_lowercase().contains("retry-after"), "{head}");
        assert!(body.contains("quota"), "{body}");
    }

    // The very same requests on a NORMAL connection are not refused.
    send_get_with(
        &mut holders[0],
        "/health",
        "Connection: Upgrade\r\nUpgrade: websocket\r\n",
    )
    .await;
    assert_eq!(read_response(&mut holders[0]).await.unwrap().0, 200);

    // A slow but active request, longer than the 1 s age, is not cut.
    let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    send_get_with(&mut c, "/slow", "").await;
    let r = tokio::time::timeout(Duration::from_secs(8), read_response(&mut c))
        .await
        .expect("no answer")
        .expect("a slow request on the quota was cut");
    assert_eq!((r.0, r.2.as_str()), (200, "done"));

    // Eight silent local connections use the whole quota: the probe is refused...
    let mut silent = Vec::new();
    for _ in 0..8 {
        silent.push(TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        get(srv.port, "/health", Duration::from_secs(3))
            .await
            .unwrap()
            .0,
        503
    );
    // ...until the safety net (read only, nothing was sent) closes them.
    eventually_ok(srv.port, "/health", Duration::from_secs(6)).await;
    for c in silent.iter_mut() {
        let mut buf = [0u8; 8];
        let n = tokio::time::timeout(Duration::from_secs(3), c.read(&mut buf)).await;
        assert!(
            matches!(n, Ok(Ok(0)) | Ok(Err(_))),
            "a silent quota connection outlived the net"
        );
    }
    // The normal connections were never touched.
    for c in holders.iter_mut() {
        send_get(c, "/health", false).await;
        assert_eq!(
            read_response(c).await.unwrap().0,
            200,
            "a normal connection was cut"
        );
    }
}

/// Un POST du transport MCP « Streamable HTTP » (`Accept: application/json,
/// text/event-stream`, imposé par la spécification) est SERVI sur le quota : seul un
/// `Accept` limité à `text/event-stream` est une demande de SSE. Et HTTP/2 en clair
/// (h2c) est refusé sur le quota : il n'a pas de `Connection: close`, une connexion
/// multiplexée pourrait y rester ouverte indéfiniment. ROUGE si tout `Accept` qui
/// contient `text/event-stream` est refusé, ou si la préface h2c est acceptée
/// (le serveur répond alors par sa trame SETTINGS, ou par une erreur 400).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_quota_serves_mcp_posts_and_refuses_h2c() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("quota-mcp-h2c", 128) {
        return;
    }
    let srv = spawn(Cfg {
        max: 3,
        loopback: 8,
        loopback_age_ms: Some(5000),
        ..Cfg::new(128, 128)
    });
    let mut holders = Vec::new();
    for _ in 0..3 {
        let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
        send_get(&mut c, "/health", false).await;
        assert_eq!(read_response(&mut c).await.unwrap().0, 200);
        holders.push(c);
    }
    // MCP POST on the quota.
    let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    let req = "POST /mcp HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
               Accept: application/json, text/event-stream\r\nContent-Length: 2\r\n\r\n{}";
    c.write_all(req.as_bytes()).await.unwrap();
    let (status, head, body) = read_response(&mut c)
        .await
        .expect("no answer to an MCP POST");
    assert_eq!(
        (status, body.as_str()),
        (200, "ok"),
        "an MCP POST was refused on the quota: {head}"
    );
    assert!(
        head.to_ascii_lowercase().contains("connection: close"),
        "{head}"
    );

    // h2c prior knowledge on the quota: the preface and an empty SETTINGS frame.
    let mut c = TcpStream::connect(("127.0.0.1", srv.port)).await.unwrap();
    let mut hello = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
    hello.extend_from_slice(&[0, 0, 0, 4, 0, 0, 0, 0, 0]);
    c.write_all(&hello).await.unwrap();
    let mut got = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(3), c.read_to_end(&mut got)).await;
    assert!(
        got.is_empty(),
        "h2c was served on the quota (answered {} bytes: {:?})",
        got.len(),
        String::from_utf8_lossy(&got[..got.len().min(40)])
    );
    drop(holders);
}

/// Mètre indisponible SANS table pleine (`/dev/fd` absent : conteneur, bac à
/// sable) : on retombe sur le plafond statique, on ne refuse pas tout. ROUGE si
/// « mesure impossible » vaut plafond 0 : même `/health` répondrait 503.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unavailable_meter_falls_back_to_the_static_cap_and_does_not_refuse_everything() {
    let _g = ONE_AT_A_TIME.lock().await;
    raise_parent_limit();
    if !runnable("meter-unavailable", 256) {
        return;
    }
    let srv = spawn(Cfg {
        meter_path: Some("/nonexistent/meter"),
        ..Cfg::new(256, 256)
    });
    assert_eq!(
        get(srv.port, "/health", Duration::from_secs(3))
            .await
            .unwrap()
            .0,
        200
    );
    // 100 concurrent connections: all admitted (static cap 192), none refused.
    let mut tasks = Vec::new();
    for _ in 0..100 {
        let port = srv.port;
        tasks.push(tokio::spawn(async move {
            let mut c = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            send_get(&mut c, "/health", false).await;
            let r = tokio::time::timeout(Duration::from_secs(5), read_response(&mut c)).await;
            (r.ok().flatten().map(|r| r.0), c)
        }));
    }
    let mut held = Vec::new();
    for t in tasks {
        let (status, c) = t.await.unwrap();
        assert_eq!(
            status,
            Some(200),
            "a connection was refused with the meter unavailable"
        );
        held.push(c);
    }
    drop(held);
}

/// Mètre aveugle SANS table pleine : le moniteur est DÉGRADÉ (warn à chaque
/// période, `/health` degraded), sans être une alerte de saturation, et le
/// message ne dit pas « impossible ». ROUGE si `degraded` ne couvre que l'alerte.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_blind_meter_is_reported_degraded_not_as_a_saturation_alert() {
    let _g = ONE_AT_A_TIME.lock().await;
    if !runnable("meter-degraded", 128) {
        return;
    }
    let srv = spawn(Cfg {
        meter_path: Some("/nonexistent/meter"),
        ..Cfg::new(128, 128)
    });
    let (_, _, out) = eventually_ok(srv.port, "/monitor", Duration::from_secs(5)).await;
    assert!(out.contains("degraded=true"), "{out}");
    assert!(out.contains("alert=false"), "{out}");
    assert!(out.contains("unavailable (errno 2)"), "{out}");
    assert!(!out.contains("impossible"), "{out}");
    assert!(!out.contains('%'), "{out}");
}

#[test]
fn the_parent_reads_its_own_usage() {
    fd_guard::init_fd_meter();
    assert!(matches!(fd_guard::fd_status(), fd_guard::FdStatus::Usage(u) if u.open > 0));
}

/// Dernier test lancé à la main avec `--test-threads=1`, ou lu dans la sortie :
/// rappelle que les lignes `FD_LIMIT scenario RAN/SKIPPED` disent ce qui a tourné.
#[test]
fn core_scenarios_need_only_a_modest_inherited_hard_limit() {
    // Saturation, refus propre, reprise de l'accept et libération ne font
    // qu'abaisser les limites (<= 256) : si même cela manque, c'est un échec
    // bruyant, pas un saut silencieux.
    let (_, hard) = inherited_limits();
    assert!(
        hard >= 256,
        "the inherited hard limit {hard} is below 256: the core scenarios cannot run"
    );
}
