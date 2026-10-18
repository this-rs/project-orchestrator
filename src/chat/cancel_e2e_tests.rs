//! End to end: `POST /api/chat/sessions/{id}/cancel-tools` and
//! `POST /api/chat/sessions/{id}/cancel-task/{task_id}` over the real router and
//! the real chat WebSocket, against `fake_claude` (the scripted Claude CLI of
//! nexus, `NEXUS_FAKES_DIR`), on both engines of Claude Code (`legacy` and
//! `agent`). No `claude`, no API key, no network: these run in CI (Unit Tests and
//! Code Coverage, which build the fakes), and a missing fake FAILS.
//!
//! They replace the stubs `tests/cancel_tools_e2e.rs` (T6 of plan 28e9afe3) and
//! `tests/cancel_task_e2e.rs` (T10 of plan 754a1379, T7 of plan fc35b25e), which
//! were `#[ignore]`d waiting for a real CLI, a key and a test server. The tool
//! processes are real: the fake CLI forks `sleep 120` as the tool would, so the
//! PIDs the backend finds, signals and reports are checked with `ps`.
//!
//! Covered here, per engine:
//! - cancel-tools: the running tool's process dies (latency measured), the
//!   `tools_cancelled` frame reaches the socket (`requested_by: "user"`), the turn
//!   goes on, and a FOLLOW-UP message is answered; past the cap (10 per 60 s) the
//!   answer is `capped: true` with nothing killed.
//! - cancel-task: two background Bash tasks and one synchronous Bash; only the two
//!   background ones are tracked (`GET .../background-tasks`), each with its PID;
//!   cancelling one kills ITS process only (the other stays alive and tracked),
//!   and `active_tasks_update` then drops it; past the cap (30 per 5 min) the
//!   answer is `capped: true`.
//!
//! Not here: the lazy recovery after a crash of the server (kill -9, restart) — an
//! in-process test cannot crash its own server; the recovery itself is covered by
//! `chat::manager::tests::test_recovery_*`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use super::agent_e2e_tests::fake_bin;
use super::config::{ChatConfig, ProviderPath};
use super::manager::ChatManager;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::GraphStore;
use crate::test_helpers::mock_app_state;

const WAIT: Duration = Duration::from_secs(30);

// ----- the fake CLI -----

/// `fake_claude` behind a wrapper that hands it `transcript.jsonl` and records
/// what it reads in `stdin.jsonl`. `--version` goes straight to the fake.
struct FakeCli {
    dir: tempfile::TempDir,
}

impl FakeCli {
    fn new(transcript: &[Value]) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let body: String = transcript.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(dir.path().join("transcript.jsonl"), body).unwrap();
        let wrapper = dir.path().join("claude");
        let script = format!(
            "#!/bin/sh\n\
             for a in \"$@\"; do if [ \"$a\" = \"--version\" ]; then exec \"{fake}\" \"$@\"; fi; done\n\
             FAKE_CLAUDE_TRANSCRIPT=\"{dir}/transcript.jsonl\" \
             FAKE_CLAUDE_STDIN_OUT=\"{dir}/stdin.jsonl\" \
             FAKE_CLAUDE_STDIN_TIMEOUT_MS=20000 \
             FAKE_CLAUDE_MAX_RUNTIME_MS=180000 \
             exec \"{fake}\" \"$@\"\n",
            fake = fake_bin("fake_claude").display(),
            dir = dir.path().display()
        );
        std::fs::write(&wrapper, script).unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        Self { dir }
    }

    fn path(&self) -> String {
        self.dir.path().join("claude").display().to_string()
    }

    /// A file next to the transcript: the go signal of a background process.
    fn go(&self, name: &str) -> String {
        self.dir.path().join(name).display().to_string()
    }
}

impl Drop for FakeCli {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let read =
                std::fs::read_to_string(self.dir.path().join("stdin.jsonl")).unwrap_or_default();
            for line in read.lines() {
                let head: String = line.chars().take(200).collect();
                eprintln!("fake_claude stdin: {head}");
            }
        }
    }
}

// ----- transcript directives (the fake's vocabulary) -----

fn await_in(needle: &str) -> Value {
    json!({"op": "await_stdin", "contains": needle, "timeout_ms": 20000, "optional": true})
}

fn emit(v: Value) -> Value {
    json!({"op": "emit_json", "json": v})
}

fn init() -> Value {
    emit(json!({
        "type": "system", "subtype": "init", "session_id": "cancel-e2e",
        "model": "fake-claude", "cwd": ".", "tools": ["Bash"],
        "permissionMode": "default", "apiKeySource": "none",
    }))
}

fn assistant(content: Value) -> Value {
    emit(json!({
        "type": "assistant",
        "message": {"id": "msg_fake", "type": "message", "role": "assistant",
                    "model": "fake-claude", "content": content, "stop_reason": "end_turn"},
    }))
}

fn text(t: &str) -> Value {
    assistant(json!([{"type": "text", "text": t}]))
}

fn tool_use(id: &str, input: Value) -> Value {
    assistant(json!([{"type": "tool_use", "id": id, "name": "Bash", "input": input}]))
}

fn tool_result(id: &str, content: &str, is_error: bool) -> Value {
    emit(json!({
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": id, "content": content, "is_error": is_error}
        ]},
        "parent_tool_use_id": null,
    }))
}

fn result_ok(t: &str) -> Value {
    emit(json!({
        "type": "result", "subtype": "success", "duration_ms": 12, "duration_api_ms": 7,
        "is_error": false, "num_turns": 1, "session_id": "cancel-e2e",
        "total_cost_usd": 0.0, "usage": {"input_tokens": 3, "output_tokens": 5},
        "result": t,
    }))
}

fn answer(key: &str, t: &str) -> Vec<Value> {
    vec![await_in(key), text(t), result_ok(t)]
}

/// A foreground tool's process: `sleep 120`, a child of the CLI from the start.
fn spawn_sleeper() -> Value {
    json!({"op": "spawn_child", "program": "sleep", "args": ["120"]})
}

/// A shell that waits for `go` to exist, then runs `sleep 120`: the sleeper is the
/// newcomer the PID claim of a background task finds (it diffs the CLI's
/// descendants around the `tool_use`, see the parity test's TURN-BG).
fn spawn_sleeper_on(go: &str) -> Value {
    json!({"op": "spawn_child", "program": "sh", "args": [
        "-c", "while [ ! -e \"$0\" ]; do sleep 0.05; done; sleep 120", go]})
}

fn wait_children() -> Value {
    json!({"op": "wait_children_exit", "timeout_ms": 20000, "optional": true})
}

fn wait_eof() -> Value {
    json!({"op": "wait_eof", "timeout_ms": 120000, "optional": true})
}

// ----- processes -----

/// Whether `pid` is a live process (a zombie is dead: it only waits to be reaped).
fn alive(pid: u32) -> bool {
    let out = std::process::Command::new("ps")
        .args(["-o", "stat=", "-p", &pid.to_string()])
        .output()
        .expect("ps");
    let stat = String::from_utf8_lossy(&out.stdout).trim().to_string();
    !stat.is_empty() && !stat.starts_with('Z')
}

async fn dead_within(pids: &[u32], within: Duration) -> Option<Duration> {
    let t0 = Instant::now();
    while t0.elapsed() < within {
        if pids.iter().all(|p| !alive(*p)) {
            return Some(t0.elapsed());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    None
}

// ----- the server and its clients -----

struct Rig {
    addr: std::net::SocketAddr,
    http: reqwest::Client,
    sid: String,
    frames: Arc<Mutex<Vec<Value>>>,
    sink: futures::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        WsMessage,
    >,
    _reader: tokio::task::JoinHandle<()>,
    _cwd: tempfile::TempDir,
}

impl Rig {
    /// A server (no-auth mode) whose Claude Code is `cli` on the engine `path`; a
    /// session opened over REST with the message `WARMUP`, answered; a WebSocket
    /// client on it whose frames are kept.
    async fn start(cli: &FakeCli, path: ProviderPath) -> Self {
        let graph = Arc::new(MockGraphStore::new());
        let config = ChatConfig {
            provider_path: path,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            nexus_browser_path: None,
            obscura_path: None,
            jwt_secret: Some("cancel-e2e-secret-cancel-e2e-secret".to_string()),
            max_sessions: 10,
            claude_cli_path: Some(cli.path()),
            ..Default::default()
        };
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
            .with_claude_cli_present(true);
        manager.update_claude_cli_path(Some(cli.path())).await;
        let manager = Arc::new(manager);
        let addr = crate::test_helpers::serve_chat(manager, graph).await;
        let http = reqwest::Client::new();
        let cwd = tempfile::tempdir().unwrap();
        let opened: Value = http
            .post(format!("http://{addr}/api/chat/sessions"))
            .json(&json!({"message": "WARMUP", "cwd": cwd.path().display().to_string()}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let sid = opened["session_id"]
            .as_str()
            .unwrap_or_else(|| panic!("no session_id in {opened}"))
            .to_string();
        let url = format!("ws://{addr}/ws/chat/{sid}?last_event=0");
        let (ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        let (mut sink, mut stream) = ws.split();
        sink.send(WsMessage::text("ready")).await.unwrap();
        let frames = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&frames);
        let reader = tokio::spawn(async move {
            while let Some(Ok(msg)) = stream.next().await {
                if let WsMessage::Text(t) = msg {
                    if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                        kept.lock().await.push(v);
                    }
                }
            }
        });
        let rig = Self {
            addr,
            http,
            sid,
            frames,
            sink,
            _reader: reader,
            _cwd: cwd,
        };
        rig.said("warm").await;
        rig
    }

    async fn send(&mut self, content: &str) {
        self.sink
            .send(WsMessage::text(
                json!({"type": "user_message", "content": content}).to_string(),
            ))
            .await
            .unwrap();
    }

    /// The first frame (from the start) satisfying `pred`, waited for.
    async fn frame(&self, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(f) = self.frames.lock().await.iter().find(|f| pred(f)) {
                return f.clone();
            }
            if Instant::now() >= deadline {
                let seen: Vec<String> = self
                    .frames
                    .lock()
                    .await
                    .iter()
                    .map(|f| f["type"].to_string())
                    .collect();
                panic!("no frame {what} within {WAIT:?}; frame types seen: {seen:?}");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The assistant said `t` (an assistant frame, never the echo of a user message).
    async fn said(&self, t: &str) {
        self.frame(&format!("saying {t:?}"), |f| {
            f["type"] != "user_message" && f.to_string().contains(t)
        })
        .await;
    }

    async fn tool_use(&self, id: &str) {
        self.frame(&format!("tool_use {id}"), |f| {
            f["type"] == "tool_use" && f["id"] == id
        })
        .await;
    }

    async fn post(&self, path: &str) -> Value {
        let res = self
            .http
            .post(format!(
                "http://{}/api/chat/sessions/{}/{path}",
                self.addr, self.sid
            ))
            .send()
            .await
            .unwrap();
        let status = res.status();
        let body: Value = res.json().await.unwrap_or(Value::Null);
        assert!(status.is_success(), "POST {path}: {status} {body}");
        body
    }

    /// `GET .../background-tasks`: the tracked tasks.
    async fn tasks(&self) -> Vec<Value> {
        let body: Value = self
            .http
            .get(format!(
                "http://{}/api/chat/sessions/{}/background-tasks",
                self.addr, self.sid
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        body["tasks"].as_array().cloned().unwrap_or_default()
    }

    /// The PID of the tracked task `id`, once its claim landed (about a second
    /// after the process starts; wide deadline for the Code Coverage job).
    async fn claimed_pid(&self, id: &str) -> u32 {
        let deadline = Instant::now() + WAIT;
        loop {
            let tasks = self.tasks().await;
            if let Some(pid) = tasks
                .iter()
                .find(|t| t["id"] == id)
                .and_then(|t| t["pid"].as_u64())
            {
                return pid as u32;
            }
            if Instant::now() >= deadline {
                panic!("no PID claimed for {id} within {WAIT:?}: {tasks:?}");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

fn pids(v: &Value) -> Vec<u32> {
    v["killed_pids"]
        .as_array()
        .unwrap_or_else(|| panic!("no killed_pids in {v}"))
        .iter()
        .map(|p| p.as_u64().unwrap() as u32)
        .collect()
}

// ----- cancel-tools -----

fn cancel_tools_transcript() -> Vec<Value> {
    let mut t = vec![
        json!({"op": "capture_hooks", "timeout_ms": 5000, "optional": true}),
        await_in("WARMUP"),
        init(),
        text("warm"),
        result_ok("warm"),
        // A foreground Bash whose process runs until the user cancels the tools.
        await_in("RUN-SLEEP"),
        spawn_sleeper(),
        tool_use("t-sleep", json!({"command": "sleep 120"})),
        wait_children(),
        tool_result("t-sleep", "interrupted by cancel_tools", true),
        text("carried on after the cancel"),
        result_ok("carried on after the cancel"),
    ];
    t.extend(answer("FOLLOW-UP", "follow-up answered"));
    t.push(wait_eof());
    t
}

async fn cancel_tools_kills_the_tool_keeps_the_turn_and_caps(path: ProviderPath) {
    let cli = FakeCli::new(&cancel_tools_transcript());
    let mut rig = Rig::start(&cli, path).await;

    rig.send("RUN-SLEEP").await;
    rig.tool_use("t-sleep").await;
    // The tool's process is up (forked before the tool_use); let the engine see it.
    tokio::time::sleep(Duration::from_millis(400)).await;

    let t0 = Instant::now();
    let cancelled = rig.post("cancel-tools").await;
    assert_eq!(cancelled["capped"], false, "{cancelled}");
    let killed = pids(&cancelled);
    assert!(!killed.is_empty(), "nothing killed: {cancelled}");
    // SLA of the stub: the process is gone within 500 ms of the request; widened
    // for the instrumented Code Coverage job, the measure is printed.
    let answered = t0.elapsed();
    let gone = dead_within(&killed, Duration::from_secs(3)).await;
    eprintln!(
        "[{path:?}] cancel-tools answered in {answered:?}, {killed:?} dead {gone:?} after the answer"
    );
    assert!(
        gone.is_some(),
        "{killed:?} still alive 3 s after cancel-tools"
    );

    let frame = rig
        .frame("tools_cancelled", |f| f["type"] == "tools_cancelled")
        .await;
    assert_eq!(frame["requested_by"], "user", "{frame}");
    assert!(frame["killed_count"].as_u64().unwrap_or(0) >= 1, "{frame}");

    // The turn goes on, and the session takes the next message.
    rig.said("carried on after the cancel").await;
    rig.send("FOLLOW-UP").await;
    rig.said("follow-up answered").await;

    // The cap: 10 per 60 s per session, one used above.
    let mut capped = None;
    for _ in 0..11 {
        let r = rig.post("cancel-tools").await;
        if r["capped"] == true {
            capped = Some(r);
            break;
        }
    }
    let capped = capped.expect("no capped:true within 11 rapid cancel-tools");
    assert!(pids(&capped).is_empty(), "{capped}");
}

#[tokio::test]
async fn cancel_tools_on_the_legacy_engine_kills_the_tool_and_the_turn_goes_on() {
    cancel_tools_kills_the_tool_keeps_the_turn_and_caps(ProviderPath::Legacy).await;
}

#[tokio::test]
async fn cancel_tools_on_the_agent_engine_kills_the_tool_and_the_turn_goes_on() {
    cancel_tools_kills_the_tool_keeps_the_turn_and_caps(ProviderPath::Agent).await;
}

// ----- cancel-task -----

fn cancel_task_transcript(cli_go_a: &str, cli_go_b: &str) -> Vec<Value> {
    let bg = |id: &str, go: &str, description: &str| -> Vec<Value> {
        vec![
            tool_use(
                id,
                json!({"command": "sleep 120", "run_in_background": true,
                       "description": description}),
            ),
            spawn_sleeper_on(go),
            tool_result(
                id,
                &format!("Command running in background with ID: {id}"),
                false,
            ),
        ]
    };
    let mut t = vec![
        json!({"op": "capture_hooks", "timeout_ms": 5000, "optional": true}),
        await_in("WARMUP"),
        init(),
        text("warm"),
        result_ok("warm"),
    ];
    t.push(await_in("BG-A"));
    t.extend(bg("t-bg-a", cli_go_a, "background A"));
    t.extend([text("A started"), result_ok("A started")]);
    t.push(await_in("BG-B"));
    t.extend(bg("t-bg-b", cli_go_b, "background B"));
    t.extend([text("B started"), result_ok("B started")]);
    // A synchronous Bash: it runs and ends inside its turn, never tracked.
    t.extend([
        await_in("SYNC"),
        tool_use("t-sync", json!({"command": "pwd"})),
        tool_result("t-sync", "/work", false),
        text("sync done"),
        result_ok("sync done"),
    ]);
    t.push(wait_eof());
    t
}

/// Starts the background task `id` (message `key`), lets its process start after
/// the backend tracked it (between the two readings of the PID claim), and
/// returns its claimed PID.
async fn start_background(rig: &mut Rig, key: &str, id: &str, go: &str, said: &str) -> u32 {
    rig.send(key).await;
    rig.frame(&format!("active_tasks_update with {id}"), |f| {
        f["type"] == "active_tasks_update"
            && f["tasks"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["id"] == id))
    })
    .await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    std::fs::write(go, b"").unwrap();
    rig.said(said).await;
    rig.claimed_pid(id).await
}

async fn cancel_task_stops_one_task_only_and_caps(path: ProviderPath) {
    let probe = FakeCli::new(&[]);
    let (go_a, go_b) = (probe.go("go-a"), probe.go("go-b"));
    let cli = FakeCli::new(&cancel_task_transcript(&go_a, &go_b));
    let mut rig = Rig::start(&cli, path).await;

    let pid_a = start_background(&mut rig, "BG-A", "t-bg-a", &go_a, "A started").await;
    let pid_b = start_background(&mut rig, "BG-B", "t-bg-b", &go_b, "B started").await;
    assert_ne!(pid_a, pid_b);
    assert!(alive(pid_a) && alive(pid_b));

    rig.send("SYNC").await;
    rig.said("sync done").await;
    let tracked: Vec<String> = rig
        .tasks()
        .await
        .iter()
        .map(|t| t["id"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        {
            let mut ids = tracked.clone();
            ids.sort();
            ids
        },
        vec!["t-bg-a".to_string(), "t-bg-b".to_string()],
        "only the two background Bash are tracked, not the synchronous one"
    );

    // Cancel A: A's process dies, B's lives.
    let stopped = rig.post("cancel-task/t-bg-a").await;
    assert_eq!(stopped["capped"], false, "{stopped}");
    assert_eq!(stopped["task_id"], "t-bg-a", "{stopped}");
    assert!(pids(&stopped).contains(&pid_a), "{stopped} (pid A {pid_a})");
    assert!(
        !pids(&stopped).contains(&pid_b),
        "{stopped} (pid B {pid_b})"
    );
    assert!(
        dead_within(&[pid_a], Duration::from_secs(3))
            .await
            .is_some(),
        "A ({pid_a}) still alive after cancel-task"
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(alive(pid_b), "B ({pid_b}) died with A: no isolation");

    // The snapshot drops A (after the grace period) and keeps B.
    rig.frame("active_tasks_update with B and without A", |f| {
        f["type"] == "active_tasks_update"
            && f["tasks"].as_array().is_some_and(|ts| {
                ts.iter().any(|t| t["id"] == "t-bg-b") && !ts.iter().any(|t| t["id"] == "t-bg-a")
            })
    })
    .await;

    // The cap: 30 per 5 min per session, one used above.
    let mut capped = None;
    for _ in 0..31 {
        let r = rig.post("cancel-task/unknown-task").await;
        if r["capped"] == true {
            capped = Some(r);
            break;
        }
    }
    let capped = capped.expect("no capped:true within 31 rapid cancel-task");
    assert!(pids(&capped).is_empty(), "{capped}");
    assert!(alive(pid_b), "the capped calls must not touch B");

    // Cleanup: B's process does not outlive the test.
    let _ = std::process::Command::new("kill")
        .args(["-INT", &pid_b.to_string()])
        .status();
}

#[tokio::test]
async fn cancel_task_on_the_legacy_engine_stops_one_background_task_only() {
    cancel_task_stops_one_task_only_and_caps(ProviderPath::Legacy).await;
}

#[tokio::test]
async fn cancel_task_on_the_agent_engine_stops_one_background_task_only() {
    cancel_task_stops_one_task_only_and_caps(ProviderPath::Agent).await;
}
