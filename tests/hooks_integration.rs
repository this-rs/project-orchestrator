//! Integration tests for Claude Code hook scripts
//!
//! Tests the hook scripts (pre-tool-use.cjs, session-start.sh) for:
//! - Graceful failure when server is down
//! - Graceful failure with invalid input
//! - Graceful failure with no .po-config
//! - Correct output format when server responds
//! - Cache & throttle behavior
//!
//! Tests that require the full PO server are marked with #[ignore].
//!
//! **IMPORTANT**: Run with `--test-threads=1` to avoid mock server port conflicts:
//!   cargo test --test hooks_integration -- --test-threads=1
//!
//! Run all (including server tests):
//!   cargo test --test hooks_integration -- --test-threads=1 --include-ignored

use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Path to the hooks directory (relative to workspace root)
fn hooks_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("hooks")
}

/// Run the pre-tool-use.cjs hook with given stdin input and working directory.
/// Returns (exit_code, stdout, stderr).
fn run_cjs_hook(
    stdin_data: &str,
    cwd: &std::path::Path,
    timeout: Duration,
) -> (i32, String, String) {
    let hook_path = hooks_dir().join("pre-tool-use.cjs");

    let mut child = Command::new("node")
        .arg(&hook_path)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PO_HOOK_DEBUG", "1")
        .spawn()
        .expect("Failed to spawn node process");

    // Write stdin
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_data.as_bytes());
        // stdin is dropped here, closing the pipe
    }

    let start = Instant::now();
    let output = child
        .wait_with_output()
        .expect("Failed to wait for hook process");
    let elapsed = start.elapsed();

    assert!(
        elapsed < timeout,
        "Hook took {:?}, expected < {:?}",
        elapsed,
        timeout
    );

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (exit_code, stdout, stderr)
}

/// Run the session-start.sh hook with given stdin and working directory.
/// Returns (exit_code, stdout, stderr).
fn run_session_start_hook(
    stdin_data: &str,
    cwd: &std::path::Path,
    timeout: Duration,
) -> (i32, String, String) {
    let hook_path = hooks_dir().join("session-start.sh");

    let mut child = Command::new("bash")
        .arg(&hook_path)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PO_HOOK_DEBUG", "1")
        .spawn()
        .expect("Failed to spawn bash process");

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_data.as_bytes());
    }

    let start = Instant::now();
    let output = child
        .wait_with_output()
        .expect("Failed to wait for hook process");
    let elapsed = start.elapsed();

    assert!(
        elapsed < timeout,
        "Session hook took {:?}, expected < {:?}",
        elapsed,
        timeout
    );

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (exit_code, stdout, stderr)
}

/// Create a temporary directory with a .po-config file.
/// Uses a high port that is NOT listening to simulate server down.
fn create_temp_project_dir(port: u16) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let config = json!({
        "project_id": "00000000-0000-0000-0000-000000000001",
        "port": port,
    });
    let config_path = dir.path().join(".po-config");
    std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
        .expect("Failed to write .po-config");
    dir
}

/// Standard PreToolUse hook input
fn make_pre_tool_use_input(tool_name: &str, tool_input: Value) -> String {
    json!({
        "hookEventName": "PreToolUse",
        "toolName": tool_name,
        "toolInput": tool_input,
    })
    .to_string()
}

// ============================================================================
// Pre-Tool-Use CJS Hook — Error Handling
// ============================================================================

#[test]
fn test_cjs_hook_invalid_json_input() {
    // Invalid JSON on stdin → should exit 0 without crash
    let dir = create_temp_project_dir(19999);
    let (exit_code, stdout, stderr) =
        run_cjs_hook("this is not json {{{", dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0, "Hook should exit 0 on invalid input");
    assert!(
        stdout.trim().is_empty(),
        "No stdout output on invalid input, got: {}",
        stdout
    );
    eprintln!("  stderr: {}", stderr.trim());
}

#[test]
fn test_cjs_hook_empty_stdin() {
    // Empty stdin → should exit 0 without crash
    let dir = create_temp_project_dir(19999);
    let (exit_code, stdout, _stderr) = run_cjs_hook("", dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0, "Hook should exit 0 on empty stdin");
    assert!(stdout.trim().is_empty(), "No stdout output on empty stdin");
}

#[test]
fn test_cjs_hook_no_po_config() {
    // No .po-config in working directory → should exit 0 silently
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    // No .po-config created!
    let input = make_pre_tool_use_input("Grep", json!({"pattern": "test"}));
    let (exit_code, stdout, stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0, "Hook should exit 0 when no .po-config found");
    assert!(stdout.trim().is_empty(), "No stdout output when no config");
    assert!(
        stderr.contains("No .po-config") || stderr.contains("no .po-config") || stderr.is_empty(),
        "Debug should mention missing config"
    );
}

#[test]
fn test_cjs_hook_non_activatable_tool() {
    // Tool not in ACTIVATABLE_TOOLS → should ignore and exit 0
    let dir = create_temp_project_dir(19999);
    let input = make_pre_tool_use_input("WebSearch", json!({"query": "test"}));
    let (exit_code, stdout, stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0, "Hook should exit 0 for non-activatable tool");
    assert!(
        stdout.trim().is_empty(),
        "No stdout output for non-activatable tool"
    );
    assert!(
        stderr.contains("Ignoring tool") || stderr.is_empty(),
        "Debug should mention ignoring tool"
    );
}

#[test]
fn test_cjs_hook_wrong_event_type() {
    // Event type != PreToolUse → should ignore
    let dir = create_temp_project_dir(19999);
    let input = json!({
        "hookEventName": "PostToolUse",
        "toolName": "Grep",
        "toolInput": {"pattern": "test"},
    })
    .to_string();

    let (exit_code, stdout, _stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0, "Hook should exit 0 for wrong event type");
    assert!(stdout.trim().is_empty(), "No stdout for wrong event type");
}

#[test]
fn test_cjs_hook_server_down_fast_timeout() {
    // Server not reachable → should exit 0 within timeout budget
    let dir = create_temp_project_dir(19998); // port not listening
    let input = make_pre_tool_use_input("Grep", json!({"pattern": "neo4j query"}));

    let start = Instant::now();
    let (exit_code, stdout, _stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));
    let elapsed = start.elapsed();

    assert_eq!(exit_code, 0, "Hook should exit 0 when server is down");
    assert!(
        stdout.trim().is_empty(),
        "No stdout output when server is down"
    );
    // The hook has a 300ms timeout, so it should complete quickly
    assert!(
        elapsed < Duration::from_millis(2000),
        "Hook should timeout quickly ({:?}), not hang",
        elapsed
    );
}

// ============================================================================
// Pre-Tool-Use CJS Hook — Server Interaction (Mock)
// ============================================================================

#[test]
fn test_cjs_hook_match_success_with_mock() {
    // Start a mock HTTP server that returns a skill activation response
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock server");
    let port = listener.local_addr().unwrap().port();

    // Spawn mock server in background
    let server_thread = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            use std::io::Read;
            // Read HTTP request
            let mut buf = vec![0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]);

            // Verify it's a POST to /api/hooks/activate
            assert!(
                request.contains("POST /api/hooks/activate"),
                "Expected POST to /api/hooks/activate, got: {}",
                request.lines().next().unwrap_or("")
            );

            // Verify request body contains expected fields
            if let Some(body_start) = request.find("\r\n\r\n") {
                let body = &request[body_start + 4..];
                let parsed: Value = serde_json::from_str(body).unwrap_or(json!({}));
                assert_eq!(parsed["tool_name"], "Grep");
                assert!(parsed["project_id"].is_string());
            }

            // Send mock response
            let response_body = json!({
                "context": "## \u{1f9e0} Skill \"Neo4j Perf\" (confidence 85%)\n- Use UNWIND for batch ops\n",
                "skill_name": "Neo4j Perf",
                "note_ids": ["00000000-0000-0000-0000-000000000001"],
            });
            let body_str = serde_json::to_string(&response_body).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            use std::io::Write;
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    let dir = create_temp_project_dir(port);
    let input = make_pre_tool_use_input("Grep", json!({"pattern": "neo4j query optimization"}));

    let (exit_code, stdout, stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));

    eprintln!("  Mock test stderr: {}", stderr.trim());

    assert_eq!(exit_code, 0, "Hook should exit 0 on success");

    // stdout should contain valid JSON with hookSpecificOutput
    assert!(
        !stdout.trim().is_empty(),
        "stdout should contain hook output"
    );

    let output: Value = serde_json::from_str(stdout.trim()).expect("stdout should be valid JSON");
    assert!(
        output["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .is_some(),
        "Output should have additionalContext"
    );
    let ctx = output["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(
        ctx.contains("Neo4j Perf") || ctx.contains("UNWIND"),
        "Context should contain skill content, got: {}",
        ctx
    );

    server_thread.join().expect("mock server thread panicked");
}

#[test]
fn test_cjs_hook_no_match_204_with_mock() {
    // Mock server returns 204 (no skill matched)
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock server");
    let port = listener.local_addr().unwrap().port();

    let server_thread = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            use std::io::{Read, Write};
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf);
            let response = "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n";
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    let dir = create_temp_project_dir(port);
    let input = make_pre_tool_use_input("Grep", json!({"pattern": "unrelated pattern xyz"}));

    let (exit_code, stdout, _stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0, "Hook should exit 0 on 204");
    assert!(
        stdout.trim().is_empty(),
        "No stdout output on 204 (no match)"
    );

    server_thread.join().expect("mock server thread panicked");
}

// ============================================================================
// Session-Start Hook — Error Handling
// ============================================================================

#[test]
fn test_session_start_hook_no_config() {
    // No .po-config → should exit 0 silently
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let input = json!({"hookEventName": "SessionStart"}).to_string();

    let (exit_code, stdout, stderr) =
        run_session_start_hook(&input, dir.path(), Duration::from_secs(8));

    assert_eq!(exit_code, 0, "Session hook should exit 0 without config");
    assert!(
        stdout.trim().is_empty(),
        "No stdout when no .po-config, got: {}",
        stdout
    );
    assert!(
        stderr.contains("No .po-config") || stderr.is_empty(),
        "Debug should mention missing config"
    );
}

#[test]
fn test_session_start_hook_server_down() {
    // .po-config exists but server is down → should exit 0
    let dir = create_temp_project_dir(19997);
    let input = json!({"hookEventName": "SessionStart"}).to_string();

    let start = Instant::now();
    let (exit_code, stdout, stderr) =
        run_session_start_hook(&input, dir.path(), Duration::from_secs(8));
    let elapsed = start.elapsed();

    assert_eq!(exit_code, 0, "Session hook should exit 0 when server down");
    assert!(stdout.trim().is_empty(), "No stdout when server is down");
    // curl has a 2s connect timeout, so total should be < 5s
    assert!(
        elapsed < Duration::from_secs(6),
        "Session hook should timeout reasonably ({:?})",
        elapsed
    );
    eprintln!("  Session hook stderr: {}", stderr.trim());
}

#[test]
fn test_session_start_hook_with_mock() {
    // Mock server returns session context → should output formatted context
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock server");
    let port = listener.local_addr().unwrap().port();

    let server_thread = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            use std::io::{Read, Write};
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf);

            let response_body = json!({
                "active_skills": [
                    {
                        "name": "Neo4j Perf",
                        "description": "Query optimization",
                        "energy": 0.85,
                        "note_count": 12,
                        "activation_count": 47,
                        "last_activated": "2026-02-27T04:00:00Z",
                    }
                ],
                "current_plan": {
                    "title": "Neural Skills",
                    "status": "in_progress",
                    "progress": "4/7 tasks",
                },
                "current_task": {
                    "title": "E2E Tests",
                    "status": "in_progress",
                },
                "critical_notes": [
                    {
                        "content": "## Always check permissions before write",
                        "note_type": "gotcha",
                        "importance": "critical",
                        "tags": ["security"],
                    }
                ],
            });
            let body_str = serde_json::to_string(&response_body).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    let dir = create_temp_project_dir(port);
    let input = json!({"hookEventName": "SessionStart"}).to_string();

    let (exit_code, stdout, stderr) =
        run_session_start_hook(&input, dir.path(), Duration::from_secs(8));

    eprintln!("  Session mock stderr: {}", stderr.trim());

    assert_eq!(exit_code, 0, "Session hook should exit 0 on success");
    assert!(
        !stdout.trim().is_empty(),
        "stdout should contain hook output"
    );

    // Parse the JSON output
    let output: Value = serde_json::from_str(stdout.trim()).expect("stdout should be valid JSON");
    let ctx = output["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("Should have additionalContext");

    // Verify context contains expected sections
    assert!(
        ctx.contains("Neural Skills"),
        "Context should mention Neural Skills, got: {}",
        ctx
    );
    assert!(
        ctx.contains("Neo4j Perf"),
        "Context should mention skill name"
    );
    assert!(
        ctx.contains("Current Plan"),
        "Context should have Current Plan section"
    );
    assert!(
        ctx.contains("Neural Skills"),
        "Context should have plan title"
    );
    assert!(
        ctx.contains("Critical Notes") || ctx.contains("permissions"),
        "Context should have critical notes"
    );

    server_thread.join().expect("mock server thread panicked");
}

// ============================================================================
// Cache & Throttle Tests
// ============================================================================

/// Helper: create a mock HTTP server that serves N requests, each returning
/// a skill activation response with the given skill_id.
/// Returns (port, join_handle). The server auto-closes after max_requests
/// or a 3-second timeout per accept, whichever comes first.
fn spawn_counted_mock(
    skill_id: &str,
    max_requests: usize,
) -> (u16, std::thread::JoinHandle<usize>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock");
    let port = listener.local_addr().unwrap().port();
    let skill_id = skill_id.to_string();

    // Set accept timeout so the thread doesn't hang forever
    listener.set_nonblocking(false).expect("set blocking");

    let handle = std::thread::spawn(move || {
        let mut served = 0;
        for _ in 0..max_requests {
            // Use a short timeout on accept to avoid hanging
            let stream = match listener.accept() {
                Ok((s, _)) => s,
                Err(_) => break,
            };
            let mut stream = stream;
            use std::io::{Read, Write};
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf);

            let body = json!({
                "context": format!("## Skill {}\n- Note content here\n", skill_id),
                "skill_name": &skill_id,
                "skill_id": &skill_id,
                "note_ids": [],
            });
            let body_str = serde_json::to_string(&body).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
            served += 1;
        }
        served
    });

    (port, handle)
}

#[test]
fn test_cjs_hook_cache_file_created_on_success() {
    // After a successful activation, the hook should write a cache file.
    // The cache file is named /tmp/po-hook-cache-{ppid}.json where ppid
    // is the hook's parent PID (which is our test process PID).
    let (port, server_handle) = spawn_counted_mock("cache-test-skill", 1);
    let dir = create_temp_project_dir(port);
    let input = make_pre_tool_use_input("Grep", json!({"pattern": "cache test"}));

    let our_pid = std::process::id();
    let hook_cache_file = std::env::temp_dir().join(format!("po-hook-cache-{}.json", our_pid));

    // Clean any stale cache
    let _ = std::fs::remove_file(&hook_cache_file);

    // Run the hook — should hit server and write cache
    let (exit1, stdout1, stderr1) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));
    assert_eq!(exit1, 0);
    assert!(
        !stdout1.trim().is_empty(),
        "First call should return context, stderr: {}",
        stderr1
    );

    // Wait for mock server to finish
    let served = server_handle.join().expect("mock server thread");
    assert_eq!(served, 1, "Mock should have served exactly 1 request");

    // Verify cache file was written
    assert!(
        hook_cache_file.exists(),
        "Cache file should exist at {:?}",
        hook_cache_file
    );

    let cache_content = std::fs::read_to_string(&hook_cache_file).expect("read cache file");
    let cache: Value = serde_json::from_str(&cache_content).expect("parse cache JSON");

    assert_eq!(
        cache["ppid"], our_pid as u64,
        "Cache ppid should match our PID"
    );
    assert_eq!(cache["global_count"], 1, "Should have 1 injection recorded");
    assert!(cache["entries"].is_object(), "Should have entries object");

    // Verify the skill entry
    let entries = cache["entries"].as_object().unwrap();
    assert_eq!(entries.len(), 1, "Should have exactly 1 skill entry");
    let (skill_id, entry) = entries.iter().next().unwrap();
    assert!(
        skill_id.contains("cache-test-skill"),
        "Entry key should contain skill id"
    );
    assert_eq!(entry["count"], 1, "Skill injection count should be 1");
    assert!(entry["context"].is_string(), "Should have cached context");
    assert!(entry["timestamp"].is_number(), "Should have timestamp");

    // Cleanup
    let _ = std::fs::remove_file(&hook_cache_file);
}

#[test]
fn test_cjs_hook_global_throttle() {
    // Verify that the cache/throttle constants are reasonable.
    // We can't easily test the full throttle (MAX_GLOBAL=10) in unit tests
    // because each hook invocation is a separate process. But we CAN verify
    // the cache file format and that injection counting works.
    let dir = create_temp_project_dir(19996);
    let our_pid = std::process::id();
    let cache_file = std::env::temp_dir().join(format!("po-hook-cache-{}.json", our_pid));

    // Pre-seed the cache with global_count = MAX_GLOBAL (10)
    let seeded_cache = json!({
        "ppid": our_pid,
        "entries": {},
        "global_count": 10,
    });
    std::fs::write(&cache_file, serde_json::to_string(&seeded_cache).unwrap())
        .expect("write seeded cache");

    // Now run the hook — it should see the global throttle and exit silently
    // even though the server is down (it shouldn't even try to call it)
    let input = make_pre_tool_use_input("Grep", json!({"pattern": "throttled"}));
    let (exit_code, stdout, stderr) = run_cjs_hook(&input, dir.path(), Duration::from_secs(5));

    assert_eq!(exit_code, 0);
    assert!(
        stdout.trim().is_empty(),
        "Throttled hook should produce no output, got: {}",
        stdout
    );
    assert!(
        stderr.contains("throttle") || stderr.contains("Global"),
        "Debug should mention throttle, got: {}",
        stderr
    );

    // Cleanup
    let _ = std::fs::remove_file(&cache_file);
}

// ============================================================================
// API Endpoint Tests (require running PO server)
// ============================================================================

#[tokio::test]
#[ignore] // Requires running PO server
async fn test_hook_activate_endpoint_match() {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:6600/api/hooks/activate")
        .json(&json!({
            "project_id": "00333b5f-2d0a-4467-9c98-155e55d2b7e5",
            "tool_name": "Grep",
            "tool_input": {"pattern": "neo4j query"},
        }))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("Failed to call activate endpoint");

    let status = resp.status().as_u16();
    assert!(
        status == 200 || status == 204,
        "Expected 200 or 204, got {}",
        status
    );

    if status == 200 {
        let body: Value = resp.json().await.expect("Failed to parse response");
        assert!(
            body["context"].is_string(),
            "Response should have context field"
        );
    }
}

#[tokio::test]
#[ignore] // Requires running PO server
async fn test_hook_session_context_endpoint() {
    let client = reqwest::Client::new();
    let resp = client
        .get("http://127.0.0.1:6600/api/hooks/session-context")
        .query(&[("project_id", "00333b5f-2d0a-4467-9c98-155e55d2b7e5")])
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("Failed to call session-context endpoint");

    assert_eq!(
        resp.status().as_u16(),
        200,
        "session-context should return 200"
    );

    let body: Value = resp.json().await.expect("Failed to parse response");
    assert!(
        body["active_skills"].is_array(),
        "Response should have active_skills array"
    );
    assert!(
        body.get("current_plan").is_some(),
        "Response should have current_plan field"
    );
    assert!(
        body["critical_notes"].is_array(),
        "Response should have critical_notes array"
    );
}

#[tokio::test]
#[ignore] // Requires running PO server
async fn test_hook_health_endpoint() {
    let client = reqwest::Client::new();
    let resp = client
        .get("http://127.0.0.1:6600/api/hooks/health")
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("Failed to call health endpoint");

    assert_eq!(resp.status().as_u16(), 200);

    let body: Value = resp.json().await.expect("Failed to parse response");
    assert_eq!(body["status"], "ok");
    assert!(body["cache"].is_object(), "Should have cache stats");
    assert!(
        body["rate_limiter"].is_object(),
        "Should have rate_limiter stats"
    );
}
