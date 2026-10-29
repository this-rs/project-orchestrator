//! V4 — `orchestrator secret get|exec` as an agent's shell actually runs them.
//!
//! These tests run the REAL `orchestrator` binary against a fake vault server
//! that knows exactly one granted secret. They pin the three acceptance
//! criteria of V4 that the unit tests of `vault::agent_cli` do not cover:
//!
//! 1. Without a grant (or without a session token), `secret get` exits with a
//!    non-zero code, prints a clear reason on stderr, and writes NOTHING on
//!    stdout — a value never leaks through a failure path.
//! 2. `secret exec` hands the value to the child through its ENVIRONMENT only:
//!    the argv of the orchestrator process names the secret, never its value,
//!    and the child does not inherit the session token.
//! 3. Without a grant, `secret exec` runs nothing at all.
use project_orchestrator::neo4j::traits::*;

use std::path::Path;
use std::process::{Command, Output};

use axum::body::Body;
use axum::extract::Json;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;

const GRANTED_NAME: &str = "db-prod";
const GRANTED_VALUE: &str = "v4-s3cr3t-value-7f3a9c";
const SESSION_TOKEN: &str = "session-token-of-the-agent";

/// The only route `orchestrator secret` talks to. Mirrors the real handler's
/// contract: the value as `text/plain` on success, `{"error": …}` otherwise.
async fn agent_read(headers: HeaderMap, Json(body): Json<serde_json::Value>) -> Response {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let name = body.get("name").and_then(|n| n.as_str()).unwrap_or("");
    if bearer != Some(SESSION_TOKEN) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "vault access needs a session token" })),
        )
            .into_response();
    }
    if name != GRANTED_NAME {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("secret `{name}` is not granted to this session")
            })),
        )
            .into_response();
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(GRANTED_VALUE))
        .expect("static response")
}

/// Starts the fake vault on an ephemeral port and returns its base URL.
async fn fake_vault() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind an ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let app = Router::new().route("/api/vault/agent/read", post(agent_read));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("fake vault serves");
    });
    format!("http://{addr}")
}

/// Runs `orchestrator secret <args…>` with the given environment, off the
/// async runtime (the child is a plain blocking process).
async fn run_secret(base_url: &str, token: Option<&str>, args: &[String]) -> Output {
    let base_url = base_url.to_string();
    let token = token.map(str::to_string);
    let args = args.to_vec();
    tokio::task::spawn_blocking(move || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_orchestrator"));
        cmd.arg("secret").args(&args);
        cmd.env_clear();
        // A minimal PATH so `/bin/sh` can run what the test asks of it.
        cmd.env("PATH", "/usr/bin:/bin");
        cmd.env("PO_SERVER_URL", &base_url);
        if let Some(t) = &token {
            cmd.env("PO_VAULT_TOKEN", t);
        }
        cmd.output().expect("the orchestrator binary runs")
    })
    .await
    .expect("spawn_blocking")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_without_a_grant_exits_non_zero_with_a_reason_and_prints_no_value() {
    let base = fake_vault().await;
    let out = run_secret(
        &base,
        Some(SESSION_TOKEN),
        &["get".into(), "other-secret".into()],
    )
    .await;

    assert_eq!(out.status.code(), Some(1), "refusal is exit code 1");
    assert!(
        out.stdout.is_empty(),
        "nothing on stdout when refused, got {:?}",
        text(&out.stdout)
    );
    let err = text(&out.stderr);
    assert!(
        err.contains("not granted to this session"),
        "the server's reason reaches the agent verbatim, got {err:?}"
    );
    assert!(
        err.contains("other-secret"),
        "the refused name is named: {err:?}"
    );
    assert!(
        !err.contains(GRANTED_VALUE),
        "no value in a refusal: {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_without_a_session_token_exits_non_zero_and_says_why() {
    let base = fake_vault().await;
    let out = run_secret(&base, None, &["get".into(), GRANTED_NAME.into()]).await;

    assert_eq!(out.status.code(), Some(1));
    assert!(
        out.stdout.is_empty(),
        "no value without a token: {:?}",
        text(&out.stdout)
    );
    let err = text(&out.stderr);
    assert!(
        err.contains("PO_VAULT_TOKEN"),
        "the message names the missing variable: {err:?}"
    );
    assert!(!err.contains(GRANTED_VALUE));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_with_a_grant_writes_the_bare_value_to_stdout_only() {
    let base = fake_vault().await;
    let out = run_secret(
        &base,
        Some(SESSION_TOKEN),
        &["get".into(), GRANTED_NAME.into()],
    )
    .await;

    assert_eq!(out.status.code(), Some(0), "stderr: {}", text(&out.stderr));
    // No trailing newline: `--password-stdin` consumers would take it as part
    // of the value.
    assert_eq!(text(&out.stdout), GRANTED_VALUE);
    assert!(
        out.stderr.is_empty(),
        "nothing on stderr on success: {:?}",
        text(&out.stderr)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_hands_the_value_through_the_environment_and_never_through_argv() {
    let base = fake_vault().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("seen-by-the-child");

    // The child reports what it sees: the mapped variable and whether the
    // session token leaked into its environment.
    let script = format!(
        "printf '%s|%s' \"$DB_PASSWORD\" \"${{PO_VAULT_TOKEN:-absent}}\" > '{}'",
        out_path.display()
    );
    let args: Vec<String> = vec![
        "exec".into(),
        "-e".into(),
        format!("DB_PASSWORD={GRANTED_NAME}"),
        "--".into(),
        "/bin/sh".into(),
        "-c".into(),
        script,
    ];

    // Criterion 2a: the argv of the orchestrator process carries the NAME of
    // the secret, never its value. (The value is only known server-side.)
    assert!(
        args.iter().all(|a| !a.contains(GRANTED_VALUE)),
        "argv must never carry a secret value: {args:?}"
    );

    let out = run_secret(&base, Some(SESSION_TOKEN), &args).await;

    assert_eq!(out.status.code(), Some(0), "stderr: {}", text(&out.stderr));
    assert!(
        out.stdout.is_empty(),
        "exec prints nothing: {:?}",
        text(&out.stdout)
    );
    assert!(
        out.stderr.is_empty(),
        "exec prints nothing: {:?}",
        text(&out.stderr)
    );

    // Criterion 2b: the child got the value through its environment…
    let seen = std::fs::read_to_string(&out_path).expect("the child ran and wrote its report");
    let (value, token) = seen.split_once('|').expect("two fields");
    assert_eq!(
        value, GRANTED_VALUE,
        "the child sees the value in $DB_PASSWORD"
    );
    // …and did NOT inherit the session token (it would let it read every
    // other granted secret on its own).
    assert_eq!(token, "absent", "the child must not inherit PO_VAULT_TOKEN");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_without_a_grant_runs_nothing() {
    let base = fake_vault().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let marker = dir.path().join("child-ran");

    let args: Vec<String> = vec![
        "exec".into(),
        "-e".into(),
        "X=other-secret".into(),
        "--".into(),
        "/bin/sh".into(),
        "-c".into(),
        format!("touch '{}'", marker.display()),
    ];
    let out = run_secret(&base, Some(SESSION_TOKEN), &args).await;

    assert_eq!(out.status.code(), Some(1));
    assert!(
        !Path::new(&marker).exists(),
        "the command must not run when a secret is refused"
    );
    let err = text(&out.stderr);
    assert!(err.contains("not granted to this session"), "{err:?}");
    assert!(out.stdout.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_rejects_a_malformed_mapping_before_contacting_the_vault() {
    let base = fake_vault().await;
    let args: Vec<String> = vec![
        "exec".into(),
        "-e".into(),
        "no-equals-sign".into(),
        "--".into(),
        "/bin/true".into(),
    ];
    let out = run_secret(&base, Some(SESSION_TOKEN), &args).await;

    assert_ne!(out.status.code(), Some(0));
    assert!(out.stdout.is_empty());
    assert!(!text(&out.stderr).is_empty(), "a reason is given");
}
