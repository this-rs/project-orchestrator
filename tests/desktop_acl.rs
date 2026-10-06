//! The permission levels of the desktop app, checked against what each window calls.
//!
//! The main window runs from the embedded backend (`http://localhost:*`), which Tauri
//! treats as a REMOTE origin: an application command is callable from it only if a
//! capability grants `allow-<command>`. When none did, the setup wizard died with
//! "Command check_docker not allowed by ACL". These tests keep four things in line:
//!
//! * the commands registered in `src/main.rs` (`generate_handler!`),
//! * the manifest in `build.rs` (`APP_COMMANDS`), which gives each one a permission,
//! * the capabilities (`capabilities/*.json`) that grant them per window,
//! * what each window really calls: `splash.html` and the web app.
//!
//! What is NOT proved here: that Tauri accepts the files (the desktop build does, it
//! validates every capability at build time) or that a window really reaches a
//! command (that needs a webview). What is: nobody is granted a command that does not
//! exist, nothing is granted by accident, and a new command forces a decision.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use regex::Regex;
use serde_json::Value;

fn tauri_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("desktop/src-tauri")
}

fn read(relative: &str) -> String {
    let path = tauri_dir().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn set<I: IntoIterator<Item = String>>(items: I) -> BTreeSet<String> {
    items.into_iter().collect()
}

/// Commands registered in `generate_handler![ … ]`, last path segment.
fn registered_commands() -> BTreeSet<String> {
    let main = read("src/main.rs");
    let start = main
        .find("generate_handler![")
        .expect("generate_handler! in main.rs")
        + "generate_handler![".len();
    let body = &main[start
        ..main[start..]
            .find("])")
            .map(|i| start + i)
            .expect("end of handler")];
    set(body
        .lines()
        .map(|l| {
            l.split("//")
                .next()
                .unwrap_or("")
                .trim()
                .trim_end_matches(',')
        })
        .filter(|l| !l.is_empty())
        .map(|l| l.rsplit("::").next().unwrap().to_string()))
}

/// `APP_COMMANDS` of `build.rs`.
fn manifest_commands() -> BTreeSet<String> {
    let build = read("build.rs");
    let start = build
        .find("const APP_COMMANDS")
        .expect("APP_COMMANDS in build.rs");
    let body = &build[start..start + build[start..].find("];").expect("end of APP_COMMANDS")];
    let re = Regex::new(r#""([a-z_]+)""#).unwrap();
    set(re.captures_iter(body).map(|c| c[1].to_string()))
}

struct Capability {
    windows: Vec<String>,
    remote: bool,
    /// Application permissions (`allow-<kebab>`), as command names.
    app_commands: BTreeSet<String>,
    all: Vec<String>,
}

fn capability(file: &str) -> Capability {
    let json: Value = serde_json::from_str(&read(&format!("capabilities/{file}")))
        .unwrap_or_else(|e| panic!("{file} is not valid JSON: {e}"));
    let permissions: Vec<String> = json["permissions"]
        .as_array()
        .unwrap_or_else(|| panic!("{file}: permissions"))
        .iter()
        .map(|p| p.as_str().expect("permission is a string").to_string())
        .collect();
    Capability {
        windows: json["windows"]
            .as_array()
            .unwrap_or_else(|| panic!("{file}: windows"))
            .iter()
            .map(|w| w.as_str().unwrap().to_string())
            .collect(),
        remote: json.get("remote").is_some(),
        app_commands: set(permissions
            .iter()
            .filter(|p| p.starts_with("allow-") || p.starts_with("deny-"))
            .map(|p| p[p.find('-').unwrap() + 1..].replace('-', "_"))),
        all: permissions,
    }
}

/// Commands `splash.html` invokes.
fn splash_calls() -> BTreeSet<String> {
    let html = read("splash.html");
    let re = Regex::new(r#"invoke\(\s*['"]([a-z_]+)['"]"#).unwrap();
    set(re.captures_iter(&html).map(|c| c[1].to_string()))
}

/// What the web app calls from the main window: the commands its code invokes
/// (`grep` of the frontend repository, 2026-10-06), plus `webview_log`, which the
/// console bridge of the initialization script in `main.rs` calls in that window.
///
/// The web app lives in another repository, so this list is kept by hand: a command
/// the web app starts to call must be added here AND to `app-main.json` and
/// `remote.json`, which is exactly the decision this test forces.
const WEB_APP_CALLS: &[&str] = &[
    "check_cli_status",
    "check_docker",
    "check_embedding_model",
    "check_update",
    "detect_shell_path",
    "download_embedding_model",
    "enable_modern_window_style",
    "generate_config",
    "get_server_port",
    "install_cli",
    "install_update",
    "open_docker_desktop",
    "open_url",
    "pick_directory",
    "read_config",
    "restart_app",
    "setup_claude_code",
    "test_connection",
    "test_connection_detailed",
    "test_embedding_endpoint",
    "verify_oidc_discovery",
    "webview_log",
];

/// Registered commands NO window may call today. Granting one is a decision: it is
/// made by moving the name out of this list, in the same change as the grant.
const NOT_EXPOSED: &[&str] = &[
    "check_config_exists",
    "detect_claude_code",
    "enable_rounded_corners",
    "get_config_path",
    "get_service_logs",
    "reposition_traffic_lights",
    "stop_docker_services",
];

fn owned(list: &[&str]) -> BTreeSet<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn the_registered_commands_and_the_manifest_are_the_same_list() {
    let registered = registered_commands();
    let manifest = manifest_commands();
    assert!(registered.len() > 20, "the parser found {registered:?}");
    let missing: Vec<_> = registered.difference(&manifest).collect();
    let stale: Vec<_> = manifest.difference(&registered).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "registered but absent from APP_COMMANDS in build.rs (uncallable): {missing:?}; \
         listed in APP_COMMANDS but not registered: {stale:?}"
    );
}

#[test]
fn no_capability_grants_a_command_that_does_not_exist() {
    let manifest = manifest_commands();
    for file in [
        "app-splash.json",
        "app-main.json",
        "remote.json",
        "default.json",
    ] {
        let unknown: Vec<_> = capability(file)
            .app_commands
            .difference(&manifest)
            .cloned()
            .collect();
        assert!(
            unknown.is_empty(),
            "{file} grants unknown commands: {unknown:?}"
        );
    }
}

#[test]
fn the_splash_is_granted_exactly_what_splash_html_calls() {
    let splash = capability("app-splash.json");
    assert_eq!(splash.windows, ["splashscreen"], "only the splash window");
    assert!(!splash.remote, "the splash is a local page");
    assert_eq!(
        splash.app_commands,
        splash_calls(),
        "app-splash.json must list exactly the invoke() calls of splash.html"
    );
}

#[test]
fn the_main_window_is_granted_exactly_what_the_web_app_calls() {
    let main = capability("app-main.json");
    assert_eq!(main.windows, ["main"]);
    assert!(!main.remote);
    assert_eq!(
        main.app_commands,
        owned(WEB_APP_CALLS),
        "app-main.json differs from the commands the web app calls (WEB_APP_CALLS)"
    );
}

#[test]
fn the_embedded_backend_origin_gets_the_same_commands_as_the_local_one() {
    // The main window is navigated to http://localhost:{port}/ after startup; it is
    // from there that the wizard calls check_docker. Same grant on both origins.
    let remote = capability("remote.json");
    assert!(remote.remote, "remote.json is the remote-origin capability");
    assert_eq!(remote.windows, ["main"]);
    assert_eq!(
        remote.app_commands,
        capability("app-main.json").app_commands
    );
}

#[test]
fn the_remote_origin_stays_limited_to_this_machine() {
    let json: Value = serde_json::from_str(&read("capabilities/remote.json")).unwrap();
    let urls = json["remote"]["urls"].as_array().expect("remote.urls");
    assert!(!urls.is_empty());
    for url in urls {
        let url = url.as_str().unwrap();
        assert!(
            url.starts_with("http://localhost") || url.starts_with("http://127.0.0.1"),
            "{url} is not a loopback origin: it would receive native commands"
        );
    }
}

#[test]
fn the_console_bridge_of_the_main_window_may_log() {
    let main_rs = read("src/main.rs");
    assert!(
        main_rs.contains("ti.invoke('webview_log'"),
        "the initialization script no longer calls webview_log: update WEB_APP_CALLS"
    );
    for file in ["app-main.json", "remote.json", "app-splash.json"] {
        assert!(
            capability(file).app_commands.contains("webview_log"),
            "{file} must allow webview_log (console bridge)"
        );
    }
}

#[test]
fn a_command_nobody_calls_is_granted_to_nobody_and_says_so() {
    let manifest = manifest_commands();
    let granted: BTreeSet<String> = ["app-splash.json", "app-main.json", "remote.json"]
        .iter()
        .flat_map(|f| capability(f).app_commands)
        .collect();
    let ungranted: BTreeSet<String> = manifest.difference(&granted).cloned().collect();
    assert_eq!(
        ungranted,
        owned(NOT_EXPOSED),
        "the commands no window can call changed: a new command must be granted or added \
         to NOT_EXPOSED on purpose (and a granted one removed from it)"
    );
}

#[test]
fn the_core_permissions_of_default_json_were_not_widened() {
    // default.json (main + splash, core and plugins) must not carry application commands:
    // those are per window, in app-splash.json and app-main.json.
    let default = capability("default.json");
    assert!(
        default.app_commands.is_empty(),
        "default.json grants application commands to both windows: {:?}",
        default.app_commands
    );
    assert!(default.all.iter().any(|p| p == "core:default"));
}
