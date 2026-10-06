use std::fs;
use std::path::Path;

/// Every command of `generate_handler![]` in src/main.rs, by name.
///
/// Declaring them here gives each one a permission of its own (`allow-<command>`,
/// kebab-case) and puts them under the ACL. Without this manifest an application
/// command has no permission to grant, and the main window, which runs from the
/// embedded backend (`http://localhost:*`, a REMOTE origin for Tauri), is refused
/// every one of them: "Command check_docker not allowed by ACL".
///
/// A command added to the handler and not listed here is not callable at all;
/// tests/desktop_acl.rs fails the build of the repository when the two disagree and
/// checks which window is granted what.
const APP_COMMANDS: &[&str] = &[
    "get_server_port",
    "proceed_to_main",
    "check_health",
    "probe_services",
    "test_connection_detailed",
    "open_url",
    "pick_directory",
    "restart_app",
    "webview_log",
    "check_dependencies",
    "check_config_exists",
    "get_config_path",
    "generate_config",
    "read_config",
    "detect_claude_code",
    "detect_shell_path",
    "check_cli_status",
    "install_cli",
    "setup_claude_code",
    "verify_oidc_discovery",
    "check_embedding_model",
    "download_embedding_model",
    "test_embedding_endpoint",
    "check_docker",
    "open_docker_desktop",
    "start_docker_services",
    "check_services_health",
    "stop_docker_services",
    "get_service_logs",
    "test_connection",
    "check_update",
    "install_update",
    "enable_rounded_corners",
    "enable_modern_window_style",
    "reposition_traffic_lights",
];

fn main() {
    // ── Guard: ensure dist/ exists and contains index.html ──────────────
    let dist_dir = Path::new("../dist");
    let index_html = dist_dir.join("index.html");

    if !dist_dir.exists() || !index_html.exists() {
        panic!(
            "\n\n\
            ╔══════════════════════════════════════════════════════════════╗\n\
            ║  ERROR: desktop/dist/ is missing or incomplete!            ║\n\
            ║                                                            ║\n\
            ║  The frontend must be built before the Tauri app.          ║\n\
            ║  Run the full build script:                                ║\n\
            ║                                                            ║\n\
            ║    ./scripts/build-desktop.sh                              ║\n\
            ║                                                            ║\n\
            ║  Or build the frontend manually:                           ║\n\
            ║    cd ~/projects/project-orchestrator/frontend && npm run build  ║\n\
            ║    cp -r dist/ <project>/desktop/dist/                     ║\n\
            ╚══════════════════════════════════════════════════════════════╝\n\n"
        );
    }

    // ── Splash: dist/ must carry the CURRENT splash.html ─────────────────
    // The splash window loads "splash.html" from frontendDist (dist/), which Tauri embeds in the
    // binary. A copy made only when the frontend is rebuilt (build-desktop.sh --skip-front does not)
    // left an OLD splash inside new builds: fixes to splash.html never shipped.
    let splash_src = Path::new("splash.html");
    if splash_src.exists() {
        if let Err(e) = fs::copy(splash_src, dist_dir.join("splash.html")) {
            panic!("cannot copy splash.html into dist/: {e}");
        }
        println!("cargo:rerun-if-changed=splash.html");
    }

    // ── Freshness check: warn if dist/ is older than 1 hour ─────────────
    if let Ok(meta) = fs::metadata(&index_html) {
        if let Ok(modified) = meta.modified() {
            if let Ok(age) = std::time::SystemTime::now().duration_since(modified) {
                let hours = age.as_secs() / 3600;
                if hours >= 1 {
                    println!(
                        "cargo:warning=desktop/dist/index.html is {} hour(s) old — \
                         consider rebuilding the frontend if you made changes.",
                        hours
                    );
                }
            }
        }
    }

    // ── Watch dist/ for cache invalidation ──────────────────────────────
    println!("cargo:rerun-if-changed=../dist");
    visit_dirs(dist_dir);

    // Also watch the bundled mcp_server binary
    let mcp_bin = Path::new("../../target/release/mcp_server");
    if mcp_bin.exists() {
        println!("cargo:rerun-if-changed={}", mcp_bin.display());
    }

    let attributes = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(APP_COMMANDS));
    if let Err(error) = tauri_build::try_build(attributes) {
        panic!("tauri-build failed: {error:#}");
    }
}

/// Recursively emit rerun-if-changed for every file in a directory.
fn visit_dirs(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            println!("cargo:rerun-if-changed={}", path.display());
            if path.is_dir() {
                visit_dirs(&path);
            }
        }
    }
}
