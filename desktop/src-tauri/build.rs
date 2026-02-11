use std::fs;
use std::path::Path;

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

    tauri_build::build();
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
