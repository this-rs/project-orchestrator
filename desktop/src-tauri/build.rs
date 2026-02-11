use std::fs;
use std::path::Path;

fn main() {
    // Watch the frontend dist directory so Cargo recompiles the binary
    // (which embeds assets via tauri::generate_context!) when dist/ changes.
    let dist_dir = Path::new("../dist");
    if dist_dir.exists() {
        println!("cargo:rerun-if-changed=../dist");
        visit_dirs(dist_dir);
    }

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
