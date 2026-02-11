#!/bin/bash
# Build script for the Project Orchestrator desktop app.
# Rebuilds: frontend → dist copy → backend binary → Tauri bundle.
#
# Usage:
#   ./scripts/build-desktop.sh              # Full build (front + back + Tauri)
#   ./scripts/build-desktop.sh --skip-front # Skip frontend rebuild (with freshness check)
#   ./scripts/build-desktop.sh --skip-back  # Skip backend binary rebuild

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$HOME/projects/project-orchestrator/frontend"
DESKTOP_DIR="$PROJECT_DIR/desktop"
DESKTOP_DIST="$DESKTOP_DIR/dist"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }
log_ok()   { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_err()  { echo -e "${RED}✗${NC} $1"; }

SKIP_FRONT=false
SKIP_BACK=false

for arg in "$@"; do
    case $arg in
        --skip-front) SKIP_FRONT=true ;;
        --skip-back)  SKIP_BACK=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-front] [--skip-back]"
            echo ""
            echo "  --skip-front  Skip frontend rebuild (with freshness check)"
            echo "  --skip-back   Skip mcp_server binary rebuild"
            exit 0
            ;;
    esac
done

cd "$PROJECT_DIR"

# ── Freshness helper ─────────────────────────────────────────────────────
# Returns 0 if frontend sources are newer than desktop/dist/index.html.
is_dist_stale() {
    local dist_index="$DESKTOP_DIST/index.html"
    if [ ! -f "$dist_index" ]; then
        return 0  # No dist = definitely stale
    fi
    if [ ! -d "$FRONTEND_DIR/src" ]; then
        return 1  # Can't check — assume fresh
    fi
    # Find any frontend source file newer than dist/index.html
    local newer
    newer=$(find "$FRONTEND_DIR/src" "$FRONTEND_DIR/index.html" "$FRONTEND_DIR/vite.config.ts" \
        -newer "$dist_index" 2>/dev/null | head -1)
    [ -n "$newer" ]
}

# ── Step 1: Build frontend ─────────────────────────────────────────────────
if [ "$SKIP_FRONT" = false ]; then
    log_step "Building frontend"

    if [ ! -d "$FRONTEND_DIR" ]; then
        log_err "Frontend directory not found: $FRONTEND_DIR"
        exit 1
    fi

    (cd "$FRONTEND_DIR" && npm run build)
    log_ok "Frontend built ($(ls "$FRONTEND_DIR/dist/assets/" | wc -l | tr -d ' ') assets)"

    # ── Step 2: Copy dist to desktop ────────────────────────────────────────
    log_step "Copying dist → desktop/dist"
    rm -rf "$DESKTOP_DIST"
    cp -r "$FRONTEND_DIR/dist" "$DESKTOP_DIST"

    # Also copy splash.html into dist if it exists alongside src-tauri
    if [ -f "$DESKTOP_DIR/src-tauri/splash.html" ]; then
        cp "$DESKTOP_DIR/src-tauri/splash.html" "$DESKTOP_DIST/splash.html"
    fi

    log_ok "dist synced to $DESKTOP_DIST"
else
    log_warn "Skipping frontend build (--skip-front)"
    if [ ! -d "$DESKTOP_DIST" ]; then
        log_err "desktop/dist/ does not exist — run without --skip-front first"
        exit 1
    fi
    # Freshness check: warn if sources are newer than dist
    if is_dist_stale; then
        echo ""
        log_warn "⚡ Frontend sources are NEWER than desktop/dist/"
        log_warn "   Your app may bundle stale assets!"
        echo -e "   Run without ${CYAN}--skip-front${NC} to rebuild, or press Enter to continue anyway."
        read -r -p "   Continue with stale dist? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY]) log_warn "Continuing with potentially stale dist..." ;;
            *)
                log_step "Rebuilding frontend (stale dist detected)"
                (cd "$FRONTEND_DIR" && npm run build)
                rm -rf "$DESKTOP_DIST"
                cp -r "$FRONTEND_DIR/dist" "$DESKTOP_DIST"
                if [ -f "$DESKTOP_DIR/src-tauri/splash.html" ]; then
                    cp "$DESKTOP_DIR/src-tauri/splash.html" "$DESKTOP_DIST/splash.html"
                fi
                log_ok "Frontend rebuilt and dist synced"
                ;;
        esac
    else
        log_ok "desktop/dist/ appears up-to-date"
    fi
fi

# ── Step 3: Build backend binary (mcp_server) ──────────────────────────────
if [ "$SKIP_BACK" = false ]; then
    log_step "Building mcp_server (release)"
    cargo build --release --bin mcp_server
    log_ok "mcp_server built: target/release/mcp_server"
else
    log_warn "Skipping backend build (--skip-back)"
    if [ ! -f "$PROJECT_DIR/target/release/mcp_server" ]; then
        log_err "target/release/mcp_server does not exist — cannot skip backend build"
        exit 1
    fi
fi

# ── Step 4: Build Tauri desktop app ─────────────────────────────────────────
log_step "Building Tauri desktop app"
cd "$DESKTOP_DIR/src-tauri"
cargo tauri build 2>&1

log_ok "Desktop app built successfully!"
echo ""

# Show output paths
APP_PATH="$DESKTOP_DIR/src-tauri/target/release/bundle/macos/Project Orchestrator.app"
DMG_PATH="$DESKTOP_DIR/src-tauri/target/release/bundle/dmg"

if [ -d "$APP_PATH" ]; then
    echo -e "${GREEN}App:${NC} $APP_PATH"
fi
if [ -d "$DMG_PATH" ]; then
    DMG_FILE=$(ls "$DMG_PATH"/*.dmg 2>/dev/null | head -1)
    if [ -n "$DMG_FILE" ]; then
        echo -e "${GREEN}DMG:${NC} $DMG_FILE"
    fi
fi
