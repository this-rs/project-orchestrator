#!/usr/bin/env bash
#
# Project Orchestrator — Hook Installer for Claude Code
#
# Installs PO hooks (pre-tool-use.cjs, session-start.sh) into the user's
# Claude Code hooks directory and generates a .po-config for project binding.
#
# Usage:
#   ./hooks/install.sh --project-id <uuid> [--port <port>] [--target-dir <dir>]
#   ./hooks/install.sh --uninstall [--target-dir <dir>]
#
# Options:
#   --project-id   UUID of the PO project to bind (required for install)
#   --port         PO server port (default: 6600)
#   --target-dir   Project directory for .po-config (default: cwd)
#   --uninstall    Remove hooks and .po-config
#   --help         Show this help
#
# The installer is idempotent: running it multiple times is safe.
# It will update existing hooks and .po-config without duplication.
#

set -euo pipefail

# ============================================================================
# Constants
# ============================================================================

HOOKS_DIR="$HOME/.claude/hooks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="1.0.0"

# Hook files to install
HOOK_FILES=(
    "pre-tool-use.cjs"
    "session-start.sh"
)

# ============================================================================
# Colors (if terminal supports it)
# ============================================================================

if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    GREEN=$(tput setaf 2)
    RED=$(tput setaf 1)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    BOLD=$(tput bold)
    RESET=$(tput sgr0)
else
    GREEN="" RED="" YELLOW="" BLUE="" BOLD="" RESET=""
fi

# ============================================================================
# Logging
# ============================================================================

info()  { echo "${GREEN}[✓]${RESET} $1"; }
warn()  { echo "${YELLOW}[!]${RESET} $1"; }
error() { echo "${RED}[✗]${RESET} $1" >&2; }
step()  { echo "${BLUE}[→]${RESET} $1"; }

# ============================================================================
# Usage
# ============================================================================

usage() {
    cat <<'USAGE'
Project Orchestrator — Hook Installer

Usage:
  ./hooks/install.sh --project-id <uuid> [--port <port>] [--target-dir <dir>]
  ./hooks/install.sh --uninstall [--target-dir <dir>]

Options:
  --project-id   UUID of the PO project (required for install)
  --port         PO server port (default: 6600)
  --target-dir   Directory for .po-config (default: current directory)
  --uninstall    Remove hooks and .po-config
  --help         Show this help

Examples:
  # Install hooks for a project
  ./hooks/install.sh --project-id 00333b5f-2d0a-4467-9c98-155e55d2b7e5

  # Install with custom port and target
  ./hooks/install.sh --project-id abc123 --port 7700 --target-dir /my/project

  # Uninstall
  ./hooks/install.sh --uninstall
USAGE
}

# ============================================================================
# Argument parsing
# ============================================================================

PROJECT_ID=""
PORT="6600"
TARGET_DIR="$(pwd)"
UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Prerequisite checks
# ============================================================================

check_prerequisites() {
    local ok=true

    # Node.js >= 18
    if command -v node >/dev/null 2>&1; then
        local node_version
        node_version=$(node -e "process.stdout.write(String(process.versions.node.split('.')[0]))" 2>/dev/null)
        if [ "${node_version:-0}" -lt 18 ]; then
            error "Node.js >= 18 required (found v${node_version})"
            ok=false
        fi
    else
        error "Node.js not found — required for pre-tool-use.cjs"
        ok=false
    fi

    # curl
    if ! command -v curl >/dev/null 2>&1; then
        error "curl not found — required for hook communication"
        ok=false
    fi

    # Verify hook source files exist
    for hook_file in "${HOOK_FILES[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$hook_file" ]; then
            error "Hook source not found: $SCRIPT_DIR/$hook_file"
            ok=false
        fi
    done

    if [ "$ok" = false ]; then
        echo ""
        error "Prerequisites check failed. Fix the issues above and retry."
        exit 1
    fi
}

# ============================================================================
# Uninstall
# ============================================================================

do_uninstall() {
    echo "${BOLD}Project Orchestrator — Uninstalling hooks${RESET}"
    echo ""

    local removed=0

    # Remove hook files
    for hook_file in "${HOOK_FILES[@]}"; do
        local target="$HOOKS_DIR/$hook_file"
        if [ -f "$target" ] || [ -L "$target" ]; then
            rm -f "$target"
            info "Removed $target"
            removed=$((removed + 1))
        fi
    done

    # Remove .po-config
    local config_file="$TARGET_DIR/.po-config"
    if [ -f "$config_file" ]; then
        rm -f "$config_file"
        info "Removed $config_file"
        removed=$((removed + 1))
    fi

    if [ "$removed" -eq 0 ]; then
        warn "Nothing to uninstall — no PO hooks found"
    else
        echo ""
        info "Uninstalled $removed file(s). PO hooks are no longer active."
    fi
}

# ============================================================================
# Install hooks
# ============================================================================

install_hooks() {
    echo "${BOLD}Project Orchestrator — Installing hooks${RESET}"
    echo ""

    # Create hooks directory
    step "Creating hooks directory: $HOOKS_DIR"
    mkdir -p "$HOOKS_DIR"
    info "Hooks directory ready"

    # Copy hook files (overwrite existing = idempotent)
    for hook_file in "${HOOK_FILES[@]}"; do
        local source="$SCRIPT_DIR/$hook_file"
        local target="$HOOKS_DIR/$hook_file"

        step "Installing $hook_file"

        # Copy the file (not symlink — avoids issues with relative paths in hooks)
        cp -f "$source" "$target"
        chmod +x "$target"

        info "Installed $target"
    done
}

# ============================================================================
# Generate .po-config
# ============================================================================

generate_config() {
    local config_file="$TARGET_DIR/.po-config"

    step "Generating .po-config in $TARGET_DIR"

    # Use Node.js for clean JSON generation
    if command -v node >/dev/null 2>&1; then
        node -e '
            const config = {
                project_id: process.argv[1],
                port: parseInt(process.argv[2], 10),
                installed_at: new Date().toISOString(),
                hooks_version: process.argv[3],
                hooks: JSON.parse(process.argv[4])
            };
            process.stdout.write(JSON.stringify(config, null, 2) + "\n");
        ' "$PROJECT_ID" "$PORT" "$VERSION" '["pre-tool-use.cjs","session-start.sh"]' > "$config_file"
    else
        # Fallback: plain text (still valid JSON)
        cat > "$config_file" <<CONF
{
  "project_id": "$PROJECT_ID",
  "port": $PORT,
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hooks_version": "$VERSION",
  "hooks": ["pre-tool-use.cjs", "session-start.sh"]
}
CONF
    fi

    info "Config written to $config_file"
}

# ============================================================================
# Health check
# ============================================================================

health_check() {
    step "Checking PO server connectivity (127.0.0.1:$PORT)"

    local response
    if response=$(curl -s --connect-timeout 2 --max-time 3 "http://127.0.0.1:${PORT}/api/hooks/health" 2>/dev/null); then
        if echo "$response" | grep -q "ok" 2>/dev/null; then
            info "PO server is healthy"
            return 0
        fi
    fi

    warn "PO server not reachable at 127.0.0.1:$PORT"
    warn "Hooks installed but will be inactive until the server starts"
    warn "Start the server with: cargo run --release -- serve --port $PORT"
    return 0  # Non-fatal — hooks still install correctly
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    echo ""
    echo "${BOLD}Installation complete!${RESET}"
    echo ""
    echo "  Hooks installed in:  $HOOKS_DIR"
    echo "  Project config:      $TARGET_DIR/.po-config"
    echo "  Server:              127.0.0.1:$PORT"
    echo "  Project ID:          $PROJECT_ID"
    echo ""
    echo "${BOLD}Next steps:${RESET}"
    echo ""
    echo "  1. Ensure PO server is running on port $PORT"
    echo "  2. Start a new Claude Code session in $TARGET_DIR"
    echo "  3. The hooks will automatically inject Neural Skills context"
    echo ""
    echo "  To uninstall:"
    echo "    $0 --uninstall --target-dir $TARGET_DIR"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    if [ "$UNINSTALL" = true ]; then
        do_uninstall
        exit 0
    fi

    # Validate required args
    if [ -z "$PROJECT_ID" ]; then
        error "Missing required option: --project-id"
        echo ""
        usage
        exit 1
    fi

    # Validate project ID looks like a UUID (8-4-4-4-12 hex format)
    if ! echo "$PROJECT_ID" | grep -qE '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'; then
        warn "Project ID doesn't look like a UUID: $PROJECT_ID"
        warn "Proceeding anyway — verify this is correct"
    fi

    # Validate target directory exists
    if [ ! -d "$TARGET_DIR" ]; then
        error "Target directory does not exist: $TARGET_DIR"
        exit 1
    fi

    check_prerequisites
    install_hooks
    generate_config
    health_check
    print_summary
}

main
