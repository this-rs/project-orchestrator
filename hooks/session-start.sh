#!/usr/bin/env bash
#
# Project Orchestrator — SessionStart Hook for Claude Code
#
# Injects session context (active skills, current plan/task, critical notes)
# at the beginning of each Claude Code session.
#
# Protocol:
#   stdin  -> JSON { hookEventName, session }
#   stdout -> JSON { hookSpecificOutput: { additionalContext: "..." } }
#
# Constraints:
#   - Pure bash control flow (no jq dependency)
#   - Node.js for JSON parsing/formatting (always available with Claude Code)
#   - curl with 2s connect timeout (never block the agent)
#   - Exit 0 on ANY error (graceful failure)
#   - stdout reserved for hook output, stderr for debug only
#
# Config: reads .po-config from cwd or parent directories
#   { "project_id": "uuid", "port": 6600 }

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DEBUG="${PO_HOOK_DEBUG:-0}"

# ============================================================================
# Logging (stderr only, never stdout)
# ============================================================================

debug() {
    if [ "$DEBUG" = "1" ]; then
        echo "[PO-SESSION] $1" >&2
    fi
}

# ============================================================================
# Global error handler — NEVER block the agent
# ============================================================================

trap 'debug "Error on line $LINENO"; exit 0' ERR

# ============================================================================
# Read stdin (consume it so the hook doesn't hang)
# ============================================================================

# Read stdin with a timeout to prevent hanging
INPUT=""
if read -t 2 -r -d '' INPUT 2>/dev/null; then
    :
fi

debug "Input received (${#INPUT} bytes)"

# ============================================================================
# Config discovery — walk up from cwd to find .po-config
# ============================================================================

find_config() {
    local dir="${1:-$(pwd)}"
    local i=0
    while [ "$i" -lt 20 ]; do
        if [ -f "$dir/.po-config" ]; then
            cat "$dir/.po-config"
            return 0
        fi
        local parent
        parent=$(dirname "$dir")
        [ "$parent" = "$dir" ] && break
        dir="$parent"
        i=$((i + 1))
    done
    return 1
}

CONFIG=$(find_config) || { debug "No .po-config found"; exit 0; }

# ============================================================================
# Extract config values — Node.js for reliable JSON parsing
# ============================================================================

if ! command -v node >/dev/null 2>&1; then
    debug "Node.js not available"
    exit 0
fi

# Extract project_id and port from config JSON
CONFIG_PARSED=$(node -e "
    try {
        const c = JSON.parse(process.argv[1]);
        process.stdout.write((c.project_id || '') + ' ' + (c.port || '6600'));
    } catch(e) {
        process.stdout.write(' 6600');
    }
" "$CONFIG" 2>/dev/null) || { debug "Failed to parse config"; exit 0; }

PROJECT_ID="${CONFIG_PARSED%% *}"
PORT="${CONFIG_PARSED##* }"

if [ -z "$PROJECT_ID" ]; then
    debug "No project_id in .po-config"
    exit 0
fi

# Validate PORT is a number (prevent injection in curl URL)
case "$PORT" in
    ''|*[!0-9]*) debug "Invalid port: $PORT"; exit 0 ;;
esac

debug "Project: $PROJECT_ID, Port: $PORT"

# ============================================================================
# Call the session-context endpoint
# ============================================================================

RESPONSE=$(curl -s --connect-timeout 2 --max-time 3 \
    "http://127.0.0.1:${PORT}/api/hooks/session-context?project_id=${PROJECT_ID}" \
    2>/dev/null) || { debug "Failed to reach server"; exit 0; }

if [ -z "$RESPONSE" ]; then
    debug "Empty response from server"
    exit 0
fi

debug "Response received (${#RESPONSE} bytes)"

# ============================================================================
# Format context and output hook response — Node.js for JSON handling
# ============================================================================

node -e '
    "use strict";

    function relativeTime(isoDate) {
        if (!isoDate) return null;
        const diff = Date.now() - new Date(isoDate).getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return "just now";
        if (mins < 60) return mins + "m ago";
        const hours = Math.floor(mins / 60);
        if (hours < 24) return hours + "h ago";
        const days = Math.floor(hours / 24);
        return days + "d ago";
    }

    const resp = JSON.parse(process.argv[1]);
    if (!resp || resp.error) process.exit(0);

    let ctx = "";

    // Active Skills
    if (resp.active_skills && resp.active_skills.length > 0) {
        ctx += "## Neural Skills (" + resp.active_skills.length + " active)\n";
        for (const s of resp.active_skills) {
            const energy = Math.round(s.energy * 100);
            ctx += "- **" + s.name + "** (" + s.note_count + " notes, " + energy + "% energy)";
            if (s.activation_count > 0) {
                const ago = relativeTime(s.last_activated);
                ctx += " — " + s.activation_count + " activations";
                if (ago) ctx += ", last " + ago;
            }
            ctx += "\n";
            if (s.description) ctx += "  " + s.description + "\n";
        }
        ctx += "\n";
    }

    // Current Plan & Task
    if (resp.current_plan) {
        const p = resp.current_plan;
        ctx += "## Current Plan\n";
        ctx += "**" + p.title + "** (" + p.status + ")";
        if (p.progress) ctx += " — " + p.progress;
        ctx += "\n";

        if (resp.current_task) {
            const t = resp.current_task;
            ctx += "Current task: **" + t.title + "** (" + t.status + ")\n";
        }
        ctx += "\n";
    }

    // Critical Notes
    if (resp.critical_notes && resp.critical_notes.length > 0) {
        ctx += "## Critical Notes\n";
        for (const n of resp.critical_notes) {
            const tag = n.note_type ? "[" + n.note_type + "] " : "";
            let content = n.content || "";
            const firstLine = content.split("\n")[0];
            content = firstLine.length > 200
                ? firstLine.substring(0, 200) + "..."
                : firstLine;
            ctx += "- " + tag + content + "\n";
        }
    }

    if (ctx === "") process.exit(0);

    const output = {
        hookSpecificOutput: {
            additionalContext: ctx
        }
    };

    process.stdout.write(JSON.stringify(output) + "\n");
' "$RESPONSE" 2>/dev/null || { debug "Failed to format context"; exit 0; }

debug "Session context injected"
exit 0
