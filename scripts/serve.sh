#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8081}"
LOG="/tmp/knowloop.log"
BINARY="./target/release/knowloop"

# Kill any existing instance
pkill -f "knowloop serve" 2>/dev/null && sleep 1 || true

# Restart backends to clear stale Neo4j connections (prevents tokio deadlock)
echo "Restarting backends..."
docker compose restart neo4j meilisearch 2>/dev/null || true
sleep 5

# Wait for Neo4j to be ready
for i in $(seq 1 15); do
    if curl -sf http://127.0.0.1:7474 -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 1
done

# Start the server
CHAT_DEFAULT_MODEL=claude-opus-4-6 "$BINARY" serve --port "$PORT" &> "$LOG" &
SERVER_PID=$!
echo "Server starting (PID $SERVER_PID, port $PORT, log $LOG)"

# Wait for the server to be ready
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:"$PORT"/api/version -o /dev/null 2>/dev/null; then
        echo "Server ready on http://127.0.0.1:$PORT"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Server did not start within 30s. Check $LOG"
        exit 1
    fi
    sleep 1
done

# Expose via Tailscale Serve (HTTPS proxy)
if tailscale status &>/dev/null; then
    tailscale serve --bg "$PORT" 2>/dev/null
    TS_HOSTNAME=$(tailscale status --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" 2>/dev/null || echo "unknown")
    echo "Tailscale: https://$TS_HOSTNAME"
else
    echo "Tailscale not running — skipping remote access setup"
fi
