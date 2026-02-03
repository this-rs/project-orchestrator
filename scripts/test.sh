#!/bin/bash
# Test script for project-orchestrator
# Usage: ./scripts/test.sh [unit|integration|api|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1

    log_info "Waiting for $name at $url..."

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_info "$name is ready!"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    log_error "$name did not become ready in time"
    return 1
}

# Start backends
start_backends() {
    log_info "Starting backends with docker-compose..."
    docker compose up -d neo4j meilisearch

    # Wait for services
    wait_for_service "http://localhost:7700/health" "Meilisearch" 60
    wait_for_service "http://localhost:7474" "Neo4j" 60

    log_info "Backends are ready!"
}

# Stop backends
stop_backends() {
    log_info "Stopping backends..."
    docker compose down
}

# Run unit tests (no external services needed)
run_unit_tests() {
    log_info "Running parser tests (unit tests)..."
    cargo test --test parser_tests -- --nocapture
}

# Run integration tests (requires Neo4j and Meilisearch)
run_integration_tests() {
    log_info "Running integration tests..."
    cargo test --test integration_tests -- --nocapture
}

# Run API tests (requires full stack)
run_api_tests() {
    log_info "Building and starting orchestrator..."
    cargo build --release

    # Start the server in background
    ./target/release/orchestrator serve &
    SERVER_PID=$!

    # Wait for server to be ready
    wait_for_service "http://localhost:8080/health" "Orchestrator API" 30

    log_info "Running API tests..."
    cargo test --test api_tests -- --nocapture
    TEST_RESULT=$?

    # Stop the server
    log_info "Stopping orchestrator..."
    kill $SERVER_PID 2>/dev/null || true

    return $TEST_RESULT
}

# Quick smoke test
smoke_test() {
    log_info "Running smoke test..."

    # Test Meilisearch
    if curl -sf "http://localhost:7700/health" > /dev/null; then
        log_info "✓ Meilisearch is running"
    else
        log_error "✗ Meilisearch is not running"
        return 1
    fi

    # Test Neo4j
    if curl -sf "http://localhost:7474" > /dev/null; then
        log_info "✓ Neo4j is running"
    else
        log_error "✗ Neo4j is not running"
        return 1
    fi

    # Build and test basic functionality
    log_info "Building project..."
    cargo build --release

    # Start server temporarily
    ./target/release/orchestrator serve &
    SERVER_PID=$!
    sleep 3

    # Test health endpoint
    if curl -sf "http://localhost:8080/health" | grep -q "ok"; then
        log_info "✓ API health check passed"
    else
        log_error "✗ API health check failed"
        kill $SERVER_PID 2>/dev/null
        return 1
    fi

    # Test creating a plan
    PLAN_RESULT=$(curl -sf -X POST "http://localhost:8080/api/plans" \
        -H "Content-Type: application/json" \
        -d '{"title":"Smoke Test Plan","description":"Testing basic functionality","priority":1}')

    if echo "$PLAN_RESULT" | grep -q "id"; then
        log_info "✓ Plan creation passed"
        PLAN_ID=$(echo "$PLAN_RESULT" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

        # Test adding a task
        TASK_RESULT=$(curl -sf -X POST "http://localhost:8080/api/plans/$PLAN_ID/tasks" \
            -H "Content-Type: application/json" \
            -d '{"description":"Smoke test task"}')

        if echo "$TASK_RESULT" | grep -q "id"; then
            log_info "✓ Task creation passed"
        else
            log_error "✗ Task creation failed"
        fi
    else
        log_error "✗ Plan creation failed"
    fi

    # Stop server
    kill $SERVER_PID 2>/dev/null

    log_info "Smoke test completed!"
}

# Main
case "${1:-all}" in
    unit)
        run_unit_tests
        ;;
    integration)
        start_backends
        run_integration_tests
        ;;
    api)
        start_backends
        run_api_tests
        ;;
    smoke)
        start_backends
        smoke_test
        ;;
    backends)
        start_backends
        log_info "Backends are running. Press Ctrl+C to stop."
        wait
        ;;
    stop)
        stop_backends
        ;;
    all)
        log_info "Running all tests..."

        # Unit tests first (no deps)
        run_unit_tests

        # Start backends
        start_backends

        # Integration tests
        run_integration_tests

        # API tests
        run_api_tests

        log_info "All tests completed!"
        ;;
    *)
        echo "Usage: $0 [unit|integration|api|smoke|backends|stop|all]"
        echo ""
        echo "Commands:"
        echo "  unit        Run unit tests (no external services needed)"
        echo "  integration Run integration tests (requires Neo4j + Meilisearch)"
        echo "  api         Run API tests (requires full stack)"
        echo "  smoke       Run quick smoke test"
        echo "  backends    Start backends and keep running"
        echo "  stop        Stop all backends"
        echo "  all         Run all tests"
        exit 1
        ;;
esac
