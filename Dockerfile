# =============================================================================
# Stage 1: Prepare the frontend assets
# =============================================================================
# This stage handles two scenarios:
#   A) Pre-built dist/ provided (CI: frontend built by separate job, placed in frontend/dist/)
#   B) Full frontend source provided (local: frontend repo cloned into ./frontend)
#
# In both cases, the output is /app/frontend/dist/ with the built assets.
# =============================================================================
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Accept frontend source path as build arg (default: ./frontend)
ARG FRONTEND_SRC=./frontend

# Copy everything from the frontend source.
# Use a wildcard so COPY doesn't fail if only dist/ is present.
COPY ${FRONTEND_SRC}/ ./

# Build only if package.json exists (full source scenario).
# If only dist/ was copied (CI scenario), skip the build entirely.
RUN if [ -f "package.json" ] && grep -q '"build"' package.json; then \
        echo "Frontend source detected — installing dependencies and building..."; \
        npm ci --ignore-scripts && npm run build; \
    elif [ -d "dist" ]; then \
        echo "Pre-built frontend dist/ detected — skipping build"; \
    else \
        echo "No frontend source or dist/ found — creating empty dist/"; \
        mkdir -p dist; \
    fi

# =============================================================================
# Stage 2: Build the backend (Rust)
# =============================================================================
FROM rust:1.88-bookworm AS builder

WORKDIR /app

# Install build dependencies (no libssl-dev — project uses rustls, not OpenSSL)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests for dependency caching (Cargo.lock required for reproducible builds)
COPY Cargo.toml Cargo.lock ./

# Create dummy source files to build dependencies.
# This must match the actual module layout so the dep-cache layer is valid.
# Module directories (11):
#   api, auth, chat, events, mcp, meilisearch, neo4j, notes, orchestrator, parser, plan
# Single files: setup_claude.rs, update.rs
# Binary: bin/mcp_server.rs
RUN mkdir -p src/bin \
             src/api src/auth src/chat src/events src/mcp \
             src/meilisearch src/neo4j src/notes src/orchestrator \
             src/parser src/plan && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/cli.rs && \
    echo "fn main() {}" > src/bin/mcp_server.rs && \
    echo "pub mod api; pub mod auth; pub mod chat; pub mod events; pub mod mcp; pub mod meilisearch; pub mod neo4j; pub mod notes; pub mod orchestrator; pub mod parser; pub mod plan; pub mod setup_claude; pub mod update;" > src/lib.rs && \
    echo "" > src/setup_claude.rs && \
    echo "" > src/update.rs && \
    for dir in api auth chat events mcp meilisearch neo4j notes orchestrator parser plan; do \
        echo "" > "src/$dir/mod.rs"; \
    done

# Create empty dist/ so rust-embed doesn't fail during dependency build
RUN mkdir -p dist

# Build dependencies only
RUN cargo build --release 2>/dev/null || true

# Remove dummy source
RUN rm -rf src

# Copy actual source code
COPY src ./src

# Copy tree-sitter queries if they exist (optional — may not be present)
COPY querie[s] ./queries/

# Copy the frontend dist/ from stage 1 (for embedded-frontend feature or ServeDir)
COPY --from=frontend-builder /app/frontend/dist ./dist

# Touch source files to trigger rebuild
RUN find src -name "*.rs" -exec touch {} \;

# Build the actual application
RUN cargo build --release

# =============================================================================
# Stage 3: Runtime image
# =============================================================================
FROM debian:bookworm-slim AS runtime

# OCI labels
LABEL org.opencontainers.image.source="https://github.com/this-rs/project-orchestrator"
LABEL org.opencontainers.image.description="AI agent orchestrator with Neo4j knowledge graph, Meilisearch, and Tree-sitter"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.vendor="OpenClaw"

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all 3 binaries
COPY --from=builder /app/target/release/orchestrator /app/orchestrator
COPY --from=builder /app/target/release/orch /app/orch
COPY --from=builder /app/target/release/mcp_server /app/mcp_server

# Copy tree-sitter queries if they exist (optional)
COPY querie[s] ./queries/

# Build arg: include frontend dist/ or not (default: true)
ARG INCLUDE_FRONTEND=true

# Copy frontend dist/ conditionally
# When INCLUDE_FRONTEND=false, dist/ will be empty (created below)
COPY --from=frontend-builder /app/frontend/dist /tmp/frontend-dist

RUN if [ "$INCLUDE_FRONTEND" = "true" ]; then \
        cp -r /tmp/frontend-dist ./dist && \
        echo "Frontend included in image"; \
    else \
        mkdir -p ./dist && \
        echo "API-only image (no frontend)"; \
    fi && \
    rm -rf /tmp/frontend-dist

# Create data directory
RUN mkdir -p /data

ENV RUST_LOG=info
ENV WORKSPACE_PATH=/workspace
ENV SERVE_FRONTEND=${INCLUDE_FRONTEND}
ENV FRONTEND_PATH=/app/dist

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./orchestrator", "serve"]
