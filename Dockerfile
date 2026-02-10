# =============================================================================
# Stage 1: Build the frontend (React SPA)
# =============================================================================
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Accept frontend source path as build arg (default: ./frontend)
# In docker-compose, this should point to the frontend directory in the build context.
# If no frontend source is available, this stage produces an empty dist/.
ARG FRONTEND_SRC=./frontend

# Copy package files first for dependency caching
COPY ${FRONTEND_SRC}/package.json ${FRONTEND_SRC}/package-lock.json* ./

# Install dependencies
RUN npm ci --ignore-scripts 2>/dev/null || echo "No frontend dependencies found, skipping"

# Copy the rest of the frontend source
COPY ${FRONTEND_SRC}/ ./

# Build the frontend (produces dist/)
RUN if [ -f "package.json" ] && grep -q '"build"' package.json; then \
        npm run build; \
    else \
        echo "No frontend build script found, creating empty dist/"; \
        mkdir -p dist; \
    fi

# =============================================================================
# Stage 2: Build the backend (Rust)
# =============================================================================
FROM rustlang/rust:nightly-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock* ./

# Create dummy source files to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/cli.rs && \
    echo "pub mod neo4j; pub mod meilisearch; pub mod parser; pub mod plan; pub mod orchestrator; pub mod api;" > src/lib.rs && \
    mkdir -p src/neo4j src/meilisearch src/parser src/plan src/orchestrator src/api && \
    echo "pub mod client; pub mod models;" > src/neo4j/mod.rs && \
    echo "" > src/neo4j/client.rs && echo "" > src/neo4j/models.rs && \
    echo "pub mod client; pub mod indexes;" > src/meilisearch/mod.rs && \
    echo "" > src/meilisearch/client.rs && echo "" > src/meilisearch/indexes.rs && \
    echo "pub mod languages;" > src/parser/mod.rs && \
    echo "" > src/parser/languages.rs && \
    echo "pub mod models; pub mod manager;" > src/plan/mod.rs && \
    echo "" > src/plan/models.rs && echo "" > src/plan/manager.rs && \
    echo "pub mod context; pub mod runner;" > src/orchestrator/mod.rs && \
    echo "" > src/orchestrator/context.rs && echo "" > src/orchestrator/runner.rs && \
    echo "pub mod routes; pub mod handlers;" > src/api/mod.rs && \
    echo "" > src/api/routes.rs && echo "" > src/api/handlers.rs

# Create empty dist/ so rust-embed doesn't fail during dependency build
RUN mkdir -p dist

# Build dependencies only
RUN cargo build --release 2>/dev/null || true

# Remove dummy source
RUN rm -rf src

# Copy actual source code
COPY src ./src
COPY queries ./queries

# Copy the frontend dist/ from stage 1 (for embedded-frontend feature or ServeDir)
COPY --from=frontend-builder /app/frontend/dist ./dist

# Touch source files to trigger rebuild
RUN find src -name "*.rs" -exec touch {} \;

# Build the actual application
RUN cargo build --release

# =============================================================================
# Stage 3: Runtime image
# =============================================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries
COPY --from=builder /app/target/release/orchestrator /app/orchestrator
COPY --from=builder /app/target/release/orch /app/orch

# Copy tree-sitter queries
COPY queries ./queries

# Copy frontend dist/ (for serve_frontend mode)
COPY --from=frontend-builder /app/frontend/dist ./dist

# Create data directory
RUN mkdir -p /data

ENV RUST_LOG=info
ENV WORKSPACE_PATH=/workspace

EXPOSE 8080

CMD ["./orchestrator", "serve"]
