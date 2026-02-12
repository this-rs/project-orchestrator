# Installation Guide

Complete setup instructions for Project Orchestrator.

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Docker | 20.10+ | Run Neo4j and Meilisearch |
| Docker Compose | 2.0+ | Orchestrate services |
| Rust | 1.75+ | Build from source (optional) |

### Required Ports

| Port | Service | Protocol |
|------|---------|----------|
| 7474 | Neo4j Browser | HTTP |
| 7687 | Neo4j Bolt | TCP |
| 7700 | Meilisearch | HTTP |
| 8080 | Orchestrator API | HTTP |
| 4222 | NATS | TCP |
| 8222 | NATS Monitoring | HTTP |

---

## Installation with Docker (Recommended)

### Step 1: Clone the repository

```bash
git clone https://github.com/this-rs/project-orchestrator.git
cd project-orchestrator
```

### Step 2: Start all services

```bash
docker compose up -d
```

This starts:
- **Neo4j** — Graph database for code structure and relationships
- **Meilisearch** — Search engine for code and decisions
- **NATS** — Message broker for inter-process event sync (optional, for multi-instance)
- **Orchestrator** — API server with 145 MCP tools

### Step 3: Verify the installation

```bash
# Check all services are running
docker compose ps

# Check API health
curl http://localhost:8080/health
```

### Step 4: Extract the MCP server binary

```bash
# For local MCP integration, extract the binary from the running container
docker cp orchestrator-server:/app/mcp_server ./mcp_server
chmod +x mcp_server

# Or build from source
cargo build --release --bin mcp_server
```

The MCP server binary is at `./mcp_server` (or `./target/release/mcp_server` if built from source).

---

## Installation from Source

### Step 1: Start backend services only

```bash
docker compose up -d neo4j meilisearch
```

### Step 2: Build the project

```bash
cargo build --release
```

### Step 3: Run the server

```bash
./target/release/orchestrator serve --port 8080
```

### Step 4: Build the MCP server

```bash
cargo build --release --bin mcp_server
```

---

## Quick Install (Pre-built Binary)

Download and install the latest pre-built binary:

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.sh | sh
```

Options:
- `--version <version>` — Install a specific version
- `--no-frontend` — Install without embedded frontend
- `--install-dir <path>` — Custom install directory (default: `~/.local/bin`)

### Homebrew (macOS)

```bash
brew install this-rs/tap/orchestrator
```

### Debian / Ubuntu

```bash
# Download the .deb package from the latest release
sudo dpkg -i orchestrator_*.deb
sudo systemctl enable --now project-orchestrator
```

### Auto-configure Claude Code

After installation, automatically register the MCP server in Claude Code:

```bash
orchestrator setup-claude
```

This detects your Claude Code installation and adds the MCP server configuration.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `orchestrator123` | Neo4j password |
| `MEILISEARCH_URL` | `http://localhost:7700` | Meilisearch URL |
| `MEILISEARCH_KEY` | `orchestrator-meili-key-change-me` | Meilisearch API key |
| `SERVER_PORT` | `8080` | HTTP API port |
| `WORKSPACE_PATH` | `.` | Default workspace for syncing |
| `RUST_LOG` | `info,project_orchestrator=debug` | Log level filter (see [env_logger syntax](https://docs.rs/env_logger)) |
| `NATS_URL` | _(none)_ | NATS server URL (e.g., `nats://localhost:4222`) |
| `SERVE_FRONTEND` | `false` | Serve embedded frontend (requires `embedded-frontend` feature) |
| `FRONTEND_PATH` | _(none)_ | Path to external frontend build directory |

### Production Configuration

For production, change the default credentials:

```bash
# .env file
NEO4J_PASSWORD=your-secure-password-here
MEILISEARCH_KEY=your-secure-api-key-here
```

Then update `docker-compose.yml` or pass environment variables:

```bash
docker compose --env-file .env up -d
```

> **Note:** Authentication (Google OAuth, JWT) is configured exclusively through
> `config.yaml`, not environment variables. See the [Configuration System](#configuration-system-configyaml)
> and [Authentication Setup](#authentication-setup) sections below.

---

## Configuration System (config.yaml)

Project Orchestrator uses a layered configuration system with the following priority (highest wins):

1. **Environment variables** -- override everything
2. **`config.yaml`** -- file-based configuration
3. **Built-in defaults** -- sensible fallbacks for development

### Quick Start

```bash
cp config.yaml.example config.yaml
# Edit config.yaml to match your environment
```

### Full Example

```yaml
server:
  port: 8080
  workspace_path: "."

neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "orchestrator123"

meilisearch:
  url: "http://localhost:7700"
  key: "orchestrator-meili-key-change-me"

nats:
  url: "nats://localhost:4222"  # optional — enables multi-instance sync

chat:
  default_model: "claude-opus-4-6"
  max_sessions: 10
  session_timeout_secs: 1800
  max_turns: 50
  prompt_builder_model: "claude-opus-4-6"

# Auth section — if absent, server denies ALL /api/* and /ws/* requests
auth:
  jwt_secret: "change-me-to-a-random-32-char-string!"
  jwt_expiry_secs: 28800

  # Root account (optional)
  root_account:
    email: "admin@example.com"
    password_hash: "$2b$12$..."

  # Google OAuth (optional)
  google_client_id: "YOUR_CLIENT_ID.apps.googleusercontent.com"
  google_client_secret: "YOUR_CLIENT_SECRET"
  google_redirect_uri: "http://localhost:3000/auth/callback"

  # Generic OIDC (optional)
  oidc:
    client_id: "your-oidc-client-id"
    client_secret: "your-oidc-secret"
    issuer_url: "https://auth.example.com/realms/main"
    redirect_uri: "http://localhost:3000/auth/oidc/callback"

  allow_registration: false
  allowed_email_domain: "example.com"
  frontend_url: "http://localhost:3000"
```

Each YAML key has a corresponding environment variable override (noted as comments
in `config.yaml.example`). For example, `neo4j.uri` is overridden by `NEO4J_URI`,
`chat.default_model` by `CHAT_DEFAULT_MODEL`, and so on.

---

## Authentication Setup

Authentication is **optional** but follows a **deny-by-default** security model:

- If `config.yaml` has **no `auth` section**, the server denies all requests to
  `/api/*` and `/ws/*` endpoints. The MCP server (stdio) is unaffected.
- If `config.yaml` has an `auth` section, the configured providers are enabled:
  - **Password login** — Root account and/or user registration
  - **Google OAuth** — If `google_client_id` is set
  - **Generic OIDC** — If `oidc` section is configured (Keycloak, Auth0, Azure AD, etc.)

For a detailed walkthrough, see the [Authentication Guide](../guides/authentication.md).

### Development Without Auth

For local development where you only use the MCP server (stdio transport),
no auth configuration is needed. The MCP binary communicates directly with
Neo4j and Meilisearch, bypassing the HTTP API entirely.

If you need the HTTP API without auth during development, you can add an
`auth` section to `config.yaml` with placeholder values and set
`jwt_secret` to any 32+ character string.

### Production With Auth

1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/apis/credentials):
   - Application type: **Web application**
   - Authorized redirect URI: `https://your-domain.com/auth/callback`
2. Fill in the `auth` section of `config.yaml` with the generated client ID
   and client secret.
3. Set a strong, random `jwt_secret` (minimum 32 characters).
4. Optionally restrict access to a specific email domain with `allowed_email_domain`.

For a detailed walkthrough, see the [Authentication Guide](../guides/authentication.md).

---

## Docker Images

Pre-built Docker images are published to GitHub Container Registry on each release:

```
ghcr.io/this-rs/project-orchestrator
```

### Image variants

| Variant | Tag examples | Description |
|---------|-------------|-------------|
| **Full** (recommended) | `:latest`, `:1.0.0`, `:1.0`, `:1` | Backend + embedded React frontend |
| **API-only** | `:latest-api`, `:1.0.0-api`, `:1.0-api`, `:1-api` | Backend only, no frontend |

### Choosing a tag

- **`:latest`** — always points to the newest full release (convenient, but may break on upgrades)
- **`:X.Y.Z`** — pinned to a specific version (recommended for production)
- **`:X.Y`** — receives patch updates automatically
- **`:X`** — receives minor and patch updates

Use `docker-compose.production.yml` with `ORCHESTRATOR_IMAGE_TAG` to set the version:

```bash
# Pin to a specific version
ORCHESTRATOR_IMAGE_TAG=1.0.0 docker compose -f docker-compose.production.yml up -d
```

---

## Docker Compose Configuration

The `docker-compose.yml` defines four services:

### Neo4j

```yaml
neo4j:
  image: neo4j:5.26.20-community
  ports:
    - "7474:7474"  # Browser UI
    - "7687:7687"  # Bolt protocol
  environment:
    - NEO4J_AUTH=neo4j/orchestrator123
    - NEO4J_PLUGINS=["apoc"]
```

Access the Neo4j Browser at http://localhost:7474

### Meilisearch

```yaml
meilisearch:
  image: getmeili/meilisearch:v1.34.2
  ports:
    - "7700:7700"
  environment:
    - MEILI_MASTER_KEY=orchestrator-meili-key-change-me
```

Access the Meilisearch dashboard at http://localhost:7700

### NATS

```yaml
nats:
  image: nats:2.11-alpine
  ports:
    - "4222:4222"  # Client connections
    - "8222:8222"  # HTTP monitoring
  command: ["--http_port", "8222", "--jetstream"]
  volumes:
    - nats_data:/data/jetstream
```

NATS is optional. It enables cross-instance event synchronization and distributed chat relay. If not running, the orchestrator operates in single-instance mode with local event broadcasting only.

### Orchestrator

```yaml
orchestrator:
  build: .
  ports:
    - "8080:8080"
  depends_on:
    neo4j:
      condition: service_healthy
    meilisearch:
      condition: service_healthy
```

### Network and Volumes

All three services communicate over a dedicated `orchestrator-net` bridge network.
Data is persisted in five named Docker volumes:

| Volume | Service | Content |
|--------|---------|---------|
| `neo4j_data` | Neo4j | Graph database files |
| `neo4j_logs` | Neo4j | Server logs |
| `meilisearch_data` | Meilisearch | Search indexes |
| `orchestrator_data` | Orchestrator | Application data |
| `nats_data` | NATS | JetStream storage |

To completely reset all data, run `docker compose down -v` (this removes all
volumes and is **destructive**).

---

## Verification

### Check Service Health

```bash
# All services status
docker compose ps

# API health check
curl http://localhost:8080/health
# Expected: {"status":"healthy"}

# Neo4j health
curl http://localhost:7474
# Expected: Neo4j Browser HTML

# Meilisearch health
curl http://localhost:7700/health
# Expected: {"status":"available"}
```

### Test MCP Server

```bash
# Test MCP server starts correctly
./target/release/mcp_server --help

# Test with debug logging
RUST_LOG=debug ./target/release/mcp_server
```

---

## Troubleshooting

### Neo4j won't start

**Error:** `Neo4j failed to start`

```bash
# Check logs
docker compose logs neo4j

# Common fix: Reset data volume
docker compose down -v
docker compose up -d
```

**Error:** `Address already in use: 7687`

```bash
# Find process using the port
lsof -i :7687
# Kill or stop the conflicting process
```

### Meilisearch connection refused

**Error:** `Connection refused to localhost:7700`

```bash
# Check if Meilisearch is running
docker compose ps meilisearch

# Check logs for errors
docker compose logs meilisearch

# Restart the service
docker compose restart meilisearch
```

### Orchestrator can't connect to Neo4j

**Error:** `Failed to connect to Neo4j`

```bash
# Ensure Neo4j is healthy first
docker compose ps neo4j

# Wait for Neo4j to be ready (can take 30s on first start)
docker compose logs -f neo4j

# Check connection from orchestrator container
docker compose exec orchestrator curl -I http://neo4j:7474
```

### MCP server not found by Claude Code

**Error:** `spawn mcp_server ENOENT`

```bash
# Use absolute path in mcp.json
"command": "/absolute/path/to/mcp_server"

# Verify binary exists and is executable
ls -la /path/to/mcp_server
chmod +x /path/to/mcp_server
```

### Permission denied on Docker volumes

**Error:** `Permission denied` when writing to volumes

```bash
# Fix permissions on host
sudo chown -R $USER:$USER ./workspace

# Or run with correct user in docker-compose.yml
user: "${UID}:${GID}"
```

### Out of memory errors

**Error:** `Neo4j out of heap space`

Increase memory limits in `docker-compose.yml`:

```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=2G
```

---

## Updating

### Update Docker images

```bash
docker compose pull
docker compose up -d
```

### Update from source

```bash
git pull
cargo build --release
cargo build --release --bin mcp_server
```

### Self-update (installed binary)

```bash
orchestrator update
```

This checks GitHub Releases for a newer version, downloads it, verifies the checksum, and replaces the binary atomically.

---

## Uninstalling

### Remove containers and volumes

```bash
# Stop and remove containers
docker compose down

# Remove data volumes (DESTRUCTIVE)
docker compose down -v
```

### Remove built binaries

```bash
rm -rf target/
```

---

## Next Steps

- [Configure your IDE integration](../integrations/claude-code.md)
- [Set up authentication](../guides/authentication.md)
- [Follow the Getting Started tutorial](../guides/getting-started.md)
- [Explore the API Reference](../api/reference.md)
