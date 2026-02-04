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

---

## Installation with Docker (Recommended)

### Step 1: Clone the repository

```bash
git clone https://github.com/your-org/project-orchestrator.git
cd project-orchestrator
```

### Step 2: Start all services

```bash
docker compose up -d
```

This starts:
- **Neo4j** — Graph database for code structure and relationships
- **Meilisearch** — Search engine for code and decisions
- **Orchestrator** — API server with 62 MCP tools

### Step 3: Verify the installation

```bash
# Check all services are running
docker compose ps

# Check API health
curl http://localhost:8080/health
```

### Step 4: Build the MCP server binary

```bash
# For local MCP integration, build the binary
docker compose exec orchestrator cat /app/mcp_server > mcp_server
chmod +x mcp_server

# Or build from source
cargo build --release --bin mcp_server
```

The MCP server binary is at `./target/release/mcp_server`.

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
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

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

---

## Docker Compose Configuration

The `docker-compose.yml` defines three services:

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
- [Follow the Getting Started tutorial](../guides/getting-started.md)
- [Explore the API Reference](../api/reference.md)
