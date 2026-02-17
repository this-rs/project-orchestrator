<p align="center">
  <img src="dist/logo-512.png" alt="Project Orchestrator" width="128" />
</p>

<h1 align="center">Project Orchestrator</h1>

<p align="center">
  <strong>Coordinate AI coding agents with a shared knowledge graph.</strong>
</p>

<p align="center">
  <a href="#desktop-app"><img src="https://img.shields.io/badge/macOS-Download-000?logo=apple&logoColor=white" alt="macOS"></a>
  <a href="#desktop-app"><img src="https://img.shields.io/badge/Windows-Download-0078D4?logo=windows&logoColor=white" alt="Windows"></a>
  <a href="#desktop-app"><img src="https://img.shields.io/badge/Linux-Download-FCC624?logo=linux&logoColor=black" alt="Linux"></a>
</p>

<p align="center">
  <a href="https://github.com/this-rs/project-orchestrator/actions/workflows/ci.yml"><img src="https://github.com/this-rs/project-orchestrator/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/this-rs/project-orchestrator"><img src="https://codecov.io/gh/this-rs/project-orchestrator/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust"></a>
  <a href="https://github.com/this-rs/project-orchestrator/releases/latest"><img src="https://img.shields.io/github/v/release/this-rs/project-orchestrator?label=release" alt="Latest Release"></a>
</p>

Project Orchestrator gives your AI agents a shared brain. Instead of each agent starting from scratch, they share code understanding, plans, decisions, and progress through a central knowledge base.

---

## Features

- **Shared Knowledge Base** — Code structure stored in Neo4j graph database, accessible to all agents
- **Semantic Code Search** — Find code by meaning, not just keywords, powered by Meilisearch
- **Plan & Task Management** — Structured workflows with dependencies, steps, and progress tracking
- **Multi-Language Parsing** — Tree-sitter support for Rust, TypeScript, Python, Go, and 8 more languages
- **Multi-Project Workspaces** — Group related projects with shared context, contracts, and milestones
- **MCP Integration** — 130 tools available for Claude Code, OpenAI Agents, and Cursor
- **Auto-Sync** — File watcher keeps the knowledge base updated as you code
- **Authentication** — Google OAuth2, OIDC, and Password login with deny-by-default security
- **Chat WebSocket** — Real-time conversational AI via Claude integration
- **Event System** — Live CRUD notifications via WebSocket
- **NATS Integration** — Inter-process event sync for multi-instance deployments

---

## Installation

### Desktop App

Download the desktop app for your platform:

| Platform | Download | Type |
|----------|----------|------|
| **macOS** (Apple Silicon) | [Download .dmg](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.1_aarch64.dmg) | M1/M2/M3/M4 |
| **macOS** (Intel) | [Download .dmg](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.1_x64.dmg) | Intel Mac |
| **Windows** (64-bit) | [Download .exe](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.1_x64-setup.exe) | Installer |
| **Windows** (64-bit MSI) | [Download .msi](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.1_x64_en-US.msi) | MSI |
| **Linux** (64-bit) | [Download .AppImage](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.1_amd64.AppImage) | Universal |
| **Linux** (Debian/Ubuntu) | [Download .deb](https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator_0.0.1-1_amd64.deb) | apt/dpkg |
| **Linux** (Fedora/RHEL) | [Download .rpm](https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator-0.0.1-1.x86_64.rpm) | dnf/rpm |

> All releases are available on the [Releases page](https://github.com/this-rs/project-orchestrator/releases/latest).

---

### Homebrew (macOS / Linux)

```bash
brew install this-rs/tap/project-orchestrator
```

This installs `orchestrator`, `orch` (CLI shorthand), and `mcp_server`.

---

### Shell Script (macOS / Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.sh | sh
```

Options:

```bash
# Install a specific version
curl -fsSL https://…/install.sh | sh -s -- --version 0.0.1

# Install without the embedded frontend (lighter)
curl -fsSL https://…/install.sh | sh -s -- --no-frontend

# Custom install directory
curl -fsSL https://…/install.sh | sh -s -- --install-dir /usr/local/bin
```

---

### PowerShell (Windows)

```powershell
irm https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.ps1 | iex
```

---

### npm / npx

```bash
# Run directly without installing
npx @anthropic/project-orchestrator

# Or install globally
npm install -g @anthropic/project-orchestrator
```

> *Coming soon — npm package not yet published.*

---

### Docker

```bash
# Full (with embedded frontend)
docker pull ghcr.io/this-rs/project-orchestrator:latest

# API-only (lighter, no frontend)
docker pull ghcr.io/this-rs/project-orchestrator:latest-api
```

Or use Docker Compose with all services (Neo4j, Meilisearch, NATS):

```bash
git clone https://github.com/this-rs/project-orchestrator.git
cd project-orchestrator
docker compose up -d
```

---

### Debian / Ubuntu (apt)

```bash
# Download and install the .deb package
curl -LO https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator_0.0.1-1_amd64.deb
sudo dpkg -i project-orchestrator_0.0.1-1_amd64.deb

# Start the service
sudo systemctl enable --now project-orchestrator
```

---

### Fedora / RHEL (rpm)

```bash
# Download and install the .rpm package
curl -LO https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator-0.0.1-1.x86_64.rpm
sudo rpm -i project-orchestrator-0.0.1-1.x86_64.rpm
```

---

### Build from Source

```bash
git clone https://github.com/this-rs/project-orchestrator.git
cd project-orchestrator
cargo build --release

# Binaries in target/release/:
#   orchestrator  — main server
#   orch          — CLI shorthand
#   mcp_server    — MCP server for AI tools
```

---

## Quick Start

### 1. Start the backend services

The server requires **Neo4j** and **Meilisearch**. The easiest way is Docker Compose:

```bash
docker compose up -d
```

Then start the orchestrator:

```bash
orchestrator serve
```

### 2. Configure your AI tool

Add to your MCP configuration (e.g., `~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "mcp_server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "orchestrator123",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me",
        "NATS_URL": "nats://localhost:4222"
      }
    }
  }
}
```

> **Note:** `NATS_URL` enables real-time event sync between the MCP server and other instances (desktop app, other agents). Without it, CRUD events from MCP tools won't propagate to the rest of the system. If some env vars are not forwarded by your AI tool, the MCP server also reads `config.yaml` as a fallback (see [Configuration](#configuration)).

### 3. Create and sync your first project

```bash
# Your AI agent can now use MCP tools to:
# - create_project: Register your codebase
# - sync_project: Parse and index your code
# - create_plan: Start a development plan
# - create_workspace: Group related projects
# - And 126 more tools...
```

That's it! Your AI agents now have shared context.

---

## Integrations

| Platform | Status | Documentation |
|----------|--------|---------------|
| **Claude Code** | Full Support | [Setup Guide](docs/integrations/claude-code.md) |
| **OpenAI Agents** | Full Support | [Setup Guide](docs/integrations/openai.md) |
| **Cursor** | Full Support | [Setup Guide](docs/integrations/cursor.md) |

---

## What Can Your Agents Do?

### Explore Code
```
"Find all functions that handle authentication"
"Show me what imports this file"
"What's the impact of changing UserService?"
```

### Manage Work
```
"Create a plan to add OAuth support"
"What's the next task I should work on?"
"Record that we chose JWT over sessions"
```

### Stay in Sync
```
"What decisions were made about caching?"
"Show me the project roadmap"
"What tasks are blocking the release?"
```

### Coordinate Multiple Projects
```
"Create a workspace for our microservices"
"Add the API contract shared by all services"
"What's the cross-project milestone progress?"
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/setup/installation.md) | Full setup instructions and configuration |
| [Getting Started](docs/guides/getting-started.md) | Step-by-step tutorial for new users |
| [API Reference](docs/api/reference.md) | Complete REST API documentation |
| [MCP Tools](docs/api/mcp-tools.md) | All 130 MCP tools with examples |
| [Workspaces](docs/guides/workspaces.md) | Multi-project coordination |
| [Multi-Agent Workflows](docs/guides/multi-agent-workflow.md) | Coordinating multiple agents |
| [Authentication](docs/guides/authentication.md) | JWT + OAuth/OIDC + Password auth setup |
| [Chat & WebSocket](docs/guides/chat-websocket.md) | Real-time chat and events |
| [Knowledge Notes](docs/guides/knowledge-notes.md) | Contextual knowledge capture |

---

## Configuration

The server uses a layered configuration system: **env vars > config.yaml > defaults**.

Copy and edit `config.yaml` at the project root:

```yaml
server:
  port: 8080                    # SERVER_PORT

neo4j:
  uri: "bolt://localhost:7687"  # NEO4J_URI
  user: "neo4j"                 # NEO4J_USER
  password: "orchestrator123"   # NEO4J_PASSWORD

meilisearch:
  url: "http://localhost:7700"              # MEILISEARCH_URL
  key: "orchestrator-meili-key-change-me"   # MEILISEARCH_KEY

nats:
  url: "nats://localhost:4222"  # NATS_URL — inter-process event sync

chat:
  default_model: "claude-opus-4-6"  # CHAT_DEFAULT_MODEL
```

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j Bolt connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `orchestrator123` |
| `MEILISEARCH_URL` | Meilisearch HTTP URL | `http://localhost:7700` |
| `MEILISEARCH_KEY` | Meilisearch API key | `orchestrator-meili-key-change-me` |
| `NATS_URL` | NATS server URL for event sync | *(optional)* |
| `CHAT_DEFAULT_MODEL` | Default Claude model for chat | `claude-opus-4-6` |
| `RUST_LOG` | Log level filter | `info` |

> **Why NATS?** The MCP server runs as a separate process (spawned by Claude Code). NATS is the pub/sub bridge that propagates CRUD events and chat messages between the MCP server, the HTTP backend, and the desktop app. Without NATS, each instance works in isolation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR AI AGENTS                          │
│           (Claude Code / OpenAI / Cursor)                   │
└──────────┬──────────────────┬───────────────────┬───────────┘
           │ MCP Protocol     │ WebSocket         │ REST API
           ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROJECT ORCHESTRATOR                        │
│                    (130 MCP Tools)                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │   Auth   │  │   Chat   │  │  Events  │  │   Config   │  │
│  │OIDC+Pass │  │  Claude  │  │ Live WS  │  │   YAML +   │  │
│  │  + JWT   │  │  Streams │  │ + NATS   │  │  env vars  │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
   ┌──────────────────────────┼──────────────────────────┐
   ▼                ▼                  ▼                  ▼
┌──────────┐  ┌──────────┐     ┌──────────┐     ┌──────────────┐
│  NEO4J   │  │MEILISEARCH│    │  NATS    │     │ TREE-SITTER  │
│          │  │          │     │          │     │              │
│• Code    │  │• Code    │     │• Event   │     │• 12 languages│
│  graph   │  │  search  │     │  sync    │     │• AST parsing │
│• Plans   │  │• Decisions│    │• Chat    │     │• Symbols     │
│• Decisions│ │          │     │  relay   │     │              │
└──────────┘  └──────────┘     └──────────┘     └──────────────┘
```

---

## Supported Languages

| Language | Extensions |
|----------|------------|
| Rust | `.rs` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx` |
| Python | `.py` |
| Go | `.go` |
| Java | `.java` |
| C/C++ | `.c`, `.h`, `.cpp`, `.cc`, `.hpp` |
| Ruby | `.rb` |
| PHP | `.php` |
| Kotlin | `.kt`, `.kts` |
| Swift | `.swift` |
| Bash | `.sh`, `.bash` |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Give your AI agents a shared brain.</i>
</p>
