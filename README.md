# Project Orchestrator

**Coordinate AI coding agents with a shared knowledge graph.**

Project Orchestrator gives your AI agents a shared brain. Instead of each agent starting from scratch, they share code understanding, plans, decisions, and progress through a central knowledge base.

[![CI](https://github.com/this-rs/project-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/this-rs/project-orchestrator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/this-rs/project-orchestrator/branch/main/graph/badge.svg)](https://codecov.io/gh/this-rs/project-orchestrator)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)

---

## Features

- **Shared Knowledge Base** — Code structure stored in Neo4j graph database, accessible to all agents
- **Semantic Code Search** — Find code by meaning, not just keywords, powered by Meilisearch
- **Plan & Task Management** — Structured workflows with dependencies, steps, and progress tracking
- **Multi-Language Parsing** — Tree-sitter support for Rust, TypeScript, Python, Go, and 8 more languages
- **Multi-Project Workspaces** — Group related projects with shared context, contracts, and milestones
- **MCP Integration** — 145 tools available for Claude Code, OpenAI Agents, and Cursor
- **Auto-Sync** — File watcher keeps the knowledge base updated as you code
- **Authentication** — Google OAuth2, OIDC, and Password login with deny-by-default security
- **Chat WebSocket** — Real-time conversational AI via Claude integration
- **Event System** — Live CRUD notifications via WebSocket
- **NATS Integration** — Inter-process event sync for multi-instance deployments
- **YAML Configuration** — Hierarchical config with env var overrides

---

## Quick Start

### 1. Start the backend services

```bash
git clone https://github.com/this-rs/project-orchestrator.git
cd project-orchestrator
docker compose up -d
```

**Or install the pre-built binary:**

```bash
curl -fsSL https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.sh | sh
```

### 2. Configure your AI tool

Add to your MCP configuration (e.g., `~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
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
# - And 141 more tools...
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
| [MCP Tools](docs/api/mcp-tools.md) | All 145 MCP tools with examples |
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
│                    (145 MCP Tools)                           │
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
