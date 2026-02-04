# Project Orchestrator

**Coordinate AI coding agents with a shared knowledge graph.**

Project Orchestrator gives your AI agents a shared brain. Instead of each agent starting from scratch, they share code understanding, plans, decisions, and progress through a central knowledge base.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)

---

## Features

- **Shared Knowledge Base** — Code structure stored in Neo4j graph database, accessible to all agents
- **Semantic Code Search** — Find code by meaning, not just keywords, powered by Meilisearch
- **Plan & Task Management** — Structured workflows with dependencies, steps, and progress tracking
- **Multi-Language Parsing** — Tree-sitter support for Rust, TypeScript, Python, Go, and 8 more languages
- **MCP Integration** — 62 tools available for Claude Code, OpenAI Agents, and Cursor
- **Auto-Sync** — File watcher keeps the knowledge base updated as you code

---

## Quick Start

### 1. Start the backend services

```bash
git clone https://github.com/your-org/project-orchestrator.git
cd project-orchestrator
docker compose up -d
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
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me"
      }
    }
  }
}
```

### 3. Create and sync your first project

```bash
# Your AI agent can now use MCP tools to:
# - create_project: Register your codebase
# - sync_project: Parse and index your code
# - create_plan: Start a development plan
# - And 59 more tools...
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

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/setup/installation.md) | Full setup instructions and configuration |
| [Getting Started](docs/guides/getting-started.md) | Step-by-step tutorial for new users |
| [API Reference](docs/api/reference.md) | Complete REST API documentation |
| [MCP Tools](docs/api/mcp-tools.md) | All 62 MCP tools with examples |
| [Multi-Agent Workflows](docs/guides/multi-agent-workflow.md) | Coordinating multiple agents |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR AI AGENTS                          │
│           (Claude Code / OpenAI / Cursor)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │ MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROJECT ORCHESTRATOR                        │
│                    (62 MCP Tools)                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    NEO4J      │     │  MEILISEARCH  │     │  TREE-SITTER  │
│               │     │               │     │               │
│ • Code graph  │     │ • Code search │     │ • 12 languages│
│ • Plans       │     │ • Decisions   │     │ • AST parsing │
│ • Decisions   │     │               │     │ • Symbols     │
└───────────────┘     └───────────────┘     └───────────────┘
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
