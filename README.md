<p align="center">
  <img src="dist/logo-512.png" alt="Project Orchestrator" width="128" />
</p>

<h1 align="center">Project Orchestrator</h1>

<p align="center">
  <strong>Coordinate AI coding agents with a shared knowledge graph.</strong>
</p>

<p align="center">
  <a href="https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_aarch64.dmg"><img src="https://img.shields.io/badge/Download_for_macOS-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Download for macOS" height="40"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_x64-setup.exe"><img src="https://img.shields.io/badge/Download_for_Windows-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Download for Windows" height="40"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_amd64.AppImage"><img src="https://img.shields.io/badge/Download_for_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Download for Linux" height="40"></a>
</p>

<p align="center">
  <a href="#desktop-app">All download options (Intel Mac, .msi, .deb, .rpm...)</a>
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
- **Protocol FSM Engine** — Define and run hierarchical finite state machines for repeatable workflows
- **RFC Lifecycle** — Propose, review, accept, and track architectural decisions through a formal protocol
- **Knowledge Fabric** — Bio-inspired neural network connecting notes, decisions, and code via synapses
- **Multi-Language Parsing** — Tree-sitter support for Rust, TypeScript, Python, Go, and 12 more languages
- **Multi-Project Workspaces** — Group related projects with shared context, contracts, and milestones
- **MCP Integration** — 22 mega-tools available for Claude Code, OpenAI Agents, and Cursor
- **Autonomous Runner** — Execute plans automatically with parallel wave dispatch and agent personas
- **Auto-Sync** — File watcher keeps the knowledge base updated as you code
- **Authentication** — Google OAuth2, OIDC, and Password login with deny-by-default security
- **Chat WebSocket** — Real-time conversational AI via Claude integration with smart context injection
- **Event System** — Live CRUD notifications via WebSocket + streaming activation events
- **NATS Integration** — Inter-process event sync for multi-instance deployments
- **Skill Federation** — Export, publish, and import neural skills across projects and instances

---

## Installation

### Desktop App

Download the desktop app for your platform:

| Platform | Download | Type |
|----------|----------|------|
| **macOS** (Apple Silicon) | [Download .dmg](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_aarch64.dmg) | M1/M2/M3/M4 |
| **macOS** (Intel) | [Download .dmg](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_x64.dmg) | Intel Mac |
| **Windows** (64-bit) | [Download .exe](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_x64-setup.exe) | Installer |
| **Windows** (64-bit MSI) | [Download .msi](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_x64_en-US.msi) | MSI |
| **Linux** (64-bit) | [Download .AppImage](https://github.com/this-rs/project-orchestrator/releases/latest/download/Project.Orchestrator_0.0.10_amd64.AppImage) | Universal |
| **Linux** (Debian/Ubuntu) | [Download .deb](https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator_0.0.10-1_amd64.deb) | apt/dpkg |
| **Linux** (Fedora/RHEL) | [Download .rpm](https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator-0.0.10-1.x86_64.rpm) | dnf/rpm |

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
curl -fsSL https://…/install.sh | sh -s -- --version 0.0.10

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
curl -LO https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator_0.0.10-1_amd64.deb
sudo dpkg -i project-orchestrator_0.0.10-1_amd64.deb

# Start the service
sudo systemctl enable --now project-orchestrator
```

---

### Fedora / RHEL (rpm)

```bash
# Download and install the .rpm package
curl -LO https://github.com/this-rs/project-orchestrator/releases/latest/download/project-orchestrator-0.0.10-1.x86_64.rpm
sudo rpm -i project-orchestrator-0.0.10-1.x86_64.rpm
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
# - And 15 more mega-tools...
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

## Protocols & FSM Engine

Protocols are reusable **finite state machines** stored in the knowledge graph. They model repeatable workflows — from code review to release pipelines — and can be composed hierarchically.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Protocol** | FSM definition with states, transitions, and optional relevance vector |
| **ProtocolState** | A node in the FSM (`start`, `intermediate`, `terminal`, or `generator`) |
| **ProtocolTransition** | An edge between states, fired by a `trigger` with an optional `guard` condition |
| **ProtocolRun** | A live execution instance tracking current state, visit history, and status |

### Creating a Protocol (one-shot compose)

The `compose` action creates a complete protocol — skill, states, transitions, and note bindings — in a single call:

```
protocol(action: "compose", project_id: "...",
  name: "code-review",
  category: "business",
  states: [
    { name: "analyze",  state_type: "start",        description: "Analyze code changes" },
    { name: "review",   state_type: "intermediate",  description: "Review findings" },
    { name: "approved", state_type: "terminal",       description: "Review passed" },
    { name: "rejected", state_type: "terminal",       description: "Review failed" }
  ],
  transitions: [
    { from_state: "analyze", to_state: "review",   trigger: "analysis_complete" },
    { from_state: "review",  to_state: "approved", trigger: "approve" },
    { from_state: "review",  to_state: "rejected", trigger: "reject" }
  ],
  relevance_vector: { phase: 0.75, structure: 0.6, domain: 0.5, resource: 0.5, lifecycle: 0.5 }
)
```

### Running a Protocol

```
# Start a run (enters the start state automatically)
protocol(action: "start_run", protocol_id: "...")

# Fire transitions to advance through the FSM
protocol(action: "transition", run_id: "...", trigger: "analysis_complete")
protocol(action: "transition", run_id: "...", trigger: "approve")

# Check current state
protocol(action: "get_run", run_id: "...")
# → { status: "completed", current_state: "approved", states_visited: [...] }
```

### Hierarchical Protocols

A state can delegate to a **child protocol** (macro-state). The parent pauses until the child completes:

```
states: [
  { name: "plan",      state_type: "start" },
  { name: "implement", state_type: "intermediate", sub_protocol_id: "<review-protocol-id>" },
  { name: "done",      state_type: "terminal" }
],
transitions: [
  { from_state: "plan",      to_state: "implement", trigger: "plan_approved" },
  { from_state: "implement", to_state: "done",      trigger: "child_completed" }
]
```

Completion strategies: `all_complete` (default), `any_complete`, `manual`.
Failure strategies: `abort` (default), `skip`, `retry(N)`.

### Context-Aware Routing

Each protocol has a 5-dimension relevance vector. Use `route` to find the best protocol for your current context:

```
protocol(action: "route", project_id: "...", plan_id: "...")
# → wave-execution: 95% (phase match + high structure)
# → code-review:    60% (phase mismatch)
```

### Example: Safe Modification Protocol

A protocol that ensures every non-trivial code change goes through proper
impact analysis, topology checks, and documentation — using the full
knowledge fabric:

```
protocol(action: "compose", project_id: "...",
  name: "safe-modification",
  category: "business",
  states: [
    { name: "gather-context",  state_type: "start",
      description: "Load notes, decisions, propagated context, active RFCs" },
    { name: "analyze-impact",  state_type: "intermediate",
      description: "Run analyze_impact + get_file_co_changers on target files" },
    { name: "check-topology",  state_type: "intermediate",
      description: "Verify new imports don't violate architectural rules" },
    { name: "check-risks",     state_type: "intermediate",
      description: "Evaluate risk via get_node_importance — PageRank, betweenness, churn" },
    { name: "implement",       state_type: "intermediate",
      description: "Make changes with full awareness of impact and constraints" },
    { name: "verify",          state_type: "intermediate",
      description: "Run tests + re-check topology for new violations" },
    { name: "document",        state_type: "intermediate",
      description: "Create notes, record decisions, link AFFECTS to changed files" },
    { name: "done",            state_type: "terminal",
      description: "Changes are safe, tested, and documented" }
  ],
  transitions: [
    { from_state: "gather-context",  to_state: "analyze-impact",  trigger: "context_loaded" },
    { from_state: "analyze-impact",  to_state: "check-topology",  trigger: "impact_assessed" },
    { from_state: "check-topology",  to_state: "check-risks",     trigger: "topology_ok" },
    { from_state: "check-topology",  to_state: "gather-context",  trigger: "topology_violation",
      guard: "New imports violate architectural rules — rethink the approach" },
    { from_state: "check-risks",     to_state: "implement",       trigger: "risks_acceptable" },
    { from_state: "check-risks",     to_state: "gather-context",  trigger: "high_risk",
      guard: "Critical risk score — need a different approach" },
    { from_state: "implement",       to_state: "verify",          trigger: "changes_made" },
    { from_state: "verify",          to_state: "document",        trigger: "verification_passed" },
    { from_state: "verify",          to_state: "implement",       trigger: "verification_failed" },
    { from_state: "document",        to_state: "done",            trigger: "documented" }
  ],
  relevance_vector: { phase: 0.5, structure: 0.7, domain: 0.5, resource: 0.5, lifecycle: 0.5 }
)
```

### Example: Knowledge Maintenance Protocol

A system protocol that runs weekly to keep the knowledge fabric healthy:

```
protocol(action: "compose", project_id: "...",
  name: "knowledge-maintenance",
  category: "system",
  states: [
    { name: "audit",           state_type: "start",
      description: "audit_gaps — find orphan notes, decisions without AFFECTS, unlinked commits" },
    { name: "health-check",    state_type: "intermediate",
      description: "get_health — hotspots, risks, homeostasis, neural metrics" },
    { name: "decay-synapses",  state_type: "intermediate",
      description: "Gentle decay (0.03) to prune dead connections — NEVER > 0.1 per pass" },
    { name: "update-scores",   state_type: "intermediate",
      description: "Recalculate staleness, energy, and fabric fusion scores" },
    { name: "review-stale",    state_type: "intermediate",
      description: "Find stale notes — confirm, invalidate, or supersede" },
    { name: "report",          state_type: "terminal",
      description: "persist_health_report — saves as note with delta vs. previous report" }
  ],
  transitions: [
    { from_state: "audit",           to_state: "health-check",    trigger: "audit_done" },
    { from_state: "health-check",    to_state: "decay-synapses",  trigger: "health_assessed" },
    { from_state: "decay-synapses",  to_state: "update-scores",   trigger: "decay_applied" },
    { from_state: "update-scores",   to_state: "review-stale",    trigger: "scores_updated" },
    { from_state: "review-stale",    to_state: "report",          trigger: "review_done" }
  ],
  relevance_vector: { phase: 0.75, structure: 0.3, domain: 0.5, resource: 0.3, lifecycle: 0.8 }
)
```

> **Full protocol guide**: See [Protocols — Building a Safe, Self-Aware Setup](docs/guides/protocols.md)
> for detailed step-by-step instructions on what the agent should do at each state,
> how to set up hierarchical protocols, auto-triggers, and a recommended production suite.

---

## RFCs (Requests for Comments)

RFCs are special knowledge notes (`note_type: "rfc"`) with a formal lifecycle managed by the Protocol FSM engine. Use them for architectural decisions that need review before implementation.

### When to Use an RFC

- Architectural changes affecting multiple components
- New patterns or conventions that change how the team works
- API redesigns, data model migrations, breaking changes
- Any decision that benefits from structured review

### Creating an RFC

```
note(action: "create",
  project_id: "...",
  note_type: "rfc",
  content: "## Problem\nOur API returns inconsistent error formats...\n\n## Proposed Solution\nStandardize on RFC 7807 Problem Details...\n\n## Alternatives\n1. Custom error envelope...\n2. GraphQL errors...\n\n## Impact\n- All HTTP handlers in src/api/\n- Client SDK needs update\n- Migration guide required",
  importance: "high",
  tags: ["rfc", "api", "error-handling"]
)
```

### RFC Lifecycle

```
draft ──propose──> proposed ──accept──> accepted ──implement──> implemented
                       │
                       └──reject──> rejected
```

```
# Advance an RFC through its lifecycle
note(action: "advance_rfc", note_id: "...", trigger: "propose")   # draft → proposed
note(action: "advance_rfc", note_id: "...", trigger: "accept")    # proposed → accepted
note(action: "advance_rfc", note_id: "...", trigger: "implement") # accepted → implemented

# Check current status
note(action: "get_rfc_status", note_id: "...")
# → { state: "accepted", protocol_run: { ... } }

# List all RFCs for a project
note(action: "list_rfcs", project_id: "...")
```

### Best Practices for RFCs

1. **Always check existing RFCs** before starting work: `note(action: "list_rfcs", project_id)`
2. **Link accepted RFCs to plans**: `note(action: "link_to_entity", note_id, "plan", plan_id)`
3. **Link to affected code**: `note(action: "link_to_entity", note_id, "file", "src/api/handlers.rs")`
4. **Advance to `implemented`** when done, and link the final commit

---

## Best Practices

### Knowledge-First Development

The most powerful feature of Project Orchestrator is its knowledge accumulation. Follow these practices to maximize value:

#### 1. Always Warm Up

Before any work, load existing knowledge to avoid redoing documented work:

```
note(action: "search_semantic", query: "authentication flow")
decision(action: "search_semantic", query: "JWT vs sessions")
note(action: "get_context", entity_type: "file", entity_id: "src/auth/mod.rs")
note(action: "list_rfcs", project_id: "...")
```

#### 2. Capture Everything

After every meaningful discovery, create a note:

| Situation | Note Type | Example |
|-----------|-----------|---------|
| Found a bug root cause | `gotcha` | "Neo4j returns null for optional rels — always use OPTIONAL MATCH" |
| Discovered a pattern | `pattern` | "All handlers follow extract→validate→execute→respond" |
| Identified a convention | `guideline` | "Use ULID for all entity IDs, never UUIDv4" |
| Found a useful trick | `tip` | "Use `code(action: 'get_communities')` before refactoring to understand module boundaries" |
| Observed behavior | `observation` | "Sync takes ~3min for repos > 50k LOC" |

Always link notes to the relevant code:

```
note(action: "create", project_id: "...", note_type: "gotcha",
  content: "neo4rs returns BoltNull for missing optional relationships — always check with .is_null() before unwrapping",
  importance: "high", tags: ["neo4j", "null-safety"])
# → note_id

note(action: "link_to_entity", note_id: "...", entity_type: "file", entity_id: "src/neo4j/client.rs")
```

#### 3. Record Architectural Decisions

For every non-trivial choice, document the alternatives and rationale:

```
decision(action: "add", task_id: "...",
  description: "Use Axum over Actix-web for HTTP framework",
  rationale: "Better tokio integration, simpler middleware, growing ecosystem",
  alternatives: ["actix-web: mature but actor model adds complexity", "warp: filter-based API is harder to read"],
  chosen_option: "axum"
)
```

#### 4. Impact Before Modification

Always check impact before changing code:

```
code(action: "analyze_impact", target: "src/api/handlers.rs")
code(action: "get_file_co_changers", file_path: "src/api/handlers.rs")
note(action: "get_propagated", file_path: "src/api/handlers.rs", slug: "my-project")
```

#### 5. Use the Knowledge Fabric

The Knowledge Fabric connects all entities via semantic relations. Leverage it:

```
# Bootstrap on a new project
admin(action: "bootstrap_knowledge_fabric", project_id: "...")

# Check codebase health (hotspots, risks, knowledge gaps)
code(action: "get_health", project_slug: "my-project")

# Understand module boundaries before refactoring
code(action: "get_communities", project_slug: "my-project")

# Evaluate risk before touching a critical file
code(action: "get_node_importance", project_slug: "my-project",
  node_path: "src/neo4j/client.rs", node_type: "File")
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v0.0.10** | 2026-03-20 | Neural Routing ML Pipeline, Plan Runner with Protocol-Driven Lifecycle (FSM + adaptive feedback), Protocol FSM enhancements (CAS, timeouts, episodes), Event Reactor & Pipeline Quality Gates, 28 mega-tools meta-prompt sync |
| **v0.0.9** | 2026-03-12 | Episodic Memory, M4 MVP P2P Knowledge Exchange, Intent-Adaptive Retrieval (biomimetic memory routing), RFC REST endpoints with lifecycle FSM, auto-roadmap endpoint, transitive knowledge propagation (FeatureGraph, Protocol, Skill) |
| **v0.0.8** | 2026-03-10 | Protocol v2 (hierarchical FSM, generator states, RFC lifecycle), Runner v2+v3 (parallel waves, agent personas), Knowledge Scars & biomimicry, streaming WebSocket events, intelligent hooks |
| **v0.0.7** | 2026-03-06 | Pattern Federation (skill registry, cross-instance discovery), compose & simulate, context-aware routing, wave execution, analytics statistical rigor |
| **v0.0.6** | 2026-02-28 | Conversational intelligence, reasoning trees, VizBlock, smart system prompt, knowledge reconstruction |
| **v0.0.5** | 2026-02-20 | Knowledge Fabric, synapses, co-change detection, community analysis, GDS integration |
| **v0.0.4** | 2026-02-12 | Multi-project workspaces, components, resources, topology |
| **v0.0.3** | 2026-02-05 | Authentication (OIDC + Password), NATS event sync, chat WebSocket |
| **v0.0.2** | 2026-01-28 | Semantic search, decisions, constraints, milestones, releases |
| **v0.0.1** | 2026-01-20 | Initial release: projects, plans, tasks, steps, Tree-sitter sync |

> All releases with full changelogs are available on the [Releases page](https://github.com/this-rs/project-orchestrator/releases).

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/setup/installation.md) | Full setup instructions and configuration |
| [Getting Started](docs/guides/getting-started.md) | Step-by-step tutorial for new users |
| [**Foundation Kit**](docs/guides/foundation-kit.md) | **Start here after setup** — seed knowledge for a self-aware instance |
| [Protocols & FSM](docs/guides/protocols.md) | Building a safe, self-aware setup with protocol guardrails |
| [API Reference](docs/api/reference.md) | Complete REST API documentation |
| [MCP Tools](docs/api/mcp-tools.md) | All 22 MCP mega-tools with examples |
| [Knowledge Fabric](docs/guides/advanced-knowledge-fabric.md) | Bio-inspired knowledge graph: scars, homeostasis, scaffolding |
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
│                   (22 MCP Mega-Tools)                        │
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
│• Code    │  │• Code    │     │• Event   │     │• 16 languages│
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
| C/C++ | `.c`, `.h`, `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` |
| Ruby | `.rb` |
| PHP | `.php` |
| Kotlin | `.kt`, `.kts` |
| Swift | `.swift` |
| Bash | `.sh`, `.bash` |
| C# | `.cs` |
| Scala | `.scala` |
| Zig | `.zig` |
| HCL/Terraform | `.tf`, `.tfvars` |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Give your AI agents a shared brain.</i>
</p>
