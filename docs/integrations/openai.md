# OpenAI Agents Integration

Guide to integrating Project Orchestrator with OpenAI's Agents SDK.

---

## Overview

OpenAI Agents SDK supports MCP (Model Context Protocol), allowing your agents to use Project Orchestrator's **145 tools** for code intelligence, plan management, and multi-agent coordination.

### What you get

- **Shared context** between multiple OpenAI agents
- **Code exploration** via semantic search and graph queries
- **Plan & task tracking** with dependencies
- **Decision history** searchable across sessions
- **Knowledge notes** anchored to code entities, with staleness tracking
- **Chat delegation** to sub-agents via MCP tools

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR OPENAI AGENTS                        │
│              (Python SDK / TypeScript SDK)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │ MCP Protocol (stdio)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                PROJECT ORCHESTRATOR MCP                      │
│                    (mcp_server binary)                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    NEO4J      │     │  MEILISEARCH  │     │  TREE-SITTER  │
│  (Knowledge   │     │   (Semantic   │     │    (Code      │
│    Graph)     │     │    Search)    │     │   Parsing)    │
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## Prerequisites

1. **Backend services running** (see [Installation Guide](../setup/installation.md))
2. **MCP server binary built**
3. **OpenAI API key**
4. **OpenAI Agents SDK installed**

```bash
# Python
pip install openai-agents

# TypeScript
npm install @openai/agents
```

---

## Configuration

### Python SDK

```python
from openai_agents import Agent, MCPServer

# Configure MCP server
mcp_server = MCPServer(
    name="project-orchestrator",
    command="/path/to/mcp_server",
    env={
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "orchestrator123",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me",
    }
)

# Create agent with MCP tools
agent = Agent(
    name="coding-assistant",
    model="gpt-4o",
    mcp_servers=[mcp_server]
)
```

### TypeScript SDK

```typescript
import { Agent, MCPServer } from '@openai/agents';

// Configure MCP server
const mcpServer = new MCPServer({
  name: 'project-orchestrator',
  command: '/path/to/mcp_server',
  env: {
    NEO4J_URI: 'bolt://localhost:7687',
    NEO4J_USER: 'neo4j',
    NEO4J_PASSWORD: 'orchestrator123',
    MEILISEARCH_URL: 'http://localhost:7700',
    MEILISEARCH_KEY: 'orchestrator-meili-key-change-me',
  },
});

// Create agent with MCP tools
const agent = new Agent({
  name: 'coding-assistant',
  model: 'gpt-4o',
  mcpServers: [mcpServer],
});
```

---

## Example: Coding Assistant Agent

### Python

```python
from openai_agents import Agent, MCPServer, Runner

# Setup MCP
mcp = MCPServer(
    name="project-orchestrator",
    command="/path/to/mcp_server",
    env={
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "orchestrator123",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me",
    }
)

# Create agent
coding_agent = Agent(
    name="code-explorer",
    model="gpt-4o",
    instructions="""You are a coding assistant with access to a project knowledge graph.

    Use the available tools to:
    - Search and explore code semantically
    - Track tasks and decisions
    - Analyze impact before making changes

    Always sync the project before exploring code.""",
    mcp_servers=[mcp]
)

# Run the agent
async def main():
    runner = Runner()

    # Start conversation
    response = await runner.run(
        agent=coding_agent,
        messages=[
            {"role": "user", "content": "List all projects, then search for authentication code"}
        ]
    )

    print(response.messages[-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### TypeScript

```typescript
import { Agent, MCPServer, Runner } from '@openai/agents';

async function main() {
  // Setup MCP
  const mcp = new MCPServer({
    name: 'project-orchestrator',
    command: '/path/to/mcp_server',
    env: {
      NEO4J_URI: 'bolt://localhost:7687',
      NEO4J_USER: 'neo4j',
      NEO4J_PASSWORD: 'orchestrator123',
      MEILISEARCH_URL: 'http://localhost:7700',
      MEILISEARCH_KEY: 'orchestrator-meili-key-change-me',
    },
  });

  // Create agent
  const codingAgent = new Agent({
    name: 'code-explorer',
    model: 'gpt-4o',
    instructions: `You are a coding assistant with access to a project knowledge graph.

    Use the available tools to:
    - Search and explore code semantically
    - Track tasks and decisions
    - Analyze impact before making changes

    Always sync the project before exploring code.`,
    mcpServers: [mcp],
  });

  // Run the agent
  const runner = new Runner();
  const response = await runner.run({
    agent: codingAgent,
    messages: [
      { role: 'user', content: 'List all projects, then search for authentication code' },
    ],
  });

  console.log(response.messages.at(-1)?.content);
}

main();
```

---

## Multi-Agent Workflow

### Scenario: Feature Development Team

```python
from openai_agents import Agent, MCPServer, Runner, Handoff

# Shared MCP server
mcp = MCPServer(
    name="project-orchestrator",
    command="/path/to/mcp_server",
    env={...}
)

# Backend Developer Agent
backend_agent = Agent(
    name="backend-developer",
    model="gpt-4o",
    instructions="""You are a backend developer.

    When assigned a task:
    1. Get the task details using get_task
    2. Mark it as in_progress using update_task
    3. Search relevant code using search_code
    4. Implement the feature
    5. Record decisions using add_decision
    6. Mark task as completed when done""",
    mcp_servers=[mcp]
)

# Frontend Developer Agent
frontend_agent = Agent(
    name="frontend-developer",
    model="gpt-4o",
    instructions="""You are a frontend developer.

    When assigned a task:
    1. Get the task details using get_task
    2. Check if backend dependencies are completed using get_task_blockers
    3. If unblocked, mark as in_progress
    4. Implement the UI
    5. Mark task as completed""",
    mcp_servers=[mcp]
)

# Orchestrator Agent
orchestrator = Agent(
    name="tech-lead",
    model="gpt-4o",
    instructions="""You are a tech lead coordinating a team.

    Use get_next_task to find available work.
    Delegate to backend-developer or frontend-developer based on task tags.
    Monitor progress using list_tasks.""",
    mcp_servers=[mcp],
    handoffs=[
        Handoff(target=backend_agent, filter=lambda m: "backend" in m.lower()),
        Handoff(target=frontend_agent, filter=lambda m: "frontend" in m.lower()),
    ]
)

# Run coordinated workflow
async def run_team():
    runner = Runner()
    await runner.run(
        agent=orchestrator,
        messages=[
            {"role": "user", "content": "Work through the tasks in plan abc-123"}
        ]
    )
```

---

## Available Tools

All 145 Project Orchestrator tools are available. See [Claude Code Integration](./claude-code.md#available-tools-145) for the complete list.

Key tools for OpenAI agents:

| Tool | Use Case |
|------|----------|
| `list_projects` | Discover available codebases |
| `sync_project` | Parse and index code |
| `search_code` | Semantic code search |
| `get_next_task` | Get work assignments |
| `update_task` | Track progress |
| `add_decision` | Record architectural choices |
| `analyze_impact` | Check before making changes |
| `create_note` | Capture knowledge about code patterns |
| `search_notes` | Find past observations and guidelines |
| `get_context_notes` | Get notes relevant to a file or function |
| `chat_send_message` | Delegate a task to a sub-agent |
| `list_chat_sessions` | Review past sub-agent conversations |

### Authentication Considerations

The MCP server runs via **stdio** as a child process of your agent, so it does not require authentication itself. However, if your agents also interact with the Project Orchestrator **HTTP REST API** (e.g., for webhooks, dashboard access, or WebSocket chat), you will need to handle authentication:

- **API Keys** — Best for programmatic access from agents. Pass the key as a Bearer token in the `Authorization` header.
- **Google OAuth** — Best for browser-based dashboard access, not typically used by agents directly.

See the [Authentication Guide](../guides/authentication.md) for setup details.

---

## Differences from Claude Code

| Feature | Claude Code | OpenAI Agents |
|---------|-------------|---------------|
| **Configuration** | `~/.claude/mcp.json` | Inline in Python/TS code |
| **Tool discovery** | `/mcp` command | Automatic at agent creation |
| **Multi-agent** | Single agent per session | Native multi-agent support |
| **Handoffs** | Manual conversation | Built-in Handoff API |
| **Streaming** | Yes | Yes |

### Key differences

1. **Configuration location**: OpenAI uses code-based configuration, not JSON files
2. **Multi-agent native**: OpenAI Agents SDK has built-in support for agent handoffs
3. **Tool selection**: OpenAI agents automatically discover and select tools

---

## Best Practices

### 1. Share MCP server across agents

Create one MCP server instance and share it:

```python
# Good: Shared MCP server
mcp = MCPServer(name="orchestrator", ...)
agent1 = Agent(mcp_servers=[mcp])
agent2 = Agent(mcp_servers=[mcp])

# Bad: Duplicate servers (resource waste)
mcp1 = MCPServer(name="orchestrator", ...)
mcp2 = MCPServer(name="orchestrator", ...)
```

### 2. Add tool usage to instructions

Tell agents how to use tools in their instructions:

```python
Agent(
    instructions="""Before exploring code, always:
    1. Call list_projects to see available projects
    2. Call sync_project to ensure data is fresh
    3. Then use search_code or get_file_symbols"""
)
```

### 3. Handle tool errors gracefully

```python
try:
    response = await runner.run(agent=agent, messages=messages)
except MCPToolError as e:
    # Handle tool-specific errors
    if "Connection refused" in str(e):
        print("Backend services not running. Start with: docker compose up -d")
```

### 4. Use task dependencies

Let the orchestrator handle task ordering:

```python
# Agent checks blockers before starting
instructions="""Before starting a task:
1. Call get_task_blockers to check dependencies
2. If blocked, report and get a different task
3. Only work on unblocked tasks"""
```

---

## Troubleshooting

### MCP server not found

```python
# Error: FileNotFoundError: mcp_server
# Fix: Use absolute path
mcp = MCPServer(command="/absolute/path/to/mcp_server")
```

### Connection refused

```bash
# Backend services not running
docker compose ps
docker compose up -d neo4j meilisearch
```

### Tool timeout

```python
# Increase timeout for long operations
mcp = MCPServer(
    command="/path/to/mcp_server",
    timeout=60  # seconds
)
```

### Debug logging

```python
mcp = MCPServer(
    command="/path/to/mcp_server",
    env={
        "RUST_LOG": "debug",
        ...
    }
)
```

---

## Next Steps

- [Getting Started Tutorial](../guides/getting-started.md)
- [Authentication Guide](../guides/authentication.md) — OAuth, API keys, and JWT setup
- [Chat & WebSocket Guide](../guides/chat-websocket.md) — Sub-agent delegation and real-time chat
- [Knowledge Notes Guide](../guides/knowledge-notes.md) — Persistent notes anchored to code
- [Multi-Agent Workflows](../guides/multi-agent-workflow.md)
- [API Reference](../api/reference.md)
- [MCP Tools Reference](../api/mcp-tools.md)
