# Plan: MCP Server pour Claude Code

## Objectif

Créer un binaire `orchestrator-mcp` qui expose l'API de l'orchestrator comme outils MCP natifs pour Claude Code.

---

## Architecture

```
src/
├── bin/
│   └── mcp_server.rs      # Point d'entrée du serveur MCP
├── mcp/
│   ├── mod.rs             # Module MCP
│   ├── protocol.rs        # JSON-RPC 2.0 sur stdio
│   ├── tools.rs           # Définitions des tools
│   └── handlers.rs        # Implémentation des handlers
```

### Communication

- **Transport**: stdio (stdin/stdout)
- **Protocole**: JSON-RPC 2.0
- **Format des messages**: `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}`

---

## Phase 1: Infrastructure MCP (T1-T2)

### T1: Protocol JSON-RPC

**Fichier**: `src/mcp/protocol.rs`

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// Error codes
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;
```

### T2: Server Loop

**Fichier**: `src/mcp/mod.rs`

```rust
use std::io::{BufRead, BufReader, Write};
use tokio::runtime::Runtime;

pub struct McpServer {
    neo4j: Neo4jClient,
    meili: MeilisearchClient,
}

impl McpServer {
    pub fn run(&self) {
        let stdin = BufReader::new(std::io::stdin());
        let mut stdout = std::io::stdout();

        for line in stdin.lines() {
            let line = line.expect("Failed to read line");
            let response = self.handle_request(&line);
            writeln!(stdout, "{}", response).expect("Failed to write");
            stdout.flush().expect("Failed to flush");
        }
    }

    fn handle_request(&self, input: &str) -> String {
        // Parse JSON-RPC, route to handler, return response
    }
}
```

---

## Phase 2: Tool Definitions (T3)

### T3: Définir tous les tools MCP

**Fichier**: `src/mcp/tools.rs`

Les tools MCP suivent le schéma:
```json
{
  "name": "tool_name",
  "description": "What this tool does",
  "inputSchema": {
    "type": "object",
    "properties": { ... },
    "required": [ ... ]
  }
}
```

#### Projects (6 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_projects` | Liste les projets avec pagination | `search?`, `limit?`, `offset?` |
| `create_project` | Crée un nouveau projet | `name`, `root_path`, `slug?`, `description?` |
| `get_project` | Récupère un projet par slug | `slug` |
| `delete_project` | Supprime un projet | `slug` |
| `sync_project` | Synchronise le code d'un projet | `slug` |
| `get_project_roadmap` | Vue agrégée roadmap | `project_id` |

#### Plans (8 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_plans` | Liste les plans avec filtres | `status?`, `priority_min?`, `priority_max?`, `search?`, `limit?`, `offset?`, `sort_by?`, `sort_order?` |
| `create_plan` | Crée un plan | `title`, `description`, `priority?`, `project_id?` |
| `get_plan` | Détails d'un plan | `plan_id` |
| `update_plan_status` | Met à jour le statut | `plan_id`, `status` |
| `link_plan_to_project` | Lie plan à projet | `plan_id`, `project_id` |
| `unlink_plan_from_project` | Délie plan du projet | `plan_id` |
| `get_plan_dependency_graph` | Graphe de dépendances | `plan_id` |
| `get_plan_critical_path` | Chemin critique | `plan_id` |

#### Tasks (12 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_tasks` | Liste globale des tâches | `plan_id?`, `status?`, `priority_min?`, `priority_max?`, `tags?`, `assigned_to?`, `limit?`, `offset?` |
| `create_task` | Ajoute une tâche à un plan | `plan_id`, `description`, `title?`, `priority?`, `tags?`, `acceptance_criteria?`, `affected_files?`, `dependencies?` |
| `get_task` | Détails d'une tâche | `task_id` |
| `update_task` | Met à jour une tâche | `task_id`, `status?`, `assigned_to?`, `priority?`, `tags?` |
| `get_next_task` | Prochaine tâche disponible | `plan_id` |
| `add_task_dependencies` | Ajoute des dépendances | `task_id`, `dependency_ids` |
| `remove_task_dependency` | Supprime une dépendance | `task_id`, `dep_id` |
| `get_task_blockers` | Tâches bloquantes | `task_id` |
| `get_tasks_blocked_by` | Tâches bloquées par celle-ci | `task_id` |
| `get_task_context` | Contexte pour agent | `plan_id`, `task_id` |
| `get_task_prompt` | Prompt généré | `plan_id`, `task_id` |
| `add_decision` | Enregistre une décision | `task_id`, `description`, `rationale`, `alternatives?`, `chosen_option?` |

#### Steps (4 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_steps` | Liste les étapes d'une tâche | `task_id` |
| `create_step` | Ajoute une étape | `task_id`, `description`, `verification?` |
| `update_step` | Met à jour une étape | `step_id`, `status?` |
| `get_step_progress` | Progression des étapes | `task_id` |

#### Constraints (3 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_constraints` | Contraintes d'un plan | `plan_id` |
| `add_constraint` | Ajoute une contrainte | `plan_id`, `constraint_type`, `description`, `severity?` |
| `delete_constraint` | Supprime une contrainte | `constraint_id` |

#### Releases (5 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_releases` | Releases d'un projet | `project_id`, `status?`, `limit?`, `offset?` |
| `create_release` | Crée une release | `project_id`, `version`, `title?`, `description?`, `target_date?` |
| `get_release` | Détails d'une release | `release_id` |
| `update_release` | Met à jour une release | `release_id`, `status?`, `target_date?`, `released_at?`, `title?`, `description?` |
| `add_task_to_release` | Ajoute tâche à release | `release_id`, `task_id` |

#### Milestones (5 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `list_milestones` | Milestones d'un projet | `project_id`, `status?`, `limit?`, `offset?` |
| `create_milestone` | Crée un milestone | `project_id`, `title`, `description?`, `target_date?` |
| `get_milestone` | Détails d'un milestone | `milestone_id` |
| `update_milestone` | Met à jour un milestone | `milestone_id`, `status?`, `target_date?`, `closed_at?`, `title?`, `description?` |
| `get_milestone_progress` | Progression | `milestone_id` |

#### Commits (4 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `create_commit` | Enregistre un commit | `sha`, `message`, `author?`, `files_changed?` |
| `link_commit_to_task` | Lie commit à tâche | `task_id`, `commit_sha` |
| `link_commit_to_plan` | Lie commit à plan | `plan_id`, `commit_sha` |
| `get_task_commits` | Commits d'une tâche | `task_id` |

#### Code Exploration (10 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `search_code` | Recherche sémantique | `query`, `limit?`, `language?` |
| `search_project_code` | Recherche dans un projet | `project_slug`, `query`, `limit?`, `language?` |
| `get_file_symbols` | Symboles d'un fichier | `file_path` |
| `find_references` | Références à un symbole | `symbol`, `limit?` |
| `get_file_dependencies` | Dépendances d'un fichier | `file_path` |
| `get_call_graph` | Graphe d'appels | `function`, `limit?` |
| `analyze_impact` | Analyse d'impact | `target` |
| `get_architecture` | Vue architecture | - |
| `find_similar_code` | Code similaire | `code_snippet`, `limit?` |
| `find_trait_implementations` | Implémentations d'un trait | `trait_name`, `limit?` |

#### Decisions (1 tool)

| Tool | Description | Params |
|------|-------------|--------|
| `search_decisions` | Recherche de décisions | `query`, `limit?` |

#### Sync & Watch (4 tools)

| Tool | Description | Params |
|------|-------------|--------|
| `sync_directory` | Sync manuel | `path`, `project_id?` |
| `start_watch` | Démarre auto-sync | `path`, `project_id?` |
| `stop_watch` | Arrête auto-sync | - |
| `watch_status` | Statut du watcher | - |

**Total: 62 tools**

---

## Phase 3: Handlers (T4-T5)

### T4: Handlers CRUD

**Fichier**: `src/mcp/handlers.rs`

```rust
impl McpServer {
    pub async fn handle_tool_call(&self, name: &str, params: Value) -> Result<Value> {
        match name {
            // Projects
            "list_projects" => self.list_projects(params).await,
            "create_project" => self.create_project(params).await,
            // ... tous les autres
            _ => Err(anyhow!("Unknown tool: {}", name)),
        }
    }

    async fn list_projects(&self, params: Value) -> Result<Value> {
        let search = params.get("search").and_then(|v| v.as_str());
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = params.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let (projects, total) = self.neo4j
            .list_projects_filtered(search, limit, offset, None, "desc")
            .await?;

        Ok(json!({
            "items": projects,
            "total": total,
            "limit": limit,
            "offset": offset
        }))
    }
    // ...
}
```

### T5: Handlers Code Exploration

Même pattern pour les 10 tools de code exploration.

---

## Phase 4: Binary & CLI (T6)

### T6: Point d'entrée

**Fichier**: `src/bin/mcp_server.rs`

```rust
use clap::Parser;
use project_orchestrator::mcp::McpServer;

#[derive(Parser)]
#[command(name = "orchestrator-mcp")]
#[command(about = "MCP server for Project Orchestrator")]
struct Args {
    /// Neo4j URI
    #[arg(long, env = "NEO4J_URI", default_value = "bolt://localhost:7687")]
    neo4j_uri: String,

    /// Neo4j user
    #[arg(long, env = "NEO4J_USER", default_value = "neo4j")]
    neo4j_user: String,

    /// Neo4j password
    #[arg(long, env = "NEO4J_PASSWORD", default_value = "password")]
    neo4j_password: String,

    /// Meilisearch URL
    #[arg(long, env = "MEILI_URL", default_value = "http://localhost:7700")]
    meili_url: String,

    /// Meilisearch API key
    #[arg(long, env = "MEILI_KEY", default_value = "")]
    meili_key: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let server = McpServer::new(
        &args.neo4j_uri,
        &args.neo4j_user,
        &args.neo4j_password,
        &args.meili_url,
        &args.meili_key,
    ).await?;

    server.run();
    Ok(())
}
```

**Cargo.toml** (ajouter):
```toml
[[bin]]
name = "orchestrator-mcp"
path = "src/bin/mcp_server.rs"
```

---

## Phase 5: MCP Protocol Methods (T7)

### T7: Méthodes MCP standard

Le protocole MCP définit des méthodes standard:

```rust
// initialize - Handshake initial
{
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": { "name": "claude-code", "version": "1.0" }
    }
}

// tools/list - Liste des tools disponibles
{
    "method": "tools/list"
}

// tools/call - Appel d'un tool
{
    "method": "tools/call",
    "params": {
        "name": "list_plans",
        "arguments": { "status": "draft", "limit": 10 }
    }
}
```

**Implémentation**:
```rust
fn handle_method(&self, method: &str, params: Option<Value>) -> Result<Value> {
    match method {
        "initialize" => self.initialize(params),
        "tools/list" => self.list_tools(),
        "tools/call" => self.call_tool(params),
        "notifications/initialized" => Ok(Value::Null), // Ack
        _ => Err(method_not_found(method)),
    }
}

fn list_tools(&self) -> Result<Value> {
    Ok(json!({
        "tools": [
            {
                "name": "list_plans",
                "description": "List plans with optional filters and pagination",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string", "description": "Filter by status (comma-separated)" },
                        "priority_min": { "type": "integer", "description": "Minimum priority" },
                        "limit": { "type": "integer", "description": "Max items", "default": 50 },
                        "offset": { "type": "integer", "description": "Skip items", "default": 0 }
                    }
                }
            },
            // ... 61 autres tools
        ]
    }))
}
```

---

## Phase 6: Tests (T8)

### T8: Tests unitaires et d'intégration

**Fichier**: `tests/mcp_tests.rs`

```rust
#[test]
fn test_parse_json_rpc_request() {
    let input = r#"{"jsonrpc":"2.0","method":"tools/list","id":1}"#;
    let req: JsonRpcRequest = serde_json::from_str(input).unwrap();
    assert_eq!(req.method, "tools/list");
}

#[test]
fn test_tool_list_response() {
    let server = create_test_server();
    let response = server.list_tools().unwrap();
    let tools = response["tools"].as_array().unwrap();
    assert!(tools.len() >= 60);
}

#[tokio::test]
async fn test_list_plans_tool() {
    let server = create_test_server().await;
    let result = server.call_tool(json!({
        "name": "list_plans",
        "arguments": { "limit": 5 }
    })).await.unwrap();

    assert!(result["items"].is_array());
}
```

---

## Phase 7: Documentation (T9)

### T9: Configuration Claude Code

**Fichier**: `docs/MCP_SETUP.md`

```markdown
# Configuration MCP pour Claude Code

## Installation

```bash
cargo build --release
cp target/release/orchestrator-mcp /usr/local/bin/
```

## Configuration

Ajouter dans `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "orchestrator": {
      "command": "/usr/local/bin/orchestrator-mcp",
      "args": [],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "MEILI_URL": "http://localhost:7700"
      }
    }
  }
}
```

## Vérification

Dans Claude Code, taper `/mcp` pour voir les tools disponibles.

## Tools disponibles

- **62 tools** couvrant: projects, plans, tasks, steps, releases, milestones, commits, code exploration
```

---

## Fichiers à créer/modifier

| Fichier | Action |
|---------|--------|
| `src/mcp/mod.rs` | CRÉER - Module principal |
| `src/mcp/protocol.rs` | CRÉER - JSON-RPC types |
| `src/mcp/tools.rs` | CRÉER - Tool definitions |
| `src/mcp/handlers.rs` | CRÉER - Tool implementations |
| `src/bin/mcp_server.rs` | CRÉER - Binary entry point |
| `src/lib.rs` | MODIFIER - Export mcp module |
| `Cargo.toml` | MODIFIER - Add binary target |
| `tests/mcp_tests.rs` | CRÉER - Tests |
| `docs/MCP_SETUP.md` | CRÉER - Documentation |

---

## Ordre d'exécution

1. **T1** - Protocol JSON-RPC (types, parsing)
2. **T2** - Server loop (stdio read/write)
3. **T3** - Tool definitions (62 tools)
4. **T4** - Handlers CRUD (projects, plans, tasks, etc.)
5. **T5** - Handlers Code Exploration
6. **T6** - Binary CLI avec clap
7. **T7** - MCP protocol methods (initialize, tools/list, tools/call)
8. **T8** - Tests
9. **T9** - Documentation

---

## Vérification

```bash
# Build
cargo build --release --bin orchestrator-mcp

# Test manuel
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | ./target/release/orchestrator-mcp

# Test avec Claude Code
# 1. Configurer ~/.claude/settings.json
# 2. Relancer Claude Code
# 3. /mcp pour voir les tools
```

---

## Estimation

- **9 tâches**
- **~1500 lignes de code**
- Dépendances: serde_json (déjà présent), clap (déjà présent)
