# Plan: OpenAPI Specification pour OpenAI & autres

## Objectif

Ajouter un endpoint `/openapi.json` qui expose la spécification OpenAPI 3.0 complète de l'API, permettant:
- **OpenAI GPT Actions**: Import direct de la spec pour créer des actions
- **OpenAI Function Calling**: Génération automatique des définitions de fonctions
- **Swagger UI**: Documentation interactive
- **Génération de clients**: SDK automatiques pour tout langage

---

## Architecture

```
src/api/
├── openapi.rs             # Génération de la spec OpenAPI
├── mod.rs                 # Export du module
└── routes.rs              # Ajout de la route /openapi.json
```

### Approche

Deux options possibles:

| Option | Avantages | Inconvénients |
|--------|-----------|---------------|
| **A: Génération manuelle** | Contrôle total, pas de dépendance | Maintenance lourde |
| **B: utoipa (macro)** | Génération automatique depuis les handlers | Dépendance, macros sur tout le code |

**Choix recommandé**: Option A (génération manuelle) car:
- La spec est stable (peu de changements)
- Permet des descriptions détaillées
- Pas de refactoring massif des handlers existants

---

## Phase 1: Structure OpenAPI (T1)

### T1: Types OpenAPI

**Fichier**: `src/api/openapi.rs`

```rust
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct OpenApiSpec {
    pub openapi: String,
    pub info: Info,
    pub servers: Vec<Server>,
    pub paths: HashMap<String, PathItem>,
    pub components: Components,
}

#[derive(Serialize)]
pub struct Info {
    pub title: String,
    pub description: String,
    pub version: String,
    pub contact: Option<Contact>,
}

#[derive(Serialize)]
pub struct Contact {
    pub name: String,
    pub url: Option<String>,
}

#[derive(Serialize)]
pub struct Server {
    pub url: String,
    pub description: String,
}

#[derive(Serialize)]
pub struct PathItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub put: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patch: Option<Operation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delete: Option<Operation>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    pub operation_id: String,
    pub summary: String,
    pub description: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_body: Option<RequestBody>,
    pub responses: HashMap<String, Response>,
}

#[derive(Serialize)]
pub struct Parameter {
    pub name: String,
    #[serde(rename = "in")]
    pub location: String, // "path", "query", "header"
    pub description: String,
    pub required: bool,
    pub schema: Schema,
}

#[derive(Serialize)]
pub struct RequestBody {
    pub required: bool,
    pub content: HashMap<String, MediaType>,
}

#[derive(Serialize)]
pub struct MediaType {
    pub schema: Schema,
}

#[derive(Serialize)]
pub struct Response {
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<HashMap<String, MediaType>>,
}

#[derive(Serialize)]
pub struct Schema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub schema_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Schema>>,
    #[serde(rename = "$ref", skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Schema>>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub required: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "default", skip_serializing_if = "Option::is_none")]
    pub default_value: Option<serde_json::Value>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct Components {
    pub schemas: HashMap<String, Schema>,
}
```

---

## Phase 2: Schémas de données (T2)

### T2: Définir tous les schemas

**Schemas requis** (dans `components.schemas`):

#### Core Models

```json
{
  "PlanNode": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "format": "uuid" },
      "title": { "type": "string" },
      "description": { "type": "string" },
      "status": { "$ref": "#/components/schemas/PlanStatus" },
      "created_at": { "type": "string", "format": "date-time" },
      "created_by": { "type": "string" },
      "priority": { "type": "integer" },
      "project_id": { "type": "string", "format": "uuid", "nullable": true }
    },
    "required": ["id", "title", "description", "status", "created_at", "created_by", "priority"]
  },
  "PlanStatus": {
    "type": "string",
    "enum": ["draft", "approved", "in_progress", "completed", "cancelled"]
  },
  "TaskNode": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "format": "uuid" },
      "title": { "type": "string", "nullable": true },
      "description": { "type": "string" },
      "status": { "$ref": "#/components/schemas/TaskStatus" },
      "assigned_to": { "type": "string", "nullable": true },
      "priority": { "type": "integer", "nullable": true },
      "tags": { "type": "array", "items": { "type": "string" } },
      "acceptance_criteria": { "type": "array", "items": { "type": "string" } },
      "affected_files": { "type": "array", "items": { "type": "string" } },
      "estimated_complexity": { "type": "integer", "nullable": true },
      "actual_complexity": { "type": "integer", "nullable": true },
      "created_at": { "type": "string", "format": "date-time" },
      "updated_at": { "type": "string", "format": "date-time", "nullable": true },
      "started_at": { "type": "string", "format": "date-time", "nullable": true },
      "completed_at": { "type": "string", "format": "date-time", "nullable": true }
    },
    "required": ["id", "description", "status", "created_at"]
  },
  "TaskStatus": {
    "type": "string",
    "enum": ["pending", "in_progress", "blocked", "completed", "failed"]
  },
  "TaskWithPlan": {
    "allOf": [
      { "$ref": "#/components/schemas/TaskNode" },
      {
        "type": "object",
        "properties": {
          "plan_id": { "type": "string", "format": "uuid" },
          "plan_title": { "type": "string" }
        },
        "required": ["plan_id", "plan_title"]
      }
    ]
  }
}
```

#### Autres schemas

| Schema | Description |
|--------|-------------|
| `ProjectNode` | Projet avec slug, root_path |
| `StepNode` | Étape d'une tâche |
| `StepStatus` | pending, in_progress, completed, skipped |
| `DecisionNode` | Décision architecturale |
| `ConstraintNode` | Contrainte (performance, security, etc.) |
| `ReleaseNode` | Release avec version, dates |
| `ReleaseStatus` | planned, in_progress, released, cancelled |
| `MilestoneNode` | Milestone avec target_date |
| `MilestoneStatus` | open, closed |
| `CommitNode` | Commit Git |
| `PaginatedResponse` | Wrapper générique avec items, total, limit, offset, has_more |
| `CreatePlanRequest` | Body pour POST /api/plans |
| `CreateTaskRequest` | Body pour POST /api/plans/{id}/tasks |
| `UpdateTaskRequest` | Body pour PATCH /api/tasks/{id} |
| `Error` | { error: string } |

**Total: ~25 schemas**

---

## Phase 3: Définition des Paths (T3-T7)

### T3: Projects Paths

```rust
fn projects_paths() -> HashMap<String, PathItem> {
    let mut paths = HashMap::new();

    // GET/POST /api/projects
    paths.insert("/api/projects".to_string(), PathItem {
        get: Some(Operation {
            operation_id: "listProjects".to_string(),
            summary: "List all projects".to_string(),
            description: "List projects with optional search and pagination".to_string(),
            tags: vec!["Projects".to_string()],
            parameters: vec![
                query_param("search", "string", "Search in name/description", false),
                query_param("limit", "integer", "Max items (default 50, max 100)", false),
                query_param("offset", "integer", "Items to skip", false),
                query_param("sort_by", "string", "Sort field", false),
                query_param("sort_order", "string", "asc or desc", false),
            ],
            request_body: None,
            responses: success_response("PaginatedProjectResponse"),
        }),
        post: Some(Operation {
            operation_id: "createProject".to_string(),
            summary: "Create a new project".to_string(),
            description: "Create a new project to track a codebase".to_string(),
            tags: vec!["Projects".to_string()],
            parameters: vec![],
            request_body: Some(json_body("CreateProjectRequest", true)),
            responses: success_response("ProjectResponse"),
        }),
        ..Default::default()
    });

    // GET/DELETE /api/projects/{slug}
    paths.insert("/api/projects/{slug}".to_string(), PathItem {
        get: Some(Operation {
            operation_id: "getProject".to_string(),
            summary: "Get project by slug".to_string(),
            description: "Retrieve a project by its URL-safe slug".to_string(),
            tags: vec!["Projects".to_string()],
            parameters: vec![path_param("slug", "Project slug")],
            request_body: None,
            responses: success_response("ProjectResponse"),
        }),
        delete: Some(Operation {
            operation_id: "deleteProject".to_string(),
            summary: "Delete a project".to_string(),
            description: "Delete a project and all its associated data".to_string(),
            tags: vec!["Projects".to_string()],
            parameters: vec![path_param("slug", "Project slug")],
            request_body: None,
            responses: no_content_response(),
        }),
        ..Default::default()
    });

    // POST /api/projects/{slug}/sync
    // GET /api/projects/{slug}/plans
    // GET /api/projects/{slug}/code/search
    // ... etc

    paths
}
```

### T4: Plans Paths

| Endpoint | Method | operation_id |
|----------|--------|--------------|
| `/api/plans` | GET | listPlans |
| `/api/plans` | POST | createPlan |
| `/api/plans/{plan_id}` | GET | getPlan |
| `/api/plans/{plan_id}` | PATCH | updatePlanStatus |
| `/api/plans/{plan_id}/project` | PUT | linkPlanToProject |
| `/api/plans/{plan_id}/project` | DELETE | unlinkPlanFromProject |
| `/api/plans/{plan_id}/next-task` | GET | getNextTask |
| `/api/plans/{plan_id}/dependency-graph` | GET | getDependencyGraph |
| `/api/plans/{plan_id}/critical-path` | GET | getCriticalPath |
| `/api/plans/{plan_id}/constraints` | GET | getConstraints |
| `/api/plans/{plan_id}/constraints` | POST | addConstraint |
| `/api/plans/{plan_id}/tasks` | POST | addTask |
| `/api/plans/{plan_id}/commits` | GET | getPlanCommits |
| `/api/plans/{plan_id}/commits` | POST | linkCommitToPlan |

### T5: Tasks Paths

| Endpoint | Method | operation_id |
|----------|--------|--------------|
| `/api/tasks` | GET | listAllTasks |
| `/api/tasks/{task_id}` | GET | getTask |
| `/api/tasks/{task_id}` | PATCH | updateTask |
| `/api/tasks/{task_id}/dependencies` | POST | addDependencies |
| `/api/tasks/{task_id}/dependencies/{dep_id}` | DELETE | removeDependency |
| `/api/tasks/{task_id}/blockers` | GET | getBlockers |
| `/api/tasks/{task_id}/blocking` | GET | getBlocking |
| `/api/tasks/{task_id}/steps` | GET | getSteps |
| `/api/tasks/{task_id}/steps` | POST | addStep |
| `/api/tasks/{task_id}/steps/progress` | GET | getStepProgress |
| `/api/tasks/{task_id}/decisions` | POST | addDecision |
| `/api/tasks/{task_id}/commits` | GET | getTaskCommits |
| `/api/tasks/{task_id}/commits` | POST | linkCommitToTask |
| `/api/steps/{step_id}` | PATCH | updateStep |
| `/api/constraints/{constraint_id}` | DELETE | deleteConstraint |

### T6: Releases & Milestones Paths

| Endpoint | Method | operation_id |
|----------|--------|--------------|
| `/api/projects/{project_id}/releases` | GET | listReleases |
| `/api/projects/{project_id}/releases` | POST | createRelease |
| `/api/releases/{release_id}` | GET | getRelease |
| `/api/releases/{release_id}` | PATCH | updateRelease |
| `/api/releases/{release_id}/tasks` | POST | addTaskToRelease |
| `/api/releases/{release_id}/commits` | POST | addCommitToRelease |
| `/api/projects/{project_id}/milestones` | GET | listMilestones |
| `/api/projects/{project_id}/milestones` | POST | createMilestone |
| `/api/milestones/{milestone_id}` | GET | getMilestone |
| `/api/milestones/{milestone_id}` | PATCH | updateMilestone |
| `/api/milestones/{milestone_id}/tasks` | POST | addTaskToMilestone |
| `/api/milestones/{milestone_id}/progress` | GET | getMilestoneProgress |
| `/api/projects/{project_id}/roadmap` | GET | getRoadmap |

### T7: Code Exploration Paths

| Endpoint | Method | operation_id |
|----------|--------|--------------|
| `/api/code/search` | GET | searchCode |
| `/api/code/symbols/{file_path}` | GET | getFileSymbols |
| `/api/code/references` | GET | findReferences |
| `/api/code/dependencies/{file_path}` | GET | getFileDependencies |
| `/api/code/callgraph` | GET | getCallGraph |
| `/api/code/impact` | GET | analyzeImpact |
| `/api/code/architecture` | GET | getArchitecture |
| `/api/code/similar` | POST | findSimilarCode |
| `/api/code/trait-impls` | GET | findTraitImplementations |
| `/api/code/type-traits` | GET | findTypeTraits |
| `/api/code/impl-blocks` | GET | getImplBlocks |
| `/api/decisions/search` | GET | searchDecisions |

---

## Phase 4: Génération de la Spec (T8)

### T8: Fonction de génération complète

```rust
pub fn generate_openapi_spec() -> OpenApiSpec {
    OpenApiSpec {
        openapi: "3.0.3".to_string(),
        info: Info {
            title: "Project Orchestrator API".to_string(),
            description: "API for coordinating AI coding agents on complex projects. \
                Provides plan management, task tracking, code exploration, and more.".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            contact: Some(Contact {
                name: "OpenClaw".to_string(),
                url: Some("https://github.com/openclaw".to_string()),
            }),
        },
        servers: vec![
            Server {
                url: "http://localhost:8080".to_string(),
                description: "Local development server".to_string(),
            },
        ],
        paths: build_all_paths(),
        components: Components {
            schemas: build_all_schemas(),
        },
    }
}

fn build_all_paths() -> HashMap<String, PathItem> {
    let mut paths = HashMap::new();
    paths.extend(projects_paths());
    paths.extend(plans_paths());
    paths.extend(tasks_paths());
    paths.extend(releases_paths());
    paths.extend(milestones_paths());
    paths.extend(code_paths());
    paths.extend(misc_paths()); // sync, watch, commits, etc.
    paths
}
```

---

## Phase 5: Handler & Route (T9)

### T9: Endpoint /openapi.json

**Fichier**: `src/api/handlers.rs` (ajouter)

```rust
use crate::api::openapi::generate_openapi_spec;

/// GET /openapi.json
pub async fn get_openapi_spec() -> Json<serde_json::Value> {
    let spec = generate_openapi_spec();
    Json(serde_json::to_value(spec).unwrap())
}
```

**Fichier**: `src/api/routes.rs` (modifier)

```rust
// Ajouter après health check
.route("/openapi.json", get(handlers::get_openapi_spec))
```

---

## Phase 6: Swagger UI (T10 - Optionnel)

### T10: Interface de documentation interactive

Option 1: Swagger UI statique
```rust
// Servir les fichiers Swagger UI
.nest_service("/docs", ServeDir::new("static/swagger-ui"))
```

Option 2: Utiliser `utoipa-swagger-ui` crate
```rust
// Dans routes.rs
.merge(SwaggerUi::new("/docs").url("/openapi.json", spec))
```

---

## Phase 7: Tests (T11)

### T11: Validation de la spec

**Fichier**: `tests/openapi_tests.rs`

```rust
#[test]
fn test_openapi_spec_valid_json() {
    let spec = generate_openapi_spec();
    let json = serde_json::to_string_pretty(&spec).unwrap();
    assert!(json.contains("\"openapi\":\"3.0.3\""));
}

#[test]
fn test_openapi_spec_has_all_paths() {
    let spec = generate_openapi_spec();

    // Vérifier les paths principaux
    assert!(spec.paths.contains_key("/api/plans"));
    assert!(spec.paths.contains_key("/api/tasks"));
    assert!(spec.paths.contains_key("/api/projects"));
    assert!(spec.paths.contains_key("/api/code/search"));
}

#[test]
fn test_openapi_spec_has_all_schemas() {
    let spec = generate_openapi_spec();

    assert!(spec.components.schemas.contains_key("PlanNode"));
    assert!(spec.components.schemas.contains_key("TaskNode"));
    assert!(spec.components.schemas.contains_key("PaginatedResponse"));
}

#[tokio::test]
async fn test_openapi_endpoint() {
    // Test HTTP endpoint
    let client = Client::new();
    let resp = client
        .get("http://localhost:8080/openapi.json")
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());

    let spec: Value = resp.json().await.unwrap();
    assert_eq!(spec["openapi"], "3.0.3");
}
```

---

## Phase 8: Documentation OpenAI (T12)

### T12: Guide d'utilisation avec OpenAI

**Fichier**: `docs/OPENAI_SETUP.md`

```markdown
# Utilisation avec OpenAI

## GPT Actions (Custom GPTs)

1. Créer un nouveau GPT sur chat.openai.com
2. Aller dans "Configure" > "Actions"
3. Cliquer "Import from URL"
4. Entrer: `https://your-server.com/openapi.json`
5. Toutes les actions sont importées automatiquement

## Function Calling (API)

```python
import openai
import requests

# Récupérer la spec
spec = requests.get("http://localhost:8080/openapi.json").json()

# Convertir en tools OpenAI
tools = []
for path, item in spec["paths"].items():
    for method, op in item.items():
        if method in ["get", "post", "patch", "put", "delete"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": op["operationId"],
                    "description": op["description"],
                    "parameters": extract_parameters(op)
                }
            })

# Utiliser avec l'API
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List all pending tasks"}],
    tools=tools,
    tool_choice="auto"
)
```

## Endpoints disponibles

L'API expose **52 endpoints** répartis en:
- Projects (6)
- Plans (14)
- Tasks (15)
- Releases & Milestones (13)
- Code Exploration (12)
- Misc (2)
```

---

## Fichiers à créer/modifier

| Fichier | Action |
|---------|--------|
| `src/api/openapi.rs` | CRÉER - Types et génération OpenAPI |
| `src/api/mod.rs` | MODIFIER - Export openapi module |
| `src/api/handlers.rs` | MODIFIER - Ajouter get_openapi_spec |
| `src/api/routes.rs` | MODIFIER - Ajouter route /openapi.json |
| `tests/openapi_tests.rs` | CRÉER - Tests |
| `docs/OPENAI_SETUP.md` | CRÉER - Documentation |

---

## Ordre d'exécution

1. **T1** - Types OpenAPI (structures Rust)
2. **T2** - Schemas (PlanNode, TaskNode, etc.)
3. **T3** - Projects paths
4. **T4** - Plans paths
5. **T5** - Tasks paths
6. **T6** - Releases & Milestones paths
7. **T7** - Code Exploration paths
8. **T8** - Fonction generate_openapi_spec()
9. **T9** - Handler et route /openapi.json
10. **T10** - Swagger UI (optionnel)
11. **T11** - Tests
12. **T12** - Documentation OpenAI

---

## Vérification

```bash
# Build
cargo build

# Test endpoint
curl http://localhost:8080/openapi.json | jq .info

# Valider avec swagger-cli (npm)
npx @apidevtools/swagger-cli validate http://localhost:8080/openapi.json

# Tester dans Swagger Editor
# https://editor.swagger.io/ -> File > Import URL
```

---

## Estimation

- **12 tâches**
- **~1200 lignes de code** (principalement des définitions de schemas/paths)
- Dépendances: serde_json (déjà présent)
- Optionnel: utoipa-swagger-ui pour Swagger UI intégré

---

## Exemple de sortie

```json
{
  "openapi": "3.0.3",
  "info": {
    "title": "Project Orchestrator API",
    "description": "API for coordinating AI coding agents...",
    "version": "0.1.0"
  },
  "paths": {
    "/api/plans": {
      "get": {
        "operationId": "listPlans",
        "summary": "List plans with filters",
        "parameters": [
          { "name": "status", "in": "query", "schema": { "type": "string" } },
          { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 50 } }
        ],
        "responses": {
          "200": {
            "description": "Paginated list of plans",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/PaginatedPlanResponse" }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "PlanNode": { ... },
      "TaskNode": { ... }
    }
  }
}
```
