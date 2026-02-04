# API Reference

Complete REST API documentation for Project Orchestrator.

**Base URL:** `http://localhost:8080`

---

## Authentication

Currently, the API does not require authentication. For production deployments, implement authentication at the reverse proxy level.

---

## Pagination

List endpoints support pagination with these query parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Max items per page (max: 100) |
| `offset` | integer | 0 | Items to skip |
| `sort_by` | string | varies | Field to sort by |
| `sort_order` | string | desc | Sort direction: `asc` or `desc` |

### Response Format

```json
{
  "items": [...],
  "total": 42,
  "limit": 50,
  "offset": 0,
  "has_more": false
}
```

---

## Health Check

### GET /health

Check if the API is running.

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Projects

### GET /api/projects

List all projects.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `search` | string | Search in name/description |
| `limit` | integer | Max items |
| `offset` | integer | Items to skip |
| `sort_by` | string | `name` or `created_at` |
| `sort_order` | string | `asc` or `desc` |

```bash
curl "http://localhost:8080/api/projects?limit=10"
```

**Response:**
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "My Project",
      "slug": "my-project",
      "root_path": "/path/to/project",
      "description": "Project description",
      "created_at": "2024-01-15T10:00:00Z",
      "last_synced": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0,
  "has_more": false
}
```

### POST /api/projects

Create a new project.

```bash
curl -X POST http://localhost:8080/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Project",
    "root_path": "/path/to/project",
    "description": "Optional description",
    "slug": "my-project"
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Project name |
| `root_path` | string | Yes | Absolute path to codebase |
| `description` | string | No | Project description |
| `slug` | string | No | URL-safe identifier (auto-generated) |

**Response:** Created project object.

### GET /api/projects/{slug}

Get project details.

```bash
curl http://localhost:8080/api/projects/my-project
```

### DELETE /api/projects/{slug}

Delete a project and all associated data.

```bash
curl -X DELETE http://localhost:8080/api/projects/my-project
```

### POST /api/projects/{slug}/sync

Sync project files to the knowledge graph.

```bash
curl -X POST http://localhost:8080/api/projects/my-project/sync
```

**Response:**
```json
{
  "files_synced": 127,
  "duration_ms": 1500
}
```

### GET /api/projects/{slug}/plans

List plans associated with a project.

```bash
curl http://localhost:8080/api/projects/my-project/plans
```

### GET /api/projects/{slug}/code/search

Search code within a specific project.

```bash
curl "http://localhost:8080/api/projects/my-project/code/search?q=authentication&limit=10"
```

### GET /api/projects/{project_id}/roadmap

Get aggregated roadmap view.

```bash
curl http://localhost:8080/api/projects/{project_id}/roadmap
```

**Response:**
```json
{
  "milestones": [...],
  "releases": [...],
  "progress": {
    "total_tasks": 45,
    "completed_tasks": 12,
    "percentage": 26.7
  }
}
```

---

## Plans

### GET /api/plans

List all plans.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Comma-separated: `draft,approved,in_progress,completed,cancelled` |
| `priority_min` | integer | Minimum priority |
| `priority_max` | integer | Maximum priority |
| `search` | string | Search in title/description |

```bash
curl "http://localhost:8080/api/plans?status=in_progress&limit=10"
```

### POST /api/plans

Create a new plan.

```bash
curl -X POST http://localhost:8080/api/plans \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Implement Feature X",
    "description": "Add new authentication system",
    "priority": 10,
    "project_id": "optional-uuid"
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Plan title |
| `description` | string | Yes | Plan description |
| `priority` | integer | No | Priority (higher = more important) |
| `project_id` | uuid | No | Associate with a project |

### GET /api/plans/{plan_id}

Get plan details with tasks, constraints, and decisions.

```bash
curl http://localhost:8080/api/plans/{plan_id}
```

**Response:**
```json
{
  "id": "uuid",
  "title": "Implement Feature X",
  "description": "...",
  "status": "in_progress",
  "priority": 10,
  "project_id": "uuid",
  "created_at": "2024-01-15T10:00:00Z",
  "tasks": [...],
  "constraints": [...],
  "decisions": [...]
}
```

### PATCH /api/plans/{plan_id}

Update plan status.

```bash
curl -X PATCH http://localhost:8080/api/plans/{plan_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "in_progress"}'
```

**Status Values:** `draft`, `approved`, `in_progress`, `completed`, `cancelled`

### PUT /api/plans/{plan_id}/project

Link plan to a project.

```bash
curl -X PUT http://localhost:8080/api/plans/{plan_id}/project \
  -H "Content-Type: application/json" \
  -d '{"project_id": "uuid"}'
```

### DELETE /api/plans/{plan_id}/project

Unlink plan from project.

```bash
curl -X DELETE http://localhost:8080/api/plans/{plan_id}/project
```

### GET /api/plans/{plan_id}/next-task

Get next available task (unblocked, highest priority).

```bash
curl http://localhost:8080/api/plans/{plan_id}/next-task
```

### GET /api/plans/{plan_id}/dependency-graph

Get task dependency graph for visualization.

```bash
curl http://localhost:8080/api/plans/{plan_id}/dependency-graph
```

**Response:**
```json
{
  "nodes": [
    {"id": "uuid", "title": "Task 1", "status": "completed", "priority": 10}
  ],
  "edges": [
    {"from": "uuid1", "to": "uuid2"}
  ]
}
```

### GET /api/plans/{plan_id}/critical-path

Get longest dependency chain.

```bash
curl http://localhost:8080/api/plans/{plan_id}/critical-path
```

**Response:**
```json
{
  "path": [
    {"id": "uuid", "title": "Task 1", "status": "completed"},
    {"id": "uuid", "title": "Task 2", "status": "pending"}
  ],
  "length": 2
}
```

---

## Tasks

### GET /api/tasks

List all tasks across plans.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `plan_id` | uuid | Filter by plan |
| `status` | string | Comma-separated: `pending,in_progress,blocked,completed,failed` |
| `priority_min` | integer | Minimum priority |
| `priority_max` | integer | Maximum priority |
| `tags` | string | Comma-separated tags |
| `assigned_to` | string | Filter by assignee |

```bash
curl "http://localhost:8080/api/tasks?status=in_progress&assigned_to=agent-1"
```

### POST /api/plans/{plan_id}/tasks

Add a task to a plan.

```bash
curl -X POST http://localhost:8080/api/plans/{plan_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Implement login",
    "description": "Add JWT-based authentication",
    "priority": 9,
    "tags": ["backend", "security"],
    "acceptance_criteria": ["Tests pass", "Docs updated"],
    "affected_files": ["src/auth.rs"],
    "dependencies": ["task-uuid-1"]
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | No | Short title |
| `description` | string | Yes | Detailed description |
| `priority` | integer | No | Priority (higher = more important) |
| `tags` | string[] | No | Categorization tags |
| `acceptance_criteria` | string[] | No | Completion conditions |
| `affected_files` | string[] | No | Files to be modified |
| `dependencies` | uuid[] | No | Task UUIDs this depends on |

### GET /api/tasks/{task_id}

Get task details.

```bash
curl http://localhost:8080/api/tasks/{task_id}
```

### PATCH /api/tasks/{task_id}

Update a task.

```bash
curl -X PATCH http://localhost:8080/api/tasks/{task_id} \
  -H "Content-Type: application/json" \
  -d '{
    "status": "in_progress",
    "assigned_to": "agent-1"
  }'
```

**Updatable Fields:** `status`, `assigned_to`, `priority`, `tags`

**Status Values:** `pending`, `in_progress`, `blocked`, `completed`, `failed`

### POST /api/tasks/{task_id}/dependencies

Add dependencies to a task.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/dependencies \
  -H "Content-Type: application/json" \
  -d '{"dependency_ids": ["uuid1", "uuid2"]}'
```

### DELETE /api/tasks/{task_id}/dependencies/{dep_id}

Remove a dependency.

```bash
curl -X DELETE http://localhost:8080/api/tasks/{task_id}/dependencies/{dep_id}
```

### GET /api/tasks/{task_id}/blockers

Get tasks blocking this task (uncompleted dependencies).

```bash
curl http://localhost:8080/api/tasks/{task_id}/blockers
```

### GET /api/tasks/{task_id}/blocking

Get tasks blocked by this task.

```bash
curl http://localhost:8080/api/tasks/{task_id}/blocking
```

---

## Steps

### GET /api/tasks/{task_id}/steps

List steps for a task.

```bash
curl http://localhost:8080/api/tasks/{task_id}/steps
```

### POST /api/tasks/{task_id}/steps

Add a step to a task.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/steps \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Setup JWT library",
    "verification": "Can generate tokens"
  }'
```

### PATCH /api/steps/{step_id}

Update step status.

```bash
curl -X PATCH http://localhost:8080/api/steps/{step_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

**Status Values:** `pending`, `in_progress`, `completed`, `skipped`

### GET /api/tasks/{task_id}/steps/progress

Get step completion progress.

```bash
curl http://localhost:8080/api/tasks/{task_id}/steps/progress
```

**Response:**
```json
{
  "completed": 3,
  "total": 5,
  "percentage": 60.0
}
```

---

## Constraints

### GET /api/plans/{plan_id}/constraints

List plan constraints.

```bash
curl http://localhost:8080/api/plans/{plan_id}/constraints
```

### POST /api/plans/{plan_id}/constraints

Add a constraint.

```bash
curl -X POST http://localhost:8080/api/plans/{plan_id}/constraints \
  -H "Content-Type: application/json" \
  -d '{
    "constraint_type": "security",
    "description": "No plaintext passwords",
    "severity": "critical"
  }'
```

**Constraint Types:** `performance`, `security`, `compatibility`, `style`, `testing`, `other`

**Severity Levels:** `low`, `medium`, `high`, `critical`

### DELETE /api/constraints/{constraint_id}

Delete a constraint.

```bash
curl -X DELETE http://localhost:8080/api/constraints/{constraint_id}
```

---

## Decisions

### POST /api/tasks/{task_id}/decisions

Record a decision.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Use JWT for authentication",
    "rationale": "Stateless, better for horizontal scaling",
    "alternatives": ["Session cookies", "OAuth tokens"],
    "chosen_option": "JWT with refresh tokens"
  }'
```

### GET /api/decisions/search

Search past decisions.

```bash
curl "http://localhost:8080/api/decisions/search?q=authentication&limit=10"
```

---

## Releases

### GET /api/projects/{project_id}/releases

List releases for a project.

```bash
curl "http://localhost:8080/api/projects/{project_id}/releases?status=planned"
```

### POST /api/projects/{project_id}/releases

Create a release.

```bash
curl -X POST http://localhost:8080/api/projects/{project_id}/releases \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0.0",
    "title": "Initial Release",
    "description": "First production release",
    "target_date": "2024-03-01T00:00:00Z"
  }'
```

### GET /api/releases/{release_id}

Get release details with tasks and commits.

```bash
curl http://localhost:8080/api/releases/{release_id}
```

### PATCH /api/releases/{release_id}

Update a release.

```bash
curl -X PATCH http://localhost:8080/api/releases/{release_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "released", "released_at": "2024-03-01T12:00:00Z"}'
```

**Status Values:** `planned`, `in_progress`, `released`, `cancelled`

### POST /api/releases/{release_id}/tasks

Add task to release.

```bash
curl -X POST http://localhost:8080/api/releases/{release_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_id": "uuid"}'
```

### POST /api/releases/{release_id}/commits

Add commit to release.

```bash
curl -X POST http://localhost:8080/api/releases/{release_id}/commits \
  -H "Content-Type: application/json" \
  -d '{"commit_sha": "abc123"}'
```

---

## Milestones

### GET /api/projects/{project_id}/milestones

List milestones for a project.

```bash
curl "http://localhost:8080/api/projects/{project_id}/milestones?status=open"
```

### POST /api/projects/{project_id}/milestones

Create a milestone.

```bash
curl -X POST http://localhost:8080/api/projects/{project_id}/milestones \
  -H "Content-Type: application/json" \
  -d '{
    "title": "MVP Complete",
    "description": "Minimum viable product ready",
    "target_date": "2024-02-15T00:00:00Z"
  }'
```

### GET /api/milestones/{milestone_id}

Get milestone details with tasks.

```bash
curl http://localhost:8080/api/milestones/{milestone_id}
```

### PATCH /api/milestones/{milestone_id}

Update a milestone.

```bash
curl -X PATCH http://localhost:8080/api/milestones/{milestone_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "closed"}'
```

**Status Values:** `open`, `closed`

### POST /api/milestones/{milestone_id}/tasks

Add task to milestone.

```bash
curl -X POST http://localhost:8080/api/milestones/{milestone_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_id": "uuid"}'
```

### GET /api/milestones/{milestone_id}/progress

Get milestone completion progress.

```bash
curl http://localhost:8080/api/milestones/{milestone_id}/progress
```

---

## Commits

### POST /api/commits

Register a commit.

```bash
curl -X POST http://localhost:8080/api/commits \
  -H "Content-Type: application/json" \
  -d '{
    "sha": "abc123def456",
    "message": "feat: add authentication",
    "author": "Developer",
    "files_changed": ["src/auth.rs", "src/lib.rs"]
  }'
```

### GET /api/tasks/{task_id}/commits

Get commits linked to a task.

```bash
curl http://localhost:8080/api/tasks/{task_id}/commits
```

### POST /api/tasks/{task_id}/commits

Link commit to task.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/commits \
  -H "Content-Type: application/json" \
  -d '{"commit_sha": "abc123"}'
```

### GET /api/plans/{plan_id}/commits

Get commits linked to a plan.

```bash
curl http://localhost:8080/api/plans/{plan_id}/commits
```

### POST /api/plans/{plan_id}/commits

Link commit to plan.

```bash
curl -X POST http://localhost:8080/api/plans/{plan_id}/commits \
  -H "Content-Type: application/json" \
  -d '{"commit_sha": "abc123"}'
```

---

## Code Exploration

### GET /api/code/search

Semantic code search.

```bash
curl "http://localhost:8080/api/code/search?q=error+handling&limit=10&language=rust"
```

**Response:**
```json
{
  "hits": [
    {
      "path": "src/error.rs",
      "language": "rust",
      "snippet": "pub struct AppError {...}",
      "symbols": ["AppError", "handle_error"],
      "score": 0.95
    }
  ]
}
```

### GET /api/code/symbols/{file_path}

Get symbols in a file.

```bash
curl "http://localhost:8080/api/code/symbols/src%2Flib.rs"
```

**Response:**
```json
{
  "path": "src/lib.rs",
  "language": "rust",
  "functions": [
    {
      "name": "main",
      "signature": "fn main() -> Result<()>",
      "line": 15,
      "is_async": true,
      "is_public": false
    }
  ],
  "structs": [...],
  "traits": [...],
  "imports": [...]
}
```

### GET /api/code/references

Find all references to a symbol.

```bash
curl "http://localhost:8080/api/code/references?symbol=AppState&limit=20"
```

### GET /api/code/dependencies/{file_path}

Get file imports and dependents.

```bash
curl "http://localhost:8080/api/code/dependencies/src%2Flib.rs"
```

**Response:**
```json
{
  "imports": ["src/config.rs", "src/db.rs"],
  "imported_by": ["src/main.rs", "src/api/mod.rs"],
  "impact_radius": 5
}
```

### GET /api/code/callgraph

Get function call graph.

```bash
curl "http://localhost:8080/api/code/callgraph?function=handle_request&depth=2&direction=both"
```

### GET /api/code/impact

Analyze change impact.

```bash
curl "http://localhost:8080/api/code/impact?target=src/models/user.rs&target_type=file"
```

**Response:**
```json
{
  "directly_affected": ["src/handlers/user.rs"],
  "transitively_affected": ["src/api/mod.rs", "src/main.rs"],
  "test_files_affected": ["tests/user_tests.rs"],
  "risk_level": "medium",
  "suggestion": "Consider updating 3 test files"
}
```

### GET /api/code/architecture

Get codebase architecture overview.

```bash
curl http://localhost:8080/api/code/architecture
```

### POST /api/code/similar

Find similar code.

```bash
curl -X POST http://localhost:8080/api/code/similar \
  -H "Content-Type: application/json" \
  -d '{"snippet": "async fn handle_error", "limit": 5}'
```

### GET /api/code/trait-impls

Find trait implementations.

```bash
curl "http://localhost:8080/api/code/trait-impls?trait_name=Handler&limit=10"
```

### GET /api/code/type-traits

Find traits implemented by a type.

```bash
curl "http://localhost:8080/api/code/type-traits?type_name=AppState"
```

### GET /api/code/impl-blocks

Get impl blocks for a type.

```bash
curl "http://localhost:8080/api/code/impl-blocks?type_name=Orchestrator"
```

---

## Notes

Knowledge Notes capture contextual knowledge about your codebase. See the [Knowledge Notes Guide](../guides/knowledge-notes.md) for detailed usage.

### GET /api/notes

List notes with filters and pagination.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `project_id` | string | Filter by project UUID |
| `note_type` | string | `guideline`, `gotcha`, `pattern`, `context`, `tip`, `observation`, `assertion` |
| `status` | string | Comma-separated: `active,needs_review,stale,obsolete,archived` |
| `importance` | string | `critical`, `high`, `medium`, `low` |
| `tags` | string | Comma-separated tags |
| `search` | string | Search in content |
| `min_staleness` | number | Min staleness score (0.0-1.0) |
| `max_staleness` | number | Max staleness score (0.0-1.0) |
| `limit` | integer | Max items (default 50) |
| `offset` | integer | Items to skip |

```bash
curl "http://localhost:8080/api/notes?note_type=guideline&status=active&limit=20"
```

### POST /api/notes

Create a new note.

```bash
curl -X POST http://localhost:8080/api/notes \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "uuid",
    "note_type": "gotcha",
    "content": "Do not use unwrap() in async contexts",
    "importance": "high",
    "tags": ["async", "error-handling"]
  }'
```

**Note Types:** `guideline`, `gotcha`, `pattern`, `context`, `tip`, `observation`, `assertion`

**Importance Levels:** `critical`, `high`, `medium`, `low`

### GET /api/notes/{note_id}

Get note details.

```bash
curl http://localhost:8080/api/notes/{note_id}
```

### PATCH /api/notes/{note_id}

Update a note.

```bash
curl -X PATCH http://localhost:8080/api/notes/{note_id} \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Updated content",
    "importance": "critical"
  }'
```

**Updatable Fields:** `content`, `importance`, `status`, `tags`

### DELETE /api/notes/{note_id}

Delete a note.

```bash
curl -X DELETE http://localhost:8080/api/notes/{note_id}
```

### GET /api/notes/search

Semantic search across notes.

```bash
curl "http://localhost:8080/api/notes/search?q=error+handling&limit=10"
```

**Response:**
```json
{
  "hits": [
    {
      "id": "uuid",
      "content": "Always wrap errors with context...",
      "note_type": "guideline",
      "importance": "high",
      "score": 0.95
    }
  ]
}
```

### GET /api/notes/context

Get contextual notes for an entity (direct + propagated through graph).

```bash
curl "http://localhost:8080/api/notes/context?entity_type=file&entity_id=src/auth/jwt.rs&max_depth=2"
```

**Response:**
```json
{
  "direct_notes": [
    {
      "id": "uuid",
      "note_type": "guideline",
      "content": "All JWT operations must use configured secret",
      "importance": "high",
      "relevance_score": 1.0
    }
  ],
  "propagated_notes": [
    {
      "note": {...},
      "source_entity": "src/auth/mod.rs",
      "relevance_score": 0.7,
      "propagation_path": ["CONTAINS", "auth/mod.rs"]
    }
  ]
}
```

### GET /api/notes/needs-review

Get notes needing human review (stale or needs_review status).

```bash
curl http://localhost:8080/api/notes/needs-review
```

### POST /api/notes/update-staleness

Recalculate staleness scores for all notes.

```bash
curl -X POST http://localhost:8080/api/notes/update-staleness
```

### POST /api/notes/{note_id}/confirm

Confirm a note is still valid (resets staleness).

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/confirm
```

### POST /api/notes/{note_id}/invalidate

Mark a note as obsolete.

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/invalidate \
  -H "Content-Type: application/json" \
  -d '{"reason": "Auth system was refactored to OAuth"}'
```

### POST /api/notes/{note_id}/supersede

Replace a note with a new one (preserves history).

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/supersede \
  -H "Content-Type: application/json" \
  -d '{
    "note_type": "guideline",
    "content": "Use OAuth tokens instead of JWT",
    "importance": "high"
  }'
```

### POST /api/notes/{note_id}/links

Link a note to a code entity.

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/links \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "function",
    "entity_id": "validate_user"
  }'
```

**Entity Types:** `file`, `function`, `struct`, `trait`, `module`, `task`, `plan`

### DELETE /api/notes/{note_id}/links/{entity_type}/{entity_id}

Remove a link between a note and entity.

```bash
curl -X DELETE http://localhost:8080/api/notes/{note_id}/links/file/src%2Fauth.rs
```

### GET /api/projects/{project_id}/notes

List notes for a specific project.

```bash
curl "http://localhost:8080/api/projects/{project_id}/notes?status=active"
```

---

## Sync & Watch

### POST /api/sync

Manually sync a directory.

```bash
curl -X POST http://localhost:8080/api/sync \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project", "project_id": "optional-uuid"}'
```

### GET /api/watch

Get file watcher status.

```bash
curl http://localhost:8080/api/watch
```

### POST /api/watch

Start file watcher.

```bash
curl -X POST http://localhost:8080/api/watch \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project"}'
```

### DELETE /api/watch

Stop file watcher.

```bash
curl -X DELETE http://localhost:8080/api/watch
```

---

## Webhooks

### POST /api/wake

Agent completion webhook.

```bash
curl -X POST http://localhost:8080/api/wake \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "uuid",
    "success": true,
    "summary": "Implemented authentication",
    "files_modified": ["src/auth.rs"]
  }'
```

---

## Meilisearch Maintenance

### GET /api/meilisearch/stats

Get code index statistics.

```bash
curl http://localhost:8080/api/meilisearch/stats
```

### DELETE /api/meilisearch/orphans

Delete documents without project_id.

```bash
curl -X DELETE http://localhost:8080/api/meilisearch/orphans
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `CONFLICT` | 409 | Resource already exists |
| `INTERNAL_ERROR` | 500 | Server error |
