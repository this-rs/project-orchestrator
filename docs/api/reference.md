# API Reference

Complete REST API documentation for Project Orchestrator.

**Base URL:** `http://localhost:8080`

**Total routes:** 110 (14 public + 96 protected)

---

## Authentication

The API uses **JWT Bearer token authentication**. When authentication is configured, it operates on a **deny-by-default** basis: all routes require a valid JWT unless explicitly marked as public.

Routes are split into two groups:

- **Public** (no auth required) -- marked with the unlocked padlock icon in this document
- **Protected** (require `Authorization: Bearer <JWT>` header) -- marked with the locked padlock icon

For detailed setup instructions (Google OAuth, JWT configuration, environment variables), see the [Authentication Guide](../guides/authentication.md).

### Route Access Summary

| Route Prefix | Access | Notes |
|--------------|--------|-------|
| `GET /health` | Public | Health check |
| `GET /api/version` | Public | Version info and feature flags |
| `GET /api/setup-status` | Public | Setup wizard status |
| `/auth/providers` | Public | List available auth methods |
| `/auth/login` | Public | Password login |
| `/auth/register` | Public | User registration |
| `/auth/google`, `/auth/google/callback` | Public | Google OAuth login flow |
| `/auth/oidc`, `/auth/oidc/callback` | Public | Generic OIDC login flow |
| `/auth/me`, `/auth/refresh` | **Protected** | User info and token refresh |
| `/ws/*` | Public | Auth via first WebSocket message |
| `/hooks/wake` | Public | Agent webhook |
| `/internal/events` | Public | Internal event receiver |
| `/api/*` | **Protected** | All API routes require JWT |

### Authenticated Request Example

```bash
curl -H "Authorization: Bearer <JWT>" http://localhost:8080/api/plans
```

---

## Auth Routes

### GET /auth/providers -- Public

List available authentication methods.

```bash
curl http://localhost:8080/auth/providers
```

**Response:**
```json
{
  "providers": ["password", "google", "oidc"],
  "allow_registration": true,
  "auth_required": true
}
```

### POST /auth/login -- Public

Login with email and password.

```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "your-password"}'
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "email": "admin@example.com",
    "name": "Admin",
    "is_root": true
  }
}
```

### POST /auth/register -- Public

Register a new user account (requires `allow_registration: true` in config).

```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure-password",
    "name": "New User"
  }'
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "email": "user@example.com",
    "name": "New User"
  }
}
```

### GET /auth/oidc -- Public

Start generic OIDC login flow. Redirects to the configured OIDC provider.

```bash
curl -v http://localhost:8080/auth/oidc
# Returns 302 redirect to OIDC provider
```

### GET /auth/oidc/callback -- Public

OIDC callback. Exchanges the authorization code for a JWT token.

```bash
curl "http://localhost:8080/auth/oidc/callback?code=AUTH_CODE&state=STATE"
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "email": "user@provider.com",
    "name": "User Name"
  }
}
```

### GET /auth/google -- Public

Start Google OAuth login flow. Redirects the user to Google's consent screen.

```bash
curl -v http://localhost:8080/auth/google
# Returns 302 redirect to Google OAuth
```

### POST /auth/google/callback -- Public

OAuth callback. Exchanges the authorization code for a JWT token.

```bash
curl -X POST http://localhost:8080/auth/google/callback \
  -H "Content-Type: application/json" \
  -d '{
    "code": "4/0AX4XfWh...",
    "redirect_uri": "http://localhost:3000/callback"
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | Yes | Authorization code from Google |
| `redirect_uri` | string | Yes | Redirect URI used in the login request |

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "email": "user@example.com",
    "name": "User Name",
    "picture": "https://lh3.googleusercontent.com/..."
  }
}
```

### GET /auth/me -- Protected

Get the currently authenticated user's information.

```bash
curl -H "Authorization: Bearer <JWT>" http://localhost:8080/auth/me
```

**Response:**
```json
{
  "email": "user@example.com",
  "name": "User Name",
  "picture": "https://lh3.googleusercontent.com/..."
}
```

### POST /auth/refresh -- Protected

Refresh the JWT token.

```bash
curl -X POST http://localhost:8080/auth/refresh \
  -H "Authorization: Bearer <JWT>"
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs..."
}
```

---

## WebSocket Routes

### GET /ws/events -- Public

Connect to the CRUD event stream. Auth is performed via the first WebSocket message.

Receives real-time notifications for all entity changes (create, update, delete). Events cover: projects, plans, tasks, steps, decisions, notes, milestones, releases, workspaces, and more.

```bash
wscat -c ws://localhost:8080/ws/events
```

### GET /ws/chat/{session_id} -- Public

WebSocket chat with Claude. Auth is performed via the first WebSocket message.

Send user messages and receive streaming response events (`AssistantText`, `Thinking`, `ToolUse`, `ToolResult`, `Error`, etc.).

```bash
wscat -c ws://localhost:8080/ws/chat/{session_id}
```

See the [Chat & WebSocket Guide](../guides/chat-websocket.md) for details.

---

## Chat Sessions -- Protected

REST endpoints for managing chat sessions. Streaming chat is handled via the WebSocket route above.

### GET /api/chat/sessions -- Protected

List chat sessions.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `project_slug` | string | Filter by project |
| `limit` | integer | Max items |
| `offset` | integer | Items to skip |

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/chat/sessions?project_slug=my-project"
```

### POST /api/chat/sessions -- Protected

Create a new chat session.

```bash
curl -X POST http://localhost:8080/api/chat/sessions \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"project_slug": "my-project"}'
```

### GET /api/chat/sessions/{id} -- Protected

Get session details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/chat/sessions/{id}
```

### DELETE /api/chat/sessions/{id} -- Protected

Delete a session.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/chat/sessions/{id}
```

### GET /api/chat/sessions/{id}/messages -- Protected

List messages in a session.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Max items |
| `offset` | integer | Items to skip |

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/chat/sessions/{id}/messages?limit=50&offset=0"
```

### GET /api/chat/search -- Protected

Search across chat messages.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/chat/search?q=authentication"
```

### POST /api/chat/sessions/backfill-previews -- Protected

Backfill preview data for existing sessions.

```bash
curl -X POST -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/chat/sessions/backfill-previews
```

---

## Health Check

### GET /health -- Public

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

## Version Info

### GET /api/version -- Public

Get server version, build information, and enabled features.

```bash
curl http://localhost:8080/api/version
```

**Response:**
```json
{
  "version": "0.1.0",
  "features": {
    "embedded_frontend": true,
    "nats": true
  }
}
```

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

## Projects

### GET /api/projects -- Protected

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
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/projects?limit=10"
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

### POST /api/projects -- Protected

Create a new project.

```bash
curl -X POST http://localhost:8080/api/projects \
  -H "Authorization: Bearer <JWT>" \
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

### GET /api/projects/{slug} -- Protected

Get project details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/projects/my-project
```

### PATCH /api/projects/{slug} -- Protected

Update a project's name, description, or root_path.

```bash
curl -X PATCH http://localhost:8080/api/projects/my-project \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description", "name": "New Name"}'
```

**Updatable Fields:** `name`, `description`, `root_path`

### DELETE /api/projects/{slug} -- Protected

Delete a project and all associated data.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/projects/my-project
```

### POST /api/projects/{slug}/sync -- Protected

Sync project files to the knowledge graph.

```bash
curl -X POST -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/projects/my-project/sync
```

**Response:**
```json
{
  "files_synced": 127,
  "duration_ms": 1500
}
```

### GET /api/projects/{slug}/plans -- Protected

List plans associated with a project.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/projects/my-project/plans
```

### GET /api/projects/{slug}/code/search -- Protected

Search code within a specific project.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/projects/my-project/code/search?q=authentication&limit=10"
```

### GET /api/projects/{project_id}/roadmap -- Protected

Get aggregated roadmap view.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/projects/{project_id}/roadmap
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

## Workspaces

Workspaces group related projects together for cross-project coordination.

### GET /api/workspaces -- Protected

List all workspaces.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `search` | string | Search in name/description |
| `limit` | integer | Max items |
| `offset` | integer | Items to skip |
| `sort_by` | string | `name` or `created_at` |
| `sort_order` | string | `asc` or `desc` |

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/workspaces?limit=10"
```

### POST /api/workspaces -- Protected

Create a new workspace.

```bash
curl -X POST http://localhost:8080/api/workspaces \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "E-Commerce Platform",
    "description": "Microservices for our e-commerce system",
    "slug": "e-commerce-platform"
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Workspace name |
| `description` | string | No | Workspace description |
| `slug` | string | No | URL-safe identifier (auto-generated) |

### GET /api/workspaces/{slug} -- Protected

Get workspace details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspaces/e-commerce-platform
```

### PATCH /api/workspaces/{slug} -- Protected

Update a workspace.

```bash
curl -X PATCH http://localhost:8080/api/workspaces/e-commerce-platform \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'
```

### DELETE /api/workspaces/{slug} -- Protected

Delete a workspace.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspaces/e-commerce-platform
```

### GET /api/workspaces/{slug}/overview -- Protected

Get workspace overview with projects, milestones, resources, and components.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspaces/e-commerce-platform/overview
```

**Response:**
```json
{
  "workspace": {...},
  "projects": [...],
  "milestones": [...],
  "resources": [...],
  "components": [...]
}
```

### GET /api/workspaces/{slug}/projects -- Protected

List projects in a workspace.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspaces/e-commerce-platform/projects
```

### POST /api/workspaces/{slug}/projects -- Protected

Add a project to a workspace.

```bash
curl -X POST http://localhost:8080/api/workspaces/e-commerce-platform/projects \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "uuid"}'
```

### DELETE /api/workspaces/{slug}/projects/{project_id} -- Protected

Remove a project from a workspace.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspaces/e-commerce-platform/projects/{project_id}
```

---

## Workspace Milestones

Cross-project milestones for coordinating tasks across multiple projects.

### GET /api/workspace-milestones -- Protected

List ALL workspace milestones across all workspaces.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `workspace_id` | uuid | Filter by workspace |
| `status` | string | `planned`, `open`, `in_progress`, `completed`, `closed` |
| `limit` | integer | Max items |
| `offset` | integer | Items to skip |

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/workspace-milestones?status=open"
```

### GET /api/workspaces/{slug}/milestones -- Protected

List workspace milestones for a specific workspace.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/workspaces/e-commerce-platform/milestones?status=open"
```

### POST /api/workspaces/{slug}/milestones -- Protected

Create a workspace milestone.

```bash
curl -X POST http://localhost:8080/api/workspaces/e-commerce-platform/milestones \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Q1 Launch",
    "description": "Cross-project launch milestone",
    "target_date": "2024-03-31T00:00:00Z",
    "tags": ["launch", "q1"]
  }'
```

### GET /api/workspace-milestones/{milestone_id} -- Protected

Get workspace milestone details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspace-milestones/{milestone_id}
```

### PATCH /api/workspace-milestones/{milestone_id} -- Protected

Update a workspace milestone.

```bash
curl -X PATCH http://localhost:8080/api/workspace-milestones/{milestone_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"status": "closed"}'
```

### DELETE /api/workspace-milestones/{milestone_id} -- Protected

Delete a workspace milestone.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspace-milestones/{milestone_id}
```

### POST /api/workspace-milestones/{milestone_id}/tasks -- Protected

Add a task from any project to a workspace milestone.

```bash
curl -X POST http://localhost:8080/api/workspace-milestones/{milestone_id}/tasks \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "uuid"}'
```

### GET /api/workspace-milestones/{milestone_id}/progress -- Protected

Get aggregated progress across all projects.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspace-milestones/{milestone_id}/progress
```

**Response:**
```json
{
  "total": 12,
  "completed": 8,
  "in_progress": 2,
  "pending": 2,
  "percentage": 66.7
}
```

---

## Resources

Shared contracts, schemas, and specifications referenced by multiple projects.

### GET /api/workspaces/{slug}/resources -- Protected

List resources in a workspace.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/workspaces/e-commerce-platform/resources?resource_type=api_contract"
```

### POST /api/workspaces/{slug}/resources -- Protected

Create a shared resource.

```bash
curl -X POST http://localhost:8080/api/workspaces/e-commerce-platform/resources \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "User API",
    "resource_type": "api_contract",
    "file_path": "specs/openapi/users.yaml",
    "format": "openapi",
    "version": "1.0.0",
    "description": "User service API contract"
  }'
```

**Resource Types:** `api_contract`, `protobuf`, `graphql_schema`, `json_schema`, `database_schema`, `shared_types`, `config`, `documentation`, `other`

### GET /api/resources/{resource_id} -- Protected

Get resource details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/resources/{resource_id}
```

### PATCH /api/resources/{resource_id} -- Protected

Update a resource.

```bash
curl -X PATCH http://localhost:8080/api/resources/{resource_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"version": "2.0.0"}'
```

### DELETE /api/resources/{resource_id} -- Protected

Delete a resource.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/resources/{resource_id}
```

### POST /api/resources/{resource_id}/projects -- Protected

Link a project to a resource as implementer or consumer.

```bash
curl -X POST http://localhost:8080/api/resources/{resource_id}/projects \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "uuid",
    "relationship": "implements"
  }'
```

**Relationship Values:** `implements` (provider), `uses` (consumer)

---

## Components & Topology

Model deployment architecture with components and their dependencies.

### GET /api/workspaces/{slug}/components -- Protected

List components in a workspace.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/workspaces/e-commerce-platform/components?component_type=service"
```

### POST /api/workspaces/{slug}/components -- Protected

Create a deployment component.

```bash
curl -X POST http://localhost:8080/api/workspaces/e-commerce-platform/components \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API Gateway",
    "component_type": "gateway",
    "description": "Main entry point for all API requests",
    "runtime": "kubernetes",
    "config": {"port": 8080, "replicas": 3},
    "tags": ["infrastructure", "gateway"]
  }'
```

**Component Types:** `service`, `frontend`, `worker`, `database`, `message_queue`, `cache`, `gateway`, `external`, `other`

### GET /api/components/{component_id} -- Protected

Get component details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/components/{component_id}
```

### PATCH /api/components/{component_id} -- Protected

Update a component.

```bash
curl -X PATCH http://localhost:8080/api/components/{component_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"runtime": "docker"}'
```

### DELETE /api/components/{component_id} -- Protected

Delete a component.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/components/{component_id}
```

### POST /api/components/{component_id}/dependencies -- Protected

Add a dependency between components.

```bash
curl -X POST http://localhost:8080/api/components/{component_id}/dependencies \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "depends_on_id": "uuid",
    "protocol": "http",
    "required": true
  }'
```

### DELETE /api/components/{component_id}/dependencies/{dep_id} -- Protected

Remove a component dependency.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/components/{component_id}/dependencies/{dep_id}
```

### PUT /api/components/{component_id}/project -- Protected

Map a component to its source code project.

```bash
curl -X PUT http://localhost:8080/api/components/{component_id}/project \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "uuid"}'
```

### GET /api/workspaces/{slug}/topology -- Protected

Get full deployment topology graph.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/workspaces/e-commerce-platform/topology
```

**Response:**
```json
{
  "components": [
    {
      "id": "uuid",
      "name": "API Gateway",
      "component_type": "gateway",
      "project_name": "api-gateway",
      "dependencies": [
        {"id": "uuid", "name": "User Service", "protocol": "http", "required": true}
      ]
    }
  ]
}
```

---

## Plans

### GET /api/plans -- Protected

List all plans.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Comma-separated: `draft,approved,in_progress,completed,cancelled` |
| `priority_min` | integer | Minimum priority |
| `priority_max` | integer | Maximum priority |
| `search` | string | Search in title/description |

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/plans?status=in_progress&limit=10"
```

### POST /api/plans -- Protected

Create a new plan.

```bash
curl -X POST http://localhost:8080/api/plans \
  -H "Authorization: Bearer <JWT>" \
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

### GET /api/plans/{plan_id} -- Protected

Get plan details with tasks, constraints, and decisions.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}
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

### PATCH /api/plans/{plan_id} -- Protected

Update plan status.

```bash
curl -X PATCH http://localhost:8080/api/plans/{plan_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"status": "in_progress"}'
```

**Status Values:** `draft`, `approved`, `in_progress`, `completed`, `cancelled`

### DELETE /api/plans/{plan_id} -- Protected

Delete a plan and all related data (tasks, steps, decisions, constraints).

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}
```

### PUT /api/plans/{plan_id}/project -- Protected

Link plan to a project.

```bash
curl -X PUT http://localhost:8080/api/plans/{plan_id}/project \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "uuid"}'
```

### DELETE /api/plans/{plan_id}/project -- Protected

Unlink plan from project.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/project
```

### GET /api/plans/{plan_id}/next-task -- Protected

Get next available task (unblocked, highest priority).

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/next-task
```

### GET /api/plans/{plan_id}/dependency-graph -- Protected

Get task dependency graph for visualization.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/dependency-graph
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

### GET /api/plans/{plan_id}/critical-path -- Protected

Get longest dependency chain.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/critical-path
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

### GET /api/tasks -- Protected

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
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/tasks?status=in_progress&assigned_to=agent-1"
```

### POST /api/plans/{plan_id}/tasks -- Protected

Add a task to a plan.

```bash
curl -X POST http://localhost:8080/api/plans/{plan_id}/tasks \
  -H "Authorization: Bearer <JWT>" \
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

### GET /api/tasks/{task_id} -- Protected

Get task details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}
```

### PATCH /api/tasks/{task_id} -- Protected

Update a task.

```bash
curl -X PATCH http://localhost:8080/api/tasks/{task_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "in_progress",
    "assigned_to": "agent-1"
  }'
```

**Updatable Fields:** `status`, `assigned_to`, `priority`, `tags`

**Status Values:** `pending`, `in_progress`, `blocked`, `completed`, `failed`

### DELETE /api/tasks/{task_id} -- Protected

Delete a task and all its steps and decisions.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}
```

### POST /api/tasks/{task_id}/dependencies -- Protected

Add dependencies to a task.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/dependencies \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"dependency_ids": ["uuid1", "uuid2"]}'
```

### DELETE /api/tasks/{task_id}/dependencies/{dep_id} -- Protected

Remove a dependency.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}/dependencies/{dep_id}
```

### GET /api/tasks/{task_id}/blockers -- Protected

Get tasks blocking this task (uncompleted dependencies).

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}/blockers
```

### GET /api/tasks/{task_id}/blocking -- Protected

Get tasks blocked by this task.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}/blocking
```

---

## Task Context

### GET /api/plans/{plan_id}/tasks/{task_id}/context -- Protected

Get full context for a task (for agent execution).

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/tasks/{task_id}/context
```

### GET /api/plans/{plan_id}/tasks/{task_id}/prompt -- Protected

Get generated prompt for a task.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/tasks/{task_id}/prompt
```

---

## Steps

### GET /api/tasks/{task_id}/steps -- Protected

List steps for a task.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}/steps
```

### POST /api/tasks/{task_id}/steps -- Protected

Add a step to a task.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/steps \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Setup JWT library",
    "verification": "Can generate tokens"
  }'
```

### GET /api/steps/{step_id} -- Protected

Get a step by ID.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/steps/{step_id}
```

### PATCH /api/steps/{step_id} -- Protected

Update step status.

```bash
curl -X PATCH http://localhost:8080/api/steps/{step_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

**Status Values:** `pending`, `in_progress`, `completed`, `skipped`

### DELETE /api/steps/{step_id} -- Protected

Delete a step.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/steps/{step_id}
```

### GET /api/tasks/{task_id}/steps/progress -- Protected

Get step completion progress.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}/steps/progress
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

### GET /api/plans/{plan_id}/constraints -- Protected

List plan constraints.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/constraints
```

### POST /api/plans/{plan_id}/constraints -- Protected

Add a constraint.

```bash
curl -X POST http://localhost:8080/api/plans/{plan_id}/constraints \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "constraint_type": "security",
    "description": "No plaintext passwords",
    "severity": "critical"
  }'
```

**Constraint Types:** `performance`, `security`, `compatibility`, `style`, `testing`, `other`

**Severity Levels:** `low`, `medium`, `high`, `critical`

### GET /api/constraints/{constraint_id} -- Protected

Get constraint details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/constraints/{constraint_id}
```

### PATCH /api/constraints/{constraint_id} -- Protected

Update a constraint.

```bash
curl -X PATCH http://localhost:8080/api/constraints/{constraint_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated constraint description"}'
```

**Updatable Fields:** `description`, `constraint_type`, `enforced_by`

### DELETE /api/constraints/{constraint_id} -- Protected

Delete a constraint.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/constraints/{constraint_id}
```

---

## Decisions

### POST /api/tasks/{task_id}/decisions -- Protected

Record a decision.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/decisions \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Use JWT for authentication",
    "rationale": "Stateless, better for horizontal scaling",
    "alternatives": ["Session cookies", "OAuth tokens"],
    "chosen_option": "JWT with refresh tokens"
  }'
```

### GET /api/decisions/{decision_id} -- Protected

Get decision details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/decisions/{decision_id}
```

### PATCH /api/decisions/{decision_id} -- Protected

Update a decision.

```bash
curl -X PATCH http://localhost:8080/api/decisions/{decision_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"rationale": "Updated rationale"}'
```

**Updatable Fields:** `description`, `rationale`, `chosen_option`

### DELETE /api/decisions/{decision_id} -- Protected

Delete a decision.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/decisions/{decision_id}
```

### GET /api/decisions/search -- Protected

Search past decisions.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/decisions/search?q=authentication&limit=10"
```

---

## Releases

### GET /api/projects/{project_id}/releases -- Protected

List releases for a project.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/projects/{project_id}/releases?status=planned"
```

### POST /api/projects/{project_id}/releases -- Protected

Create a release.

```bash
curl -X POST http://localhost:8080/api/projects/{project_id}/releases \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0.0",
    "title": "Initial Release",
    "description": "First production release",
    "target_date": "2024-03-01T00:00:00Z"
  }'
```

### GET /api/releases/{release_id} -- Protected

Get release details with tasks and commits.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/releases/{release_id}
```

### PATCH /api/releases/{release_id} -- Protected

Update a release.

```bash
curl -X PATCH http://localhost:8080/api/releases/{release_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"status": "released", "released_at": "2024-03-01T12:00:00Z"}'
```

**Status Values:** `planned`, `in_progress`, `released`, `cancelled`

### DELETE /api/releases/{release_id} -- Protected

Delete a release.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/releases/{release_id}
```

### POST /api/releases/{release_id}/tasks -- Protected

Add task to release.

```bash
curl -X POST http://localhost:8080/api/releases/{release_id}/tasks \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "uuid"}'
```

### POST /api/releases/{release_id}/commits -- Protected

Add commit to release.

```bash
curl -X POST http://localhost:8080/api/releases/{release_id}/commits \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"commit_sha": "abc123"}'
```

---

## Milestones

### GET /api/projects/{project_id}/milestones -- Protected

List milestones for a project.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/projects/{project_id}/milestones?status=open"
```

### POST /api/projects/{project_id}/milestones -- Protected

Create a milestone.

```bash
curl -X POST http://localhost:8080/api/projects/{project_id}/milestones \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "MVP Complete",
    "description": "Minimum viable product ready",
    "target_date": "2024-02-15T00:00:00Z"
  }'
```

### GET /api/milestones/{milestone_id} -- Protected

Get milestone details with tasks.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/milestones/{milestone_id}
```

### PATCH /api/milestones/{milestone_id} -- Protected

Update a milestone.

```bash
curl -X PATCH http://localhost:8080/api/milestones/{milestone_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"status": "closed"}'
```

**Status Values:** `open`, `closed`

### DELETE /api/milestones/{milestone_id} -- Protected

Delete a milestone.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/milestones/{milestone_id}
```

### POST /api/milestones/{milestone_id}/tasks -- Protected

Add task to milestone.

```bash
curl -X POST http://localhost:8080/api/milestones/{milestone_id}/tasks \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "uuid"}'
```

### GET /api/milestones/{milestone_id}/progress -- Protected

Get milestone completion progress.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/milestones/{milestone_id}/progress
```

---

## Commits

### POST /api/commits -- Protected

Register a commit.

```bash
curl -X POST http://localhost:8080/api/commits \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "sha": "abc123def456",
    "message": "feat: add authentication",
    "author": "Developer",
    "files_changed": ["src/auth.rs", "src/lib.rs"]
  }'
```

### GET /api/tasks/{task_id}/commits -- Protected

Get commits linked to a task.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/tasks/{task_id}/commits
```

### POST /api/tasks/{task_id}/commits -- Protected

Link commit to task.

```bash
curl -X POST http://localhost:8080/api/tasks/{task_id}/commits \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"commit_sha": "abc123"}'
```

### GET /api/plans/{plan_id}/commits -- Protected

Get commits linked to a plan.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/plans/{plan_id}/commits
```

### POST /api/plans/{plan_id}/commits -- Protected

Link commit to plan.

```bash
curl -X POST http://localhost:8080/api/plans/{plan_id}/commits \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"commit_sha": "abc123"}'
```

---

## Code Exploration

### GET /api/code/search -- Protected

Semantic code search.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/search?q=error+handling&limit=10&language=rust"
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

### GET /api/code/symbols/{file_path} -- Protected

Get symbols in a file.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/symbols/src%2Flib.rs"
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

### GET /api/code/references -- Protected

Find all references to a symbol.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/references?symbol=AppState&limit=20"
```

### GET /api/code/dependencies/{file_path} -- Protected

Get file imports and dependents.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/dependencies/src%2Flib.rs"
```

**Response:**
```json
{
  "imports": ["src/config.rs", "src/db.rs"],
  "imported_by": ["src/main.rs", "src/api/mod.rs"],
  "impact_radius": 5
}
```

### GET /api/code/callgraph -- Protected

Get function call graph.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/callgraph?function=handle_request&depth=2&direction=both"
```

### GET /api/code/impact -- Protected

Analyze change impact.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/impact?target=src/models/user.rs&target_type=file"
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

### GET /api/code/architecture -- Protected

Get codebase architecture overview.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/code/architecture
```

### POST /api/code/similar -- Protected

Find similar code.

```bash
curl -X POST http://localhost:8080/api/code/similar \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"snippet": "async fn handle_error", "limit": 5}'
```

### GET /api/code/trait-impls -- Protected

Find trait implementations.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/trait-impls?trait_name=Handler&limit=10"
```

### GET /api/code/type-traits -- Protected

Find traits implemented by a type.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/type-traits?type_name=AppState"
```

### GET /api/code/impl-blocks -- Protected

Get impl blocks for a type.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/code/impl-blocks?type_name=Orchestrator"
```

---

## Notes

Knowledge Notes capture contextual knowledge about your codebase. See the [Knowledge Notes Guide](../guides/knowledge-notes.md) for detailed usage.

### GET /api/notes -- Protected

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
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/notes?note_type=guideline&status=active&limit=20"
```

### POST /api/notes -- Protected

Create a new note.

```bash
curl -X POST http://localhost:8080/api/notes \
  -H "Authorization: Bearer <JWT>" \
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

### GET /api/notes/{note_id} -- Protected

Get note details.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/notes/{note_id}
```

### PATCH /api/notes/{note_id} -- Protected

Update a note.

```bash
curl -X PATCH http://localhost:8080/api/notes/{note_id} \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Updated content",
    "importance": "critical"
  }'
```

**Updatable Fields:** `content`, `importance`, `status`, `tags`

### DELETE /api/notes/{note_id} -- Protected

Delete a note.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/notes/{note_id}
```

### GET /api/notes/search -- Protected

Semantic search across notes.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/notes/search?q=error+handling&limit=10"
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

### GET /api/notes/context -- Protected

Get contextual notes for an entity (direct + propagated through graph).

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/notes/context?entity_type=file&entity_id=src/auth/jwt.rs&max_depth=2"
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

### GET /api/notes/propagated -- Protected

Get notes propagated through the graph (not directly attached to the entity).

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/notes/propagated?entity_type=function&entity_id=validate_token&max_depth=3"
```

### GET /api/notes/needs-review -- Protected

Get notes needing human review (stale or needs_review status).

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/notes/needs-review
```

### POST /api/notes/update-staleness -- Protected

Recalculate staleness scores for all notes.

```bash
curl -X POST -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/notes/update-staleness
```

### POST /api/notes/{note_id}/confirm -- Protected

Confirm a note is still valid (resets staleness).

```bash
curl -X POST -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/notes/{note_id}/confirm
```

### POST /api/notes/{note_id}/invalidate -- Protected

Mark a note as obsolete.

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/invalidate \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Auth system was refactored to OAuth"}'
```

### POST /api/notes/{note_id}/supersede -- Protected

Replace a note with a new one (preserves history).

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/supersede \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "note_type": "guideline",
    "content": "Use OAuth tokens instead of JWT",
    "importance": "high"
  }'
```

### POST /api/notes/{note_id}/links -- Protected

Link a note to a code entity.

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/links \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "function",
    "entity_id": "validate_user"
  }'
```

**Entity Types:** `file`, `function`, `struct`, `trait`, `module`, `task`, `plan`

### DELETE /api/notes/{note_id}/links/{entity_type}/{entity_id} -- Protected

Remove a link between a note and entity.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/notes/{note_id}/links/file/src%2Fauth.rs
```

### GET /api/entities/{entity_type}/{entity_id}/notes -- Protected

Get notes directly attached to an entity.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/entities/function/validate_token/notes
```

### GET /api/projects/{project_id}/notes -- Protected

List notes for a specific project.

```bash
curl -H "Authorization: Bearer <JWT>" \
  "http://localhost:8080/api/projects/{project_id}/notes?status=active"
```

---

## Sync & Watch

### POST /api/sync -- Protected

Manually sync a directory.

```bash
curl -X POST http://localhost:8080/api/sync \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project", "project_id": "optional-uuid"}'
```

### GET /api/watch -- Protected

Get file watcher status.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/watch
```

### POST /api/watch -- Protected

Start file watcher.

```bash
curl -X POST http://localhost:8080/api/watch \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project"}'
```

### DELETE /api/watch -- Protected

Stop file watcher.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/watch
```

---

## Webhooks & Internal

### POST /api/wake -- Protected

Agent completion webhook (protected variant).

```bash
curl -X POST http://localhost:8080/api/wake \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "uuid",
    "success": true,
    "summary": "Implemented authentication",
    "files_modified": ["src/auth.rs"]
  }'
```

### POST /hooks/wake -- Public

Agent completion webhook (public variant, no auth required).

```bash
curl -X POST http://localhost:8080/hooks/wake \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "uuid",
    "success": true,
    "summary": "Implemented authentication",
    "files_modified": ["src/auth.rs"]
  }'
```

### POST /internal/events -- Public

Internal event receiver. Used for inter-service communication.

```bash
curl -X POST http://localhost:8080/internal/events \
  -H "Content-Type: application/json" \
  -d '{"event_type": "task_completed", "payload": {...}}'
```

---

## Meilisearch Maintenance

### GET /api/meilisearch/stats -- Protected

Get code index statistics.

```bash
curl -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/meilisearch/stats
```

### DELETE /api/meilisearch/orphans -- Protected

Delete documents without project_id.

```bash
curl -X DELETE -H "Authorization: Bearer <JWT>" \
  http://localhost:8080/api/meilisearch/orphans
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
| `UNAUTHORIZED` | 401 | Missing or invalid JWT token |
| `FORBIDDEN` | 403 | Token valid but insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `CONFLICT` | 409 | Resource already exists |
| `INTERNAL_ERROR` | 500 | Server error |
