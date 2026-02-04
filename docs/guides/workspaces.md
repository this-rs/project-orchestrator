# Workspaces Guide

Workspaces enable coordination across multiple related projects. Use them when you have projects that share context, contracts, or need synchronized milestones.

---

## What is a Workspace?

A workspace is a container that groups related projects together. It provides:

- **Shared Context** — Notes and guidelines that apply across all projects
- **Cross-Project Milestones** — Coordinate tasks from different projects toward common goals
- **Resources** — Reference shared contracts, schemas, and specifications
- **Deployment Topology** — Model how your services connect and depend on each other

---

## When to Use Workspaces

**Good use cases:**
- Microservices architecture (API, workers, databases)
- Full-stack applications (frontend + backend + mobile)
- Multi-package monorepos
- Projects sharing API contracts or protocols

**You don't need a workspace for:**
- Single standalone projects
- Projects with no shared context
- Independent tools that don't interact

---

## Quick Start

### 1. Create a Workspace

```
Create a workspace named "E-Commerce Platform" for our microservices
```

### 2. Add Projects

```
Add project "api-service" to workspace "e-commerce-platform"
Add project "web-frontend" to workspace "e-commerce-platform"
Add project "mobile-app" to workspace "e-commerce-platform"
```

### 3. Create Shared Resources

```
Create an API contract "User API" in workspace "e-commerce-platform"
pointing to specs/openapi/users.yaml
```

### 4. Link Projects to Resources

```
Link api-service as implementer of "User API"
Link web-frontend as consumer of "User API"
Link mobile-app as consumer of "User API"
```

### 5. Create Cross-Project Milestones

```
Create workspace milestone "Q1 Launch" with target March 31st
Add the "implement OAuth" task from api-service
Add the "login flow" task from web-frontend
Add the "biometric auth" task from mobile-app
```

---

## Workspace Overview

Get a bird's-eye view of your workspace:

```
Show me the overview for workspace "e-commerce-platform"
```

This returns:
- All projects in the workspace
- Open milestones with progress
- Shared resources
- Deployment components

---

## Resources

Resources are references to shared contracts and specifications. They're not parsed — just documented pointers to files.

### Resource Types

| Type | Description |
|------|-------------|
| `api_contract` | OpenAPI, Swagger, RAML specs |
| `protobuf` | Protocol buffer definitions |
| `graphql_schema` | GraphQL schemas |
| `json_schema` | JSON Schema definitions |
| `database_schema` | Database migrations, DDL |
| `shared_types` | Shared type definitions |
| `config` | Shared configuration files |
| `documentation` | Technical documentation |

### Provider vs Consumer

Projects can have two relationships with a resource:

- **Implements** — The project provides/exposes this contract (e.g., the API server)
- **Uses** — The project consumes this contract (e.g., frontend clients)

```
Link api-service as implementer of "User API"
Link web-frontend as consumer of "User API"
```

This helps understand impact when a contract changes.

---

## Deployment Topology

Model how your services connect in production.

### Component Types

| Type | Description |
|------|-------------|
| `service` | Backend service/API |
| `frontend` | Web or mobile frontend |
| `worker` | Background job processor |
| `database` | Database instance |
| `message_queue` | Message broker (RabbitMQ, Kafka) |
| `cache` | Caching layer (Redis, Memcached) |
| `gateway` | API gateway, load balancer |
| `external` | External third-party service |

### Creating a Topology

```
# Create components
Create a service component "API Gateway" running on Kubernetes
Create a service component "User Service" running on Kubernetes
Create a database component "PostgreSQL"
Create a cache component "Redis"

# Define dependencies
API Gateway depends on User Service via HTTP
User Service depends on PostgreSQL via postgres
User Service depends on Redis via redis

# Map to source code
Map "User Service" component to project "user-api"
```

### Visualizing Topology

```
Show the deployment topology for workspace "e-commerce-platform"
```

Returns a graph of all components with their dependencies and protocols.

---

## Workspace Milestones

Workspace milestones coordinate tasks from multiple projects.

### Create a Milestone

```
Create workspace milestone "MVP Release" for workspace "e-commerce-platform"
with target date February 28th
```

### Add Tasks from Any Project

```
Add task "implement authentication" from api-service to milestone "MVP Release"
Add task "create login page" from web-frontend to milestone "MVP Release"
Add task "setup CI/CD" from devops to milestone "MVP Release"
```

### Track Cross-Project Progress

```
What's the progress on workspace milestone "MVP Release"?
```

Returns progress broken down by project:
```
Total: 12 tasks
Completed: 8 (67%)
In Progress: 2
Pending: 2

By Project:
- api-service: 4/5 complete
- web-frontend: 3/4 complete
- devops: 1/3 complete
```

---

## Notes and Knowledge Sharing

Notes created at the workspace level automatically propagate to all projects in the workspace.

### Create Workspace-Level Notes

```
Create a guideline note in workspace "e-commerce-platform":
"All services must use structured logging with correlation IDs"
```

This note will appear in the context for all projects in the workspace, with a slightly lower relevance score (0.8x) than project-specific notes.

### Note Propagation

When you get notes for a project in a workspace:

1. **Direct notes** — Notes attached directly to the project (score: 1.0)
2. **Workspace notes** — Notes from the parent workspace (score: 0.8)
3. **Propagated notes** — Notes from related code entities

---

## Best Practices

### Workspace Organization

1. **One workspace per product/system** — Group all related services together
2. **Keep resources up to date** — Update file paths when specs move
3. **Use meaningful component names** — Match your actual infrastructure

### Milestones

1. **Cross-project only** — Use workspace milestones for coordination, project milestones for single-project work
2. **Clear acceptance criteria** — Define what "done" means across projects
3. **Regular progress checks** — Monitor cross-project velocity

### Topology

1. **Model reality** — Reflect your actual deployment architecture
2. **Include external services** — Document third-party dependencies
3. **Specify protocols** — Help understand integration points

---

## Example: Microservices Setup

```
# Create workspace
Create workspace "Order System" for our e-commerce microservices

# Add projects
Add project "order-api" to workspace
Add project "inventory-service" to workspace
Add project "payment-gateway" to workspace
Add project "notification-service" to workspace

# Create shared resources
Create API contract "Order Events" at specs/events/orders.proto (protobuf)
Create API contract "Payment API" at specs/openapi/payments.yaml (openapi)

# Link resources
Link order-api as implementer of "Order Events"
Link inventory-service as consumer of "Order Events"
Link notification-service as consumer of "Order Events"
Link payment-gateway as implementer of "Payment API"
Link order-api as consumer of "Payment API"

# Create topology
Create service "Order API" (kubernetes)
Create service "Inventory Service" (kubernetes)
Create service "Payment Gateway" (kubernetes)
Create service "Notification Service" (kubernetes)
Create database "Orders DB" (postgresql)
Create message_queue "Event Bus" (kafka)

# Define dependencies
Order API depends on Orders DB
Order API depends on Event Bus
Order API depends on Payment Gateway via HTTP
Inventory Service depends on Event Bus
Notification Service depends on Event Bus

# Map to projects
Map "Order API" to project order-api
Map "Inventory Service" to project inventory-service
Map "Payment Gateway" to project payment-gateway
Map "Notification Service" to project notification-service

# Create cross-project milestone
Create workspace milestone "Inventory Sync Feature" with target March 15th
Add task "emit inventory events" from order-api
Add task "consume inventory events" from inventory-service
Add task "low stock alerts" from notification-service
```

---

## API Reference

See [MCP Tools Reference](../api/mcp-tools.md) for complete tool documentation:

- Workspace Management (9 tools)
- Workspace Milestones (6 tools)
- Resources (6 tools)
- Components & Topology (8 tools)
