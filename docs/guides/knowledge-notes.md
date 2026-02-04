# Knowledge Notes System

A system for capturing, organizing, and propagating contextual knowledge across your codebase.

---

## Overview

Knowledge Notes capture the tacit knowledge that emerges from development conversations - guidelines, gotchas, patterns, and tips that don't belong in code comments but are crucial for maintaining code quality.

Unlike traditional documentation:
- **Notes are anchored to code entities** (functions, files, modules), not line numbers
- **Notes propagate through the code graph** via imports, calls, and contains relationships
- **Notes have a lifecycle** - they're automatically flagged when the anchored code changes
- **Notes are scored for relevance** - agents receive the most relevant notes automatically

---

## Note Types

| Type | Description | Use Case |
|------|-------------|----------|
| `guideline` | Rules to follow | "All API handlers must validate input with the `validate` crate" |
| `gotcha` | Pitfalls to avoid | "Don't use `unwrap()` in this module - it's called from async contexts" |
| `pattern` | Established patterns | "Error handling in this service uses the Result<T, ServiceError> pattern" |
| `context` | Temporary context | "Currently refactoring auth - don't add new dependencies to old auth module" |
| `tip` | Useful advice | "Use `cargo expand` to debug macro issues in this crate" |
| `observation` | General observations | "This module has high coupling with the database layer" |
| `assertion` | Verifiable rules | "Function `validate_user` must exist in src/auth/" |

---

## Note Lifecycle

Notes progress through these statuses:

```
┌─────────┐     code changed      ┌─────────────┐
│ Active  │ ──────────────────────> NeedsReview │
└─────────┘                       └─────────────┘
     │                                   │
     │ time passes                       │ confirmed invalid
     v                                   v
┌─────────┐                       ┌──────────┐
│  Stale  │ ─────────────────────>│ Obsolete │
└─────────┘   marked obsolete     └──────────┘
                                        │
                                        │ archived
                                        v
                                  ┌──────────┐
                                  │ Archived │
                                  └──────────┘
```

### Status Descriptions

- **Active**: Note is valid and will be shown to agents
- **NeedsReview**: Anchored code changed; human should verify the note still applies
- **Stale**: Note hasn't been confirmed for a long time (based on note type)
- **Obsolete**: Note was explicitly marked as no longer valid
- **Archived**: Note is preserved for history but never shown

---

## Importance Levels

| Level | Description | Staleness Decay |
|-------|-------------|-----------------|
| `critical` | Must not be ignored | Slowest decay (0.5x) |
| `high` | Very important | Slow decay (0.7x) |
| `medium` | Standard importance | Normal decay (1.0x) |
| `low` | Nice to know | Fast decay (1.3x) |

---

## Staleness Calculation

Notes become stale over time based on their type:

| Note Type | Base Decay (days) | Description |
|-----------|-------------------|-------------|
| `context` | 30 | Expires quickly - temporary by nature |
| `tip` | 90 | Medium lifetime |
| `observation` | 90 | Medium lifetime |
| `gotcha` | 180 | Longer lifetime - pitfalls persist |
| `guideline` | 365 | Very stable - guidelines rarely change |
| `pattern` | 365 | Very stable - patterns are established |
| `assertion` | Never | Verified by code, not time |

The staleness score is calculated as:

```
staleness = (1 - exp(-days_since_activity / base_decay)) × importance_factor
```

When staleness exceeds 0.8, notes automatically transition to `stale` status.

---

## Creating Notes

### Via MCP Tool (Claude Code)

```
Create a guideline note for the auth module:
"All authentication functions must log failed attempts to the security audit log"
```

The `create_note` tool will be called with:
- `project_id`: Your project UUID
- `note_type`: "guideline"
- `content`: The note text
- `importance`: "high" (default: medium)
- `tags`: ["auth", "security", "logging"]

### Via REST API

```bash
curl -X POST http://localhost:8080/api/notes \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "your-project-uuid",
    "note_type": "guideline",
    "content": "All authentication functions must log failed attempts",
    "importance": "high",
    "tags": ["auth", "security"]
  }'
```

---

## Linking Notes to Code

Notes can be linked to any code entity in the knowledge graph:

### Entity Types

- `file` - A source file
- `function` - A function or method
- `struct` - A struct or class
- `trait` - A trait or interface
- `module` - A module/package
- `task` - A task in a plan
- `plan` - A development plan

### Via MCP Tool

```
Link note abc123 to the validate_user function
```

### Via REST API

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/links \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "function",
    "entity_id": "validate_user"
  }'
```

---

## Note Propagation

Notes automatically propagate through the code graph based on relationships:

### Propagation Paths

1. **CONTAINS (up)**: Note on a file applies to functions within it
2. **CONTAINS (down)**: Note on a function propagates up to its file
3. **IMPORTS**: Notes propagate to importing/imported files
4. **CALLS**: Notes propagate along the call graph

### Relevance Scoring

When notes propagate, their relevance score decreases:

```
score = (1 / distance) × type_weight × freshness × importance
```

Where:
- `distance`: Number of hops in the graph (1, 2, 3...)
- `type_weight`: Weight based on relationship type
- `freshness`: Time decay since creation
- `importance`: Note importance level

### Example

```
Note: "Always validate JWT tokens before accessing user data"
Attached to: validate_token() function

Propagation:
├── validate_token() [direct, score: 1.0]
├── auth/jwt.rs [contains, score: 0.8]
│   ├── get_current_user() [calls validate_token, score: 0.6]
│   └── middleware.rs [imports jwt.rs, score: 0.5]
└── user_handler() [calls get_current_user, score: 0.4]
```

---

## Retrieving Contextual Notes

### Get Notes for an Entity

```bash
# Get all notes relevant to a file (direct + propagated)
curl "http://localhost:8080/api/notes/context?entity_type=file&entity_id=src/auth/jwt.rs"
```

### Response

```json
{
  "direct_notes": [
    {
      "id": "uuid",
      "note_type": "guideline",
      "content": "All JWT operations must use the configured secret",
      "importance": "high",
      "relevance_score": 1.0
    }
  ],
  "propagated_notes": [
    {
      "note": { ... },
      "source_entity": "src/auth/mod.rs",
      "relevance_score": 0.7,
      "propagation_path": ["CONTAINS", "auth/mod.rs"]
    }
  ]
}
```

---

## Automatic Integration

### In Task Context

When an agent requests task context via `get_task_context`, relevant notes are automatically included:

```json
{
  "task": { ... },
  "constraints": [ ... ],
  "decisions": [ ... ],
  "notes": [
    {
      "id": "uuid",
      "note_type": "gotcha",
      "content": "This module uses async-std, not tokio",
      "importance": "high",
      "source_entity": "src/network/mod.rs",
      "propagated": true,
      "relevance_score": 0.8
    }
  ]
}
```

Notes are collected from:
1. Files affected by the task
2. The task itself
3. The parent plan
4. Related decisions

---

## Managing Note Lifecycle

### Confirm a Note

Reset the staleness score when you verify a note is still valid:

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/confirm
```

### Invalidate a Note

Mark a note as obsolete with a reason:

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/invalidate \
  -H "Content-Type: application/json" \
  -d '{"reason": "Auth system was refactored to use OAuth"}'
```

### Supersede a Note

Replace an old note with a new one (preserves history):

```bash
curl -X POST http://localhost:8080/api/notes/{note_id}/supersede \
  -H "Content-Type: application/json" \
  -d '{
    "note_type": "guideline",
    "content": "Use OAuth tokens instead of JWT for authentication",
    "importance": "high"
  }'
```

---

## Searching Notes

### Semantic Search

```bash
curl "http://localhost:8080/api/notes/search?q=authentication+security&limit=10"
```

### With Filters

```bash
curl "http://localhost:8080/api/notes/search?q=error+handling&note_type=pattern&status=active"
```

### Via MCP Tool

```
Search for notes about error handling patterns
```

---

## Reviewing Stale Notes

### List Notes Needing Review

```bash
curl "http://localhost:8080/api/notes/needs-review"
```

Returns notes that are:
- Status `needs_review` (code changed)
- Status `stale` (time-based)
- High staleness score (> 0.7)

### Update Staleness Scores

Trigger a recalculation of all staleness scores:

```bash
curl -X POST http://localhost:8080/api/notes/update-staleness"
```

---

## Assertions

Assertions are special notes that can be automatically verified during code sync.

### Assertion Types

| Type | Description | Example |
|------|-------------|---------|
| `exists` | Entity must exist | Function `validate_user` exists in `src/auth/` |
| `signature_contains` | Signature must contain | `process_payment` returns `Result<Payment, Error>` |
| `depends_on` | Import relationship | `handlers.rs` imports `auth/mod.rs` |
| `calls` | Call relationship | `create_user` calls `validate_email` |
| `no_call_to` | Must not call | `public_api` must not call `internal_db_query` |

### Creating an Assertion

```bash
curl -X POST http://localhost:8080/api/notes \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "uuid",
    "note_type": "assertion",
    "content": "The validate_user function must exist",
    "assertion_rule": {
      "check_type": "exists",
      "target": "function:validate_user",
      "file_pattern": "src/auth/*.rs"
    }
  }'
```

### Verification

Assertions are automatically verified during `sync_project`. If an assertion fails:
1. The note status changes to `needs_review`
2. A verification result is recorded
3. The failure appears in the sync report

---

## Best Practices

### When to Create Notes

1. **After debugging sessions** - Capture the gotchas you discovered
2. **During code reviews** - Document patterns and guidelines
3. **When onboarding** - Record context that's not in docs
4. **After refactoring** - Update notes for changed code

### Writing Good Notes

- **Be specific**: "Use `validate_email()` before creating users" > "Validate input"
- **Explain why**: "JWT tokens expire after 1 hour (security policy)" > "Tokens expire"
- **Include context**: "During high load, this cache may return stale data"
- **Keep it actionable**: What should the developer do?

### Maintaining Notes

- **Review stale notes weekly** - Confirm or invalidate them
- **Supersede instead of delete** - Preserve history
- **Use appropriate types** - Guidelines persist, context expires
- **Tag consistently** - Makes searching easier

---

## MCP Tools Reference

| Tool | Description |
|------|-------------|
| `list_notes` | List notes with filters and pagination |
| `create_note` | Create a new note |
| `get_note` | Get note details |
| `update_note` | Update note content/importance/tags |
| `delete_note` | Delete a note |
| `search_notes` | Semantic search across notes |
| `confirm_note` | Mark note as still valid |
| `invalidate_note` | Mark note as obsolete |
| `supersede_note` | Replace note with a new one |
| `link_note_to_entity` | Link note to code entity |
| `unlink_note_from_entity` | Remove link |
| `get_context_notes` | Get notes for an entity (direct + propagated) |
| `get_notes_needing_review` | List stale/needs_review notes |
| `update_staleness_scores` | Recalculate all staleness scores |

See [MCP Tools Reference](../api/mcp-tools.md#knowledge-notes-14-tools) for detailed documentation.

---

## REST API Reference

See [API Reference](../api/reference.md#notes) for complete endpoint documentation.

### Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/notes` | List notes |
| POST | `/api/notes` | Create note |
| GET | `/api/notes/{id}` | Get note |
| PATCH | `/api/notes/{id}` | Update note |
| DELETE | `/api/notes/{id}` | Delete note |
| GET | `/api/notes/search` | Search notes |
| GET | `/api/notes/context` | Get contextual notes |
| GET | `/api/notes/needs-review` | List notes needing review |
| POST | `/api/notes/update-staleness` | Update staleness scores |
| POST | `/api/notes/{id}/confirm` | Confirm note validity |
| POST | `/api/notes/{id}/invalidate` | Mark note obsolete |
| POST | `/api/notes/{id}/supersede` | Supersede with new note |
| POST | `/api/notes/{id}/links` | Link to entity |
| DELETE | `/api/notes/{id}/links/{type}/{entity}` | Remove link |
