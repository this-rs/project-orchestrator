# Multi-Agent Workflow Guide

Advanced guide for coordinating multiple AI agents on a single project.

---

## Why Multi-Agent?

A single AI agent can only do one thing at a time. With multiple agents, you can:

- **Parallelize work** — Backend and frontend tasks run simultaneously
- **Specialize** — Each agent focuses on what it does best
- **Scale** — Add more agents for larger projects

Project Orchestrator provides the shared context that makes this possible.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROJECT ORCHESTRATOR                             │
│                        (Shared Knowledge Base)                           │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │    Plans     │  │    Tasks     │  │  Decisions   │                   │
│  │              │  │              │  │              │                   │
│  │ Dependencies │  │   Status     │  │  Rationale   │                   │
│  │ Constraints  │  │  Assignees   │  │  History     │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      Code Knowledge Graph                         │   │
│  │  Files • Functions • Imports • Call Graph • Impact Analysis      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   AGENT 1     │     │   AGENT 2     │     │   AGENT 3     │
│   (Backend)   │     │  (Frontend)   │     │   (Tests)     │
│               │     │               │     │               │
│ get_next_task │     │ get_next_task │     │ get_next_task │
│ update_task   │     │ update_task   │     │ update_task   │
│ add_decision  │     │ add_decision  │     │ add_decision  │
└───────────────┘     └───────────────┘     └───────────────┘
```

### How it works

1. **Shared Plan** — All agents work from the same plan with defined tasks
2. **Task Assignment** — `get_next_task` returns unblocked, unassigned work
3. **Status Tracking** — Agents mark tasks `in_progress` to claim them
4. **Dependency Handling** — Completed tasks automatically unblock dependent tasks
5. **Decision Sharing** — Decisions are visible to all agents via search

---

## Setting Up Multi-Agent Work

### Step 1: Create a Plan with Parallelizable Tasks

Design tasks that can run in parallel where possible:

```
Create a plan "User Authentication Feature" with these tasks:

1. "Design API endpoints" (tags: design)
2. "Implement login backend" (tags: backend) - depends on 1
3. "Implement logout backend" (tags: backend) - depends on 1
4. "Create login UI" (tags: frontend) - depends on 1
5. "Create logout UI" (tags: frontend) - depends on 1
6. "Write backend tests" (tags: testing) - depends on 2, 3
7. "Write frontend tests" (tags: testing) - depends on 4, 5
8. "Integration testing" (tags: testing) - depends on 6, 7
```

### Dependency Graph

```
                    ┌─────────────────┐
                    │  1. Design API  │
                    │    (design)     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  2. Login BE    │ │  3. Logout BE   │ │  4. Login UI    │
│   (backend)     │ │   (backend)     │ │  (frontend)     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └─────────┬─────────┘                   │
                   │                             │
                   ▼                             ▼
         ┌─────────────────┐           ┌─────────────────┐
         │ 6. Backend Tests│           │ 5. Logout UI    │
         │   (testing)     │           │  (frontend)     │
         └────────┬────────┘           └────────┬────────┘
                  │                             │
                  │                             ▼
                  │                    ┌─────────────────┐
                  │                    │7. Frontend Tests│
                  │                    │   (testing)     │
                  │                    └────────┬────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │ 8. Integration  │
                       │   (testing)     │
                       └─────────────────┘
```

### Step 2: Assign Agent Roles

Each agent should have clear responsibilities:

**Agent 1: Backend Developer**
```
Instructions: You work on backend tasks.
- Filter tasks by tags: backend
- Focus on API implementation
- Record decisions about data structures
```

**Agent 2: Frontend Developer**
```
Instructions: You work on frontend tasks.
- Filter tasks by tags: frontend
- Focus on UI components
- Record decisions about UX choices
```

**Agent 3: Test Engineer**
```
Instructions: You write and run tests.
- Filter tasks by tags: testing
- Wait for implementation tasks to complete
- Record decisions about test coverage
```

---

## The Multi-Agent Dance

### Sequence Diagram

```
Agent 1 (Backend)          Orchestrator          Agent 2 (Frontend)
       │                        │                        │
       │   get_next_task()     │                        │
       │───────────────────────>│                        │
       │   Task 2 (Login BE)   │                        │
       │<───────────────────────│                        │
       │                        │                        │
       │   update_task(        │   get_next_task()     │
       │     in_progress)      │<───────────────────────│
       │───────────────────────>│   Task 4 (Login UI)   │
       │                        │───────────────────────>│
       │                        │                        │
       │   [Working...]         │   update_task(        │
       │                        │     in_progress)      │
       │                        │<───────────────────────│
       │                        │                        │
       │   add_decision(       │   [Working...]         │
       │     "Use bcrypt")     │                        │
       │───────────────────────>│                        │
       │                        │                        │
       │   update_task(        │   search_decisions(   │
       │     completed)        │     "password")       │
       │───────────────────────>│<───────────────────────│
       │                        │   "Use bcrypt"        │
       │   get_next_task()     │───────────────────────>│
       │───────────────────────>│                        │
       │   Task 3 (Logout BE)  │   [Uses bcrypt too]    │
       │<───────────────────────│                        │
       │                        │                        │
```

### Key Interactions

1. **Task Claiming** — When Agent 1 marks Task 2 as `in_progress`, Agent 2 won't get it
2. **Decision Sharing** — Agent 1's bcrypt decision is found by Agent 2's search
3. **Automatic Unblocking** — When Task 2 completes, Task 6 becomes available

---

## Workflow Patterns

### Pattern 1: Round-Robin Task Assignment

Agents take turns getting the next available task:

```python
# Each agent runs this loop
while True:
    task = get_next_task(plan_id)
    if not task:
        break  # All tasks done

    update_task(task.id, status="in_progress", assigned_to=agent_id)

    # Do the work
    result = execute_task(task)

    # Record what was done
    if result.decisions:
        for decision in result.decisions:
            add_decision(task.id, decision)

    update_task(task.id, status="completed")
```

### Pattern 2: Specialized Agents

Agents filter for specific task types:

```python
# Backend agent
while True:
    tasks = list_tasks(plan_id, tags="backend", status="pending")
    available = [t for t in tasks if not get_task_blockers(t.id)]

    if not available:
        # Check if there's still work coming
        all_tasks = list_tasks(plan_id)
        if all(t.status == "completed" for t in all_tasks):
            break
        time.sleep(30)  # Wait for blockers to clear
        continue

    task = available[0]
    # ... process task
```

### Pattern 3: Leader-Worker

One agent coordinates, others execute:

```python
# Leader agent
def coordinate():
    plan = get_plan(plan_id)

    while not plan_complete(plan):
        # Assign next tasks to available workers
        for worker in available_workers:
            task = get_next_task(plan_id)
            if task:
                assign_to_worker(worker, task)

        # Check progress
        time.sleep(60)
        plan = get_plan(plan_id)

# Worker agents
def work():
    while True:
        assignment = wait_for_assignment()
        execute_task(assignment.task)
        report_completion(assignment)
```

---

## Example: Full Feature Development

### Scenario

Build a "Password Reset" feature with 3 agents working in parallel.

### Plan Setup

```
Create plan "Password Reset Feature" with:

Tasks:
1. "Design reset flow" (priority: 10, tags: design)
   - Acceptance: API spec document created

2. "Create reset token model" (priority: 9, tags: backend, database)
   - Depends on: 1
   - Acceptance: Migration created, model tested

3. "Implement /forgot-password endpoint" (priority: 9, tags: backend, api)
   - Depends on: 2
   - Acceptance: Returns 200, sends email

4. "Implement /reset-password endpoint" (priority: 9, tags: backend, api)
   - Depends on: 2
   - Acceptance: Validates token, updates password

5. "Create forgot password UI" (priority: 8, tags: frontend)
   - Depends on: 1
   - Acceptance: Form submits to API

6. "Create reset password UI" (priority: 8, tags: frontend)
   - Depends on: 1
   - Acceptance: Token from URL, form works

7. "Write API tests" (priority: 7, tags: testing)
   - Depends on: 3, 4
   - Acceptance: 90% coverage

8. "Write E2E tests" (priority: 7, tags: testing)
   - Depends on: 5, 6, 7
   - Acceptance: Full flow tested
```

### Execution Timeline

```
Time    Agent 1 (Backend)      Agent 2 (Frontend)     Agent 3 (Testing)
─────   ─────────────────      ──────────────────     ─────────────────
T+0     Design reset flow      [waiting]              [waiting]
        (task 1)

T+1     Create token model     Create forgot UI       [waiting]
        (task 2)               (task 5)

T+2     /forgot-password       Create reset UI        [waiting]
        (task 3)               (task 6)

T+3     /reset-password        [done]                 Write API tests
        (task 4)                                      (task 7)

T+4     [done]                 [done]                 Write E2E tests
                                                      (task 8)
```

### Decisions Recorded

During execution, agents record decisions:

**Agent 1 (Backend):**
```
Task 2: "Use UUID v7 for reset tokens - includes timestamp for expiry check"
Task 3: "Rate limit to 3 requests per hour per email - prevent abuse"
Task 4: "Invalidate all sessions on password reset - security best practice"
```

**Agent 2 (Frontend):**
```
Task 5: "Show generic success message even if email not found - prevent enumeration"
Task 6: "Auto-focus password field, show strength meter"
```

**Agent 3 (Testing):**
```
Task 7: "Mock email service, test rate limiting separately"
Task 8: "Use mailhog for E2E email verification"
```

### Final Sync

All decisions are searchable:

```
Search decisions for "rate limit"
→ "Rate limit to 3 requests per hour per email - prevent abuse" (Task 3)

Search decisions for "security"
→ "Invalidate all sessions on password reset - security best practice" (Task 4)
→ "Show generic success message even if email not found - prevent enumeration" (Task 5)
```

---

## Best Practices

### 1. Clear Task Boundaries

Each task should be completable independently:

```
✅ Good: "Implement login API endpoint"
❌ Bad: "Work on authentication" (too vague, overlaps)
```

### 2. Explicit Dependencies

Always set dependencies to prevent conflicts:

```
✅ Good: Task "Write tests" depends on "Implement feature"
❌ Bad: Assuming agents will coordinate timing manually
```

### 3. Record Decisions Immediately

Don't wait until task completion:

```
✅ Good: Record decision as soon as you make a choice
❌ Bad: Try to remember all decisions at the end
```

### 4. Use Tags for Routing

Tags help agents find their work:

```
✅ Good: tags: [backend, api, auth]
❌ Bad: No tags (all agents compete for all tasks)
```

### 5. Small, Focused Tasks

Break large tasks into smaller ones:

```
✅ Good:
- "Create user model"
- "Add validation to user model"
- "Write user model tests"

❌ Bad:
- "Build entire user system"
```

### 6. Check for Conflicts Before Editing

Use impact analysis:

```
Before editing UserService:
→ analyze_impact("UserService")
→ Check if other agents are working on affected files
```

---

## Handling Conflicts

### File Conflicts

If two agents need the same file:

1. **Sequential tasks** — Add dependency so they don't overlap
2. **Different sections** — Coordinate via decisions
3. **Merge later** — Each works on a branch

### Decision Conflicts

If agents make contradictory decisions:

1. **Search first** — Always search existing decisions before making new ones
2. **Reference previous** — Link new decisions to related ones
3. **Escalate** — Mark task as blocked if conflict can't be resolved

---

## Monitoring Progress

### Dashboard Queries

```
# Overall progress
Show me the status of all tasks in plan xyz

# Per-agent progress
List tasks assigned to agent-1

# Blocked work
What tasks are currently blocked?

# Critical path
What's the critical path for this plan?
```

### Health Checks

```
# Are agents making progress?
List tasks that have been in_progress for more than 1 hour

# Are there orphaned tasks?
List tasks with no assignee that are unblocked
```

---

## Next Steps

- [Getting Started](./getting-started.md) — Basic tutorial
- [API Reference](../api/reference.md) — Full API documentation
- [MCP Tools](../api/mcp-tools.md) — All available tools
