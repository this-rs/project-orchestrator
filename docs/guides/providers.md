# Providers: connecting a model other than Claude Code

Claude Code is built in and needs nothing. Any other model is a **provider instance**
you register, allow per project, and (when it needs a key) grant a key to.

## Which engine runs what

| Provider | Engine | Notes |
|---|---|---|
| `claude-code` (built in) | the historical engine | hooks, message queue, auto-continue, retry, compaction, NATS and images all work |
| a registered instance (OpenAI-compatible, Codex, ACP) | the **agent engine** | always, whatever `CHAT_PROVIDER_PATH` says |

`CHAT_PROVIDER_PATH=agent` only **forces Claude Code onto the agent engine**, to try it. The
server logs a warning and every `system_init` of that session carries `engine: "agent"` and
`degraded_features` (the list of what the session does not do). Default is `legacy`.
A session opened on the agent engine cannot be resumed once that engine is switched off:
the answer is `409 engine_unavailable`.

## Before the first connection

1. **Authentication must be on.** A third-party provider is refused (`409 security_gate_closed`)
   on a server running without authentication: session tokens cannot be bound to a signing key.
2. A person (not an agent) does every step below. An agent session token gets `403`.

## First connection, step by step

1. **Open `/providers`** (or call `POST /api/chat/providers`). Body, references only:
   `{ "id": "deepseek", "kind": "openai_compatible", "preset": "deepseek", "label": "DeepSeek",
   "base_url": "https://api.deepseek.com/v1", "default_model": "deepseek-chat",
   "cost_source": "priced", "credential_ref": "vault:deepseek" }`.
   No field takes a secret value; an unknown field (an `api_key`) is refused.
   - `credential_ref` is `none`, `vault:<name>` or `env:<VAR>`. `env:` is refused unless
     `<VAR>` is declared in `CHAT_PROVIDER_ENV_CREDENTIALS` (comma separated, empty by default).
   - `kind: "codex"` runs the `codex` program of the server's PATH. `kind: "acp"` names, with
     `preset`, an agent declared in `CHAT_PROVIDER_ACP_COMMANDS`
     (`{"opencode": ["opencode", "acp"]}`). A request never carries a command line.
2. **Store the key in the vault** (unlock it first) under the name the instance's `credential_ref` gives.
3. **Grant the key to the instance**: `POST /api/vault/grants` with
   `{"secrets": {"kind":"names","names":["deepseek"]}, "scope": {"kind":"provider","value":"deepseek"}}`
   plus the unlock proof. It must name exactly the instance's key.
4. **Test the connection**: `POST /api/chat/providers/test` with the same body (a draft that carries a
   key is tested only after it is saved). The verdict says whether the endpoint is reachable, the
   model lists, **can call a tool**, and how large its context window is.
5. **Allow the project**: `PUT /api/projects/{slug}/llm-consent` with `{provider_id, origin}`, the origin
   you were shown. The consent is tied to the origin **and** the credential reference: change either
   and it stops holding. A conversation with no project can only use Claude Code.
6. **Open a session** with `provider: "deepseek"`, or set a role (`/api/chat/roles`,
   `/api/projects/{slug}/llm-roles`) so the project uses it by default.

To edit an instance, `GET /api/chat/providers/{id}` returns what was stored: the full `base_url`
(path included), `default_model`, `preset`, `cost_source` and the `credential_ref` reference, never a
secret value. It is for a signed-in person only (an agent token gets `403`, an unknown id `404`).
`GET /api/chat/providers` stays the safe listing: it shows the origin, never a path.

What was sent where is readable at `GET /api/chat/send-journal` (who, project, origin, model; never the content).

## OpenClaw (delegating a conversation to an OpenClaw agent)

OpenClaw speaks ACP through its bridge, `openclaw acp`, which connects to an OpenClaw Gateway.
Declare it on the server, then register an ACP instance that names it:

```sh
CHAT_PROVIDER_ACP_COMMANDS='{"openclaw":["openclaw","acp","--url","wss://gateway.example:18789","--token-file","/etc/po/openclaw.token"]}'
```

```json
{ "id": "openclaw", "kind": "acp", "preset": "openclaw", "label": "OpenClaw",
  "default_model": "openclaw", "cost_source": "unknown", "credential_ref": "none" }
```

- **Use `--token-file`, never `--token`**: a token on the command line is visible to every user of
  the machine in the process list (and an argument that looks like a secret is refused anyway).
  The file belongs to the server's user, mode `0600`.
- **No project-orchestrator tools in the session.** The bridge refuses MCP servers given per
  session (`session/new` answers "ACP bridge mode does not support per-session MCP servers";
  older versions ignored them without a word). The server knows it from the program's name
  (`openclaw`, whatever its path): the session is opened **without** the project-orchestrator
  MCP server and says so, `degraded_features` carries `project_orchestrator_tools` (the same
  banner as a remote Claude Code). An ACP agent declared under another name that refuses
  them the same way opens its first session again without them (and says so); the next
  sessions of that instance are opened without them from the start. To give OpenClaw the PO tools, configure them on the
  OpenClaw side (`openclaw mcp set project-orchestrator '<json>'`: a stdio `command`, or the
  server's `/mcp` over Streamable HTTP when `remote_mcp.enabled`, with its authentication).
- **Other limits of the bridge**: no `fs/*` or `terminal/*` requests (the agent uses its own
  tools), no model choice (the model name is a label), no system prompt, no knowledge-graph
  hooks, no compaction signal, images not sent; permissions are relayed only while a turn is
  running. Each session is an isolated OpenClaw session (`acp-bridge:<uuid>`) unless the
  command targets one (`--session agent:main:main`).

## Routing, roles, aliases, policy

Resolution order: existing session (frozen) > explicit request > task alias > run > project role >
global role (or an enforced policy rule) > default > Claude Code. The level that decided is stored as
`routed_by` on the session and the execution. The model policy ships `off`; `shadow` records what it
would have chosen without applying it; `enforce` applies it.

## Cost

Two counters: *marginal* (reported or priced: real spend, the only one that can stop a run) and
*notional* (subscription or free). An unknown price is never a zero. `GET /api/chat/runs/{run_id}/costs`.

## Server settings

| Variable | Default | Meaning |
|---|---|---|
| `CHAT_PROVIDER_PATH` | `legacy` | `agent` forces Claude Code onto the agent engine (degraded, warned) |
| `CHAT_PROVIDER_ENV_CREDENTIALS` | empty | server variables an instance may name as `env:<VAR>` |
| `CHAT_PROVIDER_ACP_COMMANDS` | empty | JSON object of ACP agents an instance may launch |
| `CHAT_CHILD_ENV_INHERIT` | empty | extra variables handed to agent processes (never the server's own secrets) |

## Not verified

No real Codex `app-server`, ACP agent, https endpoint or vendor API was exercised: everything was run
against the fake servers and executables of the test suites.
