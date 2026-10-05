# Providers: threat model

What a third-party provider changes: the project's content leaves the machine, a model PO does not
control drives MCP tools, and a key has to be read by the server.

| Threat | Control | Where |
|---|---|---|
| An agent (or a prompt injection in its context) registers an endpoint, widens consent, changes roles | Every mutation is human only: the middleware refuses an `agent_session` token and each handler checks again | `auth/middleware.rs`, `api/provider_handlers.rs` |
| An agent writes to, stops or deletes a human's session | An `agent_session` token acts only on the sessions it spawned (`ensure_child_of`), on every non-read `/api/chat/sessions/{id}/*` | `auth/middleware.rs` |
| A delegated session escapes its parent | Spawn envelope: parent read from the signed token, project and cwd of the parent, policy never more permissive, depth 1, 4 live children, cascade cancel | `chat/envelope.rs` |
| A model calls a tool that opens sessions, runs plans, reads the vault | Restricted tool profile = **allow-list** signed into the token; the REST routes behind withheld tools answer 403 | `auth/tool_profile.rs` |
| SSRF through a base URL | https outside loopback, no credentials in the URL, every resolved address public (rebinding), redirects refused, checked before every connection and again at open | `chat/provider/endpoint_guard.rs`, nexus `model/guard.rs` |
| A key leaks | Instances hold a reference, never a value; read per request under a grant for that instance only; masked in every provider event (fail closed) before persistence and broadcast; never in an error, a journal line or the listing | `chat/provider/credentials.rs`, `vault/`, `chat/agent_runtime.rs` |
| A reference points at a server secret | `env:` only for variables declared in `CHAT_PROVIDER_ENV_CREDENTIALS` (empty by default); the server's own secrets refused even if declared | `chat/provider/settings.rs` |
| A key is granted to the wrong instance | A provider grant must name exactly the instance's `credential_ref` secret | `api/vault_handlers.rs` |
| Content sent where nobody agreed | Consent per project, tied to the origin and the credential reference; no project = Claude Code only; a resume re-checks | `chat/manager.rs` |
| An API body runs a command | codex/acp never take a command line: `codex` from PATH, ACP by a preset declared on the server | `chat/provider/settings.rs` |
| A model with no sandbox runs free | `Trust` refused for a provider without a sandbox; a run on a third party runs under `ask` | `chat/manager.rs` |
| The child process inherits the server's secrets | Clean environment: allowlist, MCP secrets off the command line, a dedicated HOME for non-Claude providers | nexus `transport/spawn.rs`, `chat/manager.rs` |
| No trace of what was sent | Send journal (who, project, origin, model); a failed write refuses the opening | `GET /api/chat/send-journal` |
| Third party on a server without authentication | Refused (`security_gate_closed`): tokens cannot be bound | `api/provider_handlers.rs` |

## Residual risks, stated

- Without authentication (`auth_config` absent) the human-only checks pass everything.
- `/api/admin/*` stays reachable by an agent session token with the full profile (it is the target of the `admin` tool Claude sessions use); a third party's restricted profile closes it.
- The model sees the project's content: consent is the control, not isolation. A malicious model can still misuse the tools the profile allows (create notes, plans, decisions).
- Nothing was exercised against a real third-party service, a real https endpoint or a real Codex `app-server`; those paths are covered by fakes.
