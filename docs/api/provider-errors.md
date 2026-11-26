# Provider errors and codes

Two shapes. A refused **opening** (create or resume a session) answers its own status with
`{ "error", "code", "provider_id"?, "action"?, "retryable", "retry_after_ms"?, "fallbacks"?, "consent_scope"?, "candidates"? }`. The settings routes
answer `{ "error": "<code>: <message>" }` or a bare message. No message ever carries a path, a URL,
an identifier or a secret.

## Opening a session (typed)

| Status | `code` | Meaning / what to do |
|---|---|---|
| 424 | `cli_not_found` | The provider's program is not installed. Install it. |
| 424 | `auth_required` | Log in (`action` is the command) or grant the key to the instance. |
| 423 | `credentials_locked` | The vault is locked. Unlock it; nothing falls back to another provider. |
| 502 | `unauthorized` | The endpoint refused the key (never a 401: that is the user's session). |
| 502 | `endpoint_unreachable` | Cannot reach the endpoint. Retryable. |
| 422 | `model_no_tools` | The model cannot call a tool: unusable here. |
| 422 | `context_too_small` | The window cannot hold the tool schemas with room to work. |
| 429 | `rate_limited` | Retryable; `retry_after_ms` when the provider gave one. |
| 503 | `overloaded` | Retryable. |
| 504 | `timeout` | Retryable. |
| 502 | `process_exited` / `protocol` | The provider process died or spoke nonsense. |
| 422 | `unsupported` | A capability is missing (`sandbox`: Trust refused without a sandbox; `limits`; `security_gate`: authentication is off; `cost`). |
| 409 | `turn_in_progress` | One turn at a time on the agent engine. |
| 400 | `invalid_request` | The request is not acceptable. |
| 410 | `closed` | The session is closed. |
| 404 | `provider_unknown` | No such instance. |
| 409 | `provider_conflict` | A session never changes provider. |
| 403 | `endpoint_not_allowed` | No consent holds for this origin and credential here, or the endpoint guard refused it (then no `consent_scope`: change the URL, no consent helps). `consent_scope` says which consent is missing: `project` (a person allows it for the project: `PUT /api/projects/{slug}/llm-consents/{provider_id}` with `{origin}`) or `no_project` (a chat outside any project: `PUT /api/chat/llm-consents/{provider_id}` with `{origin}`, root account only). A project never inherits the no-project consent. |
| 503 | `provider_unavailable` | The instance cannot open sessions now. |
| 409 | `engine_unavailable` | No longer returned by a resume since M2: a session resumes on the engine it was opened on, whatever `CHAT_PROVIDER_PATH` says now. |
| 409 | `no_provider` | Nothing usable: no role or request names a provider, the Claude Code CLI is not on the server, and no other instance is allowed here. Install the CLI or configure (and allow) a provider. |
| 409 | `provider_not_chosen` | Nothing names a provider, the Claude Code CLI is not on the server, and several instances are usable here: `candidates` lists them. Name one in the request, or set the pilot role (`PUT /api/chat/roles {"pilot":{"provider":"<id>"}}`). |
| 502 | `provider_error` | The provider failed in a way this server version does not recognise (a provider newer than the backend). Show a generic failure with the status; retrying may help; report it with the provider id if it persists. |

## Settings routes (not in the typed table)

| Status | Text starts with | Meaning |
|---|---|---|
| 409 | `security_gate_closed` | Creating a third-party instance needs authentication on. |
| 409 | `origin_mismatch` | The origin you consented to is no longer the instance's. |
| 403 | `provider settings can only be changed by a signed-in user` | An agent token tried a human-only route. |
| 403 | `root_required` | A signed-in person other than the account that decides for everyone tried a server-wide routing setting: the global pilot role (`PUT /api/chat/roles`) or the consent for the chats outside any project (`PUT`/`DELETE /api/chat/llm-consents[/{provider_id}]`). That account is `auth.root_email` (matched on the email, so an OIDC account can be it) or the configured `root_account`; `GET /auth/providers` says whether the server names one (`has_deciding_account`). |
| 403 | `deciding_account_missing` | The same server-wide settings, on a server that names NO deciding account (`auth.root_email`, or `auth.root_account`): nobody can write them, the owner included. Set `auth.root_email` to the account that decides in `config.yaml`, then restart the server. |
| 403 | `tool_not_in_profile` | The route is behind a tool the restricted profile withholds. |
| 403 | `envelope_*` | A delegated session left its envelope (`unbound_token`, `parent_not_found`, `depth_exceeded`, `too_many_children`, `cwd_outside_parent`, `add_dir_outside_parent`, `project_mismatch`, `workspace_mismatch`, `not_a_child`). |
| 400 | `credential_ref ...` | A malformed reference, a pasted secret, or an undeclared `env:` variable. |
| 400 | `host`, `ssh_user`, `ssh_port`, `host_key`, `remote_cwd` | A `claude_code_remote` instance with a malformed machine: host or user not a plain name (never a leading `-`), port outside 1-65535, key not exactly `<type> <base64>`, no working directory, or a credential that is not `vault:<name>`. |
| 400 | `endpoint refused: ...` | The endpoint guard: scheme, credentials in the URL, host missing, private address, unresolvable, redirects. |
| 404 | `unknown provider instance` | No such instance (also for a vault grant and for a run's provider). |
| 400 | `a provider grant ...` | A provider grant must name exactly the instance's key. |

## Verdicts of `POST /api/chat/providers/test` (always 200)

`health.state` is `ok`, `auth_required`, `unreachable`, `cli_not_found` or `unknown`; `health.code` adds:
the `endpoint_*` refusals of the guard (`endpoint_invalid_url`, `endpoint_scheme_not_allowed`,
`endpoint_http_outside_loopback`, `endpoint_credentials_in_url`, `endpoint_host_missing`,
`endpoint_private_address`, `endpoint_unresolvable`, `endpoint_redirects_not_allowed`),
`credentials_locked`, `credential_test_requires_saved_instance`, `model_no_tools`.
