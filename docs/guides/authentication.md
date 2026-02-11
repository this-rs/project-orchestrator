# Authentication Guide

Multi-provider JWT-based authentication for the Project Orchestrator HTTP API and WebSocket connections. Supports password login, Google OAuth2, and generic OIDC providers (Keycloak, Auth0, Azure AD, Okta, etc.).

**Applies to:** HTTP API server (`api_server` binary)
**Does not apply to:** MCP server (`mcp_server` binary, runs locally via stdio)

---

## Overview

Project Orchestrator uses **JWT HS256 tokens** issued after authenticating through one of three supported providers. The security model has two modes:

- **No-auth mode:** If the `auth` section is absent from `config.yaml`, `auth_config` is `None` and ALL requests pass through freely with anonymous claims. This is useful for local development and single-user setups.
- **Auth mode:** If the `auth` section is present, JWT authentication is enforced on all protected routes. The specific providers available depend on which sub-sections are configured.

Key principles:

- **Three auth providers:** Password (root account + Neo4j-stored users), Google OAuth2 (legacy), and Generic OIDC (any OpenID Connect provider)
- **Provider discovery:** `GET /auth/providers` lets the frontend dynamically adapt its login UI based on which providers are configured
- **JWT tokens** are stateless and short-lived (default 8 hours)
- **Email restrictions** optionally limit access by domain or individual whitelist
- **User registration** can be enabled to allow self-service password account creation
- **WebSocket auth** uses a first-message handshake (browsers cannot set HTTP headers on WebSocket upgrades)
- **MCP server** runs locally over stdio and does not require authentication

---

## Architecture

The system supports three authentication flows, all converging on the same JWT issuance:

### Flow 1: Password Login

```
Frontend                         Server
   |                                |
   |  1. POST /auth/login           |
   |  { email, password }           |
   | -----------------------------> |
   |                                |  2. Check root account (in-memory)
   |                                |     OR lookup user in Neo4j
   |                                |  3. Verify bcrypt hash
   |                                |  4. Generate JWT (HS256)
   |  { token, user }              |
   | <----------------------------- |
   |                                |
   |  5. Authorization: Bearer <token>
   | -----------------------------> |
   |  Protected resource            |
   | <----------------------------- |
```

### Flow 2: Google OAuth2 (Legacy)

```
Frontend                    Server                        Google
   |                          |                              |
   |  1. GET /auth/google     |                              |
   | -----------------------> |                              |
   |  { auth_url: "..." }    |                              |
   | <----------------------- |                              |
   |                          |                              |
   |  2. Redirect to Google   |                              |
   | -------------------------------------------------------->
   |                          |                              |
   |  3. User consents, Google redirects with ?code=...      |
   | <--------------------------------------------------------
   |                          |                              |
   |  4. POST /auth/google/callback { code: "..." }          |
   | -----------------------> |                              |
   |                          |  5. Exchange code for token  |
   |                          | ---------------------------->|
   |                          |  6. Fetch user info          |
   |                          | ---------------------------->|
   |                          |  { email, name, picture }    |
   |                          | <----------------------------|
   |                          |                              |
   |                          |  7. Upsert user in Neo4j     |
   |                          |  8. Generate JWT (HS256)     |
   |                          |                              |
   |  { token, user }        |                              |
   | <----------------------- |                              |
```

### Flow 3: Generic OIDC

```
Frontend                    Server                     OIDC Provider
   |                          |                              |
   |  1. GET /auth/oidc       |                              |
   | -----------------------> |                              |
   |  { auth_url: "..." }    |  (built from discovery_url   |
   | <----------------------- |   or explicit endpoints)     |
   |                          |                              |
   |  2. Redirect to provider |                              |
   | -------------------------------------------------------->
   |                          |                              |
   |  3. User authenticates, provider redirects with ?code=...|
   | <--------------------------------------------------------
   |                          |                              |
   |  4. POST /auth/oidc/callback { code: "..." }            |
   | -----------------------> |                              |
   |                          |  5. Exchange code for token  |
   |                          | ---------------------------->|
   |                          |  6. Fetch user info          |
   |                          | ---------------------------->|
   |                          |  { email, name, sub }        |
   |                          | <----------------------------|
   |                          |                              |
   |                          |  7. Upsert user in Neo4j     |
   |                          |  8. Generate JWT (HS256)     |
   |                          |                              |
   |  { token, user }        |                              |
   | <----------------------- |                              |
```

### JWT Claims

The JWT payload is identical regardless of which provider was used to authenticate:

| Field | Type | Description |
|-------|------|-------------|
| `sub` | UUID | User ID (from Neo4j, or deterministic UUID for root account) |
| `email` | String | User's email address |
| `name` | String | User's display name |
| `iat` | Unix timestamp | Issued at |
| `exp` | Unix timestamp | Expiration |

---

## Configuration

Add the `auth` section to your `config.yaml`. All provider sub-sections are optional -- include only the ones you need:

```yaml
auth:
  jwt_secret: "change-me-to-a-random-32-char-string!"
  jwt_expiry_secs: 28800  # 8 hours (default)

  # ── Root account (always available if configured) ──────────
  root_account:
    email: "admin@example.com"
    name: "Admin"
    password_hash: "$2b$12$..."  # bcrypt hash (or plaintext — hashed at startup with a warning)

  # ── Google OAuth (legacy — still works) ────────────────────
  google_client_id: "YOUR_ID.apps.googleusercontent.com"
  google_client_secret: "YOUR_SECRET"
  google_redirect_uri: "http://localhost:3000/auth/callback"

  # ── Generic OIDC (any provider) ────────────────────────────
  oidc:
    client_id: "your-oidc-client-id"
    client_secret: "your-oidc-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Keycloak"    # shown in the UI (default: "SSO")
    scopes: "openid email profile"  # default
    # Auto-discovery (recommended):
    discovery_url: "https://auth.example.com/realms/main/.well-known/openid-configuration"
    # Or explicit endpoints (if no .well-known support):
    # auth_endpoint: "https://auth.example.com/authorize"
    # token_endpoint: "https://auth.example.com/token"
    # userinfo_endpoint: "https://auth.example.com/userinfo"

  # ── Access control ─────────────────────────────────────────
  allow_registration: true           # allow POST /auth/register (default: false)
  allowed_emails:                    # individual whitelist (optional)
    - "admin@company.com"
    - "dev@company.com"
  allowed_email_domain: "company.com"  # domain restriction (optional)
  frontend_url: "http://localhost:3000"  # restricts CORS origin (optional)
```

### Configuration Fields

#### Common Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `jwt_secret` | Yes | -- | Secret key for HS256 signing (minimum 32 characters) |
| `jwt_expiry_secs` | No | `28800` | Token lifetime in seconds (8 hours) |
| `allowed_email_domain` | No | `None` | If set, only emails ending with `@<domain>` are allowed |
| `allowed_emails` | No | `None` | Individual email whitelist (works alongside `allowed_email_domain`) |
| `allow_registration` | No | `false` | Enable `POST /auth/register` for self-service account creation |
| `frontend_url` | No | `None` | If set, CORS is restricted to this origin; otherwise allows any origin |

#### Root Account (`root_account`)

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `email` | Yes | -- | Root account login email |
| `name` | Yes | -- | Root account display name |
| `password_hash` | Yes | -- | Bcrypt hash (`$2b$12$...`) or plaintext (auto-hashed at startup) |

#### Generic OIDC (`oidc`)

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `client_id` | Yes | -- | OAuth2 client ID from your identity provider |
| `client_secret` | Yes | -- | OAuth2 client secret |
| `redirect_uri` | Yes | -- | Callback URL (must match provider configuration) |
| `discovery_url` | No | `None` | OIDC `.well-known/openid-configuration` URL (auto-discovers endpoints) |
| `auth_endpoint` | No | `None` | Authorization endpoint (required if no `discovery_url`) |
| `token_endpoint` | No | `None` | Token endpoint (required if no `discovery_url`) |
| `userinfo_endpoint` | No | `None` | Userinfo endpoint (optional, fetched from discovery if available) |
| `provider_name` | No | `"SSO"` | Human-readable name shown in the frontend UI |
| `provider_key` | No | `None` | Provider key for the frontend wizard (`google`, `keycloak`, `okta`, etc.) |
| `scopes` | No | `"openid email profile"` | OAuth2 scopes to request |

#### Legacy Google OAuth Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `google_client_id` | No | `None` | Deprecated -- use `oidc.client_id` instead |
| `google_client_secret` | No | `None` | Deprecated -- use `oidc.client_secret` instead |
| `google_redirect_uri` | No | `None` | Deprecated -- use `oidc.redirect_uri` instead |

> **Backward compatibility:** If the legacy `google_*` fields are present but no `oidc` section exists, they are automatically mapped to an equivalent OIDC config with Google's well-known endpoints. The `/auth/google` and `/auth/google/callback` routes continue to work.

### Config Modes Summary

| Config State | Behavior |
|-------------|----------|
| No `auth` section at all | **No-auth mode** -- all requests pass through with anonymous claims |
| `auth` present with `root_account` | Password login via `POST /auth/login` |
| `auth` present with `oidc` | Generic OIDC login via `GET /auth/oidc` + `POST /auth/oidc/callback` |
| `auth` present with `google_*` fields | Legacy Google OAuth via `GET /auth/google` + `POST /auth/google/callback` |
| `auth` present with multiple providers | All configured providers available simultaneously |
| `auth` present but no providers configured | Auth is enforced but no way to log in (effectively locked out) |

---

## Setting Up Each Provider

### Password Auth (Root Account)

The simplest setup. No external services needed.

#### Step 1: Generate a bcrypt hash

```bash
# Using htpasswd (Apache utils)
htpasswd -nbBC 12 "" "your-password-here" | cut -d: -f2

# Or using Python
python3 -c "import bcrypt; print(bcrypt.hashpw(b'your-password-here', bcrypt.gensalt(12)).decode())"
```

Alternatively, you can put the plaintext password directly in `password_hash` and the server will hash it at startup (with a warning log).

#### Step 2: Add to config.yaml

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  root_account:
    email: "admin@example.com"
    name: "Admin"
    password_hash: "$2b$12$LJ3m4sFQm1MJoUqoEZnBN.ZhYcKORSVn4nh1qZk6IP6W7yR3xtGPq"
```

> **User registration:** If `allow_registration: true` is set, anyone can create a new password-authenticated account via `POST /auth/register`. Registered users are stored in Neo4j with bcrypt-hashed passwords. You can restrict registration with `allowed_email_domain` or `allowed_emails`.

### Google OAuth2

#### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click **Select a project** at the top, then **New Project**
3. Enter a project name (e.g., "Project Orchestrator") and click **Create**

#### Step 2: Configure the OAuth Consent Screen

1. Navigate to **APIs & Services > OAuth consent screen**
2. Choose **External** (or **Internal** for Google Workspace organizations)
3. Fill in the required fields:
   - App name
   - User support email
   - Developer contact email
4. Under **Scopes**, add: `openid`, `email`, `profile`
5. Save and continue

#### Step 3: Create OAuth 2.0 Credentials

1. Navigate to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth 2.0 Client ID**
3. Select **Web application** as the application type
4. Under **Authorized redirect URIs**, add your callback URL:
   - For local development: `http://localhost:3000/auth/callback`
   - For production: `https://your-domain.com/auth/callback`
5. Click **Create**

#### Step 4: Copy Credentials to Config

You can use either the legacy fields or the new OIDC section:

**Legacy (still works):**

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  google_client_id: "123456789-abcdef.apps.googleusercontent.com"
  google_client_secret: "GOCSPX-your-secret-here"
  google_redirect_uri: "http://localhost:3000/auth/callback"
```

**Recommended (OIDC section):**

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  oidc:
    client_id: "123456789-abcdef.apps.googleusercontent.com"
    client_secret: "GOCSPX-your-secret-here"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Google"
    provider_key: "google"
    discovery_url: "https://accounts.google.com/.well-known/openid-configuration"
```

### Generic OIDC (Keycloak, Auth0, Azure AD, Okta, etc.)

Any OpenID Connect-compliant provider can be used. The recommended approach is to provide the `discovery_url` so endpoints are auto-discovered.

#### Keycloak Example

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  oidc:
    client_id: "project-orchestrator"
    client_secret: "your-keycloak-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Keycloak"
    provider_key: "keycloak"
    discovery_url: "https://auth.example.com/realms/main/.well-known/openid-configuration"
```

#### Auth0 Example

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  oidc:
    client_id: "your-auth0-client-id"
    client_secret: "your-auth0-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Auth0"
    provider_key: "auth0"
    discovery_url: "https://your-tenant.auth0.com/.well-known/openid-configuration"
```

#### Azure AD Example

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  oidc:
    client_id: "your-azure-app-id"
    client_secret: "your-azure-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Microsoft"
    provider_key: "microsoft"
    discovery_url: "https://login.microsoftonline.com/{tenant-id}/v2.0/.well-known/openid-configuration"
```

#### Explicit Endpoints (No Discovery)

If your provider does not support `.well-known/openid-configuration`:

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  oidc:
    client_id: "your-client-id"
    client_secret: "your-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Custom SSO"
    auth_endpoint: "https://sso.example.com/authorize"
    token_endpoint: "https://sso.example.com/token"
    userinfo_endpoint: "https://sso.example.com/userinfo"
```

---

## API Endpoints

### Authentication Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/auth/providers` | Public | Discovery: returns available auth methods |
| `POST` | `/auth/login` | Public | Email/password login |
| `POST` | `/auth/register` | Public | Create a new password account |
| `GET` | `/auth/google` | Public | Returns Google OAuth authorization URL (legacy) |
| `POST` | `/auth/google/callback` | Public | Exchanges Google auth code for JWT (legacy) |
| `GET` | `/auth/oidc` | Public | Returns OIDC authorization URL |
| `POST` | `/auth/oidc/callback` | Public | Exchanges OIDC auth code for JWT |
| `GET` | `/auth/me` | Protected | Returns the authenticated user's profile |
| `POST` | `/auth/refresh` | Protected | Issues a fresh JWT from a still-valid token |

---

### GET /auth/providers

Discovery endpoint. Returns which authentication providers are configured so the frontend can adapt its login UI dynamically.

```bash
curl http://localhost:8080/auth/providers
```

**Response (auth configured with password + OIDC):**

```json
{
  "auth_required": true,
  "providers": [
    {
      "id": "password",
      "name": "Email & Password",
      "type": "password"
    },
    {
      "id": "oidc",
      "name": "Google",
      "type": "oidc"
    }
  ],
  "allow_registration": true
}
```

**Response (no auth section in config -- no-auth mode):**

```json
{
  "auth_required": false,
  "providers": [],
  "allow_registration": false
}
```

---

### POST /auth/login

Email/password authentication. Checks the root account first (in-memory, no DB hit), then falls back to Neo4j-stored users.

```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "my-password"}'
```

Response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "admin@example.com",
    "name": "Admin",
    "picture_url": null,
    "is_root": true
  }
}
```

> **Security:** Error messages never reveal whether the email exists. Both "wrong email" and "wrong password" return the same `"Invalid email or password"` error to prevent user enumeration.

---

### POST /auth/register

Create a new password-authenticated account. Only available when `allow_registration` is `true`.

```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "password": "at-least-8-chars", "name": "Alice"}'
```

Response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "660a9400-f39c-51e5-b827-557766551111",
    "email": "alice@example.com",
    "name": "Alice",
    "picture_url": null,
    "is_root": false
  }
}
```

Validation rules:
- Name must not be empty
- Email must contain `@` and a valid domain
- Password must be at least 8 characters
- Email must pass the `allowed_email_domain` / `allowed_emails` restrictions (if configured)
- Email must not already be registered for the "password" provider

---

### GET /auth/google (Legacy)

Returns the Google OAuth authorization URL. The frontend should redirect the user to this URL.

```bash
curl http://localhost:8080/auth/google
```

Response:

```json
{
  "auth_url": "https://accounts.google.com/o/oauth2/v2/auth?client_id=...&redirect_uri=...&response_type=code&scope=openid%20email%20profile&access_type=offline&prompt=consent"
}
```

---

### POST /auth/google/callback (Legacy)

Exchanges a Google authorization code for a JWT token and user info.

```bash
curl -X POST http://localhost:8080/auth/google/callback \
  -H "Content-Type: application/json" \
  -d '{"code": "4/0AQlEd8x..."}'
```

Response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "alice@example.com",
    "name": "Alice Dupont",
    "picture_url": "https://lh3.googleusercontent.com/a/photo",
    "is_root": false
  }
}
```

---

### GET /auth/oidc

Returns the OIDC authorization URL. Works with any configured OIDC provider (Google, Keycloak, Auth0, Azure AD, etc.).

```bash
curl http://localhost:8080/auth/oidc
```

Response:

```json
{
  "auth_url": "https://auth.example.com/realms/main/protocol/openid-connect/auth?client_id=...&redirect_uri=...&response_type=code&scope=openid+email+profile"
}
```

---

### POST /auth/oidc/callback

Exchanges an OIDC authorization code for a JWT token and user info.

```bash
curl -X POST http://localhost:8080/auth/oidc/callback \
  -H "Content-Type: application/json" \
  -d '{"code": "abc123..."}'
```

Response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "770b9500-g49d-62f6-c938-668877662222",
    "email": "bob@example.com",
    "name": "Bob Martin",
    "picture_url": "https://auth.example.com/avatar/bob",
    "is_root": false
  }
}
```

---

### GET /auth/me

Returns the currently authenticated user's profile. Requires a valid JWT.

```bash
curl http://localhost:8080/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

Response:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "alice@example.com",
  "name": "Alice Dupont",
  "picture_url": "https://lh3.googleusercontent.com/a/photo"
}
```

---

### POST /auth/refresh

Issues a new JWT token from a still-valid (non-expired) token. The new token has a fresh expiry window.

```bash
curl -X POST http://localhost:8080/auth/refresh \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

Response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...new-token..."
}
```

> **Tip:** Call this endpoint periodically (e.g., every 7 hours for an 8-hour token) to keep the session alive without requiring the user to re-authenticate.

---

## Route Protection

### Public Routes (no authentication required)

These routes are accessible without a JWT token:

| Path | Purpose |
|------|---------|
| `/health` | Health check |
| `/auth/providers` | Auth provider discovery |
| `/auth/login` | Password login |
| `/auth/register` | User registration |
| `/auth/google` | Google OAuth login initiation (legacy) |
| `/auth/google/callback` | Google OAuth code exchange (legacy) |
| `/auth/oidc` | OIDC login initiation |
| `/auth/oidc/callback` | OIDC code exchange |
| `/ws/events` | Server-sent events WebSocket (auth via first message) |
| `/ws/chat/{session_id}` | Chat WebSocket (auth via first message) |
| `/hooks/wake` | Webhook endpoint |
| `/internal/events` | Internal event receiver (deprecated) |

### Protected Routes (JWT required)

Everything under `/api/*` requires a valid `Authorization: Bearer <token>` header:

- `/api/projects/*` -- Project management
- `/api/plans/*` -- Plan management
- `/api/tasks/*` -- Task management
- `/api/notes/*` -- Knowledge notes
- `/api/workspaces/*` -- Workspace management
- `/api/code/*` -- Code exploration
- `/api/chat/*` -- Chat session management
- `/auth/me` -- User profile
- `/auth/refresh` -- Token refresh

### How the Middleware Works

The `require_auth` middleware runs on every protected route and performs these checks in order:

1. **Auth config present?** If `auth_config` is `None` -> inject anonymous claims and pass through (**no-auth mode**)
2. **Authorization header present?** If missing -> `401 Unauthorized`
3. **Bearer token format?** Must start with `Bearer ` -> `401 Unauthorized`
4. **JWT valid and not expired?** Decoded with HS256 secret -> `401 Unauthorized`
5. **Email domain allowed?** If `allowed_email_domain` is set, email must match -> `403 Forbidden`
6. **Email in whitelist?** If `allowed_emails` is set and domain check fails, check individual whitelist -> `403 Forbidden`
7. **Success:** Claims injected into request extensions for downstream handlers

### No-Auth Mode (Anonymous Claims)

When `auth_config` is `None` (no `auth` section in `config.yaml`), the middleware injects anonymous claims into every request:

| Field | Value |
|-------|-------|
| `sub` | `00000000-0000-0000-0000-000000000000` (nil UUID) |
| `email` | `anonymous@local` |
| `name` | `Anonymous` |

This allows the entire API to function without authentication for local development. The anonymous user ID is deterministic (nil UUID) and consistent across all requests.

---

## WebSocket Authentication

WebSocket connections cannot use HTTP `Authorization` headers (browser limitation). Instead, Project Orchestrator uses a **first-message handshake** protocol.

### Handshake Protocol

```
Client                              Server
  |                                    |
  |  1. WebSocket connect              |
  | ---------------------------------->|
  |  Connection established            |
  | <----------------------------------|
  |                                    |
  |  2. Send auth message              |
  |  {"type":"auth","token":"eyJ..."}  |
  | ---------------------------------->|
  |                                    |
  |  3a. Valid token:                  |
  |  {"type":"auth_ok","user":{...}}   |
  | <----------------------------------|
  |                                    |
  |  3b. Invalid token:               |
  |  {"type":"auth_error",             |
  |   "message":"Invalid token: ..."}  |
  | <----------------------------------|
  |  Connection closed                 |
  |                                    |
```

### No-Auth Mode for WebSocket

When `auth_config` is `None`, the WebSocket handshake is skipped entirely. The server immediately sends an `auth_ok` response with the anonymous user and the connection proceeds without requiring a token:

```json
{
  "type": "auth_ok",
  "user": {
    "id": "00000000-0000-0000-0000-000000000000",
    "email": "anonymous@local",
    "name": "Anonymous"
  }
}
```

### Client-Side Implementation

After connecting, send the auth message as the **first** WebSocket message:

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/events");

ws.onopen = () => {
  // Must be the FIRST message sent
  ws.send(JSON.stringify({
    type: "auth",
    token: "eyJhbGciOiJIUzI1NiIs..."
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "auth_ok") {
    console.log("Authenticated as:", msg.user.email);
    // Now you can send/receive regular messages
  } else if (msg.type === "auth_error") {
    console.error("Auth failed:", msg.message);
    // Connection will be closed by the server
  }
};
```

### Auth Message Format

**Client sends:**

```json
{
  "type": "auth",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Server responds on success:**

```json
{
  "type": "auth_ok",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "alice@example.com",
    "name": "Alice Dupont"
  }
}
```

**Server responds on failure:**

```json
{
  "type": "auth_error",
  "message": "Invalid token: ExpiredSignature"
}
```

### Timeout

The server waits **10 seconds** for the auth message. If no auth message is received within that window, the connection is closed with an `auth_error` message.

### Token Expiry During a Stream

The JWT token is validated **once** at connection time. If the token expires during an active WebSocket stream, the connection remains open. Use the `/auth/refresh` HTTP endpoint independently to obtain a new token for future connections.

---

## Development Without Auth

### MCP Server (stdio)

The MCP server binary (`mcp_server`) communicates over **stdin/stdout** and does not use HTTP. Authentication is not needed and not supported for MCP connections. The MCP server is designed to run locally alongside your AI tool (Claude Code, Cursor, etc.).

### HTTP API Server (No-Auth Mode)

For the HTTP API server, omitting the `auth` section from `config.yaml` activates **no-auth mode**:

| Config State | Behavior |
|-------------|----------|
| `auth` section completely absent | **No-auth mode** -- anonymous claims injected, all routes accessible |
| `auth` section present with providers | Full authentication active |
| `auth` section present, no providers | Auth enforced but no login method available (locked out) |

In no-auth mode, the middleware injects anonymous claims (`anonymous@local`, nil UUID) into every request, so all handlers continue to receive valid `Claims` in their request extensions.

---

## Security Notes

### JWT Secret

- Must be at least **32 characters** long
- Generate a strong random secret: `openssl rand -hex 32`
- Never commit the secret to version control
- Store it in environment variables or a secrets manager in production

### HTTPS

- Always use **HTTPS** in production
- OAuth/OIDC redirect URIs must use HTTPS in production (providers enforce this)
- Set `redirect_uri` values to HTTPS URLs in production config
- Configure `frontend_url` to restrict CORS to your production domain

### Password Security

- Passwords are hashed with **bcrypt** (cost factor 12)
- Minimum password length is **8 characters** (enforced at registration)
- Error messages on login never reveal whether the email exists (prevents user enumeration)
- The root account `password_hash` field accepts plaintext for convenience but logs a warning -- always use a proper bcrypt hash in production

### Email Restrictions

- Use `allowed_email_domain` to restrict access to your organization
- Use `allowed_emails` to whitelist specific individual emails (works in addition to domain restriction)
- An email is allowed if it matches `allowed_email_domain` **OR** is listed in `allowed_emails`
- These checks are applied both at login/registration time (handler) and at every request (middleware)

### Token Rotation

- Rotate the `jwt_secret` periodically
- When you rotate the secret, all existing tokens are immediately invalidated
- Users will need to re-authenticate after rotation
- Plan rotations during low-traffic periods

### User Data

- User records are stored in Neo4j with: `id`, `email`, `name`, `picture_url`, `auth_provider`, `external_id`, `password_hash`, `created_at`, `last_login_at`
- The `auth_provider` field distinguishes between `password` and `oidc` accounts
- Password users store a bcrypt hash in `password_hash`; OIDC users store the provider's subject ID in `external_id`
- The root account is never stored in Neo4j -- it is verified purely from `config.yaml` values in memory

---

## Troubleshooting

### "401 Unauthorized"

- **Missing header:** Ensure you include `Authorization: Bearer <token>` in every request to protected routes
- **Expired token:** Tokens expire after `jwt_expiry_secs` (default 8 hours). Use `/auth/refresh` before expiry
- **Invalid token:** The token may have been generated with a different `jwt_secret`. Re-authenticate
- **Malformed header:** The header must be exactly `Bearer <token>` (capital B, one space)

### "403 Forbidden"

- **Auth not configured:** The `auth` section is present but the specific provider you are trying to use is not configured. Check `GET /auth/providers` to see what is available
- **Email domain restriction:** Your email does not match the `allowed_email_domain` or `allowed_emails` whitelist
- **Registration disabled:** `POST /auth/register` returns 403 if `allow_registration` is `false`
- **Password auth not configured:** `POST /auth/login` returns 403 if no `root_account` is defined and no password users exist
- **Error message check:** The response body contains a JSON `message` field with the specific reason

### "Invalid email or password"

- The email may not exist or the password is wrong (the error is intentionally vague to prevent user enumeration)
- For the root account, verify the `password_hash` in `config.yaml` matches the password you are using
- For registered users, check that the account was created with `POST /auth/register`

### "OAuth/OIDC code exchange failed"

- **Expired code:** Authorization codes are single-use and expire quickly. Retry the login flow
- **Wrong redirect URI:** The `redirect_uri` in config must exactly match the authorized redirect URI in your provider's console
- **Invalid credentials:** Verify `client_id` and `client_secret` are correct
- **Discovery URL unreachable:** If using `discovery_url`, ensure the server can reach the `.well-known/openid-configuration` endpoint

### WebSocket Disconnects

- **No auth message sent:** The server expects an auth message within 10 seconds of connection. Send it immediately on `onopen`
- **Wrong message format:** The first message must be `{"type":"auth","token":"..."}`. Check for typos in the `type` field
- **Token issues:** Same token validation as HTTP applies. Check token validity
- **Timeout:** If you see "Authentication timeout", the auth message was not sent within the 10-second window
- **No-auth mode:** If auth is not configured, no auth message is needed. The server sends `auth_ok` with the anonymous user immediately on connect

### CORS Errors

- If `frontend_url` is set, only that origin (plus `tauri://localhost` for the desktop app) is allowed in CORS headers
- For local development with a different port, ensure `frontend_url` matches your frontend's origin exactly
- If `frontend_url` is not set, CORS allows any origin (suitable for development only)

---

## Quick Reference

### Minimal Config: Password Only

```yaml
auth:
  jwt_secret: "local-dev-secret-at-least-32-characters!"
  root_account:
    email: "admin@example.com"
    name: "Admin"
    password_hash: "my-password"  # will be auto-hashed at startup
```

### Minimal Config: OIDC Only

```yaml
auth:
  jwt_secret: "local-dev-secret-at-least-32-characters!"
  oidc:
    client_id: "your-client-id"
    client_secret: "your-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    discovery_url: "https://auth.example.com/.well-known/openid-configuration"
```

### Full Config: All Providers

```yaml
auth:
  jwt_secret: "$(openssl rand -hex 32)"
  jwt_expiry_secs: 28800
  root_account:
    email: "admin@company.com"
    name: "Admin"
    password_hash: "$2b$12$..."
  oidc:
    client_id: "your-client-id"
    client_secret: "your-secret"
    redirect_uri: "https://app.company.com/auth/callback"
    discovery_url: "https://auth.company.com/.well-known/openid-configuration"
    provider_name: "Company SSO"
  allow_registration: true
  allowed_email_domain: "company.com"
  frontend_url: "https://app.company.com"
```

### Common curl Commands

```bash
# Discover available auth providers
curl http://localhost:8080/auth/providers

# Password login
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "my-password"}'

# Register a new account (if enabled)
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "password": "at-least-8-chars", "name": "Alice"}'

# Get the OIDC login URL
curl http://localhost:8080/auth/oidc

# Exchange an OIDC auth code
curl -X POST http://localhost:8080/auth/oidc/callback \
  -H "Content-Type: application/json" \
  -d '{"code": "abc123..."}'

# Get the Google login URL (legacy)
curl http://localhost:8080/auth/google

# Exchange a Google auth code (legacy)
curl -X POST http://localhost:8080/auth/google/callback \
  -H "Content-Type: application/json" \
  -d '{"code": "4/0AQlEd8x..."}'

# Get current user info
curl http://localhost:8080/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Refresh a token
curl -X POST http://localhost:8080/auth/refresh \
  -H "Authorization: Bearer $TOKEN"

# Access a protected API endpoint
curl http://localhost:8080/api/projects \
  -H "Authorization: Bearer $TOKEN"
```
