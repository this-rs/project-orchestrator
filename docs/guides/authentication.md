# Authentication Guide

JWT-based authentication with Google OAuth2 login for the Project Orchestrator HTTP API and WebSocket connections.

**Applies to:** HTTP API server (`api_server` binary)
**Does not apply to:** MCP server (`mcp_server` binary, runs locally via stdio)

---

## Overview

Project Orchestrator uses **JWT HS256 tokens** issued after a **Google OAuth2** login flow. The security model is **deny-by-default**: if the `auth` section is absent from the configuration, ALL requests to `/api/*` and protected routes are rejected with `403 Forbidden`.

Key principles:

- **Google OAuth2** is the only identity provider (no username/password)
- **JWT tokens** are stateless and short-lived (default 8 hours)
- **Email domain restriction** optionally limits access to a specific organization
- **WebSocket auth** uses a first-message handshake (browsers cannot set HTTP headers on WebSocket upgrades)
- **MCP server** runs locally over stdio and does not require authentication

---

## Architecture

The authentication flow works as follows:

```
Frontend                    Server                        Google
   │                          │                              │
   │  1. GET /auth/google     │                              │
   │ ─────────────────────>   │                              │
   │  { auth_url: "..." }     │                              │
   │ <─────────────────────   │                              │
   │                          │                              │
   │  2. Redirect to Google   │                              │
   │ ──────────────────────────────────────────────────────> │
   │                          │                              │
   │  3. User consents, Google redirects with ?code=...      │
   │ <────────────────────────────────────────────────────── │
   │                          │                              │
   │  4. POST /auth/google/callback { code: "..." }          │
   │ ─────────────────────>   │                              │
   │                          │  5. Exchange code for token  │
   │                          │ ────────────────────────────>│
   │                          │  6. Fetch user info          │
   │                          │ ────────────────────────────>│
   │                          │  { email, name, picture }    │
   │                          │ <────────────────────────────│
   │                          │                              │
   │                          │  7. Upsert user in Neo4j     │
   │                          │  8. Generate JWT (HS256)     │
   │                          │                              │
   │  { token, user }         │                              │
   │ <─────────────────────   │                              │
   │                          │                              │
   │  9. Authorization: Bearer <token>                       │
   │ ─────────────────────>   │                              │
   │  Protected resource      │                              │
   │ <─────────────────────   │                              │
```

### JWT Claims

The JWT payload contains:

| Field | Type | Description |
|-------|------|-------------|
| `sub` | UUID | User ID (from Neo4j) |
| `email` | String | User's Google email |
| `name` | String | User's display name |
| `iat` | Unix timestamp | Issued at |
| `exp` | Unix timestamp | Expiration |

---

## Configuration

Add the `auth` section to your `config.yaml`:

```yaml
auth:
  google_client_id: "YOUR_CLIENT_ID.apps.googleusercontent.com"
  google_client_secret: "YOUR_CLIENT_SECRET"
  google_redirect_uri: "http://localhost:3000/auth/callback"
  jwt_secret: "change-me-to-a-random-32-char-string!"
  jwt_expiry_secs: 28800  # 8 hours (default)
  allowed_email_domain: "example.com"  # optional — restrict to this domain
  frontend_url: "http://localhost:3000"  # optional — restricts CORS origin
```

### Configuration Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `google_client_id` | Yes | — | OAuth 2.0 client ID from Google Cloud Console |
| `google_client_secret` | Yes | — | OAuth 2.0 client secret |
| `google_redirect_uri` | Yes | — | Must match the authorized redirect URI in Google Console |
| `jwt_secret` | Yes | — | Secret key for HS256 signing (minimum 32 characters) |
| `jwt_expiry_secs` | No | `28800` | Token lifetime in seconds (8 hours) |
| `allowed_email_domain` | No | `None` | If set, only emails ending with `@<domain>` are allowed |
| `frontend_url` | No | `None` | If set, CORS is restricted to this origin; otherwise allows any origin |

> **Important:** If the entire `auth` section is omitted, deny-by-default activates and ALL protected routes return `403 Forbidden`. This is a safety measure to prevent accidental unauthenticated access.

---

## Setting Up Google OAuth

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click **Select a project** at the top, then **New Project**
3. Enter a project name (e.g., "Project Orchestrator") and click **Create**

### Step 2: Configure the OAuth Consent Screen

1. Navigate to **APIs & Services > OAuth consent screen**
2. Choose **External** (or **Internal** for Google Workspace organizations)
3. Fill in the required fields:
   - App name
   - User support email
   - Developer contact email
4. Under **Scopes**, add: `openid`, `email`, `profile`
5. Save and continue

### Step 3: Create OAuth 2.0 Credentials

1. Navigate to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth 2.0 Client ID**
3. Select **Web application** as the application type
4. Under **Authorized redirect URIs**, add your callback URL:
   - For local development: `http://localhost:3000/auth/callback`
   - For production: `https://your-domain.com/auth/callback`
5. Click **Create**

### Step 4: Copy Credentials to Config

Copy the **Client ID** and **Client Secret** from the credentials page into your `config.yaml`:

```yaml
auth:
  google_client_id: "123456789-abcdef.apps.googleusercontent.com"
  google_client_secret: "GOCSPX-your-secret-here"
  google_redirect_uri: "http://localhost:3000/auth/callback"
  jwt_secret: "$(openssl rand -hex 32)"  # generate a strong secret
```

---

## API Endpoints

### Authentication Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/auth/google` | Public | Returns the Google OAuth authorization URL |
| `POST` | `/auth/google/callback` | Public | Exchanges auth code for JWT + user info |
| `GET` | `/auth/me` | Protected | Returns the authenticated user's profile |
| `POST` | `/auth/refresh` | Protected | Issues a fresh JWT from a still-valid token |

---

### GET /auth/google

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

### POST /auth/google/callback

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
    "picture_url": "https://lh3.googleusercontent.com/a/photo"
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
| `/auth/google` | OAuth login initiation |
| `/auth/google/callback` | OAuth code exchange |
| `/ws/events` | Server-sent events WebSocket (auth via first message) |
| `/ws/chat/{session_id}` | Chat WebSocket (auth via first message) |
| `/hooks/wake` | Webhook endpoint |
| `/internal/events` | Internal event receiver |

### Protected Routes (JWT required)

Everything under `/api/*` requires a valid `Authorization: Bearer <token>` header:

- `/api/projects/*` — Project management
- `/api/plans/*` — Plan management
- `/api/tasks/*` — Task management
- `/api/notes/*` — Knowledge notes
- `/api/workspaces/*` — Workspace management
- `/api/code/*` — Code exploration
- `/api/chat/*` — Chat session management
- `/auth/me` — User profile
- `/auth/refresh` — Token refresh

### How the Middleware Works

The `require_auth` middleware runs on every protected route and performs these checks in order:

1. **Auth config present?** If `auth_config` is `None` -> `403 Forbidden`
2. **Authorization header present?** If missing -> `401 Unauthorized`
3. **Bearer token format?** Must start with `Bearer ` -> `401 Unauthorized`
4. **JWT valid and not expired?** Decoded with HS256 secret -> `401 Unauthorized`
5. **Email domain allowed?** If `allowed_email_domain` is set, email must match -> `403 Forbidden`
6. **Success:** Claims injected into request extensions for downstream handlers

---

## WebSocket Authentication

WebSocket connections cannot use HTTP `Authorization` headers (browser limitation). Instead, Project Orchestrator uses a **first-message handshake** protocol.

### Handshake Protocol

```
Client                              Server
  │                                    │
  │  1. WebSocket connect              │
  │ ──────────────────────────────────>│
  │  Connection established            │
  │ <──────────────────────────────────│
  │                                    │
  │  2. Send auth message              │
  │  {"type":"auth","token":"eyJ..."}  │
  │ ──────────────────────────────────>│
  │                                    │
  │  3a. Valid token:                  │
  │  {"type":"auth_ok","user":{...}}   │
  │ <──────────────────────────────────│
  │                                    │
  │  3b. Invalid token:               │
  │  {"type":"auth_error",             │
  │   "message":"Invalid token: ..."}  │
  │ <──────────────────────────────────│
  │  Connection closed                 │
  │                                    │
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

### HTTP API Server

For the HTTP API server, the `auth` section in `config.yaml` controls authentication behavior:

| Config State | Behavior |
|-------------|----------|
| `auth` section present with valid values | Full OAuth + JWT authentication active |
| `auth` section completely absent | **Deny-by-default** — all protected routes return `403 Forbidden` |

There is no "disable auth" mode. If you need API access, you must configure the `auth` section with valid Google OAuth credentials.

---

## Security Notes

### JWT Secret

- Must be at least **32 characters** long
- Generate a strong random secret: `openssl rand -hex 32`
- Never commit the secret to version control
- Store it in environment variables or a secrets manager in production

### HTTPS

- Always use **HTTPS** in production
- Google OAuth redirect URIs must use HTTPS in production (Google enforces this)
- Set `google_redirect_uri` to an HTTPS URL in production config
- Configure `frontend_url` to restrict CORS to your production domain

### Email Domain Restriction

- Use `allowed_email_domain` to restrict access to your organization
- The check is applied both at login time (callback handler) and at every request (middleware)
- Example: setting `allowed_email_domain: "example.com"` only allows `@example.com` emails

### Token Rotation

- Rotate the `jwt_secret` periodically
- When you rotate the secret, all existing tokens are immediately invalidated
- Users will need to re-authenticate after rotation
- Plan rotations during low-traffic periods

### User Data

- User records are stored in Neo4j with: `id`, `email`, `name`, `picture_url`, `google_id`, `created_at`, `last_login_at`
- On each login, the user record is upserted (created or updated)
- No passwords are stored — authentication is fully delegated to Google

---

## Troubleshooting

### "401 Unauthorized"

- **Missing header:** Ensure you include `Authorization: Bearer <token>` in every request to protected routes
- **Expired token:** Tokens expire after `jwt_expiry_secs` (default 8 hours). Use `/auth/refresh` before expiry
- **Invalid token:** The token may have been generated with a different `jwt_secret`. Re-authenticate via `/auth/google`
- **Malformed header:** The header must be exactly `Bearer <token>` (capital B, one space)

### "403 Forbidden"

- **Auth not configured:** The `auth` section is missing from `config.yaml`. Add it with valid Google OAuth credentials
- **Email domain restriction:** Your email does not match the `allowed_email_domain`. Check the configured domain
- **Error message check:** The response body contains a JSON `message` field with the specific reason

### "OAuth code exchange failed"

- **Expired code:** Authorization codes are single-use and expire quickly. Retry the login flow
- **Wrong redirect URI:** The `google_redirect_uri` in config must exactly match the authorized redirect URI in Google Cloud Console
- **Invalid credentials:** Verify `google_client_id` and `google_client_secret` are correct

### WebSocket Disconnects

- **No auth message sent:** The server expects an auth message within 10 seconds of connection. Send it immediately on `onopen`
- **Wrong message format:** The first message must be `{"type":"auth","token":"..."}`. Check for typos in the `type` field
- **Token issues:** Same token validation as HTTP applies. Check token validity
- **Timeout:** If you see "Authentication timeout", the auth message was not sent within the 10-second window

### CORS Errors

- If `frontend_url` is set, only that origin is allowed in CORS headers
- For local development with a different port, ensure `frontend_url` matches your frontend's origin exactly
- If `frontend_url` is not set, CORS allows any origin (suitable for development only)

---

## Quick Reference

### Minimal Config for Local Development

```yaml
auth:
  google_client_id: "YOUR_ID.apps.googleusercontent.com"
  google_client_secret: "YOUR_SECRET"
  google_redirect_uri: "http://localhost:3000/auth/callback"
  jwt_secret: "local-dev-secret-at-least-32-characters!"
```

### Common curl Commands

```bash
# Get the Google login URL
curl http://localhost:8080/auth/google

# Exchange an auth code (after Google redirect)
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
