// =============================================================================
// Migration 001 — User: google_id → auth_provider + external_id
// =============================================================================
//
// Context:
//   The UserNode model has been refactored from a Google-only auth model to a
//   multi-provider model. The `google_id` field is replaced by:
//   - `auth_provider`: "password" or "oidc"
//   - `external_id`: the provider's unique user identifier (e.g., Google "sub")
//   - `password_hash`: bcrypt hash (for password users only)
//
// This migration converts existing Google OAuth users to the new schema.
// It is idempotent — safe to run multiple times.
//
// Run with:
//   cypher-shell -u neo4j -p <password> < migrations/001_user_auth_provider.cypher
//
// Or in the Neo4j Browser, paste each statement separately.
// =============================================================================

// Step 1: Migrate existing Google users → OIDC provider
// Sets auth_provider and external_id from the legacy google_id field.
MATCH (u:User)
WHERE u.google_id IS NOT NULL AND u.auth_provider IS NULL
SET u.auth_provider = "oidc",
    u.external_id = u.google_id
RETURN count(u) AS migrated_users;

// Step 2: Remove the legacy google_id property
// Only remove after verifying Step 1 completed successfully.
MATCH (u:User)
WHERE u.google_id IS NOT NULL AND u.auth_provider IS NOT NULL AND u.external_id IS NOT NULL
REMOVE u.google_id
RETURN count(u) AS cleaned_users;

// Step 3: Create composite index for OIDC user lookup
// MERGE uses (auth_provider + external_id) for OIDC users.
CREATE INDEX user_provider_external_id IF NOT EXISTS
FOR (u:User) ON (u.auth_provider, u.external_id);

// Step 4: Create composite index for password user lookup
// MERGE uses (auth_provider + email) for password users.
CREATE INDEX user_provider_email IF NOT EXISTS
FOR (u:User) ON (u.auth_provider, u.email);

// Step 5: Verify migration
MATCH (u:User)
WHERE u.google_id IS NOT NULL
RETURN count(u) AS remaining_legacy_users;
// Expected result: 0
