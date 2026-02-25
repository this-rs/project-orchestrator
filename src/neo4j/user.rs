//! Neo4j User / Auth operations

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // User / Auth operations
    // ========================================================================

    /// Upsert a user: create if not exists, update if exists.
    ///
    /// For OIDC users: MERGE on (auth_provider + external_id).
    /// For password users: MERGE on (auth_provider + email).
    pub async fn upsert_user(&self, user: &UserNode) -> Result<UserNode> {
        use crate::neo4j::models::AuthProvider;

        let auth_provider_str = user.auth_provider.to_string();

        let q = match user.auth_provider {
            AuthProvider::Oidc => {
                let external_id = user
                    .external_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("OIDC user must have external_id"))?;
                query(
                    r#"
                    MERGE (u:User {auth_provider: $auth_provider, external_id: $external_id})
                    ON CREATE SET
                        u.id = $id,
                        u.email = $email,
                        u.name = $name,
                        u.picture_url = $picture_url,
                        u.password_hash = $password_hash,
                        u.created_at = datetime($created_at),
                        u.last_login_at = datetime($last_login_at)
                    ON MATCH SET
                        u.email = $email,
                        u.name = $name,
                        u.picture_url = $picture_url,
                        u.last_login_at = datetime($last_login_at)
                    RETURN u
                    "#,
                )
                .param("external_id", external_id.to_string())
            }
            AuthProvider::Password => query(
                r#"
                MERGE (u:User {auth_provider: $auth_provider, email: $email})
                ON CREATE SET
                    u.id = $id,
                    u.name = $name,
                    u.picture_url = $picture_url,
                    u.external_id = $external_id,
                    u.password_hash = $password_hash,
                    u.created_at = datetime($created_at),
                    u.last_login_at = datetime($last_login_at)
                ON MATCH SET
                    u.name = $name,
                    u.picture_url = $picture_url,
                    u.last_login_at = datetime($last_login_at)
                RETURN u
                "#,
            ),
        }
        .param("id", user.id.to_string())
        .param("auth_provider", auth_provider_str)
        .param("email", user.email.clone())
        .param("name", user.name.clone())
        .param("picture_url", user.picture_url.clone().unwrap_or_default())
        .param("external_id", user.external_id.clone().unwrap_or_default())
        .param(
            "password_hash",
            user.password_hash.clone().unwrap_or_default(),
        )
        .param("created_at", user.created_at.to_rfc3339())
        .param("last_login_at", user.last_login_at.to_rfc3339());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            self.node_to_user(&node)
        } else {
            anyhow::bail!("upsert_user: no row returned")
        }
    }

    /// Get a user by internal UUID
    pub async fn get_user_by_id(&self, id: Uuid) -> Result<Option<UserNode>> {
        let q = query("MATCH (u:User {id: $id}) RETURN u").param("id", id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a user by provider and external ID (for OIDC lookups)
    pub async fn get_user_by_provider_id(
        &self,
        provider: &str,
        external_id: &str,
    ) -> Result<Option<UserNode>> {
        let q =
            query("MATCH (u:User {auth_provider: $provider, external_id: $external_id}) RETURN u")
                .param("provider", provider)
                .param("external_id", external_id);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a user by email and auth provider
    pub async fn get_user_by_email_and_provider(
        &self,
        email: &str,
        provider: &str,
    ) -> Result<Option<UserNode>> {
        let q = query("MATCH (u:User {email: $email, auth_provider: $provider}) RETURN u")
            .param("email", email)
            .param("provider", provider);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Get a user by email (any provider)
    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<UserNode>> {
        let q = query("MATCH (u:User {email: $email}) RETURN u").param("email", email);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            Ok(Some(self.node_to_user(&node)?))
        } else {
            Ok(None)
        }
    }

    /// Create a password-authenticated user
    pub async fn create_password_user(
        &self,
        email: &str,
        name: &str,
        password_hash: &str,
    ) -> Result<UserNode> {
        let now = chrono::Utc::now();
        let user = UserNode {
            id: Uuid::new_v4(),
            email: email.to_string(),
            name: name.to_string(),
            picture_url: None,
            auth_provider: crate::neo4j::models::AuthProvider::Password,
            external_id: None,
            password_hash: Some(password_hash.to_string()),
            created_at: now,
            last_login_at: now,
        };
        self.upsert_user(&user).await
    }

    /// List all users
    pub async fn list_users(&self) -> Result<Vec<UserNode>> {
        let q = query("MATCH (u:User) RETURN u ORDER BY u.created_at DESC");

        let mut result = self.graph.execute(q).await?;
        let mut users = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("u")?;
            users.push(self.node_to_user(&node)?);
        }
        Ok(users)
    }

    /// Parse a Neo4j Node into a UserNode
    fn node_to_user(&self, node: &neo4rs::Node) -> Result<UserNode> {
        use crate::neo4j::models::AuthProvider;

        // Parse auth_provider with backward compat: if not present, check for google_id
        let auth_provider = node
            .get::<String>("auth_provider")
            .ok()
            .and_then(|s| s.parse::<AuthProvider>().ok())
            .unwrap_or_else(|| {
                // Legacy: if google_id exists, treat as OIDC
                if node.get::<String>("google_id").is_ok() {
                    AuthProvider::Oidc
                } else {
                    AuthProvider::Password
                }
            });

        // external_id: try new field first, fall back to legacy google_id
        let external_id = node
            .get::<String>("external_id")
            .ok()
            .and_then(|s| if s.is_empty() { None } else { Some(s) })
            .or_else(|| {
                node.get::<String>("google_id").ok().and_then(|s| {
                    if s.is_empty() {
                        None
                    } else {
                        Some(s)
                    }
                })
            });

        let password_hash = node.get::<String>("password_hash").ok().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

        Ok(UserNode {
            id: node.get::<String>("id")?.parse()?,
            email: node.get("email")?,
            name: node.get("name")?,
            picture_url: node.get::<String>("picture_url").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }),
            auth_provider,
            external_id,
            password_hash,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            last_login_at: node
                .get::<String>("last_login_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }

    // ================================================================
    // Refresh Tokens
    // ================================================================

    /// Store a new refresh token (hashed) linked to a user.
    pub async fn create_refresh_token(
        &self,
        user_id: Uuid,
        token_hash: &str,
        expires_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let q = query(
            "CREATE (rt:RefreshToken {
                token_hash: $token_hash,
                user_id: $user_id,
                expires_at: $expires_at,
                created_at: $created_at,
                revoked: false
            })",
        )
        .param("token_hash", token_hash.to_string())
        .param("user_id", user_id.to_string())
        .param("expires_at", expires_at.to_rfc3339())
        .param("created_at", chrono::Utc::now().to_rfc3339());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Validate a refresh token by its hash. Returns the token if valid
    /// (not expired, not revoked).
    pub async fn validate_refresh_token(
        &self,
        token_hash: &str,
    ) -> Result<Option<crate::neo4j::models::RefreshTokenNode>> {
        let q = query(
            "MATCH (rt:RefreshToken {token_hash: $token_hash})
             RETURN rt",
        )
        .param("token_hash", token_hash.to_string());

        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => {
                let node: neo4rs::Node = row.get("rt")?;
                let token = crate::neo4j::models::RefreshTokenNode {
                    token_hash: node.get("token_hash")?,
                    user_id: node.get::<String>("user_id")?.parse()?,
                    expires_at: node
                        .get::<String>("expires_at")?
                        .parse()
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    created_at: node
                        .get::<String>("created_at")?
                        .parse()
                        .unwrap_or_else(|_| chrono::Utc::now()),
                    revoked: node.get("revoked").unwrap_or(false),
                };

                // Check if expired or revoked
                if token.revoked || token.expires_at < chrono::Utc::now() {
                    Ok(None)
                } else {
                    Ok(Some(token))
                }
            }
            None => Ok(None),
        }
    }

    /// Revoke a single refresh token by its hash.
    pub async fn revoke_refresh_token(&self, token_hash: &str) -> Result<bool> {
        let q = query(
            "MATCH (rt:RefreshToken {token_hash: $token_hash})
             SET rt.revoked = true
             RETURN rt",
        )
        .param("token_hash", token_hash.to_string());

        let mut result = self.graph.execute(q).await?;
        Ok(result.next().await?.is_some())
    }

    /// Revoke all refresh tokens for a given user.
    pub async fn revoke_all_user_tokens(&self, user_id: Uuid) -> Result<u64> {
        let q = query(
            "MATCH (rt:RefreshToken {user_id: $user_id, revoked: false})
             SET rt.revoked = true
             RETURN count(rt) as count",
        )
        .param("user_id", user_id.to_string());

        let mut result = self.graph.execute(q).await?;
        match result.next().await? {
            Some(row) => {
                let count: i64 = row.get("count")?;
                Ok(count as u64)
            }
            None => Ok(0),
        }
    }
}
