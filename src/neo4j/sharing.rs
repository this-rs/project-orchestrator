//! Neo4j Sharing & Privacy operations

use super::client::Neo4jClient;
use crate::episodes::distill_models::{ConsentCounts, SharingConsent, SharingEvent, SharingPolicy};
use crate::reception::anchor::SignedTombstone;
use anyhow::{Context, Result};
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ========================================================================
    // Sharing policy operations
    // ========================================================================

    /// Get the sharing policy for a project.
    pub async fn get_sharing_policy(&self, project_id: Uuid) -> Result<Option<SharingPolicy>> {
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            RETURN p.sharing_policy AS policy
            "#,
        )
        .param("id", project_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let policy_json: String = row.get("policy").unwrap_or_default();
            if policy_json.is_empty() {
                return Ok(None);
            }
            let policy: SharingPolicy = serde_json::from_str(&policy_json)
                .context("Failed to deserialize sharing policy")?;
            Ok(Some(policy))
        } else {
            Ok(None)
        }
    }

    /// Update the sharing policy for a project.
    pub async fn update_sharing_policy(
        &self,
        project_id: Uuid,
        policy: &SharingPolicy,
    ) -> Result<()> {
        let policy_json = serde_json::to_string(policy)?;
        let q = query(
            r#"
            MATCH (p:Project {id: $id})
            SET p.sharing_policy = $policy
            "#,
        )
        .param("id", project_id.to_string())
        .param("policy", policy_json);

        self.graph.run(q).await?;
        Ok(())
    }

    // ========================================================================
    // Sharing consent operations
    // ========================================================================

    /// Get the sharing consent for a note.
    pub async fn get_sharing_consent(&self, note_id: Uuid) -> Result<SharingConsent> {
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            RETURN n.sharing_consent AS consent
            "#,
        )
        .param("id", note_id.to_string());

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let consent_str: String = row.get("consent").unwrap_or_default();
            let consent: SharingConsent =
                serde_json::from_str(&format!("\"{}\"", consent_str)).unwrap_or_default();
            Ok(consent)
        } else {
            Ok(SharingConsent::default())
        }
    }

    /// Update the sharing consent for a note.
    pub async fn update_sharing_consent(
        &self,
        note_id: Uuid,
        consent: &SharingConsent,
    ) -> Result<()> {
        let consent_str = serde_json::to_string(consent)?;
        // Remove quotes from the JSON string value
        let consent_str = consent_str.trim_matches('"');
        let q = query(
            r#"
            MATCH (n:Note {id: $id})
            SET n.sharing_consent = $consent
            "#,
        )
        .param("id", note_id.to_string())
        .param("consent", consent_str.to_string());

        self.graph.run(q).await?;
        Ok(())
    }

    /// Set the sharing consent of a decision.
    pub async fn update_decision_sharing_consent(
        &self,
        decision_id: Uuid,
        consent: &SharingConsent,
    ) -> Result<()> {
        let q = query("MATCH (d:Decision {id: $id}) SET d.sharing_consent = $consent")
            .param("id", decision_id.to_string())
            .param("consent", consent.as_db_str().to_string());
        self.graph.run(q).await?;
        Ok(())
    }

    /// Count notes per consent value (all statuses), for one project or all.
    /// A missing property counts as `NotSet`. Read-only.
    pub async fn count_notes_by_consent(&self, project_id: Option<Uuid>) -> Result<ConsentCounts> {
        let q = query(
            r#"
            MATCH (n:Note)
            WHERE $pid = '' OR n.project_id = $pid
            RETURN coalesce(n.sharing_consent, 'not_set') AS consent, count(n) AS k
            "#,
        )
        .param("pid", project_id.map(|p| p.to_string()).unwrap_or_default());

        let mut counts = ConsentCounts::default();
        let mut result = self.graph.execute(q).await?;
        while let Some(row) = result.next().await? {
            let consent: String = row.get("consent").unwrap_or_default();
            let k: i64 = row.get("k").unwrap_or(0);
            counts.add(SharingConsent::from_db_str(&consent), k.max(0) as u64);
        }
        Ok(counts)
    }

    // ========================================================================
    // Sharing event operations (audit trail)
    // ========================================================================

    /// Create a sharing event.
    pub async fn create_sharing_event(&self, event: &SharingEvent) -> Result<()> {
        let q = query(
            r#"
            CREATE (se:SharingEvent {
                id: $id,
                content_hash: $content_hash,
                artifact_type: $artifact_type,
                action: $action,
                source_did: $source_did,
                target_did: $target_did,
                timestamp: datetime($timestamp),
                consent: $consent,
                privacy_mode: $privacy_mode,
                reason: $reason
            })
            "#,
        )
        .param("id", event.id.clone())
        .param("content_hash", event.content_hash.clone())
        .param("artifact_type", event.artifact_type.clone())
        .param("action", event.action.clone())
        .param("source_did", event.source_did.clone())
        .param("target_did", event.target_did.clone())
        .param("timestamp", event.timestamp.to_rfc3339())
        .param(
            "consent",
            serde_json::to_string(&event.consent)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string(),
        )
        .param(
            "privacy_mode",
            serde_json::to_string(&event.privacy_mode)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string(),
        )
        .param("reason", event.reason.clone().unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List sharing events for a project (newest first).
    pub async fn list_sharing_events(
        &self,
        _project_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SharingEvent>> {
        let q = query(
            r#"
            MATCH (se:SharingEvent)
            RETURN se
            ORDER BY se.timestamp DESC
            SKIP $offset
            LIMIT $limit
            "#,
        )
        .param("offset", offset)
        .param("limit", limit);

        let mut result = self.graph.execute(q).await?;
        let mut events = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("se")?;
            let id: String = node.get("id").unwrap_or_default();
            let content_hash: String = node.get("content_hash").unwrap_or_default();
            let artifact_type: String = node.get("artifact_type").unwrap_or_default();
            let action: String = node.get("action").unwrap_or_default();
            let source_did: String = node.get("source_did").unwrap_or_default();
            let target_did: String = node.get("target_did").unwrap_or_default();
            let timestamp_str: String = node.get("timestamp").unwrap_or_default();
            let consent_str: String = node.get("consent").unwrap_or_default();
            let privacy_mode_str: String = node.get("privacy_mode").unwrap_or_default();
            let reason: String = node.get("reason").unwrap_or_default();

            events.push(SharingEvent {
                id,
                content_hash,
                artifact_type,
                action,
                source_did,
                target_did,
                timestamp: chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now()),
                consent: serde_json::from_str(&format!("\"{}\"", consent_str)).unwrap_or_default(),
                privacy_mode: serde_json::from_str(&format!("\"{}\"", privacy_mode_str))
                    .unwrap_or_default(),
                reason: if reason.is_empty() {
                    None
                } else {
                    Some(reason)
                },
            });
        }
        Ok(events)
    }

    // ========================================================================
    // Tombstone persistence
    // ========================================================================

    /// Persist a signed tombstone.
    pub async fn persist_tombstone(&self, tombstone: &SignedTombstone) -> Result<()> {
        let q = query(
            r#"
            MERGE (t:Tombstone {content_hash: $content_hash})
            SET t.issuer_did = $issuer_did,
                t.signature_hex = $signature_hex,
                t.issued_at = datetime($issued_at),
                t.reason = $reason
            "#,
        )
        .param("content_hash", tombstone.content_hash.clone())
        .param("issuer_did", tombstone.issuer_did.clone())
        .param("signature_hex", tombstone.signature_hex.clone())
        .param("issued_at", tombstone.issued_at.to_rfc3339())
        .param("reason", tombstone.reason.clone().unwrap_or_default());

        self.graph.run(q).await?;
        Ok(())
    }

    /// List all tombstones (including legacy placeholders; callers annotate
    /// them with `sharing::tombstone::annotate_tombstone`).
    pub async fn list_tombstones(&self) -> Result<Vec<SignedTombstone>> {
        // `issued_at` is stored as a Neo4j datetime: read it back as an ISO
        // string with toString(), a typed String read of a DateTime fails.
        let q = query(
            r#"
            MATCH (t:Tombstone)
            RETURN t.content_hash AS content_hash,
                   t.issuer_did AS issuer_did,
                   t.signature_hex AS signature_hex,
                   toString(t.issued_at) AS issued_at,
                   t.reason AS reason
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut tombstones = Vec::new();
        while let Some(row) = result.next().await? {
            let content_hash: String = row.get("content_hash").unwrap_or_default();
            let issuer_did: String = row.get("issuer_did").unwrap_or_default();
            let signature_hex: String = row.get("signature_hex").unwrap_or_default();
            let issued_at: String = row.get("issued_at").unwrap_or_default();
            let reason: String = row.get("reason").unwrap_or_default();
            tombstones.push(tombstone_from_stored(
                content_hash,
                issuer_did,
                signature_hex,
                &issued_at,
                reason,
            ));
        }
        Ok(tombstones)
    }

    /// Check if a content hash has been tombstoned by a non-legacy tombstone
    /// (placeholder entries with a fake signature never count).
    pub async fn is_tombstoned(&self, content_hash: &str) -> Result<bool> {
        let q = query(
            r#"
            MATCH (t:Tombstone {content_hash: $hash})
            WHERE t.signature_hex IS NOT NULL
              AND NOT t.signature_hex =~ '^0*$'
              AND t.issuer_did IS NOT NULL
              AND t.issuer_did <> ''
              AND t.issuer_did <> 'did:local:unknown'
            RETURN count(t) > 0 AS exists
            "#,
        )
        .param("hash", content_hash);

        let mut result = self.graph.execute(q).await?;
        if let Some(row) = result.next().await? {
            let exists: bool = row.get("exists").unwrap_or(false);
            Ok(exists)
        } else {
            Ok(false)
        }
    }
}

/// Rebuild a [`SignedTombstone`] from stored fields.
///
/// An unparsable `issued_at` falls back to the Unix epoch, NOT `Utc::now()`:
/// a clock-dependent fallback would change the signed payload on every read,
/// while the epoch is deterministic and simply fails verification.
fn tombstone_from_stored(
    content_hash: String,
    issuer_did: String,
    signature_hex: String,
    issued_at: &str,
    reason: String,
) -> SignedTombstone {
    SignedTombstone {
        content_hash,
        issuer_did,
        signature_hex,
        issued_at: chrono::DateTime::parse_from_rfc3339(issued_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or(chrono::DateTime::UNIX_EPOCH),
        reason: if reason.is_empty() {
            None
        } else {
            Some(reason)
        },
    }
}

#[cfg(test)]
mod tombstone_roundtrip_tests {
    use super::*;
    use crate::sharing::tombstone::{sign_tombstone, verify_tombstone};

    /// sign -> store (datetime() of the RFC 3339 param; toString() on read;
    /// None reason stored as "") -> reload -> verify.
    #[test]
    fn signature_survives_simulated_persistence() {
        let id = crate::identity::InstanceIdentity::generate();
        for reason in [None, Some("gdpr".to_string())] {
            let t = sign_tombstone(&id, "hash".into(), chrono::Utc::now(), reason);
            // Neo4j's toString(datetime) is ISO-8601 with up to 9 fractional digits.
            let stored_at = t
                .issued_at
                .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
            let back = tombstone_from_stored(
                t.content_hash.clone(),
                t.issuer_did.clone(),
                t.signature_hex.clone(),
                &stored_at,
                t.reason.clone().unwrap_or_default(),
            );
            assert_eq!(back.issued_at, t.issued_at);
            assert!(verify_tombstone(&back));
        }
    }

    #[test]
    fn unparsable_issued_at_is_deterministic_and_unverifiable() {
        let id = crate::identity::InstanceIdentity::generate();
        let t = sign_tombstone(&id, "hash".into(), chrono::Utc::now(), None);
        let a = tombstone_from_stored(
            "hash".into(),
            t.issuer_did.clone(),
            t.signature_hex.clone(),
            "",
            String::new(),
        );
        let b = tombstone_from_stored(
            "hash".into(),
            t.issuer_did.clone(),
            t.signature_hex.clone(),
            "",
            String::new(),
        );
        assert_eq!(a.issued_at, b.issued_at);
        assert!(!verify_tombstone(&a));
    }
}
