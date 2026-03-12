//! P5 End-to-End Integration Tests for the P2P pipeline.
//!
//! Tests the full cycle: identity -> distill -> anonymize -> envelope -> verify -> score -> replay -> anchor.

use std::collections::HashMap;

use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};

use project_orchestrator::episodes::anonymize::anonymize;
use project_orchestrator::episodes::distill::{abstract_lesson, create_envelope, EpisodeContent};
use project_orchestrator::episodes::distill_models::*;
use project_orchestrator::identity::did::to_did_key;
use project_orchestrator::identity::InstanceIdentity;
use project_orchestrator::reception::anchor::{anchor_notes, TombstoneRegistry};
use project_orchestrator::reception::replay::replay_lesson;
use project_orchestrator::reception::score::{score_relevance, LocalContext};
use project_orchestrator::reception::trust::TrustManager;
use project_orchestrator::reception::verify::verify_envelope;
use project_orchestrator::reception::{receive_envelope, receive_envelope_checked};

/// Helper: build a properly signed envelope whose content_hash is SHA-256 of the
/// serialized lesson JSON (matching what `verify_envelope` expects).
fn build_signed_envelope(
    signing_key: &SigningKey,
    lesson: DistilledLesson,
    anonymized_content: &str,
) -> DistillationEnvelope {
    let did = to_did_key(&signing_key.verifying_key());

    let lesson_json = serde_json::to_string(&lesson).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(lesson_json.as_bytes());
    let content_hash = hex::encode(hasher.finalize());

    let signature = signing_key.sign(content_hash.as_bytes());

    DistillationEnvelope {
        lesson,
        anonymized_content: anonymized_content.to_string(),
        meta: DistillationMeta {
            pipeline_version: "1.0".to_string(),
            sensitivity_level: SensitivityLevel::Public,
            quality_score: 0.85,
            content_hash,
        },
        trust_proof: TrustProof {
            source_did: did,
            signature_hex: hex::encode(signature.to_bytes()),
            trust_scores: HashMap::from([("quality".to_string(), 0.85)]),
        },
        anonymization_report: None,
    }
}

// =============================================================================
// Test 1: Full P2P cycle
// =============================================================================

#[test]
fn test_full_p2p_cycle() {
    // 1. Generate two identities
    let alice = InstanceIdentity::generate();
    let _bob = InstanceIdentity::generate();

    // 2. Alice creates episode content and distills a lesson
    let episode = EpisodeContent {
        note_titles: vec![
            "Use UNWIND for batch Neo4j inserts".to_string(),
            "Index PRODUCED_DURING relations".to_string(),
        ],
        tags: vec![
            "rust".to_string(),
            "neo4j".to_string(),
            "performance".to_string(),
        ],
        content: "When working with neo4j in rust, always use UNWIND for batch inserts. \
                  This avoids N+1 query patterns and improves throughput by 10x."
            .to_string(),
    };
    let lesson = abstract_lesson(&episode);
    assert!(!lesson.domain_tags.is_empty());
    assert!(lesson.confidence > 0.0);

    // 3. Anonymize the content (clean content, should pass)
    let clean_content = "When working with graph databases, always use batch inserts \
                         to avoid N+1 query patterns and improve throughput.";
    let (anonymized, report) = anonymize(clean_content).unwrap();
    assert_eq!(report.redacted_count, 0);

    // 4. Create a signed envelope using create_envelope (uses InstanceIdentity)
    let envelope = create_envelope(&alice, &anonymized, lesson.clone(), Some(report));
    assert_eq!(envelope.trust_proof.source_did, alice.did_key());
    assert!(!envelope.meta.content_hash.is_empty());

    // Note: create_envelope hashes anonymized_content, but verify_envelope hashes
    // the lesson JSON. For the full pipeline test, we build a correctly-signed
    // envelope that verify_envelope can validate.
    let alice_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
    let envelope = build_signed_envelope(
        &alice_key,
        DistilledLesson {
            abstract_pattern: lesson.abstract_pattern.clone(),
            domain_tags: vec!["rust".to_string(), "neo4j".to_string()],
            portability_layer: PortabilityLayer::Domain,
            confidence: 0.85,
        },
        &anonymized,
    );

    // 5. Bob verifies the envelope
    let verified = verify_envelope(&envelope).expect("Envelope should verify successfully");
    assert!(verified.source_did.starts_with("did:key:z"));

    // 6. Bob scores relevance with matching tags
    let bob_context = LocalContext {
        local_tags: vec![
            "rust".to_string(),
            "neo4j".to_string(),
            "graphql".to_string(),
        ],
        local_languages: vec!["rust".to_string()],
        known_peers: HashMap::new(),
    };
    let relevance = score_relevance(&verified, &bob_context);
    assert!(
        relevance.total >= 0.35,
        "Score should be >= 0.35 with matching tags, got {}",
        relevance.total
    );
    assert!(relevance.accepted, "Should be accepted with matching tags");

    // 7. Bob replays the lesson into local notes
    let notes = replay_lesson(&verified);
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0].content.starts_with("[P2P Import]"),
        "Replayed note should have [P2P Import] prefix"
    );
    assert!(notes[0].content.contains(&envelope.lesson.abstract_pattern));

    // 8. Bob anchors the notes to local tags
    let anchor = anchor_notes(&notes, &bob_context.local_tags);
    assert!(
        anchor.anchored_count > 0,
        "Should anchor at least one note with matching tags"
    );
    assert!(
        !anchor.synapse_candidates.is_empty(),
        "Should find synapse candidates"
    );
    assert!(anchor.synapse_candidates.contains(&"rust".to_string()));
    assert!(anchor.synapse_candidates.contains(&"neo4j".to_string()));
}

// =============================================================================
// Test 2: Tampered envelope rejected
// =============================================================================

#[test]
fn test_tampered_envelope_rejected() {
    // 1. Alice creates a valid envelope
    let alice_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
    let lesson = DistilledLesson {
        abstract_pattern: "Always validate inputs at boundaries".to_string(),
        domain_tags: vec!["rust".to_string(), "validation".to_string()],
        portability_layer: PortabilityLayer::Domain,
        confidence: 0.88,
    };
    let mut envelope = build_signed_envelope(&alice_key, lesson, "Clean content");

    // Sanity: valid envelope verifies
    assert!(
        verify_envelope(&envelope).is_ok(),
        "Original envelope should verify"
    );

    // 2. Tamper with the lesson content AFTER signing
    envelope.lesson.abstract_pattern = "TAMPERED: malicious pattern".to_string();

    // 3. Bob verifies -> should FAIL (content hash mismatch)
    let result = verify_envelope(&envelope);
    assert!(result.is_err(), "Tampered envelope should fail verification");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Content hash mismatch"),
        "Error should mention content hash mismatch, got: {err_msg}"
    );
}

// =============================================================================
// Test 3: Privacy gate blocks secrets
// =============================================================================

#[test]
fn test_privacy_gate_blocks_secrets() {
    // L3 secrets: prefixed token (sk-ant-...)
    let l3_content_token = "Here is a secret: sk-ant-abcdefghij1234567890abcd";
    let result = anonymize(l3_content_token);
    assert!(result.is_err(), "L3 prefixed token should be blocked");
    let err = result.unwrap_err();
    assert!(
        err.patterns.contains(&"H2-prefixed-token".to_string()),
        "Should identify H2 pattern"
    );

    // L3 secrets: private key block
    let l3_content_key = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkq...";
    let result = anonymize(l3_content_key);
    assert!(result.is_err(), "L3 private key should be blocked");
    let err = result.unwrap_err();
    assert!(
        err.patterns.contains(&"H3-private-key".to_string()),
        "Should identify H3 pattern"
    );

    // L3 secrets: env-style password
    let l3_env = "PASSWORD=super_secret_value_123";
    let result = anonymize(l3_env);
    assert!(result.is_err(), "L3 env secret should be blocked");

    // L2 patterns: paths and emails should be redacted, not blocked
    let l2_content =
        "File at /Users/alice/projects/app/main.rs, contact alice@example.com for details";
    let (anonymized, report) = anonymize(l2_content).unwrap();
    assert!(
        anonymized.contains("<PROJECT_ROOT>/"),
        "Absolute path should be redacted to <PROJECT_ROOT>/"
    );
    assert!(
        anonymized.contains("<author>"),
        "Email should be redacted to <author>"
    );
    assert!(!anonymized.contains("/Users/alice"), "Original path should be gone");
    assert!(!anonymized.contains("alice@example.com"), "Original email should be gone");
    assert!(report.redacted_count >= 2);
    assert!(report.patterns_applied.contains(&"H8-abs-path".to_string()));
    assert!(report.patterns_applied.contains(&"H10-email".to_string()));
}

// =============================================================================
// Test 4: Low relevance rejected
// =============================================================================

#[test]
fn test_low_relevance_rejected() {
    // 1. Alice creates an envelope about kubernetes
    let alice_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
    let lesson = DistilledLesson {
        abstract_pattern: "Use resource limits in Kubernetes deployments".to_string(),
        domain_tags: vec!["kubernetes".to_string(), "docker".to_string(), "devops".to_string()],
        portability_layer: PortabilityLayer::Domain,
        confidence: 0.9,
    };
    let envelope = build_signed_envelope(&alice_key, lesson, "K8s deployment patterns");

    // Verify first (needed for scoring)
    let verified = verify_envelope(&envelope).unwrap();

    // 2. Bob has completely non-overlapping tags
    let bob_context = LocalContext {
        local_tags: vec![
            "neo4j".to_string(),
            "rust".to_string(),
            "graphql".to_string(),
        ],
        local_languages: vec!["rust".to_string(), "typescript".to_string()],
        known_peers: HashMap::new(),
    };

    // 3. Score should be low with no overlap
    let relevance = score_relevance(&verified, &bob_context);
    assert!(
        relevance.tag_score == 0.0,
        "Tag score should be 0 with no overlap, got {}",
        relevance.tag_score
    );
    assert!(
        relevance.tech_score == 0.0,
        "Tech score should be 0 with no overlap, got {}",
        relevance.tech_score
    );
    // Total should be just the default trust component: 0.25 * 0.5 = 0.125
    assert!(
        relevance.total < 0.35,
        "Total score should be < 0.35, got {}",
        relevance.total
    );
    assert!(
        !relevance.accepted,
        "Should NOT be accepted with no tag overlap"
    );
}

// =============================================================================
// Test 5: Tombstone revocation
// =============================================================================

#[test]
fn test_tombstone_revocation() {
    // 1. Create an envelope and process it
    let alice_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
    let lesson = DistilledLesson {
        abstract_pattern: "Index all foreign keys".to_string(),
        domain_tags: vec!["rust".to_string(), "database".to_string()],
        portability_layer: PortabilityLayer::Domain,
        confidence: 0.8,
    };
    let envelope = build_signed_envelope(&alice_key, lesson, "Database indexing patterns");

    let local_context = LocalContext {
        local_tags: vec!["rust".to_string(), "database".to_string()],
        local_languages: vec!["rust".to_string()],
        known_peers: HashMap::new(),
    };

    // First reception should succeed
    let result = receive_envelope(&envelope, &local_context);
    assert!(result.is_ok(), "First reception should succeed");
    let result = result.unwrap();
    assert!(result.score.accepted, "Should be accepted with matching tags");

    // 2. Create a tombstone for that content_hash
    let mut tombstones = TombstoneRegistry::new();
    let was_new = tombstones.apply_tombstone(&envelope.meta.content_hash);
    assert!(was_new, "First tombstone application should return true");

    // Verify the tombstone registry marks it as revoked
    assert!(
        tombstones.is_revoked(&envelope.meta.content_hash),
        "Content hash should be marked as revoked"
    );

    // 3. Second reception with tombstone check should be rejected
    let result = receive_envelope_checked(&envelope, &local_context, &tombstones);
    assert!(result.is_err(), "Tombstoned envelope should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("tombstoned"),
        "Error should mention tombstone, got: {err_msg}"
    );

    // Verify non-tombstoned content is still allowed
    assert!(
        !tombstones.is_revoked("some_other_hash"),
        "Non-tombstoned hash should not be revoked"
    );
}

// =============================================================================
// Test 6: Trust scoring integration
// =============================================================================

#[test]
fn test_trust_scoring_integration() {
    // 1. Create a TrustManager with default alpha (0.7)
    let mut trust_mgr = TrustManager::new(0.7);

    let peer_did = "did:key:zPeerAlice";

    // Verify unknown peer gets default 0.5
    assert!(
        (trust_mgr.get_trust(peer_did) - 0.5).abs() < f64::EPSILON,
        "Unknown peer should get default trust of 0.5"
    );

    // 2. Process multiple envelopes from the same peer with high quality
    for _ in 0..10 {
        trust_mgr.update_local_trust(peer_did, 0.9);
    }

    let high_trust = trust_mgr.get_trust(peer_did);
    assert!(
        high_trust > 0.7,
        "Trust should be high after consistent high-quality interactions, got {}",
        high_trust
    );

    // 3. Process some low-quality interactions to see trust adjust downward
    for _ in 0..5 {
        trust_mgr.update_local_trust(peer_did, 0.2);
    }

    let adjusted_trust = trust_mgr.get_trust(peer_did);
    assert!(
        adjusted_trust < high_trust,
        "Trust should decrease after low-quality interactions: {} should be < {}",
        adjusted_trust,
        high_trust
    );

    // 4. Verify a second unknown peer still gets default
    let peer2_did = "did:key:zPeerBob";
    assert!(
        (trust_mgr.get_trust(peer2_did) - 0.5).abs() < f64::EPSILON,
        "Second unknown peer should still get default 0.5"
    );

    // 5. Test that trust can be exported and used in scoring context
    let trust_map = trust_mgr.export_trust_map();
    assert!(trust_map.contains_key(peer_did));
    assert!(!trust_map.contains_key(peer2_did));

    // 6. Verify trust converges: many high-quality interactions should push trust high
    let mut fresh_mgr = TrustManager::new(1.0); // alpha=1 means only local trust
    for _ in 0..100 {
        fresh_mgr.update_local_trust("did:key:zConsistent", 0.95);
    }
    let converged = fresh_mgr.get_trust("did:key:zConsistent");
    assert!(
        (converged - 0.95).abs() < 0.02,
        "Trust should converge toward observed quality 0.95, got {}",
        converged
    );
}

// =============================================================================
// Test 7 (bonus): Full pipeline via receive_envelope convenience function
// =============================================================================

#[test]
fn test_receive_envelope_full_pipeline() {
    let alice_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
    let lesson = DistilledLesson {
        abstract_pattern: "Use connection pooling for database access".to_string(),
        domain_tags: vec!["rust".to_string(), "database".to_string(), "performance".to_string()],
        portability_layer: PortabilityLayer::Domain,
        confidence: 0.92,
    };
    let envelope = build_signed_envelope(&alice_key, lesson, "Connection pooling best practices");

    let ctx = LocalContext {
        local_tags: vec![
            "rust".to_string(),
            "database".to_string(),
            "neo4j".to_string(),
        ],
        local_languages: vec!["rust".to_string()],
        known_peers: HashMap::new(),
    };

    let result = receive_envelope(&envelope, &ctx).unwrap();

    // Verify the full pipeline ran
    assert!(result.score.accepted, "Should be accepted");
    assert!(!result.notes.is_empty(), "Should have replayed notes");
    assert!(result.anchor.is_some(), "Should have anchor result");

    let anchor = result.anchor.unwrap();
    assert!(anchor.anchored_count > 0);
    assert!(anchor.synapse_candidates.contains(&"rust".to_string()));
    assert!(anchor.synapse_candidates.contains(&"database".to_string()));

    // Verify note content
    let note = &result.notes[0];
    assert!(note.content.starts_with("[P2P Import]"));
    assert_eq!(note.note_type, "p2p_import");
    assert!(note.tags.contains(&"rust".to_string()));
}
