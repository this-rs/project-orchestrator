//! P2P HTTP Exchange — Real network integration test.
//!
//! Spins up two axum servers (Alice & Bob) on random ports and tests:
//! 1. Identity discovery via GET /api/p2p/identity
//! 2. Message sending via POST /api/p2p/message
//! 3. Delta sync via POST /api/p2p/sync (HAVE → WANT → DATA)
//! 4. Full skill exchange: Alice distills → signs → syncs → Bob receives & verifies

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use chrono::Utc;
use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;

use project_orchestrator::episodes::distill_models::*;
use project_orchestrator::identity::did::to_did_key;
use project_orchestrator::identity::InstanceIdentity;
use project_orchestrator::reception::anchor::TombstoneRegistry;
use project_orchestrator::reception::score::LocalContext;
use project_orchestrator::reception::verify::verify_envelope;
use project_orchestrator::reception::{receive_envelope, receive_envelope_checked};
use project_orchestrator::transport::http::{
    HttpTransport, HttpTransportConfig, IdentityResponse, SyncRequest, SyncResponse,
};
use project_orchestrator::transport::http_handlers::{
    get_identity, handle_sync, receive_message, sync_with_peer, P2pState,
};
use project_orchestrator::transport::sync::{SyncMessage, VectorClock};
use project_orchestrator::transport::types::{Message, MessageType, PeerInfo};
use project_orchestrator::transport::TransportLayer;

// ============================================================================
// Helper: spin up a P2P node (axum server) on a random port
// ============================================================================

struct TestNode {
    #[allow(dead_code)]
    addr: SocketAddr,
    base_url: String,
    did_key: String,
    transport: Arc<HttpTransport>,
    sync_store: Arc<tokio::sync::RwLock<HashMap<String, serde_json::Value>>>,
    local_clock: Arc<tokio::sync::RwLock<VectorClock>>,
}

async fn spawn_node(name: &str) -> TestNode {
    let identity = InstanceIdentity::generate();
    let did = identity.did_key().to_string();

    // Bind to port 0 → OS assigns a random free port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://127.0.0.1:{}", addr.port());

    let config = HttpTransportConfig {
        local_base_url: base_url.clone(),
        local_did: did.clone(),
        request_timeout: Duration::from_secs(5),
        seed_peers: vec![],
    };

    let transport = Arc::new(HttpTransport::new(config).unwrap());
    let sync_store = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
    let local_clock = Arc::new(tokio::sync::RwLock::new(VectorClock::new()));

    let state = Arc::new(P2pState {
        transport: transport.clone(),
        local_peer_info: PeerInfo {
            did_key: did.clone(),
            addresses: vec![base_url.clone()],
            capabilities: vec!["sync".to_string(), "distill".to_string()],
            last_seen: Utc::now(),
            trust_score: 1.0,
        },
        sync_store: sync_store.clone(),
        local_clock: local_clock.clone(),
    });

    let app = Router::new()
        .route("/api/p2p/identity", get(get_identity))
        .route("/api/p2p/message", post(receive_message))
        .route("/api/p2p/sync", post(handle_sync))
        .with_state(state);

    let node_name = name.to_string();
    tokio::spawn(async move {
        tracing::info!(name = %node_name, addr = %addr, "P2P node started");
        axum::serve(listener, app).await.unwrap();
    });

    // Small delay to ensure server is up
    tokio::time::sleep(Duration::from_millis(50)).await;

    TestNode {
        addr,
        base_url,
        did_key: did.to_string(),
        transport,
        sync_store,
        local_clock,
    }
}

/// Build a properly signed envelope for testing.
fn build_test_envelope(
    signing_key: &SigningKey,
    pattern: &str,
    tags: Vec<String>,
) -> DistillationEnvelope {
    let did = to_did_key(&signing_key.verifying_key());
    let lesson = DistilledLesson {
        abstract_pattern: pattern.to_string(),
        domain_tags: tags,
        portability_layer: PortabilityLayer::Domain,
        confidence: 0.85,
    };

    let lesson_json = serde_json::to_string(&lesson).unwrap();
    let content_hash = hex::encode(Sha256::digest(lesson_json.as_bytes()));
    let signature = signing_key.sign(content_hash.as_bytes());

    DistillationEnvelope {
        lesson,
        anonymized_content: pattern.to_string(),
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

fn sha2_hash(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

// ============================================================================
// Test 1: Two nodes discover each other via HTTP identity endpoint
// ============================================================================

#[tokio::test]
async fn test_peer_discovery_via_http() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // Alice discovers Bob via HTTP
    let client = reqwest::Client::new();
    let resp: IdentityResponse = client
        .get(format!("{}/api/p2p/identity", bob.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert!(resp.peer_info.did_key.starts_with("did:key:z"));
    assert_eq!(resp.peer_info.did_key, bob.did_key);
    assert!(resp.peer_info.capabilities.contains(&"sync".to_string()));

    // Bob discovers Alice
    let resp: IdentityResponse = client
        .get(format!("{}/api/p2p/identity", alice.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(resp.peer_info.did_key, alice.did_key);
}

// ============================================================================
// Test 2: Alice sends a message to Bob via HTTP
// ============================================================================

#[tokio::test]
async fn test_message_sending_via_http() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // Bob subscribes to all messages
    let mut rx = bob.transport.subscribe("handshake").await.unwrap();

    // Alice sends a handshake message to Bob via HTTP POST
    let msg = Message::new(
        MessageType::Handshake,
        alice.did_key.clone(),
        b"hello bob".to_vec(),
    );

    let client = reqwest::Client::new();
    let status = client
        .post(format!("{}/api/p2p/message", bob.base_url))
        .json(&msg)
        .send()
        .await
        .unwrap()
        .status();

    assert!(
        status.is_success(),
        "POST /api/p2p/message should return 200"
    );

    // Give subscriber a moment to process
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Bob should have received the message via his subscriber
    let received = rx.try_recv().expect("Bob should have received the message");
    assert_eq!(received.header.sender_did, alice.did_key);
    assert_eq!(received.header.message_type, MessageType::Handshake);
    assert_eq!(received.payload, b"hello bob");
}

// ============================================================================
// Test 3: Delta sync — Alice has data, Bob syncs it via HAVE/WANT/DATA
// ============================================================================

#[tokio::test]
async fn test_delta_sync_have_want_data() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // Alice has 3 envelopes in her store
    let envelopes: Vec<serde_json::Value> = vec![
        serde_json::json!({"pattern": "Use UNWIND for batch inserts", "tags": ["neo4j"]}),
        serde_json::json!({"pattern": "Index foreign keys", "tags": ["database"]}),
        serde_json::json!({"pattern": "Connection pooling", "tags": ["performance"]}),
    ];

    let mut alice_hashes = std::collections::HashSet::new();
    {
        let mut store = alice.sync_store.write().await;
        for env in &envelopes {
            let hash = sha2_hash(&serde_json::to_vec(env).unwrap());
            store.insert(hash.clone(), env.clone());
            alice_hashes.insert(hash);
        }
    }

    let client = reqwest::Client::new();

    // Step 1: Alice sends her HAVE to Bob → Bob responds with WANT
    let alice_have = SyncRequest {
        sender_did: alice.did_key.clone(),
        sync_message: SyncMessage::Have {
            vector_clock: VectorClock::new(),
            content_hashes: alice_hashes.clone(),
        },
    };

    let resp: SyncResponse = client
        .post(format!("{}/api/p2p/sync", bob.base_url))
        .json(&alice_have)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Bob should respond with WANT (all 3, since he has none)
    let wanted = match &resp.sync_message {
        SyncMessage::Want { content_hashes } => content_hashes.clone(),
        other => panic!("Expected WANT, got {:?}", other),
    };
    assert_eq!(wanted.len(), 3, "Bob should want all 3 envelopes");

    // Step 2: Alice sends DATA to Bob
    let data_to_send: Vec<serde_json::Value> = {
        let store = alice.sync_store.read().await;
        wanted
            .iter()
            .filter_map(|hash| store.get(hash).cloned())
            .collect()
    };
    assert_eq!(data_to_send.len(), 3);

    let alice_data = SyncRequest {
        sender_did: alice.did_key.clone(),
        sync_message: SyncMessage::Data {
            envelopes: data_to_send,
        },
    };

    let resp: SyncResponse = client
        .post(format!("{}/api/p2p/sync", bob.base_url))
        .json(&alice_data)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Bob should respond with HAVE (confirming he now has them all)
    match &resp.sync_message {
        SyncMessage::Have { content_hashes, .. } => {
            assert_eq!(
                content_hashes.len(),
                3,
                "Bob should now have all 3 envelopes"
            );
        }
        other => panic!("Expected HAVE ack, got {:?}", other),
    }

    // Verify Bob's store actually has the data
    let bob_store = bob.sync_store.read().await;
    assert_eq!(bob_store.len(), 3);
}

// ============================================================================
// Test 4: Full skill exchange — distill → sign → sync → verify → score
// ============================================================================

#[tokio::test]
async fn test_full_skill_exchange_over_http() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // --- Alice side: create and sign an envelope ---
    let alice_signing_key = SigningKey::generate(&mut rand_core_06::OsRng);
    let envelope = build_test_envelope(
        &alice_signing_key,
        "Use UNWIND for batch Neo4j inserts to avoid N+1 patterns",
        vec![
            "rust".to_string(),
            "neo4j".to_string(),
            "performance".to_string(),
        ],
    );

    // Alice stores the envelope
    let envelope_json = serde_json::to_value(&envelope).unwrap();
    let envelope_hash = sha2_hash(&serde_json::to_vec(&envelope_json).unwrap());
    {
        let mut store = alice.sync_store.write().await;
        store.insert(envelope_hash.clone(), envelope_json.clone());
    }

    // --- Sync: Alice pushes to Bob via HTTP ---
    let client = reqwest::Client::new();

    // Step 1: Alice sends HAVE
    let alice_have = SyncRequest {
        sender_did: alice.did_key.clone(),
        sync_message: SyncMessage::Have {
            vector_clock: VectorClock::new(),
            content_hashes: std::collections::HashSet::from([envelope_hash.clone()]),
        },
    };

    let resp: SyncResponse = client
        .post(format!("{}/api/p2p/sync", bob.base_url))
        .json(&alice_have)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let wanted = match &resp.sync_message {
        SyncMessage::Want { content_hashes } => content_hashes.clone(),
        other => panic!("Expected WANT, got {:?}", other),
    };
    assert_eq!(wanted.len(), 1, "Bob should want the 1 envelope");

    // Step 2: Alice sends DATA
    let alice_data = SyncRequest {
        sender_did: alice.did_key.clone(),
        sync_message: SyncMessage::Data {
            envelopes: vec![envelope_json.clone()],
        },
    };

    let _resp: SyncResponse = client
        .post(format!("{}/api/p2p/sync", bob.base_url))
        .json(&alice_data)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // --- Bob side: extract and verify the received envelope ---
    let bob_store = bob.sync_store.read().await;
    assert_eq!(bob_store.len(), 1, "Bob should have 1 envelope");

    // Deserialize the envelope back
    let received_json = bob_store.values().next().unwrap();
    let received_envelope: DistillationEnvelope =
        serde_json::from_value(received_json.clone()).unwrap();

    // Verify cryptographic signature
    let verified = verify_envelope(&received_envelope)
        .expect("Bob should successfully verify Alice's envelope");
    assert!(verified.source_did.starts_with("did:key:z"));

    // Score relevance — Bob is a Rust+Neo4j project
    let bob_context = LocalContext {
        local_tags: vec![
            "rust".to_string(),
            "neo4j".to_string(),
            "graphql".to_string(),
        ],
        local_languages: vec!["rust".to_string()],
        known_peers: HashMap::new(),
    };

    let result = receive_envelope(&received_envelope, &bob_context)
        .expect("Full reception pipeline should succeed");

    assert!(
        result.score.accepted,
        "Bob should accept Alice's envelope (matching rust+neo4j tags), score={}",
        result.score.total
    );
    assert!(!result.notes.is_empty(), "Should produce at least 1 note");
    assert!(
        result.notes[0].content.contains("UNWIND"),
        "Note should contain the pattern content"
    );
}

// ============================================================================
// Test 5: Sync with peer helper function (3-step protocol)
// ============================================================================

#[tokio::test]
async fn test_sync_with_peer_helper() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // Alice has 2 envelopes
    {
        let mut store = alice.sync_store.write().await;
        store.insert("hash_a".to_string(), serde_json::json!({"pattern": "A"}));
        store.insert("hash_b".to_string(), serde_json::json!({"pattern": "B"}));
    }

    // Register Bob as a peer in Alice's transport
    alice
        .transport
        .registry()
        .add(PeerInfo {
            did_key: bob.did_key.clone(),
            addresses: vec![bob.base_url.clone()],
            capabilities: vec!["sync".to_string()],
            last_seen: Utc::now(),
            trust_score: 0.8,
        })
        .await;

    // Alice syncs with Bob using the helper function
    let report = sync_with_peer(
        &alice.transport,
        &bob.did_key,
        &alice.sync_store,
        &alice.local_clock,
    )
    .await
    .expect("sync_with_peer should succeed");

    assert_eq!(report.peer_did, bob.did_key);
    assert_eq!(report.items_requested, 2, "Bob should request 2 items");
    assert_eq!(report.items_sent, 2, "Alice should send 2 items");

    // Verify Bob received them
    let bob_store = bob.sync_store.read().await;
    assert_eq!(bob_store.len(), 2, "Bob should have 2 envelopes after sync");
}

// ============================================================================
// Test 6: Tombstone propagation — Alice retracts, Bob honors it
// ============================================================================

#[tokio::test]
async fn test_tombstone_propagation_over_http() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // Alice creates and signs an envelope
    let alice_key = SigningKey::generate(&mut rand_core_06::OsRng);
    let envelope = build_test_envelope(
        &alice_key,
        "Deprecated pattern — do not use",
        vec!["rust".to_string()],
    );

    // Sync the envelope to Bob (via direct HTTP calls)
    let envelope_json = serde_json::to_value(&envelope).unwrap();
    let client = reqwest::Client::new();

    // Send DATA directly to Bob
    let data_req = SyncRequest {
        sender_did: alice.did_key.clone(),
        sync_message: SyncMessage::Data {
            envelopes: vec![envelope_json],
        },
    };
    client
        .post(format!("{}/api/p2p/sync", bob.base_url))
        .json(&data_req)
        .send()
        .await
        .unwrap();

    // Verify Bob has it
    assert_eq!(bob.sync_store.read().await.len(), 1);

    // Bob successfully processes the envelope
    let bob_context = LocalContext {
        local_tags: vec!["rust".to_string()],
        local_languages: vec!["rust".to_string()],
        known_peers: HashMap::new(),
    };
    let result = receive_envelope(&envelope, &bob_context);
    assert!(result.is_ok(), "First reception should succeed");

    // Now Alice sends a tombstone for that content_hash
    let tombstone_msg = Message::new(
        MessageType::Tombstone,
        alice.did_key.clone(),
        envelope.meta.content_hash.as_bytes().to_vec(),
    );

    let status = client
        .post(format!("{}/api/p2p/message", bob.base_url))
        .json(&tombstone_msg)
        .send()
        .await
        .unwrap()
        .status();
    assert!(status.is_success(), "Tombstone message should be accepted");

    // Bob's tombstone registry should now block this envelope
    let mut tombstones = TombstoneRegistry::new();
    tombstones.apply_tombstone(&envelope.meta.content_hash);

    let result = receive_envelope_checked(&envelope, &bob_context, &tombstones);
    assert!(result.is_err(), "Tombstoned envelope should be rejected");
    assert!(
        result.unwrap_err().to_string().contains("tombstoned"),
        "Error should mention tombstone"
    );
}

// ============================================================================
// Test 7: Bidirectional sync — both nodes have unique data
// ============================================================================

#[tokio::test]
async fn test_bidirectional_sync() {
    let alice = spawn_node("alice").await;
    let bob = spawn_node("bob").await;

    // Alice has envelope A (use real SHA-256 hashes so handler dedup works correctly)
    let env_a = serde_json::json!({"from": "alice", "id": "a"});
    let hash_a = sha2_hash(&serde_json::to_vec(&env_a).unwrap());
    {
        let mut store = alice.sync_store.write().await;
        store.insert(hash_a.clone(), env_a);
    }

    // Bob has envelope B
    let env_b = serde_json::json!({"from": "bob", "id": "b"});
    let hash_b = sha2_hash(&serde_json::to_vec(&env_b).unwrap());
    {
        let mut store = bob.sync_store.write().await;
        store.insert(hash_b.clone(), env_b);
    }

    // Register peers
    alice
        .transport
        .registry()
        .add(PeerInfo {
            did_key: bob.did_key.clone(),
            addresses: vec![bob.base_url.clone()],
            capabilities: vec!["sync".to_string()],
            last_seen: Utc::now(),
            trust_score: 0.8,
        })
        .await;

    bob.transport
        .registry()
        .add(PeerInfo {
            did_key: alice.did_key.clone(),
            addresses: vec![alice.base_url.clone()],
            capabilities: vec!["sync".to_string()],
            last_seen: Utc::now(),
            trust_score: 0.8,
        })
        .await;

    // Alice pushes to Bob
    let report_a = sync_with_peer(
        &alice.transport,
        &bob.did_key,
        &alice.sync_store,
        &alice.local_clock,
    )
    .await
    .unwrap();
    assert_eq!(report_a.items_sent, 1, "Alice pushes 1 to Bob");

    // Bob pushes to Alice (Bob now has 2 items: his original + Alice's)
    let _report_b = sync_with_peer(
        &bob.transport,
        &alice.did_key,
        &bob.sync_store,
        &bob.local_clock,
    )
    .await
    .unwrap();

    // After bidirectional sync, both nodes should have both envelopes
    let alice_count = alice.sync_store.read().await.len();
    let bob_count = bob.sync_store.read().await.len();
    assert_eq!(
        alice_count, 2,
        "Alice should have 2 envelopes after bidi sync"
    );
    assert_eq!(bob_count, 2, "Bob should have 2 envelopes after bidi sync");

    // Verify both have both keys
    let alice_store = alice.sync_store.read().await;
    let bob_store = bob.sync_store.read().await;
    assert!(
        alice_store.contains_key(&hash_a),
        "Alice should still have hash_a"
    );
    assert!(
        alice_store.contains_key(&hash_b),
        "Alice should now have hash_b"
    );
    assert!(
        bob_store.contains_key(&hash_a),
        "Bob should now have hash_a"
    );
    assert!(
        bob_store.contains_key(&hash_b),
        "Bob should still have hash_b"
    );
}
