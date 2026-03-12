//! Instance Identity module for P2P authentication.
//!
//! Each Project Orchestrator instance has a unique cryptographic identity based on:
//! - **Ed25519 keypair**: for signing and verification
//! - **did:key identifier**: decentralized identifier derived from the public key
//! - **PASETO v4.public tokens**: for authenticating P2P messages
//!
//! The identity is persisted to `~/.openclaw/identity.key` and loaded on startup.

pub mod did;
pub mod rotation;
pub mod token;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use self::rotation::KeyHistory;

/// Default identity file location relative to home.
const IDENTITY_DIR: &str = ".openclaw";
const IDENTITY_FILE: &str = "identity.key";

/// Core identity of a Project Orchestrator instance.
///
/// Holds the Ed25519 signing key (private) and the derived verifying key (public).
/// The `did_key` is computed from the verifying key using the did:key method.
#[derive(Debug)]
pub struct InstanceIdentity {
    /// The Ed25519 signing key (private — never exposed in logs or errors).
    signing_key: SigningKey,
    /// The Ed25519 verifying key (public).
    verifying_key: VerifyingKey,
    /// The did:key identifier derived from the public key.
    did_key: String,
    /// When this identity was created.
    created_at: DateTime<Utc>,
    /// Key rotation history (previous keys for verifying old signatures).
    key_history: KeyHistory,
    /// Path where this identity is persisted.
    storage_path: PathBuf,
}

/// Serializable representation of the identity for disk storage.
#[derive(Serialize, Deserialize)]
struct StoredIdentity {
    /// Ed25519 signing key bytes (32 bytes, hex-encoded).
    signing_key_hex: String,
    /// Creation timestamp.
    created_at: DateTime<Utc>,
    /// Key rotation history.
    #[serde(default)]
    key_history: KeyHistory,
}

impl InstanceIdentity {
    /// Generate a new random identity.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let verifying_key = signing_key.verifying_key();
        let did_key = did::to_did_key(&verifying_key);

        info!(%did_key, "Generated new instance identity");

        Self {
            signing_key,
            verifying_key,
            did_key,
            created_at: Utc::now(),
            key_history: KeyHistory::default(),
            storage_path: default_storage_path(),
        }
    }

    /// Load identity from file, or generate a new one if not found.
    pub fn load_or_generate(path: Option<&Path>) -> Result<Self> {
        let storage_path = path
            .map(PathBuf::from)
            .unwrap_or_else(default_storage_path);

        if storage_path.exists() {
            Self::load_from_file(&storage_path)
        } else {
            let mut identity = Self::generate();
            identity.storage_path = storage_path;
            identity.save_to_file()?;
            Ok(identity)
        }
    }

    /// Load identity from a file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let data = fs::read_to_string(path)
            .with_context(|| format!("Failed to read identity file: {}", path.display()))?;

        let stored: StoredIdentity = serde_json::from_str(&data)
            .with_context(|| "Failed to parse identity file")?;

        let key_bytes = hex::decode(&stored.signing_key_hex)
            .with_context(|| "Invalid hex in signing key")?;

        if key_bytes.len() != 32 {
            anyhow::bail!("Invalid signing key length: expected 32 bytes, got {}", key_bytes.len());
        }

        let key_array: [u8; 32] = key_bytes.try_into().unwrap();
        let signing_key = SigningKey::from_bytes(&key_array);
        let verifying_key = signing_key.verifying_key();
        let did_key = did::to_did_key(&verifying_key);

        info!(%did_key, path = %path.display(), "Loaded instance identity");

        Ok(Self {
            signing_key,
            verifying_key,
            did_key,
            created_at: stored.created_at,
            key_history: stored.key_history,
            storage_path: path.to_path_buf(),
        })
    }

    /// Save identity to file with secure permissions (0o600).
    pub fn save_to_file(&self) -> Result<()> {
        let stored = StoredIdentity {
            signing_key_hex: hex::encode(self.signing_key.to_bytes()),
            created_at: self.created_at,
            key_history: self.key_history.clone(),
        };

        let json = serde_json::to_string_pretty(&stored)?;

        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write atomically: write to temp file, then rename
        let tmp_path = self.storage_path.with_extension("key.tmp");
        fs::write(&tmp_path, &json)?;

        // Set secure permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))?;
        }

        fs::rename(&tmp_path, &self.storage_path)?;

        info!(path = %self.storage_path.display(), "Saved instance identity");
        Ok(())
    }

    /// Get the did:key identifier.
    pub fn did_key(&self) -> &str {
        &self.did_key
    }

    /// Get the verifying (public) key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Get the creation timestamp.
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get the key history (for verifying old signatures).
    pub fn key_history(&self) -> &KeyHistory {
        &self.key_history
    }

    /// Sign a message with the instance's private key.
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Verify a signature against this instance's current public key.
    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        self.verifying_key.verify(message, signature).is_ok()
    }

    /// Verify a signature against the current key or any key in history.
    ///
    /// This is used for verifying signatures that may have been created
    /// before a key rotation.
    pub fn verify_with_history(&self, message: &[u8], signature: &Signature) -> bool {
        // Try current key first
        if self.verify(message, signature) {
            return true;
        }

        // Try historical keys
        self.key_history.verify_with_any(message, signature)
    }

    /// Rotate the identity key. Archives the current key and generates a new one.
    ///
    /// After rotation:
    /// - A new keypair is active
    /// - The old public key is archived in `key_history`
    /// - Old signatures can still be verified via `verify_with_history()`
    pub fn rotate(&mut self) -> Result<()> {
        let old_verifying_key = self.verifying_key;
        let old_did_key = self.did_key.clone();

        // Archive the current key
        self.key_history.archive(old_verifying_key);

        // Generate new keypair
        self.signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        self.verifying_key = self.signing_key.verifying_key();
        self.did_key = did::to_did_key(&self.verifying_key);

        // Persist atomically
        self.save_to_file()?;

        info!(
            old_did_key = %old_did_key,
            new_did_key = %self.did_key,
            archived_keys = self.key_history.len(),
            "Rotated instance identity"
        );

        Ok(())
    }
}

/// Get the default storage path for identity files.
fn default_storage_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(IDENTITY_DIR)
        .join(IDENTITY_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_identity() {
        let identity = InstanceIdentity::generate();
        assert!(identity.did_key().starts_with("did:key:z"));
        assert!(identity.created_at() <= Utc::now());
    }

    #[test]
    fn test_sign_verify() {
        let identity = InstanceIdentity::generate();
        let message = b"Hello, P2P world!";
        let signature = identity.sign(message);

        assert!(identity.verify(message, &signature));
        assert!(!identity.verify(b"tampered message", &signature));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tmp_dir = TempDir::new().unwrap();
        let key_path = tmp_dir.path().join("identity.key");

        let identity = {
            let mut id = InstanceIdentity::generate();
            id.storage_path = key_path.clone();
            id.save_to_file().unwrap();
            id
        };

        let loaded = InstanceIdentity::load_from_file(&key_path).unwrap();

        assert_eq!(identity.did_key(), loaded.did_key());
        assert_eq!(identity.verifying_key(), loaded.verifying_key());
    }

    #[test]
    fn test_load_or_generate_creates_new() {
        let tmp_dir = TempDir::new().unwrap();
        let key_path = tmp_dir.path().join("new_identity.key");

        assert!(!key_path.exists());
        let identity = InstanceIdentity::load_or_generate(Some(&key_path)).unwrap();
        assert!(key_path.exists());
        assert!(identity.did_key().starts_with("did:key:z"));
    }

    #[test]
    fn test_load_or_generate_loads_existing() {
        let tmp_dir = TempDir::new().unwrap();
        let key_path = tmp_dir.path().join("existing.key");

        let original = {
            let mut id = InstanceIdentity::generate();
            id.storage_path = key_path.clone();
            id.save_to_file().unwrap();
            id.did_key().to_string()
        };

        let loaded = InstanceIdentity::load_or_generate(Some(&key_path)).unwrap();
        assert_eq!(original, loaded.did_key());
    }

    #[cfg(unix)]
    #[test]
    fn test_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let tmp_dir = TempDir::new().unwrap();
        let key_path = tmp_dir.path().join("secure.key");

        let mut identity = InstanceIdentity::generate();
        identity.storage_path = key_path.clone();
        identity.save_to_file().unwrap();

        let perms = fs::metadata(&key_path).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o600, "Key file should be 0o600");
    }
}
