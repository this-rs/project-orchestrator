//! Key rotation and history management.
//!
//! When an identity key is rotated, the old public key is archived so that
//! signatures created before the rotation can still be verified.

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// History of previous (rotated-out) public keys.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyHistory {
    /// Archived keys, most recent first.
    pub archived_keys: Vec<ArchivedKey>,
}

/// A previously active public key that has been rotated out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedKey {
    /// The public key bytes (hex-encoded for JSON serialization).
    pub verifying_key_hex: String,
    /// When this key was retired (rotated out).
    pub retired_at: DateTime<Utc>,
    /// Sequential key ID (0 = first key, 1 = after first rotation, etc.)
    pub key_id: u32,
}

impl KeyHistory {
    /// Archive a key that is being rotated out.
    pub fn archive(&mut self, verifying_key: VerifyingKey) {
        let key_id = self.archived_keys.len() as u32;
        self.archived_keys.push(ArchivedKey {
            verifying_key_hex: hex::encode(verifying_key.to_bytes()),
            retired_at: Utc::now(),
            key_id,
        });
    }

    /// Get the number of archived keys.
    pub fn len(&self) -> usize {
        self.archived_keys.len()
    }

    /// Check if there are no archived keys.
    pub fn is_empty(&self) -> bool {
        self.archived_keys.is_empty()
    }

    /// Try to verify a signature against any archived key.
    ///
    /// Returns true if any archived key verifies the signature.
    pub fn verify_with_any(&self, message: &[u8], signature: &Signature) -> bool {
        for archived in &self.archived_keys {
            if let Ok(key_bytes) = hex::decode(&archived.verifying_key_hex) {
                if key_bytes.len() == 32 {
                    let key_array: [u8; 32] = key_bytes.try_into().unwrap();
                    if let Ok(key) = VerifyingKey::from_bytes(&key_array) {
                        if key.verify(message, signature).is_ok() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Get all archived verifying keys (for peer key exchange).
    pub fn all_verifying_keys(&self) -> Vec<VerifyingKey> {
        self.archived_keys
            .iter()
            .filter_map(|ak| {
                let bytes = hex::decode(&ak.verifying_key_hex).ok()?;
                let array: [u8; 32] = bytes.try_into().ok()?;
                VerifyingKey::from_bytes(&array).ok()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn test_archive_and_verify() {
        let mut history = KeyHistory::default();
        assert!(history.is_empty());

        // Generate key 1, sign a message, then rotate
        let key1 = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let message = b"signed before rotation";
        let sig1 = key1.sign(message);

        // Archive key 1
        history.archive(key1.verifying_key());
        assert_eq!(history.len(), 1);

        // Verify with history
        assert!(history.verify_with_any(message, &sig1));

        // Tampered message fails
        assert!(!history.verify_with_any(b"tampered", &sig1));
    }

    #[test]
    fn test_multiple_rotations() {
        let mut history = KeyHistory::default();

        let key1 = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let key2 = SigningKey::generate(&mut &mut rand_core_06::OsRng);

        let msg1 = b"message from key 1";
        let msg2 = b"message from key 2";
        let sig1 = key1.sign(msg1);
        let sig2 = key2.sign(msg2);

        history.archive(key1.verifying_key());
        history.archive(key2.verifying_key());

        assert_eq!(history.len(), 2);
        assert!(history.verify_with_any(msg1, &sig1));
        assert!(history.verify_with_any(msg2, &sig2));
    }

    #[test]
    fn test_key_ids_sequential() {
        let mut history = KeyHistory::default();

        for i in 0..3 {
            let key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
            history.archive(key.verifying_key());
            assert_eq!(history.archived_keys.last().unwrap().key_id, i);
        }
    }

    #[test]
    fn test_all_verifying_keys() {
        let mut history = KeyHistory::default();
        let key1 = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let key2 = SigningKey::generate(&mut &mut rand_core_06::OsRng);

        history.archive(key1.verifying_key());
        history.archive(key2.verifying_key());

        let keys = history.all_verifying_keys();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], key1.verifying_key());
        assert_eq!(keys[1], key2.verifying_key());
    }
}
