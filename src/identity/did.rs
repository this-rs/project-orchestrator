//! DID:key formatting and parsing for Ed25519 public keys.
//!
//! Implements the did:key method using multicodec prefix 0xed01 (Ed25519 public key)
//! and multibase base58btc encoding, producing identifiers like:
//! `did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK`

use anyhow::{anyhow, Result};
use ed25519_dalek::VerifyingKey;
use multibase::Base;

/// Multicodec prefix for Ed25519 public key (varint-encoded 0xed = 0xed01).
const ED25519_MULTICODEC_PREFIX: [u8; 2] = [0xed, 0x01];

/// DID:key method prefix.
const DID_KEY_PREFIX: &str = "did:key:";

/// Format an Ed25519 public key as a did:key identifier.
///
/// The format is: `did:key:z<base58btc(multicodec_prefix + public_key_bytes)>`
pub fn to_did_key(verifying_key: &VerifyingKey) -> String {
    let pk_bytes = verifying_key.to_bytes();

    // Prepend multicodec prefix (0xed01) to the 32-byte public key
    let mut prefixed = Vec::with_capacity(2 + pk_bytes.len());
    prefixed.extend_from_slice(&ED25519_MULTICODEC_PREFIX);
    prefixed.extend_from_slice(&pk_bytes);

    // Encode with multibase base58btc (prefix 'z')
    let encoded = multibase::encode(Base::Base58Btc, &prefixed);

    format!("{DID_KEY_PREFIX}{encoded}")
}

/// Parse a did:key identifier back to an Ed25519 public key.
///
/// Validates the multicodec prefix and extracts the 32-byte key.
pub fn from_did_key(did: &str) -> Result<VerifyingKey> {
    let encoded = did
        .strip_prefix(DID_KEY_PREFIX)
        .ok_or_else(|| anyhow!("Invalid did:key format: missing 'did:key:' prefix"))?;

    let (_base, decoded) = multibase::decode(encoded)
        .map_err(|e| anyhow!("Invalid multibase encoding in did:key: {e}"))?;

    if decoded.len() < 2 {
        return Err(anyhow!("did:key payload too short"));
    }

    if decoded[0..2] != ED25519_MULTICODEC_PREFIX {
        return Err(anyhow!(
            "Invalid multicodec prefix: expected 0xed01 (Ed25519), got 0x{:02x}{:02x}",
            decoded[0],
            decoded[1]
        ));
    }

    let key_bytes = &decoded[2..];
    if key_bytes.len() != 32 {
        return Err(anyhow!(
            "Invalid Ed25519 public key length: expected 32 bytes, got {}",
            key_bytes.len()
        ));
    }

    let key_array: [u8; 32] = key_bytes.try_into().unwrap();
    VerifyingKey::from_bytes(&key_array).map_err(|e| anyhow!("Invalid Ed25519 public key: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn test_did_key_roundtrip() {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let verifying_key = signing_key.verifying_key();

        let did = to_did_key(&verifying_key);

        // Must start with did:key:z6Mk (z = base58btc, 6Mk = Ed25519 multicodec)
        assert!(
            did.starts_with("did:key:z"),
            "did:key should start with 'did:key:z'"
        );

        // Round-trip
        let recovered = from_did_key(&did).expect("should parse valid did:key");
        assert_eq!(verifying_key, recovered, "round-trip should preserve key");
    }

    #[test]
    fn test_did_key_format() {
        let signing_key = SigningKey::generate(&mut &mut rand_core_06::OsRng);
        let did = to_did_key(&signing_key.verifying_key());

        // Pattern: did:key:z followed by base58btc characters
        assert!(did.starts_with("did:key:z"));
        assert!(did.len() > 20, "did:key should be at least 20 chars");

        // Only valid base58 characters after z
        let after_z = &did["did:key:z".len()..];
        assert!(
            after_z.chars().all(|c| c.is_ascii_alphanumeric()),
            "base58btc should only contain alphanumeric chars"
        );
    }

    #[test]
    fn test_invalid_did_key() {
        assert!(from_did_key("not-a-did").is_err());
        assert!(from_did_key("did:key:").is_err());
        assert!(from_did_key("did:web:example.com").is_err());
    }
}
