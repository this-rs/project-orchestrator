//! Cryptographic primitives of the vault.
//!
//! Two keys, two jobs:
//!
//! - The **KEK** (key-encryption key) is derived from the user's passphrase with
//!   Argon2id. It exists only for the instant it takes to unwrap the master key,
//!   and is wiped right after.
//! - The **master key** is 32 random bytes, generated once at initialisation. It
//!   encrypts every secret. It is stored only *wrapped* by the KEK, and held in
//!   memory only while the vault is unlocked.
//!
//! The indirection is what makes a passphrase change cheap (re-wrap one key, not
//! every secret) and what keeps the passphrase from ever touching a secret
//! directly.
//!
//! Every seal uses XChaCha20-Poly1305: authenticated, so a single flipped bit is
//! detected; with a 24-byte nonce, so random nonces are safe without a counter to
//! persist. Each secret's NAME is bound as associated data — a ciphertext copied
//! under another name fails to open, so an attacker who can edit the vault file
//! cannot relabel one secret as another.

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::rand_core::RngCore;
use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

/// Length of every symmetric key in the vault.
pub const KEY_LEN: usize = 32;
/// Salt length for the passphrase KDF.
pub const SALT_LEN: usize = 16;

/// A 32-byte key that is wiped when dropped and never printed.
///
/// No `Clone` on purpose: every copy of a key is one more place it has to be
/// wiped from, and one more place a bug can leave it.
pub struct SecretKey(Zeroizing<[u8; KEY_LEN]>);

impl SecretKey {
    /// A fresh random key, from the operating system's CSPRNG.
    pub fn generate() -> Self {
        let mut bytes = Zeroizing::new([0u8; KEY_LEN]);
        OsRng.fill_bytes(bytes.as_mut());
        Self(bytes)
    }

    fn from_bytes(bytes: Zeroizing<[u8; KEY_LEN]>) -> Self {
        Self(bytes)
    }

    fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
}

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the bytes, not even a prefix or a hash: a log line is the most
        // likely way for a key to leave the process.
        f.write_str("SecretKey(<redacted>)")
    }
}

/// Cost parameters of Argon2id, stored with the vault so they can be raised
/// later without locking anyone out of an existing vault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KdfParams {
    /// Memory cost in KiB.
    pub m_cost_kib: u32,
    /// Number of passes.
    pub t_cost: u32,
    /// Degree of parallelism.
    pub p_cost: u32,
}

impl Default for KdfParams {
    /// 64 MiB, 3 passes, 1 lane — the OWASP-recommended floor for Argon2id at
    /// the time of writing. Unlocking is rare (once per work session), so a
    /// fraction of a second here costs nothing and multiplies an offline
    /// attacker's work on a stolen vault file.
    fn default() -> Self {
        Self {
            m_cost_kib: 64 * 1024,
            t_cost: 3,
            p_cost: 1,
        }
    }
}

impl KdfParams {
    /// Deliberately weak parameters, for tests only: the real ones take a
    /// noticeable fraction of a second per derivation.
    #[cfg(test)]
    pub fn insecure_for_tests() -> Self {
        Self {
            m_cost_kib: 8,
            t_cost: 1,
            p_cost: 1,
        }
    }
}

/// A random salt for a new vault.
pub fn generate_salt() -> [u8; SALT_LEN] {
    let mut salt = [0u8; SALT_LEN];
    OsRng.fill_bytes(&mut salt);
    salt
}

/// Derive the key-encryption key from the passphrase.
pub fn derive_kek(
    passphrase: &str,
    salt: &[u8],
    params: KdfParams,
) -> Result<SecretKey, CryptoError> {
    let argon_params = Params::new(
        params.m_cost_kib,
        params.t_cost,
        params.p_cost,
        Some(KEY_LEN),
    )
    .map_err(|_| CryptoError::InvalidKdfParams)?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon_params);

    let mut out = Zeroizing::new([0u8; KEY_LEN]);
    argon
        .hash_password_into(passphrase.as_bytes(), salt, out.as_mut())
        .map_err(|_| CryptoError::InvalidKdfParams)?;
    Ok(SecretKey::from_bytes(out))
}

/// A ciphertext and the nonce it was sealed with, hex-encoded for storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sealed {
    pub nonce: String,
    pub ciphertext: String,
}

/// Encrypt `plaintext` under `key`, binding `associated_data` to it.
pub fn seal(
    key: &SecretKey,
    plaintext: &[u8],
    associated_data: &[u8],
) -> Result<Sealed, CryptoError> {
    let cipher = XChaCha20Poly1305::new(key.as_bytes().into());
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: plaintext,
                aad: associated_data,
            },
        )
        .map_err(|_| CryptoError::Seal)?;
    Ok(Sealed {
        nonce: hex::encode(nonce),
        ciphertext: hex::encode(ciphertext),
    })
}

/// Decrypt and authenticate. Fails if the key is wrong, if a single byte of the
/// ciphertext or nonce was altered, or if `associated_data` differs from the one
/// it was sealed with.
pub fn open(
    key: &SecretKey,
    sealed: &Sealed,
    associated_data: &[u8],
) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
    let nonce_bytes = hex::decode(&sealed.nonce).map_err(|_| CryptoError::Corrupt)?;
    if nonce_bytes.len() != 24 {
        return Err(CryptoError::Corrupt);
    }
    let ciphertext = hex::decode(&sealed.ciphertext).map_err(|_| CryptoError::Corrupt)?;
    let cipher = XChaCha20Poly1305::new(key.as_bytes().into());
    let plaintext = cipher
        .decrypt(
            XNonce::from_slice(&nonce_bytes),
            Payload {
                msg: &ciphertext,
                aad: associated_data,
            },
        )
        .map_err(|_| CryptoError::Open)?;
    Ok(Zeroizing::new(plaintext))
}

/// Wrap a master key under a KEK. The associated data pins the purpose, so a
/// wrapped key cannot be passed off as a secret or vice versa.
pub fn wrap_key(kek: &SecretKey, master: &SecretKey) -> Result<Sealed, CryptoError> {
    seal(kek, master.as_bytes(), WRAPPED_KEY_AAD)
}

/// Unwrap a master key. A wrong passphrase surfaces here, as an authentication
/// failure — the only way it can surface, since nothing about the passphrase
/// is stored.
pub fn unwrap_key(kek: &SecretKey, wrapped: &Sealed) -> Result<SecretKey, CryptoError> {
    let bytes = open(kek, wrapped, WRAPPED_KEY_AAD)?;
    if bytes.len() != KEY_LEN {
        return Err(CryptoError::Corrupt);
    }
    let mut key = Zeroizing::new([0u8; KEY_LEN]);
    key.copy_from_slice(&bytes);
    Ok(SecretKey::from_bytes(key))
}

const WRAPPED_KEY_AAD: &[u8] = b"po-vault:v1:master-key";

/// Associated data binding a secret's ciphertext to its name and format version.
pub fn secret_aad(name: &str) -> Vec<u8> {
    format!("po-vault:v1:secret:{name}").into_bytes()
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CryptoError {
    #[error("invalid key-derivation parameters")]
    InvalidKdfParams,
    #[error("encryption failed")]
    Seal,
    /// Deliberately vague: whether the key was wrong or the data tampered with,
    /// telling them apart would only help an attacker.
    #[error("decryption failed: wrong passphrase, or the vault data was altered")]
    Open,
    #[error("vault data is malformed")]
    Corrupt,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_then_open_round_trips() {
        let key = SecretKey::generate();
        let sealed = seal(&key, b"hunter2-passphrase", b"aad").unwrap();
        assert_eq!(
            open(&key, &sealed, b"aad").unwrap().as_slice(),
            b"hunter2-passphrase"
        );
    }

    #[test]
    fn a_wrong_key_cannot_open() {
        let sealed = seal(&SecretKey::generate(), b"value", b"aad").unwrap();
        assert_eq!(
            open(&SecretKey::generate(), &sealed, b"aad"),
            Err(CryptoError::Open)
        );
    }

    #[test]
    fn one_flipped_byte_is_detected() {
        let key = SecretKey::generate();
        let mut sealed = seal(&key, b"value", b"aad").unwrap();
        let mut bytes = hex::decode(&sealed.ciphertext).unwrap();
        bytes[0] ^= 0x01;
        sealed.ciphertext = hex::encode(bytes);
        assert_eq!(open(&key, &sealed, b"aad"), Err(CryptoError::Open));
    }

    #[test]
    fn a_ciphertext_moved_under_another_name_does_not_open() {
        // The name is bound as associated data: editing the vault file to swap
        // two secrets' ciphertexts must not silently relabel them.
        let key = SecretKey::generate();
        let sealed = seal(&key, b"prod-db-password", &secret_aad("prod_db")).unwrap();
        assert_eq!(
            open(&key, &sealed, &secret_aad("staging_db")),
            Err(CryptoError::Open)
        );
    }

    #[test]
    fn the_same_value_seals_differently_each_time() {
        // Random nonces: two seals of one value must not reveal they are equal.
        let key = SecretKey::generate();
        let a = seal(&key, b"same", b"aad").unwrap();
        let b = seal(&key, b"same", b"aad").unwrap();
        assert_ne!(a.nonce, b.nonce);
        assert_ne!(a.ciphertext, b.ciphertext);
    }

    #[test]
    fn a_wrong_passphrase_fails_to_unwrap() {
        let salt = generate_salt();
        let params = KdfParams::insecure_for_tests();
        let master = SecretKey::generate();
        let wrapped = wrap_key(
            &derive_kek("correct horse", &salt, params).unwrap(),
            &master,
        )
        .unwrap();

        let wrong = derive_kek("battery staple", &salt, params).unwrap();
        assert!(unwrap_key(&wrong, &wrapped).is_err());

        let right = derive_kek("correct horse", &salt, params).unwrap();
        let unwrapped = unwrap_key(&right, &wrapped).unwrap();
        assert_eq!(unwrapped.as_bytes(), master.as_bytes());
    }

    #[test]
    fn a_key_never_prints_its_bytes() {
        let key = SecretKey::generate();
        let debug = format!("{key:?}");
        assert_eq!(debug, "SecretKey(<redacted>)");
        assert!(!debug.contains(&hex::encode(key.as_bytes())));
    }

    #[test]
    fn default_kdf_parameters_are_not_the_test_ones() {
        // Guard against the weak test parameters ever becoming the default.
        assert_ne!(KdfParams::default(), KdfParams::insecure_for_tests());
        assert!(KdfParams::default().m_cost_kib >= 64 * 1024);
    }
}
