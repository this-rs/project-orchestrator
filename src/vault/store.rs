//! The vault file and its locked / unlocked lifecycle.
//!
//! The file holds only ciphertext and metadata: the KDF salt and parameters, the
//! master key wrapped by the passphrase-derived key, and each secret sealed by
//! the master key. Without the passphrase it is inert. It lives outside the
//! knowledge graph on purpose — Neo4j content is indexed, embedded, exported and
//! shared between instances, and a vault must never be reachable by any of that.
//!
//! "Unlocked" means one thing: the master key is in memory. It carries a
//! deadline, and every operation checks it first — an expired vault locks itself
//! on the next access, wiping the key, rather than waiting for a timer that
//! might not fire.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use super::crypto::{self, CryptoError, KdfParams, Sealed, SecretKey};

/// Shortest value the vault accepts.
///
/// Every stored value is also masked wherever it shows up in agent output. A
/// very short value cannot be masked without false positives — "1234" appears
/// in timestamps, ports, line numbers — and a secret that short is guessable
/// anyway. Refusing it at the door is better than masking half the logs.
pub const MIN_SECRET_LEN: usize = 8;

/// Longest a vault may stay open in one go. Past a working day, "unlocked" stops
/// being a decision and becomes a default.
pub const MAX_UNLOCK: Duration = Duration::hours(12);

const FILE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KdfRecord {
    salt: String,
    params: KdfParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SecretRecord {
    sealed: Sealed,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    #[serde(default)]
    description: Option<String>,
}

/// On-disk layout. Contains no plaintext and no key in the clear.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VaultFile {
    version: u32,
    kdf: KdfRecord,
    wrapped_key: Sealed,
    #[serde(default)]
    secrets: BTreeMap<String, SecretRecord>,
}

/// What may be said about a secret without revealing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SecretMeta {
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VaultStatus {
    pub initialized: bool,
    /// `Some` while open; the instant it locks itself.
    pub unlocked_until: Option<DateTime<Utc>>,
    pub secret_count: usize,
}

struct Unlocked {
    key: SecretKey,
    until: DateTime<Utc>,
}

impl std::fmt::Debug for Unlocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Unlocked")
            .field("until", &self.until)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct Vault {
    path: PathBuf,
    file: Option<VaultFile>,
    unlocked: Option<Unlocked>,
}

impl Vault {
    /// Load the vault at `path`, or an uninitialised one if the file is absent.
    /// Always starts LOCKED: the server cannot open the vault on its own.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, VaultError> {
        let path = path.into();
        let file = match std::fs::read(&path) {
            Ok(bytes) => {
                let file: VaultFile =
                    serde_json::from_slice(&bytes).map_err(|_| VaultError::Corrupt)?;
                if file.version != FILE_VERSION {
                    return Err(VaultError::UnsupportedVersion(file.version));
                }
                Some(file)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(VaultError::Io(e.to_string())),
        };
        Ok(Self {
            path,
            file,
            unlocked: None,
        })
    }

    /// Default location, beside the instance identity key.
    pub fn default_path() -> PathBuf {
        let base = dirs::config_dir()
            .map(|d| d.join("project-orchestrator"))
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".project-orchestrator")
            });
        base.join("vault.json")
    }

    pub fn status(&mut self, now: DateTime<Utc>) -> VaultStatus {
        self.expire_if_due(now);
        VaultStatus {
            initialized: self.file.is_some(),
            unlocked_until: self.unlocked.as_ref().map(|u| u.until),
            secret_count: self.file.as_ref().map_or(0, |f| f.secrets.len()),
        }
    }

    /// Create the vault, protected by `passphrase`, and leave it open for
    /// `duration`. Refuses to overwrite an existing vault: that would destroy
    /// every secret in it.
    pub fn init(
        &mut self,
        passphrase: &str,
        params: KdfParams,
        duration: Duration,
        now: DateTime<Utc>,
    ) -> Result<(), VaultError> {
        if self.file.is_some() {
            return Err(VaultError::AlreadyInitialized);
        }
        validate_passphrase(passphrase)?;
        let salt = crypto::generate_salt();
        let kek = crypto::derive_kek(passphrase, &salt, params)?;
        let master = SecretKey::generate();
        let wrapped_key = crypto::wrap_key(&kek, &master)?;

        let file = VaultFile {
            version: FILE_VERSION,
            kdf: KdfRecord {
                salt: hex::encode(salt),
                params,
            },
            wrapped_key,
            secrets: BTreeMap::new(),
        };
        write_atomically(&self.path, &file)?;
        self.file = Some(file);
        self.unlocked = Some(Unlocked {
            key: master,
            until: now + clamp_duration(duration),
        });
        Ok(())
    }

    /// Open the vault for `duration` (clamped to [`MAX_UNLOCK`]).
    pub fn unlock(
        &mut self,
        passphrase: &str,
        duration: Duration,
        now: DateTime<Utc>,
    ) -> Result<DateTime<Utc>, VaultError> {
        let file = self.file.as_ref().ok_or(VaultError::NotInitialized)?;
        let salt = hex::decode(&file.kdf.salt).map_err(|_| VaultError::Corrupt)?;
        let kek = crypto::derive_kek(passphrase, &salt, file.kdf.params)?;
        let key = crypto::unwrap_key(&kek, &file.wrapped_key).map_err(|e| match e {
            CryptoError::Open => VaultError::WrongPassphrase,
            other => other.into(),
        })?;
        let until = now + clamp_duration(duration);
        self.unlocked = Some(Unlocked { key, until });
        Ok(until)
    }

    /// Close now. Dropping the key wipes it (Zeroizing).
    pub fn lock(&mut self) {
        self.unlocked = None;
    }

    pub fn list(&self) -> Vec<SecretMeta> {
        self.file
            .as_ref()
            .map(|f| {
                f.secrets
                    .iter()
                    .map(|(name, r)| SecretMeta {
                        name: name.clone(),
                        description: r.description.clone(),
                        created_at: r.created_at,
                        updated_at: r.updated_at,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.file
            .as_ref()
            .is_some_and(|f| f.secrets.contains_key(name))
    }

    /// Store or replace a secret. The vault must be open.
    pub fn put(
        &mut self,
        name: &str,
        value: &str,
        description: Option<String>,
        now: DateTime<Utc>,
    ) -> Result<(), VaultError> {
        validate_name(name)?;
        if value.chars().count() < MIN_SECRET_LEN {
            return Err(VaultError::ValueTooShort);
        }
        self.expire_if_due(now);
        let key = &self.unlocked.as_ref().ok_or(VaultError::Locked)?.key;
        let sealed = crypto::seal(key, value.as_bytes(), &crypto::secret_aad(name))?;

        // Work on a copy and swap only once the write succeeded: a failed write
        // must leave the in-memory vault identical to the file on disk.
        let mut file = self.file.clone().ok_or(VaultError::NotInitialized)?;
        let created_at = file.secrets.get(name).map_or(now, |r| r.created_at);
        file.secrets.insert(
            name.to_string(),
            SecretRecord {
                sealed,
                created_at,
                updated_at: now,
                description,
            },
        );
        write_atomically(&self.path, &file)?;
        self.file = Some(file);
        Ok(())
    }

    /// Read a secret's value. The vault must be open. The returned buffer is
    /// wiped when dropped.
    pub fn get(&mut self, name: &str, now: DateTime<Utc>) -> Result<Zeroizing<String>, VaultError> {
        self.expire_if_due(now);
        let key = &self.unlocked.as_ref().ok_or(VaultError::Locked)?.key;
        let record = self
            .file
            .as_ref()
            .and_then(|f| f.secrets.get(name))
            .ok_or(VaultError::UnknownSecret)?;
        let bytes = crypto::open(key, &record.sealed, &crypto::secret_aad(name))?;
        let value = String::from_utf8(bytes.to_vec()).map_err(|_| VaultError::Corrupt)?;
        Ok(Zeroizing::new(value))
    }

    /// Every stored value, decrypted — for the output masker only. The vault
    /// must be open.
    pub fn all_values(
        &mut self,
        now: DateTime<Utc>,
    ) -> Result<Vec<(String, Zeroizing<String>)>, VaultError> {
        let names: Vec<String> = self.list().into_iter().map(|m| m.name).collect();
        names
            .into_iter()
            .map(|name| self.get(&name, now).map(|v| (name, v)))
            .collect()
    }

    pub fn delete(&mut self, name: &str) -> Result<(), VaultError> {
        let mut file = self.file.clone().ok_or(VaultError::NotInitialized)?;
        if file.secrets.remove(name).is_none() {
            return Err(VaultError::UnknownSecret);
        }
        write_atomically(&self.path, &file)?;
        self.file = Some(file);
        Ok(())
    }

    pub fn is_unlocked(&mut self, now: DateTime<Utc>) -> bool {
        self.expire_if_due(now);
        self.unlocked.is_some()
    }

    fn expire_if_due(&mut self, now: DateTime<Utc>) {
        if self.unlocked.as_ref().is_some_and(|u| now >= u.until) {
            self.unlocked = None;
        }
    }
}

fn clamp_duration(d: Duration) -> Duration {
    if d <= Duration::zero() {
        Duration::minutes(1)
    } else if d > MAX_UNLOCK {
        MAX_UNLOCK
    } else {
        d
    }
}

/// Secret names end up in environment-variable-like references and in masking
/// labels (`[secret:NAME]`); keep them boring and unambiguous.
pub fn validate_name(name: &str) -> Result<(), VaultError> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if ok {
        Ok(())
    } else {
        Err(VaultError::InvalidName)
    }
}

/// The passphrase is the only thing between a stolen vault file and its
/// contents; a short one is brute-forced regardless of the KDF cost.
fn validate_passphrase(passphrase: &str) -> Result<(), VaultError> {
    if passphrase.chars().count() < 12 {
        return Err(VaultError::WeakPassphrase);
    }
    Ok(())
}

/// Write the file so that at no instant does zero valid copy exist: write a
/// temporary file beside it (0600 from creation, never readable by others even
/// briefly), flush it to disk, then rename over the original — rename is atomic
/// on the same filesystem.
fn write_atomically(path: &Path, file: &VaultFile) -> Result<(), VaultError> {
    let dir = path
        .parent()
        .ok_or_else(|| VaultError::Io("vault path has no parent".into()))?;
    std::fs::create_dir_all(dir).map_err(|e| VaultError::Io(e.to_string()))?;
    let bytes = serde_json::to_vec_pretty(file).map_err(|_| VaultError::Corrupt)?;

    let tmp = path.with_extension("json.tmp");
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp).map_err(|e| VaultError::Io(e.to_string()))?;
        f.write_all(&bytes)
            .map_err(|e| VaultError::Io(e.to_string()))?;
        f.sync_all().map_err(|e| VaultError::Io(e.to_string()))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| VaultError::Io(e.to_string()))?;
    // Make the rename itself durable.
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VaultError {
    #[error("the vault is not initialised yet")]
    NotInitialized,
    #[error("a vault already exists here; refusing to overwrite it")]
    AlreadyInitialized,
    #[error("the vault is locked")]
    Locked,
    #[error("wrong passphrase")]
    WrongPassphrase,
    #[error("the passphrase must be at least 12 characters")]
    WeakPassphrase,
    #[error("secret names are 1-64 characters: letters, digits, '_', '-', '.'")]
    InvalidName,
    #[error("the value must be at least {MIN_SECRET_LEN} characters (shorter values cannot be masked in agent output without false positives)")]
    ValueTooShort,
    #[error("no such secret")]
    UnknownSecret,
    #[error("the vault file is malformed")]
    Corrupt,
    #[error("unsupported vault file version {0}")]
    UnsupportedVersion(u32),
    #[error("vault storage error: {0}")]
    Io(String),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
}

#[cfg(test)]
mod tests {
    use super::*;

    const PASS: &str = "correct horse battery staple";

    fn temp_vault() -> (tempfile::TempDir, Vault) {
        let dir = tempfile::tempdir().unwrap();
        let vault = Vault::load(dir.path().join("vault.json")).unwrap();
        (dir, vault)
    }

    fn t0() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-10-01T09:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn init(v: &mut Vault) {
        v.init(
            PASS,
            KdfParams::insecure_for_tests(),
            Duration::minutes(30),
            t0(),
        )
        .unwrap();
    }

    #[test]
    fn a_fresh_vault_is_uninitialised_and_locked() {
        let (_d, mut v) = temp_vault();
        let s = v.status(t0());
        assert!(!s.initialized);
        assert_eq!(s.unlocked_until, None);
    }

    #[test]
    fn store_and_read_back_while_open() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        v.put("design-tool", "the-design-tool-passphrase", None, t0())
            .unwrap();
        assert_eq!(
            v.get("design-tool", t0()).unwrap().as_str(),
            "the-design-tool-passphrase"
        );
    }

    #[test]
    fn it_locks_itself_at_the_deadline() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        v.put("design-tool", "the-design-tool-passphrase", None, t0())
            .unwrap();
        let later = t0() + Duration::minutes(30);
        assert_eq!(v.get("design-tool", later), Err(VaultError::Locked));
        assert_eq!(v.status(later).unlocked_until, None);
    }

    #[test]
    fn a_reload_starts_locked_and_opens_only_with_the_passphrase() {
        // The server restarting must not reopen the vault by itself.
        let (d, mut v) = temp_vault();
        init(&mut v);
        v.put("design-tool", "the-design-tool-passphrase", None, t0())
            .unwrap();

        let mut reloaded = Vault::load(d.path().join("vault.json")).unwrap();
        assert_eq!(reloaded.get("design-tool", t0()), Err(VaultError::Locked));
        assert_eq!(
            reloaded.unlock("not the passphrase at all", Duration::minutes(5), t0()),
            Err(VaultError::WrongPassphrase)
        );
        reloaded.unlock(PASS, Duration::minutes(5), t0()).unwrap();
        assert_eq!(
            reloaded.get("design-tool", t0()).unwrap().as_str(),
            "the-design-tool-passphrase"
        );
    }

    #[test]
    fn the_file_holds_no_plaintext_and_no_passphrase() {
        let (d, mut v) = temp_vault();
        init(&mut v);
        v.put("design-tool", "the-design-tool-passphrase", None, t0())
            .unwrap();
        let raw = std::fs::read_to_string(d.path().join("vault.json")).unwrap();
        assert!(!raw.contains("the-design-tool-passphrase"));
        assert!(!raw.contains(PASS));
        assert!(
            raw.contains("design-tool"),
            "names are metadata, stored in clear"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_readable_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let (d, mut v) = temp_vault();
        init(&mut v);
        let mode = std::fs::metadata(d.path().join("vault.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn init_refuses_to_overwrite_an_existing_vault() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        assert_eq!(
            v.init(
                PASS,
                KdfParams::insecure_for_tests(),
                Duration::minutes(5),
                t0()
            ),
            Err(VaultError::AlreadyInitialized)
        );
    }

    #[test]
    fn short_values_and_weak_passphrases_are_refused() {
        let (_d, mut v) = temp_vault();
        assert_eq!(
            v.init(
                "short",
                KdfParams::insecure_for_tests(),
                Duration::minutes(5),
                t0()
            ),
            Err(VaultError::WeakPassphrase)
        );
        init(&mut v);
        assert_eq!(
            v.put("pin", "1234", None, t0()),
            Err(VaultError::ValueTooShort)
        );
    }

    #[test]
    fn names_are_validated() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        for bad in ["", "has space", "a/b", "$(rm)", &"x".repeat(65)] {
            assert_eq!(
                v.put(bad, "long-enough-value", None, t0()),
                Err(VaultError::InvalidName)
            );
        }
    }

    #[test]
    fn unlock_duration_is_capped() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        v.lock();
        let until = v.unlock(PASS, Duration::days(30), t0()).unwrap();
        assert_eq!(until, t0() + MAX_UNLOCK);
    }

    #[test]
    fn listing_works_locked_and_reveals_no_value() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        v.put(
            "design-tool",
            "the-design-tool-passphrase",
            Some("local design tool default vault".into()),
            t0(),
        )
        .unwrap();
        v.lock();
        let listed = v.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "design-tool");
        assert!(!format!("{listed:?}").contains("the-design-tool-passphrase"));
    }

    #[test]
    fn debug_of_an_open_vault_shows_no_key_and_no_value() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        v.put("design-tool", "the-design-tool-passphrase", None, t0())
            .unwrap();
        let debug = format!("{v:?}");
        assert!(!debug.contains("the-design-tool-passphrase"));
        assert!(debug.contains("<redacted>") || !debug.contains("key: SecretKey(["));
    }

    #[test]
    fn replacing_a_secret_keeps_its_creation_date() {
        let (_d, mut v) = temp_vault();
        init(&mut v);
        v.put("design-tool", "first-value-000", None, t0()).unwrap();
        let later = t0() + Duration::minutes(3);
        v.put("design-tool", "second-value-000", None, later)
            .unwrap();
        let meta = &v.list()[0];
        assert_eq!(meta.created_at, t0());
        assert_eq!(meta.updated_at, later);
        assert_eq!(
            v.get("design-tool", later).unwrap().as_str(),
            "second-value-000"
        );
    }
}
