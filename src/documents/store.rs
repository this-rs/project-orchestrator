//! Content-addressed local blob storage for uploaded documents.
//!
//! A blob's name **is** its SHA-256. That single decision buys three things
//! that would otherwise each need machinery of their own:
//!
//! - **Deduplication** — the same bytes always land on the same path, so the
//!   second upload of a file writes nothing.
//! - **Integrity** — the name is a checksum, so corruption is detectable on
//!   read rather than silently served ([`DocumentStore::get`] checks it).
//! - **Immutability** — a blob never changes, so nothing downstream ever needs
//!   cache invalidation.
//!
//! Deliberately **not** here: the business identity of a document (its UUID,
//! filename, owning project, upload time). That lives in the graph and points
//! at a hash. Two people uploading the same PDF get two `Document` nodes and
//! one blob; deleting one node must therefore not delete the blob, which is why
//! [`DocumentStore::delete`] is a blunt primitive and not a refcount.
//!
//! Storage is local disk, not object storage: PO is self-hosted, and a remote
//! bucket would add an operational dependency for no capability this layer
//! needs.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

/// Largest blob the store will accept, in bytes.
///
/// The binding constraint is memory, not disk. Every stage downstream of this
/// one is whole-file and in-memory: extraction takes `&[u8]` and returns a
/// `String`, chunking holds that string plus its chunks. At peak, one upload
/// keeps the raw bytes, the decoded text, and the chunk vector alive at once —
/// call it three times the blob size. 64 MiB therefore bounds a single upload
/// at a few hundred MiB of resident memory, which a self-hosted single-node
/// instance can absorb without a swap storm.
///
/// It is also far above real documents: a 500-page text PDF is single-digit
/// MiB, and only scanned image PDFs reach tens of MiB. Raising it means
/// revisiting the in-memory pipeline first, not just this number.
///
/// This is a backstop, not the first line of defence — an HTTP upload handler
/// must cap the request body itself, because by the time [`DocumentStore::put`]
/// is called the bytes are already resident.
pub const MAX_BLOB_BYTES: u64 = 64 * 1024 * 1024;

/// Length of a SHA-256 rendered as lowercase hex.
const SHA256_HEX_LEN: usize = 64;

/// Hex characters consumed by each level of the directory fan-out.
///
/// Two levels of two characters give 256 top-level directories, each with up to
/// 256 children: 65 536 leaf directories. A flat layout would put every blob in
/// one directory, where tens of thousands of entries make `readdir` slow on
/// ext4 and painful on anything networked.
const SHARD_WIDTH: usize = 2;

/// Prefix for in-flight temporary files.
///
/// Leading dot so a stray temp file (killed process, full disk) is at least
/// visually distinct from a real blob, and it can never be mistaken for one:
/// a valid blob name is 64 hex characters, which cannot start with `.`.
const TEMP_PREFIX: &str = ".tmp-";

/// Distinguishes concurrent temp files inside one process. The PID separates
/// processes; this separates threads within a process.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("blob is {size} bytes, over the {limit} byte limit")]
    TooLarge { size: u64, limit: u64 },

    /// A digest that failed validation. The offending input is **not** echoed
    /// back: it may come straight from a URL, and putting attacker-controlled
    /// text into logs and error responses is how log injection starts.
    #[error("invalid sha256 digest: {reason}")]
    InvalidDigest { reason: &'static str },

    #[error("no blob stored under {sha256}")]
    NotFound { sha256: String },

    /// The bytes on disk do not hash to the name they are filed under. Either
    /// the file was tampered with, or it was written by something that did not
    /// go through [`DocumentStore::put`].
    #[error("blob {expected} is corrupt: its contents hash to {actual}")]
    Corrupted { expected: String, actual: String },

    #[error("{operation} failed on {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl StoreError {
    fn io(operation: &'static str, path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }
}

/// Default blob root, under the same per-platform application directory the
/// rest of PO already uses (see `identity::default_storage_path` and the
/// `config.yaml` search paths in `lib.rs`). Keeping every piece of local state
/// under one directory is what makes "back up PO" a single `tar`.
pub fn default_storage_dir() -> PathBuf {
    if let Some(config_dir) = dirs::config_dir() {
        #[cfg(target_os = "windows")]
        let app_dir = config_dir.join("ProjectOrchestrator");
        #[cfg(not(target_os = "windows"))]
        let app_dir = config_dir.join("project-orchestrator");
        app_dir.join("documents")
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".project-orchestrator")
            .join("documents")
    }
}

/// Content-addressed blob store rooted at a single directory.
///
/// Cheap to clone-by-reference and safe to share: every operation is a
/// standalone filesystem call, so no interior state needs locking.
#[derive(Debug, Clone)]
pub struct DocumentStore {
    root: PathBuf,
    max_blob_bytes: u64,
}

impl DocumentStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_blob_bytes: MAX_BLOB_BYTES,
        }
    }

    /// Build a store from the runtime config, honouring `documents.storage_dir`
    /// (with `~` expansion, as elsewhere in the config) and falling back to
    /// [`default_storage_dir`].
    pub fn from_config(config: &crate::Config) -> Self {
        let root = config
            .documents_storage_dir
            .as_deref()
            .map(|dir| PathBuf::from(crate::expand_tilde(dir)))
            .unwrap_or_else(default_storage_dir);
        Self::new(root)
    }

    /// Override the size cap. Exists mainly so tests can exercise the limit
    /// without allocating 64 MiB.
    pub fn with_max_blob_bytes(mut self, max_blob_bytes: u64) -> Self {
        self.max_blob_bytes = max_blob_bytes;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn max_blob_bytes(&self) -> u64 {
        self.max_blob_bytes
    }

    /// Where a blob with this digest lives, or [`StoreError::InvalidDigest`].
    ///
    /// The validation in here is the only thing standing between a digest taken
    /// from a URL path segment and `self.root.join(..)`. It runs **before** any
    /// path is built, so a hostile digest never reaches the filesystem at all.
    pub fn path_for(&self, sha256: &str) -> Result<PathBuf, StoreError> {
        let digest = validate_digest(sha256)?;
        Ok(self
            .root
            .join(&digest[..SHARD_WIDTH])
            .join(&digest[SHARD_WIDTH..SHARD_WIDTH * 2])
            // The full digest, not the remaining 60 characters: a blob found
            // loose in a backup or a support dump still says what it is, and
            // `find . -name <hash>` works without knowing the shard scheme.
            .join(digest))
    }

    /// Store `bytes`, returning their SHA-256 as lowercase hex.
    ///
    /// Idempotent by construction: storing identical bytes twice yields the
    /// same digest and leaves one file on disk.
    pub fn put(&self, bytes: &[u8]) -> Result<String, StoreError> {
        let size = bytes.len() as u64;
        if size > self.max_blob_bytes {
            return Err(StoreError::TooLarge {
                size,
                limit: self.max_blob_bytes,
            });
        }

        let digest = hex::encode(Sha256::digest(bytes));
        let target = self.path_for(&digest)?;

        // Deduplication. Content addressing makes this a correctness-free
        // shortcut rather than a cache: if the file is there, its bytes are by
        // definition these bytes, so there is nothing to write.
        if target.exists() {
            return Ok(digest);
        }

        let shard = target
            .parent()
            .expect("path_for always yields a two-level path under root");
        fs::create_dir_all(shard)
            .map_err(|e| StoreError::io("create shard directory", shard, e))?;

        self.write_atomically(shard, &target, bytes)?;
        Ok(digest)
    }

    /// Write `bytes` to `target` so that `target` never exists in a partial
    /// state.
    ///
    /// The failure this prevents is the nastiest one a content-addressed store
    /// can have. Write straight to `target` and a crash mid-write leaves a
    /// truncated file readable under a hash that promises different content.
    /// Nothing downstream can detect it: the hash is the identity, so every
    /// consumer trusts it, dedup then *skips* re-uploading the good copy
    /// because the path already exists, and the damage is permanent.
    ///
    /// So: write to a temporary file in the destination's own directory, fsync
    /// it, then rename. The temp file is a sibling of the target, which
    /// guarantees the same filesystem — `rename(2)` across filesystems fails
    /// with `EXDEV`, and a temp directory elsewhere (`/tmp`, `TMPDIR`) very
    /// often is a different one.
    fn write_atomically(
        &self,
        shard: &Path,
        target: &Path,
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        let temp = shard.join(format!(
            "{TEMP_PREFIX}{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));

        let write = (|| -> std::io::Result<()> {
            let mut file = fs::File::create(&temp)?;
            file.write_all(bytes)?;
            // Without this, `rename` only orders the *metadata*: a crash can
            // leave the final name pointing at a zero-length file, which is
            // exactly the corruption the rename was supposed to prevent.
            file.sync_all()
        })();

        if let Err(e) = write {
            // Best effort: if this fails too, we leak a temp file, which is
            // inert — it can never be read as a blob.
            let _ = fs::remove_file(&temp);
            return Err(StoreError::io("write temporary blob", &temp, e));
        }

        if let Err(e) = fs::rename(&temp, target) {
            let _ = fs::remove_file(&temp);
            // A concurrent `put` of the same content may have won the race. On
            // Windows `rename` onto an existing file fails outright, so this is
            // the normal path there, not just a rare interleaving. Either way
            // the target holds these exact bytes, so the operation succeeded.
            // `is_file`, not `exists`: a *directory* squatting on the target
            // path also "exists", and swallowing the error there would report
            // a blob as stored when nothing was written.
            if target.is_file() {
                return Ok(());
            }
            return Err(StoreError::io("rename blob into place", target, e));
        }

        Ok(())
    }

    /// Read a blob back, verifying it still hashes to its own name.
    ///
    /// The re-hash is the point of content addressing: it turns bit rot, a
    /// truncated file left by some older non-atomic writer, or a hand-edited
    /// blob into a loud [`StoreError::Corrupted`] instead of quietly wrong text
    /// flowing into extraction, chunking and eventually an agent's context.
    /// SHA-256 runs at gigabytes per second, so the cost is small next to the
    /// read that just happened.
    pub fn get(&self, sha256: &str) -> Result<Vec<u8>, StoreError> {
        let path = self.path_for(sha256)?;
        let bytes = fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StoreError::NotFound {
                    sha256: sha256.to_string(),
                }
            } else {
                StoreError::io("read blob", &path, e)
            }
        })?;

        let actual = hex::encode(Sha256::digest(&bytes));
        if actual != sha256 {
            return Err(StoreError::Corrupted {
                expected: sha256.to_string(),
                actual,
            });
        }
        Ok(bytes)
    }

    /// Whether a blob is stored. Still validates the digest first — callers
    /// reach this with untrusted input just as often as they reach [`Self::get`].
    pub fn exists(&self, sha256: &str) -> Result<bool, StoreError> {
        Ok(self.path_for(sha256)?.is_file())
    }

    /// Remove a blob. Returns whether anything was there to remove.
    ///
    /// Blunt on purpose: this store has no idea how many `Document` nodes point
    /// at a blob, so it is not the place to decide that deletion is safe. That
    /// belongs to whatever owns the graph side.
    ///
    /// Empty shard directories are left behind rather than pruned. Pruning
    /// would race a concurrent [`Self::put`], which has already run
    /// `create_dir_all` and is about to create its temp file inside that
    /// directory; 65 536 empty directories cost a few MiB at worst, an
    /// intermittent upload failure costs more.
    pub fn delete(&self, sha256: &str) -> Result<bool, StoreError> {
        let path = self.path_for(sha256)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(StoreError::io("delete blob", &path, e)),
        }
    }
}

/// Accept only a canonical SHA-256: exactly 64 lowercase hex characters.
///
/// Two rules, both load-bearing.
///
/// **Length and alphabet** are what make path traversal structurally
/// impossible rather than filtered-out. `..`, `../../etc/passwd`, an absolute
/// path, a NUL byte, a URL-encoded separator — none of them survive "64
/// characters, each one of `0-9a-f`". This is an allowlist, so it does not
/// need to enumerate the attacks.
///
/// **Lowercase only** is not cosmetic. `hex::encode` emits lowercase, so an
/// uppercase digest never names a blob this store wrote. Accepting it would
/// mean one blob has two valid spellings: on a case-sensitive filesystem that
/// is a second, separate file defeating deduplication; on a case-insensitive
/// one (macOS by default) the two spellings collide, and `exists` starts
/// answering about a file `put` never created.
fn validate_digest(sha256: &str) -> Result<&str, StoreError> {
    if sha256.is_empty() {
        return Err(StoreError::InvalidDigest { reason: "empty" });
    }
    if sha256.len() != SHA256_HEX_LEN {
        return Err(StoreError::InvalidDigest {
            reason: "must be exactly 64 characters",
        });
    }
    if !sha256
        .bytes()
        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(StoreError::InvalidDigest {
            reason: "must be lowercase hexadecimal",
        });
    }
    Ok(sha256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store() -> (TempDir, DocumentStore) {
        let dir = TempDir::new().expect("tempdir");
        let store = DocumentStore::new(dir.path());
        (dir, store)
    }

    fn sha256_of(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// Every regular file under `root`, temp files included.
    fn all_files(root: &Path) -> Vec<PathBuf> {
        walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect()
    }

    #[test]
    fn put_then_get_round_trips_the_exact_bytes() {
        let (_dir, store) = store();
        let bytes = b"# Rapport\n\nDu texte accentu\xc3\xa9 et des octets \x00\x01\xff bruts.";

        let digest = store.put(bytes).unwrap();
        assert_eq!(digest, sha256_of(bytes));
        assert_eq!(store.get(&digest).unwrap(), bytes);
        assert!(store.exists(&digest).unwrap());
    }

    #[test]
    fn stores_an_empty_blob_like_any_other() {
        // Zero bytes is a legitimate upload, and the branch most likely to be
        // confused with "nothing was written" by a later reader of this code.
        let (_dir, store) = store();
        let digest = store.put(b"").unwrap();
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(store.get(&digest).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn shards_two_levels_deep_so_no_directory_holds_every_blob() {
        let (dir, store) = store();
        let digest = store.put(b"sharding").unwrap();

        let expected = dir
            .path()
            .join(&digest[0..2])
            .join(&digest[2..4])
            .join(&digest);
        assert_eq!(store.path_for(&digest).unwrap(), expected);
        assert!(expected.is_file(), "blob should live at {expected:?}");
    }

    #[test]
    fn identical_content_is_stored_once() {
        let (dir, store) = store();
        let bytes = b"the same document, uploaded by two different people";

        let first = store.put(bytes).unwrap();
        let second = store.put(bytes).unwrap();

        assert_eq!(first, second, "content addressing must agree on the name");
        assert_eq!(
            all_files(dir.path()).len(),
            1,
            "a second upload of identical bytes must not create a second file"
        );
    }

    #[test]
    fn different_content_never_shares_a_file() {
        let (dir, store) = store();
        let a = store.put(b"alpha").unwrap();
        let b = store.put(b"beta").unwrap();

        assert_ne!(a, b);
        assert_eq!(all_files(dir.path()).len(), 2);
        assert_eq!(store.get(&a).unwrap(), b"alpha");
        assert_eq!(store.get(&b).unwrap(), b"beta");
    }

    #[test]
    fn a_successful_put_leaves_no_temporary_file_behind() {
        let (dir, store) = store();
        store.put(b"contenu").unwrap();

        let leftovers: Vec<_> = all_files(dir.path())
            .into_iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(TEMP_PREFIX))
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files survived put: {leftovers:?}"
        );
    }

    #[test]
    fn a_truncated_blob_left_by_an_interrupted_put_is_unreachable() {
        // What a crash mid-`put` actually leaves on disk: a partial file under
        // a temporary name. Reproduce that state by hand and prove the hash is
        // still unclaimed — the final name only ever comes into existence at
        // the rename, with the whole file already fsynced behind it.
        //
        // Were `put` to write straight to the target path, this same
        // interruption would leave those truncated bytes readable under the
        // full content's hash: a hash promising content it does not have, with
        // dedup then refusing to overwrite it.
        let (dir, store) = store();
        let full = b"the complete document, all of it";
        let digest = sha256_of(full);

        let shard = dir.path().join(&digest[0..2]).join(&digest[2..4]);
        fs::create_dir_all(&shard).unwrap();
        fs::write(shard.join(format!("{TEMP_PREFIX}1234-0")), &full[..8]).unwrap();

        assert!(
            !store.exists(&digest).unwrap(),
            "a partial write must not be visible under the full content's hash"
        );
        assert!(matches!(
            store.get(&digest),
            Err(StoreError::NotFound { .. })
        ));

        // And the interrupted upload is fully recoverable by simply retrying.
        assert_eq!(store.put(full).unwrap(), digest);
        assert_eq!(store.get(&digest).unwrap(), full);
    }

    #[cfg(unix)]
    #[test]
    fn the_target_is_replaced_wholesale_and_never_opened_in_place() {
        // The deterministic half of the atomicity argument. Make the target
        // unwritable: `rename(2)` cares about the *directory's* permissions,
        // not the victim file's, so a temp-then-rename writer sails through,
        // while anything that opens the target and writes into it gets EACCES.
        //
        // A write that opens the target is exactly the one that can leave it
        // truncated; this test fails the moment `write_atomically` starts doing
        // that, without needing to catch an interrupted write in the act.
        use std::os::unix::fs::PermissionsExt;

        let (dir, store) = store();
        let full = b"the complete replacement content";
        let digest = sha256_of(full);
        let target = store.path_for(&digest).unwrap();
        let shard = target.parent().unwrap();

        fs::create_dir_all(shard).unwrap();
        fs::write(&target, b"stale").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o444)).unwrap();

        store.write_atomically(shard, &target, full).unwrap();

        assert_eq!(fs::read(&target).unwrap(), full);
        assert_eq!(
            all_files(dir.path()).len(),
            1,
            "the temp file must be gone, replaced by the target"
        );
    }

    #[test]
    fn get_refuses_a_blob_whose_bytes_no_longer_match_its_name() {
        // Bit rot, a bad restore, a hand-edited file. Without the read-side
        // check this returns wrong bytes under a trusted hash and every
        // consumer believes them.
        let (_dir, store) = store();
        let digest = store.put(b"original content").unwrap();

        fs::write(store.path_for(&digest).unwrap(), b"tampered content").unwrap();

        match store.get(&digest) {
            Err(StoreError::Corrupted { expected, actual }) => {
                assert_eq!(expected, digest);
                assert_ne!(actual, digest);
            }
            other => panic!("expected Corrupted, got {other:?}"),
        }
    }

    #[test]
    fn hostile_digests_are_refused_before_any_path_is_built() {
        // A sha256 reaches this store from a URL path segment. Each of these is
        // rejected by the allowlist, and — the part that matters — rejected
        // *before* `join`, so none of them touches the filesystem.
        let (dir, store) = store();
        let hostile = [
            ("empty", ""),
            ("bare traversal", ".."),
            ("relative traversal", "../../../../etc/passwd"),
            (
                "traversal padded to 64 characters",
                "../../../../../../../../../../../../../../../../../../../../etc/p",
            ),
            ("absolute path", "/etc/passwd"),
            ("absolute path inside the root", "/tmp/blob"),
            (
                "separator in an otherwise valid digest",
                &format!("{}/{}", "a".repeat(32), "b".repeat(31)),
            ),
            (
                "backslash separator",
                &format!("{}\\{}", "a".repeat(32), "b".repeat(31)),
            ),
            ("too short", &"a".repeat(63)),
            ("too long", &"a".repeat(65)),
            ("non-hex letters", &"z".repeat(64)),
            ("uppercase hex", &"A".repeat(64)),
            ("nul byte", &format!("{}\0", "a".repeat(63))),
            ("newline", &format!("{}\n", "a".repeat(63))),
            (
                "percent-encoded traversal",
                &format!("%2e%2e%2f{}", "a".repeat(55)),
            ),
            ("non-ascii", &"é".repeat(32)),
            ("whitespace padding", &format!(" {} ", "a".repeat(62))),
            ("home shorthand", "~"),
        ];

        for (label, input) in hostile {
            assert!(
                matches!(store.path_for(input), Err(StoreError::InvalidDigest { .. })),
                "path_for accepted {label}: {input:?}"
            );
            assert!(
                matches!(store.get(input), Err(StoreError::InvalidDigest { .. })),
                "get accepted {label}: {input:?}"
            );
            assert!(
                matches!(store.exists(input), Err(StoreError::InvalidDigest { .. })),
                "exists accepted {label}: {input:?}"
            );
            assert!(
                matches!(store.delete(input), Err(StoreError::InvalidDigest { .. })),
                "delete accepted {label}: {input:?}"
            );
        }

        assert!(
            all_files(dir.path()).is_empty(),
            "rejected digests must not have created anything on disk"
        );
    }

    #[test]
    fn a_valid_looking_digest_never_escapes_the_root() {
        // The positive half of the traversal argument: anything the validator
        // accepts produces a path strictly under `root`. There is no accepted
        // input for which this is false, because `..` is not hex.
        let (dir, store) = store();
        for digest in [
            "0".repeat(64),
            "f".repeat(64),
            sha256_of(b"whatever"),
            format!("{}{}", "0".repeat(32), "f".repeat(32)),
        ] {
            let path = store.path_for(&digest).unwrap();
            assert!(
                path.starts_with(dir.path()),
                "{path:?} escaped {:?}",
                dir.path()
            );
            assert!(
                !path.components().any(|c| c.as_os_str() == ".."),
                "{path:?} contains a parent-directory component"
            );
        }
    }

    #[test]
    fn refuses_blobs_over_the_size_cap() {
        let dir = TempDir::new().unwrap();
        let store = DocumentStore::new(dir.path()).with_max_blob_bytes(16);

        assert_eq!(store.put(&[0u8; 16]).unwrap(), sha256_of(&[0u8; 16]));

        match store.put(&[0u8; 17]) {
            Err(StoreError::TooLarge { size, limit }) => {
                assert_eq!((size, limit), (17, 16));
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }

        assert_eq!(
            all_files(dir.path()).len(),
            1,
            "an oversized blob must be refused before anything is written"
        );
    }

    #[test]
    fn the_default_cap_is_the_named_constant() {
        let (_dir, store) = store();
        assert_eq!(store.max_blob_bytes(), MAX_BLOB_BYTES);
    }

    #[test]
    fn concurrent_puts_of_the_same_content_agree_and_write_one_file() {
        // Eight threads racing on the same shard directory: create_dir_all,
        // distinct temp files, then eight renames onto one target. Each rename
        // is atomic and every writer carries identical bytes, so the target is
        // always a complete blob — never a half-written one, never two files.
        let dir = TempDir::new().unwrap();
        let bytes: Vec<u8> = (0..64_000u32).map(|i| (i % 251) as u8).collect();
        let expected = sha256_of(&bytes);

        let digests: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let store = DocumentStore::new(dir.path());
                    let bytes = &bytes;
                    scope.spawn(move || store.put(bytes).unwrap())
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert!(digests.iter().all(|d| *d == expected));
        assert_eq!(
            all_files(dir.path()).len(),
            1,
            "concurrent writers must converge on one file and clean up their temps"
        );
        assert_eq!(
            DocumentStore::new(dir.path()).get(&expected).unwrap(),
            bytes,
            "the surviving file must be the complete content"
        );
    }

    #[test]
    fn concurrent_puts_of_different_content_all_survive() {
        let dir = TempDir::new().unwrap();
        let payloads: Vec<Vec<u8>> = (0..16u8).map(|i| vec![i; 1024 + i as usize]).collect();

        let digests: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = payloads
                .iter()
                .map(|payload| {
                    let store = DocumentStore::new(dir.path());
                    scope.spawn(move || store.put(payload).unwrap())
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let store = DocumentStore::new(dir.path());
        for (digest, payload) in digests.iter().zip(&payloads) {
            assert_eq!(&store.get(digest).unwrap(), payload);
        }
        assert_eq!(all_files(dir.path()).len(), payloads.len());
    }

    #[test]
    fn exists_and_delete_report_what_is_actually_there() {
        let (_dir, store) = store();
        let digest = store.put(b"deletable").unwrap();
        let absent = sha256_of(b"never stored");

        assert!(store.exists(&digest).unwrap());
        assert!(!store.exists(&absent).unwrap());

        assert!(store.delete(&digest).unwrap(), "first delete removes it");
        assert!(
            !store.delete(&digest).unwrap(),
            "deleting twice must report absence, not fail"
        );
        assert!(!store.exists(&digest).unwrap());
        assert!(matches!(
            store.get(&digest),
            Err(StoreError::NotFound { .. })
        ));

        assert!(!store.delete(&absent).unwrap());
    }

    #[test]
    fn a_blob_can_be_stored_again_after_deletion() {
        // Dedup short-circuits on `target.exists()`; after a delete that has to
        // go back to actually writing rather than silently returning the hash.
        let (_dir, store) = store();
        let digest = store.put(b"round trip").unwrap();
        store.delete(&digest).unwrap();

        assert_eq!(store.put(b"round trip").unwrap(), digest);
        assert_eq!(store.get(&digest).unwrap(), b"round trip");
    }

    #[test]
    fn get_and_exists_do_not_create_the_directories_they_look_in() {
        // A probe for a blob that was never stored must stay read-only:
        // otherwise anyone able to guess hashes can make PO mkdir 65 536
        // directories.
        let (dir, store) = store();
        let absent = sha256_of(b"absent");

        assert!(matches!(
            store.get(&absent),
            Err(StoreError::NotFound { .. })
        ));
        assert!(!store.exists(&absent).unwrap());
        assert!(!dir.path().join(&absent[0..2]).exists());
    }

    #[test]
    fn default_storage_dir_sits_under_the_app_directory() {
        let path = default_storage_dir();
        assert!(path.ends_with("documents"));
        assert!(
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.eq_ignore_ascii_case("project-orchestrator")
                    || n == "ProjectOrchestrator"
                    || n == ".project-orchestrator"),
            "unexpected parent for {path:?}"
        );
    }
}
