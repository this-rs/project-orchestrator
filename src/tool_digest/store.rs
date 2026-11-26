//! Content-addressed, immutable artifact store on disk.
//!
//! An artifact is the raw bytes of one tool result, addressed by its own
//! sha256. Two consequences of that addressing:
//!
//! - equal bytes are stored exactly once (a second `put` of the same bytes is a
//!   no-op), and
//! - a stored artifact is never rewritten under its address, so a [`super::Digest`]
//!   that points at an artifact through a [`Ref`] can always recover the full
//!   result later, even when the digest's own summary is truncated.
//!
//! A missing artifact is a *named* error ([`StoreError::Missing`]) — never a
//! silent empty read — because a digest that silently loses its artifact would
//! report a summary it can no longer back up.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};

use super::split_lines;

/// Address of an artifact: the sha256 of its bytes, hex-encoded.
///
/// Content addressing is what makes the store immutable: the address is derived
/// from the content, so the same bytes always resolve to the same [`Ref`], and
/// nothing else can be stored under that address.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ref(String);

impl Ref {
    /// The hex-encoded sha256 (64 lowercase hex characters).
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn compute(bytes: &[u8]) -> Ref {
        Ref(hex::encode(Sha256::digest(bytes)))
    }
}

impl std::fmt::Display for Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Error of the artifact store. Every failure is a name, never a silence.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    /// The artifact is not on disk: never written here, or already expired.
    /// Retention is not implemented in this first brick, so an expired artifact
    /// is reported exactly like an absent one — as an error, not as an empty
    /// read.
    #[error("artifact {0} is absent from the store (never written, or expired)")]
    Missing(Ref),
    /// A filesystem error while writing to or reading from the store directory.
    #[error("artifact store I/O error: {0}")]
    Io(String),
}

/// Content-addressed, immutable artifact store rooted at one directory.
#[derive(Debug, Clone)]
pub struct ArtifactStore {
    dir: PathBuf,
}

impl ArtifactStore {
    /// A store rooted at `dir`. The directory is created lazily by [`Self::put`].
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        ArtifactStore { dir: dir.into() }
    }

    /// The directory the store is rooted at.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, r: &Ref) -> PathBuf {
        self.dir.join(r.as_str())
    }

    /// Store `bytes` and return their address. Idempotent and immutable: bytes
    /// already present under their address are not rewritten, and the address is
    /// the sha256 of the bytes, so equal bytes always share one artifact.
    pub fn put(&self, bytes: &[u8]) -> Result<Ref, StoreError> {
        let r = Ref::compute(bytes);
        let path = self.path(&r);
        if path.exists() {
            return Ok(r);
        }
        fs::create_dir_all(&self.dir).map_err(|e| StoreError::Io(e.to_string()))?;
        // Write to a sibling temporary file, then rename into place, so a crash
        // mid-write never leaves a truncated artifact under a real address.
        let tmp = self.dir.join(format!(".tmp-{}", r.as_str()));
        if let Err(e) = fs::write(&tmp, bytes) {
            let _ = fs::remove_file(&tmp);
            return Err(StoreError::Io(e.to_string()));
        }
        if let Err(e) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            // A concurrent writer may have won the race and written the same
            // bytes (same address); that is not an error.
            if path.exists() {
                return Ok(r);
            }
            return Err(StoreError::Io(e.to_string()));
        }
        Ok(r)
    }

    /// Byte length of the artifact, or [`StoreError::Missing`] if it is absent.
    pub fn len(&self, r: &Ref) -> Result<usize, StoreError> {
        match fs::metadata(self.path(r)) {
            Ok(m) => Ok(m.len() as usize),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(StoreError::Missing(r.clone())),
            Err(e) => Err(StoreError::Io(e.to_string())),
        }
    }

    /// The lines `[from_line, to_line)` of the artifact, joined with `\n`, with
    /// no trailing newline.
    ///
    /// Indices are 0-based and half-open, like a slice: `read_portion(r, 0, 3)`
    /// returns the first three lines. `from_line >= to_line` returns `""`, and
    /// indices past the end are clamped, so a caller always gets exactly the
    /// lines that exist in the requested range. Lines are split on `\n`
    /// (`\r\n` is normalized); a trailing newline terminates the last line
    /// without starting a new one (see [`split_lines`]).
    pub fn read_portion(
        &self,
        r: &Ref,
        from_line: usize,
        to_line: usize,
    ) -> Result<String, StoreError> {
        let bytes = match fs::read(self.path(r)) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(StoreError::Missing(r.clone()))
            }
            Err(e) => return Err(StoreError::Io(e.to_string())),
        };
        let text = String::from_utf8_lossy(&bytes);
        let lines = split_lines(&text);
        if from_line >= to_line || from_line >= lines.len() {
            return Ok(String::new());
        }
        let end = to_line.min(lines.len());
        Ok(lines[from_line..end].join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_is_idempotent_and_immutable() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path());
        let a = store.put(b"same bytes").unwrap();
        let b = store.put(b"same bytes").unwrap();
        assert_eq!(a, b, "equal bytes share one address");
        assert_eq!(store.len(&a).unwrap(), 10);

        // Writing other bytes must not touch the artifact under `a`.
        store.put(b"different").unwrap();
        assert_eq!(store.len(&a).unwrap(), 10);
        assert_eq!(store.read_portion(&a, 0, 1).unwrap(), "same bytes");
    }

    #[test]
    fn missing_artifact_is_a_named_error_not_a_silence() {
        let a = ArtifactStore::new(tempfile::tempdir().unwrap().path())
            .put(b"written here")
            .unwrap();
        // A different, empty store has never seen that artifact.
        let other = ArtifactStore::new(tempfile::tempdir().unwrap().path());
        assert!(matches!(other.len(&a), Err(StoreError::Missing(_))));
        assert!(matches!(
            other.read_portion(&a, 0, 1),
            Err(StoreError::Missing(_))
        ));
    }

    #[test]
    fn read_portion_returns_exactly_the_requested_lines() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path());
        let r = store.put(b"l0\nl1\nl2\nl3\n").unwrap();

        assert_eq!(store.read_portion(&r, 0, 4).unwrap(), "l0\nl1\nl2\nl3");
        assert_eq!(store.read_portion(&r, 1, 3).unwrap(), "l1\nl2");
        assert_eq!(store.read_portion(&r, 0, 1).unwrap(), "l0");
        assert_eq!(store.read_portion(&r, 3, 4).unwrap(), "l3");
        // Empty and out-of-range requests.
        assert_eq!(store.read_portion(&r, 2, 2).unwrap(), "");
        assert_eq!(store.read_portion(&r, 4, 5).unwrap(), "");
        assert_eq!(store.read_portion(&r, 5, 6).unwrap(), "");
        assert_eq!(store.read_portion(&r, 3, 0).unwrap(), "");
        // A range past the end is clamped to what exists.
        assert_eq!(store.read_portion(&r, 2, 100).unwrap(), "l2\nl3");
    }

    #[test]
    fn crlf_endings_are_normalized_and_empty_artifacts_have_no_lines() {
        let dir = tempfile::tempdir().unwrap();
        let store = ArtifactStore::new(dir.path());
        let r = store.put(b"a\r\nb\r\n").unwrap();
        assert_eq!(store.read_portion(&r, 0, 2).unwrap(), "a\nb");

        let empty = store.put(b"").unwrap();
        assert_eq!(store.read_portion(&empty, 0, 1).unwrap(), "");
        assert_eq!(store.len(&empty).unwrap(), 0);
    }
}
