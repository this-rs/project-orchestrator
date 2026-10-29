//! Where RL2's long cumuls survive a restart.
//!
//! Only the CALENDAR counters are persisted: a budget is a day, a week, a month,
//! and a restart at noon must not hand back a fresh allowance — that is the whole
//! point of counting money. A rolling window or a token bucket lives and dies in
//! memory: it is seconds to an hour of traffic, and reloading it would claim a
//! precision the process cannot have (the calls it counted are gone).
//!
//! Two implementations: [`MemoryStore`] (tests, and a process that wants no
//! file) and [`FileStore`] (one JSON object, rewritten atomically). The trait is
//! deliberately tiny and synchronous: a save is a few hundred bytes, and the
//! ledger never holds its lock across it longer than the write itself.
use crate::neo4j::traits::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Where the ledger keeps what must outlive the process.
pub trait LedgerStore: Send + Sync + std::fmt::Debug {
    /// The stored document for a key, if any.
    fn load(&self, key: &str) -> Option<String>;
    /// Stores (or replaces) the document of a key.
    fn save(&self, key: &str, value: &str);
}

/// A store that forgets everything when the process ends.
#[derive(Debug, Default)]
pub struct MemoryStore {
    entries: Mutex<HashMap<String, String>>,
}

impl MemoryStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many documents are stored.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl LedgerStore for MemoryStore {
    fn load(&self, key: &str) -> Option<String> {
        self.lock().get(key).cloned()
    }

    fn save(&self, key: &str, value: &str) {
        self.lock().insert(key.to_string(), value.to_string());
    }
}

/// A store backed by one JSON file, rewritten through a temporary file and a
/// rename so a crash mid-write cannot leave a half document behind.
#[derive(Debug)]
pub struct FileStore {
    path: PathBuf,
    entries: Mutex<HashMap<String, String>>,
}

impl FileStore {
    /// Opens (or creates) the store at `path`. A file that does not decode is
    /// treated as empty rather than guessed: a corrupt cumul must not become a
    /// false allowance.
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let entries = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<HashMap<String, String>>(&text).unwrap_or_default(),
            Err(_) => HashMap::new(),
        };
        Ok(Self {
            path,
            entries: Mutex::new(entries),
        })
    }

    /// The file this store writes.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl LedgerStore for FileStore {
    fn load(&self, key: &str) -> Option<String> {
        self.lock().get(key).cloned()
    }

    fn save(&self, key: &str, value: &str) {
        let document = {
            let mut entries = self.lock();
            entries.insert(key.to_string(), value.to_string());
            serde_json::to_string(&*entries).unwrap_or_default()
        };
        let temporary = self.path.with_extension("json.tmp");
        if std::fs::write(&temporary, document).is_ok() {
            let _ = std::fs::rename(&temporary, &self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("po-limits-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir.join("limits.json")
    }

    #[test]
    fn the_memory_store_round_trips_and_starts_empty() {
        let store = MemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.load("a"), None);
        store.save("a", "{\"used\":1}");
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(store.load("a").as_deref(), Some("{\"used\":1}"));
        store.save("a", "{\"used\":2}");
        assert_eq!(store.load("a").as_deref(), Some("{\"used\":2}"));
    }

    #[test]
    fn the_file_store_survives_a_new_process_and_reads_what_it_wrote() {
        let path = scratch("roundtrip");
        {
            let store = FileStore::open(&path).expect("open");
            assert_eq!(store.path(), path.as_path());
            assert_eq!(store.load("deepseek|budget"), None);
            store.save("deepseek|budget", "{\"used\":3.5}");
        }
        // A second process: the cumul is there, and an unknown key is absent (not zero).
        let again = FileStore::open(&path).expect("reopen");
        assert_eq!(
            again.load("deepseek|budget").as_deref(),
            Some("{\"used\":3.5}")
        );
        assert_eq!(again.load("other"), None);
    }

    #[test]
    fn a_file_that_does_not_decode_is_empty_rather_than_guessed() {
        let path = scratch("corrupt");
        std::fs::write(&path, "{not json at all").expect("write");
        let store = FileStore::open(&path).expect("open");
        assert_eq!(store.load("anything"), None);
    }
}
