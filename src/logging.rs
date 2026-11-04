//! Persistent logging: the backend's traces outlive the terminal that started it.
//!
//! Until 03/11/2026 the server only traced to the stdout of whoever launched it, so a
//! 930 s silent turn (locked vault), the restarts and the vault refusals could not be
//! investigated afterwards. A second layer now writes the same events to a daily-rotated
//! file, capped by the number of files kept. Stdout is unchanged and `RUST_LOG` still
//! drives it.
//!
//! The file never holds a secret: every line goes through the vault's global masker
//! ([`crate::vault::mask::global`]) before it reaches the disk — values delivered to an
//! agent are replaced by `[secret:NAME]`, exactly as on the chat paths.
//!
//! Settings (`config.yaml`, section `log`; each has an env override):
//!
//! | key | env | default |
//! |---|---|---|
//! | `file_enabled` | `LOG_FILE_ENABLED` | `true` |
//! | `dir` | `LOG_DIR` | `<app dir>/logs` (next to `config.yaml`) |
//! | `level` | `LOG_LEVEL` | `info` |
//! | `max_files` | `LOG_MAX_FILES` | `14` (days) |

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Prefix of the log files: `orchestrator.YYYY-MM-DD.log`.
pub const FILE_PREFIX: &str = "orchestrator";
const FILE_SUFFIX: &str = "log";
const DEFAULT_MAX_FILES: usize = 14;
const DEFAULT_FILE_LEVEL: &str = "info";
/// What stdout shows when `RUST_LOG` is unset (unchanged since before the file layer).
const STDOUT_DEFAULT: &str = "info,project_orchestrator=debug,tower_http=debug";

/// The `log` section of `config.yaml`.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct LogYamlConfig {
    /// Write a log file (default: true).
    pub file_enabled: Option<bool>,
    /// Directory of the log files (default: `<app dir>/logs`).
    pub dir: Option<String>,
    /// Filter of the FILE only, `RUST_LOG` syntax (default: `info`).
    pub level: Option<String>,
    /// Daily files kept before the oldest is deleted (default: 14).
    pub max_files: Option<usize>,
}

/// Resolved settings (env > YAML > default).
#[derive(Debug, Clone)]
pub struct LogSettings {
    pub file_enabled: bool,
    pub dir: PathBuf,
    pub level: String,
    pub max_files: usize,
}

impl Default for LogSettings {
    fn default() -> Self {
        Self::resolve(&LogYamlConfig::default(), |_| None)
    }
}

/// `<app dir>/logs`, the app directory being the one that holds `config.yaml`.
pub fn default_log_dir() -> PathBuf {
    if let Some(config_dir) = dirs::config_dir() {
        #[cfg(target_os = "windows")]
        let app_dir = config_dir.join("ProjectOrchestrator");
        #[cfg(not(target_os = "windows"))]
        let app_dir = config_dir.join("project-orchestrator");
        app_dir.join("logs")
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".project-orchestrator")
            .join("logs")
    }
}

impl LogSettings {
    /// `env` is injected so the precedence is testable without touching the process.
    pub fn resolve(yaml: &LogYamlConfig, env: impl Fn(&str) -> Option<String>) -> Self {
        let non_empty = |v: Option<String>| v.filter(|s| !s.trim().is_empty());
        let file_enabled = env("LOG_FILE_ENABLED")
            .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            })
            .or(yaml.file_enabled)
            .unwrap_or(true);
        let dir = non_empty(env("LOG_DIR"))
            .or_else(|| non_empty(yaml.dir.clone()))
            .map(|d| PathBuf::from(crate::expand_tilde(&d)))
            .unwrap_or_else(default_log_dir);
        let level = non_empty(env("LOG_LEVEL"))
            .or_else(|| non_empty(yaml.level.clone()))
            .unwrap_or_else(|| DEFAULT_FILE_LEVEL.to_string());
        let max_files = env("LOG_MAX_FILES")
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
            .or(yaml.max_files.filter(|n| *n > 0))
            .unwrap_or(DEFAULT_MAX_FILES);
        Self {
            file_enabled,
            dir,
            level,
            max_files,
        }
    }

    pub fn from_env(yaml: &LogYamlConfig) -> Self {
        Self::resolve(yaml, |k| std::env::var(k).ok())
    }
}

/// Keeps the background writer alive: dropping it flushes the file. Hold it in `main`.
pub struct LogGuard {
    _worker: Option<tracing_appender::non_blocking::WorkerGuard>,
    /// Directory the file is written to, when the file layer is on.
    pub dir: Option<PathBuf>,
}

/// Masks every known secret value in what is written through it.
///
/// `fmt` hands over one whole formatted event per `write`, so a value is never split
/// across two calls.
pub struct MaskingWriter<W: Write> {
    inner: W,
    masker: std::sync::Arc<crate::vault::mask::Masker>,
}

impl<W: Write> MaskingWriter<W> {
    pub fn new(inner: W, masker: std::sync::Arc<crate::vault::mask::Masker>) -> Self {
        Self { inner, masker }
    }
}

impl<W: Write> Write for MaskingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.masker.is_empty() {
            return self.inner.write_all(buf).map(|()| buf.len());
        }
        let text = String::from_utf8_lossy(buf);
        let masked = self.masker.mask(&text);
        self.inner.write_all(masked.as_bytes())?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// `MakeWriter` that re-reads the global masker for every event, so a secret delivered
/// after startup is masked from then on.
#[derive(Clone)]
struct MaskedMakeWriter<M>(M);

impl<'a, M> tracing_subscriber::fmt::MakeWriter<'a> for MaskedMakeWriter<M>
where
    M: tracing_subscriber::fmt::MakeWriter<'a>,
{
    type Writer = MaskingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        MaskingWriter::new(
            self.0.make_writer(),
            crate::vault::mask::global().snapshot(),
        )
    }
}

fn file_filter(settings: &LogSettings) -> EnvFilter {
    // LOG_LEVEL / log.level were already merged in `settings.level`; an unparsable
    // directive falls back to the default instead of silencing the file.
    EnvFilter::try_new(&settings.level).unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILE_LEVEL))
}

fn stdout_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| STDOUT_DEFAULT.into())
}

/// Build the rolling appender; the directory is created if missing.
fn open_appender(
    dir: &Path,
    max_files: usize,
) -> Result<tracing_appender::rolling::RollingFileAppender, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix(FILE_PREFIX)
        .filename_suffix(FILE_SUFFIX)
        .max_log_files(max_files)
        .build(dir)
        .map_err(|e| format!("cannot open a log file in {}: {e}", dir.display()))
}

/// Install the global subscriber: stdout (as before) + the file when enabled.
/// A file that cannot be opened is reported on stderr and never stops the server.
pub fn init(settings: &LogSettings) -> LogGuard {
    let stdout = tracing_subscriber::fmt::layer().with_filter(stdout_filter());

    if settings.file_enabled {
        match open_appender(&settings.dir, settings.max_files) {
            Ok(appender) => {
                let (writer, worker) = tracing_appender::non_blocking(appender);
                let file = tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(MaskedMakeWriter(writer))
                    .with_filter(file_filter(settings));
                tracing_subscriber::registry()
                    .with(stdout)
                    .with(file)
                    .init();
                return LogGuard {
                    _worker: Some(worker),
                    dir: Some(settings.dir.clone()),
                };
            }
            Err(e) => eprintln!("warning: file logging disabled — {e}"),
        }
    }
    tracing_subscriber::registry().with(stdout).init();
    LogGuard {
        _worker: None,
        dir: None,
    }
}

/// The line that dates a (re)start in the file.
pub fn log_startup(settings: &LogSettings, guard: &LogGuard) {
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        log_dir = guard
            .dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_else(|| "(file logging off)".into()),
        log_level = %settings.level,
        max_files = settings.max_files,
        "orchestrator starting"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::mask::Masker;
    use std::sync::Arc;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn defaults_are_a_file_under_the_app_dir_at_info() {
        let s = LogSettings::resolve(&LogYamlConfig::default(), no_env);
        assert!(s.file_enabled);
        assert!(s.dir.ends_with("logs"));
        assert_eq!(s.level, "info");
        assert_eq!(s.max_files, DEFAULT_MAX_FILES);
    }

    #[test]
    fn env_beats_yaml_and_bad_values_fall_back() {
        let yaml = LogYamlConfig {
            file_enabled: Some(true),
            dir: Some("/yaml".into()),
            level: Some("warn".into()),
            max_files: Some(3),
        };
        let s = LogSettings::resolve(&yaml, no_env);
        assert_eq!(
            (s.dir.as_path(), s.level.as_str(), s.max_files),
            (Path::new("/yaml"), "warn", 3)
        );
        let s = LogSettings::resolve(&yaml, |k| match k {
            "LOG_DIR" => Some("/env".into()),
            "LOG_LEVEL" => Some("debug".into()),
            "LOG_MAX_FILES" => Some("0".into()), // invalid → yaml
            "LOG_FILE_ENABLED" => Some("off".into()),
            _ => None,
        });
        assert_eq!(s.dir, PathBuf::from("/env"));
        assert_eq!(s.level, "debug");
        assert_eq!(s.max_files, 3);
        assert!(!s.file_enabled);
    }

    #[test]
    fn a_secret_never_reaches_the_file() {
        let masker = Arc::new(Masker::from_values([("demo", "hunter2-very-secret")]));
        let mut out = Vec::new();
        let mut w = MaskingWriter::new(&mut out, masker);
        w.write_all(b"INFO login with hunter2-very-secret done\n")
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("hunter2-very-secret"));
        assert!(text.contains("[secret:demo]"));
    }

    #[test]
    fn the_file_layer_writes_notices_and_refusals_and_masks_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let appender = open_appender(dir.path(), 2).unwrap();
        let (writer, guard) = tracing_appender::non_blocking(appender);
        crate::vault::mask::global().register("log-test-secret", "s3cr3t-log-test-value");
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(MaskedMakeWriter(writer))
                .with_filter(EnvFilter::new("info")),
        );
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(kind = "credentials_locked", "provider_notice");
            tracing::warn!("vault: access refused s3cr3t-log-test-value");
            tracing::debug!("below the file level");
        });
        drop(guard); // flush
        let file = std::fs::read_dir(dir.path())
            .unwrap()
            .next()
            .expect("a log file exists")
            .unwrap()
            .path();
        let name = file.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            name.starts_with(FILE_PREFIX) && name.ends_with(FILE_SUFFIX),
            "{name}"
        );
        let text = std::fs::read_to_string(file).unwrap();
        assert!(text.contains("provider_notice") && text.contains("credentials_locked"));
        assert!(text.contains("vault: access refused"));
        assert!(
            !text.contains("s3cr3t-log-test-value"),
            "secret leaked: {text}"
        );
        assert!(!text.contains("below the file level"));
        crate::vault::mask::global().forget("log-test-secret");
    }
}
