//! User PATH detection via login shell.
//!
//! Detects the user's full PATH by spawning their login shell and reading `$PATH`.
//! This is useful when the application is launched from macOS Dock or a desktop
//! environment where the inherited PATH is minimal (`/usr/bin:/bin`).

use tracing::{debug, warn};

/// Detect the user's full PATH by spawning a login shell (Unix) or reading
/// the environment variable directly (Windows).
///
/// On Unix, launches the user's `$SHELL` (or falls back to `/bin/zsh` → `/bin/bash` → `/bin/sh`)
/// in login mode (`-l`) and reads the `PATH` environment variable.
///
/// On Windows, simply reads `%PATH%` from the environment (always fully populated).
///
/// Returns `None` on error, timeout (5s), or empty/invalid result.
/// Never panics.
#[cfg(unix)]
pub async fn detect_user_path() -> Option<String> {
    let shell = find_shell();
    debug!(shell = %shell, "Detecting user PATH via login shell");

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::process::Command::new(&shell)
            .args(["-l", "-c", "echo $PATH"])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output(),
    )
    .await
    .ok()? // timeout
    .ok()?; // spawn/io error

    if !output.status.success() {
        warn!(
            shell = %shell,
            exit_code = ?output.status.code(),
            "Login shell exited with non-zero status"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Some .zshrc/.bashrc files print text before the PATH.
    // Take the last non-empty line as the PATH value.
    let path = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
        .unwrap_or_default();

    if path.is_empty() {
        warn!(shell = %shell, "Login shell returned empty PATH");
        return None;
    }

    // Basic validation: a valid PATH should contain at least one `/`
    if !path.contains('/') {
        warn!(
            shell = %shell,
            value = %path,
            "Login shell output doesn't look like a PATH (no '/' found)"
        );
        return None;
    }

    debug!(path = %path, "Detected user PATH");
    Some(path)
}

/// Detect the user's full PATH from the environment (Windows).
///
/// On Windows the inherited `%PATH%` is always fully populated (no login-shell
/// dance required). Returns `None` only if `%PATH%` is unset or empty.
#[cfg(windows)]
pub async fn detect_user_path() -> Option<String> {
    let path = std::env::var("PATH").ok().filter(|p| !p.is_empty())?;

    // Basic validation: a valid Windows PATH should contain `\` or `:` (drive letters)
    if !path.contains('\\') && !path.contains(':') {
        warn!(
            value = %path,
            "PATH doesn't look like a Windows PATH (no '\\' or ':' found)"
        );
        return None;
    }

    debug!(path = %path, "Detected user PATH from environment");
    Some(path)
}

/// Find the user's shell, with fallbacks.
#[cfg(unix)]
fn find_shell() -> String {
    if let Ok(shell) = std::env::var("SHELL") {
        if !shell.is_empty() && std::path::Path::new(&shell).exists() {
            return shell;
        }
    }

    // Fallback chain
    for candidate in &["/bin/zsh", "/bin/bash", "/bin/sh"] {
        if std::path::Path::new(candidate).exists() {
            return (*candidate).to_string();
        }
    }

    // Last resort
    "/bin/sh".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn test_find_shell_returns_existing_path() {
        let shell = find_shell();
        assert!(
            std::path::Path::new(&shell).exists(),
            "find_shell() returned non-existent path: {}",
            shell
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_detect_user_path_returns_something() {
        let path = detect_user_path().await;
        assert!(
            path.is_some(),
            "detect_user_path() should return Some on a standard Unix system"
        );
        let path = path.unwrap();
        assert!(
            path.contains("/usr/bin"),
            "PATH should contain /usr/bin, got: {}",
            path
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_detect_user_path_validates_format() {
        let path = detect_user_path().await.expect("should detect PATH");
        // A valid PATH contains at least one `/` and one `:`
        assert!(path.contains('/'), "PATH should contain '/'");
        assert!(path.contains(':'), "PATH should contain ':'");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_detect_user_path_no_empty_segments() {
        let path = detect_user_path().await.expect("should detect PATH");
        // Check for doubled colons which indicate empty segments
        let empty_segments: Vec<&str> = path.split(':').filter(|s| s.is_empty()).collect();
        assert!(
            empty_segments.is_empty(),
            "PATH should not have empty segments (::), got: {}",
            path
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn test_detect_user_path_windows() {
        let path = detect_user_path().await;
        assert!(
            path.is_some(),
            "detect_user_path() should return Some on Windows"
        );
        let path = path.unwrap();
        assert!(
            path.contains("Windows") || path.contains("System32") || path.contains("\\"),
            "PATH should contain Windows paths, got: {}",
            path
        );
    }
}
