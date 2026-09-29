//! Version identity of the running build and comparison against releases.
//!
//! Release builds get their version injected from the git tag by CI
//! (`.github/workflows/release.yml` rewrites `Cargo.toml`), so
//! `CARGO_PKG_VERSION` is e.g. `0.0.15`. Builds from a source checkout keep the
//! `Cargo.toml` placeholder (`0.1.0`), which is *higher* than every published
//! tag (`v0.0.x`) — comparing it against GitHub releases can never report an
//! update. For those builds we use `git describe --tags` (captured by
//! `build.rs` as `PO_GIT_DESCRIBE`) to find the release the checkout is based
//! on, e.g. `v0.0.15-17-ge15ceda` → base `0.0.15`, 17 commits ahead.

use semver::Version;
use serde::Serialize;
use std::cmp::Ordering;

/// Version string baked in by Cargo (`Cargo.toml` `[package] version`).
pub const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

/// `git describe --tags --long --match 'v[0-9]*'` at build time ("" when git
/// was unavailable, e.g. Docker builds without `.git`).
pub const GIT_DESCRIBE: &str = env!("PO_GIT_DESCRIBE");

/// Parse a release tag or version (`v0.0.15`, `0.0.15`, `v1.2.0-rc.1`).
pub fn parse_version(raw: &str) -> Option<Version> {
    let trimmed = raw.trim();
    let v = trimmed
        .strip_prefix('v')
        .or_else(|| trimmed.strip_prefix('V'))
        .unwrap_or(trimmed);
    Version::parse(v).ok()
}

/// `latest` is strictly newer than `current` by SemVer precedence
/// (pre-releases sort before the release; build metadata is ignored).
pub fn is_newer(current: &Version, latest: &Version) -> bool {
    latest.cmp_precedence(current) == Ordering::Greater
}

/// What the running binary is, for update comparison and display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildVersion {
    /// Release the build corresponds to (or is based on, for source builds).
    #[serde(serialize_with = "ser_version")]
    pub base: Version,
    /// Commits on top of `base` (0 for a release build).
    pub commits_ahead: u32,
    /// Abbreviated commit hash when known from `git describe`.
    pub commit: Option<String>,
}

fn ser_version<S: serde::Serializer>(v: &Version, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&v.to_string())
}

impl BuildVersion {
    /// Human-readable version, e.g. `0.0.15` or `0.0.15+17 (e15ceda)`.
    pub fn display(&self) -> String {
        match (self.commits_ahead, &self.commit) {
            (0, _) => self.base.to_string(),
            (n, Some(c)) => format!("{}+{} ({})", self.base, n, c),
            (n, None) => format!("{}+{}", self.base, n),
        }
    }

    /// Is `latest` a newer release than this build?
    ///
    /// A source build N commits past `v0.0.15` is *not* behind `v0.0.15`;
    /// it is behind `v0.0.16`.
    pub fn is_behind(&self, latest: &Version) -> bool {
        is_newer(&self.base, latest)
    }
}

/// Parse `git describe --long` output: `<tag>-<n>-g<sha>`.
fn parse_describe(describe: &str) -> Option<(Version, u32, String)> {
    let d = describe.trim();
    let d = d.strip_suffix("-dirty").unwrap_or(d);
    // rsplitn keeps pre-release dashes inside the tag (`v1.0.0-rc.1-3-gabc`).
    let mut parts = d.rsplitn(3, '-');
    let sha = parts.next()?.strip_prefix('g')?;
    let n: u32 = parts.next()?.parse().ok()?;
    let tag = parts.next()?;
    Some((parse_version(tag)?, n, sha.to_string()))
}

/// Resolve the build version from the Cargo package version and the
/// build-time `git describe`. `git describe` wins when available: in CI it
/// matches the injected tag version, and for source builds it is the only
/// meaningful signal (the package version is a placeholder).
pub fn resolve_build_version(pkg_version: &str, git_describe: &str) -> BuildVersion {
    if let Some((base, commits_ahead, sha)) = parse_describe(git_describe) {
        return BuildVersion {
            base,
            commits_ahead,
            commit: Some(sha),
        };
    }
    BuildVersion {
        base: parse_version(pkg_version).unwrap_or_else(|| Version::new(0, 0, 0)),
        commits_ahead: 0,
        commit: None,
    }
}

/// Build version of the running binary.
pub fn current_build() -> BuildVersion {
    resolve_build_version(PKG_VERSION, GIT_DESCRIBE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        parse_version(s).unwrap()
    }

    #[test]
    fn parses_repo_tag_formats() {
        // Tags published by this repo: `v0.0.1` … `v0.0.15`.
        assert_eq!(v("v0.0.15"), Version::new(0, 0, 15));
        assert_eq!(v("0.0.15"), Version::new(0, 0, 15));
        assert_eq!(v(" V1.2.3 "), Version::new(1, 2, 3));
        assert!(parse_version("v1.2").is_none());
        assert!(parse_version("latest").is_none());
        assert!(parse_version("").is_none());
    }

    #[test]
    fn compares_by_semver_precedence() {
        assert!(is_newer(&v("0.0.9"), &v("0.0.10"))); // numeric, not lexical
        assert!(is_newer(&v("0.0.15"), &v("0.1.0")));
        assert!(!is_newer(&v("0.0.15"), &v("0.0.15")));
        assert!(!is_newer(&v("0.0.16"), &v("0.0.15")));
        // Pre-releases sort before the final release.
        assert!(is_newer(&v("1.0.0-rc.1"), &v("1.0.0")));
        assert!(!is_newer(&v("1.0.0"), &v("1.0.0-rc.2")));
        assert!(is_newer(&v("1.0.0-rc.1"), &v("1.0.0-rc.2")));
        // Build metadata is ignored.
        assert!(!is_newer(&v("1.0.0+abc"), &v("1.0.0+def")));
    }

    #[test]
    fn release_build_uses_describe_or_pkg_version() {
        let b = resolve_build_version("0.0.15", "v0.0.15-0-gabc1234");
        assert_eq!(b.base, Version::new(0, 0, 15));
        assert_eq!(b.commits_ahead, 0);
        assert_eq!(b.display(), "0.0.15");

        // CI shallow checkout: no tags → describe empty → injected pkg version.
        let b = resolve_build_version("0.0.15", "");
        assert_eq!(b.base, Version::new(0, 0, 15));
        assert_eq!(b.commit, None);
    }

    #[test]
    fn source_build_placeholder_is_not_treated_as_newer_than_releases() {
        // Root cause of "no notification ever": Cargo.toml says 0.1.0, releases are 0.0.x.
        let b = resolve_build_version("0.1.0", "v0.0.15-17-ge15ceda");
        assert_eq!(b.base, Version::new(0, 0, 15));
        assert_eq!(b.commits_ahead, 17);
        assert_eq!(b.display(), "0.0.15+17 (e15ceda)");
        assert!(!b.is_behind(&v("0.0.15")));
        assert!(b.is_behind(&v("0.0.16")));
    }

    #[test]
    fn describe_with_prerelease_tag_and_dirty_suffix() {
        let b = resolve_build_version("0.1.0", "v1.0.0-rc.1-3-gdeadbee-dirty");
        assert_eq!(b.base, v("1.0.0-rc.1"));
        assert_eq!(b.commits_ahead, 3);
        assert_eq!(b.commit.as_deref(), Some("deadbee"));
        assert!(b.is_behind(&v("1.0.0")));
    }

    #[test]
    fn garbage_describe_falls_back_to_pkg_version() {
        let b = resolve_build_version("0.0.15", "e15ceda");
        assert_eq!(b.base, Version::new(0, 0, 15));
        let b = resolve_build_version("not-a-version", "");
        assert_eq!(b.base, Version::new(0, 0, 0));
    }
}
