//! Witness — executable proof anchor for assertive Notes and Decisions.
//!
//! Implements the R\* manifesto's "no claim without witness" discipline (Rule 4.3,
//! RFC `19b89465-6e92-402b-a221-8e9d8e3a3982`). Each assertion in the graph must
//! either point to:
//!
//! - A [`WitnessKind::Proof`] — a formal proof or a piece of source code that
//!   establishes the claim (e.g. a test that asserts the property, a referenced
//!   theorem with hash).
//! - A [`WitnessKind::Certificate`] — an executable witness that can be replayed
//!   deterministically to verify the cited output (a recorded trajectory or a
//!   reproducible script).
//! - [`WitnessKind::Open`] — explicitly marked as unverified hypothesis.
//!
//! Validation is enforced at write-time by [`Witness::validate`]. v1 of the RFC
//! is warn-only at the API boundary; this module returns hard errors so callers
//! can decide whether to log-and-pass or reject.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// ============================================================================
// Witness kind
// ============================================================================

/// Epistemic status of an assertion (mapped from the R\* manifesto tags).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum WitnessKind {
    /// Demonstrated — formal proof or source code that establishes the claim.
    Proof,
    /// Certified — executable witness (trajectory or code) replayable on demand.
    Certificate,
    /// Open — explicitly unverified hypothesis. Carries no ref.
    Open,
}

impl fmt::Display for WitnessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proof => write!(f, "proof"),
            Self::Certificate => write!(f, "certificate"),
            Self::Open => write!(f, "open"),
        }
    }
}

impl std::str::FromStr for WitnessKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "proof" => Ok(Self::Proof),
            "certificate" => Ok(Self::Certificate),
            "open" => Ok(Self::Open),
            other => Err(format!("Unknown witness kind: {other}")),
        }
    }
}

// ============================================================================
// Reference types
// ============================================================================

/// Reference to a specific region of a commit.
///
/// Used by [`WitnessKind::Proof`] (e.g. a test that asserts the property) or by
/// [`WitnessKind::Certificate`] (e.g. a script that produces the cited output).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CodeRef {
    /// Full git SHA. We do not accept short SHAs to keep verification
    /// deterministic across rebases.
    pub commit_sha: String,
    /// Path relative to the repository root.
    pub file_path: String,
    /// Optional line range `(start, end)` inclusive, 1-indexed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_range: Option<(u32, u32)>,
}

/// Reference to an external document or page, anchored by content hash so that
/// drift is detectable.
///
/// The `content_hash` is mandatory: an `external_ref` without a hash is
/// rejected at validation time, regardless of the v1 warn-only API mode.
/// Rationale: a URL whose content silently changes is the worst possible
/// witness — its drift cannot be detected by replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ExternalRef {
    pub url: String,
    /// SHA-256 hex digest of the canonical fetched content at witness creation.
    pub content_hash: String,
}

// ============================================================================
// Witness
// ============================================================================

/// An assertion's epistemic anchor.
///
/// Invariants enforced by [`Witness::validate`]:
///
/// | kind          | code_ref | trajectory_ref | external_ref |
/// |---------------|----------|----------------|--------------|
/// | `Proof`       | optional | forbidden      | optional     | (at least one of code/external required)
/// | `Certificate` | optional | optional       | forbidden    | (at least one of code/trajectory required)
/// | `Open`        | forbidden| forbidden      | forbidden    |
///
/// `Open` is the only kind that carries no ref. It explicitly documents that
/// the claim is an unverified hypothesis — not a missing witness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Witness {
    pub kind: WitnessKind,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_ref: Option<CodeRef>,

    /// UUID of a stored [`Trajectory`](crate::trajectory) that, when replayed,
    /// produces the cited output. Only valid for [`WitnessKind::Certificate`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_ref: Option<Uuid>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<ExternalRef>,
}

/// Validation outcome for a witness.
#[derive(Debug, thiserror::Error)]
pub enum WitnessValidationError {
    #[error("Proof witness must carry at least one of code_ref or external_ref")]
    ProofMissingRef,

    #[error("Certificate witness must carry at least one of code_ref or trajectory_ref")]
    CertificateMissingRef,

    #[error("Open witness must not carry any ref (got {0})")]
    OpenWithRef(&'static str),

    #[error(
        "Proof witness must not carry trajectory_ref (trajectories are certificates, not proofs)"
    )]
    ProofWithTrajectory,

    #[error("Certificate witness must not carry external_ref (external pages are not replayable)")]
    CertificateWithExternal,

    #[error("external_ref must include a non-empty content_hash (SHA-256 hex) to detect drift")]
    ExternalRefMissingHash,

    #[error("code_ref.commit_sha must be a full 40-character git SHA (got length {0})")]
    CodeRefShortSha(usize),
}

impl Witness {
    /// Construct an Open witness — the only no-ref variant.
    pub fn open() -> Self {
        Self {
            kind: WitnessKind::Open,
            code_ref: None,
            trajectory_ref: None,
            external_ref: None,
        }
    }

    /// Construct a Proof witness backed by a `CodeRef`.
    pub fn proof_from_code(code_ref: CodeRef) -> Self {
        Self {
            kind: WitnessKind::Proof,
            code_ref: Some(code_ref),
            trajectory_ref: None,
            external_ref: None,
        }
    }

    /// Construct a Certificate witness backed by a stored trajectory.
    pub fn certificate_from_trajectory(trajectory_id: Uuid) -> Self {
        Self {
            kind: WitnessKind::Certificate,
            code_ref: None,
            trajectory_ref: Some(trajectory_id),
            external_ref: None,
        }
    }

    /// Validate the invariants of the (`kind`, refs) tuple.
    ///
    /// The content_hash check on `external_ref` is enforced unconditionally
    /// (see [`WitnessValidationError::ExternalRefMissingHash`]) so that even
    /// v1 warn-only callers cannot accept a hash-less external reference.
    pub fn validate(&self) -> Result<(), WitnessValidationError> {
        // Cross-cutting structural check on the refs themselves, regardless of kind.
        if let Some(ext) = &self.external_ref {
            if ext.content_hash.trim().is_empty() {
                return Err(WitnessValidationError::ExternalRefMissingHash);
            }
        }
        if let Some(code) = &self.code_ref {
            if code.commit_sha.len() != 40 {
                return Err(WitnessValidationError::CodeRefShortSha(
                    code.commit_sha.len(),
                ));
            }
        }

        match self.kind {
            WitnessKind::Proof => {
                if self.trajectory_ref.is_some() {
                    return Err(WitnessValidationError::ProofWithTrajectory);
                }
                if self.code_ref.is_none() && self.external_ref.is_none() {
                    return Err(WitnessValidationError::ProofMissingRef);
                }
                Ok(())
            }
            WitnessKind::Certificate => {
                if self.external_ref.is_some() {
                    return Err(WitnessValidationError::CertificateWithExternal);
                }
                if self.code_ref.is_none() && self.trajectory_ref.is_none() {
                    return Err(WitnessValidationError::CertificateMissingRef);
                }
                Ok(())
            }
            WitnessKind::Open => {
                if self.code_ref.is_some() {
                    return Err(WitnessValidationError::OpenWithRef("code_ref"));
                }
                if self.trajectory_ref.is_some() {
                    return Err(WitnessValidationError::OpenWithRef("trajectory_ref"));
                }
                if self.external_ref.is_some() {
                    return Err(WitnessValidationError::OpenWithRef("external_ref"));
                }
                Ok(())
            }
        }
    }

    /// Whether this witness can in principle be re-executed to produce its
    /// cited output (i.e. replayable [C] in manifesto terms).
    pub fn is_replayable(&self) -> bool {
        matches!(self.kind, WitnessKind::Certificate)
            && (self.code_ref.is_some() || self.trajectory_ref.is_some())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_code_ref() -> CodeRef {
        CodeRef {
            commit_sha: "a".repeat(40),
            file_path: "src/example.rs".to_string(),
            line_range: Some((10, 42)),
        }
    }

    fn dummy_external_ref() -> ExternalRef {
        ExternalRef {
            url: "https://example.org/spec".to_string(),
            content_hash: "deadbeef".repeat(8), // 64 hex chars, plausible SHA-256
        }
    }

    #[test]
    fn proof_with_code_ref_ok() {
        let w = Witness::proof_from_code(dummy_code_ref());
        assert!(w.validate().is_ok());
        assert!(
            !w.is_replayable(),
            "Proof is not replayable as a Certificate is"
        );
    }

    #[test]
    fn proof_with_external_ref_ok() {
        let w = Witness {
            kind: WitnessKind::Proof,
            code_ref: None,
            trajectory_ref: None,
            external_ref: Some(dummy_external_ref()),
        };
        assert!(w.validate().is_ok());
    }

    #[test]
    fn proof_without_any_ref_fails() {
        let w = Witness {
            kind: WitnessKind::Proof,
            code_ref: None,
            trajectory_ref: None,
            external_ref: None,
        };
        assert!(matches!(
            w.validate(),
            Err(WitnessValidationError::ProofMissingRef)
        ));
    }

    #[test]
    fn certificate_with_trajectory_ok_and_replayable() {
        let w = Witness::certificate_from_trajectory(Uuid::new_v4());
        assert!(w.validate().is_ok());
        assert!(w.is_replayable());
    }

    #[test]
    fn open_with_ref_fails() {
        let w = Witness {
            kind: WitnessKind::Open,
            code_ref: Some(dummy_code_ref()),
            trajectory_ref: None,
            external_ref: None,
        };
        assert!(matches!(
            w.validate(),
            Err(WitnessValidationError::OpenWithRef("code_ref"))
        ));
    }

    #[test]
    fn external_ref_without_hash_fails_even_when_kind_would_accept() {
        // Even though Proof accepts external_ref, the structural check on
        // content_hash fires unconditionally (security constraint of the RFC).
        let w = Witness {
            kind: WitnessKind::Proof,
            code_ref: None,
            trajectory_ref: None,
            external_ref: Some(ExternalRef {
                url: "https://example.org/spec".to_string(),
                content_hash: "   ".to_string(),
            }),
        };
        assert!(matches!(
            w.validate(),
            Err(WitnessValidationError::ExternalRefMissingHash)
        ));
    }

    #[test]
    fn proof_with_trajectory_fails() {
        let w = Witness {
            kind: WitnessKind::Proof,
            code_ref: Some(dummy_code_ref()),
            trajectory_ref: Some(Uuid::new_v4()),
            external_ref: None,
        };
        assert!(matches!(
            w.validate(),
            Err(WitnessValidationError::ProofWithTrajectory)
        ));
    }

    #[test]
    fn certificate_with_external_fails() {
        let w = Witness {
            kind: WitnessKind::Certificate,
            code_ref: None,
            trajectory_ref: Some(Uuid::new_v4()),
            external_ref: Some(dummy_external_ref()),
        };
        assert!(matches!(
            w.validate(),
            Err(WitnessValidationError::CertificateWithExternal)
        ));
    }

    #[test]
    fn code_ref_short_sha_fails() {
        let w = Witness {
            kind: WitnessKind::Proof,
            code_ref: Some(CodeRef {
                commit_sha: "abc1234".to_string(),
                file_path: "src/example.rs".to_string(),
                line_range: None,
            }),
            trajectory_ref: None,
            external_ref: None,
        };
        assert!(matches!(
            w.validate(),
            Err(WitnessValidationError::CodeRefShortSha(7))
        ));
    }

    #[test]
    fn witness_kind_serde_snake_case() {
        let w = Witness::open();
        let json = serde_json::to_string(&w).unwrap();
        assert!(json.contains("\"kind\":\"open\""));

        let back: Witness = serde_json::from_str(&json).unwrap();
        assert_eq!(back, w);
    }

    #[test]
    fn witness_kind_from_str_roundtrip() {
        for kind in [
            WitnessKind::Proof,
            WitnessKind::Certificate,
            WitnessKind::Open,
        ] {
            let parsed: WitnessKind = kind.to_string().parse().unwrap();
            assert_eq!(parsed, kind);
        }
        assert!("bogus".parse::<WitnessKind>().is_err());
    }
}
