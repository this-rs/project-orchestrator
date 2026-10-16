//! RL3 — the provider profiles: pure DATA, not code.
//!
//! The classifier ([`super::classify`]) owns the *rules*; this file owns the
//! *facts*. Adding a provider means adding a row to [`PROFILES`], never a branch
//! in an algorithm: the day a capture contradicts the table, the table is edited
//! and its tests with it.
//!
//! # What a profile is, and is not
//!
//! A profile says which headers an endpoint family writes, and which error codes
//! it uses for "your key/quota/balance is spent" as opposed to "I am overloaded".
//! It says NOTHING about whether our own configured limit was reached: that is
//! [`super::classify::OurStanding`], measured by RL2's ledger.
//!
//! # Evidence, and why every row carries it
//!
//! A profile that has never been checked against a real refusal is a guess. The
//! [`Evidence`] field says which it is, and a guessed profile says so in
//! [`ProviderProfile::evidence_note`] — never in a comment nobody reads at the
//! call site. NVIDIA NIM is `Unproven` (RL9 has not captured a real NIM refusal
//! yet); DeepSeek is `Unproven` for the same reason; the generic
//! OpenAI-compatible vocabulary is `Documented` because that family publishes its
//! error codes.

use std::time::Duration;

/// The generic preset: the OpenAI-compatible error vocabulary, used whenever no
/// more specific row matches.
pub const OPENAI_COMPATIBLE: &str = "openai_compatible";
/// NVIDIA NIM (`integrate.api.nvidia.com`).
pub const NIM: &str = "nim";
/// DeepSeek (`api.deepseek.com`).
pub const DEEPSEEK: &str = "deepseek";

/// How solid a profile is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    /// A real refusal from this endpoint family was captured and the profile
    /// matches it.
    Proven,
    /// The family publishes its error codes; the profile follows them, but no
    /// refusal of ours was captured against them.
    Documented,
    /// A guess. Nothing here may be presented as a fact until a capture says so.
    Unproven,
}

/// One endpoint family: which headers it writes, which codes it uses.
#[derive(Debug, Clone, Copy)]
pub struct ProviderProfile {
    /// Preset name ([`OPENAI_COMPATIBLE`], [`NIM`], [`DEEPSEEK`]).
    pub preset: &'static str,
    /// How solid this row is.
    pub evidence: Evidence,
    /// One line a human can read: where the row comes from, or what is missing.
    pub evidence_note: &'static str,
    /// Header name prefixes carrying "how much is left". A name matches when it
    /// equals the prefix or extends it with `-<variant>`
    /// (`x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`).
    pub remaining_prefixes: &'static [&'static str],
    /// Header name prefixes carrying "when it resets".
    pub reset_prefixes: &'static [&'static str],
    /// Error codes that mean the key, the quota or the balance is spent.
    pub quota_codes: &'static [&'static str],
    /// Error codes that mean the provider itself is overloaded.
    pub overload_codes: &'static [&'static str],
    /// Backoff applied when the provider refuses without saying for how long.
    pub default_backoff: Duration,
}

/// The table. First row wins for its preset; [`profile`] falls back to the
/// generic row for anything unknown.
pub const PROFILES: &[ProviderProfile] = &[
    ProviderProfile {
        preset: OPENAI_COMPATIBLE,
        evidence: Evidence::Documented,
        evidence_note:
            "the OpenAI-compatible error vocabulary (insufficient_quota, rate_limit_exceeded, \
             server_error, overloaded) is published by that family",
        remaining_prefixes: &["x-ratelimit-remaining", "ratelimit-remaining"],
        reset_prefixes: &[
            "x-ratelimit-reset",
            "ratelimit-reset",
            "x-ratelimit-reset-after",
        ],
        quota_codes: &[
            "insufficient_quota",
            "quota_exceeded",
            "billing_hard_limit_reached",
            "insufficient_balance",
            "insufficient_funds",
            "rate_limit_exceeded",
            "rate_limit_reached",
            "too_many_requests",
        ],
        overload_codes: &[
            "overloaded",
            "overloaded_error",
            "engine_overloaded",
            "server_overloaded",
            "server_error",
            "service_unavailable",
        ],
        default_backoff: Duration::from_secs(5),
    },
    ProviderProfile {
        preset: DEEPSEEK,
        evidence: Evidence::Unproven,
        evidence_note:
            "RL9 (a real capture) must confirm DeepSeek's refusal vocabulary: today the row is the \
             generic one plus `insufficient_balance`; it is not a measured fact",
        remaining_prefixes: &["x-ratelimit-remaining", "ratelimit-remaining"],
        reset_prefixes: &["x-ratelimit-reset", "ratelimit-reset"],
        quota_codes: &[
            "insufficient_balance",
            "insufficient_quota",
            "quota_exceeded",
            "rate_limit_reached",
            "rate_limit_exceeded",
            "too_many_requests",
        ],
        overload_codes: &[
            "engine_overloaded",
            "server_overloaded",
            "overloaded",
            "service_unavailable",
        ],
        default_backoff: Duration::from_secs(10),
    },
    ProviderProfile {
        preset: NIM,
        evidence: Evidence::Unproven,
        evidence_note:
            "RL9 must capture a real NVIDIA NIM refusal before any of this is claimed: headers and \
             codes below are the generic OpenAI-compatible ones, NOT a measurement",
        remaining_prefixes: &["x-ratelimit-remaining", "ratelimit-remaining"],
        reset_prefixes: &["x-ratelimit-reset", "ratelimit-reset"],
        quota_codes: &[
            "insufficient_quota",
            "quota_exceeded",
            "billing_hard_limit_reached",
            "insufficient_balance",
            "rate_limit_exceeded",
            "too_many_requests",
        ],
        overload_codes: &[
            "overloaded",
            "engine_overloaded",
            "server_overloaded",
            "server_error",
            "service_unavailable",
        ],
        default_backoff: Duration::from_secs(20),
    },
];

/// The row of a preset, falling back to the generic one. Never panics: an
/// unknown preset is classified, not refused.
pub fn profile(preset: &str) -> &'static ProviderProfile {
    PROFILES
        .iter()
        .find(|p| p.preset.eq_ignore_ascii_case(preset.trim()))
        .unwrap_or(&PROFILES[0])
}

/// The preset an endpoint family belongs to, derived from the instance's base
/// URL. A host we do not know is the generic one: the fallback is a preset, not
/// an error.
pub fn preset_for_base_url(base_url: &str) -> &'static str {
    let host = host_of(base_url);
    if host_is(&host, "nvidia.com") {
        NIM
    } else if host_is(&host, "deepseek.com") {
        DEEPSEEK
    } else {
        OPENAI_COMPATIBLE
    }
}

/// Whether a host is a domain or one of its subdomains (`api.deepseek.com` is
/// `deepseek.com`, `evil-deepseek.com` is not).
fn host_is(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// The host of a URL, lowercased, without scheme, user, port, path or query.
fn host_of(url: &str) -> String {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .rsplit('@')
        .next()
        .unwrap_or("");
    authority
        .split(':')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_has_exactly_one_row_and_the_generic_one_is_first() {
        assert_eq!(PROFILES[0].preset, OPENAI_COMPATIBLE);
        for p in PROFILES {
            let rows = PROFILES.iter().filter(|o| o.preset == p.preset).count();
            assert_eq!(rows, 1, "{} has {rows} rows", p.preset);
        }
        for p in PROFILES {
            assert!(!p.remaining_prefixes.is_empty(), "{}", p.preset);
            assert!(!p.reset_prefixes.is_empty(), "{}", p.preset);
            assert!(!p.quota_codes.is_empty(), "{}", p.preset);
            assert!(!p.overload_codes.is_empty(), "{}", p.preset);
            assert!(p.default_backoff > Duration::ZERO, "{}", p.preset);
            for prefix in p.remaining_prefixes.iter().chain(p.reset_prefixes) {
                assert_eq!(
                    *prefix,
                    prefix.to_ascii_lowercase(),
                    "a header prefix is compared lowercased: {prefix}"
                );
            }
            for code in p.quota_codes.iter().chain(p.overload_codes) {
                assert_eq!(
                    *code,
                    code.to_ascii_lowercase(),
                    "a code is compared lowercased"
                );
            }
            // A code may not mean both: the table would contradict itself.
            for code in p.quota_codes {
                assert!(
                    !p.overload_codes.contains(code),
                    "{}: '{code}' is both a quota and an overload code",
                    p.preset
                );
            }
        }
    }

    #[test]
    fn nim_and_deepseek_say_they_are_unproven_and_the_generic_one_says_what_it_rests_on() {
        // RL9 is not done: the NIM profile MUST say so, at the call site.
        assert_eq!(profile(NIM).evidence, Evidence::Unproven);
        assert_eq!(profile(DEEPSEEK).evidence, Evidence::Unproven);
        assert_eq!(profile(OPENAI_COMPATIBLE).evidence, Evidence::Documented);
        assert!(
            profile(NIM).evidence_note.contains("RL9"),
            "{}",
            profile(NIM).evidence_note
        );
        assert!(
            profile(DEEPSEEK)
                .evidence_note
                .to_lowercase()
                .contains("unproven")
                || profile(DEEPSEEK).evidence_note.contains("RL9")
        );
        assert!(profile(OPENAI_COMPATIBLE)
            .evidence_note
            .contains("published"));
    }

    #[test]
    fn an_unknown_preset_falls_back_to_the_generic_row_never_to_a_panic() {
        assert_eq!(profile("").preset, OPENAI_COMPATIBLE);
        assert_eq!(profile("  NIM  ").preset, NIM);
        assert_eq!(profile("DEEPSEEK").preset, DEEPSEEK);
        assert_eq!(profile("something-else").preset, OPENAI_COMPATIBLE);
    }

    #[test]
    fn the_preset_of_a_base_url_is_the_endpoint_family_and_never_a_lookalike() {
        assert_eq!(
            preset_for_base_url("https://integrate.api.nvidia.com/v1"),
            NIM
        );
        assert_eq!(preset_for_base_url("https://api.deepseek.com/v1"), DEEPSEEK);
        assert_eq!(
            preset_for_base_url("http://localhost:8000/v1"),
            OPENAI_COMPATIBLE
        );
        assert_eq!(
            preset_for_base_url("https://api.openai.com/v1"),
            OPENAI_COMPATIBLE
        );
        assert_eq!(preset_for_base_url(""), OPENAI_COMPATIBLE);
        // A lookalike domain is NOT the family it imitates.
        assert_eq!(
            preset_for_base_url("https://evil-deepseek.com/v1"),
            OPENAI_COMPATIBLE
        );
        assert_eq!(
            preset_for_base_url("https://deepseek.com.evil.tld/v1"),
            OPENAI_COMPATIBLE
        );
        // Scheme, credentials, port, path and query are not the host.
        assert_eq!(preset_for_base_url("api.deepseek.com"), DEEPSEEK);
        assert_eq!(
            preset_for_base_url("https://user:pw@api.deepseek.com:443/v1?x=1"),
            DEEPSEEK
        );
        assert_eq!(preset_for_base_url("HTTPS://API.DEEPSEEK.COM/v1"), DEEPSEEK);
    }

    #[test]
    fn the_proven_grade_exists_and_is_the_one_reserved_for_a_capture() {
        // Kept as a value on purpose: the day a capture proves a row, the row is
        // edited and this grade is what it takes.
        assert_ne!(Evidence::Proven, Evidence::Unproven);
        assert_eq!(format!("{:?}", Evidence::Proven), "Proven");
    }
}
