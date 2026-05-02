//! First-party statistical primitives — replaces `rs-stats` (GPL-3.0).
//!
//! See `docs/migration/rs-stats/` for the full audit + migration plan.
//! Plan id: `00f0ca9a-816f-4fcc-bc53-da88d595de34`.
//!
//! ## Conventions
//!
//! - `mean_std::std_dev_population(n)` — denominator `n` (matches `rs_stats::prob::std_dev`)
//! - `t_test::*_t_test(n-1)` — Bessel-corrected sample std (matches rs-stats internal `calculate_variance`)
//!
//! See `docs/migration/rs-stats/audit-variance-semantic.md` for why both exist.

pub mod anova;
pub mod fitting;
pub mod golden_fixtures;
pub mod mean_std;
pub mod t_test;
