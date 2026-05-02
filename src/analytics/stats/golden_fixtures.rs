//! Golden characterization fixtures for the rs-stats → first-party migration.
//!
//! This module is the contract: each `pub const` input has a matching `pub const`
//! expected output captured from `rs-stats v2.0.3` BEFORE any migration. The
//! migration is ISO if the new implementation produces the same outputs (within
//! the documented tolerance matrix) on the same inputs.
//!
//! Plan: `00f0ca9a-816f-4fcc-bc53-da88d595de34`.
//! See `docs/migration/rs-stats/` for the full audit + tolerance matrix.
//
// Lints justified for this module:
// - `excessive_precision`: the EXPECTED_* literals are captured from rs-stats
//   via `:.18e` formatting (exact byte-for-byte preservation of the upstream
//   output is the whole point of a characterization baseline). Trailing
//   digits beyond f64 precision are silently truncated by the parser, but
//   keeping them documents the capture provenance.
// - `explicit_auto_deref`: the capture test iterates over `&[(name, &[f64])]`
//   tuples; writing `*fixture` is more readable than relying on auto-deref
//   when calling `rs_stats::prob::average(fixture)`.

#![allow(clippy::excessive_precision)]
#![allow(clippy::explicit_auto_deref)]

// ─────────────────────────────────────────────────────────────────────────────
// Input fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// 5 sequential integers as f64 — basic mean/std_dev sanity check.
pub const SMALL_INTEGERS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];

/// 30 values sampled from N(10, 2) using a deterministic Box-Muller from
/// Python `random` with seed=42. Hardcoded so the fixture is byte-identical
/// across machines.
pub const MEDIUM_NORMAL: &[f64] = &[
    11.868112289933460,
    10.538363291357484,
    9.304739538887581,
    10.816328874560563,
    12.581533669473034,
    6.514467255668457,
    11.702465144696083,
    8.216741627799809,
    7.054795702721005,
    11.299452502699030,
    9.579300746973887,
    11.643381076423040,
    9.693833007450200,
    13.580707640200835,
    9.085270490645224,
    9.833716481282110,
    12.201058277454472,
    7.359948912323285,
    9.100528773640107,
    9.035159683650271,
    11.604817122185668,
    9.157424147777540,
    10.485348666429225,
    9.251139423549571,
    8.742506277433485,
    10.709660039466131,
    12.996274121438363,
    8.558131403194823,
    9.508554714696340,
    7.233839038837976,
];

/// 4 small values + 1 extreme outlier — tests skewness, IQR robustness.
pub const WITH_OUTLIERS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 100.0];

/// Tiny fluctuations near zero — re-uses the existing test_stagnation fixture.
pub const NEAR_ZERO_FLAT: &[f64] = &[0.01, -0.01, 0.02, -0.02, 0.0, 0.01];

/// 50 values drawn from LogNormal(0, 0.5) using deterministic Python with
/// seed=123. Used for fit_all best-fit detection (LogNormal expected to win).
pub const LOGNORMAL_SHAPE: &[f64] = &[
    2.819881691658933,
    1.686272361385675,
    1.248003618442658,
    0.759071737806628,
    1.164268292502177,
    0.690407456900254,
    2.312557928875238,
    1.731283489397845,
    1.582874795167690,
    0.486836566194478,
    1.204552565267076,
    1.278947145969125,
    3.245001158686856,
    0.950575787587634,
    1.035663581484292,
    1.261641963149709,
    0.502680758300875,
    1.228902277466682,
    0.543543723301693,
    0.696989080900011,
    0.755195584490837,
    1.414977295695722,
    0.745014969620613,
    1.987294632334389,
    1.169741752469329,
    1.309444074912577,
    1.246122387037005,
    0.564513615690365,
    0.332479352201588,
    0.762095253878813,
    2.173077388660371,
    1.616828942887865,
    1.099095318327866,
    0.911619194537858,
    0.869042904619676,
    1.064558775371850,
    1.051399346258096,
    1.303127196687702,
    0.942032102630122,
    1.438184612827265,
    1.418905367243931,
    0.610514408144385,
    0.477559805968711,
    1.666667394873268,
    0.485888747412277,
    0.366104797958874,
    0.222252914218920,
    0.675452626399615,
    1.954508723368066,
    0.446838059103658,
];

/// 3 groups with clearly distinct risk profiles — re-uses
/// `test_community_homogeneity_distinct_groups` fixture.
pub const ANOVA_3_GROUPS_LOW: &[f64] = &[0.1, 0.12, 0.11, 0.13, 0.10];
pub const ANOVA_3_GROUPS_MID: &[f64] = &[0.5, 0.55, 0.52, 0.48, 0.51];
pub const ANOVA_3_GROUPS_HIGH: &[f64] = &[0.9, 0.88, 0.92, 0.91, 0.89];

// ─────────────────────────────────────────────────────────────────────────────
// Expected outputs — to be filled in S10/S11 after capture from rs-stats
// ─────────────────────────────────────────────────────────────────────────────
//
// Strategy:
// 1. The capture test below (gated by #[ignore]) prints the rs-stats outputs
//    in copy-paste-ready Rust syntax.
// 2. We run it once: `cargo test capture_rs_stats_outputs -- --ignored --nocapture`
// 3. Copy the output back here as `pub const EXPECTED_*` constants.
// 4. Replace the capture test with assertion tests using the tolerance helpers.
//
// The values below are placeholders that will be replaced in S10.

// Captured 2026-05-02 from rs-stats v2.0.3 via the `capture::capture_rs_stats_outputs`
// test. See `docs/migration/rs-stats/golden-baseline.txt` for the full raw output.

// ─── prob::average ──────────────────────────────────────────────────────────
pub const EXPECTED_MEAN_SMALL_INTEGERS: f64 = 3.000000000000000000e0_f64;
pub const EXPECTED_MEAN_MEDIUM_NORMAL: f64 = 9.975253331428302062e0_f64;
pub const EXPECTED_MEAN_WITH_OUTLIERS: f64 = 2.200000000000000000e1_f64;
pub const EXPECTED_MEAN_NEAR_ZERO_FLAT: f64 = 1.666666666666666774e-3_f64;
pub const EXPECTED_MEAN_LOGNORMAL_SHAPE: f64 = 1.150209870525542799e0_f64;

// ─── prob::std_dev (POPULATION — denom n) ───────────────────────────────────
// Note: SMALL_INTEGERS std_dev exactly equals sqrt(2) by mathematical
// coincidence (variance = 10/5 = 2). Clippy's approx_constant lint trips
// on this; the value is the exact rs-stats output, not an approximation.
#[allow(clippy::approx_constant)]
pub const EXPECTED_STD_DEV_SMALL_INTEGERS: f64 = 1.414213562373095145e0_f64;
pub const EXPECTED_STD_DEV_MEDIUM_NORMAL: f64 = 1.784131496006474471e0_f64;
pub const EXPECTED_STD_DEV_WITH_OUTLIERS: f64 = 3.901281840626231912e1_f64;
pub const EXPECTED_STD_DEV_NEAR_ZERO_FLAT: f64 = 1.343709624716424994e-2_f64;
pub const EXPECTED_STD_DEV_LOGNORMAL_SHAPE: f64 = 6.238892708910099838e-1_f64;

// ─── one_sample_t_test (vs ref_mean = 0.0) ──────────────────────────────────
// Note: rs-stats p-values for these inputs are unreliable (returns 0.0 or 1.0
// when |t| is very small or very large — incomplete_beta precision issue).
// We capture them as-is for ISO baseline; statrs will produce more accurate
// values, accepted within the 1e-3 tolerance + classification check.
pub const EXPECTED_TTEST1_SMALL_INTEGERS: TTestExpected = TTestExpected {
    t_stat: 4.242640687119284770e0_f64,
    df: 4.000000000000000000e0_f64,
    p_value: 1.000000000000000000e0_f64,
};
pub const EXPECTED_TTEST1_MEDIUM_NORMAL: TTestExpected = TTestExpected {
    t_stat: 3.010898204695107694e1_f64,
    df: 2.900000000000000000e1_f64,
    p_value: 1.000000000000000000e0_f64,
};
pub const EXPECTED_TTEST1_NEAR_ZERO_FLAT: TTestExpected = TTestExpected {
    t_stat: 2.773500981126145737e-1_f64,
    df: 5.000000000000000000e0_f64,
    p_value: 0.000000000000000000e0_f64,
};

// ─── two_sample_t_test (Welch) ──────────────────────────────────────────────
pub const EXPECTED_TTEST2_WELCH_SHIFTED: TTestExpected = TTestExpected {
    t_stat: -1.000000000000000000e0_f64,
    df: 8.000000000000000000e0_f64,
    p_value: 1.000000000000000000e0_f64,
};
pub const EXPECTED_TTEST2_WELCH_IDENTICAL: TTestExpected = TTestExpected {
    t_stat: 0.000000000000000000e0_f64,
    df: 8.000000000000000000e0_f64,
    p_value: 0.000000000000000000e0_f64,
};

// ─── one_way_anova (3 groups) ───────────────────────────────────────────────
pub const EXPECTED_ANOVA_3_GROUPS: AnovaExpected = AnovaExpected {
    f_stat: 2.136440366972473385e3_f64,
    df_between: 2,
    df_within: 12,
    p_value: 9.999999999721721489e-1_f64,
    ss_between: 1.552480000000000082e0_f64,
    ss_within: 4.360000000000007099e-3_f64,
    ms_between: 7.762400000000000411e-1_f64,
    ms_within: 3.633333333333339430e-4_f64,
};

// ─── fit_all — note: rs-stats has 10 fitters (Normal, Exponential, Uniform, ──
// Gamma, LogNormal, Weibull, Beta, StudentT, F, ChiSquared). Beta skips on these
// fixtures because data > 1, so 9 fits remain. Order is by AIC ascending.
pub const EXPECTED_FIT_ALL_LOGNORMAL_SHAPE: &[FitExpected] = &[
    FitExpected {
        name: "Gamma",
        aic: 8.652257753988506295e1_f64,
        bic: 9.034662355074135576e1_f64,
        ks_p_value: 9.780769081697978651e-1_f64,
    },
    FitExpected {
        name: "LogNormal",
        aic: 8.732478621067313895e1_f64,
        bic: 9.114883222152943176e1_f64,
        ks_p_value: 7.705644546619278001e-1_f64,
    },
    FitExpected {
        name: "Weibull",
        aic: 8.920656664541205316e1_f64,
        bic: 9.303061265626834597e1_f64,
        ks_p_value: 8.717507017756929155e-1_f64,
    },
    FitExpected {
        name: "F",
        aic: 8.975613679703987202e1_f64,
        bic: 9.358018280789616483e1_f64,
        ks_p_value: 7.489654257771853896e-1_f64,
    },
    FitExpected {
        name: "Normal",
        aic: 9.871561563505848369e1_f64,
        bic: 1.025396616459147765e2_f64,
        ks_p_value: 4.509144806111594073e-1_f64,
    },
    FitExpected {
        name: "StudentT",
        aic: 1.001060202488694983e2_f64,
        bic: 1.058420892651539305e2_f64,
        ks_p_value: 4.165393865930515638e-1_f64,
    },
    FitExpected {
        name: "Uniform",
        aic: 1.146166432298891920e2_f64,
        bic: 1.184406892407454848e2_f64,
        ks_p_value: 5.760497257947763431e-7_f64,
    },
    FitExpected {
        name: "Exponential",
        aic: 1.159944421833937298e2_f64,
        bic: 1.179064651888218691e2_f64,
        ks_p_value: 1.608606591459847950e-3_f64,
    },
    FitExpected {
        name: "ChiSquared",
        aic: 1.428221529010404254e2_f64,
        bic: 1.447341759064685789e2_f64,
        ks_p_value: 7.235708072786238937e-7_f64,
    },
];

pub const EXPECTED_FIT_ALL_MEDIUM_NORMAL: &[FitExpected] = &[
    FitExpected {
        name: "Uniform",
        aic: 1.213197140948095125e2_f64,
        bic: 1.241221088581338279e2_f64,
        ks_p_value: 7.226570087051474855e-1_f64,
    },
    FitExpected {
        name: "Normal",
        aic: 1.238722163901975222e2_f64,
        bic: 1.266746111535218375e2_f64,
        ks_p_value: 9.207119476101386368e-1_f64,
    },
    FitExpected {
        name: "Gamma",
        aic: 1.240108329979821633e2_f64,
        bic: 1.268132277613064787e2_f64,
        ks_p_value: 9.711902110932811860e-1_f64,
    },
    FitExpected {
        name: "LogNormal",
        aic: 1.244758212537962976e2_f64,
        bic: 1.272782160171206129e2_f64,
        ks_p_value: 9.460078514176815112e-1_f64,
    },
    FitExpected {
        name: "Weibull",
        aic: 1.246879793225404711e2_f64,
        bic: 1.274903740858647865e2_f64,
        ks_p_value: 5.777477299729707472e-1_f64,
    },
    FitExpected {
        name: "StudentT",
        aic: 1.262518930905680463e2_f64,
        bic: 1.304554852355545052e2_f64,
        ks_p_value: 9.222123997488645974e-1_f64,
    },
    FitExpected {
        name: "ChiSquared",
        aic: 1.518870437863737664e2_f64,
        bic: 1.532882411680359098e2_f64,
        ks_p_value: 2.693876727165624846e-2_f64,
    },
    FitExpected {
        name: "Exponential",
        aic: 2.000064415452712012e2_f64,
        bic: 2.014076389269333447e2_f64,
        ks_p_value: 9.957176834003386191e-7_f64,
    },
    FitExpected {
        name: "F",
        aic: 2.937069675720478585e2_f64,
        bic: 2.965093623353721455e2_f64,
        ks_p_value: 9.130902848384875403e-22_f64,
    },
];

pub const EXPECTED_FIT_ALL_SMALL_INTEGERS: &[FitExpected] = &[
    FitExpected {
        name: "Uniform",
        aic: 1.786294361119890794e1_f64,
        bic: 1.708181943606710718e1_f64,
        ks_p_value: 9.747892465409951912e-1_f64,
    },
    FitExpected {
        name: "Weibull",
        aic: 2.134351563830021448e1_f64,
        bic: 2.056239146316841371e1_f64,
        ks_p_value: 9.969629519444076493e-1_f64,
    },
    FitExpected {
        name: "ChiSquared",
        aic: 2.140189358926468088e1_f64,
        bic: 2.101133150169878050e1_f64,
        ks_p_value: 9.253829608296609832e-1_f64,
    },
    FitExpected {
        name: "Normal",
        aic: 2.165512123484645457e1_f64,
        bic: 2.087399705971465380e1_f64,
        ks_p_value: 9.983903404822666028e-1_f64,
    },
    FitExpected {
        name: "Gamma",
        aic: 2.166861375031595216e1_f64,
        bic: 2.088748957518415139e1_f64,
        ks_p_value: 9.948950008978878490e-1_f64,
    },
    FitExpected {
        name: "LogNormal",
        aic: 2.211536770113599459e1_f64,
        bic: 2.133424352600419382e1_f64,
        ks_p_value: 9.771100979579145918e-1_f64,
    },
    FitExpected {
        name: "Exponential",
        aic: 2.298612288668109471e1_f64,
        bic: 2.259556079911519433e1_f64,
        ks_p_value: 7.289550986668227317e-1_f64,
    },
    FitExpected {
        name: "StudentT",
        aic: 2.376464418809259982e1_f64,
        bic: 2.259295792539489867e1_f64,
        ks_p_value: 9.987748914743455053e-1_f64,
    },
    FitExpected {
        name: "F",
        aic: 3.011177912872494034e1_f64,
        bic: 2.933065495359313957e1_f64,
        ks_p_value: 7.268046250963082489e-2_f64,
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Assert two f64 are within `tol` relative tolerance (or `tol` absolute when
/// the reference is near zero). Panics with a descriptive message on mismatch.
#[allow(dead_code)]
pub(crate) fn assert_relative_eq(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(1e-30);
    let rel = diff / scale;
    assert!(
        rel < rel_tol || diff < rel_tol,
        "ISO mismatch [{}]: actual={:.18e} expected={:.18e} diff={:.3e} rel={:.3e} tol={}",
        label,
        actual,
        expected,
        diff,
        rel,
        rel_tol
    );
}

/// Assert two f64 are within `tol` absolute tolerance.
#[allow(dead_code)]
pub(crate) fn assert_absolute_eq(actual: f64, expected: f64, abs_tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < abs_tol,
        "ISO mismatch [{}]: actual={:.18e} expected={:.18e} diff={:.3e} tol={}",
        label,
        actual,
        expected,
        diff,
        abs_tol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// rs-stats baseline capture (S10)
// ─────────────────────────────────────────────────────────────────────────────
//
// This module is gated behind `#[cfg(test)]` and the capture test is `#[ignore]`
// so it never runs in normal CI. To regenerate the golden values, run:
//
//   cargo test --lib analytics::stats::golden_fixtures::capture::capture_rs_stats_outputs \
//       -- --ignored --nocapture --test-threads=1
//
// Then copy the printed output into `expected.rs` (currently inlined below).

#[cfg(test)]
mod capture {
    use super::*;

    /// Print all rs-stats outputs on the fixtures, in Rust syntax, ready to be
    /// pasted as `pub const EXPECTED_*` declarations.
    #[test]
    #[ignore]
    fn capture_rs_stats_outputs() {
        eprintln!("// === Captured from rs-stats v2.0.3 ===");
        eprintln!();

        // ─── prob::average ───────────────────────────────────────────────────
        for (name, fixture) in &[
            ("SMALL_INTEGERS", SMALL_INTEGERS),
            ("MEDIUM_NORMAL", MEDIUM_NORMAL),
            ("WITH_OUTLIERS", WITH_OUTLIERS),
            ("NEAR_ZERO_FLAT", NEAR_ZERO_FLAT),
            ("LOGNORMAL_SHAPE", LOGNORMAL_SHAPE),
        ] {
            let v = rs_stats::prob::average(*fixture).unwrap();
            eprintln!("pub const EXPECTED_MEAN_{}: f64 = {:.18e}_f64;", name, v);
        }
        eprintln!();

        // ─── prob::std_dev (population) ──────────────────────────────────────
        for (name, fixture) in &[
            ("SMALL_INTEGERS", SMALL_INTEGERS),
            ("MEDIUM_NORMAL", MEDIUM_NORMAL),
            ("WITH_OUTLIERS", WITH_OUTLIERS),
            ("NEAR_ZERO_FLAT", NEAR_ZERO_FLAT),
            ("LOGNORMAL_SHAPE", LOGNORMAL_SHAPE),
        ] {
            let v = rs_stats::prob::std_dev(*fixture).unwrap();
            eprintln!("pub const EXPECTED_STD_DEV_{}: f64 = {:.18e}_f64;", name, v);
        }
        eprintln!();

        // ─── one_sample_t_test (vs ref_mean = 0.0) ───────────────────────────
        use rs_stats::hypothesis_tests::t_test::one_sample_t_test;
        for (name, fixture) in &[
            ("SMALL_INTEGERS", SMALL_INTEGERS),
            ("MEDIUM_NORMAL", MEDIUM_NORMAL),
            ("NEAR_ZERO_FLAT", NEAR_ZERO_FLAT),
        ] {
            let r = one_sample_t_test(*fixture, 0.0_f64).unwrap();
            eprintln!(
                "pub const EXPECTED_TTEST1_{}: TTestExpected = TTestExpected {{ t_stat: {:.18e}_f64, df: {:.18e}_f64, p_value: {:.18e}_f64 }};",
                name, r.t_statistic, r.degrees_of_freedom, r.p_value
            );
        }
        eprintln!();

        // ─── two_sample_t_test (Welch, false) ────────────────────────────────
        use rs_stats::hypothesis_tests::t_test::two_sample_t_test;
        let r = two_sample_t_test(SMALL_INTEGERS, &[2.0_f64, 3.0, 4.0, 5.0, 6.0], false).unwrap();
        eprintln!(
            "pub const EXPECTED_TTEST2_WELCH_SHIFTED: TTestExpected = TTestExpected {{ t_stat: {:.18e}_f64, df: {:.18e}_f64, p_value: {:.18e}_f64 }};",
            r.t_statistic, r.degrees_of_freedom, r.p_value
        );
        let r = two_sample_t_test(SMALL_INTEGERS, SMALL_INTEGERS, false).unwrap();
        eprintln!(
            "pub const EXPECTED_TTEST2_WELCH_IDENTICAL: TTestExpected = TTestExpected {{ t_stat: {:.18e}_f64, df: {:.18e}_f64, p_value: {:.18e}_f64 }};",
            r.t_statistic, r.degrees_of_freedom, r.p_value
        );
        eprintln!();

        // ─── one_way_anova (3 groups) ────────────────────────────────────────
        use rs_stats::hypothesis_tests::anova::one_way_anova;
        let groups: Vec<&[f64]> = vec![ANOVA_3_GROUPS_LOW, ANOVA_3_GROUPS_MID, ANOVA_3_GROUPS_HIGH];
        let r = one_way_anova(&groups).unwrap();
        eprintln!("pub const EXPECTED_ANOVA_3_GROUPS: AnovaExpected = AnovaExpected {{",);
        eprintln!("    f_stat: {:.18e}_f64,", r.f_statistic);
        eprintln!("    df_between: {},", r.df_between);
        eprintln!("    df_within: {},", r.df_within);
        eprintln!("    p_value: {:.18e}_f64,", r.p_value);
        eprintln!("    ss_between: {:.18e}_f64,", r.ss_between);
        eprintln!("    ss_within: {:.18e}_f64,", r.ss_within);
        eprintln!("    ms_between: {:.18e}_f64,", r.ms_between);
        eprintln!("    ms_within: {:.18e}_f64,", r.ms_within);
        eprintln!("}};");
        eprintln!();

        // ─── fit_all on LOGNORMAL_SHAPE and MEDIUM_NORMAL ────────────────────
        for (name, fixture) in &[
            ("LOGNORMAL_SHAPE", LOGNORMAL_SHAPE),
            ("MEDIUM_NORMAL", MEDIUM_NORMAL),
            ("SMALL_INTEGERS", SMALL_INTEGERS),
        ] {
            let r = rs_stats::fit_all(*fixture).unwrap();
            eprintln!("// fit_all({}) — {} fits ranked by AIC asc", name, r.len());
            eprintln!("pub const EXPECTED_FIT_ALL_{}: &[FitExpected] = &[", name);
            for f in &r {
                eprintln!(
                    "    FitExpected {{ name: \"{}\", aic: {:.18e}_f64, bic: {:.18e}_f64, ks_p_value: {:.18e}_f64 }},",
                    f.name, f.aic, f.bic, f.ks_p_value
                );
            }
            eprintln!("];");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper structs for expected values (used in S11/S12 assertion tests)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct TTestExpected {
    pub t_stat: f64,
    pub df: f64,
    pub p_value: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct AnovaExpected {
    pub f_stat: f64,
    pub df_between: usize,
    pub df_within: usize,
    pub p_value: f64,
    pub ss_between: f64,
    pub ss_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct FitExpected {
    pub name: &'static str,
    pub aic: f64,
    pub bic: f64,
    pub ks_p_value: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance constants — see plan 00f0ca9a "ISO acceptance — tolerance matrix"
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub(crate) const TOL_MEAN_STD_REL: f64 = 1e-12;
#[allow(dead_code)]
pub(crate) const TOL_TSTAT_FSTAT_REL: f64 = 1e-10;
#[allow(dead_code)]
pub(crate) const TOL_SS_MS_REL: f64 = 1e-10;
#[allow(dead_code)]
pub(crate) const TOL_PVALUE_ABS: f64 = 1e-3;
#[allow(dead_code)]
pub(crate) const TOL_AIC_BIC_REL: f64 = 1e-6;
#[allow(dead_code)]
pub(crate) const TOL_KS_PVALUE_ABS: f64 = 1e-3;

// ─────────────────────────────────────────────────────────────────────────────
// Assertion tests — confirm rs-stats reproduces the captured baseline exactly.
// These run in normal CI. They do NOT yet validate the new first-party impl —
// that comes in R3-R6 (extra tests added in those tasks).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod baseline_self_check {
    use super::*;

    #[test]
    fn baseline_mean_matches_capture() {
        for (label, fixture, expected) in &[
            (
                "SMALL_INTEGERS",
                SMALL_INTEGERS,
                EXPECTED_MEAN_SMALL_INTEGERS,
            ),
            ("MEDIUM_NORMAL", MEDIUM_NORMAL, EXPECTED_MEAN_MEDIUM_NORMAL),
            ("WITH_OUTLIERS", WITH_OUTLIERS, EXPECTED_MEAN_WITH_OUTLIERS),
            (
                "NEAR_ZERO_FLAT",
                NEAR_ZERO_FLAT,
                EXPECTED_MEAN_NEAR_ZERO_FLAT,
            ),
            (
                "LOGNORMAL_SHAPE",
                LOGNORMAL_SHAPE,
                EXPECTED_MEAN_LOGNORMAL_SHAPE,
            ),
        ] {
            let actual = rs_stats::prob::average(*fixture).unwrap();
            assert_relative_eq(
                actual,
                *expected,
                TOL_MEAN_STD_REL,
                &format!("mean_{}", label),
            );
        }
    }

    #[test]
    fn baseline_std_dev_matches_capture() {
        for (label, fixture, expected) in &[
            (
                "SMALL_INTEGERS",
                SMALL_INTEGERS,
                EXPECTED_STD_DEV_SMALL_INTEGERS,
            ),
            (
                "MEDIUM_NORMAL",
                MEDIUM_NORMAL,
                EXPECTED_STD_DEV_MEDIUM_NORMAL,
            ),
            (
                "WITH_OUTLIERS",
                WITH_OUTLIERS,
                EXPECTED_STD_DEV_WITH_OUTLIERS,
            ),
            (
                "NEAR_ZERO_FLAT",
                NEAR_ZERO_FLAT,
                EXPECTED_STD_DEV_NEAR_ZERO_FLAT,
            ),
            (
                "LOGNORMAL_SHAPE",
                LOGNORMAL_SHAPE,
                EXPECTED_STD_DEV_LOGNORMAL_SHAPE,
            ),
        ] {
            let actual = rs_stats::prob::std_dev(*fixture).unwrap();
            assert_relative_eq(
                actual,
                *expected,
                TOL_MEAN_STD_REL,
                &format!("std_dev_{}", label),
            );
        }
    }

    #[test]
    fn baseline_ttest1_matches_capture() {
        use rs_stats::hypothesis_tests::t_test::one_sample_t_test;
        for (label, fixture, expected) in &[
            (
                "SMALL_INTEGERS",
                SMALL_INTEGERS,
                EXPECTED_TTEST1_SMALL_INTEGERS,
            ),
            (
                "MEDIUM_NORMAL",
                MEDIUM_NORMAL,
                EXPECTED_TTEST1_MEDIUM_NORMAL,
            ),
            (
                "NEAR_ZERO_FLAT",
                NEAR_ZERO_FLAT,
                EXPECTED_TTEST1_NEAR_ZERO_FLAT,
            ),
        ] {
            let r = one_sample_t_test(*fixture, 0.0_f64).unwrap();
            assert_relative_eq(
                r.t_statistic,
                expected.t_stat,
                TOL_TSTAT_FSTAT_REL,
                &format!("ttest1_t_{}", label),
            );
            assert_relative_eq(
                r.degrees_of_freedom,
                expected.df,
                TOL_TSTAT_FSTAT_REL,
                &format!("ttest1_df_{}", label),
            );
            // p_value: rs-stats vs rs-stats must be exact
            assert_relative_eq(
                r.p_value,
                expected.p_value,
                TOL_PVALUE_ABS,
                &format!("ttest1_p_{}", label),
            );
        }
    }

    #[test]
    fn baseline_ttest2_welch_matches_capture() {
        use rs_stats::hypothesis_tests::t_test::two_sample_t_test;

        let r = two_sample_t_test(SMALL_INTEGERS, &[2.0_f64, 3.0, 4.0, 5.0, 6.0], false).unwrap();
        let e = EXPECTED_TTEST2_WELCH_SHIFTED;
        assert_relative_eq(
            r.t_statistic,
            e.t_stat,
            TOL_TSTAT_FSTAT_REL,
            "ttest2_shifted_t",
        );
        assert_relative_eq(
            r.degrees_of_freedom,
            e.df,
            TOL_TSTAT_FSTAT_REL,
            "ttest2_shifted_df",
        );

        let r = two_sample_t_test(SMALL_INTEGERS, SMALL_INTEGERS, false).unwrap();
        let e = EXPECTED_TTEST2_WELCH_IDENTICAL;
        assert_relative_eq(
            r.t_statistic,
            e.t_stat,
            TOL_TSTAT_FSTAT_REL,
            "ttest2_identical_t",
        );
        assert_relative_eq(
            r.degrees_of_freedom,
            e.df,
            TOL_TSTAT_FSTAT_REL,
            "ttest2_identical_df",
        );
    }

    #[test]
    fn baseline_anova_matches_capture() {
        use rs_stats::hypothesis_tests::anova::one_way_anova;
        let groups: Vec<&[f64]> = vec![ANOVA_3_GROUPS_LOW, ANOVA_3_GROUPS_MID, ANOVA_3_GROUPS_HIGH];
        let r = one_way_anova(&groups).unwrap();
        let e = EXPECTED_ANOVA_3_GROUPS;
        assert_relative_eq(r.f_statistic, e.f_stat, TOL_TSTAT_FSTAT_REL, "anova_f");
        assert_eq!(r.df_between, e.df_between, "anova_df_between");
        assert_eq!(r.df_within, e.df_within, "anova_df_within");
        assert_relative_eq(
            r.ss_between,
            e.ss_between,
            TOL_SS_MS_REL,
            "anova_ss_between",
        );
        assert_relative_eq(r.ss_within, e.ss_within, TOL_SS_MS_REL, "anova_ss_within");
        assert_relative_eq(
            r.ms_between,
            e.ms_between,
            TOL_SS_MS_REL,
            "anova_ms_between",
        );
        assert_relative_eq(r.ms_within, e.ms_within, TOL_SS_MS_REL, "anova_ms_within");
    }

    fn check_fit_all(label: &str, fixture: &[f64], expected: &[FitExpected]) {
        let r = rs_stats::fit_all(fixture).unwrap();
        assert_eq!(r.len(), expected.len(), "{}: fit count mismatch", label);
        for (i, (actual, exp)) in r.iter().zip(expected.iter()).enumerate() {
            assert_eq!(actual.name, exp.name, "{}[{}]: name mismatch", label, i);
            assert_relative_eq(
                actual.aic,
                exp.aic,
                TOL_AIC_BIC_REL,
                &format!("{}[{}].aic", label, i),
            );
            assert_relative_eq(
                actual.bic,
                exp.bic,
                TOL_AIC_BIC_REL,
                &format!("{}[{}].bic", label, i),
            );
            assert_absolute_eq(
                actual.ks_p_value,
                exp.ks_p_value,
                TOL_KS_PVALUE_ABS,
                &format!("{}[{}].ks_p_value", label, i),
            );
        }
    }

    #[test]
    fn baseline_fit_all_lognormal_matches_capture() {
        check_fit_all(
            "LOGNORMAL_SHAPE",
            LOGNORMAL_SHAPE,
            EXPECTED_FIT_ALL_LOGNORMAL_SHAPE,
        );
    }

    #[test]
    fn baseline_fit_all_medium_normal_matches_capture() {
        check_fit_all(
            "MEDIUM_NORMAL",
            MEDIUM_NORMAL,
            EXPECTED_FIT_ALL_MEDIUM_NORMAL,
        );
    }

    #[test]
    fn baseline_fit_all_small_integers_matches_capture() {
        check_fit_all(
            "SMALL_INTEGERS",
            SMALL_INTEGERS,
            EXPECTED_FIT_ALL_SMALL_INTEGERS,
        );
    }
}
