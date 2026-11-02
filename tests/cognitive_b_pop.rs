//! `B_pop` — a decision must not move when the context carries no evidence.
//!
//! The measurement itself lives in `common::metamorphic` (`b_pop`), because the
//! bench prints the same number: one implementation, two readers.
//!
//! Two checks, and neither means anything without the other:
//!
//!  - `invariance` — once every arm has observations, a declared preference
//!    changes NOTHING. The decision is a pure function of `(id, request, arms,
//!    hints)` on a fixed seed, so `B_pop` must be exactly 0, not merely small.
//!  - `control` — on FRESH arms the same preference is an honest prior and DOES
//!    move the decision. This is the non-vacuity guard: if the control ever
//!    reaches 100 % agreement, the invariance measures nothing and this file
//!    would pass while telling no one anything.
//!
//! Both directions are guarded: a prior made permanent (applied at `n > 0`) turns
//! the invariance red; a prior removed entirely turns the control red.
//!
//! What is NOT here, deliberately: no inverse-popularity sampling, no entropy
//! bonus on the utility. `B_pop` is a measurement, never a term of the score;
//! exploration stays bounded by `exploration_epsilon` and the Beta forgetting.

mod common;

use common::metamorphic::{
    b_pop, b_pop_decide, b_pop_hints, b_pop_seed, chosen, Arms, B_POP_SEEDS,
};

#[test]
fn b_pop_is_exactly_zero_once_every_arm_has_observations() {
    let measured = b_pop();
    println!(
        "\n── B_pop ({} seeds) ──\n  {}",
        B_POP_SEEDS,
        measured.line()
    );

    assert_eq!(
        measured.observed_cases,
        3 * usize::try_from(B_POP_SEEDS).unwrap_or(0),
        "three context pairs per seed"
    );
    assert_eq!(
        measured.observed_agreement(),
        1.0,
        "a declared preference moved a decision taken on observed arms"
    );
    assert_eq!(
        measured.observed_b_pop, 0.0,
        "B_pop must be exactly 0 on observed arms, got {}",
        measured.observed_b_pop
    );
    assert_eq!(
        measured.order_agree, measured.order_cases,
        "input order moved the decision"
    );
}

#[test]
fn the_prior_does_move_a_fresh_arm_so_the_invariant_is_not_vacuous() {
    let measured = b_pop();
    println!(
        "\n── B_pop control ──\n  control agreement {:.1}% on {} pairs, {} seeds moved",
        measured.control_agreement() * 100.0,
        measured.control_cases,
        measured.control_moved_seeds
    );

    assert!(
        measured.control_moved_seeds > 0,
        "the alias prior changed nothing even on fresh arms: the invariance test \
         measures nothing and must not be trusted"
    );
    assert!(
        measured.control_agreement() < 1.0,
        "the control reached 100 % agreement: B_pop = 0 above is vacuous"
    );
}

#[test]
fn the_same_id_replays_the_same_decision() {
    let contexts = b_pop_hints();
    for i in 0..B_POP_SEEDS {
        let id = b_pop_seed(i);
        let first = b_pop_decide(id, Arms::Observed, &contexts[0].1);
        let second = b_pop_decide(id, Arms::Observed, &contexts[0].1);
        assert_eq!(chosen(&first), chosen(&second), "seed {i} did not replay");
    }
}
