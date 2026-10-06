//! Cost accounting by basis (decision A21): TWO counters, never one.
//!
//! - the MARGINAL counter is real money out of pocket: a cost the provider
//!   `reported` for a model it prices, or that was `priced` from tokens. Only it
//!   can stop a run;
//! - the NOTIONAL counter is what the work would have cost: a `subscription`
//!   (already paid) or a `free` endpoint. It informs, it never cuts;
//! - an `unknown` cost has no amount: it is counted apart, never as zero.
//!
//! Pure over the facts stored on each `AgentExecution`.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::neo4j::agent_execution::AgentExecutionNode;

/// Whether a cost of this basis counts toward the budget of a run. An execution
/// recorded before bases existed (`None`) was a reported cost.
pub fn counts_toward_budget(basis: Option<&str>) -> bool {
    matches!(basis, None | Some("reported") | Some("priced"))
}

/// Totals of a set of executions.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Totals {
    /// Executions counted.
    pub executions: u64,
    /// Real spend (reported or priced).
    pub marginal_usd: f64,
    /// Subscription or free usage valued at its notional amount.
    pub notional_usd: f64,
    /// Executions whose cost is unknown: no amount, not zero.
    pub unknown_cost_executions: u64,
    /// Tokens read from the provider (`None` when no execution reported any).
    pub input_tokens: Option<u64>,
    /// Output tokens.
    pub output_tokens: Option<u64>,
}

impl Totals {
    fn add(&mut self, ae: &AgentExecutionNode) {
        self.executions += 1;
        match ae.cost_basis.as_deref() {
            None | Some("reported") | Some("priced") => self.marginal_usd += ae.cost_usd,
            Some("subscription") | Some("free") => self.notional_usd += ae.cost_usd,
            // `unknown` (and anything this version does not know): no amount.
            Some(_) => self.unknown_cost_executions += 1,
        }
        if let Some(t) = ae.tokens_in {
            *self.input_tokens.get_or_insert(0) += t;
        }
        if let Some(t) = ae.tokens_out {
            *self.output_tokens.get_or_insert(0) += t;
        }
    }
}

/// The two counters of a run, and the same split by model, provider and task class.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CostReport {
    /// Everything.
    pub total: Totals,
    /// By effective model (`unknown` when the provider did not say).
    pub by_model: BTreeMap<String, Totals>,
    /// By provider instance.
    pub by_provider: BTreeMap<String, Totals>,
    /// By task class (`unclassified` when none).
    pub by_task_class: BTreeMap<String, Totals>,
}

/// Builds the report of a set of executions.
pub fn report(executions: &[AgentExecutionNode]) -> CostReport {
    let mut r = CostReport::default();
    for ae in executions {
        r.total.add(ae);
        r.by_model
            .entry(ae.model.clone().unwrap_or_else(|| "unknown".into()))
            .or_default()
            .add(ae);
        r.by_provider
            .entry(ae.provider_id.clone())
            .or_default()
            .add(ae);
        r.by_task_class
            .entry(
                ae.task_class
                    .clone()
                    .unwrap_or_else(|| "unclassified".into()),
            )
            .or_default()
            .add(ae);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn ae(
        provider: &str,
        model: Option<&str>,
        class: &str,
        usd: f64,
        basis: Option<&str>,
    ) -> AgentExecutionNode {
        let mut a = AgentExecutionNode::new(Uuid::nil(), Uuid::new_v4());
        a.provider_id = provider.into();
        a.model = model.map(str::to_string);
        a.task_class = Some(class.into());
        a.cost_usd = usd;
        a.cost_basis = basis.map(str::to_string);
        a.tokens_in = Some(10);
        a.tokens_out = Some(5);
        a
    }

    #[test]
    fn only_a_real_spend_counts_toward_the_budget() {
        assert!(counts_toward_budget(None), "recorded before bases existed");
        assert!(counts_toward_budget(Some("reported")));
        assert!(counts_toward_budget(Some("priced")));
        assert!(!counts_toward_budget(Some("subscription")));
        assert!(!counts_toward_budget(Some("free")));
        assert!(!counts_toward_budget(Some("unknown")));
    }

    #[test]
    fn two_counters_and_an_unknown_that_is_not_a_zero() {
        let rows = vec![
            ae(
                "claude-code",
                Some("opus"),
                "complex",
                2.0,
                Some("reported"),
            ),
            ae("deepseek", Some("ds-chat"), "simple", 0.5, Some("priced")),
            ae(
                "claude-code",
                Some("opus"),
                "simple",
                3.0,
                Some("subscription"),
            ),
            ae("local", Some("llama"), "simple", 0.0, Some("free")),
            ae("local", None, "simple", 0.0, Some("unknown")),
        ];
        let r = report(&rows);
        assert_eq!(r.total.executions, 5);
        assert_eq!(r.total.marginal_usd, 2.5, "reported + priced only");
        assert_eq!(
            r.total.notional_usd, 3.0,
            "subscription, valued but not spent"
        );
        assert_eq!(r.total.unknown_cost_executions, 1);
        assert_eq!(r.total.input_tokens, Some(50));
        assert_eq!(r.by_provider["claude-code"].marginal_usd, 2.0);
        assert_eq!(r.by_provider["claude-code"].notional_usd, 3.0);
        assert_eq!(r.by_model["opus"].executions, 2);
        assert_eq!(
            r.by_model["unknown"].executions, 1,
            "a model the provider did not name"
        );
        assert_eq!(r.by_task_class["simple"].executions, 4);
        assert_eq!(r.by_task_class["complex"].marginal_usd, 2.0);
    }

    #[test]
    fn an_empty_run_has_no_tokens_rather_than_zero_tokens() {
        let r = report(&[]);
        assert_eq!(r.total.executions, 0);
        assert_eq!(r.total.input_tokens, None);
    }
}
