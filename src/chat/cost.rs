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
use crate::neo4j::traits::*;

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
    /// Input tokens served from the provider's prompt cache (`None` when no execution
    /// reported any). Apart from `input_tokens`, which counts what was NOT cached.
    pub cache_read_tokens: Option<u64>,
    /// Input tokens written to the prompt cache (`None` when none reported: an
    /// OpenAI-compatible endpoint never reports it).
    pub cache_write_tokens: Option<u64>,
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
        if let Some(t) = ae.tokens_cache_read {
            *self.cache_read_tokens.get_or_insert(0) += t;
        }
        if let Some(t) = ae.tokens_cache_write {
            *self.cache_write_tokens.get_or_insert(0) += t;
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

/// One turn of a conversation, as its stored `result` event reported it.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct TurnUsage {
    /// Sequence number of the `result` event in the session.
    pub seq: i64,
    /// Model that answered.
    pub model: Option<String>,
    /// Input tokens NOT served from the prompt cache.
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Input tokens served from the prompt cache.
    pub cache_read_tokens: Option<u64>,
    /// Input tokens written to the prompt cache (never reported by an OpenAI-compatible
    /// endpoint: absent, not zero).
    pub cache_creation_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    /// `cache_read / (input + cache_read)`, when both are known.
    pub cache_hit_ratio: Option<f64>,
}

/// The token usage of one conversation, read from the `result` events stored with its
/// turns (the path Claude Code's cache reads already take; nothing is stored twice).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SessionUsage {
    /// Turns that ended (a `result` event).
    pub turns: u64,
    /// Of those, the turns whose provider reported a usage.
    pub turns_with_usage: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_creation_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    /// Over the whole conversation: `cache_read / (input + cache_read)`.
    pub cache_hit_ratio: Option<f64>,
    /// Turn by turn, in order.
    pub by_turn: Vec<TurnUsage>,
}

fn ratio(input: Option<u64>, cached: Option<u64>) -> Option<f64> {
    let total = input? + cached?;
    (total > 0).then(|| cached.unwrap_or(0) as f64 / total as f64)
}

fn add_to(slot: &mut Option<u64>, value: Option<u64>) {
    if let Some(v) = value {
        *slot.get_or_insert(0) += v;
    }
}

/// Sums `(seq, result event payload)` pairs: the payload is the stored JSON of a
/// `ChatEvent::Result` (`usage` and `model` are read; anything else is ignored).
pub fn session_usage<'a>(
    results: impl IntoIterator<Item = (i64, &'a serde_json::Value)>,
) -> SessionUsage {
    let mut out = SessionUsage::default();
    for (seq, data) in results {
        out.turns += 1;
        let Some(usage) = data.get("usage").filter(|u| u.is_object()) else {
            continue;
        };
        let count = |key: &str| usage.get(key).and_then(serde_json::Value::as_u64);
        let turn = TurnUsage {
            seq,
            model: data
                .get("model")
                .and_then(|m| m.as_str())
                .map(str::to_string),
            input_tokens: count("input_tokens"),
            output_tokens: count("output_tokens"),
            cache_read_tokens: count("cache_read_tokens"),
            cache_creation_tokens: count("cache_creation_tokens"),
            reasoning_tokens: count("reasoning_tokens"),
            cache_hit_ratio: ratio(count("input_tokens"), count("cache_read_tokens")),
        };
        out.turns_with_usage += 1;
        add_to(&mut out.input_tokens, turn.input_tokens);
        add_to(&mut out.output_tokens, turn.output_tokens);
        add_to(&mut out.cache_read_tokens, turn.cache_read_tokens);
        add_to(&mut out.cache_creation_tokens, turn.cache_creation_tokens);
        add_to(&mut out.reasoning_tokens, turn.reasoning_tokens);
        out.by_turn.push(turn);
    }
    out.cache_hit_ratio = ratio(out.input_tokens, out.cache_read_tokens);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;
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
        assert_eq!(r.total.cache_read_tokens, None);
    }

    #[test]
    fn a_session_usage_sums_its_stored_results_and_says_its_cache_hit_ratio() {
        use serde_json::json;
        // Two native turns (DeepSeek: no cache write), an unknown usage, an older event shape.
        let first = json!({"type": "result", "model": "deepseek-v4-pro",
            "usage": {"input_tokens": 4000, "output_tokens": 100, "cache_read_tokens": 0}});
        let second = json!({"type": "result", "model": "deepseek-v4-pro",
            "usage": {"input_tokens": 200, "output_tokens": 50, "cache_read_tokens": 3800,
                "reasoning_tokens": 20}});
        let unknown = json!({"type": "result"});
        let u = session_usage([(3, &first), (9, &second), (12, &unknown)]);
        assert_eq!((u.turns, u.turns_with_usage), (3, 2));
        assert_eq!(u.input_tokens, Some(4200));
        assert_eq!(u.cache_read_tokens, Some(3800));
        assert_eq!(
            u.cache_creation_tokens, None,
            "never reported: unknown, not zero"
        );
        assert_eq!(u.reasoning_tokens, Some(20));
        assert_eq!(u.by_turn.len(), 2);
        assert_eq!(u.by_turn[0].cache_hit_ratio, Some(0.0));
        assert_eq!(u.by_turn[1].cache_hit_ratio, Some(0.95));
        assert_eq!(u.by_turn[1].seq, 9);
        assert_eq!(u.by_turn[1].model.as_deref(), Some("deepseek-v4-pro"));
        assert!((u.cache_hit_ratio.unwrap() - 3800.0 / 8000.0).abs() < 1e-12);
        let none = session_usage(std::iter::empty());
        assert_eq!(
            (none.turns, none.input_tokens, none.cache_hit_ratio),
            (0, None, None)
        );
    }

    #[test]
    fn cache_reads_are_summed_apart_and_an_unreported_cache_write_stays_unknown() {
        let mut native = ae("deepseek", Some("ds-chat"), "simple", 0.1, Some("priced"));
        native.tokens_cache_read = Some(800);
        let mut cc = ae("claude-code", Some("opus"), "simple", 1.0, Some("reported"));
        cc.tokens_cache_read = Some(200);
        cc.tokens_cache_write = Some(50);
        let without = ae("local", Some("llama"), "simple", 0.0, Some("free"));
        let r = report(&[native, cc, without]);
        assert_eq!(r.total.cache_read_tokens, Some(1000));
        assert_eq!(r.total.cache_write_tokens, Some(50));
        assert_eq!(
            r.total.input_tokens,
            Some(30),
            "the cached part is not in the input"
        );
        assert_eq!(r.by_provider["deepseek"].cache_read_tokens, Some(800));
        assert_eq!(
            r.by_provider["deepseek"].cache_write_tokens, None,
            "never reported is unknown, not zero"
        );
        assert_eq!(r.by_provider["local"].cache_read_tokens, None);
    }
}
