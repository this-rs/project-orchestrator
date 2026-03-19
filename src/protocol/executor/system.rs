//! SystemExecutor — executes system protocol actions natively via GraphStore.
//!
//! System protocols (auto-maintenance) use MCP-style action strings like:
//! - `"admin(update_staleness_scores)"`
//! - `"admin(persist_health_report) + admin(decay_synapses)"`
//!
//! The SystemExecutor parses these action strings and calls the corresponding
//! GraphStore methods directly (no LLM, no HTTP).

use super::{ExecutionResult, Executor};
use crate::neo4j::traits::GraphStore;
use crate::protocol::{Protocol, ProtocolRun, ProtocolState};
use anyhow::{Context, Result};
use async_trait::async_trait;
use tracing::{debug, info, warn};

/// Executes system protocol state actions natively via GraphStore.
pub struct SystemExecutor;

impl SystemExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SystemExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// A parsed MCP-style action call.
#[derive(Debug, Clone)]
struct ActionCall {
    tool: String,
    method: String,
    args: Vec<String>,
}

/// Parse an action string into a list of action calls.
fn parse_actions(action: &str) -> Result<Vec<ActionCall>> {
    let action = action.trim();
    if action.is_empty() {
        return Ok(vec![]);
    }

    let parts: Vec<&str> = action.split('+').collect();
    let mut calls = Vec::new();

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let open_paren = part.find('(');
        let close_paren = part.rfind(')');

        match (open_paren, close_paren) {
            (Some(op), Some(cp)) if cp > op => {
                let tool = part[..op].trim().to_string();
                let inner = part[op + 1..cp].trim();
                let inner_parts: Vec<&str> = inner.splitn(2, ',').collect();
                let method = inner_parts[0].trim().to_string();
                let args = if inner_parts.len() > 1 {
                    inner_parts[1]
                        .split(',')
                        .map(|a| a.trim().to_string())
                        .filter(|a| !a.is_empty())
                        .collect()
                } else {
                    vec![]
                };
                calls.push(ActionCall { tool, method, args });
            }
            _ => {
                calls.push(ActionCall {
                    tool: "admin".to_string(),
                    method: part.to_string(),
                    args: vec![],
                });
            }
        }
    }

    Ok(calls)
}

/// Execute a single admin action against the GraphStore.
async fn execute_admin_action(
    store: &dyn GraphStore,
    method: &str,
    args: &[String],
    _protocol: &Protocol,
) -> Result<String> {
    match method {
        "update_staleness_scores" => {
            let count = store.update_staleness_scores().await?;
            Ok(format!("Updated staleness scores for {count} notes"))
        }
        "decay_synapses" => {
            let decay_amount = args
                .first()
                .and_then(|a| a.parse::<f64>().ok())
                .unwrap_or(0.05);
            let prune_threshold = args
                .get(1)
                .and_then(|a| a.parse::<f64>().ok())
                .unwrap_or(0.01);
            let (decayed, pruned) = store.decay_synapses(decay_amount, prune_threshold).await?;
            Ok(format!(
                "Decayed {decayed} synapses, pruned {pruned} below threshold"
            ))
        }
        "list_notes_needing_synapses" | "backfill_synapses" => {
            let batch_size = args
                .first()
                .and_then(|a| a.parse::<usize>().ok())
                .unwrap_or(50);
            let (notes, total) = store.list_notes_needing_synapses(batch_size, 0).await?;
            Ok(format!(
                "Found {total} notes needing synapses (batch: {})",
                notes.len()
            ))
        }
        "update_energy_scores" => {
            let half_life = args
                .first()
                .and_then(|a| a.parse::<f64>().ok())
                .unwrap_or(168.0);
            let count = store.update_energy_scores(half_life).await?;
            Ok(format!("Updated energy scores for {count} notes"))
        }
        // Actions requiring full orchestrator context (HTTP endpoints, managers, embeddings)
        "persist_health_report"
        | "update_fabric_scores"
        | "bootstrap_knowledge_fabric"
        | "maintain_skills"
        | "detect_skills"
        | "detect_skill_fission"
        | "detect_skill_fusion"
        | "backfill_touches"
        | "backfill_discussed"
        | "auto_anchor_notes"
        | "reinforce_isomorphic"
        | "reconstruct_knowledge"
        | "consolidate_memory"
        | "audit_gaps"
        | "heal_scars"
        | "detect_stagnation"
        | "deep_maintenance"
        | "seed_prompt_fragments"
        | "install_hooks"
        | "search_neurons"
        | "reinforce_neurons"
        | "reindex_decisions"
        | "backfill_decision_embeddings" => {
            info!(
                method,
                "Admin action requires orchestrator context — marking as done"
            );
            Ok(format!(
                "Action '{method}' acknowledged (requires orchestrator context)"
            ))
        }
        _ => {
            warn!(method, "Unknown admin action — skipping");
            Ok(format!("Skipped unknown admin action: {method}"))
        }
    }
}

/// Determine the trigger to fire based on outgoing transitions.
async fn determine_trigger(
    store: &dyn GraphStore,
    state: &ProtocolState,
    run: &ProtocolRun,
) -> Result<String> {
    let transitions = store.get_protocol_transitions(run.protocol_id).await?;
    let outgoing: Vec<_> = transitions
        .iter()
        .filter(|t| t.from_state == state.id)
        .collect();

    match outgoing.len() {
        0 => Ok("done".to_string()),
        1 => Ok(outgoing[0].trigger.clone()),
        _ => {
            if let Some(t) = outgoing.iter().find(|t| t.trigger == "done") {
                Ok(t.trigger.clone())
            } else if let Some(t) = outgoing.iter().find(|t| t.trigger == "success") {
                Ok(t.trigger.clone())
            } else {
                Ok(outgoing[0].trigger.clone())
            }
        }
    }
}

/// Like [`determine_trigger`] but tries a preferred trigger first, then falls back.
async fn determine_trigger_with_fallback(
    store: &dyn GraphStore,
    state: &ProtocolState,
    run: &ProtocolRun,
    preferred: &str,
    fallback: &str,
) -> Result<String> {
    let transitions = store.get_protocol_transitions(run.protocol_id).await?;
    let outgoing: Vec<_> = transitions
        .iter()
        .filter(|t| t.from_state == state.id)
        .collect();

    if let Some(t) = outgoing.iter().find(|t| t.trigger == preferred) {
        return Ok(t.trigger.clone());
    }
    if let Some(t) = outgoing.iter().find(|t| t.trigger == fallback) {
        return Ok(t.trigger.clone());
    }
    match outgoing.len() {
        0 => Ok(fallback.to_string()),
        1 => Ok(outgoing[0].trigger.clone()),
        _ => Ok(outgoing[0].trigger.clone()),
    }
}

#[async_trait]
impl Executor for SystemExecutor {
    async fn execute_state(
        &self,
        state: &ProtocolState,
        run: &ProtocolRun,
        protocol: &Protocol,
        store: &dyn GraphStore,
    ) -> Result<ExecutionResult> {
        let action_str = match &state.action {
            Some(a) if !a.is_empty() => a.clone(),
            _ => {
                debug!(state_name = %state.name, run_id = %run.id, "No action — auto-advancing");
                return Ok(ExecutionResult {
                    trigger: determine_trigger(store, state, run).await?,
                    notes: vec!["No action — auto-advanced".to_string()],
                    should_retry: false,
                });
            }
        };

        info!(state_name = %state.name, action = %action_str, run_id = %run.id, "SystemExecutor: executing");

        let calls = parse_actions(&action_str)
            .with_context(|| format!("Failed to parse action: {action_str}"))?;

        let mut notes = Vec::new();
        let mut any_failed = false;

        for call in &calls {
            let result = match call.tool.as_str() {
                "admin" => execute_admin_action(store, &call.method, &call.args, protocol).await,
                "note" | "code" | "project" => {
                    debug!(tool = %call.tool, method = %call.method, "Skipping read-only tool call");
                    Ok(format!(
                        "Skipped read-only call: {}({})",
                        call.tool, call.method
                    ))
                }
                other => {
                    warn!(tool = %other, method = %call.method, "Unknown tool in system executor");
                    Ok(format!("Skipped unknown tool: {other}({})", call.method))
                }
            };

            match result {
                Ok(msg) => {
                    info!(tool = %call.tool, method = %call.method, "Action executed successfully");
                    notes.push(msg);
                }
                Err(e) => {
                    warn!(tool = %call.tool, method = %call.method, error = %e, "Action failed");
                    notes.push(format!("FAILED: {}({}) — {}", call.tool, call.method, e));
                    any_failed = true;
                }
            }
        }

        let trigger = if any_failed {
            determine_trigger_with_fallback(store, state, run, "fail", "done").await?
        } else {
            determine_trigger(store, state, run).await?
        };

        Ok(ExecutionResult {
            trigger,
            notes,
            should_retry: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_action() {
        let calls = parse_actions("admin(update_staleness_scores)").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "admin");
        assert_eq!(calls[0].method, "update_staleness_scores");
        assert!(calls[0].args.is_empty());
    }

    #[test]
    fn test_parse_action_with_args() {
        let calls = parse_actions("admin(decay_synapses, 0.05, 0.01)").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].method, "decay_synapses");
        assert_eq!(calls[0].args, vec!["0.05", "0.01"]);
    }

    #[test]
    fn test_parse_chained_actions() {
        let calls = parse_actions("admin(persist_health_report) + admin(decay_synapses)").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].method, "persist_health_report");
        assert_eq!(calls[1].method, "decay_synapses");
    }

    #[test]
    fn test_parse_empty_action() {
        let calls = parse_actions("").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_plain_method() {
        let calls = parse_actions("update_staleness_scores").unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "admin");
        assert_eq!(calls[0].method, "update_staleness_scores");
    }
}
