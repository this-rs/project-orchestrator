//! # Pipeline MCP handlers
//!
//! Exposes the pipeline system (composer, runner, skill injector) via MCP
//! tool actions on the `plan` mega-tool:
//!
//! - `build_pipeline`   — generate a verification pipeline from a plan
//! - `run_pipeline`     — build + create a run config for execution
//! - `pipeline_status`  — get current pipeline execution status
//! - `pipeline_history` — compare past pipeline runs

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use uuid::Uuid;

use super::http_client::McpHttpClient;
use crate::pipeline::composer::{
    compose_pipeline, PipelineSpec, PlanConstraint, PlanWave,
};
use crate::pipeline::runner::PipelineConfig;
use crate::pipeline::skill_injector::{SkillContext, SkillInjector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a UUID field from MCP args.
fn extract_plan_id(args: &Value) -> Result<Uuid> {
    let raw = args
        .get("plan_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing required parameter: plan_id"))?;
    Uuid::parse_str(raw).map_err(|e| anyhow!("invalid plan_id UUID: {e}"))
}

/// Parse a `PlanWave` list from the JSON returned by `GET /api/plans/:id/waves`.
fn parse_waves(waves_json: &Value) -> Vec<PlanWave> {
    let arr = match waves_json.as_array() {
        Some(a) => a,
        None => {
            // Some endpoints wrap in { "waves": [...] }
            match waves_json.get("waves").and_then(|v| v.as_array()) {
                Some(a) => a,
                None => return Vec::new(),
            }
        }
    };

    arr.iter()
        .filter_map(|w| {
            let wave_number = w
                .get("wave_number")
                .or_else(|| w.get("wave"))
                .and_then(|v| v.as_u64())? as usize;

            let task_ids: Vec<Uuid> = w
                .get("task_ids")
                .or_else(|| w.get("tasks"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|id| {
                            id.as_str()
                                .and_then(|s| Uuid::parse_str(s).ok())
                        })
                        .collect()
                })
                .unwrap_or_default();

            let affected_files: Vec<String> = w
                .get("affected_files")
                .or_else(|| w.get("files"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|f| f.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            Some(PlanWave {
                wave_number,
                task_ids,
                affected_files,
            })
        })
        .collect()
}

/// Parse a `PlanConstraint` list from the JSON returned by `GET /api/plans/:id/constraints`.
fn parse_constraints(constraints_json: &Value) -> Vec<PlanConstraint> {
    let arr = match constraints_json.as_array() {
        Some(a) => a,
        None => {
            match constraints_json
                .get("constraints")
                .and_then(|v| v.as_array())
            {
                Some(a) => a,
                None => return Vec::new(),
            }
        }
    };

    arr.iter()
        .filter_map(|c| {
            let constraint_type = c
                .get("constraint_type")
                .or_else(|| c.get("type"))
                .and_then(|v| v.as_str())?
                .to_string();
            let description = c
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let severity = c
                .get("severity")
                .and_then(|v| v.as_str())
                .unwrap_or("should")
                .to_string();

            Some(PlanConstraint {
                constraint_type,
                description,
                severity,
            })
        })
        .collect()
}

/// Collect all affected files across waves.
fn collect_affected_files(waves: &[PlanWave]) -> Vec<String> {
    let mut files: Vec<String> = waves
        .iter()
        .flat_map(|w| w.affected_files.iter().cloned())
        .collect();
    files.sort();
    files.dedup();
    files
}

/// Parse skill contexts from JSON returned by `GET /api/plans/:id/skills` or
/// `GET /api/projects/:slug/skills`.
fn parse_skills(skills_json: &Value) -> Vec<SkillContext> {
    let arr = match skills_json.as_array() {
        Some(a) => a,
        None => {
            match skills_json.get("skills").and_then(|v| v.as_array()) {
                Some(a) => a,
                None => return Vec::new(),
            }
        }
    };

    arr.iter()
        .filter_map(|s| serde_json::from_value::<SkillContext>(s.clone()).ok())
        .collect()
}

/// Internal: build a `PipelineSpec` from plan data fetched via HTTP.
///
/// This is shared by `handle_build_pipeline` and `handle_run_pipeline`.
async fn build_pipeline_spec(
    http: &McpHttpClient,
    plan_id: Uuid,
) -> Result<(PipelineSpec, Option<crate::pipeline::skill_injector::InjectionResult>)> {
    // 1. Fetch plan details
    let plan = http
        .get(&format!("/api/plans/{plan_id}"))
        .await
        .map_err(|e| anyhow!("failed to fetch plan: {e}"))?;

    let plan_name = plan
        .get("name")
        .or_else(|| plan.get("title"))
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed");

    // 2. Fetch waves
    let waves_json = http
        .get(&format!("/api/plans/{plan_id}/waves"))
        .await
        .map_err(|e| anyhow!("failed to fetch waves: {e}"))?;
    let waves = parse_waves(&waves_json);

    // 3. Fetch constraints
    let constraints_json = http
        .get(&format!("/api/plans/{plan_id}/constraints"))
        .await
        .unwrap_or_else(|_| json!([]));
    let constraints = parse_constraints(&constraints_json);

    // 4. Collect affected files
    let affected_files = collect_affected_files(&waves);

    // 5. Compose pipeline
    let mut spec = compose_pipeline(plan_id, plan_name, &waves, &constraints, &affected_files);

    // 6. Apply skill injections if skills are available
    let injection_result = match http
        .get(&format!("/api/plans/{plan_id}/skills"))
        .await
    {
        Ok(skills_json) => {
            let skills = parse_skills(&skills_json);
            if skills.is_empty() {
                None
            } else {
                let injector = SkillInjector::new(0.5);
                let injections = injector.analyze_skills(&skills, &spec);
                if injections.is_empty() {
                    None
                } else {
                    let result = injector.apply_injections(&mut spec, &injections);
                    Some(result)
                }
            }
        }
        Err(_) => None, // Skills endpoint not available — proceed without
    };

    Ok((spec, injection_result))
}

// ---------------------------------------------------------------------------
// Handler functions
// ---------------------------------------------------------------------------

/// **`build_pipeline`** — Generate a verification pipeline from a plan.
///
/// Input: `plan_id` (required)
///
/// Returns the composed `PipelineSpec` as JSON, including any skill injections
/// that were applied.
pub async fn handle_build_pipeline(
    http: &McpHttpClient,
    args: &Value,
) -> Result<Value> {
    let plan_id = extract_plan_id(args)?;

    let (spec, injection_result) = build_pipeline_spec(http, plan_id).await?;

    let mut result = serde_json::to_value(&spec)
        .map_err(|e| anyhow!("failed to serialize pipeline spec: {e}"))?;

    // Attach protocol states/transitions preview
    if let Value::Object(ref mut map) = result {
        map.insert(
            "protocol_states".to_string(),
            serde_json::to_value(spec.to_protocol_states())
                .unwrap_or(json!([])),
        );
        map.insert(
            "protocol_transitions".to_string(),
            serde_json::to_value(spec.to_protocol_transitions())
                .unwrap_or(json!([])),
        );
        if let Some(ref inj) = injection_result {
            map.insert(
                "skill_injections".to_string(),
                serde_json::to_value(inj).unwrap_or(json!(null)),
            );
        }
    }

    Ok(result)
}

/// **`run_pipeline`** — Build a pipeline and create a run configuration.
///
/// Input: `plan_id` (required), `cwd` (optional, defaults to `"."`).
///
/// Returns the run configuration with the embedded pipeline spec.
/// Actual execution is async and would be started by the runner service.
pub async fn handle_run_pipeline(
    http: &McpHttpClient,
    args: &Value,
) -> Result<Value> {
    let plan_id = extract_plan_id(args)?;
    let cwd = args
        .get("cwd")
        .and_then(|v| v.as_str())
        .unwrap_or(".")
        .to_string();
    let project_slug = args
        .get("project_slug")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let fail_fast = args
        .get("fail_fast")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(3600);

    // Build pipeline spec
    let (spec, injection_result) = build_pipeline_spec(http, plan_id).await?;

    // Create run config
    let config = PipelineConfig {
        cwd,
        project_slug,
        max_retries_per_gate: 2,
        timeout_secs,
        fail_fast,
    };

    let spec_json = serde_json::to_value(&spec)
        .map_err(|e| anyhow!("failed to serialize pipeline spec: {e}"))?;
    let config_json = serde_json::to_value(&config)
        .map_err(|e| anyhow!("failed to serialize run config: {e}"))?;

    let mut result = json!({
        "plan_id": plan_id.to_string(),
        "pipeline_spec": spec_json,
        "run_config": config_json,
        "stages_count": spec.stages.len(),
        "final_gates_count": spec.final_gates.len(),
        "status": "ready",
    });

    if let Some(ref inj) = injection_result {
        result.as_object_mut().unwrap().insert(
            "skill_injections".to_string(),
            serde_json::to_value(inj).unwrap_or(json!(null)),
        );
    }

    Ok(result)
}

/// **`pipeline_status`** — Get the current state of pipeline execution.
///
/// Input: `plan_id` (required)
///
/// Returns structured status from the plan's protocol runs: current wave,
/// current gate, and progress score.
pub async fn handle_pipeline_status(
    http: &McpHttpClient,
    args: &Value,
) -> Result<Value> {
    let plan_id = extract_plan_id(args)?;

    // Fetch the current run status
    let run_status = http
        .get(&format!("/api/plans/{plan_id}/run/status"))
        .await
        .unwrap_or_else(|_| json!({"status": "no_active_run"}));

    // Fetch run list to find the latest
    let runs = http
        .get(&format!("/api/plans/{plan_id}/runs"))
        .await
        .unwrap_or_else(|_| json!([]));

    let latest_run = runs
        .as_array()
        .and_then(|arr| arr.first())
        .cloned()
        .unwrap_or(json!(null));

    // Extract progress metrics from the latest run
    let current_wave = latest_run
        .get("waves_completed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total_waves = latest_run
        .get("waves_total")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let progress_score = latest_run
        .get("final_score")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let status = latest_run
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    Ok(json!({
        "plan_id": plan_id.to_string(),
        "status": status,
        "current_wave": current_wave,
        "total_waves": total_waves,
        "progress_score": progress_score,
        "active_run": run_status,
        "latest_run": latest_run,
    }))
}

/// **`pipeline_history`** — Compare past pipeline runs for a plan.
///
/// Input: `plan_id` (required)
///
/// Returns a list of protocol runs associated with the plan with metrics
/// comparison across runs.
pub async fn handle_pipeline_history(
    http: &McpHttpClient,
    args: &Value,
) -> Result<Value> {
    let plan_id = extract_plan_id(args)?;

    // Fetch all runs for the plan
    let runs = http
        .get(&format!("/api/plans/{plan_id}/runs"))
        .await
        .map_err(|e| anyhow!("failed to fetch plan runs: {e}"))?;

    let runs_arr = runs
        .as_array()
        .or_else(|| runs.get("runs").and_then(|v| v.as_array()))
        .cloned()
        .unwrap_or_default();

    if runs_arr.is_empty() {
        return Ok(json!({
            "plan_id": plan_id.to_string(),
            "total_runs": 0,
            "runs": [],
            "comparison": null,
            "message": "No pipeline runs found for this plan.",
        }));
    }

    // Build per-run summaries
    let summaries: Vec<Value> = runs_arr
        .iter()
        .map(|run| {
            json!({
                "run_id": run.get("id").or_else(|| run.get("run_id")),
                "status": run.get("status"),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
                "duration_ms": run.get("duration_ms"),
                "waves_completed": run.get("waves_completed"),
                "waves_total": run.get("waves_total"),
                "final_score": run.get("final_score"),
                "stop_reason": run.get("stop_reason"),
            })
        })
        .collect();

    // Build comparison metrics across runs
    let scores: Vec<f64> = runs_arr
        .iter()
        .filter_map(|r| r.get("final_score").and_then(|v| v.as_f64()))
        .collect();
    let durations: Vec<u64> = runs_arr
        .iter()
        .filter_map(|r| r.get("duration_ms").and_then(|v| v.as_u64()))
        .collect();

    let avg_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };
    let avg_duration = if durations.is_empty() {
        0
    } else {
        durations.iter().sum::<u64>() / durations.len() as u64
    };
    let best_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let worst_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);

    // Score trend (last 2 runs)
    let trend = if scores.len() >= 2 {
        let latest = scores[0];
        let previous = scores[1];
        if latest > previous {
            "improving"
        } else if latest < previous {
            "declining"
        } else {
            "stable"
        }
    } else {
        "insufficient_data"
    };

    Ok(json!({
        "plan_id": plan_id.to_string(),
        "total_runs": runs_arr.len(),
        "runs": summaries,
        "comparison": {
            "avg_score": avg_score,
            "best_score": if best_score.is_finite() { json!(best_score) } else { json!(null) },
            "worst_score": if worst_score.is_finite() { json!(worst_score) } else { json!(null) },
            "avg_duration_ms": avg_duration,
            "score_trend": trend,
        },
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_waves --------------------------------------------------------

    #[test]
    fn parse_waves_from_array() {
        let id1 = Uuid::new_v4();
        let json = json!([
            {
                "wave_number": 1,
                "task_ids": [id1.to_string()],
                "affected_files": ["src/lib.rs"]
            }
        ]);
        let waves = parse_waves(&json);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].wave_number, 1);
        assert_eq!(waves[0].task_ids, vec![id1]);
        assert_eq!(waves[0].affected_files, vec!["src/lib.rs".to_string()]);
    }

    #[test]
    fn parse_waves_from_wrapped_object() {
        let json = json!({
            "waves": [
                {
                    "wave_number": 1,
                    "task_ids": [],
                    "affected_files": []
                }
            ]
        });
        let waves = parse_waves(&json);
        assert_eq!(waves.len(), 1);
    }

    #[test]
    fn parse_waves_empty() {
        let waves = parse_waves(&json!(null));
        assert!(waves.is_empty());
    }

    #[test]
    fn parse_waves_alternate_keys() {
        let id1 = Uuid::new_v4();
        let json = json!([
            {
                "wave": 2,
                "tasks": [id1.to_string()],
                "files": ["main.rs"]
            }
        ]);
        let waves = parse_waves(&json);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].wave_number, 2);
        assert_eq!(waves[0].task_ids, vec![id1]);
        assert_eq!(waves[0].affected_files, vec!["main.rs".to_string()]);
    }

    // -- parse_constraints --------------------------------------------------

    #[test]
    fn parse_constraints_from_array() {
        let json = json!([
            {
                "constraint_type": "performance",
                "description": "High coverage",
                "severity": "must"
            }
        ]);
        let constraints = parse_constraints(&json);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].constraint_type, "performance");
        assert_eq!(constraints[0].severity, "must");
    }

    #[test]
    fn parse_constraints_alternate_key() {
        let json = json!([
            {
                "type": "security",
                "description": "Audit required",
                "severity": "should"
            }
        ]);
        let constraints = parse_constraints(&json);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].constraint_type, "security");
    }

    #[test]
    fn parse_constraints_defaults() {
        let json = json!([
            {
                "constraint_type": "style"
            }
        ]);
        let constraints = parse_constraints(&json);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].description, "");
        assert_eq!(constraints[0].severity, "should");
    }

    #[test]
    fn parse_constraints_empty() {
        let constraints = parse_constraints(&json!(null));
        assert!(constraints.is_empty());
    }

    // -- collect_affected_files ---------------------------------------------

    #[test]
    fn collect_affected_files_deduplicates() {
        let waves = vec![
            PlanWave {
                wave_number: 1,
                task_ids: vec![],
                affected_files: vec!["a.rs".into(), "b.rs".into()],
            },
            PlanWave {
                wave_number: 2,
                task_ids: vec![],
                affected_files: vec!["b.rs".into(), "c.rs".into()],
            },
        ];
        let files = collect_affected_files(&waves);
        assert_eq!(files, vec!["a.rs", "b.rs", "c.rs"]);
    }

    #[test]
    fn collect_affected_files_empty_waves() {
        let files = collect_affected_files(&[]);
        assert!(files.is_empty());
    }

    // -- parse_skills -------------------------------------------------------

    #[test]
    fn parse_skills_empty() {
        let skills = parse_skills(&json!(null));
        assert!(skills.is_empty());
    }

    #[test]
    fn parse_skills_from_wrapped() {
        let skills = parse_skills(&json!({"skills": []}));
        assert!(skills.is_empty());
    }
}
