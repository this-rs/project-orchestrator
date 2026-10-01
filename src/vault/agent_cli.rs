//! `orchestrator secret …` — how an agent's shell uses a granted secret.
//!
//! The value travels server → this process → a pipe or a child's environment.
//! It never needs to appear in the agent's own output, so it never needs to
//! enter the model's context or the transcript.
//!
//! - `secret exec -e VAR=NAME -- cmd args…` (preferred): runs `cmd` with the
//!   secret in its environment. Nothing is printed.
//! - `secret get NAME`: writes the value to stdout, for `… | cmd --password-stdin`.
//!   Run alone, its output WOULD reach the model; the server masks it in what is
//!   stored, but the prompt tells agents to always pipe it.
//!
//! Identity comes from `PO_VAULT_TOKEN`, injected by the chat manager and signed
//! with the session id: an agent cannot read another session's grants by
//! editing its environment.

use zeroize::Zeroizing;

const TOKEN_VAR: &str = "PO_VAULT_TOKEN";
const URL_VAR: &str = "PO_SERVER_URL";

/// Fetch one value. Errors are plain sentences meant for the agent.
pub async fn fetch(name: &str) -> Result<Zeroizing<String>, String> {
    let token = std::env::var(TOKEN_VAR).map_err(|_| {
        format!("{TOKEN_VAR} is not set — vault access exists only inside a Project Orchestrator chat session")
    })?;
    let base = std::env::var(URL_VAR).unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/api/vault/agent/read",
            base.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .json(&serde_json::json!({ "name": name }))
        .send()
        .await
        .map_err(|e| format!("cannot reach the orchestrator at {base}: {e}"))?;
    let status = resp.status();
    let body = Zeroizing::new(resp.text().await.map_err(|e| e.to_string())?);
    if status.is_success() {
        return Ok(body);
    }
    // Error bodies are `{"error": "..."}` and never contain a value.
    let reason = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
        .unwrap_or_else(|| status.to_string());
    Err(format!("secret `{name}`: {reason}"))
}

pub async fn get(name: &str) -> i32 {
    use std::io::Write;
    match fetch(name).await {
        Ok(value) => {
            let mut out = std::io::stdout().lock();
            // No trailing newline: `--password-stdin` style consumers would
            // otherwise take it as part of the value.
            if out
                .write_all(value.as_bytes())
                .and_then(|_| out.flush())
                .is_err()
            {
                return 1;
            }
            0
        }
        Err(e) => {
            eprintln!("orchestrator secret get: {e}");
            1
        }
    }
}

/// Parse `VAR=NAME`.
pub fn parse_mapping(s: &str) -> Result<(String, String), String> {
    let (var, name) = s
        .split_once('=')
        .ok_or_else(|| format!("`{s}`: expected VAR=SECRET_NAME"))?;
    let valid_var = !var.is_empty()
        && !var.starts_with(|c: char| c.is_ascii_digit())
        && var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid_var || name.is_empty() {
        return Err(format!("`{s}`: expected VAR=SECRET_NAME"));
    }
    Ok((var.to_string(), name.to_string()))
}

pub async fn exec(mappings: &[(String, String)], command: &[String]) -> i32 {
    let Some((program, args)) = command.split_first() else {
        eprintln!("orchestrator secret exec: no command given (… -- cmd args)");
        return 1;
    };
    let mut values = Vec::with_capacity(mappings.len());
    for (var, name) in mappings {
        match fetch(name).await {
            Ok(v) => values.push((var.clone(), v)),
            Err(e) => {
                eprintln!("orchestrator secret exec: {e}");
                return 1;
            }
        }
    }
    let mut cmd = std::process::Command::new(program);
    cmd.args(args);
    // The child does not need our token: it would let it read other granted
    // secrets on its own.
    cmd.env_remove(TOKEN_VAR);
    for (var, value) in &values {
        cmd.env(var, value.as_str());
    }
    match cmd.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("orchestrator secret exec: cannot run `{program}`: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mappings_are_validated() {
        assert_eq!(
            parse_mapping("DEMO_PASS=demo-secret").unwrap(),
            ("DEMO_PASS".into(), "demo-secret".into())
        );
        for bad in ["nope", "=x", "A=", "1A=x", "A-B=x"] {
            assert!(parse_mapping(bad).is_err(), "{bad}");
        }
    }
}
