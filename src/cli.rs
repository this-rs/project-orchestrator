//! Project Orchestrator - CLI Tool
//!
//! Command-line interface for interacting with the orchestrator.

use anyhow::Result;
use clap::{Parser, Subcommand};
use reqwest::Client;
use serde_json::Value;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "orch")]
#[command(about = "CLI for Project Orchestrator")]
struct Cli {
    /// Orchestrator server URL
    #[arg(
        long,
        env = "ORCHESTRATOR_URL",
        default_value = "http://localhost:8080"
    )]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Plan operations
    Plan {
        #[command(subcommand)]
        action: PlanAction,
    },

    /// Task operations
    Task {
        #[command(subcommand)]
        action: TaskAction,
    },

    /// Decision operations
    Decision {
        #[command(subcommand)]
        action: DecisionAction,
    },

    /// Sync operations
    Sync {
        /// Directory to sync
        #[arg(short, long, default_value = ".")]
        path: String,
    },

    /// Get context for a task
    Context {
        /// Plan ID
        #[arg(long)]
        plan: Uuid,

        /// Task ID
        #[arg(long)]
        task: Uuid,

        /// Output as prompt instead of JSON
        #[arg(long)]
        prompt: bool,
    },
}

#[derive(Subcommand)]
enum PlanAction {
    /// List all active plans
    List,

    /// Create a new plan
    Create {
        /// Plan title
        #[arg(short, long)]
        title: String,

        /// Plan description
        #[arg(short, long)]
        desc: String,

        /// Priority
        #[arg(short, long, default_value = "0")]
        priority: i32,
    },

    /// Show plan details
    Show {
        /// Plan ID
        id: Uuid,
    },

    /// Get next available task
    Next {
        /// Plan ID
        id: Uuid,
    },

    /// Update plan status
    Status {
        /// Plan ID
        id: Uuid,

        /// New status (draft, approved, in_progress, completed, cancelled)
        status: String,
    },
}

#[derive(Subcommand)]
enum TaskAction {
    /// Add a task to a plan
    Add {
        /// Plan ID
        #[arg(long)]
        plan: Uuid,

        /// Task description
        #[arg(short, long)]
        desc: String,

        /// Depends on task IDs (comma-separated)
        #[arg(long)]
        depends: Option<String>,
    },

    /// Show task details
    Show {
        /// Task ID
        id: Uuid,
    },

    /// Update task status
    Status {
        /// Task ID
        id: Uuid,

        /// New status (pending, in_progress, blocked, completed, failed)
        status: String,

        /// Assign to agent
        #[arg(long)]
        agent: Option<String>,
    },
}

#[derive(Subcommand)]
enum DecisionAction {
    /// Add a decision to a task
    Add {
        /// Task ID
        #[arg(long)]
        task: Uuid,

        /// Decision description
        #[arg(short, long)]
        desc: String,

        /// Rationale
        #[arg(short, long)]
        rationale: String,
    },

    /// Search decisions
    Search {
        /// Search query
        query: String,

        /// Max results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Client::new();

    match cli.command {
        Commands::Plan { action } => handle_plan(&client, &cli.server, action).await,
        Commands::Task { action } => handle_task(&client, &cli.server, action).await,
        Commands::Decision { action } => handle_decision(&client, &cli.server, action).await,
        Commands::Sync { path } => handle_sync(&client, &cli.server, &path).await,
        Commands::Context { plan, task, prompt } => {
            handle_context(&client, &cli.server, plan, task, prompt).await
        }
    }
}

async fn handle_plan(client: &Client, server: &str, action: PlanAction) -> Result<()> {
    match action {
        PlanAction::List => {
            let resp: Vec<Value> = client
                .get(format!("{}/api/plans", server))
                .send()
                .await?
                .json()
                .await?;

            println!("{:<36} {:<12} {:<5} {}", "ID", "STATUS", "PRI", "TITLE");
            println!("{}", "-".repeat(80));
            for plan in resp {
                println!(
                    "{:<36} {:<12} {:<5} {}",
                    plan["id"].as_str().unwrap_or("-"),
                    plan["status"].as_str().unwrap_or("-"),
                    plan["priority"].as_i64().unwrap_or(0),
                    plan["title"].as_str().unwrap_or("-")
                );
            }
        }

        PlanAction::Create {
            title,
            desc,
            priority,
        } => {
            let body = serde_json::json!({
                "title": title,
                "description": desc,
                "priority": priority
            });

            let resp: Value = client
                .post(format!("{}/api/plans", server))
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            println!("Created plan: {}", resp["id"]);
        }

        PlanAction::Show { id } => {
            let resp: Value = client
                .get(format!("{}/api/plans/{}", server, id))
                .send()
                .await?
                .json()
                .await?;

            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        PlanAction::Next { id } => {
            let resp: Value = client
                .get(format!("{}/api/plans/{}/next-task", server, id))
                .send()
                .await?
                .json()
                .await?;

            if resp.is_null() {
                println!("No available tasks");
            } else {
                println!("{}", serde_json::to_string_pretty(&resp)?);
            }
        }

        PlanAction::Status { id, status } => {
            let body = serde_json::json!({
                "status": status
            });

            client
                .patch(format!("{}/api/plans/{}", server, id))
                .json(&body)
                .send()
                .await?;

            println!("Updated plan status to: {}", status);
        }
    }

    Ok(())
}

async fn handle_task(client: &Client, server: &str, action: TaskAction) -> Result<()> {
    match action {
        TaskAction::Add {
            plan,
            desc,
            depends,
        } => {
            let depends_on: Option<Vec<Uuid>> =
                depends.map(|d| d.split(',').filter_map(|s| s.trim().parse().ok()).collect());

            let body = serde_json::json!({
                "description": desc,
                "depends_on": depends_on
            });

            let resp: Value = client
                .post(format!("{}/api/plans/{}/tasks", server, plan))
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            println!("Created task: {}", resp["id"]);
        }

        TaskAction::Show { id } => {
            let resp: Value = client
                .get(format!("{}/api/tasks/{}", server, id))
                .send()
                .await?
                .json()
                .await?;

            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        TaskAction::Status { id, status, agent } => {
            let body = serde_json::json!({
                "status": status,
                "assigned_to": agent
            });

            client
                .patch(format!("{}/api/tasks/{}", server, id))
                .json(&body)
                .send()
                .await?;

            println!("Updated task status to: {}", status);
        }
    }

    Ok(())
}

async fn handle_decision(client: &Client, server: &str, action: DecisionAction) -> Result<()> {
    match action {
        DecisionAction::Add {
            task,
            desc,
            rationale,
        } => {
            let body = serde_json::json!({
                "description": desc,
                "rationale": rationale
            });

            let resp: Value = client
                .post(format!("{}/api/tasks/{}/decisions", server, task))
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            println!("Created decision: {}", resp["id"]);
        }

        DecisionAction::Search { query, limit } => {
            let resp: Vec<Value> = client
                .get(format!(
                    "{}/api/decisions/search?q={}&limit={}",
                    server, query, limit
                ))
                .send()
                .await?
                .json()
                .await?;

            for decision in resp {
                println!("---");
                println!("ID: {}", decision["id"]);
                println!("Description: {}", decision["description"]);
                println!("Rationale: {}", decision["rationale"]);
                println!("By: {}", decision["decided_by"]);
            }
        }
    }

    Ok(())
}

async fn handle_sync(client: &Client, server: &str, path: &str) -> Result<()> {
    let body = serde_json::json!({
        "path": path
    });

    let resp: Value = client
        .post(format!("{}/api/sync", server))
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    println!(
        "Synced: {} files, {} skipped, {} errors",
        resp["files_synced"], resp["files_skipped"], resp["errors"]
    );

    Ok(())
}

async fn handle_context(
    client: &Client,
    server: &str,
    plan_id: Uuid,
    task_id: Uuid,
    as_prompt: bool,
) -> Result<()> {
    if as_prompt {
        let resp: Value = client
            .get(format!(
                "{}/api/plans/{}/tasks/{}/prompt",
                server, plan_id, task_id
            ))
            .send()
            .await?
            .json()
            .await?;

        println!("{}", resp["prompt"].as_str().unwrap_or(""));
    } else {
        let resp: Value = client
            .get(format!(
                "{}/api/plans/{}/tasks/{}/context",
                server, plan_id, task_id
            ))
            .send()
            .await?
            .json()
            .await?;

        println!("{}", serde_json::to_string_pretty(&resp)?);
    }

    Ok(())
}
