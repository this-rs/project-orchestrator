//! Docker management for the desktop application.
//!
//! Uses bollard to manage Neo4j and MeiliSearch containers directly
//! from the Tauri app, without requiring docker-compose.

use bollard::container::{
    Config as ContainerConfig, CreateContainerOptions, ListContainersOptions, LogsOptions,
    StopContainerOptions,
};
use bollard::image::CreateImageOptions;
use bollard::models::{HostConfig, PortBinding};
use bollard::Docker;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// Types
// ============================================================================

const NEO4J_IMAGE: &str = "neo4j:5-community";
const MEILISEARCH_IMAGE: &str = "getmeili/meilisearch:latest";
const NATS_IMAGE: &str = "nats:latest";
const NEO4J_CONTAINER: &str = "orchestrator-neo4j";
const MEILISEARCH_CONTAINER: &str = "orchestrator-meilisearch";
const NATS_CONTAINER: &str = "orchestrator-nats";
const NATS_PORT: u16 = 4222;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Starting,
    Healthy,
    Unhealthy,
    NotRunning,
    Disabled,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServicesHealth {
    pub neo4j: ServiceStatus,
    pub meilisearch: ServiceStatus,
    pub nats: ServiceStatus,
    pub docker_available: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DockerConfig {
    pub neo4j_password: String,
    pub meilisearch_key: String,
    #[serde(default = "default_true")]
    pub nats_enabled: bool,
}

// ============================================================================
// Docker Manager
// ============================================================================

pub struct DockerManager {
    docker: Option<Docker>,
}

impl DockerManager {
    /// Try to connect to Docker. Returns a manager even if Docker is unavailable.
    pub fn new() -> Self {
        let docker = Docker::connect_with_local_defaults().ok();
        Self { docker }
    }

    /// Check if Docker daemon is reachable.
    pub async fn is_available(&self) -> bool {
        match &self.docker {
            Some(docker) => docker.ping().await.is_ok(),
            None => false,
        }
    }

    fn docker(&self) -> Result<&Docker, String> {
        self.docker
            .as_ref()
            .ok_or_else(|| "Docker is not available".into())
    }

    /// Pull an image if it's not already present locally.
    async fn ensure_image(&self, image: &str) -> Result<(), String> {
        let docker = self.docker()?;

        // Check if image exists locally
        if docker.inspect_image(image).await.is_ok() {
            tracing::info!("Image {} already present", image);
            return Ok(());
        }

        tracing::info!("Pulling image {}...", image);
        let opts = CreateImageOptions {
            from_image: image,
            ..Default::default()
        };

        let mut stream = docker.create_image(Some(opts), None, None);
        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = &info.status {
                        tracing::debug!("Pull {}: {}", image, status);
                    }
                }
                Err(e) => return Err(format!("Failed to pull {}: {}", image, e)),
            }
        }

        tracing::info!("Image {} pulled successfully", image);
        Ok(())
    }

    /// Check if a container exists (running or stopped).
    async fn container_exists(&self, name: &str) -> Result<bool, String> {
        let docker = self.docker()?;
        let mut filters = HashMap::new();
        filters.insert("name", vec![name]);

        let opts = ListContainersOptions {
            all: true,
            filters,
            ..Default::default()
        };

        let containers = docker
            .list_containers(Some(opts))
            .await
            .map_err(|e| format!("Failed to list containers: {}", e))?;

        // bollard name matching includes a leading /
        Ok(containers.iter().any(|c| {
            c.names
                .as_ref()
                .map(|names| names.iter().any(|n| n == &format!("/{}", name)))
                .unwrap_or(false)
        }))
    }

    /// Check if a container is running.
    async fn container_running(&self, name: &str) -> Result<bool, String> {
        let docker = self.docker()?;
        match docker.inspect_container(name, None).await {
            Ok(info) => Ok(info
                .state
                .and_then(|s| s.running)
                .unwrap_or(false)),
            Err(_) => Ok(false),
        }
    }

    /// Start Neo4j, MeiliSearch, and optionally NATS containers.
    pub async fn start_services(&self, config: &DockerConfig) -> Result<(), String> {
        let docker = self.docker()?;

        // Pull images in parallel (only pull NATS if enabled)
        let (r1, r2) = tokio::join!(
            self.ensure_image(NEO4J_IMAGE),
            self.ensure_image(MEILISEARCH_IMAGE),
        );
        r1?;
        r2?;

        if config.nats_enabled {
            self.ensure_image(NATS_IMAGE).await?;
        }

        // --- Neo4j ---
        if !self.container_exists(NEO4J_CONTAINER).await? {
            tracing::info!("Creating Neo4j container...");

            let mut port_bindings = HashMap::new();
            port_bindings.insert(
                "7474/tcp".to_string(),
                Some(vec![PortBinding {
                    host_ip: Some("127.0.0.1".into()),
                    host_port: Some("7474".into()),
                }]),
            );
            port_bindings.insert(
                "7687/tcp".to_string(),
                Some(vec![PortBinding {
                    host_ip: Some("127.0.0.1".into()),
                    host_port: Some("7687".into()),
                }]),
            );

            let container_config = ContainerConfig {
                image: Some(NEO4J_IMAGE.to_string()),
                env: Some(vec![
                    format!("NEO4J_AUTH=neo4j/{}", config.neo4j_password),
                    "NEO4J_PLUGINS=[\"apoc\"]".into(),
                    "NEO4J_dbms_security_procedures_unrestricted=apoc.*".into(),
                ]),
                host_config: Some(HostConfig {
                    port_bindings: Some(port_bindings),
                    restart_policy: Some(bollard::models::RestartPolicy {
                        name: Some(bollard::models::RestartPolicyNameEnum::UNLESS_STOPPED),
                        maximum_retry_count: None,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };

            docker
                .create_container(
                    Some(CreateContainerOptions {
                        name: NEO4J_CONTAINER,
                        ..Default::default()
                    }),
                    container_config,
                )
                .await
                .map_err(|e| format!("Failed to create Neo4j container: {}", e))?;
        }

        if !self.container_running(NEO4J_CONTAINER).await? {
            tracing::info!("Starting Neo4j...");
            docker
                .start_container::<String>(NEO4J_CONTAINER, None)
                .await
                .map_err(|e| format!("Failed to start Neo4j: {}", e))?;
        }

        // --- MeiliSearch ---
        if !self.container_exists(MEILISEARCH_CONTAINER).await? {
            tracing::info!("Creating MeiliSearch container...");

            let mut port_bindings = HashMap::new();
            port_bindings.insert(
                "7700/tcp".to_string(),
                Some(vec![PortBinding {
                    host_ip: Some("127.0.0.1".into()),
                    host_port: Some("7700".into()),
                }]),
            );

            let container_config = ContainerConfig {
                image: Some(MEILISEARCH_IMAGE.to_string()),
                env: Some(vec![
                    format!("MEILI_MASTER_KEY={}", config.meilisearch_key),
                    "MEILI_ENV=development".into(),
                ]),
                host_config: Some(HostConfig {
                    port_bindings: Some(port_bindings),
                    restart_policy: Some(bollard::models::RestartPolicy {
                        name: Some(bollard::models::RestartPolicyNameEnum::UNLESS_STOPPED),
                        maximum_retry_count: None,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };

            docker
                .create_container(
                    Some(CreateContainerOptions {
                        name: MEILISEARCH_CONTAINER,
                        ..Default::default()
                    }),
                    container_config,
                )
                .await
                .map_err(|e| format!("Failed to create MeiliSearch container: {}", e))?;
        }

        if !self.container_running(MEILISEARCH_CONTAINER).await? {
            tracing::info!("Starting MeiliSearch...");
            docker
                .start_container::<String>(MEILISEARCH_CONTAINER, None)
                .await
                .map_err(|e| format!("Failed to start MeiliSearch: {}", e))?;
        }

        // --- NATS (conditional) ---
        if config.nats_enabled {
            if !self.container_exists(NATS_CONTAINER).await? {
                tracing::info!("Creating NATS container...");

                let mut port_bindings = HashMap::new();
                port_bindings.insert(
                    format!("{}/tcp", NATS_PORT),
                    Some(vec![PortBinding {
                        host_ip: Some("127.0.0.1".into()),
                        host_port: Some(NATS_PORT.to_string()),
                    }]),
                );

                let container_config = ContainerConfig {
                    image: Some(NATS_IMAGE.to_string()),
                    env: Some(vec![]),
                    host_config: Some(HostConfig {
                        port_bindings: Some(port_bindings),
                        restart_policy: Some(bollard::models::RestartPolicy {
                            name: Some(bollard::models::RestartPolicyNameEnum::UNLESS_STOPPED),
                            maximum_retry_count: None,
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                };

                docker
                    .create_container(
                        Some(CreateContainerOptions {
                            name: NATS_CONTAINER,
                            ..Default::default()
                        }),
                        container_config,
                    )
                    .await
                    .map_err(|e| format!("Failed to create NATS container: {}", e))?;
            }

            if !self.container_running(NATS_CONTAINER).await? {
                tracing::info!("Starting NATS...");
                docker
                    .start_container::<String>(NATS_CONTAINER, None)
                    .await
                    .map_err(|e| format!("Failed to start NATS: {}", e))?;
            }
        } else {
            tracing::info!("NATS disabled — skipping container");
        }

        tracing::info!("All Docker services started");
        Ok(())
    }

    /// Check health of all services. When `nats_enabled` is false, NATS reports as Disabled.
    pub async fn check_health(&self, nats_enabled: bool) -> ServicesHealth {
        let docker_available = self.is_available().await;

        if !docker_available {
            return ServicesHealth {
                neo4j: ServiceStatus::Unknown,
                meilisearch: ServiceStatus::Unknown,
                nats: if nats_enabled {
                    ServiceStatus::Unknown
                } else {
                    ServiceStatus::Disabled
                },
                docker_available: false,
            };
        }

        let neo4j = self.check_container_health(NEO4J_CONTAINER, 7687).await;
        let meilisearch = self
            .check_container_health_http(MEILISEARCH_CONTAINER, 7700)
            .await;
        let nats = if nats_enabled {
            self.check_container_health(NATS_CONTAINER, NATS_PORT).await
        } else {
            ServiceStatus::Disabled
        };

        ServicesHealth {
            neo4j,
            meilisearch,
            nats,
            docker_available: true,
        }
    }

    async fn check_container_health(&self, name: &str, port: u16) -> ServiceStatus {
        match self.container_running(name).await {
            Ok(true) => {
                // Try TCP connection to verify the service is actually ready
                match tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port)).await {
                    Ok(_) => ServiceStatus::Healthy,
                    Err(_) => ServiceStatus::Starting,
                }
            }
            Ok(false) => ServiceStatus::NotRunning,
            Err(_) => ServiceStatus::Unknown,
        }
    }

    async fn check_container_health_http(&self, name: &str, port: u16) -> ServiceStatus {
        match self.container_running(name).await {
            Ok(true) => {
                let url = format!("http://127.0.0.1:{}/health", port);
                match reqwest::get(&url).await {
                    Ok(resp) if resp.status().is_success() => ServiceStatus::Healthy,
                    _ => ServiceStatus::Starting,
                }
            }
            Ok(false) => ServiceStatus::NotRunning,
            Err(_) => ServiceStatus::Unknown,
        }
    }

    /// Stop both services gracefully.
    pub async fn stop_services(&self) -> Result<(), String> {
        let docker = self.docker()?;

        for name in [NEO4J_CONTAINER, MEILISEARCH_CONTAINER, NATS_CONTAINER] {
            if self.container_running(name).await.unwrap_or(false) {
                tracing::info!("Stopping {}...", name);
                let opts = StopContainerOptions { t: 10 };
                if let Err(e) = docker.stop_container(name, Some(opts)).await {
                    tracing::warn!("Failed to stop {}: {}", name, e);
                }
            }
        }

        tracing::info!("All Docker services stopped");
        Ok(())
    }

    /// Get recent logs from a container.
    pub async fn get_logs(&self, service: &str, tail: u64) -> Result<Vec<String>, String> {
        let docker = self.docker()?;

        let name = match service {
            "neo4j" => NEO4J_CONTAINER,
            "meilisearch" => MEILISEARCH_CONTAINER,
            "nats" => NATS_CONTAINER,
            _ => return Err(format!("Unknown service: {}", service)),
        };

        let opts = LogsOptions::<String> {
            stdout: true,
            stderr: true,
            tail: tail.to_string(),
            ..Default::default()
        };

        let mut stream = docker.logs(name, Some(opts));
        let mut lines = Vec::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(output) => lines.push(output.to_string()),
                Err(e) => return Err(format!("Failed to get logs: {}", e)),
            }
        }

        Ok(lines)
    }
}

// ============================================================================
// Global state (shared across Tauri commands)
// ============================================================================

pub type SharedDockerManager = Arc<RwLock<DockerManager>>;

pub fn create_docker_manager() -> SharedDockerManager {
    Arc::new(RwLock::new(DockerManager::new()))
}

// ============================================================================
// Tauri commands
// ============================================================================

/// Check if Docker is installed and accessible.
#[tauri::command]
pub async fn check_docker(
    docker: tauri::State<'_, SharedDockerManager>,
) -> Result<bool, String> {
    let mgr = docker.read().await;
    Ok(mgr.is_available().await)
}

/// Start Docker services (Neo4j + MeiliSearch).
#[tauri::command]
pub async fn start_docker_services(
    docker: tauri::State<'_, SharedDockerManager>,
    config: DockerConfig,
) -> Result<(), String> {
    let mgr = docker.read().await;
    mgr.start_services(&config).await
}

/// Get health status of Docker services.
#[tauri::command]
pub async fn check_services_health(
    docker: tauri::State<'_, SharedDockerManager>,
    nats_enabled: Option<bool>,
) -> Result<ServicesHealth, String> {
    let mgr = docker.read().await;
    Ok(mgr.check_health(nats_enabled.unwrap_or(true)).await)
}

/// Stop all Docker services.
#[tauri::command]
pub async fn stop_docker_services(
    docker: tauri::State<'_, SharedDockerManager>,
) -> Result<(), String> {
    let mgr = docker.read().await;
    mgr.stop_services().await
}

/// Get logs from a service.
#[tauri::command]
pub async fn get_service_logs(
    docker: tauri::State<'_, SharedDockerManager>,
    service: String,
    tail: Option<u64>,
) -> Result<Vec<String>, String> {
    let mgr = docker.read().await;
    mgr.get_logs(&service, tail.unwrap_or(100)).await
}

/// Test connectivity to a service.
///
/// Supports three service types:
/// - `neo4j`: TCP connect to the bolt port (extracted from bolt://host:port URL)
/// - `meilisearch`: HTTP GET /health
/// - `nats`: TCP connect to the NATS port (extracted from nats://host:port URL)
///
/// Returns `true` if the connection succeeds, `false` otherwise.
/// Timeout: 5 seconds.
#[tauri::command]
pub async fn test_connection(service: String, url: String) -> Result<bool, String> {
    let timeout = std::time::Duration::from_secs(5);

    match service.as_str() {
        "neo4j" => {
            // Parse bolt://host:port → TCP connect
            let addr = parse_host_port(&url, 7687)?;
            tracing::info!("Testing Neo4j connection to {}...", addr);
            match tokio::time::timeout(timeout, tokio::net::TcpStream::connect(&addr)).await {
                Ok(Ok(_)) => Ok(true),
                Ok(Err(e)) => {
                    tracing::warn!("Neo4j connection failed: {}", e);
                    Ok(false)
                }
                Err(_) => {
                    tracing::warn!("Neo4j connection timed out");
                    Ok(false)
                }
            }
        }
        "meilisearch" => {
            // HTTP GET /health
            let health_url = format!("{}/health", url.trim_end_matches('/'));
            tracing::info!("Testing MeiliSearch connection to {}...", health_url);
            let client = reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .map_err(|e| format!("HTTP client error: {}", e))?;
            match client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => Ok(true),
                Ok(resp) => {
                    tracing::warn!("MeiliSearch returned status {}", resp.status());
                    Ok(false)
                }
                Err(e) => {
                    tracing::warn!("MeiliSearch connection failed: {}", e);
                    Ok(false)
                }
            }
        }
        "nats" => {
            // Parse nats://host:port → TCP connect
            let addr = parse_host_port(&url, 4222)?;
            tracing::info!("Testing NATS connection to {}...", addr);
            match tokio::time::timeout(timeout, tokio::net::TcpStream::connect(&addr)).await {
                Ok(Ok(_)) => Ok(true),
                Ok(Err(e)) => {
                    tracing::warn!("NATS connection failed: {}", e);
                    Ok(false)
                }
                Err(_) => {
                    tracing::warn!("NATS connection timed out");
                    Ok(false)
                }
            }
        }
        _ => Err(format!("Unknown service: {}", service)),
    }
}

/// Parse a URL like `bolt://host:port` or `nats://host:port` into `host:port`.
/// Falls back to the given default port if the URL has no port.
fn parse_host_port(url: &str, default_port: u16) -> Result<String, String> {
    // Try to parse as a URL
    if let Ok(parsed) = url::Url::parse(url) {
        let host = parsed.host_str().unwrap_or("localhost");
        let port = parsed.port().unwrap_or(default_port);
        return Ok(format!("{}:{}", host, port));
    }
    // Fallback: treat as host:port or just host
    if url.contains(':') {
        Ok(url.to_string())
    } else {
        Ok(format!("{}:{}", url, default_port))
    }
}
