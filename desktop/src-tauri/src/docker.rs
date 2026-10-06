//! Docker management for the desktop application.
//!
//! Uses bollard to manage Neo4j and MeiliSearch containers directly
//! from the Tauri app, without requiring docker-compose.

// bollard 0.21 reorganized its API: the container-creation body moved to
// `models::ContainerCreateBody` (was `container::Config`) and every operation's
// options became a builder under `query_parameters` (was a plain struct under
// `container`/`image`). We alias ContainerCreateBody back to ContainerConfig so
// the struct-literal config blocks below stay unchanged.
use bollard::models::{ContainerCreateBody as ContainerConfig, HostConfig, PortBinding};
use bollard::query_parameters::{
    CreateContainerOptionsBuilder, CreateImageOptionsBuilder, ListContainersOptionsBuilder,
    LogsOptionsBuilder, StartContainerOptions, StopContainerOptionsBuilder,
};
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

/// Overall Docker daemon status — distinguishes "not installed" from "installed but not running".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DockerStatus {
    /// Docker daemon is reachable and responding to ping.
    Running,
    /// Docker binary/app is present on disk but the daemon is not running
    /// (e.g. Docker Desktop is installed but closed).
    Installed,
    /// No trace of Docker on this machine.
    NotInstalled,
}

impl std::fmt::Display for DockerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DockerStatus::Running => write!(f, "running"),
            DockerStatus::Installed => write!(f, "installed"),
            DockerStatus::NotInstalled => write!(f, "not_installed"),
        }
    }
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

/// How long one candidate endpoint gets to answer a ping. A Docker Desktop that is frozen
/// accepts the connection and never answers; without a bound the whole check hangs with it.
const PING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

pub struct DockerManager {
    /// The connection that last answered. Never trusted blindly: it is pinged again, and
    /// replaced when it stops answering or another runtime comes up.
    current: std::sync::Mutex<Option<Docker>>,
    /// Endpoints to try instead of the machine's own (tests).
    endpoints_override: Option<Vec<String>>,
}

impl DockerManager {
    /// A manager that finds Docker when it is asked, not once at launch.
    ///
    /// It used to connect once, at startup. When Docker Desktop was not running yet (exactly the
    /// case in which the splash offers to open it) there was no socket to connect to, the manager
    /// kept "no Docker" for the life of the app, and `status()` stayed `Installed` after Docker
    /// had come up: the splash waited for a Docker that was already running.
    pub fn new() -> Self {
        Self {
            current: std::sync::Mutex::new(None),
            endpoints_override: None,
        }
    }

    #[cfg(test)]
    fn with_endpoints(endpoints: Vec<String>) -> Self {
        Self {
            current: std::sync::Mutex::new(None),
            endpoints_override: Some(endpoints),
        }
    }

    fn candidates(&self) -> Vec<String> {
        match &self.endpoints_override {
            Some(list) => list.clone(),
            None => docker_endpoints(),
        }
    }

    /// A connection to a Docker that answers right now, or `None`.
    async fn resolve(&self) -> Option<Docker> {
        let cached = self
            .current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(docker) = cached {
            if answers(&docker).await {
                return Some(docker);
            }
            *self
                .current
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        }
        for endpoint in self.candidates() {
            let Some(docker) = connect(&endpoint) else {
                continue;
            };
            if answers(&docker).await {
                tracing::info!("Docker answers on {}", endpoint);
                *self
                    .current
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(docker.clone());
                return Some(docker);
            }
            tracing::debug!("Docker endpoint {} exists but does not answer", endpoint);
        }
        None
    }

    /// Check if Docker daemon is reachable.
    pub async fn is_available(&self) -> bool {
        self.resolve().await.is_some()
    }

    /// Return fine-grained Docker status: Running, Installed, or NotInstalled.
    ///
    /// - `Running` → a daemon answers a ping (on any known endpoint)
    /// - `Installed` → a socket or an app/CLI is on disk but nothing answers (not started, or frozen)
    /// - `NotInstalled` → no trace of Docker on this machine
    pub async fn status(&self) -> DockerStatus {
        if self.resolve().await.is_some() {
            return DockerStatus::Running;
        }
        let socket_present = self.candidates().iter().any(|e| endpoint_exists(e));
        if socket_present || Self::is_docker_installed_on_disk() {
            DockerStatus::Installed
        } else {
            DockerStatus::NotInstalled
        }
    }

    /// Check if Docker binary or app bundle exists on disk (without requiring the daemon).
    fn is_docker_installed_on_disk() -> bool {
        #[cfg(target_os = "macos")]
        {
            let home = dirs::home_dir().unwrap_or_default();
            // Docker Desktop, OrbStack, Rancher Desktop, Podman Desktop; system and per-user.
            let apps = [
                "Docker.app",
                "OrbStack.app",
                "Rancher Desktop.app",
                "Podman Desktop.app",
            ];
            for app in apps {
                if std::path::Path::new("/Applications").join(app).exists()
                    || home.join("Applications").join(app).exists()
                {
                    return true;
                }
            }
            // The CLI. A GUI app's PATH is /usr/bin:/bin:/usr/sbin:/sbin, which does NOT contain
            // Homebrew, Colima or OrbStack: `which docker` said "not installed" for all of them.
            let dirs = [
                std::path::PathBuf::from("/usr/local/bin"),
                std::path::PathBuf::from("/opt/homebrew/bin"),
                std::path::PathBuf::from("/usr/bin"),
                home.join(".orbstack/bin"),
                home.join(".rd/bin"),
                home.join(".docker/bin"),
                home.join(".colima/bin"),
            ];
            dirs.iter().any(|dir| dir.join("docker").exists())
        }

        #[cfg(target_os = "linux")]
        {
            std::process::Command::new("which")
                .arg("docker")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
                || std::path::Path::new("/usr/bin/docker").exists()
        }

        #[cfg(target_os = "windows")]
        {
            // Check for Docker Desktop in standard install locations
            let program_files = std::env::var("ProgramFiles").unwrap_or_default();
            if std::path::Path::new(&format!(
                "{}\\Docker\\Docker\\Docker Desktop.exe",
                program_files
            ))
            .exists()
            {
                return true;
            }
            // Also check PATH
            std::process::Command::new("where")
                .arg("docker")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            false
        }
    }

    /// A connection to Docker, found now.
    async fn docker(&self) -> Result<Docker, String> {
        self.resolve()
            .await
            .ok_or_else(|| "Docker is not available".to_string())
    }

    /// Pull an image if it's not already present locally.
    async fn ensure_image(&self, image: &str) -> Result<(), String> {
        let docker = self.docker().await?;

        // Check if image exists locally
        if docker.inspect_image(image).await.is_ok() {
            tracing::info!("Image {} already present", image);
            return Ok(());
        }

        tracing::info!("Pulling image {}...", image);
        let opts = CreateImageOptionsBuilder::default()
            .from_image(image)
            .build();

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
        let docker = self.docker().await?;
        let mut filters: HashMap<String, Vec<String>> = HashMap::new();
        filters.insert("name".to_string(), vec![name.to_string()]);

        let opts = ListContainersOptionsBuilder::default()
            .all(true)
            .filters(&filters)
            .build();

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
        let docker = self.docker().await?;
        match docker.inspect_container(name, None).await {
            Ok(info) => Ok(info.state.and_then(|s| s.running).unwrap_or(false)),
            Err(_) => Ok(false),
        }
    }

    /// Start Neo4j, MeiliSearch, and optionally NATS containers.
    pub async fn start_services(&self, config: &DockerConfig) -> Result<(), String> {
        let docker = self.docker().await?;

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
                    Some(
                        CreateContainerOptionsBuilder::default()
                            .name(NEO4J_CONTAINER)
                            .build(),
                    ),
                    container_config,
                )
                .await
                .map_err(|e| format!("Failed to create Neo4j container: {}", e))?;
        }

        if !self.container_running(NEO4J_CONTAINER).await? {
            tracing::info!("Starting Neo4j...");
            docker
                .start_container(NEO4J_CONTAINER, None::<StartContainerOptions>)
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
                    Some(
                        CreateContainerOptionsBuilder::default()
                            .name(MEILISEARCH_CONTAINER)
                            .build(),
                    ),
                    container_config,
                )
                .await
                .map_err(|e| format!("Failed to create MeiliSearch container: {}", e))?;
        }

        if !self.container_running(MEILISEARCH_CONTAINER).await? {
            tracing::info!("Starting MeiliSearch...");
            docker
                .start_container(MEILISEARCH_CONTAINER, None::<StartContainerOptions>)
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
                        Some(
                            CreateContainerOptionsBuilder::default()
                                .name(NATS_CONTAINER)
                                .build(),
                        ),
                        container_config,
                    )
                    .await
                    .map_err(|e| format!("Failed to create NATS container: {}", e))?;
            }

            if !self.container_running(NATS_CONTAINER).await? {
                tracing::info!("Starting NATS...");
                docker
                    .start_container(NATS_CONTAINER, None::<StartContainerOptions>)
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
        let docker = self.docker().await?;

        for name in [NEO4J_CONTAINER, MEILISEARCH_CONTAINER, NATS_CONTAINER] {
            if self.container_running(name).await.unwrap_or(false) {
                tracing::info!("Stopping {}...", name);
                let opts = StopContainerOptionsBuilder::default().t(10).build();
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
        let docker = self.docker().await?;

        let name = match service {
            "neo4j" => NEO4J_CONTAINER,
            "meilisearch" => MEILISEARCH_CONTAINER,
            "nats" => NATS_CONTAINER,
            _ => return Err(format!("Unknown service: {}", service)),
        };

        let opts = LogsOptionsBuilder::default()
            .stdout(true)
            .stderr(true)
            .tail(&tail.to_string())
            .build();

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
// Finding Docker
// ============================================================================

/// Every place a Docker daemon may listen on this machine, most specific first, without
/// repeats. A GUI app does not inherit the shell's `DOCKER_HOST`, and `docker context` may point
/// at OrbStack or Colima, so the well-known sockets of the common runtimes are listed too.
pub(crate) fn docker_endpoints() -> Vec<String> {
    endpoints_for(
        std::env::var("DOCKER_HOST").ok().as_deref(),
        dirs::home_dir().as_deref(),
        std::env::var("XDG_RUNTIME_DIR").ok().as_deref(),
    )
}

fn endpoints_for(
    docker_host: Option<&str>,
    home: Option<&std::path::Path>,
    xdg_runtime_dir: Option<&str>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // Only local endpoints: a tcp:// or ssh:// host is somebody else's daemon.
    let mut push = |endpoint: String| {
        if (endpoint.starts_with("unix://") || endpoint.starts_with("npipe://"))
            && !out.contains(&endpoint)
        {
            out.push(endpoint);
        }
    };
    if let Some(host) = docker_host.filter(|h| !h.trim().is_empty()) {
        push(host.trim().to_string());
    }
    if let Some(home) = home {
        if let Some(host) = current_context_host(home) {
            push(host);
        }
    }
    #[cfg(windows)]
    push("npipe:////./pipe/docker_engine".to_string());
    push("unix:///var/run/docker.sock".to_string());
    if let Some(home) = home {
        for relative in [
            ".docker/run/docker.sock",     // Docker Desktop (4.13+ default)
            ".docker/desktop/docker.sock", // Docker Desktop (older / alternative)
            ".orbstack/run/docker.sock",   // OrbStack
            ".colima/default/docker.sock", // Colima
            ".colima/docker.sock",
            ".rd/docker.sock",               // Rancher Desktop
            ".lima/docker/sock/docker.sock", // Lima
        ] {
            push(format!("unix://{}", home.join(relative).display()));
        }
    }
    push("unix:///var/run/docker.sock.raw".to_string());
    if let Some(dir) = xdg_runtime_dir.filter(|d| !d.is_empty()) {
        push(format!("unix://{dir}/docker.sock")); // rootless Docker on Linux
    }
    out
}

/// The endpoint of the Docker context selected with `docker context use`, if it is a local one.
/// `~/.docker/config.json` names the context; its endpoint is in
/// `~/.docker/contexts/meta/<hash>/meta.json`.
fn current_context_host(home: &std::path::Path) -> Option<String> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(home.join(".docker/config.json")).ok()?).ok()?;
    let name = config.get("currentContext")?.as_str()?;
    if name.is_empty() || name == "default" {
        return None;
    }
    for entry in std::fs::read_dir(home.join(".docker/contexts/meta"))
        .ok()?
        .flatten()
    {
        let Ok(bytes) = std::fs::read(entry.path().join("meta.json")) else {
            continue;
        };
        let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if meta.get("Name").and_then(|n| n.as_str()) == Some(name) {
            return meta
                .pointer("/Endpoints/docker/Host")
                .and_then(|h| h.as_str())
                .map(str::to_owned);
        }
    }
    None
}

fn endpoint_exists(endpoint: &str) -> bool {
    match endpoint.strip_prefix("unix://") {
        Some(path) => std::path::Path::new(path).exists(),
        // A named pipe cannot be told from a stale name without opening it.
        None => cfg!(windows),
    }
}

/// A client for `endpoint`, if there is anything there to connect to.
fn connect(endpoint: &str) -> Option<Docker> {
    if !endpoint_exists(endpoint) {
        return None;
    }
    #[cfg(windows)]
    if endpoint.starts_with("npipe://") {
        return Docker::connect_with_local_defaults().ok();
    }
    Docker::connect_with_socket(endpoint, 5, bollard::API_DEFAULT_VERSION).ok()
}

/// Whether the daemon answers a ping, within [`PING_TIMEOUT`].
async fn answers(docker: &Docker) -> bool {
    matches!(
        tokio::time::timeout(PING_TIMEOUT, docker.ping()).await,
        Ok(Ok(_))
    )
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

/// Response from the `check_docker` Tauri command.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckDockerResponse {
    /// Backward compat: true when Docker daemon is reachable.
    pub available: bool,
    /// Fine-grained status: "running", "installed", or "not_installed".
    pub status: String,
}

/// Check if Docker is installed and accessible.
///
/// Returns both a backward-compatible `available` bool and a fine-grained `status` string
/// ("running", "installed", "not_installed") so the splash screen can differentiate
/// "Docker not installed" from "Docker Desktop installed but not started".
#[tauri::command]
pub async fn check_docker(
    docker: tauri::State<'_, SharedDockerManager>,
) -> Result<CheckDockerResponse, String> {
    let mgr = docker.read().await;
    let status = mgr.status().await;
    Ok(CheckDockerResponse {
        available: status == DockerStatus::Running,
        status: status.to_string(),
    })
}

/// Attempt to launch Docker Desktop application.
///
/// On macOS: `open -a Docker`
/// On Linux: `systemctl --user start docker-desktop` (or just `docker` if available)
/// On Windows: starts Docker Desktop from Program Files
#[tauri::command]
pub async fn open_docker_desktop() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("-a")
            .arg("Docker")
            .spawn()
            .map_err(|e| format!("Failed to open Docker Desktop: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        // Try systemctl first, fall back to direct launch
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "start", "docker-desktop"])
            .spawn();
    }

    #[cfg(target_os = "windows")]
    {
        let program_files = std::env::var("ProgramFiles").unwrap_or_default();
        let docker_path = format!("{}\\Docker\\Docker\\Docker Desktop.exe", program_files);
        std::process::Command::new(&docker_path)
            .spawn()
            .map_err(|e| format!("Failed to open Docker Desktop: {}", e))?;
    }

    Ok(())
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
    // Kept for older frontends: the answer without the reason. See `net::test_connection_detailed`.
    let result = crate::net::test_service(&service, &url).await?;
    if !result.ok {
        tracing::warn!(
            "{} connection to {}:{} failed: {}",
            service,
            result.host,
            result.port,
            result.hint.as_deref().unwrap_or("no detail")
        );
    }
    Ok(result.ok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pd-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn unix(path: &Path) -> String {
        format!("unix://{}", path.display())
    }

    /// A daemon that answers `/_ping`, the way Docker does.
    fn serve_ping(path: &Path) -> tokio::task::JoinHandle<()> {
        let listener = tokio::net::UnixListener::bind(path).unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;
                    let _ = stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nApi-Version: 1.45\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nOK",
                        )
                        .await;
                });
            }
        })
    }

    /// A daemon that takes the connection and never answers: a frozen Docker Desktop.
    fn serve_frozen(path: &Path) -> tokio::task::JoinHandle<()> {
        let listener = tokio::net::UnixListener::bind(path).unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
            }
        })
    }

    #[test]
    fn endpoints_start_with_the_environment_then_the_context_then_the_well_known_sockets() {
        let home = scratch("endpoints");
        let list = endpoints_for(Some("unix:///custom/docker.sock"), Some(&home), None);
        assert_eq!(list[0], "unix:///custom/docker.sock");
        let at = |needle: &str| list.iter().position(|e| e.ends_with(needle)).unwrap();
        assert!(at("/var/run/docker.sock") > 0);
        for runtime in [
            ".docker/run/docker.sock",
            ".orbstack/run/docker.sock",
            ".colima/default/docker.sock",
            ".rd/docker.sock",
        ] {
            assert!(
                list.iter().any(|e| e.ends_with(runtime)),
                "{runtime} missing"
            );
        }
        let unique: std::collections::HashSet<_> = list.iter().collect();
        assert_eq!(unique.len(), list.len(), "no endpoint is listed twice");
    }

    #[test]
    fn a_remote_docker_host_is_never_used() {
        let home = scratch("remote");
        for host in ["tcp://10.0.0.5:2375", "ssh://user@server", "  "] {
            let list = endpoints_for(Some(host), Some(&home), None);
            assert!(
                !list
                    .iter()
                    .any(|e| e.contains("10.0.0.5") || e.starts_with("ssh")),
                "{host}"
            );
        }
    }

    #[test]
    fn the_selected_docker_context_is_followed() {
        let home = scratch("context");
        let meta = home.join(".docker/contexts/meta/abc123");
        std::fs::create_dir_all(&meta).unwrap();
        std::fs::write(
            home.join(".docker/config.json"),
            r#"{"currentContext":"colima"}"#,
        )
        .unwrap();
        std::fs::write(
            meta.join("meta.json"),
            r#"{"Name":"colima","Endpoints":{"docker":{"Host":"unix:///Users/me/.colima/default/docker.sock"}}}"#,
        )
        .unwrap();
        assert_eq!(
            current_context_host(&home).as_deref(),
            Some("unix:///Users/me/.colima/default/docker.sock")
        );
        // It comes before the defaults.
        assert_eq!(
            endpoints_for(None, Some(&home), None)[0],
            "unix:///Users/me/.colima/default/docker.sock"
        );
        // "default" and a missing config mean: no context to follow.
        std::fs::write(
            home.join(".docker/config.json"),
            r#"{"currentContext":"default"}"#,
        )
        .unwrap();
        assert_eq!(current_context_host(&home), None);
        assert_eq!(current_context_host(&scratch("nocontext")), None);
    }

    #[tokio::test]
    async fn docker_started_after_the_app_is_noticed() {
        // The splash offers to open Docker Desktop when it is not running. The manager used to
        // connect once at launch, found no socket, and never looked again.
        let dir = scratch("late");
        let socket = dir.join("docker.sock");
        let manager = DockerManager::with_endpoints(vec![unix(&socket)]);
        assert_ne!(
            manager.status().await,
            DockerStatus::Running,
            "nothing is listening yet"
        );

        let _daemon = serve_ping(&socket);
        assert_eq!(
            manager.status().await,
            DockerStatus::Running,
            "it came up: it must be seen"
        );
        assert!(manager.is_available().await);
    }

    #[tokio::test]
    async fn a_frozen_daemon_is_installed_not_running_and_the_check_returns() {
        let dir = scratch("frozen");
        let socket = dir.join("docker.sock");
        let _daemon = serve_frozen(&socket);
        let manager = DockerManager::with_endpoints(vec![unix(&socket)]);
        let started = std::time::Instant::now();
        let status = manager.status().await;
        assert_eq!(
            status,
            DockerStatus::Installed,
            "a socket is there, nobody answers"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(4),
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn the_first_endpoint_that_answers_wins_and_a_dead_one_before_it_is_skipped() {
        let dir = scratch("order");
        let frozen = dir.join("a.sock");
        let live = dir.join("b.sock");
        let missing = dir.join("c.sock");
        let _a = serve_frozen(&frozen);
        let _b = serve_ping(&live);
        let manager =
            DockerManager::with_endpoints(vec![unix(&missing), unix(&frozen), unix(&live)]);
        assert_eq!(manager.status().await, DockerStatus::Running);
    }

    #[tokio::test]
    async fn a_connection_that_stops_answering_is_replaced_by_one_that_does() {
        let dir = scratch("failover");
        let first = dir.join("first.sock");
        let second = dir.join("second.sock");
        let daemon = serve_ping(&first);
        let manager = DockerManager::with_endpoints(vec![unix(&first), unix(&second)]);
        assert_eq!(manager.status().await, DockerStatus::Running);

        // The first runtime goes away (its socket is removed), another comes up.
        daemon.abort();
        let _ = std::fs::remove_file(&first);
        let _second = serve_ping(&second);
        assert_eq!(manager.status().await, DockerStatus::Running);
    }
}
