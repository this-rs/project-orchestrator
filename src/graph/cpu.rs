//! Where the graph analytics burn CPU.
//!
//! The algorithms (PageRank, betweenness, Louvain, clustering, DNA, fingerprints, WL
//! hashes, context cards) are synchronous and last seconds to minutes on a large project.
//! Called from an `async fn` they hold a tokio worker for that whole time, and
//! rustworkx's betweenness fans out on rayon's GLOBAL pool, one thread per core. On a
//! project of 31 000 functions that froze the server (HTTP, WebSocket and the chat
//! enrichment stages timing out) for 5 to 7 minutes per pass.
//!
//! [`run_cpu`] is the one way in: the closure runs on a blocking thread (never a tokio
//! worker), inside a dedicated and BOUNDED rayon pool, so the analytics can never take
//! more than `ANALYTICS_THREADS` cores (default: a quarter of them, at least 2) and the
//! rest of the server keeps running.
use crate::neo4j::traits::*;

use anyhow::{anyhow, Result};
use std::sync::OnceLock;

/// Environment variable setting the analytics pool size.
pub const ANALYTICS_THREADS_ENV: &str = "ANALYTICS_THREADS";

/// Pool size when nothing is configured: a quarter of the cores, at least 2.
pub fn default_threads(cpus: usize) -> usize {
    (cpus / 4).max(2)
}

fn configured_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    std::env::var(ANALYTICS_THREADS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| default_threads(cpus))
}

fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = configured_threads();
        tracing::info!(threads, "analytics CPU pool initialised");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("po-analytics-{i}"))
            .build()
            .expect("the analytics rayon pool can be built")
    })
}

/// Runs `work` on a blocking thread, inside the bounded analytics pool. A panic in `work`
/// comes back as an error instead of taking the caller down.
pub async fn run_cpu<T, F>(work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || pool().install(work))
        .await
        .map_err(|e| anyhow!("analytics computation failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    #[test]
    fn the_default_pool_is_a_quarter_of_the_cores_and_never_below_two() {
        assert_eq!(default_threads(1), 2);
        assert_eq!(default_threads(8), 2);
        assert_eq!(default_threads(16), 4);
        assert_eq!(default_threads(64), 16);
    }

    #[tokio::test]
    async fn the_work_runs_off_the_async_workers_in_the_analytics_pool() {
        let name = run_cpu(|| std::thread::current().name().map(str::to_owned))
            .await
            .unwrap();
        assert!(
            name.as_deref()
                .is_some_and(|n| n.starts_with("po-analytics-")),
            "ran on {name:?}"
        );
    }

    #[tokio::test]
    async fn a_panic_in_the_work_is_an_error_not_a_crash() {
        let result = run_cpu(|| -> u32 { panic!("boom") }).await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_long_computation_does_not_stall_the_async_runtime() {
        let long = tokio::spawn(run_cpu(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            1u8
        }));
        // While it runs, a timer on the same runtime keeps its beat.
        let started = std::time::Instant::now();
        for _ in 0..5 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(started.elapsed() < std::time::Duration::from_millis(400));
        assert_eq!(long.await.unwrap().unwrap(), 1);
    }
}
