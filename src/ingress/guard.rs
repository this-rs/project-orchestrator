//! Les garde-fous EN MÉMOIRE de la route publique : drapeau, taille, délai,
//! concurrence, débit par déclencheur, et compteur des refus qui précèdent
//! l'authentification (ces refus n'écrivent RIEN en base).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::Semaphore;
use uuid::Uuid;

/// Section `ingress:` de config.yaml. Éteinte par défaut : la route n'est pas montée.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct IngressConfig {
    /// Interrupteur général (`INGRESS_ENABLED=true|1`). Défaut : false.
    pub enabled: bool,
    /// Taille maximale d'un corps, en octets. Défaut : 1 MiB.
    pub max_body_bytes: usize,
    /// Délai de lecture du corps, en secondes. Défaut : 10.
    pub body_read_timeout_secs: u64,
    /// Requêtes traitées en même temps sur la route. Défaut : 32.
    pub max_concurrent: usize,
    /// Requêtes acceptées par minute et par déclencheur (seau à jetons). Défaut : 60.
    pub rate_per_minute: u32,
}

/// Plafond dur de `max_body_bytes` (GitHub plafonne lui-même à 25 MB).
pub const MAX_BODY_CEILING: usize = 25 * 1024 * 1024;

impl Default for IngressConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_body_bytes: 1024 * 1024,
            body_read_timeout_secs: 10,
            max_concurrent: 32,
            rate_per_minute: 60,
        }
    }
}

impl IngressConfig {
    /// Validation au démarrage (sans effet quand la route est éteinte).
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.max_body_bytes == 0 || self.max_body_bytes > MAX_BODY_CEILING {
            return Err(format!(
                "ingress.max_body_bytes must be in 1..={MAX_BODY_CEILING}"
            ));
        }
        if self.body_read_timeout_secs == 0 || self.max_concurrent == 0 || self.rate_per_minute == 0
        {
            return Err(
                "ingress.body_read_timeout_secs, max_concurrent and rate_per_minute must be > 0"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn body_read_timeout(&self) -> Duration {
        Duration::from_secs(self.body_read_timeout_secs)
    }
}

/// Plafond du nombre de déclencheurs suivis par les tables en mémoire : au-delà,
/// les entrées les plus anciennes sont oubliées (la mémoire reste bornée).
const MAX_TRACKED: usize = 4096;

/// Fenêtre du compteur de refus.
const REFUSAL_WINDOW: Duration = Duration::from_secs(60);

struct Bucket {
    tokens: f64,
    at: Instant,
}

#[derive(Default)]
struct Window {
    started: Option<Instant>,
    count: u64,
}

/// L'état partagé de la route (un par serveur).
pub struct IngressRuntime {
    pub config: IngressConfig,
    permits: Arc<Semaphore>,
    buckets: std::sync::Mutex<HashMap<Uuid, Bucket>>,
    refusals: std::sync::Mutex<HashMap<Uuid, Window>>,
}

impl std::fmt::Debug for IngressRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IngressRuntime")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Default for IngressRuntime {
    fn default() -> Self {
        Self::new(IngressConfig::default())
    }
}

/// La clé des refus qui ne visent aucun déclencheur connu (un identifiant
/// inconnu ne crée pas d'entrée : la table reste bornée par les déclencheurs réels).
pub const UNKNOWN_TRIGGER: Uuid = Uuid::nil();

impl IngressRuntime {
    pub fn new(config: IngressConfig) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(config.max_concurrent.max(1))),
            config,
            buckets: std::sync::Mutex::new(HashMap::new()),
            refusals: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Une place dans la limite de concurrence, ou `None` si elle est pleine
    /// (pas d'attente : la réponse est immédiate).
    pub fn try_enter(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.permits.clone().try_acquire_owned().ok()
    }

    /// Le seau à jetons du déclencheur accorde-t-il une requête de plus ?
    pub fn allow(&self, trigger_id: Uuid) -> bool {
        let capacity = f64::from(self.config.rate_per_minute.max(1));
        let per_sec = capacity / 60.0;
        let now = Instant::now();
        let mut buckets = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        if buckets.len() >= MAX_TRACKED && !buckets.contains_key(&trigger_id) {
            // Oublie les seaux pleins (inactifs depuis assez longtemps).
            buckets.retain(|_, b| {
                b.tokens + now.duration_since(b.at).as_secs_f64() * per_sec < capacity
            });
            if buckets.len() >= MAX_TRACKED {
                return false;
            }
        }
        let bucket = buckets.entry(trigger_id).or_insert(Bucket {
            tokens: capacity,
            at: now,
        });
        bucket.tokens =
            (bucket.tokens + now.duration_since(bucket.at).as_secs_f64() * per_sec).min(capacity);
        bucket.at = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Compte un refus qui précède l'authentification : en mémoire seulement, par
    /// déclencheur et par fenêtre d'une minute, avec une trace `debug` sans secret.
    /// Rend le nombre de refus de la fenêtre courante pour ce déclencheur.
    pub fn refused(&self, trigger_id: Uuid, reason: &'static str) -> u64 {
        tracing::debug!(trigger_id = %trigger_id, reason, "external event refused before authentication");
        let now = Instant::now();
        let mut map = self.refusals.lock().unwrap_or_else(|p| p.into_inner());
        if map.len() >= MAX_TRACKED && !map.contains_key(&trigger_id) {
            map.retain(|_, w| {
                w.started
                    .is_some_and(|s| now.duration_since(s) < REFUSAL_WINDOW)
            });
        }
        let key = if map.len() >= MAX_TRACKED && !map.contains_key(&trigger_id) {
            UNKNOWN_TRIGGER
        } else {
            trigger_id
        };
        let window = map.entry(key).or_default();
        match window.started {
            Some(s) if now.duration_since(s) < REFUSAL_WINDOW => {}
            _ => {
                if window.count > 0 {
                    tracing::info!(trigger_id = %key, refusals = window.count, "external event refusals in the last window");
                }
                window.started = Some(now);
                window.count = 0;
            }
        }
        window.count += 1;
        window.count
    }

    /// Refus de la fenêtre courante pour un déclencheur (tests, diagnostic).
    pub fn refusals_in_window(&self, trigger_id: Uuid) -> u64 {
        let map = self.refusals.lock().unwrap_or_else(|p| p.into_inner());
        map.get(&trigger_id)
            .filter(|w| {
                w.started
                    .is_some_and(|s| Instant::now().duration_since(s) < REFUSAL_WINDOW)
            })
            .map_or(0, |w| w.count)
    }

    /// Nombre de déclencheurs suivis par le compteur de refus (borne mémoire).
    pub fn tracked_refusal_keys(&self) -> usize {
        self.refusals
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_by_default_with_the_documented_limits() {
        let c = IngressConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.max_body_bytes, 1024 * 1024);
        assert_eq!(c.body_read_timeout_secs, 10);
        assert_eq!(c.max_concurrent, 32);
        assert_eq!(c.rate_per_minute, 60);
        assert!(c.validate().is_ok());
        let bad = IngressConfig {
            enabled: true,
            max_body_bytes: MAX_BODY_CEILING + 1,
            ..IngressConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn the_bucket_refuses_past_its_rate_per_trigger() {
        let rt = IngressRuntime::new(IngressConfig {
            rate_per_minute: 3,
            ..IngressConfig::default()
        });
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        assert!(rt.allow(a) && rt.allow(a) && rt.allow(a));
        assert!(!rt.allow(a), "the fourth in the minute is refused");
        assert!(rt.allow(b), "another trigger has its own bucket");
    }

    #[test]
    fn the_refusal_counter_stays_bounded() {
        let rt = IngressRuntime::default();
        for _ in 0..(MAX_TRACKED + 100) {
            rt.refused(Uuid::new_v4(), "invalid_signature");
        }
        // The real triggers, plus the one shared bucket of the overflow.
        assert!(rt.tracked_refusal_keys() <= MAX_TRACKED + 1);
        assert!(rt.refusals_in_window(UNKNOWN_TRIGGER) >= 100);
    }

    #[test]
    fn the_concurrency_limit_is_immediate() {
        let rt = IngressRuntime::new(IngressConfig {
            max_concurrent: 1,
            ..IngressConfig::default()
        });
        let held = rt.try_enter();
        assert!(held.is_some());
        assert!(rt.try_enter().is_none());
        drop(held);
        assert!(rt.try_enter().is_some());
    }
}
