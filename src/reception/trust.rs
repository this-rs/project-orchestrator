//! EigenTrust-adapted peer trust scoring.
//!
//! Maintains per-peer trust scores that combine local observations with
//! network-propagated trust, using an EigenTrust-inspired formula:
//! `global_trust = α × local_trust + (1 - α) × network_trust`

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default trust score for unknown peers.
pub const DEFAULT_TRUST: f64 = 0.5;

/// Trust record for a single peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerTrust {
    /// DID:key of the peer.
    pub did_key: String,
    /// Locally-computed trust based on direct interactions.
    pub local_trust: f64,
    /// Trust propagated from the network (other peers' assessments).
    pub network_trust: f64,
    /// Computed global trust: α × local + (1-α) × network.
    pub global_trust: f64,
    /// Number of interactions observed with this peer.
    pub interactions: u32,
    /// When this peer was last seen / interacted with.
    pub last_seen: DateTime<Utc>,
}

/// Manages trust scores for all known peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustManager {
    /// Per-peer trust records, keyed by DID.
    peers: HashMap<String, PeerTrust>,
    /// EigenTrust alpha parameter: weight of local trust vs network trust.
    alpha: f64,
}

impl Default for TrustManager {
    fn default() -> Self {
        Self::new(0.7)
    }
}

impl TrustManager {
    /// Create a new trust manager with the given alpha parameter.
    ///
    /// Alpha controls the balance between local and network trust:
    /// - `alpha = 1.0` → only local trust matters
    /// - `alpha = 0.0` → only network trust matters
    /// - Typical: `alpha = 0.7` (favor local observations)
    pub fn new(alpha: f64) -> Self {
        Self {
            peers: HashMap::new(),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Update local trust for a peer based on a quality observation.
    ///
    /// Uses a rolling average weighted by interaction count.
    pub fn update_local_trust(&mut self, did_key: &str, quality_score: f64) {
        let quality_score = quality_score.clamp(0.0, 1.0);

        let peer = self
            .peers
            .entry(did_key.to_string())
            .or_insert_with(|| PeerTrust {
                did_key: did_key.to_string(),
                local_trust: DEFAULT_TRUST,
                network_trust: DEFAULT_TRUST,
                global_trust: DEFAULT_TRUST,
                interactions: 0,
                last_seen: Utc::now(),
            });

        peer.interactions += 1;
        peer.last_seen = Utc::now();

        // Rolling average: new_trust = ((old_trust * (n-1)) + quality) / n
        let n = peer.interactions as f64;
        peer.local_trust = ((peer.local_trust * (n - 1.0)) + quality_score) / n;

        // Recompute global trust
        peer.global_trust = self.alpha * peer.local_trust + (1.0 - self.alpha) * peer.network_trust;
    }

    /// Update the network trust for a peer (from external attestations).
    pub fn update_network_trust(&mut self, did_key: &str, network_score: f64) {
        let network_score = network_score.clamp(0.0, 1.0);

        let peer = self
            .peers
            .entry(did_key.to_string())
            .or_insert_with(|| PeerTrust {
                did_key: did_key.to_string(),
                local_trust: DEFAULT_TRUST,
                network_trust: DEFAULT_TRUST,
                global_trust: DEFAULT_TRUST,
                interactions: 0,
                last_seen: Utc::now(),
            });

        peer.network_trust = network_score;
        peer.last_seen = Utc::now();

        // Recompute global trust
        peer.global_trust = self.alpha * peer.local_trust + (1.0 - self.alpha) * peer.network_trust;
    }

    /// Recompute global trust for all peers using current alpha.
    pub fn compute_global_trust(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.0, 1.0);
        for peer in self.peers.values_mut() {
            peer.global_trust =
                self.alpha * peer.local_trust + (1.0 - self.alpha) * peer.network_trust;
        }
    }

    /// Get the global trust score for a peer, or [`DEFAULT_TRUST`] if unknown.
    pub fn get_trust(&self, did_key: &str) -> f64 {
        self.peers
            .get(did_key)
            .map(|p| p.global_trust)
            .unwrap_or(DEFAULT_TRUST)
    }

    /// Get the full trust record for a peer, if known.
    pub fn get_peer(&self, did_key: &str) -> Option<&PeerTrust> {
        self.peers.get(did_key)
    }

    /// Decay trust for peers not seen within `threshold_days`.
    ///
    /// Applies a 10% decay per call for inactive peers, with a floor of 0.1.
    pub fn decay_inactive(&mut self, threshold_days: i64) {
        let cutoff = Utc::now() - chrono::Duration::days(threshold_days);
        let decay_factor = 0.9;
        let floor = 0.1;

        for peer in self.peers.values_mut() {
            if peer.last_seen < cutoff {
                peer.local_trust = (peer.local_trust * decay_factor).max(floor);
                peer.network_trust = (peer.network_trust * decay_factor).max(floor);
                peer.global_trust =
                    self.alpha * peer.local_trust + (1.0 - self.alpha) * peer.network_trust;
            }
        }
    }

    /// Return the number of known peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Export trust scores as a HashMap suitable for [`super::score::LocalContext`].
    pub fn export_trust_map(&self) -> HashMap<String, f64> {
        self.peers
            .iter()
            .map(|(k, v)| (k.clone(), v.global_trust))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_trust_for_unknown() {
        let mgr = TrustManager::default();
        assert!((mgr.get_trust("did:key:zUnknown") - DEFAULT_TRUST).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_local_trust_single() {
        let mut mgr = TrustManager::new(1.0); // alpha=1 → only local matters
        mgr.update_local_trust("did:key:zPeerA", 0.9);
        let trust = mgr.get_trust("did:key:zPeerA");
        assert!((trust - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_update_local_trust_rolling_average() {
        let mut mgr = TrustManager::new(1.0);
        mgr.update_local_trust("did:key:zPeerA", 1.0);
        mgr.update_local_trust("did:key:zPeerA", 0.0);
        // After 2 interactions: (1.0 + 0.0) / 2 = ~0.5
        // But rolling: first sets to 1.0 (n=1), second: (1.0*1 + 0.0)/2 = 0.5
        let trust = mgr.get_trust("did:key:zPeerA");
        assert!((trust - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_network_trust_contribution() {
        let mut mgr = TrustManager::new(0.5); // equal weight
        mgr.update_local_trust("did:key:zPeerA", 0.8);
        mgr.update_network_trust("did:key:zPeerA", 0.4);
        // global = 0.5 * 0.8 + 0.5 * 0.4 = 0.6
        let trust = mgr.get_trust("did:key:zPeerA");
        assert!((trust - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_compute_global_trust_recomputes() {
        let mut mgr = TrustManager::new(0.7);
        mgr.update_local_trust("did:key:zPeerA", 1.0);
        mgr.update_network_trust("did:key:zPeerA", 0.0);
        // global = 0.7*1.0 + 0.3*0.0 = 0.7
        assert!((mgr.get_trust("did:key:zPeerA") - 0.7).abs() < 0.01);

        // Change alpha to 0.5 and recompute
        mgr.compute_global_trust(0.5);
        // global = 0.5*1.0 + 0.5*0.0 = 0.5
        assert!((mgr.get_trust("did:key:zPeerA") - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_decay_inactive() {
        let mut mgr = TrustManager::new(1.0);
        mgr.update_local_trust("did:key:zOldPeer", 0.9);

        // Manually backdate last_seen
        if let Some(peer) = mgr.peers.get_mut("did:key:zOldPeer") {
            peer.last_seen = Utc::now() - chrono::Duration::days(60);
        }

        let before = mgr.get_trust("did:key:zOldPeer");
        mgr.decay_inactive(30);
        let after = mgr.get_trust("did:key:zOldPeer");
        assert!(after < before, "Trust should decay for inactive peer");
    }

    #[test]
    fn test_decay_does_not_affect_active() {
        let mut mgr = TrustManager::new(1.0);
        mgr.update_local_trust("did:key:zActivePeer", 0.9);

        let before = mgr.get_trust("did:key:zActivePeer");
        mgr.decay_inactive(30);
        let after = mgr.get_trust("did:key:zActivePeer");
        assert!((after - before).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peer_count() {
        let mut mgr = TrustManager::default();
        assert_eq!(mgr.peer_count(), 0);
        mgr.update_local_trust("did:key:zA", 0.5);
        mgr.update_local_trust("did:key:zB", 0.5);
        assert_eq!(mgr.peer_count(), 2);
    }

    #[test]
    fn test_export_trust_map() {
        let mut mgr = TrustManager::new(1.0);
        mgr.update_local_trust("did:key:zA", 0.8);
        let map = mgr.export_trust_map();
        assert!(map.contains_key("did:key:zA"));
        assert!((map["did:key:zA"] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_clamp_quality_score() {
        let mut mgr = TrustManager::new(1.0);
        mgr.update_local_trust("did:key:zPeer", 1.5); // out of bounds
        assert!(mgr.get_trust("did:key:zPeer") <= 1.0);
        mgr.update_local_trust("did:key:zPeer2", -0.5); // out of bounds
        assert!(mgr.get_trust("did:key:zPeer2") >= 0.0);
    }
}
