//! Registre de récupération — instrumentation **passive** (PR1 du plan 42d49a31).
//!
//! Objectif : rendre mesurable la distinction « mémoire disponible » / « mémoire
//! mobilisable » sans changer un seul comportement. Ce module ne fait que compter.
//! Il ne peut jamais faire échouer l'appelant : `record` est infaillible et sans
//! allocation bloquante.
//!
//! Ce qui est compté ici est le premier maillon de la chaîne `E→Q→L→R→V→A` :
//! combien de fois l'index est interrogé, avec quel résultat, en combien de temps,
//! et par quel motif de déclenchement.

use std::sync::atomic::{AtomicU64, Ordering};

/// Motif du déclenchement (champ (1) du registre, formalisme v7 §X).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// L'agent a formulé la requête lui-même.
    Explicit,
    /// Frontière de tâche (nouveau plan, nouvelle tâche, nouveau fichier).
    TaskBoundary,
    /// Incertitude déclarée par l'agent.
    Uncertainty,
    /// Contradiction détectée avec une règle active.
    Contradiction,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Trigger::Explicit => "explicit",
            Trigger::TaskBoundary => "task_boundary",
            Trigger::Uncertainty => "uncertainty",
            Trigger::Contradiction => "contradiction",
        }
    }
}

/// Index interrogé (champ (2) : quelle source a servi).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Semantic,
    Bm25,
    SkillActivate,
    Other,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Semantic => "semantic",
            Source::Bm25 => "bm25",
            Source::SkillActivate => "skill_activate",
            Source::Other => "other",
        }
    }
}

/// Compteurs cumulés depuis le démarrage du processus.
pub struct Counters {
    calls: AtomicU64,
    failures: AtomicU64,
    empty_results: AtomicU64,
    hits: AtomicU64,
    latency_ms_total: AtomicU64,
    semantic_calls: AtomicU64,
    bm25_calls: AtomicU64,
    skill_calls: AtomicU64,
    other_calls: AtomicU64,
    triggers: [AtomicU64; 4],
}

impl Counters {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            empty_results: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            latency_ms_total: AtomicU64::new(0),
            semantic_calls: AtomicU64::new(0),
            bm25_calls: AtomicU64::new(0),
            skill_calls: AtomicU64::new(0),
            other_calls: AtomicU64::new(0),
            triggers: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }
}

static LEDGER: Counters = Counters::new();

/// Vue instantanée du registre — ce que lit l'agrégat de session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Snapshot {
    pub calls: u64,
    pub failures: u64,
    pub empty_results: u64,
    pub hits: u64,
    pub latency_ms_total: u64,
    pub semantic_calls: u64,
    pub bm25_calls: u64,
    pub skill_calls: u64,
    pub other_calls: u64,
    pub explicit_calls: u64,
    pub task_boundary_calls: u64,
    pub uncertainty_calls: u64,
    pub contradiction_calls: u64,
}

impl Snapshot {
    /// Coût en appels par résultat non vide. `None` s'il n'y a pas encore de
    /// résultat exploitable — jamais `0.0`, qui laisserait croire à la gratuité.
    pub fn calls_per_nonempty(&self) -> Option<f64> {
        let denom = self
            .calls
            .saturating_sub(self.empty_results)
            .saturating_sub(self.failures);
        if denom == 0 {
            None
        } else {
            Some(self.calls as f64 / denom as f64)
        }
    }

    pub fn mean_latency_ms(&self) -> Option<f64> {
        if self.calls == 0 {
            None
        } else {
            Some(self.latency_ms_total as f64 / self.calls as f64)
        }
    }
}

/// Enregistre un appel d'index. **Infaillible** : ne panique pas, n'échoue pas,
/// n'alloue pas. Appelé sur le chemin chaud, donc uniquement des additions
/// atomiques.
pub fn record(source: Source, trigger: Trigger, hits: usize, ok: bool, latency_ms: u64) {
    LEDGER.calls.fetch_add(1, Ordering::Relaxed);
    if !ok {
        LEDGER.failures.fetch_add(1, Ordering::Relaxed);
    }
    if hits == 0 {
        LEDGER.empty_results.fetch_add(1, Ordering::Relaxed);
    }
    LEDGER.hits.fetch_add(hits as u64, Ordering::Relaxed);
    LEDGER
        .latency_ms_total
        .fetch_add(latency_ms, Ordering::Relaxed);
    match source {
        Source::Semantic => LEDGER.semantic_calls.fetch_add(1, Ordering::Relaxed),
        Source::Bm25 => LEDGER.bm25_calls.fetch_add(1, Ordering::Relaxed),
        Source::SkillActivate => LEDGER.skill_calls.fetch_add(1, Ordering::Relaxed),
        Source::Other => LEDGER.other_calls.fetch_add(1, Ordering::Relaxed),
    };
    let idx = match trigger {
        Trigger::Explicit => 0,
        Trigger::TaskBoundary => 1,
        Trigger::Uncertainty => 2,
        Trigger::Contradiction => 3,
    };
    LEDGER.triggers[idx].fetch_add(1, Ordering::Relaxed);
}

/// Lit l'état courant du registre.
pub fn snapshot() -> Snapshot {
    Snapshot {
        calls: LEDGER.calls.load(Ordering::Relaxed),
        failures: LEDGER.failures.load(Ordering::Relaxed),
        empty_results: LEDGER.empty_results.load(Ordering::Relaxed),
        hits: LEDGER.hits.load(Ordering::Relaxed),
        latency_ms_total: LEDGER.latency_ms_total.load(Ordering::Relaxed),
        semantic_calls: LEDGER.semantic_calls.load(Ordering::Relaxed),
        bm25_calls: LEDGER.bm25_calls.load(Ordering::Relaxed),
        skill_calls: LEDGER.skill_calls.load(Ordering::Relaxed),
        other_calls: LEDGER.other_calls.load(Ordering::Relaxed),
        explicit_calls: LEDGER.triggers[0].load(Ordering::Relaxed),
        task_boundary_calls: LEDGER.triggers[1].load(Ordering::Relaxed),
        uncertainty_calls: LEDGER.triggers[2].load(Ordering::Relaxed),
        contradiction_calls: LEDGER.triggers[3].load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Le registre est global : les tests qui le lisent doivent être sérialisés.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn reset() {
        for c in [
            &LEDGER.calls,
            &LEDGER.failures,
            &LEDGER.empty_results,
            &LEDGER.hits,
            &LEDGER.latency_ms_total,
            &LEDGER.semantic_calls,
            &LEDGER.bm25_calls,
            &LEDGER.skill_calls,
            &LEDGER.other_calls,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for t in LEDGER.triggers.iter() {
            t.store(0, Ordering::Relaxed);
        }
    }

    #[test]
    fn record_compte_un_appel_par_champ() {
        let _g = SERIAL.lock().unwrap();
        reset();
        record(Source::Semantic, Trigger::Explicit, 3, true, 12);
        let s = snapshot();
        assert_eq!(s.calls, 1);
        assert_eq!(s.hits, 3);
        assert_eq!(s.failures, 0);
        assert_eq!(s.empty_results, 0);
        assert_eq!(s.semantic_calls, 1);
        assert_eq!(s.explicit_calls, 1);
        assert_eq!(s.mean_latency_ms(), Some(12.0));
    }

    #[test]
    fn un_appel_vide_n_est_pas_un_echec() {
        let _g = SERIAL.lock().unwrap();
        reset();
        record(Source::Bm25, Trigger::TaskBoundary, 0, true, 4);
        let s = snapshot();
        assert_eq!(s.empty_results, 1);
        assert_eq!(
            s.failures, 0,
            "un résultat vide est une réponse, pas une erreur"
        );
        assert_eq!(s.bm25_calls, 1);
        assert_eq!(s.task_boundary_calls, 1);
    }

    #[test]
    fn un_echec_n_est_pas_un_resultat_vide() {
        let _g = SERIAL.lock().unwrap();
        reset();
        record(Source::Semantic, Trigger::Uncertainty, 0, false, 1);
        let s = snapshot();
        assert_eq!(s.failures, 1);
        assert_eq!(
            s.empty_results, 0,
            "un échec et un vide ne doivent pas se confondre"
        );
        assert_eq!(s.uncertainty_calls, 1);
    }

    #[test]
    fn cout_par_resultat_non_vide_est_none_sans_resultat() {
        let _g = SERIAL.lock().unwrap();
        reset();
        assert_eq!(snapshot().calls_per_nonempty(), None);
        record(Source::Other, Trigger::Explicit, 0, true, 0);
        assert_eq!(snapshot().calls_per_nonempty(), None);
        record(Source::Other, Trigger::Explicit, 2, true, 0);
        assert_eq!(snapshot().calls_per_nonempty(), Some(2.0));
    }
}
