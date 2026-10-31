//! Registre de récupération — instrumentation **passive** (PR1 du plan 42d49a31).
//!
//! Objectif : rendre mesurable la distinction « mémoire disponible » / « mémoire
//! mobilisable » sans changer un seul comportement. Ce module n'émet aucun effet
//! de bord sur la réponse : il observe le premier maillon de la chaîne
//! `E→Q→L→R→V→A` — combien de fois l'index est interrogé, avec quelle requête,
//! quels identifiants et quels scores, par quel motif, et ce qu'il en est advenu
//! au tour suivant.
//!
//! # Les sept champs (formalisme v7 §X)
//!
//! 1. motif du déclenchement — [`Trigger`] ;
//! 2. requête exacte — [`RetrievalEvent::query`] ;
//! 3. identifiants et scores retournés — [`RetrievalEvent::results`] ;
//! 4. identifiants effectivement utilisés — [`Usage::used_ids`], rempli par
//!    [`outcome`] au tour SUIVANT (le tour qui a servi la récupération ne peut
//!    pas savoir ce qu'il en fera) ;
//! 5. preuve de validation disponible — [`Usage::validation`] ;
//! 6. résultat observable qui a suivi — [`Usage::observable`] ;
//! 7. bénéfice attribuable — [`Usage::benefit`], avec sa raison quand il est nul.
//!
//! Un champ non renseigné reste `None` : il ne vaut jamais « non » ni zéro. C'est
//! la différence entre « on ne sait pas » et « il ne s'est rien passé », et c'est
//! toute la valeur de la mesure.
//!
//! # Ce que le registre ne peut pas faire
//!
//! Il ne fait jamais échouer ni attendre l'appelant : [`record`] n'effectue
//! aucune I/O, ne panique pas et n'attend aucun verrou (`try_lock`). Un journal
//! indisponible — verrou pris, capacité nulle — **jette l'événement et le
//! compte** ([`Snapshot::dropped`]) ; la requête de l'utilisateur aboutit quand
//! même, et le compteur le dit.
//!
//! # Pourquoi en mémoire et pas dans le graphe
//!
//! Décision `8703a794` : un nœud par récupération paierait une écriture sur le
//! chemin chaud et ferait croître le graphe sans borne. Les compteurs vivent donc
//! dans le processus ; le journal borné ([`recent`]) sert la lecture immédiate, et
//! l'export durable est un autre sujet.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Motif du déclenchement (champ (1)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    /// L'agent a formulé la requête lui-même.
    Explicit,
    /// Frontière de tâche (nouveau plan, nouvelle tâche, nouveau fichier).
    TaskBoundary,
    /// Incertitude déclarée par l'agent.
    Uncertainty,
    /// Contradiction détectée avec une règle active.
    Contradiction,
    /// Déclenchement mécanique par un hook, sur un appel d'outil.
    ///
    /// Ce n'est **pas** un des quatre motifs du formalisme v7 : il est catalogué
    /// à part pour qu'un déclenchement systématique ne se déguise pas en
    /// « explicite » et ne gonfle pas ce motif.
    Hook,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Trigger::Explicit => "explicit",
            Trigger::TaskBoundary => "task_boundary",
            Trigger::Uncertainty => "uncertainty",
            Trigger::Contradiction => "contradiction",
            Trigger::Hook => "hook",
        }
    }

    const ALL: [Trigger; 5] = [
        Trigger::Explicit,
        Trigger::TaskBoundary,
        Trigger::Uncertainty,
        Trigger::Contradiction,
        Trigger::Hook,
    ];

    fn index(self) -> usize {
        match self {
            Trigger::Explicit => 0,
            Trigger::TaskBoundary => 1,
            Trigger::Uncertainty => 2,
            Trigger::Contradiction => 3,
            Trigger::Hook => 4,
        }
    }
}

/// Index interrogé (quelle source a répondu).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
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

/// Un contenu retourné par l'index : son identifiant, sa nature, son score.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RetrievedItem {
    pub id: String,
    /// `note`, `decision`, `skill`… la nature de ce qui a été rendu.
    pub kind: String,
    /// Score de l'index, quand il en existe un. `None` n'est pas `0.0`.
    pub score: Option<f64>,
}

impl RetrievedItem {
    pub fn new(id: impl Into<String>, kind: impl Into<String>, score: Option<f64>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            score,
        }
    }
}

/// La preuve qui rend une récupération vérifiable (champ (5)).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Validation {
    /// Une ancre de code (`<fichier>::<symbole>`).
    Anchor(String),
    /// Un commit.
    Sha(String),
    /// Un test qui passe.
    Test(String),
}

impl Validation {
    pub fn kind(&self) -> &'static str {
        match self {
            Validation::Anchor(_) => "anchor",
            Validation::Sha(_) => "sha",
            Validation::Test(_) => "test",
        }
    }

    pub fn value(&self) -> &str {
        match self {
            Validation::Anchor(v) | Validation::Sha(v) | Validation::Test(v) => v,
        }
    }
}

/// Bénéfice attribuable (champ (7)). Un « non » porte sa raison.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Benefit {
    Yes,
    No { reason: String },
}

impl Benefit {
    pub fn as_str(&self) -> &'static str {
        match self {
            Benefit::Yes => "yes",
            Benefit::No { .. } => "no",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Benefit::Yes => None,
            Benefit::No { reason } => Some(reason),
        }
    }
}

/// Ce qu'il est advenu d'une récupération, renseigné au tour suivant
/// (champs (4) à (7)).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Usage {
    /// Champ (4) : les identifiants réellement utilisés, parmi ceux rendus.
    pub used_ids: Vec<String>,
    /// Ceux des identifiants utilisés qui étaient hors de leur fenêtre de
    /// validité ou remplacés — le « contenu périmé » de l'agrégat.
    pub stale_ids: Vec<String>,
    /// Renseigné quand l'agent a été **bloqué** faute du contenu attendu.
    pub blocked: Option<String>,
    /// Champ (5).
    pub validation: Option<Validation>,
    /// Champ (6).
    pub observable: Option<String>,
    /// Champ (7). `None` = non attribuable, ce qui n'est pas un « non ».
    pub benefit: Option<Benefit>,
}

/// Un événement de récupération — les champs (1) à (3), puis (4) à (7) une fois
/// le tour suivant connu.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RetrievalEvent {
    pub id: u64,
    pub at: String,
    pub source: Source,
    pub trigger: Trigger,
    /// Champ (2).
    pub query: String,
    /// Champ (3).
    pub results: Vec<RetrievedItem>,
    /// La lecture a-t-elle abouti (distinct d'un résultat vide).
    pub ok: bool,
    pub latency_ms: u64,
    /// Champs (4) à (7), `None` tant que le tour suivant ne les a pas fournis.
    pub usage: Option<Usage>,
}

/// Compteurs cumulés depuis le démarrage du processus.
struct Counters {
    calls: AtomicU64,
    failures: AtomicU64,
    empty_results: AtomicU64,
    hits: AtomicU64,
    latency_ms_total: AtomicU64,
    semantic_calls: AtomicU64,
    bm25_calls: AtomicU64,
    skill_calls: AtomicU64,
    other_calls: AtomicU64,
    triggers: [AtomicU64; 5],
    /// Événements jetés faute de journal disponible (verrou pris, capacité nulle).
    dropped: AtomicU64,
    /// `outcome` appelé sur un identifiant inconnu (purgé du journal borné).
    orphan_outcomes: AtomicU64,
    used: AtomicU64,
    unused: AtomicU64,
    stale_hits: AtomicU64,
    blocked: AtomicU64,
    benefit_yes: AtomicU64,
    benefit_no: AtomicU64,
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
                AtomicU64::new(0),
            ],
            dropped: AtomicU64::new(0),
            orphan_outcomes: AtomicU64::new(0),
            used: AtomicU64::new(0),
            unused: AtomicU64::new(0),
            stale_hits: AtomicU64::new(0),
            blocked: AtomicU64::new(0),
            benefit_yes: AtomicU64::new(0),
            benefit_no: AtomicU64::new(0),
        }
    }
}

static LEDGER: Counters = Counters::new();

/// Capacité du journal borné. 0 = journal désactivé (les compteurs continuent).
static CAPACITY: AtomicUsize = AtomicUsize::new(DEFAULT_CAPACITY);
const DEFAULT_CAPACITY: usize = 256;

fn journal() -> &'static Mutex<VecDeque<RetrievalEvent>> {
    static JOURNAL: OnceLock<Mutex<VecDeque<RetrievalEvent>>> = OnceLock::new();
    JOURNAL.get_or_init(|| Mutex::new(VecDeque::with_capacity(DEFAULT_CAPACITY)))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Capacité courante du journal.
pub fn journal_capacity() -> usize {
    CAPACITY.load(Ordering::Relaxed)
}

/// Sérialise les tests qui touchent le registre : il est global, et deux tests
/// concurrents se voleraient leurs événements. Résistant à l'empoisonnement, pour
/// qu'un test qui panique ne fasse pas tomber les suivants.
#[cfg(test)]
pub(crate) fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Fixe la capacité du journal (diagnostic, tests, ou bornage mémoire).
///
/// Réduire la capacité ne détruit pas les événements déjà présents : ils sont
/// simplement évincés au fil des suivants.
pub fn set_journal_capacity(capacity: usize) {
    CAPACITY.store(capacity, Ordering::Relaxed);
}

/// Enregistre un appel d'index. **Infaillible** : n'attend aucun verrou, ne
/// panique pas, ne fait aucune I/O.
///
/// Retourne l'identifiant de l'événement, ou `None` si le journal n'a pas pu le
/// garder (compteurs incrémentés quand même, `dropped` compte le manque).
pub fn record(
    source: Source,
    trigger: Trigger,
    query: &str,
    results: Vec<RetrievedItem>,
    ok: bool,
    latency_ms: u64,
) -> Option<u64> {
    LEDGER.calls.fetch_add(1, Ordering::Relaxed);
    // Un échec et un résultat vide sont deux choses différentes : une lecture qui
    // a échoué n'a rien répondu, elle n'a pas répondu « rien ».
    if !ok {
        LEDGER.failures.fetch_add(1, Ordering::Relaxed);
    } else if results.is_empty() {
        LEDGER.empty_results.fetch_add(1, Ordering::Relaxed);
    }
    LEDGER
        .hits
        .fetch_add(results.len() as u64, Ordering::Relaxed);
    LEDGER
        .latency_ms_total
        .fetch_add(latency_ms, Ordering::Relaxed);
    match source {
        Source::Semantic => LEDGER.semantic_calls.fetch_add(1, Ordering::Relaxed),
        Source::Bm25 => LEDGER.bm25_calls.fetch_add(1, Ordering::Relaxed),
        Source::SkillActivate => LEDGER.skill_calls.fetch_add(1, Ordering::Relaxed),
        Source::Other => LEDGER.other_calls.fetch_add(1, Ordering::Relaxed),
    };
    LEDGER.triggers[trigger.index()].fetch_add(1, Ordering::Relaxed);

    let capacity = CAPACITY.load(Ordering::Relaxed);
    if capacity == 0 {
        LEDGER.dropped.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    // `try_lock` : jamais d'attente sur le chemin chaud. Un verrou pris par un
    // lecteur lent, et l'événement est jeté — compté, pas perdu en silence.
    let Ok(mut journal) = journal().try_lock() else {
        LEDGER.dropped.fetch_add(1, Ordering::Relaxed);
        return None;
    };

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    while journal.len() >= capacity {
        journal.pop_front();
    }
    journal.push_back(RetrievalEvent {
        id,
        at: chrono::Utc::now().to_rfc3339(),
        source,
        trigger,
        query: query.to_string(),
        results,
        ok,
        latency_ms,
        usage: None,
    });
    Some(id)
}

/// Renseigne ce qu'il est advenu de la récupération `event_id` (champs 4 à 7).
///
/// Retourne `false` si l'événement n'est plus dans le journal (évincé par la
/// capacité) : l'appelant n'a rien à en faire, et le manque est compté.
pub fn outcome(event_id: u64, usage: Usage) -> bool {
    if usage.used_ids.is_empty() {
        LEDGER.unused.fetch_add(1, Ordering::Relaxed);
    } else {
        LEDGER.used.fetch_add(1, Ordering::Relaxed);
    }
    if !usage.stale_ids.is_empty() {
        LEDGER
            .stale_hits
            .fetch_add(usage.stale_ids.len() as u64, Ordering::Relaxed);
    }
    if usage.blocked.is_some() {
        LEDGER.blocked.fetch_add(1, Ordering::Relaxed);
    }
    match &usage.benefit {
        Some(Benefit::Yes) => LEDGER.benefit_yes.fetch_add(1, Ordering::Relaxed),
        Some(Benefit::No { .. }) => LEDGER.benefit_no.fetch_add(1, Ordering::Relaxed),
        None => 0,
    };

    // Même règle que `recent` : ce n'est pas le chemin chaud d'une recherche, et
    // confondre « verrou pris » avec « événement évincé » ferait mentir le
    // compteur d'orphelins. On attend le verrou.
    let mut journal = journal().lock().unwrap_or_else(|e| e.into_inner());
    match journal.iter_mut().find(|e| e.id == event_id) {
        Some(event) => {
            event.usage = Some(usage);
            true
        }
        None => {
            LEDGER.orphan_outcomes.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

/// Vue instantanée des compteurs — l'agrégat de session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
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
    pub hook_calls: u64,
    /// Événements jetés : journal indisponible. Un chiffre > 0 dit que la mesure
    /// est incomplète — il ne se lit pas comme « rien à signaler ».
    pub dropped: u64,
    pub orphan_outcomes: u64,
    /// Récupérations dont au moins un contenu a servi.
    pub used: u64,
    /// Récupérations dont aucun contenu n'a servi — les appels inutiles.
    pub unused: u64,
    pub stale_hits: u64,
    pub blocked: u64,
    pub benefit_yes: u64,
    pub benefit_no: u64,
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

    /// Coût par décision correcte : appels par récupération ayant un bénéfice
    /// attribué. `None` tant qu'aucun bénéfice n'a été attribué — un coût calculé
    /// sur zéro décision correcte n'existe pas.
    pub fn cost_per_correct(&self) -> Option<f64> {
        if self.benefit_yes == 0 {
            None
        } else {
            Some(self.calls as f64 / self.benefit_yes as f64)
        }
    }
}

/// Lit l'état courant des compteurs.
pub fn snapshot() -> Snapshot {
    let t = |i: usize| LEDGER.triggers[i].load(Ordering::Relaxed);
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
        explicit_calls: t(0),
        task_boundary_calls: t(1),
        uncertainty_calls: t(2),
        contradiction_calls: t(3),
        hook_calls: t(4),
        dropped: LEDGER.dropped.load(Ordering::Relaxed),
        orphan_outcomes: LEDGER.orphan_outcomes.load(Ordering::Relaxed),
        used: LEDGER.used.load(Ordering::Relaxed),
        unused: LEDGER.unused.load(Ordering::Relaxed),
        stale_hits: LEDGER.stale_hits.load(Ordering::Relaxed),
        blocked: LEDGER.blocked.load(Ordering::Relaxed),
        benefit_yes: LEDGER.benefit_yes.load(Ordering::Relaxed),
        benefit_no: LEDGER.benefit_no.load(Ordering::Relaxed),
    }
}

/// Les `limit` derniers événements, du plus ancien au plus récent.
pub fn recent(limit: usize) -> Vec<RetrievalEvent> {
    // Lecture, hors du chemin chaud : on attend le verrou au lieu de rendre une
    // liste vide. Un `try_lock` ici confondrait « personne n'a rien écrit » et
    // « quelqu'un écrivait » — deux choses qu'un journal de mesure ne peut pas
    // se permettre de confondre.
    let journal = journal().lock().unwrap_or_else(|e| e.into_inner());
    let skip = journal.len().saturating_sub(limit);
    journal.iter().skip(skip).cloned().collect()
}

/// Les motifs, dans l'ordre de leur indice de compteur.
pub fn triggers() -> &'static [Trigger] {
    &Trigger::ALL
}

#[cfg(test)]
mod tests {
    use super::*;

    // Le registre est global : les tests qui le lisent passent par `serial_guard`,
    // partagé avec les autres modules de tests qui écrivent dedans.
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
            &LEDGER.dropped,
            &LEDGER.orphan_outcomes,
            &LEDGER.used,
            &LEDGER.unused,
            &LEDGER.stale_hits,
            &LEDGER.blocked,
            &LEDGER.benefit_yes,
            &LEDGER.benefit_no,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        for t in LEDGER.triggers.iter() {
            t.store(0, Ordering::Relaxed);
        }
        if let Ok(mut journal) = journal().lock() {
            journal.clear();
        }
        set_journal_capacity(DEFAULT_CAPACITY);
    }

    fn item(id: &str, score: f64) -> RetrievedItem {
        RetrievedItem::new(id, "note", Some(score))
    }

    #[test]
    fn record_compte_un_appel_par_champ() {
        let _g = serial_guard();
        reset();
        record(
            Source::Semantic,
            Trigger::Explicit,
            "où vit le cache ?",
            vec![item("a", 0.9)],
            true,
            12,
        );
        let s = snapshot();
        assert_eq!(s.calls, 1);
        assert_eq!(s.hits, 1);
        assert_eq!(s.failures, 0);
        assert_eq!(s.empty_results, 0);
        assert_eq!(s.semantic_calls, 1);
        assert_eq!(s.explicit_calls, 1);
        assert_eq!(s.mean_latency_ms(), Some(12.0));
    }

    /// Champ (2) et (3) : la requête exacte, les identifiants et les scores sont
    /// dans le journal — un compteur ne suffit pas à savoir *quoi* a été rendu.
    #[test]
    fn le_journal_garde_la_requete_les_ids_et_les_scores() {
        let _g = serial_guard();
        reset();
        let id = record(
            Source::Bm25,
            Trigger::TaskBoundary,
            "registre de récupération",
            vec![item("note-1", 0.42), item("note-2", 0.17)],
            true,
            7,
        )
        .expect("le journal doit accepter l'événement");

        let events = recent(10);
        let event = events
            .iter()
            .find(|e| e.id == id)
            .expect("événement présent");
        assert_eq!(event.query, "registre de récupération");
        assert_eq!(event.trigger, Trigger::TaskBoundary);
        assert_eq!(event.source, Source::Bm25);
        assert_eq!(event.results.len(), 2);
        assert_eq!(event.results[0].id, "note-1");
        assert_eq!(event.results[0].score, Some(0.42));
        assert!(!event.at.is_empty(), "l'événement est daté");
    }

    /// Champs (4) à (7) : ils arrivent au tour suivant, et seuls ceux-là sont
    /// renseignés — un champ inconnu reste `None`, jamais « non ».
    #[test]
    fn outcome_renseigne_les_champs_du_tour_suivant() {
        let _g = serial_guard();
        reset();
        let id = record(
            Source::Semantic,
            Trigger::Uncertainty,
            "q",
            vec![item("n1", 0.8)],
            true,
            3,
        )
        .unwrap();

        let before = recent(1).pop().unwrap();
        assert!(before.usage.is_none(), "le même tour ne peut pas savoir");

        assert!(outcome(
            id,
            Usage {
                used_ids: vec!["n1".into()],
                stale_ids: vec![],
                blocked: None,
                validation: Some(Validation::Anchor("src/a.rs::build".into())),
                observable: Some("le test de parité passe".into()),
                benefit: Some(Benefit::Yes),
            }
        ));

        let after = recent(1).pop().unwrap();
        let usage = after.usage.expect("renseigné au tour suivant");
        assert_eq!(usage.used_ids, vec!["n1".to_string()]);
        assert_eq!(usage.validation.unwrap().kind(), "anchor");
        assert_eq!(usage.benefit, Some(Benefit::Yes));

        let s = snapshot();
        assert_eq!(s.used, 1);
        assert_eq!(s.benefit_yes, 1);
        assert_eq!(s.unused, 0);
    }

    #[test]
    fn un_benefice_nul_porte_sa_raison() {
        let _g = serial_guard();
        reset();
        let id = record(
            Source::Other,
            Trigger::Explicit,
            "q",
            vec![item("n", 0.1)],
            true,
            1,
        )
        .unwrap();
        outcome(
            id,
            Usage {
                used_ids: vec![],
                stale_ids: vec!["n".into()],
                blocked: Some("la note était remplacée".into()),
                validation: None,
                observable: None,
                benefit: Some(Benefit::No {
                    reason: "contenu périmé".into(),
                }),
            },
        );
        let s = snapshot();
        assert_eq!(s.unused, 1, "aucun contenu n'a servi");
        assert_eq!(s.benefit_no, 1);
        assert_eq!(s.stale_hits, 1);
        assert_eq!(s.blocked, 1);

        let usage = recent(1).pop().unwrap().usage.unwrap();
        assert_eq!(
            usage.benefit.unwrap().reason(),
            Some("contenu périmé"),
            "un « non » sans raison ne s'attribue pas"
        );
    }

    /// L'identifiant d'un événement évincé ne fait pas échouer l'appelant : il est
    /// compté. Un `outcome` perdu n'est pas une erreur, c'est un manque.
    #[test]
    fn un_outcome_orphelin_est_compte_pas_une_erreur() {
        let _g = serial_guard();
        reset();
        assert!(!outcome(
            999_999,
            Usage {
                used_ids: vec!["x".into()],
                ..Default::default()
            }
        ));
        assert_eq!(snapshot().orphan_outcomes, 1);
    }

    /// LA contrainte du plan : « le registre ne fait jamais échouer une requête
    /// utilisateur […] échec silencieux et compté si le journal est indisponible ».
    /// Journal à capacité nulle : l'enregistrement est jeté, compté, et l'appelant
    /// reçoit quand même sa valeur.
    #[test]
    fn un_journal_indisponible_ne_casse_rien_et_se_compte() {
        let _g = serial_guard();
        reset();
        set_journal_capacity(0);

        let answer = {
            let id = record(
                Source::Semantic,
                Trigger::Explicit,
                "q",
                vec![item("n", 1.0)],
                true,
                2,
            );
            // L'appelant ne s'interrompt pas : il a sa réponse, pas d'erreur.
            assert!(id.is_none(), "l'événement n'a pas été gardé");
            vec!["n"]
        };

        assert_eq!(answer, vec!["n"], "la requête aboutit quand même");
        let s = snapshot();
        assert_eq!(s.dropped, 1, "le manque est compté");
        assert_eq!(s.calls, 1, "l'appel reste mesuré");
        assert!(recent(10).is_empty(), "rien n'a été gardé");
        set_journal_capacity(DEFAULT_CAPACITY);
    }

    /// Le journal est borné : au-delà de la capacité, les plus anciens sortent.
    #[test]
    fn le_journal_est_borne_et_garde_les_plus_recents() {
        let _g = serial_guard();
        reset();
        set_journal_capacity(2);
        for q in ["un", "deux", "trois"] {
            record(Source::Bm25, Trigger::Explicit, q, vec![], true, 0);
        }
        let events = recent(10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].query, "deux");
        assert_eq!(events[1].query, "trois");
        set_journal_capacity(DEFAULT_CAPACITY);
    }

    #[test]
    fn un_appel_vide_n_est_pas_un_echec() {
        let _g = serial_guard();
        reset();
        record(Source::Bm25, Trigger::TaskBoundary, "q", vec![], true, 4);
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
        let _g = serial_guard();
        reset();
        record(
            Source::Semantic,
            Trigger::Uncertainty,
            "q",
            vec![],
            false,
            1,
        );
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
        let _g = serial_guard();
        reset();
        assert_eq!(snapshot().calls_per_nonempty(), None);
        record(Source::Other, Trigger::Explicit, "q", vec![], true, 0);
        assert_eq!(snapshot().calls_per_nonempty(), None);
        record(
            Source::Other,
            Trigger::Explicit,
            "q",
            vec![item("n", 1.0)],
            true,
            0,
        );
        assert_eq!(snapshot().calls_per_nonempty(), Some(2.0));
    }

    /// Le coût par décision correcte n'existe pas tant qu'aucune n'est attribuée.
    #[test]
    fn cout_par_decision_correcte_est_none_sans_benefice() {
        let _g = serial_guard();
        reset();
        record(
            Source::Other,
            Trigger::Explicit,
            "q",
            vec![item("n", 1.0)],
            true,
            0,
        );
        assert_eq!(snapshot().cost_per_correct(), None);

        let id = record(
            Source::Other,
            Trigger::Explicit,
            "q",
            vec![item("n", 1.0)],
            true,
            0,
        )
        .unwrap();
        outcome(
            id,
            Usage {
                used_ids: vec!["n".into()],
                benefit: Some(Benefit::Yes),
                ..Default::default()
            },
        );
        assert_eq!(snapshot().cost_per_correct(), Some(2.0));
    }

    /// Le déclenchement mécanique par hook n'est pas un des quatre motifs : il ne
    /// doit pas gonfler « explicite ».
    #[test]
    fn le_hook_n_est_pas_un_declenchement_explicite() {
        let _g = serial_guard();
        reset();
        record(
            Source::SkillActivate,
            Trigger::Hook,
            "Edit",
            vec![],
            true,
            2,
        );
        let s = snapshot();
        assert_eq!(s.hook_calls, 1);
        assert_eq!(s.explicit_calls, 0);
        assert_eq!(s.skill_calls, 1);
    }
}
