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
//! # « Utilisé » : la définition mesurée
//!
//! Un identifiant rendu par l'index est **utilisé** quand la réponse de
//! l'assistant le **cite** (`#note:<uuid>`, lu par le lecteur unique
//! `refs::cite::cited_refs` : un jeton dans un bloc de code n'est pas une
//! citation). Une note *injectée* dans le prompt n'est pas pour autant utilisée :
//! c'est la mémoire *mobilisable*, pas la mémoire *mobilisée*. Aujourd'hui le
//! rendu de l'injection n'expose pas l'identifiant des notes au modèle : cette
//! mesure lira donc bas, et c'est l'information — pas un défaut du registre.
//!
//! Le règlement se fait à la **fin du tour** qui a récupéré, quand la réponse
//! est connue ([`settle_turn`], appelé par `chat::post_stream::record_turn_feedback`,
//! la couture commune aux deux moteurs). Pas « au début du tour suivant » : la
//! réponse n'y est plus sous la main (il faudrait la relire du graphe sur le
//! chemin chaud) et le dernier tour d'une session n'aurait jamais de règlement
//! — ses appels inutiles ne seraient jamais comptés. L'information est la même ;
//! le moment est le premier où elle existe.
//!
//! Les champs (5) à (7) restent `None` au règlement : aucun des deux moteurs ne
//! sait aujourd'hui attribuer une preuve, un résultat observable ou un
//! bénéfice. Les inventer ferait mentir l'agrégat.
//!
//! # Par session
//!
//! Chaque événement porte la session qui l'a déclenché quand elle est connue
//! ([`RetrievalEvent::session_id`]) : l'injection de connaissances la connaît,
//! une recherche explicite par l'outil `note` depuis le MCP ne la connaît pas
//! (`None`). [`snapshot_for_session`] est calculé depuis le journal **borné** :
//! c'est un agrégat « récent par session », pas « depuis le début » — il porte
//! la fenêtre ([`JournalWindow`]) qui dit ce qu'il a pu manquer.
//!
//! # Ce que le registre ne peut pas faire
//!
//! Il ne fait jamais échouer ni attendre l'appelant : [`record`] n'effectue
//! aucune I/O, ne panique pas et n'attend aucun verrou (`try_lock`). Verrou
//! pris, l'événement part dans un **débordement sans verrou**, borné, que le
//! prochain lecteur ou écrivain absorbe : une collision entre deux écrivains ne
//! jette rien. Un journal indisponible — débordement plein, capacité nulle —
//! **jette l'événement et le compte** ([`Snapshot::dropped`]) ; la requête de
//! l'utilisateur aboutit quand même, et le compteur le dit.
//!
//! # Pourquoi en mémoire et pas dans le graphe
//!
//! Décision `8703a794` : un nœud par récupération paierait une écriture sur le
//! chemin chaud et ferait croître le graphe sans borne. Les compteurs vivent donc
//! dans le processus ; le journal borné ([`recent`]) sert la lecture immédiate, et
//! l'export durable est un autre sujet.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use uuid::Uuid;

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
    /// Injection automatique de connaissances, à chaque tour de chat.
    ///
    /// Pas un des quatre motifs du formalisme v7, pas un hook d'outil non plus :
    /// personne n'a décidé cette récupération, elle a lieu parce qu'un tour a
    /// lieu. Cataloguée à part, comme [`Trigger::Hook`], pour que le chemin de
    /// production — qui alimente chaque tour — ne se compte pas en « explicite »
    /// et n'écrase pas les quatre motifs sous son volume.
    Injection,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Trigger::Explicit => "explicit",
            Trigger::TaskBoundary => "task_boundary",
            Trigger::Uncertainty => "uncertainty",
            Trigger::Contradiction => "contradiction",
            Trigger::Hook => "hook",
            Trigger::Injection => "injection",
        }
    }

    const ALL: [Trigger; 6] = [
        Trigger::Explicit,
        Trigger::TaskBoundary,
        Trigger::Uncertainty,
        Trigger::Contradiction,
        Trigger::Hook,
        Trigger::Injection,
    ];

    fn index(self) -> usize {
        match self {
            Trigger::Explicit => 0,
            Trigger::TaskBoundary => 1,
            Trigger::Uncertainty => 2,
            Trigger::Contradiction => 3,
            Trigger::Hook => 4,
            Trigger::Injection => 5,
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
    /// La session de chat qui a déclenché la récupération, quand l'appelant la
    /// connaît. `None` : récupération hors session (outil `note` par le MCP…), qui
    /// n'entre dans aucun agrégat par session et n'est jamais réglée.
    pub session_id: Option<Uuid>,
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
    triggers: [AtomicU64; 6],
    /// Événements jetés faute de journal disponible (verrou pris, capacité nulle).
    dropped: AtomicU64,
    /// Événements sortis du journal par la capacité : ils ont été comptés, mais
    /// ne sont plus lisibles — ni dans `recent`, ni dans un agrégat par session.
    evicted: AtomicU64,
    /// Parmi les évincés, ceux dont le tour n'avait pas encore réglé l'issue :
    /// leurs champs (4) à (7) ne seront jamais connus. Un chiffre > 0 dit que
    /// `used`/`unused` sous-comptent.
    evicted_unsettled: AtomicU64,
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
                AtomicU64::new(0),
            ],
            dropped: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
            evicted_unsettled: AtomicU64::new(0),
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

/// Le journal borné et, sous le **même** verrou, les événements de chaque session
/// qui attendent leur règlement. Un seul verrou : `record` n'a qu'un `try_lock`
/// à tenter, et aucun second mode d'échec n'apparaît.
///
/// `pending` est borné par construction : un identifiant y entre avec son
/// événement et en sort avec lui (règlement, ou éviction par la capacité). Il
/// ne peut donc jamais dépasser la capacité du journal.
struct Journal {
    events: VecDeque<RetrievalEvent>,
    pending: HashMap<Uuid, Vec<u64>>,
    /// Le côté lecture du débordement ([`Store::overflow`]) : absorbé dans
    /// `events` par quiconque tient le verrou, avant toute lecture ou écriture.
    overflow: std::sync::mpsc::Receiver<RetrievalEvent>,
}

impl Journal {
    /// Retire `id` des événements en attente de `session` — l'événement est
    /// réglé, ou évincé.
    fn forget_pending(&mut self, session: Uuid, id: u64) {
        if let Some(ids) = self.pending.get_mut(&session) {
            ids.retain(|x| *x != id);
            if ids.is_empty() {
                self.pending.remove(&session);
            }
        }
    }

    /// Ajoute un événement au journal, en évinçant les plus anciens au-delà de
    /// `capacity`, et l'inscrit à l'attente de sa session.
    fn push(&mut self, event: RetrievalEvent, capacity: usize) {
        while self.events.len() >= capacity {
            let Some(evicted) = self.events.pop_front() else {
                break;
            };
            LEDGER.evicted.fetch_add(1, Ordering::Relaxed);
            // Un événement qui sort sans règlement ne sera jamais réglé : compté,
            // et retiré de l'attente — un règlement tardif en ferait un orphelin.
            if evicted.usage.is_none() {
                LEDGER.evicted_unsettled.fetch_add(1, Ordering::Relaxed);
                if let Some(session) = evicted.session_id {
                    self.forget_pending(session, evicted.id);
                }
            }
        }
        if let Some(session) = event.session_id {
            self.pending.entry(session).or_default().push(event.id);
        }
        self.events.push_back(event);
    }

    /// Absorbe les événements partis dans le débordement pendant que le verrou
    /// était pris. Appelé par quiconque tient le verrou, avant de lire ou
    /// d'écrire : le journal est alors complet, dans l'ordre d'arrivée.
    fn absorb_overflow(&mut self) {
        let capacity = CAPACITY.load(Ordering::Relaxed);
        while let Ok(event) = self.overflow.try_recv() {
            self.push(event, capacity);
        }
    }
}

/// Le journal sous son verrou, et le côté écriture du débordement.
///
/// Le débordement est un canal **borné et sans verrou** : quand `record` trouve
/// le verrou pris — un lecteur qui pagine, un autre écrivain préempté le temps
/// d'un `push` — l'événement y part en `try_send` au lieu d'être jeté, et le
/// prochain détenteur du verrou l'absorbe. Jeter un événement du chemin
/// nominal pour une collision de quelques microsecondes ferait sous-compter la
/// mesure elle-même. Le canal plein (personne n'a tenu le verrou depuis
/// [`OVERFLOW_CAPACITY`] collisions) jette, et compte.
struct Store {
    journal: Mutex<Journal>,
    overflow: std::sync::mpsc::SyncSender<RetrievalEvent>,
}

/// Capacité du débordement. Il ne se remplit que par collisions successives
/// sans qu'aucun lecteur ni écrivain ne prenne le verrou entre-temps.
const OVERFLOW_CAPACITY: usize = 256;

fn store() -> &'static Store {
    static STORE: OnceLock<Store> = OnceLock::new();
    STORE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel(OVERFLOW_CAPACITY);
        Store {
            journal: Mutex::new(Journal {
                events: VecDeque::with_capacity(DEFAULT_CAPACITY),
                pending: HashMap::new(),
                overflow: rx,
            }),
            overflow: tx,
        }
    })
}

/// Prend le verrou du journal (en attendant) et absorbe le débordement : la vue
/// rendue est complète. Pour les lecteurs et les règlements, jamais pour `record`.
fn lock_journal() -> std::sync::MutexGuard<'static, Journal> {
    let mut journal = store().journal.lock().unwrap_or_else(|e| e.into_inner());
    journal.absorb_overflow();
    journal
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
    record_for_session(source, trigger, query, results, ok, latency_ms, None)
}

/// [`record`], avec la session qui a déclenché la récupération quand elle est
/// connue. L'événement entre alors dans l'agrégat de cette session et attend son
/// règlement ([`settle_turn`]) à la fin du tour.
///
/// Mêmes garanties que [`record`] : infaillible, non bloquant, compté s'il est jeté.
pub fn record_for_session(
    source: Source,
    trigger: Trigger,
    query: &str,
    results: Vec<RetrievedItem>,
    ok: bool,
    latency_ms: u64,
    session_id: Option<Uuid>,
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

    let event = RetrievalEvent {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        at: chrono::Utc::now().to_rfc3339(),
        session_id,
        source,
        trigger,
        query: query.to_string(),
        results,
        ok,
        latency_ms,
        usage: None,
    };
    let id = event.id;
    // `try_lock` : jamais d'attente sur le chemin chaud. Verrou pris → le
    // débordement sans verrou, que le prochain détenteur absorbe ; débordement
    // plein → jeté, et compté, pas perdu en silence.
    match store().journal.try_lock() {
        Ok(mut journal) => {
            journal.absorb_overflow();
            journal.push(event, capacity);
        }
        Err(_) => {
            if store().overflow.try_send(event).is_err() {
                LEDGER.dropped.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
    }
    Some(id)
}

/// Compte une issue dans les agrégats (champs (4) à (7)), sans toucher au journal.
fn count_usage(usage: &Usage) {
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
}

/// Ce qu'un règlement de fin de tour a fait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Settled {
    /// Événements de la session réglés par cet appel.
    pub events: usize,
    /// Parmi eux, ceux dont au moins un contenu rendu a été utilisé.
    pub used: usize,
}

/// Règle, à la fin d'un tour, tous les événements de `session_id` encore en
/// attente : `used` sont les identifiants que la réponse de l'assistant a
/// **cités** ; chaque événement reçoit l'intersection de `used` avec ce qu'il a
/// rendu (champ (4)). Les champs (5) à (7) restent `None` : on ne les sait pas.
///
/// Attend le verrou (pas le chemin chaud d'une recherche). Une session sans
/// événement en attente ne fait rien : `Settled::default()`.
pub fn settle_turn(session_id: Uuid, used: &[String]) -> Settled {
    let mut journal = lock_journal();
    let Some(ids) = journal.pending.remove(&session_id) else {
        return Settled::default();
    };
    let mut settled = Settled::default();
    for id in ids {
        // L'attente est purgée à l'éviction : un identifiant en attente est dans
        // le journal. S'il n'y était pas, ce serait un orphelin, et compté.
        let Some(event) = journal.events.iter_mut().find(|e| e.id == id) else {
            LEDGER.orphan_outcomes.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        let used_ids: Vec<String> = event
            .results
            .iter()
            .filter(|item| used.contains(&item.id))
            .map(|item| item.id.clone())
            .collect();
        let usage = Usage {
            used_ids,
            ..Default::default()
        };
        count_usage(&usage);
        settled.events += 1;
        if !usage.used_ids.is_empty() {
            settled.used += 1;
        }
        event.usage = Some(usage);
    }
    settled
}

/// Renseigne ce qu'il est advenu de la récupération `event_id` (champs 4 à 7).
///
/// Retourne `false` si l'événement n'est plus dans le journal (évincé par la
/// capacité) : l'appelant n'a rien à en faire, et le manque est compté.
pub fn outcome(event_id: u64, usage: Usage) -> bool {
    count_usage(&usage);

    // Même règle que `recent` : ce n'est pas le chemin chaud d'une recherche, et
    // confondre « verrou pris » avec « événement évincé » ferait mentir le
    // compteur d'orphelins. On attend le verrou.
    let mut journal = lock_journal();
    match journal.events.iter_mut().find(|e| e.id == event_id) {
        Some(event) => {
            // Réglé à la main : il n'attend plus la fin du tour.
            if let Some(session) = event.session_id {
                let id = event.id;
                event.usage = Some(usage);
                journal.forget_pending(session, id);
            } else {
                event.usage = Some(usage);
            }
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
    /// Injections automatiques par tour de chat — à part, comme `hook_calls`.
    pub injection_calls: u64,
    /// Événements jetés : journal indisponible (débordement plein, capacité
    /// nulle). Un chiffre > 0 dit que la mesure est incomplète — il ne se lit pas
    /// comme « rien à signaler ».
    pub dropped: u64,
    /// Événements sortis du journal par la capacité : comptés, plus lisibles.
    pub evicted: u64,
    /// Évincés avant leur règlement : `used`/`unused` sous-comptent d'autant.
    pub evicted_unsettled: u64,
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

/// `num / denom`, ou `None` quand le dénominateur est nul : un ratio sans
/// dénominateur n'est pas zéro, il n'existe pas.
fn ratio(num: u64, denom: u64) -> Option<f64> {
    if denom == 0 {
        None
    } else {
        Some(num as f64 / denom as f64)
    }
}

impl Snapshot {
    /// Coût en appels par résultat non vide. `None` s'il n'y a pas encore de
    /// résultat exploitable — jamais `0.0`, qui laisserait croire à la gratuité.
    pub fn calls_per_nonempty(&self) -> Option<f64> {
        ratio(
            self.calls,
            self.calls
                .saturating_sub(self.empty_results)
                .saturating_sub(self.failures),
        )
    }

    pub fn mean_latency_ms(&self) -> Option<f64> {
        ratio(self.latency_ms_total, self.calls)
    }

    /// Coût par décision correcte : appels par récupération ayant un bénéfice
    /// attribué. `None` tant qu'aucun bénéfice n'a été attribué — un coût calculé
    /// sur zéro décision correcte n'existe pas.
    pub fn cost_per_correct(&self) -> Option<f64> {
        ratio(self.calls, self.benefit_yes)
    }
}

/// La fenêtre dont un agrégat par session est tiré. Elle dit ce que l'agrégat a
/// pu manquer : avec `evicted_total > 0`, des événements de la session ont PU
/// sortir du journal — l'agrégat est « récent », pas « complet ».
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub struct JournalWindow {
    /// Capacité du journal borné.
    pub capacity: usize,
    /// Événements présents, toutes sessions.
    pub len: usize,
    /// Événements évincés depuis le démarrage, toutes sessions.
    pub evicted_total: u64,
}

/// L'agrégat d'UNE session — besoins (appels), déclenchements (par motif),
/// manques (échecs, vides), appels inutiles (`unused`), contenus périmés,
/// bloqués, coût par décision correcte — calculé depuis le journal borné.
///
/// Pas de `dropped` ni d'`orphan_outcomes` ici : ces compteurs sont globaux et
/// une valeur par session serait inventée. La fenêtre dit l'incomplétude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct SessionSnapshot {
    pub session_id: Uuid,
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
    pub injection_calls: u64,
    /// Événements dont le tour n'a pas encore réglé l'issue.
    pub pending: u64,
    pub used: u64,
    pub unused: u64,
    pub stale_hits: u64,
    pub blocked: u64,
    pub benefit_yes: u64,
    pub benefit_no: u64,
    pub window: JournalWindow,
}

impl SessionSnapshot {
    fn empty(session_id: Uuid, window: JournalWindow) -> Self {
        Self {
            session_id,
            calls: 0,
            failures: 0,
            empty_results: 0,
            hits: 0,
            latency_ms_total: 0,
            semantic_calls: 0,
            bm25_calls: 0,
            skill_calls: 0,
            other_calls: 0,
            explicit_calls: 0,
            task_boundary_calls: 0,
            uncertainty_calls: 0,
            contradiction_calls: 0,
            hook_calls: 0,
            injection_calls: 0,
            pending: 0,
            used: 0,
            unused: 0,
            stale_hits: 0,
            blocked: 0,
            benefit_yes: 0,
            benefit_no: 0,
            window,
        }
    }

    fn add(&mut self, event: &RetrievalEvent) {
        self.calls += 1;
        if !event.ok {
            self.failures += 1;
        } else if event.results.is_empty() {
            self.empty_results += 1;
        }
        self.hits += event.results.len() as u64;
        self.latency_ms_total += event.latency_ms;
        match event.source {
            Source::Semantic => self.semantic_calls += 1,
            Source::Bm25 => self.bm25_calls += 1,
            Source::SkillActivate => self.skill_calls += 1,
            Source::Other => self.other_calls += 1,
        }
        match event.trigger {
            Trigger::Explicit => self.explicit_calls += 1,
            Trigger::TaskBoundary => self.task_boundary_calls += 1,
            Trigger::Uncertainty => self.uncertainty_calls += 1,
            Trigger::Contradiction => self.contradiction_calls += 1,
            Trigger::Hook => self.hook_calls += 1,
            Trigger::Injection => self.injection_calls += 1,
        }
        let Some(usage) = &event.usage else {
            self.pending += 1;
            return;
        };
        if usage.used_ids.is_empty() {
            self.unused += 1;
        } else {
            self.used += 1;
        }
        self.stale_hits += usage.stale_ids.len() as u64;
        if usage.blocked.is_some() {
            self.blocked += 1;
        }
        match &usage.benefit {
            Some(Benefit::Yes) => self.benefit_yes += 1,
            Some(Benefit::No { .. }) => self.benefit_no += 1,
            None => {}
        }
    }

    /// Voir [`Snapshot::calls_per_nonempty`].
    pub fn calls_per_nonempty(&self) -> Option<f64> {
        ratio(
            self.calls,
            self.calls
                .saturating_sub(self.empty_results)
                .saturating_sub(self.failures),
        )
    }

    /// Voir [`Snapshot::mean_latency_ms`].
    pub fn mean_latency_ms(&self) -> Option<f64> {
        ratio(self.latency_ms_total, self.calls)
    }

    /// Voir [`Snapshot::cost_per_correct`].
    pub fn cost_per_correct(&self) -> Option<f64> {
        ratio(self.calls, self.benefit_yes)
    }
}

/// L'agrégat récent de `session_id`, calculé depuis le journal borné. Une
/// session inconnue rend des zéros avec sa fenêtre : « rien de récent », ce qui
/// n'est pas « rien depuis le début ».
pub fn snapshot_for_session(session_id: Uuid) -> SessionSnapshot {
    let journal = lock_journal();
    let window = JournalWindow {
        capacity: CAPACITY.load(Ordering::Relaxed),
        len: journal.events.len(),
        evicted_total: LEDGER.evicted.load(Ordering::Relaxed),
    };
    let mut snapshot = SessionSnapshot::empty(session_id, window);
    for event in journal
        .events
        .iter()
        .filter(|e| e.session_id == Some(session_id))
    {
        snapshot.add(event);
    }
    snapshot
}

/// Les `limit` derniers événements de `session_id`, du plus ancien au plus récent.
pub fn recent_for_session(session_id: Uuid, limit: usize) -> Vec<RetrievalEvent> {
    let journal = lock_journal();
    let mine: Vec<&RetrievalEvent> = journal
        .events
        .iter()
        .filter(|e| e.session_id == Some(session_id))
        .collect();
    let skip = mine.len().saturating_sub(limit);
    mine.into_iter().skip(skip).cloned().collect()
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
        injection_calls: t(5),
        dropped: LEDGER.dropped.load(Ordering::Relaxed),
        evicted: LEDGER.evicted.load(Ordering::Relaxed),
        evicted_unsettled: LEDGER.evicted_unsettled.load(Ordering::Relaxed),
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
    let journal = lock_journal();
    let skip = journal.events.len().saturating_sub(limit);
    journal.events.iter().skip(skip).cloned().collect()
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
            &LEDGER.evicted,
            &LEDGER.evicted_unsettled,
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
        let mut journal = lock_journal();
        journal.events.clear();
        journal.pending.clear();
        drop(journal);
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

    /// Un verrou pris (un lecteur, un autre écrivain) : l'appelant n'attend pas,
    /// et l'événement n'est pas jeté — il part dans le débordement et le prochain
    /// détenteur du verrou l'absorbe. Deux écrivains qui se croisent ne font
    /// pas sous-compter la mesure.
    #[test]
    fn un_verrou_pris_ne_jette_pas_l_evenement_et_n_attend_pas() {
        let _g = serial_guard();
        reset();
        let s = Uuid::new_v4();
        let held = store().journal.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        let id = record_for_session(
            Source::Bm25,
            Trigger::Injection,
            "sous verrou",
            vec![item("n", 1.0)],
            true,
            0,
            Some(s),
        );
        assert!(
            started.elapsed() < std::time::Duration::from_millis(200),
            "pas d'attente : {:?}",
            started.elapsed()
        );
        drop(held);
        let id = id.expect("gardé, pas jeté");

        let events = recent(10);
        assert_eq!(events.len(), 1, "absorbé par le lecteur suivant");
        assert_eq!(events[0].id, id);
        assert_eq!(events[0].query, "sous verrou");
        assert_eq!(snapshot().dropped, 0);
        // Et il attend bien son règlement.
        assert_eq!(
            settle_turn(s, &["n".to_string()]),
            Settled { events: 1, used: 1 }
        );
    }

    /// Le débordement est borné : plein, il jette et compte — jamais en silence.
    #[test]
    fn un_debordement_plein_jette_et_compte() {
        let _g = serial_guard();
        reset();
        let held = store().journal.lock().unwrap_or_else(|e| e.into_inner());
        let mut kept = 0;
        for i in 0..(OVERFLOW_CAPACITY + 3) {
            if record(
                Source::Other,
                Trigger::Explicit,
                &i.to_string(),
                vec![],
                true,
                0,
            )
            .is_some()
            {
                kept += 1;
            }
        }
        drop(held);
        assert_eq!(kept, OVERFLOW_CAPACITY);
        let s = snapshot();
        assert_eq!(s.dropped, 3, "le trop-plein est compté");
        assert_eq!(
            s.calls as usize,
            OVERFLOW_CAPACITY + 3,
            "chaque appel reste mesuré"
        );
        assert_eq!(
            recent(1024).len(),
            OVERFLOW_CAPACITY,
            "le reste est absorbé"
        );
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

    /// L'injection automatique par tour n'est ni un des quatre motifs ni un
    /// hook : elle a sa propre colonne, et ne gonfle ni « explicite » ni « hook ».
    #[test]
    fn l_injection_n_est_pas_un_declenchement_explicite_ni_un_hook() {
        let _g = serial_guard();
        reset();
        record(Source::Bm25, Trigger::Injection, "prompt", vec![], true, 2);
        let s = snapshot();
        assert_eq!(s.injection_calls, 1);
        assert_eq!(s.explicit_calls, 0);
        assert_eq!(s.hook_calls, 0);
        assert_eq!(Trigger::Injection.as_str(), "injection");
        assert_eq!(triggers().len(), 6, "six motifs catalogués");
        assert_eq!(triggers()[5], Trigger::Injection);
        assert_eq!(
            serde_json::to_value(Trigger::Injection).unwrap(),
            serde_json::json!("injection")
        );
    }

    fn for_session(session: Option<Uuid>, results: Vec<RetrievedItem>) -> u64 {
        record_for_session(
            Source::Semantic,
            Trigger::Injection,
            "q",
            results,
            true,
            1,
            session,
        )
        .expect("le journal accepte l'événement")
    }

    /// Le règlement de fin de tour : chaque événement de la session reçoit, en
    /// champ (4), l'intersection de ce qu'il a rendu et de ce que la réponse a
    /// cité. Les autres sessions et les événements hors session ne bougent pas ;
    /// les champs (5) à (7) restent `None`.
    #[test]
    fn settle_turn_regle_les_evenements_de_la_session_avec_les_ids_cites() {
        let _g = serial_guard();
        reset();
        let s = Uuid::new_v4();
        let other = Uuid::new_v4();
        let a = for_session(Some(s), vec![item("n1", 0.9), item("n2", 0.5)]);
        let b = for_session(Some(s), vec![item("n3", 0.4)]);
        let c = for_session(Some(other), vec![item("n1", 0.9)]);
        let d = for_session(None, vec![item("n1", 0.9)]);

        let settled = settle_turn(s, &["n1".to_string(), "zz".to_string()]);
        assert_eq!(settled, Settled { events: 2, used: 1 });

        let events = recent(10);
        let find = |id: u64| events.iter().find(|e| e.id == id).unwrap();
        let ua = find(a).usage.clone().expect("réglé");
        assert_eq!(ua.used_ids, vec!["n1".to_string()]);
        assert!(ua.validation.is_none() && ua.observable.is_none());
        assert!(
            ua.benefit.is_none(),
            "un bénéfice inconnu n'est pas « non »"
        );
        let ub = find(b).usage.clone().expect("réglé");
        assert!(ub.used_ids.is_empty(), "rien de rendu n'a été cité");
        assert!(find(c).usage.is_none(), "l'autre session attend son tour");
        assert!(find(d).usage.is_none(), "hors session : jamais réglé");
        assert_eq!(find(a).session_id, Some(s));
        assert_eq!(find(d).session_id, None);

        let snap = snapshot();
        assert_eq!(snap.used, 1);
        assert_eq!(snap.unused, 1);
        assert_eq!(snap.orphan_outcomes, 0, "chemin nominal : aucun orphelin");

        // Un second règlement de la même session n'a plus rien à régler.
        assert_eq!(settle_turn(s, &["n1".to_string()]), Settled::default());
        assert_eq!(snapshot().used, 1, "rien n'est compté deux fois");
    }

    /// Un `outcome` posé à la main retire l'événement de l'attente : le règlement
    /// de fin de tour ne le compte pas une seconde fois.
    #[test]
    fn un_outcome_manuel_retire_l_evenement_de_l_attente() {
        let _g = serial_guard();
        reset();
        let s = Uuid::new_v4();
        let id = for_session(Some(s), vec![item("n", 1.0)]);
        assert!(outcome(
            id,
            Usage {
                used_ids: vec!["n".into()],
                ..Default::default()
            }
        ));
        assert_eq!(settle_turn(s, &[]), Settled::default());
        assert_eq!(snapshot().used, 1);
        assert_eq!(snapshot().unused, 0);
    }

    /// L'éviction d'un événement non réglé est comptée et purge son attente :
    /// le règlement qui suit ne produit pas d'orphelin.
    #[test]
    fn l_eviction_purge_l_attente_et_se_compte() {
        let _g = serial_guard();
        reset();
        set_journal_capacity(2);
        let s = Uuid::new_v4();
        for_session(Some(s), vec![item("a", 1.0)]);
        for_session(Some(s), vec![item("b", 1.0)]);
        for_session(Some(s), vec![item("c", 1.0)]);

        let snap = snapshot();
        assert_eq!(snap.evicted, 1);
        assert_eq!(snap.evicted_unsettled, 1, "évincé avant son règlement");

        let settled = settle_turn(s, &["c".to_string()]);
        assert_eq!(settled, Settled { events: 2, used: 1 });
        assert_eq!(snapshot().orphan_outcomes, 0);
        set_journal_capacity(DEFAULT_CAPACITY);
    }

    /// Un événement déjà réglé qui sort du journal n'est pas « non réglé ».
    #[test]
    fn un_evince_deja_regle_ne_compte_pas_comme_non_regle() {
        let _g = serial_guard();
        reset();
        set_journal_capacity(1);
        let s = Uuid::new_v4();
        for_session(Some(s), vec![item("a", 1.0)]);
        settle_turn(s, &["a".to_string()]);
        for_session(Some(s), vec![item("b", 1.0)]);
        let snap = snapshot();
        assert_eq!(snap.evicted, 1);
        assert_eq!(snap.evicted_unsettled, 0);
        set_journal_capacity(DEFAULT_CAPACITY);
    }

    /// L'agrégat par session : calculé depuis le journal, borné, et il porte sa
    /// fenêtre. Une session inconnue rend des zéros avec la même fenêtre.
    #[test]
    fn snapshot_for_session_est_recent_et_porte_sa_fenetre() {
        let _g = serial_guard();
        reset();
        let s = Uuid::new_v4();
        let other = Uuid::new_v4();
        for_session(Some(s), vec![item("n1", 0.9)]);
        record_for_session(
            Source::Bm25,
            Trigger::Explicit,
            "q",
            vec![],
            false,
            4,
            Some(s),
        );
        record_for_session(Source::Bm25, Trigger::Hook, "q", vec![], true, 2, Some(s));
        for_session(Some(other), vec![item("x", 0.1)]);

        let before = snapshot_for_session(s);
        assert_eq!(before.calls, 3);
        assert_eq!(before.failures, 1);
        assert_eq!(before.empty_results, 1);
        assert_eq!(before.hits, 1);
        assert_eq!(before.semantic_calls, 1);
        assert_eq!(before.bm25_calls, 2);
        assert_eq!(before.injection_calls, 1);
        assert_eq!(before.explicit_calls, 1);
        assert_eq!(before.hook_calls, 1);
        assert_eq!(before.pending, 3, "rien n'est encore réglé");
        assert_eq!(before.used + before.unused, 0);
        assert_eq!(before.mean_latency_ms(), Some(7.0 / 3.0));
        assert_eq!(before.calls_per_nonempty(), Some(3.0));
        assert_eq!(before.cost_per_correct(), None);
        assert_eq!(before.window.capacity, DEFAULT_CAPACITY);
        assert_eq!(
            before.window.len, 4,
            "la fenêtre compte toutes les sessions"
        );
        assert_eq!(before.window.evicted_total, 0);

        settle_turn(s, &["n1".to_string()]);
        let after = snapshot_for_session(s);
        assert_eq!(after.pending, 0);
        assert_eq!(after.used, 1);
        assert_eq!(after.unused, 2);

        let unknown = snapshot_for_session(Uuid::new_v4());
        assert_eq!(unknown.calls, 0);
        assert_eq!(unknown.window, before.window);
        assert_eq!(unknown.calls_per_nonempty(), None);
        assert_eq!(unknown.mean_latency_ms(), None);
    }

    /// Les champs (5) à (7) posés à la main entrent aussi dans l'agrégat de la
    /// session — y compris les contenus périmés et les blocages.
    #[test]
    fn snapshot_for_session_compte_les_champs_renseignes_a_la_main() {
        let _g = serial_guard();
        reset();
        let s = Uuid::new_v4();
        let a = for_session(Some(s), vec![item("n", 1.0)]);
        let b = for_session(Some(s), vec![item("m", 1.0)]);
        outcome(
            a,
            Usage {
                used_ids: vec!["n".into()],
                stale_ids: vec!["n".into()],
                blocked: None,
                validation: Some(Validation::Sha("abc".into())),
                observable: None,
                benefit: Some(Benefit::Yes),
            },
        );
        outcome(
            b,
            Usage {
                used_ids: vec![],
                stale_ids: vec![],
                blocked: Some("note absente".into()),
                validation: None,
                observable: None,
                benefit: Some(Benefit::No {
                    reason: "rien d'utile".into(),
                }),
            },
        );
        let snap = snapshot_for_session(s);
        assert_eq!(snap.stale_hits, 1);
        assert_eq!(snap.blocked, 1);
        assert_eq!(snap.benefit_yes, 1);
        assert_eq!(snap.benefit_no, 1);
        assert_eq!(snap.cost_per_correct(), Some(2.0));
    }

    #[test]
    fn recent_for_session_filtre_et_borne() {
        let _g = serial_guard();
        reset();
        let s = Uuid::new_v4();
        for q in ["un", "deux", "trois"] {
            record_for_session(
                Source::Other,
                Trigger::Explicit,
                q,
                vec![],
                true,
                0,
                Some(s),
            );
        }
        record_for_session(
            Source::Other,
            Trigger::Explicit,
            "autre",
            vec![],
            true,
            0,
            None,
        );
        let mine = recent_for_session(s, 2);
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].query, "deux");
        assert_eq!(mine[1].query, "trois");
        assert!(mine.iter().all(|e| e.session_id == Some(s)));
    }
}
