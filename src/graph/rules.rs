//! Le MOTEUR DE RÈGLES du tissu de connaissance : une règle déclare une mesure et
//! l'arête qu'elle produit ; chaque arête écrite porte sa provenance.
//!
//! # Pourquoi un moteur, et pas du code au cas par cas
//!
//! Jusqu'ici, chaque mécanisme écrivait ses arêtes en dur — les synapses dans
//! `notes/coactivation.rs`, le co-changement ailleurs. Ajouter une arête dérivée
//! demandait donc du code neuf, et **rien ne disait pourquoi l'arête existait** : elle
//! était là, ou pas, et il fallait lire le module pour le comprendre.
//!
//! Ici, une règle est une **déclaration** : un nom, une justification en clair, le type
//! d'arête et le label qu'elle atteint. Le moteur l'exécute en appelant les primitives
//! du store — jamais un modèle de langage. La différence est ce qui rend le résultat
//! reproductible : deux passages sur les mêmes données produisent les mêmes arêtes.
//!
//! # La règle de provenance, sans exception
//!
//! Chaque arête écrite porte `rule`, `measure` et `at`. `measure` est la **mesure**
//! qui l'a justifiée, en clair, avec ses chiffres (« 3 portées lues, 7 notes
//! épinglées »). Une arête sans provenance est un bug — et ce n'est pas une convention :
//! les écrivains du store ([`GraphStore::link_agent_knows`], [`GraphStore::link_agent_uses`],
//! [`GraphStore::coactivate_session`]) n'acceptent que [`AccountableProvenance`], un type
//! qu'on n'obtient qu'en passant par [`Provenance::accountable`], qui REFUSE une règle,
//! une mesure ou une date vide avec une [`ProvenanceError`]. Une provenance vide ne
//! compile pas jusqu'à l'écriture.
//!
//! C'est ce qui rend une croyance **contestable** : on peut relire la mesure, la
//! refaire, et constater qu'elle a changé. Une note qui affirme sans mesure est une
//! opinion ; une arête qui porte sa mesure est une observation datée.
//!
//! Les SYNAPSE antérieures au moteur n'ont pas de provenance : elles sont exemptées
//! (décision 651e…), et aucune migration ne les réécrit. La garantie porte sur ce qui
//! s'écrit à partir de maintenant.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;

/// Ce qu'une arête dérivée porte, pour être contestable.
///
/// C'est le **relevé** : ce qu'on lit sur une arête, ou ce qu'on s'apprête à y écrire.
/// Ses champs sont publics parce qu'une lecture du graphe peut rendre n'importe quoi — y
/// compris une arête héritée sans règle. Pour ÉCRIRE, il faut d'abord le prouver :
/// [`Provenance::accountable`] rend l'[`AccountableProvenance`] que les écrivains exigent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// The rule that drew the edge.
    pub rule: String,
    /// The measurement that justified it, in words and numbers.
    pub measure: String,
    /// When it was measured (RFC 3339).
    pub at: String,
    /// The deterministic score that ranked it, when the rule ranks. It is what orders the
    /// read: without it, the injection would pick the twelve most RECENT knowledges instead
    /// of the twelve strongest — recency is not relevance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl Provenance {
    /// A provenance stamped now.
    pub fn new(rule: &str, measure: impl Into<String>) -> Self {
        Self {
            rule: rule.to_string(),
            measure: measure.into(),
            at: chrono::Utc::now().to_rfc3339(),
            score: None,
        }
    }

    /// A provenance that also carries the score the candidate was ranked by.
    pub fn scored(rule: &str, measure: impl Into<String>, score: f64) -> Self {
        Self {
            score: Some(score),
            ..Self::new(rule, measure)
        }
    }

    /// Whether this provenance can be contested: it must name a rule and a measure. An
    /// edge whose origin cannot be named is indistinguishable from an accident.
    pub fn is_accountable(&self) -> bool {
        self.check().is_ok()
    }

    /// Ce qui manque pour qu'elle soit contestable, nommé. `Ok(())` quand rien ne manque.
    pub fn check(&self) -> Result<(), ProvenanceError> {
        if self.rule.trim().is_empty() {
            return Err(ProvenanceError::EmptyRule);
        }
        if self.measure.trim().is_empty() {
            return Err(ProvenanceError::EmptyMeasure);
        }
        if self.at.trim().is_empty() {
            return Err(ProvenanceError::EmptyAt);
        }
        Ok(())
    }

    /// La preuve qu'un écrivain exige : la provenance nomme une règle, une mesure et une
    /// date, sinon elle est REFUSÉE ici — avant toute requête, sans panique, sans arête.
    pub fn accountable(self) -> Result<AccountableProvenance, ProvenanceError> {
        self.check()?;
        Ok(AccountableProvenance(self))
    }
}

/// Pourquoi une provenance ne peut pas être écrite. Une erreur typée : l'appelant peut
/// la distinguer d'une panne du graphe, et un test peut la nommer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceError {
    /// Aucune règle nommée : l'arête ne pourrait pas dire qui l'a tracée.
    EmptyRule,
    /// Aucune mesure : l'arête ne pourrait pas dire pourquoi.
    EmptyMeasure,
    /// Aucune date : l'observation ne serait pas datée.
    EmptyAt,
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRule => write!(f, "provenance refusée : aucune règle nommée"),
            Self::EmptyMeasure => write!(f, "provenance refusée : aucune mesure"),
            Self::EmptyAt => write!(f, "provenance refusée : aucune date"),
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// Une provenance qui a PROUVÉ qu'elle nomme une règle, une mesure et une date.
///
/// C'est le seul type que les écrivains d'arêtes dérivées acceptent, et son champ est
/// privé : la seule porte est [`Provenance::accountable`] (ou les raccourcis
/// [`AccountableProvenance::new`] et [`AccountableProvenance::scored`]), qui refusent le
/// vide. Un relevé lu sur une arête héritée ne peut donc pas être recopié tel quel sur
/// une arête neuve sans repasser par la preuve.
///
/// Un écrivain ne prend pas un `&Provenance` — c'est le compilateur qui le dit :
///
/// ```compile_fail
/// use project_orchestrator::graph::rules::{KnowledgeScope, Provenance};
/// use project_orchestrator::neo4j::mock::MockGraphStore;
/// use project_orchestrator::neo4j::traits::GraphStore;
///
/// async fn write(graph: &MockGraphStore) {
///     let empty = Provenance::new("", "");
///     let _ = graph
///         .link_agent_knows(
///             uuid::Uuid::new_v4(),
///             uuid::Uuid::new_v4(),
///             KnowledgeScope::Global,
///             None,
///             &empty, // erreur : `&Provenance` n'est pas `&AccountableProvenance`
///         )
///         .await;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccountableProvenance(Provenance);

impl AccountableProvenance {
    /// Une provenance datée de maintenant, prouvée — ou refusée.
    pub fn new(rule: &str, measure: impl Into<String>) -> Result<Self, ProvenanceError> {
        Provenance::new(rule, measure).accountable()
    }

    /// Une provenance datée de maintenant qui porte aussi son score, prouvée — ou refusée.
    pub fn scored(
        rule: &str,
        measure: impl Into<String>,
        score: f64,
    ) -> Result<Self, ProvenanceError> {
        Provenance::scored(rule, measure, score).accountable()
    }

    /// Le relevé tel qu'il sera écrit.
    pub fn as_provenance(&self) -> &Provenance {
        &self.0
    }

    /// Le relevé, en reprenant la propriété.
    pub fn into_inner(self) -> Provenance {
        self.0
    }
}

impl std::ops::Deref for AccountableProvenance {
    type Target = Provenance;

    fn deref(&self) -> &Provenance {
        &self.0
    }
}

/// A declared rule. It says what it produces, never how: the how is one function per
/// rule, below, and it may only measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    /// Stable identifier, the one written on every edge it draws.
    pub name: &'static str,
    /// Why this edge carries meaning, in one sentence a reader can disagree with.
    pub justification: &'static str,
    /// The relationship type it writes.
    pub edge: &'static str,
    /// The label it reaches.
    pub target_label: &'static str,
}

/// L'échelle à laquelle une connaissance vaut.
///
/// Une fondation par projet, et une fondation globale qui n'est pas un fourre-tout : elle
/// contient ce qui a été MESURÉ comme traversant plusieurs projets, plus ce qui est né
/// global. C'est ce qui permet d'injecter la bonne échelle au bon moment au lieu de servir
/// les douze notes les plus récentes de toute l'instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KnowledgeScope {
    /// The note belongs to one project's foundation.
    Project,
    /// The note crosses scopes, or was born global.
    Global,
}

impl KnowledgeScope {
    pub fn as_str(self) -> &'static str {
        match self {
            KnowledgeScope::Project => "project",
            KnowledgeScope::Global => "global",
        }
    }
}

/// One note a rule would put in a foundation, with the measurement that ranked it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundationCandidate {
    pub note: Uuid,
    /// Degré de co-activation : le seul signal d'usage qui existe aujourd'hui.
    pub synapses: i64,
    /// Nombre de fichiers du projet qui ancrent la note.
    pub anchors: i64,
    pub energy: f64,
    /// Le score déterministe qui a servi à classer.
    pub score: f64,
}

/// What the self knows, and how much of it was read.
///
/// The pair is the whole point: `notes` is a bounded read, `available` is the truth about
/// the size. A reader that published only `notes` would make a truncated answer look
/// complete — the failure the graph API already has (`unwrap_or(5000)` with no
/// `truncated`), and the one this type exists to stop repeating.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentKnowledge {
    /// The notes read, most recently attested first.
    pub notes: Vec<Uuid>,
    /// How many the agent knows in total.
    pub available: usize,
}

impl AgentKnowledge {
    /// The notes NOT read because of the bound. Zero means the read was whole.
    pub fn withheld(&self) -> usize {
        self.available.saturating_sub(self.notes.len())
    }
}

/// `(:Agent {kind:'self'})-[:KNOWS]->(:Note)` — le socle déclaré du sujet.
///
/// Une note épinglée est une fondation assumée : aucune question ne la rend pertinente,
/// c'est pourquoi elle est injectée sans être classée. La règle la transforme en arête,
/// pour que le sujet la **lise dans le graphe** au lieu de la tenir d'un réglage.
pub const AGENT_KNOWS_CORE: Rule = Rule {
    name: "agent-knows-core",
    justification: "Une note épinglée pour l'instance ou pour un projet est une fondation \
                    déclarée du soi : elle devient KNOWS, lisible sans configuration.",
    edge: "KNOWS",
    target_label: "Note",
};

/// `(:Agent {kind:'self'})-[:USES]->(:Persona | :Skill)` — ce que le sujet s'applique.
///
/// La mesure vient du registre d'activation : les étages de persona et d'activation de
/// skills enregistrent ce qu'ils ont injecté dans une session. Cette liste vivait en
/// mémoire et mourait avec le processus ; la règle en fait une arête durable, avec la
/// mesure (« 2 sessions sur 9 ») écrite dessus.
pub const AGENT_USES_ACTIVATED: Rule = Rule {
    name: "agent-uses-activated",
    justification: "Une persona ou un skill réellement injecté dans une session est \
                    appliqué par l'agent de cette session : c'est une mesure, pas un \
                    réglage, et elle devient USES.",
    edge: "USES",
    target_label: "Persona|Skill",
};

/// The labels a `USES` edge may reach. A label is interpolated into a query, so it is
/// matched against this closed list first.
pub const USABLE_LABELS: &[&str] = &["Persona", "Skill", "Protocol"];

/// `(:Note)-[:SYNAPSE]->(:Note)` entre les notes qu'une session a réellement eues en
/// contexte, renforcée à la clôture.
///
/// L'expérience devient structurante : jusqu'ici les 65 069 synapses reliaient des notes
/// entre elles sans qu'aucune ne dise quelle session les avait activées. La mesure vient
/// du registre d'activation (les notes rendues dans le tour), et elle est écrite sur
/// l'arête — c'est ce qui rend le renforcement contestable.
pub const SESSION_CO_ACTIVATION: Rule = Rule {
    name: "session-co-activation",
    justification: "Deux notes présentes dans le même contexte et durcies par une même \
                    session sont liées : c'est une co-occurrence mesurée, pas une \
                    similarité supposée.",
    edge: "SYNAPSE",
    target_label: "Note",
};

/// Le classement d'une note dans une fondation. Une seule définition, pure.
///
/// `2·synapses + anchors(plafonné à 50) + 10·énergie`.
///
/// - les **synapses** pèsent double : c'est le seul signal d'usage qui existe aujourd'hui
///   (`activation_count` est à 0 partout — mesuré sur 28 870 notes) ;
/// - les **ancres** sont plafonnées : une note liée à 1 741 fichiers écraserait tout le
///   reste, et « ancrée partout » n'est pas « plus vraie » — c'est souvent une note de
///   structure, pas une leçon ;
/// - l'**énergie** vaut 10 points au maximum : elle départage, elle ne domine pas.
pub fn rank_score(synapses: i64, anchors: i64, energy: f64) -> f64 {
    2.0 * synapses as f64 + anchors.clamp(0, 50) as f64 + 10.0 * energy
}

/// `(:Agent)-[:KNOWS {scope:'project'}]->(:Note)` — la fondation d'UN projet.
///
/// Ce qu'une personne qui travaille sur ce projet doit avoir sous la main : ses notes
/// actives, importantes, fraîches, ancrées dans du code du projet, classées par résonance.
pub const AGENT_KNOWS_PROJECT: Rule = Rule {
    name: "agent-knows-project",
    justification: "Une note active, importante, fraîche et ancrée dans le code d'un \
                    projet est le socle de CE projet : elle devient une connaissance \
                    d'échelle projet, classée par résonance mesurée.",
    edge: "KNOWS",
    target_label: "Note",
};

/// `(:Agent)-[:KNOWS {scope:'global'}]->(:Note)` — l'échelle qui transcende les projets.
///
/// Elle n'est pas un fourre-tout : elle contient ce qui est ancré dans des fichiers de
/// **plusieurs** projets (l'escalade est mesurée, pas décidée) et ce qui est né global.
pub const AGENT_KNOWS_GLOBAL: Rule = Rule {
    name: "agent-knows-global",
    justification: "Une note ancrée dans des fichiers de plusieurs projets a déjà traversé \
                    une frontière : elle vaut au-delà de son projet d'origine, et devient \
                    une connaissance d'échelle globale.",
    edge: "KNOWS",
    target_label: "Note",
};

/// Combien de notes par projet, et combien à l'échelle globale. Bornes basses assumées :
/// une fondation qui déborde n'est plus une fondation, c'est du contexte.
pub const FOUNDATION_PER_PROJECT: usize = 6;
pub const FOUNDATION_GLOBAL: usize = 8;
/// Combien de projets on parcourt par passage. Au-delà, la part lue est publiée.
pub const FOUNDATION_MAX_PROJECTS: usize = 50;

/// The rules this engine knows, in a stable order.
pub const RULES: &[Rule] = &[
    AGENT_KNOWS_CORE,
    AGENT_KNOWS_PROJECT,
    AGENT_KNOWS_GLOBAL,
    AGENT_USES_ACTIVATED,
    SESSION_CO_ACTIVATION,
];

/// The declared rule of that name, if any.
pub fn rule(name: &str) -> Option<&'static Rule> {
    RULES.iter().find(|r| r.name == name)
}

/// What one rule did on one run. `considered` and `written` are both published: a rule
/// that considered nothing is not a rule that succeeded silently, it is a rule with
/// nothing to say.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleOutcome {
    pub name: String,
    pub considered: usize,
    pub written: usize,
    /// Edges that were already there, re-attested rather than duplicated.
    pub skipped: usize,
    /// Edges this rule RELEASED: it had drawn them, and its measure no longer holds.
    /// A rule that only adds is not a rule, it is a log that never strikes a line.
    pub released: usize,
    pub measure: String,
}

/// The measurement a rule needs when it cannot be read from the graph alone.
///
/// Most rules read the graph and need nothing else. One does not: what an agent applied to
/// itself exists only in memory, in the activation registry, and the close of a session is
/// the only moment it can become edges. Rather than a second engine, the same engine takes
/// the measurement as an argument — and the rule stays a declaration either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Measurement {
    /// What the stages injected while one session ran.
    Activation {
        session: Uuid,
        agent: Uuid,
        personas: Vec<Uuid>,
        skills: Vec<Uuid>,
    },
    /// The notes a turn really had in context, and the project that owns them.
    SessionNotes {
        session: Uuid,
        project: Option<Uuid>,
        notes: Vec<Uuid>,
    },
}

/// The outcome of a run, rule by rule.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RulesReport {
    pub rules: Vec<RuleOutcome>,
    pub edges_written: usize,
}

impl RulesReport {
    /// The outcome of one rule, if it ran.
    pub fn of(&self, name: &str) -> Option<&RuleOutcome> {
        self.rules.iter().find(|o| o.name == name)
    }
}

/// Le budget du passage au démarrage : un graphe lent coûte la matérialisation du socle,
/// jamais le démarrage du serveur.
pub const BOOT_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// Applique toutes les règles une fois, au démarrage, sous budget.
///
/// Extrait de `main` pour être testable : un bloc dans la fonction de démarrage ne
/// s'exécute dans aucun test, donc il ne prouve rien — il était à 0 % de couverture, et
/// c'était exact.
///
/// `None` signifie « pas de rapport » : budget épuisé ou erreur. Jamais une panique, jamais
/// un démarrage refusé.
pub async fn run_at_boot(graph: &dyn GraphStore) -> Option<RulesReport> {
    match tokio::time::timeout(BOOT_BUDGET, run(graph, &[])).await {
        Ok(Ok(report)) => {
            // Le rapport se dit ici, pas dans `main` : des lignes de journalisation dans la
            // fonction de démarrage ne sont exécutées par aucun test, donc ne prouvent rien.
            for rule in &report.rules {
                tracing::info!(
                    rule = %rule.name,
                    considered = rule.considered,
                    written = rule.written,
                    released = rule.released,
                    "{}",
                    rule.measure
                );
            }
            Some(report)
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "the knowledge rules could not run at boot");
            None
        }
        Err(_) => {
            tracing::warn!("the knowledge rules timed out at boot");
            None
        }
    }
}

/// Runs the named rules (all of them when `names` is empty). Deterministic: it reads
/// counts and settings, never text, and writes nothing but the edges it declares.
pub async fn run(graph: &dyn GraphStore, names: &[&str]) -> anyhow::Result<RulesReport> {
    let selected: Vec<&'static Rule> = if names.is_empty() {
        RULES.iter().collect()
    } else {
        let mut out = Vec::new();
        for name in names {
            let r = rule(name).ok_or_else(|| anyhow::anyhow!("unknown rule: {name}"))?;
            out.push(r);
        }
        out
    };

    let mut report = RulesReport::default();
    for r in selected {
        let outcome = match r.name {
            "agent-knows-core" => run_agent_knows_core(graph).await?,
            "agent-knows-project" => run_agent_knows_project(graph).await?,
            "agent-knows-global" => run_agent_knows_global(graph).await?,
            "session-co-activation" => RuleOutcome {
                name: r.name.to_string(),
                measure: "non exécutée ici : sa mesure vient de l'appelant (les notes \
                          injectées dans la session), pas du graphe — voir `run_with`"
                    .to_string(),
                ..Default::default()
            },
            "agent-uses-activated" => RuleOutcome {
                name: r.name.to_string(),
                measure: "non exécutée ici : sa mesure vient de l'appelant (le registre \
                          d'activation), pas du graphe — voir `run_with`"
                    .to_string(),
                ..Default::default()
            },
            other => anyhow::bail!("rule {other} is declared but has no executor"),
        };
        report.edges_written += outcome.written;
        report.rules.push(outcome);
    }
    Ok(report)
}

/// Runs one rule whose measurement the caller holds. Deterministic all the same: the input
/// is a list of ids that were measured, never text.
pub async fn run_with(
    graph: &dyn GraphStore,
    rule_name: &str,
    measurement: Measurement,
) -> anyhow::Result<RuleOutcome> {
    let r = rule(rule_name).ok_or_else(|| anyhow::anyhow!("unknown rule: {rule_name}"))?;
    match (r.name, measurement) {
        (
            "agent-uses-activated",
            Measurement::Activation {
                session,
                agent,
                personas,
                skills,
            },
        ) => run_agent_uses_activated(graph, session, agent, &personas, &skills).await,
        (
            "session-co-activation",
            Measurement::SessionNotes {
                session,
                project,
                notes,
            },
        ) => run_session_co_activation(graph, session, project, &notes).await,
        (name, _) => anyhow::bail!("rule {name} does not take this measurement"),
    }
}

/// La fondation d'échelle PROJET : pour chaque projet, ses notes les plus résonnées.
async fn run_agent_knows_project(graph: &dyn GraphStore) -> anyhow::Result<RuleOutcome> {
    let self_id = graph.agent_identity_namespace().await?;
    graph.ensure_agent(self_id, None).await?;

    let projects = match graph.list_projects().await {
        Ok(p) => p,
        Err(e) => {
            // NE PAS réconcilier sur une lecture ratée : un `keep` vide effacerait tout le
            // socle de projet à cause d'une erreur transitoire. Une lecture incomplète ne
            // détruit rien ; elle dit qu'elle n'a pas pu lire, et laisse l'état en place.
            tracing::warn!(error = %e, "project list unreadable: foundation left untouched");
            return Ok(RuleOutcome {
                name: AGENT_KNOWS_PROJECT.name.to_string(),
                measure: format!(
                    "non réconciliée : la liste des projets est illisible ({e}) — le socle \
                     existant est laissé en place"
                ),
                ..Default::default()
            });
        }
    };
    let total = projects.len();
    let read: Vec<_> = projects.into_iter().take(FOUNDATION_MAX_PROJECTS).collect();

    let mut keep: Vec<Uuid> = Vec::new();
    let (mut considered, mut written, mut skipped) = (0usize, 0usize, 0usize);
    for project in &read {
        let candidates = graph
            .ranked_foundation(Some(project.id), FOUNDATION_PER_PROJECT)
            .await?;
        considered += candidates.len();
        for c in candidates {
            let provenance = AccountableProvenance::scored(
                AGENT_KNOWS_PROJECT.name,
                format!(
                    "projet {}, {} synapse(s), {} ancre(s), énergie {:.2}",
                    project.slug, c.synapses, c.anchors, c.energy
                ),
                c.score,
            )?;
            let created = graph
                .link_agent_knows(
                    self_id,
                    c.note,
                    KnowledgeScope::Project,
                    Some(project.id),
                    &provenance,
                )
                .await?;
            if created {
                written += 1;
            } else {
                skipped += 1;
            }
            keep.push(c.note);
        }
    }

    let released = graph
        .release_agent_knows(self_id, AGENT_KNOWS_PROJECT.name, &keep)
        .await?;

    // « la part lue est dite » : au-delà de FOUNDATION_MAX_PROJECTS, on ne prétend pas
    // avoir couvert l'instance.
    let truncation = if total > read.len() {
        format!(", {} projet(s) NON lus", total - read.len())
    } else {
        String::new()
    };
    let mut measure = format!(
        "{} projet(s) lu(s) sur {total}, {considered} note(s) candidate(s){truncation}",
        read.len()
    );
    if released > 0 {
        measure.push_str(&format!(", {released} arête(s) libérée(s)"));
    }

    Ok(RuleOutcome {
        name: AGENT_KNOWS_PROJECT.name.to_string(),
        considered,
        written,
        skipped,
        released,
        measure,
    })
}

/// L'échelle GLOBALE : ce qui a traversé plusieurs projets, et ce qui est né global.
async fn run_agent_knows_global(graph: &dyn GraphStore) -> anyhow::Result<RuleOutcome> {
    let self_id = graph.agent_identity_namespace().await?;
    graph.ensure_agent(self_id, None).await?;

    let candidates = graph.ranked_foundation(None, FOUNDATION_GLOBAL).await?;
    let considered = candidates.len();
    let mut keep: Vec<Uuid> = Vec::new();
    let (mut written, mut skipped) = (0usize, 0usize);
    let mut top_score = 0.0f64;

    for c in &candidates {
        top_score = top_score.max(c.score);
        let provenance = AccountableProvenance::scored(
            AGENT_KNOWS_GLOBAL.name,
            format!(
                "{} synapse(s), {} ancre(s) dans plusieurs projets, énergie {:.2}",
                c.synapses, c.anchors, c.energy
            ),
            c.score,
        )?;
        let created = graph
            .link_agent_knows(self_id, c.note, KnowledgeScope::Global, None, &provenance)
            .await?;
        if created {
            written += 1;
        } else {
            skipped += 1;
        }
        keep.push(c.note);
    }

    let released = graph
        .release_agent_knows(self_id, AGENT_KNOWS_GLOBAL.name, &keep)
        .await?;

    let mut measure = format!(
        "{considered} note(s) escaladée(s) ou née(s) globale(s), meilleur score {top_score:.1}"
    );
    if released > 0 {
        measure.push_str(&format!(", {released} arête(s) libérée(s)"));
    }

    Ok(RuleOutcome {
        name: AGENT_KNOWS_GLOBAL.name.to_string(),
        considered,
        written,
        skipped,
        released,
        measure,
    })
}

async fn run_agent_uses_activated(
    graph: &dyn GraphStore,
    session: Uuid,
    agent: Uuid,
    personas: &[Uuid],
    skills: &[Uuid],
) -> anyhow::Result<RuleOutcome> {
    let considered = personas.len() + skills.len();
    if considered == 0 {
        return Ok(RuleOutcome {
            name: AGENT_USES_ACTIVATED.name.to_string(),
            measure: format!("session {session} : rien n'a été activé"),
            ..Default::default()
        });
    }

    // The rule names a subject: it makes it. A rule that silently wrote nothing because the
    // node happened to be missing would report « 0 written » on a graph where it should have
    // written — the failure mode that looks like a successful empty run.
    graph.ensure_agent(agent, None).await?;

    let measure = format!(
        "{} persona(s) et {} skill(s) activés dans la session {session}",
        personas.len(),
        skills.len()
    );
    let provenance = AccountableProvenance::new(AGENT_USES_ACTIVATED.name, measure.clone())?;

    let mut written = 0;
    let mut skipped = 0;
    for (label, ids) in [("Persona", personas), ("Skill", skills)] {
        for id in ids {
            if graph
                .link_agent_uses(agent, label, *id, &provenance)
                .await?
            {
                written += 1;
            } else {
                skipped += 1;
            }
        }
    }

    Ok(RuleOutcome {
        name: AGENT_USES_ACTIVATED.name.to_string(),
        considered,
        written,
        skipped,
        // `agent-uses-activated` ne réconcilie pas encore : il atteste. Nommer le zéro
        // plutôt que l'oublier — un champ absent se lit comme « non mesuré ».
        released: 0,
        measure,
    })
}

/// The pins of every scope: the instance's own, then each project's. A project scope is
/// read through its slug, which is what [`crate::chat::stages::knowledge_injection`]
/// keys the setting by.
/// Les épingles de toutes les portées, lues DIRECTEMENT sur les réglages.
///
/// On n'énumère plus les projets : la portée est écrite sur le réglage lui-même, et passer
/// par `list_projects()` rendait la réconciliation otage d'un seul `(:Project)` illisible —
/// un nœud malformé désactivait silencieusement la réconciliation de toute l'instance.
/// L'identifiant de projet d'une portée n'est résolu que pour les portées qui PORTENT une
/// épingle, donc quelques appels au plus.
async fn pinned_notes_everywhere(
    graph: &dyn GraphStore,
) -> anyhow::Result<(Vec<(Uuid, Option<Uuid>)>, usize, bool)> {
    use crate::chat::stages::knowledge_injection::{PINNED_NOTES_KEY, PROJECT_SCOPE_PREFIX};

    let settings = graph.llm_settings_by_key(PINNED_NOTES_KEY).await?;
    let read = settings.len();

    let mut seen = std::collections::HashSet::new();
    let mut ids = Vec::new();
    let mut complete = true;
    for (scope, raw) in settings {
        // Une portée projet est `project:<slug>` ; tout le reste est global. On retire le
        // préfixe AVANT de résoudre : passer la portée entière ne résout jamais un projet.
        let project = match scope.strip_prefix(PROJECT_SCOPE_PREFIX) {
            None => None,
            Some(slug) => match graph.get_project_by_slug(slug).await {
                Ok(Some(p)) => Some(p.id),
                Ok(None) => None,
                Err(e) => {
                    complete = false;
                    tracing::warn!(scope = %scope, error = %e, "pin scope unresolved");
                    None
                }
            },
        };
        let parsed: Vec<Uuid> = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                complete = false;
                tracing::warn!(scope = %scope, error = %e, "unreadable pin list, skipped");
                continue;
            }
        };
        for id in parsed {
            if seen.insert(id) {
                ids.push((id, project));
            }
        }
    }
    Ok((ids, read, complete))
}

async fn run_agent_knows_core(graph: &dyn GraphStore) -> anyhow::Result<RuleOutcome> {
    let self_id = graph.agent_identity_namespace().await?;
    // The rule names a subject, so it makes it. Without this, a graph where the root has no
    // `:Agent` node yet answers « 0 written » — a successful-looking empty run on a graph
    // that should have been written to. Measured: the primitive MATCHes both ends, so a
    // missing subject is a silent no-op.
    graph.ensure_agent(self_id, None).await?;
    let (ids, scopes, complete) = pinned_notes_everywhere(graph).await?;

    let measure = format!("{scopes} portées lues, {} note(s) épinglée(s)", ids.len());

    // Reconcile FIRST: an edge the rule drew to a note that is no longer pinned must go
    // before the new set is attested. The reverse order would leave, for one instant, a
    // self that knows both what it declares and what it dropped.
    let keep: Vec<Uuid> = ids.iter().map(|(id, _)| *id).collect();
    // Même règle que pour les projets : une lecture incomplète ne supprime rien.
    let released = if complete {
        graph
            .release_agent_knows(self_id, AGENT_KNOWS_CORE.name, &keep)
            .await?
    } else {
        0
    };

    let mut written = 0;
    let mut skipped = 0;
    for (note, project) in &ids {
        let scope = match project {
            Some(_) => KnowledgeScope::Project,
            None => KnowledgeScope::Global,
        };
        // Une épingle déclarée passe devant tout ce qui est MESURÉ : c'est une décision,
        // pas une observation. Le score le dit, et la lecture le respecte.
        let declared =
            AccountableProvenance::scored(AGENT_KNOWS_CORE.name, measure.clone(), f64::MAX)?;
        // A pin that names a note the graph no longer has draws no edge: the store says
        // whether the target exists, and a missing target is a skip, not an error.
        if graph
            .link_agent_knows(self_id, *note, scope, *project, &declared)
            .await?
        {
            written += 1;
        } else {
            skipped += 1;
        }
    }

    let mut measure = measure;
    if !complete {
        measure.push_str(" — lecture incomplète : aucune arête libérée");
    }
    if released > 0 {
        measure.push_str(&format!(", {released} arête(s) libérée(s)"));
    }

    Ok(RuleOutcome {
        name: AGENT_KNOWS_CORE.name.to_string(),
        considered: ids.len(),
        written,
        skipped,
        released,
        measure,
    })
}

async fn run_session_co_activation(
    graph: &dyn GraphStore,
    session: Uuid,
    project: Option<Uuid>,
    notes: &[Uuid],
) -> anyhow::Result<RuleOutcome> {
    // A pair needs two notes, and a project: the co-activation is scoped like every other
    // reinforcement, so a session never welds the notes of two projects together.
    let name = SESSION_CO_ACTIVATION.name;
    let Some(project) = project else {
        return Ok(RuleOutcome {
            name: name.to_string(),
            measure: format!("session {session} : aucun projet, rien à renforcer"),
            ..Default::default()
        });
    };
    if notes.len() < 2 {
        return Ok(RuleOutcome {
            name: name.to_string(),
            measure: format!("session {session} : {} note(s), aucune paire", notes.len()),
            ..Default::default()
        });
    }

    let measure = format!(
        "{} note(s) dans le contexte de la session {session}",
        notes.len()
    );
    let provenance = AccountableProvenance::new(name, measure.clone())?;
    let written = graph
        .coactivate_session(project, notes, session, &provenance)
        .await?;

    Ok(RuleOutcome {
        name: name.to_string(),
        considered: notes.len(),
        written,
        skipped: 0,
        released: 0,
        measure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::*;

    /// Le refus est porté par la SIGNATURE, pas par une convention d'appel.
    ///
    /// `AccountableProvenance` a un champ privé : hors de ce module, on ne peut pas en
    /// fabriquer une sans passer par `new`/`scored`/`accountable`, qui refusent le vide.
    /// Mais ce verrou ne vaut que si les écrivains d'arêtes dérivées prennent VRAIMENT ce
    /// type : une signature revenue à `&Provenance` rouvrirait la porte, et un test
    /// d'exécution ne le verrait pas — un `Provenance` vide y est déjà impossible à
    /// construire, donc « aucune arête écrite » y serait vrai par construction. Ce test-ci
    /// lit les déclarations du trait et rougit sur la dérive.
    #[test]
    fn the_writers_of_derived_edges_accept_only_an_accountable_provenance() {
        const TRAITS: &str = include_str!("../neo4j/traits.rs");
        for writer in ["link_agent_knows", "link_agent_uses", "coactivate_session"] {
            let start = TRAITS
                .find(&format!("fn {writer}("))
                .unwrap_or_else(|| panic!("{writer} a disparu du trait de stockage"));
            let end = start
                + TRAITS[start..]
                    .find(';')
                    .expect("une déclaration de méthode se termine par ;");
            let signature = &TRAITS[start..end];
            assert!(
                signature.contains("AccountableProvenance"),
                "{writer} doit prendre un AccountableProvenance, or sa signature est :\n\
                 {signature}"
            );
            assert!(
                !signature.contains("provenance: &Provenance"),
                "{writer} accepte de nouveau une provenance nue, que rien n'oblige à \
                 justifier :\n{signature}"
            );
        }
    }

    /// Chaque refus de provenance a SON message : un `EmptyAt` annoncé comme « aucune
    /// règle nommée » enverrait corriger le mauvais champ.
    #[test]
    fn every_provenance_refusal_names_its_own_field() {
        let cases = [
            (ProvenanceError::EmptyRule, "règle"),
            (ProvenanceError::EmptyMeasure, "mesure"),
            (ProvenanceError::EmptyAt, "date"),
        ];
        for (error, expected) in cases {
            let said = error.to_string();
            assert!(
                said.contains(expected),
                "{error:?} doit nommer « {expected} », or il dit : {said}"
            );
        }
        let messages: std::collections::HashSet<String> = [
            ProvenanceError::EmptyRule,
            ProvenanceError::EmptyMeasure,
            ProvenanceError::EmptyAt,
        ]
        .iter()
        .map(|e| e.to_string())
        .collect();
        assert_eq!(messages.len(), 3, "trois refus, trois messages distincts");
    }

    /// Une lecture ratée ne doit RIEN détruire : la réconciliation est sautée, pas
    /// appliquée sur un `keep` vide — sinon une erreur transitoire effacerait le socle.
    #[tokio::test]
    async fn a_failed_project_list_wipes_nothing() {
        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let note = a_note("a foundation");
        graph.create_note(&note).await.unwrap();
        let self_id = graph.agent_identity_namespace().await.unwrap();
        let prov = AccountableProvenance::scored(AGENT_KNOWS_PROJECT.name, "posée à la main", 42.0)
            .unwrap();
        graph
            .link_agent_knows(
                self_id,
                note.id,
                KnowledgeScope::Project,
                Some(project.id),
                &prov,
            )
            .await
            .unwrap();

        graph
            .fail_list_projects
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let outcome = run(&graph, &[AGENT_KNOWS_PROJECT.name]).await.unwrap();
        graph
            .fail_list_projects
            .store(false, std::sync::atomic::Ordering::Relaxed);

        let outcome = outcome.of(AGENT_KNOWS_PROJECT.name).unwrap();
        assert_eq!(
            outcome.released, 0,
            "aucune suppression sur une lecture ratée"
        );
        assert!(outcome.measure.contains("illisible"), "{}", outcome.measure);
        assert_eq!(
            graph
                .agent_knows(self_id, Some(project.id), 10)
                .await
                .unwrap()
                .notes,
            vec![note.id],
            "l'arête est toujours là"
        );
    }

    /// Le corps de la règle PROJET : des candidats classés deviennent des arêtes d'échelle
    /// projet, et la réconciliation garde exactement ceux-là.
    #[tokio::test]
    async fn the_project_scale_writes_what_it_ranked() {
        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let mut note = a_note("the foundation of alpha");
        note.project_id = Some(project.id);
        note.status = crate::notes::NoteStatus::Active;
        note.importance = crate::notes::NoteImportance::Critical;
        graph.create_note(&note).await.unwrap();

        let outcome = run(&graph, &[AGENT_KNOWS_PROJECT.name]).await.unwrap();
        let outcome = outcome.of(AGENT_KNOWS_PROJECT.name).unwrap();
        assert_eq!(outcome.considered, 1, "{outcome:?}");
        assert_eq!(outcome.written, 1, "{outcome:?}");
        assert_eq!(outcome.released, 0);

        let self_id = graph.agent_identity_namespace().await.unwrap();
        let known = graph
            .agent_knows(self_id, Some(project.id), 10)
            .await
            .unwrap();
        assert_eq!(known.notes, vec![note.id], "the project scale serves it");
        // …et l'échelle globale ne le sert pas : il appartient à un projet.
        let global = graph.agent_knows(self_id, None, 10).await.unwrap();
        assert!(global.notes.is_empty(), "{global:?}");
    }

    /// Le corps de la règle GLOBALE : une note née globale devient une connaissance qui
    /// vaut partout, et elle apparaît même sans projet demandé.
    #[tokio::test]
    async fn the_global_scale_writes_what_crosses_projects() {
        let graph = MockGraphStore::new();
        let mut note = a_note("a lesson that crosses projects");
        note.project_id = None;
        note.status = crate::notes::NoteStatus::Active;
        note.importance = crate::notes::NoteImportance::Critical;
        graph.create_note(&note).await.unwrap();

        let outcome = run(&graph, &[AGENT_KNOWS_GLOBAL.name]).await.unwrap();
        let outcome = outcome.of(AGENT_KNOWS_GLOBAL.name).unwrap();
        assert_eq!(outcome.written, 1, "{outcome:?}");

        let self_id = graph.agent_identity_namespace().await.unwrap();
        assert_eq!(
            graph.agent_knows(self_id, None, 10).await.unwrap().notes,
            vec![note.id]
        );
        // Une connaissance globale est servie quel que soit le projet demandé : c'est sa
        // définition même. Ce qui ne se mélange pas, c'est l'inverse — une note de projet
        // n'apparaît pas dans le socle d'un autre projet (vérifié ci-dessus).
        assert_eq!(
            graph
                .agent_knows(self_id, Some(Uuid::new_v4()), 10)
                .await
                .unwrap()
                .notes,
            vec![note.id]
        );
    }

    #[test]
    fn the_scope_names_are_bare_identifiers() {
        // Le nom part dans le Cypher comme valeur, jamais comme texte inséré — mais il doit
        // rester une chaîne nue, et les deux échelles doivent être distinctes.
        assert_eq!(KnowledgeScope::Project.as_str(), "project");
        assert_eq!(KnowledgeScope::Global.as_str(), "global");
    }

    /// Deux notes dans la même échelle : c'est ce qui fait vivre le comparateur de rang.
    /// Avec une seule, le tri n'appelle jamais son comparateur et six lignes restent
    /// non couvertes — un test qui n'exerce pas ce qu'il croit exercer.
    #[tokio::test]
    async fn two_candidates_are_ranked_against_each_other() {
        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        for text in ["first", "second"] {
            let mut note = a_note(text);
            note.project_id = Some(project.id);
            note.status = crate::notes::NoteStatus::Active;
            note.importance = crate::notes::NoteImportance::Critical;
            graph.create_note(&note).await.unwrap();
        }

        let ranked = graph.ranked_foundation(Some(project.id), 6).await.unwrap();
        assert_eq!(ranked.len(), 2, "{ranked:?}");
        // Ordre déterministe : le départage se fait sur l'identifiant.
        assert!(ranked[0].note < ranked[1].note, "{ranked:?}");
    }

    /// Une échelle qui ne classe plus une note la LIBÈRE : c'est la moitié qui manquait,
    /// et sa mesure doit le dire.
    #[tokio::test]
    async fn a_scale_releases_what_it_no_longer_ranks() {
        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let mut kept = a_note("kept");
        kept.project_id = Some(project.id);
        kept.status = crate::notes::NoteStatus::Active;
        kept.importance = crate::notes::NoteImportance::Critical;
        graph.create_note(&kept).await.unwrap();
        let mut dropped = a_note("dropped");
        dropped.project_id = Some(project.id);
        dropped.status = crate::notes::NoteStatus::Active;
        dropped.importance = crate::notes::NoteImportance::Critical;
        graph.create_note(&dropped).await.unwrap();

        let first = run(&graph, &[AGENT_KNOWS_PROJECT.name]).await.unwrap();
        assert_eq!(first.of(AGENT_KNOWS_PROJECT.name).unwrap().written, 2);

        // La note sort du filtre (archivée) : la règle doit la libérer au passage suivant.
        {
            let mut notes = graph.notes.write().await;
            if let Some(n) = notes.get_mut(&dropped.id) {
                n.status = crate::notes::NoteStatus::Archived;
            }
        }
        let second = run(&graph, &[AGENT_KNOWS_PROJECT.name]).await.unwrap();
        let outcome = second.of(AGENT_KNOWS_PROJECT.name).unwrap();
        assert_eq!(outcome.released, 1, "{outcome:?}");
        assert!(outcome.measure.contains("libérée"), "{}", outcome.measure);

        let self_id = graph.agent_identity_namespace().await.unwrap();
        assert_eq!(
            graph
                .agent_knows(self_id, Some(project.id), 10)
                .await
                .unwrap()
                .notes,
            vec![kept.id]
        );
    }

    /// Un second passage atteste au lieu de réécrire : c'est ce que compte `skipped`.
    #[tokio::test]
    async fn a_second_pass_attests_instead_of_rewriting() {
        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let mut note = a_note("kept");
        note.project_id = Some(project.id);
        note.status = crate::notes::NoteStatus::Active;
        note.importance = crate::notes::NoteImportance::Critical;
        graph.create_note(&note).await.unwrap();

        run(&graph, &[AGENT_KNOWS_PROJECT.name]).await.unwrap();
        let again = run(&graph, &[AGENT_KNOWS_PROJECT.name]).await.unwrap();
        let outcome = again.of(AGENT_KNOWS_PROJECT.name).unwrap();
        assert_eq!(outcome.written, 0);
        assert_eq!(outcome.skipped, 1, "{outcome:?}");
    }

    /// Une épingle de portée PROJET garde son projet : l'échelle et le propriétaire sont
    /// écrits sur l'arête, et l'identifiant est résolu par le slug de la portée.
    #[tokio::test]
    async fn a_project_pin_keeps_its_project() {
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let note = a_note("a project foundation");
        graph.create_note(&note).await.unwrap();

        store_pinned_notes(&graph, Some("alpha"), &[note.id])
            .await
            .unwrap();

        let self_id = graph.agent_identity_namespace().await.unwrap();
        assert_eq!(
            graph
                .agent_knows(self_id, Some(project.id), 10)
                .await
                .unwrap()
                .notes,
            vec![note.id],
            "the project scale serves it"
        );
        assert!(
            graph
                .agent_knows(self_id, None, 10)
                .await
                .unwrap()
                .notes
                .is_empty(),
            "and it is not global"
        );
    }

    /// Une liste d'épingles illisible rend la lecture INCOMPLÈTE : la réconciliation est
    /// sautée, pas appliquée sur un `keep` partiel.
    #[tokio::test]
    async fn an_unreadable_pin_list_blocks_the_release() {
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let note = a_note("pinned");
        graph.create_note(&note).await.unwrap();
        store_pinned_notes(&graph, None, &[note.id]).await.unwrap();
        let self_id = graph.agent_identity_namespace().await.unwrap();
        assert_eq!(
            graph
                .agent_knows(self_id, None, 10)
                .await
                .unwrap()
                .notes
                .len(),
            1
        );

        // On corrompt la valeur stockée : la lecture devient incomplète.
        {
            let mut settings = graph.llm_settings.write().await;
            for ((_, key), value) in settings.iter_mut() {
                if key.contains("pinned") {
                    *value = "{not json".into();
                }
            }
        }
        let outcome = run(&graph, &[AGENT_KNOWS_CORE.name]).await.unwrap();
        let outcome = outcome.of(AGENT_KNOWS_CORE.name).unwrap();
        assert_eq!(outcome.released, 0, "{outcome:?}");
        assert!(
            outcome.measure.contains("incomplète"),
            "{}",
            outcome.measure
        );
        assert_eq!(
            graph.agent_knows(self_id, None, 10).await.unwrap().notes,
            vec![note.id],
            "l'arête survit à une lecture incomplète"
        );
    }

    /// Le passage au démarrage rend un rapport pour CHAQUE règle — et ne bloque jamais :
    /// c'est la seule chose que le démarrage demande.
    #[tokio::test]
    async fn the_boot_pass_reports_every_rule() {
        let graph = MockGraphStore::new();
        let report = run_at_boot(&graph).await.expect("a report");
        for rule in RULES {
            assert!(
                report.of(rule.name).is_some(),
                "{} is missing from the boot report",
                rule.name
            );
        }
    }

    /// Le départage du rang : un note sans synapse ni ancre vaut ce que vaut son énergie.
    #[test]
    fn the_rank_score_weighs_synapses_double_and_caps_anchors() {
        assert_eq!(rank_score(0, 0, 0.0), 0.0);
        assert_eq!(rank_score(1, 0, 0.0), 2.0, "a synapse counts double");
        assert_eq!(rank_score(0, 10, 0.0), 10.0);
        assert_eq!(rank_score(0, 50, 0.0), 50.0);
        assert_eq!(
            rank_score(0, 5_000, 0.0),
            50.0,
            "anchors are capped: « anchored everywhere » is not « truer »"
        );
        assert_eq!(
            rank_score(0, 0, 1.0),
            10.0,
            "energy breaks ties, it does not rule"
        );
    }

    #[test]
    fn every_rule_is_accountable_before_it_runs() {
        // The declaration is the contract: no unnamed rule, no unjustified edge. This
        // runs without a graph, because a rule that cannot state its own purpose would
        // not be fixed by one.
        for r in RULES {
            assert!(!r.name.trim().is_empty());
            assert!(
                r.justification.split_whitespace().count() >= 5,
                "{}: a justification, not a label",
                r.name
            );
            assert!(r.edge.chars().all(|c| c == '_' || c.is_ascii_uppercase()));
            assert!(!r.target_label.trim().is_empty());
        }
    }

    #[test]
    fn rule_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for r in RULES {
            assert!(seen.insert(r.name), "duplicate rule {}", r.name);
        }
        assert!(rule("agent-knows-core").is_some());
        assert!(rule("agent-knows-everything").is_none());
    }

    #[test]
    fn a_provenance_must_name_a_rule_and_a_measure() {
        // The predicate the tests below oppose to every writer.
        let good = Provenance::new("agent-knows-core", "2 portées lues, 1 note épinglée");
        assert!(good.is_accountable());
        assert!(!Provenance::new("", "measured").is_accountable());
        assert!(!Provenance::new("agent-knows-core", "  ").is_accountable());
    }

    /// C7 : « une arête sans justification est un bug ». Une provenance vide est REFUSÉE
    /// à l'écriture — par une erreur typée, sans panique, et AUCUNE arête n'existe après.
    /// La preuve se fait avant l'écrivain : celui-ci n'accepte que le type prouvé, donc
    /// le seul chemin pour écrire passe par ce refus.
    #[tokio::test]
    async fn an_empty_provenance_is_refused_and_draws_nothing() {
        let graph = MockGraphStore::new();
        let note = a_note("a note nobody can justify");
        graph.create_note(&note).await.unwrap();
        let self_id = graph.agent_identity_namespace().await.unwrap();
        graph.ensure_agent(self_id, None).await.unwrap();

        // Chaque champ vide a son erreur, nommée.
        assert_eq!(
            Provenance::new("", "measured").accountable().unwrap_err(),
            ProvenanceError::EmptyRule
        );
        assert_eq!(
            Provenance::new("   ", "measured")
                .accountable()
                .unwrap_err(),
            ProvenanceError::EmptyRule,
            "des espaces ne sont pas un nom"
        );
        assert_eq!(
            AccountableProvenance::new("agent-knows-core", "  ").unwrap_err(),
            ProvenanceError::EmptyMeasure
        );
        assert_eq!(
            AccountableProvenance::scored("agent-knows-core", "", 1.0).unwrap_err(),
            ProvenanceError::EmptyMeasure
        );
        let undated = Provenance {
            at: String::new(),
            ..Provenance::new("agent-knows-core", "measured")
        };
        assert_eq!(undated.accountable().unwrap_err(), ProvenanceError::EmptyAt);
        // Lisible comme une erreur ordinaire : elle remonte par `?` dans un `anyhow::Result`.
        let as_any: anyhow::Error = ProvenanceError::EmptyRule.into();
        assert!(as_any.to_string().contains("aucune règle"), "{as_any}");

        // Aucune arête dérivée n'a été tracée : ni KNOWS, ni USES, ni SYNAPSE.
        assert!(graph.agent_knows.read().await.is_empty());
        assert!(graph.agent_uses.read().await.is_empty());
        assert!(graph.synapses.read().await.is_empty());
        assert_eq!(
            graph.agent_knows(self_id, None, 10).await.unwrap().notes,
            Vec::<Uuid>::new()
        );

        // Et la même mesure, une fois nommée, s'écrit : le refus porte sur le vide, pas
        // sur l'écrivain.
        let proven = AccountableProvenance::new("agent-knows-core", "measured").unwrap();
        assert_eq!(proven.rule, "agent-knows-core");
        assert_eq!(proven.as_provenance().measure, "measured");
        assert!(proven.clone().into_inner().is_accountable());
        assert!(graph
            .link_agent_knows(self_id, note.id, KnowledgeScope::Global, None, &proven)
            .await
            .unwrap());
        assert_eq!(graph.agent_knows.read().await.len(), 1);
    }

    /// Une arête dérivée telle que le mock la retient : règle, mesure, source, cible.
    /// `at` est exclu à dessein — c'est la seule chose qui DOIT changer d'un passage à
    /// l'autre.
    async fn derived_edges(graph: &MockGraphStore) -> std::collections::BTreeSet<[String; 5]> {
        let mut out = std::collections::BTreeSet::new();
        for ((agent, note), (_, _, p)) in graph.agent_knows.read().await.iter() {
            out.insert([
                "KNOWS".into(),
                p.rule.clone(),
                p.measure.clone(),
                agent.to_string(),
                note.to_string(),
            ]);
        }
        for ((agent, label, target), p) in graph.agent_uses.read().await.iter() {
            out.insert([
                "USES".into(),
                p.rule.clone(),
                p.measure.clone(),
                agent.to_string(),
                format!("{label}:{target}"),
            ]);
        }
        for ((a, b), (_, _, p)) in graph.synapses.read().await.iter() {
            out.insert([
                "SYNAPSE".into(),
                p.rule.clone(),
                p.measure.clone(),
                a.to_string(),
                b.to_string(),
            ]);
        }
        out
    }

    /// Reproductibilité : deux passages du moteur sur les MÊMES données donnent le même
    /// ensemble d'arêtes (règle, mesure, source, cible). Pas de modèle de langage, pas de
    /// hasard : seule la date d'observation diffère, et elle est exclue de la comparaison.
    #[tokio::test]
    async fn two_passes_on_the_same_data_draw_the_same_edges() {
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let mut in_project = Vec::new();
        for text in ["first of alpha", "second of alpha", "third of alpha"] {
            let mut note = a_note(text);
            note.project_id = Some(project.id);
            note.status = crate::notes::NoteStatus::Active;
            note.importance = crate::notes::NoteImportance::Critical;
            graph.create_note(&note).await.unwrap();
            in_project.push(note.id);
        }
        let mut global = a_note("born global");
        global.project_id = None;
        global.status = crate::notes::NoteStatus::Active;
        global.importance = crate::notes::NoteImportance::Critical;
        graph.create_note(&global).await.unwrap();
        let pinned = a_note("pinned by hand");
        graph.create_note(&pinned).await.unwrap();
        store_pinned_notes(&graph, Some("alpha"), &[pinned.id])
            .await
            .unwrap();

        // Un passage complet : les règles qui lisent le graphe, puis les deux dont la
        // mesure vient de l'appelant — avec la MÊME mesure à chaque passage.
        let session = Uuid::new_v4();
        let agent = Uuid::new_v4();
        let (persona, skill) = (Uuid::new_v4(), Uuid::new_v4());
        let activation = Measurement::Activation {
            session,
            agent,
            personas: vec![persona],
            skills: vec![skill],
        };
        let session_notes = Measurement::SessionNotes {
            session,
            project: Some(project.id),
            notes: in_project.clone(),
        };
        async fn pass(
            graph: &MockGraphStore,
            activation: &Measurement,
            session_notes: &Measurement,
        ) -> RulesReport {
            let report = run(graph, &[]).await.unwrap();
            run_with(graph, AGENT_USES_ACTIVATED.name, activation.clone())
                .await
                .unwrap();
            run_with(graph, SESSION_CO_ACTIVATION.name, session_notes.clone())
                .await
                .unwrap();
            report
        }

        let first = pass(&graph, &activation, &session_notes).await;
        let after_first = derived_edges(&graph).await;
        let second = pass(&graph, &activation, &session_notes).await;
        let after_second = derived_edges(&graph).await;

        assert!(
            after_first.len() >= 3 + 1 + 1 + 2 + 3,
            "le jeu de données exerce chaque règle : {after_first:?}"
        );
        assert_eq!(
            after_first, after_second,
            "le même graphe, les mêmes arêtes"
        );
        // Chaque arête porte une provenance : aucune n'est vide, et chaque règle nommée est
        // une règle déclarée (ou l'épingle posée par la règle du socle).
        for edge in &after_first {
            assert!(
                rule(&edge[1]).is_some(),
                "règle inconnue sur une arête : {edge:?}"
            );
            assert!(!edge[2].trim().is_empty(), "mesure vide : {edge:?}");
        }
        // Le second passage n'écrit rien de neuf : il atteste.
        assert_eq!(
            second.edges_written, 0,
            "second passage : {second:?} (premier : {first:?})"
        );
        assert!(first.edges_written > 0, "{first:?}");
    }

    fn a_note(content: &str) -> crate::notes::models::Note {
        crate::notes::models::Note::new(
            None,
            crate::notes::models::NoteType::Guideline,
            content.into(),
            "api".into(),
        )
    }

    #[tokio::test]
    async fn the_pins_become_edges_of_the_self() {
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let note = a_note("a foundation");
        graph.create_note(&note).await.unwrap();
        store_pinned_notes(&graph, None, &[note.id]).await.unwrap();

        // The writer itself reconciles: an edge exists the moment the pin does, without
        // waiting for a restart or a cron.
        let self_id = graph.agent_identity_namespace().await.unwrap();
        assert_eq!(
            graph.agent_knows(self_id, None, 10).await.unwrap().notes,
            vec![note.id],
            "storing a pin draws its edge"
        );

        // And an explicit run attests what is already there instead of duplicating it.
        let report = run(&graph, &["agent-knows-core"]).await.unwrap();
        let outcome = report.of("agent-knows-core").unwrap();
        assert_eq!(outcome.considered, 1);
        assert_eq!(outcome.written, 0, "already attested: {report:?}");
        assert_eq!(outcome.skipped, 1);
        assert_eq!(outcome.released, 0);
    }

    #[tokio::test]
    async fn a_second_run_attests_instead_of_duplicating() {
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let note = a_note("a foundation");
        graph.create_note(&note).await.unwrap();
        store_pinned_notes(&graph, None, &[note.id]).await.unwrap();

        let first = run(&graph, &[]).await.unwrap();
        let second = run(&graph, &[]).await.unwrap();
        assert_eq!(first.edges_written, 0, "the pin already drew it");
        assert_eq!(second.edges_written, 0, "idempotent: {second:?}");
        assert_eq!(second.of("agent-knows-core").unwrap().skipped, 1);
        assert_eq!(first.of("agent-knows-core").unwrap().released, 0);
    }

    #[tokio::test]
    async fn a_pin_removed_is_released() {
        // The half that makes the rule a reconciliation: what is no longer declared stops
        // being known. Without it the self accumulates foundations it has dropped.
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let kept = a_note("kept");
        let dropped = a_note("dropped");
        graph.create_note(&kept).await.unwrap();
        graph.create_note(&dropped).await.unwrap();
        store_pinned_notes(&graph, None, &[kept.id, dropped.id])
            .await
            .unwrap();

        let self_id = graph.agent_identity_namespace().await.unwrap();
        assert_eq!(
            graph
                .agent_knows(self_id, None, 10)
                .await
                .unwrap()
                .notes
                .len(),
            2
        );

        store_pinned_notes(&graph, None, &[kept.id]).await.unwrap();

        let known = graph.agent_knows(self_id, None, 10).await.unwrap();
        assert_eq!(known.notes, vec![kept.id], "the dropped pin is gone");

        // And the measure says it: a release is published, not silent.
        let outcome = run(&graph, &["agent-knows-core"]).await.unwrap();
        assert_eq!(outcome.of("agent-knows-core").unwrap().released, 0);
    }

    #[tokio::test]
    async fn an_edge_a_person_placed_survives_the_reconciliation() {
        // Only the edges the RULE drew belong to it. A KNOWS an operator placed by hand has
        // no such rule and must survive an unpin it never asked about.
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        let pinned = a_note("pinned");
        let by_hand = a_note("a person decided this one");
        graph.create_note(&pinned).await.unwrap();
        graph.create_note(&by_hand).await.unwrap();

        let self_id = graph.agent_identity_namespace().await.unwrap();
        graph
            .link_agent_knows(
                self_id,
                by_hand.id,
                KnowledgeScope::Global,
                None,
                &AccountableProvenance::new("a person", "decided by hand").unwrap(),
            )
            .await
            .unwrap();

        store_pinned_notes(&graph, None, &[pinned.id])
            .await
            .unwrap();
        // Unpin everything: the rule has nothing left to declare.
        store_pinned_notes(&graph, None, &[]).await.unwrap();

        let known = graph.agent_knows(self_id, None, 10).await.unwrap();
        assert_eq!(
            known.notes,
            vec![by_hand.id],
            "the rule removed its own edge, not someone else's"
        );
    }

    #[tokio::test]
    async fn a_pin_that_names_nothing_draws_nothing() {
        use crate::chat::stages::knowledge_injection::store_pinned_notes;

        let graph = MockGraphStore::new();
        store_pinned_notes(&graph, None, &[Uuid::new_v4()])
            .await
            .unwrap();

        let report = run(&graph, &[]).await.unwrap();
        let outcome = report.of("agent-knows-core").unwrap();
        assert_eq!(outcome.considered, 1, "it was considered");
        assert_eq!(outcome.written, 0, "and the target did not exist");
        assert_eq!(outcome.skipped, 1);
    }

    #[tokio::test]
    async fn run_with_refuses_a_rule_it_does_not_know() {
        let graph = MockGraphStore::new();
        let err = run_with(
            &graph,
            "agent-knows-everything",
            Measurement::SessionNotes {
                session: Uuid::new_v4(),
                project: None,
                notes: vec![],
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("unknown rule"), "{err}");
    }

    #[tokio::test]
    async fn run_with_refuses_a_measurement_the_rule_does_not_take() {
        // A rule that reads the graph refuses a caller-held measurement, and says so: the
        // alternative — accepting it and ignoring it — would report a run that never happened.
        let graph = MockGraphStore::new();
        let err = run_with(
            &graph,
            AGENT_KNOWS_CORE.name,
            Measurement::SessionNotes {
                session: Uuid::new_v4(),
                project: None,
                notes: vec![],
            },
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("does not take this measurement"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_rule_refuses_a_measurement_it_does_not_take() {
        // Delivering the wrong measurement must fail loudly: a rule that silently ignored it
        // would report « 0 written » on a run that never measured anything.
        let graph = MockGraphStore::new();
        let err = run_with(
            &graph,
            AGENT_KNOWS_CORE.name,
            Measurement::SessionNotes {
                session: Uuid::new_v4(),
                project: None,
                notes: vec![Uuid::new_v4(), Uuid::new_v4()],
            },
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("does not take"), "{err}");
    }

    #[tokio::test]
    async fn an_unknown_rule_is_refused_not_ignored() {
        let graph = MockGraphStore::new();
        let err = run(&graph, &["agent-knows-everything"]).await.unwrap_err();
        assert!(err.to_string().contains("unknown rule"), "{err}");
    }
}
