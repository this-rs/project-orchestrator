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
//! épinglées »). Une arête sans provenance est un bug, et
//! [`Provenance::is_accountable`] est le prédicat que les tests opposent à tout écrivain.
//!
//! C'est ce qui rend une croyance **contestable** : on peut relire la mesure, la
//! refaire, et constater qu'elle a changé. Une note qui affirme sans mesure est une
//! opinion ; une arête qui porte sa mesure est une observation datée.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;

/// Ce qu'une arête dérivée porte, pour être contestable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The rule that drew the edge.
    pub rule: String,
    /// The measurement that justified it, in words and numbers.
    pub measure: String,
    /// When it was measured (RFC 3339).
    pub at: String,
}

impl Provenance {
    /// A provenance stamped now.
    pub fn new(rule: &str, measure: impl Into<String>) -> Self {
        Self {
            rule: rule.to_string(),
            measure: measure.into(),
            at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Whether this provenance can be contested: it must name a rule and a measure. An
    /// edge whose origin cannot be named is indistinguishable from an accident.
    pub fn is_accountable(&self) -> bool {
        !self.rule.trim().is_empty()
            && !self.measure.trim().is_empty()
            && !self.at.trim().is_empty()
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

/// The rules this engine knows, in a stable order.
pub const RULES: &[Rule] = &[
    AGENT_KNOWS_CORE,
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
    let provenance = Provenance::new(AGENT_USES_ACTIVATED.name, measure.clone());

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
        measure,
    })
}

/// The pins of every scope: the instance's own, then each project's. A project scope is
/// read through its slug, which is what [`crate::chat::stages::knowledge_injection`]
/// keys the setting by.
async fn pinned_notes_everywhere(graph: &dyn GraphStore) -> anyhow::Result<(Vec<Uuid>, usize)> {
    use crate::chat::stages::knowledge_injection::load_pinned_notes;

    let mut scopes: Vec<Option<String>> = vec![None];
    // The global pins are the socle; the project-scoped ones are a convenience. A graph that
    // holds a project the mapper cannot read (a foreign suite left a bare `(:Project)` without
    // `name`/`slug`) must not kill the rule over it — « never fail without a socle »: the
    // global pins are still applied, the unreadable scopes are said, not silently skipped.
    match graph.list_projects().await {
        Ok(projects) => {
            for project in projects {
                scopes.push(Some(project.slug));
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "project pin scopes are unreadable: only the global pins are read");
        }
    }
    let read = scopes.len();

    let mut seen = std::collections::HashSet::new();
    let mut ids = Vec::new();
    for scope in scopes {
        let pinned = load_pinned_notes(graph, scope.as_deref()).await?;
        for id in pinned {
            if seen.insert(id) {
                ids.push(id);
            }
        }
    }
    Ok((ids, read))
}

async fn run_agent_knows_core(graph: &dyn GraphStore) -> anyhow::Result<RuleOutcome> {
    let self_id = graph.agent_identity_namespace().await?;
    // The rule names a subject, so it makes it. Without this, a graph where the root has no
    // `:Agent` node yet answers « 0 written » — a successful-looking empty run on a graph
    // that should have been written to. Measured: the primitive MATCHes both ends, so a
    // missing subject is a silent no-op.
    graph.ensure_agent(self_id, None).await?;
    let (ids, scopes) = pinned_notes_everywhere(graph).await?;

    let measure = format!("{scopes} portées lues, {} note(s) épinglée(s)", ids.len());
    let provenance = Provenance::new(AGENT_KNOWS_CORE.name, measure.clone());

    let mut written = 0;
    let mut skipped = 0;
    for note in &ids {
        // A pin that names a note the graph no longer has draws no edge: the store says
        // whether the target exists, and a missing target is a skip, not an error.
        if graph.link_agent_knows(self_id, *note, &provenance).await? {
            written += 1;
        } else {
            skipped += 1;
        }
    }

    Ok(RuleOutcome {
        name: AGENT_KNOWS_CORE.name.to_string(),
        considered: ids.len(),
        written,
        skipped,
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
    let provenance = Provenance::new(name, measure.clone());
    let written = graph
        .coactivate_session(project, notes, session, &provenance)
        .await?;

    Ok(RuleOutcome {
        name: name.to_string(),
        considered: notes.len(),
        written,
        skipped: 0,
        measure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::*;

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

        let report = run(&graph, &["agent-knows-core"]).await.unwrap();
        let outcome = report.of("agent-knows-core").unwrap();
        assert_eq!(outcome.considered, 1);
        assert_eq!(outcome.written, 1, "{report:?}");
        assert_eq!(report.edges_written, 1);

        let self_id = graph.agent_identity_namespace().await.unwrap();
        let known = graph.agent_knows(self_id, 10).await.unwrap();
        assert_eq!(known.notes, vec![note.id]);
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
        assert_eq!(first.edges_written, 1);
        assert_eq!(second.edges_written, 0, "idempotent: {second:?}");
        assert_eq!(second.of("agent-knows-core").unwrap().skipped, 1);
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
    async fn an_unknown_rule_is_refused_not_ignored() {
        let graph = MockGraphStore::new();
        let err = run(&graph, &["agent-knows-everything"]).await.unwrap_err();
        assert!(err.to_string().contains("unknown rule"), "{err}");
    }
}
