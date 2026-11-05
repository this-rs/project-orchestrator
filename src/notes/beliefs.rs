//! T7 — la révision d'une croyance.
//!
//! # Le défaut que ce module existe pour tuer
//!
//! `NoteStore::verify_assertion` (`notes/lifecycle.rs`) traite quatre des sept contrôles
//! (`DependsOn`, `Calls`, `NoCalls`, `Implements`) en rendant `passed: true` avec le message
//! « requires graph access ». Une assertion de relation **ne pouvait donc jamais échouer** :
//! elle était vérifiée au sens du code, et infalsifiable au sens du monde. C'est exactement
//! la croyance ancienne et fausse que rien ne vient corriger.
//!
//! Le graphe porte pourtant tout ce qu'il faut : `CALLS` (1 000 627 arêtes mesurées),
//! `IMPLEMENTS_TRAIT`, `IMPLEMENTS_FOR`, `DEPENDS_ON`, `IMPORTS`. Ce module les interroge.
//!
//! # Trois verdicts, pas deux
//!
//! Un booléen mentirait : « je ne sais pas vérifier » n'est pas « c'est vrai ». Un contrôle
//! qui demande le texte source (une signature) n'est pas vérifiable depuis le graphe, et le
//! dire est la seule réponse honnête — `Unverifiable` n'autorise aucune sanction.
//!
//! # Le contrat de `target` et `parameters`
//!
//! `assertion_rule.target` garde le format existant (`function:<nom>`, `file:<chemin>`,
//! `struct:<nom>`). Les relations nomment leur second terme dans `parameters` :
//!
//! | contrôle | `target` | `parameters` |
//! |---|---|---|
//! | `Exists` / `NotExists` | l'entité visée | — |
//! | `Calls` / `NoCalls` | `function:<appelant>` | `{"to": "function:<appelé>"}` |
//! | `Implements` | `struct:<type>` | `{"trait": "<nom>"}` |
//! | `DependsOn` | `file:<chemin>` | `{"on": "file:<chemin>"}` |
//! | `SignatureContains` | — | **non vérifiable** depuis le graphe |
//!
//! Mesuré avant d'écrire : le graphe porte **zéro** assertions (`assertion_rule` absent sur
//! ~30 000 notes). Ce contrat n'a donc aucune donnée existante à migrer — il est neuf, et
//! documenté plutôt que deviné.

use serde::{Deserialize, Serialize};

use crate::neo4j::traits::GraphStore;
use crate::notes::models::{
    AssertionCheckType, AssertionResult, AssertionRule, Note, NoteStatus, ViolationAction,
};

/// Ce que le monde répond à une croyance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Le graphe confirme l'assertion.
    Holds(String),
    /// Le graphe la contredit.
    Fails(String),
    /// On ne peut pas la trancher ici — et ce n'est pas une confirmation.
    Unverifiable(String),
}

impl Verdict {
    /// L'assertion est-elle réfutée ? Seul `Fails` autorise une sanction.
    pub fn is_refuted(&self) -> bool {
        matches!(self, Verdict::Fails(_))
    }

    pub fn message(&self) -> &str {
        match self {
            Verdict::Holds(m) | Verdict::Fails(m) | Verdict::Unverifiable(m) => m,
        }
    }

    /// La forme persistée. `passed` reste faux pour un verdict invérifiable : on ne range
    /// pas « je ne sais pas » du côté du vrai. `details.unverifiable` le distingue.
    pub fn to_result(&self) -> AssertionResult {
        match self {
            Verdict::Holds(m) => AssertionResult::passed(m.clone()),
            Verdict::Fails(m) => AssertionResult::failed(m.clone()),
            Verdict::Unverifiable(m) => {
                // « Je ne sais pas » ne se range pas du côté du vrai : le verdict est faux
                // au sens de la vérification, et `details.unverifiable` dit pourquoi — pour
                // qu'aucune sanction ne s'applique à ce qu'on n'a pas su contrôler.
                let mut result = AssertionResult::failed(m.clone());
                result.details = Some(serde_json::json!({"unverifiable": true}));
                result
            }
        }
    }
}

/// Une croyance a-t-elle DÉRIVÉ ? La dérive est un passage de « tenue » à « réfutée » :
/// une assertion qui n'a jamais tenu n'est pas une dérive, c'est une erreur d'écriture.
pub fn has_drifted(previous: Option<&AssertionResult>, now: &Verdict) -> bool {
    match previous {
        Some(p) => p.passed && now.is_refuted(),
        None => false,
    }
}

/// Découpe `"function:foo"` en `("function", "foo")`.
fn split_target(target: &str) -> Option<(&str, &str)> {
    let (kind, name) = target.split_once(':')?;
    if name.is_empty() {
        return None;
    }
    Some((kind, name))
}

/// Vérifie une assertion CONTRE LE GRAPHE.
pub async fn verify(graph: &dyn GraphStore, note: &Note) -> anyhow::Result<Verdict> {
    let Some(rule) = note.assertion_rule.as_ref() else {
        return Ok(Verdict::Unverifiable(
            "aucune procédure déclarée : rien à vérifier".into(),
        ));
    };
    judge(graph, rule).await
}

async fn judge(graph: &dyn GraphStore, rule: &AssertionRule) -> anyhow::Result<Verdict> {
    let params = rule.parameters.clone().unwrap_or(serde_json::Value::Null);
    let Some((kind, name)) = split_target(&rule.target) else {
        return Ok(Verdict::Unverifiable(format!(
            "cible illisible « {} » : attendu « <type>:<nom> »",
            rule.target
        )));
    };

    match rule.check_type {
        AssertionCheckType::SignatureContains => Ok(Verdict::Unverifiable(
            "une signature demande le texte source, pas le graphe".into(),
        )),

        AssertionCheckType::Exists => match graph.entity_exists(kind, name).await? {
            Some(true) => Ok(Verdict::Holds(format!("{kind} « {name} » existe"))),
            Some(false) => Ok(Verdict::Fails(format!(
                "{kind} « {name} » est absent du graphe"
            ))),
            None => Ok(Verdict::Unverifiable(format!(
                "type de cible « {kind} » inconnu du graphe"
            ))),
        },

        AssertionCheckType::NotExists => match graph.entity_exists(kind, name).await? {
            Some(true) => Ok(Verdict::Fails(format!(
                "{kind} « {name} » existe alors qu'il devrait être absent"
            ))),
            Some(false) => Ok(Verdict::Holds(format!("{kind} « {name} » est bien absent"))),
            None => Ok(Verdict::Unverifiable(format!(
                "type de cible « {kind} » inconnu du graphe"
            ))),
        },

        AssertionCheckType::Calls | AssertionCheckType::NoCalls => {
            let Some(to) = params.get("to").and_then(|v| v.as_str()) else {
                return Ok(Verdict::Unverifiable(
                    "Calls sans parameters.to : la cible de l'appel n'est pas nommée".into(),
                ));
            };
            let Some((to_kind, to_name)) = split_target(to) else {
                return Ok(Verdict::Unverifiable(format!(
                    "cible d'appel illisible « {to} »"
                )));
            };
            let Some(linked) = graph
                .entities_linked(kind, name, "CALLS", to_kind, to_name)
                .await?
            else {
                return Ok(Verdict::Unverifiable(format!(
                    "type de cible inconnu (« {kind} » ou « {to_kind} »)"
                )));
            };
            let wants = rule.check_type == AssertionCheckType::Calls;
            Ok(match (linked, wants) {
                (true, true) => Verdict::Holds(format!("{name} appelle bien {to_name}")),
                (false, true) => Verdict::Fails(format!("{name} n'appelle pas {to_name}")),
                (true, false) => Verdict::Fails(format!(
                    "{name} appelle {to_name} alors qu'il ne devrait pas"
                )),
                (false, false) => Verdict::Holds(format!("{name} n'appelle pas {to_name}")),
            })
        }

        AssertionCheckType::Implements => {
            let Some(trait_name) = params.get("trait").and_then(|v| v.as_str()) else {
                return Ok(Verdict::Unverifiable(
                    "Implements sans parameters.trait".into(),
                ));
            };
            let Some(linked) = graph
                .entities_linked(kind, name, "IMPLEMENTS_TRAIT", "trait", trait_name)
                .await?
            else {
                return Ok(Verdict::Unverifiable("type de cible inconnu".into()));
            };
            Ok(if linked {
                Verdict::Holds(format!("{name} implémente {trait_name}"))
            } else {
                Verdict::Fails(format!("{name} n'implémente pas {trait_name}"))
            })
        }

        AssertionCheckType::DependsOn => {
            let Some(on) = params.get("on").and_then(|v| v.as_str()) else {
                return Ok(Verdict::Unverifiable("DependsOn sans parameters.on".into()));
            };
            let Some((on_kind, on_name)) = split_target(on) else {
                return Ok(Verdict::Unverifiable(format!("cible illisible « {on} »")));
            };
            let Some(linked) = graph
                .entities_linked(kind, name, "DEPENDS_ON", on_kind, on_name)
                .await?
            else {
                return Ok(Verdict::Unverifiable(format!(
                    "type de cible inconnu (« {kind} » ou « {on_kind} »)"
                )));
            };
            Ok(if linked {
                Verdict::Holds(format!("{name} dépend de {on_name}"))
            } else {
                Verdict::Fails(format!("{name} ne dépend pas de {on_name}"))
            })
        }
    }
}

/// Ce qu'une passe a fait. Publié, comme toute mesure de ce dépôt.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReverificationReport {
    pub checked: usize,
    pub held: usize,
    pub refuted: usize,
    pub unverifiable: usize,
    /// Passages de « tenue » à « réfutée » — la seule chose qui mérite le mot dérive.
    pub drifted: usize,
    pub flagged: usize,
    pub unsanctioned: usize,
}

/// Re-vérifie les croyances, applique la sanction d'une réfutation, et persiste le verdict.
///
/// `Unverifiable` n'est jamais sanctionné : on ne casse pas une croyance parce qu'on ne sait
/// pas la contrôler. Une réfutation applique `on_violation` — `Warn` se contente de tracer,
/// `FlagNote` et `Block` passent la note en `needs_review`.
pub async fn reverify(
    graph: &dyn GraphStore,
    limit: usize,
) -> anyhow::Result<ReverificationReport> {
    let notes = graph.list_assertion_notes(limit).await?;
    let mut report = ReverificationReport::default();

    for note in notes {
        report.checked += 1;
        let verdict = verify(graph, &note).await?;
        let drifted = has_drifted(note.last_assertion_result.as_ref(), &verdict);

        let mut status: Option<NoteStatus> = None;
        match &verdict {
            Verdict::Holds(_) => report.held += 1,
            Verdict::Unverifiable(_) => report.unverifiable += 1,
            Verdict::Fails(_) => {
                report.refuted += 1;
                if drifted {
                    report.drifted += 1;
                }
                match note.assertion_rule.as_ref().map(|r| r.on_violation) {
                    Some(ViolationAction::Warn) | None => report.unsanctioned += 1,
                    Some(ViolationAction::FlagNote) | Some(ViolationAction::Block) => {
                        status = Some(NoteStatus::NeedsReview);
                        report.flagged += 1;
                    }
                }
            }
        }

        graph
            .record_assertion_check(note.id, &verdict.to_result(), status)
            .await?;
    }

    Ok(report)
}

/// Le budget du passage au démarrage : une re-vérification lente coûte la vérification,
/// jamais le démarrage.
pub const REVERIFY_BOOT_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// Combien de croyances on re-vérifie par passage. Au-delà, la suite attend le suivant.
pub const REVERIFY_LIMIT: usize = 200;

/// Re-vérifie les croyances une fois au démarrage, sous budget. `None` = pas de rapport
/// (budget épuisé ou erreur) — jamais une panique, jamais un démarrage refusé.
pub async fn reverify_at_boot(graph: &dyn GraphStore) -> Option<ReverificationReport> {
    match tokio::time::timeout(REVERIFY_BOOT_BUDGET, reverify(graph, REVERIFY_LIMIT)).await {
        Ok(Ok(report)) => {
            tracing::info!(
                checked = report.checked,
                held = report.held,
                refuted = report.refuted,
                unverifiable = report.unverifiable,
                drifted = report.drifted,
                flagged = report.flagged,
                "beliefs re-verified"
            );
            Some(report)
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "the beliefs could not be re-verified");
            None
        }
        Err(_) => {
            tracing::warn!("the belief re-verification timed out");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::*;
    use crate::notes::models::{AssertionRule, NoteImportance, NoteType};

    fn a_belief(check: AssertionCheckType, target: &str, params: serde_json::Value) -> Note {
        let mut note = Note::new(
            None,
            NoteType::Assertion,
            "une croyance vérifiable".into(),
            "api".into(),
        );
        note.status = NoteStatus::Active;
        note.importance = NoteImportance::High;
        note.assertion_rule = Some(AssertionRule {
            check_type: check,
            target: target.into(),
            file_pattern: None,
            parameters: Some(params),
            on_violation: ViolationAction::FlagNote,
        });
        note
    }

    async fn seeded(note: &Note) -> MockGraphStore {
        let graph = MockGraphStore::new();
        graph.create_note(note).await.unwrap();
        graph
    }

    /// LE test de T7 : une assertion de relation peut enfin ÊTRE FAUSSE. Avant ce module,
    /// `verify_assertion` rendait `passed: true` pour `Calls` avec le message « requires
    /// graph access » — une croyance infalsifiable.
    #[tokio::test]
    async fn a_missing_call_refutes_the_assertion() {
        let note = a_belief(
            AssertionCheckType::Calls,
            "function:main",
            serde_json::json!({"to": "function:gone"}),
        );
        let graph = seeded(&note).await;
        graph
            .belief_entities
            .write()
            .await
            .insert(("function".into(), "main".into()));

        let verdict = verify(&graph, &note).await.unwrap();
        assert!(verdict.is_refuted(), "{verdict:?}");
        assert!(verdict.message().contains("gone"), "{}", verdict.message());
    }

    #[tokio::test]
    async fn a_present_call_holds() {
        let note = a_belief(
            AssertionCheckType::Calls,
            "function:main",
            serde_json::json!({"to": "function:helper"}),
        );
        let graph = seeded(&note).await;
        graph.belief_links.write().await.insert((
            "function".into(),
            "main".into(),
            "CALLS".into(),
            "function".into(),
            "helper".into(),
        ));

        assert_eq!(
            verify(&graph, &note).await.unwrap(),
            Verdict::Holds("main appelle bien helper".into())
        );
    }

    #[tokio::test]
    async fn no_calls_is_refuted_by_a_call_that_exists() {
        let note = a_belief(
            AssertionCheckType::NoCalls,
            "function:main",
            serde_json::json!({"to": "function:forbidden"}),
        );
        let graph = seeded(&note).await;
        graph.belief_links.write().await.insert((
            "function".into(),
            "main".into(),
            "CALLS".into(),
            "function".into(),
            "forbidden".into(),
        ));

        assert!(verify(&graph, &note).await.unwrap().is_refuted());
    }

    #[tokio::test]
    async fn implements_is_checked_against_the_real_edge() {
        let note = a_belief(
            AssertionCheckType::Implements,
            "struct:Store",
            serde_json::json!({"trait": "Clone"}),
        );
        let graph = seeded(&note).await;
        let verdict = verify(&graph, &note).await.unwrap();
        assert!(verdict.is_refuted(), "{verdict:?}");
    }

    /// Un type de cible inconnu n'est PAS « absent » : c'est « je ne sais pas », et rien
    /// ne doit être sanctionné là-dessus.
    #[tokio::test]
    async fn an_unknown_target_kind_is_unverifiable_not_refuted() {
        let note = a_belief(
            AssertionCheckType::Exists,
            "banana:anything",
            serde_json::Value::Null,
        );
        let graph = seeded(&note).await;
        let verdict = verify(&graph, &note).await.unwrap();
        assert_eq!(
            verdict,
            Verdict::Unverifiable("type de cible « banana » inconnu du graphe".into())
        );
        assert!(!verdict.is_refuted());
    }

    /// Une signature demande le texte source : le dire est la seule réponse honnête.
    #[tokio::test]
    async fn a_signature_check_is_unverifiable() {
        let note = a_belief(
            AssertionCheckType::SignatureContains,
            "function:main",
            serde_json::Value::Null,
        );
        let graph = seeded(&note).await;
        let verdict = verify(&graph, &note).await.unwrap();
        assert!(matches!(verdict, Verdict::Unverifiable(_)), "{verdict:?}");
        assert!(!verdict.is_refuted());
    }

    #[tokio::test]
    async fn a_refutation_flags_the_note_and_records_the_verdict() {
        let note = a_belief(
            AssertionCheckType::Exists,
            "function:vanished",
            serde_json::Value::Null,
        );
        let graph = seeded(&note).await;

        let report = reverify(&graph, 10).await.unwrap();
        assert_eq!(report.checked, 1);
        assert_eq!(report.refuted, 1);
        assert_eq!(report.flagged, 1);
        assert_eq!(report.drifted, 0, "elle n'a jamais tenu : pas une dérive");

        let stored = graph.notes.read().await.get(&note.id).cloned().unwrap();
        assert_eq!(stored.status, NoteStatus::NeedsReview);
        let verdict = stored
            .last_assertion_result
            .expect("le verdict est persisté");
        assert!(!verdict.passed);
    }

    #[tokio::test]
    async fn an_unverifiable_verdict_sanctions_nothing() {
        let note = a_belief(
            AssertionCheckType::SignatureContains,
            "function:main",
            serde_json::Value::Null,
        );
        let graph = seeded(&note).await;

        let report = reverify(&graph, 10).await.unwrap();
        assert_eq!(report.unverifiable, 1);
        assert_eq!(
            report.flagged, 0,
            "on ne casse pas ce qu'on ne sait pas vérifier"
        );
        assert_eq!(
            graph.notes.read().await.get(&note.id).unwrap().status,
            NoteStatus::Active
        );
    }

    /// La dérive est un PASSAGE : tenue hier, réfutée aujourd'hui.
    #[test]
    fn drift_is_a_passage_from_holding_to_refuted() {
        let held = AssertionResult::passed("ok");
        let refuted = Verdict::Fails("plus vrai".into());
        assert!(has_drifted(Some(&held), &refuted));
        assert!(
            !has_drifted(None, &refuted),
            "jamais tenu : erreur d'écriture"
        );
        assert!(!has_drifted(Some(&held), &Verdict::Holds("ok".into())));
        assert!(!has_drifted(
            Some(&AssertionResult::failed("déjà faux")),
            &refuted
        ));
    }

    #[tokio::test]
    async fn a_second_pass_sees_the_drift() {
        let note = a_belief(
            AssertionCheckType::Calls,
            "function:main",
            serde_json::json!({"to": "function:helper"}),
        );
        let graph = seeded(&note).await;
        graph.belief_links.write().await.insert((
            "function".into(),
            "main".into(),
            "CALLS".into(),
            "function".into(),
            "helper".into(),
        ));
        let first = reverify(&graph, 10).await.unwrap();
        assert_eq!(first.held, 1);

        // L'appel disparaît du graphe : la croyance dérive.
        graph.belief_links.write().await.clear();
        let second = reverify(&graph, 10).await.unwrap();
        assert_eq!(second.refuted, 1);
        assert_eq!(second.drifted, 1, "tenue puis réfutée : c'est une dérive");
    }
}
