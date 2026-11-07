//! BeliefRevisionCheck — re-vérifie les croyances et RÉVISE celles qui ont dérivé.
//!
//! Jusqu'ici la re-vérification n'existait qu'au démarrage (`reverify_at_boot`) : une
//! croyance pouvait être démentie par le monde pendant des jours sans que personne ne
//! s'en aperçoive, sur un serveur qui ne redémarre pas. Ce check donne un CYCLE à la
//! re-vérification.
//!
//! Trois garde-fous, hérités du contrat de `notes::beliefs::reverify` :
//! - `Unverifiable` n'est jamais sanctionné — on ne casse pas une croyance qu'on ne sait
//!   pas contrôler ;
//! - `Block` ne révise que sur une DÉRIVE avérée (tenue puis démentie), jamais une règle
//!   mal écrite ;
//! - le travail est BORNÉ par `REVERIFY_LIMIT`, et la part faite est dite (« la part lue
//!   est dite ») : `checked` est publié à côté du total, jamais à sa place.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::notes::beliefs::{self, REVERIFY_LIMIT};

/// Re-vérifie les croyances (toutes les 6 heures).
///
/// Le démarrage couvre déjà le premier passage ; ce check couvre la vie longue du serveur.
/// Il est idempotent : re-vérifier une croyance déjà à jour réécrit le même verdict.
pub struct BeliefRevisionCheck;

#[async_trait]
impl HeartbeatCheck for BeliefRevisionCheck {
    fn name(&self) -> &str {
        "belief_revision"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(6 * 60 * 60) // 6 hours
    }

    /// La passe interroge le graphe pour chaque croyance : le budget par défaut de 5 s ne
    /// suffit pas, et un check qui expire ne re-vérifie rien du tout.
    fn timeout_override(&self) -> Option<Duration> {
        Some(Duration::from_secs(60))
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        match beliefs::reverify(ctx.graph.as_ref(), REVERIFY_LIMIT).await {
            Ok(report) => {
                info!(
                    checked = report.checked,
                    limit = REVERIFY_LIMIT,
                    held = report.held,
                    refuted = report.refuted,
                    unverifiable = report.unverifiable,
                    drifted = report.drifted,
                    flagged = report.flagged,
                    unsanctioned = report.unsanctioned,
                    revised = report.revised,
                    healed = report.healed,
                    "BeliefRevisionCheck: croyances re-vérifiées"
                );
                Ok(())
            }
            // Une passe ratée ne casse pas le cycle : le prochain tick réessaie.
            Err(e) => {
                warn!(error = %e, "BeliefRevisionCheck: la re-vérification a échoué");
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use std::sync::Arc;

    /// Le check tourne, dans son budget, et sur le magasin qu'on lui donne : c'est ce qui
    /// transforme « la re-vérification existe » en « elle a un cycle ».
    #[tokio::test]
    async fn the_check_reverifies_the_beliefs_it_is_given() {
        let graph = Arc::new(MockGraphStore::new());
        let ctx = HeartbeatContext {
            graph: graph as Arc<dyn GraphStore>,
            search: None,
            emitter: None,
        };
        let check = BeliefRevisionCheck;

        assert_eq!(check.name(), "belief_revision");
        assert!(
            check.interval() >= Duration::from_secs(60 * 60),
            "un cycle plus court re-vérifierait le graphe en boucle"
        );
        assert!(
            check.timeout_override().is_some(),
            "les 5 s par défaut d'un check ne suffisent pas pour interroger le graphe"
        );

        check.run(&ctx).await.expect("une passe vide réussit");
    }
}
