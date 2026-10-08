//! Boot wiring of the learning side of cognitive routing: the per-turn decider
//! of mode `full` and the runner's routing handle. Both need the chat manager's
//! pool, and the manager owns them, so they hold it weakly.

use std::sync::{Arc, Weak};

use async_trait::async_trait;

use super::candidates::ModelFacts;
use crate::chat::agent_hooks::PoolSource;
use crate::chat::manager::ChatManager;
use crate::runner::routing::{RoutingHandle, RoutingPool};

/// The routing pool of a chat manager, held weakly (the manager owns the
/// per-turn routing that owns this).
pub(crate) struct ManagerPool(pub Weak<ChatManager>);

#[async_trait]
impl PoolSource for ManagerPool {
    async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
        let Some(manager) = self.0.upgrade() else {
            return Vec::new();
        };
        manager
            .routing_pool_for(None)
            .await
            .into_iter()
            .filter(|facts| facts.provider_id == provider_id)
            .collect()
    }
}

#[async_trait]
impl RoutingPool for ManagerPool {
    async fn facts(&self, project_slug: Option<&str>) -> Vec<ModelFacts> {
        match self.0.upgrade() {
            Some(manager) => manager.routing_pool_for(project_slug).await,
            None => Vec::new(),
        }
    }
}

/// Gives the per-turn router its decider and pool, and installs the runner's
/// routing handle. Does nothing when the manager has no cognitive router with a
/// store.
pub(crate) fn wire_learning(manager: &Arc<ChatManager>) {
    let Some(routing) = manager.cognitive_routing() else {
        return;
    };
    let Some(store) = routing.store.clone() else {
        return;
    };
    let pool = Arc::new(ManagerPool(Arc::downgrade(manager)));
    manager
        .turn_routing
        .configure(Arc::clone(&routing.decider), pool.clone());
    crate::runner::routing::install(Some(Arc::new(RoutingHandle::new(
        Arc::clone(&routing.decider),
        store,
        pool,
    ))));
}
