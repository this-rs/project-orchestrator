//! UserProfile Enrichment Stage for the Chat Pipeline.
//!
//! Runs BEFORE knowledge injection to load the user's adaptive profile
//! and inject it as concise context (< 50 tokens) so the LLM can adapt
//! its responses to the user's preferences (verbosity, language, expertise).
//!
//! Controlled by `ENRICHMENT_USER_PROFILE=true` (disabled by default).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::time::timeout;
use tracing::debug;

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentContext, EnrichmentInput, EnrichmentStage,
};
use crate::neo4j::traits::GraphStore;

/// Enrichment stage that loads the user's adaptive profile.
pub struct UserProfileStage {
    graph: Arc<dyn GraphStore>,
}

impl UserProfileStage {
    /// Create a new user profile stage.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

#[async_trait::async_trait]
impl EnrichmentStage for UserProfileStage {
    async fn execute(&self, input: &EnrichmentInput, ctx: &mut EnrichmentContext) -> Result<()> {
        // Derive user_id from the session_id (same convention as aggregator)
        let user_id = format!("session:{}", input.session_id);

        // Load profile with a tight timeout (50ms max — profile is just a single node)
        let stage_timeout = Duration::from_millis(50);
        let graph = self.graph.clone();
        let uid = user_id.clone();

        let profile_result = timeout(
            stage_timeout,
            async move { graph.get_user_profile(&uid).await },
        )
        .await;

        let profile = match profile_result {
            Ok(Ok(Some(p))) => p,
            Ok(Ok(None)) => {
                debug!(
                    "[user_profile_stage] No profile for '{}', skipping",
                    user_id
                );
                return Ok(());
            }
            Ok(Err(e)) => {
                debug!(
                    "[user_profile_stage] Failed to load profile for '{}': {}",
                    user_id, e
                );
                return Ok(());
            }
            Err(_) => {
                debug!(
                    "[user_profile_stage] Profile load timed out for '{}'",
                    user_id
                );
                return Ok(());
            }
        };

        // Only inject if the profile has been updated at least once
        if profile.interaction_count == 0 {
            debug!(
                "[user_profile_stage] Profile for '{}' has no interactions yet, skipping",
                user_id
            );
            return Ok(());
        }

        // Format as concise markdown (< 50 tokens)
        let markdown = profile.to_prompt_markdown();
        ctx.add_section("User Profile", markdown, self.name());

        debug!(
            "[user_profile_stage] Injected profile for '{}' (interactions: {})",
            user_id, profile.interaction_count
        );

        Ok(())
    }

    fn name(&self) -> &str {
        "user_profile"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.user_profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::enrichment::EnrichmentInput;
    use std::sync::Arc;
    use uuid::Uuid;

    fn test_input() -> EnrichmentInput {
        EnrichmentInput {
            message: "test".to_string(),
            session_id: Uuid::new_v4(),
            project_slug: None,
            project_id: None,
            cwd: None,
        }
    }

    #[tokio::test]
    async fn test_no_profile_skips() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let stage = UserProfileStage::new(mock);

        let input = test_input();
        let mut ctx = EnrichmentContext::default();
        stage.execute(&input, &mut ctx).await.unwrap();

        assert!(
            !ctx.has_content(),
            "Should not inject anything when no profile exists"
        );
    }

    #[tokio::test]
    async fn test_zero_interactions_skips() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let input = test_input();

        // Create a profile with 0 interactions
        let user_id = format!("session:{}", input.session_id);
        mock.create_or_get_user_profile(&user_id).await.unwrap();

        let stage = UserProfileStage::new(mock);
        let mut ctx = EnrichmentContext::default();
        stage.execute(&input, &mut ctx).await.unwrap();

        assert!(
            !ctx.has_content(),
            "Should not inject for 0-interaction profiles"
        );
    }

    #[tokio::test]
    async fn test_active_profile_injected() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let input = test_input();

        // Create a profile with some interactions
        let user_id = format!("session:{}", input.session_id);
        let mut profile = mock.create_or_get_user_profile(&user_id).await.unwrap();
        profile.interaction_count = 10;
        profile.verbosity = 0.8;
        profile.language = "fr".to_string();
        mock.update_user_profile(&profile).await.unwrap();

        let stage = UserProfileStage::new(mock);
        let mut ctx = EnrichmentContext::default();
        stage.execute(&input, &mut ctx).await.unwrap();

        assert!(ctx.has_content(), "Should inject profile data");
        assert_eq!(ctx.sections.len(), 1);
        assert_eq!(ctx.sections[0].title, "User Profile");
        assert!(ctx.sections[0].content.contains("fr"));
        assert!(ctx.sections[0].content.contains("detailed"));
    }
}
