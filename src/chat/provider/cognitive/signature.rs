//! Task signature: what a request asks for, in terms a router can score.
//!
//! A [`TaskSignature`] is computed without any network call and without any
//! LLM: from the request text (keyword intent, bilingual FR/EN, through the
//! canonical [`IntentDetector`]), from the task and its steps (the runner's own
//! [`profile_task`] classification), or from what a delegating agent states.
//! Its [`TaskSignature::arm_key`] names the bandit arm family a decision belongs
//! to, so it must stay stable.

use serde::{Deserialize, Serialize};

use crate::chat::provider::resolver::Role;
use crate::neo4j::models::TaskNode;
use crate::neurons::intent::{IntentDetector, QueryIntentMode};
use crate::runner::persona::{profile_task, Complexity};

/// Smallest context a task is assumed to need, in tokens.
pub const MIN_CONTEXT_TOKENS: u64 = 8_000;
/// Largest context estimate, in tokens (a router never plans above this).
pub const MAX_CONTEXT_TOKENS: u64 = 200_000;
/// Tokens assumed per affected file when its size is unknown.
const TOKENS_PER_FILE: u64 = 6_000;
/// Characters per token (the estimate the native harness uses too).
const CHARS_PER_TOKEN: u64 = 4;

/// Intent of a conversational request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatIntent {
    /// Something fails: errors, crashes, "why".
    Debug,
    /// Understanding: how it works, explain.
    Explore,
    /// Changing or reviewing existing code.
    Impact,
    /// Creating, implementing, planning.
    Plan,
    /// No specific intent detected.
    General,
}

impl ChatIntent {
    /// Stable name, used in arm keys.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Explore => "explore",
            Self::Impact => "impact",
            Self::Plan => "plan",
            Self::General => "general",
        }
    }
}

impl From<QueryIntentMode> for ChatIntent {
    fn from(mode: QueryIntentMode) -> Self {
        match mode {
            QueryIntentMode::Debug => Self::Debug,
            QueryIntentMode::Explore => Self::Explore,
            QueryIntentMode::Impact => Self::Impact,
            QueryIntentMode::Plan => Self::Plan,
            QueryIntentMode::Default => Self::General,
        }
    }
}

/// Class of work. The runner classes mirror the policy rule roles
/// (`runner.simple|complex|creative|retry`, `utility.*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    /// Single-file, few steps.
    Simple,
    /// Multi-file, many steps, refactoring.
    Complex,
    /// Design or new feature.
    Creative,
    /// Any attempt after the first.
    Retry,
    /// One-shot feature-graph proposal.
    UtilityFeatureGraph,
    /// Context compaction summary.
    UtilityCompaction,
    /// A conversation turn.
    Chat(ChatIntent),
}

impl TaskClass {
    /// Stable key of the class: the bandit arm family. Never rename.
    pub fn key(self) -> String {
        match self {
            Self::Simple => "simple".into(),
            Self::Complex => "complex".into(),
            Self::Creative => "creative".into(),
            Self::Retry => "retry".into(),
            Self::UtilityFeatureGraph => "utility.feature_graph".into(),
            Self::UtilityCompaction => "utility.compaction".into(),
            Self::Chat(intent) => format!("chat.{}", intent.as_str()),
        }
    }

    /// Parses a class stated by a caller (`simple`, `complex`, `creative`,
    /// `retry`, `utility.feature_graph`, `utility.compaction`, `chat`,
    /// `chat.<intent>`). Anything else is `None`.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim().to_lowercase();
        Some(match text.as_str() {
            "simple" => Self::Simple,
            "complex" => Self::Complex,
            "creative" => Self::Creative,
            "retry" => Self::Retry,
            "utility.feature_graph" => Self::UtilityFeatureGraph,
            "utility.compaction" => Self::UtilityCompaction,
            "chat" => Self::Chat(ChatIntent::General),
            "chat.debug" => Self::Chat(ChatIntent::Debug),
            "chat.explore" => Self::Chat(ChatIntent::Explore),
            "chat.impact" => Self::Chat(ChatIntent::Impact),
            "chat.plan" => Self::Chat(ChatIntent::Plan),
            "chat.general" => Self::Chat(ChatIntent::General),
            _ => return None,
        })
    }

    /// Base complexity (1..=10) of the class, before the task's own signals.
    fn base_complexity(self) -> u8 {
        match self {
            Self::Simple | Self::UtilityCompaction => 2,
            Self::UtilityFeatureGraph => 3,
            Self::Complex => 6,
            Self::Creative => 7,
            Self::Retry => 5,
            Self::Chat(ChatIntent::Debug) => 5,
            Self::Chat(ChatIntent::Plan | ChatIntent::Impact) => 4,
            Self::Chat(ChatIntent::Explore) => 3,
            Self::Chat(ChatIntent::General) => 2,
        }
    }
}

impl From<Complexity> for TaskClass {
    fn from(complexity: Complexity) -> Self {
        match complexity {
            Complexity::Simple => Self::Simple,
            Complexity::Complex => Self::Complex,
            Complexity::Creative => Self::Creative,
        }
    }
}

/// What the caller knows about the context a task will need. All optional:
/// nothing here is read from disk by the signature code.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ContextHints {
    /// Total size of the affected files, in bytes, when known.
    pub affected_bytes: Option<u64>,
    /// Estimated tokens of the MCP tool schemas the session will carry.
    pub mcp_schema_tokens: u64,
    /// USD left in the run's marginal budget, when a budget applies.
    pub budget_remaining_usd: Option<f64>,
}

/// A request, as a router sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSignature {
    /// Pilot (a human's conversation) or executor (runner, delegation, utility).
    pub role: Role,
    /// Class of work.
    pub class: TaskClass,
    /// 1 (trivial) ..= 10 (hardest).
    pub complexity: u8,
    /// Tokens the model must be able to hold, bounded to
    /// [`MIN_CONTEXT_TOKENS`]`..=`[`MAX_CONTEXT_TOKENS`].
    pub context_need_tokens: u64,
    /// Whether the session calls tools.
    pub needs_tools: bool,
    /// Whether the input carries images.
    pub needs_images: bool,
    /// USD left in the marginal budget, `None` when no budget applies.
    pub budget_remaining_usd: Option<f64>,
    /// Attempt number, starting at 1.
    pub attempt: u32,
    /// Project the request belongs to.
    pub project_slug: Option<String>,
    /// Persona acting, if any.
    pub persona: Option<String>,
}

impl TaskSignature {
    /// Key of the bandit arm family of this signature (the class key).
    pub fn arm_key(&self) -> String {
        self.class.key()
    }

    /// A conversation turn typed by a human.
    ///
    /// The intent comes from the canonical bilingual keyword detector. The
    /// length of the message grows the complexity by at most two points, and
    /// the context need by the message itself.
    pub fn from_chat_request(
        message: &str,
        has_images: bool,
        project_slug: Option<&str>,
        hints: ContextHints,
    ) -> Self {
        let class = TaskClass::Chat(IntentDetector::detect(message).into());
        let length_bonus = (message.chars().count() / 800).min(2) as u8;
        let message_tokens = message.chars().count() as u64 / CHARS_PER_TOKEN;
        Self {
            role: Role::Pilot,
            class,
            complexity: clamp_complexity(class.base_complexity() + length_bonus),
            context_need_tokens: context_need(None, 0, message_tokens, &hints),
            needs_tools: true,
            needs_images: has_images,
            budget_remaining_usd: hints.budget_remaining_usd,
            attempt: 1,
            project_slug: project_slug.map(str::to_owned),
            persona: None,
        }
    }

    /// A task of a plan, run by the runner.
    ///
    /// The class is the runner's own classification ([`profile_task`]), except
    /// that any attempt after the first is [`TaskClass::Retry`]: the rule role
    /// `runner.retry` never matched before, because the runner always passed
    /// the complexity class.
    pub fn from_task(
        task: &TaskNode,
        steps_count: usize,
        attempt: u32,
        project_slug: Option<&str>,
        hints: ContextHints,
    ) -> Self {
        let attempt = attempt.max(1);
        let profiled: TaskClass = profile_task(task, steps_count).complexity.into();
        let class = if attempt > 1 {
            TaskClass::Retry
        } else {
            profiled
        };
        let files = task.affected_files.len();
        let derived = profiled.base_complexity()
            + (files.min(6) / 2) as u8
            + (steps_count.min(12) / 4) as u8
            + u8::from(attempt > 1);
        let complexity = match task.estimated_complexity {
            Some(estimated) if estimated > 0 => clamp_complexity(estimated.min(10) as u8),
            _ => clamp_complexity(derived),
        };
        Self {
            role: Role::Executor,
            class,
            complexity,
            context_need_tokens: context_need(Some(files), steps_count, 0, &hints),
            needs_tools: true,
            needs_images: false,
            budget_remaining_usd: hints.budget_remaining_usd,
            attempt,
            project_slug: project_slug.map(str::to_owned),
            persona: task.persona.clone(),
        }
    }

    /// A delegated task. A class stated by the delegating agent wins; else the
    /// task's own classification; else `simple`.
    pub fn from_delegation(
        stated_class: Option<&str>,
        task: Option<(&TaskNode, usize)>,
        attempt: u32,
        project_slug: Option<&str>,
        hints: ContextHints,
    ) -> Self {
        if let Some((task, steps)) = task {
            let mut signature = Self::from_task(task, steps, attempt, project_slug, hints);
            if let Some(class) = stated_class.and_then(TaskClass::parse) {
                // A stated class keeps its complexity floor.
                signature.class = if signature.attempt > 1 {
                    TaskClass::Retry
                } else {
                    class
                };
                signature.complexity = signature.complexity.max(class.base_complexity());
            }
            return signature;
        }
        let attempt = attempt.max(1);
        let class = if attempt > 1 {
            TaskClass::Retry
        } else {
            stated_class
                .and_then(TaskClass::parse)
                .unwrap_or(TaskClass::Simple)
        };
        Self {
            role: Role::Executor,
            class,
            complexity: clamp_complexity(class.base_complexity()),
            context_need_tokens: context_need(None, 0, 0, &hints),
            needs_tools: true,
            needs_images: false,
            budget_remaining_usd: hints.budget_remaining_usd,
            attempt,
            project_slug: project_slug.map(str::to_owned),
            persona: None,
        }
    }

    /// A utility one-shot (feature-graph proposal, compaction summary): no
    /// tools, an executor, the context it is handed.
    pub fn utility(class: TaskClass, context_tokens: u64, project_slug: Option<&str>) -> Self {
        let class = match class {
            TaskClass::UtilityCompaction => TaskClass::UtilityCompaction,
            _ => TaskClass::UtilityFeatureGraph,
        };
        Self {
            role: Role::Executor,
            class,
            complexity: clamp_complexity(class.base_complexity()),
            context_need_tokens: context_tokens.clamp(MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS),
            needs_tools: false,
            needs_images: false,
            budget_remaining_usd: None,
            attempt: 1,
            project_slug: project_slug.map(str::to_owned),
            persona: None,
        }
    }
}

fn clamp_complexity(value: u8) -> u8 {
    value.clamp(1, 10)
}

/// Context a task needs: a floor, plus the affected files (their known size,
/// else a fixed estimate per file), the steps, the message and the MCP schemas.
fn context_need(
    files: Option<usize>,
    steps: usize,
    message_tokens: u64,
    hints: &ContextHints,
) -> u64 {
    let files_tokens = match (hints.affected_bytes, files) {
        (Some(bytes), _) => bytes / CHARS_PER_TOKEN,
        (None, Some(count)) => count as u64 * TOKENS_PER_FILE,
        (None, None) => 0,
    };
    let total = MIN_CONTEXT_TOKENS
        + files_tokens
        + steps as u64 * 500
        + message_tokens
        + hints.mcp_schema_tokens;
    total.clamp(MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(files: usize, tags: &[&str]) -> TaskNode {
        let mut task = TaskNode::new("work".to_string());
        task.affected_files = (0..files).map(|n| format!("src/f{n}.rs")).collect();
        task.tags = tags.iter().map(|t| (*t).to_string()).collect();
        task
    }

    fn chat(message: &str) -> TaskSignature {
        TaskSignature::from_chat_request(message, false, None, ContextHints::default())
    }

    /// The same request in French and in English lands in the same class.
    #[test]
    fn the_class_does_not_depend_on_the_language() {
        let pairs = [
            ("pourquoi ce test échoue", "why does this test fail"),
            (
                "il y a une erreur au démarrage",
                "there is an error at startup",
            ),
            ("comment fonctionne le routeur", "how does the router work"),
            (
                "je veux comprendre ce module",
                "i want to understand this module",
            ),
            ("supprimer ce fichier", "delete this file"),
            (
                "déplacer la fonction ailleurs",
                "move the function elsewhere",
            ),
            ("implémenter la pagination", "implement the pagination"),
            ("planifie la migration", "plan the migration"),
            ("bonjour, merci", "hello, thanks"),
            ("quelle heure est-il", "what time is it"),
        ];
        for (fr, en) in pairs {
            assert_eq!(chat(fr).class, chat(en).class, "{fr:?} vs {en:?}");
        }
        // And the pairs are not all the same class: the test would prove nothing.
        let classes: std::collections::HashSet<_> =
            pairs.iter().map(|(fr, _)| chat(fr).class).collect();
        assert!(classes.len() >= 4, "{classes:?}");
    }

    #[test]
    fn a_chat_request_is_a_pilot_with_tools_and_reads_images_from_the_caller() {
        let signature = TaskSignature::from_chat_request(
            "explain the build",
            true,
            Some("po"),
            ContextHints::default(),
        );
        assert_eq!(signature.role, Role::Pilot);
        assert!(signature.needs_tools && signature.needs_images);
        assert_eq!(signature.project_slug.as_deref(), Some("po"));
        assert_eq!(signature.attempt, 1);
    }

    /// `runner.retry` never matched: the runner always passed the complexity class.
    #[test]
    fn any_attempt_after_the_first_is_the_retry_class() {
        let task = task(3, &["refactor"]);
        let first = TaskSignature::from_task(&task, 8, 1, None, ContextHints::default());
        assert_eq!(first.class, TaskClass::Complex);
        let second = TaskSignature::from_task(&task, 8, 2, None, ContextHints::default());
        assert_eq!(second.class, TaskClass::Retry);
        assert_eq!(second.attempt, 2);
        assert!(second.complexity >= first.complexity);
        // Attempt 0 is a first attempt.
        let zero = TaskSignature::from_task(&task, 8, 0, None, ContextHints::default());
        assert_eq!((zero.class, zero.attempt), (TaskClass::Complex, 1));
    }

    #[test]
    fn the_task_class_is_the_runners_own_classification() {
        let hints = ContextHints::default();
        let class = |t: &TaskNode, steps| TaskSignature::from_task(t, steps, 1, None, hints).class;
        assert_eq!(class(&task(1, &[]), 2), TaskClass::Simple);
        assert_eq!(class(&task(3, &[]), 9), TaskClass::Complex);
        assert_eq!(class(&task(1, &["design"]), 1), TaskClass::Creative);
        assert_eq!(class(&task(2, &["architecture"]), 1), TaskClass::Complex);
    }

    #[test]
    fn complexity_is_the_estimate_when_given_else_derived_and_always_bounded() {
        let hints = ContextHints::default();
        let mut explicit = task(1, &[]);
        explicit.estimated_complexity = Some(9);
        assert_eq!(
            TaskSignature::from_task(&explicit, 1, 1, None, hints).complexity,
            9
        );
        explicit.estimated_complexity = Some(99);
        assert_eq!(
            TaskSignature::from_task(&explicit, 1, 1, None, hints).complexity,
            10
        );
        let small = TaskSignature::from_task(&task(1, &[]), 1, 1, None, hints).complexity;
        let big = TaskSignature::from_task(&task(6, &["refactor"]), 12, 1, None, hints).complexity;
        assert!(small < big, "{small} < {big}");
        assert!((1..=10).contains(&small) && (1..=10).contains(&big));
        // Zero means "not estimated" (every task in the graph carries 0 today).
        let mut zero = task(1, &[]);
        zero.estimated_complexity = Some(0);
        assert_eq!(
            TaskSignature::from_task(&zero, 1, 1, None, hints).complexity,
            small
        );
    }

    #[test]
    fn the_context_need_grows_with_the_files_and_stays_bounded() {
        let hints = ContextHints::default();
        let need = |files| {
            TaskSignature::from_task(&task(files, &[]), 2, 1, None, hints).context_need_tokens
        };
        assert!(need(0) >= MIN_CONTEXT_TOKENS);
        assert!(need(1) < need(4) && need(4) < need(8));
        assert_eq!(need(500), MAX_CONTEXT_TOKENS);
        // Known sizes win over the per-file estimate.
        let sized = ContextHints {
            affected_bytes: Some(400_000),
            ..ContextHints::default()
        };
        let known = TaskSignature::from_task(&task(1, &[]), 2, 1, None, sized).context_need_tokens;
        assert!(known > need(1), "{known}");
        // The MCP schemas count.
        let with_mcp = ContextHints {
            mcp_schema_tokens: 20_000,
            ..ContextHints::default()
        };
        assert!(
            TaskSignature::from_task(&task(1, &[]), 2, 1, None, with_mcp).context_need_tokens
                >= need(1) + 20_000
        );
    }

    #[test]
    fn a_stated_delegation_class_wins_over_the_default_and_a_retry_wins_over_both() {
        let hints = ContextHints::default();
        let stated = TaskSignature::from_delegation(Some("creative"), None, 1, None, hints);
        assert_eq!(stated.class, TaskClass::Creative);
        let unknown = TaskSignature::from_delegation(Some("nonsense"), None, 1, None, hints);
        assert_eq!(unknown.class, TaskClass::Simple);
        assert_eq!(
            TaskSignature::from_delegation(None, None, 1, None, hints).class,
            TaskClass::Simple
        );
        let retried = TaskSignature::from_delegation(Some("creative"), None, 3, None, hints);
        assert_eq!((retried.class, retried.attempt), (TaskClass::Retry, 3));
        // With a task, the stated class applies and never lowers the complexity floor.
        let t = task(1, &[]);
        let with_task =
            TaskSignature::from_delegation(Some("complex"), Some((&t, 1)), 1, None, hints);
        assert_eq!(with_task.class, TaskClass::Complex);
        assert!(with_task.complexity >= TaskClass::Complex.base_complexity());
        assert_eq!(with_task.role, Role::Executor);
    }

    #[test]
    fn a_utility_call_needs_no_tools_and_is_an_executor() {
        let compaction = TaskSignature::utility(TaskClass::UtilityCompaction, 50_000, Some("po"));
        assert_eq!(compaction.class, TaskClass::UtilityCompaction);
        assert!(!compaction.needs_tools && compaction.role == Role::Executor);
        assert_eq!(compaction.context_need_tokens, 50_000);
        // Any other class asked for is the feature-graph utility; the context is bounded.
        let graph = TaskSignature::utility(TaskClass::Simple, 1, None);
        assert_eq!(graph.class, TaskClass::UtilityFeatureGraph);
        assert_eq!(graph.context_need_tokens, MIN_CONTEXT_TOKENS);
    }

    #[test]
    fn arm_keys_are_stable_and_parse_round_trips() {
        let keys = [
            (TaskClass::Simple, "simple"),
            (TaskClass::Complex, "complex"),
            (TaskClass::Creative, "creative"),
            (TaskClass::Retry, "retry"),
            (TaskClass::UtilityFeatureGraph, "utility.feature_graph"),
            (TaskClass::UtilityCompaction, "utility.compaction"),
            (TaskClass::Chat(ChatIntent::Debug), "chat.debug"),
            (TaskClass::Chat(ChatIntent::General), "chat.general"),
        ];
        for (class, key) in keys {
            assert_eq!(class.key(), key);
            assert_eq!(TaskClass::parse(key), Some(class));
        }
        assert_eq!(
            TaskClass::parse(" CHAT "),
            Some(TaskClass::Chat(ChatIntent::General))
        );
        assert_eq!(TaskClass::parse("whatever"), None);
    }

    #[test]
    fn a_signature_survives_a_json_round_trip() {
        let signature = TaskSignature::from_task(
            &task(2, &["refactor"]),
            7,
            2,
            Some("po"),
            ContextHints {
                budget_remaining_usd: Some(1.5),
                ..ContextHints::default()
            },
        );
        let json = serde_json::to_string(&signature).unwrap();
        let back: TaskSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(back, signature);
    }
}
