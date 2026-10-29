//! Engine parity, MEASURED (plan 5ad54c48 « Parité totale », task P0).
//!
//! ONE scenario, played on the two engines a conversation can live on:
//!
//! - **Claude Code** on the historical engine (`ProviderPath::Legacy`):
//!   `InteractiveClient` on `fake_claude`, the scripted stand-in for the `claude` CLI
//!   of the nexus repository, behind a wrapper that hands every spawn its own
//!   transcript (the CLI is spawned again by a resume and by a switch back) and
//!   records what each spawn read;
//! - **Claude Code on the agent engine** (`ProviderPath::Agent`, the default since
//!   M2): the same CLI and transcript, driven through the nexus Claude Code façade.
//!   The scenario runs a second time for it, on its own stage (only the place of the
//!   work context restored after a compaction differs: its own turn on the historical
//!   engine, in front of the next message on the agent engine);
//! - the **native harness** (agent engine): a registered OpenAI-compatible
//!   instance on `fake_openai`, the project-orchestrator MCP server played by
//!   `fake_mcp`, and the REAL `nexus-tools` (Read, Edit, Bash...).
//!
//! The same assertion is made per function on both engines and the verdicts are
//! written to `docs/parity/engines.md`; a drift gate compares the committed file
//! with what this run measured (`UPDATE_PARITY_MATRIX=1` rewrites it).
//!
//! Every cell that is not `ok` must be DECLARED in [`EXPECTED`]: a gap names its
//! cause (harness or model) and the PO task that closes it; a function the fakes
//! cannot exercise is `not_measured`, with the reason. The test fails on a gap
//! that is not declared, and on a declared gap that is no longer measured (fixed
//! silently: the list and the table must say so). Independently of that list,
//! the native `system_init` may only announce a missing feature that is a limit
//! of the MODEL (closed list): anything else fails, it cannot be declared away.
//!
//! Offline and deterministic. `NEXUS_FAKES_DIR` must hold `fake_claude`,
//! `fake_openai`, `fake_mcp` and `nexus-tools` built at the nexus revision
//! Cargo.toml pins (CI: `.github/actions/nexus-fakes`); a missing one FAILS.
//!
//! A STALE one is refused wherever it cannot report what is measured, instead of being
//! read as a parity gap: a `fake_openai` that does not record request headers hides
//! `anthropic-version`, and the cell then accused the harness on a `main` whose CI was
//! green (28/10/2026, see [`anthropic_messages_verdict`]).
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use nexus_claude::agent::{
    AgentProvider, AgentSession, Capabilities, CompactionInfo, HookReply, HookVerdict, ModelInfo,
    NotificationInfo, PromptInfo, ProviderError, ProviderHealth, ProviderKind, ResumeToken,
    SessionHooks, SessionLimits, SessionSpec, SessionStartInfo, StopInfo, SubagentRequest,
    SubagentStopInfo, ToolCallInfo, ToolResultInfo, TurnContext, TurnDirective,
};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use uuid::Uuid;

use super::agent_e2e_tests::{
    consent, delta, fake_bin, fakes_dir, instance, request, sse_route, store_instance, FakeOpenAi,
};
use super::agent_runtime::ProviderSource;
use super::config::{ChatConfig, ProviderPath};
use super::manager::ChatManager;
use super::provider::settings::{process_origin, AcpCommand, InstanceRecord};
use super::types::ChatEvent;
use crate::documents::store::DocumentStore;
use crate::events::nats_broker_test::TestBroker;
use crate::events::NatsEmitter;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::GraphStore;

#[allow(unused_imports)]
use crate::refs::types::{EntityRef, RefKind};
use crate::test_helpers::mock_app_state;

// ============================================================================
// The matrix: functions, verdicts, the declared gaps
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    /// Claude Code on the historical engine (`ProviderPath::Legacy`).
    ClaudeCode,
    /// Claude Code on the agent engine (`ProviderPath::Agent`): the nexus façade over the
    /// same `fake_claude`, the same transcript (M2).
    ClaudeCodeAgent,
    Native,
    /// Codex provider (harness `codex`). `fake_codex` exists in the nexus repository
    /// (`src/bin/fake_codex.rs`), but it is transcript-based: each function of the
    /// parity scenario would need its own JSONL transcript. Per-function verdicts are
    /// declared in [`EXPECTED`]; functions with no transcript declare `not_measured`
    /// (ISO4, task 87eb345e).
    Codex,
    /// opencode harness. `fake_opencode` does not exist; all functions declared
    /// `not_measured` (YARDSTICK, task 87eb345e).
    OpenCode,
    /// ACP harness. `fake_acp` exists in the nexus repository. Per-function verdicts
    /// are declared in [`EXPECTED`] as `declared(harnais)` (ISO4) or `not_measured`.
    Acp,
}

/// The columns of the table, in order.
const ENGINES: [Engine; 6] = [
    Engine::ClaudeCode,
    Engine::ClaudeCodeAgent,
    Engine::Native,
    Engine::Codex,
    Engine::OpenCode,
    Engine::Acp,
];

impl Engine {
    fn label(self) -> &'static str {
        match self {
            Engine::ClaudeCode => "Claude Code",
            Engine::ClaudeCodeAgent => "Claude Code (moteur agent)",
            Engine::Native => "natif",
            Engine::Codex => "Codex",
            Engine::OpenCode => "opencode",
            Engine::Acp => "ACP",
        }
    }
}

/// What one run of the scenario measured, per engine.
type Run<'a> = [(Engine, &'a Measures)];

/// Why a function is missing on an engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cause {
    /// The harness (this backend, or the nexus harness it drives) does not do it.
    Harness,
    /// The model behind the engine cannot do it.
    Model,
}

impl Cause {
    fn label(self) -> &'static str {
        match self {
            Cause::Harness => "harnais",
            Cause::Model => "modèle",
        }
    }
}

/// What the scenario saw for one function on one engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Ok,
    Gap(Cause),
    NotMeasured,
    /// The function is declared as a gap in [`EXPECTED`] but the parity scenario was
    /// NOT run on this engine (no transcript-based fake exists yet). Distinct from
    /// [`Verdict::Gap`] (measured) and [`Verdict::NotMeasured`] (fundamentally impossible
    /// to exercise). Rendered as `declared (cause, task)` in the matrix.
    Declared(Cause),
}

/// What [`EXPECTED`] says of a cell that is not `ok`.
#[derive(Clone, Copy, Debug)]
enum Expect {
    /// A known gap: its cause, the PO task that closes it, what is missing.
    Gap {
        cause: Cause,
        task: &'static str,
        why: &'static str,
    },
    /// The fakes cannot exercise the function on that engine.
    NotMeasured { why: &'static str },
}

/// The functions of the matrix, in the order of the table: (id, what is asserted).
const FUNCTIONS: &[(&str, &str)] = &[
    (
        "hooks.before_tool",
        "le hook PreToolUse du graphe est consulté avant un outil",
    ),
    (
        "hooks.after_tool",
        "le hook PostToolUse du graphe répond après un Grep bruyant (find_references)",
    ),
    (
        "hooks.before_compaction",
        "le hook PreCompact du graphe nomme le projet dans les instructions du résumé (Claude Code : `systemMessage`, le champ que lit le CLI)",
    ),
    (
        "hooks.session_start",
        "le hook SessionStart du graphe est consulté à l'ouverture de la session (contract v7)",
    ),
    (
        "hooks.user_prompt_submit",
        "le hook UserPromptSubmit du graphe est consulté avant chaque tour (contract v7)",
    ),
    (
        "hooks.stop",
        "le hook Stop du graphe est consulté avant la fin du tour ; un block fait continuer le tour avec la raison (contract v7)",
    ),
    (
        "hooks.subagent_stop",
        "le hook SubagentStop du graphe est consulté quand un sous-agent s'arrête (contract v7)",
    ),
    (
        "hooks.notification",
        "le hook Notification du graphe est appelé quand une permission est attendue (contract v7)",
    ),
    (
        "compaction",
        "compaction_started et compact_boundary sur le fil",
    ),
    (
        "compaction.manual",
        "`/compact <instructions>` envoyé depuis le chat : Claude Code reçoit la commande telle quelle (le CLI compacte, compact_boundary trigger manual) ; le natif compacte hors tour (AgentSession::compact), les instructions et le hook PreCompact dans la requête de résumé, aucun tour envoyé au modèle, compaction_started / compact_boundary trigger manual sur le fil (ISO3)",
    ),
    (
        "message_queue",
        "un message envoyé pendant un tour l'interrompt et passe ensuite",
    ),
    (
        "auto_continue",
        "un tour arrêté sur sa limite est continué (auto_continue)",
    ),
    (
        "nats.send",
        "un message d'une autre instance (NATS) est joué ici",
    ),
    (
        "nats.interrupt",
        "un Stop d'une autre instance (NATS) arrête le tour",
    ),
    (
        "nats.permission_response",
        "une réponse de permission d'une autre instance (NATS) débloque l'outil",
    ),
    (
        "nats.cancel_tools",
        "un cancel_tools d'une autre instance (NATS) arrête l'outil, le tour continue",
    ),
    (
        "resume",
        "après redémarrage du backend, la session reprend sur le jeton relu du graphe",
    ),
    (
        "set_model",
        "set_model en cours de conversation atteint le provider",
    ),
    (
        "cancel_tools",
        "cancel_tools arrête l'outil en cours, le tour continue",
    ),
    (
        "permissions.once",
        "une permission accordée une fois débloque l'outil",
    ),
    (
        "permissions.session",
        "une permission accordée pour la session n'est pas redemandée",
    ),
    (
        "permissions.always",
        "une permission accordée pour toujours est retenue au-delà de la session",
    ),
    (
        "tool_timing",
        "un outil qui a attendu une permission porte tool_timing persisté : prise en charge, attente, exécution, fin",
    ),
    (
        "po_tools",
        "les outils project-orchestrator (MCP) sont donnés et appelables",
    ),
    (
        "tool_search",
        "un outil différé n'est pas envoyé ; ToolSearch le charge, il est ajouté EN FIN de liste et appelable dans le même tour (CTX2)",
    ),
    (
        "nexus_tools",
        "Read / Edit / Bash s'exécutent sur le projet",
    ),
    (
        "tool_result.huge",
        "un résultat d'outil MCP d'environ 1 Mo ne tue pas le tour : seul un avis borné (moins de 8 Ko) atteint le fil et le modèle, le tour se clôt et le suivant démarre ; natif : le mégaoctet est gardé sur disque, au chemin que l'avis nomme (Claude Code : ce que fait le CLI est relevé, puis rejoué par fake_claude)",
    ),
    (
        "enrichment",
        "le contexte du graphe précède le message du tour",
    ),
    (
        "refs",
        "une référence #kind:id atteint le modèle en pointeur po-context",
    ),
    (
        "provider_switch.relay",
        "la bascule vers ce moteur relaie la conversation sur le même fil (provider_switched)",
    ),
    (
        "images",
        "une image jointe atteint le provider en bloc image",
    ),
    (
        "session_record",
        "le dossier de session porte message_count, total_cost_usd et un titre",
    ),
    (
        "background_tasks",
        "une tâche d'arrière-plan est suivie (active_tasks_update)",
    ),
    ("cancel_task", "cancel_task arrête une tâche d'arrière-plan"),
    (
        "ask_user_question",
        "une question du modèle s'affiche (ask_user_question) et la réponse de l'utilisateur lui revient au tour suivant",
    ),
    (
        "subagents",
        "un sous-agent (Agent / Task) travaille et sa réponse revient en résultat ; sur le natif, une session enfant qui ne peut pas re-déléguer",
    ),
    (
        "subagents.background",
        "un sous-agent lancé en arrière-plan (run_in_background: true) rend la main au parent immédiatement ; sa fin est injectée comme task_notification au tour suivant",
    ),
    (
        "subagents.types",
        "subagent_type valide parmi les types déclarés ; un type inconnu est refusé avec la liste des types connus",
    ),
    (
        "system_init.degraded",
        "system_init n'annonce comme manquant qu'une limite du modèle (liste fermée)",
    ),
    (
        "session_limits",
        "sans réglage, aucune limite de durée par tour ni budget de session (ISO5) : le CLI ne reçoit pas --max-budget-usd, une session du moteur agent s'ouvre et reprend sans turn_timeout_ms ni max_tokens",
    ),
    (
        "anthropic_messages",
        "une instance `anthropic` (clé au coffre) : le harnais natif parle lui-même l'API Messages, et la requête reçue par le modèle porte les trois points cache_control — dernier outil, système, dernier tour (CTX5)",
    ),
];

/// Every cell of the table that is not `ok`, declared. A gap measured and not
/// listed here fails the test; an entry here that the scenario no longer
/// measures fails it too (update this list and regenerate the table).
const EXPECTED: &[(Engine, &str, Expect)] = &[
    // ── Claude Code (both engines) ────────────────────────────────────────
    (Engine::ClaudeCode, "nexus_tools", Expect::NotMeasured { why: CLI_TOOLS }),
    (Engine::ClaudeCodeAgent, "nexus_tools", Expect::NotMeasured { why: CLI_TOOLS }),
    (Engine::ClaudeCode, "tool_search", Expect::NotMeasured { why: CLI_TOOL_SEARCH }),
    (Engine::ClaudeCodeAgent, "tool_search", Expect::NotMeasured { why: CLI_TOOL_SEARCH }),
    (Engine::ClaudeCode, "anthropic_messages", Expect::NotMeasured { why: CLI_PROMPT_CACHE }),
    (Engine::ClaudeCodeAgent, "anthropic_messages", Expect::NotMeasured { why: CLI_PROMPT_CACHE }),
    // Native/anthropic_messages: ok with correct fakes (header_values["anthropic-version"] is
    // captured by fake_openai ≥ pin 0587011). No EXPECTED entry: a fake too old to report the
    // headers is REFUSED by `anthropic_messages_verdict` (it used to read as `gap (harnais)`,
    // which accused a green `main` — measured 28/10/2026).
    // ISO2: subagents.background — the parity scenario is turn-based (fake_openai), so a
    // background sub-agent whose inject_hint triggers a new turn would need a matching
    // script entry; tested by dedicated unit tests (unknown_subagent_type_is_refused,
    // denied_tools_populate_chat_request) and the background path in run_subagent.
    (Engine::Native, "subagents.background", Expect::NotMeasured { why: "le scénario turn-based ne peut pas exercer inject_hint sans entrée script ; chemin testé par les tests unitaires ISO2" }),
    (Engine::ClaudeCode, "subagents.background", Expect::NotMeasured { why: "fake_claude rejoue un transcript sans Agent(run_in_background) ; le miroir P18c le mesure séparément" }),
    (Engine::ClaudeCodeAgent, "subagents.background", Expect::NotMeasured { why: "fake_claude rejoue un transcript sans Agent(run_in_background) ; le miroir P18c le mesure séparément" }),
    (Engine::ClaudeCode, "subagents.types", Expect::NotMeasured { why: "la validation des types est propre au natif (ISO2) ; sur Claude Code le CLI gère ses propres types d'agents" }),
    (Engine::ClaudeCodeAgent, "subagents.types", Expect::NotMeasured { why: "la validation des types est propre au natif (ISO2) ; sur Claude Code le CLI gère ses propres types d'agents" }),
    // ── Codex — known harness gaps (A40 hooks, A12 images, no bg-tasks) ──
    (Engine::Codex, "hooks.before_tool", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.after_tool", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.before_compaction", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.session_start", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.user_prompt_submit", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.stop", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.subagent_stop", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "hooks.notification", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_HOOKS }),
    (Engine::Codex, "background_tasks", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_NO_BG }),
    (Engine::Codex, "cancel_task", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: CODEX_NO_BG }),
    (Engine::Codex, "images", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: "Codex v1 n'accepte pas les blocs image (A12)" }),
    (Engine::Codex, "anthropic_messages", Expect::NotMeasured { why: "Codex utilise son propre moteur LLM (API Codex v1), pas l'API Anthropic Messages ; concept sans objet sur ce moteur" }),
    (Engine::Codex, "ask_user_question", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: "requestUserInput est expérimental en v1 ; le backend le synthétise mais ne peut pas l'interroger via Codex" }),
    // ── Codex — not_measured (transcript-based fake_codex needs per-function JSONL) ──
    (Engine::Codex, "compaction", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "compaction.manual", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "message_queue", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "auto_continue", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "nats.send", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "nats.interrupt", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "nats.permission_response", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "nats.cancel_tools", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "resume", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "set_model", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "cancel_tools", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "permissions.once", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "permissions.session", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "permissions.always", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "tool_timing", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "po_tools", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "tool_search", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "nexus_tools", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "tool_result.huge", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "enrichment", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "refs", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "session_record", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "session_limits", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    (Engine::Codex, "subagents", Expect::NotMeasured { why: "collabToolCall (sous-agents Codex) NOT VERIFIED dans la suite nexus ; transcript nécessaire" }),
    (Engine::Codex, "subagents.background", Expect::NotMeasured { why: "collabToolCall (sous-agents Codex) NOT VERIFIED dans la suite nexus ; transcript nécessaire" }),
    (Engine::Codex, "subagents.types", Expect::NotMeasured { why: "collabToolCall (sous-agents Codex) NOT VERIFIED dans la suite nexus ; transcript nécessaire" }),
    (Engine::Codex, "system_init.degraded", Expect::NotMeasured { why: CODEX_TRANSCRIPT }),
    // ── ACP — known harness gaps ─────────────────────────────────────────
    (Engine::Acp, "hooks.before_tool", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.after_tool", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.before_compaction", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.session_start", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.user_prompt_submit", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.stop", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.subagent_stop", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "hooks.notification", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: ACP_HOOKS }),
    (Engine::Acp, "nexus_tools", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: "ACP utilise son propre environnement d'outils ; nexus-tools n'est pas joint à une session ACP" }),
    (Engine::Acp, "anthropic_messages", Expect::NotMeasured { why: "ACP utilise son propre moteur LLM (protocole ACP), pas l'API Anthropic Messages ; concept sans objet sur ce moteur" }),
    (Engine::Acp, "ask_user_question", Expect::NotMeasured { why: "le backend synthétise la question via le harnais natif ; non exercé dans le scénario de parité (transcripts ACP non adaptés, ISO4)" }),
    (Engine::Acp, "enrichment", Expect::NotMeasured { why: "ACP injecte le contexte PO via le bloc system du premier session/prompt ; la vérification parity cherche PLAN_TITLE dans les requêtes OpenAI qui n'existent pas pour ACP ; non exercé dans ce scénario (ISO4)" }),
    (Engine::Acp, "compaction.manual", Expect::Gap { cause: Cause::Harness, task: "ISO4", why: "/compact n'est pas transmis à un agent ACP : la compaction manuelle n'a pas de canal côté harnais" }),
    // ── ACP — ok (measured in run_acp_column, relay confirmed via FAKE_ACP_RECORD) ──────────
    // `provider_switch.relay`: verified by `run_acp_column` (this file) and by
    // `a_conversation_moves_from_*_to_acp_*` in agent_e2e_tests.rs.
    // `enrichment`: moved to not_measured — ACP uses its own system/prompt path (ISO4).
    // ── ACP — not_measured (fake_acp transcript-based, specific scenario not yet adapted) ──
    (Engine::Acp, "compaction", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "message_queue", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "auto_continue", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "nats.send", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "nats.interrupt", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "nats.permission_response", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "nats.cancel_tools", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "resume", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "set_model", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "cancel_tools", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "permissions.once", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "permissions.session", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "permissions.always", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "tool_timing", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "po_tools", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "tool_search", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "tool_result.huge", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "refs", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "session_record", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "background_tasks", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "cancel_task", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "images", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "subagents", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "subagents.background", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "subagents.types", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "system_init.degraded", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    (Engine::Acp, "session_limits", Expect::NotMeasured { why: ACP_TRANSCRIPT }),
    // ── opencode — all not_measured (no fake_opencode) ───────────────────
];

/// Why `nexus_tools` is not measured on Claude Code, whatever its engine.
const CLI_TOOLS: &str = "Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude \
                         rejoue un transcript et n'exécute aucun outil";

/// Why `tool_search` is not measured on Claude Code, whatever its engine.
const CLI_TOOL_SEARCH: &str = "les outils différés et ToolSearch sont internes au CLI Claude : la \
                               liste d'outils qu'il envoie au modèle n'est pas visible du \
                               harnais, et fake_claude rejoue un transcript";

/// Why `anthropic_messages` is not measured on Claude Code, whatever its engine.
const CLI_PROMPT_CACHE: &str =
    "le CLI Claude parle lui-même l'API Messages et pose ses propres points \
                                de cache : fake_claude rejoue un transcript, la requête au modèle \
                                n'est pas visible du harnais";

/// Why all Codex hooks are a harness gap (A40).
const CODEX_HOOKS: &str = "Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; \
     `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }`";

/// Why Codex background tasks / cancel_task are a harness gap.
const CODEX_NO_BG: &str = "`tool_cancel` et `background_tasks` ne sont pas supportés par Codex ; \
     `interrupt` est le seul mécanisme de coupe-circuit";

/// Why a Codex function is not_measured: fake_codex needs a per-function JSONL transcript.
/// Transcripts for these functions have not yet been written (manques à remonter nexus, ISO4).
const CODEX_TRANSCRIPT: &str =
    "fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; \
     cette fonction exige un transcript dédié non encore écrit (ISO4)";

/// Why all ACP hooks are a harness gap.
const ACP_HOOKS: &str =
    "L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; \
     `SessionSpec::hooks` est ignoré";

/// Why an ACP function is not_measured: fake_acp transcript not yet adapted to full scenario.
const ACP_TRANSCRIPT: &str = "fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; \
     le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4)";

/// Why every function is `not_measured` on opencode (no fake_opencode).
const OPENCODE_NOT_MEASURED: &str =
    "fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans \
     binaire faux déterministe (YARDSTICK, tâche 87eb345e)";

/// The verdicts of one engine, with what was seen (diagnostics, not in the table).
#[derive(Default)]
struct Measures {
    rows: Vec<(&'static str, Verdict, String)>,
}

impl Measures {
    fn set(&mut self, function: &'static str, verdict: Verdict, seen: impl Into<String>) {
        assert!(
            FUNCTIONS.iter().any(|(f, _)| *f == function),
            "{function} is not a function of the matrix"
        );
        assert!(
            !self.rows.iter().any(|(f, _, _)| *f == function),
            "{function} measured twice"
        );
        let seen = seen.into();
        eprintln!("parity measured {function}: {verdict:?} — {seen}");
        self.rows.push((function, verdict, seen));
    }

    /// `Ok` when `ok`, else a gap of `cause`; `seen` says what was observed.
    fn check(&mut self, function: &'static str, ok: bool, cause: Cause, seen: impl Into<String>) {
        self.set(
            function,
            if ok { Verdict::Ok } else { Verdict::Gap(cause) },
            seen,
        );
    }

    fn get(&self, function: &str) -> Option<(Verdict, &str)> {
        self.rows
            .iter()
            .find(|(f, _, _)| *f == function)
            .map(|(_, v, s)| (*v, s.as_str()))
    }
}

fn expected(engine: Engine, function: &str) -> Option<Expect> {
    // Explicit overrides (EXPECTED const) take priority.
    if let Some(x) = EXPECTED
        .iter()
        .find(|(e, f, _)| *e == engine && *f == function)
        .map(|(_, _, x)| *x)
    {
        return Some(x);
    }
    // opencode: no fake_opencode exists yet — every function is not_measured by default.
    // Codex and ACP: all per-function verdicts are in EXPECTED (see entries above).
    match engine {
        Engine::OpenCode => Some(Expect::NotMeasured {
            why: OPENCODE_NOT_MEASURED,
        }),
        Engine::Codex
        | Engine::Acp
        | Engine::ClaudeCode
        | Engine::ClaudeCodeAgent
        | Engine::Native => None,
    }
}

/// What is wrong with the measures against [`EXPECTED`]: empty when the table is
/// true. Pure, so the two ways it turns red are tested without the fakes.
fn audit(run: &Run<'_>, expected: &dyn Fn(Engine, &str) -> Option<Expect>) -> Vec<String> {
    let mut problems = Vec::new();
    for (function, _) in FUNCTIONS {
        for &(engine, m) in run {
            let Some((verdict, seen)) = m.get(function) else {
                problems.push(format!(
                    "{} / {function}: not measured at all",
                    engine.label()
                ));
                continue;
            };
            match (verdict, expected(engine, function)) {
                (Verdict::Ok, None) => {}
                (Verdict::Ok, Some(Expect::Gap { task, .. })) => problems.push(format!(
                    "{} / {function}: declared gap ({task}) is no longer measured — it was fixed: \
                     remove it from EXPECTED and regenerate the table",
                    engine.label()
                )),
                (Verdict::Ok, Some(Expect::NotMeasured { .. })) => problems.push(format!(
                    "{} / {function}: declared not_measured but measured ok: update EXPECTED",
                    engine.label()
                )),
                (Verdict::Gap(cause), None) => problems.push(format!(
                    "{} / {function}: UNDECLARED gap ({}) — seen: {seen}",
                    engine.label(),
                    cause.label()
                )),
                (
                    Verdict::Gap(cause),
                    Some(Expect::Gap {
                        cause: declared,
                        task,
                        ..
                    }),
                ) => {
                    if cause != declared {
                        problems.push(format!(
                            "{} / {function}: gap of cause {} measured, {} declared ({task})",
                            engine.label(),
                            cause.label(),
                            declared.label()
                        ));
                    }
                }
                (Verdict::Gap(cause), Some(Expect::NotMeasured { .. })) => problems.push(format!(
                    "{} / {function}: declared not_measured but a {} gap was measured — seen: \
                     {seen}",
                    engine.label(),
                    cause.label()
                )),
                (Verdict::NotMeasured, Some(Expect::NotMeasured { .. })) => {}
                (Verdict::NotMeasured, _) => problems.push(format!(
                    "{} / {function}: not measured, and not declared so",
                    engine.label()
                )),
                (Verdict::Declared(cause), Some(Expect::Gap { cause: declared, .. })) => {
                    if cause != declared {
                        problems.push(format!(
                            "{} / {function}: declared cause {} but EXPECTED says {} — update EXPECTED",
                            engine.label(),
                            cause.label(),
                            declared.label()
                        ));
                    }
                }
                (Verdict::Declared(_), None) => problems.push(format!(
                    "{} / {function}: declared but EXPECTED says ok — use Verdict::Ok or add to EXPECTED",
                    engine.label()
                )),
                (Verdict::Declared(_), Some(Expect::NotMeasured { .. })) => problems.push(format!(
                    "{} / {function}: declared gap but EXPECTED says not_measured — use Verdict::NotMeasured",
                    engine.label()
                )),
            }
        }
    }
    problems
}

fn cell(engine: Engine, function: &str, m: &Measures) -> String {
    match (m.get(function).map(|(v, _)| v), expected(engine, function)) {
        (Some(Verdict::Ok), _) => "ok".to_string(),
        (Some(Verdict::Gap(_)), Some(Expect::Gap { cause, task, .. })) => {
            format!("gap ({}, {task})", cause.label())
        }
        (Some(Verdict::Gap(cause)), _) => format!("gap ({}, NON DÉCLARÉ)", cause.label()),
        (Some(Verdict::NotMeasured), _) => "not_measured".to_string(),
        (Some(Verdict::Declared(_)), Some(Expect::Gap { cause, task, .. })) => {
            format!("declared ({}, {task})", cause.label())
        }
        (Some(Verdict::Declared(cause)), _) => format!("declared ({}, ?)", cause.label()),
        (None, _) => "?".to_string(),
    }
}

const MATRIX_PATH: &str = "docs/parity/engines.md";
const UPDATE_ENV: &str = "UPDATE_PARITY_MATRIX";
const UPDATE_COMMAND: &str =
    "UPDATE_PARITY_MATRIX=1 cargo test --lib chat::engine_parity_tests::the_same_scenario";

/// `docs/parity/engines.md`, from the verdicts and the declarations only (never
/// from what varies between runs: ids, timings, paths).
fn render(run: &Run<'_>) -> String {
    let mut out = String::new();
    out.push_str("# Parité des moteurs de chat — matrice MESURÉE\n\n");
    out.push_str(&format!(
        "<!-- Généré par src/chat/engine_parity_tests.rs : ne pas éditer à la main.\n     \
         {UPDATE_COMMAND} -->\n\n"
    ));
    out.push_str(
        "Un seul scénario, joué sur les moteurs mesurables, même assertion par fonction :\n\n\
         - **Claude Code** : moteur historique (`CHAT_PROVIDER_PATH=legacy`) sur \
         `fake_claude`, le faux CLI du dépôt nexus ;\n\
         - **Claude Code (moteur agent)** : le même CLI et le même transcript, piloté par le \
         moteur agent à travers la façade Claude Code de nexus (`CHAT_PROVIDER_PATH=agent`) ;\n\
         - **natif** : moteur agent, instance OpenAI-compatible sur `fake_openai`, outils \
         project-orchestrator sur `fake_mcp`, outils de base sur le vrai `nexus-tools` ;\n\
         - **Codex** : harnais Codex — `fake_codex` présent dans `nexus/src/bin/fake_codex.rs`, \
         replay de transcripts JSONL ; verdicts par fonction déclarés dans `EXPECTED` (ISO4) ;\n\
         - **opencode** : harnais opencode — `fake_opencode` absent, toutes les fonctions \
         `not_measured` (YARDSTICK, tâche 87eb345e) ;\n\
         - **ACP** : harnais ACP — `fake_acp` présent dans `nexus/src/bin/fake_acp.rs`, \
         replay de transcripts JSONL ; verdicts par fonction déclarés dans `EXPECTED` (ISO4).\n\n\
         `ok` : mesuré et conforme. `gap (cause, tâche)` : mesuré absent, cause `harnais` \
         (le backend ou le harnais nexus ne le fait pas) ou `modèle` (le modèle ne le peut pas), \
         fermé par la tâche PO nommée (plan 5ad54c48 « Parité totale », ou e87abcc2 « Claude Code devient optionnel » pour une tâche M…). `declared (cause, tâche)` : écart déclaré \
         dans EXPECTED mais le scénario n'a PAS tourné sur ce moteur (pas de transcript dédié) ; \
         distinct de `gap` (mesuré) et de `not_measured` (impossible à exercer). `not_measured` : \
         les faux binaires ne permettent pas de l'exercer sur ce moteur (raison ci-dessous). Un \
         écart non déclaré, ou un écart déclaré qui disparaît, fait échouer le test.\n\n",
    );
    let labels: Vec<&str> = run.iter().map(|(engine, _)| engine.label()).collect();
    out.push_str(&format!(
        "| Fonction | Ce qui est vérifié | {} |\n",
        labels.join(" | ")
    ));
    out.push_str(&format!("|---|---|{}\n", "---|".repeat(run.len())));
    for (function, what) in FUNCTIONS {
        let cells: Vec<String> = run
            .iter()
            .map(|&(engine, m)| cell(engine, function, m))
            .collect();
        out.push_str(&format!(
            "| `{function}` | {what} | {} |\n",
            cells.join(" | ")
        ));
    }
    let ok = |m: &Measures| m.rows.iter().filter(|(_, v, _)| *v == Verdict::Ok).count();
    let totals: Vec<String> = run
        .iter()
        .map(|&(engine, m)| format!("{} {}/{}", engine.label(), ok(m), FUNCTIONS.len()))
        .collect();
    out.push_str(&format!("\n`ok` : {}.\n", totals.join(", ")));
    out.push_str("\n## Écarts déclarés\n\n");
    out.push_str("| Moteur | Fonction | Cause | Tâche | Ce qui manque |\n|---|---|---|---|---|\n");
    for (engine, function, x) in declared_in_table_order() {
        if let Expect::Gap { cause, task, why } = x {
            out.push_str(&format!(
                "| {} | `{function}` | {} | {task} | {why} |\n",
                engine.label(),
                cause.label()
            ));
        }
    }
    out.push_str("\n## Non mesuré\n\n");
    out.push_str("| Moteur | Fonction | Pourquoi |\n|---|---|---|\n");
    for (engine, function, x) in declared_in_table_order() {
        if let Expect::NotMeasured { why } = x {
            out.push_str(&format!("| {} | `{function}` | {why} |\n", engine.label()));
        }
    }
    out.push_str("\n## À re-mesurer à chaque montée du CLI Claude Code\n\n");
    out.push_str(
        "`fake_claude` rejoue ce que le vrai CLI a fait à une version donnée : rien ici ne \
         verrait un CLI qui changerait d'avis sans le dire. À chaque montée de version du CLI \
         Claude Code, re-mesurer sur le vrai CLI :\n\n",
    );
    out.push_str("| Comportement | Mesuré sur | Comment | Note PO |\n|---|---|---|---|\n");
    for (what, version, how, note) in RE_MEASURE {
        out.push_str(&format!("| {what} | {version} | {how} | {note} |\n"));
    }
    out
}

/// What the parity takes from the real CLI without measuring it here (the fakes replay
/// it): to re-check on each Claude Code CLI bump. (behaviour, measured on, how, PO note)
const RE_MEASURE: &[(&str, &str, &str, &str)] = &[
    (
        "re-check PreCompact systemMessage : le CLI ajoute le `systemMessage` d'une réponse au \
         hook PreCompact aux instructions du résumé (et l'affiche à l'utilisateur dans la ligne \
         « Compacted… »), ignore `reason`, rejette `hookSpecificOutput.additionalContext`",
        "Claude Code 2.1.295",
        "`docs/parity/m2c-precompact/` (README)",
        "172c164e (tâche M2c 79cc48df)",
    ),
    (
        "re-check résultat d'outil de 1 Mo : le CLI écrit le résultat entier dans un fichier de \
         la session (`tool-results/`) et ne rend au modèle qu'un avis d'environ 1,6 Ko qui nomme \
         ce fichier, sans aperçu du contenu ni `is_error` ; le tour se clôt normalement \
         (`tool_result.huge`, rejoué par fake_claude)",
        "Claude Code 2.1.296 (18/10/2026)",
        "`docs/parity/gros1b-big-tool-result/` (README)",
        "tâche GROS-1b 4b687177",
    ),
    (
        "re-check reprise en lieu neutre : une session ouverte dans le répertoire neutre de l'hôte (`chat-neutral/<id>`), ce répertoire supprimé puis recréé au même chemin, se reprend avec `--resume` sans perte (en 2.1.296 la reprise réussit aussi depuis un AUTRE chemin) ; aucun `CLAUDE.md` ni `.claude/` dans les ancêtres hors `~/.claude`",
        "Claude Code 2.1.296 (18/10/2026)",
        "`docs/parity/t1bis-neutral-place/` (README)",
        "tâche T1-bis 6dae6085",
    ),
];

/// The size of the tool result of the row `tool_result.huge`: about a megabyte.
const BIG_RESULT_CHARS: u64 = 1_000_000;

/// What may reach the thread (and the model) for a result of [`BIG_RESULT_CHARS`]: the two
/// shapes measured are ~1.6 KB (the CLI's notice) and ~2.2 KB (the native
/// `<persisted-output>`: a 2 KB preview and the path). Anything near the megabyte fails.
const BIG_SEEN_BOUND: usize = 8 * 1024;

/// What Claude Code 2.1.296 handed the model for a 1,000,000-character MCP result, VERBATIM
/// (`claude -p`, claude-haiku-5-5, 18/10/2026, `docs/parity/gros1b-big-tool-result/`), `{path}`
/// standing for the file of the session where the CLI wrote the whole result. No `is_error`.
/// `fake_claude` replays it: the CLI runs no tool here.
const CLI_BIG_NOTICE: &str = "Error: result (1,000,000 characters across 1 line) exceeds maximum allowed tokens. Output has been saved to {path}.\n\
Format: Plain text\n\
Search within the file for specific content, and use jq if the content is JSON.\n\
REQUIREMENTS FOR SUMMARIZATION/ANALYSIS/REVIEW:\n\
- You MUST read the content from the file at {path} in sequential chunks until 100% of the content has been read.\n\
- Note: this file's lines are too long for Read's offset/limit chunking. If a shell tool is available, slice by character range (e.g. python read()[A:B], dd, or cut -c) instead.\n\
- If you receive truncation warnings when reading the file, reduce the chunk size until you have read 100% of the content without truncation.\n\
- Before producing ANY summary or analysis, you MUST explicitly describe what portion of the content you have read. ***If you did not read the entire content, you MUST explicitly state this.***\n\
- If after a few attempts you cannot read the file (file not found, lines too long for Read's offset/limit, no shell access), STOP retrying. Summarize what you were able to read, explicitly state which portion you could not read and why, and proceed.\n";

/// [`CLI_BIG_NOTICE`] for a session whose CLI keeps its tool results under `cwd`.
fn cli_big_notice(cwd: &str) -> String {
    CLI_BIG_NOTICE.replace(
        "{path}",
        &format!(
            "{}/.claude-tool-results/mcp-project-orchestrator-big-1.txt",
            cwd.trim_end_matches('/')
        ),
    )
}

/// The verdict of `tool_result.huge` from what the thread showed of the big call `id`, whether
/// its turn and the next one answered, and `extra` (the native disk check; `true` on Claude Code).
fn big_verdict(
    events: &[ChatEvent],
    id: &str,
    answered: bool,
    next: bool,
    extra: (bool, String),
) -> (bool, String) {
    let shown = events.iter().find_map(|e| match e {
        ChatEvent::ToolResult {
            id: r,
            result,
            is_error,
            ..
        } if r == id => Some((result.to_string().len(), *is_error)),
        _ => None,
    });
    let bounded = shown.is_some_and(|(len, _)| len < BIG_SEEN_BOUND);
    (
        bounded && answered && next && extra.0,
        format!(
            "résultat sur le fil (octets, is_error)={shown:?} (borne {BIG_SEEN_BOUND}), tour \
             clos={answered}, tour suivant={next}, {}",
            extra.1
        ),
    )
}

/// [`EXPECTED`] by engine, then in the order of the table.
fn declared_in_table_order() -> Vec<(Engine, &'static str, Expect)> {
    let mut out = Vec::new();
    for engine in ENGINES {
        for (function, _) in FUNCTIONS {
            if let Some(x) = expected(engine, function) {
                out.push((engine, *function, x));
            }
        }
    }
    out
}

/// The drift gate and the audit, once the scenario ran on every engine.
fn gate(run: &Run<'_>) {
    for &(engine, m) in run {
        for (function, verdict, seen) in &m.rows {
            eprintln!(
                "parity {} / {function}: {verdict:?} — {seen}",
                engine.label()
            );
        }
    }
    let table = render(run);
    let problems = audit(run, &expected);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MATRIX_PATH);
    if std::env::var(UPDATE_ENV).as_deref() == Ok("1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &table).unwrap();
    }
    assert!(
        problems.is_empty(),
        "the parity matrix is not what EXPECTED declares:\n  {}\n\nWhat this run measured:\n\n{table}",
        problems.join("\n  ")
    );
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == table,
        "{MATRIX_PATH} is out of date. What this run measured:\n\n{table}\n\
         Regenerate it and commit the result in the same commit:\n    {UPDATE_COMMAND}"
    );
}

// ============================================================================
// The system_init audit (closed list of the limits of a MODEL)
// ============================================================================

/// What a `system_init` announces as missing that is NOT a limit of the model:
/// empty when the announcement is honest. The closed list: `images` when the
/// model has no vision, and `compaction` when its `context_window` is unknown (the
/// native harness compacts against the window: `agent_runtime::kind_degraded`).
///
/// `nexus_tools` (a native session whose `nexus-tools` executable is not found,
/// `manager::lacks_nexus_tools`) is a limit of the INSTALLATION, neither of the
/// model nor of the harness. It is deliberately NOT in the list: this scenario
/// always runs with the real `nexus-tools` (`NEXUS_FAKES_DIR`, a missing binary
/// fails the test), so announcing it here means the harness failed to attach a
/// binary it had, which must fail rather than be accepted as an installation cause.
fn not_model_limits(init: &ChatEvent) -> Vec<String> {
    let ChatEvent::SystemInit {
        degraded_features,
        capabilities,
        ..
    } = init
    else {
        return vec!["not a system_init".to_string()];
    };
    let vision = capabilities
        .as_ref()
        .and_then(|c| c.get("images"))
        .and_then(Value::as_bool);
    let window_unknown = capabilities
        .as_ref()
        .is_some_and(|c| c.get("context_window").is_none_or(Value::is_null));
    degraded_features
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|f| !(f == "images" && vision == Some(false)))
        .filter(|f| !(f == "compaction" && window_unknown))
        .collect()
}

// ============================================================================
// Shared scenario data
// ============================================================================

const PLAN_TITLE: &str = "Port the enrichment";
const TASK_TITLE: &str = "Measure the parity";
const WAIT: Duration = Duration::from_secs(20);

/// Deadline for a plain engine Result on a native/ACP/Codex column. Wider than
/// `WAIT` for the same reason as `INTERRUPT_TURN_END_WAIT`: a coverage-
/// instrumented binary is several times slower, and a 20 s deadline flakes
/// under `cargo llvm-cov` (the Code Coverage job) even though it passes in the
/// plain Unit Tests job.
const ENGINE_RESULT_WAIT: Duration = Duration::from_secs(90);

/// The end of a turn AFTER an interruption gets its own, wider deadline than
/// `WAIT`. The interruption travels three hops (the other instance publishing
/// over NATS, the CLI's `interrupt` control_request, the CLI closing the
/// aborted turn) and every hop is slow when the binary is instrumented: under
/// the Code Coverage job (llvm-cov, ~976 s for the lib alone) the closing turn
/// passed the global 20 s deadline, and the `nats.interrupt` cell fell on a
/// verdict that was in fact fine — same head, red then green on a rerun (task
/// `FLAKY-PARITE`). `WAIT` itself is NOT widened: the other cells measured
/// against it are stable, and a wider global deadline would only hide their own
/// lateness.
const INTERRUPT_TURN_END_WAIT: Duration = Duration::from_secs(60);

/// The 30-line Grep result the after_tool hook redirects (`post_tool_hook`).
fn noisy_grep() -> String {
    (0..30)
        .map(|i| format!("src/parity.rs:{i}: build_agent_spec(input)\n"))
        .collect()
}

/// A 1×1 PNG.
const PIXEL: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
    0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64, 0x60, 0xf8, 0x5f,
    0x0f, 0x00, 0x02, 0x87, 0x01, 0x80, 0xeb, 0x47, 0xba, 0x92, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

fn pixel_base64() -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(PIXEL)
}

/// The keys of one engine: the same scenario, words the other engine never sees
/// (the native model receives the Claude Code conversation in the relay: a key
/// of one engine must never answer a request of the other).
#[derive(Clone, Copy)]
struct Keys(&'static str);

impl Keys {
    fn k(&self, step: &str) -> String {
        format!("{}-{step}", self.0)
    }
}

/// What both engines share: a project on disk (a file for the tools, 30 noisy
/// lines for the redirect hook), its plan (enrichment) and a task (references),
/// a NATS broker, another instance on it, the document store of attachments.
struct Stage {
    graph: Arc<MockGraphStore>,
    project: crate::neo4j::models::ProjectNode,
    task_id: Uuid,
    broker: TestBroker,
    other: NatsEmitter,
    store: DocumentStore,
    dir: tempfile::TempDir,
    /// The data directory: the documents' blobs, and the native transcripts (P14).
    data: tempfile::TempDir,
    /// The vault of every backend of the stage: the key of the `anthropic` instance
    /// (CTX5), granted to that instance only.
    vault: Arc<crate::vault::VaultService>,
}

impl Stage {
    async fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("parity.txt"), "PARITY-FILE BEFORE-EDIT\n").unwrap();
        std::fs::write(dir.path().join("noisy.rs"), noisy_grep()).unwrap();
        let graph = Arc::new(MockGraphStore::new());
        let mut project = crate::test_helpers::test_project();
        project.slug = "proj".into();
        project.root_path = dir.path().display().to_string();
        graph.create_project(&project).await.unwrap();
        crate::skills::project_resolver::seed_resolve_cache_for_tests(std::slice::from_ref(
            &project,
        ));
        let mut plan = crate::test_helpers::test_plan();
        plan.title = PLAN_TITLE.into();
        plan.status = crate::neo4j::models::PlanStatus::InProgress;
        plan.project_id = Some(project.id);
        graph.create_plan(&plan).await.unwrap();
        graph
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();
        // Completed: a pending task would make the Claude Code engine remind the model of
        // its objectives after a turn without tool use (a turn of its own, unscripted).
        let mut task = crate::test_helpers::test_task_titled(TASK_TITLE);
        task.status = crate::neo4j::models::TaskStatus::Completed;
        graph.create_task(plan.id, &task).await.unwrap();
        let broker = TestBroker::start().await;
        let other = NatsEmitter::new(broker.client().await, "events");
        let blobs = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(blobs.path());
        let vault = anthropic_vault().await;
        Self {
            graph,
            project,
            task_id: task.id,
            broker,
            other,
            store,
            dir,
            data: blobs,
            vault,
        }
    }

    fn cwd(&self) -> String {
        self.project.root_path.clone()
    }

    fn refs_message(&self, text: &str) -> String {
        crate::refs::block::encode(text, &[EntityRef::new(RefKind::Task, self.task_id)])
    }

    /// The stored form of a message `text` that attaches the pixel (uploaded as a
    /// document), as the API layer hands it to the manager.
    async fn image_message(&self, text: &str) -> String {
        let id = Uuid::new_v4();
        let sha256 = self.store.put(PIXEL).unwrap();
        self.graph.documents.write().await.insert(
            id,
            crate::neo4j::document::Document {
                id,
                filename: "pixel.png".to_string(),
                format: crate::documents::DocumentFormat::Binary,
                sha256,
                size_bytes: PIXEL.len() as u64,
                page_count: 0,
                chunk_count: 0,
                warnings: vec![],
                created_at: Utc::now(),
                project_id: None,
                session_id: None,
                extracted: false,
                mime_type: Some("image/png".to_string()),
            },
        );
        let dyn_graph: Arc<dyn GraphStore> = self.graph.clone();
        super::message_attachments::compose(&dyn_graph, text, &[id])
            .await
            .unwrap()
    }

    /// A backend process: a manager on the stage's graph and NATS, Claude Code on
    /// the CLI `cli`, the native instance tapped (its hooks recorded in `answers`).
    /// Called again to "restart" the backend: the graph is all that survives.
    async fn backend(
        &self,
        cli: &FakeClaude,
        answers: &Arc<Answers>,
        path: ProviderPath,
    ) -> Arc<ChatManager> {
        self.backend_with(cli, answers, answers, path).await
    }

    /// [`Self::backend`] with the hooks of the native instance recorded in `answers` and those
    /// of Claude Code (either engine) in `claude_answers`: a column never reads the calls of
    /// the other.
    async fn backend_with(
        &self,
        cli: &FakeClaude,
        answers: &Arc<Answers>,
        claude_answers: &Arc<Answers>,
        path: ProviderPath,
    ) -> Arc<ChatManager> {
        let config = ChatConfig {
            provider_path: path,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: Some(fake_bin("nexus-tools")),
            nexus_browser_path: None,
            obscura_path: None,
            jwt_secret: Some("parity-secret-parity-secret-parity".to_string()),
            max_sessions: 10,
            // Three model calls per turn: the turn that makes three tool calls stops
            // on its limit (auto-continue); every other turn answers within it.
            max_turns: 3,
            claude_cli_path: Some(cli.path()),
            // The lasting approvals (`always`, P11b) of the data directory: they outlive a
            // restart of the backend, as the graph does.
            lasting_rules_dir: Some(self.data.path().join("permission-rules")),
            native_turn_timeout_secs: None,
            native_session_max_tokens: None,
            ..Default::default()
        };
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = self.graph.clone();
        let owner = Arc::new(NatsEmitter::new(self.broker.client().await, "events"));
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_nats(owner)
            .with_refs_v1(true)
            .with_document_store(self.store.clone())
            .with_vault(Arc::clone(&self.vault))
            .with_native_transcripts(self.data.path().join("native-transcripts"));
        manager.update_claude_cli_path(Some(cli.path())).await;
        // As the server does at startup (CTX2): the deferred tools come from the stored
        // setting, read BEFORE the native provider is built. The scenario stores
        // `readonly` as deferred (see the native part), so the row `tool_search` can be
        // measured; every other PO tool of the fakes stays offered as before.
        manager.load_deferred_tools().await;
        let real = manager
            .provider_for("local")
            .await
            .unwrap_or_else(|e| panic!("the native instance: {e:#}"));
        let claude = super::agent_runtime::BuiltinProviders::new(Some(cli.path()))
            .get(super::provider::resolver::CLAUDE_CODE)
            .expect("the built-in Claude Code");
        let manager = Arc::new(manager.with_provider_source(Arc::new(Taps {
            native: Tap {
                inner: real,
                answers: Arc::clone(answers),
            },
            claude: Tap {
                inner: claude,
                answers: Arc::clone(claude_answers),
            },
        })));
        // The native sessions run their `Agent` calls as child sessions (P18).
        manager.enable_subagents();
        manager
    }

    /// Asks the session, from the other instance, until its owner answers. A
    /// permission answer is asked ONCE: a retry after a timeout would deliver it twice.
    async fn rpc(&self, sid: &str, message: &str, kind: &str) -> Option<bool> {
        let attempts = if kind == "control_response" { 1 } else { 10 };
        for _ in 0..attempts {
            if let Some(answer) = self.other.request_send_message(sid, message, kind).await {
                return Some(answer.success);
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        None
    }

    async fn session_node(&self, sid: &str) -> crate::neo4j::models::ChatSessionNode {
        self.graph
            .get_chat_session(Uuid::parse_str(sid).unwrap())
            .await
            .unwrap()
            .expect("the session record")
    }
}

/// The persisted `tool_timing` of the call `id`, which waited for a permission that
/// was allowed: taken up, then asked, then answered (where it starts to run), then
/// ended, in that order; stored once, after the call's result.
fn timing_verdict(persisted: &[ChatEvent], id: &str) -> (bool, String) {
    let is_timing = |e: &ChatEvent| matches!(e, ChatEvent::ToolTiming { id: t, .. } if t == id);
    let Some(at) = persisted.iter().position(&is_timing) else {
        return (false, format!("aucun tool_timing persisté pour {id}"));
    };
    let ChatEvent::ToolTiming {
        started_at,
        permission_requested_at,
        permission_resolved_at,
        permission_outcome,
        run_started_at,
        ended_at,
        ..
    } = &persisted[at]
    else {
        unreachable!("found as a tool_timing")
    };
    let after_result = persisted[..at]
        .iter()
        .any(|e| matches!(e, ChatEvent::ToolResult { id: r, .. } if r == id));
    let once = persisted.iter().filter(|e| is_timing(e)).count() == 1;
    let seen = format!(
        "pris={started_at:?} demandé={permission_requested_at:?} répondu={permission_resolved_at:?} \
         issue={permission_outcome:?} exécuté={run_started_at:?} fin={ended_at} \
         après le résultat={after_result} une fois={once}"
    );
    let ordered = match (started_at, permission_requested_at, permission_resolved_at) {
        (Some(s), Some(q), Some(r)) => s <= q && q <= r && r <= ended_at,
        _ => false,
    };
    let ok = ordered
        && run_started_at == permission_resolved_at
        && permission_outcome.as_deref() == Some("allowed")
        && after_result
        && once;
    (ok, seen)
}

/// `ok` when the record of a session carries what its conversation produced:
/// the record of a session after `sent` user messages: each counted, a cost (the
/// total the provider reported: 0 for a free endpoint, a real figure; an unknown
/// price leaves none, never an invented 0) and a title.
fn record_verdict(node: &crate::neo4j::models::ChatSessionNode, sent: i64) -> (bool, String) {
    let ok = node.message_count >= sent
        && node.total_cost_usd.is_some_and(|c| c >= 0.0)
        && node.title.as_deref().is_some_and(|t| !t.is_empty());
    (
        ok,
        format!(
            "message_count={} (≥ {sent} attendu), total_cost_usd={:?}, title={:?}",
            node.message_count, node.total_cost_usd, node.title
        ),
    )
}

// ============================================================================
// Watching a session: every live event, plus what was persisted
// ============================================================================

struct Watch {
    live: Arc<StdMutex<Vec<ChatEvent>>>,
    graph: Arc<MockGraphStore>,
    sid: String,
}

impl Watch {
    fn start(
        graph: Arc<MockGraphStore>,
        sid: &str,
        mut rx: broadcast::Receiver<ChatEvent>,
    ) -> Self {
        let live = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&live);
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => sink.lock().unwrap().push(event),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
        Self {
            live,
            graph,
            sid: sid.to_string(),
        }
    }

    async fn persisted(&self) -> Vec<ChatEvent> {
        let Ok(uuid) = Uuid::parse_str(&self.sid) else {
            return Vec::new();
        };
        self.graph
            .get_chat_events(uuid, 0, 5_000)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|r| serde_json::from_str(&r.data).ok())
            .collect()
    }

    /// Live events, then the persisted ones (an event emitted before the
    /// subscription is still found there).
    async fn all(&self) -> Vec<ChatEvent> {
        let mut out = self.live.lock().unwrap().clone();
        out.extend(self.persisted().await);
        out
    }

    async fn wait(&self, within: Duration, pred: impl Fn(&ChatEvent) -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            if self.all().await.iter().any(&pred) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn any(&self, pred: impl Fn(&ChatEvent) -> bool) -> bool {
        self.all().await.iter().any(pred)
    }

    fn turn_ends(&self) -> usize {
        self.live
            .lock()
            .unwrap()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    ChatEvent::StreamingStatus {
                        is_streaming: false
                    }
                )
            })
            .count()
    }

    /// Waits until more than `n` turns ended (live `streaming_status: false`).
    async fn turn_end_after(&self, n: usize) -> bool {
        self.turn_end_after_within(n, WAIT).await
    }

    /// `turn_end_after` with an explicit deadline, for the turns that close
    /// legitimately slower than the global `WAIT` (see
    /// [`INTERRUPT_TURN_END_WAIT`]).
    async fn turn_end_after_within(&self, n: usize, within: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            if self.turn_ends() > n {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// The end of the turn, then a pause: what follows a turn (its persistence,
    /// a queued message) is done before the next step of the scenario.
    async fn settle(&self, n: usize, manager: &ChatManager) {
        self.turn_end_after(n).await;
        // The historical engine keeps `is_streaming` while it closes the turn (its
        // post-stream work): a message sent then is queued and interrupts.
        let deadline = tokio::time::Instant::now() + WAIT;
        while manager.is_session_streaming(&self.sid).await
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    async fn system_init(&self) -> Option<ChatEvent> {
        self.all()
            .await
            .into_iter()
            .find(|e| matches!(e, ChatEvent::SystemInit { .. }))
    }

    async fn said(&self, within: Duration, text: &str) -> bool {
        self.wait(
            within,
            |e| matches!(e, ChatEvent::AssistantText { content, .. } if content.contains(text)),
        )
        .await
    }

    async fn permission_ids(&self, input_contains: &str) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for e in self.all().await {
            if let ChatEvent::PermissionRequest { id, input, .. } = e {
                if input.to_string().contains(input_contains) && !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    async fn permission_id(&self, within: Duration, input_contains: &str) -> Option<String> {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            if let Some(id) = self.permission_ids(input_contains).await.into_iter().next() {
                return Some(id);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

// ============================================================================
// Claude Code on the historical engine: fake_claude behind a wrapper
// ============================================================================

/// The fake CLI, one transcript per spawn. `--version` probes do not consume a
/// transcript; every other spawn replays `transcript.<n>.jsonl`, records its
/// stdin in `stdin.<n>.jsonl` and its invocation in `args.<n>.json`.
struct FakeClaude {
    dir: tempfile::TempDir,
}

/// On a failure, what every spawn read: the transcript stops where the backend
/// did not write what it should have.
impl Drop for FakeClaude {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        for n in 0..3 {
            for line in self.stdin_lines(n) {
                let chars: Vec<char> = line.chars().collect();
                let head: String = chars.iter().take(120).collect();
                let tail: String = chars[chars.len().saturating_sub(160)..].iter().collect();
                eprintln!("fake_claude stdin.{n}: {head} … {tail}");
            }
        }
    }
}

impl FakeClaude {
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let fake = fake_bin("fake_claude");
        let wrapper = dir.path().join("claude");
        let script = format!(
            "#!/bin/sh\n\
             for a in \"$@\"; do if [ \"$a\" = \"--version\" ]; then exec \"{fake}\" \"$@\"; fi; done\n\
             n=$(cat \"{dir}/count\" 2>/dev/null || echo 0)\n\
             echo $((n+1)) > \"{dir}/count\"\n\
             FAKE_CLAUDE_TRANSCRIPT=\"{dir}/transcript.$n.jsonl\" \
             FAKE_CLAUDE_STDIN_OUT=\"{dir}/stdin.$n.jsonl\" \
             FAKE_CLAUDE_ARGS_OUT=\"{dir}/args.$n.json\" \
             FAKE_CLAUDE_STDIN_TIMEOUT_MS=20000 \
             FAKE_CLAUDE_MAX_RUNTIME_MS=400000 \
             exec \"{fake}\" \"$@\"\n",
            fake = fake.display(),
            dir = dir.path().display()
        );
        std::fs::write(&wrapper, script).unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        Self { dir }
    }

    fn path(&self) -> String {
        self.dir.path().join("claude").display().to_string()
    }

    fn transcript(&self, n: usize, lines: &[Value]) {
        let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(self.dir.path().join(format!("transcript.{n}.jsonl")), body).unwrap();
    }

    /// The lines the fake has read on its stdin so far: only the ones whose `\n` has
    /// landed. The fake appends while the test polls every 50 ms, and a read in the
    /// middle of a write returned a truncated `control_response` (CI, 19/10/2026:
    /// `permissions.once` measured as a gap on a line cut after 52 characters).
    fn stdin_lines(&self, n: usize) -> Vec<String> {
        let text = std::fs::read_to_string(self.dir.path().join(format!("stdin.{n}.jsonl")))
            .unwrap_or_default();
        let complete = text.rfind('\n').map_or("", |i| &text[..i]);
        complete.lines().map(str::to_string).collect()
    }

    fn line(&self, n: usize, needle: &str) -> Option<String> {
        self.stdin_lines(n).into_iter().find(|l| l.contains(needle))
    }

    fn count(&self, n: usize, needle: &str) -> usize {
        self.stdin_lines(n)
            .iter()
            .filter(|l| l.contains(needle))
            .count()
    }

    async fn wait_line(&self, n: usize, needle: &str, within: Duration) -> Option<String> {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            if let Some(line) = self.line(n, needle) {
                return Some(line);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn wait_count(&self, n: usize, needle: &str, more_than: usize, within: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            if self.count(n, needle) > more_than {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn args(&self, n: usize) -> Vec<String> {
        std::fs::read_to_string(self.dir.path().join(format!("args.{n}.json")))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["argv"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(|a| a.as_str().map(str::to_string))
            .collect()
    }
}

// ----- transcript directives (the fake's vocabulary) -----

/// Waits for a stdin line containing `needle`; goes on after 15 s so a write
/// that never comes is MEASURED (the stdin recording says what came), never a hang.
fn await_in(needle: &str) -> Value {
    json!({"op": "await_stdin", "contains": needle, "timeout_ms": 15000, "optional": true})
}

fn emit(v: Value) -> Value {
    json!({"op": "emit_json", "json": v})
}

fn sleep_ms(ms: u64) -> Value {
    json!({"op": "sleep", "ms": ms})
}

fn capture_hooks() -> Value {
    json!({"op": "capture_hooks", "timeout_ms": 15000, "optional": true})
}

fn wait_eof() -> Value {
    json!({"op": "wait_eof", "timeout_ms": 60000, "optional": true})
}

fn spawn_sleeper() -> Value {
    json!({"op": "spawn_child", "program": "sleep", "args": ["120"]})
}

/// The file whose creation lets the background process of TURN-BG start
/// ([`spawn_sleeper_on`]); next to the session's directory, not in it.
fn bg_go(cwd: &str) -> String {
    format!("{}.parity-bg-go", cwd.trim_end_matches('/'))
}

/// A shell that waits for `go` to exist, then runs `sleep 120` as its child: the
/// shell is there from the start, the sleeper is the newcomer the PID claim finds.
fn spawn_sleeper_on(go: &str) -> Value {
    json!({"op": "spawn_child", "program": "sh", "args": [
        "-c", "while [ ! -e \"$0\" ]; do sleep 0.05; done; sleep 120", go]})
}

fn wait_children() -> Value {
    json!({"op": "wait_children_exit", "timeout_ms": 15000, "optional": true})
}

fn init(session_id: &str) -> Value {
    emit(json!({
        "type": "system", "subtype": "init", "session_id": session_id,
        "model": "fake-claude", "cwd": ".", "tools": ["Bash", "Read", "Edit", "Grep"],
        "permissionMode": "default", "apiKeySource": "none",
    }))
}

fn assistant(content: Value) -> Value {
    emit(json!({
        "type": "assistant",
        "message": {"id": "msg_fake", "type": "message", "role": "assistant",
                    "model": "fake-claude", "content": content, "stop_reason": "end_turn"},
    }))
}

fn text(t: &str) -> Value {
    assistant(json!([{"type": "text", "text": t}]))
}

fn cc_tool_use(id: &str, name: &str, input: Value) -> Value {
    assistant(json!([{"type": "tool_use", "id": id, "name": name, "input": input}]))
}

fn cc_tool_result(id: &str, content: &str, is_error: bool) -> Value {
    emit(json!({
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": id, "content": content, "is_error": is_error}
        ]},
        "parent_tool_use_id": null,
    }))
}

/// What a sub-agent says, inside the Task call `parent`.
fn sub_agent_text(parent: &str, t: &str) -> Value {
    emit(json!({
        "type": "assistant",
        "message": {"id": "msg_sub", "type": "message", "role": "assistant",
                    "model": "fake-claude", "content": [{"type": "text", "text": t}],
                    "stop_reason": "end_turn"},
        "parent_tool_use_id": parent,
    }))
}

/// The input of the question both engines ask (Claude Code's AskUserQuestion shape).
fn question_input() -> Value {
    json!({"questions": [{"question": "Which database?", "header": "DB",
        "options": [{"label": "Postgres"}, {"label": "SQLite"}], "multiSelect": false}]})
}

fn result(subtype: &str, t: &str, is_error: bool) -> Value {
    emit(json!({
        "type": "result", "subtype": subtype, "duration_ms": 12, "duration_api_ms": 7,
        "is_error": is_error, "num_turns": 1, "session_id": "cc-sess-1",
        "total_cost_usd": 0.0001, "usage": {"input_tokens": 3, "output_tokens": 5},
        "result": t,
    }))
}

fn result_ok(t: &str) -> Value {
    result("success", t, false)
}

fn permission_request(request_id: &str, tool: &str, input: Value, tool_use_id: &str) -> Value {
    emit(json!({
        "type": "control_request", "request_id": request_id,
        "request": {"subtype": "can_use_tool", "tool_name": tool, "input": input,
                    "tool_use_id": tool_use_id},
    }))
}

fn hook(event: &str, request_id: &str, input: Value, tool_use_id: Option<&str>) -> Value {
    json!({
        "op": "emit_hook", "event": event, "request_id": request_id, "input": input,
        "tool_use_id": tool_use_id, "await_response": true, "timeout_ms": 15000,
        "optional": true,
    })
}

fn hook_input(event: &str, cwd: &str, extra: Value) -> Value {
    let mut input = json!({
        "hook_event_name": event, "session_id": "cc-sess-1",
        "transcript_path": "/tmp/transcript.jsonl", "cwd": cwd,
    });
    for (k, v) in extra.as_object().into_iter().flatten() {
        input[k] = v.clone();
    }
    input
}

/// A plain answer to the turn whose message contains `key`.
fn answer(key: &str, t: &str) -> Vec<Value> {
    vec![await_in(key), text(t), result_ok(t)]
}

/// The first Claude Code process: the whole scenario up to the restart.
fn transcript_main(k: Keys, cwd: &str, path: ProviderPath) -> Vec<Value> {
    let grep_input = json!({"pattern": "build_agent_spec", "path": "."});
    let mut t: Vec<Value> = vec![
        // The SDK registers its hooks at connect; their callback ids are minted at
        // run time: the fake remembers them.
        capture_hooks(),
        // ISO1: SessionStart (contract v7) is NOT emitted by the CLI — the façade
        // calls hooks.session_start() directly (see ClaudeCodeSession::session_start in
        // nexus). Verified through the answers struct in the cc/na checks below.
        //
        // TURN-ONE: a Grep (before_tool hook, then after_tool hook on its noisy
        // result), a compaction announced by the CLI (before_compaction hook), its
        // boundary, then an answer.
        await_in(&k.k("TURN-ONE")),
        // ISO1: UserPromptSubmit fires when the user's input arrives (contract v7).
        hook(
            "UserPromptSubmit",
            "hook-prompt",
            hook_input("UserPromptSubmit", cwd, json!({"prompt": k.k("TURN-ONE")})),
            None,
        ),
        sleep_ms(200),
        init("cc-sess-1"),
        cc_tool_use("t1", "Grep", grep_input.clone()),
        hook(
            "PreToolUse",
            "hook-pre",
            hook_input(
                "PreToolUse",
                cwd,
                json!({"tool_name": "Grep", "tool_input": grep_input}),
            ),
            Some("t1"),
        ),
        cc_tool_result("t1", &noisy_grep(), false),
        hook(
            "PostToolUse",
            "hook-post",
            hook_input(
                "PostToolUse",
                cwd,
                json!({"tool_name": "Grep", "tool_input": grep_input,
                       "tool_response": noisy_grep()}),
            ),
            Some("t1"),
        ),
        hook(
            "PreCompact",
            "hook-compact",
            hook_input("PreCompact", cwd, json!({"trigger": "auto"})),
            None,
        ),
        emit(json!({"type": "system", "subtype": "compact_boundary",
                    "compact_metadata": {"trigger": "auto", "pre_tokens": 1200}})),
        text("answered one"),
        result_ok("answered one"),
    ];
    // After a compaction the engine re-injects the work context: the historical engine in a
    // turn of its own; the agent engine in front of the next user message (TURN-TWO's line
    // carries it, `post_compaction_recovery` of `TurnServices`).
    if path == ProviderPath::Legacy {
        t.extend(vec![
            await_in("Work Already Done"),
            text("noted"),
            result_ok("noted"),
        ]);
    }
    t.extend(vec![
        // TURN-TWO runs; TURN-THREE is sent meanwhile: the engine queues it and
        // interrupts the turn, then plays the queue.
        await_in(&k.k("TURN-TWO")),
        await_in("\"interrupt\""),
        result("error_during_execution", "", true),
    ]);
    // ISO1: TURN-THREE answers with text (no tools); the Stop hook fires before the
    // turn ends (contract v7). On the fake the response is not acted on — the blocking
    // behaviour is tested on the agent engine and the native harness.
    t.extend(vec![
        await_in(&k.k("TURN-THREE")),
        text("answered three"),
        hook(
            "Stop",
            "hook-stop",
            hook_input(
                "Stop",
                cwd,
                json!({"stop_hook_active": false,
                       "last_assistant_message": "answered three"}),
            ),
            None,
        ),
        result_ok("answered three"),
    ]);
    // TURN-FOUR stops on the turn limit; auto-continue sends the continuation.
    t.extend(vec![
        await_in(&k.k("TURN-FOUR")),
        text("partial"),
        result("error_max_turns", "partial", true),
    ]);
    t.extend(answer("Continue where you left off", "continued"));
    // set_model, between two turns: a control request on stdin.
    t.push(await_in("set_model"));
    // TURN-FIVE: a tool that needs a permission (answered once).
    t.extend(vec![
        await_in(&k.k("TURN-FIVE")),
        cc_tool_use("t5", "Bash", json!({"command": "ls"})),
        hook(
            "PreToolUse",
            "hook-t5",
            hook_input(
                "PreToolUse",
                cwd,
                json!({"tool_name": "Bash", "tool_input": {"command": "ls"}}),
            ),
            Some("t5"),
        ),
        permission_request("req-perm", "Bash", json!({"command": "ls"}), "t5"),
        await_in("req-perm"),
        // ISO1: Notification fires when a tool waits for permission (contract v7).
        // Placed AFTER await_in("req-perm"): the Notification hook wait in next_line
        // would consume the permission response otherwise (next_line discards lines
        // that don't match the needle).
        hook(
            "Notification",
            "hook-notify",
            hook_input(
                "Notification",
                cwd,
                json!({"notification_type": "permission_prompt",
                       "message": "Bash needs permission", "title": ""}),
            ),
            None,
        ),
        cc_tool_result("t5", "a.rs", false),
        text("answered five"),
        result_ok("answered five"),
    ]);
    // TURN-PERM-S: `ls -la` granted for the session, then the identical call again
    // (the backend answers it). TURN-PERM-A: `cat README.md` approved `always`, then the
    // identical call again (the backend answers it from the lasting rule).
    t.extend(vec![
        await_in(&k.k("TURN-PERM-S")),
        cc_tool_use("t5s", "Bash", json!({"command": "ls -la"})),
        permission_request("req-perm-s", "Bash", json!({"command": "ls -la"}), "t5s"),
        await_in("req-perm-s"),
        cc_tool_result("t5s", "clean", false),
        cc_tool_use("t5t", "Bash", json!({"command": "ls -la"})),
        permission_request("req-perm-t", "Bash", json!({"command": "ls -la"}), "t5t"),
        await_in("req-perm-t"),
        cc_tool_result("t5t", "clean", false),
        text("answered perm s"),
        result_ok("answered perm s"),
        await_in(&k.k("TURN-PERM-A")),
        cc_tool_use("t5a", "Bash", json!({"command": "cat README.md"})),
        permission_request(
            "req-perm-a",
            "Bash",
            json!({"command": "cat README.md"}),
            "t5a",
        ),
        await_in("req-perm-a"),
        cc_tool_result("t5a", "readme", false),
        cc_tool_use("t5b", "Bash", json!({"command": "cat README.md"})),
        permission_request(
            "req-perm-b",
            "Bash",
            json!({"command": "cat README.md"}),
            "t5b",
        ),
        await_in("req-perm-b"),
        cc_tool_result("t5b", "readme", false),
        text("answered perm a"),
        result_ok("answered perm a"),
    ]);
    // TURN-SIX: a tool whose process runs; the user cancels the tools, the tool
    // ends in error and the turn goes on.
    t.extend(vec![
        await_in(&k.k("TURN-SIX")),
        spawn_sleeper(),
        cc_tool_use("t6", "Bash", json!({"command": "sleep 120"})),
        wait_children(),
        cc_tool_result("t6", "interrupted by cancel_tools", true),
        text("answered six"),
        result_ok("answered six"),
    ]);
    // TURN-BG: a background Bash; the process appears AFTER the tool_use (the PID
    // claim diffs the CLI's descendants around it); cancel_task stops it.
    //
    // The process is forked when the test says so (`bg_go`), once the backend has
    // tracked the task, never at a fixed delay after the tool_use: the claim's first
    // reading of the descendants comes after the engine has seen the tool_use, and a
    // process forked before that reading is never claimed (task 39427f40: under Code
    // Coverage the fixed 400 ms was overtaken; forked at once, neither engine claims it).
    t.extend(vec![
        await_in(&k.k("TURN-BG")),
        cc_tool_use(
            "t7",
            "Bash",
            json!({"command": "sleep 120", "run_in_background": true,
                   "description": "parity background"}),
        ),
        spawn_sleeper_on(&bg_go(cwd)),
        cc_tool_result("t7", "Command running in background with ID: bg1", false),
        text("answered bg"),
        result_ok("answered bg"),
        wait_children(),
    ]);
    // NATS: a message from the other instance, a Stop from there, a permission
    // answered from there, a cancel_tools from there.
    t.extend(answer(&k.k("NATS-SEND"), "answered nats"));
    t.extend(vec![
        await_in(&k.k("TURN-STALL")),
        await_in("\"interrupt\""),
        result("error_during_execution", "", true),
        await_in(&k.k("NATS-PERM")),
        cc_tool_use("t8", "Bash", json!({"command": "pwd"})),
        permission_request("req-nats", "Bash", json!({"command": "pwd"}), "t8"),
        await_in("req-nats"),
        cc_tool_result("t8", "/work", false),
        text("answered nats perm"),
        result_ok("answered nats perm"),
        await_in(&k.k("NATS-CANCEL")),
        spawn_sleeper(),
        cc_tool_use("t9", "Bash", json!({"command": "sleep 121"})),
        wait_children(),
        cc_tool_result("t9", "interrupted by cancel_tools", true),
        text("answered nats cancel"),
        result_ok("answered nats cancel"),
    ]);
    // References, then an image attachment.
    t.extend(answer(&k.k("TURN-REFS"), "answered refs"));
    t.extend(answer(&k.k("TURN-IMG"), "answered img"));
    // TURN-ASK: the CLI's AskUserQuestion comes as a can_use_tool request: the engine shows
    // the question and allows it by itself; the user's answer is the next turn.
    t.extend(vec![
        await_in(&k.k("TURN-ASK")),
        cc_tool_use("t-ask", "AskUserQuestion", question_input()),
        permission_request("req-ask", "AskUserQuestion", question_input(), "t-ask"),
        await_in("req-ask"),
        cc_tool_result("t-ask", "the user was asked", false),
        text("asked"),
        result_ok("asked"),
    ]);
    t.extend(answer(&k.k("ANSWER"), "answered ask"));
    // TURN-AGENT: a sub-agent (the CLI's Task): its messages carry the parent tool_use id,
    // its answer is the tool's result.
    t.extend(vec![
        await_in(&k.k("TURN-AGENT")),
        cc_tool_use(
            "t-agent",
            "Task",
            json!({"description": "find the spec", "prompt": k.k("SUB-PROMPT")}),
        ),
        sub_agent_text("t-agent", &k.k("SUB-SAID")),
        // ISO1: SubagentStop fires when the sub-agent finishes (contract v7).
        hook(
            "SubagentStop",
            "hook-subagent-stop",
            hook_input(
                "SubagentStop",
                cwd,
                json!({"stop_hook_active": false,
                       "last_assistant_message": k.k("SUB-SAID"),
                       "agent_id": "", "agent_type": ""}),
            ),
            None,
        ),
        cc_tool_result("t-agent", &k.k("SUB-ANSWER"), false),
        text("answered agent"),
        result_ok("answered agent"),
    ]);
    // TURN-HUGE: a PO tool answers about a megabyte. What the CLI hands back is what the real
    // CLI 2.1.296 did (CLI_BIG_NOTICE, measured); then the next turn (TURN-AFTER-HUGE).
    t.extend(vec![
        await_in(&k.k("TURN-HUGE")),
        cc_tool_use("t-big", &po("big"), json!({"n": BIG_RESULT_CHARS})),
        cc_tool_result("t-big", &cli_big_notice(cwd), false),
        text("answered huge"),
        result_ok("answered huge"),
    ]);
    t.extend(answer(&k.k("TURN-AFTER-HUGE"), "answered after huge"));
    // ISO3: `/compact <instructions>` typed in the chat reaches the CLI as the command; the
    // CLI compacts (boundary `manual`) and ends the turn. The historical engine then
    // re-injects the work context in a turn of its own, as after TURN-ONE.
    t.extend(vec![
        await_in(&format!("/compact {}", k.k("MANUAL-KEEP"))),
        emit(json!({"type": "system", "subtype": "compact_boundary",
                    "compact_metadata": {"trigger": "manual", "pre_tokens": 5400}})),
        result_ok(""),
    ]);
    if path == ProviderPath::Legacy {
        t.extend(vec![
            await_in("Work Already Done"),
            text("noted again"),
            result_ok("noted again"),
        ]);
    }
    // Closed by the backend going down (EOF on stdin).
    t.push(wait_eof());
    t
}

/// The process the resume spawns after the restart (`--resume cc-sess-1`).
fn transcript_resumed(k: Keys) -> Vec<Value> {
    let mut t = vec![capture_hooks()];
    t.extend(vec![
        await_in(&k.k("RESUMED")),
        sleep_ms(200),
        init("cc-sess-1"),
        text("answered resumed"),
        result_ok("answered resumed"),
    ]);
    t.push(wait_eof());
    t
}

/// The process the switch back from the native harness spawns.
fn transcript_switched_back() -> Vec<Value> {
    vec![
        capture_hooks(),
        await_in("SWITCH-TWO"),
        sleep_ms(200),
        init("cc-sess-2"),
        text("answered switch two"),
        result_ok("answered switch two"),
        wait_eof(),
    ]
}

// ============================================================================
// The native harness: fake_openai + fake_mcp + nexus-tools, the provider tapped
// ============================================================================

/// What the graph hooks were asked and answered on the native session.
#[derive(Default)]
struct Answers {
    before_tool: StdMutex<Vec<(String, HookVerdict)>>,
    /// The tools `after_tool` was CALLED for, recorded before the hook runs: an empty
    /// `after_tool` then says whether the hook was never called or has not answered yet.
    after_tool_called: StdMutex<Vec<String>>,
    after_tool: StdMutex<Vec<Option<String>>>,
    before_compaction: StdMutex<Vec<Option<String>>>,
    /// The limits every session was opened or resumed with, by provider kind (ISO5).
    limits: StdMutex<Vec<(ProviderKind, SessionLimits)>>,
    /// ISO1: `session_start` calls, recording the source string.
    session_start_sources: StdMutex<Vec<String>>,
    /// ISO1: `user_prompt_submit` calls, recording the prompt.
    user_prompt_submit_prompts: StdMutex<Vec<String>>,
    /// ISO1: `before_stop` calls, recording the reply that was returned.
    stop_replies: StdMutex<Vec<HookReply>>,
    /// ISO1: when set, the next `before_stop` call blocks with this reason and clears
    /// the flag (the inner hooks are not called). Used to test the stop-hook-blocks path.
    stop_block_once: StdMutex<Option<String>>,
    /// ISO1: `before_subagent_stop` was called.
    subagent_stop_called: StdMutex<bool>,
    /// ISO1: `notification` calls, recording the notification type.
    notification_types: StdMutex<Vec<String>>,
}

/// The hooks the engine gave the session, every answer recorded.
struct Recording {
    inner: Arc<dyn SessionHooks>,
    answers: Arc<Answers>,
}

#[async_trait]
impl SessionHooks for Recording {
    async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
        let verdict = self.inner.before_tool(call).await;
        self.answers
            .before_tool
            .lock()
            .unwrap()
            .push((call.name.clone(), verdict.clone()));
        verdict
    }
    async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
        self.answers
            .after_tool_called
            .lock()
            .unwrap()
            .push(result.call.name.clone());
        let said = self.inner.after_tool(result).await;
        self.answers.after_tool.lock().unwrap().push(said.clone());
        said
    }
    async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
        let said = self.inner.before_compaction(info).await;
        self.answers
            .before_compaction
            .lock()
            .unwrap()
            .push(said.clone());
        said
    }
    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        self.inner.before_turn(ctx).await
    }
    fn runs_subagents(&self) -> bool {
        self.inner.runs_subagents()
    }
    async fn run_subagent(&self, request: &SubagentRequest) -> Result<String, String> {
        self.inner.run_subagent(request).await
    }
    async fn session_start(&self, info: &SessionStartInfo) -> HookReply {
        let reply = self.inner.session_start(info).await;
        self.answers
            .session_start_sources
            .lock()
            .unwrap()
            .push(info.source.clone());
        reply
    }
    async fn user_prompt_submit(&self, prompt: &PromptInfo) -> HookReply {
        let reply = self.inner.user_prompt_submit(prompt).await;
        self.answers
            .user_prompt_submit_prompts
            .lock()
            .unwrap()
            .push(prompt.prompt.clone());
        reply
    }
    async fn before_stop(&self, info: &StopInfo) -> HookReply {
        // When the test sets stop_block_once, override the inner hooks with a block.
        let override_reason = self.answers.stop_block_once.lock().unwrap().take();
        let reply = if let Some(reason) = override_reason {
            HookReply::block(reason)
        } else {
            self.inner.before_stop(info).await
        };
        self.answers
            .stop_replies
            .lock()
            .unwrap()
            .push(reply.clone());
        reply
    }
    async fn before_subagent_stop(&self, info: &SubagentStopInfo) -> HookReply {
        let reply = self.inner.before_subagent_stop(info).await;
        *self.answers.subagent_stop_called.lock().unwrap() = true;
        reply
    }
    async fn notification(&self, info: &NotificationInfo) {
        self.answers
            .notification_types
            .lock()
            .unwrap()
            .push(info.notification_type.clone());
        self.inner.notification(info).await;
    }
}

/// The real native provider under its own id, its session hooks recorded.
#[derive(Clone)]
struct Tap {
    inner: Arc<dyn AgentProvider>,
    answers: Arc<Answers>,
}

impl Tap {
    fn tap(&self, mut spec: SessionSpec) -> SessionSpec {
        self.answers
            .limits
            .lock()
            .unwrap()
            .push((self.inner.kind(), spec.limits));
        if let Some(inner) = spec.hooks.take() {
            spec.hooks = Some(Arc::new(Recording {
                inner,
                answers: Arc::clone(&self.answers),
            }));
        }
        spec
    }
}

#[async_trait]
impl AgentProvider for Tap {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn kind(&self) -> ProviderKind {
        self.inner.kind()
    }
    async fn health(&self) -> ProviderHealth {
        self.inner.health().await
    }
    async fn catalog(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        self.inner.catalog().await
    }
    fn capabilities(&self, model: Option<&str>) -> Capabilities {
        self.inner.capabilities(model)
    }
    async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn AgentSession>, ProviderError> {
        self.inner.open(self.tap(spec)).await
    }
    async fn resume(
        &self,
        spec: SessionSpec,
        token: ResumeToken,
    ) -> Result<Arc<dyn AgentSession>, ProviderError> {
        self.inner.resume(self.tap(spec), token).await
    }
}

/// The providers of a backend of the scenario: the native instance (`local`) and the
/// built-in Claude Code on the fake CLI, both tapped.
struct Taps {
    native: Tap,
    claude: Tap,
}

impl ProviderSource for Taps {
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
        match provider_id {
            "local" => Some(Arc::new(self.native.clone()) as Arc<dyn AgentProvider>),
            super::provider::resolver::CLAUDE_CODE => {
                Some(Arc::new(self.claude.clone()) as Arc<dyn AgentProvider>)
            }
            _ => None,
        }
    }

    /// The historical engine has no provider to tap: the hooks it serves Claude Code's own
    /// events with are recorded in the Claude Code column's answers.
    fn wrap_lifecycle(&self, hooks: Arc<dyn SessionHooks>) -> Arc<dyn SessionHooks> {
        Arc::new(Recording {
            inner: hooks,
            answers: Arc::clone(&self.claude.answers),
        })
    }
}

/// ISO5: whether every session of `kind` (one at least) was opened and resumed with neither a
/// turn time limit nor a session budget (tokens or USD), and what was seen.
fn no_session_limits(answers: &Answers, kind: ProviderKind) -> (bool, String) {
    let seen: Vec<SessionLimits> = answers
        .limits
        .lock()
        .unwrap()
        .iter()
        .filter(|(k, _)| *k == kind)
        .map(|(_, limits)| *limits)
        .collect();
    let ok = !seen.is_empty()
        && seen.iter().all(|l| {
            l.turn_timeout_ms.is_none() && l.max_tokens.is_none() && l.max_cost_usd.is_none()
        });
    let what: Vec<_> = seen
        .iter()
        .map(|l| (l.turn_timeout_ms, l.max_tokens, l.max_cost_usd))
        .collect();
    (
        ok,
        format!(
            "{} session(s), (turn_timeout_ms, max_tokens, max_cost_usd)={what:?}",
            seen.len()
        ),
    )
}

fn stop(usage_prompt: u64) -> Vec<Value> {
    vec![
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": usage_prompt, "completion_tokens": 4,
                                          "total_tokens": usage_prompt + 4}}),
        json!("[DONE]"),
    ]
}

fn says(key: &str, t: &str) -> Value {
    let mut events = vec![delta(json!({"content": t}))];
    events.extend(stop(10));
    sse_route(key, events)
}

/// A model step that calls `tool` (its full name) with `arguments`.
fn calls(key: &str, id: &str, tool: &str, arguments: Value) -> Value {
    sse_route(
        key,
        vec![
            delta(
                json!({"tool_calls": [{"index": 0, "id": id, "type": "function", "function": {
                "name": tool, "arguments": arguments.to_string()}}]}),
            ),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                                              "total_tokens": 14}}),
            json!("[DONE]"),
        ],
    )
}

fn po(tool: &str) -> String {
    format!("mcp__project-orchestrator__{tool}")
}

fn nexus(tool: &str) -> String {
    format!("mcp__nexus__{tool}")
}

/// A model step that never ends until it is interrupted.
fn stalls(key: &str) -> Value {
    json!({"method": "POST", "path": "/v1/chat/completions", "status": 200,
           "body_contains": key, "event_delay_ms": 60000,
           "sse": [delta(json!({"content": "late"}))]})
}

/// The routes of the native scenario, in the order of the conversation. A route
/// answers the first request whose body contains its key among the routes not
/// used yet, in list order: a key repeated answers the steps of one turn.
fn native_script(k: Keys, cwd: &str) -> Value {
    let file = format!("{cwd}/parity.txt");
    let model = |id: &str| json!({"id": id, "context_length": 32000, "capabilities": ["vision"]});
    let mut r = vec![
        // The probe at registration: the model must be able to call a tool.
        sse_route(
            "Call the ping tool now",
            vec![
                delta(
                    json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]}),
                ),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ],
        ),
        json!({"method": "GET", "path": "/v1/models", "status": 200,
               "body": {"object": "list", "data": [model("m"), model("m2")]}}),
        // SWITCH-ONE opens the native session, the Claude Code conversation relayed.
        says("SWITCH-ONE", "answered switch one"),
        // TURN-ONE: a nexus Grep (hooks before and after the tool), then an answer.
        calls(
            &k.k("TURN-ONE"),
            "c1",
            &nexus("Grep"),
            json!({"pattern": "build_agent_spec", "path": ".", "output_mode": "content"}),
        ),
        says(&k.k("TURN-ONE"), "answered one"),
        // TURN-TWO stalls until it is interrupted (TURN-THREE queued meanwhile).
        stalls(&k.k("TURN-TWO")),
        says(&k.k("TURN-THREE"), "answered three"),
        // TURN-FOUR: three tool calls, the turn limit (3) stops it; the continuation.
        calls(
            &k.k("TURN-FOUR"),
            "c4a",
            &po("echo"),
            json!({"text": "four-a"}),
        ),
        calls(
            &k.k("TURN-FOUR"),
            "c4b",
            &po("echo"),
            json!({"text": "four-b"}),
        ),
        calls(
            &k.k("TURN-FOUR"),
            "c4c",
            &po("echo"),
            json!({"text": "four-c"}),
        ),
        says("Continue where you left off", "continued"),
        // TURN-FIVE, after set_model: a PO tool that needs a permission.
        calls(
            &k.k("TURN-FIVE"),
            "c5",
            &nexus("Bash"),
            json!({"command": "echo FIVE-$((1+1))"}),
        ),
        says(&k.k("TURN-FIVE"), "answered five"),
        // TURN-PERM2: a tool asked for the first time (Write; Bash stays asked: NATS-PERM
        // needs it), granted for the session.
        calls(
            &k.k("TURN-PERM2"),
            "c5b",
            &nexus("Write"),
            json!({"file_path": format!("{cwd}/session-1.txt"), "content": "AGAIN-SESSION"}),
        ),
        says(&k.k("TURN-PERM2"), "answered perm2"),
        // TURN-PERM3: the identical call again: the session grant holds, not asked.
        calls(
            &k.k("TURN-PERM3"),
            "c5c",
            &nexus("Write"),
            json!({"file_path": format!("{cwd}/session-1.txt"), "content": "AGAIN-SESSION"}),
        ),
        says(&k.k("TURN-PERM3"), "answered perm3"),
        // TURN-ALWAYS: a read-only command approved `always`; TURN-ALWAYS2: the identical
        // call again, answered by the backend from the lasting rule.
        calls(
            &k.k("TURN-ALWAYS"),
            "c5d",
            &nexus("Bash"),
            json!({"command": "cat always.txt"}),
        ),
        says(&k.k("TURN-ALWAYS"), "answered always"),
        calls(
            &k.k("TURN-ALWAYS2"),
            "c5e",
            &nexus("Bash"),
            json!({"command": "cat always.txt"}),
        ),
        says(&k.k("TURN-ALWAYS2"), "answered always two"),
        // TURN-SIX: a tool that sleeps until cancelled; the model then answers.
        calls(&k.k("TURN-SIX"), "c6", &po("slow"), json!({})),
        says(&k.k("TURN-SIX"), "answered six"),
        // TURN-BG: a background Bash of nexus-tools.
        calls(
            &k.k("TURN-BG"),
            "c7",
            &nexus("Bash"),
            json!({"command": "sleep 30", "run_in_background": true,
                   "description": "parity background"}),
        ),
        says(&k.k("TURN-BG"), "answered bg"),
        // NATS.
        says(&k.k("NATS-SEND"), "answered nats"),
        stalls(&k.k("TURN-STALL")),
        calls(
            &k.k("NATS-PERM"),
            "c8",
            &nexus("Bash"),
            json!({"command": "echo NATS-$((3+3))"}),
        ),
        says(&k.k("NATS-PERM"), "answered nats perm"),
        calls(&k.k("NATS-CANCEL"), "c9", &po("slow"), json!({})),
        says(&k.k("NATS-CANCEL"), "answered nats cancel"),
        // nexus-tools: Read then Edit the project file; then Bash.
        calls(
            &k.k("TURN-NEXUS"),
            "c10",
            &nexus("Read"),
            json!({"file_path": file}),
        ),
        calls(
            &k.k("TURN-NEXUS"),
            "c11",
            &nexus("Edit"),
            json!({"file_path": file, "old_string": "BEFORE-EDIT", "new_string": "AFTER-EDIT"}),
        ),
        says(&k.k("TURN-NEXUS"), "answered nexus"),
        calls(
            &k.k("TURN-BASH"),
            "c12",
            &nexus("Bash"),
            json!({"command": "echo BASH-RAN-$((40+2))"}),
        ),
        says(&k.k("TURN-BASH"), "answered bash"),
        // TURN-SEARCH (CTX2): `readonly` is deferred in this scenario. The model loads it
        // with ToolSearch, calls it in the same turn, then answers.
        calls(
            &k.k("TURN-SEARCH"),
            "cs1",
            "ToolSearch",
            json!({"query": format!("select:{}", po("readonly"))}),
        ),
        calls(&k.k("TURN-SEARCH"), "cs2", &po("readonly"), json!({})),
        says(&k.k("TURN-SEARCH"), "answered search"),
        // References, the image.
        says(&k.k("TURN-REFS"), "answered refs"),
        says(&k.k("TURN-IMG"), "answered img"),
        // TURN-ASK: the harness's AskUserQuestion; the turn ends on it, the answer is a turn.
        calls(&k.k("TURN-ASK"), "cq1", "AskUserQuestion", question_input()),
        says(&k.k("ANSWER"), "answered ask"),
        // TURN-AGENT: the harness's Agent; the child session's own request (its prompt, not
        // the parent's key) is answered before the parent goes on with the result.
        calls(
            &k.k("TURN-AGENT"),
            "ca1",
            "Agent",
            json!({"description": "find the spec", "prompt": k.k("SUB-PROMPT"), "subagent_type": "explore"}),
        ),
        says(&k.k("SUB-PROMPT"), &k.k("SUB-ANSWER")),
        says(&k.k("TURN-AGENT"), "answered agent"),
        // ISO1: TURN-STOP: the model answers, the stop hook blocks once (via
        // stop_block_once), the model continues with the feedback and answers again.
        says(&k.k("TURN-STOP"), "partially done"),
        says("Stop hook feedback", "answered stop"),
        // TURN-HUGE: a PO tool answers about a megabyte (fake_mcp `big`); the model then
        // answers; the next turn (TURN-AFTER-HUGE) starts on that history.
        calls(
            &k.k("TURN-HUGE"),
            "cbig",
            &po("big"),
            json!({"n": BIG_RESULT_CHARS}),
        ),
        says(&k.k("TURN-HUGE"), "answered huge"),
        says(&k.k("TURN-AFTER-HUGE"), "answered after huge"),
    ];
    // A turn whose reported size makes the next model call compact first; the
    // summary request carries the guidance of the before_compaction hook.
    let mut prime = vec![delta(json!({"content": "primed"}))];
    prime.extend(stop(30_000));
    r.push(sse_route(&k.k("TURN-PRIME"), prime));
    r.push(says("Additional instructions", "the summary"));
    r.push(says(&k.k("TURN-COMPACT"), "answered compact"));
    // ISO3: the summary request of `/compact <instructions>` (its key is the instructions).
    r.push(says(&k.k("MANUAL-KEEP"), "the summary, on request"));
    // After the restart.
    r.push(says(&k.k("RESUMED"), "answered resumed"));
    Value::Array(r)
}

/// On a failure, the turn of every request the native model received.
struct OpenAiDump<'a>(&'a FakeOpenAi);

impl Drop for OpenAiDump<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        for (i, body) in bodies(self.0).iter().enumerate() {
            let messages = serde_json::from_str::<Value>(body)
                .ok()
                .and_then(|b| b["messages"].as_array().cloned())
                .unwrap_or_default();
            let last = messages.last().map(Value::to_string).unwrap_or_default();
            eprintln!("fake_openai request {i}: {}", excerpt(&last));
        }
    }
}

fn bodies(fake: &FakeOpenAi) -> Vec<String> {
    fake.chat_requests()
        .iter()
        .map(|r| r["body"].to_string())
        .collect()
}

/// The `n`-th request (0-based) whose body contains `key`.
async fn wait_body_nth(fake: &FakeOpenAi, key: &str, n: usize, within: Duration) -> Option<String> {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        if let Some(body) = bodies(fake).into_iter().filter(|b| b.contains(key)).nth(n) {
            return Some(body);
        }
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_body(fake: &FakeOpenAi, key: &str) -> Option<String> {
    wait_body_nth(fake, key, 0, WAIT).await
}

/// A Claude Code turn whose permission `req` is answered with `scope`: whether the
/// answer was routed, and the line the CLI read.
async fn cc_scoped(
    manager: &ChatManager,
    w: &Watch,
    cli: &FakeClaude,
    key: String,
    req: &'static str,
    scope: crate::chat::types::PermissionAnswerScope,
) -> (bool, String) {
    let ends = w.turn_ends();
    manager.send_message(&w.sid, &key).await.unwrap();
    assert!(
        w.wait(
            WAIT,
            |e| matches!(e, ChatEvent::PermissionRequest { id, .. } if id == req)
        )
        .await,
        "the Claude Code engine did not show {req}"
    );
    let routed = manager
        .route_permission_response(&w.sid, req, true, scope, false)
        .await;
    if routed.is_err() {
        // Refused (the request still waits): unblock the turn.
        let _ = manager
            .route_permission_response(
                &w.sid,
                req,
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                false,
            )
            .await;
    }
    let line = cli.wait_line(0, req, WAIT).await.unwrap_or_default();
    w.settle(ends, manager).await;
    (routed.is_ok(), line)
}

/// The person both sessions of the scenario belong to (their lasting approvals are theirs).
fn parity_person() -> crate::auth::jwt::Claims {
    crate::auth::jwt::Claims {
        sub: "parity-person".to_string(),
        email: "parity@ffs.holdings".to_string(),
        name: "Parity".to_string(),
        iat: 0,
        exp: i64::MAX / 2,
        token_type: None,
        scope: None,
        jti: None,
    }
}

/// Whether the lasting rules of [`parity_person`] on `proj` hold `rule` (what another
/// session of theirs on the project obeys: the rules are read at each request).
fn lasting_rules_of(manager: &ChatManager, rule: &str) -> bool {
    manager.lasting_rule_store().is_some_and(|store| {
        store
            .rules(&parity_person().sub)
            .iter()
            .any(|r| r.project == "proj" && r.rule == rule && r.grant(&[]).is_some())
    })
}

/// Whether the NEXT session of [`parity_person`] on `proj`, running in `cwd` (the project's
/// root), would have this call answered by a lasting rule (`session_grants::cover` on a
/// fresh session's grants, its lasting scope resolved like an opening's).
async fn next_session_covers(
    manager: &ChatManager,
    cwd: &str,
    asker: crate::chat::session_grants::Asker,
    tool: &str,
    canonical: Option<&str>,
    command: &str,
) -> bool {
    let person = parity_person();
    let Some(place) = manager
        .lasting_scope(Some(&person.sub), Some("proj"), cwd)
        .await
    else {
        return false;
    };
    let grants = std::sync::Mutex::new(
        crate::chat::session_grants::SessionGrants::declaring(Vec::new()).with_lasting(Some(place)),
    );
    let call = crate::chat::session_grants::AskedCall {
        asker,
        tool: tool.to_string(),
        canonical: canonical.map(str::to_string),
        input: json!({ "command": command }),
    };
    crate::chat::session_grants::cover(&grants, &call)
        .await
        .is_some_and(|c| c.lasting)
}

/// A Claude Code turn asking `req` then `again` (the identical call): `req` is answered
/// with `scope`, `again` is left to the backend (answered `once` when it is still asked,
/// to unblock the turn). Whether `req` was routed, and the line the CLI read for it.
async fn cc_scoped_twice(
    manager: &ChatManager,
    w: &Watch,
    cli: &FakeClaude,
    key: String,
    req: &'static str,
    again: &'static str,
    scope: crate::chat::types::PermissionAnswerScope,
) -> (bool, String) {
    let ends = w.turn_ends();
    manager.send_message(&w.sid, &key).await.unwrap();
    assert!(
        w.wait(
            WAIT,
            |e| matches!(e, ChatEvent::PermissionRequest { id, .. } if id == req)
        )
        .await,
        "the Claude Code engine did not show {req}"
    );
    let routed = manager
        .route_permission_response(&w.sid, req, true, scope, false)
        .await;
    let once = crate::chat::types::PermissionAnswerScope::Once;
    if routed.is_err() {
        let _ = manager
            .route_permission_response(&w.sid, req, true, once, false)
            .await;
    }
    let line = cli.wait_line(0, req, WAIT).await.unwrap_or_default();
    // Waited on as such, never for a fixed time: the backend's own decision about `again`
    // (it answers it, or the request waits for the user and nothing comes before WAIT).
    // The fixed 3 s it replaces was not the cause of FLAKY-PARITE-ALWAYS (the backend's
    // answer lands before `req`'s routing even returns) but still a bet on timing.
    let by_backend = w
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::PermissionDecision { id, .. } if id == again),
        )
        .await;
    if !by_backend {
        // Not answered by the backend: the user unblocks it.
        let _ = manager
            .route_permission_response(&w.sid, again, true, once, false)
            .await;
    }
    w.settle(ends, manager).await;
    (routed.is_ok(), line)
}

/// A native turn whose tool (input holding `needle`) is asked and answered with `scope`:
/// the request id, and whether the answer was accepted.
async fn native_grant(
    manager: &ChatManager,
    w: &Watch,
    fake: &FakeOpenAi,
    key: String,
    needle: &str,
    said: &str,
    scope: crate::chat::types::PermissionAnswerScope,
) -> (Option<String>, bool) {
    let ends = w.turn_ends();
    manager.send_message(&w.sid, &key).await.unwrap();
    let id = w.permission_id(WAIT, needle).await;
    let granted = match &id {
        Some(id) => manager
            .route_permission_response(&w.sid, id, true, scope, false)
            .await
            .is_ok(),
        None => false,
    };
    if let (Some(id), false) = (&id, granted) {
        // Refused (the request still waits): unblock the turn.
        let _ = manager
            .route_permission_response(
                &w.sid,
                id,
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                false,
            )
            .await;
    }
    wait_body_nth(fake, &key, 1, WAIT).await;
    w.said(WAIT, said).await;
    w.settle(ends, manager).await;
    (id, granted)
}

/// Allows the native tool whose input contains `needle` if the session asks; returns
/// whether it asked. Stops waiting once the model received the `nth` request of `key`
/// (the tool ran without asking).
async fn answer_if_asked(
    manager: &ChatManager,
    w: &Watch,
    fake: &FakeOpenAi,
    needle: &str,
    key: &str,
    nth: usize,
) -> bool {
    let deadline = tokio::time::Instant::now() + WAIT;
    while tokio::time::Instant::now() < deadline {
        if let Some(id) = w.permission_ids(needle).await.into_iter().next() {
            let _ = manager
                .route_permission_response(
                    &w.sid,
                    &id,
                    true,
                    crate::chat::types::PermissionAnswerScope::Once,
                    false,
                )
                .await;
            return true;
        }
        if bodies(fake)
            .iter()
            .filter(|b| b.contains(key))
            .nth(nth)
            .is_some()
        {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

/// The deadline of the steps that carry the megabyte of `tool_result.huge` (the PO server
/// writing it, the harness reading and persisting it): its own, like
/// [`INTERRUPT_TURN_END_WAIT`], because under the Code Coverage job every hop is slower. `WAIT`
/// itself is not widened.
const HUGE_WAIT: Duration = Duration::from_secs(60);

/// `tool_result.huge` on Claude Code: TURN-HUGE (the CLI's measured notice, replayed), then
/// TURN-AFTER-HUGE. Two messages sent.
async fn cc_huge(
    manager: &ChatManager,
    w: &Watch,
    cli: &FakeClaude,
    k: Keys,
    sid: &str,
) -> (bool, String) {
    let ends = w.turn_ends();
    manager.send_message(sid, &k.k("TURN-HUGE")).await.unwrap();
    let read = cli.wait_line(0, &k.k("TURN-HUGE"), WAIT).await.is_some();
    let shown = w
        .wait(
            HUGE_WAIT,
            |e| matches!(e, ChatEvent::ToolResult { id, .. } if id == "t-big"),
        )
        .await;
    let answered = read && shown && w.said(HUGE_WAIT, "answered huge").await;
    w.settle(ends, manager).await;
    let ends = w.turn_ends();
    manager
        .send_message(sid, &k.k("TURN-AFTER-HUGE"))
        .await
        .unwrap();
    let next = cli
        .wait_line(0, &k.k("TURN-AFTER-HUGE"), WAIT)
        .await
        .is_some()
        && w.said(WAIT, "answered after huge").await;
    w.settle(ends, manager).await;
    big_verdict(
        &w.all().await,
        "t-big",
        answered,
        next,
        (
            true,
            "ce que le CLI fait du mégaoctet (fichier de session, avis de ~1,6 Ko) est relevé \
             sur 2.1.296 et rejoué par fake_claude"
                .to_string(),
        ),
    )
}

/// `compaction.manual` on Claude Code (both engines): `/compact <instructions>` sent from the
/// chat reaches the CLI as the command, unchanged at the head of the turn's text, and the
/// CLI's boundary (`manual`) reaches the thread. One message sent.
async fn cc_manual_compact(
    manager: &ChatManager,
    w: &Watch,
    cli: &FakeClaude,
    k: Keys,
    sid: &str,
    path: ProviderPath,
) -> (bool, String) {
    let ends = w.turn_ends();
    let command = format!("/compact {}", k.k("MANUAL-KEEP"));
    manager.send_message(sid, &command).await.unwrap();
    let line = cli.wait_line(0, &command, WAIT).await;
    let text = line.as_deref().map(user_line_text).unwrap_or_default();
    let as_command = text.trim_start().starts_with("/compact");
    let boundary = w
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::CompactBoundary { trigger, .. } if trigger == "manual"),
        )
        .await;
    w.settle(ends, manager).await;
    if path == ProviderPath::Legacy {
        // The work context the historical engine re-injects in a turn of its own.
        let ends = w.turn_ends();
        if w.said(WAIT, "noted again").await {
            w.settle(ends, manager).await;
        }
    }
    (
        line.is_some() && as_command && boundary,
        format!(
            "commande lue par le CLI={}, en tête du texte={as_command} ({}), compact_boundary \
             manual={boundary}",
            line.is_some(),
            excerpt(&text)
        ),
    )
}

/// The text of a user message line written to the CLI's stdin.
fn user_line_text(line: &str) -> String {
    let v: Value = serde_json::from_str(line).unwrap_or_default();
    match &v["message"]["content"] {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => line.to_string(),
    }
}

/// `compaction.manual` on the native harness: `/compact <instructions>` is a compaction out of
/// turn — ONE request to the model, the summary, carrying the instructions and the PreCompact
/// hook's guidance (the project's name); no turn; both phases on the thread with `trigger:
/// manual`; the outcome said. One message sent.
async fn native_manual_compact(
    manager: &ChatManager,
    wn: &Watch,
    fake: &FakeOpenAi,
    k: Keys,
    nid: &str,
    project: &str,
) -> (bool, String) {
    let ends = wn.turn_ends();
    let before = bodies(fake).len();
    let instructions = k.k("MANUAL-KEEP");
    manager
        .send_message(nid, &format!("/compact {instructions}"))
        .await
        .unwrap();
    let summary = wait_body(fake, &instructions).await.unwrap_or_default();
    let with_instructions = summary.contains(&format!("Additional instructions: {instructions}"));
    let with_hook = summary.contains(project);
    let started = wn
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::CompactionStarted { trigger } if trigger == "manual"),
        )
        .await;
    let boundary = wn
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::CompactBoundary { trigger, .. } if trigger == "manual"),
        )
        .await;
    let said = wn
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::SystemHint { content }
                if content.starts_with("Conversation compacted on request"))
        })
        .await;
    wn.settle(ends, manager).await;
    let requests = bodies(fake).len() - before;
    let no_turn = !bodies(fake)[before..]
        .iter()
        .any(|b| b.contains("/compact"));
    (
        with_instructions && with_hook && started && boundary && said && requests == 1 && no_turn,
        format!(
            "instructions dans le résumé={with_instructions}, hook PreCompact={with_hook}, \
             compaction_started manual={started}, compact_boundary manual={boundary}, issue \
             dite={said}, requêtes au modèle={requests}, aucun tour /compact={no_turn}"
        ),
    )
}

/// `tool_result.huge` on the native harness: TURN-HUGE calls the PO tool `big` (fake_mcp, a
/// real megabyte through MCP), then TURN-AFTER-HUGE. Two messages sent.
async fn native_huge(
    manager: &ChatManager,
    w: &Watch,
    fake: &FakeOpenAi,
    k: Keys,
    sid: &str,
) -> (bool, String) {
    let ends = w.turn_ends();
    manager.send_message(sid, &k.k("TURN-HUGE")).await.unwrap();
    // A read-only PO tool is not asked; answered once if it ever is.
    answer_if_asked(manager, w, fake, "1000000", &k.k("TURN-HUGE"), 1).await;
    // The request after the call: what the model received for it.
    let after = wait_body_nth(fake, &k.k("TURN-HUGE"), 1, HUGE_WAIT)
        .await
        .unwrap_or_default();
    let to_model = serde_json::from_str::<Value>(&after)
        .ok()
        .and_then(|b| {
            b["messages"].as_array().and_then(|m| {
                m.iter()
                    .find(|m| m["role"] == "tool" && m["tool_call_id"] == "cbig")
                    .map(|m| match &m["content"] {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
            })
        })
        .unwrap_or_default();
    let persisted = to_model.starts_with("<persisted-output>");
    let path = to_model
        .split_once("Full output saved to: ")
        .and_then(|(_, rest)| rest.lines().next())
        .unwrap_or_default()
        .to_string();
    let on_disk = std::fs::metadata(&path).map(|m| m.len()).ok();
    let answered = !after.is_empty() && w.said(HUGE_WAIT, "answered huge").await;
    w.settle(ends, manager).await;
    let ends = w.turn_ends();
    manager
        .send_message(sid, &k.k("TURN-AFTER-HUGE"))
        .await
        .unwrap();
    let next = wait_body(fake, &k.k("TURN-AFTER-HUGE")).await.is_some()
        && w.said(WAIT, "answered after huge").await;
    w.settle(ends, manager).await;
    let kept = on_disk == Some(BIG_RESULT_CHARS);
    let _ = std::fs::remove_file(&path);
    big_verdict(
        &w.all().await,
        "cbig",
        answered,
        next,
        (
            persisted && to_model.len() < BIG_SEEN_BOUND && kept,
            format!(
                "reçu par le modèle : {} octets, <persisted-output>={persisted}, fichier nommé \
                 {} octets sur disque",
                to_model.len(),
                on_disk.map_or("absent".to_string(), |n| n.to_string())
            ),
        ),
    )
}

/// The last user message of a request body (the turn, not the history).
fn last_user(body: &str) -> Value {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    v["messages"]
        .as_array()
        .and_then(|m| m.iter().rev().find(|m| m["role"] == "user").cloned())
        .unwrap_or(Value::Null)
}

fn excerpt(s: &str) -> String {
    let s: String = s.chars().take(300).collect();
    s.replace('\n', " ")
}

/// The compaction instructions in the answer to a `PreCompact` callback: its
/// `systemMessage`, the one field the CLI appends to the summary's instructions. It
/// ignores `reason` and rejects a `hookSpecificOutput` (Claude Code 2.1.295, M2c).
fn precompact_instructions(line: &str) -> String {
    let v: Value = serde_json::from_str(line).unwrap_or(Value::Null);
    v["response"]["response"]["systemMessage"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

// ============================================================================
// The scenario
// ============================================================================

/// The scenario twice: Claude Code on the historical engine, then on the agent engine
/// (each run on its own stage, CLI and endpoint); the native column is the first run's.
#[test]
fn the_same_scenario_on_both_engines_gives_the_parity_matrix() {
    // A thread of its own with a large stack: the scenario's future outgrew the test
    // thread's (block_on keeps the future it drives on the stack), and Claude Code on the
    // agent engine nests deeper still in a debug build.
    std::thread::Builder::new()
        .name("parity".into())
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .thread_stack_size(16 << 20)
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let (cc, na) = run_naming_the_column(ProviderPath::Legacy).await;
                    let (cca, _) = run_naming_the_column(ProviderPath::Agent).await;
                    // Codex: provider_switch.relay measured via mini native→Codex run;
                    // all other functions declared (transcript per function not yet written, ISO4).
                    let codex = run_codex_column().await;
                    // opencode: no fake_opencode, fully declared not_measured (YARDSTICK).
                    let opencode = as_declared(Engine::OpenCode);
                    // ACP: provider_switch.relay measured via mini native→ACP run;
                    // all other functions declared or not_measured per EXPECTED (ISO4).
                    let acp = run_acp_column().await;
                    gate(&[
                        (Engine::ClaudeCode, &cc),
                        (Engine::ClaudeCodeAgent, &cca),
                        (Engine::Native, &na),
                        (Engine::Codex, &codex),
                        (Engine::OpenCode, &opencode),
                        (Engine::Acp, &acp),
                    ]);
                })
        })
        .unwrap()
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

/// One run of the scenario; a run that panics says WHICH run (the Claude Code column it
/// measures), so a failure on CI names its engine (review of #726).
async fn run_naming_the_column(path: ProviderPath) -> (Measures, Measures) {
    use futures::FutureExt;
    match std::panic::AssertUnwindSafe(Box::pin(the_scenario(path)))
        .catch_unwind()
        .await
    {
        Ok(measures) => measures,
        Err(panic) => {
            let why = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            let column = match path {
                ProviderPath::Legacy => Engine::ClaudeCode.label(),
                ProviderPath::Agent => Engine::ClaudeCodeAgent.label(),
            };
            panic!("parity run with Claude Code as « {column} » ({path:?}) failed: {why}");
        }
    }
}

/// The body of [`the_same_scenario_on_both_engines_gives_the_parity_matrix`], with Claude
/// Code on the engine `path` names: the measures of Claude Code, then of the native harness.
async fn the_scenario(path: ProviderPath) -> (Measures, Measures) {
    let stage = Stage::new().await;
    let cwd = stage.cwd();
    let claims = crate::auth::jwt::Claims::service_account("parity");
    let kc = Keys("CC");
    let kn = Keys("NA");

    let cli = FakeClaude::new();
    cli.transcript(0, &transcript_main(kc, &cwd, path));
    cli.transcript(1, &transcript_resumed(kc));
    cli.transcript(2, &transcript_switched_back());

    let fake = FakeOpenAi::start_for(native_script(kn, &cwd), 900_000);
    let _dump = OpenAiDump(&fake);
    store_instance(&stage.graph, &instance(&fake, "none")).await;
    consent(&stage.graph, "proj", "local", &fake.origin()).await;
    let answers: Arc<Answers> = Arc::default();
    // What the hooks of Claude Code (either engine) were asked, apart from the native harness's.
    let cc_answers: Arc<Answers> = Arc::default();

    let mut cc = Measures::default();
    let mut na = Measures::default();

    // The stored setting every backend of the scenario reads at startup (CTX2): the PO
    // tool `readonly` of fake_mcp is deferred — no other row uses it, so the rest of the
    // scenario sees exactly the tools it always saw. Measured by the row `tool_search`.
    super::deferred_tools::store(stage.graph.as_ref(), &[po("readonly")])
        .await
        .unwrap();

    // ===================================================================
    // Claude Code, historical engine (backend process #1)
    // ===================================================================
    let manager = stage.backend_with(&cli, &answers, &cc_answers, path).await;
    let mut req = request(None, Some("proj"), "default");
    req.cwd = cwd.clone();
    req.message = format!("{}: where is the spec built?", kc.k("TURN-ONE"));
    // A person's conversation: its lasting approvals are theirs (P11b).
    req.user_claims = Some(parity_person());
    let sid = manager
        .create_session(&req)
        .await
        .unwrap_or_else(|e| panic!("opening the Claude Code session: {e:#}"))
        .session_id;
    assert!(
        manager.agent_runtime.owns(&sid).await == (path == ProviderPath::Agent),
        "Claude Code runs on the engine {path:?} names"
    );
    let w = Watch::start(
        stage.graph.clone(),
        &sid,
        manager.subscribe(&sid).await.unwrap(),
    );
    let mut cc_sent = 1i64;

    // TURN-ONE: enrichment, hooks, compaction, PO tools.
    let first = cli.wait_line(0, &kc.k("TURN-ONE"), WAIT).await;
    cc.check(
        "enrichment",
        first.as_deref().is_some_and(|l| l.contains(PLAN_TITLE)),
        Cause::Harness,
        format!(
            "ligne du tour : {}",
            excerpt(first.as_deref().unwrap_or("absente"))
        ),
    );
    let pre = cli.wait_line(0, "hook-pre", WAIT).await;
    cc.check(
        "hooks.before_tool",
        pre.is_some(),
        Cause::Harness,
        format!(
            "réponse au hook PreToolUse : {:?}",
            pre.as_deref().map(excerpt)
        ),
    );
    let post = cli.wait_line(0, "hook-post", WAIT).await;
    cc.check(
        "hooks.after_tool",
        post.as_deref()
            .is_some_and(|l| l.contains("find_references") && l.contains("build_agent_spec")),
        Cause::Harness,
        format!(
            "réponse au hook PostToolUse : {:?}",
            post.as_deref().map(excerpt)
        ),
    );
    let compact = cli.wait_line(0, "hook-compact", WAIT).await;
    cc.check(
        "hooks.before_compaction",
        compact
            .as_deref()
            .is_some_and(|l| precompact_instructions(l).contains(&stage.project.name)),
        Cause::Harness,
        format!(
            "réponse au hook PreCompact : {:?}",
            compact.as_deref().map(excerpt)
        ),
    );
    // ISO1: SessionStart — the CLI sends none to SDK callbacks (measured), the host asks the
    // session's hook itself at the opening, on both engines.
    let ss = cc_answers.session_start_sources.lock().unwrap().clone();
    cc.check(
        "hooks.session_start",
        !ss.is_empty(),
        Cause::Harness,
        format!("session_start appelé {} fois, sources={ss:?}", ss.len()),
    );
    // ISO1: UserPromptSubmit — the CLI's callback reaches the session's hook, and the CLI is
    // answered (the response line carries the callback's request id).
    let prompt_answered = cli.wait_line(0, "hook-prompt", WAIT).await.is_some();
    let prompts = cc_answers
        .user_prompt_submit_prompts
        .lock()
        .unwrap()
        .clone();
    cc.check(
        "hooks.user_prompt_submit",
        prompt_answered && !prompts.is_empty(),
        Cause::Harness,
        format!(
            "CLI répondu={prompt_answered}, user_prompt_submit appelé {} fois, prompts={:?}",
            prompts.len(),
            prompts.iter().map(|p| excerpt(p)).collect::<Vec<_>>()
        ),
    );
    // Stop, SubagentStop and Notification are measured further down, after the turns in
    // which the fake CLI raises them (TURN-THREE, TURN-AGENT, TURN-FIVE): measured here they
    // read as gaps only because their turn had not happened yet.
    assert!(
        w.said(WAIT, "answered one").await,
        "the first Claude Code turn never answered"
    );
    let started = w
        .any(|e| matches!(e, ChatEvent::CompactionStarted { .. }))
        .await;
    let boundary = w
        .any(|e| matches!(e, ChatEvent::CompactBoundary { .. }))
        .await;
    cc.check(
        "compaction",
        started && boundary,
        Cause::Harness,
        format!("compaction_started={started}, compact_boundary={boundary}"),
    );
    let args = cli.args(0);
    let mcp_config = args
        .iter()
        .position(|a| a == "--mcp-config")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    // The CLI is given a file (or, on older SDKs, the JSON itself).
    let mcp_config = std::fs::read_to_string(&mcp_config).unwrap_or(mcp_config);
    cc.check(
        "po_tools",
        mcp_config.contains("project-orchestrator") && mcp_config.contains("fake_mcp"),
        Cause::Harness,
        format!("--mcp-config : {}", excerpt(&mcp_config)),
    );
    cc.set(
        "nexus_tools",
        Verdict::NotMeasured,
        "outils internes du CLI, non exécutés par fake_claude",
    );
    cc.set(
        "tool_search",
        Verdict::NotMeasured,
        "outils différés internes au CLI, liste d'outils invisible du harnais",
    );
    cc.set(
        "anthropic_messages",
        Verdict::NotMeasured,
        "cache de prompt posé par le CLI, requête au modèle invisible du harnais",
    );
    w.settle(0, &manager).await;

    // Message queue: TURN-THREE sent while TURN-TWO runs.
    let ends = w.turn_ends();
    manager.send_message(&sid, &kc.k("TURN-TWO")).await.unwrap();
    cc_sent += 1;
    assert!(cli.wait_line(0, &kc.k("TURN-TWO"), WAIT).await.is_some());
    tokio::time::sleep(Duration::from_millis(300)).await;
    // ISO1: the Stop hook of this turn blocks once (as the native TURN-STOP does).
    *cc_answers.stop_block_once.lock().unwrap() = Some("keep going".to_string());
    manager
        .send_message(&sid, &kc.k("TURN-THREE"))
        .await
        .unwrap();
    cc_sent += 1;
    let three = cli.wait_line(0, &kc.k("TURN-THREE"), WAIT).await;
    let interrupted = cli.line(0, "\"interrupt\"").is_some();
    cc.check(
        "message_queue",
        three.is_some() && interrupted && w.said(WAIT, "answered three").await,
        Cause::Harness,
        format!(
            "interrupt écrit={interrupted}, message suivant lu={}",
            three.is_some()
        ),
    );
    w.settle(ends, &manager).await;
    // ISO1: Stop — the session's hook was consulted before the turn ended, and its block
    // went to the CLI as `decision: block` + the reason: the answer the CLI reads to carry
    // the turn on (fake_claude replays a transcript, so the relaunch itself is the CLI's).
    let stop_line = cli
        .wait_line(0, "hook-stop", WAIT)
        .await
        .unwrap_or_default();
    let stop_replies = cc_answers.stop_replies.lock().unwrap().clone();
    let blocked = stop_replies
        .iter()
        .any(|r| r.block_reason() == Some("keep going"));
    cc.check(
        "hooks.stop",
        blocked && stop_line.contains("\"block\"") && stop_line.contains("keep going"),
        Cause::Harness,
        format!(
            "stop appelé {} fois, hook a bloqué={blocked}, réponse au CLI : {}",
            stop_replies.len(),
            excerpt(&stop_line)
        ),
    );

    // Auto-continue: always on (decision 7b95342a), no toggle sent.
    let ends = w.turn_ends();
    manager
        .send_message(&sid, &kc.k("TURN-FOUR"))
        .await
        .unwrap();
    cc_sent += 1;
    let continued = cli.wait_line(0, "Continue where you left off", WAIT).await;
    let announced = w
        .wait(WAIT, |e| matches!(e, ChatEvent::AutoContinue { .. }))
        .await;
    cc.check(
        "auto_continue",
        continued.is_some() && announced,
        Cause::Harness,
        format!(
            "continuation écrite={}, auto_continue={announced}",
            continued.is_some()
        ),
    );
    w.said(WAIT, "continued").await;
    w.settle(ends, &manager).await;

    // set_model between two turns.
    manager
        .set_session_model(&sid, "fake-claude-2")
        .await
        .unwrap();
    let set = cli.wait_line(0, "set_model", WAIT).await;
    let changed = w
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::ModelChanged { model } if model == "fake-claude-2"),
        )
        .await;
    cc.check(
        "set_model",
        set.as_deref().is_some_and(|l| l.contains("fake-claude-2")) && changed,
        Cause::Harness,
        format!("set_model écrit={}, model_changed={changed}", set.is_some()),
    );

    // A permission answered once.
    let ends = w.turn_ends();
    manager
        .send_message(&sid, &kc.k("TURN-FIVE"))
        .await
        .unwrap();
    cc_sent += 1;
    let asked = w
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::PermissionRequest { id, .. } if id == "req-perm"),
        )
        .await;
    assert!(
        asked,
        "the Claude Code engine did not show the permission request"
    );
    manager
        .route_permission_response(
            &sid,
            "req-perm",
            true,
            crate::chat::types::PermissionAnswerScope::Once,
            false,
        )
        .await
        .unwrap();
    let allowed = cli.wait_line(0, "req-perm", WAIT).await.unwrap_or_default();
    cc.check(
        "permissions.once",
        allowed.contains("\"behavior\":\"allow\"") && w.said(WAIT, "answered five").await,
        Cause::Harness,
        format!("réponse écrite : {}", excerpt(&allowed)),
    );
    // ISO1: Notification — a tool call waited for the user's permission: the session's hook
    // was told (the CLI sends no such notification to SDK callbacks, measured: the host tells
    // it from the `can_use_tool` request, on both engines).
    let notif = cc_answers.notification_types.lock().unwrap().clone();
    cc.check(
        "hooks.notification",
        notif.iter().any(|t| t == "permission_prompt"),
        Cause::Harness,
        format!("notification appelé {} fois, types={notif:?}", notif.len()),
    );
    w.settle(ends, &manager).await;
    let (timed, seen) = timing_verdict(&w.persisted().await, "t5");
    cc.check("tool_timing", timed, Cause::Harness, seen);
    // `session`: a plain allow for the CLI (never a rule handed to it); the identical
    // call again is answered by the backend. `always`: kept by the backend (P11b), measured below.
    // Boxed: the scenario's future is already large (a stack overflow otherwise).
    let (routed_s, line_s) = Box::pin(cc_scoped(
        &manager,
        &w,
        &cli,
        kc.k("TURN-PERM-S"),
        "req-perm-s",
        crate::chat::types::PermissionAnswerScope::Session,
    ))
    .await;
    cc_sent += 1;
    let (routed_a, line_a) = Box::pin(cc_scoped_twice(
        &manager,
        &w,
        &cli,
        kc.k("TURN-PERM-A"),
        "req-perm-a",
        "req-perm-b",
        crate::chat::types::PermissionAnswerScope::Always,
    ))
    .await;
    let line_b = cli
        .wait_line(0, "req-perm-b", WAIT)
        .await
        .unwrap_or_default();
    let events = w.all().await;
    let b_by_backend = events.iter().any(|e| {
        matches!(e, ChatEvent::PermissionDecision { id, scope: Some(s), rule: Some(r), .. }
            if id == "req-perm-b"
                && *s == crate::chat::types::PermissionAnswerScope::Always
                && r == "Bash: cat README.md")
    });
    let cc_kept = lasting_rules_of(&manager, "Bash: cat README.md")
        && Box::pin(next_session_covers(
            &manager,
            &cwd,
            crate::chat::session_grants::Asker::ClaudeCode,
            "Bash",
            None,
            "cat README.md",
        ))
        .await;
    cc_sent += 1;
    let second = cli
        .wait_line(0, "req-perm-t", WAIT)
        .await
        .unwrap_or_default();
    let plain = |line: &str| {
        line.contains("\"behavior\":\"allow\"") && !line.contains("updatedPermissions")
    };
    cc.check(
        "permissions.session",
        routed_s && plain(&line_s) && plain(&second),
        Cause::Harness,
        format!(
            "accordée={routed_s}, réponse : {}, appel identique répondu par le backend : {}",
            excerpt(&line_s),
            excerpt(&second)
        ),
    );
    cc.check(
        "permissions.always",
        routed_a && plain(&line_a) && plain(&line_b) && b_by_backend && cc_kept,
        Cause::Harness,
        format!(
            "always accepté={routed_a}, réponse : {}, appel identique répondu par le backend \
             (always)={b_by_backend} : {}, règle gardée pour la personne et le projet={cc_kept}",
            excerpt(&line_a),
            excerpt(&line_b)
        ),
    );

    // cancel_tools: the running tool's process is signalled, the turn goes on.
    let ends = w.turn_ends();
    manager.send_message(&sid, &kc.k("TURN-SIX")).await.unwrap();
    cc_sent += 1;
    assert!(
        w.wait(
            WAIT,
            |e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "t6")
        )
        .await
    );
    tokio::time::sleep(Duration::from_millis(400)).await;
    let cancelled = manager.cancel_running_tools(&sid).await.unwrap();
    let went_on = w.said(WAIT, "answered six").await;
    cc.check(
        "cancel_tools",
        !cancelled.killed_pids.is_empty() && went_on,
        Cause::Harness,
        format!(
            "killed_pids={:?}, tour poursuivi={went_on}",
            cancelled.killed_pids
        ),
    );
    w.settle(ends, &manager).await;

    // A background task, then cancel_task.
    let ends = w.turn_ends();
    manager.send_message(&sid, &kc.k("TURN-BG")).await.unwrap();
    cc_sent += 1;
    let tracked = w
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::ActiveTasksUpdate { tasks } if tasks.iter().any(|t| t.id == "t7"))
        })
        .await;
    cc.check(
        "background_tasks",
        tracked,
        Cause::Harness,
        format!("active_tasks_update avec t7={tracked}"),
    );
    // The task is tracked. The claim reads the CLI's descendants, then again a second
    // later: the process must start between the two readings. The legacy engine takes
    // the first reading before it announces the task; the agent engine (nexus) takes it
    // in a task of its own, started after the announcement, through `ps` (slow under
    // Code Coverage: a fork at once was overtaken there). The process starts 400 ms
    // after the announcement: after the first reading, well before the second.
    tokio::time::sleep(Duration::from_millis(400)).await;
    std::fs::write(bg_go(&cwd), b"").unwrap();
    w.said(WAIT, "answered bg").await;
    // The PID claim lands about a second after the process: waited on as such, with a
    // wide deadline (the Code Coverage job is several times slower).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut claimed = false;
    while !claimed && tokio::time::Instant::now() < deadline {
        claimed = manager
            .get_active_background_tasks(&sid)
            .await
            .iter()
            .any(|t| t.id == "t7" && t.pid.is_some());
        if !claimed {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    let stopped = manager.cancel_task(&sid, "t7").await.unwrap();
    cc.check(
        "cancel_task",
        !stopped.killed_pids.is_empty(),
        Cause::Harness,
        format!(
            "pid réclamé={claimed}, killed_pids={:?}",
            stopped.killed_pids
        ),
    );
    let _ = std::fs::remove_file(bg_go(&cwd));
    w.settle(ends, &manager).await;

    // NATS: a message from the other instance.
    let ends = w.turn_ends();
    let sent = stage.rpc(&sid, &kc.k("NATS-SEND"), "user_message").await;
    cc_sent += 1;
    let played = cli.wait_line(0, &kc.k("NATS-SEND"), WAIT).await.is_some();
    cc.check(
        "nats.send",
        sent == Some(true) && played,
        Cause::Harness,
        format!("réponse rpc={sent:?}, message joué={played}"),
    );
    w.settle(ends, &manager).await;
    // A Stop from the other instance.
    let ends = w.turn_ends();
    manager
        .send_message(&sid, &kc.k("TURN-STALL"))
        .await
        .unwrap();
    cc_sent += 1;
    assert!(cli.wait_line(0, &kc.k("TURN-STALL"), WAIT).await.is_some());
    tokio::time::sleep(Duration::from_millis(300)).await;
    let before = cli.count(0, "\"interrupt\"");
    stage.other.publish_interrupt(&sid);
    let stopped_there = cli.wait_count(0, "\"interrupt\"", before, WAIT).await;
    // The turn the interruption closes is the slowest step of the scenario under
    // instrumentation: waited on with its own, wider deadline, not the global WAIT.
    let turn_closed = w.turn_end_after_within(ends, INTERRUPT_TURN_END_WAIT).await;
    cc.check(
        "nats.interrupt",
        stopped_there && turn_closed,
        Cause::Harness,
        format!("interrupt écrit au CLI={stopped_there}, tour clos={turn_closed}"),
    );
    w.settle(ends, &manager).await;
    // A permission answered from the other instance.
    let ends = w.turn_ends();
    manager
        .send_message(&sid, &kc.k("NATS-PERM"))
        .await
        .unwrap();
    cc_sent += 1;
    assert!(
        w.wait(
            WAIT,
            |e| matches!(e, ChatEvent::PermissionRequest { id, .. } if id == "req-nats")
        )
        .await
    );
    let routed = stage
        .rpc(
            &sid,
            &json!({"allow": true, "request_id": "req-nats"}).to_string(),
            "control_response",
        )
        .await;
    let answered = cli.wait_line(0, "req-nats", WAIT).await.unwrap_or_default();
    cc.check(
        "nats.permission_response",
        routed == Some(true) && answered.contains("\"behavior\":\"allow\""),
        Cause::Harness,
        format!("réponse rpc={routed:?}, écrit : {}", excerpt(&answered)),
    );
    if !answered.contains("\"behavior\"") {
        // Not delivered from there: answered here, so the scenario goes on.
        let _ = manager
            .route_permission_response(
                &sid,
                "req-nats",
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                false,
            )
            .await;
    }
    w.settle(ends, &manager).await;
    // A cancel_tools from the other instance.
    let ends = w.turn_ends();
    manager
        .send_message(&sid, &kc.k("NATS-CANCEL"))
        .await
        .unwrap();
    cc_sent += 1;
    assert!(
        w.wait(
            WAIT,
            |e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "t9")
        )
        .await
    );
    tokio::time::sleep(Duration::from_millis(400)).await;
    stage.other.publish_cancel_tools(&sid);
    let went_on = w.said(WAIT, "answered nats cancel").await;
    let announced = w
        .any(|e| matches!(e, ChatEvent::ToolsCancelled { .. }))
        .await;
    cc.check(
        "nats.cancel_tools",
        went_on,
        Cause::Harness,
        format!("outil arrêté et tour poursuivi={went_on}, tools_cancelled={announced}"),
    );
    w.settle(ends, &manager).await;

    // References.
    let ends = w.turn_ends();
    manager
        .send_message(
            &sid,
            &stage.refs_message(&format!("{} see #task", kc.k("TURN-REFS"))),
        )
        .await
        .unwrap();
    cc_sent += 1;
    let refs = cli
        .wait_line(0, &kc.k("TURN-REFS"), WAIT)
        .await
        .unwrap_or_default();
    cc.check(
        "refs",
        refs.contains("po-context") && refs.contains(TASK_TITLE) && !refs.contains("po-refs"),
        Cause::Harness,
        format!("ligne : {}", excerpt(&refs)),
    );
    w.settle(ends, &manager).await;

    // An attached image.
    let ends = w.turn_ends();
    let message = stage
        .image_message(&format!("{} what is on it?", kc.k("TURN-IMG")))
        .await;
    manager.send_message(&sid, &message).await.unwrap();
    cc_sent += 1;
    let img = cli
        .wait_line(0, &kc.k("TURN-IMG"), WAIT)
        .await
        .unwrap_or_default();
    let block = serde_json::from_str::<Value>(&img).ok().is_some_and(|l| {
        l["message"]["content"].as_array().is_some_and(|blocks| {
            blocks
                .iter()
                .any(|b| b["type"] == "image" && b["source"]["data"] == pixel_base64())
        })
    });
    cc.check(
        "images",
        block,
        Cause::Harness,
        format!("bloc image sur stdin={block} ; ligne : {}", excerpt(&img)),
    );
    w.settle(ends, &manager).await;

    // A question to the user: shown, closed by a denial that ends the turn (§15.6), answered by a turn.
    let ends = w.turn_ends();
    manager.send_message(&sid, &kc.k("TURN-ASK")).await.unwrap();
    cc_sent += 1;
    let shown = w
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::AskUserQuestion { tool_call_id, questions, .. }
                if tool_call_id == "t-ask" && questions.to_string().contains("Which database?"))
        })
        .await;
    let closing = cli.wait_line(0, "req-ask", WAIT).await.unwrap_or_default();
    let closed_by_denial =
        closing.contains("\"behavior\":\"deny\"") && closing.contains("posed to the user");
    w.settle(ends, &manager).await;
    let ends = w.turn_ends();
    // What the frontend sends for the answer (`input_response`): an inert user turn.
    manager
        .send_message(
            &sid,
            &crate::refs::compose::inert(&format!("{}: Postgres", kc.k("ANSWER"))),
        )
        .await
        .unwrap();
    cc_sent += 1;
    let answer_read = cli.wait_line(0, &kc.k("ANSWER"), WAIT).await.is_some();
    cc.check(
        "ask_user_question",
        shown
            && closed_by_denial
            && answer_read
            && w.said(WAIT, "answered ask").await,
        Cause::Harness,
        format!(
            "ask_user_question={shown}, fermée par un refus qui termine le tour={closed_by_denial}, réponse lue au tour suivant={answer_read}"
        ),
    );
    w.settle(ends, &manager).await;

    // A sub-agent: its messages under the Task call, its answer as the result.
    let ends = w.turn_ends();
    manager
        .send_message(&sid, &kc.k("TURN-AGENT"))
        .await
        .unwrap();
    cc_sent += 1;
    let nested = w
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::AssistantText { content, parent_tool_use_id: Some(p) }
                if p == "t-agent" && content.contains(&kc.k("SUB-SAID")))
        })
        .await;
    let returned = w
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::ToolResult { id, result, is_error: false, .. }
                if id == "t-agent" && result.to_string().contains(&kc.k("SUB-ANSWER")))
        })
        .await;
    cc.check(
        "subagents",
        nested && returned && w.said(WAIT, "answered agent").await,
        Cause::Harness,
        format!("messages du sous-agent sous t-agent={nested}, réponse en résultat={returned}"),
    );
    // ISO1: SubagentStop — the CLI's callback reached the session's hook, and the CLI was
    // answered.
    let sub_answered = cli.wait_line(0, "hook-subagent-stop", WAIT).await.is_some();
    let sub_called = *cc_answers.subagent_stop_called.lock().unwrap();
    cc.check(
        "hooks.subagent_stop",
        sub_answered && sub_called,
        Cause::Harness,
        format!("CLI répondu={sub_answered}, before_subagent_stop appelé={sub_called}"),
    );
    w.settle(ends, &manager).await;

    // A tool result of about a megabyte: the turn closes, the next one starts.
    let (ok, seen) = Box::pin(cc_huge(&manager, &w, &cli, kc, &sid)).await;
    cc_sent += 2;
    cc.check("tool_result.huge", ok, Cause::Harness, seen);

    // ISO3: `/compact <instructions>` from the chat goes to the CLI as the command.
    let (ok, seen) = Box::pin(cc_manual_compact(&manager, &w, &cli, kc, &sid, path)).await;
    cc_sent += 1;
    cc.check("compaction.manual", ok, Cause::Harness, seen);

    // The announcement and the record.
    let init = w.system_init().await;
    let unduly = init.as_ref().map(not_model_limits);
    cc.check(
        "system_init.degraded",
        unduly.as_ref().is_some_and(Vec::is_empty),
        Cause::Harness,
        format!("manques hors limites du modèle : {unduly:?}"),
    );
    let (ok, seen) = record_verdict(&stage.session_node(&sid).await, cc_sent);
    cc.check("session_record", ok, Cause::Harness, seen);

    // ----- the backend goes down, a new one starts on the same graph -----
    let _ = manager.close_session(&sid).await;
    drop(w);
    drop(manager);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let manager = stage.backend_with(&cli, &answers, &cc_answers, path).await;
    manager
        .resume_session(&sid, &format!("{}: go on", kc.k("RESUMED")), Some(&claims))
        .await
        .unwrap_or_else(|e| panic!("resuming the Claude Code session: {e:#}"));
    let w = Watch::start(
        stage.graph.clone(),
        &sid,
        manager.subscribe(&sid).await.unwrap(),
    );
    let resumed = cli.wait_line(1, &kc.k("RESUMED"), WAIT).await.is_some();
    let args = cli.args(1);
    let on = args
        .iter()
        .position(|a| a == "--resume")
        .and_then(|i| args.get(i + 1))
        .cloned();
    cc.check(
        "resume",
        resumed && on.as_deref() == Some("cc-sess-1") && w.said(WAIT, "answered resumed").await,
        Cause::Harness,
        format!("--resume {on:?}, message lu={resumed}"),
    );
    // ISO5: nothing configured, no turn time limit and no session budget: the CLI is never
    // given --max-budget-usd, and on the agent engine the session specs carry no limit.
    let budget_flags = (0..2)
        .flat_map(|n| cli.args(n))
        .filter(|a| a.starts_with("--max-budget"))
        .count();
    let spawned = !cli.args(0).is_empty() && !cli.args(1).is_empty();
    let (specs_ok, specs_seen) = if matches!(path, ProviderPath::Agent) {
        no_session_limits(&cc_answers, ProviderKind::ClaudeCode)
    } else {
        (true, "moteur historique : aucune SessionSpec".to_string())
    };
    cc.check(
        "session_limits",
        spawned && budget_flags == 0 && specs_ok,
        Cause::Harness,
        format!(
            "CLI lancé deux fois={spawned}, --max-budget* dans argv={budget_flags}, {specs_seen}"
        ),
    );
    w.settle(0, &manager).await;

    // ===================================================================
    // Claude Code → native: the conversation relayed (backend process #2)
    // ===================================================================
    let switched = manager
        .switch_session_provider(
            &sid,
            "local",
            None,
            "SWITCH-ONE: continue here",
            Some(parity_person()),
        )
        .await
        .unwrap_or_else(|e| panic!("switching to the native instance: {e:#}"));
    let nid = switched.session_id.clone();
    assert_eq!(nid, sid, "the SAME thread continues in place");
    assert!(
        manager.agent_runtime.owns(&nid).await,
        "native: agent engine"
    );
    let wn = Watch::start(
        stage.graph.clone(),
        &nid,
        manager.subscribe(&nid).await.unwrap(),
    );
    let relayed = wait_body(&fake, "SWITCH-ONE").await.unwrap_or_default();
    let on_wire = wn
        .wait(WAIT, |e| {
            matches!(
                e,
                ChatEvent::ProviderSwitched {
                    session_id,
                    from_provider,
                    to_provider,
                    ..
                } if session_id.as_str() == nid.as_str()
                    && from_provider == "claude-code"
                    && to_provider == "local"
            )
        })
        .await;
    let history = relayed.contains("<conversation_relay") && relayed.contains("answered one");
    na.check(
        "provider_switch.relay",
        history && on_wire,
        Cause::Harness,
        format!("historique relayé au modèle={history}, provider_switched={on_wire}"),
    );
    assert!(
        wn.said(WAIT, "answered switch one").await,
        "the relayed turn never answered"
    );
    wn.settle(0, &manager).await;
    let mut na_sent = 1i64;

    // TURN-ONE: enrichment, tools given, hooks around a nexus Grep.
    let ends = wn.turn_ends();
    manager
        .send_message(
            &nid,
            &format!("{}: where is the spec built?", kn.k("TURN-ONE")),
        )
        .await
        .unwrap();
    na_sent += 1;
    let one = wait_body(&fake, &kn.k("TURN-ONE"))
        .await
        .unwrap_or_default();
    let turn = last_user(&one).to_string();
    na.check(
        "enrichment",
        turn.contains(PLAN_TITLE),
        Cause::Harness,
        format!("message du tour : {}", excerpt(&turn)),
    );
    let answered_one = wn.said(WAIT, "answered one").await;
    let tools: Vec<String> = serde_json::from_str::<Value>(&one)
        .ok()
        .and_then(|b| b["tools"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
        .collect();
    let pre: Vec<String> = answers
        .before_tool
        .lock()
        .unwrap()
        .iter()
        .map(|(name, v)| format!("{name}: {v:?}"))
        .collect();
    na.check(
        "hooks.before_tool",
        pre.iter().any(|p| p.contains("Grep")),
        Cause::Harness,
        format!("before_tool appelé : {pre:?}"),
    );
    // The answer is recorded when the graph hook RETURNS. Read once, it was `[]` three times on
    // 19/10/2026 on CI runners (Coverage of #880 and of main, Unit Tests of main) while the turn
    // had answered and `before_tool` had seen the Grep; the cause is not established (the
    // native loop awaits `after_tool` within the turn, and no `hook_timeout` was logged). The
    // cell now waits for the answer like any other observation, within WAIT, and its message
    // says whether the hook was CALLED, so the next failure tells "never called" from "called,
    // no answer". A hook never called still fails.
    let after_tool_answered = || {
        answers
            .after_tool
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|a| a.contains("find_references") && a.contains("build_agent_spec"))
    };
    let deadline = tokio::time::Instant::now() + WAIT;
    while !after_tool_answered() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let called = answers.after_tool_called.lock().unwrap().clone();
    let after = answers.after_tool.lock().unwrap().clone();
    na.check(
        "hooks.after_tool",
        after_tool_answered(),
        Cause::Harness,
        format!("after_tool appelé pour : {called:?}, a répondu : {after:?}"),
    );
    eprintln!("parity: native tools offered: {tools:?}, turn one answered={answered_one}");
    wn.settle(ends, &manager).await;

    // Message queue.
    let ends = wn.turn_ends();
    manager.send_message(&nid, &kn.k("TURN-TWO")).await.unwrap();
    na_sent += 1;
    assert!(wait_body(&fake, &kn.k("TURN-TWO")).await.is_some());
    tokio::time::sleep(Duration::from_millis(300)).await;
    manager
        .send_message(&nid, &kn.k("TURN-THREE"))
        .await
        .unwrap();
    na_sent += 1;
    let three = wait_body(&fake, &kn.k("TURN-THREE")).await.is_some();
    na.check(
        "message_queue",
        three && wn.said(WAIT, "answered three").await,
        Cause::Harness,
        format!("message suivant envoyé au modèle={three}"),
    );
    wn.settle(ends, &manager).await;

    // Auto-continue: always on (decision 7b95342a), no toggle sent.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-FOUR"))
        .await
        .unwrap();
    na_sent += 1;
    let continued = wait_body_nth(&fake, "Continue where you left off", 0, WAIT)
        .await
        .is_some_and(|b| {
            last_user(&b)
                .to_string()
                .contains("Continue where you left off")
        });
    let announced = wn
        .wait(WAIT, |e| matches!(e, ChatEvent::AutoContinue { .. }))
        .await;
    na.check(
        "auto_continue",
        continued && announced,
        Cause::Harness,
        format!("continuation envoyée={continued}, auto_continue={announced}"),
    );
    wn.said(WAIT, "continued").await;
    wn.settle(ends, &manager).await;

    // `always` (P11b): offered to a person's session on a project, kept by the BACKEND (the
    // identical call is still asked by nexus, and answered from the lasting rule).
    let offers_always = match &wn.system_init().await {
        Some(ChatEvent::SystemInit { capabilities, .. }) => capabilities
            .as_ref()
            .and_then(|c| c.get("permission_scopes").cloned())
            .is_some_and(|s| s.to_string().contains("always")),
        _ => false,
    };
    std::fs::write(stage.dir.path().join("always.txt"), "ALWAYS-ONE").unwrap();
    let (_, granted_always) = Box::pin(native_grant(
        &manager,
        &wn,
        &fake,
        kn.k("TURN-ALWAYS"),
        "cat always.txt",
        "answered always",
        crate::chat::types::PermissionAnswerScope::Always,
    ))
    .await;
    na_sent += 1;
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-ALWAYS2"))
        .await
        .unwrap();
    na_sent += 1;
    // The identical call is asked by nexus again (no rule was handed to it): wait for that
    // second request, answered by the backend or left to the user.
    let deadline = tokio::time::Instant::now() + WAIT;
    let mut asked_always = wn.permission_ids("cat always.txt").await;
    while asked_always.len() < 2 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
        asked_always = wn.permission_ids("cat always.txt").await;
    }
    // Leave the backend a moment to answer it.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = wn.all().await;
    let always_by_backend = asked_always.len() == 2
        && events.iter().any(|e| {
            matches!(e, ChatEvent::PermissionDecision { id, scope: Some(s), .. }
                if *id == asked_always[1]
                    && *s == crate::chat::types::PermissionAnswerScope::Always)
        });
    for id in asked_always.iter().skip(1).filter(|_| !always_by_backend) {
        let _ = manager
            .route_permission_response(
                &nid,
                id,
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                false,
            )
            .await;
    }
    let ran_always = wait_body_nth(&fake, &kn.k("TURN-ALWAYS2"), 1, WAIT)
        .await
        .is_some();
    wn.said(WAIT, "answered always two").await;
    wn.settle(ends, &manager).await;
    let na_kept = lasting_rules_of(&manager, "mcp__nexus__Bash: cat always.txt");
    // Measured BEFORE this session runs a command that may move its shell (`echo
    // FIVE-$((1+1))` below): from then on `always` is neither offered nor kept for a
    // command here (review of #714, A). The identical call is answered by the backend in
    // this session, and the next session of the person in the project's root is too.
    let next_session = Box::pin(next_session_covers(
        &manager,
        &cwd,
        crate::chat::session_grants::Asker::Native,
        "mcp__nexus__Bash",
        Some("Bash"),
        "cat always.txt",
    ))
    .await;
    na.check(
        "permissions.always",
        offers_always
            && granted_always
            && ran_always
            && always_by_backend
            && na_kept
            && next_session,
        Cause::Harness,
        format!(
            "portée offerte={offers_always}, accordée={granted_always}, règle gardée pour la \
             personne et la racine du projet={na_kept}, appel identique répondu par le backend \
             (always)={always_by_backend}, couvre la session suivante={next_session}"
        ),
    );

    // set_model, then a permission answered once.
    manager.set_session_model(&nid, "m2").await.unwrap();
    let changed = wn
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::ModelChanged { model } if model == "m2"),
        )
        .await;
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-FIVE"))
        .await
        .unwrap();
    na_sent += 1;
    let five = wait_body(&fake, &kn.k("TURN-FIVE"))
        .await
        .unwrap_or_default();
    let on_m2 = serde_json::from_str::<Value>(&five)
        .ok()
        .is_some_and(|b| b["model"] == "m2");
    na.check(
        "set_model",
        changed && on_m2,
        Cause::Harness,
        format!("model_changed={changed}, requête suivante sur m2={on_m2}"),
    );
    let id = wn.permission_id(WAIT, "FIVE-$").await;
    if let Some(id) = &id {
        manager
            .route_permission_response(
                &nid,
                id,
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                false,
            )
            .await
            .unwrap();
    }
    let ran = wait_body_nth(&fake, &kn.k("TURN-FIVE"), 1, WAIT)
        .await
        .is_some_and(|b| b.contains("FIVE-2"));
    na.check(
        "permissions.once",
        id.is_some() && ran && wn.said(WAIT, "answered five").await,
        Cause::Harness,
        format!(
            "demandée={}, outil exécuté après accord={ran}",
            id.is_some()
        ),
    );
    wn.settle(ends, &manager).await;
    let (timed, seen) = timing_verdict(&wn.persisted().await, "c5");
    na.check("tool_timing", timed, Cause::Harness, seen);
    // A tool granted for the session: asked once, then not again in this session.
    let (first, granted_session) = Box::pin(native_grant(
        &manager,
        &wn,
        &fake,
        kn.k("TURN-PERM2"),
        "AGAIN-SESSION",
        "answered perm2",
        crate::chat::types::PermissionAnswerScope::Session,
    ))
    .await;
    na_sent += 1;
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-PERM3"))
        .await
        .unwrap();
    na_sent += 1;
    // The identical call: the backend answers it (the tool runs, nobody is asked).
    let ran_again = wait_body_nth(&fake, &kn.k("TURN-PERM3"), 1, WAIT)
        .await
        .is_some();
    // The backend answers the second request itself (its decision says `session`); a
    // request it left to the user would have no such decision.
    let asked = wn.permission_ids("AGAIN-SESSION").await;
    let events = wn.all().await;
    let answered_by_backend = |id: &String| {
        events.iter().any(|e| {
            matches!(e, ChatEvent::PermissionDecision { id: d, scope: Some(s), .. }
                if d == id && *s == crate::chat::types::PermissionAnswerScope::Session)
        })
    };
    let asked_again = !ran_again || asked.iter().skip(1).any(|id| !answered_by_backend(id));
    for id in asked.iter().skip(1).filter(|id| !answered_by_backend(id)) {
        let _ = manager
            .route_permission_response(
                &nid,
                id,
                true,
                crate::chat::types::PermissionAnswerScope::Once,
                false,
            )
            .await;
    }
    na.check(
        "permissions.session",
        first.is_some() && granted_session && !asked_again,
        Cause::Harness,
        format!(
            "demandée={}, accordée pour la session={granted_session}, redemandée={asked_again}",
            first.is_some()
        ),
    );
    wn.said(WAIT, "answered perm3").await;
    wn.settle(ends, &manager).await;

    // cancel_tools.
    let ends = wn.turn_ends();
    manager.send_message(&nid, &kn.k("TURN-SIX")).await.unwrap();
    na_sent += 1;
    assert!(
        wn.wait(
            WAIT,
            |e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "c6")
        )
        .await
    );
    tokio::time::sleep(Duration::from_millis(400)).await;
    let cancelled = manager.cancel_running_tools(&nid).await.unwrap();
    let announced = wn
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::ToolsCancelled { killed_count, .. } if *killed_count > 0),
        )
        .await;
    let went_on = wn.said(WAIT, "answered six").await;
    na.check(
        "cancel_tools",
        announced && went_on,
        Cause::Harness,
        format!(
            "tools_cancelled={announced}, tour poursuivi={went_on}, résultat={:?}",
            cancelled.killed_pids
        ),
    );
    wn.settle(ends, &manager).await;

    // A background task, then cancel_task.
    let ends = wn.turn_ends();
    manager.send_message(&nid, &kn.k("TURN-BG")).await.unwrap();
    na_sent += 1;
    answer_if_asked(&manager, &wn, &fake, "sleep 30", &kn.k("TURN-BG"), 1).await;
    let tracked = wn
        .wait(
            WAIT,
            |e| matches!(e, ChatEvent::ActiveTasksUpdate { tasks } if !tasks.is_empty()),
        )
        .await;
    na.check(
        "background_tasks",
        tracked,
        Cause::Harness,
        format!("active_tasks_update={tracked}"),
    );
    let stopped = manager.cancel_task(&nid, "c7").await;
    na.check(
        "cancel_task",
        stopped.as_ref().is_ok_and(|s| !s.killed_pids.is_empty()),
        Cause::Harness,
        match &stopped {
            Ok(s) => format!("killed_pids={:?}", s.killed_pids),
            Err(e) => format!("refusé : {e}"),
        },
    );
    wn.said(WAIT, "answered bg").await;
    wn.settle(ends, &manager).await;

    // NATS: a message from the other instance.
    let ends = wn.turn_ends();
    let sent = stage.rpc(&nid, &kn.k("NATS-SEND"), "user_message").await;
    na_sent += 1;
    let played = wait_body(&fake, &kn.k("NATS-SEND")).await.is_some();
    na.check(
        "nats.send",
        sent == Some(true) && played,
        Cause::Harness,
        format!("réponse rpc={sent:?}, message joué={played}"),
    );
    wn.settle(ends, &manager).await;
    // A Stop from the other instance.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-STALL"))
        .await
        .unwrap();
    na_sent += 1;
    assert!(wait_body(&fake, &kn.k("TURN-STALL")).await.is_some());
    tokio::time::sleep(Duration::from_millis(300)).await;
    stage.other.publish_interrupt(&nid);
    let stopped_there = wn.turn_end_after(ends).await;
    na.check(
        "nats.interrupt",
        stopped_there,
        Cause::Harness,
        format!("tour arrêté={stopped_there}"),
    );
    wn.settle(ends, &manager).await;
    // A permission answered from the other instance.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("NATS-PERM"))
        .await
        .unwrap();
    na_sent += 1;
    let id = wn.permission_id(WAIT, "NATS-$").await;
    let routed = match &id {
        Some(id) => {
            stage
                .rpc(
                    &nid,
                    &json!({"allow": true, "request_id": id}).to_string(),
                    "control_response",
                )
                .await
        }
        None => None,
    };
    let ran = wait_body_nth(&fake, &kn.k("NATS-PERM"), 1, WAIT)
        .await
        .is_some_and(|b| b.contains("NATS-6"));
    na.check(
        "nats.permission_response",
        routed == Some(true) && ran,
        Cause::Harness,
        format!("réponse rpc={routed:?}, outil exécuté={ran}"),
    );
    wn.settle(ends, &manager).await;
    // A cancel_tools from the other instance.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("NATS-CANCEL"))
        .await
        .unwrap();
    na_sent += 1;
    assert!(
        wn.wait(
            WAIT,
            |e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "c9")
        )
        .await
    );
    tokio::time::sleep(Duration::from_millis(400)).await;
    stage.other.publish_cancel_tools(&nid);
    let went_on = wn.said(WAIT, "answered nats cancel").await;
    let announced = wn
        .any(|e| matches!(e, ChatEvent::ToolsCancelled { .. }))
        .await;
    na.check(
        "nats.cancel_tools",
        went_on,
        Cause::Harness,
        format!("outil arrêté et tour poursuivi={went_on}, tools_cancelled={announced}"),
    );
    wn.settle(ends, &manager).await;

    // nexus-tools: Read, Edit, Bash on the project.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-NEXUS"))
        .await
        .unwrap();
    na_sent += 1;
    answer_if_asked(&manager, &wn, &fake, "AFTER-EDIT", &kn.k("TURN-NEXUS"), 2).await;
    let read = wait_body_nth(&fake, &kn.k("TURN-NEXUS"), 1, WAIT)
        .await
        .is_some_and(|b| b.contains("PARITY-FILE"));
    wn.said(WAIT, "answered nexus").await;
    let edited = std::fs::read_to_string(stage.dir.path().join("parity.txt"))
        .unwrap_or_default()
        .contains("AFTER-EDIT");
    wn.settle(ends, &manager).await;
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-BASH"))
        .await
        .unwrap();
    na_sent += 1;
    answer_if_asked(&manager, &wn, &fake, "BASH-RAN", &kn.k("TURN-BASH"), 1).await;
    let bash = wait_body_nth(&fake, &kn.k("TURN-BASH"), 1, WAIT)
        .await
        .is_some_and(|b| b.contains("BASH-RAN-42"));
    na.check(
        "nexus_tools",
        read && edited && bash,
        Cause::Harness,
        format!("Read={read}, Edit={edited}, Bash={bash}, outils offerts={tools:?}"),
    );
    wn.said(WAIT, "answered bash").await;
    wn.settle(ends, &manager).await;
    // The PO tools: offered, and the calls above answered by the PO server.
    let echoed = bodies(&fake).iter().any(|b| b.contains("echo: four-a"));
    na.check(
        "po_tools",
        tools.iter().any(|t| t == &po("echo")) && echoed,
        Cause::Harness,
        format!(
            "offerts={}, appel servi={echoed}",
            tools.iter().any(|t| t == &po("echo"))
        ),
    );
    // Deferred tools (CTX2): `readonly` is deferred for this manager (see `backend`). Its
    // definition is withheld and ToolSearch offered instead; once loaded, it is sent
    // appended at the END of the list, and the call in the same turn is served.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-SEARCH"))
        .await
        .unwrap();
    na_sent += 1;
    let names = |body: &str| -> Vec<String> {
        serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|b| b["tools"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect()
    };
    let first = wait_body_nth(&fake, &kn.k("TURN-SEARCH"), 0, WAIT)
        .await
        .map(|b| names(&b))
        .unwrap_or_default();
    let withheld = !first.iter().any(|t| t == &po("readonly"));
    let offered_search = first.iter().any(|t| t == "ToolSearch");
    let second = wait_body_nth(&fake, &kn.k("TURN-SEARCH"), 1, WAIT)
        .await
        .map(|b| names(&b))
        .unwrap_or_default();
    let loaded_last = second.last().is_some_and(|t| t == &po("readonly"));
    let prefix_kept = second.len() == first.len() + 1 && second[..first.len()] == first[..];
    let served = wait_body_nth(&fake, &kn.k("TURN-SEARCH"), 2, WAIT)
        .await
        .is_some_and(|b| b.contains("readonly ok"));
    let answered = wn.said(WAIT, "answered search").await;
    na.check(
        "tool_search",
        withheld && offered_search && loaded_last && prefix_kept && served && answered,
        Cause::Harness,
        format!(
            "différé retenu={withheld}, ToolSearch offert={offered_search}, chargé en fin de liste={loaded_last}, \
             préfixe inchangé={prefix_kept}, appel servi={served}, tour clos={answered} ; \
             1re liste={} outils, 2e={}",
            first.len(),
            second.len()
        ),
    );
    wn.settle(ends, &manager).await;

    // References.
    let ends = wn.turn_ends();
    manager
        .send_message(
            &nid,
            &stage.refs_message(&format!("{} see #task", kn.k("TURN-REFS"))),
        )
        .await
        .unwrap();
    na_sent += 1;
    let refs = wait_body(&fake, &kn.k("TURN-REFS"))
        .await
        .map(|b| last_user(&b).to_string())
        .unwrap_or_default();
    na.check(
        "refs",
        refs.contains("po-context") && refs.contains(TASK_TITLE) && !refs.contains("po-refs"),
        Cause::Harness,
        format!("message du tour : {}", excerpt(&refs)),
    );
    wn.settle(ends, &manager).await;

    // An attached image (the model of the instance has vision).
    let ends = wn.turn_ends();
    let message = stage
        .image_message(&format!("{} what is on it?", kn.k("TURN-IMG")))
        .await;
    manager.send_message(&nid, &message).await.unwrap();
    na_sent += 1;
    let img = wait_body(&fake, &kn.k("TURN-IMG"))
        .await
        .map(|b| last_user(&b))
        .unwrap_or(Value::Null);
    let part = img["content"].as_array().is_some_and(|parts| {
        parts.iter().any(|p| {
            p["type"] == "image_url"
                && p["image_url"]["url"] == format!("data:image/png;base64,{}", pixel_base64())
        })
    });
    na.check(
        "images",
        part,
        Cause::Harness,
        format!("partie image_url={part}"),
    );
    wn.settle(ends, &manager).await;

    // A question to the user: the harness's AskUserQuestion, shown as a synthetic card
    // (A45); the turn ends on it and the answer is the next turn.
    let ends = wn.turn_ends();
    manager.send_message(&nid, &kn.k("TURN-ASK")).await.unwrap();
    na_sent += 1;
    let shown = wn
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::AskUserQuestion { tool_call_id, questions, synthetic: Some(true), .. }
                if tool_call_id == "cq1" && questions.to_string().contains("Which database?"))
        })
        .await;
    let ended = wn.turn_end_after(ends).await;
    // The model was not asked again within the turn: the question ended it.
    let asked_once = bodies(&fake)
        .iter()
        .filter(|b| b.contains(&kn.k("TURN-ASK")) && !b.contains(&kn.k("ANSWER")))
        .count()
        == 1;
    wn.settle(ends, &manager).await;
    let ends = wn.turn_ends();
    manager
        .send_message(
            &nid,
            &crate::refs::compose::inert(&format!("{}: Postgres", kn.k("ANSWER"))),
        )
        .await
        .unwrap();
    na_sent += 1;
    let answer_body = wait_body(&fake, &kn.k("ANSWER")).await.unwrap_or_default();
    // The answer follows the question and its result in what the model reads.
    let after_question = answer_body.contains("next user message")
        && last_user(&answer_body)
            .to_string()
            .contains(&kn.k("ANSWER"));
    na.check(
        "ask_user_question",
        shown && ended && asked_once && after_question && wn.said(WAIT, "answered ask").await,
        Cause::Harness,
        format!(
            "ask_user_question synthétique={shown}, tour terminé={ended}, une seule requête={asked_once}, \
             réponse au tour suivant après la question={after_question}"
        ),
    );
    wn.settle(ends, &manager).await;

    // A sub-agent: the harness's Agent runs as a child session (A18) that cannot delegate.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-AGENT"))
        .await
        .unwrap();
    na_sent += 1;
    let parent_body = wait_body(&fake, &kn.k("TURN-AGENT"))
        .await
        .unwrap_or_default();
    let offered = |body: &str| -> Vec<String> {
        serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|b| b["tools"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect()
    };
    let returned = wn
        .wait(WAIT, |e| {
            matches!(e, ChatEvent::ToolResult { id, result, is_error: false, .. }
                if id == "ca1" && result.to_string().contains(&kn.k("SUB-ANSWER")))
        })
        .await;
    // The child's own request: its prompt, without the parent's turn.
    let child_body = bodies(&fake)
        .into_iter()
        .find(|b| b.contains(&kn.k("SUB-PROMPT")) && !b.contains(&kn.k("TURN-AGENT")))
        .unwrap_or_default();
    let parent_offers = offered(&parent_body).iter().any(|t| t == "Agent");
    let child_tools = offered(&child_body);
    let child_cannot_delegate =
        !child_tools.is_empty() && !child_tools.iter().any(|t| t == "Agent");
    let children = stage
        .graph
        .get_session_children(Uuid::parse_str(&nid).unwrap())
        .await
        .unwrap_or_default();
    let in_tree = children.iter().any(|c| {
        c.spawned_by
            .as_deref()
            .and_then(super::types::SpawnedBy::from_json_str)
            .is_some_and(|s| {
                matches!(s, super::types::SpawnedBy::Conversation { tool_use_id: Some(ref t), .. } if t == "ca1")
            })
    });
    na.check(
        "subagents",
        parent_offers
            && returned
            && child_cannot_delegate
            && in_tree
            && wn.said(WAIT, "answered agent").await,
        Cause::Harness,
        format!(
            "Agent offert au parent={parent_offers}, réponse de l'enfant en résultat={returned}, \
             session enfant dans l'arbre={in_tree}, outils de l'enfant sans Agent={child_cannot_delegate} \
             ({} outils)",
            child_tools.len()
        ),
    );
    // ISO2: subagents.types — the Agent(subagent_type: "explore") call above succeeded:
    // the type was resolved by resolve_subagent_type, the child session was created and
    // returned a result. If the type were unknown, run_subagent would have returned an
    // error containing "unknown subagent_type" and `returned` would be false.
    na.check(
        "subagents.types",
        returned && in_tree,
        Cause::Harness,
        format!(
            "Agent(subagent_type: explore) accepté : enfant créé={in_tree}, résultat reçu={returned}"
        ),
    );
    wn.settle(ends, &manager).await;

    // ISO1: TURN-STOP — the model answers, the stop hook blocks once, the model receives
    // the feedback as a user message and answers again.
    *answers.stop_block_once.lock().unwrap() = Some("keep going".to_string());
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-STOP"))
        .await
        .unwrap();
    na_sent += 1;
    // The first model call answers "partially done"; the stop hook blocks with "keep going";
    // the native loop pushes stop_hook_feedback("keep going") as a user message and calls
    // the model again; the second call answers "answered stop" (keyed on "Stop hook feedback").
    let stop_answered = wn.said(WAIT, "answered stop").await;
    let stop_replies = answers.stop_replies.lock().unwrap().clone();
    let blocked_once = stop_replies.iter().any(|r| r.block_reason().is_some());
    // The feedback must have reached the model: the second request contains the feedback.
    let feedback_sent = wait_body(&fake, "Stop hook feedback").await.is_some();
    na.check(
        "hooks.stop",
        stop_answered && blocked_once && feedback_sent,
        Cause::Harness,
        format!(
            "answered={stop_answered}, hook a bloqué={blocked_once}, feedback envoyé au modèle={feedback_sent}, \
             stop_replies={stop_replies:?}"
        ),
    );
    wn.settle(ends, &manager).await;

    // A tool result of about a megabyte: persisted, a bounded preview reaches the model, the
    // turn closes and the next one starts.
    let (ok, seen) = Box::pin(native_huge(&manager, &wn, &fake, kn, &nid)).await;
    na_sent += 2;
    na.check("tool_result.huge", ok, Cause::Harness, seen);

    // Compaction: a turn reports a context near the window; the next one compacts.
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-PRIME"))
        .await
        .unwrap();
    na_sent += 1;
    wn.settle(ends, &manager).await;
    let ends = wn.turn_ends();
    manager
        .send_message(&nid, &kn.k("TURN-COMPACT"))
        .await
        .unwrap();
    na_sent += 1;
    let summary = wait_body_nth(&fake, "Additional instructions", 0, WAIT)
        .await
        .unwrap_or_default();
    wn.said(WAIT, "answered compact").await;
    wn.settle(ends, &manager).await;
    let guidance = answers.before_compaction.lock().unwrap().clone();
    na.check(
        "hooks.before_compaction",
        guidance
            .iter()
            .flatten()
            .any(|g| g.contains(&stage.project.name))
            && summary.contains(&stage.project.name),
        Cause::Harness,
        format!("guidance du hook : {guidance:?}"),
    );
    let started = wn
        .any(|e| matches!(e, ChatEvent::CompactionStarted { .. }))
        .await;
    let boundary = wn
        .any(|e| matches!(e, ChatEvent::CompactBoundary { .. }))
        .await;
    na.check(
        "compaction",
        started && boundary,
        Cause::Harness,
        format!("compaction_started={started}, compact_boundary={boundary}"),
    );

    // ISO3: `/compact <instructions>` from the chat: compacted out of turn on request.
    let (ok, seen) = Box::pin(native_manual_compact(
        &manager,
        &wn,
        &fake,
        kn,
        &nid,
        &stage.project.name,
    ))
    .await;
    na_sent += 1;
    na.check("compaction.manual", ok, Cause::Harness, seen);

    // ISO1: SessionStart — the hook fired at session open (contract v7).
    let ss = answers.session_start_sources.lock().unwrap().clone();
    na.check(
        "hooks.session_start",
        !ss.is_empty(),
        Cause::Harness,
        format!("session_start appelé {} fois, sources={ss:?}", ss.len()),
    );
    // ISO1: UserPromptSubmit — the hook fired before each turn (contract v7).
    let prompts = answers.user_prompt_submit_prompts.lock().unwrap().clone();
    na.check(
        "hooks.user_prompt_submit",
        !prompts.is_empty(),
        Cause::Harness,
        format!(
            "user_prompt_submit appelé {} fois, prompts={:?}",
            prompts.len(),
            prompts.iter().map(|p| excerpt(p)).collect::<Vec<_>>()
        ),
    );
    // ISO1: SubagentStop — the hook fired when the sub-agent finished (contract v7).
    let sub_called = *answers.subagent_stop_called.lock().unwrap();
    na.check(
        "hooks.subagent_stop",
        sub_called,
        Cause::Harness,
        format!("before_subagent_stop appelé={sub_called}"),
    );
    // ISO1: Notification — the hook fired when a permission was pending (contract v7).
    let notif = answers.notification_types.lock().unwrap().clone();
    na.check(
        "hooks.notification",
        notif.iter().any(|t| t == "permission_prompt"),
        Cause::Harness,
        format!("notification appelé {} fois, types={notif:?}", notif.len()),
    );

    // The announcement (hard rule, below) and the record.
    let unduly = init.as_ref().map(not_model_limits);
    na.check(
        "system_init.degraded",
        unduly.as_ref().is_some_and(Vec::is_empty),
        Cause::Harness,
        format!("manques hors limites du modèle : {unduly:?}"),
    );
    let (ok, seen) = record_verdict(&stage.session_node(&nid).await, na_sent);
    na.check("session_record", ok, Cause::Harness, seen);

    // ----- the backend goes down, a new one starts on the same graph -----
    let _ = manager.close_session(&nid).await;
    drop(wn);
    drop(w);
    drop(manager);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let manager = stage.backend(&cli, &answers, path).await;
    let reopened = manager
        .resume_session(&nid, &format!("{}: go on", kn.k("RESUMED")), Some(&claims))
        .await;
    match reopened {
        Ok(()) => {
            let wn = Watch::start(
                stage.graph.clone(),
                &nid,
                manager.subscribe(&nid).await.unwrap(),
            );
            let resumed = wait_body(&fake, &kn.k("RESUMED")).await.unwrap_or_default();
            // The resumed transcript holds the conversation before the restart.
            let knows = resumed.contains("answered compact") || resumed.contains("the summary");
            na.check(
                "resume",
                knows && wn.said(WAIT, "answered resumed").await,
                Cause::Harness,
                format!("historique connu après redémarrage={knows}"),
            );
            wn.settle(0, &manager).await;
        }
        Err(e) => na.check(
            "resume",
            false,
            Cause::Harness,
            format!("reprise refusée : {e:#}"),
        ),
    }
    // ISO5: nothing configured, the native sessions (opened, then resumed after the
    // restart) carry no turn time limit and no session budget.
    let (limits_ok, limits_seen) = no_session_limits(&answers, ProviderKind::Native);
    na.check("session_limits", limits_ok, Cause::Harness, limits_seen);

    // ===================================================================
    // native → Claude Code: the conversation relayed back (process #3)
    // ===================================================================
    let back = manager
        .switch_session_provider(&nid, "claude-code", None, "SWITCH-TWO: and back", None)
        .await
        .unwrap_or_else(|e| panic!("switching back to Claude Code: {e:#}"));
    assert_eq!(back.session_id, nid, "the SAME thread continues in place");
    let wb = Watch::start(
        stage.graph.clone(),
        &back.session_id,
        manager.subscribe(&back.session_id).await.unwrap(),
    );
    let line = cli
        .wait_line(2, "SWITCH-TWO", WAIT)
        .await
        .unwrap_or_default();
    let on_wire = wb
        .wait(WAIT, |e| {
            matches!(
                e,
                ChatEvent::ProviderSwitched {
                    session_id,
                    from_provider,
                    to_provider,
                    ..
                } if session_id.as_str() == back.session_id.as_str()
                    && from_provider == "local"
                    && to_provider == "claude-code"
            )
        })
        .await;
    let history = line.contains("<conversation_relay") && line.contains("answered compact");
    cc.check(
        "provider_switch.relay",
        history && on_wire,
        Cause::Harness,
        format!("historique relayé au CLI={history}, provider_switched={on_wire}"),
    );
    wb.said(WAIT, "answered switch two").await;
    let _ = manager.close_session(&back.session_id).await;

    // ===================================================================
    // native over the Anthropic Messages API (CTX5): an `anthropic` instance
    // ===================================================================
    let (ok, seen) = anthropic_turn(&stage, &manager, &cwd).await;
    na.check("anthropic_messages", ok, Cause::Harness, seen);

    // The hard rule: the native harness announces only limits of the MODEL.
    if let Some(unduly) = &unduly {
        assert!(
            unduly.is_empty(),
            "the native system_init announces missing features that are not limits of the \
             model: {unduly:?} (closed list: images without vision; an unknown context window)"
        );
    }
    // ISO2: cells that EXPECTED declares (NotMeasured, or a declared Gap) but that this
    // engine's scenario does not exercise must still appear in the measured matrix —
    // `audit` reports an absent cell as "not measured at all". Fill ONLY the cells the
    // scenario left unset, so a measured cell is never set twice (`Measures::set` panics).
    let cc_engine = match path {
        ProviderPath::Legacy => Engine::ClaudeCode,
        ProviderPath::Agent => Engine::ClaudeCodeAgent,
    };
    fill_missing_declared(&mut cc, cc_engine);
    fill_missing_declared(&mut na, Engine::Native);
    (cc, na)
}

// ============================================================================
// The native harness over the Anthropic Messages API (CTX5)
// ============================================================================

/// Id of the `anthropic` instance of the stage, and the vault entry of its key.
const ANTHROPIC_ID: &str = "claude-api";
const ANTHROPIC_VAULT_KEY: &str = "anthropic-key";
/// A key-shaped value that must never be seen anywhere but in the request header.
const ANTHROPIC_KEY: &str = "sk-ant-parity-0123456789abcdef";

/// The vault of the stage: unlocked, the key of the `anthropic` instance granted to it.
async fn anthropic_vault() -> Arc<crate::vault::VaultService> {
    use crate::vault::grants::{GrantScope, SecretSelector};
    let vault = crate::vault::VaultService::ephemeral();
    vault
        .init(
            "correct horse battery staple".into(),
            chrono::Duration::hours(1),
        )
        .await
        .unwrap();
    let now = Utc::now();
    vault
        .put(ANTHROPIC_VAULT_KEY, ANTHROPIC_KEY, None, now)
        .unwrap();
    vault
        .grant(
            SecretSelector::Names([ANTHROPIC_VAULT_KEY.to_string()].into()),
            GrantScope::Provider(ANTHROPIC_ID.to_string()),
            chrono::Duration::hours(1),
            None,
            now,
        )
        .unwrap();
    vault
}

/// A fake Anthropic API: the catalogue, and a Messages answer framed as the real one.
fn anthropic_routes() -> Value {
    let events = vec![
        json!({"type": "message_start", "message": {"id": "msg_p", "type": "message",
            "role": "assistant", "content": [], "model": "m",
            "usage": {"input_tokens": 20, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "answered anthropic"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3}}),
        json!({"type": "message_stop"}),
    ];
    json!([
        {"method": "GET", "path": "/v1/models", "status": 200,
         "body": {"data": [{"id": "m", "max_input_tokens": 200000,
            "capabilities": {"image_input": {"supported": true}}}], "has_more": false}},
        {"method": "POST", "path": "/v1/messages", "status": 200,
         "sse_event_names": true, "sse": events},
    ])
}

/// Every JSON path of `body` that carries a `cache_control`.
fn cache_points(body: &Value) -> Vec<String> {
    fn walk(value: &Value, path: String, out: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                for (key, inner) in map {
                    if key == "cache_control" {
                        out.push(path.clone());
                    } else {
                        walk(inner, format!("{path}.{key}"), out);
                    }
                }
            }
            Value::Array(items) => {
                for (index, inner) in items.iter().enumerate() {
                    walk(inner, format!("{path}[{index}]"), out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(body, String::new(), &mut out);
    out
}

/// One turn on an `anthropic` instance, opened by the manager like any instance (stored
/// record, consent, key in the vault): `ok` when the model received ONE request on
/// `/v1/messages` whose `cache_control` points are exactly the last tool, the system
/// block and the last block of the last message, and the answer came back.
async fn anthropic_turn(stage: &Stage, manager: &Arc<ChatManager>, cwd: &str) -> (bool, String) {
    use super::provider::settings::{InstanceRecord, KIND_ANTHROPIC};
    let fake = FakeOpenAi::start(anthropic_routes());
    let record = InstanceRecord {
        id: ANTHROPIC_ID.into(),
        kind: KIND_ANTHROPIC.into(),
        label: "Claude API".into(),
        base_url: fake.base_url(),
        origin: fake.origin(),
        default_model: Some("m".into()),
        cost_source: "unknown".into(),
        credential_ref: format!("vault:{ANTHROPIC_VAULT_KEY}"),
        ..Default::default()
    };
    store_instance(&stage.graph, &record).await;
    consent(&stage.graph, "proj", ANTHROPIC_ID, &fake.origin()).await;
    let mut req = request(Some(ANTHROPIC_ID), Some("proj"), "default");
    req.cwd = cwd.to_string();
    req.message = "ANTHROPIC-TURN: say hello".into();
    req.user_claims = Some(parity_person());
    let sid = match manager.create_session(&req).await {
        Ok(created) => created.session_id,
        Err(e) => return (false, format!("session anthropic refusée : {e:#}")),
    };
    let w = Watch::start(
        stage.graph.clone(),
        &sid,
        manager.subscribe(&sid).await.unwrap(),
    );
    let answered = w.said(WAIT, "answered anthropic").await;
    let posts: Vec<Value> = fake
        .requests()
        .into_iter()
        .filter(|r| r["method"] == "POST" && r["path"] == "/v1/messages")
        .collect();
    w.settle(0, manager).await;
    let _ = manager.close_session(&sid).await;
    match anthropic_messages_verdict(&posts, answered, ANTHROPIC_KEY) {
        Ok(verdict) => verdict,
        Err(why) => panic!(
            "{why}\n  faux utilisé : {}\n  reconstruire NEXUS_FAKES_DIR à la révision nexus \
             épinglée dans Cargo.toml (la CI le fait dans .github/actions/nexus-fakes)",
            fakes_dir().display()
        ),
    }
}

/// The verdict of the `anthropic_messages` cell, from what `fake_openai` recorded.
///
/// `Err` means the measurement is IMPOSSIBLE, not that the harness is at fault. A
/// `fake_openai` older than the nexus pin that records `header_values` reports
/// `authorization_present`, `headers`, `body`, `method` and `path` — and nothing else
/// (measured 28/10/2026 on the 17/10 build of `.target-nexus-fakes/debug`). Everything
/// that depends on headers then goes blind: `anthropic-version` and the API key read as
/// ABSENT while the harness sends both, and the cell is imputed to the HARNESS on a
/// `main` whose CI is green. The caller refuses to conclude rather than lie.
fn anthropic_messages_verdict(
    posts: &[Value],
    answered: bool,
    key: &str,
) -> Result<(bool, String), &'static str> {
    let Some(post) = posts.first() else {
        return Ok((false, "aucune requête POST /v1/messages".to_string()));
    };
    if post.get("header_values").is_none() {
        return Err(
            "fake_openai ne rapporte pas les en-têtes de la requête (`header_values` absent de \
             son relevé) : ce faux précède la révision nexus qui les enregistre, donc \
             anthropic-version et la clé se liraient ABSENTES alors que le harnais les envoie, \
             et l'écart serait imputé au harnais à tort",
        );
    }
    let body = &post["body"];
    let last = body["messages"]
        .as_array()
        .map_or(0, |m| m.len().saturating_sub(1));
    let last_block = body["messages"][last]["content"]
        .as_array()
        .map_or(0, |c| c.len().saturating_sub(1));
    let tools = body["tools"].as_array().map_or(0, Vec::len);
    let expected = vec![
        format!(".messages[{last}].content[{last_block}]"),
        ".system[0]".to_string(),
        format!(".tools[{}]", tools.saturating_sub(1)),
    ];
    let points = cache_points(body);
    let version = post["header_values"]["anthropic-version"] == "2023-06-01";
    let key_sent = post["authorization_present"] == true || post["x_api_key_present"] == true;
    let leaked = post.to_string().contains(key);
    let ok = posts.len() == 1
        && answered
        && tools > 0
        && points == expected
        && version
        && key_sent
        && !leaked;
    Ok((
        ok,
        format!(
            "requêtes /v1/messages={}, réponse={answered}, outils={tools}, points={points:?}, \
             anthropic-version={version}, clé envoyée={key_sent}, clé journalisée={leaked}",
            posts.len()
        ),
    ))
}

// ============================================================================
// The audit itself: the two ways the matrix turns red
// ============================================================================

/// A complete measure: every function `ok`, but the declared cells as declared.
/// Used by unit tests (tautological by design: mirrors EXPECTED exactly).
fn as_declared(engine: Engine) -> Measures {
    let mut m = Measures::default();
    for (function, _) in FUNCTIONS {
        let verdict = match expected(engine, function) {
            None => Verdict::Ok,
            Some(Expect::Gap { cause, .. }) => Verdict::Gap(cause),
            Some(Expect::NotMeasured { .. }) => Verdict::NotMeasured,
        };
        m.set(function, verdict, "synthetic");
    }
    m
}

/// Base measures for an engine where only a SUBSET of functions are actually run.
/// Gap entries from EXPECTED become `Declared(cause)`, not_measured stay `NotMeasured`.
/// Functions where EXPECTED returns `None` (expected ok) are NOT set here: the caller
/// must set them via the actual mini-run measurement.
fn declared_only(engine: Engine) -> Measures {
    let mut m = Measures::default();
    for (function, _) in FUNCTIONS {
        match expected(engine, function) {
            None => {} // caller sets this via real measurement
            Some(Expect::Gap { cause, .. }) => m.set(
                function,
                Verdict::Declared(cause),
                "declared in EXPECTED, scenario not run on this engine",
            ),
            Some(Expect::NotMeasured { .. }) => m.set(
                function,
                Verdict::NotMeasured,
                "not_measured (declared in EXPECTED)",
            ),
        }
    }
    m
}

/// Fills, from [`EXPECTED`], the cells the scenario did not exercise on `engine`: a
/// `NotMeasured` declaration becomes `NotMeasured`, a declared `Gap` becomes `Declared`.
/// Only MISSING cells are touched — a cell the scenario already measured is left as is,
/// so this never trips `Measures`'s "measured twice" guard.
fn fill_missing_declared(m: &mut Measures, engine: Engine) {
    for (function, _) in FUNCTIONS {
        if m.get(function).is_some() {
            continue;
        }
        match expected(engine, function) {
            Some(Expect::NotMeasured { .. }) => m.set(
                function,
                Verdict::NotMeasured,
                "not_measured (declared in EXPECTED)",
            ),
            Some(Expect::Gap { cause, .. }) => m.set(
                function,
                Verdict::Declared(cause),
                "declared in EXPECTED, scenario did not run it",
            ),
            None => {}
        }
    }
}

/// Mini-run that measures `provider_switch.relay` on ACP by opening a native session then
/// switching to fake_acp, and fills the rest via [`declared_only`].
async fn run_acp_column() -> Measures {
    use std::collections::BTreeMap;
    let dir = tempfile::TempDir::new().unwrap();
    let d = dir.path();

    // ACP transcript — "ignoring" mode: initialize → session/new → session/prompt → reply.
    let transcript_lines = concat!(
        "{\"op\":\"expect\",\"method\":\"initialize\",\"reply\":{\"protocolVersion\":1,",
        "\"agentCapabilities\":{\"loadSession\":true,\"promptCapabilities\":{\"image\":true,",
        "\"audio\":false,\"embeddedContext\":true},\"mcpCapabilities\":{\"http\":false,",
        "\"sse\":false}},\"agentInfo\":{\"name\":\"openclaw-acp\",\"version\":\"2026.9.9\"},",
        "\"authMethods\":[]}}\n",
        "{\"op\":\"expect\",\"method\":\"session/new\",\"reply\":{\"sessionId\":\"oc-1\"}}\n",
        "{\"op\":\"expect\",\"method\":\"session/prompt\",\"defer\":\"p\"}\n",
        "{\"op\":\"notify\",\"method\":\"session/update\",\"params\":{\"sessionId\":\"oc-1\",",
        "\"update\":{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"type\":\"text\",",
        "\"text\":\"hello from openclaw\"}}}}\n",
        "{\"op\":\"reply\",\"to\":\"p\",\"result\":{\"stopReason\":\"end_turn\"}}\n",
    );
    let transcript = d.join("acp-relay.jsonl");
    std::fs::write(&transcript, transcript_lines).unwrap();

    let record_path = d.join("acp-relay-record.jsonl");
    let script = d.join("fake-acp-relay.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nFAKE_ACP_TRANSCRIPT='{}' FAKE_ACP_RECORD='{}' \
             FAKE_ACP_MAX_RUNTIME_MS=60000 exec '{}' \"$@\"\n",
            transcript.display(),
            record_path.display(),
            fake_bin("fake_acp").display(),
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let acp_homes = d.join("acp-homes");
    std::fs::create_dir_all(&acp_homes).unwrap();

    // Minimal native (FakeOpenAi) script: probe + one turn.
    let native_script = json!([
        sse_route(
            "Call the ping tool now",
            vec![
                delta(json!({"tool_calls": [{"index": 0, "id": "rp1",
                "function": {"name": "ping", "arguments": "{}"}}]})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ]
        ),
        json!({"method": "GET", "path": "/v1/models", "status": 200,
               "body": {"object": "list", "data": [{"id": "m", "context_length": 32000}]}}),
        sse_route(
            "hi there",
            vec![
                delta(json!({"content": "hello from native"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 10,
                "completion_tokens": 3, "total_tokens": 13}}),
                json!("[DONE]"),
            ]
        ),
    ]);
    let fake = FakeOpenAi::start(native_script);

    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "relay-mini", "local", &fake.origin()).await;

    let acp_origin = process_origin("acp", Some("acp-bridge"));
    store_instance(
        &graph,
        &InstanceRecord {
            id: "parity-acp".into(),
            kind: "acp".into(),
            preset: Some("acp-bridge".into()),
            label: "Parity ACP relay".into(),
            base_url: String::new(),
            origin: acp_origin.clone(),
            default_model: Some("openclaw".into()),
            cost_source: "unknown".into(),
            credential_ref: "none".into(),
            ..Default::default()
        },
    )
    .await;
    consent(&graph, "relay-mini", "parity-acp", &acp_origin).await;

    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let manager = ChatManager::new_without_memory(
        dyn_graph,
        state.meili,
        ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            jwt_secret: Some("parity-secret-parity-secret-parity".to_string()),
            ..Default::default()
        },
    )
    .with_acp_commands({
        let mut cmds = BTreeMap::new();
        cmds.insert(
            "acp-bridge".to_string(),
            AcpCommand {
                argv: vec![
                    script.display().to_string(),
                    "acp".into(),
                    "--token-file".into(),
                    "/dev/null".into(),
                ],
                per_session_mcp: Some(false),
            },
        );
        cmds
    })
    .with_acp_homes(&acp_homes);

    // Open native session, wait for first turn to complete.
    let created = manager
        .create_session(&request(Some("local"), Some("relay-mini"), "default"))
        .await
        .unwrap_or_else(|e| panic!("run_acp_column: native create_session failed: {e:#}"));
    let sid = created.session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    loop {
        match tokio::time::timeout(ENGINE_RESULT_WAIT, rx.recv()).await {
            Ok(Ok(ChatEvent::Result { .. })) => break,
            Ok(Ok(_)) => continue,
            Ok(Err(_)) | Err(_) => panic!("run_acp_column: timed out waiting for native Result"),
        }
    }

    // Switch to ACP.
    manager
        .switch_session_provider(&sid, "parity-acp", None, "relay check", None)
        .await
        .unwrap_or_else(|e| panic!("run_acp_column: switch to ACP failed: {e:#}"));

    // Re-subscribe after the switch: the ACP session may use a fresh broadcast sender.
    let mut rx = manager.subscribe(&sid).await.unwrap();

    // Wait for ACP answer.
    loop {
        match tokio::time::timeout(ENGINE_RESULT_WAIT, rx.recv()).await {
            Ok(Ok(ChatEvent::Result { .. })) => break,
            Ok(Ok(_)) => continue,
            Ok(Err(_)) | Err(_) => panic!("run_acp_column: timed out waiting for ACP Result"),
        }
    }

    // Verify relay.
    let recorded: Vec<Value> = std::fs::read_to_string(&record_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect();
    let relay_ok = recorded
        .iter()
        .any(|e| e["kind"] == "in" && e["params"].to_string().contains("<conversation_relay"));

    let mut m = declared_only(Engine::Acp);
    m.set(
        "provider_switch.relay",
        if relay_ok {
            Verdict::Ok
        } else {
            Verdict::Gap(Cause::Harness)
        },
        format!(
            "ACP enregistré {} requêtes ; relay présent dans params = {}",
            recorded.len(),
            relay_ok
        ),
    );
    m
}

/// Mini-run that measures `provider_switch.relay` on Codex by opening a native session then
/// switching to fake_codex, and fills the rest via [`declared_only`].
async fn run_codex_column() -> Measures {
    let dir = tempfile::TempDir::new().unwrap();
    let d = dir.path();

    // Codex transcript (one session: initialize → thread/start → turn/start → text → turn/completed).
    let transcript = d.join("codex-relay.jsonl");
    std::fs::write(
        &transcript,
        concat!(
            "{\"op\":\"expect\",\"method\":\"initialize\",\"reply\":{\"userAgent\":\"fake_codex/0.130.0\",",
            "\"codexHome\":\"/fake/codex-home\",\"platformFamily\":\"unix\",\"platformOs\":\"macos\"}}\n",
            "{\"op\":\"expect_notification\",\"method\":\"initialized\"}\n",
            "{\"op\":\"expect\",\"method\":\"thread/start\",\"reply\":{\"thread\":{\"id\":\"thr_r1\",",
            "\"preview\":\"\",\"modelProvider\":\"openai\",\"createdAt\":1730910000},\"model\":\"gpt-fake\"}}\n",
            "{\"op\":\"notify\",\"method\":\"thread/started\",\"params\":{\"thread\":{\"id\":\"thr_r1\",",
            "\"preview\":\"\",\"modelProvider\":\"openai\",\"createdAt\":1730910000}}}\n",
            "{\"op\":\"expect\",\"method\":\"turn/start\",\"reply\":{\"turn\":{\"id\":\"turn_r1\",",
            "\"status\":\"inProgress\",\"items\":[],\"error\":null}}}\n",
            "{\"op\":\"notify\",\"method\":\"turn/started\",\"params\":{\"threadId\":\"thr_r1\",",
            "\"turn\":{\"id\":\"turn_r1\",\"status\":\"inProgress\",\"items\":[],\"error\":null}}}\n",
            "{\"op\":\"notify\",\"method\":\"item/started\",\"params\":{\"threadId\":\"thr_r1\",",
            "\"turnId\":\"turn_r1\",\"item\":{\"type\":\"agentMessage\",\"id\":\"msg_r1\",\"text\":\"\"}}}\n",
            "{\"op\":\"notify\",\"method\":\"item/agentMessage/delta\",\"params\":{\"threadId\":\"thr_r1\",",
            "\"turnId\":\"turn_r1\",\"itemId\":\"msg_r1\",\"delta\":\"Hello from codex (relay)\"}}\n",
            "{\"op\":\"notify\",\"method\":\"item/completed\",\"params\":{\"threadId\":\"thr_r1\",",
            "\"turnId\":\"turn_r1\",\"item\":{\"type\":\"agentMessage\",\"id\":\"msg_r1\",",
            "\"text\":\"Hello from codex (relay)\"}}}\n",
            "{\"op\":\"notify\",\"method\":\"thread/tokenUsage/updated\",\"params\":{\"threadId\":\"thr_r1\",",
            "\"turnId\":\"turn_r1\",\"tokenUsage\":{\"total\":{\"totalTokens\":50,\"inputTokens\":40,",
            "\"cachedInputTokens\":0,\"outputTokens\":10,\"reasoningOutputTokens\":0},",
            "\"last\":{\"totalTokens\":50,\"inputTokens\":40,\"cachedInputTokens\":0,",
            "\"outputTokens\":10,\"reasoningOutputTokens\":0},\"modelContextWindow\":272000}}}\n",
            "{\"op\":\"notify\",\"method\":\"turn/completed\",\"params\":{\"threadId\":\"thr_r1\",",
            "\"turn\":{\"id\":\"turn_r1\",\"status\":\"completed\",\"items\":[],\"error\":null}}}\n",
        ),
    )
    .unwrap();

    let record_path = d.join("codex-relay-record.jsonl");
    let home_root = d.join("codex-homes");
    let instance_home = home_root.join("parity-codex");
    std::fs::create_dir_all(&instance_home).unwrap();
    std::fs::write(instance_home.join("auth.json"), b"{}").unwrap();

    let script = d.join("fake-codex-relay.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nFAKE_CODEX_TRANSCRIPT='{}' FAKE_CODEX_RECORD='{}' \
             FAKE_CODEX_MAX_RUNTIME_MS=60000 exec '{}' \"$@\"\n",
            transcript.display(),
            record_path.display(),
            fake_bin("fake_codex").display(),
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // Minimal native (FakeOpenAi) script: probe + one turn.
    let native_script = json!([
        sse_route(
            "Call the ping tool now",
            vec![
                delta(json!({"tool_calls": [{"index": 0, "id": "cp1",
                "function": {"name": "ping", "arguments": "{}"}}]})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ]
        ),
        json!({"method": "GET", "path": "/v1/models", "status": 200,
               "body": {"object": "list", "data": [{"id": "m", "context_length": 32000}]}}),
        sse_route(
            "hi there",
            vec![
                delta(json!({"content": "hello from native (codex relay)"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 10,
                "completion_tokens": 5, "total_tokens": 15}}),
                json!("[DONE]"),
            ]
        ),
    ]);
    let fake = FakeOpenAi::start(native_script);

    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "relay-codex", "local", &fake.origin()).await;

    let codex_origin = process_origin("codex", None);
    store_instance(
        &graph,
        &InstanceRecord {
            id: "parity-codex".into(),
            kind: "codex".into(),
            label: "Parity Codex relay".into(),
            origin: codex_origin.clone(),
            default_model: Some("gpt-fake".into()),
            cost_source: "unknown".into(),
            credential_ref: "none".into(),
            ..Default::default()
        },
    )
    .await;
    consent(&graph, "relay-codex", "parity-codex", &codex_origin).await;

    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let manager = ChatManager::new_without_memory(
        dyn_graph,
        state.meili,
        ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            jwt_secret: Some("parity-secret-parity-secret-parity".to_string()),
            ..Default::default()
        },
    )
    .with_codex_program(&script)
    .with_codex_home(&home_root);

    // Open native session, wait for first turn to complete.
    let created = manager
        .create_session(&request(Some("local"), Some("relay-codex"), "default"))
        .await
        .unwrap_or_else(|e| panic!("run_codex_column: native create_session failed: {e:#}"));
    let sid = created.session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();
    loop {
        match tokio::time::timeout(ENGINE_RESULT_WAIT, rx.recv()).await {
            Ok(Ok(ChatEvent::Result { .. })) => break,
            Ok(Ok(_)) => continue,
            Ok(Err(_)) | Err(_) => panic!("run_codex_column: timed out waiting for native Result"),
        }
    }

    // Switch to Codex.
    manager
        .switch_session_provider(&sid, "parity-codex", None, "relay check", None)
        .await
        .unwrap_or_else(|e| panic!("run_codex_column: switch to Codex failed: {e:#}"));

    // Re-subscribe after the switch: the Codex session may use a fresh broadcast sender.
    let mut rx = manager.subscribe(&sid).await.unwrap();

    // Wait for Codex answer.
    loop {
        match tokio::time::timeout(ENGINE_RESULT_WAIT, rx.recv()).await {
            Ok(Ok(ChatEvent::Result { .. })) => break,
            Ok(Ok(_)) => continue,
            Ok(Err(_)) | Err(_) => panic!("run_codex_column: timed out waiting for Codex Result"),
        }
    }

    // Verify relay.
    let recorded: Vec<Value> = std::fs::read_to_string(&record_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect();
    let relay_ok = recorded
        .iter()
        .any(|e| e["kind"] == "in" && e["params"].to_string().contains("<conversation_relay"));

    let mut m = declared_only(Engine::Codex);
    m.set(
        "provider_switch.relay",
        if relay_ok {
            Verdict::Ok
        } else {
            Verdict::Gap(Cause::Harness)
        },
        format!(
            "Codex enregistré {} requêtes ; relay présent dans params = {}",
            recorded.len(),
            relay_ok
        ),
    );
    m
}

/// A run of the three columns, in table order.
fn three<'a>(cc: &'a Measures, cca: &'a Measures, na: &'a Measures) -> [(Engine, &'a Measures); 3] {
    [
        (Engine::ClaudeCode, cc),
        (Engine::ClaudeCodeAgent, cca),
        (Engine::Native, na),
    ]
}

#[test]
fn the_declared_matrix_audits_clean() {
    let measures: Vec<(Engine, Measures)> = ENGINES.iter().map(|&e| (e, as_declared(e))).collect();
    let run: Vec<(Engine, &Measures)> = measures.iter().map(|(e, m)| (*e, m)).collect();
    let problems = audit(&run, &expected);
    assert!(problems.is_empty(), "{problems:?}");
}

#[test]
fn an_undeclared_gap_turns_the_matrix_red() {
    let cc = as_declared(Engine::ClaudeCode);
    let mut na = as_declared(Engine::Native);
    na.rows
        .iter_mut()
        .find(|(f, _, _)| *f == "nats.send")
        .unwrap()
        .1 = Verdict::Gap(Cause::Harness);
    let cca = as_declared(Engine::ClaudeCodeAgent);
    let problems = audit(&three(&cc, &cca, &na), &expected);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("UNDECLARED gap"), "{problems:?}");
}

#[test]
fn a_declared_gap_fixed_silently_turns_the_matrix_red() {
    // A gap declared (synthetic, on a cell EXPECTED leaves ok) and measured ok.
    let declared = |engine: Engine, function: &str| {
        if engine == Engine::Native && function == "permissions.always" {
            return Some(Expect::Gap {
                cause: Cause::Harness,
                task: "P0",
                why: "synthetic",
            });
        }
        expected(engine, function)
    };
    let cc = as_declared(Engine::ClaudeCode);
    let na = as_declared(Engine::Native);
    let cca = as_declared(Engine::ClaudeCodeAgent);
    let problems = audit(&three(&cc, &cca, &na), &declared);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("no longer measured"), "{problems:?}");
}

#[test]
fn a_gap_of_another_cause_than_declared_turns_the_matrix_red() {
    let mut cc = as_declared(Engine::ClaudeCode);
    let na = as_declared(Engine::Native);
    cc.rows
        .iter_mut()
        .find(|(f, _, _)| *f == "images")
        .unwrap()
        .1 = Verdict::Gap(Cause::Model);
    let cca = as_declared(Engine::ClaudeCodeAgent);
    let problems = audit(&three(&cc, &cca, &na), &expected);
    assert_eq!(problems.len(), 1, "{problems:?}");
}

#[test]
fn every_declared_cell_names_a_function_of_the_matrix_once() {
    for (i, (engine, function, x)) in EXPECTED.iter().enumerate() {
        assert!(
            FUNCTIONS.iter().any(|(f, _)| f == function),
            "{function} is not in FUNCTIONS"
        );
        assert!(
            !EXPECTED[..i]
                .iter()
                .any(|(e, f, _)| e == engine && f == function),
            "{function} declared twice for {engine:?}"
        );
        if let Expect::Gap { task, .. } = x {
            // P… = plan 5ad54c48 "Parité totale"
            // M… = plan e87abcc2 "Claude Code devient optionnel"
            // ISO… = ISO1…ISO5 harness-isomorphism tasks (e.g. ISO4, id 4e2becce): these
            //        are cross-cutting harness gaps, not scoped to a single parity plan.
            assert!(
                task.starts_with('P') || task.starts_with('M') || task.starts_with("ISO"),
                "a PO task of plan 5ad54c48 (P…), e87abcc2 (M…) or an ISO harness task (ISO…): {task}"
            );
        }
    }
}

#[test]
fn only_a_limit_of_the_model_may_be_announced_missing() {
    let init = |degraded: &[&str], vision: bool| ChatEvent::SystemInit {
        cli_session_id: String::new(),
        model: None,
        tools: vec![],
        mcp_servers: vec![],
        permission_mode: None,
        provider: None,
        capabilities: Some(json!({"images": vision, "context_window": null})),
        tool_policy: None,
        policy_mode: None,
        engine: Some("agent".into()),
        degraded_features: Some(degraded.iter().map(|s| s.to_string()).collect()),
    };
    assert!(not_model_limits(&init(&[], true)).is_empty());
    assert!(not_model_limits(&init(&["images"], false)).is_empty());
    assert_eq!(not_model_limits(&init(&["images"], true)), vec!["images"]);
    // An unknown window is the model's (or its endpoint's): its compaction is not done.
    assert!(not_model_limits(&init(&["compaction"], true)).is_empty());
    // An installation limit is not a limit of the model: the scenario always has nexus-tools.
    let nexus = super::agent_runtime::NEXUS_TOOLS_FEATURE;
    assert_eq!(
        not_model_limits(&init(&["hooks", nexus], true)),
        vec!["hooks", nexus]
    );
}

// ============================================================================
// Outage of the provider instance, on the historical engine (A5)
// ============================================================================

/// The refusal the CLI prints when the provider answers 502: the historical loop
/// has only this TEXT (`classify_api_error`), not the typed `ProviderError` the
/// agent engine feeds its detector with.
const LEGACY_PROVIDER_DOWN: &str = "API Error: 502 Bad Gateway";

/// A turn the provider refuses four times, then answers. Four refusals are what it
/// takes with the default retry budget (`max_attempts = 3`): the first three spend
/// it (and the third declares the outage), the fourth finds the instance down and
/// is **held**. Every refusal is followed by `await_stdin` for the same message —
/// the loop re-sends it whole on each attempt, hold included.
fn transcript_down_then_up(turn: &str) -> Vec<Value> {
    let mut t: Vec<Value> = vec![capture_hooks(), await_in(turn)];
    for _ in 0..4 {
        t.push(result("error_during_execution", LEGACY_PROVIDER_DOWN, true));
        t.push(await_in(turn));
    }
    t.push(text("answered after the outage"));
    t.push(result_ok("answered after the outage"));
    t.push(wait_eof());
    t
}

/// A5 — the historical (Claude Code) loop against a **down** provider.
///
/// The ordinary retry budget covers a blip; once it is spent, a healthy loop would
/// close the turn on the refusal and the user would read "API Error: 502" for a
/// service that is merely away. With the outage detector wired in, the fourth
/// failure — the one the retry budget no longer covers — is **held** for the down
/// instance and probed, so the turn ends on the answer the instance gives when it
/// comes back.
///
/// The `down`/`up` pair goes to the session's own event stream
/// (`ChatEvent::ProviderOutage`, persisted and broadcast), the very channel the
/// agent engine announces its own outages on.
#[tokio::test]
async fn the_historical_loop_holds_a_turn_for_a_down_provider() {
    let stage = Stage::new().await;
    let cwd = stage.cwd();
    let k = Keys("CC");

    let cli = FakeClaude::new();
    cli.transcript(0, &transcript_down_then_up(&k.k("TURN-ONE")));
    // The backends build a provider source over the local instance, whether or not
    // the session ends up on the historical engine.
    // The local instance is only there for the provider source to build (this run
    // never reaches it): an empty script is a valid one (a JSON list).
    let fake = FakeOpenAi::start_for(json!([]), 900_000);
    let _dump = OpenAiDump(&fake);
    store_instance(&stage.graph, &instance(&fake, "none")).await;
    consent(&stage.graph, "proj", "local", &fake.origin()).await;
    let answers: Arc<Answers> = Arc::default();

    let manager = stage.backend(&cli, &answers, ProviderPath::Legacy).await;
    let mut req = request(None, Some("proj"), "default");
    req.cwd = cwd.clone();
    req.message = format!("{}: are you there?", k.k("TURN-ONE"));
    req.user_claims = Some(parity_person());
    let sid = manager
        .create_session(&req)
        .await
        .unwrap_or_else(|e| panic!("opening the Claude Code session: {e:#}"))
        .session_id;
    assert!(
        !manager.agent_runtime.owns(&sid).await,
        "the session runs on the historical engine"
    );
    let w = Watch::start(
        stage.graph.clone(),
        &sid,
        manager.subscribe(&sid).await.unwrap(),
    );

    // 1. The turn is not closed on the refusal: it answers once the instance is back.
    assert!(
        w.said(Duration::from_secs(60), "answered after the outage")
            .await,
        "the held turn ends on the answer of the instance that came back"
    );

    // 2. The hold is what bought the extra attempt: the message was re-sent once
    //    more than the ordinary budget alone allows (3 retries → 4 sends).
    let resends = cli
        .stdin_lines(0)
        .iter()
        .filter(|l| l.contains(&k.k("TURN-ONE")))
        .count();
    assert!(
        resends >= 5,
        "the spent budget was held for the down instance (sends: {resends}, ordinary budget: 4)"
    );

    // 3. The outage is announced, both ways, on the session's own channel — the
    //    `down` one naming the instance and its masked cause (our token, never the
    //    raw provider text).
    let notices = w.all().await;
    assert!(
        notices.iter().any(|e| matches!(
            e,
            ChatEvent::ProviderOutage { provider_id, phase, cause, failures, .. }
                if provider_id.as_str() == "claude-code"
                    && phase.as_str() == "down"
                    && cause.as_deref() == Some("http_5xx")
                    && *failures == 3
        )),
        "a `down` notice names the instance, its failures and its masked cause: {notices:#?}"
    );
    // The `up` notice follows the answer it announces: read right after the text it can still
    // be in flight (a loaded runner failed here, #914). Wait for it, bounded.
    assert!(
        w.wait(Duration::from_secs(10), |e| matches!(
            e,
            ChatEvent::ProviderOutage { phase, .. } if phase.as_str() == "up"
        ))
        .await,
        "the recovery is announced too: {:#?}",
        w.all().await
    );

    // 4. A held turn is not a failed one: the user never reads the refusal.
    let errors: Vec<String> = notices
        .iter()
        .filter_map(|e| match e {
            ChatEvent::Error { message, .. } => Some(message.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !errors.iter().any(|m| m.contains("502")),
        "the held turn was never closed on the refusal: {errors:?}"
    );
}

// ============================================================================
// `anthropic_messages`: a fake that cannot report headers is not a parity gap
// ============================================================================

/// A `/v1/messages` record as `fake_openai` writes it once it records headers: the
/// three `cache_control` points of CTX5, usable by both the old and the new shape.
fn anthropic_post(header_values: Option<Value>, key_header: bool) -> Value {
    let mut record = json!({
        "method": "POST",
        "path": "/v1/messages",
        "headers": {"content-type": "application/json"},
        "authorization_present": false,
        "body": {
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ]}],
            "tools": [{"name": "t", "input_schema": {"type": "object"},
                       "cache_control": {"type": "ephemeral"}}],
        },
    });
    if let Some(headers) = header_values {
        record["header_values"] = headers;
    }
    if key_header {
        record["x_api_key_present"] = json!(true);
    }
    record
}

#[test]
fn an_old_fake_is_unmeasurable_not_a_gap_of_the_harness() {
    // The exact record of the 17/10 build: headers, body, method, path, key booleans,
    // and NO `header_values`. It used to be read as `gap (harnais)` on a green `main`.
    let old = anthropic_post(None, false);
    assert!(old.get("header_values").is_none());
    let err = anthropic_messages_verdict(&[old], true, ANTHROPIC_KEY)
        .expect_err("a fake that drops the headers cannot measure the cell");
    assert!(err.contains("header_values"), "{err}");
    assert!(
        err.contains("à tort"),
        "the reason names the false accusation: {err}"
    );
}

#[test]
fn a_request_without_the_version_header_is_a_real_gap_of_the_harness() {
    // The guard must not swallow a genuine gap: headers ARE reported, the version is not
    // what the Messages API requires, so the cell is the harness's fault and says so.
    let post = anthropic_post(Some(json!({"content-type": "application/json"})), true);
    let (ok, seen) =
        anthropic_messages_verdict(&[post], true, ANTHROPIC_KEY).expect("the headers are reported");
    assert!(!ok, "a missing version header is a gap: {seen}");
    assert!(seen.contains("anthropic-version=false"), "{seen}");
}

#[test]
fn a_conforming_request_is_ok() {
    let post = anthropic_post(Some(json!({"anthropic-version": "2023-06-01"})), true);
    let (ok, seen) =
        anthropic_messages_verdict(&[post], true, ANTHROPIC_KEY).expect("the headers are reported");
    assert!(ok, "{seen}");
}

#[test]
fn a_turn_that_never_reached_the_messages_api_is_a_gap() {
    let (ok, seen) = anthropic_messages_verdict(&[], true, ANTHROPIC_KEY)
        .expect("nothing to read headers from is not the same as a blind fake");
    assert!(!ok);
    assert!(seen.contains("aucune requête"), "{seen}");
}
