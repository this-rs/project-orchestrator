//! Engine parity, MEASURED (plan 5ad54c48 « Parité totale », task P0).
//!
//! ONE scenario, played on the two engines a conversation can live on:
//!
//! - **Claude Code** on the historical engine (`ProviderPath::Legacy`, what
//!   production runs for Claude Code): `InteractiveClient` on `fake_claude`, the
//!   scripted stand-in for the `claude` CLI of the nexus repository, behind a
//!   wrapper that hands every spawn its own transcript (the CLI is spawned again
//!   by a resume and by a switch back) and records what each spawn read;
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

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use nexus_claude::agent::{
    AgentProvider, AgentSession, Capabilities, CompactionInfo, HookVerdict, ModelInfo,
    ProviderError, ProviderHealth, ProviderKind, ResumeToken, SessionHooks, SessionSpec,
    ToolCallInfo, ToolResultInfo, TurnContext, TurnDirective,
};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use uuid::Uuid;

use super::agent_e2e_tests::{
    consent, delta, fake_bin, instance, request, sse_route, store_instance, FakeOpenAi,
};
use super::agent_runtime::ProviderSource;
use super::config::{ChatConfig, ProviderPath};
use super::manager::ChatManager;
use super::types::ChatEvent;
use crate::documents::store::DocumentStore;
use crate::events::nats_broker_test::TestBroker;
use crate::events::NatsEmitter;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::GraphStore;
use crate::refs::types::{EntityRef, RefKind};
use crate::test_helpers::mock_app_state;

// ============================================================================
// The matrix: functions, verdicts, the declared gaps
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    ClaudeCode,
    Native,
}

impl Engine {
    fn label(self) -> &'static str {
        match self {
            Engine::ClaudeCode => "Claude Code",
            Engine::Native => "natif",
        }
    }
}

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
        "le hook PreCompact du graphe nomme le projet avant une compaction",
    ),
    (
        "compaction",
        "compaction_started et compact_boundary sur le fil",
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
        "po_tools",
        "les outils project-orchestrator (MCP) sont donnés et appelables",
    ),
    (
        "nexus_tools",
        "Read / Edit / Bash s'exécutent sur le projet",
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
        "la bascule vers ce moteur relaie la conversation sur le fil (conversation_relayed)",
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
        "system_init.degraded",
        "system_init n'annonce comme manquant qu'une limite du modèle (liste fermée)",
    ),
];

/// Every cell of the table that is not `ok`, declared. A gap measured and not
/// listed here fails the test; an entry here that the scenario no longer
/// measures fails it too (update this list and regenerate the table).
const EXPECTED: &[(Engine, &str, Expect)] = &[
    (
        Engine::ClaudeCode,
        "nexus_tools",
        Expect::NotMeasured {
            why: "Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue \
                  un transcript et n'exécute aucun outil",
        },
    ),
    (
        Engine::Native,
        "session_record",
        Expect::Gap {
            cause: Cause::Harness,
            task: "P8",
            why: "le moteur agent ne met pas à jour le dossier de session à la fin d'un tour \
                  (message_count / total_cost_usd)",
        },
    ),
    (
        Engine::Native,
        "background_tasks",
        Expect::Gap {
            cause: Cause::Harness,
            task: "P4",
            why: "le natif ne rapporte pas de tâche d'arrière-plan (capacité background_tasks = \
                  false) : rien n'est suivi",
        },
    ),
    (
        Engine::Native,
        "cancel_task",
        Expect::Gap {
            cause: Cause::Harness,
            task: "P12",
            why: "cancel_task n'a pas de branche moteur agent (no-op idempotent) et le natif \
                  n'a pas de tâche à annuler",
        },
    ),
];

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
    EXPECTED
        .iter()
        .find(|(e, f, _)| *e == engine && *f == function)
        .map(|(_, _, x)| *x)
}

/// What is wrong with the measures against [`EXPECTED`]: empty when the table is
/// true. Pure, so the two ways it turns red are tested without the fakes.
fn audit(
    cc: &Measures,
    native: &Measures,
    expected: &dyn Fn(Engine, &str) -> Option<Expect>,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (function, _) in FUNCTIONS {
        for (engine, m) in [(Engine::ClaudeCode, cc), (Engine::Native, native)] {
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
        (None, _) => "?".to_string(),
    }
}

const MATRIX_PATH: &str = "docs/parity/engines.md";
const UPDATE_ENV: &str = "UPDATE_PARITY_MATRIX";
const UPDATE_COMMAND: &str =
    "UPDATE_PARITY_MATRIX=1 cargo test --lib chat::engine_parity_tests::the_same_scenario";

/// `docs/parity/engines.md`, from the verdicts and the declarations only (never
/// from what varies between runs: ids, timings, paths).
fn render(cc: &Measures, native: &Measures) -> String {
    let mut out = String::new();
    out.push_str("# Parité des moteurs de chat — matrice MESURÉE\n\n");
    out.push_str(&format!(
        "<!-- Généré par src/chat/engine_parity_tests.rs : ne pas éditer à la main.\n     \
         {UPDATE_COMMAND} -->\n\n"
    ));
    out.push_str(
        "Un seul scénario, joué sur les deux moteurs, même assertion par fonction :\n\n\
         - **Claude Code** : moteur historique (chemin de production de Claude Code) sur \
         `fake_claude`, le faux CLI du dépôt nexus ;\n\
         - **natif** : moteur agent, instance OpenAI-compatible sur `fake_openai`, outils \
         project-orchestrator sur `fake_mcp`, outils de base sur le vrai `nexus-tools`.\n\n\
         `ok` : mesuré et conforme. `gap (cause, tâche)` : mesuré absent, cause `harnais` \
         (le backend ou le harnais nexus ne le fait pas) ou `modèle` (le modèle ne le peut pas), \
         fermé par la tâche PO nommée (plan 5ad54c48). `not_measured` : les faux binaires ne \
         permettent pas de l'exercer sur ce moteur (raison ci-dessous). Un écart non déclaré, \
         ou un écart déclaré qui disparaît, fait échouer le test.\n\n",
    );
    out.push_str("| Fonction | Ce qui est vérifié | Claude Code | natif |\n");
    out.push_str("|---|---|---|---|\n");
    for (function, what) in FUNCTIONS {
        out.push_str(&format!(
            "| `{function}` | {what} | {} | {} |\n",
            cell(Engine::ClaudeCode, function, cc),
            cell(Engine::Native, function, native)
        ));
    }
    let ok = |m: &Measures| m.rows.iter().filter(|(_, v, _)| *v == Verdict::Ok).count();
    out.push_str(&format!(
        "\n`ok` : Claude Code {}/{}, natif {}/{}.\n",
        ok(cc),
        FUNCTIONS.len(),
        ok(native),
        FUNCTIONS.len()
    ));
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
    out
}

/// [`EXPECTED`] by engine, then in the order of the table.
fn declared_in_table_order() -> Vec<(Engine, &'static str, Expect)> {
    let mut out = Vec::new();
    for engine in [Engine::ClaudeCode, Engine::Native] {
        for (function, _) in FUNCTIONS {
            if let Some(x) = expected(engine, function) {
                out.push((engine, *function, x));
            }
        }
    }
    out
}

/// The drift gate and the audit, once the scenario ran on both engines.
fn gate(cc: &Measures, native: &Measures) {
    for (engine, m) in [(Engine::ClaudeCode, cc), (Engine::Native, native)] {
        for (function, verdict, seen) in &m.rows {
            eprintln!(
                "parity {} / {function}: {verdict:?} — {seen}",
                engine.label()
            );
        }
    }
    let table = render(cc, native);
    let problems = audit(cc, native, &expected);
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
/// model has no vision. (An unknown `context_window` is a limit of the model
/// too, and is not a missing feature: it is never a reason to fail.)
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
    degraded_features
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|f| !(f == "images" && vision == Some(false)))
        .collect()
}

// ============================================================================
// Shared scenario data
// ============================================================================

const PLAN_TITLE: &str = "Port the enrichment";
const TASK_TITLE: &str = "Measure the parity";
const WAIT: Duration = Duration::from_secs(20);

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
        Self {
            graph,
            project,
            task_id: task.id,
            broker,
            other,
            store,
            dir,
            data: blobs,
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
    async fn backend(&self, cli: &FakeClaude, answers: &Arc<Answers>) -> ChatManager {
        let config = ChatConfig {
            provider_path: ProviderPath::Legacy,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: Some(fake_bin("nexus-tools")),
            nexus_browser_path: None,
            jwt_secret: Some("parity-secret-parity-secret-parity".to_string()),
            max_sessions: 10,
            // Three model calls per turn: the turn that makes three tool calls stops
            // on its limit (auto-continue); every other turn answers within it.
            max_turns: 3,
            claude_cli_path: Some(cli.path()),
            ..Default::default()
        };
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = self.graph.clone();
        let owner = Arc::new(NatsEmitter::new(self.broker.client().await, "events"));
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_nats(owner)
            .with_refs_v1(true)
            .with_document_store(self.store.clone())
            .with_native_transcripts(self.data.path().join("native-transcripts"))
            // The approvals `always` of the native sessions survive the restart (P11).
            .with_lasting_rules(self.data.path().join("permission-rules.json"));
        manager.update_claude_cli_path(Some(cli.path())).await;
        let real = manager
            .provider_for("local")
            .await
            .unwrap_or_else(|e| panic!("the native instance: {e:#}"));
        manager.with_provider_source(Arc::new(Tap {
            inner: real,
            answers: Arc::clone(answers),
        }))
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

/// `ok` when the record of a session carries what its conversation produced.
fn record_verdict(node: &crate::neo4j::models::ChatSessionNode, sent: i64) -> (bool, String) {
    let cost = node.total_cost_usd.unwrap_or(0.0);
    let ok = node.message_count >= sent
        && cost > 0.0
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
        let deadline = tokio::time::Instant::now() + WAIT;
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

    fn stdin_lines(&self, n: usize) -> Vec<String> {
        std::fs::read_to_string(self.dir.path().join(format!("stdin.{n}.jsonl")))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
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

/// A `can_use_tool` that carries the CLI's `permission_suggestions` (the rule a lasting
/// approval adds, as the CLI proposes it: destination `localSettings`).
fn permission_request_suggesting(
    request_id: &str,
    tool: &str,
    input: Value,
    tool_use_id: &str,
    rule_content: &str,
) -> Value {
    emit(json!({
        "type": "control_request", "request_id": request_id,
        "request": {"subtype": "can_use_tool", "tool_name": tool, "input": input,
                    "tool_use_id": tool_use_id,
                    "permission_suggestions": [{
                        "type": "addRules", "behavior": "allow", "destination": "localSettings",
                        "rules": [{"toolName": tool, "ruleContent": rule_content}]}]},
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
fn transcript_main(k: Keys, cwd: &str) -> Vec<Value> {
    let grep_input = json!({"pattern": "build_agent_spec", "path": "."});
    let mut t: Vec<Value> = vec![
        // The SDK registers its hooks at connect; their callback ids are minted at
        // run time: the fake remembers them.
        capture_hooks(),
        // TURN-ONE: a Grep (before_tool hook, then after_tool hook on its noisy
        // result), a compaction announced by the CLI (before_compaction hook), its
        // boundary, then an answer.
        await_in(&k.k("TURN-ONE")),
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
        // After a compaction the engine re-injects the work context (a turn of its own).
        await_in("Work Already Done"),
        text("noted"),
        result_ok("noted"),
        // TURN-TWO runs; TURN-THREE is sent meanwhile: the engine queues it and
        // interrupts the turn, then plays the queue.
        await_in(&k.k("TURN-TWO")),
        await_in("\"interrupt\""),
        result("error_during_execution", "", true),
    ];
    t.extend(answer(&k.k("TURN-THREE"), "answered three"));
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
        permission_request("req-perm", "Bash", json!({"command": "ls"}), "t5"),
        await_in("req-perm"),
        cc_tool_result("t5", "a.rs", false),
        text("answered five"),
        result_ok("answered five"),
    ]);
    // TURN-PERM-S / TURN-PERM-A: approvals that outlive the call (session, always).
    t.extend(vec![
        await_in(&k.k("TURN-PERM-S")),
        cc_tool_use("t5s", "Bash", json!({"command": "git status"})),
        permission_request_suggesting(
            "req-perm-s",
            "Bash",
            json!({"command": "git status"}),
            "t5s",
            "git status",
        ),
        await_in("req-perm-s"),
        cc_tool_result("t5s", "clean", false),
        text("answered perm s"),
        result_ok("answered perm s"),
        await_in(&k.k("TURN-PERM-A")),
        cc_tool_use("t5a", "Bash", json!({"command": "cargo fmt"})),
        permission_request_suggesting(
            "req-perm-a",
            "Bash",
            json!({"command": "cargo fmt"}),
            "t5a",
            "cargo fmt",
        ),
        await_in("req-perm-a"),
        cc_tool_result("t5a", "formatted", false),
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
    t.extend(vec![
        await_in(&k.k("TURN-BG")),
        cc_tool_use(
            "t7",
            "Bash",
            json!({"command": "sleep 120", "run_in_background": true,
                   "description": "parity background"}),
        ),
        sleep_ms(400),
        spawn_sleeper(),
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
    after_tool: StdMutex<Vec<Option<String>>>,
    before_compaction: StdMutex<Vec<Option<String>>>,
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
}

/// The real native provider under its own id, its session hooks recorded.
#[derive(Clone)]
struct Tap {
    inner: Arc<dyn AgentProvider>,
    answers: Arc<Answers>,
}

impl Tap {
    fn tap(&self, mut spec: SessionSpec) -> SessionSpec {
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

impl ProviderSource for Tap {
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
        (provider_id == "local").then(|| Arc::new(self.clone()) as Arc<dyn AgentProvider>)
    }
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
        // TURN-PERM3: the same tool again: the session grant holds, not asked.
        calls(
            &k.k("TURN-PERM3"),
            "c5c",
            &nexus("Write"),
            json!({"file_path": format!("{cwd}/session-2.txt"), "content": "THIRD-SESSION"}),
        ),
        says(&k.k("TURN-PERM3"), "answered perm3"),
        // TURN-ALWAYS: another tool, granted always (kept by the backend for the project).
        calls(
            &k.k("TURN-ALWAYS"),
            "c5d",
            &nexus("Edit"),
            json!({"file_path": format!("{cwd}/always.txt"), "old_string": "ALWAYS-ONE",
                   "new_string": "ALWAYS-EDITED"}),
        ),
        says(&k.k("TURN-ALWAYS"), "answered always"),
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
        // References, the image.
        says(&k.k("TURN-REFS"), "answered refs"),
        says(&k.k("TURN-IMG"), "answered img"),
    ];
    // A turn whose reported size makes the next model call compact first; the
    // summary request carries the guidance of the before_compaction hook.
    let mut prime = vec![delta(json!({"content": "primed"}))];
    prime.extend(stop(30_000));
    r.push(sse_route(&k.k("TURN-PRIME"), prime));
    r.push(says("Additional instructions", "the summary"));
    r.push(says(&k.k("TURN-COMPACT"), "answered compact"));
    // After the restart.
    r.push(says(&k.k("RESUMED"), "answered resumed"));
    // After the restart: the tool granted always runs without asking.
    r.push(calls(
        &k.k("TURN-ALWAYS2"),
        "c5e",
        &nexus("Edit"),
        json!({"file_path": format!("{cwd}/always.txt"), "old_string": "ALWAYS-EDITED",
               "new_string": "ALWAYS-TWO"}),
    ));
    r.push(says(&k.k("TURN-ALWAYS2"), "answered always2"));
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
async fn cc_lasting(
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
    let line = cli.wait_line(0, req, WAIT).await.unwrap_or_default();
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

// ============================================================================
// The scenario
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_scenario_on_both_engines_gives_the_parity_matrix() {
    let stage = Stage::new().await;
    let cwd = stage.cwd();
    let claims = crate::auth::jwt::Claims::service_account("parity");
    let kc = Keys("CC");
    let kn = Keys("NA");

    let cli = FakeClaude::new();
    cli.transcript(0, &transcript_main(kc, &cwd));
    cli.transcript(1, &transcript_resumed(kc));
    cli.transcript(2, &transcript_switched_back());

    let fake = FakeOpenAi::start_for(native_script(kn, &cwd), 900_000);
    let _dump = OpenAiDump(&fake);
    store_instance(&stage.graph, &instance(&fake, "none")).await;
    consent(&stage.graph, "proj", "local", &fake.origin()).await;
    let answers: Arc<Answers> = Arc::default();

    let mut cc = Measures::default();
    let mut na = Measures::default();

    // ===================================================================
    // Claude Code, historical engine (backend process #1)
    // ===================================================================
    let manager = stage.backend(&cli, &answers).await;
    let mut req = request(None, Some("proj"), "default");
    req.cwd = cwd.clone();
    req.message = format!("{}: where is the spec built?", kc.k("TURN-ONE"));
    let sid = manager
        .create_session(&req)
        .await
        .unwrap_or_else(|e| panic!("opening the Claude Code session: {e:#}"))
        .session_id;
    assert!(
        !manager.agent_runtime.owns(&sid).await,
        "Claude Code runs on the historical engine"
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
            .is_some_and(|l| l.contains(&stage.project.name)),
        Cause::Harness,
        format!(
            "réponse au hook PreCompact : {:?}",
            compact.as_deref().map(excerpt)
        ),
    );
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
    w.settle(0, &manager).await;

    // Message queue: TURN-THREE sent while TURN-TWO runs.
    let ends = w.turn_ends();
    manager.send_message(&sid, &kc.k("TURN-TWO")).await.unwrap();
    cc_sent += 1;
    assert!(cli.wait_line(0, &kc.k("TURN-TWO"), WAIT).await.is_some());
    tokio::time::sleep(Duration::from_millis(300)).await;
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

    // Auto-continue.
    manager.set_auto_continue(&sid, true).await.unwrap();
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
    manager.set_auto_continue(&sid, false).await.unwrap();

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
    w.settle(ends, &manager).await;
    // Approvals that outlive the call: the CLI's suggestion goes back as
    // `updatedPermissions`, moved to the scope's destination (the CLI keeps the rule).
    // Boxed: the scenario's future is already large (a stack overflow otherwise).
    let (routed_s, line_s) = Box::pin(cc_lasting(
        &manager,
        &w,
        &cli,
        kc.k("TURN-PERM-S"),
        "req-perm-s",
        crate::chat::types::PermissionAnswerScope::Session,
    ))
    .await;
    cc_sent += 1;
    let (routed_a, line_a) = Box::pin(cc_lasting(
        &manager,
        &w,
        &cli,
        kc.k("TURN-PERM-A"),
        "req-perm-a",
        crate::chat::types::PermissionAnswerScope::Always,
    ))
    .await;
    cc_sent += 1;
    let updates = |line: &str| -> Value {
        serde_json::from_str::<Value>(line).unwrap_or(Value::Null)["response"]["response"]
            ["updatedPermissions"]
            .clone()
    };
    let (up_s, up_a) = (updates(&line_s), updates(&line_a));
    cc.check(
        "permissions.session",
        routed_s
            && up_s[0]["destination"] == "session"
            && up_s[0]["rules"][0]["ruleContent"] == "git status",
        Cause::Harness,
        format!("updatedPermissions écrit : {}", excerpt(&up_s.to_string())),
    );
    cc.check(
        "permissions.always",
        routed_a
            && up_a[0]["destination"] == "localSettings"
            && up_a[0]["rules"][0]["ruleContent"] == "cargo fmt",
        Cause::Harness,
        format!("updatedPermissions écrit : {}", excerpt(&up_a.to_string())),
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
    w.said(WAIT, "answered bg").await;
    // The PID claim runs about a second after the tool_use.
    let mut claimed = false;
    for _ in 0..60 {
        claimed = manager
            .get_active_background_tasks(&sid)
            .await
            .iter()
            .any(|t| t.id == "t7" && t.pid.is_some());
        if claimed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
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
    cc.check(
        "nats.interrupt",
        stopped_there && w.turn_end_after(ends).await,
        Cause::Harness,
        format!("interrupt écrit au CLI={stopped_there}"),
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
    let manager = stage.backend(&cli, &answers).await;
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
    w.settle(0, &manager).await;

    // ===================================================================
    // Claude Code → native: the conversation relayed (backend process #2)
    // ===================================================================
    let switched = manager
        .switch_session_provider(&sid, "local", None, "SWITCH-ONE: continue here", None)
        .await
        .unwrap_or_else(|e| panic!("switching to the native instance: {e:#}"));
    let nid = switched.session_id.clone();
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
            matches!(e, ChatEvent::ConversationRelayed { from_provider, to_provider, .. }
                if from_provider == "claude-code" && to_provider == "local")
        })
        .await;
    let history = relayed.contains("<conversation_relay") && relayed.contains("answered one");
    na.check(
        "provider_switch.relay",
        history && on_wire,
        Cause::Harness,
        format!("historique relayé au modèle={history}, conversation_relayed={on_wire}"),
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
    let after = answers.after_tool.lock().unwrap().clone();
    na.check(
        "hooks.after_tool",
        after
            .iter()
            .flatten()
            .any(|a| a.contains("find_references") && a.contains("build_agent_spec")),
        Cause::Harness,
        format!("after_tool a répondu : {after:?}"),
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

    // Auto-continue.
    manager.set_auto_continue(&nid, true).await.unwrap();
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
    manager.set_auto_continue(&nid, false).await.unwrap();

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
    let asked_again = answer_if_asked(
        &manager,
        &wn,
        &fake,
        "THIRD-SESSION",
        &kn.k("TURN-PERM3"),
        1,
    )
    .await;
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

    // A tool granted always: the session offers the scope, the backend keeps the rule
    // for the project (measured after the restart, below).
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
        "ALWAYS-EDITED",
        "answered always",
        crate::chat::types::PermissionAnswerScope::Always,
    ))
    .await;
    na_sent += 1;

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
    let stopped = manager.cancel_task(&nid, "c7").await.unwrap();
    na.check(
        "cancel_task",
        !stopped.killed_pids.is_empty(),
        Cause::Harness,
        format!("killed_pids={:?}", stopped.killed_pids),
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
    let mut asked_after_restart = None;
    let manager = stage.backend(&cli, &answers).await;
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
            // The tool granted always before the restart: not asked any more.
            let ends = wn.turn_ends();
            manager
                .send_message(&nid, &kn.k("TURN-ALWAYS2"))
                .await
                .unwrap();
            let asked =
                answer_if_asked(&manager, &wn, &fake, "ALWAYS-TWO", &kn.k("TURN-ALWAYS2"), 1).await;
            let ran = wait_body_nth(&fake, &kn.k("TURN-ALWAYS2"), 1, WAIT)
                .await
                .is_some();
            asked_after_restart = Some(asked || !ran);
            wn.said(WAIT, "answered always2").await;
            wn.settle(ends, &manager).await;
        }
        Err(e) => na.check(
            "resume",
            false,
            Cause::Harness,
            format!("reprise refusée : {e:#}"),
        ),
    }
    na.check(
        "permissions.always",
        offers_always && granted_always && asked_after_restart == Some(false),
        Cause::Harness,
        format!(
            "portée offerte={offers_always}, accordée={granted_always}, \
             redemandée après redémarrage={asked_after_restart:?}"
        ),
    );

    // ===================================================================
    // native → Claude Code: the conversation relayed back (process #3)
    // ===================================================================
    let back = manager
        .switch_session_provider(&nid, "claude-code", None, "SWITCH-TWO: and back", None)
        .await
        .unwrap_or_else(|e| panic!("switching back to Claude Code: {e:#}"));
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
            matches!(e, ChatEvent::ConversationRelayed { from_provider, to_provider, .. }
                if from_provider == "local" && to_provider == "claude-code")
        })
        .await;
    let history = line.contains("<conversation_relay") && line.contains("answered compact");
    cc.check(
        "provider_switch.relay",
        history && on_wire,
        Cause::Harness,
        format!("historique relayé au CLI={history}, conversation_relayed={on_wire}"),
    );
    wb.said(WAIT, "answered switch two").await;
    let _ = manager.close_session(&back.session_id).await;

    // The hard rule: the native harness announces only limits of the MODEL.
    if let Some(unduly) = &unduly {
        assert!(
            unduly.is_empty(),
            "the native system_init announces missing features that are not limits of the \
             model: {unduly:?} (closed list: images without vision; an unknown context window)"
        );
    }
    gate(&cc, &na);
}

// ============================================================================
// The audit itself: the two ways the matrix turns red
// ============================================================================

/// A complete measure: every function `ok`, but the declared cells as declared.
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

#[test]
fn the_declared_matrix_audits_clean() {
    let problems = audit(
        &as_declared(Engine::ClaudeCode),
        &as_declared(Engine::Native),
        &expected,
    );
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
    let problems = audit(&cc, &na, &expected);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("UNDECLARED gap"), "{problems:?}");
}

#[test]
fn a_declared_gap_fixed_silently_turns_the_matrix_red() {
    let cc = as_declared(Engine::ClaudeCode);
    let mut na = as_declared(Engine::Native);
    na.rows
        .iter_mut()
        .find(|(f, _, _)| *f == "session_record")
        .unwrap()
        .1 = Verdict::Ok;
    let problems = audit(&cc, &na, &expected);
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
    let problems = audit(&cc, &na, &expected);
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
            assert!(task.starts_with('P'), "a PO task of plan 5ad54c48: {task}");
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
    // An installation limit is not a limit of the model: the scenario always has nexus-tools.
    let nexus = super::agent_runtime::NEXUS_TOOLS_FEATURE;
    assert_eq!(
        not_model_limits(&init(&["hooks", nexus], true)),
        vec!["hooks", nexus]
    );
}
