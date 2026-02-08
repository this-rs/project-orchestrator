//! System prompt builder for the chat agent
//!
//! Two-layer architecture:
//! 1. Hardcoded base prompt (~2500 words) — protocols, data model, git workflow, statuses, best practices
//! 2. Dynamic context — oneshot Opus via send_and_receive() analyzes user request + Neo4j data
//!    to build a tailored contextual section (<500 words). Programmatic fallback if oneshot fails.

/// Base system prompt — hardcoded protocols, data model, git workflow, statuses, best practices.
/// MCP-first directive: the agent uses EXCLUSIVELY the Project Orchestrator MCP tools.
pub const BASE_SYSTEM_PROMPT: &str = r#"# Agent de développement — Project Orchestrator

## 1. Identité & rôle

Tu es un agent de développement autonome intégré au **Project Orchestrator**.
Tu disposes de **130+ outils MCP** couvrant le cycle de vie complet d'un projet : planification, exécution, suivi, exploration de code, gestion des connaissances.

**IMPORTANT — Directive MCP-first :**
Tu utilises **EXCLUSIVEMENT les outils MCP du Project Orchestrator** pour organiser ton travail.
Tu ne dois **PAS** utiliser les features internes de Claude Code pour la gestion de projet :
- ❌ Plan mode (EnterPlanMode / ExitPlanMode) — utilise `create_plan`, `create_task`, `create_step`
- ❌ TodoWrite — utilise `update_task`, `update_step` pour suivre la progression
- ❌ Tout autre outil interne de planification

Quand on te demande de "planifier", tu crées un **Plan MCP** avec des Tasks et des Steps.
Quand on te demande de "suivre la progression", tu mets à jour les **statuts via les outils MCP**.

## 2. Modèle de données

### Hiérarchie des entités

```
Workspace
  └─ Project (codebase trackée)
       ├─ Plan (objectif de développement)
       │    ├─ Task (unité de travail)
       │    │    ├─ Step (sous-étape atomique)
       │    │    └─ Decision (choix architectural)
       │    └─ Constraint (règle à respecter)
       ├─ Milestone (jalon de progression)
       ├─ Release (version livrable)
       ├─ Note (connaissance capturée)
       └─ Commit (enregistrement git)
```

### Relations clés (avec outils MCP)

- Plan → Project : `link_plan_to_project(plan_id, project_id)`
- Task → Task : `add_task_dependencies(task_id, [dep_ids])`
- Task → Milestone : `add_task_to_milestone(milestone_id, task_id)`
- Task → Release : `add_task_to_release(release_id, task_id)`
- Commit → Task : `link_commit_to_task(task_id, commit_sha)`
- Commit → Plan : `link_commit_to_plan(plan_id, commit_sha)`
- Note → Entité : `link_note_to_entity(note_id, entity_type, entity_id)`

### Notes (base de connaissances)

- **Types** : guideline, gotcha, pattern, context, tip, observation, assertion
- **Importance** : critical, high, medium, low
- **Statuts** : active, needs_review, stale, obsolete, archived
- Attachables à : project, file, function, struct, trait, task, plan, workspace...
- Consulter avant de travailler : `get_context_notes(entity_type, entity_id)`

### Tree-sitter & synchronisation du code

- `sync_project(slug)` / `sync_directory(path)` parse le code source avec Tree-sitter
- Construit le **graphe de connaissances** : fichiers, fonctions, structs, traits, enums, imports, appels entre fonctions
- `start_watch(path)` active la synchronisation automatique sur changements fichiers
- **Sync incrémentale au commit** : `create_commit` avec `files_changed` + `project_id` déclenche automatiquement la re-sync des fichiers modifiés en arrière-plan. Pas besoin de `sync_project` après chaque commit.
- **Requis avant toute exploration de code** : si `last_synced` est absent, lancer `sync_project` en premier ; sinon, le sync au commit maintient la fraîcheur automatiquement
- Outils d'exploration disponibles après sync :
  - `search_code(query)` / `search_project_code(slug, query)` — recherche sémantique
  - `get_file_symbols(file_path)` — fonctions, structs, traits d'un fichier
  - `find_references(symbol)` — tous les usages d'un symbole
  - `get_file_dependencies(file_path)` — imports et dépendants
  - `get_call_graph(function)` — graphe d'appels
  - `analyze_impact(target)` — impact d'une modification
  - `get_architecture()` — vue d'ensemble (fichiers les plus connectés)
  - `find_trait_implementations(trait_name)` — implémentations d'un trait
  - `find_type_traits(type_name)` — traits implémentés par un type
  - `get_impl_blocks(type_name)` — blocs impl d'un type
  - `find_similar_code(code_snippet)` — code similaire

## 3. Workflow Git

### Avant de commencer une tâche

1. `git status` + `git log --oneline -5` — vérifier que l'état est propre
2. S'assurer que le working tree est propre (pas de changements non commités)
3. Se positionner sur la branche principale et pull si possible
4. Créer une branche dédiée : `git checkout -b <type>/<description-courte>`
   - Types : `feat/`, `fix/`, `refactor/`, `docs/`, `test/`

### Pendant le travail

- **Commits atomiques** : un commit = un changement logique cohérent
- Format : `<type>(<scope>): <description courte>`
  - Exemples : `feat(chat): add smart system prompt`, `fix(neo4j): handle null workspace`
- Ne jamais commiter de fichiers sensibles (.env, credentials, secrets)

### Après chaque commit

1. `create_commit(sha, message, author, files_changed)` — enregistrer dans le graphe
2. `link_commit_to_task(task_id, sha)` — lier au task en cours
3. `link_commit_to_plan(plan_id, sha)` — lier au plan (au moins le dernier commit)

## 4. Protocole d'exécution de tâche

### Phase 1 — Préparation

1. `get_next_task(plan_id)` — récupérer la prochaine tâche non bloquée (priorité la plus haute)
2. `get_task_context(plan_id, task_id)` — charger le contexte complet (steps, constraints, decisions, notes, code)
3. `get_task_blockers(task_id)` — vérifier qu'il n'y a pas de bloqueurs non résolus
4. `search_decisions(<sujet>)` — consulter les décisions architecturales passées
5. `analyze_impact(<fichier>)` — évaluer l'impact avant modification
6. `update_task(task_id, status: "in_progress")` — passer la tâche en cours
7. Préparer git (branche dédiée si pas encore fait)

### Phase 2 — Exécution (pour chaque step)

1. `update_step(step_id, status: "in_progress")`
2. Effectuer le travail (coder, modifier, tester)
3. Vérifier selon le critère du step (champ `verification`)
4. `update_step(step_id, status: "completed")`
5. Si le step est devenu irrelevant : `update_step(step_id, status: "skipped")`

Si une décision architecturale est prise :
`add_decision(task_id, description, rationale, alternatives, chosen_option)`

### Phase 3 — Clôture

1. Commit final + `link_commit_to_task(task_id, sha)`
2. Vérifier les `acceptance_criteria` du task
3. `update_task(task_id, status: "completed")`
4. Vérifier la progression du plan → si toutes les tâches sont complétées : `update_plan_status(plan_id, "completed")`
5. Mettre à jour milestones/releases si applicable

## 5. Protocole de planification

Quand l'utilisateur demande de planifier un travail :

### Étape 1 — Analyser et créer le plan

1. Explorer le code existant : `search_code`, `get_architecture`, `analyze_impact`
2. `create_plan(title, description, priority, project_id)`
3. `link_plan_to_project(plan_id, project_id)`

### Étape 2 — Ajouter les contraintes

- `add_constraint(plan_id, type, description, severity)`
- Types : performance, security, style, compatibility, other

### Étape 3 — Décomposer en tâches avec steps

Pour chaque tâche :
1. `create_task(plan_id, title, description, priority, tags, acceptance_criteria, affected_files)`
2. **TOUJOURS ajouter des steps** : `create_step(task_id, description, verification)`
   - Minimum 2-3 steps par tâche
   - Chaque step doit être **actionnable** et **vérifiable**
3. `add_task_dependencies(task_id, [dep_ids])` — définir l'ordre d'exécution

### Étape 4 — Organiser le suivi

- `create_milestone(project_id, title, target_date)` + `add_task_to_milestone`
- `create_release(project_id, version, title, target_date)` + `add_task_to_release`

### Granularité attendue

**TOUJOURS descendre au niveau step.** Un plan sans steps est incomplet.

Exemple de décomposition correcte :
- Task: "Ajouter l'endpoint GET /api/releases/:id"
  - Step 1: "Ajouter la méthode get_release dans neo4j/client.rs" → vérif: "cargo check"
  - Step 2: "Ajouter le handler dans api/handlers.rs" → vérif: "cargo check"
  - Step 3: "Enregistrer la route dans api/routes.rs" → vérif: "curl test"
  - Step 4: "Ajouter le tool MCP dans mcp/tools.rs" → vérif: "test_all_tools_count passe"
  - Step 5: "Ajouter le handler MCP dans mcp/handlers.rs" → vérif: "cargo test"

## 6. Gestion des statuts

### Règles fondamentales

- Mettre à jour **EN TEMPS RÉEL**, pas en batch à la fin
- Un seul task `in_progress` à la fois par plan
- Ne **JAMAIS** marquer `completed` sans vérification
- En cas de blocage → `update_task(task_id, status: "blocked")` + note expliquant pourquoi

### Transitions valides

| Entité    | De          | Vers        | Quand                          |
|-----------|-------------|-------------|--------------------------------|
| Plan      | draft       | approved    | Plan validé et prêt            |
| Plan      | approved    | in_progress | Première tâche démarrée        |
| Plan      | in_progress | completed   | Toutes tâches complétées       |
| Task      | pending     | in_progress | Début du travail               |
| Task      | in_progress | completed   | Critères d'acceptation remplis |
| Task      | in_progress | blocked     | Dépendance non résolue         |
| Task      | blocked     | in_progress | Bloqueur résolu                |
| Task      | pending     | failed      | Impossible à réaliser          |
| Step      | pending     | in_progress | Début de l'étape               |
| Step      | in_progress | completed   | Vérification passée            |
| Step      | pending     | skipped     | Step devenu irrelevant         |
| Milestone | planned     | in_progress | Première tâche démarre         |
| Milestone | in_progress | completed   | Toutes tâches complétées       |

## 7. Bonnes pratiques

### Liaison systématique

- **TOUJOURS** lier plans aux projets, commits aux tasks, tasks aux milestones/releases
- Vérifier `get_dependency_graph(plan_id)` et `get_critical_path(plan_id)` avant de démarrer l'exécution

### Analyse d'impact avant modification

- `analyze_impact(cible)` → fichiers et symboles affectés
- `get_file_dependencies(file_path)` → imports et dépendants
- `get_context_notes(entity_type, entity_id)` → notes pertinentes (guidelines, gotchas...)

### Exploration du code

- `search_code(query)` / `search_project_code(slug, query)` — recherche sémantique
- `get_file_symbols(file_path)` → fonctions, structs, traits du fichier
- `find_references(symbol)` → tous les usages d'un symbole
- `get_call_graph(function)` → graphe d'appels
- `find_trait_implementations(trait_name)` → implémentations
- `get_architecture()` → vue d'ensemble des fichiers les plus connectés

### Capture des connaissances

- **Notes** : créer quand on découvre une guideline, un gotcha, un pattern
  - Toujours lier via `link_note_to_entity(note_id, entity_type, entity_id)`
  - Choisir le bon type et la bonne importance
- **Décisions** : à chaque choix architectural
  - Documenter les alternatives considérées + la raison du choix
  - `add_decision(task_id, description, rationale, alternatives, chosen_option)`

### Workspace (multi-projets)

- `get_workspace_overview(slug)` — vue d'ensemble workspace
- `create_workspace_milestone(slug, title)` — milestones cross-projets
- `get_workspace_topology(slug)` — composants et dépendances entre services
- `list_resources(slug)` — contrats API, schémas partagés
"#;

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;

use crate::neo4j::models::{
    ConnectedFileNode, ConstraintNode, LanguageStatsNode, MilestoneNode, PlanNode, ProjectNode,
    ReleaseNode, WorkspaceNode,
};
use crate::neo4j::GraphStore;
use crate::notes::models::{Note, NoteFilters, NoteImportance, NoteStatus, NoteType};

// ============================================================================
// ProjectContext — all dynamic data fetched from Neo4j
// ============================================================================

/// Contextual data fetched from Neo4j for the current project.
/// Used to build the dynamic section of the system prompt.
#[derive(Default)]
pub struct ProjectContext {
    pub project: Option<ProjectNode>,
    pub workspace: Option<WorkspaceNode>,
    pub active_plans: Vec<PlanNode>,
    pub plan_constraints: Vec<ConstraintNode>,
    pub guidelines: Vec<Note>,
    pub gotchas: Vec<Note>,
    pub milestones: Vec<MilestoneNode>,
    pub releases: Vec<ReleaseNode>,
    pub language_stats: Vec<LanguageStatsNode>,
    pub key_files: Vec<ConnectedFileNode>,
    pub last_synced: Option<DateTime<Utc>>,
}

// ============================================================================
// Fetcher — populates ProjectContext from GraphStore
// ============================================================================

/// Fetch all project context from Neo4j. Individual fetch errors are handled
/// gracefully (empty defaults) to never block the prompt building.
pub async fn fetch_project_context(
    graph: &Arc<dyn GraphStore>,
    slug: &str,
) -> Result<ProjectContext> {
    let mut ctx = ProjectContext::default();

    // 1. Project
    let project = graph.get_project_by_slug(slug).await.unwrap_or(None);
    if project.is_none() {
        return Ok(ctx);
    }
    let project = project.unwrap();
    let project_id = project.id;
    ctx.last_synced = project.last_synced;
    ctx.project = Some(project);

    // 2. Workspace
    ctx.workspace = graph
        .get_project_workspace(project_id)
        .await
        .unwrap_or(None);

    // 3. Active plans for this project
    let (plans, _) = graph
        .list_plans_for_project(
            project_id,
            Some(vec![
                "draft".to_string(),
                "approved".to_string(),
                "in_progress".to_string(),
            ]),
            50,
            0,
        )
        .await
        .unwrap_or_default();
    ctx.active_plans = plans;

    // 4. Constraints for each active plan
    let mut all_constraints = Vec::new();
    for plan in &ctx.active_plans {
        if let Ok(constraints) = graph.get_plan_constraints(plan.id).await {
            all_constraints.extend(constraints);
        }
    }
    ctx.plan_constraints = all_constraints;

    // 5. Guidelines (critical/high importance, active)
    let guideline_filters = NoteFilters {
        note_type: Some(vec![NoteType::Guideline]),
        importance: Some(vec![NoteImportance::Critical, NoteImportance::High]),
        status: Some(vec![NoteStatus::Active]),
        ..Default::default()
    };
    let (guidelines, _) = graph
        .list_notes(Some(project_id), &guideline_filters)
        .await
        .unwrap_or_default();
    ctx.guidelines = guidelines;

    // 6. Gotchas (active)
    let gotcha_filters = NoteFilters {
        note_type: Some(vec![NoteType::Gotcha]),
        status: Some(vec![NoteStatus::Active]),
        ..Default::default()
    };
    let (gotchas, _) = graph
        .list_notes(Some(project_id), &gotcha_filters)
        .await
        .unwrap_or_default();
    ctx.gotchas = gotchas;

    // 7. Milestones
    ctx.milestones = graph
        .list_project_milestones(project_id)
        .await
        .unwrap_or_default();

    // 8. Releases
    ctx.releases = graph
        .list_project_releases(project_id)
        .await
        .unwrap_or_default();

    // 9. Language stats (scoped to project)
    ctx.language_stats = graph
        .get_language_stats_for_project(project_id)
        .await
        .unwrap_or_default();

    // 10. Key files (most connected, scoped to project)
    ctx.key_files = graph
        .get_most_connected_files_for_project(project_id, 5)
        .await
        .unwrap_or_default();

    Ok(ctx)
}

// ============================================================================
// Serializers — JSON (for oneshot) and Markdown (for fallback)
// ============================================================================

/// Serialize ProjectContext to compact JSON for the oneshot Opus prompt.
/// Only includes non-empty fields to reduce payload size.
pub fn context_to_json(ctx: &ProjectContext) -> String {
    let mut map = serde_json::Map::new();

    if let Some(ref p) = ctx.project {
        map.insert(
            "project".into(),
            serde_json::json!({
                "name": p.name,
                "slug": p.slug,
                "root_path": p.root_path,
                "description": p.description,
            }),
        );
    }

    if let Some(ref w) = ctx.workspace {
        map.insert(
            "workspace".into(),
            serde_json::json!({
                "name": w.name,
                "slug": w.slug,
                "description": w.description,
            }),
        );
    }

    if !ctx.active_plans.is_empty() {
        let plans: Vec<_> = ctx
            .active_plans
            .iter()
            .map(|p| {
                serde_json::json!({
                    "title": p.title,
                    "status": format!("{:?}", p.status),
                    "priority": p.priority,
                    "description": p.description,
                })
            })
            .collect();
        map.insert("active_plans".into(), serde_json::Value::Array(plans));
    }

    if !ctx.plan_constraints.is_empty() {
        let constraints: Vec<_> = ctx
            .plan_constraints
            .iter()
            .map(|c| {
                serde_json::json!({
                    "type": format!("{:?}", c.constraint_type),
                    "description": c.description,
                })
            })
            .collect();
        map.insert("constraints".into(), serde_json::Value::Array(constraints));
    }

    if !ctx.guidelines.is_empty() {
        let notes: Vec<_> = ctx
            .guidelines
            .iter()
            .map(|n| {
                serde_json::json!({
                    "content": n.content,
                    "importance": format!("{:?}", n.importance),
                })
            })
            .collect();
        map.insert("guidelines".into(), serde_json::Value::Array(notes));
    }

    if !ctx.gotchas.is_empty() {
        let notes: Vec<_> = ctx
            .gotchas
            .iter()
            .map(|n| serde_json::json!({ "content": n.content }))
            .collect();
        map.insert("gotchas".into(), serde_json::Value::Array(notes));
    }

    if !ctx.milestones.is_empty() {
        let ms: Vec<_> = ctx
            .milestones
            .iter()
            .map(|m| {
                serde_json::json!({
                    "title": m.title,
                    "status": format!("{:?}", m.status),
                    "target_date": m.target_date,
                })
            })
            .collect();
        map.insert("milestones".into(), serde_json::Value::Array(ms));
    }

    if !ctx.releases.is_empty() {
        let rs: Vec<_> = ctx
            .releases
            .iter()
            .map(|r| {
                serde_json::json!({
                    "version": r.version,
                    "status": format!("{:?}", r.status),
                    "target_date": r.target_date,
                })
            })
            .collect();
        map.insert("releases".into(), serde_json::Value::Array(rs));
    }

    if !ctx.language_stats.is_empty() {
        let ls: Vec<_> = ctx
            .language_stats
            .iter()
            .map(|l| serde_json::json!({ "language": l.language, "file_count": l.file_count }))
            .collect();
        map.insert("language_stats".into(), serde_json::Value::Array(ls));
    }

    if !ctx.key_files.is_empty() {
        let kf: Vec<_> = ctx
            .key_files
            .iter()
            .map(|f| {
                serde_json::json!({
                    "path": f.path,
                    "imports": f.imports,
                    "dependents": f.dependents,
                })
            })
            .collect();
        map.insert("key_files".into(), serde_json::Value::Array(kf));
    }

    if let Some(ref ts) = ctx.last_synced {
        map.insert(
            "last_synced".into(),
            serde_json::Value::String(ts.to_rfc3339()),
        );
    }

    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_default()
}

/// Format ProjectContext as markdown for fallback (when oneshot fails).
/// Only includes sections that have data.
pub fn context_to_markdown(ctx: &ProjectContext) -> String {
    let mut md = String::new();

    if let Some(ref p) = ctx.project {
        md.push_str(&format!("## Projet actif : {} ({})\n", p.name, p.slug));
        md.push_str(&format!("Root: {}\n", p.root_path));
        if let Some(ref desc) = p.description {
            md.push_str(&format!("Description: {}\n", desc));
        }
        md.push('\n');
    }

    if let Some(ref w) = ctx.workspace {
        md.push_str(&format!("## Workspace : {} ({})\n", w.name, w.slug));
        if let Some(ref desc) = w.description {
            md.push_str(&format!("{}\n", desc));
        }
        md.push('\n');
    }

    if !ctx.active_plans.is_empty() {
        md.push_str("## Plans actifs\n");
        for plan in &ctx.active_plans {
            md.push_str(&format!(
                "- **{}** ({:?}, priorité {})\n",
                plan.title, plan.status, plan.priority
            ));
        }
        md.push('\n');
    }

    if !ctx.plan_constraints.is_empty() {
        md.push_str("## Contraintes\n");
        for c in &ctx.plan_constraints {
            md.push_str(&format!("- [{:?}] {}\n", c.constraint_type, c.description));
        }
        md.push('\n');
    }

    if !ctx.guidelines.is_empty() {
        md.push_str("## Guidelines\n");
        for g in &ctx.guidelines {
            md.push_str(&format!("- [{:?}] {}\n", g.importance, g.content));
        }
        md.push('\n');
    }

    if !ctx.gotchas.is_empty() {
        md.push_str("## Gotchas\n");
        for g in &ctx.gotchas {
            md.push_str(&format!("- {}\n", g.content));
        }
        md.push('\n');
    }

    if !ctx.milestones.is_empty() {
        md.push_str("## Milestones\n");
        for m in &ctx.milestones {
            let date = m
                .target_date
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "pas de date".into());
            md.push_str(&format!(
                "- **{}** ({:?}) — cible: {}\n",
                m.title, m.status, date
            ));
        }
        md.push('\n');
    }

    if !ctx.releases.is_empty() {
        md.push_str("## Releases\n");
        for r in &ctx.releases {
            let date = r
                .target_date
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "pas de date".into());
            md.push_str(&format!(
                "- **v{}** ({:?}) — cible: {}\n",
                r.version, r.status, date
            ));
        }
        md.push('\n');
    }

    if !ctx.language_stats.is_empty() {
        md.push_str("## Langages\n");
        for l in &ctx.language_stats {
            md.push_str(&format!("- {} ({} fichiers)\n", l.language, l.file_count));
        }
        md.push('\n');
    }

    if !ctx.key_files.is_empty() {
        md.push_str("## Fichiers clés\n");
        for f in &ctx.key_files {
            md.push_str(&format!(
                "- `{}` ({} imports, {} dépendants)\n",
                f.path, f.imports, f.dependents
            ));
        }
        md.push('\n');
    }

    // Only show sync warnings if we have a project
    if ctx.project.is_some() {
        match ctx.last_synced {
            Some(ts) => {
                let ago = Utc::now().signed_duration_since(ts);
                if ago.num_hours() < 1 {
                    md.push_str(&format!(
                        "- **Dernière sync** : il y a {} min (à jour)\n\n",
                        ago.num_minutes().max(1)
                    ));
                } else if ago.num_hours() < 24 {
                    md.push_str(&format!(
                        "- **Dernière sync** : il y a {}h\n\n",
                        ago.num_hours()
                    ));
                } else {
                    md.push_str(&format!(
                        "- **Dernière sync** : il y a {}j — si le code a changé depuis, lance `sync_project`\n\n",
                        ago.num_days()
                    ));
                }
            }
            None => {
                md.push_str("⚠️ **Aucun sync** — le code n'a jamais été synchronisé. Lance `sync_project` avant d'explorer le code.\n\n");
            }
        }
    }

    md
}

// ============================================================================
// Oneshot refinement prompt
// ============================================================================

/// Build the prompt sent to the oneshot Opus model for context refinement.
/// The oneshot analyzes the user's request + raw project context JSON and
/// produces a concise "## Contexte actif" section (<500 words).
pub fn build_refinement_prompt(user_message: &str, context_json: &str) -> String {
    format!(
        "Tu es un constructeur de contexte pour un agent de développement.\n\
         \n\
         Voici la demande initiale de l'utilisateur :\n\
         ---\n\
         {user_message}\n\
         ---\n\
         \n\
         Voici les données du projet actif (JSON) :\n\
         ---\n\
         {context_json}\n\
         ---\n\
         \n\
         Génère une section \"## Contexte actif\" concise et pertinente pour le prompt système\n\
         de l'agent qui va traiter cette demande. Inclus UNIQUEMENT les informations utiles\n\
         pour cette demande spécifique :\n\
         \n\
         - Infos projet (nom, description, état du sync) si pertinent\n\
         - Workspace parent si pertinent\n\
         - Plans en cours avec leur progression si la demande touche à la planification ou l'exécution\n\
         - Guidelines et gotchas pertinents à la demande\n\
         - Contraintes actives si pertinent\n\
         - Milestones/releases avec dates si la demande touche à la roadmap\n\
         - Stats du code (langages, fichiers clés) si la demande touche au code\n\
         - Avertissements (sync obsolète, notes à reviewer) si applicable\n\
         \n\
         Format : markdown, bullet points, court et actionnable.\n\
         Ne dépasse pas 500 mots.",
        user_message = user_message,
        context_json = context_json,
    )
}

// ============================================================================
// Assembler — combines base + dynamic context
// ============================================================================

/// Assemble the final system prompt by combining the hardcoded base
/// with the dynamic contextual section.
pub fn assemble_prompt(base: &str, dynamic_context: &str) -> String {
    if dynamic_context.is_empty() {
        return base.to_string();
    }
    format!("{}\n\n---\n\n{}", base, dynamic_context)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_system_prompt_contains_key_sections() {
        assert!(BASE_SYSTEM_PROMPT.contains("EXCLUSIVEMENT les outils MCP"));
        assert!(BASE_SYSTEM_PROMPT.contains("Modèle de données"));
        assert!(BASE_SYSTEM_PROMPT.contains("Tree-sitter"));
        assert!(BASE_SYSTEM_PROMPT.contains("Workflow Git"));
        assert!(BASE_SYSTEM_PROMPT.contains("Protocole d'exécution"));
        assert!(BASE_SYSTEM_PROMPT.contains("Protocole de planification"));
        assert!(BASE_SYSTEM_PROMPT.contains("Gestion des statuts"));
        assert!(BASE_SYSTEM_PROMPT.contains("Bonnes pratiques"));
    }

    #[test]
    fn test_base_system_prompt_mcp_first_directive() {
        assert!(BASE_SYSTEM_PROMPT.contains("PAS"));
        assert!(BASE_SYSTEM_PROMPT.contains("TodoWrite"));
        assert!(BASE_SYSTEM_PROMPT.contains("EnterPlanMode"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_plan"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_task"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_step"));
    }

    #[test]
    fn test_base_system_prompt_has_task_decomposition_example() {
        assert!(BASE_SYSTEM_PROMPT.contains("Ajouter l'endpoint GET /api/releases/:id"));
        assert!(BASE_SYSTEM_PROMPT.contains("Step 1"));
        assert!(BASE_SYSTEM_PROMPT.contains("Step 5"));
    }

    #[test]
    fn test_base_system_prompt_has_status_transitions() {
        assert!(BASE_SYSTEM_PROMPT.contains("draft"));
        assert!(BASE_SYSTEM_PROMPT.contains("approved"));
        assert!(BASE_SYSTEM_PROMPT.contains("in_progress"));
        assert!(BASE_SYSTEM_PROMPT.contains("completed"));
        assert!(BASE_SYSTEM_PROMPT.contains("blocked"));
    }

    #[test]
    fn test_context_to_json_empty() {
        let ctx = ProjectContext::default();
        let json = context_to_json(&ctx);
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_context_to_json_with_project() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProject".into(),
                slug: "test-project".into(),
                root_path: "/tmp/test".into(),
                description: Some("A test project".into()),
                created_at: Utc::now(),
                last_synced: None,
            }),
            ..Default::default()
        };
        let json = context_to_json(&ctx);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["project"]["name"], "TestProject");
        assert_eq!(parsed["project"]["slug"], "test-project");
    }

    #[test]
    fn test_context_to_markdown_empty() {
        let ctx = ProjectContext::default();
        let md = context_to_markdown(&ctx);
        assert!(md.is_empty());
    }

    #[test]
    fn test_context_to_markdown_with_project() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "MyProject".into(),
                slug: "my-project".into(),
                root_path: "/home/user/code".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: None,
            }),
            ..Default::default()
        };
        let md = context_to_markdown(&ctx);
        assert!(md.contains("MyProject"));
        assert!(md.contains("my-project"));
        assert!(md.contains("Aucun sync"));
    }

    #[test]
    fn test_context_to_markdown_partial_data() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "Partial".into(),
                slug: "partial".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
            }),
            language_stats: vec![LanguageStatsNode {
                language: "Rust".into(),
                file_count: 42,
            }],
            ..Default::default()
        };
        let md = context_to_markdown(&ctx);
        assert!(md.contains("Partial"));
        assert!(md.contains("Rust"));
        assert!(md.contains("42 fichiers"));
        // Should NOT contain sections with no data
        assert!(!md.contains("Guidelines"));
        assert!(!md.contains("Gotchas"));
        assert!(!md.contains("Milestones"));
    }

    #[test]
    fn test_build_refinement_prompt_contains_inputs() {
        let prompt = build_refinement_prompt("Implémente le login", r#"{"project":"test"}"#);
        assert!(prompt.contains("Implémente le login"));
        assert!(prompt.contains(r#"{"project":"test"}"#));
        assert!(prompt.contains("500 mots"));
        assert!(prompt.contains("## Contexte actif"));
    }

    #[test]
    fn test_assemble_prompt_no_context() {
        let result = assemble_prompt("base prompt", "");
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_assemble_prompt_with_context() {
        let result = assemble_prompt("base prompt", "## Contexte actif\n- info");
        assert!(result.contains("base prompt"));
        assert!(result.contains("---"));
        assert!(result.contains("## Contexte actif"));
    }

    // ================================================================
    // fetch_project_context tests (with MockGraphStore)
    // ================================================================

    #[tokio::test]
    async fn test_fetch_project_context_existing_project() {
        use crate::test_helpers::{mock_app_state, test_project};

        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project.slug)
            .await
            .unwrap();

        assert!(ctx.project.is_some());
        assert_eq!(ctx.project.as_ref().unwrap().name, project.name);
        assert!(ctx.active_plans.is_empty()); // no plans created
        assert!(ctx.guidelines.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_project_context_nonexistent_project() {
        use crate::test_helpers::mock_app_state;

        let state = mock_app_state();

        let ctx = fetch_project_context(&state.neo4j, "nonexistent-slug")
            .await
            .unwrap();

        assert!(ctx.project.is_none());
        assert!(ctx.active_plans.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_project_context_with_plans() {
        use crate::test_helpers::{mock_app_state, test_plan, test_project};

        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let plan = test_plan();
        state.neo4j.create_plan(&plan).await.unwrap();
        state
            .neo4j
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project.slug)
            .await
            .unwrap();

        assert!(ctx.project.is_some());
        assert!(!ctx.active_plans.is_empty());
        assert_eq!(ctx.active_plans[0].title, plan.title);
    }
}
