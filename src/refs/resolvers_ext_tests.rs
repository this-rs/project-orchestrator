//! The kinds added after the first five, each proven the same ways: it is
//! found with its scope, it is refused outside the session's project, an
//! unknown id reads `not_found`, and the picker finds it by a fragment of its
//! name. The sensitive kinds get their own refusals.

use std::sync::Arc;

use uuid::Uuid;

use super::access::{AccessPolicy, Principal, Resolution, SessionScope};
use super::resolvers::GraphRefSource;
use super::search::{search, RefSearchParams};
use super::test_support::{user, world, World};
use super::types::{EntityRef, RefId, RefKind};
use super::wire::RefStatus;
use crate::neo4j::models::{FileNode, PersonaNode};
use crate::neo4j::GraphStore;
use crate::skills::SkillNode;
use crate::test_helpers::{test_chat_session, test_commit, test_milestone, test_release};

const ROOT_A: &str = "/work/alpha";
const ROOT_B: &str = "/work/beta";

struct Ext {
    w: World,
    milestone_a: Uuid,
    milestone_b: Uuid,
    release_a: Uuid,
    release_b: Uuid,
    protocol_a: Uuid,
    protocol_b: Uuid,
    persona_a: Uuid,
    persona_b: Uuid,
    persona_global: Uuid,
    skill_a: Uuid,
    skill_b: Uuid,
    conv_a: Uuid,
    conv_b: Uuid,
    conv_untitled: Uuid,
    hash_a: String,
    hash_b: String,
    hash_unlinked: String,
}

const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const HASH_U: &str = "cccccccccccccccccccccccccccccccccccccccc";

fn persona(project: Option<Uuid>, name: &str, description: &str) -> PersonaNode {
    serde_json::from_value(serde_json::json!({
        "id": Uuid::new_v4(),
        "project_id": project,
        "name": name,
        "description": description,
        "created_at": chrono::Utc::now(),
    }))
    .unwrap()
}

async fn ext() -> Ext {
    let w = world().await;
    let g = w.graph.clone();
    g.projects.write().await.get_mut(&w.a).unwrap().root_path = ROOT_A.into();
    g.projects.write().await.get_mut(&w.b).unwrap().root_path = ROOT_B.into();

    let ma = test_milestone(w.a, "Jalon fondations");
    let mb = test_milestone(w.b, "Jalon facturation");
    g.create_milestone(&ma).await.unwrap();
    g.create_milestone(&mb).await.unwrap();
    let ra = test_release(w.a, "1.4.0");
    let rb = test_release(w.b, "2.0.0");
    g.create_release(&ra).await.unwrap();
    g.create_release(&rb).await.unwrap();

    let pa = crate::protocol::Protocol::new(w.a, "Protocole de revue", Uuid::new_v4());
    let pb = crate::protocol::Protocol::new(w.b, "Protocole de facturation", Uuid::new_v4());
    g.upsert_protocol(&pa).await.unwrap();
    g.upsert_protocol(&pb).await.unwrap();

    let persona_a = persona(Some(w.a), "Expert refs", "connait les references");
    let persona_b = persona(Some(w.b), "Expert facturation", "connait la facture");
    let persona_global = persona(None, "Relecteur global", "portable");
    for p in [&persona_a, &persona_b, &persona_global] {
        g.create_persona(p).await.unwrap();
    }
    let skill_a = SkillNode::new(w.a, "Skill refs");
    let skill_b = SkillNode::new(w.b, "Skill facturation");
    g.create_skill(&skill_a).await.unwrap();
    g.create_skill(&skill_b).await.unwrap();

    // commits: A is linked to a task of alpha, B to a task of beta, U to nothing.
    for (h, m) in [
        (HASH_A, "feat(refs): fondations\n\ncorps"),
        (HASH_B, "fix(billing): arrondis"),
        (HASH_U, "chore: orphelin"),
    ] {
        g.create_commit(&test_commit(h, m)).await.unwrap();
    }
    g.link_commit_to_task(HASH_A, w.task_a.id).await.unwrap();
    g.link_commit_to_plan(HASH_B, w.plan_b.id).await.unwrap();

    // files
    for (project, path) in [
        (w.a, format!("{ROOT_A}/src/refs/mod.rs")),
        (w.a, format!("{ROOT_A}/src/refs/.env.local")),
        (w.a, format!("{ROOT_A}/.git/config")),
        (w.a, format!("{ROOT_A}/config/server.pem")),
        (w.a, format!("{ROOT_A}/docs/secrets.md")),
        (w.b, format!("{ROOT_B}/src/billing.rs")),
    ] {
        let f = FileNode {
            path: path.clone(),
            language: "rust".into(),
            hash: "h".into(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(project),
        };
        g.batch_upsert_files(&[f]).await.unwrap();
        g.link_file_to_project(&path, project).await.unwrap();
    }

    // conversations
    let mut ca = test_chat_session(Some("alpha"));
    ca.title = Some("Discussion sur les fondations".into());
    ca.preview = Some("bonjour".into());
    let mut cb = test_chat_session(Some("beta"));
    cb.title = Some("Discussion facturation".into());
    let mut cu = test_chat_session(Some("alpha"));
    cu.title = None;
    cu.preview = Some("un aperçu sans titre fondations".into());
    for c in [&ca, &cb, &cu] {
        g.create_chat_session(c).await.unwrap();
    }

    Ext {
        milestone_a: ma.id,
        milestone_b: mb.id,
        release_a: ra.id,
        release_b: rb.id,
        protocol_a: pa.id,
        protocol_b: pb.id,
        persona_a: persona_a.id,
        persona_b: persona_b.id,
        persona_global: persona_global.id,
        skill_a: skill_a.id,
        skill_b: skill_b.id,
        conv_a: ca.id,
        conv_b: cb.id,
        conv_untitled: cu.id,
        hash_a: HASH_A.into(),
        hash_b: HASH_B.into(),
        hash_unlinked: HASH_U.into(),
        w,
    }
}

fn in_project(project: Uuid, workspace: Option<Uuid>) -> Principal {
    Principal::Session(SessionScope {
        project: Some(project),
        workspace,
    })
}

async fn resolve(e: &Ext, p: &Principal, kind: RefKind, id: impl Into<RefId>) -> Resolution {
    let g: Arc<dyn GraphStore> = e.w.graph.clone();
    AccessPolicy::open_instance()
        .resolve_checked(p, &GraphRefSource::new(g), &EntityRef::new(kind, id))
        .await
}

fn found(r: Resolution) -> crate::refs::access::RefMeta {
    match r {
        Resolution::Found(m) => *m,
        other => panic!("expected a found entity, got {other:?}"),
    }
}

fn not_readable(r: &Resolution) -> bool {
    matches!(r, Resolution::Forbidden | Resolution::NotFound)
        && r.status(crate::refs::access::Disclosure::Uniform) == RefStatus::NotFound
}

async fn suggest(e: &Ext, kind: &str, q: &str, project: Option<Uuid>) -> Vec<String> {
    let query = RefSearchParams {
        q: Some(q.into()),
        kinds: Some(kind.into()),
        project_id: project.map(|p| p.to_string()),
        ..Default::default()
    }
    .into_query()
    .unwrap();
    search(
        e.w.graph.clone(),
        &AccessPolicy::open_instance(),
        &user(),
        &query,
    )
    .await
    .unwrap()
    .items
    .into_iter()
    .map(|i| i.id.to_string())
    .collect()
}

/// (kind, id in alpha, id in beta, a fragment of the alpha one, its label)
fn cases(e: &Ext) -> Vec<(RefKind, RefId, RefId, &'static str, &'static str)> {
    vec![
        (
            RefKind::Project,
            e.w.a.into(),
            e.w.b.into(),
            "alph",
            "Alpha",
        ),
        (
            RefKind::Milestone,
            e.milestone_a.into(),
            e.milestone_b.into(),
            "fondat",
            "Jalon fondations",
        ),
        (
            RefKind::Release,
            e.release_a.into(),
            e.release_b.into(),
            "release 1.4",
            "Release 1.4.0",
        ),
        (
            RefKind::Protocol,
            e.protocol_a.into(),
            e.protocol_b.into(),
            "revue",
            "Protocole de revue",
        ),
        (
            RefKind::Persona,
            e.persona_a.into(),
            e.persona_b.into(),
            "expert re",
            "Expert refs",
        ),
        (
            RefKind::Skill,
            e.skill_a.into(),
            e.skill_b.into(),
            "skill re",
            "Skill refs",
        ),
        (
            RefKind::Conversation,
            e.conv_a.into(),
            e.conv_b.into(),
            "fondat",
            "Discussion sur les fondations",
        ),
    ]
}

#[tokio::test]
async fn each_kind_is_found_with_its_scope_in_its_own_project() {
    let e = ext().await;
    for (kind, mine, _, _, label) in cases(&e) {
        let m = found(resolve(&e, &in_project(e.w.a, Some(e.w.ws.id)), kind, mine.clone()).await);
        assert_eq!(m.kind, kind);
        assert_eq!(m.id, mine);
        assert_eq!(m.label, label, "{kind}");
        assert_eq!(
            m.project.as_ref().map(|p| p.slug.as_str()),
            Some("alpha"),
            "{kind}"
        );
        assert_eq!(
            m.workspace.as_ref().map(|w| w.slug.as_str()),
            Some("po"),
            "{kind}"
        );
    }
}

#[tokio::test]
async fn each_kind_of_another_project_reads_not_found_for_a_project_session() {
    let e = ext().await;
    for (kind, _, theirs, _, _) in cases(&e) {
        let r = resolve(&e, &in_project(e.w.a, Some(e.w.ws.id)), kind, theirs).await;
        assert!(not_readable(&r), "{kind}: {r:?}");
        // and the same reference is readable from its own project.
    }
    for (kind, mine, _, _, _) in cases(&e) {
        // alpha's entity from a beta session
        let r = resolve(&e, &in_project(e.w.b, None), kind, mine).await;
        assert!(not_readable(&r), "{kind}: {r:?}");
    }
}

#[tokio::test]
async fn an_unknown_id_reads_not_found_for_each_kind() {
    let e = ext().await;
    for (kind, ..) in cases(&e) {
        let r = resolve(&e, &user(), kind, Uuid::new_v4()).await;
        assert_eq!(r, Resolution::NotFound, "{kind}");
    }
}

#[tokio::test]
async fn the_picker_finds_each_kind_by_a_fragment_of_its_name_inside_its_project() {
    let e = ext().await;
    for (kind, mine, theirs, fragment, _) in cases(&e) {
        let hits = suggest(&e, kind.as_str(), fragment, Some(e.w.a)).await;
        assert!(hits.contains(&mine.to_string()), "{kind}: {hits:?}");
        assert!(!hits.contains(&theirs.to_string()), "{kind}: {hits:?}");
        // another project's search never shows alpha's entity
        let other = suggest(&e, kind.as_str(), fragment, Some(e.w.b)).await;
        assert!(!other.contains(&mine.to_string()), "{kind}: {other:?}");
    }
}

#[tokio::test]
async fn a_workspace_resolves_and_is_found_by_its_name() {
    let e = ext().await;
    let id = e.w.ws.id;
    let m = found(resolve(&e, &in_project(e.w.a, Some(id)), RefKind::Workspace, id).await);
    assert_eq!(m.label, "PO");
    assert_eq!(m.workspace.as_ref().map(|w| w.id), Some(id));
    assert!(m.project.is_none());
    // a session of another workspace (or of a project in none) does not read it
    let other = in_project(e.w.b, Some(Uuid::new_v4()));
    assert!(not_readable(
        &resolve(&e, &other, RefKind::Workspace, id).await
    ));
    assert!(not_readable(
        &resolve(&e, &in_project(e.w.b, None), RefKind::Workspace, id).await
    ));
    assert!(suggest(&e, "workspace", "po", None)
        .await
        .contains(&id.to_string()));
    assert_eq!(
        resolve(&e, &user(), RefKind::Workspace, Uuid::new_v4()).await,
        Resolution::NotFound
    );
}

#[tokio::test]
async fn a_release_without_a_title_is_named_by_its_version() {
    let e = ext().await;
    let mut r = test_release(e.w.a, "3.1.0");
    r.title = None;
    e.w.graph.create_release(&r).await.unwrap();
    let m = found(resolve(&e, &user(), RefKind::Release, r.id).await);
    assert_eq!(m.label, "3.1.0");
}

#[tokio::test]
async fn a_persona_may_be_global_and_every_session_reads_it() {
    let e = ext().await;
    for p in [
        in_project(e.w.a, None),
        in_project(e.w.b, None),
        Principal::Session(SessionScope::default()),
    ] {
        let m = found(resolve(&e, &p, RefKind::Persona, e.persona_global).await);
        assert_eq!(m.label, "Relecteur global");
        assert!(m.project.is_none());
    }
    // `@` finds actors with no project filter at all
    let hits = suggest(&e, "persona,skill", "re", None).await;
    for id in [e.persona_a, e.persona_global, e.skill_a] {
        assert!(hits.contains(&id.to_string()), "{hits:?}");
    }
}

#[tokio::test]
async fn a_conversation_is_found_by_its_title_not_by_its_preview_alone() {
    let e = ext().await;
    let hits = suggest(&e, "conversation", "fondat", Some(e.w.a)).await;
    assert!(hits.contains(&e.conv_a.to_string()));
    // the untitled session only has the fragment in its preview: it may be
    // offered (body tier) but never before the one the title names.
    if let Some(pos) = hits.iter().position(|h| *h == e.conv_untitled.to_string()) {
        assert!(
            pos > hits
                .iter()
                .position(|h| *h == e.conv_a.to_string())
                .unwrap()
        );
    }
    let m = found(resolve(&e, &user(), RefKind::Conversation, e.conv_untitled).await);
    assert_eq!(m.label, "un aperçu sans titre fondations");
}

#[tokio::test]
async fn a_conversation_of_a_project_that_is_gone_reads_not_found() {
    let e = ext().await;
    let mut s = test_chat_session(Some("deleted"));
    s.title = Some("Orpheline".into());
    e.w.graph.create_chat_session(&s).await.unwrap();
    assert_eq!(
        resolve(&e, &user(), RefKind::Conversation, s.id).await,
        Resolution::NotFound
    );
}

// ---- commit -------------------------------------------------------------

fn commit_id(project: Uuid, hash: &str) -> RefId {
    crate::refs::types::canonical_id(RefKind::Commit, &format!("{project}:{hash}")).unwrap()
}

#[tokio::test]
async fn a_commit_is_found_through_the_plan_or_task_of_its_project() {
    let e = ext().await;
    let m = found(
        resolve(
            &e,
            &in_project(e.w.a, None),
            RefKind::Commit,
            commit_id(e.w.a, &e.hash_a),
        )
        .await,
    );
    assert_eq!(m.label, "feat(refs): fondations");
    assert_eq!(m.project.as_ref().map(|p| p.slug.as_str()), Some("alpha"));
    assert!(m.subtitle.as_deref().unwrap().starts_with("aaaaaaa"));
    // linked through a PLAN, in beta
    let b = found(
        resolve(
            &e,
            &in_project(e.w.b, None),
            RefKind::Commit,
            commit_id(e.w.b, &e.hash_b),
        )
        .await,
    );
    assert_eq!(b.label, "fix(billing): arrondis");
}

#[tokio::test]
async fn a_commit_that_belongs_to_no_task_or_plan_of_the_project_reads_not_found() {
    let e = ext().await;
    let any = Principal::User(Uuid::new_v4());
    // not linked to anything
    assert_eq!(
        resolve(
            &e,
            &any,
            RefKind::Commit,
            commit_id(e.w.a, &e.hash_unlinked)
        )
        .await,
        Resolution::NotFound
    );
    // a hash of beta claimed under alpha's project id
    assert_eq!(
        resolve(&e, &any, RefKind::Commit, commit_id(e.w.a, &e.hash_b)).await,
        Resolution::NotFound
    );
    // a hash the store has never seen
    assert_eq!(
        resolve(&e, &any, RefKind::Commit, commit_id(e.w.a, &"d".repeat(40))).await,
        Resolution::NotFound
    );
    // a project that does not exist
    assert_eq!(
        resolve(
            &e,
            &any,
            RefKind::Commit,
            commit_id(Uuid::new_v4(), &e.hash_a)
        )
        .await,
        Resolution::NotFound
    );
}

#[tokio::test]
async fn a_commit_is_not_readable_by_another_project_or_an_unattached_session() {
    let e = ext().await;
    let id = commit_id(e.w.a, &e.hash_a);
    assert!(not_readable(
        &resolve(&e, &in_project(e.w.b, None), RefKind::Commit, id.clone()).await
    ));
    assert!(not_readable(
        &resolve(
            &e,
            &Principal::Session(SessionScope::default()),
            RefKind::Commit,
            id
        )
        .await
    ));
}

#[tokio::test]
async fn commits_are_suggested_only_inside_a_project_and_by_their_message() {
    let e = ext().await;
    let id = commit_id(e.w.a, &e.hash_a).to_string();
    assert!(suggest(&e, "commit", "fondat", Some(e.w.a))
        .await
        .contains(&id));
    assert!(
        suggest(&e, "commit", "fondat", None).await.is_empty(),
        "no scope, no suggestion"
    );
    assert!(!suggest(&e, "commit", "fondat", Some(e.w.b))
        .await
        .contains(&id));
    assert!(!suggest(&e, "commit", "orphelin", Some(e.w.a))
        .await
        .iter()
        .any(|i| i.ends_with(HASH_U)));
}

// ---- file ---------------------------------------------------------------

fn file_id(project: Uuid, rel: &str) -> RefId {
    crate::refs::types::canonical_id(RefKind::File, &format!("{project}:{rel}")).unwrap()
}

#[tokio::test]
async fn a_file_is_found_by_its_path_inside_its_project() {
    let e = ext().await;
    let m = found(
        resolve(
            &e,
            &in_project(e.w.a, None),
            RefKind::File,
            file_id(e.w.a, "src/refs/mod.rs"),
        )
        .await,
    );
    assert_eq!(m.label, "mod.rs");
    assert_eq!(m.subtitle.as_deref(), Some("src/refs/mod.rs"));
    assert_eq!(m.project.as_ref().map(|p| p.slug.as_str()), Some("alpha"));
    let hits = suggest(&e, "file", "refs/mod", Some(e.w.a)).await;
    assert!(
        hits.contains(&file_id(e.w.a, "src/refs/mod.rs").to_string()),
        "{hits:?}"
    );
}

#[tokio::test]
async fn a_file_of_another_project_or_that_is_not_in_the_graph_reads_not_found() {
    let e = ext().await;
    let any = in_project(e.w.a, None);
    // beta's file claimed under alpha, and the reverse
    assert_eq!(
        resolve(&e, &any, RefKind::File, file_id(e.w.a, "src/billing.rs")).await,
        Resolution::NotFound
    );
    assert!(not_readable(
        &resolve(&e, &any, RefKind::File, file_id(e.w.b, "src/billing.rs")).await
    ));
    // not parsed into the graph
    assert_eq!(
        resolve(&e, &any, RefKind::File, file_id(e.w.a, "src/nope.rs")).await,
        Resolution::NotFound
    );
    // an unattached session cannot prove it may read a source file
    let r = resolve(
        &e,
        &Principal::Session(SessionScope::default()),
        RefKind::File,
        file_id(e.w.a, "src/refs/mod.rs"),
    )
    .await;
    assert!(not_readable(&r), "{r:?}");
}

#[tokio::test]
async fn hidden_and_secret_looking_files_are_never_named() {
    let e = ext().await;
    let me = in_project(e.w.a, None);
    for rel in [
        "src/refs/.env.local",
        ".git/config",
        "config/server.pem",
        "docs/secrets.md",
    ] {
        // they ARE in the graph...
        let abs = format!("{ROOT_A}/{rel}");
        assert!(e.w.graph.get_file(&abs).await.unwrap().is_some(), "{rel}");
        // ...and are still refused, to resolve and to suggest.
        assert_eq!(
            resolve(&e, &me, RefKind::File, file_id(e.w.a, rel)).await,
            Resolution::NotFound,
            "{rel}"
        );
    }
    for q in ["env", "config", "pem", "secrets"] {
        let hits = suggest(&e, "file", q, Some(e.w.a)).await;
        assert!(hits.is_empty(), "{q}: {hits:?}");
    }
}

#[tokio::test]
async fn files_are_suggested_only_inside_a_scope() {
    let e = ext().await;
    assert!(suggest(&e, "file", "mod", None).await.is_empty());
    assert!(!suggest(&e, "file", "mod", Some(e.w.a)).await.is_empty());
    assert!(suggest(&e, "file", "mod", Some(e.w.b)).await.is_empty());
}

#[tokio::test]
async fn a_file_is_read_only_by_the_project_the_graph_says_owns_it() {
    let e = ext().await;
    // A file whose path lies under alpha's root but that the graph attaches to
    // BETA (a moved checkout, a wrong link, or a forged id): the id claims
    // alpha, the graph says beta.
    let path = format!("{ROOT_A}/src/shared.rs");
    let f = FileNode {
        path: path.clone(),
        language: "rust".into(),
        hash: "h".into(),
        last_parsed: chrono::Utc::now(),
        project_id: Some(e.w.b),
    };
    e.w.graph.batch_upsert_files(&[f]).await.unwrap();
    e.w.graph.link_file_to_project(&path, e.w.b).await.unwrap();
    let claimed_by_alpha = file_id(e.w.a, "src/shared.rs");
    // read from alpha's own session: not found, the owner is not alpha
    let r = resolve(
        &e,
        &in_project(e.w.a, None),
        RefKind::File,
        claimed_by_alpha.clone(),
    )
    .await;
    assert_eq!(r, Resolution::NotFound);
    // the SAME file, once it belongs to alpha, is found by the same session
    let mut mine = e.w.graph.get_file(&path).await.unwrap().unwrap();
    mine.project_id = Some(e.w.a);
    e.w.graph.batch_upsert_files(&[mine]).await.unwrap();
    let m = found(
        resolve(
            &e,
            &in_project(e.w.a, None),
            RefKind::File,
            claimed_by_alpha.clone(),
        )
        .await,
    );
    assert_eq!(m.label, "shared.rs");
    // and a session of beta still cannot read it through alpha's id
    assert!(not_readable(
        &resolve(
            &e,
            &in_project(e.w.b, None),
            RefKind::File,
            claimed_by_alpha
        )
        .await
    ));
}

#[tokio::test]
async fn a_hidden_file_listed_by_the_graph_is_not_suggested_either() {
    // The picker path, apart from the resolve path.
    let e = ext().await;
    assert!(suggest(&e, "file", "mod.rs", Some(e.w.a)).await.len() == 1);
    assert!(suggest(&e, "file", ".env", Some(e.w.a)).await.is_empty());
    assert!(suggest(&e, "file", "local", Some(e.w.a)).await.is_empty());
}

#[tokio::test]
async fn without_text_each_kind_is_listed_newest_first() {
    use crate::refs::test_support::note_with;
    let e = ext().await;
    let mut ids = Vec::new();
    for i in 0..3 {
        let n = note_with(
            Some(e.w.a),
            crate::notes::NoteType::Tip,
            &format!("note {i}"),
            vec![],
        );
        e.w.graph.create_note(&n).await.unwrap();
        ids.push(n.id.to_string());
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let got: Vec<String> = suggest(&e, "note", "", Some(e.w.a))
        .await
        .into_iter()
        .filter(|i| ids.contains(i))
        .collect();
    ids.reverse();
    assert_eq!(got, ids, "empty text: no relevance, the most recent first");
}
