//! `GET /api/refs/search`: parse the query, ask each requested kind's resolver
//! for candidates, keep what is in the requested scope **and** that the access
//! policy lets this principal see, and shape the answer the golden fixture
//! `tests/fixtures/refs/search_response.json` pins.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use uuid::Uuid;

use super::access::{AccessPolicy, Principal, RefMeta, RefSource, Resolution};
use super::rank;
use super::registry::{lookup, Lookup};
use super::resolvers::{Candidates, GraphRefSource, KindResolver, Memo};
use super::types::{EntityRef, RefKind};
use super::validate::{validate_search, InvalidReason, RefsInvalid};
use super::wire::{RefSearchItem, RefSearchResponse};
use crate::neo4j::GraphStore;

/// The query string as it arrives: everything optional, everything a string, so
/// that each malformed part gets its own `refs_invalid` reason instead of a
/// generic extractor rejection.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RefSearchParams {
    pub q: Option<String>,
    pub kinds: Option<String>,
    pub project_id: Option<String>,
    pub workspace_slug: Option<String>,
    pub limit: Option<String>,
}

/// A validated search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchQuery {
    /// Trimmed, as typed (case kept).
    pub q: String,
    /// Requested kinds, deduplicated, in request order.
    pub kinds: Vec<RefKind>,
    pub project_id: Option<Uuid>,
    pub workspace_slug: Option<String>,
    pub limit: usize,
}

fn invalid(reason: InvalidReason, index: Option<usize>) -> RefsInvalid {
    RefsInvalid { reason, index }
}

impl RefSearchParams {
    pub fn into_query(self) -> Result<SearchQuery, RefsInvalid> {
        let raw_q = self.q.unwrap_or_default();
        let limit = match self.limit.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(s) => Some(
                s.parse::<usize>()
                    .map_err(|_| invalid(InvalidReason::BadLimit, None))?,
            ),
        };
        let limit = validate_search(&raw_q, limit).map_err(|r| invalid(r, None))?;

        let mut kinds: Vec<RefKind> = Vec::new();
        match self.kinds.as_deref().map(str::trim) {
            None | Some("") => kinds.extend(RefKind::HISTORICAL),
            Some(list) => {
                for (i, name) in list.split(',').enumerate() {
                    let kind = match lookup(name.trim()) {
                        Lookup::Active(k) => k,
                        Lookup::Reserved(_) => {
                            return Err(invalid(InvalidReason::KindDisabled, Some(i)))
                        }
                        Lookup::Unknown => {
                            return Err(invalid(InvalidReason::UnknownKind, Some(i)))
                        }
                    };
                    if !kinds.contains(&kind) {
                        kinds.push(kind);
                    }
                }
            }
        }

        let project_id = match self.project_id.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(s) => match Uuid::parse_str(s) {
                Ok(id) if !id.is_nil() => Some(id),
                _ => return Err(invalid(InvalidReason::BadId, None)),
            },
        };
        let workspace_slug = self
            .workspace_slug
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        Ok(SearchQuery {
            q: raw_q.trim().to_string(),
            kinds,
            project_id,
            workspace_slug,
            limit,
        })
    }
}

/// Is `meta` inside the requested project / workspace? An entity with no
/// project is outside any scope that names one (fail closed).
pub fn in_scope(meta: &RefMeta, project_id: Option<Uuid>, workspace_slug: Option<&str>) -> bool {
    if let Some(pid) = project_id {
        if meta.project.as_ref().map(|p| p.id) != Some(pid) {
            return false;
        }
    }
    if let Some(slug) = workspace_slug {
        if meta.workspace.as_ref().map(|w| w.slug.as_str()) != Some(slug) {
            return false;
        }
    }
    true
}

/// An already-loaded entity presented to the policy as a source, so that the
/// verdict on a search hit is the same call as the verdict on a pasted
/// reference.
struct Loaded(RefMeta);

#[async_trait::async_trait]
impl RefSource for Loaded {
    async fn load_unchecked(&self, _r: &EntityRef) -> anyhow::Result<Option<RefMeta>> {
        Ok(Some(self.0.clone()))
    }
}

fn item(meta: RefMeta) -> RefSearchItem {
    RefSearchItem {
        kind: meta.kind,
        id: meta.id,
        label: meta.label,
        subtitle: meta.subtitle,
        project: meta.project,
        workspace: meta.workspace,
        entity_status: meta.entity_status,
    }
}

/// Take one from each kind in turn, so a kind with many hits does not starve
/// the others, until `limit`.
fn interleave(mut lists: Vec<VecDeque<RefMeta>>, limit: usize) -> Vec<RefMeta> {
    let mut out = Vec::new();
    while out.len() < limit && lists.iter().any(|l| !l.is_empty()) {
        for l in lists.iter_mut() {
            if out.len() >= limit {
                break;
            }
            if let Some(m) = l.pop_front() {
                out.push(m);
            }
        }
    }
    out
}

/// How long one kind may take before it is left out of the page.
pub const KIND_TIMEOUT: Duration = Duration::from_secs(2);

/// What one kind contributes: its candidates in scope that the policy lets
/// `principal` see.
async fn one_kind(
    resolver: &dyn KindResolver,
    candidates: &Candidates,
    query: &SearchQuery,
    policy: &AccessPolicy,
    principal: &Principal,
) -> anyhow::Result<Vec<Ranked>> {
    // A sensitive kind is searched inside a scope the caller names, never across the instance.
    if resolver.kind().is_sensitive()
        && query.project_id.is_none()
        && query.workspace_slug.is_none()
    {
        return Ok(Vec::new());
    }
    let mut memo = Memo::default();
    let mut kept = Vec::new();
    for cand in resolver.candidates(candidates, &mut memo).await? {
        // The text decides first: a row that does not match is not suggested.
        let Some(score) = rank::score(&candidates.needle, &cand.meta.label, &cand.body) else {
            continue;
        };
        let (meta, at) = (cand.meta, cand.at);
        if !in_scope(&meta, query.project_id, query.workspace_slug.as_deref()) {
            continue;
        }
        let r = EntityRef::new(meta.kind, meta.id.clone());
        let verdict = policy.resolve_checked(principal, &Loaded(meta), &r).await;
        if let Resolution::Found(found) = verdict {
            kept.push(Ranked {
                meta: *found,
                score,
                at,
            });
        }
    }
    sort_ranked(&mut kept);
    Ok(kept)
}

/// A suggestion that passed the scope and the policy, with its relevance.
struct Ranked {
    meta: RefMeta,
    score: u32,
    at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Best score first; equal scores, the most recent first (an entity with no
/// date after the dated ones). Stable, so the result is deterministic.
fn sort_ranked(rows: &mut [Ranked]) {
    rows.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| b.at.cmp(&a.at)));
}

/// Run a validated search for `principal`.
pub async fn search(
    graph: Arc<dyn GraphStore>,
    policy: &AccessPolicy,
    principal: &Principal,
    query: &SearchQuery,
) -> anyhow::Result<RefSearchResponse> {
    let source = GraphRefSource::new(graph);
    search_with(
        &|k| source.resolver(k),
        policy,
        principal,
        query,
        KIND_TIMEOUT,
    )
    .await
}

/// [`search`] over any source of resolvers, with an explicit per-kind timeout.
///
/// The kinds are asked concurrently. A kind that fails or exceeds `timeout` is
/// left out (and logged): the picker still shows the others. Only when every
/// requested kind failed is the search an error, so an outage is never an
/// empty list.
pub async fn search_with<'s, F>(
    resolver_of: &F,
    policy: &AccessPolicy,
    principal: &Principal,
    query: &SearchQuery,
    timeout: Duration,
) -> anyhow::Result<RefSearchResponse>
where
    F: Fn(RefKind) -> &'s dyn KindResolver,
    F: Sync,
{
    // Twice the page, capped: room for what the scope or the policy drops.
    let candidates = Candidates {
        needle: query.q.to_lowercase(),
        project_id: query.project_id,
        workspace_slug: query.workspace_slug.clone(),
        fetch: (query.limit * 2).min(100),
    };

    let asked = query.kinds.iter().map(|kind| {
        let resolver = resolver_of(*kind);
        let candidates = &candidates;
        async move {
            tokio::time::timeout(
                timeout,
                one_kind(resolver, candidates, query, policy, principal),
            )
            .await
        }
    });
    let answers = futures::future::join_all(asked).await;

    let mut lists = Vec::with_capacity(answers.len());
    let mut failed = 0;
    for (kind, answer) in query.kinds.iter().zip(answers) {
        match answer {
            Ok(Ok(kept)) => lists.push(kept),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, %kind, "reference search failed for a kind");
                failed += 1;
            }
            Err(_) => {
                tracing::warn!(%kind, "reference search timed out for a kind");
                failed += 1;
            }
        }
    }
    if failed > 0 && lists.is_empty() {
        anyhow::bail!("reference search failed for every requested kind");
    }

    // With a text, ONE ranking across the kinds, cut to the limit AFTER the
    // sort. Without one there is no relevance: each kind is listed by recency
    // and the kinds take turns, so none starves the others.
    let page: Vec<RefMeta> = if candidates.needle.trim().is_empty() {
        let turns = lists
            .into_iter()
            .map(|l| l.into_iter().map(|r| r.meta).collect::<VecDeque<_>>())
            .collect();
        interleave(turns, query.limit)
    } else {
        let mut all: Vec<Ranked> = lists.into_iter().flatten().collect();
        sort_ranked(&mut all);
        all.into_iter().take(query.limit).map(|r| r.meta).collect()
    };
    let mut items = Vec::new();
    for mut meta in page {
        resolver_of(meta.kind).finish(&mut meta).await?;
        items.push(item(meta));
    }
    Ok(RefSearchResponse { items })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::PlanNode;
    use crate::neo4j::GraphStore;
    use crate::refs::access::Disclosure;
    use crate::refs::test_support::{task_titled, user, world, Spy, World};
    use crate::refs::wire::ScopeLabel;

    fn params(q: &str) -> RefSearchParams {
        RefSearchParams {
            q: Some(q.into()),
            ..Default::default()
        }
    }

    fn parsed(p: RefSearchParams) -> SearchQuery {
        p.into_query().unwrap()
    }

    async fn run(w: &World, q: SearchQuery) -> Vec<RefSearchItem> {
        search(w.graph.clone(), &AccessPolicy::open_instance(), &user(), &q)
            .await
            .unwrap()
            .items
    }

    fn ids(items: &[RefSearchItem]) -> Vec<Uuid> {
        items.iter().map(|i| i.id.uuid().unwrap()).collect()
    }

    // ---- relevance ------------------------------------------------------

    async fn add_note(w: &World, content: &str) -> Uuid {
        let n = crate::refs::test_support::note_with(
            Some(w.a),
            crate::notes::NoteType::Tip,
            content,
            vec![],
        );
        w.graph.create_note(&n).await.unwrap();
        // distinct creation instants: recency is the tie-break under test.
        tokio::time::sleep(Duration::from_millis(5)).await;
        n.id
    }

    fn notes_for(q: &str) -> SearchQuery {
        parsed(RefSearchParams {
            q: Some(q.into()),
            kinds: Some("note".into()),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn fond_ranks_fondations_before_a_text_with_fond_in_the_middle() {
        let w = world().await;
        let prefix = add_note(&w, "# Fondations refs\ncorps").await;
        let word = add_note(&w, "Sur le fond de l'affaire").await;
        let body = add_note(&w, "Autre sujet\nplus loin, on parle du fond du texte").await;
        // the most recent first would be [body, word, prefix]: relevance reverses it.
        let got = run(&w, notes_for("fond")).await;
        assert_eq!(ids(&got), [prefix, word, body]);
    }

    #[tokio::test]
    async fn accents_and_case_do_not_hide_a_match() {
        let w = world().await;
        let id = add_note(&w, "Été chaud").await;
        assert_eq!(ids(&run(&w, notes_for("ETE")).await), [id]);
        assert_eq!(ids(&run(&w, notes_for("été")).await), [id]);
    }

    #[tokio::test]
    async fn equal_scores_go_to_the_most_recent() {
        let w = world().await;
        let older = add_note(&w, "Alpha un").await;
        let newer = add_note(&w, "Alpha deux").await;
        assert_eq!(ids(&run(&w, notes_for("alpha")).await), [newer, older]);
    }

    #[tokio::test]
    async fn the_order_is_decided_before_the_cut_to_limit() {
        let w = world().await;
        let best = add_note(&w, "Fond").await;
        for i in 0..4 {
            add_note(&w, &format!("Autre sujet {i}\non parle du fond ici")).await;
        }
        let mut q = notes_for("fond");
        q.limit = 1;
        assert_eq!(ids(&run(&w, q).await), [best], "the best, not the newest");
    }
    // ---- parsing -------------------------------------------------------

    #[test]
    fn defaults_are_the_five_historical_kinds_and_a_page_of_twenty() {
        let q = parsed(RefSearchParams::default());
        assert_eq!(q.kinds, RefKind::HISTORICAL.to_vec());
        assert_eq!(q.limit, 20);
        assert_eq!(q.q, "");
        assert!(q.project_id.is_none() && q.workspace_slug.is_none());
        let blank = parsed(RefSearchParams {
            kinds: Some("  ".into()),
            limit: Some("".into()),
            project_id: Some(" ".into()),
            workspace_slug: Some("  ".into()),
            ..Default::default()
        });
        assert_eq!(blank, q);
    }

    #[test]
    fn kinds_keep_request_order_and_lose_duplicates() {
        let q = parsed(RefSearchParams {
            kinds: Some("rfc, plan,rfc".into()),
            ..Default::default()
        });
        assert_eq!(q.kinds, vec![RefKind::Rfc, RefKind::Plan]);
    }

    #[test]
    fn a_bad_kind_names_its_reason_and_its_index() {
        let e = |k: &str| {
            RefSearchParams {
                kinds: Some(k.into()),
                ..Default::default()
            }
            .into_query()
            .unwrap_err()
        };
        assert_eq!(
            e("plan,step"),
            RefsInvalid {
                reason: InvalidReason::UnknownKind,
                index: Some(1)
            }
        );
        assert_eq!(e("Plan").reason, InvalidReason::UnknownKind);
        assert_eq!(e("plan,,task").reason, InvalidReason::UnknownKind);
    }

    #[test]
    fn query_and_limit_are_validated() {
        let long = params(&"x".repeat(201)).into_query().unwrap_err();
        assert_eq!(long.reason, InvalidReason::QueryTooLong);
        assert!(params(&"x".repeat(200)).into_query().is_ok());
        for bad in ["0", "51", "abc", "-1", "2.5"] {
            let e = RefSearchParams {
                limit: Some(bad.into()),
                ..Default::default()
            }
            .into_query()
            .unwrap_err();
            assert_eq!(e.reason, InvalidReason::BadLimit, "{bad}");
            assert_eq!(e.index, None);
        }
        let ok = RefSearchParams {
            limit: Some(" 50 ".into()),
            ..Default::default()
        };
        assert_eq!(parsed(ok).limit, 50);
    }

    #[test]
    fn scope_filters_are_parsed() {
        let id = Uuid::new_v4();
        let q = parsed(RefSearchParams {
            q: Some("  Refs ".into()),
            project_id: Some(id.to_string()),
            workspace_slug: Some(" po ".into()),
            ..Default::default()
        });
        assert_eq!(q.q, "Refs");
        assert_eq!(q.project_id, Some(id));
        assert_eq!(q.workspace_slug.as_deref(), Some("po"));
        for bad in ["nope", "00000000-0000-0000-0000-000000000000"] {
            let e = RefSearchParams {
                project_id: Some(bad.into()),
                ..Default::default()
            }
            .into_query()
            .unwrap_err();
            assert_eq!(e.reason, InvalidReason::BadId, "{bad}");
        }
    }

    // ---- scope ---------------------------------------------------------

    fn meta(project: Option<Uuid>, ws: Option<&str>) -> RefMeta {
        let label = |id: Uuid, slug: &str| ScopeLabel {
            id,
            slug: slug.into(),
            name: slug.into(),
        };
        RefMeta {
            kind: RefKind::Plan,
            id: Uuid::new_v4().into(),
            label: "l".into(),
            subtitle: None,
            project: project.map(|p| label(p, "p")),
            workspace: ws.map(|s| label(Uuid::new_v4(), s)),
            entity_status: None,
        }
    }

    #[test]
    fn in_scope_requires_the_named_project_and_the_named_workspace() {
        let p = Uuid::new_v4();
        assert!(in_scope(&meta(None, None), None, None));
        assert!(in_scope(&meta(Some(p), Some("po")), Some(p), Some("po")));
        assert!(in_scope(&meta(Some(p), None), Some(p), None));
        assert!(!in_scope(
            &meta(Some(p), Some("po")),
            Some(Uuid::new_v4()),
            None
        ));
        assert!(!in_scope(&meta(None, None), Some(p), None), "no project");
        assert!(!in_scope(&meta(Some(p), Some("po")), None, Some("other")));
        assert!(
            !in_scope(&meta(Some(p), None), None, Some("po")),
            "no workspace"
        );
        assert!(!in_scope(
            &meta(Some(p), Some("po")),
            Some(p),
            Some("other")
        ));
    }

    #[tokio::test]
    async fn the_workspace_filter_holds_even_when_the_store_ignores_it() {
        // The mock ignores `workspace_slug` for plans and tasks: only our own
        // filter keeps beta (in no workspace) out.
        let w = world().await;
        let q = SearchQuery {
            kinds: vec![RefKind::Plan, RefKind::Task],
            workspace_slug: Some("po".into()),
            ..parsed(RefSearchParams::default())
        };
        let got = ids(&run(&w, q).await);
        assert!(got.contains(&w.plan_a.id) && got.contains(&w.task_a.id));
        assert!(!got.contains(&w.plan_b.id) && !got.contains(&w.task_b.id));
        assert_eq!(got.len(), 3, "plan a and its two tasks");
    }

    #[tokio::test]
    async fn the_project_filter_keeps_other_projects_and_global_notes_out() {
        let w = world().await;
        let q = SearchQuery {
            project_id: Some(w.a),
            ..parsed(RefSearchParams::default())
        };
        let got = ids(&run(&w, q).await);
        for want in [w.plan_a.id, w.task_a.id, w.note_a.id, w.decision_a.id] {
            assert!(got.contains(&want));
        }
        for out in [
            w.plan_b.id,
            w.task_b.id,
            w.decision_b.id,
            w.note_global.id,
            w.rfc.id,
        ] {
            assert!(!got.contains(&out));
        }
    }

    #[tokio::test]
    async fn the_scope_is_pushed_to_the_store_so_a_small_page_is_not_starved() {
        // A hundred plans of beta, one of alpha: with a page of one the store must
        // be asked for alpha, not for whatever it lists first.
        let w = world().await;
        for i in 0..100 {
            let p = PlanNode::new_for_project(format!("beta {i}"), "d".into(), "t".into(), 1, w.b);
            w.graph.create_plan(&p).await.unwrap();
        }
        let q = SearchQuery {
            kinds: vec![RefKind::Plan],
            project_id: Some(w.a),
            limit: 1,
            ..parsed(RefSearchParams::default())
        };
        assert_eq!(ids(&run(&w, q).await), vec![w.plan_a.id]);
    }

    // ---- text and kinds ------------------------------------------------

    #[tokio::test]
    async fn text_finds_each_kind_by_its_own_words() {
        let w = world().await;
        let one = |kind: RefKind, q: &str| SearchQuery {
            kinds: vec![kind],
            q: q.into(),
            ..parsed(RefSearchParams::default())
        };
        assert_eq!(
            ids(&run(&w, one(RefKind::Plan, "alpha")).await),
            vec![w.plan_a.id]
        );
        assert_eq!(
            ids(&run(&w, one(RefKind::Plan, "FACTURATION")).await),
            vec![w.plan_b.id]
        );
        assert_eq!(
            ids(&run(&w, one(RefKind::Task, "beta")).await),
            vec![w.task_b.id]
        );
        assert_eq!(
            ids(&run(&w, one(RefKind::Note, "piège")).await),
            vec![w.note_a.id]
        );
        assert_eq!(
            ids(&run(&w, one(RefKind::Decision, "typées")).await),
            vec![w.decision_a.id]
        );
        assert_eq!(
            ids(&run(&w, one(RefKind::Rfc, "refs")).await),
            vec![w.rfc.id]
        );
        assert!(run(&w, one(RefKind::Rfc, "piège")).await.is_empty());
        assert!(run(&w, one(RefKind::Note, "refs"))
            .await
            .iter()
            .all(|i| i.kind == RefKind::Note));
    }

    #[tokio::test]
    async fn notes_and_rfcs_never_come_back_under_each_others_kind() {
        let w = world().await;
        let notes = run(
            &w,
            SearchQuery {
                kinds: vec![RefKind::Note],
                ..parsed(RefSearchParams::default())
            },
        )
        .await;
        assert_eq!(notes.len(), 2);
        assert!(!ids(&notes).contains(&w.rfc.id));
        let rfcs = run(
            &w,
            SearchQuery {
                kinds: vec![RefKind::Rfc],
                ..parsed(RefSearchParams::default())
            },
        )
        .await;
        assert_eq!(ids(&rfcs), vec![w.rfc.id]);
    }

    #[tokio::test]
    async fn every_note_type_but_rfc_is_a_note() {
        use crate::notes::NoteType::*;
        let w = world().await;
        for t in [
            Guideline,
            Gotcha,
            Pattern,
            Context,
            Tip,
            Observation,
            Assertion,
        ] {
            let n = crate::refs::test_support::note_with(None, t, &format!("typed {t}"), vec![]);
            w.graph.create_note(&n).await.unwrap();
            let q = SearchQuery {
                kinds: vec![RefKind::Note],
                q: format!("typed {t}"),
                ..parsed(RefSearchParams::default())
            };
            assert_eq!(ids(&run(&w, q).await), vec![n.id], "{t}");
        }
    }

    #[tokio::test]
    async fn the_page_is_shared_between_kinds_in_turn_and_cut_at_the_limit() {
        let w = world().await;
        let q = SearchQuery {
            kinds: vec![RefKind::Task, RefKind::Plan],
            limit: 3,
            ..parsed(RefSearchParams::default())
        };
        let got = run(&w, q).await;
        assert_eq!(got.len(), 3);
        let kinds: Vec<_> = got.iter().map(|i| i.kind).collect();
        assert_eq!(kinds, vec![RefKind::Task, RefKind::Plan, RefKind::Task]);
        let all = run(&w, parsed(RefSearchParams::default())).await;
        assert_eq!(
            all.len(),
            2 + 3 + 2 + 2 + 1,
            "plans, tasks, notes, decisions, rfc"
        );
    }

    #[tokio::test]
    async fn plans_carry_their_task_count_and_tasks_their_plan() {
        let w = world().await;
        let got = run(
            &w,
            SearchQuery {
                kinds: vec![RefKind::Plan, RefKind::Task],
                q: "alpha".into(),
                ..parsed(RefSearchParams::default())
            },
        )
        .await;
        let plan = got.iter().find(|i| i.kind == RefKind::Plan).unwrap();
        assert_eq!(plan.subtitle.as_deref(), Some("2 tâches"));
        let task = got.iter().find(|i| i.kind == RefKind::Task).unwrap();
        assert_eq!(task.subtitle.as_deref(), Some("Plan : Plan alpha refs"));
    }

    #[tokio::test]
    async fn a_task_search_looks_past_the_first_rows_up_to_the_scan_cap() {
        let w = world().await;
        let extra = task_titled(Some("needle task"), "d");
        w.graph.create_task(w.plan_a.id, &extra).await.unwrap();
        let q = SearchQuery {
            kinds: vec![RefKind::Task],
            q: "needle".into(),
            limit: 1,
            ..parsed(RefSearchParams::default())
        };
        assert_eq!(ids(&run(&w, q).await), vec![extra.id]);
    }

    // ---- policy --------------------------------------------------------

    #[tokio::test]
    async fn what_the_policy_denies_is_not_suggested_and_the_policy_spoke() {
        let w = world().await;
        let spy = Spy::new(false);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let q = parsed(RefSearchParams::default());
        let out = search(w.graph.clone(), &policy, &user(), &q).await.unwrap();
        assert!(out.items.is_empty());
        assert_eq!(spy.asked(), 10, "one verdict per candidate");

        let spy = Spy::new(true);
        let policy = AccessPolicy::new(spy.clone(), Disclosure::Uniform);
        let out = search(w.graph.clone(), &policy, &user(), &q).await.unwrap();
        assert_eq!(out.items.len(), 10);
        assert_eq!(spy.asked(), 10);
    }

    #[tokio::test]
    async fn an_unauthenticated_caller_gets_nothing() {
        let w = world().await;
        let out = search(
            w.graph.clone(),
            &AccessPolicy::open_instance(),
            &Principal::Unauthenticated,
            &parsed(RefSearchParams::default()),
        )
        .await
        .unwrap();
        assert!(out.items.is_empty());
    }

    #[tokio::test]
    async fn a_store_failure_costs_the_page_one_kind_but_never_all_of_it_silently() {
        let w = world().await;
        w.graph
            .fail_reads
            .lock()
            .unwrap()
            .insert("list_plans_filtered");
        let run_kinds = |kinds: Vec<RefKind>| {
            let graph = w.graph.clone();
            async move {
                search(
                    graph,
                    &AccessPolicy::open_instance(),
                    &user(),
                    &SearchQuery {
                        kinds,
                        ..parsed(RefSearchParams::default())
                    },
                )
                .await
            }
        };
        let partial = run_kinds(RefKind::ALL.to_vec()).await.unwrap();
        assert!(!partial.items.is_empty());
        assert!(partial.items.iter().all(|i| i.kind != RefKind::Plan));
        assert!(run_kinds(vec![RefKind::Plan]).await.is_err());
    }

    #[tokio::test]
    async fn the_response_is_exactly_the_items_the_fixture_shape_allows() {
        let w = world().await;
        let out = search(
            w.graph.clone(),
            &AccessPolicy::open_instance(),
            &user(),
            &SearchQuery {
                kinds: vec![RefKind::Plan],
                ..parsed(RefSearchParams::default())
            },
        )
        .await
        .unwrap();
        let v = serde_json::to_value(&out).unwrap();
        assert_eq!(v.as_object().unwrap().keys().collect::<Vec<_>>(), ["items"]);
    }
}

#[cfg(test)]
mod fan_out_tests {
    use super::*;
    use crate::refs::resolvers::Candidates;
    use crate::refs::test_support::{user, world};
    use crate::refs::types::RefId;

    /// A kind that never answers.
    struct Hangs;
    /// A kind whose store is down.
    struct Down;

    #[async_trait::async_trait]
    impl KindResolver for Hangs {
        fn kind(&self) -> RefKind {
            RefKind::Task
        }
        async fn load(&self, _: &RefId, _: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
            std::future::pending().await
        }
        async fn candidates(
            &self,
            _: &Candidates,
            _: &mut Memo,
        ) -> anyhow::Result<Vec<crate::refs::resolvers::Candidate>> {
            std::future::pending().await
        }
    }

    #[async_trait::async_trait]
    impl KindResolver for Down {
        fn kind(&self) -> RefKind {
            RefKind::Task
        }
        async fn load(&self, _: &RefId, _: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
            anyhow::bail!("down")
        }
        async fn candidates(
            &self,
            _: &Candidates,
            _: &mut Memo,
        ) -> anyhow::Result<Vec<crate::refs::resolvers::Candidate>> {
            anyhow::bail!("down")
        }
    }

    fn query(kinds: Vec<RefKind>) -> SearchQuery {
        SearchQuery {
            kinds,
            ..RefSearchParams::default().into_query().unwrap()
        }
    }

    #[tokio::test]
    async fn a_kind_that_times_out_is_left_out_and_the_others_still_answer() {
        let w = world().await;
        let source = GraphRefSource::new(w.graph.clone());
        let hangs = Hangs;
        let started = std::time::Instant::now();
        let out = search_with(
            &|k| {
                if k == RefKind::Task {
                    &hangs
                } else {
                    source.resolver(k)
                }
            },
            &AccessPolicy::open_instance(),
            &user(),
            &query(vec![RefKind::Task, RefKind::Plan, RefKind::Rfc]),
            Duration::from_millis(60),
        )
        .await
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(5));
        let kinds: Vec<_> = out.items.iter().map(|i| i.kind).collect();
        assert!(kinds.contains(&RefKind::Plan) && kinds.contains(&RefKind::Rfc));
        assert!(!kinds.contains(&RefKind::Task));
    }

    #[tokio::test]
    async fn a_kind_in_error_is_left_out_and_the_others_still_answer() {
        let w = world().await;
        let source = GraphRefSource::new(w.graph.clone());
        let down = Down;
        let out = search_with(
            &|k| {
                if k == RefKind::Task {
                    &down
                } else {
                    source.resolver(k)
                }
            },
            &AccessPolicy::open_instance(),
            &user(),
            &query(vec![RefKind::Task, RefKind::Plan]),
            KIND_TIMEOUT,
        )
        .await
        .unwrap();
        assert_eq!(out.items.len(), 2);
        assert!(out.items.iter().all(|i| i.kind == RefKind::Plan));
    }

    #[tokio::test]
    async fn when_every_kind_fails_the_search_fails_instead_of_looking_empty() {
        let down = Down;
        let hangs = Hangs;
        for (resolver, timeout) in [
            (&down as &dyn KindResolver, KIND_TIMEOUT),
            (&hangs as &dyn KindResolver, Duration::from_millis(30)),
        ] {
            let r = search_with(
                &|_| resolver,
                &AccessPolicy::open_instance(),
                &user(),
                &query(vec![RefKind::Task]),
                timeout,
            )
            .await;
            assert!(r.is_err());
        }
    }

    #[tokio::test]
    async fn a_kind_that_is_down_does_not_mask_a_page_that_is_legitimately_empty() {
        let w = world().await;
        let source = GraphRefSource::new(w.graph.clone());
        let mut q = query(vec![RefKind::Plan]);
        q.q = "nothing matches this".into();
        let out = search_with(
            &|k| source.resolver(k),
            &AccessPolicy::open_instance(),
            &user(),
            &q,
            KIND_TIMEOUT,
        )
        .await
        .unwrap();
        assert!(out.items.is_empty());
    }
}

#[cfg(test)]
mod pushdown_tests {
    use super::*;
    use crate::refs::test_support::{note_with, user, world};

    /// The mock honours `workspace_slug` for notes (not for plans or tasks),
    /// so the push-down of the workspace is observable on notes.
    #[tokio::test]
    async fn the_workspace_is_pushed_to_the_store_for_notes() {
        let w = world().await;
        for i in 0..100 {
            let n = note_with(
                Some(w.b),
                crate::notes::NoteType::Tip,
                &format!("beta {i}"),
                vec![],
            );
            w.graph.create_note(&n).await.unwrap();
        }
        let q = SearchQuery {
            kinds: vec![RefKind::Note],
            workspace_slug: Some("po".into()),
            limit: 1,
            ..RefSearchParams::default().into_query().unwrap()
        };
        let out = search(w.graph.clone(), &AccessPolicy::open_instance(), &user(), &q)
            .await
            .unwrap();
        assert_eq!(out.items.len(), 1);
        assert_eq!(out.items[0].id, w.note_a.id);
    }
}
