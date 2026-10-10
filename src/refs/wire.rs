//! The bodies the frontend reads: search results, the `refs_resolved` event
//! and the error body. Their JSON is pinned by `tests/fixtures/refs/*.json`.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::kinds::{KindClass, LabelSource, KINDS};
use super::registry;
use super::types::{canonical_id, IdFormat, MapOnly, RefId, RefKind};
use super::validate::{InvalidReason, RefsInvalid};

/// The outcome of resolving one reference, as the user sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefStatus {
    Ok,
    NotFound,
    /// Only emitted when the policy discloses existence (see `access::Disclosure`).
    Forbidden,
    /// Resolved, but its content was cut to fit the prompt budget.
    Truncated,
}

/// A project or a workspace, for display next to a result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeLabel {
    pub id: Uuid,
    pub slug: String,
    pub name: String,
}

/// One row of `GET /api/refs/search`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefSearchItem {
    pub kind: RefKind,
    pub id: RefId,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<ScopeLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<ScopeLabel>,
    /// The entity's own status (`in_progress`, `active`, `accepted`...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_status: Option<String>,
}

/// The rows as objects only: a positional row is refused.
fn de_items<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<RefSearchItem>, D::Error> {
    let rows = Vec::<MapOnly<RefSearchItem>>::deserialize(d)?;
    let mut items = Vec::with_capacity(rows.len());
    for MapOnly(r) in rows {
        // the id must be valid, and spelled canonically, for its kind.
        if canonical_id(r.kind, r.id.as_str()).as_ref() != Some(&r.id) {
            return Err(serde::de::Error::custom("not a valid reference id"));
        }
        items.push(r);
    }
    Ok(items)
}

/// Body of `GET /api/refs/search`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefSearchResponse {
    #[serde(deserialize_with = "de_items")]
    pub items: Vec<RefSearchItem>,
}

/// One reference in a `refs_resolved` event. `label`, `subtitle`, scope and
/// `entity_status` are present only when `status` is `ok` or `truncated`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefResolution {
    pub kind: RefKind,
    pub id: RefId,
    pub status: RefStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<ScopeLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<ScopeLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_status: Option<String>,
}

/// The chat event sent once per turn, before the model answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefsEvent {
    RefsResolved { refs: Vec<RefResolution> },
}

/// One kind the server resolves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KindInfo {
    pub kind: RefKind,
    /// `data` (cited with `#`) or `actor` (invocable, cited with `@`).
    pub class: KindClass,
    /// The character that starts its token in the composer: `#` or `@`.
    pub sigil: String,
    /// How its `id` is spelled.
    pub id_format: IdFormat,
    /// Sensitive kinds (file, commit) are suggested only inside a project or
    /// workspace (`project_id` / `workspace_slug` on the search) and are
    /// readable by a session only inside a project it is attached to.
    pub sensitive: bool,
    /// What the label is made of.
    pub label_source: LabelSource,
    /// Offered by `GET /api/refs/search`.
    pub searchable: bool,
    /// Where the app opens it, relative to `/workspace/:slug/` (`{id}` is the
    /// reference id, `{slug}` the slug of the result's `project`); `null`: no page.
    pub open_route: Option<String>,
    /// Opened outside the app.
    pub opens_external: bool,
    /// A key the frontend maps to an icon.
    pub icon: String,
}

/// Body of `GET /api/refs/kinds`: the kinds this server resolves, so that a
/// client offers only those. A server older than this route answers 404: the
/// client then offers the five historical kinds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KindsResponse {
    pub contract_version: u32,
    pub kinds: Vec<KindInfo>,
}

impl KindsResponse {
    /// The enabled kinds of the registry, in registry order.
    pub fn active() -> Self {
        Self {
            contract_version: super::CONTRACT_VERSION,
            kinds: KINDS
                .iter()
                .filter(|d| registry::spec(d.kind).enabled)
                .map(|d| KindInfo {
                    kind: d.kind,
                    class: d.class,
                    sigil: d.class.sigil().to_string(),
                    id_format: d.id_format,
                    sensitive: d.kind.is_sensitive(),
                    label_source: d.label,
                    searchable: d.searchable,
                    open_route: d.open_route.map(str::to_string),
                    opens_external: d.opens_external,
                    icon: d.icon.to_string(),
                })
                .collect(),
        }
    }
}

/// Error body for a refused request (HTTP 400, or a WS error frame).
/// `error` is the human line every other API error already carries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefsErrorBody {
    pub error: String,
    pub code: String,
    pub reason: InvalidReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

pub const REFS_INVALID_CODE: &str = "refs_invalid";

impl From<RefsInvalid> for RefsErrorBody {
    fn from(e: RefsInvalid) -> Self {
        Self::from_reason(e.reason, e.index)
    }
}

impl RefsErrorBody {
    pub fn from_reason(reason: InvalidReason, index: Option<usize>) -> Self {
        Self {
            error: reason.message().to_string(),
            code: REFS_INVALID_CODE.to_string(),
            reason,
            index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_body_carries_the_stable_code_and_the_index() {
        let body: RefsErrorBody = RefsInvalid {
            reason: InvalidReason::UnknownKind,
            index: Some(2),
        }
        .into();
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["code"], "refs_invalid");
        assert_eq!(v["reason"], "unknown_kind");
        assert_eq!(v["index"], 2);
        let no_index =
            serde_json::to_value(RefsErrorBody::from_reason(InvalidReason::TooMany, None)).unwrap();
        assert!(no_index.get("index").is_none());
    }

    #[test]
    fn the_event_is_tagged_refs_resolved() {
        let ev = RefsEvent::RefsResolved { refs: vec![] };
        assert_eq!(
            serde_json::to_value(&ev).unwrap(),
            serde_json::json!({"type": "refs_resolved", "refs": []})
        );
    }

    #[test]
    fn statuses_use_snake_case() {
        let all = [
            (RefStatus::Ok, "ok"),
            (RefStatus::NotFound, "not_found"),
            (RefStatus::Forbidden, "forbidden"),
            (RefStatus::Truncated, "truncated"),
        ];
        for (s, name) in all {
            assert_eq!(serde_json::to_value(s).unwrap(), name);
        }
    }

    #[test]
    fn a_search_item_is_an_object_with_a_real_hyphenated_id() {
        let id = "3adeffc9-c8b0-4e2f-a674-55bfcb293433";
        let ok = format!(r#"{{"items":[{{"kind":"plan","id":"{id}","label":"L"}}]}}"#);
        assert_eq!(
            serde_json::from_str::<RefSearchResponse>(&ok)
                .unwrap()
                .items
                .len(),
            1
        );
        for bad in [
            r#"{"items":[["plan","3adeffc9-c8b0-4e2f-a674-55bfcb293433","L"]]}"#.to_string(),
            r#"{"items":[{"kind":"plan","id":"00000000-0000-0000-0000-000000000000","label":"L"}]}"#.to_string(),
            format!(r#"{{"items":[{{"kind":"plan","id":"{}","label":"L"}}]}}"#, id.replace('-', "")),
            format!(r#"{{"items":[{{"kind":"plan","id":"{{{id}}}","label":"L"}}]}}"#),
        ] {
            assert!(serde_json::from_str::<RefSearchResponse>(&bad).is_err(), "{bad}");
        }
    }
}
