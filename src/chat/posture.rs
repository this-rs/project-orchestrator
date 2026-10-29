//! Posture of a conversation, derived from its focus anchors — SHADOW mode (T7).
//!
//! A conversation anchored on a plan is better framed as planning work, one anchored
//! on a file as code reading. The **posture** is that framing: a persona, a tone, a
//! level of detail, the read-only tools and the skills worth bringing forward, and a
//! scaffolding (checklist, step plan). It only frames and costs: it never grants
//! anything.
//!
//! # Invariants versus posture
//!
//! What a session may do lives in the tool policy (T0e: [`SessionAccess`], the
//! permission mode), never in the posture. [`Invariants`] is that part, rendered as
//! a segment no posture can touch: [`compose`] always writes it first, byte for byte
//! the same whatever the posture (a property test holds it). A posture's tools are
//! read-only by construction and are clipped again by [`Posture::within`]: a posture
//! can narrow what is put forward, never widen what is allowed.
//!
//! # Derivation
//!
//! Only the **live focus** anchors count ([`derive`]); mentions, origins and broken
//! targets do not. When the resolver ran, only the anchors it admitted count (a
//! refused anchor never shapes the framing). Several focus anchors that disagree:
//! the tools are INTERSECTED, a field that differs falls back to its neutral value,
//! the skills are united. The result does not depend on the anchors' order.
//!
//! # Fit, hysteresis, override
//!
//! - [`Posture::fit`] adapts a posture to what the session's model can do
//!   (`nexus_claude::agent::Capabilities`); every change is a [`Degradation`],
//!   never a silent drop. Unknown capabilities are a degradation too.
//! - [`PostureTracker`] keeps the posture stable: a derived change is adopted after
//!   [`STABLE_TURNS`] identical turns, at once when the user changed an explicit
//!   focus anchor, and every switch comes with a one-line announcement.
//! - [`PostureOverride`] `Auto | Pin | Off`: the user's word wins; every change of
//!   override is journaled.
//!
//! # Shadow mode
//!
//! Nothing here reaches a prompt yet. Each turn, the manager derives, fits and
//! tracks the posture of the session and journals ONE JSON line (target
//! [`JOURNAL_TARGET`], [`PostureObservation`]); the turn's prompt, tools and
//! enrichment are unchanged. The book of the trackers is [`book`].
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use crate::chat::anchor::{Anchor, AnchorActor, AnchorRole, AnchorState, AnchorTargetType};
use crate::chat::anchor_resolver::ResolvedScope;
use crate::chat::provider::policy::SessionAccess;
use nexus_claude::agent::Capabilities;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use uuid::Uuid;

/// Tracing target of the shadow journal.
pub const JOURNAL_TARGET: &str = "posture_shadow";
/// Identical derived turns needed before a posture inferred from the anchors is adopted.
pub const STABLE_TURNS: u32 = 3;
/// Below this context window (tokens), a fine level of detail is lowered to normal.
pub const SMALL_WINDOW_TOKENS: u64 = 32_000;
/// Sessions the book keeps (the oldest observed is dropped beyond).
const BOOK_CAPACITY: usize = 1024;

/// Who the assistant is, for this conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Persona {
    Generalist,
    Developer,
    Architect,
    Planner,
    Releaser,
    Archivist,
    ProjectLead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tone {
    Neutral,
    Technical,
    Concise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    Overview,
    Normal,
    Fine,
}

/// The frame the answer is given in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Scaffolding {
    None,
    Checklist,
    StepPlan,
}

/// Prefix of the project-orchestrator mega-tools (read actions only) in a posture;
/// the other tool names are native Claude Code ones (`Read`, `Grep`, `Glob`).
const PO_PREFIX: &str = "po:";

/// The framing of a conversation. Never a permission.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Posture {
    pub persona: Persona,
    pub tone: Tone,
    pub detail: Detail,
    /// Read-only tools put forward (a hint, not an allow list).
    pub tools: BTreeSet<String>,
    /// Skills worth suggesting.
    pub skills: BTreeSet<String>,
    pub scaffolding: Scaffolding,
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

impl Posture {
    /// The posture of a conversation with no live focus anchor.
    pub fn neutral() -> Self {
        Self {
            persona: Persona::Generalist,
            tone: Tone::Neutral,
            detail: Detail::Normal,
            tools: BTreeSet::new(),
            skills: BTreeSet::new(),
            scaffolding: Scaffolding::None,
        }
    }

    /// The posture one focus anchor of this type calls for (the table of the 11 types).
    pub fn for_target(t: AnchorTargetType) -> Self {
        use AnchorTargetType as T;
        let (persona, tone, detail, tools, skills, scaffolding): (_, _, _, &[&str], &[&str], _) =
            match t {
                T::File | T::Function => (
                    Persona::Developer,
                    Tone::Technical,
                    Detail::Fine,
                    &["Read", "Grep", "Glob", "po:code"],
                    &["impact"],
                    Scaffolding::None,
                ),
                T::Component => (
                    Persona::Architect,
                    Tone::Technical,
                    Detail::Normal,
                    &["Read", "Grep", "po:code", "po:component"],
                    &["impact", "diagram-drift"],
                    Scaffolding::None,
                ),
                T::Feature => (
                    Persona::Architect,
                    Tone::Technical,
                    Detail::Normal,
                    &["Read", "Grep", "po:code", "po:feature_graph"],
                    &["impact"],
                    Scaffolding::None,
                ),
                T::Release => (
                    Persona::Releaser,
                    Tone::Concise,
                    Detail::Overview,
                    &["po:release", "po:milestone", "po:plan", "po:task"],
                    &[],
                    Scaffolding::Checklist,
                ),
                T::Plan => (
                    Persona::Planner,
                    Tone::Concise,
                    Detail::Overview,
                    &["po:plan", "po:task", "po:step", "po:decision"],
                    &["plan-review"],
                    Scaffolding::StepPlan,
                ),
                T::Task => (
                    Persona::Developer,
                    Tone::Technical,
                    Detail::Fine,
                    &[
                        "Read", "Grep", "Glob", "po:task", "po:step", "po:plan", "po:code",
                    ],
                    &["task-run"],
                    Scaffolding::StepPlan,
                ),
                T::Note => (
                    Persona::Archivist,
                    Tone::Neutral,
                    Detail::Normal,
                    &["po:note", "po:decision"],
                    &["capture"],
                    Scaffolding::None,
                ),
                T::Decision => (
                    Persona::Architect,
                    Tone::Neutral,
                    Detail::Normal,
                    &["po:decision", "po:note"],
                    &["rfc"],
                    Scaffolding::None,
                ),
                T::Project => (
                    Persona::ProjectLead,
                    Tone::Neutral,
                    Detail::Overview,
                    &["po:project", "po:plan", "po:task", "po:note", "po:code"],
                    &["warmup"],
                    Scaffolding::None,
                ),
                T::Workspace => (
                    Persona::ProjectLead,
                    Tone::Neutral,
                    Detail::Overview,
                    &["po:workspace", "po:project", "po:plan"],
                    &["warmup"],
                    Scaffolding::None,
                ),
            };
        Self {
            persona,
            tone,
            detail,
            tools: set(tools),
            skills: set(skills),
            scaffolding,
        }
    }

    /// Two postures merged: tools intersected, skills united, any other field that
    /// differs falls back to its neutral value. Commutative and associative.
    fn merge(self, other: Self) -> Self {
        let n = Self::neutral();
        Self {
            persona: if self.persona == other.persona {
                self.persona
            } else {
                n.persona
            },
            tone: if self.tone == other.tone {
                self.tone
            } else {
                n.tone
            },
            detail: if self.detail == other.detail {
                self.detail
            } else {
                n.detail
            },
            tools: self.tools.intersection(&other.tools).cloned().collect(),
            skills: self.skills.union(&other.skills).cloned().collect(),
            scaffolding: if self.scaffolding == other.scaffolding {
                self.scaffolding
            } else {
                n.scaffolding
            },
        }
    }

    /// The posture with every tool the invariants refuse removed: a posture never
    /// puts forward a tool the session may not call.
    pub fn within(mut self, inv: &Invariants) -> Self {
        self.tools.retain(|t| inv.admits(t));
        self
    }

    /// The posture adapted to the model's capabilities, with what had to change.
    /// `None`: the capabilities are unknown, which is said, never assumed.
    pub fn fit(&self, caps: Option<&Capabilities>) -> Fitted {
        let mut posture = self.clone();
        let mut degradations = Vec::new();
        let Some(caps) = caps else {
            degradations.push(Degradation::CapabilitiesUnknown);
            return Fitted {
                posture,
                degradations,
            };
        };
        if !caps.tools && !posture.tools.is_empty() {
            degradations.push(Degradation::NoTools {
                dropped: std::mem::take(&mut posture.tools).into_iter().collect(),
            });
        }
        if !caps.per_session_mcp {
            let dropped: Vec<String> = posture
                .tools
                .iter()
                .filter(|t| t.starts_with(PO_PREFIX))
                .cloned()
                .collect();
            if !dropped.is_empty() {
                posture.tools.retain(|t| !t.starts_with(PO_PREFIX));
                degradations.push(Degradation::NoProjectTools { dropped });
            }
        }
        if (!caps.tools || !caps.per_session_mcp) && !posture.skills.is_empty() {
            degradations.push(Degradation::NoSkills {
                dropped: std::mem::take(&mut posture.skills).into_iter().collect(),
            });
        }
        if posture.detail == Detail::Fine {
            if let Some(w) = caps.context_window.as_ref().map(|w| w.value) {
                if w < SMALL_WINDOW_TOKENS {
                    posture.detail = Detail::Normal;
                    degradations.push(Degradation::DetailLowered { window: w });
                }
            }
        }
        Fitted {
            posture,
            degradations,
        }
    }

    /// The posture segment of a prompt. Deterministic.
    pub fn render(&self) -> String {
        let join = |s: &BTreeSet<String>| {
            if s.is_empty() {
                "-".to_string()
            } else {
                s.iter().cloned().collect::<Vec<_>>().join(", ")
            }
        };
        format!(
            "## Posture\npersona: {}\ntone: {}\ndetail: {}\nscaffolding: {}\ntools: {}\nskills: {}\n",
            json_word(&self.persona),
            json_word(&self.tone),
            json_word(&self.detail),
            json_word(&self.scaffolding),
            join(&self.tools),
            join(&self.skills),
        )
    }

    /// The one-line announcement of a switch to this posture.
    pub fn announce(&self, cause: SwitchCause) -> String {
        format!(
            "Posture : {} · {} · {} ({})",
            json_word(&self.persona),
            json_word(&self.detail),
            json_word(&self.scaffolding),
            match cause {
                SwitchCause::ExplicitAnchor => "ancre explicite",
                SwitchCause::Stable => "ancres stables",
                SwitchCause::Override => "choix de l'utilisateur",
            }
        )
    }
}

fn json_word<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// What [`Posture::fit`] had to change. Each is journaled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Degradation {
    /// The session carries no capability snapshot: nothing was adapted.
    CapabilitiesUnknown,
    /// The model cannot call tools.
    NoTools { dropped: Vec<String> },
    /// The project-orchestrator MCP server is not attached to the session.
    NoProjectTools { dropped: Vec<String> },
    /// No skill can be loaded without tools and the MCP server.
    NoSkills { dropped: Vec<String> },
    /// The context window is too small for a fine level of detail.
    DetailLowered { window: u64 },
}

/// A posture adapted to a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Fitted {
    pub posture: Posture,
    pub degradations: Vec<Degradation>,
}

/// What no posture can change: the tool policy of the session (T0e).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invariants {
    pub access: SessionAccess,
    /// The session's permission mode as stored (`None`: the default).
    pub permission_mode: Option<String>,
}

impl Invariants {
    /// Whether the session may call `tool` (a posture tool name: native, or `po:<name>`).
    pub fn admits(&self, tool: &str) -> bool {
        let wire = match tool.strip_prefix(PO_PREFIX) {
            Some(name) => format!("mcp__project-orchestrator__{name}"),
            None => tool.to_string(),
        };
        !self.access.denied_tools().contains(&wire)
    }

    /// The invariant segment of a prompt. Depends on the invariants ONLY.
    pub fn render(&self) -> String {
        format!(
            "## Invariants\naccess: {}\npermission_mode: {}\nLes outils et permissions de cette session sont fixés par le harnais ; la posture ne les change pas.\n",
            self.access.as_str(),
            self.permission_mode.as_deref().unwrap_or("default"),
        )
    }
}

/// The two segments, invariants first. With `None` (override `Off`), the invariant
/// segment alone. The invariant segment is a byte-identical prefix whatever the posture.
pub fn compose(inv: &Invariants, posture: Option<&Posture>) -> String {
    match posture {
        Some(p) => format!("{}\n{}", inv.render(), p.render()),
        None => inv.render(),
    }
}

/// The focus anchors that shape the posture: `focus` role, `live`, and admitted by
/// the resolver when it ran (`admitted`).
fn shaping<'a>(
    anchors: &'a [Anchor],
    admitted: Option<&'a ResolvedScope>,
) -> impl Iterator<Item = &'a Anchor> + 'a {
    anchors.iter().filter(move |a| {
        a.has_role(AnchorRole::Focus)
            && a.state == AnchorState::Live
            && admitted.is_none_or(|s| s.anchors.iter().any(|x| x.anchor_id == a.id))
    })
}

/// The posture the anchors call for.
pub fn derive(anchors: &[Anchor], admitted: Option<&ResolvedScope>) -> Posture {
    shaping(anchors, admitted)
        .map(|a| Posture::for_target(a.target_type))
        .reduce(Posture::merge)
        .unwrap_or_else(Posture::neutral)
}

/// The explicit focus anchors (put by the user, not inferred): a change of this set
/// switches the posture at once.
pub fn explicit_focus(anchors: &[Anchor], admitted: Option<&ResolvedScope>) -> BTreeSet<Uuid> {
    shaping(anchors, admitted)
        .filter(|a| a.by == AnchorActor::User && !a.inferred)
        .map(|a| a.id)
        .collect()
}

/// Why the posture switched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SwitchCause {
    ExplicitAnchor,
    Stable,
    Override,
}

/// The user's word on the posture of a session.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
#[serde(tag = "mode", content = "posture", rename_all = "snake_case")]
pub enum PostureOverride {
    /// Derived from the anchors, with hysteresis.
    #[default]
    Auto,
    /// This posture, whatever the anchors.
    Pin(Posture),
    /// No posture at all: the invariant segment alone.
    Off,
}

/// A switch of the effective posture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Switch {
    pub cause: SwitchCause,
    /// The one-line announcement (`None` for a switch to `Off`).
    pub announce: Option<String>,
}

/// The posture of one session across its turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostureTracker {
    current: Posture,
    /// A derived posture different from `current`, and for how many turns in a row.
    candidate: Option<(Posture, u32)>,
    explicit: BTreeSet<Uuid>,
    over: PostureOverride,
    pub turns: u64,
    pub switches: u64,
}

impl Default for PostureTracker {
    fn default() -> Self {
        Self {
            current: Posture::neutral(),
            candidate: None,
            explicit: BTreeSet::new(),
            over: PostureOverride::Auto,
            turns: 0,
            switches: 0,
        }
    }
}

impl PostureTracker {
    /// The posture in force (`None`: override `Off`).
    pub fn effective(&self) -> Option<&Posture> {
        match &self.over {
            PostureOverride::Auto => Some(&self.current),
            PostureOverride::Pin(p) => Some(p),
            PostureOverride::Off => None,
        }
    }

    pub fn override_mode(&self) -> &PostureOverride {
        &self.over
    }

    /// One turn: the derived posture and the explicit focus anchors of this turn.
    /// Returns the switch of the effective posture, if any. Under `Pin` / `Off`, the
    /// derived posture is still tracked (a return to `Auto` resumes from it) but the
    /// effective posture does not move.
    pub fn observe(&mut self, derived: Posture, explicit: BTreeSet<Uuid>) -> Option<Switch> {
        self.turns += 1;
        let explicit_changed = explicit != self.explicit;
        self.explicit = explicit;
        if derived == self.current {
            self.candidate = None;
            return None;
        }
        let cause = if explicit_changed {
            SwitchCause::ExplicitAnchor
        } else {
            let n = match &self.candidate {
                Some((p, n)) if *p == derived => n + 1,
                _ => 1,
            };
            if n < STABLE_TURNS {
                self.candidate = Some((derived, n));
                return None;
            }
            SwitchCause::Stable
        };
        self.candidate = None;
        self.current = derived;
        if self.over != PostureOverride::Auto {
            return None;
        }
        self.switches += 1;
        Some(Switch {
            cause,
            announce: Some(self.current.announce(cause)),
        })
    }

    /// Set the override. Journaled by the caller ([`PostureBook::set_override`]).
    pub fn set_override(&mut self, over: PostureOverride) -> Option<Switch> {
        if over == self.over {
            return None;
        }
        self.over = over;
        self.switches += 1;
        Some(Switch {
            cause: SwitchCause::Override,
            announce: self.effective().map(|p| p.announce(SwitchCause::Override)),
        })
    }
}

/// The journal line of one shadow turn.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PostureObservation {
    pub session_id: String,
    pub turn: u64,
    /// Types of the focus anchors that shaped the posture (sorted).
    pub focus_types: Vec<&'static str>,
    pub derived: Posture,
    pub over: PostureOverride,
    /// The posture in force after fitting (`None`: override `Off`).
    pub effective: Option<Posture>,
    pub degradations: Vec<Degradation>,
    pub switch: Option<Switch>,
    pub switches: u64,
    /// Estimated tokens the posture segment would cost (0 under `Off`).
    pub segment_tokens: usize,
}

/// The trackers of the sessions of this process.
#[derive(Default)]
pub struct PostureBook {
    sessions: std::sync::Mutex<HashMap<Uuid, (u64, PostureTracker)>>,
    clock: std::sync::atomic::AtomicU64,
}

/// The book of this process.
pub fn book() -> &'static PostureBook {
    static BOOK: std::sync::OnceLock<PostureBook> = std::sync::OnceLock::new();
    BOOK.get_or_init(PostureBook::default)
}

impl PostureBook {
    fn with<R>(&self, session: Uuid, f: impl FnOnce(&mut PostureTracker) -> R) -> R {
        let tick = self
            .clock
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut map = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        if map.len() >= BOOK_CAPACITY && !map.contains_key(&session) {
            if let Some(oldest) = map.iter().min_by_key(|(_, (t, _))| *t).map(|(k, _)| *k) {
                map.remove(&oldest);
            }
        }
        let entry = map.entry(session).or_default();
        entry.0 = tick;
        f(&mut entry.1)
    }

    /// One shadow turn of a session: derive, track, fit, journal. Changes nothing
    /// the session does.
    pub fn observe_turn(
        &self,
        session: Uuid,
        anchors: &[Anchor],
        admitted: Option<&ResolvedScope>,
        inv: &Invariants,
        caps: Option<&Capabilities>,
    ) -> PostureObservation {
        let derived = derive(anchors, admitted).within(inv);
        let explicit = explicit_focus(anchors, admitted);
        let mut focus_types: Vec<&'static str> = shaping(anchors, admitted)
            .map(|a| a.target_type.as_str())
            .collect();
        focus_types.sort_unstable();
        let (switch, over, effective, turn, switches) = self.with(session, |t| {
            let switch = t.observe(derived.clone(), explicit);
            (
                switch,
                t.override_mode().clone(),
                t.effective().cloned(),
                t.turns,
                t.switches,
            )
        });
        let (effective, degradations) = match effective {
            Some(p) => {
                let f = p.within(inv).fit(caps);
                (Some(f.posture), f.degradations)
            }
            None => (None, Vec::new()),
        };
        let obs = PostureObservation {
            session_id: session.to_string(),
            turn,
            focus_types,
            derived,
            over,
            segment_tokens: effective
                .as_ref()
                .map_or(0, |p| crate::chat::anchor_resolver::est_tokens(&p.render())),
            effective,
            degradations,
            switch,
            switches,
        };
        journal(&obs);
        obs
    }

    /// Set the override of a session; journaled (one line, `event: override`).
    pub fn set_override(&self, session: Uuid, over: PostureOverride) -> Option<Switch> {
        let switch = self.with(session, |t| t.set_override(over.clone()));
        match serde_json::to_string(&serde_json::json!({
            "event": "override",
            "session_id": session.to_string(),
            "over": over,
            "switch": switch,
        })) {
            Ok(json) => tracing::info!(target: JOURNAL_TARGET, "{json}"),
            Err(e) => {
                tracing::warn!(target: JOURNAL_TARGET, error = %e, "override not serializable")
            }
        }
        switch
    }

    /// The tracker of a session, as it stands (a copy).
    pub fn tracker(&self, session: Uuid) -> Option<PostureTracker> {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&session)
            .map(|(_, t)| t.clone())
    }

    /// Forget a closed session.
    pub fn forget(&self, session: Uuid) {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&session);
    }
}

fn journal(obs: &PostureObservation) {
    match serde_json::to_string(obs) {
        Ok(json) => tracing::info!(target: JOURNAL_TARGET, "{json}"),
        Err(e) => {
            tracing::warn!(target: JOURNAL_TARGET, error = %e, "posture observation not serializable")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::anchor::AnchorState;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use chrono::Utc;
    use nexus_claude::agent::{ContextWindow, ContextWindowSource};

    fn anchor(t: AnchorTargetType, roles: &[AnchorRole], by: AnchorActor) -> Anchor {
        Anchor {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            target_type: t,
            target_id: "x".into(),
            roles: roles.iter().copied().collect(),
            state: AnchorState::Live,
            by,
            inferred: false,
            confidence: 1.0,
            snapshot_name: None,
            snapshot_path: None,
            snapshot_type: None,
            rev: None,
            version: 1,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn focus(t: AnchorTargetType) -> Anchor {
        anchor(t, &[AnchorRole::Focus], AnchorActor::User)
    }

    fn inv(access: SessionAccess) -> Invariants {
        Invariants {
            access,
            permission_mode: Some("default".into()),
        }
    }

    fn full_caps() -> Capabilities {
        let mut c = Capabilities::none();
        c.tools = true;
        c.per_session_mcp = true;
        c.context_window = Some(ContextWindow {
            value: 200_000,
            source: ContextWindowSource::Catalog,
        });
        c
    }

    #[test]
    fn each_of_the_eleven_anchor_types_and_no_anchor_give_their_posture() {
        use AnchorTargetType as T;
        let table = [
            (
                T::File,
                Persona::Developer,
                Detail::Fine,
                Scaffolding::None,
                "Read",
            ),
            (
                T::Function,
                Persona::Developer,
                Detail::Fine,
                Scaffolding::None,
                "Grep",
            ),
            (
                T::Component,
                Persona::Architect,
                Detail::Normal,
                Scaffolding::None,
                "po:component",
            ),
            (
                T::Feature,
                Persona::Architect,
                Detail::Normal,
                Scaffolding::None,
                "po:feature_graph",
            ),
            (
                T::Release,
                Persona::Releaser,
                Detail::Overview,
                Scaffolding::Checklist,
                "po:release",
            ),
            (
                T::Plan,
                Persona::Planner,
                Detail::Overview,
                Scaffolding::StepPlan,
                "po:plan",
            ),
            (
                T::Task,
                Persona::Developer,
                Detail::Fine,
                Scaffolding::StepPlan,
                "po:task",
            ),
            (
                T::Note,
                Persona::Archivist,
                Detail::Normal,
                Scaffolding::None,
                "po:note",
            ),
            (
                T::Decision,
                Persona::Architect,
                Detail::Normal,
                Scaffolding::None,
                "po:decision",
            ),
            (
                T::Project,
                Persona::ProjectLead,
                Detail::Overview,
                Scaffolding::None,
                "po:project",
            ),
            (
                T::Workspace,
                Persona::ProjectLead,
                Detail::Overview,
                Scaffolding::None,
                "po:workspace",
            ),
        ];
        assert_eq!(table.len(), T::ALL.len(), "every anchor type has a row");
        for (t, persona, detail, scaffolding, tool) in table {
            let p = derive(&[focus(t)], None);
            assert_eq!(
                (p.persona, p.detail, p.scaffolding),
                (persona, detail, scaffolding),
                "{t}"
            );
            assert!(p.tools.contains(tool), "{t}: {:?}", p.tools);
        }
        assert_eq!(derive(&[], None), Posture::neutral());
    }

    #[test]
    fn only_live_focus_anchors_admitted_by_the_resolver_shape_the_posture() {
        let mention = anchor(
            AnchorTargetType::Plan,
            &[AnchorRole::Mention],
            AnchorActor::Agent,
        );
        let origin = anchor(
            AnchorTargetType::Plan,
            &[AnchorRole::Origin],
            AnchorActor::System,
        );
        let mut dangling = focus(AnchorTargetType::Plan);
        dangling.state = AnchorState::Dangling;
        assert_eq!(
            derive(&[mention, origin, dangling], None),
            Posture::neutral()
        );

        let plan = focus(AnchorTargetType::Plan);
        let refused = ResolvedScope::default(); // the resolver admitted nothing
        assert_eq!(
            derive(std::slice::from_ref(&plan), Some(&refused)),
            Posture::neutral()
        );
        assert_eq!(derive(&[plan], None).persona, Persona::Planner);
    }

    #[test]
    fn conflicting_focus_anchors_intersect_the_tools_whatever_their_order() {
        let file = focus(AnchorTargetType::File);
        let task = focus(AnchorTargetType::Task);
        let plan = focus(AnchorTargetType::Plan);
        let ft = derive(&[file.clone(), task.clone()], None);
        assert_eq!(ft, derive(&[task.clone(), file.clone()], None));
        assert_eq!(ft.tools, set(&["Glob", "Grep", "Read", "po:code"]));
        assert_eq!(ft.persona, Persona::Developer);
        assert_eq!(ft.scaffolding, Scaffolding::None, "differs: neutral");
        assert_eq!(ft.skills, set(&["impact", "task-run"]));
        let all = derive(&[plan.clone(), file.clone(), task.clone()], None);
        assert_eq!(all, derive(&[task, plan, file], None));
        assert!(all.tools.is_empty(), "nothing common to file and plan");
        assert_eq!(all.persona, Persona::Generalist);
    }

    fn every_posture() -> Vec<Posture> {
        let mut out = vec![Posture::neutral()];
        for a in AnchorTargetType::ALL {
            out.push(Posture::for_target(*a));
            for b in AnchorTargetType::ALL {
                out.push(Posture::for_target(*a).merge(Posture::for_target(*b)));
            }
        }
        out
    }

    #[test]
    fn the_invariant_segment_is_byte_identical_whatever_the_posture() {
        for access in [SessionAccess::Normal, SessionAccess::ReadOnly] {
            let inv = inv(access);
            let segment = inv.render();
            assert_eq!(compose(&inv, None).as_bytes(), segment.as_bytes());
            for p in every_posture() {
                for fitted in [p.fit(Some(&full_caps())).posture, p.fit(None).posture] {
                    let prompt = compose(&inv, Some(&fitted.clone().within(&inv)));
                    assert_eq!(
                        &prompt.as_bytes()[..segment.len()],
                        segment.as_bytes(),
                        "{fitted:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_posture_never_puts_forward_a_tool_a_read_only_session_may_not_call() {
        let ro = inv(SessionAccess::ReadOnly);
        let mut p = Posture::for_target(AnchorTargetType::File);
        p.tools.insert("Edit".into());
        p.tools.insert("po:admin".into());
        let clipped = p.within(&ro);
        assert!(!clipped.tools.contains("Edit") && !clipped.tools.contains("po:admin"));
        assert!(clipped.tools.contains("Read"));
        for p in every_posture() {
            assert_eq!(p.clone().within(&ro), p, "the table holds read tools only");
        }
    }

    #[test]
    fn fit_says_every_change_it_makes() {
        let task = Posture::for_target(AnchorTargetType::Task);
        assert_eq!(task.fit(Some(&full_caps())).degradations, vec![]);
        assert_eq!(
            task.fit(None),
            Fitted {
                posture: task.clone(),
                degradations: vec![Degradation::CapabilitiesUnknown]
            }
        );

        let mut no_mcp = full_caps();
        no_mcp.per_session_mcp = false;
        let f = task.fit(Some(&no_mcp));
        assert_eq!(f.posture.tools, set(&["Glob", "Grep", "Read"]));
        assert!(f.posture.skills.is_empty());
        assert!(
            matches!(&f.degradations[..], [Degradation::NoProjectTools { dropped }, Degradation::NoSkills { .. }] if dropped.len() == 4)
        );

        let mut no_tools = full_caps();
        no_tools.tools = false;
        let f = task.fit(Some(&no_tools));
        assert!(f.posture.tools.is_empty());
        assert!(matches!(f.degradations[0], Degradation::NoTools { .. }));

        let mut small = full_caps();
        small.context_window = Some(ContextWindow {
            value: 8_000,
            source: ContextWindowSource::Configured,
        });
        let f = task.fit(Some(&small));
        assert_eq!(f.posture.detail, Detail::Normal);
        assert_eq!(
            f.degradations,
            vec![Degradation::DetailLowered { window: 8_000 }]
        );
    }

    fn ids(a: &[&Anchor]) -> BTreeSet<Uuid> {
        a.iter().map(|a| a.id).collect()
    }

    #[test]
    fn an_inferred_change_is_adopted_only_after_stable_turns() {
        let mut t = PostureTracker::default();
        let plan = Posture::for_target(AnchorTargetType::Plan);
        for _ in 1..STABLE_TURNS {
            assert_eq!(t.observe(plan.clone(), BTreeSet::new()), None);
            assert_eq!(t.effective(), Some(&Posture::neutral()));
        }
        let s = t
            .observe(plan.clone(), BTreeSet::new())
            .expect("stable: adopted");
        assert_eq!(s.cause, SwitchCause::Stable);
        assert_eq!(
            s.announce.as_deref(),
            Some("Posture : planner · overview · step_plan (ancres stables)")
        );
        assert!(!s.announce.unwrap().contains('\n'), "one line");
        assert_eq!(t.effective(), Some(&plan));
    }

    #[test]
    fn a_flapping_derivation_switches_less_than_once_every_five_turns() {
        let mut t = PostureTracker::default();
        let a = Posture::for_target(AnchorTargetType::Plan);
        let b = Posture::for_target(AnchorTargetType::File);
        let mut switches = 0;
        for turn in 0..100 {
            // two turns on a, one on b, ... (never STABLE_TURNS in a row)
            let p = if turn % 3 == 2 { b.clone() } else { a.clone() };
            switches += t.observe(p, BTreeSet::new()).is_some() as u32;
        }
        assert_eq!(switches, 0, "never stable long enough");
        assert!(t.switches * 5 < t.turns);
    }

    #[test]
    fn a_new_explicit_focus_anchor_switches_at_once() {
        let mut t = PostureTracker::default();
        let file = focus(AnchorTargetType::File);
        let s = t
            .observe(derive(std::slice::from_ref(&file), None), ids(&[&file]))
            .expect("explicit: at once");
        assert_eq!(s.cause, SwitchCause::ExplicitAnchor);
        assert_eq!(t.effective().unwrap().persona, Persona::Developer);
        // the same anchors next turn: nothing
        assert_eq!(
            t.observe(derive(std::slice::from_ref(&file), None), ids(&[&file])),
            None
        );
        // a system anchor (backfill) is not explicit: hysteresis applies
        let mut backfill = anchor(
            AnchorTargetType::Project,
            &[AnchorRole::Focus],
            AnchorActor::System,
        );
        backfill.inferred = true;
        let both = [file.clone(), backfill];
        assert_eq!(explicit_focus(&both, None), ids(&[&file]));
        assert_eq!(
            t.observe(derive(&both, None), explicit_focus(&both, None)),
            None
        );
    }

    #[test]
    fn the_override_wins_and_auto_resumes_from_the_anchors() {
        let mut t = PostureTracker::default();
        let file = focus(AnchorTargetType::File);
        let pinned = Posture::for_target(AnchorTargetType::Decision);
        let s = t
            .set_override(PostureOverride::Pin(pinned.clone()))
            .unwrap();
        assert_eq!(s.cause, SwitchCause::Override);
        assert_eq!(
            t.set_override(PostureOverride::Pin(pinned.clone())),
            None,
            "no-op"
        );
        // anchors move: the pinned posture stays, the derived one is tracked
        assert_eq!(
            t.observe(derive(std::slice::from_ref(&file), None), ids(&[&file])),
            None
        );
        assert_eq!(t.effective(), Some(&pinned));
        let off = t.set_override(PostureOverride::Off).unwrap();
        assert_eq!(off.announce, None);
        assert_eq!(t.effective(), None);
        t.set_override(PostureOverride::Auto).unwrap();
        assert_eq!(t.effective().unwrap().persona, Persona::Developer);
    }

    #[test]
    fn the_book_observes_a_turn_without_touching_anything_and_journals_it() {
        let b = PostureBook::default();
        let sid = Uuid::new_v4();
        let plan = focus(AnchorTargetType::Plan);
        let o = b.observe_turn(sid, &[plan], None, &inv(SessionAccess::Normal), None);
        assert_eq!(o.turn, 1);
        assert_eq!(o.focus_types, vec!["plan"]);
        assert_eq!(o.derived.persona, Persona::Planner);
        assert_eq!(
            o.switch.as_ref().unwrap().cause,
            SwitchCause::ExplicitAnchor
        );
        assert_eq!(o.degradations, vec![Degradation::CapabilitiesUnknown]);
        assert!(o.segment_tokens > 0);
        let json = serde_json::to_value(&o).unwrap();
        assert_eq!(json["derived"]["persona"], "planner");
        assert_eq!(json["over"]["mode"], "auto");
        b.set_override(sid, PostureOverride::Off);
        let o = b.observe_turn(sid, &[], None, &inv(SessionAccess::Normal), None);
        assert_eq!((o.effective, o.segment_tokens), (None, 0));
        assert_eq!(b.tracker(sid).unwrap().turns, 2);
        b.forget(sid);
        assert!(b.tracker(sid).is_none());
    }
}
