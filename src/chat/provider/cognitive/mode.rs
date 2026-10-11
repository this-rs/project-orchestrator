//! Routing mode and learning stage (decision R2): how much of the model choice
//! the user delegates to PO, and how far the learnt choices are trusted.
//!
//! Settings only. Nothing here takes a decision: the cognitive router that
//! reads these settings comes in a later slice (B-R4). The default, `primary`
//! + `shadow`, leaves the resolver's observable behaviour strictly unchanged.
//!
//! Stored as one JSON document per scope (`LlmSetting {scope, key: "routing"}`,
//! see [`super::load_routing`]); every field is read with a default, so an
//! older document, or no document at all, is valid (no migration).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::provider::settings::RoleTarget;

/// Key of the routing document, in the `global` scope or a `project:<slug>` scope.
pub const ROUTING_KEY: &str = "routing";

/// Default exploration rate (share of decisions that try a non-best arm).
pub const DEFAULT_EXPLORATION_EPSILON: f64 = 0.05;
/// Upper bound of the exploration rate: beyond it, "learning" is gambling.
pub const MAX_EXPLORATION_EPSILON: f64 = 0.25;
/// Default weight of the normalised cost in the utility.
pub const DEFAULT_COST_WEIGHT: f64 = 0.3;
/// Default weight of the normalised latency in the utility.
pub const DEFAULT_LATENCY_WEIGHT: f64 = 0.1;
/// Default window (decisions) after which a learnt arm that does worse than the
/// declarative choice demotes the stage back to `shadow`.
pub const DEFAULT_DEMOTE_AFTER: u32 = 20;
/// Smallest meaningful demotion window.
pub const MIN_DEMOTE_AFTER: u32 = 5;

/// Who chooses the provider and model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderRoutingMode {
    /// One exclusive primary provider: the current behaviour. PO only records
    /// what it would have chosen.
    #[default]
    Primary,
    /// The primary pilots the conversation; PO routes the executor sessions
    /// (runner, delegation, utilities).
    Mixed,
    /// PO routes everything, the pilot included.
    Full,
}

impl ProviderRoutingMode {
    /// Every mode, in the order the interface shows them.
    pub const ALL: [Self; 3] = [Self::Primary, Self::Mixed, Self::Full];

    /// Stable serialised name (identical to the serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Mixed => "mixed",
            Self::Full => "full",
        }
    }
}

/// How far the learnt choices are trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LearningStage {
    /// Everything is computed and recorded, nothing is applied.
    #[default]
    Shadow,
    /// The learnt choice is proposed, a person confirms.
    Advisory,
    /// The learnt choice is applied.
    Auto,
}

impl LearningStage {
    /// Every stage, from the most cautious to the most trusting.
    pub const ALL: [Self; 3] = [Self::Shadow, Self::Advisory, Self::Auto];

    /// Stable serialised name (identical to the serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Advisory => "advisory",
            Self::Auto => "auto",
        }
    }
}

fn default_epsilon() -> f64 {
    DEFAULT_EXPLORATION_EPSILON
}
fn default_cost_weight() -> f64 {
    DEFAULT_COST_WEIGHT
}
fn default_latency_weight() -> f64 {
    DEFAULT_LATENCY_WEIGHT
}
fn default_demote_after() -> u32 {
    DEFAULT_DEMOTE_AFTER
}

/// The routing settings of one scope. Every field has a default so that a
/// partial or older document still reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingSettings {
    /// Who chooses.
    #[serde(default)]
    pub mode: ProviderRoutingMode,
    /// How far the learnt choices are trusted.
    #[serde(default)]
    pub stage: LearningStage,
    /// The primary instance (and model or alias), when the user names one here
    /// rather than through the role assignments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<RoleTarget>,
    /// Exploration rate, `0.0..=0.25`.
    #[serde(default = "default_epsilon")]
    pub exploration_epsilon: f64,
    /// Weight of the normalised cost in the utility, `0.0..=1.0`.
    #[serde(default = "default_cost_weight")]
    pub cost_weight: f64,
    /// Weight of the normalised latency in the utility, `0.0..=1.0`.
    #[serde(default = "default_latency_weight")]
    pub latency_weight: f64,
    /// Decisions after which a worse learnt arm demotes the stage, `>= 5`.
    #[serde(default = "default_demote_after")]
    pub demote_after: u32,
}

impl Default for RoutingSettings {
    fn default() -> Self {
        Self {
            mode: ProviderRoutingMode::Primary,
            stage: LearningStage::Shadow,
            primary: None,
            exploration_epsilon: DEFAULT_EXPLORATION_EPSILON,
            cost_weight: DEFAULT_COST_WEIGHT,
            latency_weight: DEFAULT_LATENCY_WEIGHT,
            demote_after: DEFAULT_DEMOTE_AFTER,
        }
    }
}

/// Why a routing body was refused. The `code` is what the API answers; the
/// messages never echo a value the caller sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    /// `mode` is not `primary`, `mixed` or `full`.
    InvalidMode,
    /// `stage` is not `shadow`, `advisory` or `auto`.
    InvalidStage,
    /// A numeric knob is out of its bounds; the text names the field and the bounds.
    InvalidWeight(String),
    /// `primary` does not name a usable target.
    InvalidPrimary(String),
    /// The body does not have the shape of routing settings at all.
    Malformed,
}

impl RoutingError {
    /// Stable error code (`400` body: `{"code": ...}`).
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidMode => "invalid_routing_mode",
            Self::InvalidStage => "invalid_learning_stage",
            Self::InvalidWeight(_) => "invalid_routing_weight",
            Self::InvalidPrimary(_) => "invalid_routing_primary",
            Self::Malformed => "invalid_routing_settings",
        }
    }
}

impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMode => {
                f.write_str("unknown routing mode: expected primary, mixed or full")
            }
            Self::InvalidStage => {
                f.write_str("unknown learning stage: expected shadow, advisory or auto")
            }
            Self::InvalidWeight(why) | Self::InvalidPrimary(why) => f.write_str(why),
            Self::Malformed => f.write_str("the body does not have the shape of routing settings"),
        }
    }
}

impl std::error::Error for RoutingError {}

fn in_bounds(name: &str, value: f64, max: f64) -> Result<(), RoutingError> {
    if value.is_finite() && (0.0..=max).contains(&value) {
        Ok(())
    } else {
        Err(RoutingError::InvalidWeight(format!(
            "{name} must be a number between 0 and {max}"
        )))
    }
}

/// Refuses settings whose knobs are out of bounds. The mode and stage are
/// already typed here; a body with an unknown one is refused by
/// [`parse_routing_settings`] before it gets this far.
pub fn validate_routing_settings(settings: &RoutingSettings) -> Result<(), RoutingError> {
    in_bounds(
        "exploration_epsilon",
        settings.exploration_epsilon,
        MAX_EXPLORATION_EPSILON,
    )?;
    in_bounds("cost_weight", settings.cost_weight, 1.0)?;
    in_bounds("latency_weight", settings.latency_weight, 1.0)?;
    if settings.demote_after < MIN_DEMOTE_AFTER {
        return Err(RoutingError::InvalidWeight(format!(
            "demote_after must be at least {MIN_DEMOTE_AFTER}"
        )));
    }
    if let Some(primary) = &settings.primary {
        if primary.provider.trim().is_empty() {
            return Err(RoutingError::InvalidPrimary(
                "primary must name a provider instance".to_string(),
            ));
        }
        if primary.model.is_some() && primary.alias.is_some() {
            return Err(RoutingError::InvalidPrimary(
                "primary names a model OR an alias, not both".to_string(),
            ));
        }
    }
    Ok(())
}

fn enum_field(
    body: &Value,
    field: &str,
    allowed: &[&str],
    err: RoutingError,
) -> Result<(), RoutingError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(s)) if allowed.contains(&s.as_str()) => Ok(()),
        Some(_) => Err(err),
    }
}

/// Parses and validates a request body. The mode and stage are checked by
/// name first so that an unknown one answers its own code instead of a
/// generic deserialisation failure; a field that is absent takes its default.
pub fn parse_routing_settings(body: &Value) -> Result<RoutingSettings, RoutingError> {
    if !body.is_object() {
        return Err(RoutingError::Malformed);
    }
    let modes: Vec<&str> = ProviderRoutingMode::ALL
        .iter()
        .map(|m| m.as_str())
        .collect();
    let stages: Vec<&str> = LearningStage::ALL.iter().map(|s| s.as_str()).collect();
    enum_field(body, "mode", &modes, RoutingError::InvalidMode)?;
    enum_field(body, "stage", &stages, RoutingError::InvalidStage)?;
    let settings: RoutingSettings =
        serde_json::from_value(body.clone()).map_err(|_| RoutingError::Malformed)?;
    validate_routing_settings(&settings)?;
    Ok(settings)
}

/// Which scope the effective settings come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoutingScope {
    /// The server-wide document.
    Global,
    /// The project's own document (overrides the global one).
    Project,
    /// No document anywhere: the built-in default.
    Default,
}

impl RoutingScope {
    /// Stable serialised name (identical to the serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Project => "project",
            Self::Default => "default",
        }
    }
}

/// The settings that apply, and where they come from: project > global > default.
pub fn effective_routing(
    global: Option<RoutingSettings>,
    project: Option<RoutingSettings>,
) -> (RoutingSettings, RoutingScope) {
    match (global, project) {
        (_, Some(p)) => (p, RoutingScope::Project),
        (Some(g), None) => (g, RoutingScope::Global),
        (None, None) => (RoutingSettings::default(), RoutingScope::Default),
    }
}

/// Whether the user handed THIS conversation's routing to PO (decision R-S1): Auto in the
/// menu (`routing_mode: full`) or two models ticked or more (a pool). One model ticked is
/// strict (a pin), and a conversation that asked nothing follows the settings.
pub fn conversation_routes(mode: Option<ProviderRoutingMode>, pool_len: usize) -> bool {
    mode == Some(ProviderRoutingMode::Full) || pool_len > 1
}

/// The stage a conversation's decisions are taken at (decision R-S1): a choice the user
/// made on the conversation ([`conversation_routes`]) counts as `auto` for it, whatever the
/// settings say. The settings' stage (`shadow` by default) is a LEARNING setting: it
/// governs the sessions that asked nothing, the runner's executors and delegations, and
/// the shadow report; it never vetoes an explicit choice of the user.
pub fn conversation_stage(settings: LearningStage, conversation_routes: bool) -> LearningStage {
    if conversation_routes {
        LearningStage::Auto
    } else {
        settings
    }
}

/// Body of `GET /api/chat/routing` and `GET /api/projects/{slug}/routing`:
/// the settings, flattened, plus the scope they come from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectiveRouting {
    /// The settings that apply.
    #[serde(flatten)]
    pub settings: RoutingSettings,
    /// Where they come from.
    pub scope: RoutingScope,
}

impl EffectiveRouting {
    /// Pairs settings with their scope.
    pub fn new(settings: RoutingSettings, scope: RoutingScope) -> Self {
        Self { settings, scope }
    }
}

impl From<(RoutingSettings, RoutingScope)> for EffectiveRouting {
    fn from((settings, scope): (RoutingSettings, RoutingScope)) -> Self {
        Self::new(settings, scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_choice_on_the_conversation_is_the_auto_stage_for_it_only() {
        use ProviderRoutingMode::*;
        // Auto, or a pool of two models or more: the user handed the routing to PO.
        assert!(conversation_routes(Some(Full), 0));
        assert!(conversation_routes(Some(Mixed), 2));
        assert!(conversation_routes(None, 3));
        // Nothing asked, one model ticked (a pin), or a mode without a pool: the settings.
        assert!(!conversation_routes(None, 0));
        assert!(!conversation_routes(Some(Primary), 1));
        assert!(!conversation_routes(Some(Mixed), 0));
        for stage in LearningStage::ALL {
            assert_eq!(conversation_stage(stage, true), LearningStage::Auto);
            assert_eq!(conversation_stage(stage, false), stage);
        }
    }

    #[test]
    fn the_default_is_primary_and_shadow_with_the_documented_knobs() {
        let d = RoutingSettings::default();
        assert_eq!(d.mode, ProviderRoutingMode::Primary);
        assert_eq!(d.stage, LearningStage::Shadow);
        assert_eq!(d.primary, None);
        assert_eq!(d.exploration_epsilon, 0.05);
        assert_eq!(d.cost_weight, 0.3);
        assert_eq!(d.latency_weight, 0.1);
        assert_eq!(d.demote_after, 20);
        assert!(validate_routing_settings(&d).is_ok());
        // An empty document reads as the default: no migration needed.
        assert_eq!(parse_routing_settings(&json!({})).unwrap(), d);
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["mode"], "primary");
        assert_eq!(v["stage"], "shadow");
        assert!(v.get("primary").is_none(), "absent, not null");
    }

    #[test]
    fn modes_and_stages_round_trip_through_serde_and_as_str() {
        for m in ProviderRoutingMode::ALL {
            let s = serde_json::to_string(&m).unwrap();
            assert_eq!(s, format!("\"{}\"", m.as_str()));
            assert_eq!(serde_json::from_str::<ProviderRoutingMode>(&s).unwrap(), m);
        }
        for st in LearningStage::ALL {
            let s = serde_json::to_string(&st).unwrap();
            assert_eq!(s, format!("\"{}\"", st.as_str()));
            assert_eq!(serde_json::from_str::<LearningStage>(&s).unwrap(), st);
        }
        assert_eq!(
            ProviderRoutingMode::ALL.map(|m| m.as_str()),
            ["primary", "mixed", "full"]
        );
        assert_eq!(
            LearningStage::ALL.map(|s| s.as_str()),
            ["shadow", "advisory", "auto"]
        );
        for sc in [
            RoutingScope::Global,
            RoutingScope::Project,
            RoutingScope::Default,
        ] {
            let s = serde_json::to_string(&sc).unwrap();
            assert_eq!(s, format!("\"{}\"", sc.as_str()));
            assert_eq!(serde_json::from_str::<RoutingScope>(&s).unwrap(), sc);
        }
    }

    #[test]
    fn a_full_document_round_trips_with_its_primary() {
        let s = RoutingSettings {
            mode: ProviderRoutingMode::Full,
            stage: LearningStage::Advisory,
            primary: Some(RoleTarget {
                provider: "deepseek".into(),
                model: Some("deepseek-chat".into()),
                alias: None,
            }),
            exploration_epsilon: 0.2,
            cost_weight: 1.0,
            latency_weight: 0.0,
            demote_after: 5,
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["primary"]["provider"], "deepseek");
        assert_eq!(parse_routing_settings(&v).unwrap(), s);
    }

    #[test]
    fn an_unknown_mode_or_stage_is_refused_with_its_own_code() {
        let e = parse_routing_settings(&json!({"mode": "turbo"})).unwrap_err();
        assert_eq!(e, RoutingError::InvalidMode);
        assert_eq!(e.code(), "invalid_routing_mode");
        assert!(!e.to_string().contains("turbo"), "never echoes the value");
        let e = parse_routing_settings(&json!({"mode": 3})).unwrap_err();
        assert_eq!(e.code(), "invalid_routing_mode");
        let e = parse_routing_settings(&json!({"stage": "yolo"})).unwrap_err();
        assert_eq!(e, RoutingError::InvalidStage);
        assert_eq!(e.code(), "invalid_learning_stage");
        // Serde alone refuses them too (a stored document is never widened).
        assert!(serde_json::from_str::<ProviderRoutingMode>("\"turbo\"").is_err());
        assert!(serde_json::from_str::<LearningStage>("\"yolo\"").is_err());
        assert!(serde_json::from_str::<ProviderRoutingMode>("\"Primary\"").is_err());
        // Not an object at all.
        assert_eq!(
            parse_routing_settings(&json!([1, 2])).unwrap_err().code(),
            "invalid_routing_settings"
        );
        assert_eq!(
            parse_routing_settings(&json!({"cost_weight": "a lot"}))
                .unwrap_err()
                .code(),
            "invalid_routing_settings"
        );
    }

    #[test]
    fn the_knobs_are_bounded() {
        let weight = |body: Value| {
            let e = parse_routing_settings(&body).unwrap_err();
            assert_eq!(e.code(), "invalid_routing_weight", "{body}");
            e.to_string()
        };
        assert!(weight(json!({"exploration_epsilon": 0.26})).contains("exploration_epsilon"));
        assert!(weight(json!({"exploration_epsilon": -0.01})).contains("exploration_epsilon"));
        assert!(weight(json!({"cost_weight": 1.01})).contains("cost_weight"));
        assert!(weight(json!({"latency_weight": -1.0})).contains("latency_weight"));
        assert!(weight(json!({"demote_after": 4})).contains("demote_after"));
        // The bounds themselves are accepted.
        for body in [
            json!({"exploration_epsilon": 0.0}),
            json!({"exploration_epsilon": 0.25}),
            json!({"cost_weight": 0.0, "latency_weight": 1.0}),
            json!({"demote_after": 5}),
        ] {
            assert!(parse_routing_settings(&body).is_ok(), "{body}");
        }
        // NaN and infinity are not "in bounds".
        let mut s = RoutingSettings {
            cost_weight: f64::NAN,
            ..Default::default()
        };
        assert_eq!(
            validate_routing_settings(&s).unwrap_err().code(),
            "invalid_routing_weight"
        );
        s.cost_weight = f64::INFINITY;
        assert!(validate_routing_settings(&s).is_err());
    }

    #[test]
    fn a_primary_names_an_instance_and_a_model_or_an_alias() {
        let e = parse_routing_settings(&json!({"primary": {"provider": "  "}})).unwrap_err();
        assert_eq!(e.code(), "invalid_routing_primary");
        let e = parse_routing_settings(
            &json!({"primary": {"provider": "ds", "model": "m", "alias": "fast"}}),
        )
        .unwrap_err();
        assert_eq!(e.code(), "invalid_routing_primary");
        assert!(
            parse_routing_settings(&json!({"primary": {"provider": "ds", "alias": "fast"}}))
                .is_ok()
        );
        assert!(parse_routing_settings(&json!({"primary": null})).is_ok());
    }

    #[test]
    fn effective_routing_prefers_project_then_global_then_default() {
        let global = RoutingSettings {
            mode: ProviderRoutingMode::Mixed,
            ..Default::default()
        };
        let project = RoutingSettings {
            mode: ProviderRoutingMode::Full,
            stage: LearningStage::Auto,
            ..Default::default()
        };
        assert_eq!(
            effective_routing(None, None),
            (RoutingSettings::default(), RoutingScope::Default)
        );
        assert_eq!(
            effective_routing(Some(global.clone()), None),
            (global.clone(), RoutingScope::Global)
        );
        assert_eq!(
            effective_routing(Some(global.clone()), Some(project.clone())),
            (project.clone(), RoutingScope::Project)
        );
        // A project document wins even when there is no global one.
        assert_eq!(
            effective_routing(None, Some(project.clone())),
            (project, RoutingScope::Project)
        );
        let v = serde_json::to_value(EffectiveRouting::from(effective_routing(
            Some(global),
            None,
        )))
        .unwrap();
        assert_eq!(v["mode"], "mixed");
        assert_eq!(v["scope"], "global");
        assert_eq!(v["demote_after"], 20);
    }
}
