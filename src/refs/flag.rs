//! The `refs_v1` switch.
//!
//! **On by default.** The feature is delivered with the release; the switch is
//! there to turn it OFF (`REFS_V1=0`) if it misbehaves in the field. Off, the
//! server behaves exactly as before: the `refs` field is ignored, `auth_ok`
//! carries no `features`, nothing is announced.

/// Environment variable of the switch.
pub const REFS_V1_VAR: &str = "REFS_V1";

/// The name a server announces (`auth_ok.features`, `GET /api/version`).
pub const FEATURE_REFS_V1: &str = "refs_v1";

/// Read the switch from a raw value. Absent or unrecognised → ON; only an
/// explicit "off" word turns it off, so a typo never silently disables it.
pub fn parse(value: Option<&str>) -> bool {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) => !matches!(v.as_str(), "0" | "false" | "off" | "no" | "disabled"),
        None => true,
    }
}

/// The switch as the process environment sets it.
pub fn from_env() -> bool {
    parse(std::env::var(REFS_V1_VAR).ok().as_deref())
}

/// What `auth_ok.features` carries: the name when on, `None` (field omitted)
/// when off.
pub fn features(enabled: bool) -> Option<Vec<&'static str>> {
    enabled.then(|| vec![FEATURE_REFS_V1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_unless_explicitly_turned_off() {
        assert!(parse(None));
        assert!(parse(Some("1")));
        assert!(parse(Some("true")));
        assert!(parse(Some("")));
        assert!(parse(Some("flase")), "a typo must not disable the feature");
        for off in ["0", "false", "FALSE", " off ", "No", "disabled"] {
            assert!(!parse(Some(off)), "{off:?}");
        }
    }

    #[test]
    fn announced_only_when_on() {
        assert_eq!(features(true), Some(vec!["refs_v1"]));
        assert_eq!(features(false), None);
    }
}
