//! La règle de réception d'un déclencheur `webhook` qui déclare une `source`, et sa
//! validation à la création.
//!
//! Le secret n'est JAMAIS dans la configuration : la règle porte une référence
//! `secret_ref: "vault:<nom>"`, résolue dans le coffre au moment de la réception
//! (grant de portée `Provider("trigger:<id>")`, voir `api::vault_handlers`). Une clé
//! `secret` en clair est refusée à la création et ignorée à la réception.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use crate::runner::{Trigger, TriggerType};

/// En-tête de signature quand le déclencheur n'en désigne pas un autre (format GitHub).
pub const DEFAULT_SIGNATURE_HEADER: &str = "x-hub-signature-256";

/// Préfixe de l'instance sous laquelle un secret de déclencheur est accordé dans le
/// coffre (`GrantScope::Provider("trigger:<uuid>")`).
pub const TRIGGER_GRANT_PREFIX: &str = "trigger:";

/// Longueur minimale (octets) d'un secret résolu : en deçà, la clé HMAC se devine.
pub const MIN_SECRET_BYTES: usize = 16;

/// Longueur maximale d'un identifiant de livraison conservé.
const MAX_DELIVERY_ID: usize = 128;

/// En-têtes qu'un déclencheur ne peut pas désigner comme signature ou identifiant.
const FORBIDDEN_HEADERS: &[&str] = &[
    "authorization",
    "cookie",
    "content-length",
    "content-type",
    "host",
    "transfer-encoding",
    "connection",
    "proxy-authorization",
];

/// La règle de réception d'un déclencheur. Ne porte que le NOM du secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub source: String,
    /// Nom du secret dans le coffre (sans le préfixe `vault:`), s'il est déclaré.
    pub secret_name: Option<String>,
    pub signature_header: String,
    pub delivery_header: Option<String>,
}

impl Rule {
    /// Une règle existe seulement pour un déclencheur `webhook` dont la
    /// configuration déclare une `source`. Une clé `secret` en clair est ignorée.
    pub fn of(trigger: &Trigger) -> Option<Self> {
        if trigger.trigger_type != TriggerType::Webhook {
            return None;
        }
        let config = &trigger.config;
        Some(Self {
            source: config.get("source")?.as_str()?.to_string(),
            secret_name: config
                .get("secret_ref")
                .and_then(|v| v.as_str())
                .and_then(|r| r.strip_prefix("vault:"))
                .filter(|n| crate::vault::store::validate_name(n).is_ok())
                .map(str::to_string),
            signature_header: config
                .get("signature_header")
                .and_then(|v| v.as_str())
                .unwrap_or(DEFAULT_SIGNATURE_HEADER)
                .to_ascii_lowercase(),
            delivery_header: config
                .get("delivery_header")
                .and_then(|v| v.as_str())
                .map(str::to_ascii_lowercase),
        })
    }

    /// L'instance sous laquelle le secret de ce déclencheur est accordé.
    pub fn grant_instance(trigger_id: uuid::Uuid) -> String {
        format!("{TRIGGER_GRANT_PREFIX}{trigger_id}")
    }
}

fn valid_source(source: &str) -> bool {
    (1..=32).contains(&source.len())
        && source
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

fn valid_header_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    (1..=64).contains(&name.len())
        && axum::http::HeaderName::from_bytes(name.as_bytes()).is_ok()
        && !FORBIDDEN_HEADERS.contains(&lower.as_str())
}

/// Valide la configuration d'un déclencheur à sa création. Renvoie le message
/// d'erreur (sans valeur de secret) à rendre en 400.
///
/// - toute clé `secret` dont la valeur commence par `vault:` est refusée : elle
///   serait prise au pied de la lettre comme clé ;
/// - un déclencheur `webhook` qui déclare une `source` (réception publique) :
///   `source` en `[a-z0-9-]{1,32}`, `secret_ref` obligatoire en `vault:<nom>`,
///   AUCUNE clé `secret` en clair, en-têtes désignés valides et non sensibles.
pub fn validate_trigger_config(
    trigger_type: &TriggerType,
    config: &serde_json::Value,
) -> Result<(), String> {
    if config
        .get("secret")
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.starts_with("vault:"))
    {
        return Err(
            "config.secret holds a vault reference: name it in config.secret_ref instead".into(),
        );
    }
    let Some(source) = config.get("source") else {
        return Ok(());
    };
    if *trigger_type != TriggerType::Webhook {
        return Err("config.source is only meaningful on a webhook trigger".into());
    }
    let source = source.as_str().unwrap_or_default();
    if !valid_source(source) {
        return Err("config.source is 1 to 32 lowercase letters, digits or '-'".into());
    }
    if config.get("secret").is_some() {
        return Err(
            "a webhook trigger with a source never holds its secret in clear: store it in the vault and set config.secret_ref = \"vault:<name>\"".into(),
        );
    }
    let secret_ref = config.get("secret_ref").and_then(|v| v.as_str());
    let name_ok = secret_ref
        .and_then(|r| r.strip_prefix("vault:"))
        .is_some_and(|n| crate::vault::store::validate_name(n).is_ok());
    if !name_ok {
        return Err(
            "a webhook trigger with a source needs config.secret_ref = \"vault:<name>\"".into(),
        );
    }
    for key in ["signature_header", "delivery_header"] {
        if let Some(value) = config.get(key) {
            if !value.as_str().is_some_and(valid_header_name) {
                return Err(format!(
                    "config.{key} must be a plain header name (not authorization, cookie, host, content-*)"
                ));
            }
        }
    }
    Ok(())
}

/// Une copie de la configuration où toute valeur `secret` est masquée : ce qui
/// sort par l'API et par `Debug`.
pub fn redact_config(config: &serde_json::Value) -> serde_json::Value {
    let mut out = config.clone();
    if let Some(obj) = out.as_object_mut() {
        if obj.contains_key("secret") {
            obj.insert("secret".into(), serde_json::Value::from("[redacted]"));
        }
    }
    out
}

/// L'en-tête de signature est-il bien formé (`sha256=` puis 64 hex) ? Vérifié
/// avant toute lecture du corps.
pub fn signature_well_formed(value: &str) -> bool {
    value
        .strip_prefix("sha256=")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// L'identifiant de livraison, borné et assaini : `None` s'il est absent, trop
/// long ou s'il porte autre chose que `[A-Za-z0-9._-]`.
pub fn sanitize_delivery_id(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    let ok = (1..=MAX_DELIVERY_ID).contains(&raw.len())
        && raw
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    ok.then(|| raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use serde_json::json;

    #[test]
    fn a_plain_secret_on_a_source_trigger_is_refused() {
        let cfg = json!({"source": "github", "secret": "x", "secret_ref": "vault:gh"});
        let err = validate_trigger_config(&TriggerType::Webhook, &cfg).unwrap_err();
        assert!(err.contains("never holds its secret"), "{err}");
    }

    #[test]
    fn a_vault_reference_in_the_secret_key_is_refused_everywhere() {
        let cfg = json!({"secret": "vault:gh-hook"});
        assert!(validate_trigger_config(&TriggerType::Webhook, &cfg).is_err());
    }

    #[test]
    fn a_source_trigger_needs_a_vault_reference() {
        for cfg in [
            json!({"source": "github"}),
            json!({"source": "github", "secret_ref": "gh"}),
            json!({"source": "github", "secret_ref": "vault:"}),
            json!({"source": "github", "secret_ref": "vault:bad name"}),
        ] {
            assert!(
                validate_trigger_config(&TriggerType::Webhook, &cfg).is_err(),
                "{cfg}"
            );
        }
        let ok = json!({"source": "github", "secret_ref": "vault:gh-hook"});
        assert!(validate_trigger_config(&TriggerType::Webhook, &ok).is_ok());
    }

    #[test]
    fn sources_and_headers_are_checked() {
        for cfg in [
            json!({"source": "GitHub", "secret_ref": "vault:k"}),
            json!({"source": "", "secret_ref": "vault:k"}),
            json!({"source": "g", "secret_ref": "vault:k", "signature_header": "Authorization"}),
            json!({"source": "g", "secret_ref": "vault:k", "delivery_header": "cookie"}),
            json!({"source": "g", "secret_ref": "vault:k", "delivery_header": "a b"}),
        ] {
            assert!(
                validate_trigger_config(&TriggerType::Webhook, &cfg).is_err(),
                "{cfg}"
            );
        }
        let cfg = json!({"source": "g", "secret_ref": "vault:k"});
        assert!(validate_trigger_config(&TriggerType::Schedule, &cfg).is_err());
    }

    #[test]
    fn redaction_hides_the_secret_value() {
        let red = redact_config(&json!({"secret": "abc", "x": 1}));
        assert_eq!(red, json!({"secret": "[redacted]", "x": 1}));
        assert_eq!(redact_config(&json!({"x": 1})), json!({"x": 1}));
    }

    #[test]
    fn signature_and_delivery_id_shapes() {
        assert!(signature_well_formed(&format!("sha256={}", "a".repeat(64))));
        assert!(!signature_well_formed("sha256=00"));
        assert!(!signature_well_formed(&format!("sha1={}", "a".repeat(64))));
        assert!(!signature_well_formed(&format!(
            "sha256={}",
            "g".repeat(64)
        )));
        assert_eq!(
            sanitize_delivery_id(Some("abc-1.2_3")),
            Some("abc-1.2_3".into())
        );
        assert_eq!(sanitize_delivery_id(Some("<script>")), None);
        assert_eq!(sanitize_delivery_id(Some(&"a".repeat(129))), None);
        assert_eq!(sanitize_delivery_id(None), None);
    }
}
