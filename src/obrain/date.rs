//! The ONE decoder of a stored date.
//!
//! A date reaches an obrain base under three forms, and every reader of this
//! backend goes through here so that all three read as the same instant:
//!
//! - `Value::String`, RFC 3339: what this backend WRITES (`to_rfc3339()`, and
//!   the fixed-width nanosecond form of `trigger::instant`);
//! - `Value::ZonedDatetime`: what the Neo4j migration tool writes for a Neo4j
//!   `DateTime` (microseconds, plus an offset that does not change the instant);
//! - `Value::Timestamp`: what a Cypher `datetime()` leaves (nanoseconds).
//!
//! Reading by `Value::as_str()` only answered for the first: on a migrated base
//! every node whose `created_at` was required was quarantined, and an optional
//! date read as absent, or as "now", without a word.
//!
//! Two kinds of property, two rules:
//!
//! - an OPTIONAL date (`facultative`): absent, null or the empty string is `None`.
//!   A value that does not read as a date is read ABSENT, as the readers always
//!   did, and a warning names the property, the entity and what was found — so
//!   the node is still served, and nothing is lost without a trace;
//! - a REQUIRED date (`requise`, `ou_maintenant_si_absente`): absent is an error
//!   (or, where the Neo4j adapter defaults an absent property to "now", that
//!   default — and ONLY for an absent property), and a value present that does
//!   not read as a date is an error that NAMES the property, the entity and the
//!   variant received. The caller quarantines the node: a stored date is never
//!   replaced by "now", and the node is never served with a date it does not have.

use anyhow::{anyhow, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::Value;
use tracing::warn;

/// A property lookup: `None` for an absent property (the lookups of this backend
/// also answer `None` for a null).
pub(super) type Lire<'a> = &'a dyn Fn(&str) -> Option<Value>;

/// The instant a stored value denotes, or WHY it denotes none: the variant
/// received (and, for a string, its text and the parse error).
pub(super) fn decoder(v: &Value) -> std::result::Result<DateTime<Utc>, String> {
    match v {
        Value::String(s) => s
            .parse::<DateTime<Utc>>()
            .map_err(|e| format!("STRING « {s} » qui n'est pas une date RFC 3339 ({e})")),
        Value::Timestamp(t) => Ok(DateTime::from_timestamp_nanos(t.as_nanos())),
        // The offset is how the instant was WRITTEN, not another instant: the
        // domain values are all in UTC.
        Value::ZonedDatetime(z) => Ok(DateTime::from_timestamp_nanos(z.as_timestamp().as_nanos())),
        autre => Err(format!(
            "variante {} reçue, là où une date est une STRING RFC 3339, un TIMESTAMP ou un \
             ZONED DATETIME",
            autre.type_name()
        )),
    }
}

fn illisible(quoi: &str, cle: &str, pourquoi: String) -> anyhow::Error {
    anyhow!("{quoi} : `{cle}` illisible : {pourquoi}")
}

/// A REQUIRED date: absent is an error, and so is a value that is not a date
/// (the empty string included). The error names the property and what was found.
pub(super) fn requise(get: Lire<'_>, cle: &str, quoi: &str) -> Result<DateTime<Utc>> {
    match get(cle) {
        None | Some(Value::Null) => Err(anyhow!("{quoi} sans propriété `{cle}`")),
        Some(v) => decoder(&v).map_err(|pourquoi| illisible(quoi, cle, pourquoi)),
    }
}

/// An OPTIONAL date: absent, null or the empty string is `None`. A value that is
/// present and does not read as a date reads ABSENT (what the readers did before
/// the three forms were read), and a warning names the property, the entity and
/// what was found, so the loss is visible in the logs.
pub(super) fn facultative(get: Lire<'_>, cle: &str, quoi: &str) -> Result<Option<DateTime<Utc>>> {
    match get(cle) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(v) => match decoder(&v) {
            Ok(d) => Ok(Some(d)),
            Err(pourquoi) => {
                warn!(
                    propriete = cle,
                    entite = quoi,
                    "date optionnelle illisible, lue absente : {pourquoi}"
                );
                Ok(None)
            }
        },
    }
}

/// A REQUIRED date that the Neo4j adapter reads as "now" when the property is
/// ABSENT (`node_to_decision`, the chat sessions and events). That default is
/// kept, for an absent property ONLY: a value that is PRESENT is read under its
/// three forms, and one that is not a date, the empty string included, is an
/// error, never replaced by "now".
pub(super) fn ou_maintenant_si_absente(
    get: Lire<'_>,
    cle: &str,
    quoi: &str,
) -> Result<DateTime<Utc>> {
    match get(cle) {
        None | Some(Value::Null) => Ok(Utc::now()),
        Some(v) => decoder(&v).map_err(|pourquoi| illisible(quoi, cle, pourquoi)),
    }
}

/// A date already looked up on an EDGE, for a field that is required (the edge's
/// `last_verified`): absent or null is `None`; a value present that is not a date,
/// the empty string included, is an error, naming the property and the variant.
pub(super) fn de_valeur(v: Option<&Value>, cle: &str, quoi: &str) -> Result<Option<DateTime<Utc>>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => decoder(v)
            .map(Some)
            .map_err(|pourquoi| illisible(quoi, cle, pourquoi)),
    }
}

/// A date kept as TEXT by its domain value (a sort key, a `toString(…)` of the
/// Neo4j adapter): a string is returned AS STORED, a temporal value as RFC 3339
/// in UTC with a `Z`. Absent, null, or another variant: `None`.
pub(super) fn en_texte(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.to_string()),
        v @ (Value::Timestamp(_) | Value::ZonedDatetime(_)) => decoder(v)
            .ok()
            .map(|d| d.to_rfc3339_opts(SecondsFormat::AutoSi, true)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_common::types::{Timestamp, ZonedDatetime};

    const NANOS: i64 = 1_770_747_886_858_105_123;

    #[test]
    fn les_trois_formes_d_un_meme_instant_se_lisent_pareil() {
        let attendu = DateTime::from_timestamp_nanos(NANOS);
        assert_eq!(decoder(&Value::from(attendu.to_rfc3339())), Ok(attendu));
        assert_eq!(
            decoder(&Value::Timestamp(Timestamp::from_nanos(NANOS))),
            Ok(attendu)
        );
        // A `ZonedDatetime` keeps microseconds, and its offset is not the instant.
        let micro = DateTime::from_timestamp_micros(NANOS / 1000).unwrap();
        for decalage in [0, 7200, -19800] {
            assert_eq!(
                decoder(&Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                    Timestamp::from_nanos(NANOS),
                    decalage
                ))),
                Ok(micro),
                "décalage {decalage}"
            );
        }
        // A string with an offset is the same instant in UTC.
        assert_eq!(
            decoder(&Value::from("2026-02-10T20:24:46.858105123+02:00")),
            Ok(attendu)
        );
    }

    #[test]
    fn une_date_d_avant_1970_se_lit() {
        let avant = DateTime::from_timestamp_nanos(-1_500_000_000);
        assert_eq!(
            decoder(&Value::Timestamp(Timestamp::from_nanos(-1_500_000_000))),
            Ok(avant)
        );
        assert_eq!(
            decoder(&Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_nanos(-1_500_000_000),
                0
            ))),
            Ok(avant)
        );
    }

    #[test]
    fn un_requis_illisible_nomme_la_propriete_et_la_variante() {
        let get = |cle: &str| match cle {
            "entier" => Some(Value::Int64(5)),
            "texte" => Some(Value::from("pas une date")),
            "vide" => Some(Value::from("")),
            _ => None,
        };
        let e = format!("{:#}", requise(&get, "entier", "projet X").unwrap_err());
        assert!(
            e.contains("projet X") && e.contains("`entier`") && e.contains("INT64"),
            "{e}"
        );
        let e = format!("{:#}", requise(&get, "texte", "projet X").unwrap_err());
        assert!(
            e.contains("`texte`") && e.contains("STRING") && e.contains("pas une date"),
            "{e}"
        );
        let e = format!("{:#}", requise(&get, "absente", "projet X").unwrap_err());
        assert!(e.contains("sans propriété `absente`"), "{e}");
        let e = format!("{:#}", requise(&get, "vide", "projet X").unwrap_err());
        assert!(
            e.contains("`vide` illisible") && e.contains("STRING"),
            "{e}"
        );
    }

    #[test]
    fn un_optionnel_illisible_est_absent_et_absent_ou_vide_aussi() {
        let get = |cle: &str| match cle {
            "entier" => Some(Value::Int64(5)),
            "texte" => Some(Value::from("pas une date")),
            "vide" => Some(Value::from("")),
            _ => None,
        };
        // Read as absent (the service keeps going), not quarantined.
        assert_eq!(facultative(&get, "entier", "projet X").unwrap(), None);
        assert_eq!(facultative(&get, "texte", "projet X").unwrap(), None);
        assert_eq!(facultative(&get, "absente", "projet X").unwrap(), None);
        assert_eq!(facultative(&get, "vide", "projet X").unwrap(), None);
        // A valid value is still read, whatever its form.
        let valide =
            |cle: &str| (cle == "t").then(|| Value::Timestamp(Timestamp::from_nanos(NANOS)));
        assert_eq!(
            facultative(&valide, "t", "projet X").unwrap(),
            Some(DateTime::from_timestamp_nanos(NANOS))
        );
    }

    #[test]
    fn un_defaut_maintenant_ne_s_applique_qu_a_l_absence() {
        let get = |cle: &str| match cle {
            "texte" => Some(Value::from("pas une date")),
            "vide" => Some(Value::from("")),
            _ => None,
        };
        let avant = Utc::now();
        let absente = ou_maintenant_si_absente(&get, "absente", "décision X").unwrap();
        assert!(absente >= avant);
        // Present and not a date, the empty string included: an error naming it, never
        // "now" (a stored value is never replaced).
        let e = format!(
            "{:#}",
            ou_maintenant_si_absente(&get, "texte", "décision X").unwrap_err()
        );
        assert!(e.contains("`texte`") && e.contains("pas une date"), "{e}");
        let e = format!(
            "{:#}",
            ou_maintenant_si_absente(&get, "vide", "décision X").unwrap_err()
        );
        assert!(e.contains("`vide`"), "{e}");
    }

    #[test]
    fn une_date_gardee_en_texte_garde_la_chaine_et_ecrit_les_autres_formes_en_z() {
        assert_eq!(
            en_texte(Some(&Value::from("2026-01-02T03:04:05+00:00"))).as_deref(),
            Some("2026-01-02T03:04:05+00:00")
        );
        assert_eq!(
            en_texte(Some(&Value::ZonedDatetime(
                ZonedDatetime::from_timestamp_offset(Timestamp::from_nanos(NANOS), 0)
            )))
            .as_deref(),
            Some("2026-02-10T18:24:46.858105Z")
        );
        assert_eq!(en_texte(Some(&Value::Int64(5))), None);
        assert_eq!(en_texte(None), None);
    }
}
