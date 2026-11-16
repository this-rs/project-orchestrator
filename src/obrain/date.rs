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
//! date read as absent — or as "now".
//!
//! The rules, the same for every domain:
//!
//! - ABSENT (no property, or null) is `None` for an optional date and an error
//!   for a required one;
//! - the EMPTY STRING is absent for an optional date (the repo convention: `""`
//!   means "none"), and unreadable for a required one;
//! - PRESENT but not a date (another variant, or a string that does not parse) is
//!   an error that NAMES the property and the variant received. It is never read
//!   as absent, and never replaced by `Utc::now()`: the caller quarantines the
//!   node (or refuses the read).

use anyhow::{anyhow, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use obrain_common::types::Value;

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

/// An optional date already looked up (a node or an EDGE property).
pub(super) fn de_valeur(v: Option<&Value>, cle: &str, quoi: &str) -> Result<Option<DateTime<Utc>>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(v) => decoder(v)
            .map(Some)
            .map_err(|pourquoi| illisible(quoi, cle, pourquoi)),
    }
}

/// An optional date: absent, null or the empty string is `None`; anything else
/// that is not a date is an error naming `cle` and the variant.
pub(super) fn facultative(get: Lire<'_>, cle: &str, quoi: &str) -> Result<Option<DateTime<Utc>>> {
    de_valeur(get(cle).as_ref(), cle, quoi)
}

/// A required date: absent is an error, and so is anything that is not a date
/// (the empty string included).
pub(super) fn requise(get: Lire<'_>, cle: &str, quoi: &str) -> Result<DateTime<Utc>> {
    match get(cle) {
        None | Some(Value::Null) => Err(anyhow!("{quoi} sans propriété `{cle}`")),
        Some(v) => decoder(&v).map_err(|pourquoi| illisible(quoi, cle, pourquoi)),
    }
}

/// A date whose domain value is not optional and that the Neo4j adapter reads as
/// "now" when the property is ABSENT (`node_to_decision`, the chat sessions and
/// events): that default is kept, for an absent property ONLY. A stored value is
/// read under its three forms, and one that is not a date is an error: what is
/// in the base is never replaced by "now".
pub(super) fn ou_maintenant_si_absente(
    get: Lire<'_>,
    cle: &str,
    quoi: &str,
) -> Result<DateTime<Utc>> {
    Ok(facultative(get, cle, quoi)?.unwrap_or_else(Utc::now))
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
    fn ce_qui_n_est_pas_une_date_nomme_la_propriete_et_la_variante() {
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
        let e = format!("{:#}", facultative(&get, "entier", "projet X").unwrap_err());
        assert!(e.contains("`entier`") && e.contains("INT64"), "{e}");
        let e = format!("{:#}", facultative(&get, "texte", "projet X").unwrap_err());
        assert!(
            e.contains("`texte`") && e.contains("STRING") && e.contains("pas une date"),
            "{e}"
        );
        // Absent and empty: `None` when optional, an error when required.
        assert_eq!(facultative(&get, "absente", "projet X").unwrap(), None);
        assert_eq!(facultative(&get, "vide", "projet X").unwrap(), None);
        let e = format!("{:#}", requise(&get, "absente", "projet X").unwrap_err());
        assert!(e.contains("sans propriété `absente`"), "{e}");
        let e = format!("{:#}", requise(&get, "vide", "projet X").unwrap_err());
        assert!(
            e.contains("`vide` illisible") && e.contains("STRING"),
            "{e}"
        );
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
