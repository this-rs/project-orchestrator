//! Bolt value → obrain value, with no silent loss.
//!
//! Everything that cannot be represented exactly is either COUNTED in
//! [`Losses`] — and then [`Losses::refusal`] makes the copy FAIL with the cause,
//! except for the one documented narrowing that keeps the exact value (an
//! inexact f32 vector kept as a list) — or returned as an error — never turned
//! into text. Ported from the measured spike (`spike-migrateur/src/conv.rs`), minus
//! its measuring-only oracle.

use neo4rs::BoltType;
use obrain_common::types::{Date, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime};
use std::collections::BTreeMap;
use std::sync::Arc;

/// What the conversion could not keep identically, by cause.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Losses {
    /// A datetime whose nanoseconds are not a multiple of 1 000, truncated to
    /// the microsecond: obrain's `ZonedDatetime` stores microseconds. An
    /// ACCEPTED, documented loss — PO writes its own dates at nanosecond
    /// precision (`to_rfc3339()`), so on a Linux runner almost every node
    /// carries one. It is counted and reported, but it does NOT refuse the copy.
    pub sub_microsecond: u64,
    /// A datetime carrying a zone NAME (`[Europe/Paris]`): obrain keeps the offset only.
    pub named_zone: u64,
    /// A local datetime (no zone): stored as a UTC `Timestamp`.
    pub local_datetime: u64,
    /// Types with no equivalent (point, node, relationship, path): refused.
    pub no_equivalent: u64,
    /// An embedding with an element not representable in f32: kept as a list.
    pub inexact_vector: u64,
}

impl Losses {
    /// The named cause that must fail the copy, `None` when nothing was lost.
    /// A real loss is refused: the NAME of a zone, the "local" nature of a
    /// datetime, a value with no equivalent. Two narrowings are accepted and
    /// only counted/logged: `inexact_vector` loses nothing (the exact floats
    /// are kept, as a list rather than an f32 vector), and `sub_microsecond`
    /// truncates a datetime to the microsecond — obrain stores no finer, and
    /// refusing it would make the real base (PO writes dates at nanosecond
    /// precision) unmigrable.
    pub fn refusal(&self) -> Option<String> {
        let mut causes = Vec::new();
        for (n, what) in [
            (
                self.named_zone,
                "datetime(s) carrying a zone NAME (obrain keeps the offset only)",
            ),
            (
                self.local_datetime,
                "local datetime(s) (obrain stores a UTC instant)",
            ),
            (self.no_equivalent, "value(s) with no equivalent type"),
        ] {
            if n > 0 {
                causes.push(format!("{n} {what}"));
            }
        }
        (!causes.is_empty()).then(|| causes.join("; "))
    }

    pub fn add(&mut self, o: &Losses) {
        self.sub_microsecond += o.sub_microsecond;
        self.named_zone += o.named_zone;
        self.local_datetime += o.local_datetime;
        self.no_equivalent += o.no_equivalent;
        self.inexact_vector += o.inexact_vector;
    }
}

/// Properties that are index vectors in obrain (f32), not lists of floats.
pub const VECTOR_KEYS: &[&str] = &["embedding", "context_embedding", "query_embedding"];

/// A property read from Neo4j, converted for the bulk import. A list of floats
/// under a vector key becomes `Value::Vector` (f32) ONLY when exact for every
/// element; otherwise it stays a list and is counted.
pub fn import_value(key: &str, b: &BoltType, losses: &mut Losses) -> Result<Value, String> {
    if let (true, BoltType::List(l)) = (VECTOR_KEYS.contains(&key), b) {
        let floats: Option<Vec<f64>> = l
            .value
            .iter()
            .map(|e| match e {
                BoltType::Float(x) => Some(x.value),
                _ => None,
            })
            .collect();
        if let Some(xs) = floats.filter(|xs| !xs.is_empty()) {
            let f32s: Vec<f32> = xs.iter().map(|x| *x as f32).collect();
            if xs.iter().zip(&f32s).all(|(x, f)| f64::from(*f) == *x) {
                return Ok(Value::Vector(Arc::from(f32s)));
            }
            losses.inexact_vector += 1;
        }
    }
    bolt_to_value(b, losses)
}

/// neo4rs 0.8 hides a duration's months and days (private fields; the public
/// conversion flattens them). Its `Debug` rendering is the only lossless access.
fn duration_from_debug(d: &neo4rs::BoltDuration) -> Option<Duration> {
    let text = format!("{d:?}");
    let mut numbers = Vec::new();
    for piece in text.split("value: ").skip(1) {
        let end = piece
            .find(|c: char| !(c.is_ascii_digit() || c == '-'))
            .unwrap_or(piece.len());
        numbers.push(piece[..end].parse::<i64>().ok()?);
    }
    if numbers.len() != 4 {
        return None;
    }
    let (months, days, seconds, nanos) = (numbers[0], numbers[1], numbers[2], numbers[3]);
    Some(Duration::new(
        months,
        days,
        seconds.checked_mul(1_000_000_000)?.checked_add(nanos)?,
    ))
}

pub fn bolt_to_value(b: &BoltType, losses: &mut Losses) -> Result<Value, String> {
    Ok(match b {
        BoltType::Null(_) => Value::Null,
        BoltType::Boolean(x) => Value::Bool(x.value),
        BoltType::Integer(x) => Value::Int64(x.value),
        BoltType::Float(x) => Value::Float64(x.value),
        BoltType::String(x) => Value::String(x.value.as_str().into()),
        BoltType::Bytes(x) => Value::Bytes(Arc::from(&x.value[..])),
        BoltType::List(l) => {
            let mut v = Vec::with_capacity(l.value.len());
            for e in &l.value {
                v.push(bolt_to_value(e, losses)?);
            }
            Value::List(v.into())
        }
        BoltType::Map(m) => {
            let mut bm = BTreeMap::new();
            for (k, e) in &m.value {
                bm.insert(
                    PropertyKey::new(k.value.as_str()),
                    bolt_to_value(e, losses)?,
                );
            }
            Value::Map(Arc::new(bm))
        }
        BoltType::DateTime(d) => {
            let c: chrono::DateTime<chrono::FixedOffset> = d
                .try_into()
                .map_err(|e| format!("unreadable datetime: {e:?}"))?;
            zoned(c, losses)?
        }
        BoltType::DateTimeZoneId(d) => {
            let c: chrono::DateTime<chrono::FixedOffset> = d
                .try_into()
                .map_err(|e| format!("unreadable datetime with a zone name: {e:?}"))?;
            losses.named_zone += 1;
            zoned(c, losses)?
        }
        BoltType::LocalDateTime(d) => {
            let c: chrono::NaiveDateTime = d
                .try_into()
                .map_err(|e| format!("unreadable local datetime: {e:?}"))?;
            losses.local_datetime += 1;
            let n = c
                .and_utc()
                .timestamp_nanos_opt()
                .ok_or("local datetime outside the i64 nanosecond range")?;
            Value::Timestamp(Timestamp::from_nanos(n))
        }
        BoltType::Date(d) => {
            let c: chrono::NaiveDate = d
                .try_into()
                .map_err(|e| format!("unreadable date: {e:?}"))?;
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let days = c.signed_duration_since(epoch).num_days();
            Value::Date(Date::from_days(
                i32::try_from(days).map_err(|_| "date outside the i32 range".to_string())?,
            ))
        }
        BoltType::Time(t) => {
            use chrono::Timelike;
            let (h, offset): (chrono::NaiveTime, chrono::FixedOffset) = t.into();
            let n = h.num_seconds_from_midnight() as u64 * 1_000_000_000 + h.nanosecond() as u64;
            Value::Time(
                Time::from_nanos(n)
                    .ok_or("time outside the day")?
                    .with_offset(offset.local_minus_utc()),
            )
        }
        BoltType::LocalTime(t) => {
            use chrono::Timelike;
            let h: chrono::NaiveTime = t.into();
            let n = h.num_seconds_from_midnight() as u64 * 1_000_000_000 + h.nanosecond() as u64;
            Value::Time(Time::from_nanos(n).ok_or("time outside the day")?)
        }
        BoltType::Duration(d) => Value::Duration(
            duration_from_debug(d).ok_or("unreadable duration (unexpected Debug rendering)")?,
        ),
        BoltType::Point2D(_) | BoltType::Point3D(_) => {
            losses.no_equivalent += 1;
            return Err("point: no equivalent type in obrain".into());
        }
        BoltType::Node(_)
        | BoltType::Relation(_)
        | BoltType::UnboundedRelation(_)
        | BoltType::Path(_) => {
            losses.no_equivalent += 1;
            return Err("node/relationship/path: not a property value".into());
        }
    })
}

/// The sub-microsecond remainder (0..=999 ns) that [`zoned`] drops: obrain's
/// `Timestamp` / `ZonedDatetime` hold microseconds only (`obrain-common`
/// `zoned_datetime.rs`), so no native representation keeps it. The reversible
/// option is a companion integer property (`<key>__nanos`) next to the
/// microsecond value: `micros * 1000 + remainder` is the source instant.
///
/// NOT wired into the production copy: it adds a property PO can see, which is
/// a decision (see docs/migration/GATES.md, G7). It exists so the round trip is
/// demonstrated, in a test, before that decision.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn sub_micro_remainder(c: chrono::DateTime<chrono::FixedOffset>) -> Option<i64> {
    Some(c.timestamp_nanos_opt()?.rem_euclid(1000))
}

pub(super) fn zoned(
    c: chrono::DateTime<chrono::FixedOffset>,
    losses: &mut Losses,
) -> Result<Value, String> {
    let n = c
        .timestamp_nanos_opt()
        .ok_or("datetime outside the i64 nanosecond range")?;
    if n % 1000 != 0 {
        losses.sub_microsecond += 1;
    }
    Ok(Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
        Timestamp::from_nanos(n),
        c.offset().local_minus_utc(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo4rs::{BoltFloat, BoltInteger, BoltList, BoltString};

    #[test]
    fn only_real_losses_refuse_the_copy() {
        assert_eq!(Losses::default().refusal(), None);
        let inexact = Losses {
            inexact_vector: 3,
            ..Default::default()
        };
        assert_eq!(inexact.refusal(), None, "an exact list loses nothing");
        // `sub_microsecond` is the ONE accepted truncation (to the microsecond):
        // counted and reported, but it does NOT refuse — PO writes its own dates
        // at nanosecond precision, and refusing would make the real base
        // unmigrable.
        assert_eq!(
            Losses {
                sub_microsecond: 7,
                ..Default::default()
            }
            .refusal(),
            None,
            "microsecond truncation is accepted"
        );
        for l in [
            Losses {
                named_zone: 1,
                ..Default::default()
            },
            Losses {
                local_datetime: 1,
                ..Default::default()
            },
            Losses {
                no_equivalent: 1,
                ..Default::default()
            },
        ] {
            assert!(l.refusal().is_some(), "{l:?}");
        }
    }

    #[test]
    fn a_nanosecond_datetime_is_truncated_to_the_microsecond_and_not_refused() {
        // PO writes its own dates at nanosecond precision (`to_rfc3339()`); on a
        // Linux runner `Utc::now()` is nanosecond, so the real base carries
        // sub-microsecond datetimes. The copy must truncate, count, and NOT
        // refuse — otherwise the real base is unmigrable.
        let dt =
            chrono::DateTime::parse_from_rfc3339("2026-11-20T10:00:00.123456789+01:00").unwrap();
        let mut l = Losses::default();
        let v = zoned(dt, &mut l).unwrap();
        assert_eq!(l.refusal(), None, "sub-microsecond is an accepted loss");
        assert_eq!(l.sub_microsecond, 1, "it is still counted");
        match v {
            Value::ZonedDatetime(z) => {
                let nanos = dt.timestamp_nanos_opt().unwrap();
                assert_eq!(z.as_timestamp().as_nanos(), nanos - (nanos % 1000));
                assert_eq!(z.offset_seconds(), 3600);
            }
            other => panic!("expected a ZonedDatetime, got {other:?}"),
        }
    }

    fn floats(xs: &[f64]) -> BoltType {
        BoltType::List(BoltList {
            value: xs
                .iter()
                .map(|x| BoltType::Float(BoltFloat::new(*x)))
                .collect(),
        })
    }

    #[test]
    fn negative_integers_keep_their_sign() {
        // The neo4rs 0.8.0 defect (vendor/neo4rs/PATCH.md) would read -1 as 255.
        let mut l = Losses::default();
        let v = bolt_to_value(&BoltType::Integer(BoltInteger::new(-1)), &mut l).unwrap();
        assert!(matches!(v, Value::Int64(-1)));
    }

    #[test]
    fn an_exact_f32_list_under_a_vector_key_becomes_a_vector() {
        let mut l = Losses::default();
        let v = import_value("embedding", &floats(&[0.5, -1.25]), &mut l).unwrap();
        assert!(matches!(v, Value::Vector(_)));
        assert_eq!(l, Losses::default());
    }

    #[test]
    fn an_inexact_list_stays_a_list_and_is_counted() {
        let mut l = Losses::default();
        let v = import_value("embedding", &floats(&[0.1]), &mut l).unwrap();
        assert!(matches!(v, Value::List(_)), "0.1 is not an exact f32");
        assert_eq!(l.inexact_vector, 1);
    }

    #[test]
    fn the_same_list_under_another_key_is_a_list() {
        let mut l = Losses::default();
        let v = import_value("scores", &floats(&[0.5]), &mut l).unwrap();
        assert!(matches!(v, Value::List(_)));
    }

    #[test]
    fn strings_and_nulls_round_trip() {
        let mut l = Losses::default();
        let s = BoltType::String(BoltString::new("é"));
        assert!(
            matches!(bolt_to_value(&s, &mut l).unwrap(), Value::String(x) if x.as_str() == "é")
        );
    }
}
