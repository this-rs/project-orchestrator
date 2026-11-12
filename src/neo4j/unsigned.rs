//! Unsigned values read from integer columns.
//!
//! Neo4j integers are signed; a Rust field that is a count, a size or a duration
//! is not. `v as usize` on a negative reads as an astronomical value (`-1` is
//! `usize::MAX`), and -1 is exactly what the graph holds for "none" in several
//! properties. Every read of a count goes through here instead: a negative is
//! the named safe value (`0`, or `None` for an optional), a value too large for
//! the target saturates, nothing wraps.

/// A count / size / duration: negative -> 0, too large for `T` -> `max`.
pub(crate) fn unsigned<T: TryFrom<i64>>(v: i64, max: T) -> T {
    T::try_from(v.max(0)).unwrap_or(max)
}

/// An optional unsigned property: negative (the graph's "none") or too large -> `None`.
pub(crate) fn unsigned_opt<T: TryFrom<i64>>(v: i64) -> Option<T> {
    if v < 0 {
        None
    } else {
        T::try_from(v).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_negative_count_is_zero_never_a_huge_number() {
        assert_eq!(unsigned(-1, usize::MAX), 0);
        assert_eq!(unsigned(i64::MIN, u64::MAX), 0);
        assert_eq!(unsigned(-16, u32::MAX), 0);
        assert_eq!(unsigned(0, usize::MAX), 0);
        assert_eq!(unsigned(42, usize::MAX), 42);
        assert_eq!(unsigned(i64::MAX, u64::MAX), i64::MAX as u64);
    }

    #[test]
    fn a_count_too_large_for_its_type_saturates() {
        assert_eq!(unsigned(300, u8::MAX), 255);
        assert_eq!(unsigned(i64::MAX, u32::MAX), u32::MAX);
        assert_eq!(unsigned(255, u8::MAX), 255);
    }

    #[test]
    fn an_optional_unsigned_is_none_for_a_negative_or_an_overflow() {
        assert_eq!(unsigned_opt::<u64>(-1), None);
        assert_eq!(unsigned_opt::<u8>(-16), None);
        assert_eq!(unsigned_opt::<u8>(256), None);
        assert_eq!(unsigned_opt::<u8>(0), Some(0));
        assert_eq!(unsigned_opt::<u8>(255), Some(255));
        assert_eq!(unsigned_opt::<u64>(1800), Some(1800));
    }
}
