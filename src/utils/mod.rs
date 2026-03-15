//! Shared utility helpers.

pub mod paths;

/// Stable equivalent of the nightly-only `str::floor_char_boundary`.
/// Returns the largest byte index `<= index` that is a valid UTF-8 char boundary.
///
/// Useful for safely truncating a `&str` at a maximum byte length without
/// panicking on multi-byte characters.
pub(crate) fn floor_char_boundary(s: &str, index: usize) -> usize {
    let index = index.min(s.len());
    (0..=index)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(floor_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_floor_char_boundary_multibyte() {
        // "é" is 2 bytes (0xC3 0xA9); boundary at byte 1 is invalid
        let s = "héllo";
        assert_eq!(floor_char_boundary(s, 2), 1); // step back to valid boundary
        assert_eq!(floor_char_boundary(s, 3), 3);
    }

    #[test]
    fn test_floor_char_boundary_beyond_len() {
        let s = "hi";
        assert_eq!(floor_char_boundary(s, 100), 2);
    }
}
