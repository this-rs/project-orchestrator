//! The `link` kind: an external web address used as a reference.
//!
//! **Inert by construction.** Nothing in this module can reach the network:
//! it imports `url` (a parser) and `std::net` address types, and nothing that
//! resolves a name or opens a socket. Validating a link, normalizing it,
//! labelling it and showing it never leave the process. That is a property of
//! the code, pinned by a test that reads this very file for the identifiers
//! that would break it (`inert_by_construction`).
//!
//! What the agent does with a link afterwards (reading the page) is a separate
//! act that goes through the existing per-origin consent; it is not here.
//!
//! The id of a `link` reference IS the normalized address, so the same address
//! always names the same reference. A link is refused when:
//!
//! * it is not `http` or `https` (`javascript:`, `data:`, `file:`, `ftp:`...);
//! * it carries credentials (`user:pass@host`);
//! * it is longer than [`MAX_LINK_BYTES`], or contains whitespace or control
//!   characters (the parser would silently strip some of them);
//! * its host is an IP literal that is not public (loopback, private,
//!   link-local, unique-local, CGNAT, unspecified, multicast), or a name that is local by convention (`localhost`,
//!   `*.localhost`, `*.local`, `*.internal`, `*.lan`, `*.home.arpa`, or a bare
//!   single-label name).
//!
//! Known limit, by design: a public-looking NAME that a DNS server answers with
//! a private address cannot be caught without a lookup, and a lookup is a
//! network request. The check at read time (consent per origin) is where that
//! belongs.

use std::net::{Ipv4Addr, Ipv6Addr};

use url::{Host, Url};

/// Longest accepted link, before and after normalization.
pub const MAX_LINK_BYTES: usize = 1024;

/// Why a link was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkError {
    TooLong,
    /// Whitespace or a control character anywhere in the text.
    BadCharacter,
    Unparsable,
    /// Not `http` / `https`.
    Scheme,
    /// `user:pass@` in the address.
    Credentials,
    NoHost,
    /// A host that is the user's own machine or network.
    PrivateHost,
}

fn private_v4(ip: Ipv4Addr) -> bool {
    let [a, b, ..] = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || a == 0
        || (a == 100 && (64..=127).contains(&b)) // CGNAT 100.64.0.0/10
}

fn private_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return private_v4(v4);
    }
    let first = ip.segments()[0];
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (first & 0xffc0) == 0xfe80 // link-local fe80::/10
        || (first & 0xfe00) == 0xfc00 // unique-local fc00::/7
        || (first & 0xffc0) == 0xfec0 // deprecated site-local
        // NAT64 / 6to4 wrappers of a private v4 are not special-cased: they
        // are public-routable prefixes; the v4 inside is the gateway's business.
        || ip.to_ipv4().is_some_and(private_v4) // ::a.b.c.d (deprecated compat)
}

fn local_name(name: &str) -> bool {
    let name = name.trim_end_matches('.');
    !name.contains('.')
        || name == "localhost"
        || [
            ".localhost",
            ".local",
            ".internal",
            ".lan",
            ".home.arpa",
            ".localdomain",
        ]
        .iter()
        .any(|s| name.ends_with(s))
}

/// Parse and normalize a link. The result is what the reference id holds.
pub fn normalize(input: &str) -> Result<String, LinkError> {
    if input.len() > MAX_LINK_BYTES {
        return Err(LinkError::TooLong);
    }
    if input.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(LinkError::BadCharacter);
    }
    let mut url = Url::parse(input).map_err(|_| LinkError::Unparsable)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(LinkError::Scheme);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(LinkError::Credentials);
    }
    match url.host() {
        None => return Err(LinkError::NoHost),
        Some(Host::Ipv4(ip)) if private_v4(ip) => return Err(LinkError::PrivateHost),
        Some(Host::Ipv6(ip)) if private_v6(ip) => return Err(LinkError::PrivateHost),
        Some(Host::Domain(d)) if local_name(d) => return Err(LinkError::PrivateHost),
        Some(_) => {}
    }
    // The fragment is client-side only: it does not change what is designated.
    url.set_fragment(None);
    let out = url.to_string();
    if out.len() > MAX_LINK_BYTES {
        return Err(LinkError::TooLong);
    }
    Ok(out)
}

/// The label of a link: its host, as the address names it. Never fetched.
pub fn host_label(normalized: &str) -> Option<String> {
    let url = Url::parse(normalized).ok()?;
    let host = url.host_str()?.to_string();
    Some(crate::refs::label::truncate(&host))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_public_https_address_is_normalized() {
        assert_eq!(
            normalize("HTTPS://Example.COM:443/a/../b?q=1#frag").unwrap(),
            "https://example.com/b?q=1"
        );
        assert_eq!(
            normalize("http://example.com").unwrap(),
            "http://example.com/"
        );
        assert_eq!(
            normalize("http://example.com:8080/x").unwrap(),
            "http://example.com:8080/x"
        );
        let once = normalize("https://Example.com/%7Euser").unwrap();
        assert_eq!(
            normalize(&once).unwrap(),
            once,
            "normalization is idempotent"
        );
    }

    #[test]
    fn dangerous_schemes_are_refused() {
        for s in [
            "javascript:alert(1)",
            "data:text/html,<script>1</script>",
            "file:///etc/passwd",
            "ftp://example.com/x",
            "vbscript:x",
            "blob:https://example.com/uuid",
            "ws://example.com/",
            "mailto:a@example.com",
            "//example.com/x",
            "example.com/x",
        ] {
            assert!(normalize(s).is_err(), "{s}");
        }
        assert_eq!(normalize("javascript:alert(1)"), Err(LinkError::Scheme));
        assert_eq!(normalize("data:text/html,x"), Err(LinkError::Scheme));
    }

    #[test]
    fn credentials_in_the_address_are_refused() {
        for s in [
            "https://user:pass@example.com/",
            "https://user@example.com/",
            "https://:pass@example.com/",
            "https://example.com@evil.example.org/",
        ] {
            assert_eq!(normalize(s), Err(LinkError::Credentials), "{s}");
        }
    }

    #[test]
    fn private_loopback_and_link_local_addresses_are_refused() {
        for s in [
            "http://127.0.0.1/",
            "http://127.1/",
            "http://2130706433/",
            "http://0x7f.0.0.1/",
            "http://0.0.0.0/",
            "http://10.0.0.5/",
            "http://172.16.0.1/",
            "http://172.31.255.255/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://100.64.0.1/",
            "http://[::1]/",
            "http://[::]/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
            "http://[fd12:3456::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://localhost/",
            "http://LOCALHOST:3000/",
            "http://app.localhost/",
            "http://printer.local/",
            "http://db.internal/",
            "http://nas/",
            "http://localhost./",
        ] {
            assert_eq!(normalize(s), Err(LinkError::PrivateHost), "{s}");
        }
    }

    #[test]
    fn public_addresses_next_to_private_ranges_are_accepted() {
        for s in [
            "http://172.32.0.1/",
            "http://172.15.0.1/",
            "http://100.63.0.1/",
            "http://100.128.0.1/",
            "http://8.8.8.8/",
            "http://[2001:4860:4860::8888]/",
            "https://sub.example.co.uk/path",
        ] {
            assert!(normalize(s).is_ok(), "{s}");
        }
    }

    #[test]
    fn a_link_that_is_too_long_is_refused() {
        let ok = format!("https://example.com/{}", "a".repeat(MAX_LINK_BYTES - 20));
        assert_eq!(ok.len(), MAX_LINK_BYTES);
        assert!(normalize(&ok).is_ok());
        let long = format!("https://example.com/{}", "a".repeat(MAX_LINK_BYTES));
        assert_eq!(normalize(&long), Err(LinkError::TooLong));
    }

    #[test]
    fn whitespace_and_control_characters_are_refused_not_stripped() {
        for s in [
            "https://example.com/a b",
            "https://exa\tmple.com/",
            "https://example.com/\n",
            " https://example.com/",
            "https://example.com/\u{0}",
            "java\nscript:alert(1)",
        ] {
            assert_eq!(normalize(s), Err(LinkError::BadCharacter), "{s:?}");
        }
    }

    #[test]
    fn an_empty_or_hostless_address_is_refused() {
        for s in ["", "https://", "https:///x", "http:"] {
            assert!(normalize(s).is_err(), "{s:?}");
        }
    }

    #[test]
    fn the_label_is_the_host() {
        assert_eq!(
            host_label("https://docs.example.com/a/b?c").as_deref(),
            Some("docs.example.com")
        );
        assert_eq!(host_label("not a url"), None);
    }

    /// Nothing here may reach the network: no socket, no name lookup, no HTTP
    /// client. A guard on the source, because a network call cannot be observed
    /// for a public host from a unit test.
    #[test]
    fn inert_by_construction() {
        let source = include_str!("link.rs");
        let code = source
            .split("#[cfg(test)]")
            .next()
            .expect("source before the tests");
        for forbidden in [
            "reqwest",
            "hyper",
            "TcpStream",
            "UdpSocket",
            "ToSocketAddrs",
            "lookup_host",
            "tokio::net",
            "std::net::TcpListener",
            "getaddrinfo",
            "curl",
            ".send(",
            ".await",
        ] {
            // the module comment names some of these on purpose: only code counts.
            let in_code = code
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .any(|l| l.contains(forbidden));
            assert!(!in_code, "link.rs must stay inert: found `{forbidden}`");
        }
    }
}
