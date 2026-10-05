//! Endpoint guard: SSRF and TLS rules for provider base URLs (decision A36,
//! task B29 / review item B23).
//!
//! A provider instance points at an HTTP endpoint chosen by an administrator
//! or, later, by a user. Left unchecked, that endpoint can target the host's
//! own services (`http://127.0.0.1:7474`), a cloud metadata service
//! (`http://169.254.169.254/latest/meta-data/`) or any machine on the private
//! network, and the `status`/`models` probes would act as a server-side
//! request forgery. This module decides, with no network of its own, whether
//! an endpoint may be contacted:
//!
//! * [`validate_url`] — syntax, scheme (`https` everywhere, `http` only
//!   towards an explicit loopback literal), no embedded credentials, a host.
//! * [`classify_ip`] — the non-public address families, including the four
//!   known cloud metadata addresses. IPv4 addresses mapped into IPv6
//!   (`::ffff:a.b.c.d`) are classified as their IPv4.
//! * [`validate_resolved`] — EVERY resolved address must be public (a single
//!   private answer refuses: that is DNS rebinding), with two administrator
//!   exceptions ([`EndpointPolicy::allowed_private_hosts`],
//!   [`EndpointPolicy::allow_private_ranges`]). A loopback answer is accepted
//!   only when the URL host is itself a loopback literal.
//! * [`validate_endpoint_with`] / [`validate_endpoint`] — the whole check, with
//!   an injected or a real (blocking, `ToSocketAddrs`) resolver. Called when
//!   an instance is created or edited AND before every connection.
//! * [`http_client_builder_for_endpoints`] — a `reqwest` builder with
//!   redirects disabled and a bounded connect timeout; a 3xx answer is then
//!   refused through [`refuse_redirect`].
//! * [`redact_error_body`] — what may be kept of an upstream error body:
//!   truncated and with anything that looks like a credential replaced.
//!
//! Refusals never carry the URL. Their [`fmt::Display`] names the rule only;
//! [`EndpointRefusal::message_for`] prefixes the normalised origin (scheme,
//! host, non-default port — never the path or the query), the same origin the
//! user consent is tied to (decision A28, [`origin_of`]).

use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
// `url` is not a direct dependency of this crate; `reqwest::Url` is the very
// same `url::Url` type, re-exported.
use reqwest::Url;

/// Upper bound on the connect phase of any request towards an endpoint.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// What an administrator may relax. The default is the strict posture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointPolicy {
    /// Allow `http://` (no TLS) — only towards the local loopback
    /// (`localhost`, `127.0.0.0/8`, `::1`). Local inference servers (ollama,
    /// llama-server, vLLM) rarely speak TLS, hence `true` by default.
    pub allow_insecure_loopback: bool,
    /// Exact host names or IP literals the administrator vouches for, as they
    /// appear in the URL (case-insensitive, trailing dot and IPv6 brackets
    /// ignored). Such a host may resolve to a private, link-local, loopback,
    /// metadata or reserved address. It never lifts the scheme rules.
    pub allowed_private_hosts: Vec<String>,
    /// Accept endpoints resolving to RFC 1918 / unique-local / link-local
    /// addresses (an on-premises deployment). Does NOT lift loopback,
    /// metadata, reserved, unspecified, multicast or broadcast refusals.
    pub allow_private_ranges: bool,
}

impl Default for EndpointPolicy {
    fn default() -> Self {
        Self {
            allow_insecure_loopback: true,
            allowed_private_hosts: Vec::new(),
            allow_private_ranges: false,
        }
    }
}

/// Which non-public family an address belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrivateKind {
    /// `127.0.0.0/8`, `::1`.
    Loopback,
    /// `169.254.0.0/16`, `fe80::/10`.
    LinkLocal,
    /// Cloud metadata services (AWS/GCP/Azure `169.254.169.254`, AWS IPv6
    /// `fd00:ec2::254`, Alibaba `100.100.100.200`, Oracle `192.0.0.192`).
    Metadata,
    /// `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fc00::/7`.
    Private,
    /// `0.0.0.0`, `::`.
    Unspecified,
    /// `224.0.0.0/4`, `ff00::/8`.
    Multicast,
    /// `255.255.255.255`.
    Broadcast,
    /// Everything else that is not routable on the public Internet:
    /// `240.0.0.0/4`, `0.0.0.0/8`, documentation ranges, `100.64.0.0/10`
    /// (shared address space), `192.0.0.0/24`, `198.18.0.0/15`, `2001:db8::/32`,
    /// `2001:2::/48`, deprecated site-local and IPv4-compatible IPv6, `100::/64`.
    Reserved,
}

impl PrivateKind {
    /// Short stable label.
    pub fn label(self) -> &'static str {
        match self {
            PrivateKind::Loopback => "loopback",
            PrivateKind::LinkLocal => "link-local",
            PrivateKind::Metadata => "metadata",
            PrivateKind::Private => "private",
            PrivateKind::Unspecified => "unspecified",
            PrivateKind::Multicast => "multicast",
            PrivateKind::Broadcast => "broadcast",
            PrivateKind::Reserved => "reserved",
        }
    }
}

impl fmt::Display for PrivateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Why an endpoint is refused. Carries no URL, path, query or credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointRefusal {
    /// The string is not a URL.
    InvalidUrl,
    /// A scheme other than `https` (or `http` towards loopback when allowed).
    SchemeNotAllowed {
        /// The scheme found, lowercased by the parser.
        scheme: String,
    },
    /// `http://` towards a host that is not a loopback literal.
    HttpOutsideLoopback,
    /// `user:password@` in the URL.
    CredentialsInUrl,
    /// No host component.
    HostMissing,
    /// The host (or one of its resolved addresses) is not public.
    PrivateAddress {
        /// The family the address belongs to.
        kind: PrivateKind,
    },
    /// DNS resolution failed or returned nothing.
    UnresolvableHost,
    /// The endpoint answered with a redirect; redirects are never followed.
    RedirectsNotAllowed,
}

impl EndpointRefusal {
    /// Stable code, for the API and the frontend error cards.
    pub fn code(&self) -> &'static str {
        match self {
            EndpointRefusal::InvalidUrl => "endpoint_invalid_url",
            EndpointRefusal::SchemeNotAllowed { .. } => "endpoint_scheme_not_allowed",
            EndpointRefusal::HttpOutsideLoopback => "endpoint_http_outside_loopback",
            EndpointRefusal::CredentialsInUrl => "endpoint_credentials_in_url",
            EndpointRefusal::HostMissing => "endpoint_host_missing",
            EndpointRefusal::PrivateAddress { .. } => "endpoint_private_address",
            EndpointRefusal::UnresolvableHost => "endpoint_unresolvable",
            EndpointRefusal::RedirectsNotAllowed => "endpoint_redirects_not_allowed",
        }
    }

    /// The refusal prefixed with the normalised origin of `url` (scheme, host,
    /// non-default port) — never its path, query or credentials. An
    /// unparseable `url` is named `<invalid url>`.
    pub fn message_for(&self, url: &str) -> String {
        let origin = origin_of(url).unwrap_or_else(|| "<invalid url>".to_string());
        format!("{origin}: {self}")
    }
}

impl fmt::Display for EndpointRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EndpointRefusal::InvalidUrl => f.write_str("the endpoint URL is not valid"),
            EndpointRefusal::SchemeNotAllowed { scheme } => {
                write!(f, "scheme `{scheme}` is not allowed (https is required)")
            }
            EndpointRefusal::HttpOutsideLoopback => {
                f.write_str("plain http is only allowed towards the local loopback")
            }
            EndpointRefusal::CredentialsInUrl => {
                f.write_str("credentials must not be embedded in the endpoint URL")
            }
            EndpointRefusal::HostMissing => f.write_str("the endpoint URL has no host"),
            EndpointRefusal::PrivateAddress { kind } => {
                write!(f, "the endpoint resolves to a non-public address ({kind})")
            }
            EndpointRefusal::UnresolvableHost => {
                f.write_str("the endpoint host cannot be resolved")
            }
            EndpointRefusal::RedirectsNotAllowed => {
                f.write_str("the endpoint answered with a redirect, which is not followed")
            }
        }
    }
}

impl std::error::Error for EndpointRefusal {}

// ---------------------------------------------------------------------------
// Origin
// ---------------------------------------------------------------------------

/// Normalised origin of a URL: `scheme://host[:port]`, lowercased, the port
/// written only when it is not the scheme's default. `None` when the string
/// is not a URL with a host. This is the identity the user consent is tied to
/// (decision A28): `HTTPS://Api.Example.com:443/v1/` → `https://api.example.com`.
pub fn origin_of(url: &str) -> Option<String> {
    let parsed = Url::parse(url.trim()).ok()?;
    let host = parsed.host_str()?;
    if host.is_empty() {
        return None;
    }
    let mut origin = format!(
        "{}://{}",
        parsed.scheme().to_ascii_lowercase(),
        host.to_ascii_lowercase()
    );
    // `Url::port` is `None` when the port equals the scheme's known default.
    if let Some(port) = parsed.port() {
        origin.push(':');
        origin.push_str(&port.to_string());
    }
    Some(origin)
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/// The host of a URL, as a literal address or a name.
#[derive(Debug, Clone, PartialEq, Eq)]
enum HostKind {
    /// An IPv4 or IPv6 literal (brackets removed).
    Ip(IpAddr),
    /// A DNS name, lowercased, trailing dot removed.
    Domain(String),
}

/// Lowercase, drop a trailing dot and IPv6 brackets.
fn normalize_host(host: &str) -> String {
    let trimmed = host.trim().trim_end_matches('.');
    let unbracketed = trimmed
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(trimmed);
    unbracketed.to_ascii_lowercase()
}

/// Classify the host of an already-parsed URL.
fn host_of(url: &Url) -> Result<HostKind, EndpointRefusal> {
    let raw = url.host_str().ok_or(EndpointRefusal::HostMissing)?;
    let normalized = normalize_host(raw);
    if normalized.is_empty() {
        return Err(EndpointRefusal::HostMissing);
    }
    if let Ok(ip) = normalized.parse::<IpAddr>() {
        return Ok(HostKind::Ip(ip));
    }
    Ok(HostKind::Domain(normalized))
}

/// Whether the URL host is, by itself, an explicit loopback: `localhost`,
/// `*.localhost` (RFC 6761), `127.0.0.0/8` or `::1` (also IPv4-mapped).
fn is_loopback_literal(host: &HostKind) -> bool {
    match host {
        HostKind::Ip(ip) => classify_ip(*ip) == Some(PrivateKind::Loopback),
        HostKind::Domain(name) => name == "localhost" || name.ends_with(".localhost"),
    }
}

/// Whether the parse failure is best explained by an empty authority
/// (`https://`, `https:///v1`, `https://user:pw@`, `https://:443`).
fn authority_is_empty(raw: &str) -> bool {
    let Some(idx) = raw.find("://") else {
        return false;
    };
    let rest = &raw[idx + 3..];
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, host_port)| host_port)
        .unwrap_or(authority);
    host_port.is_empty() || host_port.starts_with(':')
}

// ---------------------------------------------------------------------------
// URL rules
// ---------------------------------------------------------------------------

/// Static checks on the URL string: syntax, scheme, embedded credentials,
/// host presence, and the `http`-only-towards-loopback rule. No resolution.
pub fn validate_url(url: &str, policy: &EndpointPolicy) -> Result<Url, EndpointRefusal> {
    let raw = url.trim();
    // Checked on the raw text: the WHATWG parser swallows extra slashes of a
    // special scheme, so `https:///v1` would otherwise become the host `v1`.
    let lower = raw.to_ascii_lowercase();
    if (lower.starts_with("http:") || lower.starts_with("https:")) && authority_is_empty(raw) {
        return Err(EndpointRefusal::HostMissing);
    }
    let parsed = Url::parse(raw).map_err(|_| EndpointRefusal::InvalidUrl)?;

    let scheme = parsed.scheme().to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(EndpointRefusal::SchemeNotAllowed { scheme });
    }

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(EndpointRefusal::CredentialsInUrl);
    }

    let host = host_of(&parsed)?;

    if scheme == "http" {
        if !is_loopback_literal(&host) {
            return Err(EndpointRefusal::HttpOutsideLoopback);
        }
        if !policy.allow_insecure_loopback {
            return Err(EndpointRefusal::SchemeNotAllowed { scheme });
        }
    }

    Ok(parsed)
}

// ---------------------------------------------------------------------------
// Address classification
// ---------------------------------------------------------------------------

/// The four cloud metadata addresses reachable over IPv4.
const METADATA_V4: [Ipv4Addr; 3] = [
    Ipv4Addr::new(169, 254, 169, 254), // AWS, GCP, Azure, OpenStack
    Ipv4Addr::new(100, 100, 100, 200), // Alibaba Cloud
    Ipv4Addr::new(192, 0, 0, 192),     // Oracle Cloud
];

/// AWS IMDS over IPv6.
const METADATA_V6: Ipv6Addr = Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x254);

fn classify_ipv4(ip: Ipv4Addr) -> Option<PrivateKind> {
    let [a, b, c, _] = ip.octets();
    if ip.is_unspecified() {
        return Some(PrivateKind::Unspecified);
    }
    if ip.is_broadcast() {
        return Some(PrivateKind::Broadcast);
    }
    if METADATA_V4.contains(&ip) {
        return Some(PrivateKind::Metadata);
    }
    if a == 127 {
        return Some(PrivateKind::Loopback);
    }
    if a == 169 && b == 254 {
        return Some(PrivateKind::LinkLocal);
    }
    if a == 10 || (a == 172 && (16..=31).contains(&b)) || (a == 192 && b == 168) {
        return Some(PrivateKind::Private);
    }
    if (224..=239).contains(&a) {
        return Some(PrivateKind::Multicast);
    }
    let reserved = a >= 240 // 240.0.0.0/4 (broadcast handled above)
        || a == 0 // 0.0.0.0/8 "this network"
        || (a == 192 && b == 0 && c == 2) // 192.0.2.0/24 TEST-NET-1
        || (a == 198 && b == 51 && c == 100) // 198.51.100.0/24 TEST-NET-2
        || (a == 203 && b == 0 && c == 113) // 203.0.113.0/24 TEST-NET-3
        || (a == 100 && (64..=127).contains(&b)) // 100.64.0.0/10 shared address space
        || (a == 192 && b == 0 && c == 0) // 192.0.0.0/24 IETF protocol assignments
        || (a == 198 && (b == 18 || b == 19)); // 198.18.0.0/15 benchmarking
    if reserved {
        return Some(PrivateKind::Reserved);
    }
    None
}

/// IPv4 address embedded in the low 32 bits of an IPv6 address.
fn embedded_low_ipv4(segments: [u16; 8]) -> Ipv4Addr {
    let hi = segments[6].to_be_bytes();
    let lo = segments[7].to_be_bytes();
    Ipv4Addr::new(hi[0], hi[1], lo[0], lo[1])
}

fn classify_ipv6(ip: Ipv6Addr) -> Option<PrivateKind> {
    // `::ffff:a.b.c.d` — classified as the IPv4 it carries. (Not `to_ipv4`,
    // which would also turn `::1` into `0.0.0.1`.)
    if let Some(mapped) = ip.to_ipv4_mapped() {
        return classify_ipv4(mapped);
    }
    if ip.is_unspecified() {
        return Some(PrivateKind::Unspecified);
    }
    if ip.is_loopback() {
        return Some(PrivateKind::Loopback);
    }
    if ip == METADATA_V6 {
        return Some(PrivateKind::Metadata);
    }
    let segments = ip.segments();
    let first = segments[0];
    if (first & 0xffc0) == 0xfe80 {
        return Some(PrivateKind::LinkLocal);
    }
    if (first & 0xfe00) == 0xfc00 {
        return Some(PrivateKind::Private);
    }
    if (first & 0xff00) == 0xff00 {
        return Some(PrivateKind::Multicast);
    }
    let reserved = (first == 0x2001 && segments[1] == 0x0db8) // 2001:db8::/32 documentation
        || (first == 0x2001 && segments[1] == 0x0002 && segments[2] == 0) // 2001:2::/48 benchmarking
        || (first & 0xffc0) == 0xfec0 // fec0::/10 deprecated site-local
        || (first == 0x0100 && segments[1..4] == [0, 0, 0]) // 100::/64 discard-only
        || segments[..6] == [0, 0, 0, 0, 0, 0]; // ::a.b.c.d deprecated IPv4-compatible (:: and ::1 handled above)
    if reserved {
        return Some(PrivateKind::Reserved);
    }
    // Addresses that carry a public IPv4 inside are as public as that IPv4.
    if segments[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        // 64:ff9b::/96 NAT64 well-known prefix
        return classify_ipv4(embedded_low_ipv4(segments));
    }
    if first == 0x2002 {
        // 2002::/16 6to4: IPv4 in the next 32 bits
        let hi = segments[1].to_be_bytes();
        let lo = segments[2].to_be_bytes();
        return classify_ipv4(Ipv4Addr::new(hi[0], hi[1], lo[0], lo[1]));
    }
    None
}

/// `None` when the address is public (routable on the Internet), otherwise
/// the non-public family it belongs to. IPv4 mapped into IPv6 is classified
/// as its IPv4.
pub fn classify_ip(ip: IpAddr) -> Option<PrivateKind> {
    match ip {
        IpAddr::V4(v4) => classify_ipv4(v4),
        IpAddr::V6(v6) => classify_ipv6(v6),
    }
}

// ---------------------------------------------------------------------------
// Resolved-address rules
// ---------------------------------------------------------------------------

/// Whether the URL host is one of the administrator's exact exceptions.
fn host_is_exempt(host: &HostKind, policy: &EndpointPolicy) -> bool {
    let url_host = match host {
        HostKind::Ip(ip) => ip.to_string(),
        HostKind::Domain(name) => name.clone(),
    };
    policy.allowed_private_hosts.iter().any(|allowed| {
        let allowed = normalize_host(allowed);
        if allowed.is_empty() {
            return false;
        }
        match allowed.parse::<IpAddr>() {
            // Compare addresses, not spellings (`::FFFF:10.0.0.5` vs `::ffff:a00:5`).
            Ok(allowed_ip) => matches!(host, HostKind::Ip(ip) if *ip == allowed_ip),
            Err(_) => allowed == url_host,
        }
    })
}

/// Check every resolved address of `url`'s host. All must be public; one
/// non-public answer refuses (DNS rebinding returns a mix on purpose).
///
/// * Loopback is accepted only when the URL host is itself a loopback literal
///   (`localhost`, `127.x`, `::1`) or an exact administrator exception. A
///   public name resolving to `127.0.0.1` is refused, even with
///   `allow_private_ranges`.
/// * Private and link-local are accepted with `allow_private_ranges` or an
///   exact exception.
/// * Metadata and reserved are accepted only with an exact exception.
/// * Unspecified, multicast and broadcast are never accepted.
/// * An empty slice is [`EndpointRefusal::UnresolvableHost`].
pub fn validate_resolved(
    url: &Url,
    addrs: &[IpAddr],
    policy: &EndpointPolicy,
) -> Result<(), EndpointRefusal> {
    if addrs.is_empty() {
        return Err(EndpointRefusal::UnresolvableHost);
    }
    let host = host_of(url)?;
    let exempt = host_is_exempt(&host, policy);
    let loopback_literal = is_loopback_literal(&host);

    for ip in addrs {
        let Some(kind) = classify_ip(*ip) else {
            continue;
        };
        let allowed = match kind {
            PrivateKind::Unspecified | PrivateKind::Multicast | PrivateKind::Broadcast => false,
            PrivateKind::Loopback => loopback_literal || exempt,
            PrivateKind::Private | PrivateKind::LinkLocal => exempt || policy.allow_private_ranges,
            PrivateKind::Metadata | PrivateKind::Reserved => exempt,
        };
        if !allowed {
            return Err(EndpointRefusal::PrivateAddress { kind });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Whole check
// ---------------------------------------------------------------------------

/// A resolver: host name and port in, addresses out. Injected so the tests
/// run without network.
pub type Resolver<'a> = &'a dyn Fn(&str, u16) -> io::Result<Vec<IpAddr>>;

/// [`validate_url`], then resolution of a named host through `resolve`, then
/// [`validate_resolved`]. A host that is already an IP literal is not
/// resolved. A resolver error is [`EndpointRefusal::UnresolvableHost`].
pub fn validate_endpoint_with(
    url: &str,
    policy: &EndpointPolicy,
    resolve: Resolver<'_>,
) -> Result<Url, EndpointRefusal> {
    let parsed = validate_url(url, policy)?;
    let addrs = match host_of(&parsed)? {
        HostKind::Ip(ip) => vec![ip],
        HostKind::Domain(name) => {
            let port = parsed.port_or_known_default().unwrap_or(443);
            resolve(&name, port).map_err(|_| EndpointRefusal::UnresolvableHost)?
        }
    };
    validate_resolved(&parsed, &addrs, policy)?;
    Ok(parsed)
}

/// Real resolution through the system resolver (`ToSocketAddrs`), off the
/// async runtime. Called before every connection, so a name that started to
/// answer a private address is caught at the next attempt.
pub async fn validate_endpoint(url: &str, policy: &EndpointPolicy) -> Result<Url, EndpointRefusal> {
    let url = url.to_string();
    let policy = policy.clone();
    tokio::task::spawn_blocking(move || {
        let resolve = |host: &str, port: u16| -> io::Result<Vec<IpAddr>> {
            (host, port)
                .to_socket_addrs()
                .map(|addrs| addrs.map(|addr| addr.ip()).collect())
        };
        validate_endpoint_with(&url, &policy, &resolve)
    })
    .await
    // A panicked or cancelled blocking task is a failed check, not a pass.
    .unwrap_or(Err(EndpointRefusal::UnresolvableHost))
}

// ---------------------------------------------------------------------------
// HTTP client posture
// ---------------------------------------------------------------------------

/// The `reqwest` builder every endpoint probe and chat connection must start
/// from: redirects are never followed (a 302 towards an internal address is
/// the classic bypass — the 3xx answer comes back and [`refuse_redirect`]
/// turns it into a refusal), the connect phase is bounded by
/// [`CONNECT_TIMEOUT`], and no option accepting invalid certificates is set.
/// An additional CA, when the administrator provides one, is added by the
/// caller with `add_root_certificate`.
pub fn http_client_builder_for_endpoints() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
}

/// With redirects disabled, a 3xx status reaches the caller as a normal
/// response: refuse it instead of looking at `Location`.
pub fn refuse_redirect(status: u16) -> Result<(), EndpointRefusal> {
    if (300..400).contains(&status) {
        Err(EndpointRefusal::RedirectsNotAllowed)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error body redaction
// ---------------------------------------------------------------------------

/// Replacement for anything that looks like a credential.
pub const REDACTED: &str = "[redacted]";

/// Patterns replaced by [`REDACTED`], applied in order. Written to err on the
/// side of hiding too much: an error body is diagnostic, never load-bearing.
static REDACTIONS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        // `Authorization: Bearer xxx`, `authorization=...`
        r#"(?i)\bauthorization\s*[:=]\s*[^\r\n"',;]+"#,
        // `Bearer xxx`
        r#"(?i)\bbearer\s+[^\s"',;]+"#,
        // `key=...`, `token: ...`, `"api_key": "..."`, `password=...`
        r#"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|secret|token|key|password|passwd|pwd)\b["']?\s*[=:]\s*["']?[^\s"'&,;]+"#,
        // OpenAI / Anthropic / Stripe style prefixes: `sk-...`, `sk_live_...`
        r"\bsk[-_][A-Za-z0-9_-]{8,}",
        // Long hexadecimal runs (digests, hex-encoded tokens)
        r"\b[A-Fa-f0-9]{24,}\b",
        // Long base64 / url-safe base64 runs
        r"[A-Za-z0-9+/_-]{24,}={0,2}",
    ]
    .into_iter()
    .map(|pattern| Regex::new(pattern).expect("redaction pattern is valid"))
    .collect()
});

/// Longest prefix of `s` that is at most `max` bytes and ends on a character
/// boundary.
fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// What may be kept of an upstream error body: credentials-looking fragments
/// replaced by [`REDACTED`] (on the full body, so a secret cut in two by the
/// truncation is still caught), then truncated to at most `max` bytes on a
/// UTF-8 boundary. The result never exceeds `max` bytes.
pub fn redact_error_body(body: &str, max: usize) -> String {
    let mut redacted = body.to_string();
    for pattern in REDACTIONS.iter() {
        if pattern.is_match(&redacted) {
            redacted = pattern.replace_all(&redacted, REDACTED).into_owned();
        }
    }
    truncate_utf8(&redacted, max).to_string()
}

// ---------------------------------------------------------------------------
// Tests (no network)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    fn v6(s: &str) -> IpAddr {
        IpAddr::V6(s.parse::<Ipv6Addr>().unwrap())
    }

    fn strict() -> EndpointPolicy {
        EndpointPolicy::default()
    }

    /// A resolver answering `addrs` for any name.
    fn resolving(addrs: Vec<IpAddr>) -> impl Fn(&str, u16) -> io::Result<Vec<IpAddr>> {
        move |_host, _port| Ok(addrs.clone())
    }

    fn public_v4() -> IpAddr {
        v4(93, 184, 216, 34)
    }

    // --- classify_ip: one test per family -------------------------------

    #[test]
    fn classify_loopback_v4_and_v6() {
        assert_eq!(classify_ip(v4(127, 0, 0, 1)), Some(PrivateKind::Loopback));
        assert_eq!(classify_ip(v4(127, 200, 3, 4)), Some(PrivateKind::Loopback));
        assert_eq!(classify_ip(v6("::1")), Some(PrivateKind::Loopback));
        assert_eq!(
            classify_ip(v6("::ffff:127.0.0.1")),
            Some(PrivateKind::Loopback)
        );
    }

    #[test]
    fn classify_link_local() {
        assert_eq!(
            classify_ip(v4(169, 254, 1, 1)),
            Some(PrivateKind::LinkLocal)
        );
        assert_eq!(classify_ip(v6("fe80::1")), Some(PrivateKind::LinkLocal));
        assert_eq!(classify_ip(v6("febf::1")), Some(PrivateKind::LinkLocal));
    }

    #[test]
    fn classify_metadata_four_addresses_and_mapped() {
        assert_eq!(
            classify_ip(v4(169, 254, 169, 254)),
            Some(PrivateKind::Metadata)
        );
        assert_eq!(
            classify_ip(v6("fd00:ec2::254")),
            Some(PrivateKind::Metadata)
        );
        assert_eq!(
            classify_ip(v4(100, 100, 100, 200)),
            Some(PrivateKind::Metadata)
        );
        assert_eq!(classify_ip(v4(192, 0, 0, 192)), Some(PrivateKind::Metadata));
        assert_eq!(
            classify_ip(v6("::ffff:169.254.169.254")),
            Some(PrivateKind::Metadata)
        );
    }

    #[test]
    fn classify_private_ranges() {
        assert_eq!(classify_ip(v4(10, 0, 0, 5)), Some(PrivateKind::Private));
        assert_eq!(classify_ip(v4(172, 16, 0, 1)), Some(PrivateKind::Private));
        assert_eq!(
            classify_ip(v4(172, 31, 255, 255)),
            Some(PrivateKind::Private)
        );
        assert_eq!(classify_ip(v4(192, 168, 1, 1)), Some(PrivateKind::Private));
        assert_eq!(classify_ip(v6("fc00::1")), Some(PrivateKind::Private));
        assert_eq!(classify_ip(v6("fd12:3456::1")), Some(PrivateKind::Private));
        // Neighbours of 172.16/12 are public.
        assert_eq!(classify_ip(v4(172, 15, 0, 1)), None);
        assert_eq!(classify_ip(v4(172, 32, 0, 1)), None);
    }

    #[test]
    fn classify_unspecified() {
        assert_eq!(classify_ip(v4(0, 0, 0, 0)), Some(PrivateKind::Unspecified));
        assert_eq!(classify_ip(v6("::")), Some(PrivateKind::Unspecified));
    }

    #[test]
    fn classify_multicast() {
        assert_eq!(classify_ip(v4(224, 0, 0, 1)), Some(PrivateKind::Multicast));
        assert_eq!(
            classify_ip(v4(239, 255, 255, 255)),
            Some(PrivateKind::Multicast)
        );
        assert_eq!(classify_ip(v6("ff02::1")), Some(PrivateKind::Multicast));
    }

    #[test]
    fn classify_broadcast() {
        assert_eq!(
            classify_ip(v4(255, 255, 255, 255)),
            Some(PrivateKind::Broadcast)
        );
    }

    #[test]
    fn classify_reserved_ranges() {
        assert_eq!(classify_ip(v4(240, 0, 0, 1)), Some(PrivateKind::Reserved));
        assert_eq!(classify_ip(v4(192, 0, 2, 10)), Some(PrivateKind::Reserved));
        assert_eq!(
            classify_ip(v4(198, 51, 100, 7)),
            Some(PrivateKind::Reserved)
        );
        assert_eq!(classify_ip(v4(203, 0, 113, 9)), Some(PrivateKind::Reserved));
        assert_eq!(classify_ip(v4(100, 64, 0, 1)), Some(PrivateKind::Reserved));
        assert_eq!(
            classify_ip(v4(100, 127, 255, 254)),
            Some(PrivateKind::Reserved)
        );
        assert_eq!(classify_ip(v4(0, 1, 2, 3)), Some(PrivateKind::Reserved));
        assert_eq!(classify_ip(v6("2001:db8::1")), Some(PrivateKind::Reserved));
        // 100.128/9 is public (just past the shared range).
        assert_eq!(classify_ip(v4(100, 128, 0, 1)), None);
    }

    #[test]
    fn classify_public_addresses_are_none() {
        assert_eq!(classify_ip(public_v4()), None);
        assert_eq!(classify_ip(v4(8, 8, 8, 8)), None);
        assert_eq!(classify_ip(v6("2606:4700:4700::1111")), None);
        assert_eq!(classify_ip(v6("::ffff:8.8.8.8")), None);
        // NAT64 and 6to4 carrying a public IPv4 are public.
        assert_eq!(classify_ip(v6("64:ff9b::808:808")), None);
        // ... and carrying a private one are not.
        assert_eq!(
            classify_ip(v6("64:ff9b::a00:5")),
            Some(PrivateKind::Private)
        );
        assert_eq!(classify_ip(v6("2002:a00:5::1")), Some(PrivateKind::Private));
    }

    // --- validate_url --------------------------------------------------

    #[test]
    fn validate_url_accepts_public_https() {
        let url = validate_url("https://api.example.com/v1", &strict()).unwrap();
        assert_eq!(url.host_str(), Some("api.example.com"));
    }

    #[test]
    fn validate_url_accepts_http_towards_loopback_literals() {
        assert!(validate_url("http://localhost:11434", &strict()).is_ok());
        assert!(validate_url("http://127.0.0.1:8000", &strict()).is_ok());
        assert!(validate_url("http://[::1]:8000/v1", &strict()).is_ok());
        assert!(validate_url("http://ollama.localhost:11434", &strict()).is_ok());
    }

    #[test]
    fn validate_url_refuses_http_outside_loopback() {
        let err = validate_url("http://example.com", &strict()).unwrap_err();
        assert_eq!(err, EndpointRefusal::HttpOutsideLoopback);
        assert_eq!(err.code(), "endpoint_http_outside_loopback");
        // A private IP literal over http is still "outside loopback".
        assert_eq!(
            validate_url("http://10.0.0.5:8000", &strict()).unwrap_err(),
            EndpointRefusal::HttpOutsideLoopback
        );
    }

    #[test]
    fn validate_url_refuses_http_loopback_when_policy_forbids_it() {
        let policy = EndpointPolicy {
            allow_insecure_loopback: false,
            ..EndpointPolicy::default()
        };
        let err = validate_url("http://localhost:11434", &policy).unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::SchemeNotAllowed {
                scheme: "http".to_string()
            }
        );
        assert_eq!(err.code(), "endpoint_scheme_not_allowed");
    }

    #[test]
    fn validate_url_refuses_other_schemes() {
        let err = validate_url("ftp://example.com", &strict()).unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::SchemeNotAllowed {
                scheme: "ftp".to_string()
            }
        );
        assert!(matches!(
            validate_url("file:///etc/passwd", &strict()).unwrap_err(),
            EndpointRefusal::SchemeNotAllowed { .. }
        ));
    }

    #[test]
    fn validate_url_refuses_credentials() {
        let err = validate_url("https://user:pw@h", &strict()).unwrap_err();
        assert_eq!(err, EndpointRefusal::CredentialsInUrl);
        assert_eq!(err.code(), "endpoint_credentials_in_url");
        assert_eq!(
            validate_url("https://token@api.example.com", &strict()).unwrap_err(),
            EndpointRefusal::CredentialsInUrl
        );
    }

    #[test]
    fn validate_url_refuses_missing_host_and_garbage() {
        assert_eq!(
            validate_url("https://", &strict()).unwrap_err(),
            EndpointRefusal::HostMissing
        );
        assert_eq!(
            validate_url("https:///v1", &strict()).unwrap_err(),
            EndpointRefusal::HostMissing
        );
        let err = validate_url("not a url", &strict()).unwrap_err();
        assert_eq!(err, EndpointRefusal::InvalidUrl);
        assert_eq!(err.code(), "endpoint_invalid_url");
        assert_eq!(
            validate_url("https://example.com:99999", &strict()).unwrap_err(),
            EndpointRefusal::InvalidUrl
        );
    }

    // --- validate_resolved / validate_endpoint_with ------------------------

    #[test]
    fn endpoint_public_name_resolving_to_public_is_accepted() {
        let url = validate_endpoint_with(
            "https://api.example.com/v1",
            &strict(),
            &resolving(vec![public_v4(), v6("2606:4700:4700::1111")]),
        )
        .unwrap();
        assert_eq!(url.host_str(), Some("api.example.com"));
    }

    #[test]
    fn endpoint_mixed_public_and_private_answers_is_refused() {
        let err = validate_endpoint_with(
            "https://api.example.com",
            &strict(),
            &resolving(vec![public_v4(), v4(10, 0, 0, 5)]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Private
            }
        );
        assert_eq!(err.code(), "endpoint_private_address");
    }

    #[test]
    fn endpoint_public_name_rebinding_to_loopback_is_refused() {
        let err = validate_endpoint_with(
            "https://api.example.com",
            &strict(),
            &resolving(vec![v4(127, 0, 0, 1)]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Loopback
            }
        );
        // Not even with private ranges allowed: loopback needs a loopback literal.
        let lenient = EndpointPolicy {
            allow_private_ranges: true,
            ..EndpointPolicy::default()
        };
        assert!(validate_endpoint_with(
            "https://api.example.com",
            &lenient,
            &resolving(vec![v4(127, 0, 0, 1)]),
        )
        .is_err());
    }

    #[test]
    fn endpoint_public_name_resolving_to_metadata_is_refused() {
        let err = validate_endpoint_with(
            "https://api.example.com",
            &strict(),
            &resolving(vec![v4(169, 254, 169, 254)]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Metadata
            }
        );
    }

    #[test]
    fn endpoint_empty_resolution_is_unresolvable() {
        let err = validate_endpoint_with("https://api.example.com", &strict(), &resolving(vec![]))
            .unwrap_err();
        assert_eq!(err, EndpointRefusal::UnresolvableHost);
        assert_eq!(err.code(), "endpoint_unresolvable");
    }

    #[test]
    fn endpoint_resolver_error_is_unresolvable() {
        let failing =
            |_: &str, _: u16| -> io::Result<Vec<IpAddr>> { Err(io::Error::other("nxdomain")) };
        assert_eq!(
            validate_endpoint_with("https://api.example.com", &strict(), &failing).unwrap_err(),
            EndpointRefusal::UnresolvableHost
        );
    }

    #[test]
    fn endpoint_loopback_literal_resolving_to_loopback_is_accepted() {
        assert!(validate_endpoint_with(
            "http://localhost:11434/v1",
            &strict(),
            &resolving(vec![v4(127, 0, 0, 1), v6("::1")]),
        )
        .is_ok());
        // IP literals are not resolved at all.
        let never = |_: &str, _: u16| -> io::Result<Vec<IpAddr>> {
            panic!("an IP literal must not be resolved")
        };
        assert!(validate_endpoint_with("http://127.0.0.1:8000", &strict(), &never).is_ok());
        assert!(validate_endpoint_with("https://[::1]:8443", &strict(), &never).is_ok());
    }

    #[test]
    fn endpoint_loopback_literal_resolving_elsewhere_is_refused() {
        // A tampered hosts file: `localhost` → 10.0.0.5.
        let err = validate_endpoint_with(
            "http://localhost:11434",
            &strict(),
            &resolving(vec![v4(10, 0, 0, 5)]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Private
            }
        );
    }

    #[test]
    fn endpoint_private_ip_literal_is_refused_by_default() {
        let err = validate_endpoint_with(
            "https://10.0.0.5:8443",
            &strict(),
            &resolving(vec![public_v4()]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Private
            }
        );
    }

    #[test]
    fn endpoint_host_in_allowed_private_hosts_is_accepted() {
        let policy = EndpointPolicy {
            allowed_private_hosts: vec!["LLM.Corp.Internal.".to_string(), "10.0.0.7".to_string()],
            ..EndpointPolicy::default()
        };
        assert!(validate_endpoint_with(
            "https://llm.corp.internal/v1",
            &policy,
            &resolving(vec![v4(10, 0, 0, 5)]),
        )
        .is_ok());
        assert!(validate_endpoint_with("https://10.0.0.7", &policy, &resolving(vec![])).is_ok());
        // The exception is exact: a sibling host is still refused.
        assert!(validate_endpoint_with(
            "https://other.corp.internal",
            &policy,
            &resolving(vec![v4(10, 0, 0, 5)]),
        )
        .is_err());
        // And it never lifts the scheme rule.
        assert_eq!(
            validate_endpoint_with(
                "http://llm.corp.internal",
                &policy,
                &resolving(vec![v4(10, 0, 0, 5)]),
            )
            .unwrap_err(),
            EndpointRefusal::HttpOutsideLoopback
        );
    }

    #[test]
    fn endpoint_allow_private_ranges_lifts_private_but_not_metadata() {
        let policy = EndpointPolicy {
            allow_private_ranges: true,
            ..EndpointPolicy::default()
        };
        assert!(validate_endpoint_with(
            "https://llm.corp.internal",
            &policy,
            &resolving(vec![v4(192, 168, 1, 20)]),
        )
        .is_ok());
        assert_eq!(
            validate_endpoint_with(
                "https://llm.corp.internal",
                &policy,
                &resolving(vec![v4(169, 254, 169, 254)]),
            )
            .unwrap_err(),
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Metadata
            }
        );
    }

    #[test]
    fn validate_resolved_direct_call_checks_all_addresses() {
        let url = Url::parse("https://api.example.com").unwrap();
        assert!(validate_resolved(&url, &[public_v4()], &strict()).is_ok());
        assert_eq!(
            validate_resolved(&url, &[public_v4(), v4(0, 0, 0, 0)], &strict()).unwrap_err(),
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Unspecified
            }
        );
        assert_eq!(
            validate_resolved(&url, &[], &strict()).unwrap_err(),
            EndpointRefusal::UnresolvableHost
        );
    }

    #[tokio::test]
    async fn validate_endpoint_async_handles_ip_literals_without_dns() {
        assert!(validate_endpoint("http://127.0.0.1:8000", &strict())
            .await
            .is_ok());
        assert_eq!(
            validate_endpoint("https://10.0.0.5", &strict())
                .await
                .unwrap_err(),
            EndpointRefusal::PrivateAddress {
                kind: PrivateKind::Private
            }
        );
        assert_eq!(
            validate_endpoint("http://example.com", &strict())
                .await
                .unwrap_err(),
            EndpointRefusal::HttpOutsideLoopback
        );
    }

    // --- origin_of -------------------------------------------------------

    #[test]
    fn origin_of_normalises_case_default_port_and_path() {
        assert_eq!(
            origin_of("HTTPS://Api.Example.com:443/v1/").as_deref(),
            Some("https://api.example.com")
        );
        assert_eq!(
            origin_of("http://localhost:11434/v1").as_deref(),
            Some("http://localhost:11434")
        );
        assert_eq!(
            origin_of("https://api.example.com:8443/v1?x=1#f").as_deref(),
            Some("https://api.example.com:8443")
        );
        assert_eq!(
            origin_of("http://example.com:80/").as_deref(),
            Some("http://example.com")
        );
        assert_eq!(
            origin_of("https://[::1]:8443/x").as_deref(),
            Some("https://[::1]:8443")
        );
        assert_eq!(origin_of("not a url"), None);
        assert_eq!(origin_of("https://"), None);
    }

    // --- Display / message_for ------------------------------------------

    #[test]
    fn refusal_display_never_contains_path_or_query() {
        let url = "https://api.example.com/secret/path?api_key=sk-abcdefghijklmnop";
        let err =
            validate_endpoint_with(url, &strict(), &resolving(vec![v4(10, 0, 0, 5)])).unwrap_err();
        for text in [err.to_string(), format!("{err:?}"), err.message_for(url)] {
            assert!(!text.contains("/secret"), "{text}");
            assert!(!text.contains("api_key"), "{text}");
            assert!(!text.contains("sk-"), "{text}");
        }
        assert_eq!(
            err.message_for(url),
            "https://api.example.com: the endpoint resolves to a non-public address (private)"
        );
        let creds = "https://user:pw@api.example.com/v1";
        let err = validate_url(creds, &strict()).unwrap_err();
        let text = err.message_for(creds);
        assert!(!text.contains("user"), "{text}");
        assert!(!text.contains("pw@"), "{text}");
        assert!(!text.contains("/v1"), "{text}");
    }

    #[test]
    fn refusal_codes_are_stable() {
        let all = [
            (EndpointRefusal::InvalidUrl, "endpoint_invalid_url"),
            (
                EndpointRefusal::SchemeNotAllowed {
                    scheme: "ftp".into(),
                },
                "endpoint_scheme_not_allowed",
            ),
            (
                EndpointRefusal::HttpOutsideLoopback,
                "endpoint_http_outside_loopback",
            ),
            (
                EndpointRefusal::CredentialsInUrl,
                "endpoint_credentials_in_url",
            ),
            (EndpointRefusal::HostMissing, "endpoint_host_missing"),
            (
                EndpointRefusal::PrivateAddress {
                    kind: PrivateKind::Reserved,
                },
                "endpoint_private_address",
            ),
            (EndpointRefusal::UnresolvableHost, "endpoint_unresolvable"),
            (
                EndpointRefusal::RedirectsNotAllowed,
                "endpoint_redirects_not_allowed",
            ),
        ];
        for (refusal, code) in all {
            assert_eq!(refusal.code(), code);
        }
    }

    // --- redirects / client builder ------------------------------------

    #[test]
    fn refuse_redirect_rejects_3xx_only() {
        assert_eq!(
            refuse_redirect(302).unwrap_err(),
            EndpointRefusal::RedirectsNotAllowed
        );
        assert_eq!(
            refuse_redirect(307).unwrap_err().code(),
            "endpoint_redirects_not_allowed"
        );
        assert!(refuse_redirect(200).is_ok());
        assert!(refuse_redirect(401).is_ok());
        assert!(refuse_redirect(503).is_ok());
    }

    #[test]
    fn http_client_builder_builds_without_network() {
        assert!(http_client_builder_for_endpoints().build().is_ok());
        assert!(CONNECT_TIMEOUT <= Duration::from_secs(10));
    }

    // --- redact_error_body ----------------------------------------------

    #[test]
    fn redact_hides_openai_style_key_in_json_error() {
        let body = r#"{"error":"Incorrect API key provided: sk-abc123def456ghi789jkl012mno345"}"#;
        let out = redact_error_body(body, 4096);
        assert!(!out.contains("sk-abc"), "{out}");
        assert!(out.contains(REDACTED), "{out}");
        assert!(out.contains("Incorrect API key provided"), "{out}");
    }

    #[test]
    fn redact_hides_bearer_authorization_and_key_value_pairs() {
        let out = redact_error_body(
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload; token=abc; key=xyz; \"api_key\": \"q1w2e3\"",
            4096,
        );
        assert!(!out.contains("eyJ"), "{out}");
        assert!(!out.contains("token=abc"), "{out}");
        assert!(!out.contains("key=xyz"), "{out}");
        assert!(!out.contains("q1w2e3"), "{out}");
        let out = redact_error_body("bearer abcdef", 4096);
        assert!(!out.contains("abcdef"), "{out}");
    }

    #[test]
    fn redact_hides_long_hex_and_base64_runs() {
        let hex = "0123456789abcdef0123456789abcdef";
        let b64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=";
        let out = redact_error_body(&format!("digest {hex} and blob {b64} end"), 4096);
        assert!(!out.contains(hex), "{out}");
        assert!(!out.contains(b64), "{out}");
        assert!(out.ends_with(" end"), "{out}");
        // Short tokens and ordinary words survive.
        assert_eq!(
            redact_error_body("model not found", 4096),
            "model not found"
        );
    }

    #[test]
    fn redact_truncates_on_utf8_boundary() {
        assert_eq!(redact_error_body("héllo", 2), "h");
        assert_eq!(redact_error_body("héllo", 3), "hé");
        assert_eq!(redact_error_body("héllo", 100), "héllo");
        assert_eq!(redact_error_body("héllo", 0), "");
        let long = "x".repeat(50);
        assert_eq!(redact_error_body(&long, 10).len(), 10);
    }

    #[test]
    fn redact_catches_a_secret_that_truncation_would_have_split() {
        let body = format!("prefix sk-{}", "a".repeat(40));
        let out = redact_error_body(&body, 12);
        assert!(out.len() <= 12, "{out}");
        assert!(!out.contains("sk-a"), "{out}");
    }

    #[test]
    fn default_policy_is_strict_except_insecure_loopback() {
        let policy = EndpointPolicy::default();
        assert!(policy.allow_insecure_loopback);
        assert!(policy.allowed_private_hosts.is_empty());
        assert!(!policy.allow_private_ranges);
    }
}
