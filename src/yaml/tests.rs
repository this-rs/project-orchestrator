//! Tests of the YAML module: the real shapes we read, the behaviour pinned at the
//! move off serde_yaml, and hostile input.
//!
//! "Pinned" below means: checked against serde_yaml 0.9 on the same input before
//! it was removed. Where the behaviour changed, the test says so.
use crate::neo4j::traits::*;

use super::*;
use crate::YamlConfig;
use std::time::{Duration, Instant};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/yaml")
            .join(name),
    )
    .unwrap()
}

fn config(yaml: &str) -> Result<YamlConfig, YamlError> {
    from_str(yaml)
}

// ── The production config.yaml (anonymised shape) ───────────────────────────

#[test]
fn the_production_config_shape_loads_every_section() {
    let c = config(&fixture("config.prod-shape.yaml")).unwrap();
    assert!(c.setup_completed, "absent → true (backward compat)");
    assert_eq!(c.server.port, 8080);
    assert_eq!(
        c.server.public_url.as_deref(),
        Some("https://po.example.org")
    );
    assert_eq!(c.neo4j.uri, "bolt://localhost:7687");
    assert_eq!(c.neo4j.password, "PLACEHOLDER-neo4j-password");
    assert_eq!(c.meilisearch.key, "PLACEHOLDER-meilisearch-key");

    let perms = c.chat.permissions.as_ref().unwrap();
    assert_eq!(perms.mode, "bypassPermissions");
    assert_eq!(perms.allowed_tools.len(), 9);
    assert_eq!(perms.allowed_tools[0], "mcp__project-orchestrator__*");
    assert_eq!(perms.allowed_tools[1], "Bash(git *)");
    assert!(perms.disallowed_tools.is_empty());
    assert_eq!(c.chat.auto_update_cli, Some(true));
    assert_eq!(c.chat.auto_update_app, Some(false));
    assert!(c
        .chat
        .process_path
        .as_deref()
        .unwrap()
        .contains(":/usr/bin:"));

    let auth = c.auth.as_ref().unwrap();
    assert_eq!(
        auth.jwt_secret,
        "PLACEHOLDER-jwt-secret-at-least-32-characters"
    );
    assert_eq!(
        auth.access_token_expiry_secs, 28800,
        "jwt_expiry_secs alias"
    );
    assert_eq!(auth.allowed_emails.as_ref().unwrap().len(), 4);
    let oidc = auth.oidc.as_ref().unwrap();
    assert_eq!(oidc.client_secret, "PLACEHOLDER-oidc-client-secret");
    assert_eq!(oidc.scopes, "openid email profile");

    let nr = &c.neural_routing;
    assert!(nr.enabled);
    assert!((nr.nn.min_similarity - 0.7).abs() < 1e-6);
    assert!((nr.cpu_guard.pause_threshold - 80.0).abs() < 1e-6);

    assert!(!c.remote_mcp.enabled);
    assert_eq!(c.remote_mcp.origin_allowlist.len(), 2);
    assert_eq!(c.remote_mcp.rate_limit.max_requests, 120);
}

/// The persist paths (`chat.permissions`, `chat.process_path`…) rewrite
/// config.yaml through a [`Value`] tree: every other value, and the key order,
/// must come back as it was.
#[test]
fn rewriting_the_production_config_shape_keeps_every_value_and_the_order() {
    let original = value_from_str(&fixture("config.prod-shape.yaml")).unwrap();
    let rewritten = to_string(&original).unwrap();
    assert_eq!(value_from_str(&rewritten).unwrap(), original);

    let keys: Vec<_> = original.as_mapping().unwrap().keys().cloned().collect();
    assert_eq!(
        keys,
        [
            "server",
            "neo4j",
            "meilisearch",
            "nats",
            "chat",
            "auth",
            "neural_routing",
            "remote_mcp"
        ]
    );
    // And the typed loader reads the rewritten file the same way.
    let c = config(&rewritten).unwrap();
    assert_eq!(c.server.port, 8080);
    assert_eq!(
        c.auth.unwrap().jwt_secret,
        "PLACEHOLDER-jwt-secret-at-least-32-characters"
    );
}

// ── config.yaml.example (copied as the config by the .deb and .rpm) ──────────

fn example() -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.yaml.example"),
    )
    .unwrap()
}

/// The shipped example loads, and its chat section names its provider.
#[test]
fn the_example_config_loads_with_its_chat_provider() {
    let c = config(&example()).unwrap();
    assert_eq!(c.chat.provider.as_deref(), Some("claude-code"));
    assert_eq!(c.chat.default_model.as_deref(), Some("claude-opus-4-7"));
    assert_eq!(c.server.port, 8080);
}

/// The example's other choice, `provider: "none"` with its Claude-only lines
/// removed as the comment says, loads too: no Claude field is required.
#[test]
fn the_example_config_loads_without_claude_code() {
    let without_claude: String = example()
        .replace(r#"provider: "claude-code""#, r#"provider: "none""#)
        .lines()
        .filter(|l| !l.trim_start().starts_with("default_model:"))
        .map(|l| format!("{l}\n"))
        .collect();
    let c = config(&without_claude).unwrap();
    assert_eq!(c.chat.provider.as_deref(), Some("none"));
    assert_eq!(c.chat.default_model, None);
    assert_eq!(c.chat.claude_cli_path, None);
}

// ── The config loader: behaviour pinned against serde_yaml ───────────────────

#[test]
fn an_empty_document_is_the_defaults_as_before() {
    for empty in ["", "# only a comment\n", "---\n", "{}\n"] {
        let c = config(empty).unwrap();
        assert!(c.setup_completed, "{empty:?}");
        assert_eq!(c.server.port, 8080, "{empty:?}");
        assert!(c.auth.is_none(), "{empty:?}");
    }
    // Changed: a bare `~` document was an error under serde_yaml; it is the
    // defaults now, like an empty file.
    assert!(config("~\n").is_ok());
}

#[test]
fn an_unquoted_number_or_boolean_reads_into_a_string_field_as_before() {
    let c = config("auth:\n  jwt_secret: 123456\ninfra_mode: 12\n").unwrap();
    assert_eq!(c.auth.unwrap().jwt_secret, "123456");
    assert_eq!(c.infra_mode.as_deref(), Some("12"));
    let c = config("auth:\n  jwt_secret: true\n").unwrap();
    assert_eq!(c.auth.unwrap().jwt_secret, "true");
}

#[test]
fn numbers_read_as_before() {
    assert_eq!(
        config("server:\n  port: 0x2382\n").unwrap().server.port,
        9090
    );
    assert_eq!(config("server:\n  port: +80\n").unwrap().server.port, 80);
    assert!(config("server:\n  port: 010\n").is_err(), "no legacy octal");
    assert!(config("server:\n  port: 80.0\n").is_err());
    // Changed: serde_yaml refused a quoted number and `_` separators for a
    // number field; both read now.
    assert_eq!(
        config("server:\n  port: \"9090\"\n").unwrap().server.port,
        9090
    );
    assert_eq!(
        config("server:\n  port: 9_090\n").unwrap().server.port,
        9090
    );
}

#[test]
fn booleans_are_yaml_1_2_as_before() {
    assert!(config("setup_completed: True\n").unwrap().setup_completed);
    assert!(config("setup_completed: TRUE\n").unwrap().setup_completed);
    assert!(
        config("setup_completed: yes\n").is_err(),
        "yes is not a boolean"
    );
    assert!(
        config("setup_completed: on\n").is_err(),
        "on is not a boolean"
    );
    // Changed: a quoted "false" was an error under serde_yaml.
    assert!(
        !config("setup_completed: \"false\"\n")
            .unwrap()
            .setup_completed
    );
}

#[test]
fn a_key_given_twice_is_an_error_as_before() {
    assert!(config("server:\n  port: 1\nserver:\n  port: 2\n").is_err());
    assert!(config("server:\n  port: 1\n  port: 2\n").is_err());
}

#[test]
fn enums_read_plain_or_tagged_as_before() {
    use crate::runner::models::PerTaskCap;
    let cap = |y: &str| config(y).map(|c| c.runner.per_task_cap);
    assert_eq!(
        cap("runner:\n  per_task_cap: enforce\n").unwrap(),
        PerTaskCap::Enforce
    );
    assert_eq!(
        cap("runner:\n  per_task_cap: !enforce\n").unwrap(),
        PerTaskCap::Enforce
    );
    assert!(cap("runner:\n  per_task_cap: abort\n").is_err());
}

#[test]
fn tags_unknown_fields_and_nulls_as_before() {
    // A custom tag on a mapping is ignored; an unknown field too.
    let c = config("server: !custom\n  port: 9091\n  nope: x\nwhatever: 1\n").unwrap();
    assert_eq!(c.server.port, 9091);
    for null in ["infra_mode: null\n", "infra_mode:\n", "infra_mode: ~\n"] {
        assert_eq!(config(null).unwrap().infra_mode, None, "{null:?}");
    }
    // Changed: `!!str 9092` into a number field read under serde_yaml; the tag
    // says string, so it is refused now. Same for `~` into a required string,
    // which serde_yaml read as the text "~".
    assert!(config("server:\n  port: !!str 9092\n").is_err());
    assert!(config("auth:\n  jwt_secret: ~\n").is_err());
}

#[test]
fn malformed_documents_are_errors_as_before() {
    for bad in [
        "server:\n  port: 1\n---\nserver:\n  port: 2\n",
        "- a\n",
        "hello\n",
        "server:\n\tport: 1\n",
        "setup_completed: [oops\n",
        "auth:\n  jwt_secret: x\n  allowed_emails: a@b.c\n",
        "auth: {}\n",
    ] {
        assert!(config(bad).is_err(), "{bad:?}");
    }
}

/// Changed: serde_yaml 0.9 left `<<` as an ordinary key; merges are expanded now.
#[test]
fn merge_keys_are_expanded_in_typed_reads() {
    let c = config("base: &b\n  port: 9093\nserver:\n  <<: *b\n").unwrap();
    assert_eq!(c.server.port, 9093);
}

/// The input may be config.yaml: an error must not carry its secrets.
#[test]
fn a_parse_error_does_not_quote_the_input() {
    for yaml in [
        "auth:\n  jwt_secret: SECRET-VALUE-42\n  jwt_expiry_secs: [oops\n",
        "server:\n  port: SECRET-VALUE-42\n",
        "auth:\n  jwt_secret: SECRET-VALUE-42\nauth:\n  jwt_secret: x\n",
    ] {
        let message = config(yaml).unwrap_err().to_string();
        assert!(!message.contains("SECRET-VALUE-42"), "{message}");
    }
}

// ── The Value tree used to rewrite config.yaml ───────────────────────────────

#[test]
fn value_scalars_resolve_as_serde_yaml_did() {
    let a = |yaml: &str| value_from_str(yaml).unwrap()["a"].clone();
    let s = |v: &str| Value::String(v.to_owned());
    assert_eq!(a("a: yes"), s("yes"));
    assert_eq!(a("a: on"), s("on"));
    assert_eq!(a("a: 017"), s("017"), "a leading zero keeps it a string");
    assert_eq!(a("a: 0123"), s("0123"));
    assert_eq!(a("a: 1_000"), s("1_000"));
    assert_eq!(a("a: 2026-10-18"), s("2026-10-18"));
    assert_eq!(a("a: 12:30"), s("12:30"));
    assert_eq!(a("a: '123'"), s("123"));
    assert_eq!(a("a: !!str 12"), s("12"));
    assert_eq!(a("a: !!binary aGVsbG8="), s("aGVsbG8="));
    assert_eq!(a("a: 0x1F"), Value::Int(31));
    assert_eq!(a("a: 0o17"), Value::Int(15));
    assert_eq!(a("a: 0b101"), Value::Int(5));
    assert_eq!(a("a: -5"), Value::Int(-5));
    assert_eq!(a("a: +80"), Value::Int(80));
    assert_eq!(a("a: !!int \"12\""), Value::Int(12));
    assert_eq!(a("a: 18446744073709551615"), Value::UInt(u64::MAX));
    assert_eq!(a("a: 1e3"), Value::Float(1000.0));
    assert_eq!(a("a: 017.5"), Value::Float(17.5));
    assert_eq!(a("a: .5"), Value::Float(0.5));
    assert_eq!(a("a: -.inf"), Value::Float(f64::NEG_INFINITY));
    assert!(matches!(a("a: .nan"), Value::Float(f) if f.is_nan()));
    assert_eq!(a("a: True"), Value::Bool(true));
    assert_eq!(a("a: NULL"), Value::Null);
    assert_eq!(a("a:"), Value::Null);
    assert_eq!(a("a: ~"), Value::Null);
}

#[test]
fn value_documents_keep_their_shape() {
    assert_eq!(value_from_str("").unwrap(), Value::Null);
    let v = value_from_str("z: 1\na: 2\nm: 3\n").unwrap();
    let keys: Vec<_> = v.as_mapping().unwrap().keys().cloned().collect();
    assert_eq!(keys, ["z", "a", "m"], "document order");
    // Aliases are expanded; `<<` stays an ordinary key, as with serde_yaml, so a
    // rewrite does not flatten a merge.
    let v = value_from_str("base: &b {x: 1}\nm:\n  <<: *b\n  y: 2\nl: &l [1]\nk: *l\n").unwrap();
    assert_eq!(v["m"]["<<"]["x"], Value::Int(1));
    assert_eq!(v["k"], Value::Sequence(vec![Value::Int(1)]));
    // Scalar keys are kept as their text.
    let v = value_from_str("1: x\ntrue: y\n").unwrap();
    assert_eq!(v["1"].as_str(), Some("x"));
    assert_eq!(v["true"].as_str(), Some("y"));
    // A missing key indexes to Null.
    assert_eq!(v["absent"]["deeper"], Value::Null);
}

#[test]
fn value_refuses_what_it_cannot_represent() {
    // Changed: serde_yaml accepted a collection as a key.
    assert!(value_from_str("? [a, b]\n: c\n").is_err());
    assert!(value_from_str("a: 1\na: 2\n").is_err());
    assert!(value_from_str("a: 1\n---\nb: 2\n").is_err());
    assert!(value_from_str("a: [oops\n").is_err());
}

/// Strings that look like another type are quoted when written, so a rewrite
/// of config.yaml never turns a password into a number or a boolean.
#[test]
fn strings_that_look_like_other_types_survive_a_rewrite() {
    let mut m = Mapping::new();
    for s in [
        "yes",
        "no",
        "on",
        "off",
        "y",
        "n",
        "true",
        "False",
        "null",
        "~",
        "",
        "123",
        "0123",
        "017",
        "1_000",
        "0x1F",
        "0o17",
        "1e3",
        ".inf",
        "-.nan",
        "+1",
        "2026-10-18",
        "12:30",
        " lead",
        "trail ",
        "a: b",
        "- x",
        "#c",
        "multi\nline",
        "*star",
        "&amp",
        "!bang",
        "@at",
        "`tick",
        "quote'",
        "dq\"",
        "back\\slash",
        "tab\tx",
        "é ⚠️",
        "/opt/a:/usr/b:/c",
        "Bash(git *)",
        "mcp__x__*",
        "0 2 * * *",
        "%percent",
        "{a}",
        "[a]",
        "a,b",
        "?q",
        "|pipe",
        ">gt",
        "PLACEHOLDER-jwt-secret",
    ] {
        m.insert(format!("k{}", m.len()), Value::String(s.into()));
    }
    m.insert("int".into(), Value::Int(-42));
    m.insert("uint".into(), Value::UInt(u64::MAX));
    m.insert("float".into(), Value::Float(0.7));
    m.insert("whole_float".into(), Value::Float(80.0));
    m.insert("bool".into(), Value::Bool(false));
    m.insert("null".into(), Value::Null);
    m.insert("empty_seq".into(), Value::Sequence(vec![]));
    m.insert("empty_map".into(), Value::Mapping(Mapping::new()));
    let v = Value::Mapping(m);
    let written = to_string(&v).unwrap();
    assert_eq!(value_from_str(&written).unwrap(), v, "{written}");
}

// ── Hostile input: every limit holds, and fails fast ─────────────────────────

/// Checks that every way in refuses `yaml`, quickly.
fn assert_refused(yaml: &str) {
    let started = Instant::now();
    assert!(
        from_str::<serde_json::Value>(yaml).is_err(),
        "typeless read"
    );
    assert!(value_from_str(yaml).is_err(), "value tree");
    assert!(from_str::<IgnoredAny>(yaml).is_err(), "skipped content");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "took {:?}",
        started.elapsed()
    );
}

/// The classic "billion laughs": 9 levels of 9 aliases, 9^9 strings once expanded.
fn billion_laughs() -> String {
    let mut yaml = String::from(
        "a: &a [\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\",\"lol\"]\n",
    );
    let names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
    for pair in names.windows(2) {
        let refs = vec![format!("*{}", pair[0]); 9].join(",");
        yaml.push_str(&format!("{0}: &{0} [{1}]\n", pair[1], refs));
    }
    yaml
}

#[test]
fn an_alias_bomb_is_refused() {
    let bomb = billion_laughs();
    assert!(bomb.len() < 1_000, "tiny input");
    assert_refused(&bomb);
    // Hidden under a key the config loader does not know, too.
    let started = Instant::now();
    let hidden = format!("server:\n  port: 1\nbomb:\n{}", indent(&bomb));
    assert!(config(&hidden).is_err());
    assert!(started.elapsed() < Duration::from_secs(5));
}

fn indent(yaml: &str) -> String {
    yaml.lines().map(|l| format!("  {l}\n")).collect()
}

#[test]
fn a_modest_use_of_aliases_still_reads() {
    let yaml = "base: &b {x: 1, y: [1, 2, 3]}\nl: [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]\n";
    let v = value_from_str(yaml).unwrap();
    assert_eq!(v["l"].as_sequence().unwrap().len(), 10);
    assert!(from_str::<serde_json::Value>(yaml).is_ok());
}

#[test]
fn too_many_aliases_or_anchors_are_refused() {
    let aliases = format!("a: &a x\nl: [{}]\n", vec!["*a"; MAX_ALIASES + 1].join(","));
    assert_refused(&aliases);
    let anchors: String = (0..=MAX_ANCHORS)
        .map(|i| format!("k{i}: &a{i} v\n"))
        .collect();
    assert_refused(&anchors);
}

#[test]
fn an_alias_to_itself_or_to_nothing_is_refused() {
    assert_refused("a: &a [*a]\n");
    assert_refused("a: *nowhere\n");
}

#[test]
fn deep_nesting_is_refused() {
    let depth = MAX_DEPTH * 4;
    let flow = format!("{}{}", "[".repeat(depth), "]".repeat(depth));
    assert_refused(&flow);
    let block: String = (0..depth)
        .map(|i| format!("{}k:\n", "  ".repeat(i)))
        .collect();
    assert_refused(&block);
    // One level past the limit is enough, flow or block.
    let past = MAX_DEPTH + 1;
    assert_refused(&format!("{}{}", "[".repeat(past), "]".repeat(past)));
    let block: String = (0..past)
        .map(|i| format!("{}k:\n", "  ".repeat(i)))
        .collect();
    assert_refused(&format!("{block}{}v\n", "  ".repeat(past)));
    // Within the limit, it reads.
    let ok = MAX_DEPTH - 2;
    let flow = format!("{}{}", "[".repeat(ok), "]".repeat(ok));
    assert!(value_from_str(&flow).is_ok());
}

#[test]
fn too_many_nodes_are_refused() {
    let wide = format!("[{}]", vec!["1"; MAX_NODES + 1].join(","));
    assert!(wide.len() < MAX_INPUT_BYTES);
    assert_refused(&wide);
}

#[test]
fn an_oversized_document_is_refused_before_parsing() {
    let big = format!("a: \"{}\"\n", "x".repeat(MAX_INPUT_BYTES));
    let started = Instant::now();
    assert!(matches!(
        from_str::<serde_json::Value>(&big),
        Err(YamlError::TooLarge { .. })
    ));
    assert!(matches!(
        value_from_str(&big),
        Err(YamlError::TooLarge { .. })
    ));
    assert!(matches!(config(&big), Err(YamlError::TooLarge { .. })));
    assert!(started.elapsed() < Duration::from_secs(1));
}
