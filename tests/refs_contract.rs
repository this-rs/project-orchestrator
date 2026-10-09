//! The golden fixtures in `tests/fixtures/refs/` are the contract with the
//! frontend. These tests keep them and the Rust types in step: change a type
//! and a fixture goes red; change a fixture and the type refuses it.

use project_orchestrator::chat::message_attachments::{self, MessageAttachment};
use project_orchestrator::refs::access::{Disclosure, Resolution};
use project_orchestrator::refs::block;
use project_orchestrator::refs::types::RawRef;
use project_orchestrator::refs::validate::{
    validate_one, validate_token, InvalidReason, MAX_REFS_PER_MESSAGE,
};
use project_orchestrator::refs::wire::{
    RefSearchResponse, RefsErrorBody, RefsEvent, REFS_INVALID_CODE,
};
use project_orchestrator::refs::{EntityRef, RefKind, CONTRACT_VERSION};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/refs/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(
        v["_contract_version"],
        u64::from(CONTRACT_VERSION),
        "{name}: _contract_version"
    );
    v
}

fn refs_of(v: &Value) -> Vec<EntityRef> {
    serde_json::from_value(v.clone()).expect("refs")
}

#[test]
fn every_fixture_declares_the_contract_version_first() {
    for name in [
        "entity_ref.json",
        "tokens.json",
        "po_refs_block.json",
        "search_response.json",
        "refs_resolved_event.json",
        "errors.json",
        "auth_ok_features.json",
        "kinds_response.json",
        "kind_ids.json",
    ] {
        let path = format!("{}/tests/fixtures/refs/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(path).unwrap();
        let first_key = text.lines().nth(1).unwrap().trim();
        assert!(
            first_key.starts_with("\"_contract_version\": 1"),
            "{name}: _contract_version must be the first field"
        );
        fixture(name);
    }
}

#[test]
fn entity_ref_fixture() {
    let v = fixture("entity_ref.json");
    let kinds: Vec<&str> = v["kinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    let historical: Vec<&str> = RefKind::HISTORICAL.iter().map(|k| k.as_str()).collect();
    assert_eq!(
        kinds, historical,
        "`kinds` is the fallback of a client: the first five"
    );
    let all: Vec<&str> = v["all_kinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    let ours: Vec<&str> = RefKind::ALL.iter().map(|k| k.as_str()).collect();
    assert_eq!(all, ours);
    assert_eq!(v["max_refs_per_message"], MAX_REFS_PER_MESSAGE as u64);
    for case in v["cases"].as_array().unwrap() {
        let parsed: EntityRef = serde_json::from_value(case["ref"].clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), case["ref"]);
        let raw: RawRef = serde_json::from_value(case["ref"].clone()).unwrap();
        assert_eq!(validate_one(&raw), Ok(parsed.clone()));
    }
    for name in v["reserved_kinds_refused_as_kind_disabled"]
        .as_array()
        .unwrap()
    {
        let raw = RawRef {
            kind: name.as_str().unwrap().into(),
            id: "57cf05c9-25b6-495d-ab07-de4b11d64736".into(),
        };
        assert_eq!(validate_one(&raw), Err(InvalidReason::KindDisabled));
    }
}

#[test]
fn token_fixture() {
    let v = fixture("tokens.json");
    for case in v["valid"].as_array().unwrap() {
        let r = validate_token(case["token"].as_str().unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&r).unwrap(), case["ref"]);
        assert_eq!(r.token(), case["token"].as_str().unwrap());
    }
    for case in v["invalid"].as_array().unwrap() {
        let reason = validate_token(case["token"].as_str().unwrap()).unwrap_err();
        assert_eq!(
            serde_json::to_value(reason).unwrap(),
            case["reason"],
            "{}",
            case["token"]
        );
    }
}

#[test]
fn block_fixture() {
    let v = fixture("po_refs_block.json");
    for case in v["cases"].as_array().unwrap() {
        let refs = refs_of(&case["refs"]);
        let text = case["text"].as_str().unwrap();
        let encoded = block::encode(text, &refs);
        assert_eq!(
            encoded,
            case["encoded"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        let (visible, got) = block::split(&encoded);
        assert_eq!(got, refs, "{}", case["name"]);
        let expected_text = case
            .get("decoded_text")
            .and_then(Value::as_str)
            .unwrap_or(text);
        assert_eq!(visible, expected_text, "{}", case["name"]);
    }

    let w = &v["with_attachments"];
    let refs = refs_of(&w["refs"]);
    let atts: Vec<MessageAttachment> = serde_json::from_value(w["attachments"].clone()).unwrap();
    let encoded =
        message_attachments::encode(&block::encode(w["text"].as_str().unwrap(), &refs), &atts);
    assert_eq!(encoded, w["encoded"].as_str().unwrap());
    let (after_att, got_atts) = message_attachments::split(&encoded);
    assert_eq!(got_atts, atts);
    assert_eq!(
        block::split(&after_att),
        (w["text"].as_str().unwrap().to_string(), refs)
    );

    for s in v["left_in_text_when_unparsable"].as_array().unwrap() {
        let s = s.as_str().unwrap();
        assert_eq!(block::split(s), (s.to_string(), vec![]));
    }
}

#[test]
fn search_fixture() {
    let v = fixture("search_response.json");
    let parsed: RefSearchResponse = serde_json::from_value(v["response"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), v["response"]);
    let empty: RefSearchResponse = serde_json::from_value(v["empty_response"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&empty).unwrap(), v["empty_response"]);
}

#[test]
fn refs_resolved_fixture() {
    let v = fixture("refs_resolved_event.json");
    let parsed: RefsEvent = serde_json::from_value(v["event"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), v["event"]);
    let RefsEvent::RefsResolved { refs } = parsed;
    let statuses: Vec<String> = refs
        .iter()
        .map(|r| {
            serde_json::to_value(r.status)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(statuses, ["ok", "truncated", "not_found", "forbidden"]);
}

#[test]
fn uniform_disclosure_never_emits_forbidden() {
    // The fixture shows `forbidden` so the client handles it; the server's
    // default policy must never produce it.
    let r = EntityRef::new(
        RefKind::Note,
        "9f1c2b7e-4d3a-4e58-8a61-0b2c7d9e1a10"
            .parse::<uuid::Uuid>()
            .unwrap(),
    );
    for res in [
        Resolution::Forbidden,
        Resolution::NotFound,
        Resolution::Failed,
    ] {
        let wire = serde_json::to_value(res.to_wire(&r, Disclosure::Uniform)).unwrap();
        assert_eq!(wire["status"], "not_found");
    }
}

#[test]
fn error_fixture() {
    let v = fixture("errors.json");
    assert_eq!(v["code"], REFS_INVALID_CODE);
    let mut seen = Vec::new();
    for case in v["cases"].as_array().unwrap() {
        let body: RefsErrorBody = serde_json::from_value(case["body"].clone()).unwrap();
        assert_eq!(serde_json::to_value(&body).unwrap(), case["body"]);
        assert_eq!(body.code, REFS_INVALID_CODE);
        assert_eq!(body.error, body.reason.message());
        seen.push(
            serde_json::to_value(body.reason)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    let declared: Vec<String> = v["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        seen, declared,
        "every reason has an example, in the declared order"
    );
}

#[test]
fn strict_fixture() {
    let v = fixture("strict.json");
    for case in v["valid"].as_array().unwrap() {
        let r = validate_token(case["token"].as_str().unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(r).unwrap(),
            case["ref"],
            "{}",
            case["token"]
        );
    }
    for case in v["invalid_tokens"].as_array().unwrap() {
        let reason = validate_token(case["token"].as_str().unwrap()).unwrap_err();
        assert_eq!(
            serde_json::to_value(reason).unwrap(),
            case["reason"],
            "{}",
            case["token"]
        );
    }
    // Wire shapes that must be refused by every deserializer, not just the token path.
    for bad in v["invalid_json"].as_array().unwrap() {
        let s = bad.as_str().unwrap();
        assert!(
            serde_json::from_str::<EntityRef>(s).is_err(),
            "EntityRef {s}"
        );
        // RawRef keeps any string for the id (validate_one names the failure) but not the shape.
        match serde_json::from_str::<RawRef>(s) {
            Ok(raw) => assert_eq!(validate_one(&raw), Err(InvalidReason::BadId), "RawRef {s}"),
            Err(_) => assert!(s.starts_with('['), "RawRef {s}"),
        }
    }
    for bad in v["invalid_blocks"].as_array().unwrap() {
        let s = bad.as_str().unwrap();
        assert_eq!(block::split(s), (s.to_string(), vec![]), "{s}");
    }
}

#[test]
fn auth_ok_features_fixture_is_what_the_server_builds() {
    use project_orchestrator::api::ws_auth::auth_ok_frame;
    use project_orchestrator::auth::jwt::Claims;
    let f = fixture("auth_ok_features.json");
    let claims = Claims {
        sub: f["auth_ok_on"]["user"]["id"].as_str().unwrap().into(),
        email: f["auth_ok_on"]["user"]["email"].as_str().unwrap().into(),
        name: f["auth_ok_on"]["user"]["name"].as_str().unwrap().into(),
        iat: 0,
        exp: 0,
        token_type: None,
        scope: None,
        jti: None,
    };
    assert_eq!(
        auth_ok_frame(
            &claims,
            project_orchestrator::refs::flag::features(true).as_deref()
        ),
        f["auth_ok_on"]
    );
    assert_eq!(
        auth_ok_frame(
            &claims,
            project_orchestrator::refs::flag::features(false).as_deref()
        ),
        f["auth_ok_off"]
    );
}

#[test]
fn the_prompt_section_teaches_exactly_the_fixture_kinds_and_the_token_shape() {
    let section = project_orchestrator::refs::cite::prompt_section();
    let kinds = fixture("entity_ref.json");
    for k in kinds["kinds"].as_array().expect("kinds") {
        assert!(
            section.contains(k.as_str().unwrap()),
            "{k} missing: {section}"
        );
    }
    for k in kinds["reserved_kinds_refused_as_kind_disabled"]
        .as_array()
        .expect("reserved")
    {
        assert!(!section.contains(k.as_str().unwrap()), "{k} is reserved");
    }
    // The shape it teaches is the one the golden tokens have: `#kind:` + a 36-character id.
    let tokens = fixture("tokens.json");
    for t in tokens["valid"].as_array().expect("valid") {
        let token = t["token"].as_str().unwrap();
        assert_eq!(token.len(), 1 + token.find(':').unwrap() + 36, "{token}");
        assert!(validate_token(token).is_ok(), "{token}");
    }
    assert!(section.contains("#kind:uuid") && section.contains("36"));
}

#[test]
fn kinds_response_fixture_is_what_the_server_builds() {
    use project_orchestrator::refs::wire::KindsResponse;
    let v = fixture("kinds_response.json");
    let parsed: KindsResponse = serde_json::from_value(v["response"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), v["response"]);
    assert_eq!(parsed, KindsResponse::active());
    assert_eq!(parsed.contract_version, CONTRACT_VERSION);
    let fallback: Vec<&str> = v["fallback_kinds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k.as_str().unwrap())
        .collect();
    let historical: Vec<&str> = RefKind::HISTORICAL.iter().map(|k| k.as_str()).collect();
    assert_eq!(fallback, historical);
    // every historical kind is announced: a new server never drops one.
    for kind in RefKind::HISTORICAL {
        assert!(parsed.kinds.iter().any(|k| k.kind == kind), "{kind}");
    }
    // the sensitive ones are exactly those the description names.
    let sensitive: Vec<&str> = parsed
        .kinds
        .iter()
        .filter(|k| k.sensitive)
        .map(|k| k.kind.as_str())
        .collect();
    assert_eq!(sensitive, ["commit", "file"]);
    // persona and skill are actors, cited with @; every other kind is data, cited with #.
    for k in &parsed.kinds {
        let actor = matches!(k.kind, RefKind::Persona | RefKind::Skill);
        assert_eq!(k.sigil, if actor { "@" } else { "#" }, "{}", k.kind);
    }
    // only the link has nothing to search, and only it opens outside the app.
    let unsearchable: Vec<_> = parsed
        .kinds
        .iter()
        .filter(|k| !k.searchable)
        .map(|k| k.kind)
        .collect();
    assert_eq!(unsearchable, [RefKind::Link]);
    let external: Vec<_> = parsed
        .kinds
        .iter()
        .filter(|k| k.opens_external)
        .map(|k| k.kind)
        .collect();
    assert_eq!(external, [RefKind::Link]);
}

#[test]
fn kind_ids_fixture() {
    let v = fixture("kind_ids.json");
    for case in v["valid"].as_array().unwrap() {
        let token = case["token"].as_str().unwrap();
        let r = validate_token(token).unwrap_or_else(|e| panic!("{token}: {e:?}"));
        assert_eq!(serde_json::to_value(&r).unwrap(), case["ref"], "{token}");
        assert_eq!(r.token(), token, "a valid token is its own canonical form");
        // the wire form is accepted as is, and refused when the id is changed.
        let back: EntityRef = serde_json::from_value(case["ref"].clone()).unwrap();
        assert_eq!(back, r);
    }
    for case in v["normalized"].as_array().unwrap() {
        let token = case["token"].as_str().unwrap();
        let r = validate_token(token).unwrap_or_else(|e| panic!("{token}: {e:?}"));
        assert_eq!(serde_json::to_value(&r).unwrap(), case["ref"], "{token}");
        assert_ne!(r.token(), token, "{token} is not canonical");
    }
    for case in v["invalid"].as_array().unwrap() {
        let token = case["token"].as_str().unwrap();
        let reason = validate_token(token).unwrap_err();
        assert_eq!(
            serde_json::to_value(reason).unwrap(),
            case["reason"],
            "{token}"
        );
    }
    let long = &v["link_too_long"];
    let token = format!(
        "#link:{}{}",
        long["prefix"].as_str().unwrap(),
        long["filler"]
            .as_str()
            .unwrap()
            .repeat(long["count"].as_u64().unwrap() as usize)
    );
    assert_eq!(
        serde_json::to_value(validate_token(&token).unwrap_err()).unwrap(),
        long["reason"]
    );
}
