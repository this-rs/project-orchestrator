//! YAML, read and written in safe Rust, with bounded resources.
//!
//! Every YAML document the backend reads goes through this module: `config.yaml`,
//! runner settings, skill packages (which may come from a third party) and the
//! `docker-compose.yml` of analysed projects (which come from any repository).
//!
//! The parser is `serde-saphyr` (on `granit-parser`), both `#![forbid(unsafe_code)]`.
//! It replaced `serde_yaml` 0.9, deprecated, whose parser was `unsafe-libyaml` — a
//! C-to-Rust translation of libyaml, archived, and flagged RUSTSEC-2024-0449.
//!
//! The limits below hold whatever the input: a document is refused, with an error,
//! before it can cost more than a bounded amount of memory or time.
//! - size: [`MAX_INPUT_BYTES`], checked before parsing;
//! - depth: [`MAX_DEPTH`] nested collections;
//! - aliases ("billion laughs"): at most [`MAX_ALIASES`] aliases, [`MAX_ANCHORS`]
//!   anchors and [`MAX_REPLAYED_EVENTS`] events replayed through aliases in total;
//! - volume: [`MAX_NODES`] nodes and [`MAX_EVENTS`] parser events.
//!
//! Parse errors never quote the input: the parser's source snippets are turned off,
//! because the input may be `config.yaml`, which holds secrets.
//!
//! Two ways in:
//! - [`from_str`] into a Rust type: the type is the schema (`port: 8080` into a
//!   `String` gives `"8080"`);
//! - [`value_from_str`] into a [`Value`] tree, for code that rewrites a file it
//!   does not fully know (`config.yaml`, one section at a time). Scalars there are
//!   resolved as serde_yaml did, so a rewrite keeps every other value as it was:
//!   `password: 0123` stays the string `"0123"`, never the number 123.

use indexmap::IndexMap;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Serialize, Serializer};
use serde_saphyr::granit_parser::{self, Event, ScalarStyle, Tag};
use std::collections::HashMap;

/// Largest document accepted, in bytes (4 MiB). A config file or a compose file is
/// a few kilobytes.
pub const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
/// Deepest nesting of collections (mappings and sequences).
pub const MAX_DEPTH: usize = 64;
/// Most `*alias` references in one document.
pub const MAX_ALIASES: usize = 1_000;
/// Most `&anchor` definitions in one document.
pub const MAX_ANCHORS: usize = 1_000;
/// Most events replayed through aliases, all aliases together: what an alias
/// bomb multiplies.
pub const MAX_REPLAYED_EVENTS: usize = 100_000;
/// Most nodes (scalars, mappings, sequences) in one document.
pub const MAX_NODES: usize = 100_000;
/// Most parser events in one document.
pub const MAX_EVENTS: usize = 250_000;

/// Why a YAML document could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum YamlError {
    /// The input is larger than [`MAX_INPUT_BYTES`]; it was not parsed.
    #[error("YAML document is {size} bytes, over the {limit}-byte limit")]
    TooLarge { size: usize, limit: usize },
    /// Malformed YAML, a shape that does not match the target type, or a limit
    /// (depth, aliases, nodes) exceeded.
    #[error("{0}")]
    Parse(Box<serde_saphyr::DeserializeError>),
    /// A document [`value_from_str`] cannot represent (a collection as a mapping
    /// key, a key given twice, several documents).
    #[error("{0}")]
    Shape(String),
    /// The value could not be written as YAML.
    #[error("{0}")]
    Serialize(Box<serde_saphyr::SerializeError>),
}

impl From<serde_saphyr::DeserializeError> for YamlError {
    fn from(e: serde_saphyr::DeserializeError) -> Self {
        YamlError::Parse(Box::new(e))
    }
}

impl From<serde_saphyr::SerializeError> for YamlError {
    fn from(e: serde_saphyr::SerializeError) -> Self {
        YamlError::Serialize(Box::new(e))
    }
}

/// The parser options: our limits, strict YAML 1.2 booleans, no input in errors.
fn options() -> serde_saphyr::Options {
    let mut budget = serde_saphyr::Budget::default();
    budget.max_depth = MAX_DEPTH;
    budget.max_aliases = MAX_ALIASES;
    budget.max_anchors = MAX_ANCHORS;
    budget.max_nodes = MAX_NODES;
    budget.max_events = MAX_EVENTS;
    budget.max_total_scalar_bytes = MAX_INPUT_BYTES;
    budget.max_reader_input_bytes = Some(MAX_INPUT_BYTES);

    let mut alias_limits = serde_saphyr::options::AliasLimits::default();
    alias_limits.max_total_replayed_events = MAX_REPLAYED_EVENTS;
    alias_limits.max_replay_stack_depth = MAX_DEPTH;

    let mut options = serde_saphyr::Options::default();
    options.budget = Some(budget);
    options.alias_limits = alias_limits;
    // As serde_yaml did: `yes`/`no`/`on`/`off` are not booleans.
    options.strict_booleans = true;
    // As serde_yaml did: a key given twice is an error.
    options.duplicate_keys = serde_saphyr::DuplicateKeyPolicy::Error;
    // Never echo the input in an error: it may be a config file holding secrets.
    options.with_snippet = false;
    // As serde_yaml did: `.inf` and `.nan` are floats, also where no type is fixed.
    options.reject_non_finite_typeless_float = false;
    options
}

fn check_size(input: &str) -> Result<(), YamlError> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(YamlError::TooLarge {
            size: input.len(),
            limit: MAX_INPUT_BYTES,
        });
    }
    Ok(())
}

/// Parse one YAML document into `T`, within the limits of this module.
pub fn from_str<T: DeserializeOwned>(input: &str) -> Result<T, YamlError> {
    check_size(input)?;
    Ok(serde_saphyr::from_str_with_options(input, options())?)
}

/// Write `value` as a YAML document.
///
/// Strings that a reader could take for a number, a boolean or null are quoted,
/// so a document written here reads back to the same values.
pub fn to_string<T: Serialize>(value: &T) -> Result<String, YamlError> {
    let mut options = serde_saphyr::SerializerOptions::default();
    // Plain or quoted scalars, never `>` folding of long lines: what serde_yaml wrote.
    options.prefer_block_scalars = false;
    Ok(serde_saphyr::to_string_with_options(value, options)?)
}

/// `value` as a YAML [`Value`] tree (written, then read back).
pub fn to_value<T: Serialize>(value: &T) -> Result<Value, YamlError> {
    value_from_str(&to_string(value)?)
}

/// A YAML mapping, keys in document order.
pub type Mapping = IndexMap<String, Value>;

/// Any YAML document, for code that edits a file without knowing its whole shape
/// (`config.yaml`, rewritten one section at a time). Read it with [`value_from_str`].
///
/// Mapping keys keep their order, so rewriting a file does not reorder it. Keys
/// are strings: a scalar key (`8080:`, `true:`) is kept as its text; a collection
/// used as a key is refused.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
    Sequence(Vec<Value>),
    Mapping(Mapping),
}

impl Value {
    /// The value under `key`, when this is a mapping that has it.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_mapping().and_then(|m| m.get(key))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(i) => u64::try_from(*i).ok(),
            Value::UInt(u) => Some(*u),
            _ => None,
        }
    }

    pub fn as_sequence(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Sequence(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_mapping(&self) -> Option<&Mapping> {
        match self {
            Value::Mapping(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_mapping_mut(&mut self) -> Option<&mut Mapping> {
        match self {
            Value::Mapping(m) => Some(m),
            _ => None,
        }
    }
}

/// `value["key"]`: the value under `key`, or [`Value::Null`] when there is none.
impl std::ops::Index<&str> for Value {
    type Output = Value;

    fn index(&self, key: &str) -> &Value {
        static NULL: Value = Value::Null;
        self.get(key).unwrap_or(&NULL)
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(b) => serializer.serialize_bool(*b),
            Value::Int(i) => serializer.serialize_i64(*i),
            Value::UInt(u) => serializer.serialize_u64(*u),
            Value::Float(f) => serializer.serialize_f64(*f),
            Value::String(s) => serializer.serialize_str(s),
            Value::Sequence(items) => items.serialize(serializer),
            Value::Mapping(map) => {
                use serde::ser::SerializeMap;
                let mut out = serializer.serialize_map(Some(map.len()))?;
                for (key, value) in map {
                    out.serialize_entry(key, value)?;
                }
                out.end()
            }
        }
    }
}

/// Parse one YAML document into a [`Value`] tree, within the limits of this module.
///
/// An empty document is [`Value::Null`]. Aliases are expanded; a `<<` merge key is
/// kept as an ordinary key (as serde_yaml did), so a rewrite does not flatten it.
/// Tags are dropped, the tagged value kept.
pub fn value_from_str(input: &str) -> Result<Value, YamlError> {
    // Every limit (size, depth, aliases, nodes) is checked by a first pass that
    // builds nothing; the tree is built only from a document that passed it.
    from_str::<IgnoredAny>(input)?;

    let mut parser_options = granit_parser::Options::default();
    parser_options.emit_comments = false;
    let mut builder = TreeBuilder {
        parser: granit_parser::Parser::new_from_str_with_options(input, parser_options),
        anchors: HashMap::new(),
    };
    builder.document()
}

struct TreeBuilder<'a> {
    parser: granit_parser::Parser<'a, granit_parser::StrInput<'a>>,
    anchors: HashMap<usize, Value>,
}

impl<'a> TreeBuilder<'a> {
    fn next(&mut self) -> Result<Event<'a>, YamlError> {
        loop {
            match self.parser.next_event() {
                Some(Ok((Event::Comment(..), _))) => continue,
                Some(Ok((event, _))) => return Ok(event),
                Some(Err(e)) => return Err(YamlError::Shape(e.to_string())),
                None => return Err(YamlError::Shape("unexpected end of YAML".into())),
            }
        }
    }

    fn document(&mut self) -> Result<Value, YamlError> {
        let mut value = Value::Null;
        let mut documents = 0;
        loop {
            match self.next()? {
                Event::StreamStart | Event::DocumentEnd => {}
                Event::StreamEnd => return Ok(value),
                Event::DocumentStart(..) => {
                    documents += 1;
                    if documents > 1 {
                        return Err(YamlError::Shape(
                            "more than one YAML document is not supported".into(),
                        ));
                    }
                }
                event => value = self.node(event)?,
            }
        }
    }

    fn remember(&mut self, anchor: usize, value: &Value) {
        if anchor != 0 {
            self.anchors.insert(anchor, value.clone());
        }
    }

    fn node(&mut self, event: Event<'a>) -> Result<Value, YamlError> {
        match event {
            Event::Scalar(text, style, anchor, tag) => {
                let value = resolve_scalar(&text, style, tag.as_deref());
                self.remember(anchor, &value);
                Ok(value)
            }
            Event::Alias(id) => self
                .anchors
                .get(&id)
                .cloned()
                .ok_or_else(|| YamlError::Shape("alias to an unknown anchor".into())),
            Event::SequenceStart(_, anchor, _) => {
                let mut items = Vec::new();
                loop {
                    match self.next()? {
                        Event::SequenceEnd => break,
                        event => items.push(self.node(event)?),
                    }
                }
                let value = Value::Sequence(items);
                self.remember(anchor, &value);
                Ok(value)
            }
            Event::MappingStart(_, anchor, _) => {
                let mut map = Mapping::new();
                loop {
                    let key = match self.next()? {
                        Event::MappingEnd => break,
                        Event::Scalar(text, style, key_anchor, tag) => {
                            let value = resolve_scalar(&text, style, tag.as_deref());
                            self.remember(key_anchor, &value);
                            text.into_owned()
                        }
                        event => match self.node(event)? {
                            Value::String(s) => s,
                            Value::Null => "null".to_owned(),
                            Value::Bool(b) => b.to_string(),
                            Value::Int(i) => i.to_string(),
                            Value::UInt(u) => u.to_string(),
                            Value::Float(f) => f.to_string(),
                            Value::Sequence(_) | Value::Mapping(_) => {
                                return Err(YamlError::Shape(
                                    "a collection as a mapping key is not supported".into(),
                                ));
                            }
                        },
                    };
                    let event = self.next()?;
                    let value = self.node(event)?;
                    if map.contains_key(&key) {
                        return Err(YamlError::Shape(format!("duplicate mapping key `{key}`")));
                    }
                    map.insert(key, value);
                }
                let value = Value::Mapping(map);
                self.remember(anchor, &value);
                Ok(value)
            }
            _ => Err(YamlError::Shape("unexpected YAML event".into())),
        }
    }
}

/// A scalar as serde_yaml resolved it (YAML 1.2 core schema, minus the forms
/// it kept as strings: integers with a leading zero, `_` separators).
fn resolve_scalar(text: &str, style: ScalarStyle, tag: Option<&Tag>) -> Value {
    let core = tag.and_then(Tag::core_suffix);
    if core == Some("str") || (tag.is_none() && !matches!(style, ScalarStyle::Plain)) {
        return Value::String(text.to_owned());
    }
    if tag.is_some() && core.is_none() && !matches!(style, ScalarStyle::Plain) {
        // A custom tag on a quoted scalar: still a string.
        return Value::String(text.to_owned());
    }
    match text {
        "" | "~" | "null" | "Null" | "NULL" => return Value::Null,
        "true" | "True" | "TRUE" => return Value::Bool(true),
        "false" | "False" | "FALSE" => return Value::Bool(false),
        _ => {}
    }
    if let Some(v) = parse_int(text) {
        return v;
    }
    if let Some(f) = parse_float(text) {
        return Value::Float(f);
    }
    Value::String(text.to_owned())
}

fn parse_int(text: &str) -> Option<Value> {
    let (negative, body) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    let (radix, digits) = if let Some(d) = body.strip_prefix("0x") {
        (16, d)
    } else if let Some(d) = body.strip_prefix("0o") {
        (8, d)
    } else if let Some(d) = body.strip_prefix("0b") {
        (2, d)
    } else {
        // `017` is a string for serde_yaml, not 17 nor octal 15.
        if body.len() > 1 && body.starts_with('0') {
            return None;
        }
        (10, body)
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    let magnitude = u64::from_str_radix(digits, radix).ok()?;
    if negative {
        if magnitude <= i64::MAX as u64 {
            Some(Value::Int(-(magnitude as i64)))
        } else if magnitude == i64::MAX as u64 + 1 {
            Some(Value::Int(i64::MIN))
        } else {
            None
        }
    } else {
        Some(match i64::try_from(magnitude) {
            Ok(i) => Value::Int(i),
            Err(_) => Value::UInt(magnitude),
        })
    }
}

fn parse_float(text: &str) -> Option<f64> {
    let unsigned = text.strip_prefix(['-', '+']).unwrap_or(text);
    let negative = text.starts_with('-');
    match unsigned {
        ".inf" | ".Inf" | ".INF" => {
            return Some(if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            });
        }
        ".nan" | ".NaN" | ".NAN" if unsigned.len() == text.len() => return Some(f64::NAN),
        _ => {}
    }
    // YAML 1.2 core: [-+]? ( \.[0-9]+ | [0-9]+ ( \.[0-9]* )? ) ( [eE] [-+]? [0-9]+ )?
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(i) => (&unsigned[..i], Some(&unsigned[i + 1..])),
        None => (unsigned, None),
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (mantissa, None),
    };
    let all_digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    let mantissa_ok = match frac_part {
        None => !int_part.is_empty() && all_digits(int_part),
        Some(frac) => {
            all_digits(int_part) && all_digits(frac) && !(int_part.is_empty() && frac.is_empty())
        }
    };
    let exponent_ok = exponent.is_none_or(|e| {
        let e = e.strip_prefix(['-', '+']).unwrap_or(e);
        !e.is_empty() && all_digits(e)
    });
    // `017` stays a string, as for serde_yaml (`017.5` is a number).
    let leading_zero = frac_part.is_none()
        && exponent.is_none()
        && int_part.len() > 1
        && int_part.starts_with('0');
    if !mantissa_ok || !exponent_ok || leading_zero {
        return None;
    }
    text.parse().ok()
}

#[cfg(test)]
mod tests;
