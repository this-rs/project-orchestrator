//! `cargo xtask ts-dto` — engendre les DTO TypeScript à partir des types serde
//! du backend (garde-fou du contrat REST front ↔ back, objectif O1 de
//! `docs/BOUCLE.md`).
//!
//! Le front a 473 sites d'appel REST et 507 DTO copiés À LA MAIN depuis les
//! types Rust ; sur 16 paires comparées, 10 divergent. Ce sous-programme lit
//! `src/**/*.rs`, extrait les `struct`/`enum` dérivant `Serialize`, et écrit
//! `frontend/src/types/generated/*.ts`. Périmètre du premier lot : `Note`,
//! `Task` (+ `TaskStatus`), `Project`, `Plan`.
//!
//! Deux modes :
//!   `--write` régénère le dossier `generated/` ;
//!   `--check` compare le généré au commité et sort 1 en nommant le fichier et
//!   le champ qui dérivent.
//!
//! Ce que le parseur std sait lire : `struct` (champs simples), `enum` sans
//! données, `Option<T>` → `T | null`, `Vec<T>` → `T[]`, `HashMap<K, V>` →
//! `Record<K, V>`, `#[serde(rename_all)]`, `#[serde(default)]` /
//! `#[serde(skip_serializing_if)]` → champ optionnel. Toute autre forme est
//! LISTÉE dans `xtask/XTASK-TS-DTO-NON-SUPORTE.md` (jamais devinée).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const HELP: &str = "\
usage: cargo xtask ts-dto [options]

  --write              régénère frontend/src/types/generated/*.ts
  --check              compare le généré au commité (défaut), exit 1 si dérive
  --repo <path>        checkout backend, défaut : remonter depuis cwd
  --frontend <path>    checkout frontend, défaut : <repo>/../frontend

exit 0 conforme, 1 dérive (--check), 2 non vérifiable.";

/// (nom TypeScript, nom Rust) du premier lot. `TaskNode`/`ProjectNode`/`PlanNode`
/// sont les types de réponse réellement sérialisés par les routes liste/lecture ;
/// le front les appelle `Task`/`Project`/`Plan`.
const TARGETS: &[(&str, &str)] = &[
    ("Note", "Note"),
    ("Task", "TaskNode"),
    ("TaskStatus", "TaskStatus"),
    ("Project", "ProjectNode"),
    ("Plan", "PlanNode"),
];

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct Options {
    pub write: bool,
    pub check: bool,
    pub repo: Option<PathBuf>,
    pub frontend: Option<PathBuf>,
    pub help: bool,
}

pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} attend une valeur"))
        };
        match a.as_str() {
            "--write" => o.write = true,
            "--check" => o.check = true,
            "--repo" => o.repo = Some(PathBuf::from(value("--repo")?)),
            "--frontend" => o.frontend = Some(PathBuf::from(value("--frontend")?)),
            "-h" | "--help" => o.help = true,
            other => return Err(format!("option inconnue « {other} »")),
        }
    }
    if !o.write && !o.check {
        o.check = true;
    }
    Ok(o)
}

// ---------------------------------------------------------------------------
// Modèle intermédiaire
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub rust_type: String,
    pub optional: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub rename: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Struct {
        fields: Vec<Field>,
    },
    Enum {
        variants: Vec<Variant>,
        rename_all: Option<String>,
    },
    /// Forme reconnue mais non engendrable (p. ex. enum à variantes à données).
    Unsupported {
        reason: String,
    },
}

impl Decl {
    fn field_types(&self) -> Vec<String> {
        match self {
            Decl::Struct { fields } => fields.iter().map(|f| f.rust_type.clone()).collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unsupported {
    pub where_: String,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Parsing (std uniquement, ligne à ligne)
// ---------------------------------------------------------------------------

/// Parse le contenu d'un fichier `.rs` : nom → déclaration, pour tout
/// `struct`/`enum` qui dérive `Serialize`.
pub fn parse_rust_source(content: &str) -> BTreeMap<String, Decl> {
    let lines: Vec<&str> = content.lines().collect();
    let mut decls = BTreeMap::new();
    let mut pending_attrs: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let raw = lines[i];
        let t = raw.trim();
        if t.starts_with("#[") {
            pending_attrs.push(t.to_string());
            i += 1;
            continue;
        }
        if t.starts_with("//") || t.is_empty() {
            // Un commentaire (///…) entre un attribut et la déclaration ne
            // casse pas l'accumulation ; une ligne vide non plus.
            i += 1;
            continue;
        }
        if let Some((kw, name)) = detect_decl(t) {
            if pending_attrs
                .iter()
                .any(|a| a.contains("derive") && a.contains("Serialize"))
            {
                let rename_all = extract_rename_all(&pending_attrs);
                let (body, end) = extract_body(&lines, i);
                let decl = match kw {
                    "struct" => parse_struct(&name, &body),
                    _ => parse_enum(&name, &body, rename_all),
                };
                decls.insert(name, decl);
                i = end;
                pending_attrs.clear();
                continue;
            }
        }
        pending_attrs.clear();
        i += 1;
    }
    decls
}

fn detect_decl(line: &str) -> Option<(&'static str, String)> {
    let l = line.trim_start();
    let l = l.strip_prefix("pub ").unwrap_or(l);
    for kw in ["struct", "enum"] {
        if let Some(rest) = l.strip_prefix(kw) {
            let next_is_ws = rest.chars().next().is_some_and(|c| c.is_whitespace());
            if !next_is_ws {
                continue; // p. ex. « structural »
            }
            let rest = rest.trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some((kw, name));
            }
        }
    }
    None
}

fn extract_rename_all(attrs: &[String]) -> Option<String> {
    for a in attrs {
        if let Some(rest) = a.split_once("rename_all") {
            // le premier littéral "..." après « rename_all = »
            if let Some(eq) = rest.1.find('=') {
                let after = &rest.1[eq + 1..];
                if let Some(open) = after.find('"') {
                    let rest = &after[open + 1..];
                    if let Some(close) = rest.find('"') {
                        return Some(rest[..close].to_string());
                    }
                }
            }
        }
    }
    None
}

/// Retourne les lignes intérieures d'un corps `{ … }` et l'index de la ligne de
/// fermeture. Compte les accolades sur les lignes non-commentaire.
fn extract_body<'a>(lines: &[&'a str], decl_idx: usize) -> (Vec<&'a str>, usize) {
    let mut depth = 0i32;
    let mut started = false;
    let mut body = Vec::new();
    let mut i = decl_idx;
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim_start();
        if !t.starts_with("//") {
            let opens = line.chars().filter(|&c| c == '{').count() as i32;
            let closes = line.chars().filter(|&c| c == '}').count() as i32;
            depth += opens;
            depth -= closes;
            if opens > 0 {
                started = true;
            }
        }
        if started {
            if depth == 0 {
                return (body, i);
            }
            if i > decl_idx {
                body.push(lines[i]);
            }
        }
        i += 1;
    }
    (body, i)
}

fn parse_struct(_name: &str, body: &[&str]) -> Decl {
    let mut fields = Vec::new();
    let mut pending_attr: Option<String> = None;
    for line in body {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with('}') {
            continue;
        }
        if t.starts_with("#[") {
            pending_attr = Some(t.to_string());
            continue;
        }
        if let Some(f) = parse_field(t, pending_attr.as_deref()) {
            fields.push(f);
        }
        pending_attr = None;
    }
    Decl::Struct { fields }
}

fn parse_field(line: &str, attr: Option<&str>) -> Option<Field> {
    let l = line.trim();
    let l = l.strip_prefix("pub ").unwrap_or(l);
    let (name, ty) = l.split_once(':')?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    // Strip a trailing `// comment` (p. ex. `pub slug: String, // URL-safe identifier`).
    let ty = ty.split("//").next().unwrap_or(ty).trim();
    let ty = ty.trim_end_matches(',').trim().to_string();
    let mut optional = false;
    if let Some(a) = attr {
        // #[serde(skip)] : le champ n'est pas sérialisé du tout.
        let is_skip = a.contains("skip")
            && !a.contains("skip_serializing_if")
            && !a.contains("skip_deserializing");
        if is_skip {
            return None;
        }
        optional = a.contains("default") || a.contains("skip_serializing_if");
    }
    Some(Field {
        name: name.to_string(),
        rust_type: ty,
        optional,
    })
}

fn parse_enum(_name: &str, body: &[&str], rename_all: Option<String>) -> Decl {
    let mut variants = Vec::new();
    let mut has_data = false;
    let mut pending_rename: Option<String> = None;
    for line in body {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with('}') {
            continue;
        }
        if t.starts_with("#[") {
            if let Some(r) = parse_serde_rename(t) {
                pending_rename = Some(r);
            }
            continue;
        }
        if t.starts_with("impl") || t.starts_with("fn") {
            break;
        }
        let name: String = t
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after = t[name.len()..].trim_start();
        if after.starts_with('(') || after.starts_with('{') {
            has_data = true;
        }
        variants.push(Variant {
            name,
            rename: pending_rename.take(),
        });
    }
    if has_data {
        Decl::Unsupported {
            reason: "enum avec variantes à données (union discriminée non engendrée)".to_string(),
        }
    } else {
        Decl::Enum {
            variants,
            rename_all,
        }
    }
}

fn parse_serde_rename(attr: &str) -> Option<String> {
    if !attr.contains("rename") || attr.contains("rename_all") {
        return None;
    }
    let eq = attr.find('=')?;
    let after = &attr[eq + 1..];
    let open = after.find('"')?;
    let rest = &after[open + 1..];
    let close = rest.find('"')?;
    Some(rest[..close].to_string())
}

// ---------------------------------------------------------------------------
// Mapping Rust → TypeScript (fonctions pures)
// ---------------------------------------------------------------------------

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

fn generic_inner<'a>(t: &'a str, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}<");
    if t.starts_with(&prefix) && t.ends_with('>') {
        Some(&t[prefix.len()..t.len() - 1])
    } else {
        None
    }
}

fn generic_inner2<'a>(t: &'a str, name: &str) -> Option<(&'a str, &'a str)> {
    let inner = generic_inner(t, name)?;
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                return Some((inner[..i].trim(), inner[i + 1..].trim()));
            }
            _ => {}
        }
    }
    None
}

fn scalar_ts(t: &str) -> Option<String> {
    let s = match t {
        "String" | "str" | "char" | "Uuid" | "NaiveDate" | "NaiveDateTime" | "NaiveTime" => {
            "string"
        }
        "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
        | "isize" | "f32" | "f64" => "number",
        "bool" => "boolean",
        "serde_json::Value" | "Value" => "unknown",
        _ => return None,
    };
    Some(s.to_string())
}

fn path_last_ident(t: &str) -> Option<&str> {
    let mut last = None;
    for seg in t.split("::") {
        if !is_ident(seg) {
            return None;
        }
        last = Some(seg);
    }
    last
}

/// Traduit un type Rust en type TypeScript. `registry` sert à résoudre les
/// références nommées (et à donner la raison quand le type est non supporté).
pub fn ts_type(rust: &str, registry: &BTreeMap<String, Decl>) -> Result<String, String> {
    let t = rust.trim();
    if let Some(inner) = generic_inner(t, "Option") {
        return Ok(format!("{} | null", ts_type(inner, registry)?));
    }
    if let Some(inner) = generic_inner(t, "Vec") {
        return Ok(format!("{}[]", ts_type(inner, registry)?));
    }
    if let Some((k, v)) = generic_inner2(t, "HashMap") {
        return Ok(format!(
            "Record<{}, {}>",
            ts_type(k, registry)?,
            ts_type(v, registry)?
        ));
    }
    if generic_inner(t, "DateTime").is_some() {
        return Ok("string".to_string());
    }
    if let Some(s) = scalar_ts(t) {
        return Ok(s);
    }
    if let Some(name) = path_last_ident(t) {
        return match registry.get(name) {
            Some(Decl::Unsupported { reason }) => Err(format!("{name} : {reason}")),
            Some(_) => Ok(name.to_string()),
            None => Err(format!("type {name} introuvable (pas un type serde connu)")),
        };
    }
    Err(format!("type non supporté : {t}"))
}

/// Tous les identifiants de type (hors scalaires et enveloppes) référencés par
/// un type Rust — sert à calculer la fermeture des types à engendrer.
pub fn referenced_names(rust: &str) -> Vec<String> {
    const NON_TYPE: &[&str] = &[
        "String",
        "str",
        "char",
        "Uuid",
        "NaiveDate",
        "NaiveDateTime",
        "NaiveTime",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "isize",
        "f32",
        "f64",
        "bool",
        "Option",
        "Vec",
        "HashMap",
        "DateTime",
        "Utc",
        "FixedOffset",
        "Local",
        "Value",
        "serde_json",
        "json",
        "Box",
        "Result",
        "Cow",
        "Duration",
    ];
    let mut names: Vec<String> = Vec::new();
    for token in rust.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':')) {
        if token.is_empty() {
            continue;
        }
        let last = token.rsplit("::").next().unwrap_or(token);
        if last.is_empty() || NON_TYPE.contains(&last) {
            continue;
        }
        names.push(last.to_string());
    }
    names.sort();
    names.dedup();
    names
}

/// Applique une règle `#[serde(rename_all = "...")]` à un nom de variante.
pub fn apply_rename_all(name: &str, rule: Option<&str>) -> String {
    match rule {
        None | Some("none") => name.to_string(),
        Some("snake_case") => to_snake_case(name),
        Some("camelCase") => to_camel_case(name),
        Some("kebab-case") => to_kebab_case(name),
        Some("lowercase") => name.to_ascii_lowercase(),
        Some("UPPERCASE") => name.to_ascii_uppercase(),
        Some(other) => {
            // Règle inconnue : on ne devine pas, on laisse le nom tel quel
            // (le cas ne se produit pas dans le périmètre actuel).
            let _ = other;
            name.to_string()
        }
    }
}

fn to_snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn to_camel_case(name: &str) -> String {
    let mut out = String::new();
    let mut cap_next = false;
    for c in name.chars() {
        if c == '_' {
            cap_next = true;
        } else if cap_next {
            out.push(c.to_ascii_uppercase());
            cap_next = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn to_kebab_case(name: &str) -> String {
    to_snake_case(name).replace('_', "-")
}

// ---------------------------------------------------------------------------
// Fermeture + rendu
// ---------------------------------------------------------------------------

fn render_type(
    ts_name: &str,
    decl: &Decl,
    registry: &BTreeMap<String, Decl>,
    unsupported: &mut Vec<Unsupported>,
) -> String {
    match decl {
        Decl::Struct { fields } => {
            let mut s = format!(
                "// Généré par `cargo xtask ts-dto --write`. Ne pas éditer.\n\nexport interface {ts_name} {{\n"
            );
            for f in fields {
                match ts_type(&f.rust_type, registry) {
                    Ok(t) => {
                        let q = if f.optional { "?" } else { "" };
                        s.push_str(&format!("  {}{q}: {t}\n", f.name));
                    }
                    Err(reason) => {
                        unsupported.push(Unsupported {
                            where_: format!("{ts_name}.{}", f.name),
                            reason,
                        });
                        s.push_str(&format!("  {}: unknown\n", f.name));
                    }
                }
            }
            s.push_str("}\n");
            s
        }
        Decl::Enum {
            variants,
            rename_all,
        } => {
            let mut parts = Vec::new();
            for v in variants {
                let wire = match &v.rename {
                    Some(r) => r.clone(),
                    None => apply_rename_all(&v.name, rename_all.as_deref()),
                };
                parts.push(format!("'{wire}'"));
            }
            format!(
                "// Généré par `cargo xtask ts-dto --write`. Ne pas éditer.\n\nexport type {ts_name} = {}\n",
                parts.join(" | ")
            )
        }
        Decl::Unsupported { .. } => String::new(),
    }
}

/// Calcule la fermeture des types à engendrer depuis `targets` et les rend.
/// Retourne (nom TS → texte) et la liste des formes non supportées.
pub fn generate(
    registry: &BTreeMap<String, Decl>,
    targets: &[(&str, &str)],
) -> (BTreeMap<String, String>, Vec<Unsupported>) {
    let mut ts_of: BTreeMap<String, String> = BTreeMap::new();
    for (ts, rust) in targets {
        ts_of.insert(rust.to_string(), ts.to_string());
    }

    let mut included: BTreeSet<String> = BTreeSet::new();
    let mut unsupported: Vec<Unsupported> = Vec::new();
    let mut queue: Vec<String> = targets.iter().map(|(_, r)| r.to_string()).collect();
    while let Some(name) = queue.pop() {
        if included.contains(&name) {
            continue;
        }
        match registry.get(&name) {
            None => unsupported.push(Unsupported {
                where_: name.clone(),
                reason: "type introuvable dans src/".to_string(),
            }),
            Some(Decl::Unsupported { reason }) => unsupported.push(Unsupported {
                where_: name.clone(),
                reason: reason.clone(),
            }),
            Some(decl) => {
                included.insert(name.clone());
                for ft in decl.field_types() {
                    for r in referenced_names(&ft) {
                        if !included.contains(&r) && !queue.contains(&r) {
                            queue.push(r);
                        }
                    }
                }
            }
        }
    }

    let mut rendered: BTreeMap<String, String> = BTreeMap::new();
    for rust_name in &included {
        let decl = registry.get(rust_name).unwrap();
        let ts_name = ts_of
            .get(rust_name)
            .cloned()
            .unwrap_or_else(|| rust_name.clone());
        let text = render_type(&ts_name, decl, registry, &mut unsupported);
        rendered.insert(ts_name, text);
    }
    unsupported.sort_by(|a, b| a.where_.cmp(&b.where_));
    unsupported.dedup_by(|a, b| a.where_ == b.where_ && a.reason == b.reason);
    (rendered, unsupported)
}

fn render_non_supporte(unsupported: &[Unsupported]) -> String {
    let mut s = String::from(
        "# XTASK-TS-DTO — formes non supportées\n\n\
         Généré par `cargo xtask ts-dto --write`. Le parseur std ne lit que les formes\n\
         courantes : `struct` (champs simples), `enum` sans données, `Option<T>` →\n\
         `T | null`, `Vec<T>` → `T[]`, `HashMap<K, V>` → `Record<K, V>`,\n\
         `#[serde(rename_all)]`, `#[serde(default)]` → champ optionnel. Toute autre forme\n\
         est LISTÉE ici plutôt que devinée ; le champ correspondant est engendré comme\n\
         `unknown`.\n\n\
         | Où | Raison |\n\
         |---|---|\n",
    );
    for u in unsupported {
        s.push_str(&format!("| {} | {} |\n", u.where_, u.reason));
    }
    s
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk_rs(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
}

fn locate_root(o: &Options) -> Result<PathBuf, String> {
    if let Some(r) = &o.repo {
        return Ok(r.clone());
    }
    let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        if dir.join("src").is_dir() && dir.join("xtask").is_dir() {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    Err("aucun checkout backend trouvé (src/ + xtask/) en remontant depuis cwd".to_string())
}

fn locate_frontend(o: &Options, root: &Path) -> Option<PathBuf> {
    if let Some(f) = &o.frontend {
        return Some(f.clone());
    }
    let candidates = [
        root.join("frontend"),
        root.parent().unwrap_or(root).join("frontend"),
        root.parent()
            .and_then(|p| p.parent())
            .unwrap_or(root)
            .join("frontend"),
    ];
    candidates
        .into_iter()
        .find(|c| c.join("src/types/index.ts").is_file())
}

fn write_files(
    out_dir: &Path,
    rendered: &BTreeMap<String, String>,
    unsupported: &[Unsupported],
    root: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{} : {e}", out_dir.display()))?;
    for (name, text) in rendered {
        let path = out_dir.join(format!("{name}.ts"));
        std::fs::write(&path, text).map_err(|e| format!("{} : {e}", path.display()))?;
    }
    let md = root.join("xtask").join("XTASK-TS-DTO-NON-SUPORTE.md");
    std::fs::write(&md, render_non_supporte(unsupported))
        .map_err(|e| format!("{} : {e}", md.display()))?;
    Ok(())
}

fn extract_interface_fields(text: &str) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty()
            || t.starts_with("//")
            || t.starts_with("export")
            || t.starts_with('{')
            || t.starts_with('}')
        {
            continue;
        }
        let t = t.trim_end_matches(',').trim();
        if let Some((name, ty)) = t.split_once(':') {
            fields.insert(name.trim().to_string(), ty.trim().to_string());
        }
    }
    fields
}

fn diff_type(name: &str, rendered: &str, committed: &str) -> Vec<String> {
    if rendered == committed {
        return Vec::new();
    }
    let r_iface = rendered.contains("interface");
    let c_iface = committed.contains("interface");
    if r_iface && c_iface {
        let r = extract_interface_fields(rendered);
        let c = extract_interface_fields(committed);
        let mut diffs = Vec::new();
        for (fname, fty) in &r {
            match c.get(fname) {
                None => diffs.push(format!(
                    "{name}.ts : champ « {fname} » présent côté généré, absent côté commité"
                )),
                Some(cty) if cty != fty => diffs.push(format!(
                    "{name}.ts : champ « {fname} » : généré `{fty}` ≠ commité `{cty}`"
                )),
                _ => {}
            }
        }
        for fname in c.keys() {
            if !r.contains_key(fname) {
                diffs.push(format!(
                    "{name}.ts : champ « {fname} » présent côté commité, absent côté généré"
                ));
            }
        }
        if diffs.is_empty() {
            diffs.push(format!("{name}.ts : diffère hors des champs"));
        }
        diffs
    } else {
        vec![format!("{name}.ts : diffère")]
    }
}

fn compare_files(out_dir: &Path, rendered: &BTreeMap<String, String>) -> Vec<String> {
    let mut diffs = Vec::new();
    for (name, text) in rendered {
        let path = out_dir.join(format!("{name}.ts"));
        match std::fs::read_to_string(&path) {
            Err(_) => diffs.push(format!("{name}.ts : manquant")),
            Ok(committed) => diffs.extend(diff_type(name, text, &committed)),
        }
    }
    diffs
}

// ---------------------------------------------------------------------------
// Entrée du sous-programme
// ---------------------------------------------------------------------------

pub fn run(args: &[String]) -> i32 {
    let o = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ts-dto : {e}\n\n{HELP}");
            return 2;
        }
    };
    if o.help {
        println!("{HELP}");
        return 0;
    }
    let root = match locate_root(&o) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ts-dto : non vérifiable — {e}");
            return 2;
        }
    };

    let registry = {
        let mut files = Vec::new();
        walk_rs(&root.join("src"), &mut files);
        let mut reg: BTreeMap<String, Decl> = BTreeMap::new();
        for f in files {
            let content = match std::fs::read_to_string(&f) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("ts-dto : non vérifiable — {} : {e}", f.display());
                    return 2;
                }
            };
            let parsed = parse_rust_source(&content);
            for (name, decl) in parsed {
                reg.entry(name).or_insert(decl);
            }
        }
        reg
    };

    let (rendered, unsupported) = generate(&registry, TARGETS);
    let frontend = locate_frontend(&o, &root);

    if o.write {
        let frontend = match frontend {
            Some(f) => f,
            None => {
                eprintln!("ts-dto : non vérifiable — frontend introuvable (--frontend <path>)");
                return 2;
            }
        };
        let out_dir = frontend.join("src/types/generated");
        if let Err(e) = write_files(&out_dir, &rendered, &unsupported, &root) {
            eprintln!("ts-dto : non vérifiable — {e}");
            return 2;
        }
        eprintln!(
            "ts-dto : {} types écrits dans {} — {} forme(s) non supportée(s) listées dans xtask/XTASK-TS-DTO-NON-SUPORTE.md",
            rendered.len(),
            out_dir.display(),
            unsupported.len()
        );
        return 0;
    }

    // --check (défaut)
    match frontend {
        None => {
            eprintln!(
                "ts-dto : frontend absent (CI) — {} types engendrés, rien à comparer",
                rendered.len()
            );
            0
        }
        Some(_) => {
            let out_dir = frontend.unwrap().join("src/types/generated");
            let diffs = compare_files(&out_dir, &rendered);
            if diffs.is_empty() {
                eprintln!("ts-dto : conforme — {} fichiers identiques", rendered.len());
                0
            } else {
                for d in &diffs {
                    eprintln!("ts-dto : DÉRIVE {d}");
                }
                1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> BTreeMap<String, Decl> {
        BTreeMap::new()
    }

    #[test]
    fn option_uuid_maps_to_string_or_null() {
        assert_eq!(ts_type("Option<Uuid>", &reg()).unwrap(), "string | null");
    }

    #[test]
    fn vec_string_maps_to_string_array() {
        assert_eq!(ts_type("Vec<String>", &reg()).unwrap(), "string[]");
    }

    #[test]
    fn nested_option_vec_maps_correctly() {
        assert_eq!(
            ts_type("Option<Vec<String>>", &reg()).unwrap(),
            "string[] | null"
        );
    }

    #[test]
    fn hashmap_maps_to_record() {
        assert_eq!(
            ts_type("HashMap<String, String>", &reg()).unwrap(),
            "Record<string, string>"
        );
    }

    #[test]
    fn snake_case_rename_all_transforms_variants() {
        assert_eq!(
            apply_rename_all("InProgress", Some("snake_case")),
            "in_progress"
        );
        assert_eq!(
            apply_rename_all("NeedsReview", Some("snake_case")),
            "needs_review"
        );
        assert_eq!(
            apply_rename_all("Interrupted", Some("snake_case")),
            "interrupted"
        );
    }

    #[test]
    fn unsupported_box_is_refused_not_ignored() {
        assert!(ts_type("Box<String>", &reg()).is_err());
        assert!(ts_type("(String, u32)", &reg()).is_err());
    }

    #[test]
    fn parse_extracts_a_serialize_struct() {
        let src = "\
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Foo {
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub tags: Vec<String>,
}
";
        let decls = parse_rust_source(src);
        match decls.get("Foo") {
            Some(Decl::Struct { fields }) => {
                assert_eq!(fields.len(), 3);
                assert_eq!(fields[0].name, "id");
                assert_eq!(fields[2].rust_type, "Vec<String>");
                assert!(fields[2].optional);
            }
            other => panic!("attendu un struct Foo, reçu {other:?}"),
        }
    }

    #[test]
    fn parse_extracts_an_enum_with_rename_all() {
        let src = "\
#[derive(Serialize, Deserialize)]
#[serde(rename_all = \"snake_case\")]
pub enum Status {
    Pending,
    InProgress,
    Completed,
}
";
        let decls = parse_rust_source(src);
        match decls.get("Status") {
            Some(Decl::Enum {
                variants,
                rename_all,
            }) => {
                assert_eq!(rename_all.as_deref(), Some("snake_case"));
                assert_eq!(variants.len(), 3);
                assert_eq!(variants[1].name, "InProgress");
            }
            other => panic!("attendu un enum Status, reçu {other:?}"),
        }
    }

    #[test]
    fn enum_with_data_is_unsupported() {
        let src = "\
#[derive(Serialize, Deserialize)]
pub enum NoteScope {
    Workspace,
    Module(String),
}
";
        let decls = parse_rust_source(src);
        assert!(matches!(
            decls.get("NoteScope"),
            Some(Decl::Unsupported { .. })
        ));
    }

    #[test]
    fn a_struct_without_serialize_is_ignored() {
        let src = "\
#[derive(Debug)]
pub struct NotSerialized {
    pub x: i32,
}
";
        assert!(parse_rust_source(src).is_empty());
    }

    #[test]
    fn render_struct_produces_an_interface() {
        let mut r = reg();
        r.insert(
            "Foo".to_string(),
            Decl::Struct {
                fields: vec![
                    Field {
                        name: "id".to_string(),
                        rust_type: "Uuid".to_string(),
                        optional: false,
                    },
                    Field {
                        name: "project_id".to_string(),
                        rust_type: "Option<Uuid>".to_string(),
                        optional: false,
                    },
                ],
            },
        );
        let (rendered, unsupported) = generate(&r, &[("Foo", "Foo")]);
        assert!(unsupported.is_empty());
        let text = rendered.get("Foo").unwrap();
        assert!(text.contains("export interface Foo {"));
        assert!(text.contains("  id: string"));
        assert!(text.contains("  project_id: string | null"));
    }
}
