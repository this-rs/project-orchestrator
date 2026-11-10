# XTASK-TS-DTO — formes non supportées

Généré par `cargo xtask ts-dto --write`. Le parseur std ne lit que les formes
courantes : `struct` (champs simples), `enum` sans données, `Option<T>` →
`T | null`, `Vec<T>` → `T[]`, `HashMap<K, V>` → `Record<K, V>`,
`#[serde(rename_all)]`, `#[serde(default)]` → champ optionnel. Toute autre forme
est LISTÉE ici plutôt que devinée ; le champ correspondant est engendré comme
`unknown`.

| Où | Raison |
|---|---|
| Note.scope | NoteScope : enum avec variantes à données (union discriminée non engendrée) |
| NoteScope | enum avec variantes à données (union discriminée non engendrée) |
