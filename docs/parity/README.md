# `docs/parity/` — l'inventaire opposable du contrat de l'API

`inventory.tsv` est GÉNÉRÉ par `cargo xtask parity --write`, jamais écrit à la
main. Sources : `src/api/routes.rs` (production seule, avant le premier
`#[cfg(test)]`) et `MEGA_TOOL_ACTIONS` (`src/mcp/handlers.rs`). Colonnes : kind,
path_or_tool, action, file:line, puis tier=/h= (route) ou internal= (action).
Par défaut `--check` : régénère en mémoire et compare octet à octet (0 identique,
1 écart constaté, 2 non vérifiable).

CE QU'IL PROUVE : que le contrat a bougé et que personne ne l'a dit — une route
ou une action ajoutée, retirée, renommée ou déplacée rend la CI rouge, et une
ligne écrite à la main aussi (second comptage indépendant des `.route(` : un
parseur qui perdrait des routes ne passe pas en silence).

CE QU'IL NE PROUVE PAS : que le routeur sert ce qu'il monte, ni qu'un corps de
réponse corresponde (c'est le rejeu de `tests/parity.rs`, qui exige le vrai
routeur) ; il ne compare pas les `enum` de `src/mcp/tools.rs` (déjà testé côté
backend) ; et un commentaire citant `.route(` le fait sortir en 2 — c'est voulu :
mieux vaut un faux rouge qu'un faux vert.
