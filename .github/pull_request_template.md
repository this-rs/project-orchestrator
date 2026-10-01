<!--
Une PR = une tranche verticale : correctif + test de régression (rouge sans le correctif) + diagramme + doc.
Une case non cochée dans la checklist = PR non fusionnable.
-->

## Domaine

<!-- Domaine concerné (ex. chat, neo4j, auth, today, ...) -->

## Diagramme(s) touché(s)

<!--
Indiquer le fichier docs/diagrams/<nom>.mmd modifié,
ou « Diagram-Unchanged: <nom> — <raison> » si le diagramme existant reste exact,
ou « sans objet » si aucun diagramme ne couvre ce code.
-->

## Bug(s) corrigé(s)

<!-- Description du/des bug(s) et lien d'issue (Fixes #123). « Aucun » si la PR ne corrige pas de bug. -->

## Preuve du test de régression

<!--
Le test doit ÉCHOUER sans le correctif. Coller :
- la commande exécutée sans le correctif (correctif retiré/annulé),
- la sortie montrant l'échec,
- puis la même commande avec le correctif (verte).
-->

```text
```

## Couverture du module avant / après

<!--
Donner les deux chiffres quand ils existent : « brut » (tout le code du module)
ET « gated » (hors exclusions de la configuration de couverture). Voir docs/COVERAGE.md.
-->

| Module | Brut avant | Brut après | Gated avant | Gated après |
|---|---|---|---|---|
|  |  |  |  |  |

## Checklist

- [ ] CI verte (`cargo fmt --check, cargo clippy -- -D warnings, cargo test`)
- [ ] Test de régression présent et prouvé rouge sans le correctif (section ci-dessus)
- [ ] Doc à jour (et diagramme, ou `Diagram-Unchanged` / « sans objet » justifié)
- [ ] Pas de release, de tag ni de bump de version
- [ ] Poussé par `git push origin HEAD:refs/heads/<branche>` (jamais de push direct sur `main`)
