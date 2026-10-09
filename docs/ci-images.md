# Images tierces de la CI (miroir ghcr.io)

## Pourquoi

Les jobs de CI qui démarrent `neo4j` et `meilisearch` tiraient leurs images
depuis Docker Hub sans authentification. Docker Hub limite ces téléchargements
(`toomanyrequests`), d'où des échecs intermittents sans rapport avec le code.
Les images sont donc copiées vers `ghcr.io/this-rs/ci-mirror/`, lu avec le
`GITHUB_TOKEN` éphémère des workflows (aucun secret à gérer).

## Fonctionnement

- `.github/ci-images.txt` liste les images : `source@sha256:<digest> cible`.
  Le digest est celui de l'index multi-architecture ; la copie le préserve.
- `.github/workflows/mirror-ci-images.yml` (`workflow_dispatch` + hebdomadaire,
  jamais sur pull request) copie chaque ligne avec
  `docker buildx imagetools create`, avec relances, puis vérifie que le digest
  de la cible est identique à celui de la source. Le dépôt doit être
  `this-rs/project-orchestrator` et chaque cible doit être sous
  `ghcr.io/<propriétaire>/ci-mirror/`.
- Les paquets sont PRIVÉS par défaut (Neo4j Community est sous GPL). Les rendre
  publics se fait dans les réglages du paquet sur GitHub.
- Le workflow n'existe pour GitHub qu'une fois fusionné sur `main`.

## Mettre à jour un digest

1. Relever le nouveau digest de l'index (lecture seule, sans tirer l'image) :
   `docker buildx imagetools inspect docker.io/library/neo4j:5` (champ `Digest`),
   et vérifier que `linux/amd64` figure dans les manifestes.
2. Modifier la ligne dans `.github/ci-images.txt` (digest, et tag cible).
3. Fusionner via PR, puis lancer :
   `gh workflow run mirror-ci-images.yml --ref main`
4. Mettre à jour les références dans `ci.yml` (voir ci-dessous).

## Passer la CI au miroir (PR 2, après le premier remplissage)

Dans `.github/workflows/ci.yml`, pour les trois jobs ayant `services:` :

1. Remplacer `image: neo4j:5` par
   `image: ghcr.io/this-rs/ci-mirror/neo4j:5.26.31@sha256:<digest>` et
   `image: getmeili/meilisearch:v1.34` par
   `image: ghcr.io/this-rs/ci-mirror/meilisearch:v1.34.3@sha256:<digest>`.
2. Ajouter `credentials: { username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }`
   à chaque service (paquet privé), et `packages: read` aux `permissions` du job.
3. Lier chaque paquet au dépôt (réglages du paquet, « Manage Actions access »)
   pour que le `GITHUB_TOKEN` puisse le lire.
4. Attention : les PR venant de forks n'ont pas ce droit ; si elles doivent
   passer la CI, rendre les paquets publics (en connaissance de la licence GPL).
