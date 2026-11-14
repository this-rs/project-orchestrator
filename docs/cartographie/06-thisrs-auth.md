# 06 — Remplacer l'authentification de PO par WAMI (this-rs) : ce que ça coûte

Analyse faite en lecture seule le 10/11/2026. Aucun build, aucun git.
Abréviations : `T/` = `/Users/triviere/projects/this-rs/this/`, `B/` = `/Users/triviere/projects/project-orchestrator/backend/`.

> Correction du brief : il n'y a pas de fichier `core/auth/oidc.rs` dans this-rs. Le fichier de 970 lignes s'appelle `T/src/core/auth/oidc_provider.rs`.

---

## 1. Ce que WAMI fait exactement dans this-rs

### 1.1 Les trois modes
`T/src/config/auth.rs:40-50` définit `AuthMode` :
- **`sts`** (défaut, `:42-43`) : WAMI tourne comme un service séparé ; this-rs ne fait que **vérifier** les jetons. Il faut la clé publique et pas la clé privée (`:172-181`).
- **`embedded`** (`:45-46`) : le STS est dans le processus. Il expose les routes de login et de vérification et a besoin de la clé privée.
- **`bootstrap`** (`:48-49`) : WAMI minimal en mémoire, pour le développement. Une paire de clés est générée au démarrage si elle manque (`:173`, `:181`). L'audience y est facultative (`T/src/core/auth/wami_provider.rs:166-171`) et la fenêtre « legacy » ouverte par défaut (`T/src/config/auth.rs:207`).

### 1.2 Émission : `StsService` (`T/src/core/auth/sts.rs`)
- Clé Ed25519 en PEM PKCS8, chargée dans `wami::sts::jwt::KeyManager` (`sts.rs:151-160`). Chaque jeton porte le `kid` actif dans son en-tête (`sts.rs:445-452`).
- Le `kid` est l'empreinte RFC 7638 de la clé publique (`sts.rs:57-63`).
- **Le login n'accepte qu'un nom d'utilisateur et un mot de passe** (`LoginCredentials`, `sts.rs:76-80` ; `login`, `sts.rs:257-284`). Quand un annuaire est branché (trait `AnnuaireVerification`, argon2id, `sts.rs:102-114`), il est la seule source de vérité. Sans annuaire, la comparaison se fait en clair, ce qui n'est prévu que pour les tests (`sts.rs:117-124`).
- Un appel émet une **paire** de jetons : accès et rafraîchissement (`issue_token_pair`, `sts.rs:404-460`). Cette fonction est **`pub(crate)`** : aucun code hors du crate ne peut faire émettre un jeton pour une identité vérifiée ailleurs, par exemple après un login OIDC.
- Routes (`T/src/server/exposure/rest/auth_routes.rs:233-245`) : `POST /auth/token`, `GET /auth/keys` (JWKS), `POST /auth/refresh`, `POST /auth/revoke`, `GET|POST /auth/verify`. Cette dernière est une porte `forward_auth` pour un proxy inverse (`:126-200`).
- Le login pose un cookie `HttpOnly; SameSite=Lax; Path=/` qui contient le **jeton d'accès**, pas le jeton de rafraîchissement (`auth_routes.rs:51-62`, `:75-86`).

### 1.3 Vérification : `WamiAuthProvider` (`T/src/core/auth/wami_provider.rs`)
- Une seule clé publique Ed25519, en PEM ou en base64 brut de 32 octets (`:78-80`, `:200-232`). Elle ne lit pas de JWKS et ne choisit pas la clé selon le `kid`.
- `aud` est obligatoire en mode `sts` et en mode `embedded` (`:158-165`). L'expiration est toujours vérifiée (`:176`).
- Un jeton de rafraîchissement ne peut pas authentifier une requête (`:250-251`). Un jeton sans `aud` n'est accepté que pendant la fenêtre legacy (ADR D4, `:253-262`).
- Correspondance avec `AuthContext` (`:284-320`) :
  - `is_admin` donne `Admin` ;
  - `service_name` donne `Service` ;
  - sinon `User`, qui exige un `tenant_id` UUID (refus explicite sinon, `:306-308`) ;
  - `scopes` est toujours vide pour un jeton WAMI (`:316`).
- Le fournisseur **ne consulte pas la liste de révocation** : le mot « revok » n'apparaît nulle part dans `wami_provider.rs`.

### 1.4 Politiques `perm:` (`T/src/core/auth/wami_policy.rs`)
- Point d'entrée synchrone `PermissionFn = Fn(&str) -> Option<bool>` (`:1-14`). `perm:orders:write` se traduit en `this:orders:write` (`:55`, `:86`).
- Règles de combinaison (ADR D2, `:6-11`) :
  - le refus explicite l'emporte ;
  - sans autorisation, c'est un refus implicite ;
  - une limite de permissions (*boundary*) restreint et n'accorde jamais.
- Le résultat `Verdict::Unevaluable` correspond à `None` et doit être traité comme un refus (`:58-76`).
- Ressources désignées par ARN : `…:orders/<uuid>`, `…:link/owns/users/orders`, `…:resource/*` (`:16-23`). Les rôles du jeton correspondent à des ensembles de documents (`:25-29`).
- La notion de **propriétaire** (`owner`) reste du côté de `PolicyExpr` : WAMI ne sait pas qui possède une ligne (`:31-35`).
- **Limite écrite dans le code** : les documents et le registre d'identités sont en mémoire (`:36-38`) ; il faut persister soi-même (`identity_path`, `T/src/config/auth.rs:224-228`).
- Passerelle UUID ↔ segment `u64` de tenant, avec allocation sans réutilisation (`T/src/core/auth/wami_bridge.rs:9-25`).

### 1.5 `T/src/storage/iam.rs`
Ce fichier n'est **pas** du WAMI. C'est une ACL matérialisée sur Obrain : l'union des permissions est calculée à l'écriture, puis lue par index de propriété (`iam.rs:1-21`), avec une époque pour détecter le périmé (`:31-35`). Il « ne dit pas quelles permissions sont légitimes » (`:37-41`). C'est un accélérateur de décision, pas une source d'identité.

### 1.6 OIDC (`T/src/core/auth/oidc_provider.rs`)
- **Serveur de ressources seulement** : il vérifie des jetons émis ailleurs et n'en émet jamais (ADR D11, `:1-6`).
- Découverte paresseuse ; JWKS en cache 5 minutes, relu une seule fois sur un `kid` inconnu (`:52-56`, `:204-240`).
- Il refuse `HS*` et `none` (`:40-50`), un mauvais `aud` ou un mauvais `iss`, et un jeton sans tenant UUID (`:16-23`, `:367-378`).
- Un `sub` opaque devient un UUID v5 calculé **sans l'émetteur** : deux fournisseurs d'identité qui émettent le même `sub` donnent le même utilisateur (`:356-364`).

---

## 2. Le modèle de jetons : this-rs comparé à PO

| | this-rs (WAMI) | PO aujourd'hui |
|---|---|---|
| Algorithme | EdDSA / Ed25519, asymétrique, `kid` (`sts.rs:151-160`) | **HS256** à secret partagé ; tout autre algorithme est refusé par le fournisseur cryptographique (`B/src/auth/jwt_crypto.rs:1-11`, `:108-123`) |
| Revendications | `sub, tenant_id, roles, exp, iat, jti, iss, aud, token_use` (`sts.rs:16-45`) | `sub, email, name, iat, exp, token_type, scope, jti` (`B/src/auth/jwt.rs:27-50`). Pas de `aud`, pas d'`iss`, pas de tenant |
| Durée de l'accès | 15 min avec `StsService::new` (`sts.rs:165`), mais **3600 s** par défaut dans la configuration (`T/src/config/auth.rs:198-199`, `:225-227`) | 900 s (`B/src/lib.rs:644-647`, `:743-745`) |
| Rafraîchissement | **JWT** signé, valable 24 h, rotation par révocation du `jti` (`sts.rs:166`, `:290-323`) | **Valeur opaque** de 256 bits, **stockée hachée** (SHA-256) en base, dans un cookie `HttpOnly; Path=/` ; 7 jours ; rotation à usage unique et atomique (`B/src/auth/refresh.rs:1-35`, `B/src/lib.rs:648-651`, `:747-749`, `B/src/api/auth_handlers.rs:700-770`) |
| Révocation | `HashSet<jti>` **en mémoire**, perdu au redémarrage (`sts.rs:138`, `:326-338`). Consultée par `/auth/verify` et `/auth/refresh` seulement (`auth_routes.rs:189`), **pas par le middleware** (§1.3). `POST /auth/revoke` prend n'importe quel `jti` sans vérifier à qui il appartient (`auth_routes.rs:116-125`) | Jetons MCP : `jti` vérifié en Neo4j **à chaque requête**, refus si la base ne répond pas (`B/src/auth/middleware.rs:446-468`). Jetons d'agent : liste d'autorisation, le jeton meurt avec sa session (`B/src/auth/agent_tokens.rs:1-12`, `middleware.rs:470-480`). Rafraîchissement : hash supprimé à la déconnexion (`auth_handlers.rs:784`) |
| Multi-tenant | Natif : `tenant_id` obligatoire, segments WAMI (`wami_provider.rs:306-308`, `wami_bridge.rs`) | Aucun tenant ; restriction par `allowed_email_domain` et `allowed_emails` (`B/src/lib.rs:652-657`, `middleware.rs:440-444`) |
| Jetons longs | Aucun | Jeton MCP : 90 jours par défaut, 365 au plus (`auth_handlers.rs:975-977`, `:1028-1039`). Jeton OAuth : 30 jours (`B/src/auth/oauth_server.rs:62-65`). Jeton de session d'agent : 24 h (`B/src/chat/config.rs:221-222`, `:425`, `:674`). Jetons de coffre et de profil d'outils (`jwt.rs:331-357`, `B/src/auth/tool_profile.rs:1-20`) |

---

## 3. Ce qui casse

1. **Pas de serveur d'autorisation OAuth.** PO est son propre serveur d'autorisation OAuth 2.1 pour les connecteurs MCP de Claude.ai :
   - métadonnées RFC 8414 et RFC 9728 ;
   - enregistrement dynamique des clients RFC 7591, avec des `client_id` sans état signés en HS256 ;
   - code d'autorisation + PKCE S256, page de consentement (`oauth_server.rs:1-35`, `:104-107`, `:390-480`).

   La recherche de `authorization_code`, `code_challenge`, `PKCE` et `/authorize` dans `T/src` ne trouve rien. L'ADR D11 exclut ce rôle volontairement (`oidc_provider.rs:1-6`, `T/docs/oidc-resource-server.md` §1).
2. **Pas de client OIDC.** PO mène le flux complet comme client : redirection, échange du code, `userinfo`, avec Google, Microsoft, Okta ou Keycloak (`B/src/auth/oidc.rs:1-7`, `:197-290` ; `auth_handlers.rs:469`, `:571`). this-rs ne sait que **vérifier** le jeton d'un fournisseur d'identité, et son STS ne sait émettre qu'à partir d'un mot de passe (§1.2). Même avec un login OIDC fait par PO, `issue_token_pair` est `pub(crate)` : PO devrait signer lui-même avec `KeyManager`, ce qui le ferait redevenir émetteur.
3. **Pas d'équivalent aux jetons d'agent.**
   - `AuthContext::Service` existe côté vérification (`wami_provider.rs:290-296`), mais le STS n'en émet jamais : il ne produit que des paires utilisateur (`sts.rs:404-460`).
   - Il n'y a ni liaison à une session (`AgentSessionBinding`, `jwt.rs:187-231`), ni plafond de permission, ni profil d'outils signé, ni liste d'autorisation qui tue le jeton à la fermeture de la session.
   - Rien non plus pour les jetons MCP longs, révocables et inventoriés.
4. **La révocation recule.** Avec WAMI, un `jti` révoqué continue d'ouvrir toutes les routes protégées par le middleware jusqu'à son `exp`, et la liste de révocation est perdue au redémarrage (§2). Pour PO, dont les jetons durent 90 jours, ce serait une régression de sécurité.
5. **Le modèle d'identité change.** WAMI exige un `tenant_id` et ne porte pas `email` : PO perd le filtre par domaine (`middleware.rs:440`), sauf à ajouter une revendication. Il n'y a pas non plus de `/auth/register` (PO : `auth_handlers.rs:340`), pas d'équivalent au ticket WebSocket (`auth_handlers.rs:840`), et pas de mot de passe racine bcrypt (`B/src/lib.rs:848-856`).

---

## 4. Le chemin de migration

**Ce qui est incompatible au niveau des octets.** Un vérificateur WAMI n'accepte que EdDSA. Un jeton PO HS256 y est refusé à la signature, et la fenêtre legacy de l'ADR D4 n'y change rien : elle tolère seulement l'absence d'`aud` et de `token_use`, pas un autre algorithme (`T/docs/wami-core-migration.md:79-84`, `wami_provider.rs:253-262`). Aucun jeton existant de PO ne peut donc être accepté tel quel. Il faut **réémettre**, ou **vérifier les deux formats** pendant une période.

**Ce qui se réémet de lui-même, sans déconnecter personne :**
- **Humains.** Le cookie de rafraîchissement de PO est opaque et vit en base (`refresh.rs:1-13`). Il n'a pas à devenir un JWT. Au prochain `POST /auth/refresh`, PO consomme le cookie comme aujourd'hui et rend un **jeton d'accès Ed25519** à la place du HS256. Au bout de 15 minutes (la durée de vie d'un accès), tous les clients actifs ont changé de format **sans nouveau login**.
- **Jetons de session d'agent.** Ils sont déjà invalidés à chaque redémarrage et refrappés à chaque reprise (`agent_tokens.rs:9-12`). Ils se migrent au premier démarrage, sans coût.

**Ce qui ne se réémet pas :** les jetons MCP manuels (jusqu'à 365 jours) et ceux d'OAuth (30 jours). Ils sont posés dans la configuration des connecteurs et des outils tiers. Les invalider force chaque utilisateur à refaire la connexion de ses connecteurs.

**La plus petite migration qui ne déconnecte personne deux fois :**
1. **Double vérification dans le middleware de PO**, en un seul lot :
   - EdDSA, avec `aud`, `token_use` et un `iss` attendu, en premier ;
   - HS256 hérité ensuite, **seulement** si `token_type` vaut `mcp` ou si le `jti` est connu du magasin MCP ;
   - la vérification Neo4j de révocation reste la même pour les deux formats.
2. L'émission des accès humains et des jetons d'agent passe en Ed25519 dès le même lot. Les rafraîchissements opaques restent inchangés.
3. Les **nouveaux** jetons MCP et OAuth sont émis en Ed25519. Les anciens vivent jusqu'à leur `exp` naturel. La branche HS256 est retirée quand le dernier jeton MCP actif en Neo4j a expiré (au plus 365 jours, ou plus tôt après une relance demandée aux utilisateurs).
4. Le serveur OAuth, le client OIDC et la logique d'agent **restent dans PO**. On ne remplace que la *signature* et la *vérification*.

Une rupture franche (tout invalider) déconnecte tout le monde une fois et casse tous les connecteurs Claude.ai. Une coexistence sans double vérification les déconnecte deux fois. Le plan ci-dessus ne déconnecte personne.

---

## 5. Ce que chaque côté a et que l'autre n'a pas

**Ce que PO a et que WAMI n'a pas** (le retirer serait une régression) :
- serveur d'autorisation OAuth 2.1 avec enregistrement dynamique, PKCE S256 et consentement (`oauth_server.rs`) ;
- client OIDC multi-fournisseurs, avec la configuration Google héritée (`oidc.rs`, `google.rs`) ;
- inscription, mot de passe racine bcrypt, ticket WebSocket ;
- rafraîchissement opaque haché et persistant, avec rotation atomique ;
- jetons MCP longs, à portée limitée, inventoriés, **révoqués à chaque requête** depuis une base persistante ;
- jetons d'agent liés à une session, avec plafond de permission, profil d'outils signé, sortie de trust (`agent_tokens.rs:23-27`) et lignée tierce ;
- jetons de coffre limités à un préfixe de chemin (`middleware.rs:17`) ;
- filtre par domaine d'e-mail.

**Ce que WAMI a et que PO n'a pas :**
- signature asymétrique Ed25519, JWKS et `kid` : un tiers peut vérifier sans détenir de secret capable d'émettre ;
- `aud` et `token_use`, qui séparent l'accès du rafraîchissement ;
- porte `forward_auth` (`/auth/verify`) pour un proxy ;
- multi-tenant natif ;
- moteur de politiques à refus prioritaire, avec limites de permissions et ARN de ressource (`wami_policy.rs`) ;
- ACL matérialisée en O(1) indexé (`iam.rs`) ;
- vérification OIDC de serveur de ressources avec rotation des JWKS ;
- annuaire de mots de passe en argon2id.

**Limites propres à WAMI à prendre en compte :**
- révocation en mémoire et ignorée par le middleware ;
- `/auth/revoke` sans contrôle de propriétaire ;
- une seule clé de vérification, donc pas de rotation à plusieurs clés côté vérificateur ;
- politiques et registre d'identités en mémoire, sauf `identity_path` ;
- durée par défaut incohérente (900 s dans `StsService::new`, 3600 s dans la configuration).

---

## Verdict

Un **remplacement complet** de `backend/src/auth/` par WAMI serait une **régression** : il ferait perdre le serveur OAuth (connecteurs Claude.ai), le login OIDC, les jetons d'agent et la révocation persistante.

Le gain réel est plus étroit : adopter **le format de jeton de WAMI** (Ed25519, `kid`, `aud`, `token_use`, JWKS) et éventuellement **son moteur `perm:`**, en gardant dans PO :
- les flux d'identité (OAuth, OIDC, mot de passe) ;
- le rafraîchissement opaque ;
- les magasins de révocation.

Migration des jetons existants : **double vérification HS256 et EdDSA**. Les humains et les agents basculent sans s'en apercevoir en 15 minutes, et la branche HS256 ne sert plus qu'aux jetons MCP et OAuth jusqu'à leur expiration naturelle. Personne n'est déconnecté.
