# Diagram publishing helper (mermaid.ffs.dev)

`publish.mjs` lints a Mermaid diagram locally, then creates/updates it through the
mermaid.ffs.dev API. Background and linter traps: note 47c540b9.

## Setup

    cd scripts/diagrams && npm install
    node publish.mjs fetch-vendor     # /vendor/mermaid.min.js is public; saved NEXT TO the script (gitignored)

## Commands

    node publish.mjs lint diagram.mmd                 # local only: negative control + trap checks + exact mermaid parse
    node publish.mjs create diagram.mmd --workspace <id> --session <s>
    node publish.mjs update <id> diagram.mmd --by <who>
    node publish.mjs archive <id>                     # endpoint unverified, confirm with an open session

Every create/update first runs the negative control (a deliberately broken diagram
MUST be rejected, otherwise the helper aborts), the trap checks (bare `%%`,
`state "x" as Y`), the JSDOM parse, then the server `/lint`.

## Authentication (the service is locked)

The service is locked until someone POSTs `/login {vault, passphrase}`. The helper
never handles the passphrase. Provide the resulting session cookie via
`MERMAID_COOKIE` or `MERMAID_COOKIE_FILE` (a file outside the repo). Never commit it.

## Read back exactly and verify

    curl -s -H "cookie: $MERMAID_COOKIE" "https://mermaid.ffs.dev/version?id=<id>&n=<v>" | jq -r .code
    # n+1 must not exist (expect 404):
    curl -s -o /dev/null -w "%{http_code}\n" -H "cookie: $MERMAID_COOKIE" "https://mermaid.ffs.dev/version?id=<id>&n=$((v+1))"

## Tests

    npm test
