#!/usr/bin/env python3
"""Sonde de coût de contexte — GIFLUX-A (plan b00d80c1, tâche GEO-1).

CE QU'ELLE LIT : les transcripts JSONL écrits par le CLI Claude Code sous
`~/Library/Application Support/project-orchestrator/chat-neutral-<uuid>/`
(surchargeables par PO_TRANSCRIPTS). Lecture seule, aucune API, aucun secret.

CE QU'ELLE NE LIT PAS : le graphe. La jointure par tâche / run / provider
demande Neo4j et reste à faire (GEO-1).

CE QUE CHAQUE NOMBRE EST (la sonde étiquette chaque ligne) :
  [M] MESURÉ     — lu tel quel dans le transcript (usage par requête, cost-state).
  [R] RECONSTRUIT— dérivé par un modèle explicite (résidence FIFO, prix résolus).
  [S] SUPPOSÉ    — barème ou convention non prouvés sur ce corpus.

TROIS ERREURS DÉJÀ COMMISES ICI, À NE PAS REFAIRE :
 1. `result.usage` est CUMULATIF sur le tour : on agrège par requête dédupliquée
    par `requestId`, jamais par somme de tour.
 2. Un volume INJECTÉ ne se compare pas à un volume RELU. Ce qui coûte est la
    RÉSIDENCE : taille du bloc × nombre de requêtes qui l'ont encore.
 3. Une image se facture au PIXEL, pas à la longueur de son base64.

USAGE :
    python3 scripts/geo_usage_probe.py            # rapport
    python3 scripts/geo_usage_probe.py --check    # assertions, sortie non nulle si faux
    python3 scripts/geo_usage_probe.py --selftest  # fixtures synthétiques, hors corpus
"""
import json, glob, os, sys, collections, tempfile

TRANSCRIPTS = os.environ.get(
    "PO_TRANSCRIPTS",
    os.path.expanduser("~/Library/Application Support/project-orchestrator/chat-neutral-*"))
# [S] Barème NON validé. L'ajustement tombe à 0,2 % sur ses propres données : c'est
# du surapprentissage, pas une preuve. En leave-one-out l'erreur empirique de
# prédiction est de 12,7 % — une erreur de prédiction ponctuelle, PAS un intervalle
# de confiance. Les prix déclarés par le provider restent la référence.
R_CACHE_WRITE, R_CACHE_READ, R_OUTPUT = 1.25, 0.10, 5.00
CHARS_PER_TOKEN = 3.6                                       # [S] ordre de grandeur
SEP = "\n\n---\n\n"                                         # frontière enrichissement PO / texte


def sessions(root=None):
    for d in sorted(glob.glob(root or TRANSCRIPTS)):
        for f in glob.glob(os.path.join(d, "*.jsonl")):
            yield os.path.basename(d).split("chat-neutral-")[-1], f


def scan(f):
    """Compte les requêtes, le coût réel, la forme du préfixe, les blobs système."""
    req, api_ids, turns, shapes, blobs, cost = set(), set(), set(), {}, [], None
    for line in open(f, errors="ignore"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        if t == "assistant":
            if (o.get("message") or {}).get("usage") and o.get("requestId"):
                req.add(o["requestId"])
        elif t == "api-request":
            if o.get("id"):
                api_ids.add(o["id"])
        elif t == "api-request-blob":
            blobs.append((o.get("hash"), len(json.dumps(o.get("message") or ""))))
        elif t == "user" and not o.get("isSidechain"):
            turns.add(o.get("promptId"))
        elif t == "api-request-shape":
            sh = o.get("shape") or {}
            tools = sh.get("tools") or []
            system = sum(len(x.get("text", "")) for x in (sh.get("system") or [])
                         if isinstance(x, dict))
            shapes[o.get("shapeHash")] = (system + len(json.dumps(tools))) / CHARS_PER_TOKEN
        elif t == "cost-state":
            cost = o
    return {"req": req, "api_ids": api_ids, "turns": turns, "shapes": shapes,
            "blobs": blobs, "cost": cost}


def residence(f, floor):
    """[R] Coût de résidence par catégorie, simulation FIFO des compactions.

    La taille en jetons d'un bloc vient du DELTA observé du préfixe (jetons
    réels), réparti au prorata des caractères des blocs ajoutés entre deux
    requêtes. Ce qui dépasse le contexte observé sort par le début (FIFO) :
    hypothèse explicite, la compaction oublie le plus ancien d'abord.
    Les blocs image ne sont jamais comptés ici (facturation au pixel).
    """
    blocks, resid = [], collections.Counter()
    names, prev, pending = {}, None, []
    for line in open(f, errors="ignore"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        if t == "assistant":
            m = o.get("message") or {}
            pending = []
            for b in (m.get("content") or []) if isinstance(m.get("content"), list) else []:
                k = b.get("type")
                if k == "text":
                    pending.append(("assistant_texte", len(b.get("text") or "")))
                elif k == "thinking":
                    pending.append(("raisonnement", len(b.get("thinking") or "")))
                elif k == "tool_use":
                    names[b.get("id")] = b.get("name") or "?"
                    pending.append((f"args:{b.get('name') or '?'}",
                                    len(json.dumps(b.get("input") or ""))))
            u = m.get("usage")
            if not u:
                continue
            ctx = ((u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                   + (u.get("cache_creation_input_tokens") or 0))
            if prev is None:
                prev = ctx
                continue
            delta = ctx - prev
            if delta > 0:
                total = sum(c for _, c in pending) or 1
                for cat, chars in pending:
                    blocks.append([cat, delta * chars / total])
            for cat, size in blocks:
                resid[cat] += size
            excess = (floor + sum(s for _, s in blocks)) - ctx
            while excess > 0 and blocks:
                _, size = blocks.pop(0)
                excess -= size
            prev = ctx
        elif t == "user":
            ct = (o.get("message") or {}).get("content")
            items = ct if isinstance(ct, list) else [{"type": "text", "text": ct or ""}]
            for b in items:
                if b.get("type") == "tool_result":
                    blocks.append(["resultat:" + names.get(b.get("tool_use_id"), "?"),
                                   len(json.dumps(b.get("content"))) / CHARS_PER_TOKEN])
                elif b.get("type") == "text":
                    txt = b.get("text") or ""
                    head, sep, rest = txt.partition(SEP)
                    if sep:
                        blocks.append(["enrichissement_PO", len(head) / CHARS_PER_TOKEN])
                        blocks.append(["texte_utilisateur", len(rest) / CHARS_PER_TOKEN])
                    else:
                        blocks.append(["texte_utilisateur", len(txt) / CHARS_PER_TOKEN])
    return resid


def units(mu):
    return (mu.get("inputTokens", 0) + R_CACHE_READ * mu.get("cacheReadInputTokens", 0)
            + R_CACHE_WRITE * mu.get("cacheCreationInputTokens", 0)
            + R_OUTPUT * mu.get("outputTokens", 0)) / 1_000_000


def solve_prices(rows):
    """[R] Prix d'entrée par modèle, moindres carrés sur la facture réelle."""
    models = sorted({m for r in rows for m in r["mu"] if m != "<synthetic>"})
    if not models or not rows:
        return {}, []
    A = [[units(r["mu"].get(m, {})) for m in models] for r in rows]
    b = [r["usd"] for r in rows]
    n = len(models)
    ata = [[sum(k[i] * k[j] for k in A) for j in range(n)] for i in range(n)]
    atb = [sum(k[i] * v for k, v in zip(A, b)) for i in range(n)]
    M = [row[:] + [atb[i]] for i, row in enumerate(ata)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                fac = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= fac * M[c][k]
    prices = {m: (M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else float("nan"))
              for i, m in enumerate(models)}
    err = [abs(sum(A[k][i] * prices[m] for i, m in enumerate(models)) - b[k])
           for k in range(len(b))]
    return prices, err


def report(rows, label=""):
    prices, err = solve_prices(rows)
    total = sum(r["usd"] for r in rows)
    rel = 100 * sum(err) / max(1e-9, total)
    print(f"[R] prix dérivés ({label}) : "
          + "  ".join(f"{m.replace('claude-','')}={v:.2f}" for m, v in prices.items())
          + f"   erreur={rel:.1f}%")
    return rel


def leave_one_session_out(rows):
    """Estime le coût d'un transcript avec les prix ajustés sur d'autres sessions."""
    by_session = collections.defaultdict(list)
    for row in rows:
        by_session[row["sid"]].append(row)

    error_usd = comparable_usd = scored = unnotable = singular = 0
    for sid, held_out in by_session.items():
        prices, _ = solve_prices([row for row in rows if row["sid"] != sid])
        for row in held_out:
            needed = {model for model, usage in row["mu"].items()
                      if model != "<synthetic>" and units(usage) > 0}
            if not needed or not needed.issubset(prices):
                unnotable += 1
                continue
            if any(not 0 <= prices[model] < float("inf") for model in needed):
                singular += 1
                continue
            estimated = sum(units(row["mu"][model]) * prices[model] for model in needed)
            error_usd += abs(estimated - row["usd"])
            comparable_usd += row["usd"]
            scored += 1

    error_pct = 100 * error_usd / comparable_usd if comparable_usd else None
    return error_pct, scored, unnotable, singular


def selftest():
    """Fixtures synthétiques : aucune dépendance au corpus réel."""
    d = tempfile.mkdtemp()
    f = os.path.join(d, "s.jsonl")
    use = lambda i, cr, cw, o: {"input_tokens": i, "cache_read_input_tokens": cr,
                                "cache_creation_input_tokens": cw, "output_tokens": o}
    lines = [
        {"type": "api-request", "id": "A"},
        {"type": "api-request", "id": "B"},
        {"type": "api-request", "id": "ORPHELIN"},          # tentative sans usage
        {"type": "api-request-blob", "hash": "h1", "message": {"role": "user"}},
        {"type": "api-request-shape", "shapeHash": "sh",
         "shape": {"system": [{"text": "x" * 3600}], "tools": [{"name": "t"}]}},
        {"type": "user", "promptId": "p1", "isSidechain": False,
         "message": {"content": "enrichissement" + SEP + "question"}},
        {"type": "assistant", "requestId": "A", "message": {
            "model": "m1", "usage": use(0, 1000, 100, 10),
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "y" * 360},
            {"type": "image", "source": {"data": "z" * 100000}}]}},
        {"type": "assistant", "requestId": "B", "message": {
            "model": "m1", "usage": use(0, 1200, 50, 20), "content": [{"type": "text", "text": "ok"}]}},
        {"type": "assistant", "requestId": "B", "message": {
            "model": "m1", "usage": use(0, 1200, 50, 20), "content": []}},   # doublon
        {"type": "cost-state", "totalCostUSD": 1.0,
         "modelUsage": {"m1": {"inputTokens": 0, "outputTokens": 20,
                               "cacheReadInputTokens": 1200, "cacheCreationInputTokens": 50}}},
        {"type": "cost-state", "totalCostUSD": 1.0,
         "modelUsage": {"m2": {"inputTokens": 0, "outputTokens": 20,
                               "cacheReadInputTokens": 1200, "cacheCreationInputTokens": 50}}},
    ]
    with open(f, "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")
    s = scan(f)
    fails = []
    if len(s["req"]) != 2:
        fails.append(f"dédup par requestId : attendu 2, obtenu {len(s['req'])}")
    orph = s["api_ids"] - s["req"]
    if orph != {"ORPHELIN"}:
        fails.append(f"tentatives sans usage : attendu {{'ORPHELIN'}}, obtenu {orph}")
    if not any(abs(v - 3600 / CHARS_PER_TOKEN) < 20 for v in s["shapes"].values()):
        fails.append(f"plancher mal calculé : {s['shapes']}")
    r = residence(f, floor=1000)
    if "enrichissement_PO" not in r:
        fails.append("l'enrichissement PO n'est pas séparé du texte utilisateur")
    if "resultat:Bash" not in r:
        fails.append("le résultat d'outil n'est pas attribué à son outil (tool_use_id)")
    if any("image" in c for c in r):
        fails.append("une image a été comptée comme du texte (facturation au pixel)")
    if r["enrichissement_PO"] <= 0:
        fails.append("résidence d'enrichissement nulle alors qu'un bloc existe")
    # 1 M et 2 M de cache_read à 0,10 u/M => 0,1 et 0,2 : prix d'entrée 1,0
    rows = [{"mu": {"m1": {"inputTokens": 0, "outputTokens": 0,
                           "cacheReadInputTokens": 1_000_000, "cacheCreationInputTokens": 0}},
             "usd": 0.10},
            {"mu": {"m1": {"inputTokens": 0, "outputTokens": 0,
                           "cacheReadInputTokens": 2_000_000, "cacheCreationInputTokens": 0}},
             "usd": 0.20}]
    p, err = solve_prices(rows)
    if not abs(p.get("m1", 0) - 1.0) < 0.01:
        fails.append(f"prix mal résolu : attendu 1.0, obtenu {p.get('m1')}")
    # Deux factures à usage égal. L'ajustement à 6 % masque une erreur de
    # prédiction à 12 %. Un autre fichier du même dossier doit rester exclu.
    usage = {"m1": {"inputTokens": 1_000_000}}
    holdout = [{"sid": "a", "usd": 0.94, "mu": usage},
               {"sid": "b", "usd": 1.06, "mu": usage}]
    _, fit_errors = solve_prices(holdout)
    error_pct, scored, _, _ = leave_one_session_out(holdout)
    if not (sum(fit_errors) / 2 < 0.10 and error_pct is not None
            and abs(error_pct - 12) < 0.01 and scored == 2):
        fails.append("l'ajustement et la validation hors session ne sont pas distingués")
    error_pct, scored, _, _ = leave_one_session_out([holdout[0], holdout[0]])
    if error_pct is not None or scored:
        fails.append("un autre fichier de la session a servi à ajuster ses propres prix")
    for msg in fails:
        print("ÉCHEC:", msg)
    print("selftest OK" if not fails else f"selftest : {len(fails)} échec(s)")
    return 1 if fails else 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    check = "--check" in sys.argv
    rows = []
    for sid, f in sessions():
        s = scan(f)
        usd = (s["cost"] or {}).get("totalCostUSD") or 0
        if not s["cost"] or not s["req"] or usd <= 0:
            continue
        rows.append({"sid": sid, "path": f, "n": len(s["req"]), "api": len(s["api_ids"]),
                     "turns": len(s["turns"]), "usd": usd,
                     "mu": s["cost"].get("modelUsage") or {},
                     "inter": len(s["api_ids"] & s["req"]),
                     "blobs": s["blobs"],
                     "floor": (sorted(s["shapes"].values())[len(s["shapes"]) // 2]
                               if s["shapes"] else 0)})
    if not rows:
        print("aucun transcript trouvé", file=sys.stderr)
        return 2
    total = sum(r["usd"] for r in rows)
    print(f"[M] sessions={len(rows)}  requêtes uniques={sum(r['n'] for r in rows)}  "
          f"tours={sum(r['turns'] for r in rows)}  coût réel={total:.2f} USD")
    api_n, inter = sum(r["api"] for r in rows), sum(r["inter"] for r in rows)
    print(f"[M] lignes api-request={api_n}  requêtes avec usage={sum(r['n'] for r in rows)}  "
          f"identifiants communs={inter}")
    if inter == 0:
        print("[M] les deux compteurs ne sont PAS joignables par identifiant : l'écart "
              f"({api_n - sum(r['n'] for r in rows):+d}) est un écart de VOLUME. "
              "Rien ne prouve que ce sont des reprises.")
    allb = [b for r in rows for b in r["blobs"]]
    if allb:
        hashes = {h for h, _ in allb}
        sizes = sorted(sz for _, sz in allb)
        print(f"[M] blobs d'injection système : lignes={len(allb)}, empreintes distinctes={len(hashes)}, "
              f"taille médiane={sizes[len(sizes) // 2]:,} carr. (non attribuables par requête depuis ces lignes)")
    rows.sort(key=lambda r: r["sid"])
    error_pct, scored, unnotable, singular = leave_one_session_out(rows)
    error_label = f"{error_pct:.1f}%" if error_pct is not None else "indisponible"
    print(f"[R] leave-one-out par session : transcriptions comparables={scored}/{len(rows)}, "
          f"usages non comparables={unnotable}, ajustements inutilisables={singular}, "
          f"erreur={error_label}")
    print("[R] NB : l'erreur sur le corpus complet (ci-dessous) n'est PAS une validation, "
          "c'est un ajustement sur ses propres données.")
    report(rows, "corpus complet")

    big = max(rows, key=lambda r: r["usd"])
    res = residence(big["path"], big["floor"])
    tots = sum(res.values())
    print(f"\n[R] RÉSIDENCE — session {big['sid'][:8]} ({big['usd']:.2f} USD, {big['n']} requêtes, "
          f"plancher {big['floor']:,.0f} j)")
    for cat, v in res.most_common(8):
        print(f"   {cat:44} {v/1e6:7.1f} M jetons-lus  {100*v/tots:5.1f}%")

    if check:
        fails = []
        if scored != len(rows):
            fails.append(f"recoupement hors session incomplet : {scored}/{len(rows)} transcriptions")
        if error_pct is not None and error_pct > 10:
            fails.append(f"erreur de coût hors session à {error_pct:.1f}% (>10%)")
        if not (1 <= tots / max(1, big["n"] * big["floor"]) <= 60):
            fails.append("résidence incohérente avec le contexte observé")
        for f_ in fails:
            print("ÉCHEC:", f_)
        print("OK" if not fails else f"{len(fails)} échec(s)")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
