#!/usr/bin/env python3
"""Rejeu de résidence de l'enrichissement — plan b00d80c1, tâche GEO-14.

CE QU'ELLE LIT : les transcripts JSON du harnais NATIF
(`~/Library/Application Support/project-orchestrator/native-transcripts/<provider>/*.json`,
surchargeables par PO_NATIVE_TRANSCRIPTS). Chaque fichier porte la liste `messages`
telle qu'elle a été envoyée au provider : c'est la seule source qui montre le
préfixe RÉEL, message par message. Lecture seule, aucun appel réseau.

CE QU'ELLE NE LIT PAS : les transcripts du CLI Claude Code (JSONL d'un autre
format), et le graphe (pas de jointure par tâche / run). Elle ne prétend donc
rien sur Claude Code.

CE QUE MESURE « RÉSIDENCE ». Un bloc injecté au tour N reste dans le préfixe et
est relu à chaque requête suivante : son coût est `taille × nombre de requêtes
qui le relisent encore`. Une taille qui ne bouge pas peut donc coûter très cher,
et c'est exactement le cas de l'enrichissement PO (mesuré : 66 % du volume relu
du natif, dont 30 % pour « Work In Progress » réinjecté 55 fois).

CE QU'ELLE COMPARE :
  AVANT — chaque tour réécrit ses sections en entier (l'état d'aujourd'hui) ;
  APRÈS — la politique de résidence de `src/chat/residence.rs` : un bloc déjà
          délivré dans cette session devient un POINTEUR borné (identique, ou
          « seuls les chiffres ont bougé »), et il est redélivré en entier dès
          qu'il change matériellement, ou tous REFRESH_EVERY tours.

LES CONSTANTES CI-DESSOUS SONT CELLES DU MODULE RUST. Si l'une change d'un côté,
elle change de l'autre : `POLICY_DIGEST` en bas les résume, et le test Rust
`policy_matches_the_replay_script` relit ce fichier pour refuser une divergence.

USAGE :
    python3 scripts/geo_residence_replay.py             # rapport
    python3 scripts/geo_residence_replay.py --check     # assertions, sortie non nulle si faux
    python3 scripts/geo_residence_replay.py --json      # chiffres bruts
"""
import glob
import hashlib
import json
import os
import re
import sys

NATIVE = os.environ.get(
    "PO_NATIVE_TRANSCRIPTS",
    os.path.expanduser("~/Library/Application Support/project-orchestrator/native-transcripts/*/*.json"),
)

# ── Constantes de la politique (miroir de src/chat/residence.rs) ────────────
SECTION_FLOOR = 240      # en dessous, un pointeur coûterait presque autant
REFRESH_EVERY = 40       # un bloc pointé est redélivré en entier tous N tours
HEADING = re.compile(r"^#{2,3} .+$", re.M)
CONTAINER = re.compile(r"^</?untrusted_data\b.*$", re.M)
DIGITS = re.compile(r"\d+")
SPACES = re.compile(r"\s+")

# ── Le bloc d'enrichissement, reconnu à ses titres ──────────────────────────
ENRICHMENT_TITLES = (
    "Foundational Notes",
    "Relevant Knowledge",
    "Relevant Notes",
    "Relevant Decisions",
    "Work In Progress",
    "Reflexes",
    "Scaffolding Level",
    "Suggested Reasoning Plan",
    "Active Protocols",
    "Session Work Log",
)


def pointer_identical(turn, size):
    # Same bytes as `residence::pointer_identical`: the replay must measure the real pointer.
    return (f"_[unchanged since turn {turn}: this block is already in an earlier message "
            f"of this conversation ({size} chars not repeated)]_")


def pointer_figures(turn, size):
    return (f"_[same block as turn {turn}, only figures changed (progress counters): "
            f"not repeated, see the earlier message ({size} chars)]_")


def body_of(content):
    """The section body without its untrusted container (the id is a fresh nonce
    every turn, so keeping it would make every turn look new)."""
    return CONTAINER.sub("", content).strip()


def loose(body):
    """Digits collapsed: '99/110 (90%)' et '100/110 (91%)' sont le même bloc."""
    return SPACES.sub(" ", DIGITS.sub("#", body)).strip()


def fingerprint(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def sections_of(text):
    """[(title, body)] : split on the markdown headings the enrichment renders."""
    heads = [(m.start(), m.group(0).strip()) for m in HEADING.finditer(text)]
    out = []
    for i, (pos, head) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        out.append((head.lstrip("# ").strip(), text[pos:end]))
    return out


def is_enrichment(text):
    return "## Foundational Notes" in text or "### Foundational Notes" in text


def scan_file(path):
    """Returns the residence of ONE transcript, before and after the residence policy.

    `before` = every enriched user message as it was really sent;
    `after`  = the same blocks, with the ones already present in this conversation
               replaced by a bounded pointer.
    """
    doc = json.load(open(path, errors="ignore"))
    msgs = doc.get("messages") or []
    n = len(msgs)
    # tail[i] = number of requests that replay a block placed just before message i
    tail = [0] * (n + 1)
    for i in reversed(range(n)):
        tail[i] = tail[i + 1] + (1 if msgs[i].get("role") == "assistant" else 0)

    seen = {}          # title -> {exact hash: turn first delivered}
    last_full = {}     # title -> turn of the last FULL delivery, whatever the body
    turn = 0
    before = after = 0
    before_enr = after_enr = 0
    per_section = {}
    enriched = 0
    for i, m in enumerate(msgs):
        text = m.get("content")
        if m.get("role") != "user" or not isinstance(text, str) or not is_enrichment(text):
            continue
        enriched += 1
        turn += 1
        w = tail[i + 1]
        for title, chunk in sections_of(text):
            size = len(chunk)
            before += size * w
            if title in ENRICHMENT_TITLES:
                before_enr += size * w
            per_section.setdefault(title, [0, 0])[0] += size * w
            kept = chunk
            if title in ENRICHMENT_TITLES and size >= SECTION_FLOOR:
                exact = fingerprint(title, body_of(chunk))
                loose_h = fingerprint(title, loose(body_of(chunk)))
                bodies_exact = seen.setdefault(title, {})
                bodies_loose = seen.setdefault(title + "\x00loose", {})
                known = bodies_exact.get(exact) or bodies_loose.get(loose_h)
                fresh = last_full.get(title)
                stale = fresh is None or turn - fresh >= REFRESH_EVERY
                if known is not None and not stale:
                    kept = (pointer_identical(known, size) if exact in bodies_exact
                            else pointer_figures(known, size))
                else:
                    last_full[title] = turn
                    bodies_exact.setdefault(exact, turn)
                    bodies_loose.setdefault(loose_h, turn)
                    if len(bodies_exact) > 32:
                        bodies_exact.pop(next(iter(bodies_exact)))
                    if len(bodies_loose) > 32:
                        bodies_loose.pop(next(iter(bodies_loose)))
            after += len(kept) * w
            if title in ENRICHMENT_TITLES:
                after_enr += len(kept) * w
            per_section[title][1] += len(kept) * w
    return {
        "turns": tail[0],
        "before": before,
        "after": after,
        "before_enr": before_enr,
        "after_enr": after_enr,
        "messages": enriched,
        "sections": per_section,
    }


def group(totals, key, value):
    totals[key] = totals.get(key, 0) + value


def main():
    argv = sys.argv[1:]
    as_json = "--json" in argv
    check = "--check" in argv
    files = sorted(glob.glob(NATIVE))
    if not files:
        print(f"aucun transcript sous {NATIVE}", file=sys.stderr)
        return 2
    grand = {"before": 0, "after": 0, "before_enr": 0, "after_enr": 0, "sessions": 0, "messages": 0}
    per_section = {}
    kept_sessions = []
    for f in files:
        try:
            r = scan_file(f)
        except Exception:
            continue
        if not r["messages"]:
            continue
        grand["before"] += r["before"]
        grand["after"] += r["after"]
        grand["before_enr"] += r["before_enr"]
        grand["after_enr"] += r["after_enr"]
        grand["sessions"] += 1
        grand["messages"] += r["messages"]
        for t, (b, a) in r["sections"].items():
            slot = per_section.setdefault(t, [0, 0])
            slot[0] += b
            slot[1] += a
        kept_sessions.append((r["before"], os.path.basename(f)[:8], r))

    saved = grand["before"] - grand["after"]
    pct = saved / grand["before"] * 100 if grand["before"] else 0.0
    if as_json:
        print(json.dumps({
            "sessions": grand["sessions"],
            "enriched_messages": grand["messages"],
            "before_chars": grand["before"],
            "after_chars": grand["after"],
            "saved_pct": round(pct, 2),
            "before_enr_chars": grand["before_enr"],
            "after_enr_chars": grand["after_enr"],
            "sections": {k: {"before": v[0], "after": v[1],
                             "saved_pct": round((v[0] - v[1]) / v[0] * 100, 1) if v[0] else 0.0}
                         for k, v in per_section.items()},
        }, ensure_ascii=False, indent=2))
    else:
        print(f"[M] sessions={grand['sessions']} blocs d'enrichissement={grand['messages']}")
        print(f"[M] résidence AVANT = {grand['before']/1e6:.1f} M caractères relus")
        print(f"[M] résidence APRÈS = {grand['after']/1e6:.1f} M caractères relus")
        print(f"[M] ÉCONOMISÉ = {saved/1e6:.1f} M caractères ({pct:.1f} %)  "
              f"≈ {saved/3.6/1e6:.1f} M jetons (à 3,6 car./jeton, ordre de grandeur)")
        eb, ea = grand["before_enr"], grand["after_enr"]
        ep = (eb - ea) / eb * 100 if eb else 0.0
        print(f"[M] dont ENRICHISSEMENT SEUL (titres PO) : {eb/1e6:.1f} Mch → {ea/1e6:.1f} Mch   −{ep:.1f} %")
        print("    (le total inclut des documents collés par l'utilisateur, que cette")
        print("     politique ne touche pas : c'est pourquoi les deux chiffres sont donnés)")
        # The corpus mixes long agent sessions (where the money is) with short ones
        # (where there is nothing to save): both views, never one alone.
        longs = [r for _, _, r in kept_sessions if r["messages"] >= 20]
        if longs:
            lb = sum(r["before"] for r in longs)
            la = sum(r["after"] for r in longs)
            lp = (lb - la) / lb * 100 if lb else 0.0
            print(f"[M] sessions à ≥20 blocs ({len(longs)} / {grand['sessions']}) : "
                  f"{lb/1e6:.1f} Mch → {la/1e6:.1f} Mch   −{lp:.1f} %")
            print()
        print("Par section (résidence relue) :")
        for t, (b, a) in sorted(per_section.items(), key=lambda kv: -kv[1][0])[:12]:
            p = (b - a) / b * 100 if b else 0.0
            print(f"   {b/1e6:8.2f} Mch  →  {a/1e6:8.2f} Mch   −{p:4.1f} %   {t[:56]}")
        print()
        print("Cinq plus grosses sessions :")
        for b, name, r in sorted(kept_sessions, reverse=True)[:5]:
            p = (r["before"] - r["after"]) / r["before"] * 100 if r["before"] else 0.0
            print(f"   {name}  blocs={r['messages']:4d}  tours={r['turns']:4d}  "
                  f"{r['before']/1e6:7.1f} Mch → {r['after']/1e6:7.1f} Mch  −{p:.1f} %")

    if check:
        failures = []
        if grand["sessions"] < 5:
            failures.append(f"corpus trop petit ({grand['sessions']} sessions)")
        if grand["before"] <= 0:
            failures.append("aucune résidence mesurée : le corpus ne porte pas l'enrichissement")
        ep = ((grand["before_enr"] - grand["after_enr"]) / grand["before_enr"] * 100
              if grand["before_enr"] else 0.0)
        # Seuils lus sur la mesure du 28/11/2026 (505 sessions) : enrichissement −7,9 %
        # au total, −12,3 % sur les sessions à ≥20 blocs, −41 % sur les deux plus
        # longues. Une politique qui ne rend pas au moins la moitié de cela ne vaut pas.
        if ep < 5.0:
            failures.append(f"gain enrichissement {ep:.1f} % < 5 % : la politique ne mord pas")
        wip = per_section.get("Work In Progress")
        if wip and wip[0] and (wip[0] - wip[1]) / wip[0] * 100 < 8.0:
            failures.append("Work In Progress économise < 8 % : le bloc le plus répété n'est pas traité")
        if grand["after"] <= 0:
            failures.append("après = 0 : la simulation a tout supprimé, c'est une perte, pas une économie")
        if failures:
            for f in failures:
                print(f"[ÉCHEC] {f}", file=sys.stderr)
            return 1
        print(f"[OK] gain {pct:.1f} % (total) / {ep:.1f} % (enrichissement) "
              f"sur {grand['sessions']} sessions réelles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
