#!/usr/bin/env python3
"""Banc MCP de la récupération partielle des réponses (GEO-6, revue de #1337).

CE QU'IL FAIT : il lance le VRAI serveur MCP (`mcp_server`, stdio) autant de fois que
de sessions, et parle JSON-RPC avec lui. Rien n'est simulé sauf le lanceur : il pose
l'environnement que `nexus_tools::launch` pose (répertoire temporaire et magasin
d'artefacts PROPRES À LA SESSION, `PO_VALIDATE_STORE_DIR`).

TROIS CAS, ceux de la revue :
 1. AUTORISATION — la lecture d'un artefact est autorisée à la session qui l'a reçu,
    et refusée à une autre qui connaît pourtant l'adresse : le sha256 seul n'est pas
    une capacité.
 2. TRANCHES ET RECONSTRUCTION — une tranche est réellement bornée, et la suite des
    tranches relit l'artefact BYTE POUR OCTET (avant et après réouverture du serveur,
    sans réexécuter l'outil qui a produit la réponse). C'est le cas que la revue
    pointe : un payload `serde_json::to_vec` tient sur UNE ligne, donc toute tranche
    valait le tout.
 3. `full` ET CONTRAT — `full: true` rend la charge brute (`_budget` absent), et la
    réponse projetée garde les mêmes clés de premier niveau que la brute, plus
    `_budget` : rien n'est renommé, rien n'est retiré.

USAGE :
    python3 scripts/geo_mcp_bench.py --server <chemin/mcp_server> \
        --po-server http://127.0.0.1:8080 --plan-id <uuid>

Sortie nulle si les trois cas passent ; chaque échec est nommé.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Session:
    """Un serveur MCP réel, avec SON magasin d'artefacts."""

    def __init__(self, server, po_server, root, label):
        self.label = label
        self.home = os.path.join(root, label)
        self.store = os.path.join(self.home, "artefacts")
        os.makedirs(self.store, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self.home,
            "TMPDIR": self.home,
            "PO_SERVER_URL": po_server,
            "PO_VALIDATE_STORE_DIR": self.store,
            "PO_VALIDATE_ROOTS": REPO,
            "RUST_LOG": "error",
        }
        # Le service local exige un jeton : le banc passe celui de son propre
        # environnement, jamais sur une ligne de commande, jamais affiché.
        for forwarded in ("PO_AUTH_TOKEN", "PO_JWT_SECRET"):
            if os.environ.get(forwarded):
                env[forwarded] = os.environ[forwarded]
        self.proc = subprocess.Popen(
            [server], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
        self._id = 0
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "geo-bench", "version": "1"}})
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0",
                                          "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method, params):
        self._id += 1
        rid = self._id
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                                          "params": params}) + "\n")
        self.proc.stdin.flush()
        box = {}
        def read():
            box["line"] = self.proc.stdout.readline()
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(120)
        if "line" not in box or not box["line"]:
            raise RuntimeError(f"[{self.label}] no answer to {method}")
        return json.loads(box["line"])

    def call(self, tool, args):
        answer = self._rpc("tools/call", {"name": tool, "arguments": args})
        if "error" in answer:
            return None, answer["error"].get("message", "error")
        result = answer.get("result") or {}
        text = "".join(block.get("text", "") for block in result.get("content") or []
                       if isinstance(block, dict))
        return text, None

    def address_of(self, text):
        for line in text.split("\n"):
            if "artefact:" in line:
                candidate = line.split("artefact:")[1].strip().strip('"').split()[0]
                candidate = candidate.rstrip(",}]")
                if len(candidate) == 64:
                    return candidate
        return None

    def artefact_text(self, address):
        with open(os.path.join(self.store, address), "rb") as fh:
            return fh.read()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.kill()
        except Exception:
            pass


def check(label, ok, detail=""):
    print(("  ✔ " if ok else "  ✘ ") + label + (f" — {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--po-server", default="http://127.0.0.1:8080")
    ap.add_argument("--plan-id", required=True, help="un plan réel, pour une réponse > 8 000 octets")
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="geo-mcp-bench-")
    ok = True
    a = b = None
    try:
        a = Session(args.server, args.po_server, root, "session-a")
        text, err = a.call("plan", {"action": "get", "plan_id": args.plan_id})
        if err or not text:
            print(f"✘ la réponse de plan.get est vide ou en erreur: {err}")
            return 1
        address = a.address_of(text)
        if not check("une réponse volumineuse rend une poignée", address is not None,
                     f"{len(text)} octets, artefact {address or '—'}"):
            return 1
        full_text = a.artefact_text(address).decode()
        lines = full_text.count("\n") + 1

        # ── CAS 2 : tranches bornées, reconstruction exacte, réouverture ────────
        print("2. tranches et reconstruction")
        ok &= check("l'artefact est multi-ligne", lines > 20, f"{lines} lignes, {len(full_text)} octets")
        ok &= check("la réponse projetée est plus petite que l'originale",
                    len(text) < len(full_text), f"{len(text)} < {len(full_text)}")
        slice_text, err = a.call("validate", {"action": "read", "artefact": address,
                                              "from_line": 0, "to_line": 10})
        ok &= check("une tranche de 10 lignes est lisible par la vraie route MCP",
                    err is None, err or "")
        if err is None:
            expected = "\n".join(full_text.split("\n")[0:10])
            ok &= check("la tranche vaut exactement les 10 premières lignes",
                        slice_text == expected, f"{len(slice_text)} octets")
            ok &= check("la tranche est réellement bornée",
                        len(slice_text) < len(full_text) // 4,
                        f"{len(slice_text)} sur {len(full_text)}")
        rebuilt, at, step = "", 0, 50
        while at < lines:
            chunk, err = a.call("validate", {"action": "read", "artefact": address,
                                             "from_line": at, "to_line": min(at + step, lines)})
            if err:
                ok &= check(f"tranche {at}-{at+step} lisible", False, err)
                break
            if at:
                rebuilt += "\n"
            rebuilt += chunk
            at += step
        ok &= check("la suite des tranches reconstruit l'artefact byte pour byte",
                    rebuilt.encode() == a.artefact_text(address),
                    f"{len(rebuilt)} octets reconstruits")

        # réouverture : même magasin, nouveau processus, AUCUN nouvel appel à plan.get
        store_a = a.store
        a.close()
        a = Session(args.server, args.po_server, root, "session-a")
        reopened, err = a.call("validate", {"action": "read", "artefact": address,
                                            "from_line": 0, "to_line": 10})
        ok &= check("après réouverture du serveur, la poignée résout encore",
                    err is None and reopened == "\n".join(full_text.split("\n")[0:10]),
                    err or "identique")

        # ── CAS 1 : un AUTRE appelant qui connaît l'adresse est refusé ─────────
        print("1. autorisation des artefacts")
        b = Session(args.server, args.po_server, root, "session-b")
        theirs, err = b.call("validate", {"action": "read", "artefact": address,
                                          "from_line": 0, "to_line": 10})
        refused = err is not None or not theirs or "absent from the store" in theirs
        ok &= check("une autre session, même adresse connue, est refusée",
                    refused and (not theirs or "absent from the store" in theirs),
                    (err or (theirs or "réponse vide"))[:120])
        ok &= check("le magasin de l'autre session ne contient pas ces octets",
                    not os.path.exists(os.path.join(b.store, address)), b.store)

        # ── CAS 3 : full intact, contrat de réponse préservé ───────────────────
        print("3. full et contrat")
        raw, err = a.call("plan", {"action": "get", "plan_id": args.plan_id, "full": True})
        # `_budget` is a TOP-LEVEL key, never a substring: the payload itself may talk
        # about it (this very task does).
        has_budget_key = any(line.startswith("_budget:") for line in (raw or "").split("\n"))
        ok &= check("full: true rend la charge brute (pas de clé _budget)",
                    err is None and not has_budget_key, err or f"{len(raw or '')} octets")
        if raw:
            ok &= check("la brute est plus grosse que la projetée", len(raw) > len(text),
                        f"{len(raw)} > {len(text)}")

            def keys(value):
                return sorted(line.split(":")[0].strip() for line in value.split("\n")
                              if line and not line.startswith(" ") and ":" in line)
            projected_keys = [k for k in keys(text) if k != "_budget"]
            ok &= check("mêmes clés de premier niveau, plus _budget seulement",
                        projected_keys == keys(raw),
                        f"{projected_keys} vs {keys(raw)}")
    finally:
        for s in (a, b):
            if s:
                s.close()
        shutil.rmtree(root, ignore_errors=True)

    print("\nBANC " + ("VERT" if ok else "ROUGE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
