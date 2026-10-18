#!/usr/bin/env python3
"""Aligner PO / CBM / lecteur par symbole, en results/report.md.

Aucun vainqueur n'est infere : les trois quantites different (PO resout des aretes
CALLS, CBM des aretes de graphe, le lecteur liste des sites d'appel syntaxiques).
La verite terrain (ground-truth.json) est la reference du LECTEUR, pas une cible
pour PO ni CBM.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
GT_PATH = BASE / 'ground-truth.json'


def load(results_dir, name):
    p = results_dir / (name + '.json')
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _status_count(entry):
    """(status, result) — None result quand absent/invalide."""
    if entry is None:
        return ('missing', None)
    if entry.get('status') != 'ok':
        return (entry.get('status', 'error'), None)
    return ('ok', entry.get('result'))


def po_callers(results_dir, sym):
    st, r = _status_count(load(results_dir, 'po-callers-' + sym))
    if st != 'ok':
        return st, None
    return 'ok', {'resolved': len(r.get('caller_details') or []),
                  'names': len(r.get('callers') or [])}


def reader_callers(results_dir, sym):
    st, r = _status_count(load(results_dir, 'reader-callers-' + sym))
    if st != 'ok':
        return st, None
    sites = r.get('call_sites') or []
    return 'ok', {'call_sites': len(sites), 'limit_reached': bool(r.get('limit_reached'))}


def cbm_callers(results_dir, sym):
    st, r = _status_count(load(results_dir, 'cbm-callers-' + sym))
    if st != 'ok':
        return st, None
    data = r.get('structuredContent')
    if data is None and isinstance(r.get('content'), list) and r.get('content'):
        try:
            data = json.loads(r['content'][0].get('text', ''))
        except Exception:
            data = None
    if isinstance(data, dict):
        for key in ('callers', 'inbound', 'nodes', 'trace', 'results', 'edges', 'paths'):
            v = data.get(key)
            if isinstance(v, list):
                return 'ok', {'count': len(v), 'key': key}
        return 'ok', {'count': None, 'keys': sorted(data.keys())}
    return 'ok', {'count': None, 'note': 'structuredContent absent — voir le fichier brut'}


def build_report(results_dir=None, gt_path=None):
    results_dir = Path(results_dir or OUT)
    gt = json.loads((Path(gt_path or GT_PATH)).read_text())
    rows = []
    for sym, g in gt['symbols'].items():
        pst, po = po_callers(results_dir, sym)
        rst, rd = reader_callers(results_dir, sym)
        cst, cb = cbm_callers(results_dir, sym)
        rows.append({'symbol': sym, 'gt': g['call_sites'],
                     'po': po, 'po_status': pst,
                     'reader': rd, 'reader_status': rst,
                     'cbm': cb, 'cbm_status': cst})
    return gt, rows


def _cell(value, status, fmt):
    if value is None:
        return status
    return fmt(value)


def render(gt, rows):
    L = []
    L.append('# Comparaison PO / CBM / lecteur — rapport aligné')
    L.append('')
    L.append('Source indexée `%s` — vérité terrain vérifiée le %s.' % (gt['source_revision'][:12], gt['verified_at']))
    L.append('')
    L.append('> Les trois quantités diffèrent : le **lecteur** compte des **sites d\'appel syntaxiques**, '
             '**PO** résout des arêtes **CALLS**, **CBM** résout des arêtes de graphe. La colonne '
             '« vérité terrain » est la référence du lecteur, pas une cible pour PO ni CBM.')
    L.append('')
    L.append('| Symbole | Vérité terrain (syntaxe) | Lecteur (syntaxe) | PO (CALLS résolus) | CBM (graphe) |')
    L.append('|---|---:|---:|---:|---:|')
    for r in rows:
        rd = ('%d%s' % (r['reader']['call_sites'], ' ⚠' if r['reader'].get('limit_reached') else '')) if r['reader'] else r['reader_status']
        po = ('%d' % r['po']['resolved']) if r['po'] else r['po_status']
        if r['cbm']:
            c = r['cbm'].get('count')
            cb = ('%d' % c) if c is not None else ('n/a (%s)' % (', '.join(r['cbm'].get('keys', [])) if r['cbm'].get('keys') else 'voir brut'))
        else:
            cb = r['cbm_status']
        L.append('| `%s` | %d | %s | %s | %s |' % (r['symbol'], r['gt'], rd, po, cb))
    L.append('')
    L.append('`⚠` = limite atteinte (compte tronqué). `n/a` = forme non reconnue — voir le JSON brut.')
    L.append('')
    L.append('## Comment lire')
    L.append('')
    L.append('- **Lecteur ≈ vérité terrain** : même quantité. Un écart signale un bug de lecture (macros, génériques, `Self::`).')
    L.append('- **PO < vérité terrain** n\'est pas forcément faux : PO résout le dispatch et peut exclure les fichiers de test de ses arêtes CALLS.')
    L.append('- **CBM** tourne en « mode full » avec son propre parcours : son graphe n\'est pas la même quantité.')
    L.append('- Statuts `missing` / `error` : la mesure n\'a pas abouti (voir `results/*.json`).')
    return '\n'.join(L) + '\n'


def main():
    gt, rows = build_report()
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.md').write_text(render(gt, rows))
    print(render(gt, rows))


if __name__ == '__main__':
    main()
