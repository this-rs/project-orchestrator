#!/usr/bin/env python3
"""Mocked report alignment: fixture results, no Docker."""
import importlib.util, json, tempfile
from pathlib import Path

def run():
    spec = importlib.util.spec_from_file_location('report', Path(__file__).with_name('report.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / 'results'; res.mkdir()
        fixtures = {
            'with_cognitive_routing': (12, 4, 3),
            'decision_confidence': (7, 7, 7),
            'build_enrichment_pipeline': (3, 3, 3),
            'close_decision_with': (3, 2, 3),
        }
        for sym, (gt, po, cbm) in fixtures.items():
            (res / f'po-callers-{sym}.json').write_text(json.dumps({'status': 'ok', 'result': {
                'callers': ['x'] * po, 'caller_details': [{'name': 'x'}] * po}}))
            (res / f'reader-callers-{sym}.json').write_text(json.dumps({'status': 'ok', 'result': {
                'call_sites': [{'path': 'p'}] * gt, 'limit_reached': False}}))
            (res / f'cbm-callers-{sym}.json').write_text(json.dumps({'status': 'ok', 'result': {
                'structuredContent': {'callers': [{'name': 'x'}] * cbm}}}))
        gt, rows = m.build_report(res, Path(__file__).with_name('ground-truth.json'))
        by = {r['symbol']: r for r in rows}
        assert by['with_cognitive_routing']['reader']['call_sites'] == 12
        assert by['with_cognitive_routing']['po']['resolved'] == 4
        assert by['with_cognitive_routing']['cbm']['count'] == 3
        assert by['decision_confidence']['gt'] == 7
        md = m.render(gt, rows)
        assert 'Comparaison PO / CBM / lecteur' in md
        assert '| 12 | 12 | 4 | 3 |' in md, md
        # fichier manquant -> statut honnête 'missing'
        (res / 'po-callers-close_decision_with.json').unlink()
        _, rows2 = m.build_report(res, Path(__file__).with_name('ground-truth.json'))
        assert {r['symbol']: r['po_status'] for r in rows2}['close_decision_with'] == 'missing'
    print('PASS: aligned report renders counts and honest missing statuses')

if __name__ == '__main__':
    run()
