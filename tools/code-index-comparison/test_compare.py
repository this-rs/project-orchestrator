"""Mocked orchestration checks only: this test does not start Docker or index code."""
import contextlib,io,importlib.util,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch

def run():
    spec=importlib.util.spec_from_file_location('comparison',Path(__file__).with_name('compare.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    requests=[];commands=[]
    def fake_http(url,body=None,headers=None):
        requests.append((url,body))
        if '/sync?' in url:return {'files_synced':1,'errors':0}
        if '/api/projects/po-self' in url:return {'id':'fixture-project','file_count':0,'last_synced':None}
        if '/tx/commit' in url:return {'results':[],'errors':[]}
        return {'fixture':True}
    def fake_command(service,args):
        commands.append((service,args))
        if args[0]=='curl':return {'total':0}
        return {'fixture':True}
    with tempfile.TemporaryDirectory() as tmp:
        module.BASE=Path(tmp);module.OUT=Path(tmp)/'results'
        (module.BASE/'.env').write_text('NEO4J_PASSWORD=fixture-password\nMEILI_MASTER_KEY=fixture-key\n')
        with patch.object(module,'http',fake_http),patch.object(module,'command',fake_command),contextlib.redirect_stdout(io.StringIO()):
            module.main()
        summary=json.loads((module.OUT/'summary.json').read_text())
        assert summary['po_warm_state'] is False
        assert any('/api/projects/po-self/sync?force=true' in url for url,body in requests)
        assert all('project_slug=po-self' in url for url,body in requests
                   if any(path in url for path in ['/api/code/search?','/api/code/callgraph?','/api/code/architecture?']))
        assert any(service=='cbm' and 'index_repository' in args and '/workspace/po' in args for service,args in commands)
        assert any(service=='reader' and '/opt/reader/code_reader.py' in args for service,args in commands)
        assert len(list(module.OUT.glob('*.json')))>=28
        fake=module.BASE/'fake_mcp.py'
        fake.write_text('''import json,sys
for line in sys.stdin:
 r=json.loads(line)
 if 'id' in r:
  print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':{'ok':True}}),flush=True)
''')
        client=module.McpClient([sys.executable,str(fake)],module.BASE/'mcp.log')
        try:assert client.request('tools/call',{'name':'stats','arguments':{}})=={'ok':True}
        finally:client.close()
        assert any('/api/notes/context?' in url for url,body in requests)
        assert any(service=='cbm' and 'trace_path' in args and args[args.index('--depth')+1]=='1' for service,args in commands)
    print('PASS: mocked project sync, project-scoped queries, identical container source path, independent outputs')
if __name__=='__main__':run()
