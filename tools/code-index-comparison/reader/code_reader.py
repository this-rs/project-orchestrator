#!/usr/bin/env python3
"""Read-only AST code index. Call sites are syntax, never proven dispatch edges."""
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'python'))
DB = BASE / 'code-index.sqlite'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript('''
    CREATE TABLE IF NOT EXISTS projects(name TEXT PRIMARY KEY, root TEXT, revision TEXT);
    CREATE TABLE IF NOT EXISTS files(project TEXT, path TEXT, sha TEXT, bytes INTEGER,
      lines INTEGER, parse_errors INTEGER, PRIMARY KEY(project,path));
    CREATE TABLE IF NOT EXISTS symbols(id INTEGER PRIMARY KEY, project TEXT, path TEXT,
      name TEXT, qualified TEXT, kind TEXT, start INTEGER, end INTEGER, test INTEGER);
    CREATE TABLE IF NOT EXISTS calls(symbol_id INTEGER, project TEXT, path TEXT,
      line INTEGER, expression TEXT, target_name TEXT);
    CREATE INDEX IF NOT EXISTS symbol_name ON symbols(project,name);
    CREATE INDEX IF NOT EXISTS call_target ON calls(project,target_name);
    CREATE INDEX IF NOT EXISTS call_owner ON calls(symbol_id);
    CREATE TABLE IF NOT EXISTS imports(project TEXT,path TEXT,line INTEGER,expression TEXT);
    ''')
    return c

def text(node, data):
    return data[node.start_byte:node.end_byte].decode('utf-8', 'replace')

def named(node, field, data):
    n = node.child_by_field_name(field)
    return text(n, data) if n else ''

def parsers():
    from tree_sitter import Language, Parser
    import tree_sitter_rust as rust
    import tree_sitter_typescript as ts
    import tree_sitter_javascript as js
    return {'.rs': Parser(Language(rust.language())),
            '.ts': Parser(Language(ts.language_typescript())),
            '.tsx': Parser(Language(ts.language_tsx())),
            '.js': Parser(Language(js.language())),
            '.jsx': Parser(Language(js.language()))}

KINDS = {'function_item', 'function_signature_item', 'function_declaration',
         'method_definition', 'method_signature', 'struct_item', 'enum_item',
         'trait_item', 'type_item', 'class_declaration', 'interface_declaration',
         'type_alias_declaration', 'const_item', 'static_item', 'mod_item'}
FUNCTIONS = {'function_item', 'function_declaration', 'method_definition',
             'arrow_function', 'function_expression'}

def index_project(project, root):
    root = Path(root).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).split(b'\0')
    ps = parsers()
    c = connect()
    with c:
        for table in ['calls', 'symbols', 'files', 'imports']:
            c.execute(f'DELETE FROM {table} WHERE project=?', (project,))
        c.execute('INSERT OR REPLACE INTO projects VALUES(?,?,?)', (project, str(root), revision))
        for raw in tracked:
            if not raw:
                continue
            path = raw.decode('utf-8')
            p = root / path
            if p.suffix not in ps or p.is_symlink() or not p.is_file():
                continue
            data = p.read_bytes()
            tree = ps[p.suffix].parse(data)
            c.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',
                      (project,path,digest(data),len(data),len(data.splitlines()),int(tree.root_node.has_error)))
            # Iterative DFS, carrying the nearest function, scope and test context.
            test_file = ('tests' in p.parts or '__tests__' in p.parts or '_test' in p.stem or '.test' in p.stem or '.spec' in p.stem)
            stack = [(tree.root_node,None,[],test_file)]
            while stack:
                node, owner, scopes, test = stack.pop()
                typ = node.type
                name = named(node, 'name', data)
                if typ == 'impl_item':
                    scopes = scopes + [named(node, 'type', data)]
                if typ == 'mod_item' and name == 'tests':
                    test = True
                prev = node.prev_named_sibling
                while prev and prev.type == 'attribute_item':
                    if 'cfg(test)' in text(prev,data) or '#[test]' in text(prev,data) or '::test]' in text(prev,data):
                        test = True
                    prev = prev.prev_named_sibling
                if typ in {'arrow_function','function_expression'} and node.parent:
                    parent = node.parent
                    if parent.type in {'variable_declarator','pair'}:
                        name = named(parent, 'name', data) or named(parent, 'key', data)
                if (typ in KINDS or typ in FUNCTIONS) and name:
                    qualified = '::'.join([project,path,*scopes,name])
                    cur = c.execute('INSERT INTO symbols(project,path,name,qualified,kind,start,end,test) VALUES(?,?,?,?,?,?,?,?)',
                                    (project,path,name,qualified,typ,node.start_point.row+1,node.end_point.row+1,int(test)))
                    if typ in FUNCTIONS:
                        owner = cur.lastrowid
                    if typ in {'mod_item','class_declaration','trait_item','interface_declaration'}:
                        scopes = scopes + [name]
                if typ == 'call_expression':
                    fn = node.child_by_field_name('function')
                    if fn:
                        # Unwrap turbofish/generic arguments to preserve the callee name.
                        while fn.type == 'generic_function':
                            child = fn.child_by_field_name('function')
                            if not child:
                                break
                            fn = child
                        expression = text(fn,data)
                        target = named(fn,'field',data) or named(fn,'property',data) or named(fn,'name',data)
                        if not target:
                            target = expression.split('::')[-1] if fn.type in {'identifier','scoped_identifier'} else ''
                        c.execute('INSERT INTO calls VALUES(?,?,?,?,?,?)',
                                  (owner,project,path,node.start_point.row+1,expression,target))
                if typ in {'use_declaration','import_statement'}:
                    c.execute('INSERT INTO imports VALUES(?,?,?,?)',
                              (project,path,node.start_point.row+1,text(node,data)))
                stack.extend((child,owner,scopes,test) for child in reversed(node.named_children))
    return stats({'project': project})

def stats(args):
    c = connect()
    result = []
    for p in c.execute('SELECT * FROM projects WHERE name LIKE ?', (args.get('project','%'),)):
        row = dict(p)
        for table in ['files','symbols','calls','imports']:
            row[table] = c.execute(f'SELECT COUNT(*) FROM {table} WHERE project=?',(p['name'],)).fetchone()[0]
        row['lines'] = c.execute('SELECT COALESCE(SUM(lines),0) FROM files WHERE project=?',(p['name'],)).fetchone()[0]
        row['files_with_parse_errors'] = c.execute('SELECT COUNT(*) FROM files WHERE project=? AND parse_errors>0',(p['name'],)).fetchone()[0]
        result.append(row)
    return {'projects':result,'call_semantics':'syntactic call sites; name matches are candidates, not resolved dispatch',
            'coverage':'git-tracked .rs/.ts/.tsx/.js/.jsx; no dependency sources or runtime traces'}

def limit(args):
    return min(100,max(1,int(args.get('limit',15))))

def search(args):
    c = connect()
    query = '%'+args.get('name','')+'%'
    rows = c.execute('SELECT * FROM symbols WHERE project=? AND name LIKE ? AND path LIKE ? AND (test=0 OR ?=1) ORDER BY test,path,start LIMIT ?',
                     (args['project'],query,'%'+args.get('path','')+'%',int(args.get('include_tests',False)),limit(args))).fetchall()
    return {'symbols':[dict(r) for r in rows], 'limit':limit(args)}

def source_file(c, project, path):
    p = c.execute('SELECT root FROM projects WHERE name=?',(project,)).fetchone()
    f = c.execute('SELECT sha FROM files WHERE project=? AND path=?',(project,path)).fetchone()
    if not p or not f:
        raise ValueError('Unknown indexed project/file')
    root = Path(p['root']).resolve()
    full = (root/path).resolve()
    if not full.is_relative_to(root):
        raise ValueError('Path outside project')
    data = full.read_bytes()
    if digest(data) != f['sha']:
        raise ValueError('Stale index: file changed; reindex before reading')
    return data.decode('utf-8','replace').splitlines()

def snippet(args):
    c = connect()
    s = c.execute('SELECT * FROM symbols WHERE id=?',(args['id'],)).fetchone()
    if not s:
        raise ValueError('Unknown symbol id')
    lines = source_file(c,s['project'],s['path'])
    start = max(s['start'],int(args.get('start',s['start'])))
    end = min(s['end'],start+min(400,max(1,int(args.get('max_lines',80))))-1)
    return {'symbol':dict(s),'code':'\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1,end)),
            'has_more':end<s['end'],'next_start':end+1 if end<s['end'] else None}

def outline(args):
    c = connect()
    source_file(c,args['project'],args['path'])
    rows = c.execute('SELECT id,name,qualified,kind,start,end,test FROM symbols WHERE project=? AND path=? ORDER BY start LIMIT ? OFFSET ?',
                     (args['project'],args['path'],limit(args),int(args.get('offset',0)))).fetchall()
    return {'symbols':[dict(r) for r in rows]}

def calls(args):
    c = connect()
    direction = args.get('direction','inbound')
    if direction == 'outbound':
        if 'id' not in args:
            raise ValueError('Outbound calls require a symbol id')
        rows = c.execute('SELECT * FROM calls WHERE symbol_id=? ORDER BY line LIMIT ?',
                         (args['id'],limit(args))).fetchall()
    elif direction == 'inbound':
        rows = c.execute('SELECT c.*,s.name AS caller,s.qualified AS caller_qualified,s.test FROM calls c LEFT JOIN symbols s ON s.id=c.symbol_id WHERE c.project=? AND c.target_name=? AND (COALESCE(s.test,0)=0 OR ?=1) ORDER BY COALESCE(s.test,0),c.path,c.line LIMIT ?',
                         (args['project'],args['name'],int(args.get('include_tests',False)),limit(args))).fetchall()
    else:
        raise ValueError('direction must be inbound or outbound')
    sites=[]
    for r in rows:
        row=dict(r)
        expression=row['expression']
        if len(expression)>360:
            row['expression']=expression[:180]+' … '+expression[-180:]
            row['expression_truncated']=True
            row['expression_characters']=len(expression)
        sites.append(row)
    return {'semantics':'syntax only; inbound rows are name-match candidates, including possible homonyms',
            'call_sites':sites,'limit':limit(args),'limit_reached':len(sites)==limit(args)}

def search_code(args):
    c = connect()
    query = args['query']
    if not query:
        raise ValueError('Non-empty literal query required')
    matches=[]
    for f in c.execute('SELECT path FROM files WHERE project=? AND path LIKE ? ORDER BY path',
                       (args['project'],'%'+args.get('path','')+'%')):
        for i,line in enumerate(source_file(c,args['project'],f['path'])):
            if query in line:
                at=line.index(query)
                matches.append({'path':f['path'],'line':i+1,'text':line[max(0,at-120):at+len(query)+200]})
                if len(matches)>=limit(args):
                    return {'matches':matches,'limit_reached':True}
    return {'matches':matches,'limit_reached':False}

OPERATIONS={'stats':stats,'search':search,'snippet':snippet,'outline':outline,'calls':calls,'search_code':search_code}

def dispatch(operation, args):
    # Reject stale commit metadata and tracked worktree edits rather than use old edges.
    if operation != 'stats':
        c = connect()
        project = args.get('project')
        if operation == 'snippet' or (operation == 'calls' and args.get('direction') == 'outbound'):
            s = c.execute('SELECT project FROM symbols WHERE id=?',(args.get('id'),)).fetchone()
            if not s:
                raise ValueError('Unknown symbol id')
            project = s['project']
        p = c.execute('SELECT * FROM projects WHERE name=?',(project,)).fetchone()
        if not p:
            raise ValueError('Unknown project')
        revision = subprocess.check_output(['git','-C',p['root'],'rev-parse','HEAD'],text=True).strip()
        dirty = subprocess.check_output(['git','-C',p['root'],'status','--porcelain','--untracked-files=no'],text=True)
        if revision != p['revision'] or dirty:
            raise ValueError('Stale index: repository changed; reindex before querying')
    return OPERATIONS[operation](args)

def bind(args):
    c=connect()
    p=c.execute('SELECT revision FROM projects WHERE name=?',(args['project'],)).fetchone()
    root=Path(args['root']).resolve()
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if not p or revision!=p['revision']:
        raise ValueError('Binding requires the exact indexed revision; otherwise reindex')
    with c:
        c.execute('UPDATE projects SET root=? WHERE name=?',(str(root),args['project']))
    return stats({'project':args['project']})

def serve():
    # Standard newline-delimited stdio MCP; no network listener or mutation tools.
    schemas={
      'stats': {'project':{'type':'string'}},
      'search': {'project':{'type':'string'},'name':{'type':'string'},'path':{'type':'string'},'limit':{'type':'integer'},'include_tests':{'type':'boolean'}},
      'snippet': {'id':{'type':'integer'},'start':{'type':'integer'},'max_lines':{'type':'integer'}},
      'outline': {'project':{'type':'string'},'path':{'type':'string'},'limit':{'type':'integer'},'offset':{'type':'integer'}},
      'calls': {'project':{'type':'string'},'name':{'type':'string'},'id':{'type':'integer'},'direction':{'enum':['inbound','outbound']},'limit':{'type':'integer'},'include_tests':{'type':'boolean'}},
      'search_code': {'project':{'type':'string'},'query':{'type':'string'},'path':{'type':'string'},'limit':{'type':'integer'}}}
    for line in sys.stdin:
        req={}
        try:
            req=json.loads(line)
            if 'id' not in req:
                continue
            method=req['method']
            if method=='initialize':
                result={'protocolVersion':req.get('params',{}).get('protocolVersion','2024-11-05'),
                        'capabilities':{'tools':{}},'serverInfo':{'name':'po-code-reader','version':'1.0.0'}}
            elif method=='ping':
                result={}
            elif method=='tools/list':
                result={'tools':[{'name':name,'description':f'Read-only AST index: {name}. Call edges are syntactic candidates.',
                                 'inputSchema':{'type':'object','properties':props},
                                 'annotations':{'readOnlyHint':True}} for name,props in schemas.items()]}
            elif method=='tools/call':
                params=req['params']
                data=dispatch(params['name'],params.get('arguments',{}))
                result={'content':[{'type':'text','text':json.dumps(data,ensure_ascii=False)}]}
            else:
                raise ValueError('Unsupported method')
            response={'jsonrpc':'2.0','id':req['id'],'result':result}
        except Exception as e:
            response={'jsonrpc':'2.0','id':req.get('id'),'error':{'code':-32603,'message':str(e)}}
        print(json.dumps(response,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['index','bind','serve',*OPERATIONS])
    parser.add_argument('args',nargs='?',default='{}',help='JSON arguments')
    args=parser.parse_args()
    if args.operation=='serve':
        serve();return
    data=json.loads(args.args)
    result=index_project(data['project'],data['root']) if args.operation=='index' else bind(data) if args.operation=='bind' else dispatch(args.operation,data)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
